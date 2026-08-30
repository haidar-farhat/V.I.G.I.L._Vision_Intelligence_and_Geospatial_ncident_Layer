#!/usr/bin/env node
/**
 * Architectural checks.
 *
 * These are the invariants from ARCHITECTURE.md, expressed as something a machine
 * can fail on. A rule that lives only in a document is a rule that erodes: the
 * layering holds for a year and then one import in a hurry inverts it, and nobody
 * notices until the worker cannot be built without the desktop app.
 *
 *   node scripts/dev.mjs lint
 */

import { readFileSync } from 'node:fs';
import { readdir } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

const SKIP_DIRS = new Set(['node_modules', '.git', 'dist', 'build', 'target', 'data', 'coverage']);

/** Every TypeScript source file in the workspace. */
const sources = async (dir = ROOT, found = []) => {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    if (entry.name.startsWith('.') && entry.name !== '.github') continue;
    const full = path.join(dir, entry.name);

    if (entry.isDirectory()) {
      if (SKIP_DIRS.has(entry.name)) continue;
      await sources(full, found);
    } else if (entry.name.endsWith('.ts') || entry.name.endsWith('.tsx')) {
      found.push(full);
    }
  }
  return found;
};

const relative = (file) => path.relative(ROOT, file).split(path.sep).join('/');

/** Which architectural layer a file belongs to. */
const layerOf = (rel) => {
  if (rel.startsWith('packages/')) return 'package';
  if (rel.startsWith('services/')) return 'service';
  if (rel.startsWith('apps/')) return 'app';
  if (rel.startsWith('simulator/')) return 'simulator';
  return 'other';
};

const IMPORT_PATTERN = /(?:import|export)[\s\S]*?from\s+['"]([^'"]+)['"]|import\s*\(\s*['"]([^'"]+)['"]/g;

const importsOf = (text) => {
  const found = [];
  for (const match of text.matchAll(IMPORT_PATTERN)) {
    const specifier = match[1] ?? match[2];
    if (specifier !== undefined) found.push(specifier);
  }
  return found;
};

/**
 * Source with comments removed.
 *
 * Several checks would otherwise fire on prose that *describes* the anti-pattern
 * it is guarding against - the docs for `buildRtspUrl` necessarily show what a
 * credentialed RTSP URL looks like. Documenting a hazard is the opposite of
 * committing one, so the scanners look at code.
 */
const withoutComments = (text) =>
  text
    // Blank out comment bodies rather than deleting them, so reported line
    // numbers still match the file on disk.
    .replace(/\/\*[\s\S]*?\*\//g, (block) => block.replace(/[^\n]/g, ' '))
    .replace(
      /(^|[^:])\/\/[^\n]*/g,
      (line, prefix) => prefix + ' '.repeat(line.length - prefix.length),
    );

const problems = [];
const fail = (file, line, rule, detail) =>
  problems.push({ file: relative(file), line, rule, detail });

const lineOf = (text, index) => text.slice(0, index).split('\n').length;

// ---------------------------------------------------------------------------

/**
 * Dependencies point inward. A package may not reach up into a service or an app,
 * and a service may not reach into an app. Violating this is what turns a
 * separable system into a monolith one import at a time.
 */
const checkLayering = (file, rel, text) => {
  const layer = layerOf(rel);

  for (const specifier of importsOf(text)) {
    const target =
      specifier.startsWith('@sentinel/')
        ? specifier.slice('@sentinel/'.length).split('/')[0]
        : null;

    const targetLayer =
      target === null
        ? null
        : ['api', 'worker', 'inference', 'recorder', 'event-engine'].includes(target)
          ? 'service'
          : target === 'desktop'
            ? 'app'
            : 'package';

    if (layer === 'package' && (targetLayer === 'service' || targetLayer === 'app')) {
      fail(file, 1, 'layering', `a package must not import from ${targetLayer} "${specifier}"`);
    }
    if (layer === 'service' && targetLayer === 'app') {
      fail(file, 1, 'layering', `a service must not import from the desktop app "${specifier}"`);
    }
    if (specifier.includes('../../apps/') || specifier.includes('../apps/')) {
      fail(file, 1, 'layering', `reaches into the desktop app by relative path: "${specifier}"`);
    }
  }
};

/**
 * Nothing in the product may depend on a cloud service, and nothing may reach the
 * Internet. The guarantee is only worth something if it is checked.
 */
const BANNED_MODULES = [
  ['aws-sdk', 'cloud SDK'],
  ['@aws-sdk/', 'cloud SDK'],
  ['@google-cloud/', 'cloud SDK'],
  ['firebase', 'cloud SDK'],
  ['@azure/', 'cloud SDK'],
  ['@sentry/', 'remote telemetry'],
  ['posthog', 'remote analytics'],
  ['mixpanel', 'remote analytics'],
  ['@amplitude/', 'remote analytics'],
  ['mapbox-gl', 'online map SDK'],
  ['@googlemaps/', 'online map SDK'],
  ['openai', 'cloud AI'],
  ['@anthropic-ai/', 'cloud AI'],
  ['@google/generative-ai', 'cloud AI'],
];

const checkOfflineOnly = (file, rel, text) => {
  for (const specifier of importsOf(text)) {
    for (const [banned, why] of BANNED_MODULES) {
      if (specifier === banned || specifier.startsWith(banned)) {
        fail(file, 1, 'zero-wan', `imports "${specifier}" (${why}); the platform must never require the Internet`);
      }
    }
  }

  // A hard-coded public URL in product code is a latent Internet dependency, even
  // when nothing calls it yet. Tests are exempt: proving the egress guard refuses
  // an external URL requires naming one.
  if (rel.includes('/test/') || rel.endsWith('.test.ts')) return;

  const code = withoutComments(text);
  const urlPattern = /['"`](https?:\/\/[^'"`\s]+)['"`]/g;
  for (const match of code.matchAll(urlPattern)) {
    const url = match[1];
    if (url === undefined) continue;

    const host = (() => {
      try {
        return new URL(url).hostname;
      } catch {
        return '';
      }
    })();

    const isLocal =
      host === 'localhost' ||
      host === '127.0.0.1' ||
      host === '::1' ||
      host.endsWith('.local') ||
      /^(10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)/.test(host) ||
      // Schema and spec identifiers are namespaces, not endpoints.
      host === 'json.schemastore.org' ||
      host === 'www.w3.org' ||
      host === 'schemas.xmlsoap.org' ||
      host === 'www.onvif.org';

    if (!isLocal) {
      fail(
        file,
        lineOf(code, match.index ?? 0),
        'zero-wan',
        `hard-coded external URL "${url}"; imported resources must come from local files`,
      );
    }
  }
};

/**
 * Credentials must never be committed, and a plausible-looking literal in source
 * is how they usually are.
 */
const SECRET_PATTERNS = [
  [/(password|passwd|secret|api[_-]?key|token)\s*[:=]\s*['"][^'"]{8,}['"]/i, 'possible hard-coded credential'],
  [/-----BEGIN (RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----/, 'private key material'],
];

/**
 * Words that stand in for a credential rather than being one.
 *
 * The schema and the security package both have to *name* the shape of a
 * credentialed RTSP URL in order to forbid it, including inside SQL comments the
 * JavaScript comment stripper cannot see. A real password still trips the check;
 * `user:pass@host` does not.
 */
const PLACEHOLDER_SECRETS = new Set([
  'pass',
  'password',
  'passwd',
  'secret',
  'credentials',
  'xxx',
  'yyy',
  'changeme',
  'redacted',
]);

const checkRtspCredentials = (file, code) => {
  for (const match of code.matchAll(/rtsp:\/\/([^\s'"/:]+):([^\s'"/@]+)@/g)) {
    const password = (match[2] ?? '').toLowerCase().replace(/[<>${}*]/g, '');
    if (PLACEHOLDER_SECRETS.has(password)) continue;

    fail(file, lineOf(code, match.index ?? 0), 'secrets', 'RTSP URL with embedded credentials');
  }
};

const checkSecrets = (file, rel, text) => {
  // Test fixtures deliberately contain credential-shaped strings in order to
  // prove they are redacted. Their whole job is to fail loudly if that stops
  // working, so scanning them would invert the intent.
  if (rel.includes('/test/') || rel.endsWith('.test.ts')) return;

  const code = withoutComments(text);
  for (const [pattern, why] of SECRET_PATTERNS) {
    const match = pattern.exec(code);
    if (match !== null) {
      fail(file, lineOf(code, match.index), 'secrets', why);
    }
  }

  // Scans the raw text: SQL comments inside template literals are invisible to
  // the JavaScript comment stripper, so placeholder filtering does the work here.
  checkRtspCredentials(file, text);
};

/**
 * Unfinished work must be visible as unfinished.
 *
 * A TODO is fine. A TODO that a status table describes as complete is not, so
 * this counts them and STATUS.md is expected to stay honest about them.
 */
const checkPlaceholders = (file, rel, text) => {
  const markers = [...text.matchAll(/\b(TODO|FIXME|XXX|HACK)\b/g)];
  for (const marker of markers) {
    fail(file, lineOf(text, marker.index ?? 0), 'placeholder', `${marker[1]} left in source`);
  }
};

/**
 * The whole workspace runs on Node's type-stripping loader, which erases types
 * but cannot transform syntax. An enum or a parameter property compiles under
 * tsc and then fails at runtime, so it is worth catching directly.
 */
const checkErasableSyntax = (file, rel, text) => {
  const nonErasable = [
    [/^\s*(export\s+)?(const\s+)?enum\s+\w+/m, 'enum (use a const object plus a union type)'],
    [/^\s*(export\s+)?namespace\s+\w+/m, 'namespace'],
    // A modifier immediately after "(" or "," is a parameter property. A
    // `readonly` inside a type annotation (`x: readonly T[]`) follows a colon and
    // must not be flagged - an earlier version of this rule did exactly that.
    [
      /[(,]\s*(?:public|private|protected|readonly)\s+(?:readonly\s+)?[A-Za-z_$][\w$]*\s*[?:]/,
      'parameter property',
    ],
  ];

  const code = withoutComments(text);
  for (const [pattern, why] of nonErasable) {
    const match = pattern.exec(code);
    if (match !== null) {
      fail(file, lineOf(code, match.index), 'erasable-syntax', `${why} does not erase`);
    }
  }
};

// ---------------------------------------------------------------------------

const files = await sources();

for (const file of files) {
  const rel = relative(file);
  const text = readFileSync(file, 'utf8');

  checkLayering(file, rel, text);
  checkOfflineOnly(file, rel, text);
  checkSecrets(file, rel, text);
  checkPlaceholders(file, rel, text);
  checkErasableSyntax(file, rel, text);
}

const byRule = new Map();
for (const problem of problems) {
  byRule.set(problem.rule, [...(byRule.get(problem.rule) ?? []), problem]);
}

if (problems.length === 0) {
  console.log(`architectural checks passed across ${files.length} files`);
  console.log('  layering        dependencies point inward');
  console.log('  zero-wan        no cloud SDK, no external URL');
  console.log('  secrets         no credential-shaped literals');
  console.log('  placeholder     no unfinished markers');
  console.log('  erasable-syntax runs on the type-stripping loader');
  process.exit(0);
}

for (const [rule, list] of byRule) {
  console.error(`\n${rule}  (${list.length})`);
  for (const problem of list) {
    console.error(`  ${problem.file}:${problem.line}  ${problem.detail}`);
  }
}
console.error(`\n${problems.length} problem(s) across ${files.length} files`);
process.exit(1);
