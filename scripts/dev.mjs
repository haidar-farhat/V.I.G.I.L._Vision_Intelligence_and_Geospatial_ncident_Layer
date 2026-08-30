#!/usr/bin/env node
/**
 * One-command development entry point.
 *
 * Identical on Windows, Linux and macOS: everything routes through Node rather
 * than a shell script, so there is exactly one set of instructions for the whole
 * team and no `.sh` / `.ps1` pair to drift apart.
 *
 *   node scripts/dev.mjs <command>
 *
 * Also reachable as `npm run <command>` for the common ones.
 */

import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import process from 'node:process';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

/** Glob set covering every test in the workspace. */
const TEST_GLOBS = [
  'packages/*/test/**/*.test.ts',
  'services/*/test/**/*.test.ts',
  'simulator/test/**/*.test.ts',
];

const run = (command, args, options = {}) =>
  new Promise((resolve) => {
    const child = spawn(command, args, {
      cwd: ROOT,
      stdio: 'inherit',
      // Required on Windows, where npm/npx are batch shims rather than binaries.
      shell: process.platform === 'win32',
      ...options,
    });
    child.on('close', (code) => resolve(code ?? 1));
    child.on('error', (error) => {
      process.stderr.write(`\n  cannot run ${command}: ${error.message}\n`);
      resolve(127);
    });
  });

const node = (args, options) => run(process.execPath, args, options);

const HELP = `
Sentinel Vision - development commands

  up            start the local stack (api + worker + simulator) in standalone mode
  down          stop anything started by 'up'
  test          run the full test suite
  test:watch    re-run tests on change
  lint          static checks: layering rules, banned imports, secret scanning
  typecheck     strict TypeScript check across the workspace
  build         typecheck, then build the desktop bundle
  simulator     run the camera simulator on its own
  slice         run the end-to-end vertical slice and print the incident it produces
  demo          seed and launch the scripted demonstration scenario
  db <cmd>      migrate | rollback | status

Every command works with the network cable unplugged.
`;

const commands = {
  async test() {
    return node(['--test', ...TEST_GLOBS]);
  },

  async 'test:watch'() {
    return node(['--test', '--watch', ...TEST_GLOBS]);
  },

  async typecheck() {
    return run('npx', ['tsc', '--noEmit', '-p', 'tsconfig.json']);
  },

  async lint() {
    return node(['scripts/lint.mjs']);
  },

  async build() {
    const typecheck = await commands.typecheck();
    if (typecheck !== 0) return typecheck;
    return run('npm', ['run', 'build', '--workspace', 'apps/desktop']);
  },

  async up() {
    return node(['scripts/stack.mjs', 'up']);
  },

  async down() {
    return node(['scripts/stack.mjs', 'down']);
  },

  async simulator() {
    return node(['simulator/src/cli.ts', ...process.argv.slice(3)]);
  },

  async slice() {
    return node(['scripts/slice.mjs', ...process.argv.slice(3)]);
  },

  async demo() {
    return node(['simulator/src/cli.ts', 'demo', ...process.argv.slice(3)]);
  },

  async db() {
    return node(['packages/database/src/cli.ts', ...process.argv.slice(3)]);
  },
};

const requested = process.argv[2];

if (requested === undefined || requested === 'help' || requested === '--help') {
  process.stdout.write(HELP);
  process.exit(0);
}

const command = commands[requested];
if (command === undefined) {
  process.stderr.write(`unknown command: ${requested}\n${HELP}`);
  process.exit(2);
}

process.exit(await command());
