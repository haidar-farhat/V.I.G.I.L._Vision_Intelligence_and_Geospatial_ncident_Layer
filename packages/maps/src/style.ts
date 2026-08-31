import { classifyHost, isLocalScope } from '@sentinel/security';

/**
 * MapLibre style validation.
 *
 * This is the check that catches the failure mode the whole offline architecture
 * exists to prevent: a style that imports and validates cleanly, then reaches for
 * a font server the first time it renders a label. The map appears to work in the
 * lab, where the network is up, and shows unlabelled roads on the isolated site
 * where it matters.
 *
 * So every URL a style can carry is checked - sources, sprites, glyphs and any
 * absolute reference buried in a layer. A style may point at files inside its own
 * package, or at a private address. Anything else is refused.
 */

export type StyleReference = {
  /** Where in the style the URL appeared, e.g. `sources.basemap.url`. */
  readonly path: string;
  readonly url: string;
  readonly kind: 'source' | 'sprite' | 'glyphs' | 'other';
};

export type StyleValidation = {
  readonly valid: boolean;
  readonly name: string | null;
  readonly references: readonly StyleReference[];
  /** References that would require the Internet. Any one of these fails the style. */
  readonly external: readonly StyleReference[];
  /** Local references naming a file the package does not contain. */
  readonly missing: readonly StyleReference[];
  readonly errors: readonly string[];
  readonly warnings: readonly string[];
  /** Vector source layers the style draws from, for checking against the archive. */
  readonly sourceLayers: readonly string[];
};

const MAX_STYLE_BYTES = 4 * 1024 * 1024;

/**
 * Whether a URL is satisfiable without the Internet.
 *
 * `pmtiles://`, `mbtiles://` and relative paths resolve inside the package.
 * `file://` and private-range HTTP are local. Everything else - including a
 * hostname this build cannot prove is local - is external, because resolving it
 * to find out would itself require a DNS server.
 */
export const isLocalStyleUrl = (url: string): boolean => {
  const trimmed = url.trim();
  if (trimmed === '') return true;

  // Style tokens the renderer substitutes; not addresses.
  if (trimmed.startsWith('{') || trimmed.startsWith('mapbox://')) return trimmed.startsWith('{');

  if (/^(pmtiles|mbtiles|file):/i.test(trimmed)) return true;

  // Relative or root-relative: resolved against the package directory.
  if (!/^[a-z][a-z0-9+.-]*:\/\//i.test(trimmed)) return true;

  try {
    return isLocalScope(classifyHost(new URL(trimmed).hostname));
  } catch {
    return false;
  }
};

/** Strip a pmtiles:// wrapper and any template placeholders to get a filename. */
export const styleUrlToLocalPath = (url: string): string | null => {
  const trimmed = url.trim().replace(/^pmtiles:\/\//i, '').replace(/^mbtiles:\/\//i, '');
  if (/^[a-z][a-z0-9+.-]*:\/\//i.test(trimmed)) return null;
  if (trimmed.includes('{')) return null;

  const withoutQuery = trimmed.split(/[?#]/)[0] ?? '';
  return withoutQuery.replace(/^\.?\//, '');
};

/**
 * Validate a style document.
 *
 * `packageFiles` is what the package actually contains; a local reference naming
 * something absent is reported as missing rather than assumed fine. Passing an
 * empty list skips that check, for callers validating a style in isolation.
 */
export const validateStyle = (
  raw: string,
  packageFiles: readonly string[] = [],
): StyleValidation => {
  const errors: string[] = [];
  const warnings: string[] = [];
  const references: StyleReference[] = [];
  const sourceLayers: string[] = [];

  if (raw.length > MAX_STYLE_BYTES) {
    return fail(`The style is larger than ${MAX_STYLE_BYTES} bytes.`);
  }

  let style: Record<string, unknown>;
  try {
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
      return fail('The style is not a JSON object.');
    }
    style = parsed as Record<string, unknown>;
  } catch (error) {
    return fail(`The style is not valid JSON: ${error instanceof Error ? error.message : 'parse failed'}`);
  }

  if (style['version'] !== 8) {
    errors.push(`The style declares version ${String(style['version'])}; MapLibre requires version 8.`);
  }

  const name = typeof style['name'] === 'string' ? style['name'] : null;

  // ------------------------------------------------------------------ sources
  const sources = style['sources'];
  if (typeof sources !== 'object' || sources === null) {
    errors.push('The style declares no sources, so it would render an empty map.');
  } else {
    for (const [id, value] of Object.entries(sources as Record<string, unknown>)) {
      if (typeof value !== 'object' || value === null) continue;
      const source = value as Record<string, unknown>;

      if (typeof source['url'] === 'string') {
        references.push({ path: `sources.${id}.url`, url: source['url'], kind: 'source' });
      }

      const tiles = source['tiles'];
      if (Array.isArray(tiles)) {
        tiles.forEach((tile, index) => {
          if (typeof tile === 'string') {
            references.push({ path: `sources.${id}.tiles[${index}]`, url: tile, kind: 'source' });
          }
        });
      }

      if (source['url'] === undefined && tiles === undefined && source['data'] === undefined) {
        warnings.push(`Source "${id}" names no url, tiles or data and will render nothing.`);
      }

      if (typeof source['data'] === 'string') {
        references.push({ path: `sources.${id}.data`, url: source['data'], kind: 'source' });
      }
    }
  }

  // -------------------------------------------------------- sprites and glyphs
  if (typeof style['sprite'] === 'string') {
    references.push({ path: 'sprite', url: style['sprite'], kind: 'sprite' });
  }
  if (typeof style['glyphs'] === 'string') {
    references.push({ path: 'glyphs', url: style['glyphs'], kind: 'glyphs' });
  } else {
    // Without glyphs a style renders no text at all. Worth saying plainly,
    // because "the map has no labels" is otherwise a mystifying symptom.
    warnings.push(
      'The style names no glyph source, so no text will be drawn. Include a local glyph ' +
        'pack if labels are wanted.',
    );
  }

  // ------------------------------------------------------------------- layers
  const layers = style['layers'];
  if (!Array.isArray(layers) || layers.length === 0) {
    errors.push('The style declares no layers, so it would render an empty map.');
  } else {
    const sourceIds = new Set(
      typeof sources === 'object' && sources !== null ? Object.keys(sources) : [],
    );

    for (const [index, value] of layers.entries()) {
      if (typeof value !== 'object' || value === null) continue;
      const layer = value as Record<string, unknown>;

      const layerSource = layer['source'];
      if (typeof layerSource === 'string' && !sourceIds.has(layerSource)) {
        errors.push(
          `Layer ${index} ("${String(layer['id'] ?? index)}") draws from source ` +
            `"${layerSource}", which the style does not define.`,
        );
      }

      const sourceLayer = layer['source-layer'];
      if (typeof sourceLayer === 'string' && !sourceLayers.includes(sourceLayer)) {
        sourceLayers.push(sourceLayer);
      }
    }
  }

  // Catch any absolute URL anywhere else in the document - a layer property, an
  // expression, a vendor extension. A style is a large open format and an
  // Internet dependency can hide in a corner nothing else inspects.
  for (const found of raw.matchAll(/"(https?:\/\/[^"]+)"/g)) {
    const url = found[1];
    if (url === undefined) continue;
    if (references.some((reference) => reference.url === url)) continue;
    references.push({ path: 'embedded', url, kind: 'other' });
  }

  const external = references.filter((reference) => !isLocalStyleUrl(reference.url));

  const missing =
    packageFiles.length === 0
      ? []
      : references.filter((reference) => {
          if (!isLocalStyleUrl(reference.url)) return false;
          const path = styleUrlToLocalPath(reference.url);
          if (path === null || path === '') return false;

          // `sprite` is a base name, not a file. MapLibre appends the extensions
          // and pixel-ratio suffixes itself, fetching sprite.json, sprite.png and
          // optionally sprite@2x variants. Checking the base name literally
          // reports every correctly-built package as missing its sprite.
          if (reference.kind === 'sprite') {
            return !packageFiles.some(
              (file) => file === `${path}.json` || file === `${path}.png`,
            );
          }

          return !packageFiles.includes(path);
        });

  for (const reference of external) {
    errors.push(
      `${reference.path} points at "${reference.url}", which requires the Internet. ` +
        'Every style reference must resolve inside the package or on the local network.',
    );
  }
  for (const reference of missing) {
    errors.push(
      `${reference.path} points at "${reference.url}", which the package does not contain.`,
    );
  }

  return {
    valid: errors.length === 0,
    name,
    references,
    external,
    missing,
    errors,
    warnings,
    sourceLayers,
  };

  function fail(message: string): StyleValidation {
    return {
      valid: false,
      name: null,
      references: [],
      external: [],
      missing: [],
      errors: [message],
      warnings: [],
      sourceLayers: [],
    };
  }
};
