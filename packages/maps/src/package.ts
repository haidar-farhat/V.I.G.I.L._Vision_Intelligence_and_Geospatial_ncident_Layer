import { createHash } from 'node:crypto';
import type { GeoBounds, MapPackageId, UtcMillis } from '@sentinel/shared-types';
import { asId } from '@sentinel/shared-types';
import type { PmtilesHeader, PmtilesMetadata } from './pmtiles.ts';
import {
  PmtilesError,
  TileType,
  boundsAreaSquareKm,
  compressionName,
  readPmtilesHeader,
  readPmtilesMetadata,
  tileTypeName,
} from './pmtiles.ts';
import type { StyleValidation } from './style.ts';
import { validateStyle } from './style.ts';

/**
 * Offline map package import.
 *
 * A package is a directory an operator brings on a USB stick:
 *
 *   region.pmtiles   tile archive
 *   style.json       MapLibre style, referencing only files in this package
 *   metadata.json    region, version, provenance (optional)
 *
 * Import is the only way map data enters the system. Nothing is ever downloaded,
 * and there is no code path that could - which means import validation is the
 * single point where a bad package can be caught. It is therefore thorough and
 * loud rather than permissive.
 *
 * The distinction that matters throughout: an **error** means the package will
 * not work and is refused; a **warning** means it will work but an operator
 * should know something. Refusing a usable map over a cosmetic defect strands a
 * site with no map at all, which is worse than the defect.
 */

export type MapPackageInput = {
  readonly name: string;
  /** Raw bytes of the .pmtiles archive. */
  readonly tiles: Uint8Array;
  /** Contents of style.json. */
  readonly style: string;
  /** Contents of metadata.json, if present. */
  readonly metadata?: string;
  /** File names the package directory contains, for resolving style references. */
  readonly files: readonly string[];
  /** Path the archive will be stored at, relative to the map-data root. */
  readonly relativePath: string;
  readonly importedAt: UtcMillis;
};

export type MapPackageManifest = {
  readonly id: MapPackageId;
  readonly name: string;
  readonly region: string;
  readonly bounds: GeoBounds;
  readonly minZoom: number;
  readonly maxZoom: number;
  readonly tileType: string;
  readonly tileCompression: string;
  readonly sizeBytes: number;
  readonly areaSquareKm: number;
  readonly tileCount: number;
  readonly sha256: string;
  readonly relativePath: string;
  readonly importedAt: UtcMillis;
  readonly attribution: string | null;
  readonly version: string | null;
};

export type MapPackageValidation = {
  readonly valid: boolean;
  readonly manifest: MapPackageManifest | null;
  readonly header: PmtilesHeader | null;
  readonly tileMetadata: PmtilesMetadata | null;
  readonly style: StyleValidation | null;
  readonly errors: readonly string[];
  readonly warnings: readonly string[];
};

/** Renderable tile types. An archive of anything else cannot be drawn. */
const RENDERABLE = new Set<number>([
  TileType.Mvt,
  TileType.Png,
  TileType.Jpeg,
  TileType.Webp,
]);

export const sha256Of = (bytes: Uint8Array): string =>
  createHash('sha256').update(bytes).digest('hex');

/**
 * Derive a stable package id from the archive's content hash.
 *
 * Importing the same region twice produces the same id, so a re-import replaces
 * rather than duplicating - and a package can be recognised as the same one after
 * being copied between machines, which is how a multi-site deployment keeps its
 * map data consistent.
 */
export const derivePackageId = (sha256: string): MapPackageId =>
  asId<MapPackageId>(`map-${sha256.slice(0, 16)}`);

export const validateMapPackage = (input: MapPackageInput): MapPackageValidation => {
  const errors: string[] = [];
  const warnings: string[] = [];

  // ------------------------------------------------------------- tile archive
  let header: PmtilesHeader;
  try {
    header = readPmtilesHeader(input.tiles, input.tiles.length);
  } catch (error) {
    return {
      valid: false,
      manifest: null,
      header: null,
      tileMetadata: null,
      style: null,
      errors: [error instanceof PmtilesError ? error.message : String(error)],
      warnings: [],
    };
  }

  const tileMetadata = readPmtilesMetadata(input.tiles, header);

  if (!RENDERABLE.has(header.tileType)) {
    errors.push(
      `The archive contains ${tileTypeName(header.tileType)} tiles, which this build cannot render.`,
    );
  }

  if (header.tileEntries === 0 || header.tileDataLength === 0) {
    errors.push('The archive contains no tiles.');
  }

  if (header.maxZoom < header.minZoom) {
    errors.push(
      `The archive declares a zoom range of ${header.minZoom} to ${header.maxZoom}, which is inverted.`,
    );
  }

  // ------------------------------------------------------------------- bounds
  const { bounds } = header;
  const boundsSane =
    bounds.minLat >= -90 &&
    bounds.maxLat <= 90 &&
    bounds.minLon >= -180 &&
    bounds.maxLon <= 180 &&
    bounds.minLat < bounds.maxLat &&
    bounds.minLon < bounds.maxLon;

  if (!boundsSane) {
    errors.push(
      `The archive declares an impossible coverage area ` +
        `(${bounds.minLat}, ${bounds.minLon}) to (${bounds.maxLat}, ${bounds.maxLon}).`,
    );
  }

  // A site camera sits at 20 m per pixel or better. An archive that stops at zoom
  // 10 is a country map and will show a beige rectangle when the operator zooms
  // to the perimeter - a package worth importing, but not worth being surprised by.
  if (boundsSane && header.maxZoom < 14) {
    warnings.push(
      `The archive stops at zoom ${header.maxZoom}. Site-level detail usually needs zoom 16 ` +
        'or better; expect the map to blur when zoomed to a perimeter.',
    );
  }

  if (boundsSane && boundsAreaSquareKm(bounds) > 2_000_000) {
    warnings.push('This package covers a very large area. Check the available disk space.');
  }

  // -------------------------------------------------------------------- style
  const style = validateStyle(input.style, input.files);
  errors.push(...style.errors);
  warnings.push(...style.warnings);

  // A style drawing from a layer the archive does not contain renders nothing for
  // that layer. Cross-checking here is the only place the two halves of a package
  // are ever compared.
  if (tileMetadata.vectorLayers.length > 0 && style.sourceLayers.length > 0) {
    const absent = style.sourceLayers.filter(
      (layer) => !tileMetadata.vectorLayers.includes(layer),
    );
    if (absent.length > 0) {
      warnings.push(
        `The style draws from source layers the archive does not contain: ${absent.join(', ')}. ` +
          'Those layers will render nothing.',
      );
    }
  }

  // ------------------------------------------------------------- storage path
  if (
    input.relativePath.includes('..') ||
    input.relativePath.startsWith('/') ||
    input.relativePath.startsWith('\\') ||
    /^[a-zA-Z]:/.test(input.relativePath)
  ) {
    errors.push(
      `The storage path "${input.relativePath}" is absolute or traverses outside the map ` +
        'data directory.',
    );
  }

  const sha256 = sha256Of(input.tiles);

  const manifest: MapPackageManifest = {
    id: derivePackageId(sha256),
    name: input.name,
    region: tileMetadata.name ?? input.name,
    bounds,
    minZoom: header.minZoom,
    maxZoom: header.maxZoom,
    tileType: tileTypeName(header.tileType),
    tileCompression: compressionName(header.tileCompression),
    sizeBytes: input.tiles.length,
    areaSquareKm: boundsSane ? boundsAreaSquareKm(bounds) : 0,
    tileCount: header.addressedTiles,
    sha256,
    relativePath: input.relativePath,
    importedAt: input.importedAt,
    attribution: tileMetadata.attribution ?? null,
    version: tileMetadata.version ?? null,
  };

  return {
    valid: errors.length === 0,
    manifest: errors.length === 0 ? manifest : null,
    header,
    tileMetadata,
    style,
    errors,
    warnings,
  };
};

/**
 * Verify an installed package still matches what was imported.
 *
 * Run from diagnostics. A silently corrupted archive renders blank tiles rather
 * than failing, so an operator would see an empty map and assume the region was
 * never imported.
 */
export const verifyIntegrity = (
  bytes: Uint8Array,
  expectedSha256: string,
): { readonly intact: boolean; readonly actualSha256: string; readonly detail: string } => {
  const actual = sha256Of(bytes);

  return {
    intact: actual === expectedSha256,
    actualSha256: actual,
    detail:
      actual === expectedSha256
        ? 'The archive matches the hash recorded at import.'
        : 'The archive does not match the hash recorded at import. It has been modified or ' +
          'corrupted since; re-import it from the original media.',
  };
};

/** The message shown when no package is installed. Stated, never worked around. */
export const NO_MAP_DATA_MESSAGE = 'OFFLINE MAP DATA NOT INSTALLED';

/**
 * Which installed package covers a point, preferring the most detailed.
 *
 * A deployment may hold a wide regional package and a detailed site package that
 * overlap. The site one is what an operator wants when looking at a camera.
 */
export const packageForPoint = (
  packages: readonly MapPackageManifest[],
  lat: number,
  lon: number,
): MapPackageManifest | null => {
  const covering = packages.filter(
    (candidate) =>
      lat >= candidate.bounds.minLat &&
      lat <= candidate.bounds.maxLat &&
      lon >= candidate.bounds.minLon &&
      lon <= candidate.bounds.maxLon,
  );

  if (covering.length === 0) return null;

  return covering.reduce((best, candidate) =>
    candidate.maxZoom > best.maxZoom ? candidate : best,
  );
};
