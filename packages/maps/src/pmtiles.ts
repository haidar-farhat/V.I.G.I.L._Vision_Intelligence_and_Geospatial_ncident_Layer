import type { GeoBounds } from '@sentinel/shared-types';

/**
 * PMTiles v3 header reading.
 *
 * PMTiles is a single-file tile archive: a fixed 127-byte header, a JSON metadata
 * blob, a directory structure, and the tile data. It is the format this platform
 * uses because it is the only mainstream one that works from a local file with no
 * server, which is the entire requirement.
 *
 * This reads and validates the header and metadata. It does **not** traverse the
 * tile directories to fetch individual tiles - MapLibre does that in the renderer,
 * from the same file. What the application needs here is the answer to "is this
 * archive usable, and what does it cover?", asked once at import time.
 *
 * The archive comes from an operator's USB stick. It is untrusted input: every
 * offset is checked against the actual file length before anything is read, and a
 * declared length that runs past the end of the file is a rejection rather than a
 * read of whatever happens to be adjacent in memory.
 */

export const PMTILES_MAGIC = 'PMTiles';
export const PMTILES_HEADER_BYTES = 127;
export const PMTILES_VERSION = 3;

/** Tile payload formats. Only vector and raster are renderable here. */
export const TileType = {
  Unknown: 0,
  /** Mapbox Vector Tile. What a vector basemap ships as. */
  Mvt: 1,
  Png: 2,
  Jpeg: 3,
  Webp: 4,
  Avif: 5,
} as const;
export type TileType = (typeof TileType)[keyof typeof TileType];

export const Compression = {
  Unknown: 0,
  None: 1,
  Gzip: 2,
  Brotli: 3,
  Zstd: 4,
} as const;
export type Compression = (typeof Compression)[keyof typeof Compression];

export const tileTypeName = (type: number): string => {
  switch (type) {
    case TileType.Mvt:
      return 'vector (MVT)';
    case TileType.Png:
      return 'raster (PNG)';
    case TileType.Jpeg:
      return 'raster (JPEG)';
    case TileType.Webp:
      return 'raster (WebP)';
    case TileType.Avif:
      return 'raster (AVIF)';
    default:
      return 'unknown';
  }
};

export const compressionName = (compression: number): string => {
  switch (compression) {
    case Compression.None:
      return 'none';
    case Compression.Gzip:
      return 'gzip';
    case Compression.Brotli:
      return 'brotli';
    case Compression.Zstd:
      return 'zstd';
    default:
      return 'unknown';
  }
};

export type PmtilesHeader = {
  readonly version: number;
  readonly rootDirectoryOffset: number;
  readonly rootDirectoryLength: number;
  readonly metadataOffset: number;
  readonly metadataLength: number;
  readonly leafDirectoryOffset: number;
  readonly leafDirectoryLength: number;
  readonly tileDataOffset: number;
  readonly tileDataLength: number;
  readonly addressedTiles: number;
  readonly tileEntries: number;
  readonly tileContents: number;
  readonly clustered: boolean;
  readonly internalCompression: Compression;
  readonly tileCompression: Compression;
  readonly tileType: TileType;
  readonly minZoom: number;
  readonly maxZoom: number;
  readonly bounds: GeoBounds;
  readonly centerZoom: number;
  readonly centerLon: number;
  readonly centerLat: number;
};

export class PmtilesError extends Error {
  readonly code: string;
  readonly recoverable = false;

  constructor(message: string, code: string) {
    super(message);
    this.name = 'PmtilesError';
    this.code = code;
  }
}

/**
 * Read a uint64 as a JavaScript number.
 *
 * PMTiles offsets are 64-bit. Above 2^53 a double cannot represent them exactly,
 * and a silently-wrong offset would read the wrong bytes rather than fail. An
 * archive that large is not something an operator carries on a stick, so the
 * limit is enforced rather than papered over.
 */
const readUint64 = (view: DataView, offset: number, field: string): number => {
  const value = view.getBigUint64(offset, true);
  if (value > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new PmtilesError(
      `${field} is ${value}, beyond the range this reader can address exactly.`,
      'PMTILES_OFFSET_TOO_LARGE',
    );
  }
  return Number(value);
};

/** Coordinates are stored as degrees times 10^7, as signed 32-bit integers. */
const readE7 = (view: DataView, offset: number): number => view.getInt32(offset, true) / 1e7;

/**
 * Parse the header.
 *
 * `fileLength` is required so every declared offset can be checked against the
 * file that actually exists rather than the one the header claims.
 */
export const readPmtilesHeader = (bytes: Uint8Array, fileLength: number): PmtilesHeader => {
  if (bytes.length < PMTILES_HEADER_BYTES) {
    throw new PmtilesError(
      `The file is ${bytes.length} bytes, shorter than a ${PMTILES_HEADER_BYTES}-byte PMTiles header.`,
      'PMTILES_TRUNCATED',
    );
  }

  const magic = new TextDecoder().decode(bytes.subarray(0, 7));
  if (magic !== PMTILES_MAGIC) {
    throw new PmtilesError(
      `This is not a PMTiles archive: it begins with "${magic.replace(/[^\x20-\x7e]/g, '.')}" ` +
        `rather than "${PMTILES_MAGIC}".`,
      'PMTILES_BAD_MAGIC',
    );
  }

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const version = view.getUint8(7);

  if (version !== PMTILES_VERSION) {
    throw new PmtilesError(
      `This archive is PMTiles version ${version}; this build reads version ${PMTILES_VERSION}. ` +
        'Re-export the region with a current tool.',
      'PMTILES_BAD_VERSION',
    );
  }

  const header: PmtilesHeader = {
    version,
    rootDirectoryOffset: readUint64(view, 8, 'root directory offset'),
    rootDirectoryLength: readUint64(view, 16, 'root directory length'),
    metadataOffset: readUint64(view, 24, 'metadata offset'),
    metadataLength: readUint64(view, 32, 'metadata length'),
    leafDirectoryOffset: readUint64(view, 40, 'leaf directory offset'),
    leafDirectoryLength: readUint64(view, 48, 'leaf directory length'),
    tileDataOffset: readUint64(view, 56, 'tile data offset'),
    tileDataLength: readUint64(view, 64, 'tile data length'),
    addressedTiles: readUint64(view, 72, 'addressed tile count'),
    tileEntries: readUint64(view, 80, 'tile entry count'),
    tileContents: readUint64(view, 88, 'tile content count'),
    clustered: view.getUint8(96) === 1,
    internalCompression: view.getUint8(97) as Compression,
    tileCompression: view.getUint8(98) as Compression,
    tileType: view.getUint8(99) as TileType,
    minZoom: view.getUint8(100),
    maxZoom: view.getUint8(101),
    bounds: {
      minLon: readE7(view, 102),
      minLat: readE7(view, 106),
      maxLon: readE7(view, 110),
      maxLat: readE7(view, 114),
    },
    centerZoom: view.getUint8(118),
    centerLon: readE7(view, 119),
    centerLat: readE7(view, 123),
  };

  assertRangesWithinFile(header, fileLength);
  return header;
};

/**
 * Every declared region must lie inside the file.
 *
 * A truncated download and a deliberately crafted header look identical from the
 * inside; both are refused here rather than at read time, when a length that
 * overruns the file would otherwise be a read of adjacent memory.
 */
const assertRangesWithinFile = (header: PmtilesHeader, fileLength: number): void => {
  const ranges: readonly [string, number, number][] = [
    ['root directory', header.rootDirectoryOffset, header.rootDirectoryLength],
    ['metadata', header.metadataOffset, header.metadataLength],
    ['leaf directories', header.leafDirectoryOffset, header.leafDirectoryLength],
    ['tile data', header.tileDataOffset, header.tileDataLength],
  ];

  for (const [name, offset, length] of ranges) {
    if (offset < 0 || length < 0) {
      throw new PmtilesError(`The ${name} region has a negative offset or length.`, 'PMTILES_BAD_RANGE');
    }
    if (offset + length > fileLength) {
      throw new PmtilesError(
        `The ${name} region ends at byte ${offset + length}, past the end of a ${fileLength}-byte ` +
          'file. The archive is truncated or corrupt.',
        'PMTILES_BAD_RANGE',
      );
    }
  }
};

/** Metadata a tile archive carries about itself. All fields are optional in practice. */
export type PmtilesMetadata = {
  readonly name?: string;
  readonly description?: string;
  readonly attribution?: string;
  readonly version?: string;
  /** Layer names present in a vector archive, used to check a style against it. */
  readonly vectorLayers: readonly string[];
  /** Everything else, retained but not interpreted. */
  readonly raw: Readonly<Record<string, unknown>>;
};

/**
 * Read the JSON metadata blob.
 *
 * Returns empty metadata rather than throwing when the blob is absent, unreadable
 * or compressed with something this build cannot decompress. Metadata is
 * descriptive: an archive whose tiles are fine but whose metadata is malformed is
 * still a usable map, and refusing it would be a worse outcome than showing an
 * unnamed region.
 */
export const readPmtilesMetadata = (
  bytes: Uint8Array,
  header: PmtilesHeader,
): PmtilesMetadata => {
  const empty: PmtilesMetadata = { vectorLayers: [], raw: {} };

  if (header.metadataLength === 0) return empty;
  if (header.internalCompression !== Compression.None) {
    // Gzip and friends are handled by the renderer, which has the archive open
    // anyway. Import does not need the metadata badly enough to justify pulling a
    // decompressor into this layer.
    return empty;
  }

  const slice = bytes.subarray(header.metadataOffset, header.metadataOffset + header.metadataLength);

  let parsed: unknown;
  try {
    parsed = JSON.parse(new TextDecoder().decode(slice));
  } catch {
    return empty;
  }

  if (typeof parsed !== 'object' || parsed === null) return empty;
  const raw = parsed as Record<string, unknown>;

  const vectorLayers: string[] = [];
  const layers = raw['vector_layers'];
  if (Array.isArray(layers)) {
    for (const layer of layers) {
      if (typeof layer === 'object' && layer !== null) {
        const id = (layer as Record<string, unknown>)['id'];
        if (typeof id === 'string') vectorLayers.push(id);
      }
    }
  }

  return {
    vectorLayers,
    raw,
    ...(typeof raw['name'] === 'string' ? { name: raw['name'] } : {}),
    ...(typeof raw['description'] === 'string' ? { description: raw['description'] } : {}),
    ...(typeof raw['attribution'] === 'string' ? { attribution: raw['attribution'] } : {}),
    ...(typeof raw['version'] === 'string' ? { version: raw['version'] } : {}),
  };
};

/** Whether a point falls inside the archive's declared coverage. */
export const boundsCover = (bounds: GeoBounds, lat: number, lon: number): boolean =>
  lat >= bounds.minLat && lat <= bounds.maxLat && lon >= bounds.minLon && lon <= bounds.maxLon;

/**
 * Approximate ground area of the covered region, in square kilometres.
 *
 * Shown at import so an operator can tell a city-sized package from a
 * country-sized one before committing the disk space.
 */
export const boundsAreaSquareKm = (bounds: GeoBounds): number => {
  const meanLat = ((bounds.minLat + bounds.maxLat) / 2) * (Math.PI / 180);
  const latKm = (bounds.maxLat - bounds.minLat) * 110.574;
  const lonKm = (bounds.maxLon - bounds.minLon) * 111.32 * Math.cos(meanLat);
  return Math.abs(latKm * lonKm);
};
