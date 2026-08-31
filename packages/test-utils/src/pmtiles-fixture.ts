/**
 * Builds real PMTiles v3 archives for tests.
 *
 * The alternative is stubbing the reader, which would test the reader against
 * itself. A binary format is exactly the case where that fails: an off-by-one in
 * a field offset produces a parser that agrees perfectly with a mock built from
 * the same misunderstanding, and disagrees with every real file.
 *
 * So this writes actual bytes at actual offsets, and can be told to write them
 * wrongly - a bad magic, a wrong version, an offset past the end of the file - so
 * the rejection paths are exercised on genuinely malformed archives.
 */

export const PMTILES_HEADER_BYTES = 127;

export type PmtilesFixtureOptions = {
  readonly magic?: string;
  readonly version?: number;
  /** 1 = MVT vector, 2 = PNG, 3 = JPEG, 4 = WebP, 5 = AVIF, 0 = unknown. */
  readonly tileType?: number;
  /** 1 = none, 2 = gzip, 3 = brotli, 4 = zstd. */
  readonly tileCompression?: number;
  readonly internalCompression?: number;
  readonly minZoom?: number;
  readonly maxZoom?: number;
  readonly minLat?: number;
  readonly minLon?: number;
  readonly maxLat?: number;
  readonly maxLon?: number;
  readonly addressedTiles?: number;
  readonly tileEntries?: number;
  /** JSON metadata blob. Written uncompressed so the reader can parse it. */
  readonly metadata?: Record<string, unknown>;
  /** Declare a tile-data length that runs past the end of the file. */
  readonly overrunTileData?: boolean;
  /** Declare a metadata region past the end of the file. */
  readonly overrunMetadata?: boolean;
  /** Truncate the file below a full header. */
  readonly truncate?: boolean;
  /** Bytes of tile payload to append. */
  readonly tileDataBytes?: number;
};

const writeUint64 = (view: DataView, offset: number, value: number): void => {
  view.setBigUint64(offset, BigInt(value), true);
};

/** Degrees times 10^7, signed 32-bit, as the format stores coordinates. */
const writeE7 = (view: DataView, offset: number, degrees: number): void => {
  view.setInt32(offset, Math.round(degrees * 1e7), true);
};

export const buildPmtiles = (options: PmtilesFixtureOptions = {}): Uint8Array => {
  const metadataJson = JSON.stringify(
    options.metadata ?? {
      name: 'Test Region',
      attribution: 'Test data',
      version: '1.0',
      vector_layers: [{ id: 'roads' }, { id: 'buildings' }, { id: 'water' }],
    },
  );
  const metadataBytes = new TextEncoder().encode(metadataJson);
  const tileDataBytes = options.tileDataBytes ?? 512;

  // Layout: header, then metadata, then a stand-in root directory, then tiles.
  const metadataOffset = PMTILES_HEADER_BYTES;
  const rootDirectoryOffset = metadataOffset + metadataBytes.length;
  const rootDirectoryLength = 16;
  const tileDataOffset = rootDirectoryOffset + rootDirectoryLength;
  const totalLength = tileDataOffset + tileDataBytes;

  const bytes = new Uint8Array(totalLength);
  const view = new DataView(bytes.buffer);

  const magic = options.magic ?? 'PMTiles';
  bytes.set(new TextEncoder().encode(magic).subarray(0, 7), 0);
  view.setUint8(7, options.version ?? 3);

  writeUint64(view, 8, rootDirectoryOffset);
  writeUint64(view, 16, rootDirectoryLength);
  writeUint64(view, 24, metadataOffset);
  writeUint64(
    view,
    32,
    options.overrunMetadata === true ? totalLength + 1000 : metadataBytes.length,
  );
  writeUint64(view, 40, 0); // leaf directory offset
  writeUint64(view, 48, 0); // leaf directory length
  writeUint64(view, 56, tileDataOffset);
  writeUint64(view, 64, options.overrunTileData === true ? totalLength + 5000 : tileDataBytes);
  writeUint64(view, 72, options.addressedTiles ?? 4096);
  writeUint64(view, 80, options.tileEntries ?? 4096);
  writeUint64(view, 88, options.tileEntries ?? 4096);

  view.setUint8(96, 1); // clustered
  view.setUint8(97, options.internalCompression ?? 1); // none, so metadata is readable
  view.setUint8(98, options.tileCompression ?? 2); // gzip, as real archives use
  view.setUint8(99, options.tileType ?? 1); // MVT
  view.setUint8(100, options.minZoom ?? 0);
  view.setUint8(101, options.maxZoom ?? 16);

  writeE7(view, 102, options.minLon ?? 35.48);
  writeE7(view, 106, options.minLat ?? 33.87);
  writeE7(view, 110, options.maxLon ?? 35.53);
  writeE7(view, 114, options.maxLat ?? 33.92);

  view.setUint8(118, 14); // center zoom
  writeE7(view, 119, 35.5018);
  writeE7(view, 123, 33.8938);

  bytes.set(metadataBytes, metadataOffset);

  // Tile payload: arbitrary but non-zero, so a hash is meaningful.
  for (let i = 0; i < tileDataBytes; i += 1) {
    bytes[tileDataOffset + i] = (i * 31) % 251;
  }

  return options.truncate === true ? bytes.subarray(0, 40) : bytes;
};

/** A MapLibre style referencing only files inside its own package. */
export const localStyleJson = (overrides: Record<string, unknown> = {}): string =>
  JSON.stringify({
    version: 8,
    name: 'Sentinel Offline',
    glyphs: 'glyphs/{fontstack}/{range}.pbf',
    sprite: 'sprite',
    sources: {
      basemap: { type: 'vector', url: 'pmtiles://region.pmtiles' },
    },
    layers: [
      { id: 'background', type: 'background', paint: { 'background-color': '#0d1117' } },
      { id: 'water', type: 'fill', source: 'basemap', 'source-layer': 'water' },
      { id: 'roads', type: 'line', source: 'basemap', 'source-layer': 'roads' },
      { id: 'buildings', type: 'fill', source: 'basemap', 'source-layer': 'buildings' },
    ],
    ...overrides,
  });

/** The files a well-formed package directory contains. */
export const PACKAGE_FILES: readonly string[] = Object.freeze([
  'region.pmtiles',
  'style.json',
  'metadata.json',
  'sprite.json',
  'sprite.png',
  'glyphs/Noto Sans Regular/0-255.pbf',
]);
