import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { utcMillis } from '@sentinel/shared-types';
import { PACKAGE_FILES, buildPmtiles, localStyleJson } from '@sentinel/test-utils';
import {
  Compression,
  PmtilesError,
  TileType,
  boundsAreaSquareKm,
  boundsCover,
  readPmtilesHeader,
  readPmtilesMetadata,
} from '../src/pmtiles.ts';
import { isLocalStyleUrl, styleUrlToLocalPath, validateStyle } from '../src/style.ts';
import {
  NO_MAP_DATA_MESSAGE,
  derivePackageId,
  packageForPoint,
  sha256Of,
  validateMapPackage,
  verifyIntegrity,
} from '../src/package.ts';
import type { MapPackageManifest } from '../src/package.ts';

const importPackage = (options: Parameters<typeof buildPmtiles>[0] = {}, style?: string) =>
  validateMapPackage({
    name: 'Beirut Site',
    tiles: buildPmtiles(options),
    style: style ?? localStyleJson(),
    files: PACKAGE_FILES,
    relativePath: 'beirut/region.pmtiles',
    importedAt: utcMillis(1_718_158_440_000),
  });

// -------------------------------------------------------------------- pmtiles

describe('PMTiles header', () => {
  test('reads a well-formed archive', () => {
    const bytes = buildPmtiles();
    const header = readPmtilesHeader(bytes, bytes.length);

    assert.equal(header.version, 3);
    assert.equal(header.tileType, TileType.Mvt);
    assert.equal(header.tileCompression, Compression.Gzip);
    assert.equal(header.minZoom, 0);
    assert.equal(header.maxZoom, 16);
    assert.equal(header.clustered, true);
    assert.equal(header.tileEntries, 4096);
  });

  test('decodes coordinates from their E7 integer encoding', () => {
    const bytes = buildPmtiles({ minLat: 33.87, minLon: 35.48, maxLat: 33.92, maxLon: 35.53 });
    const header = readPmtilesHeader(bytes, bytes.length);

    // Round-trip through int32 tenths-of-a-microdegree is exact to 7 places.
    assert.ok(Math.abs(header.bounds.minLat - 33.87) < 1e-7);
    assert.ok(Math.abs(header.bounds.maxLon - 35.53) < 1e-7);
  });

  test('rejects a file that is not PMTiles', () => {
    const notPmtiles = new TextEncoder().encode('#!/bin/sh\necho hello\n'.padEnd(200, ' '));

    assert.throws(
      () => readPmtilesHeader(notPmtiles, notPmtiles.length),
      (error: unknown) => {
        assert.ok(error instanceof PmtilesError);
        assert.equal(error.code, 'PMTILES_BAD_MAGIC');
        // The message quotes what it found, so a mis-picked file is obvious.
        assert.match(error.message, /begins with/);
        return true;
      },
    );
  });

  test('rejects an archive from a future format version', () => {
    const bytes = buildPmtiles({ version: 4 });

    assert.throws(
      () => readPmtilesHeader(bytes, bytes.length),
      (error: unknown) => {
        assert.ok(error instanceof PmtilesError);
        assert.equal(error.code, 'PMTILES_BAD_VERSION');
        assert.match(error.message, /Re-export the region/);
        return true;
      },
    );
  });

  test('rejects a truncated file rather than reading past its end', () => {
    const bytes = buildPmtiles({ truncate: true });

    assert.throws(
      () => readPmtilesHeader(bytes, bytes.length),
      (error: unknown) => {
        assert.ok(error instanceof PmtilesError);
        assert.equal(error.code, 'PMTILES_TRUNCATED');
        return true;
      },
    );
  });

  test('rejects a declared region that runs past the end of the file', () => {
    // A truncated download and a crafted header look identical from the inside.
    // Both are refused before any read, not at read time.
    for (const options of [{ overrunTileData: true }, { overrunMetadata: true }]) {
      const bytes = buildPmtiles(options);
      assert.throws(
        () => readPmtilesHeader(bytes, bytes.length),
        (error: unknown) => {
          assert.ok(error instanceof PmtilesError);
          assert.equal(error.code, 'PMTILES_BAD_RANGE');
          assert.match(error.message, /truncated or corrupt/);
          return true;
        },
      );
    }
  });

  test('reads the metadata blob and its vector layers', () => {
    const bytes = buildPmtiles();
    const header = readPmtilesHeader(bytes, bytes.length);
    const metadata = readPmtilesMetadata(bytes, header);

    assert.equal(metadata.name, 'Test Region');
    assert.equal(metadata.version, '1.0');
    assert.deepEqual(metadata.vectorLayers, ['roads', 'buildings', 'water']);
  });

  test('unreadable metadata degrades rather than failing the archive', () => {
    // An archive whose tiles are fine but whose metadata is malformed is still a
    // usable map. Refusing it would be worse than showing an unnamed region.
    const bytes = buildPmtiles({ metadata: {} });
    const header = readPmtilesHeader(bytes, bytes.length);

    const withBadCompression = readPmtilesMetadata(bytes, {
      ...header,
      internalCompression: Compression.Gzip,
    });
    assert.deepEqual(withBadCompression.vectorLayers, []);
    assert.equal(withBadCompression.name, undefined);
  });

  test('reports coverage and area', () => {
    const bytes = buildPmtiles();
    const header = readPmtilesHeader(bytes, bytes.length);

    assert.ok(boundsCover(header.bounds, 33.8938, 35.5018), 'the site is inside the region');
    assert.ok(!boundsCover(header.bounds, 51.5, -0.12), 'London is not');

    const area = boundsAreaSquareKm(header.bounds);
    assert.ok(area > 5 && area < 100, `a 0.05-degree box should be tens of km2, got ${area}`);
  });
});

// ---------------------------------------------------------------------- style

describe('style URL classification', () => {
  test('package-relative and local references are local', () => {
    for (const url of [
      'pmtiles://region.pmtiles',
      'mbtiles://region.mbtiles',
      'glyphs/{fontstack}/{range}.pbf',
      './sprite',
      '/sprite.png',
      'file:///data/maps/region.pmtiles',
      'http://127.0.0.1:8787/tiles',
      'http://192.168.1.10/tiles',
      'http://tiles.local/style.json',
    ]) {
      assert.ok(isLocalStyleUrl(url), `${url} should be local`);
    }
  });

  test('anything requiring the Internet is not', () => {
    for (const url of [
      'https://api.mapbox.com/styles/v1/mapbox/streets-v11',
      'https://fonts.googleapis.com/css?family=Roboto',
      'https://tiles.example.com/{z}/{x}/{y}.pbf',
      'https://demotiles.maplibre.org/style.json',
      'http://8.8.8.8/tiles',
    ]) {
      assert.ok(!isLocalStyleUrl(url), `${url} should be refused`);
    }
  });

  test('resolves a local reference to a package file name', () => {
    assert.equal(styleUrlToLocalPath('pmtiles://region.pmtiles'), 'region.pmtiles');
    assert.equal(styleUrlToLocalPath('./sprite.png'), 'sprite.png');
    assert.equal(styleUrlToLocalPath('/sprite.png'), 'sprite.png');
    assert.equal(styleUrlToLocalPath('sprite.png?v=2'), 'sprite.png');
    // Templated URLs expand at render time and cannot be checked against a listing.
    assert.equal(styleUrlToLocalPath('glyphs/{fontstack}/{range}.pbf'), null);
    assert.equal(styleUrlToLocalPath('https://example.com/x'), null);
  });
});

describe('style validation', () => {
  test('accepts a fully local style', () => {
    const result = validateStyle(localStyleJson(), PACKAGE_FILES);

    assert.equal(result.valid, true, result.errors.join('; '));
    assert.equal(result.name, 'Sentinel Offline');
    assert.deepEqual(result.external, []);
    assert.deepEqual([...result.sourceLayers].sort(), ['buildings', 'roads', 'water']);
  });

  test('refuses a style that would fetch tiles from the Internet', () => {
    // The failure the whole offline architecture exists to prevent.
    const style = localStyleJson({
      sources: { basemap: { type: 'vector', url: 'https://tiles.example.com/tiles.json' } },
    });

    const result = validateStyle(style, PACKAGE_FILES);
    assert.equal(result.valid, false);
    assert.equal(result.external.length, 1);
    assert.match(result.errors[0] ?? '', /requires the Internet/);
  });

  test('refuses a remote glyph server, which is how labels vanish on site', () => {
    // Renders perfectly in the lab, shows unlabelled roads on an isolated network.
    const result = validateStyle(
      localStyleJson({ glyphs: 'https://fonts.example.com/{fontstack}/{range}.pbf' }),
      PACKAGE_FILES,
    );

    assert.equal(result.valid, false);
    assert.ok(result.external.some((reference) => reference.kind === 'glyphs'));
  });

  test('finds an external URL hidden anywhere in the document', () => {
    // A style is a large open format and a dependency can hide in a corner
    // nothing else inspects.
    const style = localStyleJson({
      metadata: { 'vendor:helpUrl': 'https://vendor.example.com/help' },
    });

    const result = validateStyle(style, PACKAGE_FILES);
    assert.equal(result.valid, false);
    assert.ok(result.external.some((reference) => reference.kind === 'other'));
  });

  test('reports a local reference the package does not contain', () => {
    const result = validateStyle(
      localStyleJson({ sources: { basemap: { type: 'vector', url: 'pmtiles://missing.pmtiles' } } }),
      PACKAGE_FILES,
    );

    assert.equal(result.valid, false);
    assert.equal(result.missing.length, 1);
    assert.match(result.errors.join(' '), /does not contain/);
  });

  test('rejects a layer drawing from an undefined source', () => {
    const result = validateStyle(
      localStyleJson({
        layers: [{ id: 'roads', type: 'line', source: 'nonexistent', 'source-layer': 'roads' }],
      }),
      PACKAGE_FILES,
    );

    assert.equal(result.valid, false);
    assert.match(result.errors.join(' '), /which the style does not define/);
  });

  test('rejects a style with no sources or no layers', () => {
    // Emptying the sources leaves every layer pointing at one that is gone.
    const noSources = validateStyle(localStyleJson({ sources: {} }), PACKAGE_FILES);
    assert.equal(noSources.valid, false);
    assert.match(noSources.errors.join(' '), /which the style does not define/);

    assert.equal(validateStyle(localStyleJson({ layers: [] }), PACKAGE_FILES).valid, false);
    assert.match(
      validateStyle(JSON.stringify({ version: 8, layers: [] }), []).errors.join(' '),
      /no sources/,
    );
  });

  test('a sprite base name resolves to the files MapLibre actually fetches', () => {
    // "sprite": "sprite" means sprite.json and sprite.png, not a file called
    // "sprite". Checking it literally reports every correct package as broken.
    assert.equal(validateStyle(localStyleJson({ sprite: 'sprite' }), PACKAGE_FILES).valid, true);

    const absent = validateStyle(localStyleJson({ sprite: 'icons/missing' }), PACKAGE_FILES);
    assert.equal(absent.valid, false);
    assert.match(absent.errors.join(' '), /does not contain/);
  });

  test('rejects a style that is not version 8', () => {
    assert.match(
      validateStyle(localStyleJson({ version: 7 }), PACKAGE_FILES).errors.join(' '),
      /requires version 8/,
    );
  });

  test('warns rather than fails when a style would draw no text', () => {
    const style = JSON.parse(localStyleJson()) as Record<string, unknown>;
    delete style['glyphs'];

    const result = validateStyle(JSON.stringify(style), PACKAGE_FILES);
    assert.equal(result.valid, true, 'a map without labels still works');
    assert.match(result.warnings.join(' '), /no text will be drawn/);
  });

  test('rejects unparseable input without throwing', () => {
    for (const raw of ['', 'not json', '[]', 'null']) {
      const result = validateStyle(raw, []);
      assert.equal(result.valid, false);
      assert.ok(result.errors.length > 0);
    }
  });
});

// -------------------------------------------------------------------- package

describe('map package import', () => {
  test('accepts a well-formed package and describes what it covers', () => {
    const result = importPackage();

    assert.equal(result.valid, true, result.errors.join('; '));
    assert.notEqual(result.manifest, null);

    const manifest = result.manifest!;
    assert.equal(manifest.region, 'Test Region');
    assert.equal(manifest.tileType, 'vector (MVT)');
    assert.equal(manifest.tileCompression, 'gzip');
    assert.equal(manifest.minZoom, 0);
    assert.equal(manifest.maxZoom, 16);
    assert.equal(manifest.attribution, 'Test data');
    assert.ok(manifest.areaSquareKm > 0);
    assert.match(manifest.sha256, /^[0-9a-f]{64}$/);
  });

  test('the package id is derived from content, so a re-import replaces', () => {
    const first = importPackage();
    const second = importPackage();

    assert.equal(first.manifest?.id, second.manifest?.id);
    assert.match(String(first.manifest?.id), /^map-[0-9a-f]{16}$/);

    // Different content, different package.
    const other = importPackage({ tileDataBytes: 1024 });
    assert.notEqual(first.manifest?.id, other.manifest?.id);
  });

  test('refuses an archive whose tiles cannot be rendered', () => {
    const result = importPackage({ tileType: 5 }); // AVIF
    assert.equal(result.valid, false);
    assert.match(result.errors.join(' '), /cannot render/);
  });

  test('refuses an empty archive', () => {
    const result = importPackage({ tileEntries: 0, tileDataBytes: 0 });
    assert.equal(result.valid, false);
    assert.match(result.errors.join(' '), /contains no tiles/);
  });

  test('refuses impossible coverage bounds', () => {
    const result = importPackage({ minLat: 40, maxLat: 30 });
    assert.equal(result.valid, false);
    assert.match(result.errors.join(' '), /impossible coverage area/);
  });

  test('warns when the archive is too coarse for site-level work', () => {
    // A country map is worth importing; being surprised by it at the perimeter
    // is not.
    const result = importPackage({ maxZoom: 10 });

    assert.equal(result.valid, true, 'still usable');
    assert.match(result.warnings.join(' '), /Site-level detail usually needs zoom 16/);
  });

  test('refuses a storage path that escapes the map data directory', () => {
    for (const relativePath of ['../../etc/passwd', '/etc/passwd', 'C:\\windows\\x', '..\\x']) {
      const result = validateMapPackage({
        name: 'Evil',
        tiles: buildPmtiles(),
        style: localStyleJson(),
        files: PACKAGE_FILES,
        relativePath,
        importedAt: utcMillis(0),
      });

      assert.equal(result.valid, false, `${relativePath} should be refused`);
      assert.match(result.errors.join(' '), /absolute or traverses outside/);
    }
  });

  test('an external style reference fails the whole package', () => {
    const result = importPackage(
      {},
      localStyleJson({ glyphs: 'https://fonts.example.com/{fontstack}/{range}.pbf' }),
    );

    assert.equal(result.valid, false);
    assert.equal(result.manifest, null, 'nothing is recorded for a package that cannot work');
  });

  test('warns when the style draws layers the archive does not contain', () => {
    // The only place the two halves of a package are compared.
    const result = importPackage(
      { metadata: { name: 'Roads Only', vector_layers: [{ id: 'roads' }] } },
      localStyleJson(),
    );

    assert.equal(result.valid, true, 'the map still renders, just not those layers');
    assert.match(result.warnings.join(' '), /source layers the archive does not contain/);
    assert.match(result.warnings.join(' '), /buildings/);
  });

  test('a bad archive reports the archive problem and stops', () => {
    // Reporting six style warnings about a file that is not a PMTiles archive
    // buries the one line that matters.
    const result = validateMapPackage({
      name: 'Broken',
      tiles: new TextEncoder().encode('this is not a tile archive at all, not even close'),
      style: localStyleJson(),
      files: PACKAGE_FILES,
      relativePath: 'x/region.pmtiles',
      importedAt: utcMillis(0),
    });

    assert.equal(result.valid, false);
    assert.equal(result.errors.length, 1);
    assert.equal(result.style, null, 'the style was not even examined');
  });
});

describe('integrity', () => {
  test('confirms an untouched archive', () => {
    const bytes = buildPmtiles();
    const result = verifyIntegrity(bytes, sha256Of(bytes));

    assert.equal(result.intact, true);
    assert.match(result.detail, /matches the hash recorded at import/);
  });

  test('detects a modified archive and says what to do', () => {
    // A corrupted archive renders blank tiles rather than failing, so an operator
    // would see an empty map and assume the region was never imported.
    const bytes = buildPmtiles();
    const expected = sha256Of(bytes);

    const tampered = Uint8Array.from(bytes);
    tampered[tampered.length - 1] = (tampered[tampered.length - 1] ?? 0) ^ 0xff;

    const result = verifyIntegrity(tampered, expected);
    assert.equal(result.intact, false);
    assert.match(result.detail, /re-import it from the original media/);
  });
});

describe('selecting a package', () => {
  const manifest = (
    id: string,
    bounds: MapPackageManifest['bounds'],
    maxZoom: number,
  ): MapPackageManifest => ({
    id: derivePackageId(id.padEnd(64, '0')),
    name: id,
    region: id,
    bounds,
    minZoom: 0,
    maxZoom,
    tileType: 'vector (MVT)',
    tileCompression: 'gzip',
    sizeBytes: 1000,
    areaSquareKm: 10,
    tileCount: 100,
    sha256: id.padEnd(64, '0'),
    relativePath: `${id}/region.pmtiles`,
    importedAt: utcMillis(0),
    attribution: null,
    version: null,
  });

  const country = manifest('country', { minLat: 33, minLon: 35, maxLat: 35, maxLon: 37 }, 12);
  const site = manifest('site', { minLat: 33.87, minLon: 35.48, maxLat: 33.92, maxLon: 35.53 }, 18);

  test('prefers the most detailed package covering a point', () => {
    // A deployment may hold a wide regional package and a detailed site package
    // that overlap. The site one is what an operator wants at a camera.
    assert.equal(packageForPoint([country, site], 33.8938, 35.5018)?.name, 'site');
    assert.equal(packageForPoint([site, country], 33.8938, 35.5018)?.name, 'site');
  });

  test('falls back to the wider package outside the detailed one', () => {
    assert.equal(packageForPoint([country, site], 34.5, 36.0)?.name, 'country');
  });

  test('reports no coverage rather than guessing', () => {
    assert.equal(packageForPoint([country, site], 51.5, -0.12), null);
    assert.equal(packageForPoint([], 33.89, 35.5), null);
  });

  test('states the absence plainly when nothing is installed', () => {
    assert.equal(NO_MAP_DATA_MESSAGE, 'OFFLINE MAP DATA NOT INSTALLED');
  });
});
