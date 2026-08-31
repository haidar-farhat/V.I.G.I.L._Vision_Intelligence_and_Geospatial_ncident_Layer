import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import type { MapPackageId } from '@sentinel/shared-types';
import { asId, utcMillis } from '@sentinel/shared-types';
import { openMemoryDatabase } from '../src/sqlite.ts';
import { MigrationRunner } from '../src/migrations.ts';
import { MIGRATIONS } from '../src/schema.ts';
import { MapPackageRepository } from '../src/repositories/map-packages.ts';
import type { InstalledMapPackage } from '../src/repositories/map-packages.ts';

const setup = () => {
  const db = openMemoryDatabase();
  new MigrationRunner(db, MIGRATIONS).migrate(1000);
  return { db, repo: new MapPackageRepository(db) };
};

const pkg = (
  id: string,
  overrides: Partial<InstalledMapPackage> = {},
): InstalledMapPackage => ({
  id: asId<MapPackageId>(id),
  name: id,
  region: id,
  bounds: { minLat: 33.87, minLon: 35.48, maxLat: 33.92, maxLon: 35.53 },
  minZoom: 0,
  maxZoom: 16,
  tileType: 'vector (MVT)',
  sizeBytes: 50_000_000,
  sha256: 'a'.repeat(64),
  relativePath: `${id}/region.pmtiles`,
  isDefault: false,
  importedAt: utcMillis(1000),
  ...overrides,
});

describe('map package installation', () => {
  test('round-trips a package including its bounds', () => {
    const { db, repo } = setup();
    repo.install(pkg('map-site'));

    const loaded = repo.get(asId<MapPackageId>('map-site'));
    assert.equal(loaded?.region, 'map-site');
    assert.equal(loaded?.maxZoom, 16);
    assert.equal(loaded?.bounds.minLat, 33.87);
    assert.equal(loaded?.tileType, 'vector (MVT)');
    db.close();
  });

  test('re-importing the same package replaces rather than duplicating', () => {
    // The id comes from the archive's content hash, so this is what happens when
    // an operator imports the same USB stick twice.
    const { db, repo } = setup();

    repo.install(pkg('map-site', { name: 'Site' }));
    repo.install(pkg('map-site', { name: 'Site (re-exported)', maxZoom: 18 }));

    assert.equal(repo.list().length, 1);
    assert.equal(repo.get(asId<MapPackageId>('map-site'))?.name, 'Site (re-exported)');
    assert.equal(repo.get(asId<MapPackageId>('map-site'))?.maxZoom, 18);
    db.close();
  });

  test('the first package installed becomes the default', () => {
    // A fresh site gets a working map without an extra step nobody would know
    // to take.
    const { db, repo } = setup();
    repo.install(pkg('map-first'));

    assert.equal(repo.default()?.id, 'map-first');
    db.close();
  });

  test('a second package does not steal the default', () => {
    const { db, repo } = setup();
    repo.install(pkg('map-first'));
    repo.install(pkg('map-second'));

    assert.equal(repo.default()?.id, 'map-first');
    db.close();
  });

  test('exactly one package is default at a time', () => {
    const { db, repo } = setup();
    repo.install(pkg('map-a'));
    repo.install(pkg('map-b'));

    assert.equal(repo.setDefault(asId<MapPackageId>('map-b')), true);

    const defaults = repo.list().filter((entry) => entry.isDefault);
    assert.equal(defaults.length, 1);
    assert.equal(defaults[0]?.id, 'map-b');
    db.close();
  });

  test('setting an unknown package as default reports failure rather than clearing', () => {
    const { db, repo } = setup();
    repo.install(pkg('map-a'));

    assert.equal(repo.setDefault(asId<MapPackageId>('nope')), false);
    assert.equal(repo.default()?.id, 'map-a', 'the existing default is untouched');
    db.close();
  });

  test('a corrupt bounds column lists but never matches, rather than crashing', () => {
    const { db, repo } = setup();
    repo.install(pkg('map-a'));
    db.exec('UPDATE map_packages SET bounds = ? WHERE id = ?', ['not json', 'map-a']);

    const loaded = repo.get(asId<MapPackageId>('map-a'));
    assert.notEqual(loaded, undefined, 'the package still lists');
    assert.equal(loaded?.bounds.minLat, 0);
    db.close();
  });
});

describe('removal', () => {
  test('returns the stored path so the archive can be deleted', () => {
    // Dropping the row alone leaves a multi-gigabyte file nothing references and
    // nothing will ever look at again.
    const { db, repo } = setup();
    repo.install(pkg('map-a'));

    const result = repo.remove(asId<MapPackageId>('map-a'));
    assert.equal(result.removed, true);
    assert.equal(result.relativePath, 'map-a/region.pmtiles');
    assert.equal(repo.get(asId<MapPackageId>('map-a')), undefined);
    db.close();
  });

  test('removing the default promotes the most detailed remaining package', () => {
    // Removing the default must not leave a site with packages but no map.
    const { db, repo } = setup();
    repo.install(pkg('map-default', { maxZoom: 14 }));
    repo.install(pkg('map-coarse', { maxZoom: 10 }));
    repo.install(pkg('map-detailed', { maxZoom: 18 }));

    assert.equal(repo.default()?.id, 'map-default');
    repo.remove(asId<MapPackageId>('map-default'));

    assert.equal(repo.default()?.id, 'map-detailed');
    db.close();
  });

  test('removing the last package leaves no default rather than a dangling one', () => {
    const { db, repo } = setup();
    repo.install(pkg('map-only'));
    repo.remove(asId<MapPackageId>('map-only'));

    assert.equal(repo.default(), undefined);
    assert.equal(repo.list().length, 0);
    db.close();
  });

  test('removing an unknown package reports it rather than pretending', () => {
    const { db, repo } = setup();
    const result = repo.remove(asId<MapPackageId>('nope'));

    assert.equal(result.removed, false);
    assert.equal(result.relativePath, null);
    db.close();
  });
});

describe('storage accounting', () => {
  test('totals the disk occupied by installed packages', () => {
    const { db, repo } = setup();
    repo.install(pkg('map-a', { sizeBytes: 1_000_000 }));
    repo.install(pkg('map-b', { sizeBytes: 2_500_000 }));

    assert.equal(repo.totalSizeBytes(), 3_500_000);
    db.close();
  });

  test('an empty installation totals zero rather than null', () => {
    const { db, repo } = setup();
    assert.equal(repo.totalSizeBytes(), 0);
    db.close();
  });
});
