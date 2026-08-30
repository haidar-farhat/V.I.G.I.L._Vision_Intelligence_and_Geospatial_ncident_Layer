import type { GeoBounds, MapPackageId, UtcMillis } from '@sentinel/shared-types';
import { asId, utcMillis } from '@sentinel/shared-types';
import type { SqlDriver } from '../driver.ts';

/**
 * Installed offline map packages.
 *
 * Mirrors `MapPackageManifest` from `@sentinel/maps`, declared structurally here
 * so the database package does not depend on the maps package - a repository
 * should not need a PMTiles reader to store a row.
 */
export type InstalledMapPackage = {
  readonly id: MapPackageId;
  readonly name: string;
  readonly region: string;
  readonly bounds: GeoBounds;
  readonly minZoom: number;
  readonly maxZoom: number;
  readonly tileType: string;
  readonly sizeBytes: number;
  readonly sha256: string;
  readonly relativePath: string;
  readonly isDefault: boolean;
  readonly importedAt: UtcMillis;
};

type Row = {
  id: string;
  name: string;
  region: string;
  tile_type: string;
  min_zoom: number;
  max_zoom: number;
  bounds: string;
  size_bytes: number;
  sha256: string;
  relative_path: string;
  is_default: number;
  imported_at: number;
};

const DEFAULT_BOUNDS: GeoBounds = { minLat: 0, minLon: 0, maxLat: 0, maxLon: 0 };

export class MapPackageRepository {
  readonly #db: SqlDriver;

  constructor(db: SqlDriver) {
    this.#db = db;
  }

  #toDomain(row: Row): InstalledMapPackage {
    let bounds: GeoBounds = DEFAULT_BOUNDS;
    try {
      bounds = JSON.parse(row.bounds) as GeoBounds;
    } catch {
      // A corrupt bounds column leaves a package that lists but never matches a
      // point, which is visible and fixable. Throwing would take the map screen
      // down entirely.
    }

    return {
      id: asId<MapPackageId>(row.id),
      name: row.name,
      region: row.region,
      bounds,
      minZoom: row.min_zoom,
      maxZoom: row.max_zoom,
      tileType: row.tile_type,
      sizeBytes: row.size_bytes,
      sha256: row.sha256,
      relativePath: row.relative_path,
      isDefault: row.is_default === 1,
      importedAt: utcMillis(row.imported_at),
    };
  }

  list(): readonly InstalledMapPackage[] {
    return this.#db
      .query<Row>('SELECT * FROM map_packages ORDER BY name')
      .map((row) => this.#toDomain(row));
  }

  get(id: MapPackageId): InstalledMapPackage | undefined {
    const row = this.#db.queryOne<Row>('SELECT * FROM map_packages WHERE id = ?', [id]);
    return row === undefined ? undefined : this.#toDomain(row);
  }

  /**
   * Install, or replace an existing package with the same id.
   *
   * The id is derived from the archive's content hash, so re-importing the same
   * region updates in place rather than accumulating duplicates an operator would
   * then have to reason about.
   */
  install(pkg: InstalledMapPackage): void {
    this.#db.transaction(() => {
      this.#db.exec(
        `INSERT INTO map_packages
           (id, name, region, tile_type, min_zoom, max_zoom, bounds, size_bytes, sha256,
            relative_path, is_default, imported_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(id) DO UPDATE SET
           name = excluded.name,
           region = excluded.region,
           tile_type = excluded.tile_type,
           min_zoom = excluded.min_zoom,
           max_zoom = excluded.max_zoom,
           bounds = excluded.bounds,
           size_bytes = excluded.size_bytes,
           sha256 = excluded.sha256,
           relative_path = excluded.relative_path,
           imported_at = excluded.imported_at`,
        [
          pkg.id,
          pkg.name,
          pkg.region,
          pkg.tileType,
          pkg.minZoom,
          pkg.maxZoom,
          JSON.stringify(pkg.bounds),
          pkg.sizeBytes,
          pkg.sha256,
          pkg.relativePath,
          pkg.isDefault ? 1 : 0,
          pkg.importedAt,
        ],
      );

      // The very first package installed becomes the default, so a fresh site has
      // a working map without an extra step nobody would know to take.
      const anyDefault = this.#db.queryOne<{ n: number }>(
        'SELECT COUNT(*) AS n FROM map_packages WHERE is_default = 1',
      );
      if ((anyDefault?.n ?? 0) === 0) this.#setDefaultUnsafe(pkg.id);
    });
  }

  /** Exactly one package is default at a time. */
  setDefault(id: MapPackageId): boolean {
    return this.#db.transaction(() => {
      const exists = this.#db.queryOne('SELECT id FROM map_packages WHERE id = ?', [id]);
      if (exists === undefined) return false;

      this.#setDefaultUnsafe(id);
      return true;
    });
  }

  #setDefaultUnsafe(id: MapPackageId): void {
    this.#db.exec('UPDATE map_packages SET is_default = 0 WHERE is_default = 1');
    this.#db.exec('UPDATE map_packages SET is_default = 1 WHERE id = ?', [id]);
  }

  default(): InstalledMapPackage | undefined {
    const row = this.#db.queryOne<Row>('SELECT * FROM map_packages WHERE is_default = 1');
    return row === undefined ? undefined : this.#toDomain(row);
  }

  /**
   * Remove a package.
   *
   * Returns the stored path so the caller can delete the archive. Dropping the
   * row alone leaves a multi-gigabyte file nothing references and nothing will
   * ever look at again.
   */
  remove(id: MapPackageId): { readonly removed: boolean; readonly relativePath: string | null } {
    return this.#db.transaction(() => {
      const existing = this.#db.queryOne<Row>('SELECT * FROM map_packages WHERE id = ?', [id]);
      if (existing === undefined) return { removed: false, relativePath: null };

      const wasDefault = existing.is_default === 1;
      this.#db.exec('DELETE FROM map_packages WHERE id = ?', [id]);

      // Removing the default must not leave a site with packages but no map.
      if (wasDefault) {
        const next = this.#db.queryOne<{ id: string }>(
          'SELECT id FROM map_packages ORDER BY max_zoom DESC, name LIMIT 1',
        );
        if (next !== undefined) this.#setDefaultUnsafe(asId<MapPackageId>(next.id));
      }

      return { removed: true, relativePath: existing.relative_path };
    });
  }

  /** Total disk occupied by installed packages, for the storage screen. */
  totalSizeBytes(): number {
    const row = this.#db.queryOne<{ total: number | null }>(
      'SELECT SUM(size_bytes) AS total FROM map_packages',
    );
    return row?.total ?? 0;
  }
}
