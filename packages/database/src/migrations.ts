import { createHash } from 'node:crypto';
import type { SqlDriver } from './driver.ts';
import { DatabaseError } from './driver.ts';

/**
 * Schema migrations.
 *
 * Every schema change is a numbered, checksummed migration. Nothing modifies a
 * live schema by hand: an operator upgrading an air-gapped deployment must get a
 * deterministic result, and an evidence export whose schema cannot be identified
 * is not evidence.
 *
 * Each migration carries its own `down`, because an upgrade that cannot be undone
 * on a machine with no Internet access and no spare hardware is not an upgrade,
 * it is a gamble.
 */

export type Migration = {
  readonly version: number;
  readonly name: string;
  readonly up: string;
  readonly down: string;
};

export type AppliedMigration = {
  readonly version: number;
  readonly name: string;
  readonly checksum: string;
  readonly appliedAt: number;
};

const MIGRATIONS_TABLE = `
CREATE TABLE IF NOT EXISTS schema_migrations (
  version    INTEGER PRIMARY KEY,
  name       TEXT    NOT NULL,
  checksum   TEXT    NOT NULL,
  applied_at INTEGER NOT NULL
)`;

export const checksumOf = (migration: Migration): string =>
  createHash('sha256').update(`${migration.version} ${migration.name} ${migration.up}`).digest('hex');

export class MigrationRunner {
  readonly #driver: SqlDriver;
  readonly #migrations: readonly Migration[];

  constructor(driver: SqlDriver, migrations: readonly Migration[]) {
    this.#driver = driver;

    // Ordering is by version, not by declaration order, so a migration added on a
    // branch and merged out of sequence still applies deterministically.
    this.#migrations = [...migrations].sort((a, b) => a.version - b.version);

    const versions = new Set<number>();
    for (const migration of this.#migrations) {
      if (versions.has(migration.version)) {
        throw new DatabaseError(
          `duplicate migration version ${migration.version}`,
          'migration validation',
          false,
        );
      }
      versions.add(migration.version);
    }

    this.#driver.script(MIGRATIONS_TABLE);
  }

  applied(): readonly AppliedMigration[] {
    return this.#driver
      .query<{ version: number; name: string; checksum: string; applied_at: number }>(
        'SELECT version, name, checksum, applied_at FROM schema_migrations ORDER BY version',
      )
      .map((row) => ({
        version: row.version,
        name: row.name,
        checksum: row.checksum,
        appliedAt: row.applied_at,
      }));
  }

  pending(): readonly Migration[] {
    const done = new Set(this.applied().map((m) => m.version));
    return this.#migrations.filter((m) => !done.has(m.version));
  }

  /**
   * Confirm that applied migrations still match the code that produced them.
   *
   * A changed checksum means someone edited a migration that has already run
   * somewhere. That is how two deployments silently diverge, so it is reported as
   * a hard, unrecoverable error rather than a warning nobody reads.
   */
  verify(): readonly string[] {
    const problems: string[] = [];
    const known = new Map(this.#migrations.map((m) => [m.version, m]));

    for (const applied of this.applied()) {
      const migration = known.get(applied.version);
      if (migration === undefined) {
        problems.push(
          `migration ${applied.version} (${applied.name}) is recorded in the database but ` +
            'is not present in this build - the database is newer than the application',
        );
        continue;
      }
      if (checksumOf(migration) !== applied.checksum) {
        problems.push(
          `migration ${applied.version} (${applied.name}) has been modified since it was ` +
            'applied; the schema in this database is not the schema this build expects',
        );
      }
    }
    return problems;
  }

  /** Apply every pending migration, each in its own transaction. */
  migrate(now: number = Date.now()): readonly Migration[] {
    const problems = this.verify();
    if (problems.length > 0) {
      throw new DatabaseError(problems.join('; '), 'migration verification', false);
    }

    const applied: Migration[] = [];
    for (const migration of this.pending()) {
      this.#driver.transaction(() => {
        this.#driver.script(migration.up);
        this.#driver.exec(
          'INSERT INTO schema_migrations (version, name, checksum, applied_at) VALUES (?, ?, ?, ?)',
          [migration.version, migration.name, checksumOf(migration), now],
        );
      });
      applied.push(migration);
    }
    return applied;
  }

  /** Roll back the most recently applied migration. */
  rollback(): Migration | null {
    const applied = this.applied();
    const last = applied[applied.length - 1];
    if (last === undefined) return null;

    const migration = this.#migrations.find((m) => m.version === last.version);
    if (migration === undefined) {
      throw new DatabaseError(
        `cannot roll back migration ${last.version}: it is not present in this build`,
        'rollback',
        false,
      );
    }

    this.#driver.transaction(() => {
      this.#driver.script(migration.down);
      this.#driver.exec('DELETE FROM schema_migrations WHERE version = ?', [migration.version]);
    });

    return migration;
  }

  status(): {
    readonly applied: readonly AppliedMigration[];
    readonly pending: readonly Migration[];
    readonly problems: readonly string[];
    readonly currentVersion: number;
  } {
    const applied = this.applied();
    return {
      applied,
      pending: this.pending(),
      problems: this.verify(),
      currentVersion: applied[applied.length - 1]?.version ?? 0,
    };
  }
}
