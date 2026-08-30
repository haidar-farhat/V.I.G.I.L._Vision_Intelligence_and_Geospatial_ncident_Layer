import { DatabaseSync } from 'node:sqlite';
import { mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import type { SqlDriver, SqlRow, SqlValue } from './driver.ts';
import { DatabaseError } from './driver.ts';

/**
 * Embedded SQLite driver.
 *
 * `node:sqlite` ships inside Node, so standalone mode has a real transactional
 * SQL database with no service to install, no daemon to supervise and no
 * third-party dependency in the supply chain - all three of which matter for
 * something that has to run unattended on an isolated network.
 */

export type SqliteOptions = {
  /** File path, or ':memory:' for a throwaway database. */
  readonly path: string;
  /**
   * Write-ahead logging. Lets the recorder and the API read while the event
   * engine writes, which is the normal state of this system. Not available for
   * in-memory databases.
   */
  readonly wal?: boolean;
  readonly readOnly?: boolean;
};

export class SqliteDriver implements SqlDriver {
  readonly #db: DatabaseSync;
  #depth = 0;

  constructor(options: SqliteOptions) {
    if (options.path !== ':memory:') {
      mkdirSync(dirname(options.path), { recursive: true });
    }

    this.#db = new DatabaseSync(options.path, {
      readOnly: options.readOnly ?? false,
    });

    // Foreign keys are off by default in SQLite, which silently permits orphaned
    // evidence and dangling incident references. Turn them on before anything
    // else touches the connection.
    this.#db.exec('PRAGMA foreign_keys = ON');

    if (options.path !== ':memory:' && (options.wal ?? true)) {
      this.#db.exec('PRAGMA journal_mode = WAL');
      // NORMAL is the right trade with WAL: durable across process crashes, and
      // only at risk in a power loss, which a recorder tolerates far better than
      // it tolerates fsync on every frame's worth of metadata.
      this.#db.exec('PRAGMA synchronous = NORMAL');
    }

    this.#db.exec('PRAGMA busy_timeout = 5000');
  }

  exec(sql: string, params: readonly SqlValue[] = []): void {
    try {
      this.#db.prepare(sql).run(...params);
    } catch (error) {
      throw new DatabaseError(messageOf(error), sql);
    }
  }

  query<T extends SqlRow = SqlRow>(sql: string, params: readonly SqlValue[] = []): T[] {
    try {
      return this.#db.prepare(sql).all(...params) as T[];
    } catch (error) {
      throw new DatabaseError(messageOf(error), sql);
    }
  }

  queryOne<T extends SqlRow = SqlRow>(
    sql: string,
    params: readonly SqlValue[] = [],
  ): T | undefined {
    try {
      return this.#db.prepare(sql).get(...params) as T | undefined;
    } catch (error) {
      throw new DatabaseError(messageOf(error), sql);
    }
  }

  /**
   * Transactions, with savepoints for nesting.
   *
   * Nested calls are common once repositories compose (writing an incident also
   * writes its events), and a naive implementation would either commit early or
   * fail outright. Savepoints make the inner scope roll back independently while
   * the outer transaction still governs the whole unit.
   */
  transaction<T>(work: () => T): T {
    const nested = this.#depth > 0;
    const savepoint = `sp_${this.#depth}`;

    this.#db.exec(nested ? `SAVEPOINT ${savepoint}` : 'BEGIN');
    this.#depth += 1;

    try {
      const result = work();
      this.#depth -= 1;
      this.#db.exec(nested ? `RELEASE ${savepoint}` : 'COMMIT');
      return result;
    } catch (error) {
      this.#depth -= 1;
      try {
        this.#db.exec(nested ? `ROLLBACK TO ${savepoint}` : 'ROLLBACK');
      } catch {
        // A failed rollback must not mask the original error, which is the one
        // that actually explains what went wrong.
      }
      throw error;
    }
  }

  script(sql: string): void {
    try {
      this.#db.exec(sql);
    } catch (error) {
      throw new DatabaseError(messageOf(error), sql.slice(0, 200));
    }
  }

  close(): void {
    this.#db.close();
  }
}

const messageOf = (error: unknown): string =>
  error instanceof Error ? error.message : String(error);

/** Open the standalone-mode database. */
export const openDatabase = (path: string): SqlDriver => new SqliteDriver({ path });

/** An in-memory database, for tests. */
export const openMemoryDatabase = (): SqlDriver => new SqliteDriver({ path: ':memory:' });
