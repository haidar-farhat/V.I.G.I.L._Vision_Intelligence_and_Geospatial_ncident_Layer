/**
 * Storage abstraction.
 *
 * Repositories speak this interface and nothing below it. Standalone deployments
 * back it with `node:sqlite` - embedded, transactional, no daemon, shipped inside
 * Node itself, so a single-machine install needs no database to be installed at
 * all. Larger multi-node deployments point the same interface at PostgreSQL.
 *
 * The seam exists because deployment shape is not knowable in advance, and a
 * codebase that has `SELECT` statements scattered through its services cannot
 * change its mind later.
 */

/** A value a parameterised statement can carry. */
export type SqlValue = string | number | bigint | null | Uint8Array;

/**
 * A result row.
 *
 * Rows come back with a **null prototype**: column values are addressable, but
 * `Object.prototype` is not in the chain. That is deliberate rather than
 * incidental - it means a column named `__proto__` or `constructor` can never
 * reach the prototype chain, and it makes `deepEqual` against a plain object
 * literal fail, which is a small price for closing a prototype-pollution path in
 * a component that reads attacker-influenced data.
 */
export type SqlRow = Record<string, SqlValue>;

export type SqlDriver = {
  /** Run a statement that returns no rows. */
  exec(sql: string, params?: readonly SqlValue[]): void;
  /** Run a query and return every row. */
  query<T extends SqlRow = SqlRow>(sql: string, params?: readonly SqlValue[]): T[];
  /** Run a query and return the first row, or undefined. */
  queryOne<T extends SqlRow = SqlRow>(sql: string, params?: readonly SqlValue[]): T | undefined;
  /**
   * Run `work` inside a transaction, rolling back on any thrown error.
   *
   * Deliberately synchronous: SQLite's transactions are connection-scoped, and an
   * `await` inside one would interleave other work into the same transaction.
   */
  transaction<T>(work: () => T): T;
  /** Execute a multi-statement script, e.g. a migration. */
  script(sql: string): void;
  close(): void;
};

export class DatabaseError extends Error {
  readonly code = 'DATABASE_ERROR';
  readonly sql: string;
  readonly recoverable: boolean;

  constructor(message: string, sql: string, recoverable = true) {
    super(message);
    this.name = 'DatabaseError';
    // The SQL text is retained for diagnosis; parameters are not, because they
    // may carry operational data that has no business in a log.
    this.sql = sql;
    this.recoverable = recoverable;
  }
}
