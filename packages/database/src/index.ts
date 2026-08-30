/**
 * `@sentinel/database` - storage abstraction, migrations and repositories.
 *
 * Standalone deployments run on node:sqlite, which ships inside Node, so a
 * single-machine install needs no database service. Multi-node deployments point
 * the same driver interface at PostgreSQL.
 */

export * from './driver.ts';
export * from './sqlite.ts';
export * from './migrations.ts';
export * from './schema.ts';
