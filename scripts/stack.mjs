#!/usr/bin/env node
/**
 * Run the local stack.
 *
 *   node scripts/dev.mjs up
 *
 * Standalone mode: one process holding the API, the realtime hub and the embedded
 * database. Binds loopback. Needs no network, no services and no configuration
 * beyond a data directory.
 *
 * On first run it creates an administrator and prints the password once. That is
 * a development convenience and says so; the first-run wizard will own this in a
 * real installation.
 */

import { mkdirSync } from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { randomBytes } from 'node:crypto';
import { fileURLToPath } from 'node:url';

import { asId, utcMillis } from '../packages/shared-types/src/index.ts';
import { secret } from '../packages/security/src/index.ts';
import { openDatabase } from '../packages/database/src/sqlite.ts';
import { MigrationRunner } from '../packages/database/src/migrations.ts';
import { MIGRATIONS } from '../packages/database/src/schema.ts';
import { createInitialAdmin } from '../services/api/src/auth.ts';
import { startApiServer } from '../services/api/src/server.ts';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

const dataDir = process.env['SENTINEL_DATA_DIR'] ?? path.join(ROOT, 'data');
const host = process.env['SENTINEL_API_HOST'] ?? '127.0.0.1';
const port = Number.parseInt(process.env['SENTINEL_API_PORT'] ?? '8787', 10);

mkdirSync(dataDir, { recursive: true });

const database = openDatabase(path.join(dataDir, 'sentinel.db'));
const applied = new MigrationRunner(database, MIGRATIONS).migrate();
if (applied.length > 0) {
  console.log(`applied ${applied.length} migration(s)`);
}

/*
 * Users live in memory for now.
 *
 * The schema has a users table and the API reads accounts through an injected
 * lookup, so wiring persistence is a small change - but claiming it works before
 * it does is the failure STATUS.md exists to prevent. This says what it is.
 */
const accounts = new Map();

const generatedPassword = randomBytes(12).toString('base64url');
const admin = createInitialAdmin('admin', secret(generatedPassword), utcMillis(Date.now()));
accounts.set('admin', admin);

const server = await startApiServer({
  nodeId: asId(process.env['SENTINEL_NODE_NAME'] ?? 'node-standalone'),
  database,
  host,
  port,
  findUser: (username) => accounts.get(username),
  onAudit: (entry) => {
    // Structured, one line per record, no secrets. The real sink writes to
    // audit_logs; this is the development view of the same stream.
    if (entry.permission === null && entry.outcome === 'SUCCESS') return;
    console.log(
      JSON.stringify({
        at: new Date().toISOString(),
        kind: 'audit',
        ...entry,
      }),
    );
  },
  onLog: (entry) => {
    console.log(
      JSON.stringify({ at: new Date().toISOString(), kind: 'log', ...entry }),
    );
  },
});

const line = '─'.repeat(72);
console.log('');
console.log(line);
console.log('  SENTINEL VISION — standalone');
console.log(line);
console.log(`  API        http://${server.host}:${server.port}`);
console.log(`  Realtime   ws://${server.host}:${server.port}/ws?token=<session>`);
console.log(`  Database   ${path.join(dataDir, 'sentinel.db')}`);
console.log('');
console.log('  Development administrator (in memory, regenerated each start):');
console.log(`    username  admin`);
console.log(`    password  ${generatedPassword}`);
console.log('');
console.log('  Accounts are not yet persisted — see STATUS.md.');
console.log('  No outbound network access is required or attempted.');
console.log(line);
console.log('');

const shutdown = async (signal) => {
  console.log(`\n${signal} received, shutting down.`);
  await server.close();
  database.close();
  process.exit(0);
};

process.on('SIGINT', () => void shutdown('SIGINT'));
process.on('SIGTERM', () => void shutdown('SIGTERM'));
