#!/usr/bin/env node
/**
 * Schema management.
 *
 *   node scripts/dev.mjs db migrate | rollback | status
 *
 * Never modify a live schema by hand: an air-gapped operator upgrading in the
 * field must get a deterministic, reversible result.
 */

import process from 'node:process';
import { openDatabase } from './sqlite.ts';
import { MigrationRunner } from './migrations.ts';
import { MIGRATIONS } from './schema.ts';

const path = process.env['SENTINEL_DB_PATH'] ?? './data/sentinel.db';
const command = process.argv[2] ?? 'status';

const driver = openDatabase(path);
const runner = new MigrationRunner(driver, MIGRATIONS);

try {
  if (command === 'migrate') {
    const applied = runner.migrate();
    if (applied.length === 0) {
      console.log('schema is up to date');
    } else {
      for (const migration of applied) {
        console.log(`applied ${migration.version}  ${migration.name}`);
      }
    }
  } else if (command === 'rollback') {
    const rolled = runner.rollback();
    console.log(rolled === null ? 'nothing to roll back' : `rolled back ${rolled.version}  ${rolled.name}`);
  } else if (command === 'status') {
    const status = runner.status();
    console.log(`database        ${path}`);
    console.log(`current version ${status.currentVersion}`);
    console.log(`applied         ${status.applied.length}`);
    console.log(`pending         ${status.pending.length}`);
    for (const migration of status.pending) {
      console.log(`  pending ${migration.version}  ${migration.name}`);
    }
    for (const problem of status.problems) {
      console.error(`  PROBLEM ${problem}`);
    }
    if (status.problems.length > 0) process.exitCode = 1;
  } else {
    console.error(`unknown command: ${command}`);
    console.error('usage: db migrate | rollback | status');
    process.exitCode = 2;
  }
} finally {
  driver.close();
}
