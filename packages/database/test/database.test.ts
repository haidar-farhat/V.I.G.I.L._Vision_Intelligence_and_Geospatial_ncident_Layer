import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { DatabaseError } from '../src/driver.ts';
import { openMemoryDatabase } from '../src/sqlite.ts';
import { MigrationRunner, checksumOf } from '../src/migrations.ts';
import type { Migration } from '../src/migrations.ts';
import { MIGRATIONS } from '../src/schema.ts';

const migrated = () => {
  const driver = openMemoryDatabase();
  new MigrationRunner(driver, MIGRATIONS).migrate(1000);
  return driver;
};

describe('sqlite driver', () => {
  test('executes statements and reads rows back', () => {
    const db = openMemoryDatabase();
    db.script('CREATE TABLE t (id TEXT PRIMARY KEY, n INTEGER)');
    db.exec('INSERT INTO t (id, n) VALUES (?, ?)', ['a', 1]);
    db.exec('INSERT INTO t (id, n) VALUES (?, ?)', ['b', 2]);

    assert.equal(db.query('SELECT * FROM t').length, 2);
    assert.equal(db.queryOne<{ n: number }>('SELECT n FROM t WHERE id = ?', ['b'])?.n, 2);
    assert.equal(db.queryOne('SELECT n FROM t WHERE id = ?', ['zzz']), undefined);
    db.close();
  });

  test('rows have a null prototype, closing a prototype-pollution path', () => {
    const db = openMemoryDatabase();
    db.script('CREATE TABLE t (id TEXT PRIMARY KEY)');
    db.exec('INSERT INTO t (id) VALUES (?)', ['a']);

    const row = db.queryOne('SELECT * FROM t');
    assert.notEqual(row, undefined);
    assert.equal(Object.getPrototypeOf(row), null, 'Object.prototype must not be in the chain');
    db.close();
  });

  test('reports a failing statement as a DatabaseError with its SQL', () => {
    const db = openMemoryDatabase();
    assert.throws(
      () => db.query('SELECT * FROM does_not_exist'),
      (error: unknown) => {
        assert.ok(error instanceof DatabaseError);
        assert.equal(error.code, 'DATABASE_ERROR');
        assert.match(error.sql, /does_not_exist/);
        return true;
      },
    );
    db.close();
  });

  test('enforces foreign keys', () => {
    // Off by default in SQLite, which would silently permit orphaned evidence.
    const db = migrated();
    assert.throws(
      () =>
        db.exec(
          'INSERT INTO camera_profiles (id, camera_id, kind, name, path, codec, width, height, fps, bitrate_kbps) ' +
            "VALUES ('p1', 'no-such-camera', 'MAIN', 'main', '/live', 'h264', 1920, 1080, 25, 4000)",
        ),
      DatabaseError,
    );
    db.close();
  });
});

describe('transactions', () => {
  test('commits on success', () => {
    const db = openMemoryDatabase();
    db.script('CREATE TABLE t (id TEXT PRIMARY KEY)');

    const result = db.transaction(() => {
      db.exec('INSERT INTO t (id) VALUES (?)', ['a']);
      return 'done';
    });

    assert.equal(result, 'done');
    assert.equal(db.query('SELECT * FROM t').length, 1);
    db.close();
  });

  test('rolls back on a thrown error and rethrows it', () => {
    const db = openMemoryDatabase();
    db.script('CREATE TABLE t (id TEXT PRIMARY KEY)');

    assert.throws(
      () =>
        db.transaction(() => {
          db.exec('INSERT INTO t (id) VALUES (?)', ['a']);
          throw new Error('business rule violated');
        }),
      /business rule violated/,
    );

    assert.equal(db.query('SELECT * FROM t').length, 0, 'the insert must not survive');
    db.close();
  });

  test('nested transactions roll back independently via savepoints', () => {
    // Repositories compose - writing an incident also writes its events - so an
    // inner failure must not abandon the whole outer unit of work.
    const db = openMemoryDatabase();
    db.script('CREATE TABLE t (id TEXT PRIMARY KEY)');

    db.transaction(() => {
      db.exec('INSERT INTO t (id) VALUES (?)', ['outer']);

      try {
        db.transaction(() => {
          db.exec('INSERT INTO t (id) VALUES (?)', ['inner']);
          throw new Error('inner failed');
        });
      } catch {
        // Handled: the outer transaction continues.
      }

      db.exec('INSERT INTO t (id) VALUES (?)', ['after']);
    });

    const ids = db.query<{ id: string }>('SELECT id FROM t ORDER BY id').map((r) => r.id);
    assert.deepEqual(ids, ['after', 'outer'], 'only the inner scope was discarded');
    db.close();
  });

  test('the connection is usable after a rollback', () => {
    const db = openMemoryDatabase();
    db.script('CREATE TABLE t (id TEXT PRIMARY KEY)');

    try {
      db.transaction(() => {
        throw new Error('nope');
      });
    } catch {
      // expected
    }

    db.exec('INSERT INTO t (id) VALUES (?)', ['a']);
    assert.equal(db.query('SELECT * FROM t').length, 1);
    db.close();
  });
});

describe('migrations', () => {
  const simple: readonly Migration[] = [
    {
      version: 1,
      name: 'first',
      up: 'CREATE TABLE a (id TEXT PRIMARY KEY)',
      down: 'DROP TABLE a',
    },
    {
      version: 2,
      name: 'second',
      up: 'CREATE TABLE b (id TEXT PRIMARY KEY)',
      down: 'DROP TABLE b',
    },
  ];

  test('applies pending migrations in version order', () => {
    const db = openMemoryDatabase();
    // Deliberately out of declaration order: ordering must come from the version.
    const runner = new MigrationRunner(db, [simple[1]!, simple[0]!]);

    const applied = runner.migrate(1000);
    assert.deepEqual(
      applied.map((m) => m.version),
      [1, 2],
    );
    assert.equal(runner.status().currentVersion, 2);
    db.close();
  });

  test('is idempotent - a second run applies nothing', () => {
    const db = openMemoryDatabase();
    const runner = new MigrationRunner(db, simple);

    assert.equal(runner.migrate(1000).length, 2);
    assert.equal(runner.migrate(2000).length, 0, 'already applied');
    assert.equal(runner.pending().length, 0);
    db.close();
  });

  test('rolls back the most recent migration', () => {
    const db = openMemoryDatabase();
    const runner = new MigrationRunner(db, simple);
    runner.migrate(1000);

    const rolled = runner.rollback();
    assert.equal(rolled?.version, 2);
    assert.equal(runner.status().currentVersion, 1);
    assert.equal(runner.pending().length, 1);

    // And the table it created is gone.
    assert.throws(() => db.query('SELECT * FROM b'), DatabaseError);
    db.close();
  });

  test('rollback on an empty database is a no-op, not an error', () => {
    const db = openMemoryDatabase();
    assert.equal(new MigrationRunner(db, simple).rollback(), null);
    db.close();
  });

  test('detects a migration edited after it was applied', () => {
    // This is how two deployments silently diverge, so it must be loud.
    const db = openMemoryDatabase();
    new MigrationRunner(db, simple).migrate(1000);

    const tampered: Migration[] = [
      simple[0]!,
      { ...simple[1]!, up: 'CREATE TABLE b (id TEXT PRIMARY KEY, extra TEXT)' },
    ];

    const problems = new MigrationRunner(db, tampered).verify();
    assert.equal(problems.length, 1);
    assert.match(problems[0]!, /has been modified since it was applied/);

    assert.throws(
      () => new MigrationRunner(db, tampered).migrate(2000),
      (error: unknown) => {
        assert.ok(error instanceof DatabaseError);
        assert.equal(error.recoverable, false, 'a divergent schema is not recoverable at runtime');
        return true;
      },
    );
    db.close();
  });

  test('detects a database newer than the application', () => {
    const db = openMemoryDatabase();
    new MigrationRunner(db, simple).migrate(1000);

    // An older build that has never heard of migration 2.
    const problems = new MigrationRunner(db, [simple[0]!]).verify();
    assert.equal(problems.length, 1);
    assert.match(problems[0]!, /newer than the application/);
    db.close();
  });

  test('rejects duplicate migration versions at construction', () => {
    const db = openMemoryDatabase();
    assert.throws(
      () => new MigrationRunner(db, [simple[0]!, { ...simple[1]!, version: 1 }]),
      /duplicate migration version 1/,
    );
    db.close();
  });

  test('a failing migration leaves no partial schema behind', () => {
    const db = openMemoryDatabase();
    const broken: Migration = {
      version: 1,
      name: 'broken',
      up: 'CREATE TABLE ok (id TEXT); THIS IS NOT SQL;',
      down: 'DROP TABLE IF EXISTS ok',
    };

    assert.throws(() => new MigrationRunner(db, [broken]).migrate(1000), DatabaseError);
    assert.equal(
      new MigrationRunner(db, [broken]).status().currentVersion,
      0,
      'a partially applied migration must not be recorded as applied',
    );
    db.close();
  });

  test('checksums are stable and content-sensitive', () => {
    assert.equal(checksumOf(simple[0]!), checksumOf(simple[0]!));
    assert.notEqual(checksumOf(simple[0]!), checksumOf({ ...simple[0]!, up: 'CREATE TABLE c (x)' }));
  });
});

describe('the real schema', () => {
  test('applies cleanly and reports its version', () => {
    const db = openMemoryDatabase();
    const runner = new MigrationRunner(db, MIGRATIONS);

    const applied = runner.migrate(1000);
    assert.equal(applied.length, MIGRATIONS.length);
    assert.deepEqual(runner.verify(), []);
    db.close();
  });

  test('holds no column that could store a credential', () => {
    const db = migrated();
    const columns = db.query<{ name: string; tbl: string }>(
      `SELECT p.name AS name, m.name AS tbl
       FROM sqlite_master m JOIN pragma_table_info(m.name) p
       WHERE m.type = 'table'`,
    );

    for (const column of columns) {
      const name = column.name.toLowerCase();
      // password_hash is the one legitimate exception: it is a one-way hash of a
      // local operator password, never a camera or node credential.
      if (column.tbl === 'users' && name === 'password_hash') continue;

      assert.ok(
        !/password|passwd|secret|apikey|api_key|passphrase|private_key/.test(name),
        `${column.tbl}.${column.name} looks like it could hold a credential`,
      );
    }
    db.close();
  });

  test('stores camera credentials only as an opaque reference', () => {
    const db = migrated();
    const columns = db
      .query<{ name: string }>('SELECT name FROM pragma_table_info(?)', ['cameras'])
      .map((c) => c.name);

    assert.ok(columns.includes('credentials_ref'));
    assert.ok(!columns.includes('password'));
    db.close();
  });

  test('keeps camera time and node time as separate columns', () => {
    // Clock skew between a camera and a node is evidence about the deployment.
    // Reconciling them into one column would destroy that.
    const db = migrated();
    const columns = db
      .query<{ name: string }>('SELECT name FROM pragma_table_info(?)', ['events'])
      .map((c) => c.name);

    assert.ok(columns.includes('occurred_at'));
    assert.ok(columns.includes('recorded_at'));
    db.close();
  });

  test('never separates a position from its uncertainty', () => {
    const db = migrated();
    for (const table of ['events', 'incidents', 'track_observations']) {
      const columns = db
        .query<{ name: string }>('SELECT name FROM pragma_table_info(?)', [table])
        .map((c) => c.name);

      if (columns.includes('latitude')) {
        assert.ok(
          columns.includes('uncertainty_meters'),
          `${table} stores a position with no uncertainty, which is false precision`,
        );
      }
    }
    db.close();
  });

  test('protects incident evidence and audit logs from routine cleanup', () => {
    const db = migrated();
    const protectedClasses = db
      .query<{ data_class: string }>('SELECT data_class FROM retention_policies WHERE protected = 1')
      .map((r) => r.data_class);

    assert.ok(protectedClasses.includes('INCIDENT_EVIDENCE'));
    assert.ok(protectedClasses.includes('AUDIT_LOG'));
    db.close();
  });

  test('round-trips an incident with its events', () => {
    const db = migrated();

    db.transaction(() => {
      db.exec(
        'INSERT INTO incidents (id, title, severity, opened_at, updated_at, distinct_object_count, risk_score, risk_contributions) ' +
          'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
        ['INC-1', '3 people in Restricted Zone A', 'CRITICAL', 1000, 2000, 3, 100, '[]'],
      );
      db.exec(
        'INSERT INTO events (id, type, severity, occurred_at, recorded_at, node_id, confidence, summary, detail, incident_id) ' +
          'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
        ['e1', 'PersonEnteredZone', 'HIGH', 1000, 1050, 'node-1', 0.9, 'a person entered', '{}', 'INC-1'],
      );
      db.exec('INSERT INTO incident_events (incident_id, event_id) VALUES (?, ?)', ['INC-1', 'e1']);
    });

    const incident = db.queryOne<{ title: string; distinct_object_count: number }>(
      'SELECT title, distinct_object_count FROM incidents WHERE id = ?',
      ['INC-1'],
    );
    assert.equal(incident?.distinct_object_count, 3);

    const events = db.query('SELECT e.* FROM events e JOIN incident_events ie ON ie.event_id = e.id WHERE ie.incident_id = ?', ['INC-1']);
    assert.equal(events.length, 1);

    // Deleting the incident takes its links, not its events' history.
    db.exec('DELETE FROM incidents WHERE id = ?', ['INC-1']);
    assert.equal(db.query('SELECT * FROM incident_events').length, 0);
    db.close();
  });
});
