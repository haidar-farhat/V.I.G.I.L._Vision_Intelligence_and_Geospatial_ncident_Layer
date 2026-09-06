"""The schema, as migrations. Users, audit and alerts are in migration 1.

A migration has a way back or it does not ship: an upgrade with no way back
is a gamble on an air-gapped machine. `password_hash` is the one column that
may look like a secret — a salted one-way hash of a local operator's
password, never a device credential — and the schema test names it as the
single exemption.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    name: str
    up: str
    down: str


MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        version=1,
        name="foundation",
        up="""
        CREATE TABLE site (
            id            TEXT PRIMARY KEY,
            name          TEXT NOT NULL,
            timezone      TEXT NOT NULL DEFAULT 'UTC',
            updated_at    INTEGER NOT NULL
        );
        CREATE TABLE users (
            name          TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            role          TEXT NOT NULL CHECK (role IN ('VIEWER','OPERATOR','ANALYST','ADMIN')),
            active        INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0,1)),
            created_at    INTEGER NOT NULL,
            updated_at    INTEGER NOT NULL
        );
        -- Append-only. No code path updates or deletes a row; a test asserts it.
        CREATE TABLE audit (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            at            INTEGER NOT NULL,
            principal     TEXT NOT NULL,
            action        TEXT NOT NULL,
            subject       TEXT,
            detail        TEXT,
            before        TEXT,
            after         TEXT
        );
        CREATE INDEX audit_by_time ON audit (at);
        CREATE TABLE cameras (
            id              TEXT PRIMARY KEY,
            name            TEXT NOT NULL,
            source          TEXT NOT NULL,           -- never carries a password
            credentials_ref TEXT,                     -- opaque keychain handle
            lat REAL, lon REAL, mount_height REAL, heading REAL, pitch REAL, roll REAL,
            horizontal_fov REAL, vertical_fov REAL, range_meters REAL,
            record          INTEGER NOT NULL DEFAULT 0 CHECK (record IN (0,1)),
            created_at      INTEGER NOT NULL,
            updated_at      INTEGER NOT NULL
        );
        CREATE TABLE zones (
            id                 TEXT PRIMARY KEY,
            name               TEXT NOT NULL,
            kind               TEXT NOT NULL,
            ring               TEXT NOT NULL,         -- JSON [[lat, lon], ...]
            watch              TEXT NOT NULL DEFAULT '[]',
            enter_after_millis INTEGER NOT NULL DEFAULT 600,
            exit_after_millis  INTEGER NOT NULL DEFAULT 2000,
            min_membership     TEXT NOT NULL DEFAULT 'INSIDE',
            closed_from        INTEGER,
            closed_until       INTEGER,
            created_at         INTEGER NOT NULL,
            updated_at         INTEGER NOT NULL
        );
        CREATE TABLE events (
            id           TEXT PRIMARY KEY,
            type         TEXT NOT NULL,
            severity     TEXT NOT NULL,
            summary      TEXT NOT NULL,
            occurred_at  INTEGER NOT NULL,
            recorded_at  INTEGER NOT NULL,
            node_id      TEXT NOT NULL,
            rule_id      TEXT NOT NULL,
            confidence   REAL NOT NULL,
            camera_id    TEXT NOT NULL,
            track_id     INTEGER NOT NULL,
            zone_id      TEXT,
            zone_name    TEXT,
            evidence     TEXT NOT NULL              -- JSON
        );
        CREATE INDEX events_by_time ON events (occurred_at);
        CREATE TABLE incidents (
            id            TEXT PRIMARY KEY,
            severity      TEXT NOT NULL,
            summary       TEXT NOT NULL,
            opened_at     INTEGER NOT NULL,
            closed_at     INTEGER NOT NULL,
            distinct_objects INTEGER NOT NULL,
            cameras       TEXT NOT NULL,
            zones         TEXT NOT NULL,
            risk          TEXT NOT NULL,             -- JSON
            associations  TEXT NOT NULL,             -- JSON
            updated_at    INTEGER NOT NULL
        );
        CREATE TABLE incident_events (
            incident_id TEXT NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
            event_id    TEXT NOT NULL REFERENCES events(id),
            PRIMARY KEY (incident_id, event_id)
        );
        CREATE TABLE recordings (
            path           TEXT PRIMARY KEY,
            camera_id      TEXT NOT NULL,
            started_at     INTEGER NOT NULL,
            ended_at       INTEGER NOT NULL,
            frames         INTEGER NOT NULL,
            width INTEGER NOT NULL, height INTEGER NOT NULL, nominal_fps REAL NOT NULL,
            size_bytes     INTEGER NOT NULL,
            sha256         TEXT NOT NULL,
            preserved      INTEGER NOT NULL DEFAULT 0 CHECK (preserved IN (0,1))
        );
        CREATE INDEX recordings_by_age ON recordings (preserved, started_at);
        CREATE TABLE alerts (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            kind       TEXT NOT NULL,
            subject    TEXT NOT NULL,
            detail     TEXT NOT NULL,
            raised_at  INTEGER NOT NULL,
            cleared_at INTEGER
        );
        CREATE INDEX alerts_open ON alerts (cleared_at);
        """,
        down="""
        DROP TABLE alerts; DROP TABLE recordings; DROP TABLE incident_events; DROP TABLE incidents;
        DROP TABLE events; DROP TABLE zones; DROP TABLE cameras; DROP TABLE audit; DROP TABLE users;
        DROP TABLE site;
        """,
    ),
    Migration(
        version=2,
        name="incident_review",
        up="""
        -- An operator works a queue. Without a state every incident stays new
        -- for ever, the list only grows, and the one that matters is buried
        -- under the ones somebody already looked at.
        ALTER TABLE incidents ADD COLUMN state TEXT NOT NULL DEFAULT 'NEW'
            CHECK (state IN ('NEW', 'ACKNOWLEDGED', 'DISMISSED'));
        ALTER TABLE incidents ADD COLUMN reviewed_by TEXT;
        ALTER TABLE incidents ADD COLUMN reviewed_at INTEGER;
        ALTER TABLE incidents ADD COLUMN note TEXT;
        CREATE INDEX incidents_by_state ON incidents (state, opened_at);
        """,
        down="""
        -- A build without review shows every incident, which is what it did
        -- before. The judgements are lost; the incidents are not.
        DROP INDEX incidents_by_state;
        ALTER TABLE incidents DROP COLUMN note;
        ALTER TABLE incidents DROP COLUMN reviewed_at;
        ALTER TABLE incidents DROP COLUMN reviewed_by;
        ALTER TABLE incidents DROP COLUMN state;
        """,
    ),
    Migration(
        version=3,
        name="site_threats",
        up="""
        -- Which labels this site treats as dangerous. Empty by default and
        -- deliberately so: the shipped model names `knife` and `scissors`,
        -- and a kitchen raising a critical alert every evening teaches an
        -- operator to ignore the word within a week.
        ALTER TABLE site ADD COLUMN threat_labels TEXT NOT NULL DEFAULT '[]';
        """,
        down="""
        -- A build without it treats nothing as a threat, which is what it did.
        ALTER TABLE site DROP COLUMN threat_labels;
        """,
    ),
    Migration(
        version=4,
        name="site_detection",
        up="""
        -- What this site watches for, and how sure the detector must be.
        -- Settings rather than flags: a service started at boot has nobody
        -- to type `--watch` at it, and until now it silently analysed with
        -- the defaults while the operator believed the console's choices
        -- applied everywhere.
        ALTER TABLE site ADD COLUMN watch_labels TEXT NOT NULL DEFAULT '[]';
        ALTER TABLE site ADD COLUMN min_confidence REAL;
        """,
        down="""
        -- A build without them uses the built-in watch list, which is what
        -- every run did before this migration.
        ALTER TABLE site DROP COLUMN min_confidence;
        ALTER TABLE site DROP COLUMN watch_labels;
        """,
    ),
    Migration(
        version=5,
        name="site_detect_every",
        up="""
        -- Run the detector on one frame in N and track through the rest.
        --
        -- Measured on this machine: detecting every third frame costs 3.0x
        -- less and moves a track 0.008 box heights from where full-rate
        -- detection put it, against a projection error of over a metre at
        -- range. It is only safe because the tracker was rebuilt around a
        -- Kalman filter, a weak-detection recovery pass and re-identification
        -- across a gap; on the exponential-average tracker this replaced it
        -- would have been reckless.
        --
        -- 1 is every frame, which is what every run did before this.
        ALTER TABLE site ADD COLUMN detect_every INTEGER NOT NULL DEFAULT 1;
        """,
        down="""
        -- A build without it detects on every frame, which is what it did.
        ALTER TABLE site DROP COLUMN detect_every;
        """,
    ),
)

SCHEMA_VERSION = MIGRATIONS[-1].version
