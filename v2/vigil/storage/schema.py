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
)

SCHEMA_VERSION = MIGRATIONS[-1].version
