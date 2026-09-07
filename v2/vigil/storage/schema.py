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
    Migration(
        version=6,
        name="camera_calibration",
        up="""
        -- What the lens does to a straight line, and how well the pose is
        -- actually known.
        --
        -- Both replace an assumption. Until now every camera was assumed
        -- rectilinear and its pose assumed good to +/- 2 degrees, and that
        -- assumption dominated every position at range: at 40 m, two degrees
        -- of heading is 1.4 m of sideways error before the detector has
        -- contributed anything. `vigil cameras calibrate` measures both.
        --
        -- Zero coefficients mean a rectilinear lens, which is what every
        -- camera was before this, so the default reproduces today exactly.
        ALTER TABLE cameras ADD COLUMN k1 REAL NOT NULL DEFAULT 0;
        ALTER TABLE cameras ADD COLUMN k2 REAL NOT NULL DEFAULT 0;
        ALTER TABLE cameras ADD COLUMN p1 REAL NOT NULL DEFAULT 0;
        ALTER TABLE cameras ADD COLUMN p2 REAL NOT NULL DEFAULT 0;
        ALTER TABLE cameras ADD COLUMN k3 REAL NOT NULL DEFAULT 0;

        -- NULL, not a default, and the distinction is the point: NULL means
        -- "nobody measured this, use the stated assumption", and a number
        -- means "this was measured". A default of 2.0 here would make an
        -- uncalibrated camera indistinguishable from one that measured badly.
        ALTER TABLE cameras ADD COLUMN sigma_heading REAL;
        ALTER TABLE cameras ADD COLUMN sigma_pitch REAL;
        ALTER TABLE cameras ADD COLUMN sigma_roll REAL;
        ALTER TABLE cameras ADD COLUMN sigma_height REAL;

        -- Provenance. A measured uncertainty with no record of where it came
        -- from is only a more precise-looking assumption; these say when it
        -- was measured, from how many points, and how well those points were
        -- explained, so a suspicious number can be traced instead of trusted.
        ALTER TABLE cameras ADD COLUMN calibrated_at INTEGER;
        ALTER TABLE cameras ADD COLUMN calibration_rms REAL;
        ALTER TABLE cameras ADD COLUMN calibration_points INTEGER;
        """,
        down="""
        -- A build without these assumes a rectilinear lens and the stated
        -- pose uncertainty, which is what every run did before. The
        -- measurements are lost and the cameras keep working.
        ALTER TABLE cameras DROP COLUMN calibration_points;
        ALTER TABLE cameras DROP COLUMN calibration_rms;
        ALTER TABLE cameras DROP COLUMN calibrated_at;
        ALTER TABLE cameras DROP COLUMN sigma_height;
        ALTER TABLE cameras DROP COLUMN sigma_roll;
        ALTER TABLE cameras DROP COLUMN sigma_pitch;
        ALTER TABLE cameras DROP COLUMN sigma_heading;
        ALTER TABLE cameras DROP COLUMN k3;
        ALTER TABLE cameras DROP COLUMN p2;
        ALTER TABLE cameras DROP COLUMN p1;
        ALTER TABLE cameras DROP COLUMN k2;
        ALTER TABLE cameras DROP COLUMN k1;
        """,
    ),
    Migration(
        version=7,
        name="camera_ground_tilt",
        up="""
        -- The ground this camera projects onto, once the site has solved it.
        --
        -- Every projection before this assumed a level plane at the camera's
        -- mount height. On a yard with a 3% fall that put every position out
        -- along the line of sight by 3% of its range -- over a metre at 40 m
        -- -- and `terrain_slope` only ever widened the error bar around that
        -- bias without removing it. `service.triangulation` fits the plane
        -- from what two cameras see and writes the slope back here.
        --
        -- Zero is the level plane, which is what every camera had, and it is
        -- the exact identity in the projection rather than an approximation.
        ALTER TABLE cameras ADD COLUMN ground_tilt_east REAL NOT NULL DEFAULT 0;
        ALTER TABLE cameras ADD COLUMN ground_tilt_north REAL NOT NULL DEFAULT 0;
        -- When, and from how many observations, so a suspicious slope can be
        -- traced rather than trusted.
        ALTER TABLE cameras ADD COLUMN ground_solved_at INTEGER;
        ALTER TABLE cameras ADD COLUMN ground_observations INTEGER;
        """,
        down="""
        -- A build without these assumes a level yard, which is what every
        -- run did before. The measurement is lost; the cameras keep working.
        ALTER TABLE cameras DROP COLUMN ground_observations;
        ALTER TABLE cameras DROP COLUMN ground_solved_at;
        ALTER TABLE cameras DROP COLUMN ground_tilt_north;
        ALTER TABLE cameras DROP COLUMN ground_tilt_east;
        """,
    ),
    Migration(
        version=8,
        name="site_tiling",
        up="""
        -- Run the detector over crops of the far half of the frame as well as
        -- over the whole of it.
        --
        -- A 1080p frame letterboxed into 640x640 shrinks a person at 40 m to
        -- about 24 pixels, which is at or below what a nano-scale model can
        -- find. A crop at native scale puts them back at full height. It
        -- costs one inference per tile, which is what a GPU provider buys.
        --
        -- NULL means decide from the provider: on where inference is cheap,
        -- off on CPU, where four extra inferences a frame would take a
        -- camera below usable. 0 and 1 are an operator overriding that.
        ALTER TABLE site ADD COLUMN tile_far INTEGER;
        """,
        down="""
        -- A build without it runs the whole frame only, which is what every
        -- run did before.
        ALTER TABLE site DROP COLUMN tile_far;
        """,
    ),
    Migration(
        version=9,
        name="identity",
        up="""
        -- Faces, plates and a register of enrolled subjects. DECISIONS.md
        -- D-08, whose gate was waived deliberately; that entry says by whom.
        --
        -- OFF. `identity_enabled` defaults to 0 and every read path checks it,
        -- so a build that ships with these tables and a site that has never
        -- touched the switch behave exactly as they did before: no face is
        -- embedded, no plate is read, nothing is written here.
        ALTER TABLE site ADD COLUMN identity_enabled INTEGER NOT NULL DEFAULT 0
            CHECK (identity_enabled IN (0,1));
        -- Days after which biometric rows are deleted. NULL is not "keep for
        -- ever" -- `vigil doctor` FAILS when the switch is on and this is
        -- unset, because an unbounded biometric store is the single worst
        -- thing this feature can become, and the failure has to be loud.
        ALTER TABLE site ADD COLUMN biometric_retention_days INTEGER;

        -- Enrolled people. `label` is what an operator typed, never a legal
        -- identity; `embedding` is JSON floats from an operator-supplied
        -- model. Deleting a row deletes the observations that reference it.
        CREATE TABLE subjects (
            id          TEXT PRIMARY KEY,
            label       TEXT NOT NULL,
            note        TEXT,
            embedding   TEXT NOT NULL,
            model_sha256 TEXT NOT NULL,
            enrolled_by TEXT NOT NULL,
            created_at  INTEGER NOT NULL,
            updated_at  INTEGER NOT NULL
        );

        -- One face seen once. `subject_id` is NULL when nothing matched, and
        -- that is the common row: a face that matched nobody is still a face
        -- that was processed, and the retention sweep has to be able to find
        -- it. `distance` and `margin` are stored because a match is a
        -- similarity and the numbers behind it are what make it reviewable.
        CREATE TABLE face_observations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id   TEXT NOT NULL,
            track_id    INTEGER,
            at          INTEGER NOT NULL,
            subject_id  TEXT REFERENCES subjects(id) ON DELETE CASCADE,
            distance    REAL,
            margin      REAL,
            quality     REAL NOT NULL,
            model_sha256 TEXT NOT NULL
        );
        CREATE INDEX face_observations_by_age ON face_observations (at);

        -- One plate read once. `text` is what the OCR produced and
        -- `characters` its per-character confidence, because "ABC123 at 0.71"
        -- hides that the 8 was read at 0.31 and could be a B.
        CREATE TABLE plate_observations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id   TEXT NOT NULL,
            track_id    INTEGER,
            at          INTEGER NOT NULL,
            text        TEXT NOT NULL,
            characters  TEXT NOT NULL,
            confidence  REAL NOT NULL,
            model_sha256 TEXT NOT NULL
        );
        CREATE INDEX plate_observations_by_age ON plate_observations (at);
        """,
        down="""
        -- A build without identity does not process faces or plates, which is
        -- what every run did before. The biometric data is DESTROYED rather
        -- than orphaned, which is the correct direction for this one: rolling
        -- back a feature that should not have been enabled must not leave its
        -- data lying in a table nothing reads any more.
        DROP INDEX plate_observations_by_age;
        DROP TABLE plate_observations;
        DROP INDEX face_observations_by_age;
        DROP TABLE face_observations;
        DROP TABLE subjects;
        ALTER TABLE site DROP COLUMN biometric_retention_days;
        ALTER TABLE site DROP COLUMN identity_enabled;
        """,
    ),
)

SCHEMA_VERSION = MIGRATIONS[-1].version
