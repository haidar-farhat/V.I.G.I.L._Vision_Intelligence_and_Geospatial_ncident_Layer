"""Persistence.

Without this the system forgets everything the moment it stops, which for
something whose output is evidence is not a limitation but a disqualification.
An incident an operator cannot go back to a week later did not really happen as
far as anyone reviewing it is concerned.

Six conventions hold across every table here, each of them a decision rather than
a habit:

**Timestamps are integer milliseconds since the epoch, UTC.** Local time exists
in the presentation layer and nowhere else. A timezone is a display setting; it
is not a storage format.

**Observed time and recorded time are separate columns, and are never
reconciled.** ``occurred_at`` is when the observing node says it happened;
``recorded_at`` is when this node durably accepted it. The gap between them is
evidence about the deployment — a camera with a drifting clock is a fact worth
keeping, not noise to smooth away.

**No table holds a credential.** A camera stores an opaque handle into the
operating system's keychain, never a password. A test walks every column in the
schema and fails on anything credential-shaped.

**A position is never stored without its uncertainty.** Any row carrying a
latitude also carries its radius and its source. A test enforces it, because
storing a coordinate without its error is how false precision gets into a map and
then into a decision.

**Writes are idempotent.** Ids are deterministic, derived from what a thing *is*,
so replaying footage or receiving a re-sent batch after an outage upserts rather
than duplicating. This is what makes at-least-once delivery safe.

**Foreign keys are on.** SQLite leaves them off by default, which silently
permits orphaned evidence and incidents pointing at events that were deleted.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from .auditing import AuditRecord
from .core import LatLon
from .events import Event, Evidence, EventType, Severity, utc_from_millis
from .incidents import Association, Incident, Risk, RiskFactor
from .registry import DEFAULT_PLATE_FORMAT, PlateFormat, Register
from .registry import SCHEMA as _REGISTER_SCHEMA
from .site import DEFAULT_SITE_ID, FrameKind, Site
from .zones import Schedule, Zone, ZoneKind

from . import paths
from .logs import get as _get_logger

_log = _get_logger(__name__)

#: Schema version this build expects. A database at a different version is
#: migrated forward, never opened as-is: opening a schema you do not understand
#: and hoping the columns line up is how evidence is silently corrupted.
SCHEMA_VERSION = 6


class StoreError(RuntimeError):
    """The store could not be opened, migrated, or written."""


def default_database_path() -> Path:
    """Where the database lives when nobody says otherwise.

    Under the per-OS application data directory rather than beside the code,
    because an operator running from a read-only install, or from a directory
    they do not own, must still get a working system. ``SENTINEL_DATA_DIR``
    overrides it for a deployment that keeps its data on a specific volume —
    which is the normal case for a security appliance with a dedicated disk.

    The directory logic lives in `paths.py` so the database, the log and the
    evidence export cannot disagree about where this deployment keeps its files.
    """
    return paths.database_path()


# ------------------------------------------------------------------ migrations


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    name: str
    up: str
    #: Every migration carries a way back. An upgrade that cannot be undone on a
    #: machine with no Internet and no spare hardware is a gamble, not an upgrade.
    down: str


def _register_schema_sql() -> str:
    """The register's tables, taken from the module that owns them.

    Spelled here as a reference rather than as a copy of the DDL. Two spellings
    of one schema drift — and the half that drifts would be the one holding
    face templates, which is the half nobody may get wrong. A copied
    ``CREATE TABLE`` missing the ``CHECK`` that keeps a ``MATCH`` sighting from
    losing its score would make a migrated database accept a claim about a
    person that a fresh one refuses, and only the deployments that have been
    upgraded would hold it.

    Every statement in `registry.SCHEMA` is ``IF NOT EXISTS``, so applying this
    to a database a `Register` has already touched is a no-op rather than a
    conflict.
    """
    separator = ";\n\n"
    return separator.join(statement.strip() for statement in _REGISTER_SCHEMA) + ";"


MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        version=1,
        name="initial",
        up="""
        CREATE TABLE cameras (
            id                  TEXT PRIMARY KEY,
            name                TEXT NOT NULL,
            source              TEXT NOT NULL,
            -- An opaque handle into the OS keychain. NEVER a password.
            credentials_ref     TEXT,
            latitude            REAL,
            longitude           REAL,
            mount_height        REAL,
            heading             REAL,
            pitch               REAL,
            horizontal_fov      REAL,
            vertical_fov        REAL,
            range_meters        REAL,
            created_at          INTEGER NOT NULL,
            updated_at          INTEGER NOT NULL
        );

        CREATE TABLE zones (
            id                  TEXT PRIMARY KEY,
            name                TEXT NOT NULL,
            kind                TEXT NOT NULL,
            ring                TEXT NOT NULL,
            schedule_start      TEXT,
            schedule_end        TEXT,
            schedule_days       TEXT,
            enter_after_millis  INTEGER NOT NULL,
            exit_after_millis   INTEGER NOT NULL,
            accept_uncertain    INTEGER NOT NULL,
            created_at          INTEGER NOT NULL
        );

        CREATE TABLE events (
            id                  TEXT PRIMARY KEY,
            type                TEXT NOT NULL,
            severity            TEXT NOT NULL,
            summary             TEXT NOT NULL,
            rule_id             TEXT NOT NULL,
            zone_id             TEXT,
            zone_name           TEXT,
            -- When the observing node says it happened, in media time.
            occurred_at_millis  INTEGER NOT NULL,
            -- The same instant as wall clock, per the observing node.
            occurred_at         INTEGER NOT NULL,
            -- When this node durably accepted it. Deliberately not reconciled
            -- with the above: the difference is evidence about the deployment.
            recorded_at         INTEGER NOT NULL,
            confidence          REAL NOT NULL,
            conditions          TEXT NOT NULL,

            camera_id           TEXT NOT NULL,
            track_id            INTEGER NOT NULL,
            first_seen_millis   INTEGER NOT NULL,
            last_seen_millis    INTEGER NOT NULL,
            observations        INTEGER NOT NULL,
            detector            TEXT NOT NULL,
            detector_classifies INTEGER NOT NULL,
            model_digest        TEXT,
            class_label         TEXT NOT NULL,
            latitude            REAL,
            longitude           REAL,
            uncertainty_meters  REAL,
            position_source     TEXT,
            speed_mps           REAL,
            heading_degrees     REAL,
            frame_indices       TEXT NOT NULL
        );

        CREATE INDEX events_by_time   ON events (occurred_at);
        CREATE INDEX events_by_camera ON events (camera_id, occurred_at);
        CREATE INDEX events_by_zone   ON events (zone_id, occurred_at);

        CREATE TABLE incidents (
            id                    TEXT PRIMARY KEY,
            severity              TEXT NOT NULL,
            summary               TEXT NOT NULL,
            opened_at_millis      INTEGER NOT NULL,
            closed_at_millis      INTEGER NOT NULL,
            opened_at             INTEGER NOT NULL,
            recorded_at           INTEGER NOT NULL,
            -- How many objects the contributing track segments are believed to
            -- represent. Separate from the segment count on purpose: conflating
            -- them titles an incident "6 people" when three walked past.
            distinct_object_count INTEGER NOT NULL,
            cameras               TEXT NOT NULL,
            zones                 TEXT NOT NULL,
            risk_score            REAL NOT NULL,
            risk_factors          TEXT NOT NULL,
            associations          TEXT NOT NULL
        );

        CREATE INDEX incidents_by_time ON incidents (opened_at);

        CREATE TABLE incident_events (
            incident_id TEXT NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
            event_id    TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
            PRIMARY KEY (incident_id, event_id)
        );

        -- Append-only. No code path updates or deletes a row here.
        CREATE TABLE audit_logs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            at          INTEGER NOT NULL,
            actor       TEXT NOT NULL,
            action      TEXT NOT NULL,
            subject     TEXT,
            detail      TEXT
        );

        CREATE INDEX audit_by_time ON audit_logs (at);
        """,
        down="""
        DROP TABLE IF EXISTS audit_logs;
        DROP TABLE IF EXISTS incident_events;
        DROP TABLE IF EXISTS incidents;
        DROP TABLE IF EXISTS events;
        DROP TABLE IF EXISTS zones;
        DROP TABLE IF EXISTS cameras;
        """,
    ),
    Migration(
        version=2,
        name="camera_roll",
        up="""
        -- Roll was accepted by CameraPose, dropped on the way into the database
        -- and defaulted to 0.0 on the way out, so camera_pose() handed back a
        -- pose that differed from the one saved.
        --
        -- Added as a new migration rather than by editing version 1: an applied
        -- migration is a fact about every deployment that has run it, and
        -- editing one is how two of them silently diverge.
        ALTER TABLE cameras ADD COLUMN roll REAL;
        """,
        down="""
        ALTER TABLE cameras DROP COLUMN roll;
        """,
    ),
    Migration(
        version=3,
        name="recordings",
        up="""
        -- The index over recorded segments. The files are the evidence; this is
        -- how anything finds them.
        --
        -- Without it a segment can only be located by listing a directory and
        -- parsing filenames, which means retention cannot know what an incident
        -- depends on and evidence cannot ask "what covers this window".
        CREATE TABLE recordings (
            path              TEXT PRIMARY KEY,
            camera_id         TEXT NOT NULL,
            -- Wall clock, always. Media time is meaningless across cameras and
            -- retention works in days.
            started_millis    INTEGER NOT NULL,
            ended_millis      INTEGER NOT NULL,
            frames            INTEGER NOT NULL,
            width             INTEGER NOT NULL,
            height            INTEGER NOT NULL,
            -- What the container header claims, and what was actually measured.
            -- They differ for a live source, where the rate has to be chosen
            -- before the camera has revealed it.
            nominal_fps       REAL NOT NULL,
            measured_fps      REAL NOT NULL,
            codec             TEXT NOT NULL,
            size_bytes        INTEGER NOT NULL,
            sha256            TEXT NOT NULL,
            -- 0 when the writer was closed by a failure rather than by
            -- rotation, so the clip may be short or unplayable. Recorded rather
            -- than hidden: a gap an operator knows about is a different thing
            -- from one they do not.
            complete          INTEGER NOT NULL DEFAULT 1,
            -- Set when an incident depends on this segment. Retention will not
            -- delete a preserved segment however old it is or however full the
            -- disk gets: losing the footage of the one thing that happened, in
            -- order to keep the footage of everything that did not, is the
            -- failure this column exists to prevent.
            preserved         INTEGER NOT NULL DEFAULT 0,
            recorded_at       INTEGER NOT NULL
        );

        -- The two questions actually asked: what covers this window, and what
        -- is oldest.
        CREATE INDEX recordings_by_camera_time
            ON recordings (camera_id, started_millis, ended_millis);
        CREATE INDEX recordings_by_age ON recordings (preserved, started_millis);
        """,
        down="""
        DROP INDEX recordings_by_age;
        DROP INDEX recordings_by_camera_time;
        DROP TABLE recordings;
        """,
    ),
    Migration(
        version=4,
        name="sites",
        up="""
        -- The site: the fixed thing every geographic answer is measured from.
        --
        -- Until now nothing recorded it, so each screen anchored its own frame
        -- on whatever it happened to have: the plan view on the first placed
        -- camera, coverage on the first vertex of the boundary it was handed.
        -- Removing that camera therefore re-anchored the whole view and every
        -- zone, footprint and track jumped — nothing had moved, the ruler had.
        -- An origin in a row cannot be deleted by removing a camera.
        --
        -- One row per site, and one site per node today. The table exists
        -- anyway, because "there is exactly one" is the kind of assumption that
        -- otherwise ends up compiled into forty queries.
        CREATE TABLE sites (
            id             TEXT PRIMARY KEY,
            name           TEXT NOT NULL,
            -- The anchor of the local metric frame. Not the centroid of
            -- anything: it must not move when what it was computed from does.
            origin_lat     REAL NOT NULL,
            origin_lon     REAL NOT NULL,
            -- GEOGRAPHIC: the origin is a real coordinate, so latitudes shown
            -- against it mean what they say. LOCAL: a floor plan or sketch
            -- whose origin is fixed but arbitrary, where distances are real and
            -- coordinates are not. Stored rather than guessed, because a screen
            -- that guesses eventually prints an invented coordinate beside a
            -- surveyed one with nothing to tell them apart. Constrained here so
            -- a third spelling cannot reach the database and be interpreted as
            -- neither.
            frame          TEXT NOT NULL DEFAULT 'GEOGRAPHIC'
                           CHECK (frame IN ('GEOGRAPHIC', 'LOCAL')),
            -- An IANA name, never an offset. An offset is right for half the
            -- year: a site saved as UTC+3 in August is UTC+2 in January, and an
            -- after-hours window that shifts by an hour on the night the clocks
            -- change disarms the site at the hour nobody is watching it.
            timezone       TEXT NOT NULL DEFAULT 'UTC',
            -- The outline, as JSON [[lat, lon], ...] — the same shape zones
            -- store their ring in, so one reader serves both. NULL, not '[]',
            -- when nobody has drawn one: "not drawn yet" and "encloses nothing"
            -- are different answers, and only the second is worth alarming on.
            boundary_ring  TEXT,
            created_at     INTEGER NOT NULL,
            updated_at     INTEGER NOT NULL
        );
        """,
        down="""
        DROP TABLE sites;
        """,
    ),
    Migration(
        version=5,
        name="register",
        # The register: the people and vehicles somebody deliberately named, and
        # the machinery for taking a name away again.
        #
        # In the ladder as well as in `registry.create_schema` because the two
        # must produce the same database. A `Register` handed a bare connection
        # creates its own tables, which is right for a migration tool and wrong
        # as the only path: a deployment where the register exists because
        # something happened to open it has a schema whose presence depends on
        # what ran, and a fresh install would then differ from an upgraded one.
        # The statements are read from `registry.SCHEMA` rather than copied, for
        # the reason `_register_schema_sql` gives.
        up=_register_schema_sql(),
        down="""
        -- Children first: a subject whose identifiers outlive it is an
        -- enrolment nobody can find to delete, which is the one failure this
        -- half of the schema exists to make impossible.
        DROP INDEX IF EXISTS register_sightings_by_subject_time;
        DROP TABLE IF EXISTS register_sightings;
        DROP INDEX IF EXISTS register_identifiers_by_age;
        DROP INDEX IF EXISTS register_identifiers_by_subject;
        DROP INDEX IF EXISTS register_template_unique;
        DROP INDEX IF EXISTS register_plate_unique;
        DROP TABLE IF EXISTS register_identifiers;
        DROP TABLE IF EXISTS register_subjects;
        """,
    ),
    Migration(
        version=6,
        name="audit_records",
        up="""
        -- The structured half of an audit row. Until now an edit reached this
        -- table as one prose string — "kind RESTRICTED -> EXCLUSION" — which is
        -- readable and nothing else: it cannot be filtered, replayed or
        -- checked, and it keeps only the fields somebody wrote a branch for.
        --
        -- Every column is nullable, and that is load-bearing rather than
        -- lenient. Every row already written has none of them, and an audit log
        -- is append-only: there is no pass that can go back and fill these in,
        -- so a NOT NULL here would either fail the migration or force this code
        -- to invent a before-state for an edit made last year.
        ALTER TABLE audit_logs ADD COLUMN before_json TEXT;
        ALTER TABLE audit_logs ADD COLUMN after_json TEXT;
        -- Which node wrote the row. Two nodes' logs merged without it are one
        -- log in which nobody can say where an entry came from.
        ALTER TABLE audit_logs ADD COLUMN node_id TEXT;
        -- SHA-256 over this record and the hash before it. It detects
        -- alteration; it does not prevent it, and it is not a signature —
        -- anybody able to rewrite a row can recompute every hash after it. What
        -- it buys is that an alteration has to be complete to go unnoticed.
        ALTER TABLE audit_logs ADD COLUMN chain_hash TEXT;
        """,
        down="""
        ALTER TABLE audit_logs DROP COLUMN chain_hash;
        ALTER TABLE audit_logs DROP COLUMN node_id;
        ALTER TABLE audit_logs DROP COLUMN after_json;
        ALTER TABLE audit_logs DROP COLUMN before_json;
        """,
    ),
)


# ------------------------------------------------------------------- the store


def _now() -> int:
    return int(time.time() * 1000)


def _statements(script: str) -> list[str]:
    """Split a migration into individual statements.

    Deliberately not ``executescript``: that method commits any open
    transaction before it runs, so a migration executed through it is not
    covered by the transaction wrapped around it — and a failure halfway
    through would leave a partial schema with nothing recorded as applied,
    which is the exact outcome the transaction exists to prevent.

    The splitting is naive because it only ever sees these migrations, which
    contain no semicolons inside string literals. A migration that needs one
    must add a real parser rather than hoping.
    """
    without_comments = "\n".join(
        line for line in script.splitlines() if not line.strip().startswith("--")
    )
    return [part.strip() for part in without_comments.split(";") if part.strip()]


class Store:
    """A local SQLite database holding cameras, zones, events and incidents.

    Synchronous by design. SQLite transactions are connection-scoped, so an
    ``await`` inside one would interleave unrelated work into the same
    transaction — a subtle way to commit half of somebody else's write.
    """

    __slots__ = ("_connection", "_path", "_plate_format", "_register")

    def __init__(
        self,
        path: str | Path = ":memory:",
        *,
        auto_migrate: bool = True,
        plate_format: PlateFormat = DEFAULT_PLATE_FORMAT,
    ):
        """
        ``auto_migrate`` is on for application use: an operator starting the
        console should not have to run a command first. It is off for
        maintenance, because a rollback that the next open silently re-applies
        is not a rollback — somebody stepping back a version to diagnose a
        problem would find the step undone underneath them.

        ``plate_format`` is how this site's country writes a registration down,
        and it is set here because :attr:`register` is the only way to reach the
        register: a caller that could not name the format would have to build
        its own `Register`, which is the thing this store exists to stop.
        """
        self._path = str(path)
        self._plate_format = plate_format
        self._register: Register | None = None
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)

        self._connection = sqlite3.connect(self._path, isolation_level=None)
        self._connection.row_factory = sqlite3.Row

        # WAL lets a reader run while a writer commits, which matters when the
        # interface is querying incidents while a pipeline is inserting events.
        # Not available in memory, where it is also unnecessary.
        if self._path != ":memory:":
            self._connection.execute("PRAGMA journal_mode = WAL")
        # Off by default in SQLite, which silently permits orphaned evidence.
        self._connection.execute("PRAGMA foreign_keys = ON")

        self._ensure_schema(auto_migrate)

    @property
    def path(self) -> str:
        return self._path

    @property
    def register(self) -> Register:
        """Who is enrolled, over this store's own connection.

        A property rather than something a caller constructs, because a
        `Register` needs a connection and the obvious way to get one is to open
        a second connection to the same file. That fails three ways at once, all
        of them quietly: the second connection creates the register's tables
        outside the migration ladder, so a database's schema comes to depend on
        which process opened it first; it cannot see anything written inside a
        transaction this store has open, so an enrolment and the audit row
        proving it was made can disagree about whether it happened; and two
        writers on one SQLite file take turns, so an enrolment during a
        correlation pass waits for a lock or fails on one.

        Built once and kept, since constructing one runs its DDL, and reusing
        the object is what makes "the register" a single thing on this node.
        """
        if self._register is None:
            self._register = Register(
                self._connection, plate_format=self._plate_format
            )
        return self._register

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------ transactions

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """A unit of work that either lands completely or not at all.

        Nesting uses savepoints, because repositories compose — writing an
        incident also writes its events — and an inner failure must be able to
        roll back without abandoning the outer unit of work.
        """
        in_transaction = self._connection.in_transaction
        if in_transaction:
            name = f"sp_{id(self)}_{_now()}"
            self._connection.execute(f"SAVEPOINT {name}")
            try:
                yield self._connection
            except Exception:
                self._connection.execute(f"ROLLBACK TO {name}")
                raise
            finally:
                self._connection.execute(f"RELEASE {name}")
            return

        self._connection.execute("BEGIN")
        try:
            yield self._connection
            self._connection.execute("COMMIT")
        except Exception:
            self._connection.execute("ROLLBACK")
            raise

    # -------------------------------------------------------------- migrations

    def _ensure_schema(self, auto_migrate: bool = True) -> None:
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "  version INTEGER PRIMARY KEY,"
            "  name    TEXT NOT NULL,"
            "  applied_at INTEGER NOT NULL"
            ")"
        )
        if auto_migrate:
            self.migrate()

    def applied_versions(self) -> list[int]:
        rows = self._connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        return [row["version"] for row in rows]

    def pending(self) -> list[Migration]:
        applied = set(self.applied_versions())
        # Sorted by version rather than declaration order, so one added on a
        # branch and merged out of sequence still applies deterministically.
        return sorted(
            (m for m in MIGRATIONS if m.version not in applied), key=lambda m: m.version
        )

    def migrate(self) -> list[Migration]:
        """Apply everything pending, each in its own transaction.

        A failure therefore leaves nothing partially applied, and nothing
        recorded as applied — an operator upgrading an air-gapped deployment
        must get a deterministic result or a clean refusal.
        """
        done: list[Migration] = []
        for migration in self.pending():
            try:
                with self.transaction() as connection:
                    for statement in _statements(migration.up):
                        connection.execute(statement)
                    connection.execute(
                        "INSERT INTO schema_migrations (version, name, applied_at) "
                        "VALUES (?, ?, ?)",
                        (migration.version, migration.name, _now()),
                    )
            except sqlite3.Error as error:
                raise StoreError(
                    f"Migration {migration.version} ({migration.name}) failed: {error}"
                ) from error
            done.append(migration)
        return done

    def rollback(self) -> Migration | None:
        """Undo the most recently applied migration."""
        applied = self.applied_versions()
        if not applied:
            return None

        version = applied[-1]
        migration = next((m for m in MIGRATIONS if m.version == version), None)
        if migration is None:
            raise StoreError(
                f"The database is at version {version}, which this build does not "
                "know how to undo. It was probably written by a newer version."
            )

        with self.transaction() as connection:
            for statement in _statements(migration.down):
                connection.execute(statement)
            connection.execute(
                "DELETE FROM schema_migrations WHERE version = ?", (version,)
            )
        return migration

    # ----------------------------------------------------------------- cameras

    def save_camera(
        self,
        camera_id: str,
        name: str,
        source: str,
        pose=None,
        credentials_ref: str | None = None,
    ) -> None:
        """Record a camera.

        ``source`` must already be redacted — pass ``VideoSource.display_url``,
        never the raw URL. ``credentials_ref`` is an opaque handle into the
        operating system's keychain; a password must never reach this function,
        and a test asserts that nothing stored here looks like one.
        """
        now = _now()
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO cameras (
                    id, name, source, credentials_ref,
                    latitude, longitude, mount_height, heading, pitch, roll,
                    horizontal_fov, vertical_fov, range_meters,
                    created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    source = excluded.source,
                    credentials_ref = excluded.credentials_ref,
                    latitude = excluded.latitude,
                    longitude = excluded.longitude,
                    mount_height = excluded.mount_height,
                    heading = excluded.heading,
                    pitch = excluded.pitch,
                    roll = excluded.roll,
                    horizontal_fov = excluded.horizontal_fov,
                    vertical_fov = excluded.vertical_fov,
                    range_meters = excluded.range_meters,
                    updated_at = excluded.updated_at
                """,
                (
                    camera_id, name, source, credentials_ref,
                    pose.position.lat if pose else None,
                    pose.position.lon if pose else None,
                    pose.mount_height if pose else None,
                    pose.heading if pose else None,
                    pose.pitch if pose else None,
                    pose.roll if pose else None,
                    pose.horizontal_fov if pose else None,
                    pose.vertical_fov if pose else None,
                    pose.range_meters if pose else None,
                    now, now,
                ),
            )

    def cameras(self) -> list[sqlite3.Row]:
        return self._connection.execute(
            "SELECT * FROM cameras ORDER BY id"
        ).fetchall()

    def delete_camera(self, camera_id: str) -> bool:
        """Forget a camera. Returns whether there was one to forget.

        Its events and incidents stay: they are evidence of what was seen,
        and a camera being taken down does not unmake what it saw. They carry
        the camera id as text, not a foreign key, for exactly this reason.
        """
        with self.transaction() as connection:
            cursor = connection.execute("DELETE FROM cameras WHERE id = ?", (camera_id,))
            return cursor.rowcount > 0

    def camera_pose(self, camera_id: str):
        """The stored pose, or ``None`` if the camera was never placed."""
        from .core import CameraPose

        row = self._connection.execute(
            "SELECT * FROM cameras WHERE id = ?", (camera_id,)
        ).fetchone()
        if row is None or row["latitude"] is None:
            return None

        return CameraPose(
            position=LatLon(row["latitude"], row["longitude"]),
            mount_height=row["mount_height"],
            heading=row["heading"],
            pitch=row["pitch"],
            # Stored and returned rather than dropped on the way in and
            # defaulted to 0.0 on the way out — which silently handed back a
            # different pose than the one that was saved.
            roll=row["roll"] if row["roll"] is not None else 0.0,
            horizontal_fov=row["horizontal_fov"],
            vertical_fov=row["vertical_fov"],
            range_meters=row["range_meters"],
        )

    # ------------------------------------------------------------------- sites

    def save_site(self, site: Site) -> None:
        """Record the place being watched: its origin, outline and clock.

        Idempotent on the id, like every other write here, so a console that
        saves the site on each edit updates one row rather than accumulating a
        history nobody asked for. ``created_at`` is deliberately absent from the
        update: it is when this site was first recorded, and re-saving the
        boundary must not rewrite that any more than a re-sent event may rewrite
        when it was accepted.

        An empty boundary is stored as NULL rather than ``[]``. "Nobody has
        drawn the outline yet" and "the outline encloses nothing" lead to
        different screens — the first says coverage cannot be computed, the
        second says none of the site is covered — and collapsing them is how a
        site with no outline gets reported as entirely unwatched.
        """
        now = _now()
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO sites (
                    id, name, origin_lat, origin_lon, frame, timezone,
                    boundary_ring, created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    origin_lat = excluded.origin_lat,
                    origin_lon = excluded.origin_lon,
                    frame = excluded.frame,
                    timezone = excluded.timezone,
                    boundary_ring = excluded.boundary_ring,
                    updated_at = excluded.updated_at
                """,
                (
                    site.id,
                    site.name,
                    site.origin.lat,
                    site.origin.lon,
                    site.frame.value,
                    site.timezone,
                    (
                        json.dumps([[p.lat, p.lon] for p in site.boundary])
                        if site.boundary
                        else None
                    ),
                    now, now,
                ),
            )

    def site(self, site_id: str = DEFAULT_SITE_ID) -> Site | None:
        """The stored site, or ``None`` if this deployment has never named one.

        ``None`` rather than a site invented from the cameras, which is the
        behaviour this table exists to remove: an origin derived from whatever
        was placed first moves the moment that camera is deleted, and every
        object on the plan view moves with it. A caller with no site has to
        decide what to do about it in the open.
        """
        row = self._connection.execute(
            "SELECT * FROM sites WHERE id = ?", (site_id,)
        ).fetchone()
        return None if row is None else _site_from_row(row)

    def sites(self) -> list[Site]:
        """Every site, for the tooling that must not assume there is one."""
        rows = self._connection.execute("SELECT * FROM sites ORDER BY id").fetchall()
        return [_site_from_row(row) for row in rows]

    # ------------------------------------------------------------------- zones

    def save_zone(self, zone: Zone) -> None:
        schedule = zone.schedule
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO zones (
                    id, name, kind, ring,
                    schedule_start, schedule_end, schedule_days,
                    enter_after_millis, exit_after_millis, accept_uncertain,
                    created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    kind = excluded.kind,
                    ring = excluded.ring,
                    schedule_start = excluded.schedule_start,
                    schedule_end = excluded.schedule_end,
                    schedule_days = excluded.schedule_days,
                    enter_after_millis = excluded.enter_after_millis,
                    exit_after_millis = excluded.exit_after_millis,
                    accept_uncertain = excluded.accept_uncertain
                """,
                (
                    zone.id,
                    zone.name,
                    zone.kind.value,
                    json.dumps([[p.lat, p.lon] for p in zone.ring]),
                    schedule.start.isoformat() if schedule else None,
                    schedule.end.isoformat() if schedule else None,
                    json.dumps(sorted(schedule.days)) if schedule else None,
                    zone.enter_after_millis,
                    zone.exit_after_millis,
                    int(zone.accept_uncertain),
                    _now(),
                ),
            )

    def zones(self) -> list[Zone]:
        rows = self._connection.execute("SELECT * FROM zones ORDER BY id").fetchall()
        return [_zone_from_row(row) for row in rows]

    def delete_zone(self, zone_id: str) -> bool:
        """Forget a zone. Returns whether there was one to forget.

        Events raised inside it keep its name in their own text; the zone
        table is the geography as it is now, not as it was.
        """
        with self.transaction() as connection:
            cursor = connection.execute("DELETE FROM zones WHERE id = ?", (zone_id,))
            return cursor.rowcount > 0

    # ------------------------------------------------------------------ events

    def save_events(self, events: Iterable[Event]) -> int:
        """Persist events, idempotently.

        Ids are deterministic, so a replayed or re-sent event collides with the
        row it already produced and updates it rather than making a second one.
        That is what allows a worker to resend everything after the last
        acknowledged sequence without the control node accumulating duplicates.

        ``recorded_at`` is set here, on write, and is never taken from the event.
        """
        recorded_at = _now()
        written = 0

        with self.transaction() as connection:
            for event in events:
                evidence = event.evidence
                connection.execute(
                    """
                    INSERT INTO events (
                        id, type, severity, summary, rule_id, zone_id, zone_name,
                        occurred_at_millis, occurred_at, recorded_at,
                        confidence, conditions,
                        camera_id, track_id, first_seen_millis, last_seen_millis,
                        observations, detector, detector_classifies, model_digest,
                        class_label, latitude, longitude, uncertainty_meters,
                        position_source, speed_mps, heading_degrees, frame_indices
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(id) DO UPDATE SET
                        -- Every column that can differ between two observations
                        -- of the same event, not just the two that were obvious.
                        -- Refreshing `observations` while leaving the position,
                        -- motion, class and frames from an earlier pass produced
                        -- a row that was true of neither observation — a blend,
                        -- presented as evidence.
                        --
                        -- `recorded_at` is deliberately absent: it is when this
                        -- node FIRST durably accepted the event, and a re-send
                        -- must not rewrite that.
                        severity = excluded.severity,
                        summary = excluded.summary,
                        confidence = excluded.confidence,
                        conditions = excluded.conditions,
                        zone_id = excluded.zone_id,
                        zone_name = excluded.zone_name,
                        observations = excluded.observations,
                        first_seen_millis = excluded.first_seen_millis,
                        last_seen_millis = excluded.last_seen_millis,
                        detector = excluded.detector,
                        detector_classifies = excluded.detector_classifies,
                        model_digest = excluded.model_digest,
                        class_label = excluded.class_label,
                        latitude = excluded.latitude,
                        longitude = excluded.longitude,
                        uncertainty_meters = excluded.uncertainty_meters,
                        position_source = excluded.position_source,
                        speed_mps = excluded.speed_mps,
                        heading_degrees = excluded.heading_degrees,
                        frame_indices = excluded.frame_indices
                    """,
                    (
                        event.id,
                        event.type.value,
                        event.severity.value,
                        event.summary,
                        event.rule_id,
                        event.zone_id,
                        event.zone_name,
                        event.occurred_at_millis,
                        int(event.occurred_at.timestamp() * 1000),
                        recorded_at,
                        event.confidence,
                        json.dumps(list(event.triggering_conditions)),
                        evidence.camera_id,
                        evidence.track_id,
                        evidence.first_seen_millis,
                        evidence.last_seen_millis,
                        evidence.observations,
                        evidence.detector,
                        int(evidence.detector_classifies),
                        evidence.model_digest,
                        evidence.class_label,
                        evidence.latitude,
                        evidence.longitude,
                        evidence.position_uncertainty_meters,
                        evidence.position_source,
                        evidence.speed_mps,
                        evidence.heading_degrees,
                        json.dumps(list(evidence.frame_indices)),
                    ),
                )
                written += 1

        return written

    def events(
        self,
        *,
        camera_id: str | None = None,
        since_millis: int | None = None,
        limit: int = 500,
    ) -> list[Event]:
        clauses, params = [], []
        if camera_id is not None:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        if since_millis is not None:
            clauses.append("occurred_at >= ?")
            params.append(since_millis)

        # Filtered and ordered on the SAME column. `since_millis` is a
        # wall-clock instant and `occurred_at_millis` is media time — an earlier
        # version filtered on one and ordered by the other, so a query across
        # two cameras came back interleaved by two incompatible clocks.
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._connection.execute(
            f"SELECT * FROM events {where} ORDER BY occurred_at, id LIMIT ?",
            (*params, limit),
        ).fetchall()
        return [_event_from_row(row) for row in rows]

    def event_count(self) -> int:
        return self._connection.execute("SELECT COUNT(*) AS n FROM events").fetchone()["n"]

    def clock_skew(self, camera_id: str) -> list[int]:
        """How far each event's observed time sat from when it was recorded.

        Kept queryable rather than averaged away. A camera whose clock drifts is
        a fact about the deployment, and an incident timeline assembled from two
        nodes that disagree about the time is worth being able to explain.
        """
        rows = self._connection.execute(
            "SELECT recorded_at - occurred_at AS skew FROM events WHERE camera_id = ?",
            (camera_id,),
        ).fetchall()
        return [row["skew"] for row in rows]

    # --------------------------------------------------------------- incidents

    def save_incident(self, incident: Incident) -> None:
        """Persist an incident and its events as one unit of work.

        Both, together: an incident referring to events that were not written is
        a dangling reference, and events written without the incident that
        explains them lose the reasoning. The savepoint nesting exists for
        exactly this composition.
        """
        with self.transaction() as connection:
            self.save_events(incident.events)

            connection.execute(
                """
                INSERT INTO incidents (
                    id, severity, summary, opened_at_millis, closed_at_millis,
                    opened_at, recorded_at, distinct_object_count,
                    cameras, zones, risk_score, risk_factors, associations
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    severity = excluded.severity,
                    summary = excluded.summary,
                    closed_at_millis = excluded.closed_at_millis,
                    distinct_object_count = excluded.distinct_object_count,
                    cameras = excluded.cameras,
                    zones = excluded.zones,
                    risk_score = excluded.risk_score,
                    risk_factors = excluded.risk_factors,
                    associations = excluded.associations
                """,
                (
                    incident.id,
                    incident.severity.value,
                    incident.summary,
                    incident.opened_at_millis,
                    incident.closed_at_millis,
                    int(incident.opened_at.timestamp() * 1000),
                    _now(),
                    incident.distinct_objects,
                    json.dumps(list(incident.cameras)),
                    json.dumps(list(incident.zones)),
                    incident.risk.score,
                    json.dumps(
                        [
                            {"name": f.name, "points": f.points, "because": f.because}
                            for f in incident.risk.factors
                        ]
                    ),
                    json.dumps(
                        [
                            {
                                "a": list(link.a),
                                "b": list(link.b),
                                "score": link.score,
                                "separation_meters": link.separation_meters,
                                "allowance_meters": link.allowance_meters,
                                "time_gap_millis": link.time_gap_millis,
                                "reasons": list(link.reasons),
                            }
                            for link in incident.associations
                        ]
                    ),
                ),
            )

            # An event belongs to exactly one incident. Correlation is not
            # stable across runs — a later batch can merge two incidents into
            # one, and that one has a different deterministic id — so without
            # this the superseded row survives forever and the same event is
            # linked to both, double-counting on every screen that reads them.
            event_ids = [event.id for event in incident.events]
            if event_ids:
                placeholders = ",".join("?" * len(event_ids))
                superseded = connection.execute(
                    f"""
                    SELECT DISTINCT incident_id FROM incident_events
                    WHERE event_id IN ({placeholders}) AND incident_id != ?
                    """,
                    (*event_ids, incident.id),
                ).fetchall()

                for row in superseded:
                    connection.execute(
                        "DELETE FROM incidents WHERE id = ?", (row["incident_id"],)
                    )

            for event in incident.events:
                connection.execute(
                    "INSERT OR IGNORE INTO incident_events (incident_id, event_id) "
                    "VALUES (?, ?)",
                    (incident.id, event.id),
                )

    def incidents(self, *, limit: int = 100) -> list[sqlite3.Row]:
        return self._connection.execute(
            "SELECT * FROM incidents ORDER BY opened_at DESC LIMIT ?", (limit,)
        ).fetchall()

    def incident(self, incident_id: str) -> Incident | None:
        """Rebuild one stored incident, events and reasoning included.

        Until this existed, an incident could only be exported while the process
        that raised it was still running — which makes an evidence package a
        thing you must remember to produce at the time, rather than a thing you
        can produce when somebody asks. Every field is on the row or in the
        linked events; nothing is recomputed, because re-deriving risk from a
        newer rule set would silently rewrite what was concluded at the time.
        """
        row = self._connection.execute(
            "SELECT * FROM incidents WHERE id = ?", (incident_id,)
        ).fetchone()
        if row is None:
            return None

        events = tuple(self.incident_events(incident_id))
        factors = tuple(
            RiskFactor(name=f["name"], points=f["points"], because=f["because"])
            for f in json.loads(row["risk_factors"])
        )
        associations = tuple(
            Association(
                a=(link["a"][0], int(link["a"][1])),
                b=(link["b"][0], int(link["b"][1])),
                score=link["score"],
                separation_meters=link["separation_meters"],
                allowance_meters=link["allowance_meters"],
                time_gap_millis=link["time_gap_millis"],
                reasons=tuple(link["reasons"]),
            )
            for link in json.loads(row["associations"])
        )

        return Incident(
            id=row["id"],
            severity=Severity(row["severity"]),
            summary=row["summary"],
            opened_at_millis=row["opened_at_millis"],
            closed_at_millis=row["closed_at_millis"],
            opened_at=utc_from_millis(row["opened_at"]),
            distinct_objects=row["distinct_object_count"],
            cameras=tuple(json.loads(row["cameras"])),
            zones=tuple(json.loads(row["zones"])),
            events=events,
            associations=associations,
            risk=Risk(score=row["risk_score"], factors=factors),
        )

    def incident_events(self, incident_id: str) -> list[Event]:
        rows = self._connection.execute(
            """
            SELECT events.* FROM events
            JOIN incident_events ON incident_events.event_id = events.id
            WHERE incident_events.incident_id = ?
            ORDER BY events.occurred_at_millis, events.id
            """,
            (incident_id,),
        ).fetchall()
        return [_event_from_row(row) for row in rows]

    def incident_count(self) -> int:
        return self._connection.execute(
            "SELECT COUNT(*) AS n FROM incidents"
        ).fetchone()["n"]

    # -------------------------------------------------------------- recordings

    @staticmethod
    def segment_key(path: "str | Path") -> str:
        """One spelling of a segment's path, everywhere.

        `str(Path("/rec/a.mp4"))` is `\rec\a.mp4` on Windows and `/rec/a.mp4`
        everywhere else, so a path written by one call and looked up by another
        that skipped the normalisation simply does not match. That failed
        *silently* — `preserve_segments` reported nothing preserved and returned
        0, and the next retention pass would have deleted the footage an
        incident depended on. Resolved as well as normalised, so a relative path
        and an absolute one to the same file are the same row.
        """
        return str(Path(path).resolve())

    def save_segment(self, segment: "Segment") -> None:
        """Index one recorded clip.

        Idempotent on the path, so re-indexing a directory after a crash
        replaces rather than duplicates. `preserved` is deliberately absent from
        the update: re-indexing must never un-preserve evidence somebody is
        relying on.
        """
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO recordings (
                    path, camera_id, started_millis, ended_millis, frames,
                    width, height, nominal_fps, measured_fps, codec,
                    size_bytes, sha256, complete, preserved, recorded_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,0,?)
                ON CONFLICT(path) DO UPDATE SET
                    camera_id = excluded.camera_id,
                    started_millis = excluded.started_millis,
                    ended_millis = excluded.ended_millis,
                    frames = excluded.frames,
                    width = excluded.width,
                    height = excluded.height,
                    nominal_fps = excluded.nominal_fps,
                    measured_fps = excluded.measured_fps,
                    codec = excluded.codec,
                    size_bytes = excluded.size_bytes,
                    sha256 = excluded.sha256,
                    complete = excluded.complete
                """,
                (
                    self.segment_key(segment.path), segment.camera_id,
                    segment.started_millis, segment.ended_millis, segment.frames,
                    segment.width, segment.height,
                    segment.nominal_fps, segment.measured_fps, segment.codec,
                    segment.size_bytes, segment.sha256, int(segment.complete),
                    _now(),
                ),
            )

    def segments(
        self,
        *,
        camera_id: str | None = None,
        start_millis: int | None = None,
        end_millis: int | None = None,
        limit: int = 1000,
    ) -> list["Segment"]:
        """Segments overlapping a window, oldest first.

        Overlap, not containment. A ten-second incident inside a sixty-second
        segment is contained by nothing and covered by one, and asking for
        containment would return an empty set for the commonest case there is.
        """
        clauses, params = [], []
        if camera_id is not None:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        if end_millis is not None:
            clauses.append("started_millis <= ?")
            params.append(end_millis)
        if start_millis is not None:
            clauses.append("ended_millis >= ?")
            params.append(start_millis)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._connection.execute(
            f"SELECT * FROM recordings {where} "
            "ORDER BY started_millis, camera_id LIMIT ?",
            (*params, limit),
        ).fetchall()
        return [_segment_from_row(row) for row in rows]

    def preserve_segments(self, paths: Iterable[str | Path]) -> int:
        """Mark segments as evidence, so retention will never delete them."""
        listed = [self.segment_key(path) for path in paths]
        if not listed:
            return 0
        placeholders = ",".join("?" * len(listed))
        with self.transaction() as connection:
            cursor = connection.execute(
                f"UPDATE recordings SET preserved = 1 WHERE path IN ({placeholders})",
                listed,
            )

        preserved = cursor.rowcount
        if preserved != len(listed):
            # Never silent. Failing to preserve is the one outcome here that
            # destroys evidence, and it destroys it later, on a retention pass,
            # where nothing connects the deletion back to this call.
            _log.warning(
                "asked to preserve %d segment(s) but matched %d in the index; "
                "the unmatched ones are not protected from retention",
                len(listed), preserved,
            )
        return preserved

    def preserved_paths(self) -> set[str]:
        """Every segment an incident depends on, as normalised keys.

        Read once per retention pass rather than per segment: a query per file
        over a fortnight of recordings is twenty thousand round trips to answer
        a question with one answer.
        """
        rows = self._connection.execute(
            "SELECT path FROM recordings WHERE preserved = 1"
        ).fetchall()
        return {row["path"] for row in rows}

    def recorded_bytes(self, *, preserved: bool | None = None) -> int:
        clause = "" if preserved is None else f"WHERE preserved = {int(preserved)}"
        row = self._connection.execute(
            f"SELECT COALESCE(SUM(size_bytes), 0) AS total FROM recordings {clause}"
        ).fetchone()
        return int(row["total"])

    def forget_segment(self, path: str | Path) -> None:
        """Drop one segment from the index. The file is the caller's business."""
        with self.transaction() as connection:
            connection.execute(
                "DELETE FROM recordings WHERE path = ?", (self.segment_key(path),)
            )

    def recording_count(self) -> int:
        return self._connection.execute(
            "SELECT COUNT(*) AS n FROM recordings"
        ).fetchone()["n"]

    # ------------------------------------------------------------------- audit

    def audit(
        self, actor: str, action: str, subject: str | None = None, detail: str | None = None
    ) -> None:
        """Record that somebody did something. Append-only.

        No code path in this module updates or deletes an audit row, and there is
        deliberately no method to. An audit log that can be edited is not one.

        The prose-only path, and it stays. Most actions have no before-state to
        record — a node starting, analysis stopping — and forcing every caller
        through :meth:`audit_record` would make them invent one. A row written
        here carries no structured before/after and no chain hash, which
        :meth:`audit_chain_head` is explicit about.
        """
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO audit_logs (at, actor, action, subject, detail) "
                "VALUES (?,?,?,?,?)",
                (_now(), actor, action, subject, detail),
            )

    def audit_record(self, record: AuditRecord, *, detail: str | None = None) -> str:
        """Record a change in both forms: the prose and the states behind it.

        The prose goes in ``detail`` exactly as before, so a person reading the
        Audit tab sees the line they have always seen; the canonical JSON of
        both states goes beside it, so the same edit can now be filtered,
        replayed and checked. Both come from one comparison — `AuditRecord`
        renders its own changes — which is what stops the two halves drifting
        into disagreeing about what happened.

        ``detail`` overrides that rendering, for a call whose existing line says
        more than a generic diff would: a camera's placement reads as
        ``33.893800,35.501800 h=6.0 hdg=145.0`` and a diff of two poses would
        replace that with JSON. **An overridden line is outside the hash**, which
        covers the record's own fields and not this column. That is a real limit
        and it is named here rather than implied away: the states are protected,
        the sentence rendered from them is not.

        Returns the chain hash written, so a caller can record the head
        somewhere this process cannot reach — which is the only thing that turns
        the chain into evidence of tampering rather than an integrity check.

        ``record.at`` should be timezone-aware. A naive one is read in this
        machine's local zone, which puts the row hours away from where it
        belongs on a node whose clock is not UTC.
        """
        with self.transaction() as connection:
            previous = self.audit_chain_head()
            chain_hash = record.chain(previous)
            connection.execute(
                "INSERT INTO audit_logs "
                "(at, actor, action, subject, detail, before_json, after_json, "
                " node_id, chain_hash) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    int(record.at.timestamp() * 1000),
                    record.actor,
                    record.action,
                    record.subject,
                    record.describe() if detail is None else detail,
                    record.before_json,
                    record.after_json,
                    record.node_id,
                    chain_hash,
                ),
            )
        return chain_hash

    def audit_chain_head(self) -> str | None:
        """The most recent chain hash, or ``None`` if nothing carries one.

        By insertion order rather than by ``at``, because that is the order the
        chain was folded in. Two rows written in the same millisecond — which
        happens whenever an edit writes more than one — would otherwise be
        chained one way and verified the other.

        Rows written by :meth:`audit` are skipped, because they have no hash.
        The chain therefore covers the structured records and reports nothing
        about the prose-only rows between them: an honest chain over part of the
        log beats a claim of coverage over all of it.
        """
        row = self._connection.execute(
            "SELECT chain_hash FROM audit_logs WHERE chain_hash IS NOT NULL "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return None if row is None else row["chain_hash"]

    def audit_trail(self, *, limit: int = 200) -> list[sqlite3.Row]:
        return self._connection.execute(
            "SELECT * FROM audit_logs ORDER BY at DESC, id DESC LIMIT ?", (limit,)
        ).fetchall()

    # ------------------------------------------------------------ introspection

    def table_names(self) -> list[str]:
        rows = self._connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall()
        return [row["name"] for row in rows]

    def column_names(self, table: str) -> list[str]:
        rows = self._connection.execute(f"PRAGMA table_info({table})").fetchall()
        return [row["name"] for row in rows]


# ------------------------------------------------------------- reconstruction




def _segment_from_row(row: sqlite3.Row) -> "Segment":
    from .recording import Segment

    return Segment(
        camera_id=row["camera_id"],
        path=Path(row["path"]),
        started_millis=row["started_millis"],
        ended_millis=row["ended_millis"],
        frames=row["frames"],
        width=row["width"],
        height=row["height"],
        nominal_fps=row["nominal_fps"],
        measured_fps=row["measured_fps"],
        codec=row["codec"],
        size_bytes=row["size_bytes"],
        sha256=row["sha256"],
        complete=bool(row["complete"]),
    )



def _event_from_row(row: sqlite3.Row) -> Event:
    """Rebuild an event, losing nothing that was stored.

    A read that quietly drops a field is worse than one that fails: the evidence
    looks complete and is not.
    """
    evidence = Evidence(
        camera_id=row["camera_id"],
        track_id=row["track_id"],
        first_seen_millis=row["first_seen_millis"],
        last_seen_millis=row["last_seen_millis"],
        observations=row["observations"],
        detector=row["detector"],
        detector_classifies=bool(row["detector_classifies"]),
        model_digest=row["model_digest"],
        class_label=row["class_label"],
        latitude=row["latitude"],
        longitude=row["longitude"],
        position_uncertainty_meters=row["uncertainty_meters"],
        position_source=row["position_source"],
        speed_mps=row["speed_mps"],
        heading_degrees=row["heading_degrees"],
        frame_indices=tuple(json.loads(row["frame_indices"])),
    )
    return Event(
        id=row["id"],
        type=EventType(row["type"]),
        severity=Severity(row["severity"]),
        summary=row["summary"],
        occurred_at_millis=row["occurred_at_millis"],
        occurred_at=datetime.fromtimestamp(row["occurred_at"] / 1000.0, tz=timezone.utc),
        zone_id=row["zone_id"],
        zone_name=row["zone_name"],
        rule_id=row["rule_id"],
        evidence=evidence,
        triggering_conditions=tuple(json.loads(row["conditions"])),
        confidence=row["confidence"],
    )


def _site_from_row(row: sqlite3.Row) -> Site:
    """Rebuild a site, losing neither its frame kind nor its clock.

    Both have been dropped by a reader before, elsewhere in this file, and both
    fail quietly: a site read back as GEOGRAPHIC when it is a floor plan prints
    coordinates that mean nothing, and one read back as UTC evaluates an
    after-hours schedule in the wrong clock.
    """
    ring = row["boundary_ring"]
    return Site(
        id=row["id"],
        name=row["name"],
        origin=LatLon(row["origin_lat"], row["origin_lon"]),
        frame=FrameKind(row["frame"]),
        timezone=row["timezone"],
        # NULL stays empty: a site nobody has outlined has no boundary, which is
        # not the same as one whose boundary is empty.
        boundary=(
            tuple(LatLon(lat, lon) for lat, lon in json.loads(ring)) if ring else ()
        ),
    )


def _zone_from_row(row: sqlite3.Row) -> Zone:
    from datetime import time as clock

    schedule = None
    if row["schedule_start"] is not None:
        schedule = Schedule(
            start=clock.fromisoformat(row["schedule_start"]),
            end=clock.fromisoformat(row["schedule_end"]),
            days=frozenset(json.loads(row["schedule_days"] or "[]")),
        )

    return Zone(
        id=row["id"],
        name=row["name"],
        kind=ZoneKind(row["kind"]),
        ring=tuple(LatLon(lat, lon) for lat, lon in json.loads(row["ring"])),
        schedule=schedule,
        enter_after_millis=row["enter_after_millis"],
        exit_after_millis=row["exit_after_millis"],
        accept_uncertain=bool(row["accept_uncertain"]),
    )


def risk_from_row(row: sqlite3.Row) -> Risk:
    factors = tuple(
        RiskFactor(name=f["name"], points=f["points"], because=f["because"])
        for f in json.loads(row["risk_factors"])
    )
    return Risk(score=row["risk_score"], factors=factors)
