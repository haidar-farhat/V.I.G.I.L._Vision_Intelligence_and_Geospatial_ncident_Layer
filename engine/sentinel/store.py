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

from .core import LatLon
from .events import Event, Evidence, EventType, Severity, utc_from_millis
from .incidents import Association, Incident, Risk, RiskFactor
from .zones import Schedule, Zone, ZoneKind

from . import paths
from .logs import get as _get_logger

_log = _get_logger(__name__)

#: Schema version this build expects. A database at a different version is
#: migrated forward, never opened as-is: opening a schema you do not understand
#: and hoping the columns line up is how evidence is silently corrupted.
SCHEMA_VERSION = 2


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

    __slots__ = ("_connection", "_path")

    def __init__(self, path: str | Path = ":memory:", *, auto_migrate: bool = True):
        """
        ``auto_migrate`` is on for application use: an operator starting the
        console should not have to run a command first. It is off for
        maintenance, because a rollback that the next open silently re-applies
        is not a rollback — somebody stepping back a version to diagnose a
        problem would find the step undone underneath them.
        """
        self._path = str(path)
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

    # ------------------------------------------------------------------- audit

    def audit(
        self, actor: str, action: str, subject: str | None = None, detail: str | None = None
    ) -> None:
        """Record that somebody did something. Append-only.

        No code path in this module updates or deletes an audit row, and there is
        deliberately no method to. An audit log that can be edited is not one.
        """
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO audit_logs (at, actor, action, subject, detail) "
                "VALUES (?,?,?,?,?)",
                (_now(), actor, action, subject, detail),
            )

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
