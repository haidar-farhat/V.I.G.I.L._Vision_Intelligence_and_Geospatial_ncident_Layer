"""The store: persistence for every entity, an append-only audit, backups.

Owned by one thread. Every public method asserts it is called from the
thread that opened the connection — v1 enforced this by comment and was
wrong at least once.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

from ..adapters.recorder import Segment
from ..domain.detection import DetectorInfo
from ..domain.events import Event, EventType, Evidence, Severity
from ..domain.geo import CameraPose, Distortion, LatLon, PoseUncertainty
from ..domain.incidents import Association, Incident, Review, ReviewState, Risk, RiskFactor
from ..domain.zones import Membership, Schedule, Zone, ZoneKind
from ..logs import get as _get_logger
from .rows import _camera_dict, _evidence_dict, _event_of, _incident_of, _review_of, _zone_of
from .schema import MIGRATIONS, SCHEMA_VERSION, Migration

#: "Not given", so that clearing a value and leaving it alone are different
#: requests. The same sentinel `service.site` uses, for the same reason.
KEEP = object()

_log = _get_logger(__name__)


class StoreError(RuntimeError):
    pass


class ThreadOwnership(StoreError):
    pass


def _now() -> int:
    return int(time.time() * 1000)


class Store:
    def __init__(self, path: str | Path, *, migrate: bool = True):
        self.path = Path(path) if str(path) != ":memory:" else None
        target = str(path)
        self._owner = threading.get_ident()
        try:
            self._connection = sqlite3.connect(target, isolation_level=None, check_same_thread=True)
        except sqlite3.Error as error:
            raise StoreError(f"could not open {target}: {error}") from error
        self._connection.row_factory = sqlite3.Row
        try:
            self._connection.execute("PRAGMA foreign_keys = ON")
            if self.path is not None:
                self._connection.execute("PRAGMA journal_mode = WAL")
                self._connection.execute("PRAGMA synchronous = NORMAL")
            self._require_intact()
        except sqlite3.Error as error:
            self._connection.close()
            raise StoreError(f"{target} is not a usable database: {error}; restore a backup") from error
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at INTEGER NOT NULL)"
        )
        if migrate:
            self.migrate()

    # ------------------------------------------------------------- plumbing

    def _check_thread(self) -> None:
        if threading.get_ident() != self._owner:
            raise ThreadOwnership("the store belongs to the thread that opened it")

    def _require_intact(self) -> None:
        row = self._connection.execute("PRAGMA quick_check").fetchone()
        if row is None or row[0] != "ok":
            raise StoreError(f"database failed its integrity check: {row[0] if row else 'no answer'}; restore a backup")

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        self._check_thread()
        self._connection.execute("BEGIN")
        try:
            yield self._connection
        except BaseException:
            self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ----------------------------------------------------------- migrations

    def applied_versions(self) -> list[int]:
        return [r[0] for r in self._connection.execute("SELECT version FROM schema_migrations ORDER BY version")]

    def migrate(self) -> list[Migration]:
        self._check_thread()
        applied = set(self.applied_versions())
        newest = max(applied) if applied else 0
        if newest > SCHEMA_VERSION:
            raise StoreError(f"this database is from a newer build (schema {newest}, this build knows {SCHEMA_VERSION}); refusing to touch it")
        done = []
        for migration in MIGRATIONS:
            if migration.version in applied:
                continue
            # `executescript` commits any open transaction before it runs, so
            # the migration and its bookkeeping row are one script, one transaction.
            self._connection.executescript(
                f"BEGIN;\n{migration.up}\nINSERT INTO schema_migrations VALUES "
                f"({migration.version}, '{migration.name}', {_now()});\nCOMMIT;"
            )
            done.append(migration)
            _log.info("applied migration %d %s", migration.version, migration.name)
        return done

    def rollback(self) -> Migration | None:
        applied = self.applied_versions()
        if not applied:
            return None
        migration = next(m for m in MIGRATIONS if m.version == applied[-1])
        self._connection.executescript(
            f"BEGIN;\n{migration.down}\nDELETE FROM schema_migrations WHERE version = {migration.version};\nCOMMIT;"
        )
        return migration

    def table_names(self) -> list[str]:
        return [r[0] for r in self._connection.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]

    def column_names(self, table: str) -> list[str]:
        return [r[1] for r in self._connection.execute(f"PRAGMA table_info({table})")]

    # ---------------------------------------------------------------- audit

    def audit(self, principal: str, action: str, subject: str | None = None, detail: str | None = None,
              *, before: dict | None = None, after: dict | None = None) -> None:
        """Append. There is deliberately no method to update or delete a row."""
        self._check_thread()
        self._connection.execute(
            "INSERT INTO audit (at, principal, action, subject, detail, before, after) VALUES (?,?,?,?,?,?,?)",
            (_now(), principal, action, subject, detail,
             json.dumps(before, sort_keys=True) if before is not None else None,
             json.dumps(after, sort_keys=True) if after is not None else None),
        )

    def audit_trail(self, *, limit: int = 200, since: int | None = None) -> list[sqlite3.Row]:
        self._check_thread()
        if since is not None:
            return self._connection.execute("SELECT * FROM audit WHERE at >= ? ORDER BY id DESC LIMIT ?", (since, limit)).fetchall()
        return self._connection.execute("SELECT * FROM audit ORDER BY id DESC LIMIT ?", (limit,)).fetchall()

    # ---------------------------------------------------------------- users

    def save_user(self, name: str, password_hash: str, role: str, *, active: bool = True) -> None:
        now = _now()
        with self.transaction() as c:
            c.execute("INSERT INTO users VALUES (?,?,?,?,?,?)", (name, password_hash, role, 1 if active else 0, now, now))

    def user(self, name: str) -> sqlite3.Row | None:
        self._check_thread()
        return self._connection.execute("SELECT * FROM users WHERE name = ?", (name,)).fetchone()

    def users(self) -> list[sqlite3.Row]:
        self._check_thread()
        return self._connection.execute("SELECT * FROM users ORDER BY name").fetchall()

    def update_user(self, name: str, *, password_hash: str | None = None, active: bool | None = None, role: str | None = None) -> None:
        with self.transaction() as c:
            if password_hash is not None:
                c.execute("UPDATE users SET password_hash=?, updated_at=? WHERE name=?", (password_hash, _now(), name))
            if active is not None:
                c.execute("UPDATE users SET active=?, updated_at=? WHERE name=?", (1 if active else 0, _now(), name))
            if role is not None:
                c.execute("UPDATE users SET role=?, updated_at=? WHERE name=?", (role, _now(), name))

    # -------------------------------------------------------------- cameras

    def save_camera(self, camera_id: str, name: str, source: str, *, credentials_ref: str | None = None,
                    pose: CameraPose | None = None, record: bool = False,
                    calibration: tuple[int, float, int] | None | object = KEEP,
                    ground: tuple[int, int] | None | object = KEEP) -> None:
        """Save a camera, and the lens and pose uncertainty that came with it.

        A sigma equal to the stated assumption is written as NULL, per
        parameter, so "nobody measured this" survives a round trip instead of
        arriving back as a measurement that happens to agree with the default.
        Per parameter because a fit that pinned the heading down and left the
        roll alone is worth keeping as exactly that.

        `calibration` is `(when, rms, points)` — provenance for those sigmas.
        It defaults to `KEEP` rather than `None` because this method is how
        *every* edit to a camera is written: `set_recording` re-saves the pose
        it just read, and if omitting the provenance meant erasing it, turning
        recording on would silently throw away a calibration. Pass `None` to
        clear it deliberately.
        """
        now = _now()
        p = pose
        lens = p.lens if p else None
        assumed = PoseUncertainty()
        u = p.uncertainty if p else None

        def sigma(name: str) -> float | None:
            if u is None:
                return None
            value = getattr(u, name)
            return None if value == getattr(assumed, name) else value

        when, rms, points = (None, None, None) if calibration in (KEEP, None) else calibration
        solved, observations = (None, None) if ground in (KEEP, None) else ground
        with self.transaction() as c:
            was = None
            if calibration is KEEP or ground is KEEP:
                was = c.execute("SELECT calibrated_at, calibration_rms, calibration_points, "
                                "ground_solved_at, ground_observations FROM cameras WHERE id = ?",
                                (camera_id,)).fetchone()
            if calibration is KEEP and was is not None:
                when, rms, points = was["calibrated_at"], was["calibration_rms"], was["calibration_points"]
            if ground is KEEP and was is not None:
                solved, observations = was["ground_solved_at"], was["ground_observations"]
            c.execute(
                """INSERT INTO cameras (id, name, source, credentials_ref, lat, lon, mount_height, heading, pitch, roll,
                   horizontal_fov, vertical_fov, range_meters, record, created_at, updated_at,
                   k1, k2, p1, p2, k3, sigma_heading, sigma_pitch, sigma_roll, sigma_height,
                   calibrated_at, calibration_rms, calibration_points,
                   ground_tilt_east, ground_tilt_north, ground_solved_at, ground_observations)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET name=excluded.name, source=excluded.source,
                   credentials_ref=excluded.credentials_ref, lat=excluded.lat, lon=excluded.lon,
                   mount_height=excluded.mount_height, heading=excluded.heading, pitch=excluded.pitch, roll=excluded.roll,
                   horizontal_fov=excluded.horizontal_fov, vertical_fov=excluded.vertical_fov,
                   range_meters=excluded.range_meters, record=excluded.record, updated_at=excluded.updated_at,
                   k1=excluded.k1, k2=excluded.k2, p1=excluded.p1, p2=excluded.p2, k3=excluded.k3,
                   sigma_heading=excluded.sigma_heading, sigma_pitch=excluded.sigma_pitch,
                   sigma_roll=excluded.sigma_roll, sigma_height=excluded.sigma_height,
                   calibrated_at=excluded.calibrated_at, calibration_rms=excluded.calibration_rms,
                   calibration_points=excluded.calibration_points,
                   ground_tilt_east=excluded.ground_tilt_east,
                   ground_tilt_north=excluded.ground_tilt_north,
                   ground_solved_at=excluded.ground_solved_at,
                   ground_observations=excluded.ground_observations""",
                (camera_id, name, source, credentials_ref,
                 p.position.lat if p else None, p.position.lon if p else None, p.mount_height if p else None,
                 p.heading if p else None, p.pitch if p else None, p.roll if p else None,
                 p.horizontal_fov if p else None, p.vertical_fov if p else None, p.range_meters if p else None,
                 1 if record else 0, now, now,
                 lens.k1 if lens else 0.0, lens.k2 if lens else 0.0, lens.p1 if lens else 0.0,
                 lens.p2 if lens else 0.0, lens.k3 if lens else 0.0,
                 sigma("heading_deg"), sigma("pitch_deg"), sigma("roll_deg"), sigma("mount_height_m"),
                 when, rms, points,
                 p.ground_tilt_east if p else 0.0, p.ground_tilt_north if p else 0.0,
                 solved, observations),
            )

    def camera(self, camera_id: str) -> dict | None:
        self._check_thread()
        row = self._connection.execute("SELECT * FROM cameras WHERE id = ?", (camera_id,)).fetchone()
        return _camera_dict(row) if row else None

    def cameras(self) -> list[dict]:
        self._check_thread()
        return [_camera_dict(r) for r in self._connection.execute("SELECT * FROM cameras ORDER BY id")]

    def delete_camera(self, camera_id: str) -> None:
        with self.transaction() as c:
            c.execute("DELETE FROM cameras WHERE id = ?", (camera_id,))

    # ---------------------------------------------------------------- zones

    def save_zone(self, zone: Zone) -> None:
        now = _now()
        with self.transaction() as c:
            c.execute(
                """INSERT INTO zones (id, name, kind, ring, watch, enter_after_millis, exit_after_millis, min_membership,
                   closed_from, closed_until, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET name=excluded.name, kind=excluded.kind, ring=excluded.ring,
                   watch=excluded.watch, enter_after_millis=excluded.enter_after_millis,
                   exit_after_millis=excluded.exit_after_millis, min_membership=excluded.min_membership,
                   closed_from=excluded.closed_from, closed_until=excluded.closed_until, updated_at=excluded.updated_at""",
                (zone.id, zone.name, zone.kind.value, json.dumps([[p.lat, p.lon] for p in zone.ring]),
                 json.dumps(sorted(zone.watch)), zone.enter_after_millis, zone.exit_after_millis, zone.min_membership.value,
                 zone.schedule.closed_from if zone.schedule else None, zone.schedule.closed_until if zone.schedule else None,
                 now, now),
            )

    def zones(self) -> list[Zone]:
        self._check_thread()
        return [_zone_of(r) for r in self._connection.execute("SELECT * FROM zones ORDER BY id")]

    def zone(self, zone_id: str) -> Zone | None:
        self._check_thread()
        row = self._connection.execute("SELECT * FROM zones WHERE id = ?", (zone_id,)).fetchone()
        return _zone_of(row) if row else None

    def delete_zone(self, zone_id: str) -> None:
        with self.transaction() as c:
            c.execute("DELETE FROM zones WHERE id = ?", (zone_id,))

    # --------------------------------------------------------------- events

    def save_events(self, events: Sequence[Event]) -> int:
        if not events:
            return 0
        now = _now()
        with self.transaction() as c:
            before = c.total_changes
            c.executemany(
                """INSERT OR IGNORE INTO events (id, type, severity, summary, occurred_at, recorded_at, node_id, rule_id,
                   confidence, camera_id, track_id, zone_id, zone_name, evidence) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                [(e.id, e.type.value, e.severity.value, e.summary, e.occurred_at_millis, now, e.node_id, e.rule_id,
                  e.confidence, e.evidence.camera_id, e.evidence.track_id, e.zone_id, e.zone_name,
                  json.dumps(_evidence_dict(e.evidence))) for e in events],
            )
            return c.total_changes - before

    def events(self, *, since: int | None = None, until: int | None = None, camera_id: str | None = None,
               zone_id: str | None = None, severities: Sequence[str] | None = None, contains: str | None = None,
               limit: int = 10_000) -> list[Event]:
        """Events, filtered in SQL. A search must not read the history into memory."""
        self._check_thread()
        clauses, args = [], []
        if since is not None:
            clauses.append("occurred_at >= ?"); args.append(since)
        if until is not None:
            clauses.append("occurred_at <= ?"); args.append(until)
        if camera_id:
            clauses.append("camera_id = ?"); args.append(camera_id)
        if zone_id:
            clauses.append("zone_id = ?"); args.append(zone_id)
        if severities:
            clauses.append(f"severity IN ({','.join('?' * len(severities))})")
            args.extend(str(s) for s in severities)
        if contains:
            # The summary and the evidence, because "person" is in one and
            # "north-gate" may only be in the other.
            clauses.append("(summary LIKE ? OR evidence LIKE ?)")
            args.extend([f"%{contains}%", f"%{contains}%"])
        sql = "SELECT * FROM events"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY occurred_at LIMIT ?"
        args.append(limit)
        return [_event_of(r) for r in self._connection.execute(sql, args)]

    def event_count(self) -> int:
        self._check_thread()
        return int(self._connection.execute("SELECT COUNT(*) FROM events").fetchone()[0])

    # ------------------------------------------------------------ incidents

    def save_incidents(self, incidents: Sequence[Incident]) -> None:
        now = _now()
        with self.transaction() as c:
            for inc in incidents:
                c.execute(
                    """INSERT INTO incidents (id, severity, summary, opened_at, closed_at, distinct_objects, cameras, zones,
                       risk, associations, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(id) DO UPDATE SET severity=excluded.severity, summary=excluded.summary,
                       closed_at=excluded.closed_at, distinct_objects=excluded.distinct_objects, cameras=excluded.cameras,
                       zones=excluded.zones, risk=excluded.risk, associations=excluded.associations, updated_at=excluded.updated_at
                       -- The review columns are deliberately absent: a
                       -- re-correlation refines what the system concluded and
                       -- must never undo what a person decided about it.""",
                    (inc.id, inc.severity.value, inc.summary, inc.opened_at_millis, inc.closed_at_millis, inc.distinct_objects,
                     json.dumps(list(inc.cameras)), json.dumps(list(inc.zones)),
                     json.dumps({"score": inc.risk.score, "factors": [asdict(f) for f in inc.risk.factors]}),
                     json.dumps([asdict(a) for a in inc.associations]), now),
                )
                c.execute("DELETE FROM incident_events WHERE incident_id = ?", (inc.id,))
                c.executemany("INSERT OR IGNORE INTO incident_events VALUES (?,?)", [(inc.id, e.id) for e in inc.events])

    def incidents(self, *, limit: int = 500, states: Sequence[str] | None = None, since: int | None = None,
                  until: int | None = None, camera_id: str | None = None, zone: str | None = None,
                  severities: Sequence[str] | None = None, contains: str | None = None) -> list[Incident]:
        """Incidents, newest first, filtered in SQL.

        `cameras` and `zones` are JSON arrays on the row, so a camera or zone
        is matched with LIKE on the quoted name — exact enough because both
        are ids the site controls, and it keeps the filter in the database.
        """
        self._check_thread()
        out = []
        clauses, args = [], []
        if states:
            clauses.append(f"state IN ({','.join('?' * len(states))})")
            args.extend(str(s) for s in states)
        if since is not None:
            clauses.append("closed_at >= ?"); args.append(since)
        if until is not None:
            clauses.append("opened_at <= ?"); args.append(until)
        if camera_id:
            clauses.append("cameras LIKE ?"); args.append(f'%"{camera_id}"%')
        if zone:
            clauses.append("zones LIKE ?"); args.append(f'%"{zone}"%')
        if severities:
            clauses.append(f"severity IN ({','.join('?' * len(severities))})")
            args.extend(str(s) for s in severities)
        if contains:
            clauses.append("(summary LIKE ? OR note LIKE ?)")
            args.extend([f"%{contains}%", f"%{contains}%"])
        sql = "SELECT * FROM incidents"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY opened_at DESC LIMIT ?"
        args.append(limit)
        for row in self._connection.execute(sql, args):
            events = [_event_of(r) for r in self._connection.execute(
                "SELECT e.* FROM events e JOIN incident_events ie ON ie.event_id = e.id WHERE ie.incident_id = ? ORDER BY e.occurred_at",
                (row["id"],))]
            out.append(_incident_of(row, events))
        return out

    def incident(self, incident_id: str) -> Incident | None:
        self._check_thread()
        row = self._connection.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()
        if row is None:
            return None
        events = [_event_of(r) for r in self._connection.execute(
            "SELECT e.* FROM events e JOIN incident_events ie ON ie.event_id = e.id WHERE ie.incident_id = ? ORDER BY e.occurred_at",
            (incident_id,))]
        return _incident_of(row, events)

    def newest_event_millis(self) -> int | None:
        """When the newest event happened, or ``None``. Read to bound a correlation.

        Not the wall clock: a file source stamps its events from the file's
        own timeline, so "the last hour" has to be measured from the events
        themselves or a whole run falls outside it.
        """
        self._check_thread()
        row = self._connection.execute("SELECT MAX(occurred_at) FROM events").fetchone()
        return int(row[0]) if row and row[0] is not None else None

    def set_incident_review(self, incident_id: str, state: str, *, by: str, at: int, note: str | None) -> None:
        with self.transaction() as c:
            c.execute("UPDATE incidents SET state=?, reviewed_by=?, reviewed_at=?, note=?, updated_at=? WHERE id=?",
                      (state, by, at, note, _now(), incident_id))

    def delete_incidents(self) -> int:
        with self.transaction() as c:
            before = c.total_changes
            c.execute("DELETE FROM incident_events")
            c.execute("DELETE FROM incidents")
            return c.total_changes - before

    # ----------------------------------------------------------- recordings

    def save_segment(self, segment: Segment) -> None:
        with self.transaction() as c:
            c.execute(
                """INSERT INTO recordings (path, camera_id, started_at, ended_at, frames, width, height, nominal_fps, size_bytes, sha256)
                   VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET ended_at=excluded.ended_at, frames=excluded.frames,
                   size_bytes=excluded.size_bytes, sha256=excluded.sha256""",
                (str(segment.path), segment.camera_id, segment.started_millis, segment.ended_millis, segment.frames,
                 segment.width, segment.height, segment.nominal_fps, segment.size_bytes, segment.sha256),
            )

    def segments(self, *, camera_id: str | None = None, between: tuple[int, int] | None = None) -> list[Segment]:
        self._check_thread()
        sql, args = "SELECT * FROM recordings", []
        clauses = []
        if camera_id is not None:
            clauses.append("camera_id = ?")
            args.append(camera_id)
        if between is not None:
            clauses.append("ended_at >= ? AND started_at <= ?")
            args.extend(between)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY started_at"
        return [Segment(r["camera_id"], Path(r["path"]), r["started_at"], r["ended_at"], r["frames"], r["width"], r["height"],
                        r["nominal_fps"], r["size_bytes"], r["sha256"]) for r in self._connection.execute(sql, args)]

    def preserve_segments(self, paths: Sequence[str | Path]) -> int:
        with self.transaction() as c:
            before = c.total_changes
            c.executemany("UPDATE recordings SET preserved = 1 WHERE path = ?", [(str(p),) for p in paths])
            return c.total_changes - before

    def preserved_paths(self) -> set[str]:
        self._check_thread()
        return {r[0] for r in self._connection.execute("SELECT path FROM recordings WHERE preserved = 1")}

    def forget_segment(self, path: str | Path) -> None:
        with self.transaction() as c:
            c.execute("DELETE FROM recordings WHERE path = ?", (str(path),))

    def recorded_bytes(self) -> int:
        self._check_thread()
        return int(self._connection.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM recordings").fetchone()[0])

    # ---------------------------------------------------------------- alerts

    def open_alert(self, kind: str, subject: str, detail: str, at: int) -> int:
        with self.transaction() as c:
            return int(c.execute("INSERT INTO alerts (kind, subject, detail, raised_at) VALUES (?,?,?,?)", (kind, subject, detail, at)).lastrowid)

    def close_alert(self, kind: str, subject: str, at: int) -> None:
        with self.transaction() as c:
            c.execute("UPDATE alerts SET cleared_at = ? WHERE kind = ? AND subject = ? AND cleared_at IS NULL", (at, kind, subject))

    def open_alerts(self) -> list[sqlite3.Row]:
        self._check_thread()
        return self._connection.execute("SELECT * FROM alerts WHERE cleared_at IS NULL ORDER BY raised_at").fetchall()

    def alert_history(self, *, limit: int = 200) -> list[sqlite3.Row]:
        self._check_thread()
        return self._connection.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)).fetchall()

    # ----------------------------------------------------------------- site

    def site(self) -> dict:
        self._check_thread()
        row = self._connection.execute("SELECT * FROM site WHERE id = 'site'").fetchone()
        if row is None:
            return {"id": "site", "name": "Unnamed site", "timezone": "UTC", "threat_labels": [],
                    "watch_labels": [], "min_confidence": None, "detect_every": 1, "updated_at": 0}
        site = dict(row)
        site["threat_labels"] = json.loads(site.get("threat_labels") or "[]")
        site["watch_labels"] = json.loads(site.get("watch_labels") or "[]")
        return site

    def set_threat_labels(self, labels: Sequence[str]) -> None:
        with self.transaction() as c:
            c.execute("INSERT INTO site (id, name, timezone, threat_labels, updated_at) "
                      "VALUES ('site', 'Unnamed site', 'UTC', ?, ?) "
                      "ON CONFLICT(id) DO UPDATE SET threat_labels=excluded.threat_labels, updated_at=excluded.updated_at",
                      (json.dumps(sorted({str(l).strip().lower() for l in labels if str(l).strip()})), _now()))

    def set_detection(self, labels: Sequence[str], confidence: float | None,
                      detect_every: int = 1, tile: bool | None = None) -> None:
        """What to watch for, how sure to be, how often to look, and whether to
        look closely at the far ground as well as at the whole frame."""
        with self.transaction() as c:
            c.execute("INSERT INTO site (id, name, timezone, watch_labels, min_confidence, detect_every, "
                      "tile_far, updated_at) VALUES ('site', 'Unnamed site', 'UTC', ?, ?, ?, ?, ?) "
                      "ON CONFLICT(id) DO UPDATE SET watch_labels=excluded.watch_labels, "
                      "min_confidence=excluded.min_confidence, detect_every=excluded.detect_every, "
                      "tile_far=excluded.tile_far, updated_at=excluded.updated_at",
                      (json.dumps(sorted({str(l).strip().lower() for l in labels if str(l).strip()})),
                       None if confidence is None else float(confidence), int(detect_every),
                       None if tile is None else int(bool(tile)), _now()))

    # ------------------------------------------------------------- identity

    def set_identity(self, enabled: bool, retention_days: int | None) -> None:
        """The identity switch and the biometric retention limit.

        One method for both, because they are one decision. Turning face and
        plate recognition on without saying how long the data lives is the
        state `vigil doctor` fails on, and making them separate calls would
        make that state reachable one statement at a time.
        """
        with self.transaction() as c:
            c.execute("INSERT INTO site (id, name, timezone, identity_enabled, "
                      "biometric_retention_days, updated_at) VALUES ('site', 'Unnamed site', 'UTC', ?, ?, ?) "
                      "ON CONFLICT(id) DO UPDATE SET identity_enabled=excluded.identity_enabled, "
                      "biometric_retention_days=excluded.biometric_retention_days, "
                      "updated_at=excluded.updated_at",
                      (1 if enabled else 0,
                       None if retention_days is None else int(retention_days), _now()))

    def save_subject(self, subject_id: str, label: str, embedding: Sequence[float],
                     model_sha256: str, *, enrolled_by: str, note: str | None = None) -> None:
        now = _now()
        with self.transaction() as c:
            c.execute("INSERT INTO subjects (id, label, note, embedding, model_sha256, enrolled_by, "
                      "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?) "
                      "ON CONFLICT(id) DO UPDATE SET label=excluded.label, note=excluded.note, "
                      "embedding=excluded.embedding, model_sha256=excluded.model_sha256, "
                      "updated_at=excluded.updated_at",
                      (subject_id, label, note, json.dumps([float(v) for v in embedding]),
                       model_sha256, enrolled_by, now, now))

    def subjects(self) -> list[sqlite3.Row]:
        self._check_thread()
        return self._connection.execute("SELECT * FROM subjects ORDER BY label").fetchall()

    def delete_subject(self, subject_id: str) -> int:
        """Remove a subject and, by cascade, every observation of them.

        The cascade is the point and it is in the schema rather than here: a
        person who asks to be removed from a biometric register is not served
        by having their row deleted and ten thousand observations of them left
        pointing at nothing.
        """
        with self.transaction() as c:
            return c.execute("DELETE FROM subjects WHERE id = ?", (subject_id,)).rowcount

    def save_face_observation(self, camera_id: str, track_id: int | None, at: int, *,
                              subject_id: str | None, distance: float | None,
                              margin: float | None, quality: float, model_sha256: str) -> None:
        with self.transaction() as c:
            c.execute("INSERT INTO face_observations (camera_id, track_id, at, subject_id, distance, "
                      "margin, quality, model_sha256) VALUES (?,?,?,?,?,?,?,?)",
                      (camera_id, track_id, at, subject_id, distance, margin, float(quality),
                       model_sha256))

    def save_plate_observation(self, camera_id: str, track_id: int | None, at: int, *,
                               text: str, characters: Sequence[float], confidence: float,
                               model_sha256: str) -> None:
        with self.transaction() as c:
            c.execute("INSERT INTO plate_observations (camera_id, track_id, at, text, characters, "
                      "confidence, model_sha256) VALUES (?,?,?,?,?,?,?)",
                      (camera_id, track_id, at, text,
                       json.dumps([round(float(v), 4) for v in characters]),
                       float(confidence), model_sha256))

    def face_observations(self, *, limit: int = 200) -> list[sqlite3.Row]:
        self._check_thread()
        return self._connection.execute(
            "SELECT * FROM face_observations ORDER BY at DESC LIMIT ?", (limit,)).fetchall()

    def plate_observations(self, *, limit: int = 200) -> list[sqlite3.Row]:
        self._check_thread()
        return self._connection.execute(
            "SELECT * FROM plate_observations ORDER BY at DESC LIMIT ?", (limit,)).fetchall()

    def biometric_counts(self) -> dict[str, int]:
        self._check_thread()
        return {
            "subjects": self._connection.execute("SELECT COUNT(*) FROM subjects").fetchone()[0],
            "faces": self._connection.execute("SELECT COUNT(*) FROM face_observations").fetchone()[0],
            "plates": self._connection.execute("SELECT COUNT(*) FROM plate_observations").fetchone()[0],
        }

    def sweep_biometrics(self, older_than_millis: int) -> dict[str, int]:
        """Delete face and plate observations older than a cut-off.

        Observations only. An enrolled subject is not swept: somebody put them
        there deliberately and removing them silently on a timer would be a
        different feature, and a surprising one. `delete_subject` is how a
        subject goes.
        """
        with self.transaction() as c:
            faces = c.execute("DELETE FROM face_observations WHERE at < ?", (older_than_millis,)).rowcount
            plates = c.execute("DELETE FROM plate_observations WHERE at < ?", (older_than_millis,)).rowcount
        return {"faces": max(0, faces), "plates": max(0, plates)}

    def save_site(self, name: str, timezone_name: str) -> None:
        with self.transaction() as c:
            # The columns are named on purpose: `INSERT INTO site VALUES (…)`
            # broke the moment the table grew a column, and every site row
            # with it. The threat labels are absent so naming a site cannot
            # quietly drop them.
            c.execute("INSERT INTO site (id, name, timezone, updated_at) VALUES ('site', ?, ?, ?) "
                      "ON CONFLICT(id) DO UPDATE SET name=excluded.name, timezone=excluded.timezone, "
                      "updated_at=excluded.updated_at",
                      (name, timezone_name, _now()))

    # -------------------------------------------------------------- backups

    def backup_to(self, destination: str | Path) -> Path:
        self._check_thread()
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        target = sqlite3.connect(str(destination))
        try:
            self._connection.backup(target)
        finally:
            target.close()
        digest = sha256_of(destination)
        destination.with_suffix(destination.suffix + ".sha256").write_text(f"{digest}  {destination.name}\n", encoding="utf-8")
        self.audit("system", "database.backup", str(destination), digest[:12])
        return destination


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_backup(path: str | Path) -> str:
    """The digest, or a `StoreError` saying which of the three checks failed."""
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file():
        raise StoreError(f"no backup at {path}")
    if not sidecar.is_file():
        raise StoreError(f"no checksum beside {path.name}; a backup without one cannot be trusted")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    actual = sha256_of(path)
    if actual != expected:
        raise StoreError(f"{path.name} does not match its checksum")
    probe = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        result = probe.execute("PRAGMA quick_check").fetchone()[0]
    finally:
        probe.close()
    if result != "ok":
        raise StoreError(f"{path.name} failed its integrity check: {result}")
    return actual


def restore_backup(backup: str | Path, database: str | Path) -> Path:
    """Verify, keep the current file beside, then replace. Never while the store is open."""
    verify_backup(backup)
    database = Path(database)
    if database.exists():
        aside = database.with_name(database.name + f".before-restore-{int(time.time())}")
        shutil.move(str(database), str(aside))
    for suffix in ("-wal", "-shm"):
        Path(str(database) + suffix).unlink(missing_ok=True)
    shutil.copy2(str(backup), str(database))
    return database


