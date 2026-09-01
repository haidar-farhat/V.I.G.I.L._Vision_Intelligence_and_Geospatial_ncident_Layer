"""Tests for persistence.

Three groups, in descending order of how badly a failure would matter:

**The schema must not hold secrets.** A test walks every column in the database
and fails on anything credential-shaped. That check is worth more than any
review, because it keeps working when somebody adds a table in eighteen months.

**Writes must be idempotent.** Ids are deterministic, so a replayed batch after
an outage has to upsert rather than duplicate. If that stops being true, a
control node reconnecting after a weekend produces a second copy of every
incident.

**Reads must lose nothing.** A read that quietly drops a field is worse than one
that fails: the evidence looks complete and is not.
"""

from __future__ import annotations

from datetime import datetime, time, timezone
from pathlib import Path

import pytest

from sentinel.core import CameraPose, LatLon, destination_point
from sentinel.events import Event, Evidence, EventType, Severity
from sentinel.incidents import Correlator
from sentinel.store import MIGRATIONS, Store, StoreError
from sentinel.zones import Schedule, Zone, ZoneKind

SITE = LatLon(33.8938, 35.5018)


@pytest.fixture
def store() -> Store:
    with Store(":memory:") as db:
        yield db


def make_event(
    *,
    camera: str = "cam-07",
    track: int = 1,
    at_millis: int = 1000,
    event_id: str | None = None,
    placed: bool = True,
    classifies: bool = False,
) -> Event:
    evidence = Evidence(
        camera_id=camera,
        track_id=track,
        first_seen_millis=at_millis,
        last_seen_millis=at_millis + 3000,
        observations=17,
        detector="yolo-test" if classifies else "MOG2 background subtraction",
        detector_classifies=classifies,
        model_digest="c" * 64 if classifies else None,
        class_label="person" if classifies else "unclassified",
        latitude=SITE.lat if placed else None,
        longitude=SITE.lon if placed else None,
        position_uncertainty_meters=1.75 if placed else None,
        position_source="GROUND_PROJECTION" if placed else None,
        speed_mps=1.35,
        heading_degrees=88.5,
        frame_indices=(12, 13, 14),
    )
    return Event(
        id=event_id or f"ev_{camera}_{track}_{at_millis}",
        type=EventType.ZONE_ENTRY,
        severity=Severity.HIGH,
        summary="An object entered Restricted Area A",
        occurred_at_millis=at_millis,
        occurred_at=datetime(2026, 8, 30, 3, 0, tzinfo=timezone.utc),
        zone_id="zone-a",
        zone_name="Restricted Area A",
        rule_id="zone-entry",
        evidence=evidence,
        triggering_conditions=("membership held for 600 ms", "zone kind is RESTRICTED"),
        confidence=0.875,
    )


# ------------------------------------------------------------------- no secrets


#: Substrings that must never name a column. If one is genuinely needed, this
#: list is the place the argument has to be had.
FORBIDDEN = ("password", "passwd", "secret", "credential_value", "token", "api_key", "private_key")


def test_no_column_in_the_schema_is_credential_shaped(store: Store):
    """The check that keeps working after everyone here has forgotten the rule.

    ``cameras.credentials_ref`` is allowed because it is a handle, not a value —
    an opaque key into the operating system's keychain. Everything else that
    reads like a secret is refused.
    """
    offending = []
    for table in store.table_names():
        for column in store.column_names(table):
            lowered = column.lower()
            if lowered == "credentials_ref":
                continue
            if any(word in lowered for word in FORBIDDEN):
                offending.append(f"{table}.{column}")

    assert offending == [], f"credential-shaped columns: {offending}"


def test_a_camera_stores_a_handle_not_a_password(store: Store):
    """The credential must not survive the trip into the database.

    An earlier version of this test asserted `"hunter2" not in joined` while
    never putting "hunter2" anywhere — it checked for the absence of a string it
    had not introduced, which is true of almost any database and proves nothing.
    It now pushes a real credential through the real redactor and looks for that
    credential, so a redactor that stopped working would fail here.
    """
    from sentinel.decode import contains_credential, redact_url

    raw = "rtsp://admin:hunter2-not-a-real-password@10.20.30.40:554/Streaming/Channels/101"
    store.save_camera(
        "cam-07",
        "North gate",
        source=redact_url(raw),
        credentials_ref="keychain://sentinel/cam-07",
    )

    row = store.cameras()[0]
    joined = " ".join(str(value) for value in tuple(row))

    assert not contains_credential(joined, raw), (
        "the camera row carries a secret from the source URL"
    )
    assert "10.20.30.40:554" in row["source"], "the row must still identify the camera"
    assert row["credentials_ref"].startswith("keychain://")


def test_the_store_refuses_nothing_and_that_is_the_point(store: Store):
    """`save_camera` cannot validate what it is given, so the boundary is above it.

    This is recorded as a test because it is a real design decision rather than
    an oversight: the store takes a string. Redaction happens at the one place
    that has the raw URL — decode.py — and every caller is expected to have gone
    through it. The test above proves the console's path does; this one states
    plainly that the store itself is not a second line of defence.
    """
    store.save_camera("cam-99", "Careless", source="rtsp://admin:leaked@10.0.0.1/s")

    assert "leaked" in store.cameras()[-1]["source"], (
        "if this ever starts passing by redaction inside the store, the comment "
        "above is wrong and the boundary has moved"
    )


def test_every_stored_position_carries_its_uncertainty(store: Store):
    # Storing a coordinate without its error is how false precision gets into a
    # map, and from there into a decision about where to send somebody.
    store.save_events([make_event()])

    columns = store.column_names("events")
    assert "latitude" in columns
    assert "uncertainty_meters" in columns
    assert "position_source" in columns

    stored = store.events()[0]
    assert stored.evidence.position_uncertainty_meters is not None
    assert stored.evidence.position_source == "GROUND_PROJECTION"


# ------------------------------------------------------------------ migrations


def test_a_new_database_is_migrated_to_the_current_schema(store: Store):
    assert store.applied_versions() == [m.version for m in MIGRATIONS]
    assert store.pending() == []


def test_migrating_twice_changes_nothing(store: Store):
    assert store.migrate() == []


def test_a_migration_can_be_undone(store: Store):
    """An upgrade that cannot be reversed on an air-gapped machine is a gamble.

    Asserted as the general property rather than against one migration's
    contents. An earlier version checked that rolling back removed the `events`
    table, which was only true while `events` happened to be in the newest
    migration — it broke the moment a second one was added, which is exactly
    when a rollback test matters most.
    """
    before = store.applied_versions()
    assert before, "nothing was applied, so nothing was checked"

    undone = store.rollback()

    assert undone is not None
    assert undone.version == before[-1], "rollback must undo the most recent"
    assert store.applied_versions() == before[:-1]

    store.migrate()
    assert store.applied_versions() == before, "re-applying did not restore the schema"


def test_every_migration_can_be_undone_and_reapplied(store: Store):
    # All the way down and all the way back, so a `down` that was never run is
    # not discovered to be broken during an actual downgrade.
    original = store.applied_versions()

    while store.rollback() is not None:
        pass
    assert store.applied_versions() == []

    store.migrate()
    assert store.applied_versions() == original


def test_a_camera_pose_keeps_its_roll(store: Store):
    # Roll was accepted, dropped on the way in, and defaulted to 0.0 on the way
    # out — so the store handed back a different pose than it was given.
    pose = CameraPose(
        position=SITE, mount_height=6.0, heading=90.0, pitch=-20.0, roll=-3.5,
    )
    store.save_camera("cam-tilt", "Tilted", "file:///media/tilt.mp4", pose)

    restored = store.camera_pose("cam-tilt")
    assert restored is not None
    assert restored.roll == pytest.approx(-3.5)


def test_migrations_are_applied_in_version_order(store: Store):
    versions = [m.version for m in MIGRATIONS]
    assert versions == sorted(versions)
    assert len(set(versions)) == len(versions), "two migrations share a version"


def test_every_migration_carries_a_way_back():
    for migration in MIGRATIONS:
        assert migration.down.strip(), f"migration {migration.version} has no down"


def test_foreign_keys_are_enforced(store: Store):
    # Off by default in SQLite, which silently permits an incident pointing at
    # evidence that was deleted.
    with pytest.raises(Exception):
        with store.transaction() as connection:
            connection.execute(
                "INSERT INTO incident_events (incident_id, event_id) VALUES (?, ?)",
                ("inc_nothing", "ev_nothing"),
            )


# -------------------------------------------------------------- idempotency


def test_writing_the_same_event_twice_stores_it_once(store: Store):
    # The property that makes at-least-once delivery safe. A worker resending
    # everything after the last acknowledged sequence must not double the
    # database.
    event = make_event()

    store.save_events([event])
    store.save_events([event])

    assert store.event_count() == 1


def test_replaying_a_whole_batch_converges(store: Store):
    batch = [make_event(track=n, at_millis=n * 1000) for n in range(1, 6)]

    store.save_events(batch)
    store.save_events(batch)
    store.save_events(list(reversed(batch)))

    assert store.event_count() == 5


def test_an_updated_event_replaces_rather_than_duplicates(store: Store):
    first = make_event()
    store.save_events([first])

    revised = Event(**{**first.__dict__, "confidence": 0.5}) if False else Event(
        id=first.id,
        type=first.type,
        severity=Severity.CRITICAL,
        summary=first.summary,
        occurred_at_millis=first.occurred_at_millis,
        occurred_at=first.occurred_at,
        zone_id=first.zone_id,
        zone_name=first.zone_name,
        rule_id=first.rule_id,
        evidence=first.evidence,
        triggering_conditions=first.triggering_conditions,
        confidence=0.5,
    )
    store.save_events([revised])

    assert store.event_count() == 1
    assert store.events()[0].severity is Severity.CRITICAL
    assert store.events()[0].confidence == pytest.approx(0.5)


def test_saving_an_incident_twice_stores_it_once(store: Store):
    incident = Correlator().correlate([make_event(track=n) for n in (1, 2)])[0]

    store.save_incident(incident)
    store.save_incident(incident)

    assert store.incident_count() == 1
    assert len(store.incident_events(incident.id)) == 2


# ------------------------------------------------------------ nothing is lost


def test_an_event_survives_a_round_trip_intact(store: Store):
    original = make_event(classifies=True)
    store.save_events([original])

    restored = store.events()[0]

    assert restored.id == original.id
    assert restored.type is original.type
    assert restored.severity is original.severity
    assert restored.summary == original.summary
    assert restored.zone_id == original.zone_id
    assert restored.rule_id == original.rule_id
    assert restored.confidence == pytest.approx(original.confidence)
    assert restored.triggering_conditions == original.triggering_conditions
    assert restored.occurred_at == original.occurred_at

    assert restored.evidence == original.evidence, (
        "the evidence bundle lost or changed a field on the way through"
    )


def test_an_unplaced_event_stays_unplaced(store: Store):
    # None must not become 0.0 in the database and then a coordinate off West
    # Africa on the map.
    store.save_events([make_event(placed=False)])
    restored = store.events()[0]

    assert restored.evidence.latitude is None
    assert restored.evidence.position_uncertainty_meters is None
    assert restored.evidence.position_source is None


def test_a_zone_survives_a_round_trip(store: Store):
    zone = Zone(
        id="zone-a",
        name="Restricted Area A",
        kind=ZoneKind.RESTRICTED,
        ring=tuple(destination_point(SITE, b, 30.0) for b in (0.0, 90.0, 180.0, 270.0)),
        schedule=Schedule(time(18, 0), time(6, 0), frozenset({1, 2, 3})),
        enter_after_millis=750,
        exit_after_millis=2500,
        accept_uncertain=True,
    )
    store.save_zone(zone)

    restored = store.zones()[0]
    assert restored.id == zone.id
    assert restored.kind is zone.kind
    assert restored.enter_after_millis == 750
    assert restored.accept_uncertain is True
    assert restored.schedule is not None
    assert restored.schedule.start == time(18, 0)
    assert restored.schedule.days == frozenset({1, 2, 3})
    assert len(restored.ring) == len(zone.ring)


def test_a_camera_pose_survives_a_round_trip(store: Store):
    pose = CameraPose(
        position=SITE, mount_height=6.5, heading=145.0, pitch=-24.0,
        horizontal_fov=58.0, vertical_fov=33.0, range_meters=110.0,
    )
    store.save_camera("cam-07", "North gate", "file:///media/north.mp4", pose)

    restored = store.camera_pose("cam-07")
    assert restored is not None
    assert restored.mount_height == pytest.approx(6.5)
    assert restored.heading == pytest.approx(145.0)
    assert restored.range_meters == pytest.approx(110.0)


def test_an_unplaced_camera_has_no_pose(store: Store):
    store.save_camera("cam-08", "Yard", "file:///media/yard.mp4")
    assert store.camera_pose("cam-08") is None


# -------------------------------------------------------------------- time


def test_observed_and_recorded_time_are_kept_apart(store: Store):
    # Never reconciled. A camera with a drifting clock is a fact about the
    # deployment worth keeping, not noise to smooth away.
    store.save_events([make_event()])

    columns = store.column_names("events")
    assert "occurred_at" in columns
    assert "recorded_at" in columns

    skew = store.clock_skew("cam-07")
    assert len(skew) == 1
    assert skew[0] != 0, "the two clocks were silently made to agree"


# ------------------------------------------------------------------- incidents


def test_an_incident_and_its_events_are_written_together(store: Store):
    # An incident referring to events that were not written is a dangling
    # reference; events without the incident that explains them lose the
    # reasoning.
    incident = Correlator().correlate([make_event(track=n, at_millis=n * 800) for n in (1, 2, 3)])[0]
    store.save_incident(incident)

    assert store.incident_count() == 1
    assert store.event_count() == 3
    assert len(store.incident_events(incident.id)) == 3


def test_the_object_count_is_stored_separately_from_the_segments(store: Store):
    # Conflating them titles an incident "6 people" when three walked past.
    incident = Correlator().correlate([make_event(track=n) for n in (1, 2, 3)])[0]
    store.save_incident(incident)

    row = store.incidents()[0]
    assert row["distinct_object_count"] == incident.distinct_objects
    assert len(store.incident_events(incident.id)) == 3


def test_an_incident_keeps_its_risk_reasoning(store: Store):
    from sentinel.store import risk_from_row

    incident = Correlator().correlate([make_event()])[0]
    store.save_incident(incident)

    risk = risk_from_row(store.incidents()[0])
    assert risk.score == pytest.approx(incident.risk.score)
    assert [f.name for f in risk.factors] == [f.name for f in incident.risk.factors]
    assert all(f.because for f in risk.factors)


# ----------------------------------------------------------------------- audit


def test_the_audit_log_records_who_did_what(store: Store):
    store.audit("operator:alice", "camera.placed", "cam-07", "6 m mast, bearing 145")

    trail = store.audit_trail()
    assert len(trail) == 1
    assert trail[0]["actor"] == "operator:alice"
    assert trail[0]["action"] == "camera.placed"


def test_the_store_offers_no_way_to_edit_the_audit_log(store: Store):
    # An audit log that can be edited is not one. There is deliberately no
    # method for it, and this test fails if somebody adds one.
    methods = [name for name in dir(store) if not name.startswith("__")]
    dangerous = [
        name for name in methods
        if "audit" in name and any(word in name for word in ("delete", "update", "clear", "purge"))
    ]
    assert dangerous == []


# ------------------------------------------------------------------- on disk


def test_a_database_on_disk_survives_being_reopened(tmp_path: Path):
    # The entire point. Nothing above this line proves anything survives a
    # restart, because an in-memory database never has to.
    path = tmp_path / "sentinel.db"

    with Store(path) as first:
        first.save_events([make_event()])
        incident = Correlator().correlate([make_event(track=2, at_millis=2000)])[0]
        first.save_incident(incident)

    with Store(path) as second:
        assert second.event_count() == 2
        assert second.incident_count() == 1
        assert second.events()[0].evidence.detector


def test_the_database_directory_is_created_if_absent(tmp_path: Path):
    path = tmp_path / "nested" / "deeper" / "sentinel.db"
    with Store(path) as store:
        assert store.event_count() == 0
    assert path.exists()


def test_a_failed_transaction_leaves_nothing_behind(store: Store):
    store.save_events([make_event()])

    with pytest.raises(RuntimeError):
        with store.transaction() as connection:
            connection.execute(
                "INSERT INTO audit_logs (at, actor, action) VALUES (1, 'a', 'b')"
            )
            raise RuntimeError("something went wrong halfway through")

    assert store.audit_trail() == []
    assert store.event_count() == 1, "the earlier write was rolled back too"


# --------------------------------------------------- correlation is not stable


def test_merging_two_incidents_does_not_leave_the_old_ones_behind(store: Store):
    """Correlation is not stable across runs, and the database has to survive that.

    A later batch can merge two incidents into one, and that one has a different
    deterministic id. Without cleanup the superseded rows survive forever and
    the same event is linked to two incidents — double-counting on every screen
    that reads them.
    """
    from sentinel.core import destination_point

    far = make_event(camera="cam-07", track=1, at_millis=0)
    near = make_event(camera="cam-08", track=1, at_millis=200_000)

    # Two separate incidents first: far apart in time.
    separate = Correlator().correlate([far])
    separate += Correlator().correlate([near])
    assert len(separate) == 2
    for incident in separate:
        store.save_incident(incident)
    assert store.incident_count() == 2

    # Now a run that sees both together and merges them.
    merged = Correlator(window_millis=10_000_000).correlate([far, near])
    assert len(merged) == 1, "the fixture did not actually merge them"
    store.save_incident(merged[0])

    assert store.incident_count() == 1, "a superseded incident survived the merge"
    assert len(store.incident_events(merged[0].id)) == 2

    # And no event belongs to two incidents.
    rows = store._connection.execute(
        "SELECT event_id, COUNT(*) AS n FROM incident_events GROUP BY event_id"
    ).fetchall()
    assert all(row["n"] == 1 for row in rows), "an event is linked to two incidents"


def test_an_updated_event_refreshes_everything_that_can_change(store: Store):
    """A re-sent event must replace the row, not blend with it.

    The upsert used to refresh two columns and leave the position, motion,
    class, zone and frames from the earlier pass — producing a row that was true
    of neither observation and was then exported as evidence.
    """
    from dataclasses import replace

    first = make_event(track=1, at_millis=1000)
    store.save_events([first])

    revised = replace(
        first,
        zone_name="Restricted Area B",
        evidence=replace(
            first.evidence,
            observations=99,
            latitude=34.0,
            longitude=36.0,
            position_uncertainty_meters=9.5,
            speed_mps=None,
            heading_degrees=None,
            class_label="person",
            detector_classifies=True,
            frame_indices=(99,),
        ),
    )
    store.save_events([revised])

    stored = store.events()[0]
    assert store.event_count() == 1
    assert stored.zone_name == "Restricted Area B"
    assert stored.evidence.observations == 99
    assert stored.evidence.latitude == pytest.approx(34.0)
    assert stored.evidence.position_uncertainty_meters == pytest.approx(9.5)
    assert stored.evidence.speed_mps is None, "stale motion survived the update"
    assert stored.evidence.class_label == "person"
    assert stored.evidence.frame_indices == (99,)


def test_recorded_at_is_not_rewritten_by_a_re_send(store: Store):
    # `recorded_at` is when this node FIRST durably accepted the event. A worker
    # replaying its buffer after an outage must not move it.
    event = make_event()
    store.save_events([event])
    first = store._connection.execute("SELECT recorded_at FROM events").fetchone()[0]

    store.save_events([event])
    second = store._connection.execute("SELECT recorded_at FROM events").fetchone()[0]

    assert first == second, "a re-send rewrote when the event was first accepted"
