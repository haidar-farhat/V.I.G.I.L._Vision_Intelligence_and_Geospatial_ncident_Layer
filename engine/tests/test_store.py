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

from datetime import datetime, time, timedelta, timezone, tzinfo
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


# ------------------------------------------------------------------- the site


def populate(store: Store) -> None:
    """A database with something in it, so a migration is tested against one.

    A migration that applies to an empty schema and destroys a populated one is
    the failure worth catching, and it is invisible to every test that migrates
    a database with no rows in it.
    """
    pose = CameraPose(position=SITE, mount_height=6.0, heading=90.0, pitch=-20.0)
    store.save_camera("cam-07", "North gate", "file:///media/north.mp4", pose)
    store.save_zone(
        Zone(
            id="zone-a",
            name="Restricted Area A",
            kind=ZoneKind.RESTRICTED,
            ring=tuple(destination_point(SITE, b, 30.0) for b in (0.0, 90.0, 180.0)),
        )
    )
    store.save_incident(Correlator().correlate([make_event(track=n) for n in (1, 2)])[0])
    store.audit("operator:alice", "camera.placed", "cam-07")


def make_site(**overrides) -> "Site":
    from sentinel.site import FrameKind, Site

    fields = dict(
        id="default",
        name="Beirut yard",
        origin=SITE,
        frame=FrameKind.GEOGRAPHIC,
        timezone="Asia/Beirut",
        boundary=tuple(
            destination_point(SITE, bearing, 60.0)
            for bearing in (45.0, 135.0, 225.0, 315.0)
        ),
    )
    fields.update(overrides)
    return Site(**fields)


def test_the_site_table_arrives_and_leaves_without_touching_the_evidence(store: Store):
    """The migration must apply, and undo, on a database that has rows in it.

    An air-gapped deployment steps back a version to diagnose something and
    steps forward again afterwards. If either direction took the cameras, zones
    or incidents with it, the diagnosis would cost the evidence — and no test
    over an empty schema would ever have shown it.
    """
    populate(store)
    store.save_site(make_site())
    before = store.applied_versions()
    events, incidents = store.event_count(), store.incident_count()

    # Down to and including `sites`, rather than one step. A single `rollback()`
    # only reached this migration while it happened to be the newest, and this
    # test broke the moment one was added after it — which is exactly when a
    # rollback test matters most.
    undone = store.rollback()
    while undone is not None and undone.name != "sites":
        undone = store.rollback()

    assert undone is not None and undone.name == "sites"
    assert "sites" not in store.table_names(), "the table survived its own down"
    assert store.event_count() == events, "rolling back the site took the events"
    assert store.incident_count() == incidents
    assert len(store.cameras()) == 1
    assert len(store.zones()) == 1
    assert len(store.audit_trail()) == 1

    store.migrate()

    assert store.applied_versions() == before
    assert "sites" in store.table_names()
    assert store.site() is None, "the site row is not resurrected by re-applying"
    store.save_site(make_site())
    assert store.site() is not None


def test_a_site_survives_a_round_trip_with_its_boundary_and_its_clock(store: Store):
    # The frame kind and the time zone are the two fields a reader can drop
    # silently: the first prints invented coordinates for a floor plan, the
    # second evaluates an after-hours schedule in the wrong clock.
    original = make_site()
    store.save_site(original)

    restored = store.site()

    assert restored is not None
    assert restored.id == original.id
    assert restored.name == original.name
    assert restored.origin.lat == pytest.approx(SITE.lat)
    assert restored.origin.lon == pytest.approx(SITE.lon)
    assert restored.frame is original.frame
    assert restored.timezone == "Asia/Beirut", "the site's clock did not survive"
    assert len(restored.boundary) == len(original.boundary)
    for restored_point, original_point in zip(restored.boundary, original.boundary):
        assert restored_point.lat == pytest.approx(original_point.lat)
        assert restored_point.lon == pytest.approx(original_point.lon)


def test_a_site_nobody_has_outlined_is_not_a_site_enclosing_nothing(store: Store):
    # Stored as NULL rather than '[]'. "Not drawn yet" means coverage cannot be
    # computed; "encloses nothing" means none of the site is covered, which is
    # an alarm — and a reader that conflates them raises the second for the
    # first.
    store.save_site(make_site(boundary=()))

    restored = store.site()
    assert restored is not None
    assert restored.boundary == ()
    assert restored.has_boundary is False

    stored = store._connection.execute("SELECT boundary_ring FROM sites").fetchone()
    assert stored["boundary_ring"] is None


def test_saving_a_site_twice_stores_it_once(store: Store):
    store.save_site(make_site(name="Beirut yard"))
    store.save_site(make_site(name="Beirut yard, north half"))

    assert len(store.sites()) == 1
    assert store.site().name == "Beirut yard, north half"


def test_the_origin_does_not_move_when_the_first_camera_is_removed(store: Store):
    """The bug this whole record exists for.

    The plan view anchored its frame on the first placed camera, so deleting
    that camera re-anchored everything and every zone, footprint and track
    jumped on screen. Nothing had moved; the ruler had. An origin in a row
    cannot be deleted by removing a camera.
    """
    first = CameraPose(
        position=destination_point(SITE, 90.0, 120.0),
        mount_height=6.0, heading=270.0, pitch=-20.0,
    )
    store.save_camera("cam-07", "North gate", "file:///media/north.mp4", first)
    store.save_site(make_site())
    origin = store.site().origin

    assert store.delete_camera("cam-07") is True

    after = store.site().origin
    assert after.lat == pytest.approx(origin.lat)
    assert after.lon == pytest.approx(origin.lon)
    assert after.lat == pytest.approx(SITE.lat), "the origin followed the camera"


def test_a_local_site_does_not_claim_to_be_a_place(store: Store):
    # A floor plan's origin is fixed but arbitrary. Distances on it are real;
    # its coordinates are not, and a screen that prints them is inventing
    # precision the geometry cannot support.
    from sentinel.site import FrameKind

    store.save_site(make_site(frame=FrameKind.LOCAL))

    restored = store.site()
    assert restored.frame is FrameKind.LOCAL
    assert restored.is_georeferenced is False


def test_a_two_point_boundary_is_refused_before_it_reaches_the_database():
    # Two points reach shapely as a line, whose area is zero, so every coverage
    # figure computed against it is a division by zero or a confident 0% — a
    # site reported as entirely unwatched because somebody clicked twice.
    from sentinel.site import SiteError

    with pytest.raises(SiteError):
        make_site(boundary=(SITE, destination_point(SITE, 90.0, 40.0)))


def test_a_site_declares_a_clock_that_can_actually_be_read():
    """The default site's clock has to work on the machine it ships to.

    Every fresh deployment starts on DEFAULT_TIMEZONE, and node.py asks the site
    for its clock before it can evaluate a single schedule. On Windows the tz
    database is an optional package, so a clock() that always went through
    ZoneInfo would leave a brand-new node with no clock at all — and tell the
    operator to install tzdata for the one zone that has no rules to look up.

    The two offsets are asserted six months apart so this cannot pass for a
    summer-shifting zone that merely happens to sit on zero in January.
    """
    from sentinel.site import DEFAULT_TIMEZONE

    site = make_site(timezone=DEFAULT_TIMEZONE)

    clock = site.clock()

    winter = datetime(2026, 1, 15, 12, 0)
    summer = datetime(2026, 7, 15, 12, 0)
    print("clock:", clock)
    print("utcoffset January:", clock.utcoffset(winter))
    print("utcoffset July:", clock.utcoffset(summer))

    assert isinstance(clock, tzinfo), "the site handed back something unusable"
    assert clock.utcoffset(winter) == timedelta(0)
    assert clock.utcoffset(summer) == timedelta(0), "the site's clock moved in July"
    assert winter.replace(tzinfo=clock).utcoffset() == timedelta(0)


def test_the_site_clock_says_so_rather_than_falling_back_to_utc():
    """An unresolvable zone must not become UTC in silence.

    That fallback is how a window typed as 18:00 armed at 21:00 local, with
    nothing on screen saying why. On Windows the tz database is not shipped with
    Python, so this is the ordinary case, not an exotic one.
    """
    from sentinel.site import SiteError

    site = make_site(timezone="Mars/Olympus_Mons")

    with pytest.raises(SiteError) as raised:
        site.clock()
    assert "Mars/Olympus_Mons" in str(raised.value)
    assert site.timezone == "Mars/Olympus_Mons", "the declared zone is still recorded"


# ------------------------------------------------------------- the site frame


def test_a_site_frame_round_trips_a_point_to_under_a_centimetre():
    """Metres out and coordinates back, without walking the site off its fence.

    A boundary is drawn in one direction and stored in the other, edit after
    edit, so a conversion that lost a millimetre a trip would move a fence over
    a season. Measured first and floored, never guessed.
    """
    from sentinel.core import haversine_distance
    from sentinel.site import SiteFrame

    frame = SiteFrame(SITE)
    worst = 0.0
    for bearing in range(0, 360, 7):
        for distance in (0.5, 25.0, 250.0, 1000.0, 5000.0):
            point = destination_point(SITE, float(bearing), distance)
            east, north = frame.to_xy(point)
            error = haversine_distance(point, frame.to_latlon(east, north))
            worst = max(worst, error)

    # Measured at 1.4e-9 m out to 5 km; the bound is six orders of magnitude
    # looser than that, and still far inside the centimetre this has to hold.
    print(f"worst round-trip error over 5 km: {worst * 1000:.9f} mm")
    assert worst < 1e-3, "a round trip lost more than a millimetre"


def test_the_origin_maps_to_the_origin():
    # The zero-distance branch: the bearing from a point to itself is
    # arbitrary, and an arbitrary bearing times a zero distance is harmless
    # only until somebody changes the multiplication.
    from sentinel.site import SiteFrame

    frame = SiteFrame(SITE)
    assert frame.to_xy(SITE) == (0.0, 0.0)
    assert frame.to_latlon(0.0, 0.0) == SITE


def test_the_site_frame_and_the_coverage_frame_are_the_same_conversion():
    """Two tangent planes that came to differ would be a bug nobody could find.

    `SiteFrame` is meant to replace `coverage._Frame`; while both exist they
    must agree exactly, or the plan view and the coverage report would draw the
    same gap in two places. Imported read-only — nothing in coverage is changed
    here.
    """
    from sentinel.coverage import _Frame
    from sentinel.site import SiteFrame

    mine = SiteFrame(SITE)
    theirs = _Frame(SITE)

    worst_xy = 0.0
    worst_latlon = 0.0
    for bearing in (0.0, 37.0, 90.0, 143.0, 180.0, 271.0, 355.0):
        for distance in (0.5, 25.0, 250.0, 1000.0):
            point = destination_point(SITE, bearing, distance)
            a, b = mine.to_xy(point), theirs.to_xy(point)
            worst_xy = max(worst_xy, abs(a[0] - b[0]), abs(a[1] - b[1]))

            back_mine = mine.to_latlon(*a)
            back_theirs = theirs.to_latlon(*b)
            worst_latlon = max(
                worst_latlon,
                abs(back_mine.lat - back_theirs.lat),
                abs(back_mine.lon - back_theirs.lon),
            )

    print(f"worst disagreement: {worst_xy:.3e} m, {worst_latlon:.3e} degrees")
    assert worst_xy == 0.0, "the two frames no longer do the same arithmetic"
    assert worst_latlon == 0.0


def test_a_stored_site_hands_out_the_frame_everything_should_share(store: Store):
    # Constructed from the site rather than per caller: two frames on two
    # origins are two answers to the same question, and the place they disagree
    # is the far corner of a large site, which is the corner nobody checks.
    from sentinel.site import SiteFrame

    store.save_site(make_site())
    restored = store.site()

    frame = restored.metric_frame()
    east, north = frame.to_xy(destination_point(SITE, 90.0, 100.0))

    print(f"100 m due east reads as east={east:.4f} m, north={north:.4f} m")
    assert east == pytest.approx(100.0, abs=0.01)
    assert abs(north) < 0.01
    assert SiteFrame.of(restored).origin == frame.origin


# ------------------------------------------------------------------ the register


def register_schema(connection) -> dict[str, str]:
    """Every register object's definition, with comments and spacing removed.

    Compared as text rather than as a list of table names, because the parts
    that carry the weight are the CHECK constraints. A migrated database that
    accepted a MATCH sighting with no score while a fresh one refused it would
    hold — on upgraded deployments only — exactly the claim-without-evidence the
    register exists to make unrepresentable, and a comparison of table names
    would call the two databases identical.
    """
    rows = connection.execute(
        "SELECT name, sql FROM sqlite_master WHERE name LIKE 'register%' "
        "ORDER BY name"
    ).fetchall()
    return {row[0]: flattened(row[1]) for row in rows}


def flattened(sql: str) -> str:
    """One line, no comments: the schema as SQLite will enforce it."""
    kept = [line for line in sql.splitlines() if not line.strip().startswith("--")]
    return " ".join(" ".join(kept).split())


def test_a_fresh_database_and_an_upgraded_one_hold_the_same_register(tmp_path: Path):
    """The migration must build what `registry.create_schema` builds, exactly.

    Two ways into one schema is two schemas eventually, and the half that drifts
    is the half holding face templates. Checked against a database that predates
    the register — rolled back below it and brought forward again — because that
    is the deployment the difference would appear on, and never on the developer
    machine where every database is fresh.
    """
    import sqlite3

    from sentinel.registry import create_schema

    with Store(tmp_path / "old.db") as upgraded:
        populate(upgraded)
        while any(name.startswith("register_") for name in upgraded.table_names()):
            assert upgraded.rollback() is not None, "the register was never undone"
        assert "register_subjects" not in upgraded.table_names()

        upgraded.migrate()
        migrated = register_schema(upgraded._connection)

    bare = sqlite3.connect(":memory:")
    try:
        create_schema(bare)
        fresh = register_schema(bare)
    finally:
        bare.close()

    print(sorted(migrated))
    assert "register_subjects" in migrated, "the migration created no register"
    assert migrated == fresh, "an upgraded database is not the schema a new one gets"


def test_the_register_arrives_and_leaves_without_touching_the_evidence(store: Store):
    """The migration must apply, and undo, on a database that has rows in it.

    An air-gapped deployment steps back a version to diagnose something and
    steps forward again afterwards. If either direction took the cameras, zones,
    incidents or the audit log with it, the diagnosis would cost the evidence.
    """
    from sentinel.registry import Plate

    populate(store)
    store.register.enrol(
        subject_id="veh-1",
        display_name="Contractor van",
        identifier=Plate("B 7421"),
        actor="operator:alice",
        basis="site access list",
    )
    before = store.applied_versions()
    events, incidents = store.event_count(), store.incident_count()

    undone = store.rollback()
    while undone is not None and undone.name != "register":
        undone = store.rollback()

    assert undone is not None and undone.name == "register"
    assert [n for n in store.table_names() if n.startswith("register_")] == [], (
        "a register table survived its own down"
    )
    assert store.event_count() == events, "rolling back the register took the events"
    assert store.incident_count() == incidents
    assert len(store.cameras()) == 1
    assert len(store.zones()) == 1
    assert len(store.audit_trail()) == 1

    store.migrate()

    assert store.applied_versions() == before
    assert store.register.subjects() == (), (
        "an enrolment came back from a table that had been dropped"
    )
    store.register.enrol(
        subject_id="veh-1",
        display_name="Contractor van",
        identifier=Plate("B 7421"),
        actor="operator:alice",
        basis="site access list",
    )
    assert store.register.find_plate("B-7421") is not None


def test_an_enrolment_made_through_the_store_survives_a_reopen(tmp_path: Path):
    """The register is reachable from the store, and what it writes is durable.

    Reachability is the point. Until this property existed the register was a
    tested module with no way to an operator, because building one needs a
    connection and the only honest connection is the store's own.
    """
    from sentinel.registry import Plate

    database = tmp_path / "n.db"
    with Store(database) as store:
        enrolment = store.register.enrol(
            subject_id="veh-1",
            display_name="Contractor van",
            identifier=Plate("b 7421", frames_agreeing=6),
            actor="operator:alice",
            basis="site access list",
        )
        assert enrolment.created_subject
        # The audit row gets ids, kinds and counts. The plate and the name stay
        # in the register, where forgetting can reach them.
        store.audit(
            "operator:alice", enrolment.action, enrolment.subject.id, enrolment.detail()
        )
        assert "7421" not in (store.audit_trail()[0]["detail"] or "")

    with Store(database) as again:
        subject = again.register.find_plate("B-7421")

        assert subject is not None, "the enrolment did not survive the reopen"
        assert subject.display_name == "Contractor van"
        identifiers = again.register.identifiers("veh-1")
        assert [identifier.plate for identifier in identifiers] == ["B7421"]
        assert identifiers[0].raw_text == "b 7421", "the characters read were lost"
        assert identifiers[0].basis == "site access list"
        assert identifiers[0].enrolled_by == "operator:alice"


def test_the_register_writes_inside_the_stores_own_unit_of_work(store: Store):
    """One connection, so an enrolment and the row proving it land together.

    A `Register` built by a caller on a second connection to the same file would
    commit on its own: the enrolment would survive a failure that rolled back
    everything written beside it, and the database would hold a template with no
    audit row saying who put it there.
    """
    from sentinel.registry import Plate

    assert store.register is store.register, "each call built another register"

    with pytest.raises(RuntimeError, match="the paperwork failed"):
        with store.transaction():
            store.register.enrol(
                subject_id="veh-1",
                display_name="Contractor van",
                identifier=Plate("B 7421"),
                actor="operator:alice",
                basis="site access list",
            )
            raise RuntimeError("the paperwork failed")

    assert store.register.subjects() == (), (
        "the enrolment committed on its own connection, outside the failed unit "
        "of work"
    )


# -------------------------------------------------------- structured audit rows


def zone_pair() -> tuple[Zone, Zone]:
    """One zone before and after an edit that changes exactly one field."""
    ring = tuple(destination_point(SITE, bearing, 30.0) for bearing in (0.0, 90.0, 180.0))
    before = Zone(id="zone-a", name="Yard", kind=ZoneKind.RESTRICTED, ring=ring)
    return before, Zone(id="zone-a", name="Yard", kind=ZoneKind.EXCLUSION, ring=ring)


def make_record(**overrides):
    from sentinel.auditing import AuditRecord

    before, after = zone_pair()
    fields = dict(
        actor="operator:alice",
        action="zone.changed",
        subject="zone-a",
        node_id="gatehouse",
        at=datetime(2026, 3, 1, 9, 30, tzinfo=timezone.utc),
        before=before,
        after=after,
    )
    fields.update(overrides)
    return AuditRecord.of(**fields)


def test_a_structured_audit_row_keeps_the_prose_and_the_states_behind_it(store: Store):
    # The sentence is what an operator reads; the states are what makes the row
    # answerable to a question nobody asked at the time.
    import json

    record = make_record()

    head = store.audit_record(record)

    row = store.audit_trail()[0]
    print(row["detail"])
    assert row["detail"] == "kind RESTRICTED -> EXCLUSION"
    assert row["actor"] == "operator:alice" and row["node_id"] == "gatehouse"
    assert row["at"] == int(record.at.timestamp() * 1000), "the row moved in time"
    before = json.loads(row["before_json"])
    after = json.loads(row["after_json"])
    assert {key for key in before if before[key] != after[key]} == {"kind"}
    assert row["chain_hash"] == head


def test_the_chain_folds_each_record_into_the_next(store: Store):
    """Editing one row must break every hash after it, and say which row.

    A chain that only covered the newest row would detect nothing: altering an
    audit log means altering something old.
    """
    from sentinel.auditing import verify_chain

    first = make_record(subject="zone-a")
    second = make_record(subject="zone-b")
    hashes = [store.audit_record(first), store.audit_record(second)]

    assert hashes[0] != hashes[1]
    assert store.audit_chain_head() == hashes[1]
    assert verify_chain([first, second], hashes) is None

    tampered = make_record(subject="zone-a", actor="operator:mallory")
    assert verify_chain([tampered, second], hashes) == 0, (
        "an edited record verified, so the chain protects nothing"
    )


def test_a_prose_only_row_leaves_the_chain_where_it_was(store: Store):
    # `audit` has no before-state to record and writes no hash, and the chain is
    # honest about covering only the rows that carry one. Claiming the whole log
    # is chained when half of it is not is the failure this pins.
    head = store.audit_record(make_record())

    store.audit("node", "node.started", "gatehouse")

    assert store.audit_chain_head() == head, "a row with no hash broke the chain"
    assert store.audit_trail()[0]["chain_hash"] is None


def test_an_audit_row_written_before_the_columns_existed_still_reads(store: Store):
    """Nullable is load-bearing here, not lenient.

    The audit log is append-only, so there is no pass that could go back and
    fill a before-state in for an edit made last year. A NOT NULL column would
    either fail the migration or force this code to invent one.
    """
    undone = store.rollback()
    while undone is not None and undone.name != "audit_records":
        undone = store.rollback()
    assert undone is not None and undone.name == "audit_records"
    assert "before_json" not in store.column_names("audit_logs")

    store.audit("operator:alice", "camera.placed", "cam-07", "6 m mast, bearing 145")
    store.migrate()

    row = store.audit_trail()[0]
    assert row["actor"] == "operator:alice"
    assert row["detail"] == "6 m mast, bearing 145", "the old row lost its line"
    assert row["before_json"] is None and row["after_json"] is None
    assert row["node_id"] is None and row["chain_hash"] is None
    assert store.audit_chain_head() is None

    # And a structured row written after it starts the chain rather than failing
    # on a predecessor that has no hash.
    head = store.audit_record(make_record())
    assert head and store.audit_chain_head() == head
