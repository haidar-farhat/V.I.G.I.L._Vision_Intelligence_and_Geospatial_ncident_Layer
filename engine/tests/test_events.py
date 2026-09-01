"""Tests for the event layer.

An event is the first thing this system produces that is meant to interrupt a
person, so these tests are mostly about restraint and about proof:

- an event carries the grounds for itself, not a promise that grounds exist;
- an ongoing condition produces one event, not one per frame;
- identity is derived from what the event is, so replay is idempotent;
- a claim is never more specific than the detector can support;
- a rule declines to fire when the measurement cannot carry it.
"""

from __future__ import annotations

from datetime import datetime, time, timezone

import pytest

from sentinel.core import (
    BoundingBox,
    LatLon,
    PositionEstimate,
    Track,
    destination_point,
)
from sentinel.detect import UNCLASSIFIED, DetectorInfo
from sentinel.events import (
    AfterHoursRule,
    Event,
    EventEngine,
    EventType,
    LoiteringRule,
    RapidMovementRule,
    Rule,
    Severity,
    ZoneEntryRule,
    event_id,
    severity_rank,
    utc_from_millis,
)
from sentinel.zones import Presence, PresenceChange, Schedule, Zone, ZoneKind
from test_zones import make_track, ring_around

SITE = LatLon(33.8938, 35.5018)
MOMENT = datetime(2026, 8, 30, 3, 0, tzinfo=timezone.utc)

MOTION = DetectorInfo(kind="motion", name="MOG2 background subtraction", classifies=False)
MODEL = DetectorInfo(
    kind="onnx",
    name="yolo-test",
    model_path="/models/yolo-test.onnx",
    model_sha256="a" * 64,
    class_names={0: "person"},
    classifies=True,
)


def restricted(**overrides) -> Zone:
    defaults = dict(
        id="zone-a",
        name="Restricted Area A",
        kind=ZoneKind.RESTRICTED,
        ring=ring_around(SITE, 40.0),
        schedule=None,
        enter_after_millis=600,
    )
    defaults.update(overrides)
    return Zone(**defaults)  # type: ignore[arg-type]


def presence(track_id: int = 1, *, started: int = 0, last: int = 2000, uncertain: int = 0) -> Presence:
    return Presence(
        zone_id="zone-a",
        track_id=track_id,
        started_millis=started,
        last_present_millis=last,
        confirmed=True,
        observations=10,
        uncertain_observations=uncertain,
    )


def engine(*rules, camera: str = "cam-07") -> EventEngine:
    return EventEngine(list(rules), node_id="nd_test", camera_id=camera)


def fire_entry(rules, zone: Zone, track: Track, detector=MOTION, at: int = 2000) -> list[Event]:
    change = PresenceChange("ENTERED", presence(track.id), at)
    return engine(*rules).on_presence_changes(
        [change], {zone.id: zone}, {track.id: track},
        at_millis=at, moment=MOMENT, detector=detector, frame_index=30,
    )


# ------------------------------------------------------------------- identity


def test_the_same_event_gets_the_same_id_every_time():
    # Replay must be idempotent, or a re-examined incident becomes two incidents.
    args = ("nd_1", "cam-07", "zone-entry", EventType.ZONE_ENTRY, 4, 12_345)
    assert event_id(*args) == event_id(*args)


def test_different_events_get_different_ids():
    base = ("nd_1", "cam-07", "zone-entry", EventType.ZONE_ENTRY, 4, 12_345)

    variants = [
        ("nd_2", *base[1:]),
        (base[0], "cam-08", *base[2:]),
        (*base[:2], "loitering", *base[3:]),
        (*base[:3], EventType.LOITERING, *base[4:]),
        (*base[:4], 5, base[5]),
        (*base[:5], 99_999),
    ]
    ids = {event_id(*base)} | {event_id(*v) for v in variants}

    assert len(ids) == len(variants) + 1


def test_timestamps_within_the_same_bucket_collapse():
    # Two nodes observing the same event will not agree on the millisecond.
    base = ("nd_1", "cam-07", "zone-entry", EventType.ZONE_ENTRY, 4)

    assert event_id(*base, 12_000) == event_id(*base, 12_999)
    assert event_id(*base, 12_000) != event_id(*base, 13_000)


def test_an_id_does_not_depend_on_when_it_was_recorded():
    # Nothing in the material is a wall clock or a counter, which is what makes
    # the same footage produce the same ids on a different day.
    first = event_id("nd_1", "cam-07", "r", EventType.ZONE_ENTRY, 1, 5000)
    second = event_id("nd_1", "cam-07", "r", EventType.ZONE_ENTRY, 1, 5000)

    assert first == second and first.startswith("ev_")


# ------------------------------------------------------------------- evidence


def test_an_event_carries_the_grounds_for_itself():
    track = make_track(1, SITE, uncertainty=1.4, first=0, last=2000, speed=1.2)
    events = fire_entry([ZoneEntryRule()], restricted(), track, detector=MODEL)

    assert len(events) == 1
    evidence = events[0].evidence

    assert evidence.camera_id == "cam-07"
    assert evidence.track_id == 1
    assert evidence.detector == "yolo-test"
    assert evidence.model_digest == "a" * 64
    assert evidence.class_label == "person"
    assert evidence.position_uncertainty_meters == pytest.approx(1.4)
    assert evidence.position_source == "GROUND_PROJECTION"
    assert evidence.speed_mps == pytest.approx(1.2)
    assert evidence.frame_indices == (30,)


def test_an_event_states_why_it_fired():
    track = make_track(1, SITE)
    events = fire_entry([ZoneEntryRule()], restricted(), track)

    assert events[0].triggering_conditions
    assert any("RESTRICTED" in c for c in events[0].triggering_conditions)


def test_an_event_reports_its_confidence_from_the_presence():
    # A presence built half from uncertain positions is a weaker claim, and the
    # event must say so rather than presenting it as equally solid.
    change = PresenceChange("ENTERED", presence(uncertain=5), 2000)
    zone = restricted()
    track = make_track(1, SITE)

    events = engine(ZoneEntryRule()).on_presence_changes(
        [change], {zone.id: zone}, {1: track},
        at_millis=2000, moment=MOMENT, detector=MOTION, frame_index=1,
    )

    assert events[0].confidence == pytest.approx(0.5)


def test_a_position_that_is_unknown_is_recorded_as_unknown():
    track = make_track(1, None)
    events = fire_entry([ZoneEntryRule()], restricted(), track)

    # The zone rule fired from the presence, which is what the caller asserted.
    # The evidence must still not invent a location for it.
    assert events[0].evidence.latitude is None
    assert events[0].evidence.position_uncertainty_meters is None
    assert "position unknown" in events[0].evidence.describe()


# ------------------------------------------------------------------- restraint


def test_a_claim_is_never_more_specific_than_the_detector_supports():
    # Under a motion detector this must not say "a person entered". A blob is
    # not a person, and the summary is what an operator reads.
    track = make_track(1, SITE)

    blob = fire_entry([ZoneEntryRule()], restricted(), track, detector=MOTION)
    classified = fire_entry([ZoneEntryRule()], restricted(), track, detector=MODEL)

    assert blob[0].summary == "An object entered Restricted Area A"
    assert "person" not in blob[0].summary
    assert "person" in classified[0].summary


def test_an_unclassified_track_is_labelled_unclassified_in_the_evidence():
    track = make_track(1, SITE)
    object.__setattr__(track, "class_id", UNCLASSIFIED)

    events = fire_entry([ZoneEntryRule()], restricted(), track, detector=MOTION)
    assert events[0].evidence.class_label == "unclassified"
    assert events[0].evidence.detector_classifies is False


def test_loitering_fires_once_for_an_ongoing_condition():
    # The alert-fatigue failure mode. One person loitering for four minutes is
    # one event, not two hundred and forty.
    rule = LoiteringRule(dwell_millis=2000)
    zone = restricted()
    track = make_track(1, SITE)
    active = engine(rule)

    produced = []
    for step in range(30):
        at = step * 200
        held = Presence("zone-a", 1, 0, at, confirmed=True, observations=step + 1)
        produced += active.on_frame(
            [held], {zone.id: zone}, {1: track},
            at_millis=at, moment=MOMENT, detector=MOTION, frame_index=step,
        )

    assert len(produced) == 1


def test_the_engine_suppresses_a_duplicate_id():
    # Belt and braces on top of the rules' own debouncing: rules are the part
    # most likely to be edited by someone who does not know the whole system.
    zone = restricted()
    track = make_track(1, SITE)
    active = engine(ZoneEntryRule())
    change = PresenceChange("ENTERED", presence(), 2000)

    first = active.on_presence_changes(
        [change], {zone.id: zone}, {1: track},
        at_millis=2000, moment=MOMENT, detector=MOTION, frame_index=1,
    )
    second = active.on_presence_changes(
        [change], {zone.id: zone}, {1: track},
        at_millis=2000, moment=MOMENT, detector=MOTION, frame_index=1,
    )

    assert len(first) == 1 and second == []
    assert active.stats.suppressed_duplicates == 1


def test_a_presence_whose_track_has_gone_produces_no_event():
    # There would be nothing to attribute it to, and inventing an attribution is
    # worse than losing the event.
    zone = restricted()
    change = PresenceChange("LEFT", presence(), 5000)

    events = engine(ZoneEntryRule()).on_presence_changes(
        [change], {zone.id: zone}, {},
        at_millis=5000, moment=MOMENT, detector=MOTION, frame_index=1,
    )
    assert events == []


def test_leaving_a_zone_is_not_an_entry():
    zone = restricted()
    track = make_track(1, SITE)
    change = PresenceChange("LEFT", presence(), 5000)

    events = engine(ZoneEntryRule()).on_presence_changes(
        [change], {zone.id: zone}, {1: track},
        at_millis=5000, moment=MOMENT, detector=MOTION, frame_index=1,
    )
    assert events == []


def test_a_zone_of_the_wrong_kind_does_not_fire_an_entry_rule():
    watching = restricted(kind=ZoneKind.INTEREST)
    track = make_track(1, SITE)

    assert fire_entry([ZoneEntryRule()], watching, track) == []


# ------------------------------------------------------- declining to conclude


def test_rapid_movement_declines_when_the_position_is_too_vague():
    # 12 m/s from a position known to ±9 m is a measurement of the projection,
    # not of the object. Firing on it is how speed rules become distrusted.
    rule = RapidMovementRule(speed_mps=6.0, max_uncertainty_meters=3.0)
    zone = restricted()

    vague = make_track(1, SITE, uncertainty=9.0, speed=12.0)
    confident = make_track(2, SITE, uncertainty=1.0, speed=12.0)

    active = engine(rule)
    events = active.on_frame(
        [], {zone.id: zone}, {1: vague, 2: confident},
        at_millis=1000, moment=MOMENT, detector=MOTION, frame_index=1,
    )

    assert len(events) == 1
    assert events[0].evidence.track_id == 2


def test_rapid_movement_declines_when_the_speed_is_unknown():
    # `None` means unknown, which is not the same as slow. Treating it as either
    # is an invention.
    rule = RapidMovementRule(speed_mps=6.0)
    zone = restricted()
    track = make_track(1, SITE, uncertainty=1.0, speed=None)

    events = engine(rule).on_frame(
        [], {zone.id: zone}, {1: track},
        at_millis=1000, moment=MOMENT, detector=MOTION, frame_index=1,
    )
    assert events == []


def test_after_hours_does_not_fire_for_a_zone_with_no_schedule():
    track = make_track(1, SITE)
    assert fire_entry([AfterHoursRule()], restricted(schedule=None), track) == []


def test_after_hours_names_the_time_as_the_reason():
    night = restricted(schedule=Schedule(time(18, 0), time(6, 0)))
    track = make_track(1, SITE)

    events = fire_entry([AfterHoursRule()], night, track)

    assert len(events) == 1
    assert events[0].severity is Severity.HIGH
    assert any("03:00" in c for c in events[0].triggering_conditions)
    assert any("18:00" in c for c in events[0].triggering_conditions)


# --------------------------------------------------------------------- shape


def test_severity_is_ordered():
    assert severity_rank(Severity.CRITICAL) > severity_rank(Severity.HIGH)
    assert severity_rank(Severity.HIGH) > severity_rank(Severity.INFO)


def test_an_event_describes_itself_completely():
    track = make_track(1, SITE, uncertainty=1.5, speed=1.1)
    events = fire_entry([ZoneEntryRule()], restricted(), track, detector=MODEL)
    text = events[0].describe()

    for expected in ("HIGH", "Restricted Area A", "because", "evidence", "confidence", "ev_"):
        assert expected in text


def test_the_engine_counts_what_it_produced():
    zone = restricted()
    track = make_track(1, SITE)
    active = engine(ZoneEntryRule(), AfterHoursRule())
    night = restricted(schedule=Schedule(time(18, 0), time(6, 0)))

    active.on_presence_changes(
        [PresenceChange("ENTERED", presence(), 2000)],
        {night.id: night}, {1: track},
        at_millis=2000, moment=MOMENT, detector=MOTION, frame_index=1,
    )

    assert active.stats.events == 2
    assert active.stats.by_severity["HIGH"] == 2


def test_media_time_converts_to_wall_clock():
    assert utc_from_millis(0) == datetime(1970, 1, 1, tzinfo=timezone.utc)


# ------------------------------------------------ produced kinds versus reserved
#
# `EventType` has members no rule raises. They cannot simply be deleted: a stored
# event names its type as a string, so removing one makes an old database
# unreadable and re-using one for something else silently reinterprets history.
# What they can be is *labelled* — and a label nothing checks is a label that
# rots, so this is the check.

#: The kinds a rule in this module actually raises today.
PRODUCED = {
    EventType.ZONE_ENTRY,
    EventType.LOITERING,
    EventType.AFTER_HOURS_PRESENCE,
    EventType.RAPID_MOVEMENT,
}

#: Declared, documented as not built, and asserted to stay that way.
RESERVED = {EventType.ZONE_EXIT, EventType.PERIMETER_BREACH}


def _rule_classes() -> list[type]:
    import inspect

    from sentinel import events as module

    return [
        value
        for value in vars(module).values()
        if inspect.isclass(value)
        and issubclass(value, Rule)
        and value is not Rule
    ]


def test_every_event_type_is_either_produced_or_declared_reserved():
    # A new member added without deciding which it is fails here rather than
    # appearing in the documentation as a capability nobody wrote.
    assert PRODUCED | RESERVED == set(EventType)
    assert not (PRODUCED & RESERVED)


def test_the_produced_set_is_exactly_what_the_rules_raise():
    raised = {rule.event_type for rule in _rule_classes()}

    assert raised == PRODUCED, (
        "the rules and the documented set disagree; update EventType's comment "
        "and this set together"
    )


def test_no_rule_raises_a_reserved_kind():
    # The other direction: writing the rule without removing the RESERVED label
    # would leave the documentation understating what the system does.
    for rule in _rule_classes():
        assert rule.event_type not in RESERVED, (
            f"{rule.__name__} now raises {rule.event_type}; move it out of "
            "RESERVED and out of the 'not built' comment on EventType"
        )
