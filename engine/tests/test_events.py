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

import scene

SITE = LatLon(33.8938, 35.5018)
MOMENT = datetime(2026, 8, 30, 3, 0, tzinfo=timezone.utc)

MOTION = DetectorInfo(kind="motion", name="MOG2 background subtraction", classifies=False)
MODEL = DetectorInfo(
    kind="onnx",
    name="yolo-test",
    model_path="/models/yolo-test.onnx",
    model_sha256="a" * 64,
    # COCO's numbering, because that is what the sofa incident was raised in.
    class_names={0: "person", 39: "bottle", 57: "couch"},
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


def test_the_after_hours_condition_is_written_in_the_site_clock():
    # The reader checks the condition against a wall clock, so it carries the
    # site's time and offset; the event's own timestamp stays UTC beside it.
    from datetime import time as clock
    from datetime import timedelta

    from sentinel.events import AfterHoursRule
    from sentinel.zones import Schedule

    from sentinel.core import BoundingBox

    night = restricted(schedule=Schedule(clock(18, 0), clock(6, 0)))
    track = Track(
        id=1, class_id=0, bbox=BoundingBox(0.4, 0.5, 0.1, 0.2), confidence=0.9, hits=10,
        first_seen_millis=0, last_seen_millis=2000, position=None, speed_mps=None,
        heading_degrees=None,
    )
    moment = datetime(2026, 8, 30, 16, 30, tzinfo=timezone.utc)

    beirut = EventEngine([AfterHoursRule()], node_id="nd_test", camera_id="cam-07",
                         site_tz=timezone(timedelta(hours=3)))
    (event,) = beirut.on_presence_changes(
        [PresenceChange("ENTERED", presence(track.id), 2000)], {night.id: night},
        {track.id: track}, at_millis=2000, moment=moment, detector=MOTION, frame_index=60,
    )
    assert any(c.startswith("19:30 UTC+0300") for c in event.triggering_conditions), event.triggering_conditions
    assert event.occurred_at == moment and event.occurred_at.tzinfo is timezone.utc

    plain = EventEngine([AfterHoursRule()], node_id="nd_test", camera_id="cam-07")
    (event,) = plain.on_presence_changes(
        [PresenceChange("ENTERED", presence(track.id), 2000)], {night.id: night},
        {track.id: track}, at_millis=2000, moment=moment, detector=MOTION, frame_index=60,
    )
    assert any(c.startswith("16:30 UTC+0000") for c in event.triggering_conditions)


# ------------------------------------------------------------ the class filter
#
# On a real camera a RESTRICTED zone raised "1 couch in Room (HIGH, risk 55)" and
# "A bottle entered Room", because it fired on any class the segmenter named. A
# zone now carries a class filter and every rule that acts on a presence asks
# it. These tests are the couch, the person, and the detector that cannot tell
# them apart.

PERSON, BOTTLE, COUCH = 0, 39, 57


def people_only(**overrides) -> Zone:
    return restricted(classes=frozenset({"person"}), **overrides)


def test_a_couch_entering_a_person_only_zone_raises_nothing():
    zone = people_only()

    couch = fire_entry([ZoneEntryRule()], zone, make_track(1, SITE, class_id=COUCH), detector=MODEL)
    bottle = fire_entry([ZoneEntryRule()], zone, make_track(2, SITE, class_id=BOTTLE), detector=MODEL)
    person = fire_entry([ZoneEntryRule()], zone, make_track(3, SITE, class_id=PERSON), detector=MODEL)

    assert couch == [] and bottle == []
    assert len(person) == 1
    assert person[0].summary == "A person entered Restricted Area A"
    assert person[0].severity is Severity.HIGH


def test_an_empty_filter_keeps_firing_for_everything_including_unclassified():
    # What every zone did before the filter existed, and what a site with no
    # model relies on. The couch still fires here — that is the operator's
    # choice to make by setting a filter, not this code's to make for them.
    zone = restricted()

    couch = fire_entry([ZoneEntryRule()], zone, make_track(1, SITE, class_id=COUCH), detector=MODEL)
    unnamed = fire_entry(
        [ZoneEntryRule()], zone, make_track(2, SITE, class_id=UNCLASSIFIED), detector=MODEL
    )
    blob = fire_entry([ZoneEntryRule()], zone, make_track(3, SITE, class_id=UNCLASSIFIED), detector=MOTION)

    assert couch[0].summary == "A couch entered Restricted Area A"
    assert unnamed[0].evidence.class_label == "unclassified"
    assert blob[0].summary == "An object entered Restricted Area A"


def test_a_filtered_zone_never_fires_from_a_motion_detector():
    """A blob is not a person, however confidently it moved.

    The motion detector labels nothing, so it cannot say the thing in the zone
    was a person, so a person-only zone must not fire on it — whatever class id
    happens to be on the track. Checked for every rule that acts on a presence,
    because one of them firing would be the sofa incident under another name.
    """
    night = people_only(schedule=Schedule(time(18, 0), time(6, 0)))

    for class_id in (PERSON, COUCH, UNCLASSIFIED):
        track = make_track(1, SITE, class_id=class_id)
        assert fire_entry([ZoneEntryRule()], night, track, detector=MOTION) == []
        assert fire_entry([AfterHoursRule()], night, track, detector=MOTION) == []

        held = Presence("zone-a", 1, 0, 60_000, confirmed=True, observations=300)
        loitering = engine(LoiteringRule(dwell_millis=2000)).on_frame(
            [held], {night.id: night}, {1: track},
            at_millis=60_000, moment=MOMENT, detector=MOTION, frame_index=900,
        )
        assert loitering == []


def test_the_context_names_the_class_only_when_the_detector_can():
    # The one place a rule reads the label. `None` under a motion detector,
    # not "unclassified": the filter is asked with this value, and the string
    # is reserved for a classifying detector that looked and could not decide.
    from sentinel.events import RuleContext

    def context(track: Track, detector: DetectorInfo) -> RuleContext:
        return RuleContext(
            node_id="nd_test", camera_id="cam-07", zone=None, track=track, presence=None,
            at_millis=0, moment=MOMENT, detector=detector, frame_index=0,
        )

    assert context(make_track(1, SITE, class_id=PERSON), MOTION).class_label is None
    assert context(make_track(1, SITE, class_id=UNCLASSIFIED), MOTION).class_label is None
    assert context(make_track(1, SITE, class_id=PERSON), MODEL).class_label == "person"
    assert context(make_track(1, SITE, class_id=COUCH), MODEL).class_label == "couch"
    assert context(make_track(1, SITE, class_id=UNCLASSIFIED), MODEL).class_label == "unclassified"


def test_loitering_honours_the_filter_without_spending_the_presences_one_firing():
    # Fires once per presence — and a couch must not use that one up, so a
    # filter edited while the presence is open still lets the person fire.
    rule = LoiteringRule(dwell_millis=2000)
    zone = people_only()
    active = engine(rule)

    def run(track: Track) -> list[Event]:
        produced = []
        for step in range(30):
            at = step * 200
            held = Presence("zone-a", track.id, 0, at, confirmed=True, observations=step + 1)
            produced += active.on_frame(
                [held], {zone.id: zone}, {track.id: track},
                at_millis=at, moment=MOMENT, detector=MODEL, frame_index=step,
            )
        return produced

    assert run(make_track(1, SITE, class_id=COUCH)) == []
    assert len(run(make_track(2, SITE, class_id=PERSON))) == 1


def test_after_hours_honours_the_filter():
    night = people_only(schedule=Schedule(time(18, 0), time(6, 0)))

    couch = fire_entry([AfterHoursRule()], night, make_track(1, SITE, class_id=COUCH), detector=MODEL)
    person = fire_entry([AfterHoursRule()], night, make_track(2, SITE, class_id=PERSON), detector=MODEL)

    assert couch == []
    assert len(person) == 1 and person[0].type is EventType.AFTER_HOURS_PRESENCE


def test_rapid_movement_is_about_the_track_and_ignores_the_filter():
    # Stated so the omission cannot be read as an oversight: the zone in this
    # context is incidental, and something moving at 12 m/s is worth a LOW
    # event whatever the detector calls it.
    zone = people_only()
    couch = make_track(1, SITE, uncertainty=1.0, speed=12.0, class_id=COUCH)
    held = Presence("zone-a", 1, 0, 1000, confirmed=True, observations=5)

    events = engine(RapidMovementRule(speed_mps=6.0)).on_frame(
        [held], {zone.id: zone}, {1: couch},
        at_millis=1000, moment=MOMENT, detector=MODEL, frame_index=1,
    )

    assert len(events) == 1 and events[0].type is EventType.RAPID_MOVEMENT


# ----------------------------------------------- the filter on the reference video


class OracleLabeller:
    """The motion detector, with each blob named from the scene's own truth.

    The reference scene has no classes — its walkers are rectangles — so a
    filter over it has nothing to filter unless something supplies labels. The
    labels here are taken from ``scene.ground_truth``: the walker whose box is
    nearest a blob names it. Two walkers are called ``person`` and the third
    ``couch``, not because it resembles one but because a filter needs
    something to exclude, and a fixture that labelled everything ``person``
    would prove the filter equal to no filter and nothing else.

    Everything downstream of `detect()` — tracking, projection, presences, the
    rules — is the real pipeline, so what this measures is whether the filter
    removes exactly the events it should from real pipeline output, and not a
    hand-built presence. Frames are counted rather than passed in because the
    `Detector` protocol sees only the image; the pipeline calls `detect` once
    per frame in order, which `test_pipeline.py` asserts.
    """

    NAMES = {PERSON: "person", COUCH: "couch"}
    WALKER_CLASS = {"approaching": PERSON, "crossing": PERSON, "loiterer": COUCH}

    def __init__(self, inner):
        from dataclasses import replace

        self._inner = inner
        self._frame = 0
        self._info = replace(
            inner.info, kind="oracle", name="motion + scene oracle",
            class_names=dict(self.NAMES), classifies=True,
        )

    @property
    def info(self) -> DetectorInfo:
        return self._info

    def detect(self, image):
        from dataclasses import replace

        truth = scene.ground_truth(self._frame)
        self._frame += 1

        labelled = []
        for detection in self._inner.detect(image):
            cx = (detection.bbox.x + detection.bbox.w / 2) * scene.WIDTH
            cy = (detection.bbox.y + detection.bbox.h / 2) * scene.HEIGHT
            nearest, distance = None, None
            for name, (x, y, w, h) in truth.items():
                gap = ((cx - (x + w / 2)) ** 2 + (cy - (y + h / 2)) ** 2) ** 0.5
                if distance is None or gap < distance:
                    nearest, distance = name, gap
            # Within the walker's own size, or the oracle honestly says it
            # cannot name the blob — a merged pair, a shadow.
            if nearest is not None and distance <= max(truth[nearest][2], truth[nearest][3]):
                class_id = self.WALKER_CLASS[nearest]
            else:
                class_id = UNCLASSIFIED
            labelled.append(replace(detection, class_id=class_id))
        return labelled


def reference_zone(pose, classes: frozenset[str] = frozenset()) -> Zone:
    """The restricted area `test_pipeline.py` lays across the walkers' paths."""
    centre = destination_point(pose.position, 180.0, 14.0)
    return Zone(
        id="zone-a",
        name="Restricted Area A",
        kind=ZoneKind.RESTRICTED,
        ring=tuple(destination_point(centre, b, 9.0) for b in (0.0, 90.0, 180.0, 270.0)),
        enter_after_millis=600,
        classes=classes,
    )


def run_reference(video, pose, zone: Zone) -> list[Event]:
    from sentinel.decode import VideoSource
    from sentinel.detect import MotionDetector
    from sentinel.pipeline import Pipeline

    # A fixed clock, so the same footage produces the same ids on any day —
    # which is what lets the two runs below be compared id for id.
    epoch = int(datetime(2026, 8, 30, 2, 0, tzinfo=timezone.utc).timestamp() * 1000)
    with Pipeline(
        VideoSource(video, source_id="cam-07"),
        OracleLabeller(MotionDetector()),
        pose=pose,
        zones=[zone],
        rules=[ZoneEntryRule(), LoiteringRule(dwell_millis=4000)],
        node_id="nd_test",
        wall_clock_epoch_millis=epoch,
    ) as pipeline:
        return [event for result in pipeline.run() for event in result.events]


@pytest.fixture(scope="module")
def filtered_and_not(reference_video, reference_pose):
    """One pass with no filter and one with ``person`` only, over one video."""
    return (
        run_reference(reference_video, reference_pose, reference_zone(reference_pose)),
        run_reference(
            reference_video, reference_pose,
            reference_zone(reference_pose, frozenset({"person"})),
        ),
    )


def test_a_person_only_zone_on_the_reference_video_raises_no_more_than_an_unfiltered_one(
    filtered_and_not,
):
    """Measured: 9 events unfiltered (7 person, 2 couch) against 7 person-only.

    The floors are below that so a detector or tracker change moves the
    numbers without failing this — what must hold is the direction, and that
    the comparison was not vacuous: the unfiltered run has to have raised
    something the filter could remove.
    """
    unfiltered, people = filtered_and_not
    labels = {event.evidence.class_label for event in unfiltered}

    assert len(unfiltered) >= 4, "the reference walkers never crossed the zone"
    assert "couch" in labels, "the oracle never labelled a blob couch, so there was nothing to filter"
    assert len(people) <= len(unfiltered)
    assert len(people) < len(unfiltered), "the couch's events survived the filter"
    assert len(people) >= 1, "the filter silenced the people too"


def test_a_person_only_zone_never_raises_for_a_non_person_label(filtered_and_not):
    _, people = filtered_and_not

    assert people, "nothing to check"
    assert {event.evidence.class_label for event in people} == {"person"}
    assert all("couch" not in event.summary for event in people)


def test_the_filter_removes_exactly_the_non_person_events_and_nothing_else(filtered_and_not):
    # Ids are deterministic and the presences are unaffected by the filter, so
    # the person-only run must be the person subset of the unfiltered run —
    # not fewer events for some other reason, and not different ones.
    unfiltered, people = filtered_and_not

    expected = {event.id for event in unfiltered if event.evidence.class_label == "person"}
    assert {event.id for event in people} == expected


def test_every_rule_names_the_class_the_same_way():
    """"A person entered" beside "An object remained" reads as doubt.

    Seen on the laptop camera with a person-only zone: the entry events said
    "A person" and the loitering event for the same person said "An object",
    because two rules carried a fixed subject. All three now share one
    phrasing, and under a motion detector all three still say "An object".
    """
    from sentinel.events import AfterHoursRule, LoiteringRule, RuleContext, _subject
    from sentinel.zones import Schedule
    from datetime import time as clock

    person = make_track(1, SITE, class_id=0)
    yard = restricted()

    def context(detector, track):
        return RuleContext(
            node_id="nd", camera_id="cam-07", zone=yard, track=track,
            presence=presence(track.id), at_millis=2000,
            moment=datetime(2026, 8, 30, 3, 0, tzinfo=timezone.utc),
            detector=detector, frame_index=60,
        )

    assert _subject(context(MODEL, person)) == "A person"
    assert _subject(context(MOTION, person)) == "An object"

    # And the two rules that used to say "An object" regardless now agree
    # with the entry rule when the detector can name the class.
    night = restricted(schedule=Schedule(clock(18, 0), clock(6, 0)))
    (after_hours,) = AfterHoursRule().on_presence_change(
        PresenceChange("ENTERED", presence(person.id), 2000),
        RuleContext(node_id="nd", camera_id="cam-07", zone=night, track=person,
                    presence=presence(person.id), at_millis=2000,
                    moment=datetime(2026, 8, 30, 3, 0, tzinfo=timezone.utc),
                    detector=MODEL, frame_index=60),
    )
    assert after_hours.summary.startswith("A person was in")
