from datetime import datetime, timezone

from vigil.domain.detection import BoundingBox, DetectorInfo
from vigil.domain.events import AfterHoursRule, EventType, LoiteringRule, RuleContext, Severity, ZoneEntryRule, default_rules
from vigil.domain.geo import LatLon, PositionEstimate, PositionSource, Vec2, destination_point
from vigil.domain.tracking import Track
from vigil.domain.zones import Membership, PresenceTracker, Schedule, Zone, ZoneKind

CLASSIFYING = DetectorInfo("onnx-detect", "test", class_names={0: "person", 1: "couch"}, classifies=True)
MOTION = DetectorInfo("motion", "MOG2", classifies=False)


def square(centre: LatLon, half: float = 5.0) -> tuple[LatLon, ...]:
    return tuple(destination_point(centre, b, half * 1.4142) for b in (45, 135, 225, 315))


def track(track_id: int, point: LatLon, *, class_id: int = 0, radius: float = 0.5, speed=None) -> Track:
    t = Track.observing(track_id, class_id, BoundingBox(0.4, 0.5, 0.1, 0.2), contact=Vec2(0.45, 0.7),
                        confidence=0.9,
                        position=PositionEstimate(point, radius, PositionSource.GROUND_PROJECTION))
    t.speed_mps = speed
    return t


def test_membership_is_uncertain_near_the_edge_and_the_watch_list_filters():
    centre = LatLon(33.8938, 35.5018)
    zone = Zone("z", "Yard", ZoneKind.RESTRICTED, square(centre), watch=frozenset({"person"}))
    assert zone.membership(centre, 0.5) is Membership.INSIDE
    assert zone.membership(destination_point(centre, 0, 4.8), 1.0) is Membership.UNCERTAIN
    assert zone.membership(destination_point(centre, 0, 30), 0.5) is Membership.OUTSIDE
    assert zone.watches("person") and not zone.watches("couch") and not zone.watches(None)
    assert Zone("o", "Open", ZoneKind.INTEREST, square(centre)).watches(None), "an empty list watches everything"


def test_presence_needs_the_entry_hold_and_survives_a_flicker():
    centre = LatLon(33.8938, 35.5018)
    zone = Zone("z", "Yard", ZoneKind.RESTRICTED, square(centre), enter_after_millis=500, exit_after_millis=1000)
    presence = PresenceTracker([zone])
    inside = track(1, centre)
    assert presence.update([inside], 0) == []
    assert presence.update([inside], 300) == []
    changes = presence.update([inside], 600)
    assert [c.kind for c in changes] == ["ENTERED"]
    assert presence.update([], 900) == [], "one missing frame is not a departure"
    assert presence.update([inside], 1000) == []
    assert [c.kind for c in presence.update([], 2100)] == ["LEFT"]


def test_zone_entry_says_object_under_motion_and_person_under_a_classifier():
    centre = LatLon(33.8938, 35.5018)
    zone = Zone("z", "Yard", ZoneKind.RESTRICTED, square(centre), enter_after_millis=0)
    presence = PresenceTracker([zone])
    presence.update([track(1, centre)], 0)
    change = presence.update([track(1, centre)], 100)[0]
    rule = ZoneEntryRule()
    for detector, word in ((MOTION, "An object"), (CLASSIFYING, "A person")):
        context = RuleContext("node", "cam", zone, track(1, centre), change.presence, 100, datetime.now(timezone.utc), detector, 3)
        events = rule.on_presence_change(change, context)
        assert len(events) == 1 and events[0].summary.startswith(word) and events[0].severity is Severity.HIGH
        assert events[0].evidence.latitude is not None and events[0].zone_name == "Yard"


def test_loitering_needs_dwell_and_stillness_and_fires_once():
    centre = LatLon(33.8938, 35.5018)
    zone = Zone("z", "Yard", ZoneKind.INTEREST, square(centre), enter_after_millis=0)
    presence = PresenceTracker([zone])
    presence.update([track(1, centre)], 0)
    presence.update([track(1, centre)], 100)
    rule = LoiteringRule(dwell_millis=5000, still_speed_mps=0.5)
    p = presence.presence_of("z", 1)
    early = RuleContext("node", "cam", zone, track(1, centre, speed=0.1), p, 2000, datetime.now(timezone.utc), CLASSIFYING, 1)
    assert rule.on_frame(early) == []
    moving = RuleContext("node", "cam", zone, track(1, centre, speed=2.0), p, 6000, datetime.now(timezone.utc), CLASSIFYING, 1)
    assert rule.on_frame(moving) == []
    still = RuleContext("node", "cam", zone, track(1, centre, speed=0.1), p, 6000, datetime.now(timezone.utc), CLASSIFYING, 1)
    events = rule.on_frame(still)
    assert len(events) == 1 and events[0].type is EventType.LOITERING
    assert rule.on_frame(still) == [], "once per presence"


def test_after_hours_reads_the_site_clock():
    centre = LatLon(33.8938, 35.5018)
    zone = Zone("z", "Yard", ZoneKind.INTEREST, square(centre), enter_after_millis=0, schedule=Schedule(22, 6))
    presence = PresenceTracker([zone])
    presence.update([track(1, centre)], 0)
    change = presence.update([track(1, centre)], 100)[0]
    rule = AfterHoursRule()
    night = RuleContext("node", "cam", zone, track(1, centre), change.presence, 100, datetime(2026, 9, 6, 2, 30, tzinfo=timezone.utc), CLASSIFYING, 1)
    day = RuleContext("node", "cam", zone, track(1, centre), change.presence, 100, datetime(2026, 9, 6, 14, 0, tzinfo=timezone.utc), CLASSIFYING, 1)
    assert len(rule.on_presence_change(change, night)) == 1
    assert rule.on_presence_change(change, day) == []
    assert Schedule(22, 6).is_closed_at(23) and Schedule(22, 6).is_closed_at(3) and not Schedule(22, 6).is_closed_at(12)


def test_default_rules_cover_every_event_type():
    assert {r.event_type for r in default_rules()} == set(EventType)


def test_a_zone_entry_says_what_they_were_carrying_and_shows_its_working():
    """The relation is inferred, so the event that quotes it must carry its conditions."""
    from vigil.domain.detection import BoundingBox
    from vigil.domain.relations import Relation, RelationKind

    centre = LatLon(33.8938, 35.5018)
    zone = Zone("z", "Yard", ZoneKind.RESTRICTED, square(centre), enter_after_millis=0)
    presence = PresenceTracker([zone])
    presence.update([track(1, centre)], 0)
    change = presence.update([track(1, centre)], 100)[0]
    carried = Relation(RelationKind.CARRIED, 1, 9, confidence=0.7,
                       conditions=("62% of the backpack's box lay within the person's",))
    context = RuleContext("node", "cam", zone, track(1, centre), change.presence, 100,
                          datetime.now(timezone.utc), CLASSIFYING, 3, None, (carried,), {1: "person", 9: "backpack"})
    events = ZoneEntryRule().on_presence_change(change, context)
    assert len(events) == 1
    assert events[0].summary == "A person entered Yard carrying backpack"
    assert any("62% of the backpack's box" in c for c in events[0].evidence.conditions)

    # With nothing carried the sentence is unchanged from before.
    plain = RuleContext("node", "cam", zone, track(1, centre), change.presence, 100,
                        datetime.now(timezone.utc), CLASSIFYING, 3)
    assert ZoneEntryRule().on_presence_change(change, plain)[0].summary == "A person entered Yard"
    assert plain.carrying() == () and plain.name_of(4) == "track 4"


def test_a_vehicle_entering_a_zone_says_how_many_people_appear_to_be_in_it():
    """One event whether it holds a driver or five, and the difference is the point."""
    from vigil.domain.relations import Relation, RelationKind, describe_group, occupants_of

    centre = LatLon(33.8938, 35.5018)
    zone = Zone("z", "Yard", ZoneKind.RESTRICTED, square(centre), enter_after_millis=0)
    presence = PresenceTracker([zone])
    presence.update([track(7, centre)], 0)
    change = presence.update([track(7, centre)], 100)[0]
    # Three people inside the car, and the car itself carrying nothing.
    inside = tuple(Relation(RelationKind.INSIDE, subject, 7, confidence=0.6,
                            conditions=(f"{70 + subject}% of the person's box lay within the car's",))
                   for subject in (1, 2, 3))
    labels = {7: "car", 1: "person", 2: "person", 3: "person"}
    vehicles = DetectorInfo("onnx-detect", "test", class_names={0: "person", 2: "car"}, classifies=True)
    context = RuleContext("node", "cam", zone, track(7, centre, class_id=2), change.presence, 100,
                          datetime.now(timezone.utc), vehicles, 3, None, inside, labels)
    assert occupants_of(inside, 7) == (1, 2, 3)
    assert context.group() == "3 people"
    event = ZoneEntryRule().on_presence_change(change, context)[0]
    assert event.summary == "A car entered Yard, apparently with 3 people inside"
    assert "apparently" in event.summary, "one camera cannot see inside a car; the wording must hedge"
    assert any("box lay within the car's" in c for c in event.evidence.conditions), "the count must show its working"

    # Counted once each, however many frames repeated the relation, and the
    # phrase is the one an operator would use.
    assert occupants_of((*inside, inside[0]), 7) == (1, 2, 3)
    assert describe_group(["person"]) == "a person" and describe_group(["person", "dog"]) == "a dog and a person"

    # Nobody inside: the sentence is the one it always was.
    alone = RuleContext("node", "cam", zone, track(7, centre, class_id=2), change.presence, 100,
                        datetime.now(timezone.utc), vehicles, 3, None, (), labels)
    assert alone.occupants() == () and alone.group() == ""
    assert ZoneEntryRule().on_presence_change(change, alone)[0].summary == "A car entered Yard"


def _approaching(distance_m: float, zone: Zone, *, class_id: int = 0, confidence: float = 0.8):
    """A track that far outside the zone, and the relation saying it is closing."""
    from vigil.domain.relations import Relation, RelationKind

    centre = LatLon(33.8938, 35.5018)
    point = destination_point(centre, 0.0, 7.07 + distance_m)
    subject = track(1, point, class_id=class_id)
    subject.confidence = confidence
    relation = Relation(RelationKind.APPROACHING, 1, None, zone.id, confidence=0.6, observations=8,
                        conditions=("the gap to the Yard closed 4.2 m in 2.1 s",))
    context = RuleContext("node", "cam", zone, subject, None, 1000, datetime.now(timezone.utc), CLASSIFYING, 5,
                          None, (relation,), {1: "person"})
    return relation, context


def test_an_approach_to_a_restricted_zone_warns_before_the_breach():
    from vigil.domain.events import ApproachRule

    centre = LatLon(33.8938, 35.5018)
    zone = Zone("z", "Yard", ZoneKind.RESTRICTED, square(centre))
    rule = ApproachRule()
    relation, context = _approaching(4.0, zone)
    events = rule.on_relation(relation, context)
    assert len(events) == 1
    assert events[0].type is EventType.APPROACHING and events[0].severity is Severity.MEDIUM
    assert events[0].summary.startswith("A person is approaching Yard, ")
    assert "±" in events[0].summary, "a distance without its error"
    assert any("closed 4.2 m in 2.1 s" in c for c in events[0].evidence.conditions)
    assert any("inside the 10 m this rule watches" in c for c in events[0].evidence.conditions)
    assert rule.on_relation(relation, context) == [], "a warning repeats every frame otherwise"
    rule.forget(1)
    assert rule.on_relation(relation, context), "a new track of the same id starts again"


def test_an_approach_from_far_away_or_to_an_ordinary_zone_says_nothing():
    from vigil.domain.events import ApproachRule
    from vigil.domain.relations import RelationKind

    centre = LatLon(33.8938, 35.5018)
    restricted = Zone("z", "Yard", ZoneKind.RESTRICTED, square(centre))
    relation, far = _approaching(40.0, restricted)
    assert ApproachRule().on_relation(relation, far) == [], "walking towards it from forty metres is walking"

    interest = Zone("z", "Yard", ZoneKind.INTEREST, square(centre))
    relation, close = _approaching(4.0, interest)
    assert ApproachRule().on_relation(relation, close) == [], "an interest zone is not a place to warn about"

    # A relation of another kind, or for another zone, is not this rule's business.
    other, close = _approaching(4.0, restricted)
    from dataclasses import replace

    assert ApproachRule().on_relation(replace(other, kind=RelationKind.NEAR), close) == []
    assert ApproachRule().on_relation(replace(other, zone_id="somewhere-else"), close) == []


def test_a_zone_watch_list_silences_an_approach_too():
    from vigil.domain.events import ApproachRule

    centre = LatLon(33.8938, 35.5018)
    zone = Zone("z", "Yard", ZoneKind.RESTRICTED, square(centre), watch=frozenset({"vehicle"}))
    relation, context = _approaching(4.0, zone)
    assert ApproachRule().on_relation(relation, context) == [], "a person approached a vehicle-only zone"
