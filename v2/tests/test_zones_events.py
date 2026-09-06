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
    t = Track(track_id, class_id, 0, 0, 0, BoundingBox(0.4, 0.5, 0.1, 0.2), Vec2(0.45, 0.7), confidence=0.9, confirmed=True,
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
