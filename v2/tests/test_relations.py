"""What tracked things are doing together — and every way that must not fire."""

from __future__ import annotations

import pytest

from vigil.domain.detection import BoundingBox
from vigil.domain.geo import LatLon, PositionEstimate, PositionSource, destination_point
from vigil.domain.relations import (
    Relation, RelationKind, RelationRules, RelationTracker, overlap_fraction,
)
from vigil.domain.tracking import Track
from vigil.domain.zones import Zone, ZoneKind

ORIGIN = LatLon(33.8938, 35.5018)
LABELS = {0: "person", 1: "car", 2: "backpack", 3: "bench"}


def label_of(class_id: int) -> str | None:
    return LABELS.get(class_id)


def track(track_id: int, box: BoundingBox, class_id: int = 0, *, point: LatLon | None = None,
          radius: float = 0.4) -> Track:
    position = None
    if point is not None:
        position = PositionEstimate(point, radius, PositionSource.GROUND_PROJECTION)
    return Track(track_id, class_id, 0, 0, 0, box, box.bottom_center, confidence=0.9, confirmed=True, position=position)


def _hold(tracker: RelationTracker, tracks, *, frames: int = 6, step: int = 200, zones=()):
    """Run the same geometry for a while, the way a real second of video does."""
    found = []
    for index in range(frames):
        found = tracker.update(tracks, index * step, label_of=label_of, zones=zones)
    return found


def test_overlap_is_measured_against_the_inner_box():
    outer = BoundingBox(0.0, 0.0, 1.0, 1.0)
    assert overlap_fraction(BoundingBox(0.25, 0.25, 0.5, 0.5), outer) == pytest.approx(1.0)
    assert overlap_fraction(BoundingBox(-0.5, 0.0, 1.0, 1.0), outer) == pytest.approx(0.5)
    assert overlap_fraction(BoundingBox(2.0, 2.0, 0.1, 0.1), outer) == 0.0
    assert overlap_fraction(BoundingBox(0, 0, 0, 0), outer) == 0.0


# ---------------------------------------------------------------- inside


def test_a_person_mostly_within_a_vehicle_is_probably_inside_it_and_says_so():
    car = track(1, BoundingBox(0.2, 0.3, 0.5, 0.4), class_id=1)
    person = track(2, BoundingBox(0.30, 0.34, 0.10, 0.30), class_id=0)
    tracker = RelationTracker()
    assert tracker.update([car, person], 0, label_of=label_of) == [], "one frame is a coincidence"
    found = _hold(tracker, [car, person])
    inside = [r for r in found if r.kind is RelationKind.INSIDE]
    assert len(inside) == 1 and inside[0].subject == 2 and inside[0].object == 1
    assert 0 < inside[0].confidence <= 0.95
    said = inside[0].describe(lambda t: {1: "the car", 2: "a person"}[t])
    assert said == "a person is probably in the car", said
    assert any("cannot tell being inside from passing in front" in c for c in inside[0].conditions)
    assert any("held for" in c for c in inside[0].conditions)


def test_a_person_passing_in_front_of_a_car_is_not_called_inside_it():
    car = track(1, BoundingBox(0.2, 0.3, 0.5, 0.4), class_id=1)
    passing = track(2, BoundingBox(0.62, 0.30, 0.12, 0.45), class_id=0)  # mostly outside the car's box
    found = _hold(RelationTracker(), [car, passing])
    assert [r for r in found if r.kind is RelationKind.INSIDE] == []


def test_a_car_is_never_said_to_be_inside_a_person():
    car = track(1, BoundingBox(0.30, 0.34, 0.10, 0.10), class_id=1)
    person = track(2, BoundingBox(0.2, 0.3, 0.5, 0.4), class_id=0)
    found = _hold(RelationTracker(), [car, person])
    assert [r for r in found if r.kind is RelationKind.INSIDE] == []


# --------------------------------------------------------------- carried


def test_a_small_thing_held_off_the_ground_is_reported_as_carried():
    person = track(1, BoundingBox(0.40, 0.30, 0.12, 0.50), class_id=0)
    bag = track(2, BoundingBox(0.44, 0.45, 0.06, 0.10), class_id=2)
    found = _hold(RelationTracker(), [person, bag])
    carried = [r for r in found if r.kind is RelationKind.CARRIED]
    assert len(carried) == 1 and carried[0].subject == 1 and carried[0].object == 2
    said = carried[0].describe(lambda t: {1: "a person", 2: "a backpack"}[t])
    assert said == "a person appears to be carrying a backpack"
    assert any("not on the ground" in c for c in carried[0].conditions)


def test_a_bag_on_the_ground_beside_somebody_is_not_carried():
    person = track(1, BoundingBox(0.40, 0.30, 0.12, 0.50), class_id=0)
    on_the_ground = track(2, BoundingBox(0.44, 0.74, 0.06, 0.06), class_id=2)  # its bottom is at the feet
    found = _hold(RelationTracker(), [person, on_the_ground])
    assert [r for r in found if r.kind is RelationKind.CARRIED] == []


def test_something_as_big_as_the_person_is_not_carried():
    person = track(1, BoundingBox(0.40, 0.30, 0.12, 0.50), class_id=0)
    bench = track(2, BoundingBox(0.41, 0.32, 0.10, 0.40), class_id=3)
    found = _hold(RelationTracker(), [person, bench])
    assert [r for r in found if r.kind is RelationKind.CARRIED] == []


# ------------------------------------------------------------------ near


def test_two_people_close_on_the_ground_are_together_and_far_ones_are_not():
    near_point = destination_point(ORIGIN, 90.0, 1.5)
    far_point = destination_point(ORIGIN, 90.0, 12.0)
    a = track(1, BoundingBox(0.1, 0.5, 0.05, 0.2), point=ORIGIN)
    b = track(2, BoundingBox(0.3, 0.5, 0.05, 0.2), point=near_point)
    together = [r for r in _hold(RelationTracker(), [a, b]) if r.kind is RelationKind.NEAR]
    assert len(together) == 1 and together[0].subject == 1 and together[0].object == 2
    assert together[0].describe(lambda t: f"person {t}") == "person 1 is with person 2"

    apart = track(2, BoundingBox(0.3, 0.5, 0.05, 0.2), point=far_point)
    assert [r for r in _hold(RelationTracker(), [a, apart]) if r.kind is RelationKind.NEAR] == []


def test_a_pair_whose_error_swamps_the_gap_is_not_called_together():
    """Two positions each known to ±3 m are not "within three metres" of anything."""
    b_point = destination_point(ORIGIN, 90.0, 2.0)
    a = track(1, BoundingBox(0.1, 0.5, 0.05, 0.2), point=ORIGIN, radius=3.0)
    b = track(2, BoundingBox(0.3, 0.5, 0.05, 0.2), point=b_point, radius=3.0)
    assert [r for r in _hold(RelationTracker(), [a, b]) if r.kind is RelationKind.NEAR] == []


def test_an_unprojected_track_takes_part_in_no_ground_relation():
    a = track(1, BoundingBox(0.1, 0.5, 0.05, 0.2), point=ORIGIN)
    nowhere = track(2, BoundingBox(0.3, 0.5, 0.05, 0.2))
    found = _hold(RelationTracker(), [a, nowhere])
    assert [r for r in found if r.kind in (RelationKind.NEAR, RelationKind.APPROACHING)] == []


# ----------------------------------------------------------- approaching


def _zone_around(centre: LatLon, half: float = 5.0) -> Zone:
    ring = tuple(destination_point(centre, b, half * 1.4142) for b in (45, 135, 225, 315))
    return Zone("yard", "the Yard", ZoneKind.RESTRICTED, ring)


def test_somebody_walking_towards_a_zone_is_reported_once_the_gap_really_closes():
    zone = _zone_around(ORIGIN)
    tracker = RelationTracker()
    found = []
    for step in range(10):
        # Twenty-five metres out, closing a metre and a half each step.
        point = destination_point(ORIGIN, 0.0, 25.0 - step * 1.5)
        walker = track(1, BoundingBox(0.4, 0.5, 0.1, 0.3), point=point)
        found = tracker.update([walker], step * 300, label_of=label_of, zones=[zone])
    approaching = [r for r in found if r.kind is RelationKind.APPROACHING]
    assert len(approaching) == 1 and approaching[0].zone_id == "yard"
    assert approaching[0].describe(lambda t: "a person") == "a person is moving towards zone yard"
    assert any("closed" in c for c in approaching[0].conditions)
    assert any("from the edge now" in c for c in approaching[0].conditions)


def test_standing_still_is_not_approaching_and_neither_is_jitter():
    zone = _zone_around(ORIGIN)
    tracker = RelationTracker()
    found = []
    for step in range(10):
        # A tenth of a metre of wobble, well inside the position error.
        point = destination_point(ORIGIN, 0.0, 25.0 + (0.1 if step % 2 else -0.1))
        still = track(1, BoundingBox(0.4, 0.5, 0.1, 0.3), point=point)
        found = tracker.update([still], step * 300, label_of=label_of, zones=[zone])
    assert [r for r in found if r.kind is RelationKind.APPROACHING] == []


def test_somebody_already_inside_the_zone_is_a_presence_not_an_approach():
    zone = _zone_around(ORIGIN)
    inside = track(1, BoundingBox(0.4, 0.5, 0.1, 0.3), point=ORIGIN)
    found = _hold(RelationTracker(), [inside], zones=[zone])
    assert [r for r in found if r.kind is RelationKind.APPROACHING] == []


# ------------------------------------------------------------ the rules


def test_the_thresholds_are_one_object_a_site_can_argue_with():
    car = track(1, BoundingBox(0.2, 0.3, 0.5, 0.4), class_id=1)
    # Two thirds of this person's box lies inside the car's, which is the
    # awkward middle a site has to make its own mind up about.
    partly = track(2, BoundingBox(0.60, 0.34, 0.15, 0.30), class_id=0)
    strict = RelationTracker(RelationRules(inside_overlap=0.9, hold_millis=0))
    assert [r for r in _hold(strict, [car, partly]) if r.kind is RelationKind.INSIDE] == []
    lenient = RelationTracker(RelationRules(inside_overlap=0.5, hold_millis=0))
    assert [r for r in _hold(lenient, [car, partly]) if r.kind is RelationKind.INSIDE]


def test_a_track_that_ends_takes_its_relations_with_it():
    tracker = RelationTracker()
    car = track(1, BoundingBox(0.2, 0.3, 0.5, 0.4), class_id=1)
    person = track(2, BoundingBox(0.30, 0.34, 0.10, 0.30), class_id=0)
    assert _hold(tracker, [car, person])
    tracker.forget_track(2)
    assert tracker.update([car, person], 5000, label_of=label_of) == [], "the hold must start again"


def test_a_relation_without_a_namer_still_reads():
    relation = Relation(RelationKind.CARRIED, 4, 9)
    assert relation.describe() == "track 4 appears to be carrying track 9"
    assert Relation(RelationKind.APPROACHING, 4, None, "yard").describe() == "track 4 is moving towards zone yard"
