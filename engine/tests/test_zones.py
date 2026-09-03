"""Tests for zones, schedules and presence.

The stage where a position becomes a reason to wake someone. Most of these tests
are about *not* doing that: a zone boundary is where false alarms are
manufactured, and almost every way of manufacturing one has a test here.
"""

from __future__ import annotations

from datetime import datetime, time, timezone

import pytest

from sentinel.core import (
    BoundingBox,
    LatLon,
    PositionEstimate,
    Track,
    ZoneMembership,
    destination_point,
    zone_membership,
)
from sentinel.zones import Schedule, Zone, ZoneEvaluator, ZoneKind

SITE = LatLon(33.8938, 35.5018)


def ring_around(centre: LatLon, edge_distance: float) -> tuple[LatLon, ...]:
    """An axis-aligned square whose *edges* sit ``edge_distance`` from the centre.

    Corners go out by a further factor of root two. Placing the corners at the
    named distance instead would put the edges 29 percent closer, which is the
    kind of quiet arithmetic slip that makes a boundary test assert the opposite
    of what it reads as.
    """
    import math

    diagonal = edge_distance * math.sqrt(2)
    return tuple(
        destination_point(centre, bearing, diagonal)
        for bearing in (45.0, 135.0, 225.0, 315.0)
    )


def make_track(
    track_id: int,
    point: LatLon | None,
    uncertainty: float = 1.0,
    *,
    first: int = 0,
    last: int = 0,
    speed: float | None = None,
) -> Track:
    position = (
        PositionEstimate(point=point, radius_meters=uncertainty, source="GROUND_PROJECTION")
        if point is not None
        else None
    )
    return Track(
        id=track_id,
        class_id=0,
        bbox=BoundingBox(0.4, 0.5, 0.1, 0.2),
        confidence=0.9,
        hits=10,
        first_seen_millis=first,
        last_seen_millis=last,
        position=position,
        speed_mps=speed,
        heading_degrees=None,
    )


def zone(**overrides) -> Zone:
    defaults = dict(
        id="zone-a",
        name="Restricted Area A",
        kind=ZoneKind.RESTRICTED,
        ring=ring_around(SITE, 40.0),
        schedule=None,
        enter_after_millis=600,
        exit_after_millis=2000,
    )
    defaults.update(overrides)
    return Zone(**defaults)  # type: ignore[arg-type]


# ------------------------------------------------------------------ membership


def test_a_confident_position_inside_is_inside():
    assert zone_membership(ring_around(SITE, 40.0), SITE, 1.0) is ZoneMembership.INSIDE


def test_an_uncertain_position_near_the_boundary_is_neither():
    # 3 m outside a fence, known to ±8 m. Calling that "outside" is a guess
    # dressed as a measurement; calling it "inside" is an alarm nobody can
    # justify. The third answer exists so a rule can decline to fire.
    near = destination_point(SITE, 0.0, 36.0)  # 4 m inside the northern edge
    ring = ring_around(SITE, 40.0)

    assert zone_membership(ring, near, 8.0) is ZoneMembership.UNCERTAIN
    assert zone_membership(ring, near, 0.5) is ZoneMembership.INSIDE

    outside = destination_point(SITE, 0.0, 44.0)  # 4 m beyond it
    assert zone_membership(ring, outside, 8.0) is ZoneMembership.UNCERTAIN
    assert zone_membership(ring, outside, 0.5) is ZoneMembership.OUTSIDE


def test_a_track_with_no_position_is_never_in_a_zone():
    # An unplaced camera cannot support any claim about where its objects are.
    assert zone().membership_of(make_track(1, None)) is ZoneMembership.OUTSIDE


def test_a_restricted_zone_rejects_an_uncertain_position_by_default():
    restricted = zone()
    assert restricted.accepts(ZoneMembership.INSIDE) is True
    assert restricted.accepts(ZoneMembership.UNCERTAIN) is False


def test_a_zone_can_opt_into_accepting_uncertainty():
    # Right for an interest or coverage zone, wrong for an alarm.
    watching = zone(kind=ZoneKind.INTEREST, accept_uncertain=True)
    assert watching.accepts(ZoneMembership.UNCERTAIN) is True


def test_a_zone_needs_three_points():
    with pytest.raises(ValueError, match="line, not an area"):
        Zone(id="z", name="Z", kind=ZoneKind.RESTRICTED, ring=(SITE, SITE))


# -------------------------------------------------------------------- schedule


def test_a_schedule_that_wraps_midnight_covers_the_night():
    # The case that silently disarms a site every night if it is wrong, so it
    # is the one to lead with.
    after_hours = Schedule(time(18, 0), time(6, 0))

    assert after_hours.covers(datetime(2026, 8, 30, 23, 0, tzinfo=timezone.utc)) is True
    assert after_hours.covers(datetime(2026, 8, 30, 3, 0, tzinfo=timezone.utc)) is True
    assert after_hours.covers(datetime(2026, 8, 30, 14, 0, tzinfo=timezone.utc)) is False


def test_a_daytime_schedule_does_not_wrap():
    working = Schedule(time(9, 0), time(17, 0))

    assert working.covers(datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)) is True
    assert working.covers(datetime(2026, 8, 30, 22, 0, tzinfo=timezone.utc)) is False


def test_the_window_excludes_its_end():
    # Half-open, so two adjacent windows cannot both claim the same instant.
    window = Schedule(time(9, 0), time(17, 0))

    assert window.covers(datetime(2026, 8, 30, 9, 0, tzinfo=timezone.utc)) is True
    assert window.covers(datetime(2026, 8, 30, 17, 0, tzinfo=timezone.utc)) is False


def test_a_schedule_can_be_limited_to_certain_days():
    weekend = Schedule(time(0, 0), time(23, 59), days=frozenset({6, 7}))

    assert weekend.covers(datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)) is True   # Saturday
    assert weekend.covers(datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc)) is False  # Monday


def test_a_zone_with_no_schedule_is_always_active():
    assert zone(schedule=None).is_active(datetime(2026, 8, 30, 3, 0, tzinfo=timezone.utc))


# -------------------------------------------------------------------- presence


def test_a_brief_touch_of_the_boundary_is_not_a_presence():
    # Somebody walking along a fence clips it as the estimate jitters. Without
    # the entry delay that is an event, and then another, and another.
    evaluator = ZoneEvaluator([zone(enter_after_millis=1000)])
    track = make_track(1, SITE)

    assert evaluator.update([track], 0) == []
    assert evaluator.update([track], 200) == []
    assert evaluator.update([], 400) == []
    assert evaluator.open_presences() == ()


def test_sustained_membership_becomes_a_presence():
    evaluator = ZoneEvaluator([zone(enter_after_millis=600)])
    track = make_track(1, SITE)

    changes = []
    for step in range(6):
        changes.extend(evaluator.update([track], step * 200))

    entered = [c for c in changes if c.kind == "ENTERED"]
    assert len(entered) == 1, "a presence must be reported exactly once"
    assert entered[0].presence.track_id == 1


def test_a_presence_survives_a_detector_dropout():
    # The whole reason exit is slower than entry. One person loitering for four
    # minutes must not become eight two-minute intrusions.
    evaluator = ZoneEvaluator([zone(enter_after_millis=400, exit_after_millis=2000)])
    track = make_track(1, SITE)

    for step in range(5):
        evaluator.update([track], step * 200)

    # Two frames where the detector lost it, but the tracker still holds it.
    evaluator.update([track], 1000)
    changes = evaluator.update([track], 1600)

    assert not any(c.kind == "LEFT" for c in changes)
    assert len(evaluator.open_presences()) == 1


def test_a_presence_ends_once_the_absence_is_real():
    evaluator = ZoneEvaluator([zone(enter_after_millis=400, exit_after_millis=1000)])
    track = make_track(1, SITE)

    for step in range(5):
        evaluator.update([track], step * 200)

    changes = evaluator.update([], 10_000)
    left = [c for c in changes if c.kind == "LEFT"]

    assert len(left) == 1
    assert left[0].presence.duration_millis == pytest.approx(800, abs=1)


def test_a_presence_that_never_confirmed_is_never_reported_as_leaving():
    # Otherwise a boundary touch produces no entry and a spurious exit.
    evaluator = ZoneEvaluator([zone(enter_after_millis=5000, exit_after_millis=200)])
    evaluator.update([make_track(1, SITE)], 0)

    assert evaluator.update([], 10_000) == []


def test_a_deactivated_schedule_closes_what_it_was_holding():
    # Otherwise a presence spanning the end of the window reopens hours later
    # with a duration covering the whole night.
    night = zone(schedule=Schedule(time(18, 0), time(6, 0)), enter_after_millis=400)
    evaluator = ZoneEvaluator([night])
    track = make_track(1, SITE)

    at_night = datetime(2026, 8, 30, 3, 0, tzinfo=timezone.utc)
    for step in range(5):
        evaluator.update([track], step * 200, at_night)
    assert len(evaluator.open_presences()) == 1

    morning = datetime(2026, 8, 30, 8, 0, tzinfo=timezone.utc)
    changes = evaluator.update([track], 2000, morning)

    assert any(c.kind == "LEFT" for c in changes)
    assert evaluator.open_presences() == ()


def test_presence_confidence_reflects_how_much_was_confidently_observed():
    # A presence built from uncertain positions is a weaker claim, and the
    # number that says so must travel with it into the event.
    watching = zone(kind=ZoneKind.INTEREST, accept_uncertain=True, enter_after_millis=200)
    evaluator = ZoneEvaluator([watching])

    confident = make_track(1, SITE, uncertainty=0.5)
    vague = make_track(1, destination_point(SITE, 0.0, 39.0), uncertainty=15.0)

    for step in range(3):
        evaluator.update([confident], step * 200)
    for step in range(3, 6):
        evaluator.update([vague], step * 200)

    presence = evaluator.open_presences()[0]
    assert 0.0 < presence.confidence < 1.0
    assert presence.uncertain_observations == 3


def test_two_tracks_in_one_zone_are_two_presences():
    evaluator = ZoneEvaluator([zone(enter_after_millis=200)])
    tracks = [make_track(1, SITE), make_track(2, destination_point(SITE, 90.0, 5.0))]

    changes = []
    for step in range(4):
        changes.extend(evaluator.update(tracks, step * 200))

    assert len({c.presence.track_id for c in changes if c.kind == "ENTERED"}) == 2


def test_one_track_in_two_zones_is_two_presences():
    overlapping = [
        zone(id="a", name="A", enter_after_millis=200),
        zone(id="b", name="B", ring=ring_around(SITE, 60.0), enter_after_millis=200),
    ]
    evaluator = ZoneEvaluator(overlapping)
    track = make_track(1, SITE)

    changes = []
    for step in range(4):
        changes.extend(evaluator.update([track], step * 200))

    assert {c.presence.zone_id for c in changes if c.kind == "ENTERED"} == {"a", "b"}


# ------------------------------------------------------------- a usable ring


def test_a_self_intersecting_outline_is_refused():
    # A figure of eight has no inside: point-in-polygon flips depending on the
    # lobe, so events would fire at random. Shapely decides, not a hand-rolled
    # segment test.
    from sentinel.zones import Zone, ZoneKind, ring_problem
    from sentinel.core import LatLon

    bow_tie = (
        LatLon(33.8938, 35.5018), LatLon(33.8939, 35.5019),
        LatLon(33.8938, 35.5019), LatLon(33.8939, 35.5018),
    )
    assert ring_problem(bow_tie) is not None
    with pytest.raises(ValueError, match="self-intersection"):
        Zone(id="z", name="Bow tie", kind=ZoneKind.RESTRICTED, ring=bow_tie)


def test_collinear_points_are_not_an_area():
    from sentinel.zones import ring_problem
    from sentinel.core import LatLon

    line = (LatLon(33.8938, 35.5018), LatLon(33.8939, 35.5019), LatLon(33.8940, 35.5020))
    assert "no area" in (ring_problem(line) or "")


def test_a_simple_outline_is_accepted_whatever_its_winding():
    from sentinel.zones import ring_problem
    from sentinel.core import LatLon

    square = (
        LatLon(33.8938, 35.5018), LatLon(33.8939, 35.5018),
        LatLon(33.8939, 35.5019), LatLon(33.8938, 35.5019),
    )
    assert ring_problem(square) is None
    assert ring_problem(tuple(reversed(square))) is None
