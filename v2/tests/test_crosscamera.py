"""Linking one object across two cameras, and knowing when that cannot be done.

The interesting part is not the linking. It is that the threshold is measured
on this site, from pairs nobody labelled, and that when the measurement says
appearance cannot separate objects here the answer is to stop using it rather
than to use it anyway.
"""

from dataclasses import replace
from datetime import datetime, timezone

import numpy as np

from vigil.domain.appearance import (
    MIN_BALANCE_SAMPLES, MIN_SEPARATION_SAMPLES, Appearance, ColourBalance, CrossCameraSeparation,
)
from vigil.domain.detection import DetectorInfo
from vigil.domain.events import Event, EventType, Evidence, Severity
from vigil.domain.incidents import associate

BINS = 32


#: Real colour histograms of a person are broad — skin, hair, a coat, the
#: ground showing through — not a spike at one hue. Two dark coats differ by a
#: few bins, and that is the case where a camera's own tint dominates.
DESCRIPTOR_SPREAD = 8.0


def _look(peak: float, spread: float = DESCRIPTOR_SPREAD, gain=None) -> Appearance:
    """A descriptor with its mass around one bin — a coat of one colour."""
    x = np.arange(BINS, dtype=np.float64)
    vector = np.exp(-((x - peak) ** 2) / (2 * spread ** 2))
    if gain is not None:
        vector = vector * gain
    return Appearance(vector / np.linalg.norm(vector), 50_000)


def _tint(strength: float) -> np.ndarray:
    """What one camera does to every colour it sees: a smooth per-bin gain."""
    x = np.arange(BINS, dtype=np.float64) / BINS
    return 1.0 + strength * np.sin(x * 3.1)


def test_a_camera_tint_can_dominate_the_comparison_it_is_meant_to_survive():
    """The problem, as a number, before the fix is applied.

    One coat through two cameras must not look further apart than two
    different coats through one — and without correction it does.
    """
    warm, cool = _tint(1.4), _tint(-0.7)
    same_object_two_cameras = _look(12, gain=warm).distance(_look(12, gain=cool))
    two_objects_one_camera = _look(12, gain=warm).distance(_look(16, gain=warm))
    assert same_object_two_cameras > two_objects_one_camera, (
        f"this test has stopped demonstrating the problem: {same_object_two_cameras:.3f} "
        f"against {two_objects_one_camera:.3f}")


def test_normalising_each_camera_makes_two_cameras_comparable():
    warm, cool = _tint(1.4), _tint(-0.7)
    a, b = ColourBalance(), ColourBalance()
    rng = np.random.default_rng(3)
    for _ in range(MIN_BALANCE_SAMPLES + 20):
        peak = float(rng.uniform(0, BINS))
        a.observe(_look(peak, gain=warm))
        b.observe(_look(peak, gain=cool))
    assert a.measured and b.measured and "balanced" in a.describe()

    same = a.normalise(_look(12, gain=warm)).distance(b.normalise(_look(12, gain=cool)))
    different = a.normalise(_look(12, gain=warm)).distance(b.normalise(_look(16, gain=cool)))
    assert same < different, (
        f"after normalising, one object across two cameras ({same:.3f}) must look closer than "
        f"two objects ({different:.3f})")


def test_an_unmeasured_balance_passes_the_descriptor_through_untouched():
    """A correction nobody has measured is a guess, and a guessed gain on one
    bin is worse than no gain at all."""
    balance = ColourBalance()
    look = _look(8)
    for _ in range(MIN_BALANCE_SAMPLES - 1):
        balance.observe(_look(4))
    assert not balance.measured
    assert balance.normalise(look) is look


def test_the_cross_camera_threshold_is_measured_from_pairs_nobody_labelled():
    """Geometry decides which pairs are one object; appearance is then scored
    against that decision. No labels anywhere."""
    separation = CrossCameraSeparation()
    assert not separation.measured and separation.ceiling() is None
    assert "not yet measured" in separation.describe()

    rng = np.random.default_rng(11)
    for _ in range(MIN_SEPARATION_SAMPLES + 20):
        separation.observe_same(float(abs(rng.normal(0.04, 0.02))))
        separation.observe_different(float(abs(rng.normal(0.45, 0.10))))
    assert separation.measured and separation.usable
    ceiling = separation.ceiling()
    assert 0.02 < ceiling < 0.25, ceiling
    assert "linking below" in separation.describe()


def test_when_the_two_distributions_overlap_appearance_is_refused():
    """A corridor of people in dark coats. No threshold both links one object
    and keeps two apart, so the honest answer is to decline."""
    separation = CrossCameraSeparation()
    rng = np.random.default_rng(5)
    for _ in range(MIN_SEPARATION_SAMPLES + 20):
        separation.observe_same(float(abs(rng.normal(0.30, 0.12))))
        separation.observe_different(float(abs(rng.normal(0.33, 0.12))))
    assert separation.measured
    assert separation.ceiling() is None and not separation.usable
    assert "cannot separate" in separation.describe()


# ------------------------------------------------------ through the correlator


def _event(camera: str, track: int, at: int, lat: float, lon: float, look=()):
    evidence = Evidence(camera, track, 1,
                        DetectorInfo("m", "sha", {0: "person"}, True, (640, 640), "CPU"),
                        "person", lat, lon, 1.0, 5, (), "GROUND_PROJECTION", tuple(look))
    return Event(f"{camera}-{track}-{at}", EventType.ZONE_ENTRY, Severity.MEDIUM, "in the yard", at,
                 datetime.fromtimestamp(at / 1000, tz=timezone.utc), "node", "rule", 0.9, evidence,
                 "yard", "Yard")


def test_two_people_who_look_different_are_not_linked_across_cameras():
    """Place alone would join them: they are within the allowance and within
    the window. Appearance is what says they are two."""
    red, blue = tuple(_look(4, spread=2.0).vector), tuple(_look(20, spread=2.0).vector)
    events = [_event("west", 1, 1000, 33.8938, 35.5018, red),
              _event("east", 2, 1300, 33.89381, 35.50181, blue)]
    assert len(associate(events)) == 1, "place alone links them, which is the point"
    assert associate(events, appearance_ceiling=0.10) == []


def test_two_sightings_that_look_alike_are_linked_and_say_so():
    red = tuple(_look(4, spread=2.0).vector)
    events = [_event("west", 1, 1000, 33.8938, 35.5018, red),
              _event("east", 2, 1300, 33.89381, 35.50181, red)]
    links = associate(events, appearance_ceiling=0.10)
    assert len(links) == 1
    assert any("similarity, not an identity" in r for r in links[0].reasons)


def test_without_a_measured_ceiling_the_correlator_behaves_exactly_as_before():
    """Every site is in this state until two cameras have overlapped for a
    while, so it is the common path and must not have changed."""
    red, blue = tuple(_look(4, spread=2.0).vector), tuple(_look(20, spread=2.0).vector)
    events = [_event("west", 1, 1000, 33.8938, 35.5018, red),
              _event("east", 2, 1300, 33.89381, 35.50181, blue)]
    assert len(associate(events, appearance_ceiling=None)) == 1

    bare = [_event("west", 1, 1000, 33.8938, 35.5018),
            _event("east", 2, 1300, 33.89381, 35.50181)]
    assert len(associate(bare, appearance_ceiling=0.10)) == 1, (
        "an event with no descriptor must fall back to time and place, not be rejected")


def test_an_event_says_how_its_position_was_arrived_at():
    """Three sources worth very different amounts; an event that does not say
    which cannot be weighed."""
    from vigil.domain.geo import PositionSource

    event = _event("west", 1, 1000, 33.8938, 35.5018)
    assert event.evidence.position().source is PositionSource.GROUND_PROJECTION

    triangulated = replace(event.evidence, position_source="TRIANGULATED")
    assert triangulated.position().source is PositionSource.TRIANGULATED

    # A row written by a newer build names a source this one does not know.
    unknown = replace(event.evidence, position_source="LIDAR")
    assert unknown.position().source is PositionSource.GROUND_PROJECTION
    assert unknown.position().point.lat == 33.8938, "the position is still a position"
