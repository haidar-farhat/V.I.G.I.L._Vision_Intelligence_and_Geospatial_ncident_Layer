"""Tests for coverage and blind-spot analysis.

The question this answers is the one a plan view cannot: six cameras drawn as
six overlapping wedges look like thorough coverage, and the four-metre corridor
between two of them looks like nothing at all until somebody walks down it.

Two properties carry the weight here, and neither is about geometry being
pretty:

- **The numbers are an upper bound and say so.** Nothing models occlusion or
  resolution, both of which only ever make real coverage smaller. A security
  tool may err towards optimism only if it is labelled.
- **A gap's area is the polygon's, not its outline's.** An uncovered region is
  usually the site with a camera-shaped hole in it, so measuring the outline
  reported a 10,446 m² gap on a 14,400 m² site as 14,400 m² — alarming, and
  false.
"""

from __future__ import annotations

import math

import pytest

from sentinel import logs
from sentinel.core import CameraPose, LatLon, destination_point, haversine_distance
from sentinel.coverage import ARC_SEGMENTS, Coverage, CoverageError, Gap, analyse

SITE_METRES = 120.0
ORIGIN = LatLon(33.8938, 35.5018)


@pytest.fixture(autouse=True)
def clean_logging():
    logs.reset()
    yield
    logs.reset()


@pytest.fixture
def site() -> list[LatLon]:
    """A 120 m square, built with the same geodesy the analysis uses."""
    north_east = destination_point(ORIGIN, 90.0, SITE_METRES)
    return [
        ORIGIN,
        north_east,
        destination_point(north_east, 180.0, SITE_METRES),
        destination_point(ORIGIN, 180.0, SITE_METRES),
    ]


def camera(at: LatLon, heading: float, *, pitch: float = -22.0,
           range_meters: float = 90.0) -> CameraPose:
    return CameraPose(
        position=at, mount_height=6.0, heading=heading, pitch=pitch,
        horizontal_fov=62.0, vertical_fov=36.0, range_meters=range_meters,
    )


def on_north_edge(offset: float) -> LatLon:
    return destination_point(ORIGIN, 90.0, offset)


def on_south_edge(offset: float) -> LatLon:
    return destination_point(destination_point(ORIGIN, 180.0, SITE_METRES), 90.0, offset)


# ------------------------------------------------------------------- the area


def test_the_site_area_is_right(site):
    result = analyse(site, {})

    # 120 m square. A percent of slack for the projection, which is a great deal
    # tighter than any real site boundary is surveyed.
    assert result.site_area_m2 == pytest.approx(SITE_METRES**2, rel=0.01)


def test_no_cameras_means_nothing_is_covered(site):
    result = analyse(site, {})

    assert result.covered_area_m2 == 0.0
    assert result.covered_fraction == 0.0
    assert len(result.gaps) == 1
    assert result.gaps[0].area_m2 == pytest.approx(result.site_area_m2, rel=0.01)


def test_a_camera_covers_something_but_not_everything(site):
    result = analyse(site, {"north": camera(on_north_edge(60.0), 180.0)})

    assert 0.0 < result.covered_fraction < 1.0
    assert result.uncovered_area_m2 > 0
    assert result.gaps


def test_a_gap_is_measured_with_its_holes_subtracted(site):
    # The defect this pins. A camera in the middle of a site punches a hole in
    # the uncovered region, so that region's *outline* is the site boundary.
    # Measuring the outline reported the gap as the whole site.
    result = analyse(site, {"middle": camera(on_north_edge(60.0), 180.0)})

    gap = result.gaps[0]
    assert gap.holes, "the covered wedge should be a hole in the gap"
    assert gap.area_m2 < result.site_area_m2
    assert gap.area_m2 == pytest.approx(result.uncovered_area_m2, rel=0.02)
    assert result.largest_gap_m2 == gap.area_m2


def test_more_cameras_cover_more(site):
    one = analyse(site, {"north": camera(on_north_edge(60.0), 180.0)})
    two = analyse(site, {
        "north": camera(on_north_edge(60.0), 180.0),
        "south": camera(on_south_edge(60.0), 0.0),
    })

    assert two.covered_area_m2 > one.covered_area_m2
    assert two.uncovered_area_m2 < one.uncovered_area_m2


def test_overlapping_cameras_are_not_counted_twice(site):
    # A union, not a sum. Two cameras looking at the same ground cover that
    # ground once, and a sum would report more coverage than the site has.
    alone = analyse(site, {"a": camera(on_north_edge(60.0), 180.0)})
    twice = analyse(site, {
        "a": camera(on_north_edge(60.0), 180.0),
        # The same pose under another name.
        "b": camera(on_north_edge(60.0), 180.0),
    })

    assert twice.covered_area_m2 == pytest.approx(alone.covered_area_m2, rel=1e-6)
    assert twice.covered_fraction <= 1.0


def test_coverage_never_exceeds_the_site(site):
    # Cameras ringing the site see far beyond it. Only what is inside counts,
    # or a small site with long-range cameras would report 300% covered.
    cameras = {
        f"c{index}": camera(on_north_edge(offset), 180.0, range_meters=400.0)
        for index, offset in enumerate((0.0, 40.0, 80.0, 120.0))
    }

    result = analyse(site, cameras)

    assert result.covered_area_m2 <= result.site_area_m2 * 1.001
    assert result.covered_fraction <= 1.0


# ---------------------------------------------------------- cameras that do not


def test_a_camera_pointed_above_the_horizon_is_named(site):
    # It sees no ground at all. Not a broken camera — a camera pointed at the
    # sky — and the report says which one rather than quietly omitting it.
    result = analyse(site, {"sky": camera(ORIGIN, 180.0, pitch=25.0)})

    assert result.blind_cameras == ("sky",)
    assert result.covered_area_m2 == 0.0


def test_a_camera_looking_away_from_the_site_is_named(site):
    # It sees plenty of ground, none of it here. Just as much a finding: an
    # installer has mounted a camera that contributes nothing to this site.
    away = destination_point(ORIGIN, 0.0, 300.0)
    result = analyse(site, {"elsewhere": camera(away, 0.0)})

    assert result.blind_cameras == ("elsewhere",)
    assert result.covered_area_m2 == 0.0


def test_a_working_camera_is_not_called_blind(site):
    result = analyse(site, {"north": camera(on_north_edge(60.0), 180.0)})

    assert result.blind_cameras == ()


# ------------------------------------------------------------------ boundaries


def test_a_boundary_needs_three_points():
    with pytest.raises(CoverageError, match="three points"):
        analyse([ORIGIN, destination_point(ORIGIN, 90.0, 10.0)], {})


def test_a_boundary_enclosing_nothing_is_refused():
    line = [ORIGIN, destination_point(ORIGIN, 90.0, 10.0), ORIGIN]

    with pytest.raises(CoverageError, match="no area"):
        analyse(line, {})


def test_a_self_intersecting_boundary_is_repaired_rather_than_refused():
    # A figure of eight, which is what a typo in one vertex produces. Refusing
    # gives an operator nothing to act on; repairing gives them a number and a
    # shape they can look at and correct.
    north_east = destination_point(ORIGIN, 90.0, 100.0)
    south_east = destination_point(north_east, 180.0, 100.0)
    south_west = destination_point(ORIGIN, 180.0, 100.0)
    bowtie = [ORIGIN, south_east, north_east, south_west]

    result = analyse(bowtie, {})

    assert result.site_area_m2 > 0


def test_a_sliver_is_not_reported_as_a_gap(site):
    # A wedge of arc-approximation error between two overlapping footprints is
    # not a place somebody can stand, and reporting one teaches an operator to
    # ignore the field that reports real ones.
    cameras = {
        f"c{index}": camera(on_north_edge(offset), 180.0, range_meters=400.0)
        for index, offset in enumerate(range(0, 130, 15))
    }

    result = analyse(site, cameras, minimum_gap_m2=4.0)

    assert all(gap.area_m2 >= 4.0 for gap in result.gaps)


# -------------------------------------------------------------- what it claims


def test_the_report_says_it_is_an_upper_bound(site):
    # Nothing here models occlusion or resolution, and both only ever make real
    # coverage smaller. Erring towards optimism is defensible only when said.
    text = analyse(site, {"north": camera(on_north_edge(60.0), 180.0)}).describe()

    assert "upper bound" in text
    assert "occludes" in text
    assert "never larger" in text


def test_a_gap_ring_is_a_real_place_on_the_ground(site):
    # The coordinates have to be somewhere an operator can walk to, not an
    # artefact of the projection. Every vertex of a gap should be within a site
    # diagonal of the origin.
    result = analyse(site, {"north": camera(on_north_edge(60.0), 180.0)})

    diagonal = SITE_METRES * math.sqrt(2) * 1.05
    for point in result.gaps[0].ring:
        assert haversine_distance(ORIGIN, point) <= diagonal


def test_the_projection_round_trips(site):
    # The frame converts to metres and back. If it did not round-trip, every
    # gap would be reported somewhere slightly wrong — and nothing else in the
    # system would notice.
    from sentinel.coverage import _Frame

    frame = _Frame(ORIGIN)
    for point in site:
        x, y = frame.to_xy(point)
        back = frame.to_latlon(x, y)
        assert haversine_distance(point, back) < 0.01
