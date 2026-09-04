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
from sentinel.coverage import (
    ARC_SEGMENTS,
    SIGMA_THRESHOLDS_M,
    Coverage,
    CoverageError,
    Gap,
    analyse,
    sigma_bands,
    zone_report,
)

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


# ------------------------------------------- what the camera can actually rule on
#
# Coverage says a camera can *reach* this ground. These say how well it knows
# where anything on it is — which is the question that decides whether a zone
# drawn there can be adjudicated or will only ever report UNCERTAIN.


GATE = CameraPose(
    position=ORIGIN, mount_height=6.0, heading=180.0, pitch=-22.0,
    horizontal_fov=62.0, vertical_fov=36.0, range_meters=90.0,
)


def square_at(distance: float, half: float, pose: CameraPose = GATE) -> tuple[LatLon, ...]:
    """A square of side ``2 * half``, centred ``distance`` ahead of the camera."""
    centre = destination_point(pose.position, pose.heading, distance)
    return tuple(
        destination_point(centre, bearing, half * math.sqrt(2))
        for bearing in (45.0, 135.0, 225.0, 315.0)
    )


def band_reach(band) -> float:
    """How far from the mast the band extends."""
    return max(haversine_distance(GATE.position, point) for point in band.ring)


def test_sigma_bands_lie_further_out_the_looser_they_are():
    # The whole premise: position error grows with distance, so the ground a
    # camera knows to half a metre is a band hugging its near edge, and the
    # ground it knows to five metres reaches much further.
    bands = sigma_bands(GATE)

    assert [band.threshold_m for band in bands] == list(SIGMA_THRESHOLDS_M)
    reaches = [band_reach(band) for band in bands]
    assert reaches == sorted(reaches), f"bands are not nested: {reaches}"

    # Measured on the reference pose (6 m mast, -22°, 36° vertical field), then
    # floored: 8.2 m for the tightest band and 33.0 m for the loosest.
    assert 6.0 < reaches[0] <= 10.0, f"the half-metre band reaches {reaches[0]:.1f} m"
    assert 25.0 <= reaches[-1] <= 40.0, f"the five-metre band reaches {reaches[-1]:.1f} m"
    # And none of them beyond the range the pose claims.
    assert reaches[-1] < GATE.range_meters


def test_sigma_bands_are_computed_once_per_pose():
    # Roughly two thousand calls across the FFI. Once per placement that is
    # nothing; once per repaint it would make the map unusable, so the cache is
    # a correctness property of the drawing path, not an optimisation.
    sigma_bands.cache_clear()
    first = sigma_bands(GATE)
    hits_before = sigma_bands.cache_info().hits

    again = sigma_bands(
        CameraPose(
            position=ORIGIN, mount_height=6.0, heading=180.0, pitch=-22.0,
            horizontal_fov=62.0, vertical_fov=36.0, range_meters=90.0,
        )
    )
    assert again is first, "an equal pose recomputed the bands"
    assert sigma_bands.cache_info().hits == hits_before + 1

    steeper = sigma_bands(
        CameraPose(
            position=ORIGIN, mount_height=6.0, heading=180.0, pitch=-23.0,
            horizontal_fov=62.0, vertical_fov=36.0, range_meters=90.0,
        )
    )
    assert steeper is not first, "a different pose returned a cached answer"


def test_a_camera_pointed_at_the_sky_has_no_bands():
    blind = CameraPose(
        position=ORIGIN, mount_height=6.0, heading=180.0, pitch=10.0,
        horizontal_fov=62.0, vertical_fov=36.0, range_meters=90.0,
    )
    assert sigma_bands(blind) == ()


def test_a_zone_at_the_near_edge_can_be_adjudicated():
    # Measured: covered 0.982 (a corner falls in the blind foreground under the
    # mast), confident 0.982, best 0.5 m, worst 1.0 m.
    report = zone_report(square_at(9.0, 2.0), {"gate": GATE})

    assert report.covered_fraction >= 0.95
    assert report.confident_fraction >= 0.9
    assert report.cameras == ("gate",)
    assert report.best_sigma_m == 0.5
    assert report.worst_sigma_m is not None and report.worst_sigma_m <= 2.0
    assert 15.0 <= report.area_m2 <= 17.0


def test_a_zone_at_the_far_edge_is_covered_and_unadjudicable():
    # The failure this exists to make visible: fully covered, and the system
    # still cannot say which side of a four-metre line somebody is on.
    report = zone_report(square_at(70.0, 2.0), {"gate": GATE})

    assert report.covered_fraction >= 0.9, "the far zone is inside the footprint"
    assert report.confident_fraction <= 0.1
    assert report.best_sigma_m is None, "no band reaches 70 m"
    assert report.worst_sigma_m is None, "which reads as 'beyond five metres'"


def test_a_zone_outside_every_footprint_can_never_fire():
    behind = destination_point(GATE.position, GATE.heading + 180.0, 40.0)
    ring = tuple(
        destination_point(behind, bearing, 3.0 * math.sqrt(2))
        for bearing in (45.0, 135.0, 225.0, 315.0)
    )
    report = zone_report(ring, {"gate": GATE})

    assert report.covered_fraction == 0.0
    assert report.outside_fraction == 1.0
    assert report.cameras == ()
    assert report.confident_fraction == 0.0


def test_a_zone_the_width_of_a_threshold_is_not_dropped_by_rounding():
    """A four-metre square at 20 m must report the confidence it has.

    Its half-width is 2 m, exactly a threshold — but built from geodesic
    destination points the square measures 3.999990 m, so an exact `<=`
    excluded the two-metre band and reported 0% confident on a zone that is
    half inside it. Five microns, one wrong answer, and invisible in every
    zone whose width does not happen to match a threshold.
    """
    report = zone_report(square_at(20.0, 2.0), {"gate": GATE})

    assert report.covered_fraction >= 0.99
    assert report.confident_fraction >= 0.4, "the two-metre band was dropped"
    assert report.best_sigma_m == 2.0


def test_the_zone_report_agrees_with_the_coverage_analysis(site):
    # One union code path. Two that came to differ would have the zone panel
    # calling a zone covered while the blind-spot report showed a hole in
    # exactly that place.
    cameras = {"north": camera(on_north_edge(60.0), 180.0)}

    assert zone_report(site, cameras).covered_fraction == pytest.approx(
        analyse(site, cameras).covered_fraction, abs=1e-6
    )


def test_a_zone_needs_three_points():
    with pytest.raises(CoverageError, match="three points"):
        zone_report((ORIGIN, on_north_edge(10.0)), {"gate": GATE})
