"""Coverage: what these cameras reach, and — the useful half — what they miss.

Every number this produces is an upper bound, because nothing models occlusion
or resolution. The tests hold that: a camera aimed at a wall covers the wall's
ground as far as this is concerned, and it must say so rather than pretend
otherwise.
"""

import pytest

from vigil.domain.geo import CameraPose, LatLon, destination_point
from vigil.service.coverage import (
    MINIMUM_GAP_M2, SIGMA_BANDS_M, Band, CoverageError, Gap, analyse, boundary_from_cameras,
    unwatched_share,
)

pytest.importorskip("shapely")

SITE = LatLon(33.8938, 35.5018)


def _pose(bearing=0.0, distance=0.0, heading=0.0, pitch=-25.0, **kwargs):
    position = destination_point(SITE, bearing, distance) if distance else SITE
    return CameraPose(position, kwargs.pop("mount_height", 4.0), heading, pitch,
                      kwargs.pop("roll", 0.0), kwargs.pop("horizontal_fov", 62.0),
                      kwargs.pop("vertical_fov", 36.0), kwargs.pop("range_meters", 40.0))


def _square(side_m: float, centre: LatLon = SITE):
    half = side_m / 2
    return [
        destination_point(centre, b, half * 1.41421356)
        for b in (225.0, 135.0, 45.0, 315.0)
    ]


def test_one_camera_covers_part_of_a_yard_and_the_rest_is_reported_as_gaps():
    boundary = _square(60.0)
    coverage = analyse(boundary, {"gate": _pose()})
    assert 3400 < coverage.boundary_m2 < 3700, coverage.boundary_m2
    assert 0.05 < coverage.fraction < 0.6, coverage.describe()
    assert coverage.gaps, "a 62-degree camera cannot cover a square yard"
    assert isinstance(coverage.gaps[0], Gap)
    assert coverage.gaps[0].area_m2 >= MINIMUM_GAP_M2
    assert coverage.gaps[0].span_m > 5.0
    assert "across" in coverage.gaps[0].describe()
    assert coverage.cameras == ("gate",)
    assert "upper bound" in coverage.describe()


def test_covered_plus_uncovered_is_the_whole_boundary():
    boundary = _square(50.0)
    coverage = analyse(boundary, {"a": _pose(), "b": _pose(heading=180.0)})
    accounted = coverage.covered_m2 + sum(g.area_m2 for g in coverage.gaps)
    # Gaps under the minimum are not reported, so the sum is a little short —
    # never over, which would mean coverage was being double-counted.
    assert accounted <= coverage.boundary_m2 + 1.0
    assert accounted > coverage.boundary_m2 * 0.95


def test_overlapping_cameras_are_a_union_not_a_sum():
    """Two cameras pointed at the same ground cover it once. A sum would tell
    an installer that four cameras in a corner had covered four times the
    corner."""
    boundary = _square(60.0)
    one = analyse(boundary, {"a": _pose()})
    two = analyse(boundary, {"a": _pose(), "b": _pose(heading=2.0)})
    assert two.covered_m2 < one.covered_m2 * 1.3, (
        f"nearly-identical cameras must not double-count: {one.covered_m2:.0f} -> {two.covered_m2:.0f}"
    )
    assert two.covered_m2 >= one.covered_m2 - 1.0


def test_more_cameras_never_reduce_coverage():
    boundary = _square(80.0)
    cameras = {}
    previous = 0.0
    for i, heading in enumerate((0.0, 90.0, 180.0, 270.0)):
        cameras[f"c{i}"] = _pose(heading=heading)
        covered = analyse(boundary, cameras).covered_m2
        assert covered >= previous - 1e-6, "adding a camera reduced coverage"
        previous = covered
    assert previous > analyse(boundary, {"c0": _pose()}).covered_m2 * 2


def test_error_bands_are_nested_and_the_tightest_is_the_smallest():
    """"Covered" and "covered well enough to say which side of a line somebody
    was on" are different questions, and only the second decides whether a
    zone can be adjudicated at all."""
    coverage = analyse(_square(80.0), {"gate": _pose()})
    assert all(isinstance(b, Band) for b in coverage.bands)
    areas = [b.area_m2 for b in coverage.bands]
    assert [b.sigma_m for b in coverage.bands] == list(SIGMA_BANDS_M)
    assert "or better" in coverage.bands[0].describe()
    assert areas == sorted(areas), f"a looser error band must cover at least as much: {areas}"
    assert areas[-1] <= coverage.covered_m2 + 1.0, "no band can exceed the footprint"
    assert areas[0] < coverage.covered_m2, (
        "half-metre accuracy over the whole footprint would mean the error model is not working"
    )


def test_a_camera_that_sees_no_ground_is_named_rather_than_silently_dropped():
    coverage = analyse(_square(60.0), {"gate": _pose(), "sky": _pose(pitch=20.0)})
    assert coverage.cameras == ("gate",)
    assert any(name == "sky" for name, _ in coverage.skipped)
    assert "horizon" in dict(coverage.skipped)["sky"]
    assert "sky" in coverage.describe()


def test_a_camera_pointed_off_site_is_named_too():
    far = _pose(bearing=0.0, distance=500.0, heading=0.0)
    coverage = analyse(_square(40.0), {"gate": _pose(), "far": far})
    assert "far" not in coverage.cameras
    assert "outside the boundary" in dict(coverage.skipped)["far"]


def test_a_boundary_that_encloses_nothing_is_refused():
    with pytest.raises(CoverageError):
        analyse([SITE, SITE], {"gate": _pose()})
    with pytest.raises(CoverageError):
        analyse([SITE, SITE, SITE], {"gate": _pose()})


def test_a_boundary_can_be_derived_from_the_cameras_and_flatters_them():
    cameras = {"a": _pose(), "b": _pose(heading=180.0)}
    boundary = boundary_from_cameras(cameras)
    assert len(boundary) == 4
    coverage = analyse(boundary, cameras)
    # It is defined by where the cameras point, so it always looks good. The
    # docstring says so; this holds it to saying so.
    assert coverage.fraction > 0.2
    with pytest.raises(CoverageError):
        boundary_from_cameras({})


def test_the_one_line_check_agrees_with_the_full_report():
    boundary = _square(60.0)
    cameras = {"gate": _pose()}
    assert abs(unwatched_share(boundary, cameras) - (1 - analyse(boundary, cameras).fraction)) < 1e-9


def test_roll_changes_the_footprint_and_therefore_the_coverage():
    """v1's coverage inherited the decoupled camera model's error and ignored
    roll entirely. A rolled camera covers different ground, and the number has
    to move with it."""
    boundary = _square(60.0)
    level = analyse(boundary, {"gate": _pose()}).covered_m2
    rolled = analyse(boundary, {"gate": _pose(roll=25.0)}).covered_m2
    assert abs(level - rolled) > 1.0, f"roll changed nothing: {level:.1f} vs {rolled:.1f}"
