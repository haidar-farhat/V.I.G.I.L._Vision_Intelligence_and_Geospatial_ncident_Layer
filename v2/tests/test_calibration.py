"""Measuring a pose, and measuring how well it was measured.

The second is the point. A refined pose that does not say what it is worth has
replaced one assumption with another; the covariance is what makes it a
measurement, so the test that matters here is the Monte Carlo below, which
checks that the reported sigma actually brackets the true error.
"""

from dataclasses import replace

import numpy as np
import pytest

from vigil.domain.geo import CameraPose, LatLon, PoseUncertainty, Vec2, project_to_ground
from vigil.service.calibration import (
    MIN_POINTS, CalibrationError, Correspondence, calibrate_lens, calibrate_pose,
)

TRUTH = CameraPose(LatLon(33.8938, 35.5018), 4.0, 37.0, -22.0, 3.0, 62.0, 36.0, 60.0)

#: What an operator clicking a point on a 1080-line frame is worth.
CLICK_PIXELS = 4.0
CLICK_NOISE = CLICK_PIXELS / 1080.0


def _spread(pose=TRUTH, columns=(0.12, 0.35, 0.5, 0.68, 0.9), rows=(0.6, 0.75, 0.95)):
    """Points across the frame and across the range, which is what a fit needs."""
    out = []
    for u in columns:
        for v in rows:
            ground = project_to_ground(pose, u, v, enforce_range=False)
            if ground is not None:
                out.append(Correspondence(Vec2(u, v), ground.position, f"{u},{v}"))
    return out


def _typed_by_an_operator(pose=TRUTH):
    """The pose somebody would have typed: a few degrees and half a metre out."""
    return replace(pose, heading=34.0, pitch=-25.0, roll=0.0, mount_height=4.5)


def _noisy(rng, points):
    return [Correspondence(Vec2(c.image.x + rng.normal(0, CLICK_NOISE),
                                c.image.y + rng.normal(0, CLICK_NOISE)), c.ground)
            for c in points]


def test_a_pose_is_recovered_exactly_from_perfect_points():
    points = _spread()
    assert len(points) >= 12
    result = calibrate_pose(_typed_by_an_operator(), points)
    assert abs(result.pose.heading - TRUTH.heading) < 1e-3
    assert abs(result.pose.pitch - TRUTH.pitch) < 1e-3
    assert abs(result.pose.roll - TRUTH.roll) < 1e-3
    assert abs(result.pose.mount_height - TRUTH.mount_height) < 1e-3
    assert result.rms < 1e-6
    # Zero residuals mean zero covariance, which is the honest answer to
    # noiseless data and never happens on a real site.
    assert result.uncertainty.heading_deg < 1e-3


def test_the_reported_uncertainty_actually_brackets_the_true_error():
    """The claim the whole module rests on.

    A covariance nobody has checked is decoration. This adds realistic click
    noise, fits many times, and asks how often the true value falls inside the
    reported one sigma -- which for a correctly scaled covariance is about 68%
    of the time. Measured at 62-68% across the four parameters.
    """
    base = _spread()
    rng = np.random.default_rng(7)
    names = ("heading", "pitch", "roll", "mount_height")
    truth = {"heading": TRUTH.heading, "pitch": TRUTH.pitch, "roll": TRUTH.roll,
             "mount_height": TRUTH.mount_height}
    inside = dict.fromkeys(names, 0)
    runs = 120
    for _ in range(runs):
        result = calibrate_pose(_typed_by_an_operator(), _noisy(rng, base))
        got = {"heading": result.pose.heading, "pitch": result.pose.pitch,
               "roll": result.pose.roll, "mount_height": result.pose.mount_height}
        sigma = {"heading": result.uncertainty.heading_deg,
                 "pitch": result.uncertainty.pitch_deg,
                 "roll": result.uncertainty.roll_deg,
                 "mount_height": result.uncertainty.mount_height_m}
        for name in names:
            if abs(got[name] - truth[name]) <= sigma[name]:
                inside[name] += 1
    for name in names:
        coverage = inside[name] / runs
        assert 0.5 <= coverage <= 0.85, (
            f"{name}: the true value fell inside the reported 1 sigma {coverage:.0%} of the "
            f"time; a correctly scaled covariance gives about 68%"
        )


def test_measuring_the_pose_beats_assuming_it_by_more_than_an_order_of_magnitude():
    """The reason this module exists. At 40 m, two degrees of heading is 1.4 m
    of sideways error before the detector has contributed anything."""
    rng = np.random.default_rng(11)
    result = calibrate_pose(_typed_by_an_operator(), _noisy(rng, _spread()))
    assumed = PoseUncertainty()
    assert result.better_than(assumed)
    assert result.uncertainty.heading_deg < assumed.heading_deg / 10, (
        f"measured {result.uncertainty.heading_deg:.3f} deg against an assumed "
        f"{assumed.heading_deg} deg"
    )
    assert result.uncertainty.mount_height_m < assumed.mount_height_m


def test_too_few_points_is_refused_with_the_reason():
    with pytest.raises(CalibrationError, match="not enough"):
        calibrate_pose(TRUTH, _spread()[: MIN_POINTS - 1])


def test_points_clustered_in_one_corner_cannot_separate_the_parameters():
    """The arrangement that is actually degenerate.

    A cluster spanning 0.02% of the frame conditions at ~9.5e9, past the 1e8
    where an inverse stops carrying information and starts carrying rounding
    error, so it is refused rather than returned as a confident covariance.
    """
    cluster = _spread(columns=(0.5000, 0.5002), rows=(0.8000, 0.8002))
    assert len(cluster) >= MIN_POINTS
    with pytest.raises(CalibrationError, match="cannot separate"):
        calibrate_pose(_typed_by_an_operator(), cluster)


def test_points_down_one_image_column_are_not_degenerate():
    """The intuition this module originally encoded, and it was wrong.

    "Collinear points are degenerate" is true of a homography from a plane to a
    plane. It is *not* true here, because the fit also knows each point's
    range: heading and roll move a near point and a far point by different
    amounts even when the two lie along one ray. Measured at 1.6e4 -- four
    orders of magnitude inside the refusal -- and the pose comes back exact.
    Kept as a test so nobody re-tightens the check to catch a phantom.
    """
    down_one_column = _spread(columns=(0.5,), rows=(0.55, 0.65, 0.75, 0.85, 0.95))
    assert len(down_one_column) >= MIN_POINTS
    result = calibrate_pose(_typed_by_an_operator(), down_one_column)
    assert result.condition < 1e6
    assert abs(result.pose.heading - TRUTH.heading) < 1e-3
    assert abs(result.pose.mount_height - TRUTH.mount_height) < 1e-3


def test_a_mis_clicked_point_shows_up_as_the_worst_rather_than_hiding_in_the_rms():
    """A good RMS with one terrible point is a mis-clicked correspondence, not
    a bad pose, and the two need telling apart."""
    points = list(_spread())
    bad = len(points) // 2
    points[bad] = Correspondence(Vec2(points[bad].image.x + 0.25, points[bad].image.y),
                                 points[bad].ground, "mis-clicked")
    result = calibrate_pose(_typed_by_an_operator(), points)
    assert result.worst_index == bad, "the outlier must be named"
    assert result.worst > result.rms * 2


def test_a_calibration_knows_when_it_is_worse_than_the_assumption():
    """A fit that makes the pose less certain is not a calibration."""
    from vigil.service.calibration import Calibration

    poor = Calibration(replace(TRUTH, uncertainty=PoseUncertainty(5.0, 5.0, 5.0, 1.0, 0.02)),
                       (0.1,), 0.1, 0.1, 0, 3, 1e3)
    assert not poor.better_than(PoseUncertainty())
    good = Calibration(replace(TRUTH, uncertainty=PoseUncertainty(0.1, 0.1, 0.1, 0.01, 0.02)),
                       (0.001,), 0.001, 0.001, 0, 5, 1e3)
    assert good.better_than(PoseUncertainty())
    assert "heading" in good.describe()


def test_position_can_be_solved_when_there_are_control_points_for_it():
    """Off by default -- an operator's click on a map is worth about a metre
    and the orientation is worth two degrees, and at 40 m the second matters
    more. On, for a site with surveyed control points."""
    from vigil.domain.geo import destination_point, distance_meters

    moved = replace(TRUTH, position=destination_point(TRUTH.position, 45.0, 1.5))
    result = calibrate_pose(replace(moved, heading=34.0), _spread(), solve_position=True)
    assert distance_meters(result.pose.position, TRUTH.position) < 0.3, (
        "solving position should walk the camera back towards where the points say it is"
    )
    assert result.position_error_m is not None
    assert "position" in result.describe()


def test_position_is_solved_in_metres_so_the_degeneracy_check_measures_geometry():
    """Regression on a bug that made `solve_position` unusable.

    Solving latitude and longitude in *degrees* makes those Jacobian columns
    about 1e5 times the size of the angular ones and `J^T J` about 1e10 out,
    so its condition number measured the choice of units instead of the
    geometry: the same well-spread points came out at 3.3e13 in degrees
    against 2.2e3 in metres, and every position fit was refused as degenerate.
    """
    result = calibrate_pose(_typed_by_an_operator(), _spread(), solve_position=True)
    assert result.condition < 1e8, (
        f"six well-conditioned parameters came out at {result.condition:.1e}; position is being "
        f"solved in degrees again"
    )
    four = calibrate_pose(_typed_by_an_operator(), _spread())
    assert result.condition < four.condition * 1e4, (
        "adding two position parameters should worsen conditioning by a modest factor, not by "
        "the ten orders of magnitude a unit mismatch produces"
    )


def test_a_lens_calibration_refuses_when_it_cannot_see_the_board():
    with pytest.raises(CalibrationError, match="no images"):
        calibrate_lens([])
    blank = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(4)]
    with pytest.raises(CalibrationError, match="at least three"):
        calibrate_lens(blank)
