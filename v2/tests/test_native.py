"""The Rust core and the NumPy that mirrors it must agree.

Two implementations of one model are safe exactly as long as something proves
they agree, and this is that proof. Where there is no NumPy mirror — the
ground rasteriser — the test checks the core against the *independent* exact
inverse in `vigil.domain.geo` instead, which is a stronger check than a
mirror would be.
"""

import math
from dataclasses import replace

import numpy as np
import pytest

from vigil.domain.geo import (
    CameraPose, LatLon, PoseUncertainty, distance_meters, image_coordinates, project_to_ground,
)
from vigil.kernel import filtering, native

pytestmark = pytest.mark.skipif(not native.available(),
                                reason=f"the engine core is not built: {native.fault()}")

POSES = [
    CameraPose(LatLon(33.8938, 35.5018), 4.0, 0.0, -25.0, 0.0, 62.0, 36.0, 60.0),
    CameraPose(LatLon(33.8938, 35.5018), 8.0, 137.0, -40.0, 7.0, 90.0, 50.0, 80.0),
    CameraPose(LatLon(-33.86, 151.20), 2.5, 271.0, -12.0, -11.0, 45.0, 25.0, 40.0),
    CameraPose(LatLon(64.14, -21.94), 6.0, 359.0, -60.0, 0.0, 110.0, 70.0, 30.0),
]
POINTS = [(u / 8, v / 8) for u in range(1, 8) for v in range(4, 9)]

#: The same poses with a real lens fitted. Without these the two
#: implementations agree only because the distortion is the identity on both
#: sides, which proves nothing about the code that applies it.
from dataclasses import replace as _replace  # noqa: E402

from vigil.domain.lens import Distortion  # noqa: E402

LENSED = [
    _replace(POSES[0], lens=Distortion(k1=-0.28, k2=0.09, p1=0.0006, p2=-0.0004, k3=-0.012)),
    _replace(POSES[2], lens=Distortion(k1=-0.15, k2=0.02)),
    _replace(POSES[0], lens=Distortion(p1=0.002, p2=-0.003)),
]


def test_the_core_and_the_python_project_to_the_same_place():
    """The one that matters: `core/src/camera.rs` exists only because the map
    builder cannot cross the boundary a thousand times a frame, and a second
    copy of a projection is a second answer waiting to differ."""
    worst_position = 0.0
    worst_sigma = 0.0
    compared = 0
    for pose in POSES + LENSED:
        results, status = native.project_batch(
            pose, POINTS, 0.75, pose.uncertainty, enforce_range=False
        )
        for (u, v), row, code in zip(POINTS, results, status):
            reference = project_to_ground(pose, u, v, 0.75, enforce_range=False)
            if reference is None:
                assert code != 0, f"the core projected {u},{v} where Python refused"
                continue
            assert code == 0, f"the core refused {u},{v} (code {code}) where Python projected"
            got = LatLon(row[0], row[1])
            worst_position = max(worst_position, distance_meters(got, reference.position))
            worst_sigma = max(worst_sigma, abs(row[4] - reference.uncertainty.along_meters))
            worst_sigma = max(worst_sigma, abs(row[5] - reference.uncertainty.across_meters))
            assert abs(row[2] - reference.ground_distance_meters) < 1e-9
            compared += 1
    assert compared > 100, f"only {compared} points compared"
    assert worst_position < 1e-6, f"positions differ by {worst_position} m"
    assert worst_sigma < 1e-6, f"uncertainties differ by {worst_sigma} m"


def test_the_core_and_the_python_invert_the_projection_to_the_same_pixel():
    worst = 0.0
    for pose in POSES + LENSED:
        points = []
        for u, v in POINTS:
            projection = project_to_ground(pose, u, v, enforce_range=False)
            if projection is not None:
                points.append((projection.position.lat, projection.position.lon))
        if not points:
            continue
        uv, status = native.image_coordinates_batch(pose, points)
        for (lat, lon), row, code in zip(points, uv, status):
            reference = image_coordinates(pose, LatLon(lat, lon))
            assert (reference is None) == (code != 0)
            if reference is not None:
                worst = max(worst, abs(row[0] - reference.x), abs(row[1] - reference.y))
    assert worst < 1e-9, f"the two inverses differ by {worst} of a frame"


def test_the_two_kalman_filters_stay_together_over_a_long_run():
    """The two must agree, and — the stronger claim — must not *drift* apart.

    They are not bit-identical and cannot be: the Rust filter takes its gain
    through a Cholesky factor and the NumPy one through a general solve, and
    `F P F'` is two rank updates on one side and two matrix products on the
    other. Measured, one warp applied to one identical state leaves the means
    bit-identical and the covariance differing by 1e-23.

    That 1e-23 then grows, because the velocity state is the least observable
    part of this filter and amplifies a covariance perturbation hard: 400
    steps take it to about 3e-8. So the test holds two things — a tolerance
    far below anything physical (1e-6 of a frame is a thousandth of a pixel),
    and that the gap stops growing, which is what separates rounding from a
    modelling difference.
    """
    rust = native.kalman_initiate(0.5, 0.6, 0.5, 0.2)
    numpy_state = filtering.initiate(np.zeros(filtering.STATE_VALUES), 0.5, 0.6, 0.5, 0.2)
    worst = 0.0
    early = 0.0
    rng = np.random.default_rng(11)
    for i in range(400):
        # Ragged timing on purpose: a dropped frame is where a filter with a
        # per-frame time constant would drift away from one with a real dt.
        dt = 1 / 15 if i % 7 else 0.31
        native.kalman_predict(rust, dt)
        filtering.predict(numpy_state, dt)
        if i % 11 == 3:
            warp = np.array([[1.0, 0.004, 0.002], [-0.004, 1.0, -0.001]])
            assert native.kalman_warp(rust, warp) == filtering.warp(numpy_state, warp)
        z = (0.5 + 0.002 * i + rng.normal(0, 0.002), 0.6 + rng.normal(0, 0.002), 0.5, 0.2)
        assert native.kalman_update(rust, *z) == filtering.update(numpy_state, *z)
        worst = max(worst, float(np.max(np.abs(rust - numpy_state))))
        if i == 99:
            early = worst
    assert worst < 1e-6, f"the two filters differ by {worst}"
    assert worst < early * 4 + 1e-12, (
        f"the gap is growing rather than settling: {early:.2e} at 100 steps, {worst:.2e} at 400"
    )
    # The box itself — the only part anything downstream reads — must agree to
    # far better than a pixel.
    assert abs(rust[0] - numpy_state[0]) < 1e-9

    # And the gate — where the two filters' small disagreement could actually
    # change an outcome. Relative, because a Mahalanobis distance of 3 500 is
    # a normal reading for a box on the other side of the frame, and what
    # matters is that both sides reach the same verdict about it.
    boxes = np.array([[0.5, 0.6, 0.5, 0.2], [0.9, 0.2, 0.5, 0.2], [0.51, 0.61, 0.49, 0.21]])
    for position_only in (False, True):
        limit = filtering.CHI2_GATE_2DOF if position_only else filtering.CHI2_GATE_4DOF
        a = native.kalman_gate(rust, boxes, position_only)
        b = filtering.gate(numpy_state, boxes, position_only)
        assert np.allclose(a, b, rtol=1e-6, atol=1e-9), f"gates differ: {a} vs {b}"
        assert ((a < limit) == (b < limit)).all(), "the two reached different verdicts"


def test_the_two_assignment_solvers_return_the_same_matching():
    rng = np.random.default_rng(5)
    for rows in range(1, 9):
        for cols in range(1, 9):
            for _ in range(6):
                cost = rng.random((rows, cols))
                # A forbidden pair now and then, which is the case where a
                # solver that treats infinity carelessly goes wrong.
                if rows > 1 and cols > 1:
                    cost[rng.integers(rows), rng.integers(cols)] = 1.0e9
                rust = native.assign(cost)
                numpy_result = native._assign_numpy(cost)
                assert (rust == numpy_result).all(), f"{rows}x{cols}: {rust} vs {numpy_result}"


def test_the_rasteriser_agrees_with_the_exact_inverse():
    """No NumPy mirror exists for this, so it is checked against the
    independent exact inverse instead — a stronger test than a mirror."""
    from vigil.service.mapping import Grid, site_lattice

    pose = POSES[0]
    grid = site_lattice(pose.position, 0.25, pose)
    # A frame whose blue channel encodes the column and green the row, at a
    # width where one colour level is exactly one pixel.
    width = height = 255
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(width, dtype=np.uint8)[None, :]
    image[:, :, 1] = np.arange(height, dtype=np.uint8)[:, None]
    image[:, :, 2] = 128

    cells = grid.cells
    colour = np.zeros(cells * 3, dtype=np.uint8)
    valid = np.zeros(cells, dtype=np.uint8)
    resolution = np.full(cells, np.inf, dtype=np.float32)
    filled = native.ortho_sample(pose, grid.values(), image, grid.rows, grid.cols, 33,
                                 colour, valid, resolution)
    assert filled > 1000, f"only {filled} cells filled"

    colour = colour.reshape(grid.rows, grid.cols, 3)
    valid = valid.reshape(grid.rows, grid.cols)
    worst = 0.0
    compared = 0
    for row in range(0, grid.rows, 3):
        for col in range(0, grid.cols, 3):
            if not valid[row, col]:
                continue
            point = grid.centre_of(row, col)
            # Only the near two thirds: past that one image row is metres of
            # ground and the comparison stops meaning anything.
            if distance_meters(pose.position, point) > 25.0:
                continue
            exact = image_coordinates(pose, point)
            if exact is None or not (0.02 <= exact.x <= 0.98 and 0.02 <= exact.y <= 0.98):
                continue
            worst = max(worst, abs(colour[row, col, 0] / 255.0 - exact.x) * width,
                        abs(colour[row, col, 1] / 255.0 - exact.y) * height)
            compared += 1
    assert compared > 200, f"only {compared} cells compared"
    assert worst < 1.5, f"the lattice inverse is {worst:.2f} px from the exact one"


def test_the_binding_refuses_a_core_that_is_not_the_right_core():
    # The ABI check is the thing standing between a moved signature and
    # plausible, wrong geometry. It is asserted rather than assumed.
    #
    # ABI 2 since the pose grew its lens: a core built for ABI 1 would read
    # fourteen values where nine were sent and invent five from whatever was
    # next in memory — a lens made of stack garbage, applied to every ray.
    assert native.ABI_VERSION == 2
    assert native.EXPECTED_LAYOUT == (14, 5, 6, 5, 72)
    assert native.loaded_from() is not None and native.loaded_from().is_file()


def test_a_bad_pose_is_refused_at_the_boundary_rather_than_computed():
    broken = replace(POSES[0], mount_height=float("nan"))
    with pytest.raises(native.NativeError):
        native.project_batch(broken, [(0.5, 0.8)], 0.75, broken.uncertainty)


def test_an_accumulator_releases_its_memory_when_it_is_closed():
    accumulator = native.MedianAccumulator(16, 4)
    colour = np.full(48, 100, dtype=np.uint8)
    valid = np.ones(16, dtype=np.uint8)
    for _ in range(4):
        accumulator.add(colour, valid)
    out_colour = np.zeros(48, dtype=np.uint8)
    out_valid = np.zeros(16, dtype=np.uint8)
    samples = np.zeros(16, dtype=np.uint16)
    deviation = np.zeros(16, dtype=np.uint8)
    disturbed = np.zeros(16, dtype=np.uint8)
    assert accumulator.result(1, out_colour, out_valid, samples, deviation, disturbed) == 16
    assert (out_colour == 100).all() and (samples == 4).all()
    accumulator.close()
    assert accumulator.closed
    with pytest.raises(native.NativeError):
        accumulator.add(colour, valid)
    accumulator.close()  # idempotent


def test_the_median_is_what_makes_a_map_out_of_samples():
    # The claim the whole mapping module rests on, at the smallest scale that
    # shows it: a cell is ground for most of its samples and a person for a
    # few, and the median is the ground.
    accumulator = native.MedianAccumulator(1, 11)
    for i in range(11):
        colour = np.array([200, 210, 220] if i in (4, 7) else [40, 45, 50], dtype=np.uint8)
        accumulator.add(colour, np.ones(1, dtype=np.uint8))
    out_colour = np.zeros(3, dtype=np.uint8)
    out_valid = np.zeros(1, dtype=np.uint8)
    samples = np.zeros(1, dtype=np.uint16)
    deviation = np.zeros(1, dtype=np.uint8)
    disturbed = np.zeros(1, dtype=np.uint8)
    accumulator.result(1, out_colour, out_valid, samples, deviation, disturbed)
    accumulator.close()
    assert list(out_colour) == [40, 45, 50], "the walker survived the median"
    # The median absolute deviation is zero and that is *correct*: two
    # outliers in eleven cannot move it, which is why the median is
    # trustworthy. The number that notices the walker is the disturbance.
    assert deviation[0] == 0
    assert disturbed[0] == 18, "2 of 11 samples were blocked"


def test_the_two_lens_models_agree_on_a_real_calibration():
    """The distortion is the identity on both sides until a camera is
    calibrated, so agreeing without a lens fitted proves nothing about the
    code that applies one. This fits three and checks the whole path."""
    for pose in LENSED:
        assert not pose.lens.is_identity
        # Rust's `ray` undistorts; Python's does too. Compare where the two
        # put the ground point, which is the only thing anything downstream
        # reads.
        results, status = native.project_batch(pose, POINTS, 0.75, pose.uncertainty,
                                               enforce_range=False)
        worst = 0.0
        seen = 0
        for (u, v), row, code in zip(POINTS, results, status):
            reference = project_to_ground(pose, u, v, 0.75, enforce_range=False)
            if reference is None:
                assert code != 0
                continue
            assert code == 0
            worst = max(worst, distance_meters(LatLon(row[0], row[1]), reference.position))
            seen += 1
        assert seen > 10, f"only {seen} points projected through {pose.lens.describe()}"
        assert worst < 1e-6, f"{pose.lens.describe()}: the two lenses differ by {worst} m"
