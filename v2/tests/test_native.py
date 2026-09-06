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
    # ABI 2 was the pose growing its lens: a core built for ABI 1 would read
    # fourteen values where nine were sent and invent five from whatever was
    # next in memory — a lens made of stack garbage, applied to every ray.
    # ABI 3 added triangulation and the ground plane, which an older core does
    # not export at all. ABI 4 grew the pose by the two ground tilts, and that
    # one is the dangerous kind: a short pose gives the core a ground plane
    # tilted by whatever was next in memory.
    assert native.ABI_VERSION == 5
    assert native.EXPECTED_LAYOUT == (16, 5, 6, 5, 72, 8, 8)
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


def test_the_two_triangulators_agree_over_a_field_of_geometries():
    """Both implementations, over pairs that succeed and pairs that are
    refused, including the refusal itself: a mirror that agrees on the answers
    and disagrees on which questions have one is not a mirror."""
    rng = np.random.default_rng(19)
    agreed = refused = 0
    for _ in range(400):
        target = np.array([rng.uniform(-40, 40), rng.uniform(5, 60), rng.uniform(0, 3)])
        a_o = np.array([rng.uniform(-30, 30), rng.uniform(-5, 5), rng.uniform(2, 9)])
        b_o = np.array([rng.uniform(-30, 30), rng.uniform(-5, 5), rng.uniform(2, 9)])
        # Half the pairs get a nudge, so some fail the gap check rather than
        # every pair meeting exactly.
        nudge = np.array([rng.normal(0, 2.0), rng.normal(0, 2.0), rng.normal(0, 0.5)])
        a_d, b_d = target - a_o, target + nudge - b_o
        rust_code, rust = native.triangulate(a_o, a_d, b_o, b_d, 0.25, native.MIN_PARALLAX_DEG)
        py_code, py = native._triangulate_numpy(
            a_o.astype(float), a_d.astype(float), b_o.astype(float), b_d.astype(float),
            0.25, native.MIN_PARALLAX_DEG, np.zeros(8))
        assert rust_code == py_code, f"one refused and the other did not: {rust_code} vs {py_code}"
        if rust_code == 0:
            assert np.allclose(rust, py, rtol=1e-9, atol=1e-9), f"{rust} vs {py}"
            agreed += 1
        else:
            refused += 1
    assert agreed > 50 and refused > 20, f"{agreed} agreed, {refused} refused — a thin test"


def test_the_two_plane_fits_draw_the_same_plane_including_the_random_draw():
    """The RANSAC draw is seeded and the generator is written out in both, so
    the two agree exactly rather than nearly. A library generator would make
    this test the loosest thing in the suite."""
    rng = np.random.default_rng(23)
    for trial in range(12):
        east, north = rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05)
        n = 60
        cloud = np.empty((n, 3))
        cloud[:, 0] = rng.uniform(-25, 25, n)
        cloud[:, 1] = rng.uniform(0, 50, n)
        cloud[:, 2] = east * cloud[:, 0] + north * cloud[:, 1] + rng.normal(0, 0.05, n)
        # A tenth of the points stand on something that is not the ground.
        cloud[: n // 10, 2] += 1.4
        rust = native.fit_plane(cloud, 0.3, 200, 1000 + trial)
        py = native._fit_plane_numpy(cloud, 0.3, 200, 1000 + trial, np.zeros(8))
        assert rust is not None and py is not None
        assert np.allclose(rust, py, rtol=1e-9, atol=1e-9), f"{rust} vs {py}"
        assert abs(rust[6] - east) < 0.01 and abs(rust[7] - north) < 0.01, (
            f"tilt came out {rust[6]:.4f},{rust[7]:.4f} against {east:.4f},{north:.4f}")
        assert rust[4] >= n * 0.8, "the ground should be the majority"


def test_a_triangulated_point_beats_flat_ground_when_the_object_is_not_on_it():
    """The reason this exists. A person standing 1.4 m up on a dock is placed
    metres past themselves by a flat-ground projection; two rays put them
    where they are."""
    truth = np.array([10.0, 35.0, 1.4])
    a_o, b_o = np.array([-15.0, 0.0, 5.0]), np.array([20.0, -2.0, 4.5])
    code, values = native.triangulate(a_o, truth - a_o, b_o, truth - b_o, 0.25,
                                      native.MIN_PARALLAX_DEG)
    assert code == 0
    assert np.allclose(values[:3], truth, atol=1e-9)

    # What the flat-ground assumption does with the same ray: it follows it to
    # z = 0 and lands long.
    direction = truth - a_o
    flat = a_o + direction * (a_o[2] / -direction[2])
    assert abs(flat[2]) < 1e-9
    error = float(np.linalg.norm(flat[:2] - truth[:2]))
    assert error > 5.0, f"the flat-ground error here is {error:.2f} m"


def test_the_core_and_the_python_agree_on_ground_that_is_not_level():
    """The tilt has to arrive on both sides of the boundary and mean the same
    thing. Sixteen values sent where fourteen were read would give the core a
    plane tilted by whatever was next in memory, and every position from that
    camera would be quietly, plausibly wrong."""
    for pose in POSES:
        for east, north in ((0.0, 0.0), (0.03, -0.02), (-0.05, 0.04), (0.08, 0.08)):
            tilted = replace(pose, ground_tilt_east=east, ground_tilt_north=north)
            for u, v in POINTS:
                mine = project_to_ground(tilted, u, v, enforce_range=False)
                theirs = native.project_batch(tilted, [(u, v)], 0.75, tilted.uncertainty)
                if mine is None:
                    assert theirs[1][0] != 0, "one refused and the other did not"
                    continue
                assert theirs[1][0] == 0, f"the core refused {u},{v} that Python projected"
                assert distance_meters(mine.position, LatLon(theirs[0][0][0], theirs[0][0][1])) < 1e-6


def test_zero_tilt_reproduces_the_level_plane_exactly_on_both_sides():
    """The guarantee every uncalibrated camera depends on: the ground tilt
    arriving in the projection must not move one existing answer."""
    for pose in POSES:
        explicit = replace(pose, ground_tilt_east=0.0, ground_tilt_north=0.0)
        for u, v in POINTS:
            a = project_to_ground(pose, u, v, enforce_range=False)
            b = project_to_ground(explicit, u, v, enforce_range=False)
            assert (a is None) == (b is None)
            if a is not None:
                assert a.position.lat == b.position.lat and a.position.lon == b.position.lon
                assert a.ground_distance_meters == b.ground_distance_meters


def test_the_two_suppressors_keep_exactly_the_same_boxes():
    """Including which of two equal scores survives.

    A quantised model emits equal scores constantly, and `argsort` is not
    stable by default — so without a defined tie-break the Rust and the NumPy
    would disagree now and then for a reason neither could be blamed for, on
    a frame nobody could reproduce.
    """
    rng = np.random.default_rng(31)
    for trial in range(30):
        count = int(rng.integers(1, 120))
        xyxy = np.empty((count, 4))
        xyxy[:, 0] = rng.uniform(0, 0.9, count)
        xyxy[:, 1] = rng.uniform(0, 0.9, count)
        xyxy[:, 2] = xyxy[:, 0] + rng.uniform(0.01, 0.15, count)
        xyxy[:, 3] = xyxy[:, 1] + rng.uniform(0.01, 0.15, count)
        # Quantised to two decimals on purpose, so ties are common.
        scores = np.round(rng.uniform(0.2, 0.99, count), 2)
        classes = rng.integers(0, 4, count)
        for soft in (False, True):
            rust = native.suppress(xyxy, scores, classes, 0.45, soft=soft)
            numpy_result = native._suppress_numpy(xyxy, scores, classes, 0.45, soft, 0.5, 0.2)
            assert list(rust) == list(numpy_result), (
                f"trial {trial}, soft={soft}: {list(rust)} vs {list(numpy_result)}")


def test_suppression_is_fast_enough_that_tiling_can_afford_it():
    """The measurement the kernel exists for. NumPy took 4.9 ms on 300
    proposals and tiling runs suppression once per tile, so a five-pass frame
    was spending most of its budget deciding what to throw away."""
    import time

    rng = np.random.default_rng(5)
    count = 300
    xyxy = np.empty((count, 4))
    xyxy[:, 0] = rng.uniform(0, 0.9, count)
    xyxy[:, 1] = rng.uniform(0, 0.9, count)
    xyxy[:, 2] = xyxy[:, 0] + rng.uniform(0.02, 0.1, count)
    xyxy[:, 3] = xyxy[:, 1] + rng.uniform(0.02, 0.1, count)
    scores = rng.uniform(0.25, 0.99, count)
    classes = rng.integers(0, 8, count)

    native.suppress(xyxy, scores, classes, 0.45, soft=True)
    started = time.perf_counter()
    for _ in range(50):
        native.suppress(xyxy, scores, classes, 0.45, soft=True)
    each = (time.perf_counter() - started) * 1000 / 50
    assert each < 1.0, f"soft suppression took {each:.3f} ms; NumPy did it in 4.9"
