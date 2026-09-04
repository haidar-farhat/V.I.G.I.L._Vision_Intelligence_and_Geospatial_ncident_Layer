"""Tests for the basemap the cameras draw themselves.

Four claims carry the weight here, and each of them is a way a generated
basemap goes wrong in a way nobody notices until an incident is being argued
about:

- **The inverse map is the pose's, not an approximation of somebody's own.**
  The patch is checked against :func:`~sentinel.core.image_coordinates`, the
  exact inverse of the projection every position in this system comes from. If
  the two ever disagreed, the basemap would be offset from the tracks drawn on
  top of it and every judgement made against it would be wrong by that offset.
- **Empty stays empty.** A cell no camera has looked at is never coloured, and
  emptiness is carried in a mask rather than inferred from a black pixel —
  because asphalt at dusk is a black pixel.
- **The median deletes what moves.** An object crossing a cell in a minority of
  frames contributes nothing to that cell's final colour. The converse is
  tested too: something parked there for most of the window *becomes* the map,
  which is the honest limit of the method rather than a bug in it.
- **The mosaic ranks per cell, not per camera.** The overlap is decided cell by
  cell by whichever camera's position error is smaller *there*.

No model file is needed for any of this. The images are synthetic and the
geometry is the real one.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from sentinel import logs
from sentinel.core import (
    CameraPose,
    LatLon,
    camera_sees,
    destination_point,
    field_of_view,
    image_coordinates,
)
from sentinel.coverage import _Frame
from sentinel.orthophoto import (
    DEFAULT_CAPACITY,
    MINIMUM_SAMPLES,
    GroundGrid,
    GroundPatch,
    MedianAccumulator,
    OrthophotoError,
    mosaic,
    sample_frame,
)

WIDTH = 640
HEIGHT = 480
ORIGIN = LatLon(33.8938, 35.5018)


@pytest.fixture(autouse=True)
def clean_logging():
    logs.reset()
    yield
    logs.reset()


def camera(at: LatLon, heading: float, *, pitch: float = -22.0) -> CameraPose:
    return CameraPose(
        position=at, mount_height=6.0, heading=heading, pitch=pitch,
        horizontal_fov=62.0, vertical_fov=36.0, range_meters=90.0,
    )


@pytest.fixture
def grid(reference_pose: CameraPose) -> GroundGrid:
    """A one-metre raster over the reference camera's footprint, with room around it.

    The margin matters: without ground *outside* the footprint on the raster
    there is nowhere for "this cell must stay empty" to be asserted.
    """
    return GroundGrid.covering(
        field_of_view(reference_pose), cell_size_m=1.0, margin_m=20.0
    )


def flat(colour: tuple[int, int, int]) -> np.ndarray:
    """A frame of one colour, so a sampled cell's provenance is unambiguous."""
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:, :] = colour
    return frame


def synthetic_patch(
    grid: GroundGrid,
    camera_id: str,
    *,
    colour: tuple[int, int, int],
    valid: np.ndarray,
    sigma_m: float,
    when: float,
) -> GroundPatch:
    """A patch built by hand, so mosaic logic is tested without any geometry.

    The point of building one is that the compositing rules — who wins a cell,
    what happens to a cell nobody covers — are decidable from the arrays alone,
    and a test that went through a real pose to reach them would be testing the
    projection again instead.
    """
    pixels = np.zeros((*grid.shape, 3), dtype=np.uint8)
    pixels[valid] = colour
    return GroundPatch(
        camera_id=camera_id,
        grid=grid,
        colour=pixels,
        valid=valid,
        sigma_m=np.where(valid, np.float32(sigma_m), np.nan).astype(np.float32),
        updated_at=np.where(valid, when, np.nan),
        samples=valid.astype(np.uint32),
    )


# ------------------------------------------------------------------- the grid


def test_a_cell_round_trips_from_indices_to_a_map_position_and_back(grid):
    for row, column in (
        (0, 0),
        (grid.rows - 1, grid.columns - 1),
        (grid.rows // 2, grid.columns // 3),
        (1, grid.columns - 2),
    ):
        assert grid.cell_of(grid.cell_centre(row, column)) == (row, column)


def test_a_cell_centre_is_the_centre_and_not_a_corner(grid):
    """Half a cell of offset is the difference between two sides of a doorway."""
    corner_x, corner_y = 0.0, 0.0
    x, y = grid.cell_xy(0, 0)
    print(f"cell (0,0) centre at {x:.3f} m east, {y:.3f} m north of the origin")
    assert x == pytest.approx(grid.cell_size_m / 2.0)
    assert y == pytest.approx(-grid.cell_size_m / 2.0)
    assert (x, y) != (corner_x, corner_y)


def test_the_grid_row_axis_runs_south_so_the_raster_is_north_up(grid):
    north_row = grid.cell_centre(0, grid.columns // 2)
    south_row = grid.cell_centre(grid.rows - 1, grid.columns // 2)
    print(f"row 0 at {north_row.lat:.6f}, last row at {south_row.lat:.6f}")
    assert north_row.lat > south_row.lat


def test_a_position_off_the_grid_gets_no_cell_rather_than_a_clamped_edge(grid):
    """Clamping would pile everything beyond the site onto its boundary row."""
    far = destination_point(grid.origin, 315.0, 5_000.0)
    assert grid.cell_of(far) is None


def test_a_grid_built_to_cover_a_footprint_contains_every_point_of_it(reference_pose):
    footprint = field_of_view(reference_pose)
    covering = GroundGrid.covering(footprint, cell_size_m=1.0, margin_m=1.0)
    missed = [point for point in footprint if covering.cell_of(point) is None]
    print(f"{covering.rows}×{covering.columns} cells, {len(missed)} footprint point(s) off it")
    assert missed == []


def test_a_grid_with_no_area_is_refused():
    with pytest.raises(OrthophotoError, match="positive size"):
        GroundGrid(origin=ORIGIN, cell_size_m=0.0, columns=10, rows=10)
    with pytest.raises(OrthophotoError, match="at least one cell"):
        GroundGrid(origin=ORIGIN, cell_size_m=1.0, columns=0, rows=10)


# ---------------------------------------------------------------- the sampling


def test_a_sampled_patch_is_populated_inside_the_footprint(reference_pose, grid):
    patch = sample_frame(reference_pose, flat((10, 200, 30)), grid, camera_id="cam-a")

    frame = _Frame(grid.origin)
    footprint = Polygon(
        [frame.to_xy(point) for point in field_of_view(reference_pose)]
    )
    well_inside = footprint.buffer(-1.5)
    xs, ys = grid.centres_xy()

    inside = 0
    holes = 0
    for row in range(grid.rows):
        for column in range(grid.columns):
            if well_inside.contains(Point(xs[row, column], ys[row, column])):
                inside += 1
                if not patch.valid[row, column]:
                    holes += 1

    print(
        f"{patch.covered_cells} of {grid.cell_count} cells sampled "
        f"({patch.covered_fraction:.1%}); {inside} cells well inside the "
        f"footprint, {holes} of them unfilled"
    )
    assert inside > 2_000
    assert holes == 0
    assert np.all(patch.colour[patch.valid] == (10, 200, 30))


def test_a_sampled_patch_is_empty_outside_the_footprint(reference_pose, grid):
    """No cell is coloured from ground the pose cannot reach.

    Checked two ways, because they fail differently: against the footprint
    polygon (a wrong inverse fills the wrong place) and against
    :func:`~sentinel.core.camera_sees` (a wrong range or horizon fills too far).
    """
    patch = sample_frame(reference_pose, flat((10, 200, 30)), grid, camera_id="cam-a")

    frame = _Frame(grid.origin)
    footprint = Polygon(
        [frame.to_xy(point) for point in field_of_view(reference_pose)]
    ).buffer(grid.cell_size_m)
    xs, ys = grid.centres_xy()

    rows, columns = np.nonzero(patch.valid)
    strays = sum(
        1
        for row, column in zip(rows, columns)
        if not footprint.contains(Point(xs[row, column], ys[row, column]))
    )
    unseen = sum(
        1
        for row, column in list(zip(rows, columns))[::17]
        if not camera_sees(reference_pose, grid.cell_centre(int(row), int(column)))
    )
    print(
        f"{rows.size} sampled cells; {strays} outside the footprint, "
        f"{unseen} the core says are not visible"
    )
    assert strays == 0
    assert unseen == 0

    # The blind ground under the camera's own mast, and the ground behind it.
    behind = grid.cell_of(destination_point(reference_pose.position, 0.0, 15.0))
    assert behind is not None, "the grid should extend behind the camera"
    assert not patch.valid[behind]
    assert np.all(patch.colour[behind] == 0)


def test_the_inverse_map_agrees_with_the_exact_inverse(reference_pose, grid):
    """The residual against ``image_coordinates``, measured and then floored.

    A patch built from a *different* inverse than the one positions come from
    would be a basemap offset from the tracks drawn on it — an error that looks
    like nothing at all until somebody argues about which side of a line
    somebody was standing.

    The frame encodes its own column index in sixteen bits across two channels,
    so what comes back out of the patch is the exact pixel the sampler chose
    rather than a quantised guess at it. The residual therefore includes the
    half-pixel of nearest-neighbour rounding, which is the honest thing to
    measure: it is the error a caller actually gets.
    """
    columns = np.rint(np.linspace(0.0, 65535.0, WIDTH)).astype(np.uint32)
    rows_16 = np.rint(np.linspace(0.0, 65535.0, HEIGHT)).astype(np.uint32)

    horizontal = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    horizontal[:, :, 0] = (columns >> 8).astype(np.uint8)[None, :]
    horizontal[:, :, 1] = (columns & 0xFF).astype(np.uint8)[None, :]

    vertical = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    vertical[:, :, 0] = (rows_16 >> 8).astype(np.uint8)[:, None]
    vertical[:, :, 1] = (rows_16 & 0xFF).astype(np.uint8)[:, None]

    across = sample_frame(reference_pose, horizontal, grid, camera_id="u")
    down = sample_frame(reference_pose, vertical, grid, camera_id="v")

    rows, cols = np.nonzero(across.valid)
    picked = np.random.default_rng(7).choice(rows.size, size=400, replace=False)

    du: list[float] = []
    dv: list[float] = []
    for index in picked:
        row, column = int(rows[index]), int(cols[index])
        exact = image_coordinates(reference_pose, grid.cell_centre(row, column))
        if exact is None or not exact.in_frame:
            continue
        u = (int(across.colour[row, column, 0]) * 256
             + int(across.colour[row, column, 1])) / 65535.0
        v = (int(down.colour[row, column, 0]) * 256
             + int(down.colour[row, column, 1])) / 65535.0
        du.append(abs(u - exact.u))
        dv.append(abs(v - exact.v))

    horizontal_px = np.array(du) * (WIDTH - 1)
    vertical_px = np.array(dv) * (HEIGHT - 1)
    print(
        f"{len(du)} cells checked against the exact inverse; "
        f"median {np.median(horizontal_px):.3f} px across / "
        f"{np.median(vertical_px):.3f} px down, "
        f"95th {np.percentile(horizontal_px, 95):.3f} / "
        f"{np.percentile(vertical_px, 95):.3f}, "
        f"worst {horizontal_px.max():.3f} / {vertical_px.max():.3f}"
    )
    assert len(du) > 300
    assert np.median(horizontal_px) < 1.0
    assert np.median(vertical_px) < 1.0
    assert np.percentile(horizontal_px, 95) < 3.0
    assert np.percentile(vertical_px, 95) < 3.0


def test_an_empty_cell_is_not_a_black_one(reference_pose, grid):
    """Asphalt at dusk is a black pixel, and it is still ground."""
    patch = sample_frame(reference_pose, flat((0, 0, 0)), grid, camera_id="cam-a")
    print(f"{patch.covered_cells} cells sampled from an all-black frame")
    assert patch.covered_cells > 2_000
    assert np.all(patch.colour == 0)
    assert patch.valid.any()
    assert not patch.valid.all()


def test_uncertainty_travels_with_every_sampled_cell(reference_pose, grid):
    patch = sample_frame(reference_pose, flat((90, 90, 90)), grid, camera_id="cam-a")

    assert np.all(np.isfinite(patch.sigma_m[patch.valid]))
    assert np.all(np.isnan(patch.sigma_m[~patch.valid]))

    frame = _Frame(grid.origin)
    camera_x, camera_y = frame.to_xy(reference_pose.position)
    xs, ys = grid.centres_xy()
    distance = np.hypot(xs - camera_x, ys - camera_y)

    near = patch.valid & (distance < 15.0)
    far = patch.valid & (distance > 60.0)
    print(
        f"1σ near (<15 m): {patch.sigma_m[near].mean():.2f} m, "
        f"far (>60 m): {patch.sigma_m[far].mean():.2f} m"
    )
    assert near.any() and far.any()
    assert patch.sigma_m[far].mean() > patch.sigma_m[near].mean() * 3.0


def test_a_camera_pointed_at_the_sky_produces_an_entirely_empty_patch(grid):
    """Not an exception and not a guess: it sees no ground, so it maps none."""
    patch = sample_frame(
        camera(ORIGIN, 180.0, pitch=10.0), flat((200, 0, 0)), grid, camera_id="sky"
    )
    assert patch.covered_cells == 0
    assert patch.covered_fraction == 0.0
    assert np.all(np.isnan(patch.updated_at))


def test_freshness_comes_back_with_the_patch(reference_pose, grid):
    patch = sample_frame(
        reference_pose, flat((5, 5, 5)), grid, camera_id="cam-a", captured_at=1_000.0
    )
    ages = patch.age_seconds(now=1_030.0)
    print(f"age of a sampled cell {np.nanmax(ages):.1f} s; unseen cells are NaN")
    assert np.all(ages[patch.valid] == pytest.approx(30.0))
    assert np.all(np.isnan(ages[~patch.valid]))


def test_sampling_a_real_grid_is_fast_enough_to_feed_a_median(reference_pose):
    """A per-cell loop across the FFI is what this timing exists to rule out."""
    fine = GroundGrid.covering(
        field_of_view(reference_pose), cell_size_m=0.5, margin_m=2.0
    )
    image = flat((30, 60, 90))
    sample_frame(reference_pose, image, fine, camera_id="warm")

    started = time.perf_counter()
    sample_frame(reference_pose, image, fine, camera_id="cam-a")
    elapsed = time.perf_counter() - started
    print(f"{fine.cell_count:,} cells sampled in {elapsed * 1000:.0f} ms")
    assert elapsed < 1.0


# ------------------------------------------------------------------ the median


def test_the_median_removes_an_object_that_appears_in_a_minority_of_frames(
    reference_pose, grid
):
    """The whole reason the accumulator exists.

    A bright patch is painted over part of the frame in four of eleven passes.
    Fewer than half, so it cannot move any cell's median — and what is left is
    the site as it was when nobody was on it.
    """
    background = (40, 80, 120)
    accumulator = MedianAccumulator(grid, capacity=11)
    for index in range(11):
        image = flat(background)
        if index % 3 == 0:  # four of eleven
            image[200:400, 150:450] = 255
        accumulator.update(
            sample_frame(reference_pose, image, grid, camera_id="cam-a",
                         captured_at=1_000.0 + index)
        )

    result = accumulator.result(now=1_100.0)
    colours = np.unique(result.colour[result.valid].reshape(-1, 3), axis=0)
    print(
        f"{accumulator.frames} frames, {result.covered_cells} cells; "
        f"distinct colours surviving: {colours.tolist()}"
    )
    assert result.covered_cells > 2_000
    assert colours.tolist() == [list(background)]
    assert np.all(result.samples[result.valid] == 11)


def test_the_median_keeps_something_that_was_there_for_most_of_the_window(
    reference_pose, grid
):
    """The honest converse: a majority is the ground, as far as this can tell.

    Something parked in a cell for more than half the observation window is
    indistinguishable from ground by any median, and pretending otherwise would
    be the more dangerous claim. So it is asserted rather than papered over.
    """
    accumulator = MedianAccumulator(grid, capacity=9)
    for index in range(9):
        image = flat((40, 80, 120))
        if index >= 3:  # six of nine
            image[:, :] = (200, 210, 220)
        accumulator.update(
            sample_frame(reference_pose, image, grid, camera_id="cam-a",
                         captured_at=2_000.0 + index)
        )

    result = accumulator.result(now=2_100.0)
    colours = np.unique(result.colour[result.valid].reshape(-1, 3), axis=0)
    print(f"colours surviving a six-of-nine majority: {colours.tolist()}")
    assert colours.tolist() == [[200, 210, 220]]


def test_a_cell_seen_too_few_times_is_left_empty(reference_pose, grid):
    """Two samples cannot outvote a passer-by, so two samples are not a map."""
    accumulator = MedianAccumulator(grid, minimum_samples=MINIMUM_SAMPLES)
    for index in range(MINIMUM_SAMPLES - 1):
        accumulator.update(
            sample_frame(reference_pose, flat((7, 7, 7)), grid, camera_id="cam-a",
                         captured_at=float(index))
        )
    thin = accumulator.result(now=100.0)
    print(f"after {accumulator.frames} frames: {thin.covered_cells} cells reported")
    assert thin.covered_cells == 0

    accumulator.update(
        sample_frame(reference_pose, flat((7, 7, 7)), grid, camera_id="cam-a",
                     captured_at=99.0)
    )
    enough = accumulator.result(now=100.0)
    print(f"after {accumulator.frames} frames: {enough.covered_cells} cells reported")
    assert enough.covered_cells > 2_000


def test_the_accumulator_memory_is_bounded_by_the_ring_not_the_frame_count(
    reference_pose, grid
):
    """A thousand frames must cost exactly what fifteen do."""
    accumulator = MedianAccumulator(grid, capacity=DEFAULT_CAPACITY)
    image = flat((11, 22, 33))

    accumulator.update(
        sample_frame(reference_pose, image, grid, camera_id="cam-a", captured_at=0.0)
    )
    after_one = accumulator.memory_bytes
    for index in range(1, 60):
        accumulator.update(
            sample_frame(reference_pose, image, grid, camera_id="cam-a",
                         captured_at=float(index))
        )
    after_sixty = accumulator.memory_bytes

    print(
        f"{accumulator.bytes_per_cell} bytes per cell × {grid.cell_count:,} cells = "
        f"{accumulator.memory_bytes / 1e6:.2f} MB after 1 frame and after "
        f"{accumulator.frames} frames"
    )
    assert after_sixty == after_one
    assert accumulator.bytes_per_cell * grid.cell_count == accumulator.memory_bytes
    assert accumulator.bytes_per_cell < 200
    # A cell only ever holds its last `capacity` samples, so the count reported
    # beside the colour saturates there rather than growing with the footage.
    result = accumulator.result(now=100.0)
    assert result.samples.max() == DEFAULT_CAPACITY


def test_retention_drops_samples_older_than_the_window(reference_pose, grid):
    """Ground the camera saw an hour ago is not evidence about ground now."""
    accumulator = MedianAccumulator(grid, capacity=9, retain_seconds=30.0)
    for index in range(6):
        accumulator.update(
            sample_frame(reference_pose, flat((10, 10, 10)), grid, camera_id="cam-a",
                         captured_at=1_000.0 + index)
        )

    fresh = accumulator.result(now=1_010.0)
    stale = accumulator.result(now=1_200.0)
    print(
        f"{fresh.covered_cells} cells within the window, "
        f"{stale.covered_cells} once every sample is older than it"
    )
    assert fresh.covered_cells > 2_000
    assert stale.covered_cells == 0


def test_the_median_reports_the_newest_sample_behind_each_cell(reference_pose, grid):
    accumulator = MedianAccumulator(grid, capacity=5)
    for index in range(5):
        accumulator.update(
            sample_frame(reference_pose, flat((60, 60, 60)), grid, camera_id="cam-a",
                         captured_at=500.0 + index * 10.0)
        )
    result = accumulator.result(now=600.0)
    newest = np.nanmax(result.updated_at)
    print(f"newest contributing sample at t={newest:.1f}, oldest at t=500.0")
    assert newest == pytest.approx(540.0, abs=0.01)
    assert np.all(result.age_seconds(now=600.0)[result.valid]
                  == pytest.approx(60.0, abs=0.01))


def test_a_patch_from_another_grid_is_refused(reference_pose, grid):
    """Two rasters that do not share an origin silently shift the map."""
    other = GroundGrid(
        origin=destination_point(grid.origin, 90.0, 10.0),
        cell_size_m=grid.cell_size_m, columns=grid.columns, rows=grid.rows,
    )
    accumulator = MedianAccumulator(grid)
    with pytest.raises(OrthophotoError, match="different grid"):
        accumulator.update(
            sample_frame(reference_pose, flat((1, 2, 3)), other, camera_id="cam-a")
        )


# ------------------------------------------------------------------ the mosaic


def test_the_mosaic_prefers_the_lower_sigma_camera_where_two_overlap(grid):
    """Same cells, same colours, different pose error: the surveyed mast wins."""
    everywhere = np.ones(grid.shape, dtype=bool)
    sharp = synthetic_patch(grid, "cam-sharp", colour=(0, 255, 0),
                            valid=everywhere, sigma_m=0.4, when=100.0)
    vague = synthetic_patch(grid, "cam-vague", colour=(255, 0, 0),
                            valid=everywhere, sigma_m=0.4, when=200.0)

    result = mosaic([sharp, vague], {"cam-sharp": 0.3, "cam-vague": 4.0})
    won = result.cells_from("cam-sharp")
    print(f"cam-sharp won {won} of {grid.cell_count} contested cells")
    assert won == grid.cell_count
    assert np.all(result.colour[result.valid] == (0, 255, 0))
    assert np.all(result.updated_at[result.valid] == 100.0)
    # The reported error is the two contributions in quadrature, not either one.
    assert result.sigma_m[0, 0] == pytest.approx(math.hypot(0.4, 0.3), rel=1e-5)


def test_the_mosaic_takes_each_cell_from_whichever_camera_knows_it_best(grid):
    """Per cell, not per camera — which is the point of the whole exercise.

    Two identical masts at opposite ends of an overlap: neither is the better
    camera, and each should win the half of the overlap nearer to itself.
    """
    north = np.zeros(grid.shape, dtype=bool)
    north[: grid.rows // 2 + 10, :] = True
    south = np.zeros(grid.shape, dtype=bool)
    south[grid.rows // 2 - 10:, :] = True

    # Error rising away from each camera's own end of the site.
    rows = np.arange(grid.rows, dtype=np.float32)[:, None]
    north_sigma = np.broadcast_to(0.5 + rows * 0.1, grid.shape).copy()
    south_sigma = np.broadcast_to(0.5 + (grid.rows - 1 - rows) * 0.1, grid.shape).copy()

    def patch(camera_id, colour, valid, sigma, when):
        pixels = np.zeros((*grid.shape, 3), dtype=np.uint8)
        pixels[valid] = colour
        return GroundPatch(
            camera_id=camera_id, grid=grid, colour=pixels, valid=valid,
            sigma_m=np.where(valid, sigma, np.nan).astype(np.float32),
            updated_at=np.where(valid, when, np.nan),
            samples=valid.astype(np.uint32),
        )

    result = mosaic(
        [patch("cam-n", (0, 0, 255), north, north_sigma, 10.0),
         patch("cam-s", (255, 0, 0), south, south_sigma, 20.0)],
        {"cam-n": 1.0, "cam-s": 1.0},
    )

    overlap = north & south
    expected = np.where(north_sigma < south_sigma,
                        result.cameras.index("cam-n"),
                        result.cameras.index("cam-s"))
    wrong = int(np.count_nonzero(result.source[overlap] != expected[overlap]))
    print(
        f"{int(overlap.sum())} contested cells, split "
        f"{result.cells_from('cam-n')} / {result.cells_from('cam-s')}; "
        f"{wrong} decided against the smaller error"
    )
    assert overlap.sum() > 0
    assert wrong == 0
    assert result.cells_from("cam-n") > 0
    assert result.cells_from("cam-s") > 0


def test_cells_no_camera_covers_stay_empty(grid):
    """Never interpolated. Inventing ground nobody has seen is the one rule."""
    left = np.zeros(grid.shape, dtype=bool)
    left[:, : grid.columns // 3] = True
    right = np.zeros(grid.shape, dtype=bool)
    right[:, -grid.columns // 3:] = True

    result = mosaic(
        [synthetic_patch(grid, "cam-l", colour=(0, 200, 0), valid=left,
                         sigma_m=1.0, when=5.0),
         synthetic_patch(grid, "cam-r", colour=(0, 0, 200), valid=right,
                         sigma_m=1.0, when=6.0)],
        {"cam-l": 1.0, "cam-r": 1.0},
    )

    nobody = ~(left | right)
    print(
        f"{int(nobody.sum())} cells between the two footprints; "
        f"{result.covered_fraction:.0%} of the raster is mapped"
    )
    assert nobody.sum() > 0
    assert not result.valid[nobody].any()
    assert np.all(result.colour[nobody] == 0)
    assert np.all(result.source[nobody] == -1)
    assert np.all(np.isnan(result.updated_at[nobody]))
    assert np.all(np.isnan(result.sigma_m[nobody]))
    assert "never interpolated" in result.describe()


def test_a_camera_with_no_stated_pose_error_is_refused(grid):
    """Defaulting it to zero would let an unsurveyed mast win every cell."""
    everywhere = np.ones(grid.shape, dtype=bool)
    patch = synthetic_patch(grid, "cam-a", colour=(1, 1, 1), valid=everywhere,
                            sigma_m=1.0, when=0.0)
    with pytest.raises(OrthophotoError, match="no pose error"):
        mosaic([patch], {})
    with pytest.raises(OrthophotoError, match="non-negative"):
        mosaic([patch], {"cam-a": -1.0})


def test_a_mosaic_of_patches_on_different_grids_is_refused(grid):
    everywhere = np.ones(grid.shape, dtype=bool)
    shifted = GroundGrid(
        origin=grid.origin, cell_size_m=grid.cell_size_m * 2.0,
        columns=grid.columns, rows=grid.rows,
    )
    with pytest.raises(OrthophotoError, match="different grid"):
        mosaic(
            [synthetic_patch(grid, "cam-a", colour=(1, 1, 1), valid=everywhere,
                             sigma_m=1.0, when=0.0),
             synthetic_patch(shifted, "cam-b", colour=(2, 2, 2), valid=everywhere,
                             sigma_m=1.0, when=0.0)],
            {"cam-a": 1.0, "cam-b": 1.0},
        )


def test_an_empty_mosaic_is_refused():
    with pytest.raises(OrthophotoError, match="at least one patch"):
        mosaic([], {})


def test_two_real_cameras_facing_each_other_compose_into_one_basemap(reference_pose):
    """End to end, with the real geometry, from two poses to one raster.

    The claim under test is that the pieces fit: two patches sampled through
    real projections onto one shared grid, composited, cover more ground than
    either alone and leave the rest honestly empty.
    """
    north = reference_pose
    south = camera(destination_point(reference_pose.position, 180.0, 60.0), 0.0)
    shared = GroundGrid.covering(
        field_of_view(north) + field_of_view(south), cell_size_m=1.0, margin_m=3.0
    )

    patches = [
        sample_frame(north, flat((0, 0, 255)), shared, camera_id="cam-north",
                     captured_at=100.0),
        sample_frame(south, flat((255, 0, 0)), shared, camera_id="cam-south",
                     captured_at=200.0),
    ]
    result = mosaic(patches, {"cam-north": 0.2, "cam-south": 0.2})

    both = patches[0].valid & patches[1].valid
    print(
        f"{shared.rows}×{shared.columns} raster: "
        f"north {patches[0].covered_cells}, south {patches[1].covered_cells}, "
        f"{int(both.sum())} seen by both, {result.covered_cells} mapped, "
        f"{shared.cell_count - result.covered_cells} left empty"
    )
    assert both.sum() > 100
    assert result.covered_cells > max(p.covered_cells for p in patches)
    assert result.covered_cells < shared.cell_count
    assert np.all(result.valid == (patches[0].valid | patches[1].valid))
    assert result.cells_from("cam-north") > 0
    assert result.cells_from("cam-south") > 0
