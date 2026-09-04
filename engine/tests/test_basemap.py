"""Tests for the basemap builder, its files, and its fingerprint.

`test_orthophoto` proves the pieces: the inverse map, the median, the mosaic.
This proves the product path that finally calls them, and three claims carry
the weight:

- **What is fed is what is mapped, and nothing else.** Twenty frames of flat
  ground from the reference pose produce ground exactly inside that pose's
  footprint, in that colour, with a finite age and error on every mapped cell
  and ``inf`` on every empty one — and a bright square crossing a minority of
  those frames leaves no trace.
- **A second camera enlarges the union without touching the first.** Each
  camera samples onto a grid of its own on one shared cell lattice; the build
  lays each median onto the union by an integer shift, and the laid patch is
  held to the original cell for cell. A wrong offset, a lattice that is not
  whole cells, or a rebuild that quietly dropped the first camera's minutes,
  fails here — and so does what the first shape of this module did: grow the
  grid on the *second frame of the same camera*, because a footprint edge
  re-measured in another frame sat half a millimetre past a lattice line.
- **The files are the asset, or they are refused.** Save and load round-trip
  every array and every pose exactly; a PNG edited afterwards, a JSON with one
  number changed, and the half-written pair a crash leaves are all refused
  with the same error — because the plan view must never draw ground that is
  not the ground somebody's judgement was made against.

No model, no camera, no network. The frames are synthetic; the geometry is the
real one.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from sentinel import logs, paths
import sentinel.basemap as basemap_module
from sentinel.basemap import (
    DEFAULT_POSE_SIGMA_M,
    FORMAT,
    JSON_NAME,
    PNG_NAME,
    BasemapAsset,
    BasemapBuilder,
    BasemapError,
    _placed,
    basemap_directory,
    load_basemap,
    save_basemap,
)
from sentinel.core import CameraPose, LatLon, destination_point, field_of_view
from sentinel.coverage import _Frame
from sentinel.orthophoto import (
    DEFAULT_CAPACITY,
    MINIMUM_SAMPLES,
    GroundGrid,
    mosaic,
    sample_frame,
)

WIDTH = 640
HEIGHT = 480
ORIGIN = LatLon(33.8938, 35.5018)
GROUND = (40, 80, 120)


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


def flat(colour: tuple[int, int, int]) -> np.ndarray:
    """A frame of one colour, so a mapped cell's provenance is unambiguous."""
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:, :] = colour
    return frame


def feed(
    builder: BasemapBuilder,
    camera_id: str,
    pose: CameraPose,
    frames: int,
    *,
    colour: tuple[int, int, int] = GROUND,
    start: float = 1_000.0,
    paint=None,
) -> None:
    """Feed ``frames`` flat frames a second apart, ``paint(index, image)`` on each."""
    for index in range(frames):
        image = flat(colour)
        if paint is not None:
            paint(index, image)
        builder.feed(camera_id, pose, image, start + index)


def one_camera_asset(pose: CameraPose, *, frames: int = 20) -> BasemapAsset:
    builder = BasemapBuilder(cell_size_m=1.0)
    feed(builder, "cam-a", pose, frames)
    return builder.build(now=1_100.0)


def facing_pair(reference_pose: CameraPose) -> tuple[CameraPose, CameraPose]:
    """The reference camera and one 60 m south of it looking back north."""
    return reference_pose, camera(
        destination_point(reference_pose.position, 180.0, 60.0), 0.0
    )


# ---------------------------------------------------------------- the builder


def test_twenty_flat_frames_from_the_reference_pose_map_the_footprint_and_nothing_else(
    reference_pose,
):
    builder = BasemapBuilder(cell_size_m=1.0)
    feed(builder, "cam-a", reference_pose, 20)
    asset = builder.build(now=1_100.0)

    frame = _Frame(asset.grid.origin)
    footprint = Polygon([frame.to_xy(point) for point in field_of_view(reference_pose)])
    well_inside = footprint.buffer(-1.5)
    generous = footprint.buffer(asset.grid.cell_size_m)
    xs, ys = asset.grid.centres_xy()

    inside = holes = strays = 0
    for row in range(asset.grid.rows):
        for column in range(asset.grid.columns):
            centre = Point(xs[row, column], ys[row, column])
            if well_inside.contains(centre):
                inside += 1
                if not asset.valid[row, column]:
                    holes += 1
            if asset.valid[row, column] and not generous.contains(centre):
                strays += 1

    print(
        f"{asset.grid.rows}x{asset.grid.columns} grid, {asset.covered_cells} cells "
        f"mapped ({asset.covered_fraction:.0%}); {inside} well inside the "
        f"footprint, {holes} of them empty, {strays} mapped outside it"
    )
    assert inside > 2_000
    assert holes == 0
    assert strays == 0
    assert np.all(asset.colour[asset.valid] == GROUND)
    assert np.all(asset.colour[~asset.valid] == 0)
    assert asset.cameras == ("cam-a",)
    assert asset.poses == {"cam-a": reference_pose}
    assert np.all(asset.source[asset.valid] == 0)
    assert np.all(asset.source[~asset.valid] == -1)
    assert asset.frames_used == 20
    assert builder.frames_fed() == {"cam-a": 20}
    assert builder.frames_sampled() == {"cam-a": 20}
    assert builder.blind_cameras() == ()


def test_age_and_error_are_finite_where_mapped_and_infinite_everywhere_else(reference_pose):
    """A renderer that fades by age must read "absent", never "very old"."""
    asset = one_camera_asset(reference_pose)

    ages = asset.age_seconds[asset.valid]
    print(
        f"mapped cells are {ages.min():.1f}-{ages.max():.1f} s old; "
        f"error {asset.sigma_m[asset.valid].min():.2f}-{asset.sigma_m[asset.valid].max():.2f} m"
    )
    # Newest sample at t=1019, built at t=1100: a fixed camera sees every one
    # of its cells in every frame, so every mapped cell is exactly that old.
    assert np.all(ages == pytest.approx(81.0, abs=0.01))
    assert np.all(np.isinf(asset.age_seconds[~asset.valid]))
    assert np.all(np.isfinite(asset.sigma_m[asset.valid]))
    assert np.all(np.isinf(asset.sigma_m[~asset.valid]))
    # The pose error is in quadrature with the projection error, so no cell
    # can be known better than the mast it was seen from.
    assert np.all(asset.sigma_m[asset.valid] >= np.float32(DEFAULT_POSE_SIGMA_M))
    assert asset.built_at_millis == 1_100_000


def test_a_bright_square_crossing_in_a_minority_of_frames_leaves_no_trace(reference_pose):
    """The whole reason the median sits between the camera and the map.

    Seven of twenty frames carry a white square, at a different place each
    time and including the very last frame — so a builder that kept the latest
    frame, or averaged, would print it into the ground. The square is shown to
    land on real cells first, so the absence afterwards is the median's doing
    and not the square missing the footprint.
    """
    def crossing(index: int, image: np.ndarray) -> None:
        if index % 3 == 1:  # 1, 4, ..., 19: seven of twenty
            top, left = 120 + 6 * index, 100 + 10 * index
            image[top:top + 200, left:left + 300] = 255

    builder = BasemapBuilder(cell_size_m=1.0)
    feed(builder, "cam-a", reference_pose, 20, paint=crossing)
    asset = builder.build(now=1_100.0)

    def struck_by(index: int) -> int:
        painted = flat(GROUND)
        crossing(index, painted)
        alone = sample_frame(reference_pose, painted, asset.grid, camera_id="cam-a")
        return int(np.count_nonzero(np.all(alone.colour == 255, axis=2) & alone.valid))

    # Far ground is many cells per pixel and near ground the reverse, so the
    # first pass strikes over a hundred cells and the last, lower in the
    # frame, a few dozen. Both land; neither survives.
    first, last = struck_by(1), struck_by(19)
    colours = np.unique(asset.colour[asset.valid].reshape(-1, 3), axis=0)
    print(
        f"the square covers {first} cells in its first frame alone and {last} in "
        f"its last; colours surviving the median: {colours.tolist()}"
    )
    assert first > 100
    assert last > 20
    assert colours.tolist() == [list(GROUND)]


def test_the_grid_grows_for_a_second_camera_and_the_first_keeps_every_cell(reference_pose):
    """The union is an index shift of each camera's own grid, and this holds it to that.

    The first camera is built alone, then a camera whose footprint lies
    outside its grid is fed and the pair is built. Every cell the first asset
    mapped must be mapped in the second at the shifted index, with the same
    colour, age and error wherever the first camera still owns it. A rebuild
    that dropped the first camera's samples fails here with no north cells at
    all; an offset wrong by one cell fails at the footprint's edge. And the
    first camera's ring is the same object on the same grid afterwards —
    nothing was moved or copied to make room — so the two rings together cost
    their own cells, not twice the union.
    """
    north, south = facing_pair(reference_pose)
    builder = BasemapBuilder(cell_size_m=1.0)
    feed(builder, "cam-n", north, 5, colour=(0, 0, 255))
    first = builder.build(now=1_100.0)
    small = first.grid
    ring_n = builder.grid_of("cam-n")
    assert ring_n is small, "one camera: its own grid is the union"

    feed(builder, "cam-s", south, 5, colour=(255, 0, 0))
    second = builder.build(now=1_100.0)
    large = second.grid
    ring_s = builder.grid_of("cam-s")
    assert builder.grid_of("cam-n") is ring_n, "the first camera's grid was replaced"
    assert ring_s is not None and ring_s != large and ring_n != large
    assert builder.memory_bytes == 113 * (ring_n.cell_count + ring_s.cell_count)
    assert builder.memory_bytes < 2 * 113 * large.cell_count

    dx, dy = _Frame(large.origin).to_xy(small.origin)
    column_offset = dx / large.cell_size_m
    row_offset = -dy / large.cell_size_m
    misregistration = max(
        abs(column_offset - round(column_offset)), abs(row_offset - round(row_offset))
    )
    print(
        f"{small.rows}x{small.columns} grew to {large.rows}x{large.columns}; the old "
        f"origin sits {row_offset:.4f} rows down and {column_offset:.4f} columns "
        f"across the new one ({misregistration * large.cell_size_m * 1000:.2f} mm off a "
        "whole cell)"
    )
    assert large.cell_size_m == small.cell_size_m
    assert large.rows > small.rows
    assert misregistration < 0.01
    rows = slice(round(row_offset), round(row_offset) + small.rows)
    columns = slice(round(column_offset), round(column_offset) + small.columns)

    window_valid = second.valid[rows, columns]
    assert np.all(window_valid[first.valid]), "a cell the first camera mapped went missing"

    index_n = second.cameras.index("cam-n")
    still_north = second.source[rows, columns] == index_n
    assert np.all(first.valid[still_north]), "the north camera won a cell it never saw"
    assert np.array_equal(second.colour[rows, columns][still_north], first.colour[still_north])
    assert np.array_equal(
        second.age_seconds[rows, columns][still_north], first.age_seconds[still_north]
    )
    assert np.array_equal(second.sigma_m[rows, columns][still_north], first.sigma_m[still_north])

    assert second.cells_from("cam-n") > 0 and second.cells_from("cam-s") > 0
    assert second.cells_from("cam-n") + second.cells_from("cam-s") == second.covered_cells
    assert second.covered_cells > first.covered_cells
    assert second.frames_used == 10


def test_a_patch_laid_on_the_union_grid_is_the_same_patch_at_the_shifted_index(reference_pose):
    """The build lays each median onto the union through the patch's public fields.

    After the shift, the colour, the mask, the count, the error and the newest
    sample of every cell are identical at the shifted index; everything
    outside the window is what an empty cell is — not ground, zero colour,
    NaN error and time, zero samples — so the mosaic underneath reads "no
    ground" there rather than "black ground", and accepts the laid patch as
    it would the original. A window that does not fit is refused, not
    clipped.
    """
    builder = BasemapBuilder(cell_size_m=1.0)
    feed(builder, "cam-a", reference_pose, 1, colour=(10, 20, 30))
    small = builder.grid_of("cam-a")
    assert small is not None
    before = sample_frame(
        reference_pose, flat((10, 20, 30)), small, camera_id="cam-a", captured_at=500.0
    )

    large = GroundGrid(
        origin=_Frame(small.origin).to_latlon(-2.0, 3.0),
        cell_size_m=1.0, columns=small.columns + 5, rows=small.rows + 7,
    )
    after = _placed(before, large, row_offset=3, column_offset=2)

    window = (slice(3, 3 + small.rows), slice(2, 2 + small.columns))
    outside = np.ones(large.shape, dtype=bool)
    outside[window] = False
    print(
        f"{before.covered_cells} cells before the shift, {after.covered_cells} after; "
        f"{int(after.valid[outside].sum())} outside the window"
    )
    assert after.grid == large and after.camera_id == "cam-a"
    assert after.covered_cells == before.covered_cells > 2_000
    assert np.array_equal(after.valid[window], before.valid)
    assert np.array_equal(after.colour[window], before.colour)
    assert np.array_equal(after.samples[window], before.samples)
    assert np.array_equal(after.sigma_m[window], before.sigma_m, equal_nan=True)
    assert np.array_equal(after.updated_at[window], before.updated_at, equal_nan=True)
    assert not after.valid[outside].any()
    assert not after.colour[outside].any() and not after.samples[outside].any()
    assert np.all(np.isnan(after.sigma_m[outside])) and np.all(np.isnan(after.updated_at[outside]))
    assert mosaic([after], {"cam-a": 1.0}).covered_cells == before.covered_cells

    # Already on the grid: handed back as it is, nothing copied.
    assert _placed(before, small, row_offset=0, column_offset=0) is before
    with pytest.raises(BasemapError, match="does not fit"):
        _placed(before, large, row_offset=8, column_offset=2)


def test_feeding_a_placed_camera_again_never_touches_the_grid(reference_pose):
    """A camera's grid is placed once; its second frame costs what its first did.

    The first shape of this module grew the grid on the *second* frame of the
    same camera — 82×91 to 83×91 at a metre, 327×362 to 328×362 at the CLI's
    quarter metre — copying the whole ring to do it, because the footprint
    was measured again in the grid's own frame and found half a millimetre
    outside. Now the footprint is measured once: the grid is the same object
    after forty frames as after one, the memory is the same number, and the
    footprint is computed once per camera and not once per frame.
    """
    calls: list[str] = []
    real = basemap_module.field_of_view

    def counting(pose, *args, **kwargs):
        calls.append("footprint")
        return real(pose, *args, **kwargs)

    for cell, frames in ((1.0, 40), (0.25, 3)):
        calls.clear()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(basemap_module, "field_of_view", counting)
            builder = BasemapBuilder(cell_size_m=cell)
            feed(builder, "cam-a", reference_pose, 1)
            first = builder.grid
            after_one = builder.memory_bytes
            feed(builder, "cam-a", reference_pose, frames - 1, start=1_001.0)
        print(
            f"{cell} m: {first.rows}x{first.columns} after one frame, "
            f"{builder.grid.rows}x{builder.grid.columns} after {frames}; "
            f"{after_one / 1e6:.2f} MB then {builder.memory_bytes / 1e6:.2f} MB; "
            f"footprint computed {len(calls)} time(s)"
        )
        assert builder.grid is first
        assert builder.grid_of("cam-a") is first
        assert builder.memory_bytes == after_one
        assert builder.frames_fed() == {"cam-a": frames}
        assert calls == ["footprint"]


@pytest.mark.parametrize(
    "heading, cell",
    [(180.0, 1.0), (45.0, 1.0), (90.0, 1.0), (135.0, 1.0), (180.0, 0.25)],
)
def test_two_cameras_on_one_mast_are_given_one_grid_exactly(heading: float, cell: float):
    """The same footprint, measured from the lattice origin, lands on the same cells.

    The second camera is measured in the first grid's frame, sixty-odd metres
    from where the first was measured, and the two frames disagree by the
    convergence of meridians — half a millimetre over this footprint. With
    the lattice anchored at a footprint point, or at the corner
    :meth:`GroundGrid.covering` chooses, a footprint edge plus a whole-cell
    margin sat *exactly* on a lattice line and that half millimetre decided
    which side of it the second camera fell: these four headings each got a
    grid one cell wider or taller. Anchored at the camera, the extremes sit
    mid-cell and the two grids are equal.
    """
    pose = camera(ORIGIN, heading)
    builder = BasemapBuilder(cell_size_m=cell)
    feed(builder, "cam-a", pose, 1)
    feed(builder, "cam-b", pose, 1)
    first, second = builder.grid_of("cam-a"), builder.grid_of("cam-b")
    print(f"heading {heading:g} at {cell} m: {first.shape} and {second.shape}")
    assert second == first
    assert builder.grid is first, "two equal grids: the union is still the first"


def test_a_camera_grid_laid_on_the_union_is_where_its_cells_are_to_a_fraction_of_a_millimetre(
    reference_pose,
):
    """The lattice's one approximation, measured rather than assumed.

    Each camera's grid is a local frame at its own origin and the union is
    another; the index shift claims that cell ``(r, c)`` of a camera's grid
    *is* cell ``(r + dr, c + dc)`` of the union. Two such frames disagree by
    the convergence of meridians, ``D·d·tan(lat)/R`` for origins ``D`` apart
    east-west and a cell ``d`` away, so the pair here is east-west — a
    north-south pair measures nothing — and every grid corner is checked
    against where the union says it is. The offsets must be whole cells to a
    millionth, and the cell centres agree to under a millimetre.
    """
    west = reference_pose
    east = camera(destination_point(reference_pose.position, 90.0, 60.0), 270.0)
    builder = BasemapBuilder(cell_size_m=1.0)
    feed(builder, "cam-w", west, 1)
    feed(builder, "cam-e", east, 1)
    union = builder.grid
    frame = _Frame(union.origin)

    worst = 0.0
    for camera_id in ("cam-w", "cam-e"):
        grid = builder.grid_of(camera_id)
        ox, oy = frame.to_xy(grid.origin)
        column0, row0 = round(ox / union.cell_size_m), round(-oy / union.cell_size_m)
        assert abs(ox / union.cell_size_m - column0) < 1e-6
        assert abs(-oy / union.cell_size_m - row0) < 1e-6
        assert row0 >= 0 and column0 >= 0
        assert row0 + grid.rows <= union.rows and column0 + grid.columns <= union.columns
        for row, column in (
            (0, 0), (0, grid.columns - 1), (grid.rows - 1, 0),
            (grid.rows - 1, grid.columns - 1), (grid.rows // 2, grid.columns // 2),
        ):
            x, y = frame.to_xy(grid.cell_centre(row, column))
            expected_x, expected_y = union.cell_xy(row0 + row, column0 + column)
            worst = max(worst, math.hypot(x - expected_x, y - expected_y))
    print(
        f"union {union.rows}x{union.columns}; worst cell-centre disagreement between "
        f"a camera's frame and the union's: {worst * 1000:.3f} mm"
    )
    assert worst < 1e-3


def test_a_frame_that_cannot_be_sampled_leaves_nothing_behind(reference_pose):
    """Refused whole: not counted, no pose recorded, no grid placed for it.

    An image with no pixels is one the sampler refuses. Before this, the
    frame was counted and the pose recorded *before* sampling, so a refused
    frame was reported as fed and its pose bound the camera to a placement
    that never produced a sample.
    """
    builder = BasemapBuilder(cell_size_m=1.0)
    with pytest.raises(BasemapError, match="no pixels"):
        builder.feed("cam-a", reference_pose, np.zeros((0, 0, 3), np.uint8), 1_000.0)
    assert builder.frames_fed() == {}
    assert builder.frames_sampled() == {}
    assert builder.grid is None and builder.grid_of("cam-a") is None
    assert builder.memory_bytes == 0

    # The pose was not recorded: the camera may be fed under another one.
    elsewhere = camera(destination_point(reference_pose.position, 90.0, 5.0), 180.0)
    feed(builder, "cam-a", elsewhere, 1)
    assert builder.frames_fed() == {"cam-a": 1}
    placed = builder.grid

    # And a placed camera's refused frame changes nothing either.
    with pytest.raises(BasemapError, match="no pixels"):
        builder.feed("cam-a", elsewhere, np.zeros((0, 0, 3), np.uint8), 1_001.0)
    assert builder.frames_fed() == {"cam-a": 1}
    assert builder.frames_sampled() == {"cam-a": 1}
    assert builder.grid is placed


def test_cells_from_refuses_a_camera_the_asset_was_not_built_from(reference_pose):
    """A mistyped id must not read as "saw no ground"."""
    asset = one_camera_asset(reference_pose)
    assert asset.cells_from("cam-a") == asset.covered_cells
    with pytest.raises(BasemapError, match="not one this basemap was built from") as refusal:
        asset.cells_from("cam-b")
    assert "cam-a" in str(refusal.value)


def test_a_cell_two_cameras_cover_comes_from_the_one_that_knows_it_better(reference_pose):
    """Per cell, by the smaller position error there — the mosaic's rule, reached.

    Each camera's per-cell error is re-derived on the asset's own grid and
    combined with the pose error the way the mosaic does; the asset's choice
    must match it on every contested cell. Cells within a millimetre of a tie
    are not judged, and the contested region is eroded by one cell so a
    boundary cell that one camera's footprint reaches on one grid origin and
    not the other cannot turn a real check into a flaky one.
    """
    north, south = facing_pair(reference_pose)
    builder = BasemapBuilder(cell_size_m=1.0)
    feed(builder, "cam-n", north, 4, colour=(0, 0, 255))
    feed(builder, "cam-s", south, 4, colour=(255, 0, 0))
    asset = builder.build(now=1_100.0)

    grid = asset.grid
    patch_n = sample_frame(north, flat((0, 0, 255)), grid, camera_id="cam-n")
    patch_s = sample_frame(south, flat((255, 0, 0)), grid, camera_id="cam-s")
    effective_n = np.hypot(patch_n.sigma_m.astype(np.float64), DEFAULT_POSE_SIGMA_M)
    effective_s = np.hypot(patch_s.sigma_m.astype(np.float64), DEFAULT_POSE_SIGMA_M)

    contested = cv2.erode(
        (patch_n.valid & patch_s.valid).astype(np.uint8), np.ones((3, 3), np.uint8)
    ).astype(bool)
    decisive = contested & (np.abs(effective_n - effective_s) > 1e-3)

    index_n, index_s = asset.cameras.index("cam-n"), asset.cameras.index("cam-s")
    # Strictly better wins; a tie goes to the first camera by name, as in the mosaic.
    expected = np.where(effective_s < effective_n, index_s, index_n)
    wrong = int(np.count_nonzero(asset.source[decisive] != expected[decisive]))
    colours_wrong = int(np.count_nonzero(
        np.any(asset.colour[decisive] != np.where(
            (expected[decisive] == index_s)[:, None], (255, 0, 0), (0, 0, 255)
        ), axis=1)
    ))
    print(
        f"{int(contested.sum())} contested cells, {int(decisive.sum())} decisive; "
        f"north won {int((asset.source[decisive] == index_n).sum())}, south "
        f"{int((asset.source[decisive] == index_s).sum())}; {wrong} decided against "
        f"the smaller error, {colours_wrong} coloured by the loser"
    )
    assert decisive.sum() > 100
    assert wrong == 0
    assert colours_wrong == 0
    assert (asset.source[decisive] == index_n).any()
    assert (asset.source[decisive] == index_s).any()
    assert np.all(asset.valid[contested])


def test_a_camera_pointed_at_the_sky_feeds_nothing_and_alone_builds_nothing():
    """Counted, never sampled, never sized for — and not a basemap by itself."""
    builder = BasemapBuilder(cell_size_m=1.0)
    feed(builder, "sky", camera(ORIGIN, 180.0, pitch=30.0), 5)

    print(f"fed {builder.frames_fed()}, sampled {builder.frames_sampled()}, blind {builder.blind_cameras()}")
    assert builder.frames_fed() == {"sky": 5}
    assert builder.frames_sampled() == {}
    assert builder.blind_cameras() == ("sky",)
    assert builder.grid is None
    assert builder.memory_bytes == 0
    with pytest.raises(BasemapError, match="sees any ground") as refusal:
        builder.build(now=1_100.0)
    assert "sky" in str(refusal.value)

    with pytest.raises(BasemapError, match="nothing was fed"):
        BasemapBuilder().build()


def test_a_sky_camera_beside_a_real_one_is_recorded_as_contributing_nothing(
    reference_pose, tmp_path: Path
):
    """Provenance, not an absence: the asset names it, with zero cells.

    The blind camera's id sorts *before* the real one's on purpose. ``source``
    indexes the asset's full camera list, and the mosaic underneath knows only
    the cameras that produced a patch — so without the re-indexing in
    ``build()`` every mapped cell here would name the camera that saw nothing.
    """
    aloft = camera(ORIGIN, 180.0, pitch=30.0)
    builder = BasemapBuilder(cell_size_m=1.0)
    feed(builder, "cam-a", reference_pose, 4)
    feed(builder, "aloft", aloft, 4)
    asset = builder.build(now=1_100.0)

    assert asset.cameras == ("aloft", "cam-a")
    assert asset.poses == {"aloft": aloft, "cam-a": reference_pose}
    assert asset.cells_from("aloft") == 0
    assert not (asset.source == asset.cameras.index("aloft")).any()
    assert np.all(asset.source[asset.valid] == asset.cameras.index("cam-a"))
    assert asset.cells_from("cam-a") == asset.covered_cells > 2_000

    _, json_path = save_basemap(asset, tmp_path)
    document = json.loads(json_path.read_text(encoding="utf-8"))
    print(f"cells per camera on disk: {document['coverage']['cells_per_camera']}")
    assert document["coverage"]["cells_per_camera"] == {"aloft": 0, "cam-a": asset.covered_cells}
    assert set(document["poses"]) == {"aloft", "cam-a"}
    assert load_basemap(tmp_path).cameras == ("aloft", "cam-a")


def test_too_few_frames_is_refused_rather_than_saved_as_an_empty_map(reference_pose):
    """Two samples cannot outvote a passer-by, so two samples are not a map."""
    builder = BasemapBuilder(cell_size_m=1.0)
    feed(builder, "cam-a", reference_pose, MINIMUM_SAMPLES - 1)
    with pytest.raises(BasemapError, match=f"seen {MINIMUM_SAMPLES} times") as refusal:
        builder.build(now=1_100.0)
    print(f"refused: {refusal.value}")
    assert f"cam-a {MINIMUM_SAMPLES - 1}" in str(refusal.value)

    builder.feed("cam-a", reference_pose, flat(GROUND), 1_050.0)
    asset = builder.build(now=1_100.0)
    print(f"with {asset.frames_used} frames: {asset.covered_cells} cells")
    assert asset.covered_cells > 2_000


def test_a_camera_that_moved_mid_build_is_refused(reference_pose):
    """Samples already in the ring were placed by the old pose."""
    builder = BasemapBuilder(cell_size_m=1.0)
    feed(builder, "cam-a", reference_pose, 2)
    moved = camera(destination_point(reference_pose.position, 90.0, 2.0), 180.0)
    with pytest.raises(BasemapError, match="different pose"):
        builder.feed("cam-a", moved, flat(GROUND), 1_010.0)
    assert builder.frames_fed() == {"cam-a": 2}


def test_a_builder_refuses_what_it_could_never_report(reference_pose):
    with pytest.raises(BasemapError, match="positive size"):
        BasemapBuilder(cell_size_m=0.0)
    with pytest.raises(BasemapError, match=f"at least {MINIMUM_SAMPLES}"):
        BasemapBuilder(capacity=MINIMUM_SAMPLES - 1)
    with pytest.raises(BasemapError, match="non-negative"):
        BasemapBuilder(pose_sigma_m=-1.0)

    builder = BasemapBuilder(cell_size_m=1.0)
    with pytest.raises(BasemapError, match="uint8"):
        builder.feed("cam-a", reference_pose, np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32), 1.0)
    with pytest.raises(BasemapError, match="finite"):
        builder.feed("cam-a", reference_pose, flat(GROUND), math.nan)
    with pytest.raises(BasemapError, match="needs an id"):
        builder.feed("", reference_pose, flat(GROUND), 1.0)
    assert builder.frames_fed() == {}


def test_a_greyscale_frame_is_mapped_as_grey_ground(reference_pose):
    builder = BasemapBuilder(cell_size_m=1.0)
    for index in range(4):
        builder.feed("cam-a", reference_pose, np.full((HEIGHT, WIDTH), 77, np.uint8), 1_000.0 + index)
    asset = builder.build(now=1_100.0)
    assert asset.colour.shape[2] == 3
    assert np.all(asset.colour[asset.valid] == (77, 77, 77))


def test_memory_is_bounded_by_the_ring_and_is_the_stated_bytes_per_cell(reference_pose):
    """Forty frames must cost exactly what five do, and the docstring's number must be true."""
    builder = BasemapBuilder(cell_size_m=1.0)
    feed(builder, "cam-a", reference_pose, 5)
    after_five = builder.memory_bytes
    feed(builder, "cam-a", reference_pose, 35, start=1_005.0)
    after_forty = builder.memory_bytes

    assert builder.grid is not None
    per_cell = after_forty / builder.grid.cell_count
    print(
        f"{builder.grid.cell_count:,} cells: {after_five / 1e6:.2f} MB after 5 frames, "
        f"{after_forty / 1e6:.2f} MB after 40 — {per_cell:.0f} bytes per cell"
    )
    assert after_forty == after_five
    # The number the module docstring states, derived the way it states it.
    assert per_cell == DEFAULT_CAPACITY * (3 + 4) + 8 == 113


def test_describe_is_one_honest_line(reference_pose):
    asset = one_camera_asset(reference_pose)
    line = asset.describe()
    print(line)
    assert "\n" not in line
    assert f"{asset.covered_cells:,}" in line
    assert "%" in line
    assert "cam-a" in line
    assert "81–81 s old" in line
    assert "1970-01-01 00:18:20 UTC" in line
    assert "20 frame(s)" in line


# ------------------------------------------------------------------ the files


def test_save_and_load_round_trip_everything_exactly(reference_pose, tmp_path: Path):
    asset = one_camera_asset(reference_pose)
    directory = tmp_path / "basemap"

    png_path, json_path = save_basemap(asset, directory)
    loaded = load_basemap(directory)

    print(
        f"{png_path.name} {png_path.stat().st_size:,} bytes, {json_path.name} "
        f"{json_path.stat().st_size:,} bytes; fingerprint {asset.fingerprint[:16]}"
    )
    assert (png_path.name, json_path.name) == (PNG_NAME, JSON_NAME)
    assert sorted(path.name for path in directory.iterdir()) == [JSON_NAME, PNG_NAME], (
        "two files, no temporaries, and nothing else"
    )
    assert loaded is not None
    assert loaded.grid == asset.grid
    assert loaded.poses == asset.poses
    assert loaded.cameras == asset.cameras
    assert loaded.fingerprint == asset.fingerprint
    assert loaded.built_at_millis == asset.built_at_millis
    assert loaded.frames_used == asset.frames_used
    assert np.array_equal(loaded.colour, asset.colour)
    assert np.array_equal(loaded.valid, asset.valid)
    assert np.array_equal(loaded.age_seconds, asset.age_seconds)
    assert np.array_equal(loaded.sigma_m, asset.sigma_m)
    assert np.array_equal(loaded.source, asset.source)
    assert (loaded.colour.dtype, loaded.source.dtype) == (np.uint8, np.int16)
    assert loaded.sigma_m.dtype == loaded.age_seconds.dtype == np.float32

    # A rebuilt asset from the loaded one hashes the same: the identity survives.
    assert load_basemap(directory).fingerprint == asset.fingerprint


def test_the_fingerprint_is_the_documented_hash_of_the_two_files(reference_pose, tmp_path: Path):
    """Recomputed here from scratch, so the format is pinned for any other reader.

    SHA-256 over the PNG bytes, a newline, then the JSON without its
    ``fingerprint`` key, dumped with sorted keys, no spaces and ASCII escapes.
    """
    asset = one_camera_asset(reference_pose)
    png_path, json_path = save_basemap(asset, tmp_path)

    document = json.loads(json_path.read_text(encoding="utf-8"))
    recorded = document.pop("fingerprint")
    canonical = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    recomputed = hashlib.sha256(png_path.read_bytes() + b"\n" + canonical).hexdigest()

    print(f"recorded {recorded[:16]}…, recomputed {recomputed[:16]}…")
    assert recomputed == recorded == asset.fingerprint
    assert document["format"] == FORMAT
    assert set(document) == {
        "format", "grid", "cameras", "poses", "built_at_millis", "frames_used",
        "coverage", "layers",
    }


def test_two_different_grounds_have_different_fingerprints(reference_pose):
    builder = BasemapBuilder(cell_size_m=1.0)
    feed(builder, "cam-a", reference_pose, 4, colour=(1, 2, 3))
    first = builder.build(now=1_100.0)
    other = BasemapBuilder(cell_size_m=1.0)
    feed(other, "cam-a", reference_pose, 4, colour=(3, 2, 1))
    second = other.build(now=1_100.0)
    assert first.fingerprint != second.fingerprint
    assert len(first.fingerprint) == 64


def test_a_png_edited_after_save_fails_to_load(reference_pose, tmp_path: Path):
    """One pixel of one mapped cell, re-encoded: refused, and named as such."""
    asset = one_camera_asset(reference_pose)
    png_path, _ = save_basemap(asset, tmp_path)
    original = png_path.read_bytes()

    picture = cv2.imdecode(np.frombuffer(original, np.uint8), cv2.IMREAD_UNCHANGED)
    row, column = np.argwhere(asset.valid)[0]
    picture[row, column, 0] ^= 0x40
    ok, edited = cv2.imencode(".png", picture)
    assert ok
    png_path.write_bytes(edited.tobytes())
    assert png_path.read_bytes() != original, "the edit must be real"

    with pytest.raises(BasemapError, match="fingerprint") as refusal:
        load_basemap(tmp_path)
    print(f"refused: {refusal.value}")
    assert "rebuild" in str(refusal.value)


def test_a_json_edited_after_save_fails_to_load(reference_pose, tmp_path: Path):
    """The metadata is under the same hash: one changed number is a different asset."""
    asset = one_camera_asset(reference_pose)
    _, json_path = save_basemap(asset, tmp_path)

    document = json.loads(json_path.read_text(encoding="utf-8"))
    document["frames_used"] += 1
    json_path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(BasemapError, match="fingerprint"):
        load_basemap(tmp_path)


def test_a_pose_edited_after_save_fails_to_load(reference_pose, tmp_path: Path):
    """The poses are what "inside the fence" was judged with; they are hashed too."""
    asset = one_camera_asset(reference_pose)
    _, json_path = save_basemap(asset, tmp_path)

    document = json.loads(json_path.read_text(encoding="utf-8"))
    document["poses"]["cam-a"]["heading"] += 1.0
    json_path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(BasemapError, match="fingerprint"):
        load_basemap(tmp_path)


def test_a_half_written_pair_is_refused_and_an_orphan_png_is_not_an_asset(
    reference_pose, tmp_path: Path
):
    """A crash between the two renames leaves a new PNG beside the old JSON."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    builder = BasemapBuilder(cell_size_m=1.0)
    feed(builder, "cam-a", reference_pose, 4, colour=(1, 2, 3))
    save_basemap(builder.build(now=1_100.0), first)
    other = BasemapBuilder(cell_size_m=1.0)
    feed(other, "cam-a", reference_pose, 4, colour=(3, 2, 1))
    save_basemap(other.build(now=1_100.0), second)

    shutil.copyfile(second / PNG_NAME, first / PNG_NAME)
    with pytest.raises(BasemapError, match="fingerprint"):
        load_basemap(first)

    (first / JSON_NAME).unlink()
    assert load_basemap(first) is None, "a PNG with no record is not a basemap"

    (second / PNG_NAME).unlink()
    with pytest.raises(BasemapError, match="missing"):
        load_basemap(second)


def test_nothing_there_loads_as_none(tmp_path: Path):
    assert load_basemap(tmp_path / "absent") is None
    assert load_basemap(tmp_path) is None


def test_a_failed_save_leaves_the_previous_asset_intact_or_refused_never_wrong(
    reference_pose, tmp_path: Path, monkeypatch
):
    """The atomic claim, both halves.

    A failure before the PNG is renamed leaves the old pair untouched and
    loadable. A failure between the two renames leaves a new PNG and an old
    JSON — and that pair is refused, not drawn.
    """
    import sentinel.basemap as module

    builder = BasemapBuilder(cell_size_m=1.0)
    feed(builder, "cam-a", reference_pose, 4, colour=(1, 2, 3))
    kept = builder.build(now=1_100.0)
    save_basemap(kept, tmp_path)
    other = BasemapBuilder(cell_size_m=1.0)
    feed(other, "cam-a", reference_pose, 4, colour=(3, 2, 1))
    replacement = other.build(now=1_100.0)

    real_replace = module.os.replace
    calls: list[int] = []

    def failing(source, target, *, fail_on: int):
        calls.append(1)
        if len(calls) == fail_on:
            raise OSError("the disk went away")
        return real_replace(source, target)

    monkeypatch.setattr(module.os, "replace", lambda s, t: failing(s, t, fail_on=1))
    with pytest.raises(OSError):
        save_basemap(replacement, tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == [JSON_NAME, PNG_NAME], (
        "no temporary left behind"
    )
    assert load_basemap(tmp_path).fingerprint == kept.fingerprint

    calls.clear()
    monkeypatch.setattr(module.os, "replace", lambda s, t: failing(s, t, fail_on=2))
    with pytest.raises(OSError):
        save_basemap(replacement, tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == [JSON_NAME, PNG_NAME]
    with pytest.raises(BasemapError, match="fingerprint"):
        load_basemap(tmp_path)


def test_an_asset_altered_after_it_was_built_cannot_be_saved_under_its_fingerprint(
    reference_pose, tmp_path: Path
):
    asset = one_camera_asset(reference_pose)
    row, column = np.argwhere(asset.valid)[0]
    asset.colour[row, column] = (9, 9, 9)
    with pytest.raises(BasemapError, match="fingerprint"):
        save_basemap(asset, tmp_path)
    assert not (tmp_path / PNG_NAME).exists()


def test_the_files_carry_the_ground_and_its_provenance_and_nothing_else(
    reference_pose, tmp_path: Path
):
    """No frame, no source. The PNG is the raster; the JSON is the record."""
    asset = one_camera_asset(reference_pose)
    png_path, json_path = save_basemap(asset, tmp_path)

    picture = cv2.imdecode(np.frombuffer(png_path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
    assert picture.shape == (asset.grid.rows, asset.grid.columns, 4)
    assert np.array_equal(picture[:, :, 3] > 0, asset.valid), "alpha is the valid mask"

    document = json.loads(json_path.read_text(encoding="utf-8"))
    assert document["grid"] == {
        "origin": {"lat": asset.grid.origin.lat, "lon": asset.grid.origin.lon},
        "cell_size_m": 1.0, "rows": asset.grid.rows, "columns": asset.grid.columns,
    }
    assert document["poses"]["cam-a"] == {
        "lat": reference_pose.position.lat, "lon": reference_pose.position.lon,
        "mount_height": 6.0, "heading": 180.0, "pitch": -22.0, "roll": 0.0,
        "horizontal_fov": 62.0, "vertical_fov": 36.0, "range_meters": 90.0,
    }
    assert document["coverage"]["covered_cells"] == asset.covered_cells
    assert set(document["layers"]) == {"sigma_m", "age_seconds", "source"}
    # No NaN or Infinity written as a *number*: Python's encoder would, and a
    # strict reader refuses both. Checked through the parser rather than as a
    # substring, because the base64 of a layer can spell "NaN" by chance —
    # and did, the day the grid changed shape by one cell.
    def refuse(constant: str) -> None:
        raise AssertionError(f"{constant} was written as a number")

    json.loads(json_path.read_text(encoding="utf-8"), parse_constant=refuse)


def test_basemap_directory_is_under_the_data_directory():
    assert basemap_directory(Path("elsewhere")) == Path("elsewhere") / "basemap"
    # The suite's conftest points the data directory at a temporary folder;
    # the default follows it, so a test can never write a real site's basemap.
    assert basemap_directory() == paths.data_directory() / "basemap"
    assert basemap_directory().is_relative_to(paths.data_directory())
