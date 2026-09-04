"""Tests for the basemap builder, its files, and its fingerprint.

`test_orthophoto` proves the pieces: the inverse map, the median, the mosaic.
This proves the product path that finally calls them, and three claims carry
the weight:

- **What is fed is what is mapped, and nothing else.** Twenty frames of flat
  ground from the reference pose produce ground exactly inside that pose's
  footprint, in that colour, with a finite age and error on every mapped cell
  and ``inf`` on every empty one — and a bright square crossing a minority of
  those frames leaves no trace.
- **A second camera extends the grid without disturbing the first.** The
  accumulator is moved by an index shift, not resampled, and the moved result
  is held to the unmoved one cell for cell. A wrong offset, or a rebuild that
  quietly dropped the first camera's minutes, fails here.
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
from sentinel.basemap import (
    DEFAULT_POSE_SIGMA_M,
    FORMAT,
    JSON_NAME,
    PNG_NAME,
    BasemapAsset,
    BasemapBuilder,
    BasemapError,
    _moved,
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
    MedianAccumulator,
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
            top = 180 + 8 * index
            image[top:top + 160, 120 + 12 * index:420 + 12 * index] = 255

    builder = BasemapBuilder(cell_size_m=1.0)
    feed(builder, "cam-a", reference_pose, 20, paint=crossing)
    asset = builder.build(now=1_100.0)

    painted = flat(GROUND)
    crossing(19, painted)
    alone = sample_frame(reference_pose, painted, asset.grid, camera_id="cam-a")
    struck = int(np.count_nonzero(np.all(alone.colour == 255, axis=2) & alone.valid))

    colours = np.unique(asset.colour[asset.valid].reshape(-1, 3), axis=0)
    print(
        f"the square covers {struck} cells in one frame alone; colours surviving "
        f"the median: {colours.tolist()}"
    )
    assert struck > 200
    assert colours.tolist() == [list(GROUND)]


def test_the_grid_grows_for_a_second_camera_and_the_first_keeps_every_cell(reference_pose):
    """Extension is an index shift of the ring, and this holds it to that.

    The first camera is built alone, then a camera whose footprint lies
    outside the grid is fed and the pair is built. Every cell the first asset
    mapped must be mapped in the second at the shifted index, with the same
    colour, age and error wherever the first camera still owns it. A rebuild
    that dropped the first camera's samples fails here with no north cells at
    all; an offset wrong by one cell fails at the footprint's edge.
    """
    north, south = facing_pair(reference_pose)
    builder = BasemapBuilder(cell_size_m=1.0)
    feed(builder, "cam-n", north, 5, colour=(0, 0, 255))
    first = builder.build(now=1_100.0)
    small = first.grid

    feed(builder, "cam-s", south, 5, colour=(255, 0, 0))
    second = builder.build(now=1_100.0)
    large = second.grid

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


def test_a_moved_accumulator_reports_exactly_what_the_unmoved_one_did(reference_pose):
    """The extension copies the ring through the accumulator's own slots.

    That coupling is the one place this module knows the accumulator's layout,
    and this is what keeps it honest: after the move, the median, the count,
    the error and the newest sample of every cell are identical at the shifted
    index, and nothing outside the shifted window is ground.
    """
    small = GroundGrid.covering(field_of_view(reference_pose), cell_size_m=1.0, margin_m=1.0)
    accumulator = MedianAccumulator(small, capacity=6)
    for index in range(4):
        image = flat((10, 20, 30))
        if index == 2:
            image[:, :] = (200, 200, 200)
        accumulator.update(
            sample_frame(reference_pose, image, small, camera_id="cam-a",
                         captured_at=500.0 + index)
        )
    before = accumulator.result(camera_id="cam-a", now=600.0)

    large = GroundGrid(
        origin=_Frame(small.origin).to_latlon(-2.0, 3.0),
        cell_size_m=1.0, columns=small.columns + 5, rows=small.rows + 7,
    )
    moved = _moved(accumulator, large, row_offset=3, column_offset=2)
    after = moved.result(camera_id="cam-a", now=600.0)

    window = (slice(3, 3 + small.rows), slice(2, 2 + small.columns))
    outside = np.ones(large.shape, dtype=bool)
    outside[window] = False
    print(
        f"{before.covered_cells} cells before the move, {after.covered_cells} after; "
        f"{int(after.valid[outside].sum())} outside the window"
    )
    assert moved.frames == accumulator.frames == 4
    assert after.covered_cells == before.covered_cells > 2_000
    assert np.array_equal(after.valid[window], before.valid)
    assert np.array_equal(after.colour[window], before.colour)
    assert np.array_equal(after.samples[window], before.samples)
    assert np.array_equal(after.sigma_m[window], before.sigma_m, equal_nan=True)
    assert np.array_equal(after.updated_at[window], before.updated_at, equal_nan=True)
    assert not after.valid[outside].any()


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
    """Provenance, not an absence: the asset names it, with zero cells."""
    sky = camera(ORIGIN, 180.0, pitch=30.0)
    builder = BasemapBuilder(cell_size_m=1.0)
    feed(builder, "cam-a", reference_pose, 4)
    feed(builder, "sky", sky, 4)
    asset = builder.build(now=1_100.0)

    assert asset.cameras == ("cam-a", "sky")
    assert asset.poses == {"cam-a": reference_pose, "sky": sky}
    assert asset.cells_from("sky") == 0
    assert not (asset.source == asset.cameras.index("sky")).any()
    assert asset.cells_from("cam-a") == asset.covered_cells > 2_000

    _, json_path = save_basemap(asset, tmp_path)
    document = json.loads(json_path.read_text(encoding="utf-8"))
    print(f"cells per camera on disk: {document['coverage']['cells_per_camera']}")
    assert document["coverage"]["cells_per_camera"] == {"cam-a": asset.covered_cells, "sky": 0}
    assert set(document["poses"]) == {"cam-a", "sky"}


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
    text = json_path.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text


def test_basemap_directory_is_under_the_data_directory():
    assert basemap_directory(Path("elsewhere")) == Path("elsewhere") / "basemap"
    # The suite's conftest points the data directory at a temporary folder;
    # the default follows it, so a test can never write a real site's basemap.
    assert basemap_directory() == paths.data_directory() / "basemap"
    assert basemap_directory().is_relative_to(paths.data_directory())
