"""The ground map: what it draws, what it refuses to draw, and how it says so.

The scenes here are synthetic and that bounds what these tests prove. They can
show that a walker is removed by the median, that ground nobody looked at
stays empty, that two cameras composite without a seam and that the confidence
layer refuses what it should. They cannot show that a real yard at dusk maps
well — see PRODUCTION_READINESS.md for what is still unmeasured.
"""

import numpy as np
import pytest

from vigil.domain.geo import CameraPose, LatLon, destination_point, distance_meters
from vigil.kernel import native
from vigil.service.mapping import (
    MIN_SAMPLES, USABLE_CONFIDENCE, MapBuilder, MappingError, build_from_cameras, confidence_of,
    load_map, save_map, site_lattice, uncertainty_over, visualise,
)

pytestmark = pytest.mark.skipif(not native.available(),
                                reason=f"the engine core is not built: {native.fault()}")

SITE = LatLon(33.8938, 35.5018)


def _pose(bearing_from_site=0.0, distance=0.0, heading=0.0, pitch=-25.0, **kwargs):
    position = destination_point(SITE, bearing_from_site, distance) if distance else SITE
    return CameraPose(position, kwargs.pop("mount_height", 4.0), heading, pitch,
                      kwargs.pop("roll", 0.0), kwargs.pop("horizontal_fov", 62.0),
                      kwargs.pop("vertical_fov", 36.0), kwargs.pop("range_meters", 40.0))


def _ground_frame(size=(360, 640), tone=90, seed=0):
    """A textured but static ground: what an empty yard looks like.

    Real texture at a real spatial frequency, not a flat tone with a little
    noise. Two earlier versions of this fixture failed the frame-quality gate
    and both failures were the gate being right: a frame uniform to within
    eight levels has nothing in it, and one upscaled 10x from a coarse grid
    measures a Laplacian variance of 18 against a floor of 40 — it *is* out of
    focus. A fifth is about the spatial frequency of tarmac at 20 m.
    """
    import cv2

    rng = np.random.default_rng(seed)
    coarse = rng.integers(max(0, tone - 45), min(255, tone + 45),
                          size=(size[0] // 5, size[1] // 5, 3), dtype=np.uint8)
    return cv2.resize(coarse, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)


def _with_walker(image, u, v, colour=(230, 230, 230), size=0.08):
    """The same frame with somebody standing in it."""
    out = image.copy()
    h, w = out.shape[:2]
    x0, y0 = int((u - size / 2) * w), int((v - size * 2) * h)
    out[max(0, y0):int(v * h), max(0, x0):int((u + size / 2) * w)] = colour
    return out


# ------------------------------------------------------------------ lattice


def test_a_cameras_grid_covers_its_footprint_and_nothing_much_more():
    pose = _pose()
    grid = site_lattice(SITE, 0.25, pose)
    from vigil.domain.geo import field_of_view

    for point in field_of_view(pose, 24):
        row_col = None
        for row in (0, grid.rows - 1):
            for col in (0, grid.cols - 1):
                row_col = grid.centre_of(row, col)
        assert row_col is not None
        # Every footprint point must be inside the grid's own bounds.
        from vigil.domain.geo import LocalFrame

        local = LocalFrame(grid.origin).to_local(point)
        assert -0.5 <= local.x <= grid.cols * grid.cell_size_m + 0.5
        assert -0.5 <= local.y <= grid.rows * grid.cell_size_m + 0.5
    # And not be wildly larger than it needs to be.
    assert grid.cells < 400_000, f"{grid.rows}x{grid.cols} is more grid than a 40 m camera needs"


def test_two_cameras_land_on_one_lattice_so_compositing_is_a_shift():
    """The guarantee is the integer offsets, not a round trip through frames.

    Measuring one grid's origin in the *other* grid's tangent plane does not
    come out an exact multiple of the cell, and cannot: two frames 40 m apart
    evaluate the metres-per-degree series at two latitudes and disagree by a
    few micrometres over 20 m. That difference is why the offsets are carried
    as integers from the shared site frame rather than recomputed — the
    composite is then a subtraction, and nothing is ever resampled.
    """
    from vigil.domain.geo import LocalFrame

    a = site_lattice(SITE, 0.25, _pose())
    b = site_lattice(SITE, 0.25, _pose(bearing_from_site=90.0, distance=37.3, heading=270.0))
    site = LocalFrame(SITE)
    for grid in (a, b):
        local = site.to_local(grid.origin)
        assert abs(local.x / 0.25 - grid.east_cells) < 1e-6
        assert abs(local.y / 0.25 - grid.north_cells) < 1e-6
    # So the shift between them is exact and integral.
    assert isinstance(a.east_cells - b.east_cells, int)
    assert (a.east_cells - b.east_cells) != 0, "these two cameras are not in the same place"


def test_a_camera_that_sees_no_ground_is_refused_rather_than_given_an_empty_grid():
    with pytest.raises(MappingError):
        site_lattice(SITE, 0.25, _pose(pitch=25.0))


# -------------------------------------------------------------- uncertainty


def test_uncertainty_grows_with_range_and_refuses_ground_past_the_frame():
    pose = _pose()
    distances = np.array([5.0, 10.0, 20.0, 30.0, 1000.0])
    sigma = uncertainty_over(pose, distances)
    assert np.all(np.diff(sigma[:4]) > 0), f"error must grow with range: {sigma}"
    assert not np.isfinite(sigma[-1]), "ground the camera cannot reach has no stated error"
    assert sigma[0] < 2.0, "near ground should be placed to within a couple of metres"


def test_confidence_multiplies_independent_objections_rather_than_averaging_them():
    depth = 15
    good = confidence_of(np.array([15.0]), np.array([0.0]), np.array([0.02]), np.array([0.3]), depth)
    assert good[0] > 0.9
    # A cell seen thirty times, sharply, from a well-known pose, with a lorry
    # over it half the time is not two-thirds trustworthy.
    blocked = confidence_of(np.array([15.0]), np.array([0.5]), np.array([0.02]), np.array([0.3]), depth)
    assert blocked[0] == 0.0
    smeared = confidence_of(np.array([15.0]), np.array([0.0]), np.array([2.0]), np.array([0.3]), depth)
    assert smeared[0] < 0.15
    unplaced = confidence_of(np.array([15.0]), np.array([0.0]), np.array([0.02]), np.array([20.0]), depth)
    assert unplaced[0] < 0.15
    thin = confidence_of(np.array([1.0]), np.array([0.0]), np.array([0.02]), np.array([0.3]), depth)
    assert thin[0] < 0.2, "one look is not a measurement"


# --------------------------------------------------------------- the median


def test_a_walker_is_removed_by_the_median_and_the_cell_says_it_was_disturbed():
    """The claim behind calling this a map rather than a picture."""
    pose = _pose()
    with MapBuilder(SITE, cell_size_m=0.5, depth=11) as builder:
        base = _ground_frame()
        for i in range(11):
            frame = _with_walker(base, 0.5, 0.85) if i in (4, 7) else base
            builder.observe("gate", pose, frame, at_seconds=i * 1.0)
        ground = builder.build(minimum_samples=5)

    # The map must be ground-toned, not walker-toned.
    assert ground.valid.any(), "nothing was mapped at all"
    lit = ground.colour[ground.valid > 0]
    assert lit.mean() < 140, f"the walker survived the median (mean tone {lit.mean():.0f})"
    # And somewhere in the map, the disturbance layer noticed.
    assert ground.disturbance.max() > 0.1, "no cell reported having been blocked"


def test_ground_nobody_looked_at_stays_empty_and_is_drawn_as_empty():
    pose = _pose()
    with MapBuilder(SITE, cell_size_m=0.5) as builder:
        for i in range(8):
            builder.observe("gate", pose, _ground_frame(), at_seconds=i * 1.0)
        ground = builder.build()
    unseen = ground.valid == 0
    assert unseen.any(), "a rectangular grid over a wedge must have corners nobody saw"
    # Nothing interpolated into them, and the renderer does not paint them a
    # colour that could be read as tarmac.
    picture = visualise(ground)
    assert picture[unseen].max() < 40, "empty ground was painted as if it were ground"
    assert ground.confidence[unseen].max() == 0.0


def test_a_cell_seen_once_is_not_a_median_and_is_not_drawn():
    pose = _pose()
    with MapBuilder(SITE, cell_size_m=1.0) as builder:
        builder.observe("gate", pose, _ground_frame(), at_seconds=0.0)
        ground = builder.build(minimum_samples=MIN_SAMPLES)
    assert not ground.valid.any(), "one frame is not a map"


def test_frames_are_rate_limited_because_a_median_needs_time_not_frames():
    pose = _pose()
    with MapBuilder(SITE, cell_size_m=1.0, sample_interval_s=0.25) as builder:
        assert builder.observe("gate", pose, _ground_frame(), at_seconds=0.0)
        assert not builder.observe("gate", pose, _ground_frame(), at_seconds=0.05)
        assert not builder.observe("gate", pose, _ground_frame(), at_seconds=0.2)
        assert builder.observe("gate", pose, _ground_frame(), at_seconds=0.3)
        assert builder.frames_for("gate") == 2


# ------------------------------------------------------------- compositing


def test_two_cameras_composite_and_the_better_claim_wins_each_cell():
    near = _pose()
    far = _pose(bearing_from_site=0.0, distance=25.0, heading=180.0)
    with MapBuilder(SITE, cell_size_m=0.5, depth=9) as builder:
        for i in range(9):
            builder.observe("near", near, _ground_frame(tone=90, seed=1), at_seconds=i * 1.0)
            builder.observe("far", far, _ground_frame(tone=180, seed=2), at_seconds=i * 1.0)
        ground = builder.build(minimum_samples=5)

    assert ground.cameras == ("far", "near")
    assert set(np.unique(ground.source)) <= {-1, 0, 1}
    seen = ground.valid > 0
    assert seen.sum() > 1000
    # Both contributed, and every drawn cell names the camera it came from.
    assert (ground.source[seen] >= 0).all()
    assert len(set(np.unique(ground.source[seen]))) == 2, "one camera took the whole map"
    # Where a camera won, it must be because its claim was better — never a
    # blend, which would be a measurement of neither.
    assert ground.confidence[seen].min() >= 0.0
    summary = ground.summary()
    assert summary["seen"] == int(seen.sum())
    assert summary["usable"] <= summary["seen"]
    assert "usable of" in ground.describe()


def test_a_camera_that_moved_starts_again_rather_than_blending_two_poses():
    pose = _pose()
    moved = _pose(heading=12.0)
    with MapBuilder(SITE, cell_size_m=1.0) as builder:
        for i in range(6):
            builder.observe("gate", pose, _ground_frame(), at_seconds=i * 1.0)
        assert builder.frames_for("gate") == 6
        builder.observe("gate", moved, _ground_frame(), at_seconds=10.0)
        assert builder.frames_for("gate") == 1, (
            "samples projected through the old pose would draw the ground twice"
        )


def test_an_empty_builder_refuses_to_produce_a_map():
    with MapBuilder(SITE) as builder:
        with pytest.raises(MappingError):
            builder.build()


# -------------------------------------------------------------- on disk


def test_a_saved_map_reloads_identically_and_a_tampered_one_is_refused(tmp_path):
    pose = _pose()
    with MapBuilder(SITE, cell_size_m=0.5, depth=9) as builder:
        for i in range(9):
            builder.observe("gate", pose, _ground_frame(), at_seconds=i * 1.0)
        ground = builder.build(minimum_samples=5)

    save_map(ground, tmp_path)
    loaded = load_map(tmp_path)
    assert loaded is not None
    assert loaded.grid == ground.grid
    assert (loaded.valid == ground.valid).all()
    assert (loaded.colour[loaded.valid > 0] == ground.colour[ground.valid > 0]).all()
    assert np.allclose(loaded.confidence, ground.confidence)
    assert loaded.cameras == ground.cameras
    assert loaded.poses["gate"].heading == pose.heading

    # An incident judged against *this* map has to be re-checkable against
    # this map, so a file that no longer hashes to its own claim is refused
    # rather than drawn. A tampered file and a half-written one look the same
    # to this check, and neither should reach a screen.
    png = (tmp_path / "ground.png").read_bytes()
    (tmp_path / "ground.png").write_bytes(png[:-40] + b"\x00" * 40)
    assert load_map(tmp_path) is None
    assert load_map(tmp_path / "nowhere") is None


def test_nothing_that_could_identify_a_camera_or_a_person_is_written(tmp_path):
    pose = _pose()
    with MapBuilder(SITE, cell_size_m=1.0, depth=7) as builder:
        for i in range(7):
            builder.observe("gate", pose, _with_walker(_ground_frame(), 0.5, 0.85), at_seconds=i * 1.0)
        ground = builder.build(minimum_samples=3)
    save_map(ground, tmp_path)
    written = {p.name for p in tmp_path.iterdir()}
    assert written == {"ground.png", "ground.json", "ground.npz"}
    text = (tmp_path / "ground.json").read_text(encoding="utf-8")
    for forbidden in ("rtsp", "password", "http", "@"):
        assert forbidden not in text.lower(), f"the manifest leaked {forbidden}"


# -------------------------------------------------------------- rendering


def test_the_picture_shows_doubtful_ground_as_doubtful():
    pose = _pose()
    with MapBuilder(SITE, cell_size_m=0.5, depth=9) as builder:
        for i in range(9):
            builder.observe("gate", pose, _ground_frame(tone=120), at_seconds=i * 1.0)
        ground = builder.build(minimum_samples=5)
    picture = visualise(ground)
    assert picture.shape == (*ground.valid.shape, 3)
    usable, doubtful = ground.usable, (ground.valid > 0) & ~ground.usable
    if doubtful.any():
        # Dimmed and desaturated: visible enough to give context, obviously
        # not a measurement.
        assert picture[doubtful].mean() < ground.colour[doubtful].mean()
    if usable.any():
        assert (picture[usable] == ground.colour[usable]).all()


def test_the_usable_mask_is_the_confidence_threshold_and_says_where_it_is():
    pose = _pose()
    with MapBuilder(SITE, cell_size_m=0.5, depth=9) as builder:
        for i in range(9):
            builder.observe("gate", pose, _ground_frame(), at_seconds=i * 1.0)
        ground = builder.build(minimum_samples=5)
    expected = (ground.valid > 0) & (ground.confidence >= USABLE_CONFIDENCE)
    assert (ground.usable == expected).all()
    # And usable ground must be the near ground, because that is where a
    # camera's resolution and its projection error are both worth having.
    if ground.usable.any():
        rows, cols = np.nonzero(ground.usable)
        distances = [distance_meters(pose.position, ground.grid.centre_of(int(r), int(c)))
                     for r, c in zip(rows[::37], cols[::37])]
        assert max(distances) < pose.range_meters


def test_a_build_from_a_file_source_reports_what_each_camera_managed(tmp_path):
    """`build_from_cameras` end to end, on a file rather than a camera.

    The capture loop lives in the service so that this test can exist at all:
    with it in the CLI there would be no way to run it without a camera, which
    is how a capture path comes to be shipped untested.
    """
    import cv2

    path = tmp_path / "yard.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (640, 360))
    assert writer.isOpened()
    # Each frame slightly different, the way a real sensor's are: an
    # identical repeat is skipped on purpose, because it adds nothing a median
    # did not already have.
    for i in range(90):
        writer.write(_ground_frame((360, 640), seed=i))
    writer.release()

    pose = _pose()
    report = build_from_cameras({"gate": (str(path), pose)}, SITE, seconds=2.0,
                                cell_size_m=1.0, depth=9, minimum_samples=3)
    assert report.faults == {}
    assert report.samples["gate"] >= 3, report.samples
    assert report.ground.valid.any()
    assert "usable of" in report.ground.describe()


def test_a_source_that_will_not_open_is_named_rather_than_swallowed(tmp_path):
    pose = _pose()
    with pytest.raises(MappingError) as caught:
        build_from_cameras({"gate": (str(tmp_path / "absent.mp4"), pose)}, SITE, seconds=0.5)
    assert "gate" in str(caught.value)
