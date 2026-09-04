"""Tests for the basemap drawn under the plan view.

The map the site draws of itself is read back here **as pixels**, at the screen
points the view itself computes for the grid's cells. A text assertion on a
layer that exists only as a picture proves nothing; these grab the rendered
widget and look at it, the way the footprint and marker tests already do.

What is checked is what would mislead an operator if it were wrong:

- **Empty is transparent.** A cell no camera covered shows the panel, never
  the asset's colour and never black. The mask is the authority, not the pixel.
- **The georeference is the view's own.** A cell's centre, taken from the grid
  in lat/lon and pushed through the same `_to_local` and `_to_screen` as every
  track, lands where that cell is drawn, to the pixel. A basemap a metre off
  the tracks on top of it would put every judgement made against it a metre
  out.
- **The grid stays on top.** The metric lines are what make a distance
  readable; a photograph painted over them takes that away.
- **Stale ground looks stale**, and staleness is judged by the clock, not by
  the moment the asset was built.
- **Switched off means nothing**: not drawn, not in the legend, not under the
  pointer.

The asset is synthetic — a solid colour over a known rectangle of cells — so
every expected pixel is computable. The engine half of this slice
(``sentinel.basemap``) builds real ones; its dataclass is used when it is
importable, and a stand-in with the same fields, built by name, when the two
halves are being written in parallel. Nothing here depends on which.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QImage, QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from sentinel.core import (  # noqa: E402
    BoundingBox,
    CameraPose,
    LatLon,
    PositionEstimate,
    Track,
    destination_point,
)
from sentinel.orthophoto import GroundGrid  # noqa: E402
from sentinel_console import theme  # noqa: E402
from sentinel_console.map_view import BASEMAP_STALE_AFTER_SECONDS, MapView  # noqa: E402

try:
    from sentinel.basemap import BasemapAsset
except ImportError:  # the engine half of the slice is being written alongside this
    BasemapAsset = None


@pytest.fixture(scope="session")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


# The same camera the console tests place: 6 m up, facing south, 22° down.
SITE_POSE = CameraPose(
    position=LatLon(33.8938, 35.5018), mount_height=6.0, heading=180.0,
    pitch=-22.0, horizontal_fov=62.0, vertical_fov=36.0, range_meters=90.0,
)

#: The ground colour, as the asset carries it (BGR — it comes from OpenCV) and
#: as the screen must show it. Red and blue are far apart on purpose: a
#: renderer that forgot the swap paints the yard the wrong colour, and this is
#: the test that would notice.
GROUND_BGR = (20, 160, 90)
GROUND_RGB = (90, 160, 20)
PANEL_RGB = (theme.PANEL.red(), theme.PANEL.green(), theme.PANEL.blue())


@dataclass(frozen=True, slots=True)
class _StandInAsset:
    """The shared contract's fields, verbatim, for when ``sentinel.basemap`` is
    not importable yet. Built by keyword only, so a renamed field on either
    side fails loudly at construction rather than silently drawing nothing."""

    grid: GroundGrid
    colour: np.ndarray
    valid: np.ndarray
    age_seconds: np.ndarray
    sigma_m: np.ndarray
    source: np.ndarray
    cameras: tuple[str, ...]
    poses: dict
    built_at_millis: int
    frames_used: int
    fingerprint: str

    @property
    def covered_fraction(self) -> float:
        return float(np.count_nonzero(self.valid)) / self.valid.size

    def describe(self) -> str:
        return f"{self.valid.size} cells, {self.covered_fraction:.0%} covered"


def _grid(*, columns: int = 20, rows: int = 20, cell: float = 1.0) -> GroundGrid:
    """A metre grid whose north-west corner is 10 m north and 10 m west of the
    camera, so the camera sits on the boundary between cells 9 and 10 in both
    directions and the crosshair falls on a cell edge, never through a centre."""
    north = destination_point(SITE_POSE.position, 0.0, 10.0)
    origin = destination_point(north, 270.0, 10.0)
    return GroundGrid(origin=origin, cell_size_m=cell, columns=columns, rows=rows)


def _rectangle(grid: GroundGrid, rows=slice(2, 15), columns=slice(2, 10)) -> np.ndarray:
    """A block of ground west of the camera's meridian, crossing its parallel."""
    valid = np.zeros(grid.shape, dtype=bool)
    valid[rows, columns] = True
    return valid


def _asset(
    grid: GroundGrid | None = None,
    valid: np.ndarray | None = None,
    *,
    colour=GROUND_BGR,
    age: float = 0.0,
    sigma: float = 0.8,
    cameras: tuple[str, ...] = ("gate",),
    source: int = 0,
    built_at: float | None = None,
    cls=None,
):
    """A synthetic asset: one colour over the valid cells, nothing elsewhere.

    Everything outside the mask is inf / -1 / zero, as the contract says it
    must be, so a renderer that reads those instead of the mask draws
    something visible and fails the empty-cell checks.
    """
    grid = grid or _grid()
    valid = _rectangle(grid) if valid is None else valid
    pixels = np.zeros((*grid.shape, 3), dtype=np.uint8)
    pixels[valid] = colour
    cls = cls or BasemapAsset or _StandInAsset
    built = time.time() if built_at is None else built_at
    return cls(
        grid=grid,
        colour=pixels,
        valid=valid,
        age_seconds=np.where(valid, float(age), np.inf),
        sigma_m=np.where(valid, float(sigma), np.inf),
        source=np.where(valid, source, -1).astype(np.int16),
        cameras=tuple(cameras),
        poses={camera: SITE_POSE for camera in cameras},
        built_at_millis=int(built * 1000),
        frames_used=40,
        fingerprint=hashlib.sha256(pixels.tobytes()).hexdigest(),
    )


def _shown_map(asset=None, *, legend: bool = False) -> MapView:
    """A shown 600x600 plan view of one placed camera with a basemap under it.

    Shown and the events processed, because `grab()` on a widget that was
    never realised is a picture of nothing. The legend is off unless a test is
    about it: it overlays the bottom-right corner and several samples land there.
    """
    view = MapView()
    view.resize(600, 600)
    view.set_cameras({"gate": SITE_POSE})
    if asset is not None:
        view.set_basemap(asset)
    view.show_legend = legend
    view.show()
    QApplication.processEvents()
    # A cell has to be several pixels across for a pixel at its centre to be
    # that cell's and not its neighbour's. Measured: 5.3 px/m after the fit.
    assert view._scale() >= 4.0, f"a metre is only {view._scale():.1f} px here"
    return view


def _render(view: MapView) -> QImage:
    return view.grab().toImage().convertToFormat(QImage.Format.Format_RGB888)


def _at(image: QImage, point: QPointF) -> tuple[int, int, int]:
    colour = image.pixelColor(int(point.x()), int(point.y()))
    return (colour.red(), colour.green(), colour.blue())


def _cell_screen(view: MapView, row: int, column: int) -> QPointF:
    """Where the view says a cell's centre is, via the same path as a track."""
    return view._to_screen(*view._to_local(view.basemap.grid.cell_centre(row, column)))


def _composite(rgb, opacity: float) -> tuple[int, int, int]:
    """What the ground colour looks like at this opacity over the panel."""
    return tuple(round(v * opacity + p * (1.0 - opacity)) for v, p in zip(rgb, PANEL_RGB))


def _close(a, b, tolerance: int = 4) -> bool:
    return all(abs(x - y) <= tolerance for x, y in zip(a, b))


def _move(view, at):
    """Move the pointer to a widget position, delivered straight to the widget
    (QTest routes by global position, and in a full run another window sits
    over the same screen coordinates)."""
    view.mouseMoveEvent(QMouseEvent(
        QEvent.Type.MouseMove, at, view.mapToGlobal(at),
        Qt.MouseButton.NoButton, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
    ))
    QApplication.processEvents()


# ------------------------------------------------------------------ drawing


def test_ground_shows_the_basemap_colour_and_empty_cells_show_the_panel(qt_app):
    """The asset's colour where the mask says ground; the panel where it does
    not — never the zero the colour array holds there, which is also black."""
    view = _shown_map(_asset())
    image = _render(view)

    ground = _at(image, _cell_screen(view, 5, 5))
    empty = _at(image, _cell_screen(view, 5, 14))
    expected = _composite(GROUND_RGB, view.basemap_opacity)
    print(f"ground {ground} (expected {expected}), empty {empty} (panel {PANEL_RGB})")

    assert _close(ground, expected), f"a covered cell is {ground}, not {expected}"
    assert empty == PANEL_RGB, f"an uncovered cell is {empty}: the mask is not the authority"


def test_taking_the_basemap_away_removes_it(qt_app):
    view = _shown_map(_asset())
    assert view.basemap_rect() is not None

    view.set_basemap(None)
    QApplication.processEvents()

    assert view.basemap is None
    assert view.basemap_rect() is None
    assert _at(_render(view), _cell_screen(_shown_map(_asset()), 5, 5)) == PANEL_RGB


def test_the_opacity_is_applied_and_nought_draws_nothing(qt_app):
    view = _shown_map(_asset())
    where = _cell_screen(view, 5, 5)

    assert view.basemap_opacity == 0.85
    assert _close(_at(_render(view), where), _composite(GROUND_RGB, 0.85))

    view.basemap_opacity = 1.0
    assert _at(_render(view), where) == GROUND_RGB, "opaque is not the colour itself"

    view.basemap_opacity = 0.0
    assert _at(_render(view), where) == PANEL_RGB, "fully transparent still drew something"
    assert view.basemap_rect() is None, "an invisible map still claims a place on screen"

    # Clamped, not trusted: a slider off the end must not go negative or over.
    view.basemap_opacity = 7.0
    assert view.basemap_opacity == 1.0


def test_switching_the_basemap_off_draws_describes_and_reports_nothing(qt_app):
    """Off is off. Not fainter, not still in the legend, not still under the
    pointer — a layer that is "off" and still answers questions is a layer
    whose state the operator cannot trust."""
    view = _shown_map(_asset(), legend=True)
    where = _cell_screen(view, 5, 5)
    assert any(row.startswith("map:") for row in view._legend_captions())
    assert view.basemap_text_at(where) is not None

    view.show_basemap = False
    QApplication.processEvents()

    assert _at(_render(view), where) == PANEL_RGB
    assert not any(row.startswith("map:") for row in view._legend_captions())
    assert view.basemap_text_at(where) is None
    assert view.basemap_rect() is None
    assert view.basemap is not None, "off should not have thrown the asset away"


def test_the_basemap_is_georeferenced_to_its_grid(qt_app):
    """One valid cell, and it is drawn exactly where the grid says it is.

    Checked two ways: the rectangle the view draws into puts the cell's centre
    within a pixel of `_to_screen(_to_local(grid.cell_centre(...)))`, and the
    pixel at that point is ground while the pixels a cell and a half away in
    every direction are panel. The second check is the one that catches a
    mirrored raster: a south-up drawing of row 7 lands at row 12.
    """
    grid = _grid()
    lone = np.zeros(grid.shape, dtype=bool)
    lone[7, 4] = True
    view = _shown_map(_asset(grid, lone))
    rect = view.basemap_rect()
    scale = view._scale()

    for row, column in ((0, 0), (7, 4), (19, 19), (3, 15)):
        drawn = QPointF(
            rect.left() + (column + 0.5) * grid.cell_size_m * scale,
            rect.top() + (row + 0.5) * grid.cell_size_m * scale,
        )
        said = _cell_screen(view, row, column)
        off = math.hypot(drawn.x() - said.x(), drawn.y() - said.y())
        assert off <= 1.0, f"cell ({row}, {column}) is drawn {off:.2f} px from where it is"

    image = _render(view)
    centre = _cell_screen(view, 7, 4)
    assert _close(_at(image, centre), _composite(GROUND_RGB, view.basemap_opacity))
    step = 1.5 * grid.cell_size_m * scale
    for dx, dy in ((step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step)):
        beside = QPointF(centre.x() + dx, centre.y() + dy)
        assert _at(image, beside) == PANEL_RGB, f"ground leaked to {(dx, dy)} of the only cell"


def test_the_grid_is_drawn_over_the_basemap(qt_app):
    """The crosshair through the camera stays visible across covered ground.

    Sampled on the horizontal grid line where it crosses the valid block, and
    compared with the same colour of ground a few cells north of it. Drawn
    under the map at 85 % opacity the line would show through at 15 % — a few
    units of green — so the bar is a large drop, not any drop. Measured: 95
    on the line against 139 off it, with the line antialiased over two rows.
    """
    view = _shown_map(_asset())
    image = _render(view)
    camera = view._to_screen(*view._to_local(SITE_POSE.position))
    x = _cell_screen(view, 5, 5).x()

    off_line = _at(image, _cell_screen(view, 5, 5))[1]
    on_line = min(
        _at(image, QPointF(x, int(camera.y()) + dy))[1] for dy in (-1, 0, 1)
    )
    print(f"green: {on_line} on the grid line, {off_line} off it")

    assert on_line < 0.75 * off_line, "the grid line is not drawn over the basemap"


# ------------------------------------------------------------------- legend


def test_the_legend_says_where_the_map_came_from_and_stays_narrow(qt_app):
    view = _shown_map(_asset(cameras=("gate", "yard")), legend=True)

    captions = view._legend_captions()
    print(captions, view.legend_rect())
    assert any(row.startswith("map: built ") for row in captions)
    assert "26% of grid" in captions, "104 of 400 cells are ground"
    assert "from gate, yard" in captions

    legend = view.legend_rect()
    assert legend.width() <= view.width() * 0.45, f"the legend is {legend.width():.0f} px wide"
    assert not legend.intersects(view.scale_bar_rect())
    assert legend.top() >= 0 and legend.bottom() <= view.height()

    # Nine cameras with long names: elided, not stretched across the map.
    many = tuple(f"loading-bay-camera-{index}" for index in range(9))
    view.set_basemap(_asset(cameras=many))
    captions = view._legend_captions()
    cameras_row = next(row for row in captions if row.startswith("from "))
    print(captions)
    assert cameras_row.endswith("…"), "a nine-camera list was not elided"
    assert view.legend_rect().width() <= view.width() * 0.45

    view.set_basemap(None)
    assert not any(row.startswith("map:") for row in view._legend_captions())


# ---------------------------------------------------------------- staleness


def test_stale_ground_is_faded_and_hatched(qt_app):
    """Ground older than a day is still ground — it is drawn — but visibly
    not today's: fainter, with a diagonal hatch cut through it."""
    fresh = _shown_map(_asset(age=0.0))
    stale = _shown_map(_asset(age=BASEMAP_STALE_AFTER_SECONDS + 3600.0))

    # (5, 6) is off the hatch diagonals, (5, 7) is on one: 11 and 12 mod 4.
    plain, hatched = _cell_screen(fresh, 5, 6), _cell_screen(fresh, 5, 7)
    fresh_image, stale_image = _render(fresh), _render(stale)
    fresh_plain, fresh_hatched = _at(fresh_image, plain), _at(fresh_image, hatched)
    stale_plain, stale_hatched = _at(stale_image, plain), _at(stale_image, hatched)
    print(f"fresh {fresh_plain} / {fresh_hatched}; stale {stale_plain} / {stale_hatched}")

    assert _close(fresh_plain, fresh_hatched), "fresh ground has a hatch in it"
    assert stale_plain != PANEL_RGB, "stale ground vanished — it is still ground"
    assert PANEL_RGB[1] < stale_plain[1] < fresh_plain[1] - 30, "stale ground is not faded"
    assert stale_hatched[1] < stale_plain[1] - 20, "stale ground carries no hatch"


def test_staleness_is_judged_against_the_clock_and_revisited(qt_app):
    """An asset built yesterday with every cell "0 s old" is a day old.

    And a map that is fresh when the console opens does not stay fresh
    forever: once the clock passes the moment its oldest cell turns a day
    old, the next repaint shows it stale without the asset being touched.
    """
    t0 = 1_800_000_000.0
    where = None

    def frozen(view, *, built_ago: float, age: float = 0.0, later: float = 0.0):
        nonlocal where
        view.clock = lambda: t0 + later
        view.set_basemap(_asset(age=age, built_at=t0 - built_ago))
        view.show_legend = False
        view.show()
        QApplication.processEvents()
        where = where or _cell_screen(view, 5, 6)
        return _at(_render(view), where)[1]

    def bare_view():
        view = MapView()
        view.resize(600, 600)
        view.set_cameras({"gate": SITE_POSE})
        return view

    fresh_green = frozen(bare_view(), built_ago=60.0)
    old_build = frozen(bare_view(), built_ago=BASEMAP_STALE_AFTER_SECONDS + 3600.0)
    print(f"green: fresh {fresh_green}, built a day ago {old_build}")
    assert old_build < fresh_green - 30, "a day-old build is drawn as fresh"

    # Built 90 s short of a day ago: fresh now, stale two minutes from now.
    view = bare_view()
    now_green = frozen(view, built_ago=BASEMAP_STALE_AFTER_SECONDS - 90.0)
    assert now_green == fresh_green
    view.clock = lambda: t0 + 120.0
    later_green = _at(_render(view), where)[1]
    print(f"green: {now_green} now, {later_green} two minutes on")
    assert later_green == old_build, "the map did not go stale when the clock said so"


def test_the_rendering_is_cached_until_the_asset_changes(qt_app):
    """Repaints, an opacity change and the same asset again all reuse the
    image; only a different asset rebuilds it. The console re-sends what it
    holds on every refresh, and a rebuild per refresh is a rebuild per second."""
    asset = _asset()
    view = _shown_map(asset)
    image = view._basemap_image
    assert image is not None

    _render(view)
    view.basemap_opacity = 0.5
    _render(view)
    assert view._basemap_image is image, "a repaint or an opacity change rebuilt the image"

    view.set_basemap(asset)
    assert view._basemap_image is image, "the same asset again rebuilt the image"

    view.set_basemap(_asset())
    assert view._basemap_image is not image


# -------------------------------------------------------------------- hover


def test_hovering_bare_ground_reports_what_the_basemap_knows(qt_app):
    view = _shown_map(_asset(cameras=("gate", "yard"), source=1, sigma=0.8, age=300.0))

    _move(view, _cell_screen(view, 5, 5))
    assert view.hovered is None, "nothing of the console's own is there"
    assert view.toolTip() == "ground from yard, ±0.8 m, sampled 5 min ago"

    # Off the ground: nothing to say, and the last cell's text does not linger.
    _move(view, _cell_screen(view, 5, 14))
    assert view.toolTip() == ""
    assert view.basemap_text_at(_cell_screen(view, 5, 14)) is None


def test_a_thing_on_the_ground_outranks_the_ground_under_it(qt_app):
    """A track standing on a covered cell gets the tooltip; the ground does not."""
    view = _shown_map(_asset())
    point = view.basemap.grid.cell_centre(5, 5)
    track = Track(
        id=3, class_id=0, bbox=BoundingBox(0.4, 0.3, 0.2, 0.4), confidence=0.9, hits=12,
        first_seen_millis=0, last_seen_millis=1000,
        position=PositionEstimate(point=point, radius_meters=1.0, source="GROUND_PROJECTION"),
        speed_mps=1.2, heading_degrees=90.0,
    )
    view.set_tracks((track,), "gate")

    _move(view, _cell_screen(view, 5, 5))
    assert view.hovered is not None and view.hovered.kind == "track"
    assert "#3 on gate" in view.toolTip()
    assert "ground from" not in view.toolTip()

    # And stepping off the track onto bare ground hands the tooltip back.
    _move(view, _cell_screen(view, 5, 3))
    assert view.toolTip().startswith("ground from gate")


# ------------------------------------------------------------------ fitting


def test_fitting_the_view_includes_the_ground_the_site_has_drawn(qt_app):
    """A basemap reaching 300 m east of the footprint is inside the fitted view."""
    wide = _grid(columns=300)
    asset = _asset(wide, np.ones(wide.shape, dtype=bool))

    def corners_inside(view) -> list[bool]:
        return [
            view.rect().contains(_cell_screen(view, row, column).toPoint())
            for row in (0, 19) for column in (0, 299)
        ]

    # Set after the cameras: not refitted on its own (see `set_basemap`), and
    # then fitted on request — the double-click path.
    view = _shown_map(asset)
    assert not all(corners_inside(view)), "the far edge was already on screen; the test is moot"
    view._fit_view()
    assert all(corners_inside(view)), "fitting left the basemap off the edge"

    # Set before the cameras, as at start-up: the fit that placing them runs
    # already frames it.
    early = MapView()
    early.resize(600, 600)
    early.set_basemap(asset)
    early.set_cameras({"gate": SITE_POSE})
    assert all(corners_inside(early))

    # Switched off, it is not fitted to either — off is off.
    early.show_basemap = False
    early._fit_view()
    assert not all(corners_inside(early))


# ------------------------------------------------------------------ refusals


def test_a_malformed_asset_is_refused_before_it_is_stored(qt_app):
    """Arrays that do not match their grid never reach the paint path, where an
    IndexError would keep the widget alive. Built with the stand-in on purpose:
    the view has to refuse it whether or not the engine's dataclass would."""
    view = _shown_map()
    grid = _grid()
    wrong = _asset(grid, cls=_StandInAsset)
    torn = _StandInAsset(
        grid=grid, colour=np.zeros((20, 40, 3), dtype=np.uint8), valid=wrong.valid,
        age_seconds=wrong.age_seconds, sigma_m=wrong.sigma_m, source=wrong.source,
        cameras=wrong.cameras, poses=wrong.poses, built_at_millis=wrong.built_at_millis,
        frames_used=wrong.frames_used, fingerprint=wrong.fingerprint,
    )

    with pytest.raises(ValueError, match="20×40×3"):
        view.set_basemap(torn)
    assert view.basemap is None
    assert view.basemap_rect() is None
    _render(view)  # and the view still paints


def test_the_basemap_waits_for_a_placed_camera(qt_app):
    """With no camera the view has no origin, so there is nowhere to put the
    map — it is held, drawn as nothing, and appears once a camera is placed."""
    view = MapView()
    view.resize(600, 600)
    view.set_basemap(_asset())
    view.show()
    QApplication.processEvents()

    assert view.basemap is not None
    assert not view.basemap_shown()
    assert view.basemap_rect() is None
    _render(view)  # no origin, no exception

    view.set_cameras({"gate": SITE_POSE})
    assert view.basemap_shown()
    assert view.basemap_rect() is not None
