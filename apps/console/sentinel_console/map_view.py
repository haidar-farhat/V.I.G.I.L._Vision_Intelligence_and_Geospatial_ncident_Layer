"""The plan view: where things are on the ground.

**There are no map tiles here, and no request is made for any.** The requirement
is that this works with the network cable unplugged, and a map that silently
fetches tiles is a map that shows an empty grey rectangle at the one site that
has no Internet — which is most of them. What is drawn instead is a metric grid,
the camera, its real ground footprint, and the objects projected onto it. An
operator can import a local map package to sit underneath this; nothing is
fetched to produce it.

Two properties this view exists to make visible:

**The footprint is an annular sector, not a pie slice.** A camera tilted
downwards cannot see the ground at its own mast, and drawing a wedge from the
camera outwards would claim coverage that does not exist. The hole in the middle
is real.

**Uncertainty is drawn, not annotated.** Every object is a disc whose radius is
the actual 1-sigma horizontal error from the projection. Near the camera that
disc is smaller than the marker; toward the horizon it is metres across. Drawing
both at the same size would be the single most misleading thing this view could
do — it would present a guess and a measurement identically.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetricsF,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QPolygonF,
    QWheelEvent,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from sentinel.core import (
    CameraPose,
    LatLon,
    Track,
    bearing_degrees,
    destination_point,
    field_of_view,
    haversine_distance,
)

from . import theme
from .selection import Selection


#: What the next click on the map will do. A click that sometimes pans,
#: sometimes moves a camera and sometimes drops a zone corner is the single most
#: dangerous ambiguity in the console: every one of those is destructive in a
#: different way and none of them is undoable.
MODE_SELECT = "select"
MODE_DRAW = "draw"
MODE_PLACE = "place"
MODE_MEASURE = "measure"


class MapView(QWidget):
    """A north-up plan view in metres, centred on the camera."""

    #: A point the operator clicked while the view was asked to pick one, as a
    #: `LatLon`. The view knows nothing about what it is for.
    picked = Signal(object)
    #: A closed outline the operator drew: a list of `LatLon`, three or more.
    drawn = Signal(object)
    #: ``(zone_id, ring)`` once the operator finished reshaping a zone.
    edited = Signal(object, object)
    #: The id of a zone the operator clicked.
    zone_clicked = Signal(str)
    #: A `Selection` the operator clicked, or ``None`` for empty ground.
    selected = Signal(object)
    #: The ground point under the pointer, or ``None`` when it leaves the view.
    ground_moved = Signal(object)
    #: The mode changed — including because a gesture finished on its own.
    mode_changed = Signal(str)

    #: How close, in pixels, a click must be to a vertex to grab it, and to an
    #: edge to split it. Generous: a cross-hair on a 4K panel is small.
    HANDLE_PIXELS = 9.0
    EDGE_PIXELS = 6.0

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        #: While set, the next left click is a choice of ground point rather
        #: than the start of a drag. The text is what is drawn across the top.
        self._pick_prompt: str | None = None
        #: Vertices of an outline being drawn, in local metres. `None` when not
        #: drawing; an empty list when drawing has begun and nothing is placed.
        self._draw_points: list[tuple[float, float]] | None = None
        self._hover: QPointF | None = None
        #: The outline being reshaped: its zone id, vertices in local metres,
        #: the original ring (to tell a no-op from a change), and drag state.
        self._edit: dict | None = None
        self._selected_zone: str | None = None
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(280, 280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # A site has cameras, plural. One is the common case and keeps its own
        # convenience accessor, but the view is built around the general one:
        # the whole point of a plan view is seeing where coverage overlaps and
        # where it does not, and that needs more than one camera on it.
        self._cameras: dict[str, CameraPose] = {}
        self._footprints: dict[str, list[LatLon]] = {}
        #: Per camera, its position-error bands, tightest first (see
        #: `coverage.sigma_bands`). Set by the owner; never computed here.
        self._bands: dict[str, tuple] = {}
        #: The last live coverage report and the outline it was computed for.
        self._report_cache: tuple | None = None
        self.show_legend = True
        #: What is selected across the whole console, and what the pointer is
        #: over. Hover is a separate, weaker thing: it follows the mouse and is
        #: forgotten, where a selection persists until something replaces it.
        self._selection: Selection | None = None
        self._hover: Selection | None = None
        #: Measuring is read-only, so it is allowed in Monitor mode. The flag
        #: is separate from the points because "measuring, nothing placed yet"
        #: is a real state and an empty list is falsy — inferring the mode from
        #: the list ended measure mode the instant it began.
        self._measuring = False
        self._measure: list = []
        self._measure_to: QPointF | None = None
        # Always on, so hovering reports without a button held. It was
        # previously switched on only while drawing.
        self.setMouseTracking(True)
        self._tracks: tuple[Track, ...] = ()
        # Trails are keyed by camera and track, because a track id is only
        # unique within one camera. Merging them would draw one path jumping
        # between two different people.
        self._trails: dict[tuple[str, int], list[tuple[float, float]]] = {}
        self._origin: LatLon | None = None
        # East, north, and metres-per-pixel of the current view. Recomputed from
        # the content rather than fixed on the camera: a camera looking south
        # puts its whole footprint in one half of the widget, so centring on the
        # camera wastes half the view and shrinks everything in it.
        self._view_centre = (0.0, 0.0)
        self._span_meters = 60.0
        self._drag_from: QPoint | None = None
        self._zones: list = []
        #: Live tracks per camera, so one camera's update does not erase another's.
        self._live: dict[str, tuple[Track, ...]] = {}

        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setToolTip(
            "Wheel to zoom, drag to pan, double-click to fit. "
            "No tiles are fetched: this view works with the network unplugged."
        )

    # ------------------------------------------------------------- interaction

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt naming
        """Zoom about the pointer.

        A camera's claimed range and the ground its objects actually occupy are
        routinely an order of magnitude apart — a camera stating 90 m may put
        everything it sees between 8 m and 20 m. Fitting to the footprint is the
        honest default, but an operator watching that band needs to get closer
        to it, and no automatic heuristic beats being able to zoom.
        """
        steps = event.angleDelta().y() / 120.0
        if not steps:
            return

        before = self._from_screen(event.position())
        self._span_meters = max(2.0, min(5000.0, self._span_meters * (0.85 ** steps)))
        after = self._from_screen(event.position())

        # Hold the point under the pointer still, which is what makes zooming
        # feel like moving a map rather than resizing a picture.
        self._view_centre = (
            self._view_centre[0] + (before[0] - after[0]),
            self._view_centre[1] + (before[1] - after[1]),
        )
        self.update()
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        position = event.position()
        left = event.button() == Qt.MouseButton.LeftButton
        right = event.button() == Qt.MouseButton.RightButton

        if self._pick_prompt is not None:
            if left:
                point = self.point_at(position)
                self.cancel_pick()
                if point is not None:
                    self.picked.emit(point)
            elif right:
                self.cancel_pick()
            event.accept()
            return

        if self.measuring:
            if left:
                point = self.point_at(position)
                if point is not None:
                    if len(self._measure) >= 2:
                        self._measure = [point]
                    else:
                        self._measure.append(point)
            elif right:
                self.cancel_measure()
            self.update()
            event.accept()
            return

        if self._draw_points is not None:
            if left:
                self._draw_points.append(self._from_screen(position))
            elif right:
                # Undo the last vertex; with nothing placed, stop drawing.
                if self._draw_points:
                    self._draw_points.pop()
                else:
                    self.cancel_draw()
            self.update()
            event.accept()
            return

        if self._edit is not None:
            edit = self._edit
            handle = self._handle_at(position)
            if left and handle is not None:
                edit["drag"] = handle
            elif right and handle is not None:
                # Never below three: two points are a line, not an area, and the
                # engine would refuse the result anyway — better refused here,
                # where the operator can see the vertex stay put.
                if len(edit["points"]) > 3:
                    del edit["points"][handle]
            elif left and (edge := self._edge_at(position)) is not None:
                # Split the edge where the operator clicked and start dragging
                # the new vertex, so one gesture both adds and places it.
                edit["points"].insert(edge + 1, self._from_screen(position))
                edit["drag"] = edge + 1
            elif left and self._polygon_of(edit["points"]).containsPoint(
                position, Qt.FillRule.OddEvenFill
            ):
                edit["moving"] = position
            else:
                # Outside the outline: an ordinary pan, so the operator can
                # bring a far vertex into view without leaving edit mode.
                if left:
                    self._drag_from = position.toPoint()
            self.update()
            event.accept()
            return

        if left:
            hit = self.hit_test(position)
            # Emitted even when nothing was hit: clicking bare ground is how an
            # operator says "never mind", and it must clear the selection rather
            # than leave a stale highlight on four panels.
            self.selected.emit(hit)
            if hit is not None and hit.zone_id is not None:
                self.zone_clicked.emit(hit.zone_id)
            self._drag_from = position.toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        position = event.position()
        self._report_ground(position)
        if self.measuring:
            self._measure_to = position
            self.update()
            return
        if self._draw_points is None and self._edit is None and self._drag_from is None:
            # Hover only while nothing else is going on. A halo that followed
            # the pointer through a drag would compete with the thing being
            # dragged for the operator's attention.
            hover = self.hit_test(position)
            if hover != self._hover:
                self._hover = hover
                self.setToolTip(self._hover_text(hover) or "")
                self.update()
        if self._draw_points is not None:
            self._hover = position
            self.update()
            return
        if self._edit is not None:
            edit = self._edit
            if edit.get("drag") is not None:
                edit["points"][edit["drag"]] = self._from_screen(position)
                self.update()
                return
            if edit.get("moving") is not None:
                scale = self._scale()
                delta = position - edit["moving"]
                edit["moving"] = position
                edit["points"] = [
                    (e + delta.x() / scale, n - delta.y() / scale) for e, n in edit["points"]
                ]
                self.update()
                return
        if self._drag_from is None:
            return
        scale = self._scale()
        delta = position.toPoint() - self._drag_from
        self._drag_from = position.toPoint()
        self._view_centre = (
            self._view_centre[0] - delta.x() / scale,
            self._view_centre[1] + delta.y() / scale,
        )
        self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._hover is not None:
            self._hover = None
            self.update()
        self.ground_moved.emit(None)
        super().leaveEvent(event)

    def _report_ground(self, position: QPointF) -> None:
        """Tell whoever is listening where the pointer is on the ground."""
        self.ground_moved.emit(self.point_at(position))

    def _hover_text(self, hover: Selection | None) -> str | None:
        """Everything known about the thing under the pointer.

        Assembled here rather than in a tooltip string built by the caller,
        because the numbers that matter — the uncertainty and how the position
        was obtained — live on the objects this view already holds, and a
        second copy of that formatting would come to disagree with the table.
        """
        if hover is None:
            return None

        if hover.kind == "track":
            for track in self._live.get(hover.camera_id, ()):
                if track.id != hover.track_id:
                    continue
                lines = [f"#{track.id} on {hover.camera_id}"]
                if track.position is not None:
                    metres = haversine_distance(
                        self._cameras[hover.camera_id].position, track.position.point
                    ) if hover.camera_id in self._cameras else None
                    if metres is not None:
                        bearing = bearing_degrees(
                            self._cameras[hover.camera_id].position, track.position.point
                        )
                        lines.append(f"{metres:.1f} m at {bearing:.0f}° from the camera")
                    lines.append(f"±{track.position.radius_meters:.1f} m (1σ)")
                    lines.append(
                        "projected onto the ground"
                        if track.position.source == "GROUND_PROJECTION"
                        else "projection failed — shown at the camera, not located"
                    )
                else:
                    lines.append("not located: the camera is not placed")
                if track.speed_mps:
                    lines.append(f"{track.speed_mps:.1f} m/s")
                if track.heading_degrees is not None:
                    lines.append(f"heading {track.heading_degrees:.0f}°")
                return "\n".join(lines)
            return None

        if hover.kind == "camera":
            pose = self._cameras.get(hover.camera_id)
            if pose is None:
                return None
            footprint = self._footprints.get(hover.camera_id) or []
            lines = [hover.camera_id]
            if footprint:
                reach = [haversine_distance(pose.position, p) for p in footprint]
                lines.append(f"sees {min(reach):.0f} m to {max(reach):.0f} m ahead")
            lines.append(f"{pose.mount_height:.1f} m up, facing {pose.heading:.0f}°")
            return "\n".join(lines)

        if hover.kind == "zone":
            zone = next((z for z in self._zones if z.id == hover.zone_id), None)
            if zone is None:
                return None
            kind = getattr(zone.kind, "value", str(zone.kind)).lower()
            lines = [f"{zone.name} · {kind}", f"{len(zone.ring)} corners"]
            schedule = getattr(zone, "schedule", None)
            lines.append(
                f"active {schedule.describe()}" if schedule is not None else "active at all times"
            )
            return "\n".join(lines)
        return None

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._edit is not None:
            self._edit["drag"] = None
            self._edit["moving"] = None
        self._drag_from = None
        if self._draw_points is None and self._edit is None and self._pick_prompt is None:
            self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._draw_points is not None:
            # The double-click's own first press placed a vertex on top of the
            # previous one; drop it before closing.
            if len(self._draw_points) >= 2:
                last, before = self._draw_points[-1], self._draw_points[-2]
                if math.hypot(last[0] - before[0], last[1] - before[1]) * self._scale() < 3.0:
                    self._draw_points.pop()
            self.finish_draw()
            return
        if self._edit is not None:
            self.finish_edit()
            return
        self._fit_view()
        self.update()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        key = event.key()
        if key == Qt.Key.Key_Escape:
            if self._draw_points is not None:
                self.cancel_draw()
            elif self._edit is not None:
                self.cancel_edit()
            elif self._pick_prompt is not None:
                self.cancel_pick()
            elif self.measuring:
                self.cancel_measure()
            else:
                self.select_zone(None)
            self.mode_changed.emit(self.mode)
            return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self._draw_points is not None:
                self.finish_draw()
            elif self._edit is not None:
                self.finish_edit()
            return
        if key == Qt.Key.Key_Backspace and self._draw_points:
            self._draw_points.pop()
            self.update()
            return
        super().keyPressEvent(event)

    # ------------------------------------------------------- drawing outlines

    @property
    def mode(self) -> str:
        """What the next click will do, derived from what is in progress.

        Derived rather than stored, because the gestures already carry their own
        state and a second copy would be the thing that disagrees: a mode that
        said "drawing" after the drawing finished is exactly the ambiguity this
        is here to remove.
        """
        if self._draw_points is not None:
            return MODE_DRAW
        if self._pick_prompt is not None:
            return MODE_PLACE
        if self._measuring:
            return MODE_MEASURE
        return MODE_SELECT

    def begin_measure(self) -> bool:
        """Measure a distance on the ground. Read-only, so always allowed."""
        if self._origin is None:
            return False
        self.cancel_draw()
        self.cancel_edit()
        self._measuring = True
        self._measure = []
        self._measure_to = None
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setFocus()
        self.mode_changed.emit(MODE_MEASURE)
        self.update()
        return True

    def cancel_measure(self) -> None:
        if not self._measuring:
            return
        self._measuring = False
        self._measure = []
        self._measure_to = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.mode_changed.emit(self.mode)
        self.update()

    @property
    def measuring(self) -> bool:
        return self.mode == MODE_MEASURE

    def measured_metres(self) -> float | None:
        """The distance between the two placed points, once both are placed."""
        if len(self._measure) < 2:
            return None
        return haversine_distance(self._measure[0], self._measure[1])

    def to_select(self) -> None:
        """Abandon whatever gesture is running and go back to selecting."""
        self.cancel_draw()
        self.cancel_edit()
        self.cancel_pick()
        self.cancel_measure()
        self.mode_changed.emit(MODE_SELECT)

    def begin_draw(self, prompt: str = "Draw a zone") -> bool:
        """Start an outline. Each left click places a vertex; double-click or
        Enter closes it; right-click or Backspace removes the last vertex;
        Escape abandons it. Needs a placed camera, like picking."""
        if self._origin is None:
            return False
        self.cancel_edit()
        self.cancel_measure()
        self._draw_points = []
        self._draw_prompt = prompt
        self._hover = None
        self.mode_changed.emit(MODE_DRAW)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setFocus()
        self.update()
        return True

    @property
    def drawing(self) -> bool:
        return self._draw_points is not None

    @property
    def draw_points(self) -> int:
        return len(self._draw_points or ())

    def finish_draw(self) -> bool:
        """Close the outline. Returns whether one was produced.

        Fewer than three vertices is not an area, so the outline stays open and
        the operator keeps drawing — a half-drawn zone must never be created.
        """
        if self._draw_points is None or len(self._draw_points) < 3:
            return False
        ring = [self._from_local(e, n) for e, n in self._draw_points]
        self.cancel_draw()
        self.mode_changed.emit(self.mode)
        self.drawn.emit(ring)
        return True

    def cancel_draw(self) -> None:
        self._draw_points = None
        self._hover = None
        # Tracking stays on: hover inspection needs it whether or not anything
        # is being drawn, and switching it off here disabled hover for the rest
        # of the session after the first zone.
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.update()

    # ------------------------------------------------------ editing outlines

    def begin_edit(self, zone_id: str) -> bool:
        """Show a zone's vertices as handles. Drag one to move it, click an edge
        to add one, right-click a handle to remove it, drag inside to move the
        whole outline. Enter or double-click applies; Escape reverts."""
        zone = next((z for z in self._zones if z.id == zone_id), None)
        if zone is None or self._origin is None:
            return False
        self.cancel_draw()
        self._edit = {
            "zone_id": zone_id,
            "points": [self._to_local(p) for p in zone.ring],
            "original": tuple(zone.ring),
            "drag": None,
            "moving": None,
        }
        self.select_zone(zone_id)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setFocus()
        self.update()
        return True

    @property
    def editing(self) -> str | None:
        return None if self._edit is None else self._edit["zone_id"]

    def finish_edit(self) -> bool:
        """Apply the reshaped outline. Returns whether anything changed."""
        if self._edit is None:
            return False
        edit, self._edit = self._edit, None
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.update()
        ring = tuple(self._from_local(e, n) for e, n in edit["points"])
        if ring == edit["original"]:
            return False
        self.edited.emit(edit["zone_id"], list(ring))
        return True

    def cancel_edit(self) -> None:
        if self._edit is None:
            return
        self._edit = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.update()

    def edit_vertex_count(self) -> int:
        return 0 if self._edit is None else len(self._edit["points"])

    # ------------------------------------------------------------- selection

    def select_zone(self, zone_id: str | None) -> None:
        if zone_id != self._selected_zone:
            self._selected_zone = zone_id
            self.update()

    @property
    def selected_zone(self) -> str | None:
        return self._selected_zone

    def hit_test(self, position: QPointF) -> Selection | None:
        """What is under this point: a track, a camera, or a zone.

        In that order, and the order is the whole design. A track disc is a few
        pixels across and sits inside both a footprint and often a zone; a zone
        is hundreds of pixels across. Testing the largest thing first would make
        the small ones unclickable, so the test runs smallest-first and the
        operator can always reach what they can see.
        """
        if self._origin is None:
            return None

        # Tracks, newest camera first, and within a camera the closest to the
        # pointer rather than the first that happens to be within reach.
        best: tuple[float, Selection] | None = None
        for camera_id, tracks in self._live.items():
            for track in tracks:
                if track.position is None:
                    continue
                point = self._to_screen(*self._to_local(track.position.point))
                distance = math.hypot(point.x() - position.x(), point.y() - position.y())
                if distance <= self.HANDLE_PIXELS and (best is None or distance < best[0]):
                    best = (distance, Selection.track(camera_id, track.id))
        if best is not None:
            return best[1]

        for camera_id, pose in self._cameras.items():
            point = self._to_screen(*self._to_local(pose.position))
            if math.hypot(point.x() - position.x(), point.y() - position.y()) <= 12.0:
                return Selection.camera(camera_id)

        zone_id = self.zone_at(position)
        return None if zone_id is None else Selection.zone(zone_id)

    def set_selection(self, selection: Selection | None) -> None:
        """Show what is selected. Never emits — this is the way *in*.

        A setter that re-emitted would turn the bus into a loop: the map tells
        the bus, the bus tells the map, the map tells the bus.
        """
        self._selection = selection
        # The zone highlight predates the bus and is kept working, so a zone
        # selected either way is drawn the same.
        self._selected_zone = selection.zone_id if selection is not None else None
        self.update()

    @property
    def selection(self) -> Selection | None:
        return self._selection

    @property
    def hovered(self) -> Selection | None:
        return self._hover

    def zone_at(self, position: QPointF) -> str | None:
        """The topmost zone under a widget position, or ``None``."""
        if self._origin is None:
            return None
        for zone in reversed(self._zones):
            polygon = QPolygonF([self._to_screen(*self._to_local(p)) for p in zone.ring])
            if polygon.containsPoint(position, Qt.FillRule.OddEvenFill):
                return zone.id
        return None

    def vertex_screen_position(self, index: int) -> QPointF:
        """Where the edited outline's vertex `index` is on screen (for tests and
        for anything that wants to point at it)."""
        assert self._edit is not None
        return self._to_screen(*self._edit["points"][index])

    def _polygon_of(self, points) -> QPolygonF:
        return QPolygonF([self._to_screen(e, n) for e, n in points])

    def _handle_at(self, position: QPointF) -> int | None:
        if self._edit is None:
            return None
        best, best_distance = None, self.HANDLE_PIXELS
        for index, (e, n) in enumerate(self._edit["points"]):
            screen = self._to_screen(e, n)
            distance = math.hypot(screen.x() - position.x(), screen.y() - position.y())
            if distance <= best_distance:
                best, best_distance = index, distance
        return best

    def _edge_at(self, position: QPointF) -> int | None:
        """Index of the edge (from vertex i to i+1) within reach, or ``None``."""
        if self._edit is None:
            return None
        points = [self._to_screen(e, n) for e, n in self._edit["points"]]
        best, best_distance = None, self.EDGE_PIXELS
        for index in range(len(points)):
            a, b = points[index], points[(index + 1) % len(points)]
            distance = _point_to_segment(position, a, b)
            if distance <= best_distance:
                best, best_distance = index, distance
        return best

    def _from_screen(self, point: QPointF) -> tuple[float, float]:
        scale = self._scale()
        return (
            self._view_centre[0] + (point.x() - self.width() / 2) / scale,
            self._view_centre[1] - (point.y() - self.height() / 2) / scale,
        )

    # ------------------------------------------------------------------ picking

    def begin_pick(self, prompt: str) -> bool:
        """Ask for one ground point. Returns whether the view can give one.

        It cannot until a camera is placed: with no placed camera the view has
        no origin, and a click on it is a click on nothing. A zone or a camera
        cannot be put on ground the map does not yet know where it is.
        """
        if self._origin is None:
            return False
        self._pick_prompt = prompt
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.update()
        return True

    def cancel_pick(self) -> None:
        self._pick_prompt = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.update()

    @property
    def picking(self) -> bool:
        return self._pick_prompt is not None

    def point_at(self, position: QPointF) -> LatLon | None:
        """The ground point under a widget position, or ``None`` with no origin."""
        if self._origin is None:
            return None
        return self._from_local(*self._from_screen(position))

    def _from_local(self, east: float, north: float) -> LatLon:
        """The inverse of `_to_local`: metres east and north of the origin, back
        to a position on the ground."""
        assert self._origin is not None
        distance = math.hypot(east, north)
        if distance < 1e-9:
            return self._origin
        bearing = math.degrees(math.atan2(east, north)) % 360.0
        return destination_point(self._origin, bearing, distance)

    # ------------------------------------------------------------------ inputs

    @property
    def _pose(self) -> CameraPose | None:
        """The first camera, for the single-camera case."""
        return next(iter(self._cameras.values()), None)

    @property
    def _footprint(self) -> list[LatLon]:
        return next(iter(self._footprints.values()), [])

    def set_pose(self, pose: CameraPose | None, camera_id: str = "camera") -> None:
        """Show exactly one camera. Convenience for the single-camera case."""
        self.set_cameras({camera_id: pose} if pose is not None else {})

    def set_cameras(self, cameras: dict[str, CameraPose]) -> None:
        """Show every placed camera and its ground footprint."""
        self._cameras = dict(cameras)
        self._trails.clear()
        self._footprints = {
            camera_id: field_of_view(pose, arc_segments=28)
            for camera_id, pose in self._cameras.items()
        }
        # The local frame is anchored on the first camera, so every camera is
        # drawn in one consistent metric space rather than each relative to
        # itself.
        self._origin = self._pose.position if self._pose else None
        self._fit_view()
        self.update()

    def set_sigma_bands(self, bands: dict) -> None:
        """Each camera's position-error bands, from `coverage.sigma_bands`."""
        self._bands = dict(bands)
        self.update()

    def live_report(self):
        """What the cameras can rule on for the outline being drawn or reshaped,
        or ``None`` when nothing is in progress or it has fewer than three
        corners. The bands are cached per pose upstream, so only the Shapely
        intersection runs here."""
        points = None
        if self._draw_points is not None and len(self._draw_points) >= 3:
            points = self._draw_points
        elif self._edit is not None and len(self._edit["points"]) >= 3:
            points = self._edit["points"]
        if points is None or self._origin is None or not self._cameras:
            return None

        # Cached on the outline itself. This runs from the paint path, and
        # measured at 1 ms for one camera, 9.6 ms for eight and 21 ms for
        # sixteen — most of it Shapely, not the projection. Most repaints while
        # drawing move only the pointer and leave the corners alone, and those
        # now cost nothing. A reshape drag does change them every frame, and on
        # a large site that is the 21 ms; it is bounded, and it happens only
        # inside an explicit edit gesture.
        key = (
            tuple((round(east, 3), round(north, 3)) for east, north in points),
            tuple(sorted(self._cameras)),
        )
        if self._report_cache is not None and self._report_cache[0] == key:
            return self._report_cache[1]

        from sentinel.coverage import zone_report

        try:
            report = zone_report([self._from_local(e, n) for e, n in points], self._cameras)
        except Exception:  # noqa: BLE001 - a half-drawn bow tie is not an error worth a dialog
            report = None
        self._report_cache = (key, report)
        return report

    #: The legend's two lines. Swatches are drawn inline on the first.
    _LEGEND_HEADING = "1σ ≤"
    _LEGEND_SWATCHES = ("0.5", "1", "2", "5 m")
    _LEGEND_CAPTION = "unshaded: beyond 5 m"

    def legend_rect(self) -> QRectF:
        """Sized to the text it holds, and sitting clear of the scale bar.

        A fixed 236 px box clipped its own last line to "…not de" in the first
        photograph of it, and the measured replacement was wider than a narrow
        view. Anything with text in it has to measure that text *and* be told
        where the edges are.
        """
        metrics = QFontMetricsF(self._legend_font())
        swatches = sum(
            metrics.horizontalAdvance(label) + 20.0 for label in self._LEGEND_SWATCHES
        )
        width = max(
            metrics.horizontalAdvance(self._LEGEND_HEADING) + 8.0 + swatches,
            metrics.horizontalAdvance(self._LEGEND_CAPTION),
        ) + 16.0
        width = min(width, max(80.0, self.width() - 20.0))
        height = metrics.height() * 2 + 12.0
        # Above the scale bar, never over it: two overlaid captions in the same
        # corner are unreadable, and the scale bar is the one an operator needs
        # to judge a distance by eye.
        bottom = self.scale_bar_rect().top() - 8.0
        return QRectF(
            max(6.0, self.width() - width - 10.0), bottom - height, width, height
        )

    def _legend_font(self) -> QFont:
        font = QFont(self.font())
        font.setPointSize(8)
        font.setBold(False)
        return font

    def scale_bar_rect(self) -> QRectF:
        step = _nice_step(self._span_meters)
        return QRectF(8.0, self.height() - 40.0, step * self._scale() + 8.0, 36.0)

    def set_zones(self, zones) -> None:
        """Areas whose boundaries mean something.

        Drawn under the tracks and over the footprint, so an operator can see at
        a glance whether an object is inside one — and, because uncertainty
        discs are drawn too, whether the system could possibly know.
        """
        self._zones = list(zones)
        self.update()

    def set_tracks(self, tracks: tuple[Track, ...], camera_id: str = "camera") -> None:
        """Replace the tracks belonging to one camera.

        Per camera, because the console runs a pipeline per source and their
        updates arrive independently. Replacing everything from one camera's
        update would erase the others between frames.
        """
        self._live[camera_id] = tuple(tracks)
        self._tracks = tuple(t for group in self._live.values() for t in group)

        if self._origin is not None:
            for track in tracks:
                if track.position is None:
                    continue
                key = (camera_id, track.id)
                trail = self._trails.setdefault(key, [])
                point = self._to_local(track.position.point)
                if not trail or _distance(trail[-1], point) > 0.25:
                    trail.append(point)
                # Bounded: a trail is context, not a recording. Persistence holds
                # the history; this is what happened just now.
                if len(trail) > 160:
                    del trail[0]

            seen = {(camera_id, t.id) for t in tracks}
            stale = [k for k in self._trails if k[0] == camera_id and k not in seen]
            for key in stale:
                del self._trails[key]

        self.update()

    def _fit_view(self) -> None:
        """Frame everything worth seeing: the camera, its footprint, the tracks.

        Fitted once when the pose is set rather than continuously. A view that
        rescaled every frame as objects moved would make the ground itself
        appear to breathe, and an operator judging distance by eye would be
        judging against a moving ruler.
        """
        points = [
            self._to_local(point)
            for footprint in self._footprints.values()
            for point in footprint
        ]
        points += [self._to_local(pose.position) for pose in self._cameras.values()]
        # Zones too. A zone drawn behind the camera — the very case the coverage
        # warning exists for — was off the edge of the only view that could have
        # shown the operator why it will never fire.
        points += [self._to_local(point) for zone in self._zones for point in zone.ring]
        if len(points) < 2:
            self._view_centre = (0.0, 0.0)
            self._span_meters = 60.0
            return

        east = [p[0] for p in points]
        north = [p[1] for p in points]
        self._view_centre = ((min(east) + max(east)) / 2, (min(north) + max(north)) / 2)

        extent = max(max(east) - min(east), max(north) - min(north))
        # A tenth of headroom, so nothing sits on the widget boundary.
        self._span_meters = max(10.0, extent * 1.1)

    def clear(self) -> None:
        self._tracks = ()
        self._live.clear()
        self._trails.clear()
        self.update()

    # -------------------------------------------------------------- projection

    def _to_local(self, point: LatLon) -> tuple[float, float]:
        """Metres east and north of the view origin."""
        if self._origin is None:
            return 0.0, 0.0
        distance = haversine_distance(self._origin, point)
        bearing = math.radians(bearing_degrees(self._origin, point))
        return distance * math.sin(bearing), distance * math.cos(bearing)

    def _scale(self) -> float:
        """Pixels per metre."""
        return min(self.width(), self.height()) / max(self._span_meters, 1e-6)

    def _to_screen(self, east: float, north: float) -> QPointF:
        scale = self._scale()
        return QPointF(
            self.width() / 2 + (east - self._view_centre[0]) * scale,
            # North is up, so the sign flips: screen y grows downwards.
            self.height() / 2 - (north - self._view_centre[1]) * scale,
        )

    # ----------------------------------------------------------------- drawing

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), theme.PANEL)

        self._paint_grid(painter)

        if not self._cameras:
            painter.setPen(QPen(theme.TEXT_FAINT))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "This camera has not been placed.\nTracks are found but cannot be located.",
            )
            painter.end()
            return

        self._paint_footprint(painter)
        self._paint_zones(painter)
        self._paint_trails(painter)
        self._paint_tracks(painter)
        self._paint_camera(painter)
        self._paint_outline_in_progress(painter)
        self._paint_edit_handles(painter)
        self._paint_scale_bar(painter)
        self._paint_measure(painter)
        if self.show_legend and self._bands:
            self._paint_legend(painter)
        banner = self._banner()
        if banner is not None:
            self._paint_banner(painter, banner)
        painter.end()

    def _outside_parts(self, zone) -> list[QPolygonF]:
        """Screen polygons of the zone's area outside every footprint."""
        if not self._footprints:
            return []
        try:
            from shapely.geometry import Polygon
            from shapely.ops import unary_union
        except ImportError:  # pragma: no cover - shapely is a dependency
            return []
        try:
            area = Polygon([self._to_local(p) for p in zone.ring])
            union = unary_union([
                Polygon([self._to_local(p) for p in ring]) for ring in self._footprints.values() if len(ring) >= 3
            ])
            outside = area.difference(union)
        except Exception:  # noqa: BLE001 - degenerate geometry is not worth a repaint failure
            return []
        parts = getattr(outside, "geoms", [outside])
        return [
            QPolygonF([self._to_screen(x, y) for x, y in part.exterior.coords])
            for part in parts
            if not part.is_empty and part.area > 0.05
        ]

    def _paint_measure(self, painter: QPainter) -> None:
        """The measuring line, its ends, and the distance on it."""
        if not self._measure:
            return
        start = self._to_screen(*self._to_local(self._measure[0]))
        if len(self._measure) >= 2:
            end = self._to_screen(*self._to_local(self._measure[1]))
            metres = self.measured_metres()
        elif self._measure_to is not None:
            end = self._measure_to
            to = self.point_at(end)
            metres = haversine_distance(self._measure[0], to) if to is not None else None
        else:
            end, metres = start, None

        painter.setPen(QPen(theme.SELECTION, 1.6, Qt.PenStyle.DashLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawLine(start, end)
        painter.setBrush(QBrush(theme.SELECTION))
        painter.setPen(Qt.PenStyle.NoPen)
        for point in (start, end):
            painter.drawEllipse(point, 3.0, 3.0)

        if metres is not None:
            font = QFont(painter.font())
            font.setPointSize(9)
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(QPen(theme.SELECTION))
            middle = QPointF((start.x() + end.x()) / 2, (start.y() + end.y()) / 2)
            painter.drawText(QPointF(middle.x() + 6, middle.y() - 6), f"{metres:.1f} m")

    def _paint_legend(self, painter: QPainter) -> None:
        """What the shading means: two lines, bottom-right, clear of the scale bar.

        Deliberately small. It overlays the ground, and a legend that hides the
        site it explains is worse than no legend — so the swatches sit inline on
        one row rather than stacked down the corner of the map.
        """
        rect = self.legend_rect()
        painter.setPen(QPen(theme.BORDER))
        painter.setBrush(QBrush(QColor(0, 0, 0, 170)))
        painter.drawRoundedRect(rect, 4, 4)

        font = self._legend_font()
        painter.setFont(font)
        metrics = QFontMetricsF(font)
        baseline = rect.top() + metrics.ascent() + 4.0

        painter.setPen(QPen(theme.TEXT_MUTED))
        heading = self._LEGEND_HEADING
        painter.drawText(QPointF(rect.left() + 8, baseline), heading)
        x = rect.left() + 8 + metrics.horizontalAdvance(heading) + 8.0

        # Loosest alpha first, to match the order of the labels.
        for label, alpha in zip(self._LEGEND_SWATCHES, reversed(theme.SIGMA_BANDS)):
            swatch = QColor(theme.FOOTPRINT)
            swatch.setAlpha(alpha + 60)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(swatch))
            painter.drawRect(QRectF(x, baseline - 8.0, 12.0, 9.0))
            painter.setPen(QPen(theme.TEXT_MUTED))
            painter.drawText(QPointF(x + 14.0, baseline), label)
            x += metrics.horizontalAdvance(label) + 20.0

        painter.setPen(QPen(theme.TEXT_FAINT))
        painter.drawText(
            QPointF(rect.left() + 8, baseline + metrics.height()), self._LEGEND_CAPTION
        )

    def _banner(self) -> str | None:
        if self.measuring:
            if len(self._measure) >= 2:
                metres = self.measured_metres() or 0.0
                bearing = bearing_degrees(self._measure[0], self._measure[1])
                return (
                    f"Measured {metres:.1f} m at {bearing:.0f}° — "
                    "click to start again, Esc to finish"
                )
            if self._measure:
                live = ""
                if self._measure_to is not None:
                    to = self.point_at(self._measure_to)
                    if to is not None:
                        live = (
                            f": {haversine_distance(self._measure[0], to):.1f} m at "
                            f"{bearing_degrees(self._measure[0], to):.0f}°"
                        )
                return f"Measure — click the far end{live}, Esc to cancel"
            return "Measure — click the first point, Esc to cancel"
        if self._pick_prompt is not None:
            return f"{self._pick_prompt} — click the map, right-click to cancel"
        if self._draw_points is not None:
            placed = len(self._draw_points)
            return (
                f"{getattr(self, '_draw_prompt', 'Draw a zone')}: {placed} point(s) — "
                "click to add, double-click or Enter to close, right-click to undo, Esc to abandon"
                + self._report_line()
            )
        if self._edit is not None:
            return (
                "Reshape: drag a corner, click an edge to add one, right-click a "
                "corner to remove it, drag inside to move — Enter to apply, Esc to revert"
                + self._report_line()
            )
        return None

    def _report_line(self) -> str:
        """A second banner line: what the cameras could rule on for the outline
        as it stands. Empty until there are three corners."""
        report = self.live_report()
        if report is None:
            return ""
        seen = ", ".join(report.cameras) if report.cameras else "no camera"
        return (
            f"\ncovered {report.covered_fraction:.0%} · confident {report.confident_fraction:.0%}"
            f" · {report.area_m2:.0f} m² · seen by {seen}"
        )

    def _paint_banner(self, painter: QPainter, text: str) -> None:
        """Say what the next click will do, across the top of the view."""
        font = QFont(painter.font())
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)
        lines = text.count("\n") + 1
        band = self.rect().adjusted(0, 0, 0, -(self.height() - 16 - 16 * lines))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 170)))
        painter.drawRect(band)
        painter.setPen(QPen(theme.TEXT))
        painter.drawText(band, Qt.AlignmentFlag.AlignCenter, text)

    def _paint_outline_in_progress(self, painter: QPainter) -> None:
        if self._draw_points is None:
            return
        points = [self._to_screen(e, n) for e, n in self._draw_points]
        pen = QPen(theme.TEXT, 1.6, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if len(points) >= 2:
            painter.drawPolyline(QPolygonF(points))
        if points and self._hover is not None:
            painter.drawLine(points[-1], self._hover)
            if len(points) >= 2:
                faint = QPen(theme.TEXT_FAINT, 1.0, Qt.PenStyle.DotLine)
                painter.setPen(faint)
                painter.drawLine(self._hover, points[0])
        painter.setPen(QPen(theme.TEXT, 1.0))
        painter.setBrush(QBrush(theme.PANEL))
        for point in points:
            painter.drawEllipse(point, 4.0, 4.0)

    def _paint_edit_handles(self, painter: QPainter) -> None:
        if self._edit is None:
            return
        zone = next((z for z in self._zones if z.id == self._edit["zone_id"]), None)
        colour = theme.zone_colour(zone.kind) if zone is not None else theme.TEXT
        points = [self._to_screen(e, n) for e, n in self._edit["points"]]
        painter.setPen(QPen(colour, 2.0))
        fill = QColor(colour)
        fill.setAlpha(40)
        painter.setBrush(QBrush(fill))
        painter.drawPolygon(QPolygonF(points))
        painter.setBrush(QBrush(theme.PANEL))
        half = 4.5
        for point in points:
            painter.drawRect(QRectF(point.x() - half, point.y() - half, 2 * half, 2 * half))

    def _paint_grid(self, painter: QPainter) -> None:
        """A metric grid, so distances are readable rather than implied."""
        scale = self._scale()
        step = _nice_step(self._span_meters)

        # Range rings are centred on the first camera rather than on the view,
        # because their whole purpose is reading distance from a camera. With
        # several cameras a full set each would become a moire, so only the
        # first carries them and the scale bar covers the rest.
        first = self._pose
        origin = (
            self._to_screen(*self._to_local(first.position))
            if first
            else self._to_screen(0.0, 0.0)
        )
        rings = int(self._span_meters / step) + 2
        for index in range(1, rings + 1):
            radius = index * step * scale
            painter.setPen(QPen(theme.GRID_MAJOR if index % 5 == 0 else theme.GRID, 1))
            painter.drawEllipse(origin, radius, radius)

        painter.setPen(QPen(theme.GRID, 1))
        painter.drawLine(0, int(origin.y()), self.width(), int(origin.y()))
        painter.drawLine(int(origin.x()), 0, int(origin.x()), self.height())

        font = QFont(painter.font())
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QPen(theme.TEXT_FAINT))
        painter.drawText(int(origin.x()) + 4, 14, "N")

        # Label a couple of rings, so the rings are a measurement rather than
        # decoration.
        painter.setPen(QPen(theme.TEXT_FAINT))
        for index in (1, 2, 5, 10):
            radius = index * step * scale
            if radius > min(self.width(), self.height()):
                break
            painter.drawText(QPointF(origin.x() + 3, origin.y() - radius - 2), f"{index * step:g}")

    def _paint_footprint(self, painter: QPainter) -> None:
        """Every camera's ground coverage.

        Drawn with the same translucent fill, so where two footprints overlap
        the ground is visibly brighter. That overlap is not decoration: it is
        where a hand-off between cameras can happen, and so where an operator
        should expect one object rather than two.
        """
        painter.setPen(QPen(theme.FOOTPRINT_EDGE, 1.5))
        painter.setBrush(QBrush(theme.FOOTPRINT))

        for footprint in self._footprints.values():
            if len(footprint) < 3:
                continue
            painter.drawPolygon(
                QPolygonF([self._to_screen(*self._to_local(p)) for p in footprint])
            )

        # The bands: how well a position inside the footprint is actually known.
        # Widest first so each tighter band paints over the looser one, and the
        # far half of a long footprint stays as pale as the footprint itself —
        # "beyond 5 m" is the honest reading there, and the legend says so.
        painter.setPen(Qt.PenStyle.NoPen)
        for bands in self._bands.values():
            for index, band in enumerate(reversed(tuple(bands))):
                alpha = theme.SIGMA_BANDS[min(index, len(theme.SIGMA_BANDS) - 1)]
                colour = QColor(theme.FOOTPRINT)
                colour.setAlpha(alpha)
                painter.setBrush(QBrush(colour))
                ring = getattr(band, "ring", band)
                if len(ring) < 3:
                    continue
                painter.drawPolygon(
                    QPolygonF([self._to_screen(*self._to_local(p)) for p in ring])
                )

    def _paint_zones(self, painter: QPainter) -> None:
        font = QFont(painter.font())
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)

        for zone in self._zones:
            polygon = QPolygonF([self._to_screen(*self._to_local(p)) for p in zone.ring])
            # Coloured by kind. Every zone used to be red, which made an
            # exclusion zone — "ignore this" — look like a restricted area.
            colour = theme.zone_colour(zone.kind)
            edge = QColor(colour)
            edge.setAlpha(150)
            fill = QColor(colour)
            fill.setAlpha(26)
            selected = zone.id == self._selected_zone
            if self._edit is not None and zone.id == self._edit["zone_id"]:
                # The handles draw the live outline; the stored one would only
                # confuse, so it is skipped while being reshaped.
                continue
            painter.setPen(QPen(edge, 3.0 if selected else 1.6, Qt.PenStyle.SolidLine if selected else Qt.PenStyle.DashLine))
            painter.setBrush(QBrush(fill))
            painter.drawPolygon(polygon)
            if selected:
                # The part of the selected zone no camera can see, hatched: a
                # zone half outside every footprint is half a zone, and an
                # operator drawing one should see which half.
                for outside in self._outside_parts(zone):
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QBrush(theme.OUTSIDE_HATCH, Qt.BrushStyle.BDiagPattern))
                    painter.drawPolygon(outside)
                painter.setPen(QPen(edge, 3.0))

            painter.setPen(QPen(edge))
            centroid = polygon.boundingRect().center()
            kind = getattr(zone.kind, "value", str(zone.kind)).lower()
            # Centred on the zone, not started at its centre: a label that runs
            # off to the right reads as belonging to whatever is beside it.
            painter.drawText(
                QRectF(centroid.x() - 200, centroid.y() - 10, 400, 20),
                Qt.AlignmentFlag.AlignCenter,
                f"{zone.name} · {kind}",
            )

    def _paint_camera(self, painter: QPainter) -> None:
        font = QFont(painter.font())
        font.setPointSize(8)
        painter.setFont(font)

        for camera_id, pose in self._cameras.items():
            centre = self._to_screen(*self._to_local(pose.position))
            chosen = self._selection is not None and self._selection.kind == "camera" \
                and self._selection.camera_id == camera_id
            hovered = self._hover is not None and self._hover.kind == "camera" \
                and self._hover.camera_id == camera_id
            if chosen or hovered:
                painter.setPen(QPen(theme.SELECTION, 2 if chosen else 1))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(centre, 13, 13)

            painter.setPen(QPen(theme.CAMERA, 2))
            painter.setBrush(QBrush(theme.PANEL))
            painter.drawEllipse(centre, 5, 5)

            # A short stalk showing where it is pointed.
            heading = math.radians(pose.heading)
            painter.drawLine(
                centre,
                QPointF(
                    centre.x() + math.sin(heading) * 16,
                    centre.y() - math.cos(heading) * 16,
                ),
            )

            painter.setPen(QPen(theme.TEXT_MUTED))
            painter.drawText(QPointF(centre.x() + 8, centre.y() + 12), camera_id)

    def _paint_trails(self, painter: QPainter) -> None:
        painter.setBrush(Qt.BrushStyle.NoBrush)
        pen = QPen(theme.TRACK.darker(160), 1.5)
        painter.setPen(pen)

        for trail in self._trails.values():
            if len(trail) < 2:
                continue
            painter.drawPolyline(QPolygonF([self._to_screen(*p) for p in trail]))

    def _paint_tracks(self, painter: QPainter) -> None:
        scale = self._scale()
        font = QFont(painter.font())
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)

        # Per camera, because a track id is only unique within one and the
        # selection is keyed on both halves. Iterating the flattened tuple meant
        # searching every camera's list to find out where each track came from.
        for camera_id, tracks in self._live.items():
            for track in tracks:
                if track.position is None:
                    continue

                point = self._to_screen(*self._to_local(track.position.point))

                # The uncertainty disc first, so the marker sits on top of it.
                radius = max(2.0, track.position.radius_meters * scale)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(theme.UNCERTAINTY))
                painter.drawEllipse(point, radius, radius)

                # How the position was obtained, drawn differently. A fallback is
                # not a position: the projection failed and all the system can say
                # is "something, at this camera". Drawn as a filled dot beside a
                # real one it would be a claim the geometry never made.
                fallback = track.position.source != "GROUND_PROJECTION"
                if self._selection is not None and self._selection.is_track(
                    camera_id, track.id
                ):
                    painter.setPen(QPen(theme.SELECTION, 2))
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawEllipse(point, 11, 11)
                elif self._hover is not None and self._hover.is_track(
                    camera_id, track.id
                ):
                    painter.setPen(QPen(theme.SELECTION.lighter(130), 1))
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawEllipse(point, 9, 9)

                if fallback:
                    painter.setPen(QPen(theme.TRACK, 1.5, Qt.PenStyle.DashLine))
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawEllipse(point, 6, 6)
                else:
                    painter.setPen(QPen(theme.TRACK, 2))
                    painter.setBrush(QBrush(theme.TRACK.darker(220)))
                    painter.drawEllipse(point, 4, 4)

                # Heading, only when there is one. A stationary object gets no arrow,
                # because an arrow drawn from jitter is an invented direction.
                if track.heading_degrees is not None and track.speed_mps:
                    angle = math.radians(track.heading_degrees)
                    length = 10 + min(20.0, track.speed_mps * 6)
                    painter.setPen(QPen(theme.TRACK, 2))
                    painter.drawLine(
                        point,
                        QPointF(
                            point.x() + math.sin(angle) * length,
                            point.y() - math.cos(angle) * length,
                        ),
                    )

                painter.setPen(QPen(theme.TEXT))
                painter.drawText(QPointF(point.x() + 7, point.y() - 6), f"#{track.id}")

    def _paint_scale_bar(self, painter: QPainter) -> None:
        step = _nice_step(self._span_meters)
        length = step * self._scale()

        left = 12.0
        bottom = self.height() - 16.0

        painter.setPen(QPen(theme.TEXT_MUTED, 2))
        painter.drawLine(QPointF(left, bottom), QPointF(left + length, bottom))
        painter.drawLine(QPointF(left, bottom - 4), QPointF(left, bottom + 4))
        painter.drawLine(QPointF(left + length, bottom - 4), QPointF(left + length, bottom + 4))

        font = QFont(painter.font())
        font.setPointSize(8)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QPen(theme.TEXT_MUTED))
        painter.drawText(QPointF(left, bottom - 8), f"{step:g} m")

        painter.setPen(QPen(theme.TEXT_FAINT))
        painter.drawText(
            QPointF(left, bottom + 14), "wheel: zoom   drag: pan   double-click: fit"
        )


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _nice_step(span: float) -> float:
    """A round grid interval for this span: 1, 2, 5, 10, 20, 50 ..."""
    raw = span / 4.0
    magnitude = 10.0 ** math.floor(math.log10(max(raw, 1e-6)))
    for multiple in (1.0, 2.0, 5.0):
        if raw <= multiple * magnitude:
            return multiple * magnitude
    return 10.0 * magnitude


def _point_to_segment(point: QPointF, a: QPointF, b: QPointF) -> float:
    """Pixel distance from a point to the segment a-b."""
    ax, ay, bx, by = a.x(), a.y(), b.x(), b.y()
    dx, dy = bx - ax, by - ay
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-12:
        return math.hypot(point.x() - ax, point.y() - ay)
    t = max(0.0, min(1.0, ((point.x() - ax) * dx + (point.y() - ay) * dy) / length_squared))
    return math.hypot(point.x() - (ax + t * dx), point.y() - (ay + t * dy))
