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

**A footprint's far edge is one of two different facts.** It is either the
range the operator typed — a clamp, which they can raise — or the ground
running out, which no setting will change. Drawn identically, an operator who
wants to see further raises a range that was never what stopped them. The
clamped edge is solid and the ground-limited one dashed, and the legend says
which is which.

**A camera delivering nothing covers nothing.** Its footprint is hatched rather
than filled and carries no error bands, because a filled wedge under a camera
that has stopped is a claim that ground is being watched.
"""

from __future__ import annotations

import math
from dataclasses import replace
from functools import lru_cache

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
    project_to_ground,
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

#: What stops a footprint at its far edge. `FAR_EDGE_RANGE` is the pose's
#: stated range, a number somebody chose; `FAR_EDGE_HORIZON` is the top of the
#: frame meeting the ground before that range, which is geometry and not a
#: setting. See `_far_edge_kind`.
FAR_EDGE_RANGE = "range"
FAR_EDGE_HORIZON = "horizon"

#: Metres of slack when comparing the two. The core takes `min(far, range)`, so
#: an edge within a few centimetres of the range is the range.
_FAR_EDGE_TOLERANCE_M = 0.05

#: The hatch a dark camera's ground is drawn in: the idle grey rather than the
#: footprint blue, because blue in this view means a camera is seeing.
_DARK_HATCH = QColor(theme.IDLE.red(), theme.IDLE.green(), theme.IDLE.blue(), 110)


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
    #: ``(camera_id, LatLon)`` once a camera has been dragged and let go. Emitted
    #: once, on release, and never during the drag: a camera half-way through a
    #: gesture is not where the operator is putting it, and every position that
    #: camera has ever reported is derived from where it is said to be.
    camera_moved = Signal(str, object)
    #: ``(camera_id, heading_degrees)`` once its heading handle has been let go.
    #: Separate from `camera_moved` because aiming and moving are different
    #: mistakes to make, and an operator who turned a camera has not moved it.
    camera_aimed = Signal(str, float)

    #: How close, in pixels, a click must be to a vertex to grab it, and to an
    #: edge to split it. Generous: a cross-hair on a 4K panel is small.
    HANDLE_PIXELS = 9.0
    EDGE_PIXELS = 6.0
    #: How close a click must be to a camera marker to count as a click on it.
    #: Larger than the marker: the marker is 5 px across and the mast it stands
    #: for is a real thing an operator will want to grab.
    CAMERA_PIXELS = 12.0

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        #: While set, the next left click is a choice of ground point rather
        #: than the start of a drag. The text is what is drawn across the top.
        self._pick_prompt: str | None = None
        #: Vertices of an outline being drawn, in local metres. `None` when not
        #: drawing; an empty list when drawing has begun and nothing is placed.
        self._draw_points: list[tuple[float, float]] | None = None
        #: Where the pointer is while an outline is being drawn, for the
        #: rubber band. Named apart from `_hover`, which is a `Selection`:
        #: the two shared a name, and every repaint during a draw called
        #: `.kind` on a QPointF and killed the paint.
        self._draw_cursor: QPointF | None = None
        #: The outline being reshaped: its zone id, vertices in local metres,
        #: the original ring (to tell a no-op from a change), and drag state.
        self._edit: dict | None = None
        self._selected_zone: str | None = None
        #: Whether a camera may be dragged on the map. False until the console
        #: says otherwise: moving a camera is a configuration change and the
        #: console has a lock for those. See `set_editable`.
        self._editable = False
        #: The camera being dragged: the pose it had, which of position and
        #: heading the gesture changes, and the bands taken down for it. Nothing
        #: leaves this dictionary until the button comes up.
        self._camera_drag: dict | None = None
        #: Cameras that are placed but delivering nothing. Their ground is
        #: hatched and carries no bands. See `set_dark_cameras`.
        self._dark: set[str] = set()
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

        if left and self._editable:
            # Before the hit test, because the handle sits out on bare ground
            # where the test would find nothing and start a pan.
            aimed = self._heading_handle_at(position)
            if aimed is not None:
                self._begin_camera_drag(aimed, "aim", position)
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
            # A press on a camera picks the camera up rather than the ground
            # under it — but only where that is allowed, and only after the
            # selection has gone out, so a click that moves nothing still
            # selects. Through `hit_test`, so a track standing on the mast is
            # still the thing nearest the pointer and still wins.
            if (
                self._editable
                and hit is not None
                and hit.kind == "camera"
                and hit.camera_id in self._cameras
            ):
                self._begin_camera_drag(hit.camera_id, "move", position)
                event.accept()
                return
            self._drag_from = position.toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        position = event.position()
        self._report_ground(position)
        if self.measuring:
            self._measure_to = position
            self.update()
            return
        if self._camera_drag is not None:
            self._drag_camera_to(position)
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
            self._draw_cursor = position
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
                    projected = track.position.source == "GROUND_PROJECTION"
                    # Range and bearing only for a real projection. A fallback
                    # sits *on* the camera, so it would read "0.0 m at 0°" — a
                    # measurement, of nothing, that the operator would believe.
                    if projected and hover.camera_id in self._cameras:
                        origin = self._cameras[hover.camera_id].position
                        lines.append(
                            f"{haversine_distance(origin, track.position.point):.1f} m "
                            f"at {bearing_degrees(origin, track.position.point):.0f}° from the camera"
                        )
                    lines.append(f"±{track.position.radius_meters:.1f} m (1σ)")
                    lines.append(
                        "projected onto the ground"
                        if projected
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
            # Which of the two things stopped it there. An operator who wants to
            # see further reaches for the range setting, and on a camera the
            # ground already runs out under, raising it changes nothing at all.
            kind = self.far_edge_kind(hover.camera_id)
            if kind == FAR_EDGE_RANGE:
                lines.append(f"stopped by its {pose.range_meters:.0f} m range")
            elif kind == FAR_EDGE_HORIZON:
                lines.append(
                    f"stopped by the ground, short of its {pose.range_meters:.0f} m range"
                )
            if hover.camera_id in self._dark:
                lines.append("delivering nothing — this ground is not being watched")
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
        if self._camera_drag is not None:
            self._end_camera_drag()
            return
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
            if self._camera_drag is not None:
                # The mode never changed — dragging a camera is a Select-mode
                # gesture — so nothing is announced, the camera simply goes
                # back to where it was and the release commits nothing.
                self._revert_camera_drag()
                return
            if self._draw_points is not None:
                self.cancel_draw()
            elif self._edit is not None:
                self.cancel_edit()
            elif self._pick_prompt is not None:
                self.cancel_pick()
            elif self.measuring:
                self.cancel_measure()
            else:
                # Nothing of the map's own to abandon, so the key belongs to
                # the window, which clears the selection. Accepting it here
                # made Escape do nothing whenever the map had focus.
                self.mode_changed.emit(self.mode)
                super().keyPressEvent(event)
                return
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
        if self._edit is not None:
            # Reshaping is a drawing gesture: the next click adds or grabs a
            # corner, and reporting Select while it does that is the ambiguity
            # this property exists to remove.
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
        self.cancel_pick()
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
        self.cancel_pick()
        self._draw_points = []
        self._draw_prompt = prompt
        self._draw_cursor = None
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
        self._draw_cursor = None
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

    # -------------------------------------------------------- moving a camera

    def set_editable(self, editable: bool) -> None:
        """Allow, or forbid, dragging a camera on this view.

        Off by default, and off whenever the console is in Monitor. Where a
        camera is said to be is the input to every position it will ever
        report, so a sleeve across a touchscreen must not be able to change it
        while somebody is watching the wall. Taking the permission away
        mid-gesture reverts the gesture rather than committing it, because the
        lock coming back is not the operator saying yes.
        """
        editable = bool(editable)
        if editable == self._editable:
            return
        self._editable = editable
        if not editable:
            self._revert_camera_drag()
        self.update()

    @property
    def editable(self) -> bool:
        return self._editable

    @property
    def dragging_camera(self) -> str | None:
        """The camera being dragged right now, or ``None``.

        Exposed so the owner can tell an uncommitted pose from a stored one —
        nothing is persisted until the button comes up.
        """
        return None if self._camera_drag is None else self._camera_drag["camera_id"]

    def set_dark_cameras(self, ids) -> None:
        """The placed cameras that are delivering nothing.

        A stopped or faulted camera keeps its pose, and until this existed it
        kept the full blue wedge that goes with one: an operator reading the
        plan view saw the yard covered by a camera that had not produced a
        frame in an hour. Dark ground is hatched and carries no error bands,
        because coverage is what a camera is doing and not where it points.
        """
        dark = set(ids)
        if dark == self._dark:
            return
        self._dark = dark
        self.update()

    @property
    def dark_cameras(self) -> frozenset:
        return frozenset(self._dark)

    def far_edge_kind(self, camera_id: str) -> str | None:
        """What stops this camera's footprint: `FAR_EDGE_RANGE`,
        `FAR_EDGE_HORIZON`, or ``None`` when the core cannot say.

        Worked out from the same two facts the core builds the wedge from —
        where the top row of the frame lands on the ground, and the pose's
        stated range — rather than guessed from the drawn shape.
        """
        pose = self._cameras.get(camera_id)
        return None if pose is None else _far_edge_kind(pose)

    def far_edge_metres(self, camera_id: str) -> float | None:
        """How far the drawn footprint reaches, down the camera's axis.

        Read off the footprint the view is holding rather than recomputed, so
        the grip and the edge styling can never disagree with the polygon they
        are drawn on.
        """
        pose = self._cameras.get(camera_id)
        ring = self._footprints.get(camera_id) or []
        if pose is None or len(ring) < 3:
            return None
        return max(haversine_distance(pose.position, point) for point in ring)

    def heading_handle(self, camera_id: str) -> QPointF | None:
        """Where the heading grip sits, or ``None`` when there is none to grab.

        On the footprint's own axis, at its far edge: an operator turning a
        camera is aiming the wedge, and a grip anywhere else asks them to think
        about the mast instead of about the ground. ``None`` when the view may
        not be edited, when the camera has no footprint, or when the grip would
        land on the marker — a rotate handle inside the thing it rotates steals
        the drag that moves it.
        """
        if not self._editable:
            return None
        pose = self._cameras.get(camera_id)
        distance = self.far_edge_metres(camera_id)
        if pose is None or distance is None:
            return None
        centre = self._to_screen(*self._to_local(pose.position))
        point = self._to_screen(
            *self._to_local(destination_point(pose.position, pose.heading, distance))
        )
        reach = math.hypot(point.x() - centre.x(), point.y() - centre.y())
        if reach <= self.CAMERA_PIXELS + self.HANDLE_PIXELS:
            return None
        return point

    def _heading_handle_at(self, position: QPointF) -> str | None:
        """The camera whose heading grip is under this point, nearest first."""
        best: tuple[float, str] | None = None
        for camera_id in self._cameras:
            point = self.heading_handle(camera_id)
            if point is None:
                continue
            distance = math.hypot(point.x() - position.x(), point.y() - position.y())
            if distance <= self.HANDLE_PIXELS and (best is None or distance < best[0]):
                best = (distance, camera_id)
        return None if best is None else best[1]

    def _begin_camera_drag(self, camera_id: str, kind: str, position: QPointF) -> None:
        """Pick a camera up. Nothing is emitted and nothing is stored."""
        pose = self._cameras[camera_id]
        east, north = self._to_local(pose.position)
        cursor_east, cursor_north = self._from_screen(position)
        self._camera_drag = {
            "camera_id": camera_id,
            "kind": kind,
            "original": pose,
            # Where the pointer grabbed it, so the mast does not jump to the
            # cursor on the first pixel of movement.
            "grab": (east - cursor_east, north - cursor_north),
            # The bands come down for the duration. Recomputing them costs 1750
            # calls across the FFI per camera — measured at 7.9 ms warm, 75 ms
            # on the first — which is a slideshow at mouse-move rate. The
            # footprint alone is one call at 0.03 ms, so that stays live.
            "bands": self._bands.pop(camera_id, None),
        }
        self._hover = None
        self.setCursor(
            Qt.CursorShape.ClosedHandCursor if kind == "move" else Qt.CursorShape.CrossCursor
        )
        # So Escape reaches this view rather than the window behind it.
        self.setFocus()
        self.update()

    def _drag_camera_to(self, position: QPointF) -> None:
        """Show the camera where the pointer is now, committing nothing."""
        drag = self._camera_drag
        if drag is None:
            return
        pose = self._cameras.get(drag["camera_id"])
        if pose is None:
            return
        if drag["kind"] == "move":
            east, north = self._from_screen(position)
            moved = replace(
                pose,
                position=self._from_local(east + drag["grab"][0], north + drag["grab"][1]),
            )
        else:
            point = self._from_local(*self._from_screen(position))
            # A bearing taken from a point on top of the mast is noise, and the
            # camera would spin to whatever the rounding said. Ignore it.
            if haversine_distance(pose.position, point) < 0.5:
                return
            moved = replace(pose, heading=bearing_degrees(pose.position, point) % 360.0)
        self._show_uncommitted_pose(drag["camera_id"], moved)

    def _show_uncommitted_pose(self, camera_id: str, pose: CameraPose) -> None:
        """Draw a pose that exists nowhere but this gesture.

        The footprint is rebuilt with it, because the wedge is the thing the
        operator is dragging *for*: they are watching where the coverage lands,
        not where the marker is.
        """
        self._cameras[camera_id] = pose
        try:
            self._footprints[camera_id] = field_of_view(pose, arc_segments=28)
        except Exception:  # noqa: BLE001 - a pose mid-drag is not worth killing the gesture
            self._footprints[camera_id] = []
        self.update()

    def _end_camera_drag(self) -> None:
        """Let the camera go, and say so exactly once.

        A drag that put the camera back where it started says nothing at all:
        an operator who thought better of it half way must not leave a
        re-placement and an audit entry behind.
        """
        drag, self._camera_drag = self._camera_drag, None
        if drag is None:
            return
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        camera_id = drag["camera_id"]
        pose = self._cameras.get(camera_id)
        original = drag["original"]
        if pose is None:
            return

        if drag["kind"] == "move":
            changed = haversine_distance(original.position, pose.position) >= 0.01
        else:
            changed = abs(_angle_difference(original.heading, pose.heading)) >= 0.05

        if not changed:
            self._restore_bands(camera_id, drag)
            self.update()
            return

        # The bands stay down. They say where this camera's error crosses each
        # threshold, and the ones taken down at the start belong to the pose it
        # no longer has — put back now they would draw the confident ground
        # where the camera used to be. The owner recomputes them off this
        # signal, in practice before the next repaint; until it does, the bare
        # footprint is the honest drawing.
        if drag["kind"] == "move":
            self.camera_moved.emit(camera_id, pose.position)
        else:
            self.camera_aimed.emit(camera_id, pose.heading)
        self.update()

    def _revert_camera_drag(self) -> None:
        """Put the camera back exactly as it was, and emit nothing."""
        if self._camera_drag is None:
            return
        drag, self._camera_drag = self._camera_drag, None
        self._show_uncommitted_pose(drag["camera_id"], drag["original"])
        self._restore_bands(drag["camera_id"], drag)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.update()

    def _restore_bands(self, camera_id: str, drag: dict) -> None:
        if drag.get("bands") is not None:
            self._bands[camera_id] = drag["bands"]

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
            if (
                math.hypot(point.x() - position.x(), point.y() - position.y())
                <= self.CAMERA_PIXELS
            ):
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
        # Whatever was being dragged is gone: the owner has just said where the
        # cameras are, and a half-finished gesture holding a pose from before
        # that would commit it on release over the top of the new one.
        self._camera_drag = None
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

    #: The legend's rows. Swatches are drawn inline on the sigma heading.
    _LEGEND_HEADING = "1σ ≤"
    _LEGEND_SWATCHES = ("0.5", "1", "2", "5 m")
    _LEGEND_CAPTION = "unshaded: beyond 5 m"
    #: The two far edges, one row each. The distinction is the whole reason
    #: these lines exist: one of them is a number in a dialog and the other is
    #: the site. Kept to two short rows because the measured one-line version
    #: came out 451 px wide on a 600 px view and the legend sits on the ground.
    _LEGEND_FAR_EDGE = ("solid edge: range", "dashed edge: horizon")
    _LEGEND_DARK = "hatched: no frames"

    def _legend_captions(self) -> tuple[str, ...]:
        """The caption rows, and only the ones describing something on screen.

        A legend line for a case the view is not drawing is one more thing to
        read at three in the morning, and the legend sits on the ground it
        explains.
        """
        captions = [self._LEGEND_CAPTION] if self._bands else []
        if any(len(ring) >= 3 for ring in self._footprints.values()):
            captions.extend(self._LEGEND_FAR_EDGE)
        if any(camera_id in self._footprints for camera_id in self._dark):
            captions.append(self._LEGEND_DARK)
        return tuple(captions) or (self._LEGEND_CAPTION,)

    def legend_rect(self) -> QRectF:
        """Sized to the text it holds, and sitting clear of the scale bar.

        A fixed 236 px box clipped its own last line to "…not de" in the first
        photograph of it, and the measured replacement was wider than a narrow
        view. Anything with text in it has to measure that text *and* be told
        where the edges are.
        """
        metrics = QFontMetricsF(self._legend_font())
        captions = self._legend_captions()
        rows = len(captions)
        width = max(metrics.horizontalAdvance(caption) for caption in captions)
        if self._bands:
            swatches = sum(
                metrics.horizontalAdvance(label) + 20.0 for label in self._LEGEND_SWATCHES
            )
            width = max(
                width, metrics.horizontalAdvance(self._LEGEND_HEADING) + 8.0 + swatches
            )
            rows += 1
        width = min(width + 16.0, max(80.0, self.width() - 20.0))
        height = metrics.height() * rows + 12.0
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

    def forget_camera(self, camera_id: str) -> None:
        """Drop everything belonging to a camera that has gone.

        Its tracks stayed in `_live` after the camera was removed: still
        drawn, still hit-testable, and still described by the hover text as
        located — by a camera the node no longer has.
        """
        self._live.pop(camera_id, None)
        self._dark.discard(camera_id)
        if self._camera_drag is not None and self._camera_drag["camera_id"] == camera_id:
            # Not reverted — there is nothing left to revert it onto.
            self._camera_drag = None
        self._tracks = tuple(t for group in self._live.values() for t in group)
        for key in [k for k in self._trails if k[0] == camera_id]:
            del self._trails[key]
        if self._selection is not None and self._selection.camera_id == camera_id:
            self._selection = None
        if self._hover is not None and self._hover.camera_id == camera_id:
            self._hover = None
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
        try:
            self._paint(painter)
        finally:
            # An exception in here would otherwise leave an active QPainter
            # attached to the widget — and Qt keeps the traceback, which keeps
            # the widget, which is the leak this codebase already knows well.
            painter.end()

    def _paint(self, painter: QPainter) -> None:
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
        # Footprints without bands still need the legend: the far edge means
        # two different things whether or not the error has been computed.
        if self.show_legend and (self._bands or self._footprints):
            self._paint_legend(painter)
        banner = self._banner()
        if banner is not None:
            self._paint_banner(painter, banner)

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

        # The swatches explain the shading, so they are drawn only when there is
        # shading. Without bands the footprint is one flat blue, and a row
        # reading "unshaded: beyond 5 m" over it would be a claim about the
        # error out there that nobody has computed.
        if self._bands:
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
            baseline += metrics.height()

        painter.setPen(QPen(theme.TEXT_FAINT))
        for caption in self._legend_captions():
            painter.drawText(QPointF(rect.left() + 8, baseline), caption)
            baseline += metrics.height()

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
        if points and self._draw_cursor is not None:
            painter.drawLine(points[-1], self._draw_cursor)
            if len(points) >= 2:
                faint = QPen(theme.TEXT_FAINT, 1.0, Qt.PenStyle.DotLine)
                painter.setPen(faint)
                painter.drawLine(self._draw_cursor, points[0])
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
        for camera_id, footprint in self._footprints.items():
            if len(footprint) < 3:
                continue
            local = [self._to_local(point) for point in footprint]
            points = [self._to_screen(east, north) for east, north in local]
            dark = camera_id in self._dark
            painter.setPen(Qt.PenStyle.NoPen)
            # Hatched, never filled. A fill is how this view says "seen", and a
            # camera producing no frames is not seeing this ground — it is the
            # ground nobody is watching, drawn the way the uncovered part of a
            # zone is drawn.
            painter.setBrush(
                QBrush(_DARK_HATCH, Qt.BrushStyle.FDiagPattern)
                if dark
                else QBrush(theme.FOOTPRINT)
            )
            painter.drawPolygon(QPolygonF(points))
            self._paint_far_edge(painter, camera_id, local, points, dark)

        # The bands: how well a position inside the footprint is actually known.
        # Widest first so each tighter band paints over the looser one, and the
        # far half of a long footprint stays as pale as the footprint itself —
        # "beyond 5 m" is the honest reading there, and the legend says so.
        painter.setPen(Qt.PenStyle.NoPen)
        for camera_id, bands in self._bands.items():
            # A dark camera's bands say how well it *would* locate something.
            # It is locating nothing, so they are not drawn.
            if camera_id in self._dark:
                continue
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

    def _far_arc_count(self, camera_id: str, local: list) -> int | None:
        """How many of the footprint's leading vertices lie on its far edge.

        Measured off the ring the view is holding, not assumed from the segment
        count the core was asked for. If the core ever stopped emitting the far
        arc first this answers ``None``, and the outline is drawn undecorated —
        a dash across the wrong side of a footprint would say the operator's
        range clamp is the thing they cannot change.
        """
        pose = self._cameras.get(camera_id)
        if pose is None or len(local) < 3:
            return None
        origin = self._to_local(pose.position)
        # Planar, off coordinates already computed for the screen points: a
        # site is metres across, and this is a comparison rather than a
        # measurement anybody reads.
        distances = [math.hypot(e - origin[0], n - origin[1]) for e, n in local]
        furthest = max(distances)
        if furthest <= 0.0:
            return None
        floor = furthest - max(0.25, furthest * 0.002)
        count = 0
        for distance in distances:
            if distance < floor:
                break
            count += 1
        return count if 2 <= count < len(local) else None

    def _paint_far_edge(
        self, painter: QPainter, camera_id: str, local: list, points: list, dark: bool
    ) -> None:
        """The footprint's outline, with its far edge drawn for what bounds it.

        Solid where the pose's stated range clamps the view — a number somebody
        typed, and can raise — and dashed where the ground runs out first,
        which no setting will move. An operator who wants another twenty metres
        needs to know whether to change the range or the mast.
        """
        colour = QColor(theme.IDLE if dark else theme.FOOTPRINT_EDGE)
        if dark:
            colour.setAlpha(150)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        count = self._far_arc_count(camera_id, local)
        kind = self.far_edge_kind(camera_id)
        if count is None or kind is None:
            # Nothing honest to say about which edge is which, so the outline is
            # drawn in one weight and claims nothing.
            painter.setPen(QPen(colour, 1.5))
            painter.drawPolygon(QPolygonF(points))
            return

        # The near edge and the two sides first, then the far arc over them.
        painter.setPen(QPen(colour, 1.5))
        painter.drawPolyline(QPolygonF(points[count - 1:] + points[:1]))
        painter.setPen(QPen(
            colour,
            2.2,
            Qt.PenStyle.SolidLine if kind == FAR_EDGE_RANGE else Qt.PenStyle.DashLine,
        ))
        painter.drawPolyline(QPolygonF(points[:count]))

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

            # A dark camera is hollow, struck through, and in the colour the
            # console uses for "nothing is running". The same shape, so it is
            # still recognisably a camera; unmistakably not this one's evidence.
            dark = camera_id in self._dark
            painter.setPen(QPen(theme.IDLE if dark else theme.CAMERA, 2))
            painter.setBrush(Qt.BrushStyle.NoBrush if dark else QBrush(theme.PANEL))
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
            if dark:
                painter.drawLine(
                    QPointF(centre.x() - 6.0, centre.y() - 6.0),
                    QPointF(centre.x() + 6.0, centre.y() + 6.0),
                )

            self._paint_heading_handle(painter, camera_id)

            painter.setPen(QPen(theme.TEXT_MUTED))
            painter.drawText(
                QPointF(centre.x() + 8, centre.y() + 12),
                f"{camera_id} · dark" if dark else camera_id,
            )

    def _paint_heading_handle(self, painter: QPainter, camera_id: str) -> None:
        """The grip that turns a camera, out on the far edge of its own wedge.

        Drawn only while the view is editable, and that is the point: a handle
        offered in Monitor advertises a gesture the lock is going to refuse.
        """
        point = self.heading_handle(camera_id)
        if point is None:
            return
        turning = (
            self._camera_drag is not None
            and self._camera_drag["camera_id"] == camera_id
            and self._camera_drag["kind"] == "aim"
        )
        painter.setPen(QPen(theme.SELECTION, 2.0 if turning else 1.4))
        painter.setBrush(QBrush(theme.PANEL))
        painter.drawEllipse(point, 5.0, 5.0)

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


@lru_cache(maxsize=64)
def _far_edge_kind(pose: CameraPose) -> str | None:
    """Whether this pose's footprint is stopped by its range or by the ground.

    The core builds the wedge out to ``min(top of frame, range_meters)``, so the
    question is which of the two won, and it is answered by asking for the top
    row of the frame with the range clamp switched off. No answer means that row
    is at or above the horizon and only the range is holding the wedge in; an
    answer shorter than the range means the ground ran out first. ``None`` only
    when the core cannot be asked, and then nothing is claimed either way.

    Cached on the pose — frozen, so it hashes — because a repaint asks for it.
    """
    try:
        far = project_to_ground(pose, 0.5, 0.0, enforce_range=False)
    except Exception:  # noqa: BLE001 - a broken core must not take a repaint with it
        return None
    if far is None:
        return FAR_EDGE_RANGE
    if far.ground_distance_meters >= pose.range_meters - _FAR_EDGE_TOLERANCE_M:
        return FAR_EDGE_RANGE
    return FAR_EDGE_HORIZON


def _angle_difference(a: float, b: float) -> float:
    """Degrees from `a` to `b`, signed, in -180..180.

    So that 359° and 1° are two degrees apart rather than three hundred and
    fifty eight, which is what decided whether a nudged camera was re-aimed.
    """
    return (b - a + 180.0) % 360.0 - 180.0


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
