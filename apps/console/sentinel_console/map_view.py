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

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QBrush, QFont, QMouseEvent, QPainter, QPen, QPolygonF, QWheelEvent
from PySide6.QtWidgets import QSizePolicy, QWidget

from sentinel.core import (
    CameraPose,
    LatLon,
    Track,
    bearing_degrees,
    field_of_view,
    haversine_distance,
)

from . import theme


class MapView(QWidget):
    """A north-up plan view in metres, centred on the camera."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumSize(280, 280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._pose: CameraPose | None = None
        self._footprint: list[LatLon] = []
        self._tracks: tuple[Track, ...] = ()
        self._trails: dict[int, list[tuple[float, float]]] = {}
        self._footprint_local: list[tuple[float, float]] = []
        # East, north, and metres-per-pixel of the current view. Recomputed from
        # the content rather than fixed on the camera: a camera looking south
        # puts its whole footprint in one half of the widget, so centring on the
        # camera wastes half the view and shrinks everything in it.
        self._view_centre = (0.0, 0.0)
        self._span_meters = 60.0
        self._drag_from: QPoint | None = None
        self._zones: list = []

        self.setMouseTracking(False)
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
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_from = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag_from is None:
            return
        scale = self._scale()
        delta = event.position().toPoint() - self._drag_from
        self._drag_from = event.position().toPoint()
        self._view_centre = (
            self._view_centre[0] - delta.x() / scale,
            self._view_centre[1] + delta.y() / scale,
        )
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self._drag_from = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self._fit_view()
        self.update()

    def _from_screen(self, point: QPointF) -> tuple[float, float]:
        scale = self._scale()
        return (
            self._view_centre[0] + (point.x() - self.width() / 2) / scale,
            self._view_centre[1] - (point.y() - self.height() / 2) / scale,
        )

    # ------------------------------------------------------------------ inputs

    def set_pose(self, pose: CameraPose | None) -> None:
        self._pose = pose
        self._trails.clear()
        self._footprint = field_of_view(pose, arc_segments=28) if pose else []
        self._footprint_local = (
            [self._to_local(p) for p in self._footprint] if pose else []
        )
        self._fit_view()
        self.update()

    def set_zones(self, zones) -> None:
        """Areas whose boundaries mean something.

        Drawn under the tracks and over the footprint, so an operator can see at
        a glance whether an object is inside one — and, because uncertainty
        discs are drawn too, whether the system could possibly know.
        """
        self._zones = list(zones)
        self.update()

    def set_tracks(self, tracks: tuple[Track, ...]) -> None:
        self._tracks = tracks

        if self._pose is not None:
            for track in tracks:
                if track.position is None:
                    continue
                trail = self._trails.setdefault(track.id, [])
                point = self._to_local(track.position.point)
                if not trail or _distance(trail[-1], point) > 0.25:
                    trail.append(point)
                # Bounded: a trail is context, not a recording. The database
                # holds the history; this is what happened just now.
                if len(trail) > 160:
                    del trail[0]

            live = {t.id for t in tracks}
            for track_id in list(self._trails):
                if track_id not in live:
                    del self._trails[track_id]

        self.update()

    def _fit_view(self) -> None:
        """Frame everything worth seeing: the camera, its footprint, the tracks.

        Fitted once when the pose is set rather than continuously. A view that
        rescaled every frame as objects moved would make the ground itself
        appear to breathe, and an operator judging distance by eye would be
        judging against a moving ruler.
        """
        points = list(self._footprint_local) + [(0.0, 0.0)]
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
        self._trails.clear()
        self.update()

    # -------------------------------------------------------------- projection

    def _to_local(self, point: LatLon) -> tuple[float, float]:
        """Metres east and north of the camera."""
        assert self._pose is not None
        distance = haversine_distance(self._pose.position, point)
        bearing = math.radians(bearing_degrees(self._pose.position, point))
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

        if self._pose is None:
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
        self._paint_scale_bar(painter)
        painter.end()

    def _paint_grid(self, painter: QPainter) -> None:
        """A metric grid, so distances are readable rather than implied."""
        scale = self._scale()
        step = _nice_step(self._span_meters)

        # Range rings are centred on the camera, not on the view, because their
        # whole purpose is reading distance from the camera.
        origin = self._to_screen(0.0, 0.0)
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
        if len(self._footprint) < 3:
            return

        polygon = QPolygonF([self._to_screen(*self._to_local(p)) for p in self._footprint])
        painter.setPen(QPen(theme.FOOTPRINT_EDGE, 1.5))
        painter.setBrush(QBrush(theme.FOOTPRINT))
        painter.drawPolygon(polygon)

    def _paint_zones(self, painter: QPainter) -> None:
        font = QFont(painter.font())
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)

        for zone in self._zones:
            polygon = QPolygonF([self._to_screen(*self._to_local(p)) for p in zone.ring])
            painter.setPen(QPen(theme.ZONE_EDGE, 1.6, Qt.PenStyle.DashLine))
            painter.setBrush(QBrush(theme.ZONE_FILL))
            painter.drawPolygon(polygon)

            painter.setPen(QPen(theme.ZONE_EDGE))
            centroid = polygon.boundingRect().center()
            painter.drawText(centroid, zone.name)

    def _paint_camera(self, painter: QPainter) -> None:
        assert self._pose is not None
        centre = self._to_screen(0.0, 0.0)

        painter.setPen(QPen(theme.CAMERA, 2))
        painter.setBrush(QBrush(theme.PANEL))
        painter.drawEllipse(centre, 5, 5)

        # A short stalk showing where it is pointed.
        heading = math.radians(self._pose.heading)
        painter.setPen(QPen(theme.CAMERA, 2))
        painter.drawLine(
            centre,
            QPointF(centre.x() + math.sin(heading) * 16, centre.y() - math.cos(heading) * 16),
        )

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

        for track in self._tracks:
            if track.position is None:
                continue

            point = self._to_screen(*self._to_local(track.position.point))

            # The uncertainty disc first, so the marker sits on top of it.
            radius = max(2.0, track.position.radius_meters * scale)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(theme.UNCERTAINTY))
            painter.drawEllipse(point, radius, radius)

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
