"""The ground plan: where things are, drawn from the site's own geometry.

No external tiles, ever — that is the product's central promise, and a map
that needs the Internet would break it silently the first time a site was
air-gapped. What is drawn is what the system knows: camera positions, the
ground each camera can actually see, zones, and projected tracks with their
uncertainty. A scale bar says what a pixel is worth, because a plan with no
scale is a picture.
"""

from __future__ import annotations

import math
from typing import Sequence

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from ...domain.geo import CameraPose, LatLon, LocalFrame, field_of_view
from . import theme

#: Metres of padding around whatever is being shown, so nothing touches the edge.
MARGIN_METERS = 8.0


class PlanView(QWidget):
    """Cameras, coverage, zones and tracks on one local ground frame."""

    #: A click on the plan, in latitude/longitude. The window uses it to draw
    #: a zone; nothing else in the console has a use for a map click yet.
    clicked = Signal(object)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._frame: LocalFrame | None = None
        self._cameras: dict[str, CameraPose] = {}
        self._zones: list = []
        self._tracks: dict[str, tuple] = {}
        #: Per camera, what its tracks are doing together. Drawn as a link so
        #: a group reads as a group and not as three unrelated dots.
        self._relations: dict[str, tuple] = {}
        self._draft: list[LatLon] = []
        #: The site's own ground, when one has been built. See `set_ground`.
        self._ground = None
        self._ground_pixmap = None
        self._ground_key: tuple | None = None
        self._drawing = False
        self._selected: str | None = None
        self._zoom = 1.0
        self.setMinimumSize(240, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

    # -------------------------------------------------------------- inputs

    def set_cameras(self, poses: dict[str, CameraPose]) -> None:
        self._cameras = dict(poses)
        self._reframe()
        self.update()

    def set_ground(self, ground) -> None:
        """The map the cameras built, drawn under everything else.

        `vigil.service.mapping` opens by saying a plan view needs ground under
        it. It has been able to produce that ground since it was written and
        nothing drew it — correct, tested code no product path reached, which
        is this repository's recurring defect.

        Only the cells the map is confident about are drawn. A cell it cannot
        vouch for is left as background rather than shaded in, because the
        whole reason an operator looks at this view is to judge which side of
        a line somebody was on, and ground that might be a smeared wall is
        worse than no ground at all.
        """
        self._ground = ground
        self._ground_pixmap = None
        self._ground_key = None
        self.update()

    def set_zones(self, zones: Sequence) -> None:
        self._zones = list(zones)
        self._reframe()
        self.update()

    def set_tracks(self, camera_id: str, tracks: Sequence, relations: Sequence = ()) -> None:
        self._tracks[camera_id] = tuple(tracks)
        self._relations[camera_id] = tuple(relations)
        self.update()

    def clear_tracks(self) -> None:
        self._tracks.clear()
        self._relations.clear()
        self.update()

    def select(self, camera_id: str | None) -> None:
        self._selected = camera_id
        self.update()

    def begin_zone(self) -> None:
        self._drawing = True
        self._draft = []
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.update()

    def end_zone(self) -> list[LatLon]:
        ring, self._draft = list(self._draft), []
        self._drawing = False
        self.unsetCursor()
        self.update()
        return ring

    @property
    def drawing(self) -> bool:
        return self._drawing

    @property
    def draft(self) -> list[LatLon]:
        return list(self._draft)

    def zoom_by(self, factor: float) -> None:
        self._zoom = min(8.0, max(0.2, self._zoom * factor))
        self.update()

    # ------------------------------------------------------------ geometry

    def _points_of_interest(self) -> list[LatLon]:
        points: list[LatLon] = []
        for pose in self._cameras.values():
            points.append(pose.position)
            points.extend(field_of_view(pose, 8) or [])
        for zone in self._zones:
            points.extend(zone.ring)
        points.extend(self._draft)
        return points

    def _reframe(self) -> None:
        points = self._points_of_interest()
        if not points:
            self._frame = None
            return
        self._frame = LocalFrame(LatLon(sum(p.lat for p in points) / len(points),
                                        sum(p.lon for p in points) / len(points)))

    def _scale(self) -> float:
        """Pixels per metre, so everything of interest fits with a margin."""
        if self._frame is None:
            return 1.0
        points = self._points_of_interest()
        extent = 1.0
        for point in points:
            local = self._frame.to_local(point)
            extent = max(extent, abs(local.x), abs(local.y))
        extent += MARGIN_METERS
        return (min(self.width(), self.height()) / 2) / extent * self._zoom

    def _to_screen(self, point: LatLon) -> QPointF:
        assert self._frame is not None
        local = self._frame.to_local(point)
        scale = self._scale()
        # North is up: the local frame's +y is north, the screen's is down.
        return QPointF(self.width() / 2 + local.x * scale, self.height() / 2 - local.y * scale)

    def _to_lat_lon(self, x: float, y: float) -> LatLon | None:
        if self._frame is None:
            return None
        scale = self._scale()
        from ...domain.geo import Vec2

        return self._frame.to_lat_lon(Vec2((x - self.width() / 2) / scale, (self.height() / 2 - y) / scale))

    # -------------------------------------------------------------- events

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt's name
        point = self._to_lat_lon(event.position().x(), event.position().y())
        if point is None:
            return
        if self._drawing:
            self._draft.append(point)
            self.update()
        self.clicked.emit(point)

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt's name
        self.zoom_by(1.15 if event.angleDelta().y() > 0 else 1 / 1.15)

    # ------------------------------------------------------------ painting

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt's name
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), theme.BACKGROUND)
        if self._frame is None:
            painter.setPen(theme.TEXT_FAINT)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "Nothing is placed yet.\nPlace a camera to see the ground it covers.")
            painter.end()
            return
        self._draw_ground(painter)
        self._draw_zones(painter)
        self._draw_coverage(painter)
        self._draw_cameras(painter)
        self._draw_tracks(painter)
        self._draw_draft(painter)
        self._draw_scale(painter)
        painter.end()

    def _draw_ground(self, painter: QPainter) -> None:
        """The map, warped onto this view's frame.

        Cached against the geometry it was drawn for: the raster is hundreds
        of thousands of cells and rebuilding it on every repaint would make
        panning unusable. The key is everything that changes where a cell
        lands on screen.
        """
        ground = self._ground
        if ground is None or self._frame is None:
            return
        key = (id(ground), self.width(), self.height(), round(self._scale(), 4),
               self._frame.origin.lat, self._frame.origin.lon)
        if self._ground_pixmap is None or self._ground_key != key:
            self._ground_pixmap = self._render_ground(ground)
            self._ground_key = key
        if self._ground_pixmap is None:
            return
        # Where the raster's own corners land on this view. Its rows run north
        # to south, so the top-left cell is the north-west corner.
        grid = ground.grid
        north_west = grid.centre_of(0, 0)
        south_east = grid.centre_of(grid.rows - 1, grid.cols - 1)
        top_left = self._to_screen(north_west)
        bottom_right = self._to_screen(south_east)
        target = QRectF(top_left, bottom_right).normalized()
        if target.width() < 1 or target.height() < 1:
            return
        painter.drawPixmap(target, self._ground_pixmap, QRectF(self._ground_pixmap.rect()))

    @staticmethod
    def _render_ground(ground):
        """The usable cells as an image with transparency everywhere else."""
        try:
            import numpy as np
            from PySide6.QtGui import QImage

            usable = ground.usable
            if not usable.any():
                return None
            rows, cols = usable.shape
            rgba = np.zeros((rows, cols, 4), dtype=np.uint8)
            # The map is BGR; Qt wants RGB, and the alpha carries the
            # confidence so ground the map is less sure of fades rather than
            # claiming the same standing as ground it is sure of.
            rgba[..., 0] = ground.colour[..., 2]
            rgba[..., 1] = ground.colour[..., 1]
            rgba[..., 2] = ground.colour[..., 0]
            alpha = np.clip(ground.confidence, 0.0, 1.0) * 255.0
            rgba[..., 3] = np.where(usable, alpha, 0).astype(np.uint8)
            buffer = np.ascontiguousarray(rgba)
            image = QImage(buffer.data, cols, rows, cols * 4, QImage.Format.Format_RGBA8888)
            return QPixmap.fromImage(image.copy())
        except Exception:  # noqa: BLE001 - a map that will not draw must not take the window with it
            return None

    def _draw_coverage(self, painter: QPainter) -> None:
        for camera_id, pose in self._cameras.items():
            ring = field_of_view(pose, 24)
            if not ring:
                continue
            polygon = QPolygonF([self._to_screen(p) for p in ring])
            colour = QColor(theme.ACCENT if camera_id == self._selected else theme.LIVE)
            fill = QColor(colour)
            fill.setAlpha(34 if camera_id == self._selected else 20)
            painter.setBrush(QBrush(fill))
            outline = QColor(colour)
            outline.setAlpha(120)
            painter.setPen(QPen(outline, 1))
            painter.drawPolygon(polygon)
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _draw_cameras(self, painter: QPainter) -> None:
        # Where a label has already been written, so two masts a few metres
        # apart do not print one name over another — which they did.
        taken: list[QRectF] = []
        for camera_id, pose in sorted(self._cameras.items()):
            centre = self._to_screen(pose.position)
            colour = theme.ACCENT if camera_id == self._selected else theme.TEXT
            painter.setPen(QPen(colour, 2))
            painter.setBrush(QBrush(theme.PANEL_RAISED))
            painter.drawEllipse(centre, 5, 5)
            # A short spike in the heading, so which way a camera looks is
            # readable without reading a number.
            heading = math.radians(pose.heading)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawLine(centre, QPointF(centre.x() + math.sin(heading) * 14, centre.y() - math.cos(heading) * 14))
            painter.setPen(colour)
            painter.drawText(self._free_label_spot(painter, QPointF(centre.x() + 9, centre.y() - 7), camera_id, taken),
                             camera_id)

    @staticmethod
    def _free_label_spot(painter: QPainter, wanted: QPointF, text: str, taken: list[QRectF]) -> QPointF:
        """The first place this label fits without covering one already drawn."""
        metrics = painter.fontMetrics()
        width, height = metrics.horizontalAdvance(text), metrics.height()
        spot = QPointF(wanted)
        for _ in range(8):
            box = QRectF(spot.x(), spot.y() - height, width, height)
            if not any(box.intersects(other) for other in taken):
                break
            spot = QPointF(spot.x(), spot.y() + height)
        taken.append(QRectF(spot.x(), spot.y() - height, width, height))
        return spot

    def _draw_zones(self, painter: QPainter) -> None:
        for zone in self._zones:
            polygon = QPolygonF([self._to_screen(p) for p in zone.ring])
            colour = theme.FAULT if str(zone.kind) == "RESTRICTED" else theme.STALE
            fill = QColor(colour)
            fill.setAlpha(28)
            painter.setBrush(QBrush(fill))
            painter.setPen(QPen(colour, 1, Qt.PenStyle.DashLine))
            painter.drawPolygon(polygon)
            painter.setPen(colour)
            box = polygon.boundingRect()
            watch = f" · {', '.join(sorted(zone.watch))}" if zone.watch else ""
            text = f"{zone.name}{watch}"
            # Below the ring and centred on it. Above put it on the camera
            # marker's own label, which the first console photograph showed.
            width = painter.fontMetrics().horizontalAdvance(text)
            painter.drawText(QPointF(box.center().x() - width / 2, box.bottom() + 14), text)
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _draw_tracks(self, painter: QPainter) -> None:
        for camera_id, tracks in self._tracks.items():
            unprojected = 0
            for track in tracks:
                position = track.position
                if position is None:
                    continue
                if not position.is_projected:
                    # The geometry could not put this on the ground, so the
                    # position is the camera's own with the whole field of
                    # view as its error. That is still worth showing — an
                    # operator learns "something is happening at this camera"
                    # — but it must be impossible to mistake for a fix, so it
                    # is drawn dashed, unfilled, and counted in words.
                    unprojected += 1
                    continue
                centre = self._to_screen(position.point)
                colour = theme.track_colour(track.id)
                radius = max(3.0, position.radius_meters * self._scale())
                halo = QColor(colour)
                halo.setAlpha(46)
                painter.setBrush(QBrush(halo))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(centre, radius, radius)
                painter.setBrush(QBrush(colour))
                painter.setPen(QPen(theme.BACKGROUND, 1))
                painter.drawEllipse(centre, 4, 4)
                if track.heading_degrees is not None and track.speed_mps:
                    heading = math.radians(track.heading_degrees)
                    painter.setPen(QPen(colour, 2))
                    length = 8 + min(24, track.speed_mps * 4)
                    painter.drawLine(centre, QPointF(centre.x() + math.sin(heading) * length,
                                                     centre.y() - math.cos(heading) * length))
            self._draw_links(painter, camera_id)
            if unprojected:
                self._draw_unprojected(painter, camera_id, unprojected)
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _draw_links(self, painter: QPainter, camera_id: str) -> None:
        """A thin line between two tracks a relation joins, and what it says.

        Dashed and faint on purpose: the relation is inferred, and a solid
        line between two dots reads as a fact about the ground.
        """
        placed = {t.id: t.position.point for t in self._tracks.get(camera_id, ())
                  if t.position is not None and t.position.is_projected}
        for relation in self._relations.get(camera_id, ()):
            if relation.object is None:
                continue
            here, there = placed.get(relation.subject), placed.get(relation.object)
            if here is None or there is None:
                continue
            a, b = self._to_screen(here), self._to_screen(there)
            painter.setPen(QPen(theme.STALE, 1, Qt.PenStyle.DotLine))
            painter.drawLine(a, b)
            painter.setPen(theme.STALE)
            middle = QPointF((a.x() + b.x()) / 2, (a.y() + b.y()) / 2 - 3)
            painter.drawText(middle, str(relation.kind).lower())

    def _draw_unprojected(self, painter: QPainter, camera_id: str, count: int) -> None:
        """Something is at this camera that the geometry could not place.

        Drawn as the camera's own range, dashed and unfilled, with the count
        in words beside it. Never as a dot: a dot is a fix, and this is the
        opposite of one.
        """
        pose = self._cameras.get(camera_id)
        if pose is None:
            return
        centre = self._to_screen(pose.position)
        radius = max(6.0, pose.range_meters * self._scale())
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(theme.STALE, 1, Qt.PenStyle.DashLine))
        painter.drawEllipse(centre, radius, radius)
        painter.setPen(theme.STALE)
        painter.drawText(QPointF(centre.x() + 9, centre.y() + 16),
                         f"{count} not placed on the ground — somewhere at {camera_id}")

    def _draw_draft(self, painter: QPainter) -> None:
        if not self._draft:
            return
        painter.setPen(QPen(theme.ACCENT, 2, Qt.PenStyle.DashLine))
        points = [self._to_screen(p) for p in self._draft]
        if len(points) > 1:
            painter.drawPolyline(QPolygonF(points))
        painter.setBrush(QBrush(theme.ACCENT))
        painter.setPen(QPen(theme.BACKGROUND, 1))
        for point in points:
            painter.drawEllipse(point, 3, 3)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(theme.ACCENT)
        painter.drawText(8, 18, f"drawing a zone: {len(points)} point(s) — three or more, then Finish")

    def _draw_scale(self, painter: QPainter) -> None:
        scale = self._scale()
        target = self.width() / 5
        metres = 1.0
        for candidate in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000):
            if candidate * scale <= target:
                metres = float(candidate)
        length = metres * scale
        y = self.height() - 14
        painter.setPen(QPen(theme.TEXT_MUTED, 1))
        painter.drawLine(QPointF(12, y), QPointF(12 + length, y))
        painter.drawLine(QPointF(12, y - 3), QPointF(12, y + 3))
        painter.drawLine(QPointF(12 + length, y - 3), QPointF(12 + length, y + 3))
        font = QFont(painter.font())
        font.setPointSizeF(max(7.5, font.pointSizeF() - 1))
        painter.setFont(font)
        painter.drawText(QPointF(16 + length, y + 4), f"{metres:g} m")
