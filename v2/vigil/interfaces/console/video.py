"""The camera view: the frame the conclusions were drawn from, and the conclusions on it.

The image and the boxes travel together in one `FrameResult` and are drawn
together. Drawing a track box over a *different* frame than the one it was
computed from misrepresents what the system saw, so this widget never keeps
one without the other.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget

from ...domain.detection import DetectorInfo
from . import theme


class VideoView(QWidget):
    """One camera. Draws the frame scaled to fit, with boxes in image coordinates."""

    def __init__(self, camera_id: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.camera_id = camera_id
        self._pixmap: QPixmap | None = None
        self._tracks: tuple = ()
        self._detector: DetectorInfo | None = None
        self._caption = "waiting for the first frame"
        self._fps = 0.0
        self.setMinimumSize(220, 165)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_detector_info(self, info: DetectorInfo | None) -> None:
        self._detector = info

    def show_result(self, result, fps: float = 0.0) -> None:
        self._tracks = tuple(result.tracks)
        self._fps = fps
        image = getattr(result, "image", None)
        if image is not None:
            self._pixmap = QPixmap.fromImage(_to_qimage(image))
        self._caption = f"{len(self._tracks)} tracked"
        self.update()

    def set_caption(self, text: str) -> None:
        self._caption = text
        self.update()

    def clear(self) -> None:
        self._pixmap = None
        self._tracks = ()
        self.update()

    # ------------------------------------------------------------ painting

    def _frame_rect(self) -> QRectF:
        """Where the image sits inside the widget, letterboxed and centred."""
        if self._pixmap is None or self._pixmap.isNull():
            return QRectF(self.rect())
        w, h = self.width(), self.height()
        pw, ph = self._pixmap.width(), self._pixmap.height()
        scale = min(w / pw, h / ph)
        sw, sh = pw * scale, ph * scale
        return QRectF((w - sw) / 2, (h - sh) / 2, sw, sh)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt's name
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), theme.BACKGROUND)
        frame = self._frame_rect()
        if self._pixmap is not None and not self._pixmap.isNull():
            painter.drawPixmap(frame, self._pixmap, QRectF(self._pixmap.rect()))
        else:
            painter.setPen(theme.TEXT_FAINT)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "no frame")

        for track in self._tracks:
            self._draw_track(painter, frame, track)

        painter.setPen(theme.TEXT_MUTED)
        font = QFont(painter.font())
        font.setPointSizeF(max(7.5, font.pointSizeF() - 1))
        painter.setFont(font)
        rate = f"  ·  {self._fps:.0f} fps" if self._fps else ""
        painter.drawText(6, self.height() - 6, f"{self.camera_id}  ·  {self._caption}{rate}")
        painter.end()

    def _draw_track(self, painter: QPainter, frame: QRectF, track) -> None:
        colour = theme.track_colour(track.id)
        box = track.bbox
        rect = QRectF(frame.x() + box.x * frame.width(), frame.y() + box.y * frame.height(),
                      box.width * frame.width(), box.height * frame.height())
        pen = QPen(colour, 2)
        if track.coasting:
            # Dashed while the detector cannot see it: the box is where the
            # tracker believes the object is, not where it was measured.
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawRect(rect)

        # The contact point, which is what was actually projected to the map.
        contact = QPointF(frame.x() + track.contact.x * frame.width(), frame.y() + track.contact.y * frame.height())
        painter.setBrush(colour)
        painter.setPen(QPen(theme.BACKGROUND, 1))
        painter.drawEllipse(contact, 3.5, 3.5)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        label = self._label_for(track)
        painter.setPen(colour)
        painter.drawText(QPointF(rect.left(), max(frame.y() + 10, rect.top() - 4)), label)

    def _label_for(self, track) -> str:
        name = self._detector.label_for(track.class_id) if self._detector is not None else None
        # "unclassified" is never invented here: a detector that does not
        # classify says nothing, and the track is named by its id alone.
        head = f"{name} " if name else ""
        speed = f" {track.speed_mps:.1f} m/s" if track.speed_mps is not None else ""
        return f"{head}#{track.id} {track.confidence:.2f}{speed}"


def _to_qimage(image: np.ndarray) -> QImage:
    """BGR ndarray to QImage, copied — the buffer belongs to the camera thread."""
    if image.ndim == 2:
        height, width = image.shape
        return QImage(image.data, width, height, width, QImage.Format.Format_Grayscale8).copy()
    height, width, channels = image.shape
    if channels == 4:
        return QImage(image.data, width, height, width * 4, QImage.Format.Format_ARGB32).copy()
    return QImage(image.data, width, height, width * 3, QImage.Format.Format_BGR888).copy()
