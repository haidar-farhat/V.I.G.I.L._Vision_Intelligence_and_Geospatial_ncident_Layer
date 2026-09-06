"""The camera view: the frame the conclusions were drawn from, and the conclusions on it.

The image and the boxes travel together in one `FrameResult` and are drawn
together. Drawing a track box over a *different* frame than the one it was
computed from misrepresents what the system saw, so this widget never keeps
one without the other.

For the same reason it draws the frame's **own verdict on itself**. A frame
that is out of focus, blown out, or the same frame the decoder handed over a
second ago still produces boxes, and boxes drawn over it look exactly like
boxes drawn over a good frame. An operator watching a camera whose lens has
been sprayed sees a calm, empty scene and no indication that calm is all it
can ever show. The banner is that indication.

The camera's own motion is shown for the same reason: when a mast is moving,
every position on the map is being computed from a pose that is no longer
true, and that is worth knowing while it is happening rather than afterwards.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget

from ...domain.detection import DetectorInfo
from ...domain.geo import Vec2
from . import theme


class VideoView(QWidget):
    """One camera. Draws the frame scaled to fit, with boxes in image coordinates."""

    #: A click on the picture, as a `Vec2` in frame fractions — the same
    #: coordinates a detection and a projection use, so what is clicked and
    #: what is computed are in one system. Emitted only for a click that
    #: landed on the image: the widget is letterboxed, and a click on the bar
    #: beside the picture is not a point in the picture.
    clicked = Signal(object)

    def __init__(self, camera_id: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.camera_id = camera_id
        self._pixmap: QPixmap | None = None
        self._tracks: tuple = ()
        self._detector: DetectorInfo | None = None
        self._caption = "waiting for the first frame"
        self._fps = 0.0
        self._quality = None
        self._motion = None
        #: `(Vec2, label)` crosses drawn over the picture. Used by the
        #: calibration dialog to show what has been marked so far.
        self._marks: tuple = ()
        self.setMinimumSize(220, 165)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def show_still(self, pixmap) -> None:
        """One frozen frame, with no tracks over it.

        The tracks go because they belong to a moment and this widget is now
        showing a still: boxes left over a frame nobody is updating are boxes
        drawn over a scene that has moved on, which is the exact
        misrepresentation `show_result` exists to prevent.
        """
        self._pixmap = pixmap if pixmap is not None and not pixmap.isNull() else None
        self._tracks = ()
        self._fps = 0.0
        self.update()

    def still(self):
        """The frame currently on screen, or `None` before the first one."""
        return self._pixmap

    def set_marks(self, marks) -> None:
        self._marks = tuple(marks)
        self.update()

    def frame_point(self, x: float, y: float) -> Vec2 | None:
        """Widget pixels to frame fractions, or `None` off the picture."""
        rect = self._frame_rect()
        if self._pixmap is None or self._pixmap.isNull() or not rect.contains(QPointF(x, y)):
            return None
        return Vec2((x - rect.x()) / rect.width(), (y - rect.y()) / rect.height())

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt's name
        point = self.frame_point(event.position().x(), event.position().y())
        if point is not None:
            self.clicked.emit(point)

    def set_detector_info(self, info: DetectorInfo | None) -> None:
        self._detector = info

    def show_result(self, result, fps: float = 0.0) -> None:
        self._tracks = tuple(result.tracks)
        self._fps = fps
        self._quality = getattr(result, "quality", None)
        self._motion = getattr(result, "camera_motion", None)
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
        self._quality = None
        self._motion = None
        self.update()

    def warnings(self) -> list[str]:
        """What this frame cannot support, in an operator's words.

        Empty when the frame is fine, which is the common case and must cost
        nothing to draw.
        """
        out: list[str] = []
        quality = self._quality
        if quality is not None:
            degraded = getattr(quality, "degraded", None)
            if degraded:
                out.append(degraded)
        motion = self._motion
        if motion is not None and getattr(motion, "measured", False) and not motion.still:
            # A camera that is moving is a camera whose stored pose is wrong
            # for as long as it moves, and every position on the plan is
            # computed from that pose.
            out.append(f"the camera is moving ({motion.magnitude:.0%} of a frame, "
                       f"{motion.rotation_degrees:+.1f}°): positions are unreliable while it does")
        return out

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
        self._draw_marks(painter, frame)

        painter.setPen(theme.TEXT_MUTED)
        font = QFont(painter.font())
        font.setPointSizeF(max(7.5, font.pointSizeF() - 1))
        painter.setFont(font)
        rate = f"  ·  {self._fps:.0f} fps" if self._fps else ""
        painter.drawText(6, self.height() - 6, f"{self.camera_id}  ·  {self._caption}{rate}")
        self._draw_warnings(painter, font)
        painter.end()

    def _draw_marks(self, painter: QPainter, frame: QRectF) -> None:
        """Numbered crosses where points have been marked.

        A cross rather than a filled dot, because the thing being marked is
        under the cursor and a dot large enough to see would hide it.
        """
        if not self._marks:
            return
        font = QFont(painter.font())
        font.setBold(True)
        painter.setFont(font)
        for point, label in self._marks:
            x = frame.x() + point.x * frame.width()
            y = frame.y() + point.y * frame.height()
            painter.setPen(QPen(theme.BACKGROUND, 3))
            painter.drawLine(QPointF(x - 7, y), QPointF(x + 7, y))
            painter.drawLine(QPointF(x, y - 7), QPointF(x, y + 7))
            painter.setPen(QPen(theme.ACCENT, 1.4))
            painter.drawLine(QPointF(x - 7, y), QPointF(x + 7, y))
            painter.drawLine(QPointF(x, y - 7), QPointF(x, y + 7))
            painter.drawText(QPointF(x + 9, y - 4), label)

    def _draw_warnings(self, painter: QPainter, font: QFont) -> None:
        """A banner across the top when the frame cannot support what is drawn
        on it.

        Across the top and opaque, not a subtle tint: the failure being warned
        about is precisely the one that *looks* like a working camera, so it
        has to be the thing an operator notices first.
        """
        warnings = self.warnings()
        if not warnings:
            return
        painter.setFont(font)
        metrics = painter.fontMetrics()
        line = metrics.height() + 4
        height = line * len(warnings) + 4
        banner = QRectF(0, 0, self.width(), height)
        painter.fillRect(banner, theme.FAULT.darker(180))
        painter.setPen(theme.TEXT)
        for index, text in enumerate(warnings):
            painter.drawText(QRectF(8, 2 + index * line, self.width() - 16, line),
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                             metrics.elidedText(text, Qt.TextElideMode.ElideRight,
                                                int(self.width() - 16)))

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
