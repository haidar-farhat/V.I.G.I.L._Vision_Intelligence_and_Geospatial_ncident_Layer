"""The camera view: the frame, with what the system concluded drawn over it.

The overlay is where an operator decides whether to trust the system, so it draws
the distinction between evidence and inference rather than flattening them:

- A **detection** is what the detector found in this frame. Thin, blue, unlabelled.
- A **track** is a claim that persists. Solid green, with an identity.
- A **coasting track** is one the tracker is holding open with no detection
  behind it right now. Amber and dashed, because it is a memory rather than an
  observation, and an operator watching a box glide along with nothing under it
  should be able to see that immediately.

Nothing is labelled with a class the detector cannot produce. Where the detector
does not classify, the overlay says "unclassified" rather than "person".
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from sentinel.detect import DetectorInfo

from . import theme
from sentinel.node import Update


class VideoView(QWidget):
    """Renders the current frame and its overlay."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAutoFillBackground(False)

        self._pixmap: QPixmap | None = None
        self._update: Update | None = None
        self._info: DetectorInfo | None = None
        self._show_detections = True
        self._placeholder = "No source running"

    # ------------------------------------------------------------------ inputs

    def set_detector_info(self, info: DetectorInfo | None) -> None:
        self._info = info

    def set_placeholder(self, text: str) -> None:
        self._placeholder = text
        self.update()

    def clear(self) -> None:
        self._pixmap = None
        self._update = None
        self.update()

    def set_show_detections(self, show: bool) -> None:
        self._show_detections = show
        self.update()

    def show_update(self, update: Update) -> None:
        image = update.image
        if image is None:
            return

        height, width = image.shape[:2]
        # OpenCV gives BGR; Qt's Format_BGR888 reads it directly, avoiding a
        # per-frame colour conversion. `.copy()` is required: the QImage would
        # otherwise reference a numpy buffer that is freed underneath it.
        frame = QImage(image.data, width, height, image.strides[0], QImage.Format.Format_BGR888)
        self._pixmap = QPixmap.fromImage(frame.copy())
        self._update = update
        self.update()

    # ----------------------------------------------------------------- drawing

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), theme.BACKGROUND)

        if self._pixmap is None:
            self._paint_placeholder(painter)
            painter.end()
            return

        target = self._fit_rect()
        # The source rect is required with a QRectF target; without it Qt has no
        # overload to match and the paint fails silently every frame.
        painter.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))

        if self._update is not None:
            self._paint_overlay(painter, target)
            self._paint_readout(painter, target)

        painter.end()

    def _fit_rect(self) -> QRectF:
        """Letterbox the frame. Stretching it would distort every measurement an
        operator makes by eye."""
        assert self._pixmap is not None
        available = QRectF(self.rect())
        scale = min(
            available.width() / self._pixmap.width(),
            available.height() / self._pixmap.height(),
        )
        width = self._pixmap.width() * scale
        height = self._pixmap.height() * scale
        return QRectF(
            available.x() + (available.width() - width) / 2,
            available.y() + (available.height() - height) / 2,
            width,
            height,
        )

    def _paint_placeholder(self, painter: QPainter) -> None:
        painter.setPen(QPen(theme.TEXT_FAINT))
        font = QFont(painter.font())
        font.setPointSize(11)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._placeholder)

    def _paint_overlay(self, painter: QPainter, target: QRectF) -> None:
        assert self._update is not None
        result = self._update.result

        def to_screen(x: float, y: float, w: float, h: float) -> QRectF:
            return QRectF(
                target.x() + x * target.width(),
                target.y() + y * target.height(),
                w * target.width(),
                h * target.height(),
            )

        if self._show_detections:
            pen = QPen(theme.DETECTION, 1.0, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for detection in result.detections:
                box = detection.bbox
                painter.drawRect(to_screen(box.x, box.y, box.w, box.h))

        font = QFont(painter.font())
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)

        for track in result.tracks:
            gap = result.timestamp_millis - track.last_seen_millis
            coasting = gap > theme.COASTING_AFTER_MILLIS
            colour = theme.TRACK_COASTING if coasting else theme.TRACK

            pen = QPen(colour, 2.0)
            pen.setStyle(Qt.PenStyle.DashLine if coasting else Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            box = track.bbox
            rect = to_screen(box.x, box.y, box.w, box.h)
            painter.drawRect(rect)

            label = self._label_for(track)
            self._draw_label(painter, rect, label, colour)

    def _label_for(self, track) -> str:
        """What to write beside a track.

        Short on purpose. With four or five objects in shot, labels carrying
        every attribute overlap each other and the boxes they belong to, and an
        overlay that obscures the scene defeats itself. The full record — class,
        confidence, position, uncertainty, provenance — is one row per object in
        the table below, where there is room for it.

        A class is only named when the detector can actually produce one. Under
        the motion detector the label is the identity alone, which claims
        nothing beyond "this is the same thing as before".
        """
        parts = [f"#{track.id}"]

        if self._info is not None and self._info.classifies:
            parts.append(self._info.label_for(track.class_id))

        if track.speed_mps is not None and track.speed_mps >= 0.3:
            parts.append(f"{track.speed_mps:.1f} m/s")

        return " ".join(parts)

    def _draw_label(self, painter: QPainter, rect: QRectF, text: str, colour: QColor) -> None:
        metrics = painter.fontMetrics()
        width = metrics.horizontalAdvance(text) + 10
        height = metrics.height() + 4

        # Above the box normally, inside it when the box is against the top edge,
        # so a label never leaves the frame.
        top = rect.top() - height - 2
        if top < 0:
            top = rect.top() + 2

        background = QRectF(rect.left(), top, width, height)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 170)))
        painter.drawRoundedRect(background, 3, 3)

        painter.setPen(QPen(colour))
        painter.drawText(background.adjusted(5, 0, 0, 0),
                         Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)

    def _paint_readout(self, painter: QPainter, target: QRectF) -> None:
        """Frame index, media time and rate, burned into the corner.

        Present because a still frame with a track box on it is not evidence
        unless you can say which frame it was.
        """
        assert self._update is not None
        result = self._update.result
        seconds = result.timestamp_millis / 1000.0

        lines = [
            f"{result.source_id}",
            f"frame {result.index}   t+{seconds:07.3f}s",
            f"{self._update.analysis_fps:.0f} fps analysed"
            + (f"   {self._update.skipped} frames not drawn" if self._update.skipped else ""),
        ]

        font = QFont(painter.font())
        font.setPointSize(9)
        font.setBold(False)
        painter.setFont(font)
        metrics = painter.fontMetrics()

        width = max(metrics.horizontalAdvance(line) for line in lines) + 16
        height = metrics.height() * len(lines) + 12
        panel = QRectF(target.left() + 8, target.bottom() - height - 8, width, height)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 150)))
        painter.drawRoundedRect(panel, 4, 4)

        painter.setPen(QPen(theme.TEXT_MUTED))
        y = panel.top() + 6 + metrics.ascent()
        for line in lines:
            painter.drawText(QPointF(panel.left() + 8, y), line)
            y += metrics.height()
