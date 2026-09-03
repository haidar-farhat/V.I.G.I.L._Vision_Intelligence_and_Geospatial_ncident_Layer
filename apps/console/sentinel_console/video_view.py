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

from datetime import datetime, timezone

import numpy as np

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


#: Media time from a file will not reach this in any recording anybody will
#: ever make; a live camera's wall-clock stamp is always past it. The two are
#: therefore distinguishable, which they have to be, because they mean entirely
#: different things and the same format made one of them nonsense.
_EPOCH_THRESHOLD_MILLIS = 1_000_000_000_000


def _stamp(millis: int) -> str:
    """The frame's time, in whichever form is meaningful for its source.

    A file's frames are counted from the start of the recording, so `t+11.933s`
    is exactly right. A live camera's are stamped with the wall clock, and the
    same format produced `t+1788428138.044s` — seen in a screenshot of the real
    thing, and meaning nothing to anybody.
    """
    if millis >= _EPOCH_THRESHOLD_MILLIS:
        moment = datetime.fromtimestamp(millis / 1000.0, tz=timezone.utc)
        return moment.strftime("%H:%M:%S.") + f"{moment.microsecond // 1000:03d} UTC"
    return f"t+{millis / 1000.0:07.3f}s"


#: How strongly a mask tints the frame underneath it. Low on purpose: this is
#: evidence, and an overlay that hides the pixels it is describing makes the
#: frame useless for the one job it has.
MASK_ALPHA = 90


def _mask_image(mask: "np.ndarray", colour: QColor) -> QImage:
    """A translucent, single-colour image of one instance's silhouette.

    Built per frame rather than cached: the mask changes every frame, and at
    thirty a second a cache keyed on anything would miss every time while
    holding a reference to every frame it had ever seen.
    """
    height, width = mask.shape
    # ARGB32 is BGRA in memory on a little-endian machine, which every platform
    # this runs on is. Writing the channels in the wrong order costs nothing at
    # runtime and turns every person blue-green, which reads as a rendering
    # style rather than as the bug it is.
    buffer = np.empty((height, width, 4), dtype=np.uint8)
    buffer[..., 0] = colour.blue()
    buffer[..., 1] = colour.green()
    buffer[..., 2] = colour.red()
    buffer[..., 3] = (mask > 0) * MASK_ALPHA
    # `.copy()` because QImage does not take ownership of the buffer, and
    # `buffer` is a local that dies at the end of this function.
    return QImage(
        buffer.data, width, height, width * 4, QImage.Format.Format_ARGB32
    ).copy()


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
            # The readout's rectangle is computed first and handed to the
            # overlay, so a label can move out of its way. Painting the readout
            # last and hoping is what produced `1.6gate` — a speed drawn under a
            # camera name, both illegible, in a screenshot of the real thing.
            reserved = self._readout_rect(painter, target)
            self._paint_overlay(painter, target, reserved)
            self._paint_readout(painter, target, reserved)

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

    def _paint_overlay(
        self, painter: QPainter, target: QRectF, reserved: QRectF | None = None
    ) -> None:
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
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for detection in result.detections:
                box = detection.bbox
                rect = to_screen(box.x, box.y, box.w, box.h)
                # The silhouette when there is one, the rectangle when there is
                # not. Showing both would draw a box around every mask and hide
                # the one difference the operator is being shown: whether this
                # detector knows the object's shape or only its extent.
                mask = getattr(detection, "mask", None)
                if mask is not None and mask.size:
                    painter.drawImage(rect, _mask_image(mask, theme.DETECTION))
                else:
                    painter.setPen(pen)
                    painter.drawRect(rect)

        font = QFont(painter.font())
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)

        placed: list[QRectF] = []
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
            self._draw_label(
                painter, rect, label, colour, target, reserved, placed
            )

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

    def _draw_label(
        self, painter: QPainter, rect: QRectF, text: str, colour: QColor,
        frame: QRectF | None = None, reserved: QRectF | None = None,
        placed: list[QRectF] | None = None,
    ) -> None:
        """Draw one track's label, avoiding the readout and the labels already
        drawn.

        ``placed`` accumulates what has been drawn this frame. Without it two
        objects standing near each other get their labels stacked in the same
        few pixels, which is unreadable at exactly the moment the operator most
        needs to tell them apart — a person beside a bag reads as one smear.
        """
        metrics = painter.fontMetrics()
        width = metrics.horizontalAdvance(text) + 10
        height = metrics.height() + 4

        # Above the box normally, inside it when the box is against the top edge.
        top = rect.top() - height - 2
        if frame is not None and top < frame.top():
            top = rect.top() + 2

        background = QRectF(rect.left(), top, width, height)

        if frame is not None:
            # Clamped on every edge, not just the top. Only the top was checked,
            # so a track against the left of the frame drew its label at a
            # negative x and ran off the picture — visible in a screenshot,
            # invisible to every test.
            if background.right() > frame.right():
                background.moveRight(frame.right() - 2)
            if background.left() < frame.left():
                background.moveLeft(frame.left() + 2)
            if background.bottom() > frame.bottom():
                background.moveBottom(frame.bottom() - 2)
            if background.top() < frame.top():
                background.moveTop(frame.top() + 2)

        if reserved is not None and background.intersects(reserved):
            # The readout is provenance — which frame, at what time — and an
            # operator cannot recover it from anywhere else on screen. A track
            # label can: it is a row in the table. So the label moves.
            below = QRectF(background)
            below.moveTop(rect.bottom() + 2)
            if frame is not None and below.bottom() > frame.bottom():
                below.moveBottom(frame.bottom() - 2)

            if not below.intersects(reserved):
                background = below
            else:
                # Both above and below are blocked, so go sideways, to the far
                # edge of the panel rather than on top of it.
                background.moveLeft(reserved.right() + 4)
                if frame is not None and background.right() > frame.right():
                    background.moveRight(frame.right() - 2)

        if placed is not None:
            # Step down past anything already drawn. Bounded: after a few tries
            # the labels are further apart than they are tall, and going on
            # would push a label further from the box it names than from the one
            # it does not — a label in the wrong place is worse than a crowded
            # one.
            for _ in range(6):
                clash = next(
                    (other for other in placed if background.intersects(other)), None
                )
                if clash is None:
                    break
                background.moveTop(clash.bottom() + 2)
                if frame is not None and background.bottom() > frame.bottom():
                    background.moveBottom(frame.bottom() - 2)
                    break
            placed.append(QRectF(background))

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 170)))
        painter.drawRoundedRect(background, 3, 3)

        painter.setPen(QPen(colour))
        painter.drawText(background.adjusted(5, 0, 0, 0),
                         Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)

    def _readout_lines(self) -> list[str]:
        assert self._update is not None
        result = self._update.result
        return [
            f"{result.source_id}",
            f"frame {result.index}   {_stamp(result.timestamp_millis)}",
            f"{self._update.analysis_fps:.0f} fps analysed"
            + (f"   {self._update.skipped} frames not drawn" if self._update.skipped else ""),
        ]

    def _readout_rect(self, painter: QPainter, target: QRectF) -> QRectF:
        """Where the readout will go, worked out before anything is drawn.

        Separate from painting it so the overlay can be told to keep off.
        """
        if self._update is None:
            return QRectF()

        font = QFont(painter.font())
        font.setPointSize(9)
        font.setBold(False)
        metrics = painter.fontMetrics() if painter.font() == font else None

        painter.save()
        painter.setFont(font)
        metrics = painter.fontMetrics()
        lines = self._readout_lines()
        width = max(metrics.horizontalAdvance(line) for line in lines) + 16
        height = metrics.height() * len(lines) + 12
        painter.restore()

        return QRectF(target.left() + 8, target.bottom() - height - 8, width, height)

    def _paint_readout(self, painter: QPainter, target: QRectF, panel: QRectF) -> None:
        """Frame index, media time and rate, burned into the corner.

        Present because a still frame with a track box on it is not evidence
        unless you can say which frame it was.
        """
        assert self._update is not None
        lines = self._readout_lines()

        font = QFont(painter.font())
        font.setPointSize(9)
        font.setBold(False)
        painter.setFont(font)
        metrics = painter.fontMetrics()

        painter.setPen(Qt.PenStyle.NoPen)
        # Nearly opaque. At alpha 150 a bright scene showed straight through the
        # panel and the provenance became unreadable over exactly the frames an
        # operator would want it for.
        painter.setBrush(QBrush(QColor(0, 0, 0, 215)))
        painter.drawRoundedRect(panel, 4, 4)

        painter.setPen(QPen(theme.TEXT_MUTED))
        y = panel.top() + 6 + metrics.ascent()
        for line in lines:
            painter.drawText(QPointF(panel.left() + 8, y), line)
            y += metrics.height()
