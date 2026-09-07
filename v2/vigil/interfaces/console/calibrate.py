"""Marking points on the picture and on the plan, to measure a camera's pose.

# Why a dialog and not a command

`vigil cameras calibrate --points "u,v,lat,lon; …"` works and is what a script
uses, but it asks an operator to read a fraction off a frame and a latitude off
a map and type both without transposing anything. The way this is actually
done is by pointing: click the corner of the loading bay in the picture, then
click the same corner on the plan. That is what this is.

# What it shows while you work

The fit, after every pair, before anything is saved. Four points is the
arithmetic minimum and it is thin; the number that matters is not how many
points there are but what the covariance came out at, so the covariance is on
screen the whole time and the Save button is disabled until it beats the
assumption it would replace.

The worst point is named for the same reason. A good RMS with one bad residual
is a mis-clicked pair -- almost always a transposition, mark 3 in the picture
matched to mark 4 on the plan -- and that is a different problem from a bad
pose, fixed by removing one row rather than by starting again.

# The frame is frozen

The picture is the last frame that arrived when the dialog opened, and it does
not update while the dialog is open. A live view would move the thing being
pointed at between the click and the release, and worse, a pair could be
marked against one frame and another pair against a frame taken after the mast
swung. Every mark here is against one image.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QListWidget, QPushButton,
    QSplitter, QVBoxLayout, QWidget,
)

from ...domain.geo import LatLon, Vec2
from ...service.calibration import MIN_POINTS, Correspondence
from . import theme
from .plan import PlanView
from .video import VideoView


class CalibrateDialog(QDialog):
    """Click a point in the picture, then the same point on the plan."""

    def __init__(self, camera, commands, pixmap=None, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(f"Measure {camera.id}'s pose")
        self.setModal(True)
        self.resize(1000, 620)
        self._camera = camera
        self._commands = commands
        self._pairs: list[Correspondence] = []
        #: The half-made pair: an image point waiting for its match, or the
        #: other way round. Either order is allowed because an operator who
        #: has just found a corner on the map should not have to remember
        #: which half the dialog wanted first.
        self._pending_image: Vec2 | None = None
        self._pending_ground: LatLon | None = None
        self._result = None

        self.video = VideoView(camera.id)
        self.video.show_still(pixmap)
        self.video.set_caption("click a point you can also find on the plan")
        self.video.clicked.connect(self._image_clicked)

        self.plan = PlanView()
        self.plan.set_cameras({camera.id: camera.pose} if camera.pose else {})
        self.plan.set_zones(commands.zones())
        self.plan.set_ground(commands.ground())
        self.plan.clicked.connect(self._ground_clicked)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(_panel("PICTURE", self.video))
        split.addWidget(_panel("PLAN — NO EXTERNAL TILES", self.plan))
        split.setSizes([500, 500])

        self.pairs = QListWidget()
        self.pairs.setMaximumHeight(110)
        self.report = QLabel()
        self.report.setWordWrap(True)
        self.report.setFont(QFont("Consolas", 8))
        self.report.setTextFormat(Qt.TextFormat.PlainText)

        self.remove = QPushButton("Remove selected")
        self.remove.clicked.connect(self._remove_selected)
        self.solve_position = QCheckBox("Also solve where the camera is")
        self.solve_position.setToolTip(
            "Off by default. A click on a map is worth about a metre and the orientation is worth "
            "two degrees, and at 40 m the second matters more, so solving both from a handful of "
            "points spends the geometry on the term that matters less. Turn it on when these are "
            "surveyed control points rather than map clicks."
        )
        self.solve_position.toggled.connect(self._refit)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save
                                   | QDialogButtonBox.StandardButton.Cancel)
        self.save = buttons.button(QDialogButtonBox.StandardButton.Save)
        self.save.setText("Save this measurement")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        controls = QHBoxLayout()
        controls.addWidget(self.remove)
        controls.addWidget(self.solve_position)
        controls.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addWidget(split, 1)
        layout.addWidget(self.pairs)
        layout.addLayout(controls)
        layout.addWidget(self.report)
        layout.addWidget(buttons)
        self._refresh()

    # ------------------------------------------------------------- marking

    def _image_clicked(self, point: Vec2) -> None:
        if self._pending_ground is not None:
            self._add(point, self._pending_ground)
            self._pending_ground = None
        else:
            self._pending_image = point
        self._refresh()

    def _ground_clicked(self, point: LatLon) -> None:
        if self._pending_image is not None:
            self._add(self._pending_image, point)
            self._pending_image = None
        else:
            self._pending_ground = point
        self._refresh()

    def _add(self, image: Vec2, ground: LatLon) -> None:
        self._pairs.append(Correspondence(image, ground, f"point {len(self._pairs) + 1}"))

    def _remove_selected(self) -> None:
        row = self.pairs.currentRow()
        if 0 <= row < len(self._pairs):
            del self._pairs[row]
            # Renumber, so the labels on screen keep matching the report's
            # "worst was point N" -- a stale number sends somebody to the
            # wrong row to fix a mis-click.
            self._pairs = [Correspondence(c.image, c.ground, f"point {i}")
                           for i, c in enumerate(self._pairs, start=1)]
        self._refresh()

    # ------------------------------------------------------------ the fit

    def _refit(self) -> None:
        self._result = None
        if len(self._pairs) >= MIN_POINTS:
            outcome = self._commands.try_calibration(
                self._camera.id, self._pairs, solve_position=self.solve_position.isChecked())
            self._result = outcome.value if outcome else None
            self._message = "" if outcome else outcome.message
        else:
            need = MIN_POINTS - len(self._pairs)
            self._message = (f"{need} more point{'s' if need != 1 else ''} before this can be "
                             f"fitted. Spread them across the frame and across the range: near and "
                             f"far, left and right.")

    def _refresh(self) -> None:
        self._refit()
        self.video.set_marks([(c.image, str(i)) for i, c in enumerate(self._pairs, start=1)])
        self.plan.set_marks([(c.ground, str(i)) for i, c in enumerate(self._pairs, start=1)])
        self.pairs.clear()
        for i, c in enumerate(self._pairs, start=1):
            residual = ""
            if self._result is not None and i <= len(self._result.residuals):
                residual = f"   off by {self._result.residuals[i - 1] * 100:.2f}% of the frame"
            self.pairs.addItem(f"{i}.  picture {c.image.x:.3f},{c.image.y:.3f}"
                               f"   ground {c.ground.lat:.6f},{c.ground.lon:.6f}{residual}")

        if self._pending_image is not None:
            self.video.set_caption("now click the same spot on the plan")
        elif self._pending_ground is not None:
            self.video.set_caption("now click the same spot in the picture")
        else:
            self.video.set_caption("click a point you can also find on the plan")

        assumed = self._camera.pose.uncertainty if self._camera.pose else None
        better = self._result is not None and assumed is not None and self._result.better_than(assumed)
        self.save.setEnabled(better)
        if self._result is None:
            self.report.setStyleSheet(f"color: {theme.TEXT_MUTED.name()};")
            self.report.setText(self._message)
            return
        text = self._result.describe()
        if better:
            self.report.setStyleSheet(f"color: {theme.TEXT.name()};")
            self.report.setText(f"{text}\n\nThis beats the "
                                f"+/-{assumed.heading_deg:.1f}° this camera assumes today.")
        else:
            # Not an error. It is a fit that has not earned its place yet, and
            # the difference matters: the operator's next move is more points,
            # not a bug report.
            self.report.setStyleSheet(f"color: {theme.STALE.name()};")
            self.report.setText(f"{text}\n\nNot yet better than the +/-{assumed.heading_deg:.1f}° "
                                f"this camera already assumes, so it cannot be saved. More points, "
                                f"wider apart -- or check point {self._result.worst_index + 1}, "
                                f"which fits worst.")

    def value(self):
        """`(points, solve_position)` for the caller to save. The dialog never
        writes anything itself; `Commands` does, with the principal."""
        return list(self._pairs), self.solve_position.isChecked()


def _panel(title: str, widget: QWidget) -> QWidget:
    box = QWidget()
    layout = QVBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(3)
    label = QLabel(title)
    label.setStyleSheet(f"color: {theme.TEXT_FAINT.name()}; font-size: 9px; letter-spacing: 1px;")
    layout.addWidget(label)
    layout.addWidget(widget, 1)
    return box
