"""Where a camera is and where it points.

A camera that has not been placed cannot say where anything is. The console
refuses to guess: until an operator fills this in, tracks are found and followed
but reported as *not placed*, and the plan view says so rather than drawing them
somewhere plausible.

That refusal is the point. A position on a map is acted on — someone is sent to
it — so a fabricated one is worse than none. The alternative, defaulting to a
nominal origin, produces coordinates that look exactly like measured ones.

Every field here is something an installer can actually determine on site with a
tape measure, a compass and the camera's own specification sheet. Nothing asks
for a value that would have to be guessed.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from sentinel.core import CameraPose, LatLon

from . import theme


def _spin(
    minimum: float, maximum: float, value: float, suffix: str, decimals: int = 1, step: float = 1.0
) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(minimum, maximum)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setValue(value)
    box.setSuffix(suffix)
    box.setAlignment(Qt.AlignmentFlag.AlignRight)
    return box


class PlacementDialog(QDialog):
    """Collects a camera pose."""

    def __init__(self, current: CameraPose | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Camera placement")
        self.setStyleSheet(theme.STYLESHEET)
        self.setMinimumWidth(420)

        pose = current

        # Six decimal places is about 0.1 m, which is finer than any consumer GPS
        # and far finer than the projection's own uncertainty. More would imply a
        # precision the rest of the system does not have.
        self.latitude = _spin(-90.0, 90.0, pose.position.lat if pose else 0.0, "°", 6, 0.000_1)
        self.longitude = _spin(-180.0, 180.0, pose.position.lon if pose else 0.0, "°", 6, 0.000_1)
        self.mount_height = _spin(0.5, 60.0, pose.mount_height if pose else 6.0, " m", 2, 0.1)
        self.heading = _spin(0.0, 359.9, pose.heading if pose else 0.0, "°", 1, 5.0)
        self.pitch = _spin(-89.0, 30.0, pose.pitch if pose else -20.0, "°", 1, 1.0)
        self.horizontal_fov = _spin(5.0, 180.0, pose.horizontal_fov if pose else 62.0, "°", 1, 1.0)
        self.vertical_fov = _spin(3.0, 140.0, pose.vertical_fov if pose else 36.0, "°", 1, 1.0)
        self.range_meters = _spin(5.0, 500.0, pose.range_meters if pose else 90.0, " m", 0, 5.0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        intro = QLabel(
            "Until a camera is placed, objects are tracked but not located.\n"
            "Nothing here is sent anywhere; it is stored with the camera."
        )
        intro.setObjectName("Caption")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        form.setSpacing(8)
        form.addRow("Latitude", self.latitude)
        form.addRow("Longitude", self.longitude)
        form.addRow("Mount height", self.mount_height)
        form.addRow("Heading (0 = north)", self.heading)
        form.addRow("Pitch (negative = down)", self.pitch)
        form.addRow("Horizontal field of view", self.horizontal_fov)
        form.addRow("Vertical field of view", self.vertical_fov)
        form.addRow("Useful range", self.range_meters)
        layout.addLayout(form)

        self._warning = QLabel("")
        self._warning.setObjectName("Caption")
        self._warning.setWordWrap(True)
        layout.addWidget(self._warning)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        for widget in (self.pitch, self.vertical_fov, self.mount_height, self.range_meters):
            widget.valueChanged.connect(self._describe_coverage)
        self._describe_coverage()

    def pose(self) -> CameraPose:
        return CameraPose(
            position=LatLon(self.latitude.value(), self.longitude.value()),
            mount_height=self.mount_height.value(),
            heading=self.heading.value(),
            pitch=self.pitch.value(),
            horizontal_fov=self.horizontal_fov.value(),
            vertical_fov=self.vertical_fov.value(),
            range_meters=self.range_meters.value(),
        )

    def _describe_coverage(self) -> None:
        """Say what this pose can actually see, as the operator types.

        The stated range is routinely not what a camera covers: a downward tilt
        and a finite vertical field of view bound the ground it sees at both
        ends. An installer who enters 90 m and is covering 7 m to 19 m should
        find that out here, not from an intrusion nobody was alerted to.
        """
        import math

        pose = self.pose()
        half = pose.vertical_fov / 2.0
        depression_near = -pose.pitch + half
        depression_far = -pose.pitch - half

        def ground(depression: float) -> float | None:
            if depression <= 0.5:
                return None
            return pose.mount_height / math.tan(math.radians(depression))

        near = ground(depression_near)
        far = ground(depression_far)

        if near is None:
            self._warning.setText(
                "This camera is pointed at or above the horizon. Nothing it sees "
                "can be placed on the ground."
            )
            self._warning.setStyleSheet(f"color: {theme.FAULT.name()};")
            return

        if far is None:
            self._warning.setText(
                f"Sees the ground from {near:.0f} m outwards; the top of its view is "
                f"above the horizon, so the far edge is limited only by the stated "
                f"range of {pose.range_meters:.0f} m."
            )
            self._warning.setStyleSheet(f"color: {theme.STALE.name()};")
            return

        far = min(far, pose.range_meters)
        self._warning.setText(
            f"Covers the ground from about {near:.0f} m to {far:.0f} m — a band "
            f"{far - near:.0f} m deep. It is blind closer than {near:.0f} m."
        )
        self._warning.setStyleSheet(f"color: {theme.TEXT_MUTED.name()};")
