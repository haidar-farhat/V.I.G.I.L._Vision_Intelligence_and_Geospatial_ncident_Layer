"""Every dialog the console opens.

None of them sets `WA_DeleteOnClose`. In v1 four dialogs did, and Qt deleted
them the moment `exec()` returned — so *Place…* and *Add zone…* read a
destroyed object and did nothing at all, while every test passed because the
tests called the slot beneath the dialog. Here the caller reads the dialog
and then calls `deleteLater()`, and `ask()` below is the only way any of them
is opened.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QSpinBox, QVBoxLayout, QWidget,
)
from PySide6.QtCore import Qt

from ...domain.geo import CameraPose, LatLon
from ...domain.zones import Schedule, ZoneKind
from ...service.auth import AuthError, Role

MAX_SIGN_IN_ATTEMPTS = 5


def ask(dialog: QDialog):
    """Show a dialog, read it, then let Qt delete it. The only way to open one."""
    accepted = dialog.exec() == QDialog.DialogCode.Accepted
    value = dialog.value() if accepted and hasattr(dialog, "value") else None
    dialog.deleteLater()
    return accepted, value


class _Dialog(QDialog):
    def __init__(self, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self._outer = QVBoxLayout(self)
        self.message = QLabel("")
        self.message.setObjectName("Caption")
        self.message.setWordWrap(True)

    def _finish(self, ok_text: str = "OK", cancel_text: str = "Cancel") -> None:
        self._outer.addWidget(self.message)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(ok_text)
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(cancel_text)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        self._outer.addWidget(buttons)
        self.buttons = buttons

    def _accept(self) -> None:
        problem = self.check()
        if problem:
            self.message.setText(problem)
            return
        self.accept()

    def check(self) -> str | None:
        return None


# ------------------------------------------------------------------ people


class SignInDialog(_Dialog):
    def __init__(self, accounts, parent: QWidget | None = None):
        super().__init__("Sentinel Vision — sign in", parent)
        self._accounts = accounts
        self._principal = None
        self._attempts = 0
        caption = QLabel("Every change you make is written to the audit trail under your name.")
        caption.setWordWrap(True)
        self._outer.addWidget(caption)
        form = QFormLayout()
        self.name = QLineEdit()
        self.password = QLineEdit()
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("Name", self.name)
        form.addRow("Password", self.password)
        self._outer.addLayout(form)
        self._finish("Sign in")
        self.name.setFocus()

    def check(self) -> str | None:
        try:
            self._principal = self._accounts.authenticate(self.name.text(), self.password.text())
        except AuthError as error:
            self._attempts += 1
            self.password.clear()
            if self._attempts >= MAX_SIGN_IN_ATTEMPTS:
                self.reject()
                return None
            return f"{error}. {MAX_SIGN_IN_ATTEMPTS - self._attempts} attempt(s) left."
        return None

    def value(self):
        return self._principal


class FirstAdminDialog(_Dialog):
    """No account exists. Offer to create the first administrator, declinably."""

    def __init__(self, accounts, parent: QWidget | None = None):
        super().__init__("Sentinel Vision — first administrator", parent)
        self._accounts = accounts
        self._principal = None
        caption = QLabel(
            "No account exists yet. Create the first administrator so every change to this site is "
            "recorded under a name. You can skip this: the console then opens with nothing gated, and "
            "the status bar will keep saying that the audit trail names nobody."
        )
        caption.setWordWrap(True)
        self._outer.addWidget(caption)
        form = QFormLayout()
        self.name = QLineEdit()
        self.name.setPlaceholderText("one word")
        self.password = QLineEdit()
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        self.confirm = QLineEdit()
        self.confirm.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("Name", self.name)
        form.addRow("Password", self.password)
        form.addRow("Again", self.confirm)
        self._outer.addLayout(form)
        self._finish("Create", "Skip for now")

    def check(self) -> str | None:
        if self.password.text() != self.confirm.text():
            self.confirm.clear()
            return "The two passwords differ."
        if len(self.password.text()) < 8:
            return "Use at least eight characters."
        from ...service.auth import Principal

        try:
            self._principal = self._accounts.add(self.name.text(), self.password.text(), Role.ADMIN,
                                                 by=Principal.open_site("first-run"))
        except AuthError as error:
            return str(error)
        return None

    def value(self):
        return self._principal


# ----------------------------------------------------------------- cameras


class AddCameraDialog(_Dialog):
    def __init__(self, parent: QWidget | None = None):
        super().__init__("Add a camera", parent)
        form = QFormLayout()
        self.identifier = QLineEdit()
        self.identifier.setPlaceholderText("north-gate")
        self.source = QLineEdit()
        self.source.setPlaceholderText("a file, device:0, or an address on the local network")
        browse = QWidget()
        row = QHBoxLayout(browse)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.source, 1)
        from PySide6.QtWidgets import QPushButton

        self.browse = QPushButton("File…")
        self.browse.clicked.connect(self._choose_file)
        self.camera = QPushButton("This laptop")
        self.camera.clicked.connect(self._choose_device)
        row.addWidget(self.browse)
        row.addWidget(self.camera)
        self.password = QLineEdit()
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        self.password.setPlaceholderText("only for a network camera; kept in the OS keychain")
        self.record = QCheckBox("Record this camera to disk")
        form.addRow("Name", self.identifier)
        form.addRow("Source", browse)
        form.addRow("Password", self.password)
        form.addRow("", self.record)
        self._outer.addLayout(form)
        hint = QLabel("The password is stored in this machine's keychain under a random handle. "
                      "It is never written to the database, the log, an export or this screen again.")
        hint.setObjectName("Caption")
        hint.setWordWrap(True)
        self._outer.addWidget(hint)
        self._finish("Add")

    def _choose_file(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(self, "Choose a video file", "", "Video (*.mp4 *.mkv *.avi *.mov);;Any file (*)")
        if chosen:
            self.source.setText(chosen)
            if not self.identifier.text():
                self.identifier.setText(Path(chosen).stem)

    def _choose_device(self) -> None:
        self.source.setText("device:0")
        if not self.identifier.text():
            self.identifier.setText("laptop")

    def check(self) -> str | None:
        if not self.identifier.text().strip():
            return "A camera needs a name."
        if not self.source.text().strip():
            return "A camera needs a source: a file, device:0, or an address."
        return None

    def value(self) -> dict:
        return {"id": self.identifier.text().strip(), "source": self.source.text().strip(),
                "password": self.password.text() or None, "record": self.record.isChecked()}


class PlaceCameraDialog(_Dialog):
    """Where a camera is and where it looks. Without this nothing can be located."""

    def __init__(self, camera_id: str, pose: CameraPose | None = None, parent: QWidget | None = None):
        super().__init__(f"Place {camera_id}", parent)
        caption = QLabel("Until a camera is placed, what it sees cannot be put on the ground — "
                         "no zone can act on it and no incident can say where.")
        caption.setWordWrap(True)
        self._outer.addWidget(caption)
        form = QFormLayout()
        self.fields: dict[str, QDoubleSpinBox] = {}
        for key, label, low, high, decimals, default in (
            ("lat", "Latitude", -90.0, 90.0, 6, 0.0), ("lon", "Longitude", -180.0, 180.0, 6, 0.0),
            ("height", "Mount height (m)", 0.5, 60.0, 2, 3.0), ("heading", "Heading (° from north)", 0.0, 359.9, 1, 0.0),
            ("pitch", "Pitch (° below level)", -89.0, 89.0, 1, -20.0), ("hfov", "Horizontal field of view (°)", 5.0, 170.0, 1, 62.0),
            ("vfov", "Vertical field of view (°)", 5.0, 170.0, 1, 36.0), ("range", "Useful range (m)", 5.0, 500.0, 0, 60.0),
        ):
            box = QDoubleSpinBox()
            box.setRange(low, high)
            box.setDecimals(decimals)
            box.setValue(default)
            self.fields[key] = box
            form.addRow(label, box)
        self._outer.addLayout(form)
        if pose is not None:
            self.set_pose(pose)
        self._finish("Place")

    def set_pose(self, pose: CameraPose) -> None:
        for key, value in (("lat", pose.position.lat), ("lon", pose.position.lon), ("height", pose.mount_height),
                           ("heading", pose.heading), ("pitch", pose.pitch), ("hfov", pose.horizontal_fov),
                           ("vfov", pose.vertical_fov), ("range", pose.range_meters)):
            self.fields[key].setValue(value)

    def check(self) -> str | None:
        try:
            self.value().validate()
        except ValueError as error:
            return str(error)
        if self.fields["pitch"].value() >= 0:
            return "A camera pointed at or above the horizon sees no ground; pitch must be below level."
        return None

    def value(self) -> CameraPose:
        f = self.fields
        return CameraPose(LatLon(f["lat"].value(), f["lon"].value()), f["height"].value(), f["heading"].value(),
                          f["pitch"].value(), 0.0, f["hfov"].value(), f["vfov"].value(), f["range"].value())


# ------------------------------------------------------------------- zones


class ZoneDialog(_Dialog):
    """What a drawn ring means: its name, kind, watch list and closed hours.

    The same dialog edits one, because the questions are the same; only the
    ring is untouchable there, since it is the part that took care to draw.
    """

    def __init__(self, ring: Sequence[LatLon], labels: Sequence[str] = (), parent: QWidget | None = None,
                 existing=None):
        super().__init__("Edit a zone" if existing is not None else "Add a zone", parent)
        self._ring = list(existing.ring) if existing is not None else list(ring)
        self._existing = existing
        self._outer.addWidget(QLabel(f"{len(self._ring)} point(s) on the ground."
                                     + (" The ring itself is not changed here." if existing is not None else "")))
        form = QFormLayout()
        self.identifier = QLineEdit()
        self.identifier.setPlaceholderText("yard")
        self.name = QLineEdit()
        self.kind = QComboBox()
        for kind in ZoneKind:
            self.kind.addItem(str(kind).title(), kind.value)
        form.addRow("Id", self.identifier)
        form.addRow("Name", self.name)
        form.addRow("Kind", self.kind)
        self._outer.addLayout(form)
        self._outer.addWidget(QLabel("Act on these classes (none ticked means every class):"))
        self.watch = QListWidget()
        self.watch.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.watch.setMaximumHeight(140)
        for label in labels:
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.watch.addItem(item)
        if not labels:
            self.watch.setEnabled(False)
            self.watch.addItem("This detector names no classes, so a class filter would silence the zone.")
        self._outer.addWidget(self.watch)
        hours = QWidget()
        row = QHBoxLayout(hours)
        row.setContentsMargins(0, 0, 0, 0)
        self.closed = QCheckBox("Presence is after-hours between")
        self.closed_from = QSpinBox()
        self.closed_from.setRange(0, 23)
        self.closed_from.setValue(22)
        self.closed_until = QSpinBox()
        self.closed_until.setRange(0, 23)
        self.closed_until.setValue(6)
        row.addWidget(self.closed)
        row.addWidget(self.closed_from)
        row.addWidget(QLabel("and"))
        row.addWidget(self.closed_until)
        row.addStretch(1)
        self._outer.addWidget(hours)
        if existing is not None:
            self.identifier.setText(existing.id)
            self.identifier.setReadOnly(True)
            self.name.setText(existing.name)
            self.kind.setCurrentIndex(max(0, self.kind.findData(existing.kind.value)))
            for index in range(self.watch.count()):
                item = self.watch.item(index)
                if item.text().lower() in existing.watch:
                    item.setCheckState(Qt.CheckState.Checked)
            if existing.schedule is not None:
                self.closed.setChecked(True)
                self.closed_from.setValue(existing.schedule.closed_from)
                self.closed_until.setValue(existing.schedule.closed_until)
        self._finish("Save" if existing is not None else "Add zone")

    def check(self) -> str | None:
        if len(self._ring) < 3:
            return "A zone needs at least three points."
        if not self.identifier.text().strip():
            return "A zone needs an id."
        return None

    def value(self) -> dict:
        watched = []
        if self.watch.isEnabled():
            watched = [self.watch.item(i).text() for i in range(self.watch.count())
                       if self.watch.item(i).checkState() == Qt.CheckState.Checked]
        schedule = Schedule(self.closed_from.value(), self.closed_until.value()) if self.closed.isChecked() else None
        return {"id": self.identifier.text().strip(), "name": self.name.text().strip() or self.identifier.text().strip(),
                "kind": self.kind.currentData(), "ring": self._ring, "watch": watched, "schedule": schedule}


class PasswordDialog(_Dialog):
    def __init__(self, camera_id: str, parent: QWidget | None = None):
        super().__init__(f"Password for {camera_id}", parent)
        self._outer.addWidget(QLabel("Kept in this machine's keychain under a random handle, never in the database."))
        form = QFormLayout()
        self.password = QLineEdit()
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("Password", self.password)
        self._outer.addLayout(form)
        self._finish("Store")

    def check(self) -> str | None:
        return None if self.password.text() else "A password cannot be empty."

    def value(self) -> str:
        return self.password.text()


class NoteDialog(_Dialog):
    """One line of why. Required where a judgement without a reason means nothing."""

    def __init__(self, title: str, prompt: str, parent: QWidget | None = None):
        super().__init__(title, parent)
        caption = QLabel(prompt)
        caption.setWordWrap(True)
        self._outer.addWidget(caption)
        self.note = QLineEdit()
        self.note.setMaxLength(500)
        self._outer.addWidget(self.note)
        self._finish("Save")
        self.note.setFocus()

    def check(self) -> str | None:
        return None if self.note.text().strip() else "A reason is required."

    def value(self) -> str:
        return self.note.text().strip()
