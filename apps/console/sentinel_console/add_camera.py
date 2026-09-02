"""Choosing what a camera actually is.

Three kinds of source, and they are genuinely different things rather than three
spellings of one:

- **A camera attached to this machine.** Found through the operating system's
  own device interface — the PnP registry on Windows, the V4L2 tree on Linux,
  the system profiler on macOS — and opened through that platform's native
  capture API. See `sentinel.devices`.
- **A camera on the network.** An RTSP URL, which usually carries a password.
- **A video file.** Not a camera at all, and the only one of the three that is
  *evidence*: every frame is processed in order, so replaying it reproduces the
  original result exactly.

Two things this dialog is careful about.

**It does not switch a camera on to list them.** Enumeration reads metadata and
captures nothing, so opening this dialog does not light the webcam light or —
on macOS — trigger a permission prompt for a camera nobody asked to use.
*Detect* is a separate button because opening a camera is a deliberate act.

**It does not pretend to know which camera is which.** The operating system
knows the *names*; OpenCV opens by *index*; there is no supported mapping
between them, so the pairing is an assumption and the list says so. Two
identical webcams cannot be told apart any other way than by looking at the
picture, which is what the camera wall is for.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from sentinel import devices, logs
from sentinel.decode import redact_url

from . import theme

_log = logs.get(__name__)


@dataclass(frozen=True, slots=True)
class ChosenSource:
    """What the operator picked.

    ``source`` is what :class:`~sentinel.decode.VideoSource` is given and may
    carry a credential; ``display`` never does and is what goes on screen, in
    the log and in the database.
    """

    source: str
    display: str
    suggested_id: str

    @property
    def is_device(self) -> bool:
        return devices.is_device_source(self.source)


def _hint(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet(f"color: {theme.TEXT_MUTED.name()}; font-size: 11px;")
    return label


class AddCameraDialog(QDialog):
    """Pick a local camera, a network camera, or a file."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Add camera")
        self.setStyleSheet(theme.STYLESHEET)
        self.setMinimumWidth(560)

        self._chosen: list[ChosenSource] = []
        self._cameras: list[devices.LocalCamera] = []

        layout = QVBoxLayout(self)
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_local(), "This machine")
        self._tabs.addTab(self._build_network(), "Network camera")
        self._tabs.addTab(self._build_file(), "Video file")
        layout.addWidget(self._tabs)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.accepted.connect(self._accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self._tabs.currentChanged.connect(lambda _: self._refresh_ok())
        # Listing reads metadata only, so it is safe to do as the dialog opens.
        self._list_devices()

    # ------------------------------------------------------------- this machine

    def _build_local(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        layout.addWidget(
            _hint(
                f"Cameras attached to this machine, as {devices.preferred_backend()} "
                "reports them. Listing does not switch any of them on."
            )
        )

        self._device_list = QListWidget()
        self._device_list.setSelectionMode(
            QListWidget.SelectionMode.ExtendedSelection
        )
        self._device_list.itemSelectionChanged.connect(self._refresh_ok)
        layout.addWidget(self._device_list)

        row = QHBoxLayout()
        self._detect = QPushButton("Detect")
        self._detect.setToolTip(
            "Opens each camera briefly to confirm which index is which and what "
            "resolution it gives. This switches the cameras on, which is why it "
            "is a button rather than something the dialog does by itself."
        )
        self._detect.clicked.connect(self._probe_devices)
        row.addWidget(self._detect)

        refresh = QPushButton("Refresh list")
        refresh.clicked.connect(self._list_devices)
        row.addWidget(refresh)
        row.addStretch(1)
        layout.addLayout(row)

        self._device_note = _hint("")
        layout.addWidget(self._device_note)
        return page

    def _list_devices(self) -> None:
        """Ask the operating system. Opens nothing."""
        self._device_list.clear()
        try:
            self._cameras = devices.discover(probe_indices=False)
        except Exception as error:  # noqa: BLE001
            # A device subsystem that cannot be queried is not a reason for the
            # dialog to fail: the other two tabs still work.
            _log.warning("could not enumerate cameras: %s", type(error).__name__)
            self._cameras = []

        for camera in self._cameras:
            item = QListWidgetItem(camera.label)
            item.setData(Qt.ItemDataRole.UserRole, camera)
            self._device_list.addItem(item)

        self._describe_devices()
        self._refresh_ok()

    def _probe_devices(self) -> None:
        """Open each camera briefly, and report what actually happened."""
        self._detect.setEnabled(False)
        self._device_note.setText("Opening each camera…")
        # Painted before the probe, which blocks: otherwise the operator sees a
        # frozen dialog and no explanation for it.
        self.repaint()

        try:
            self._cameras = devices.discover(probe_indices=True)
        except Exception as error:  # noqa: BLE001
            _log.warning("probing cameras failed: %s", type(error).__name__)
            self._cameras = []
        finally:
            self._detect.setEnabled(True)

        self._device_list.clear()
        for camera in self._cameras:
            item = QListWidgetItem(camera.label)
            item.setData(Qt.ItemDataRole.UserRole, camera)
            self._device_list.addItem(item)

        self._describe_devices(probed=True)
        self._refresh_ok()

    def _describe_devices(self, *, probed: bool = False) -> None:
        if not self._cameras:
            self._device_note.setText(
                "No cameras. The operating system reports none attached. Some "
                "cameras are not listed by the device registry but do open — "
                "press Detect to look."
                if not probed
                else "No camera on this machine would open. One may be in use by "
                "another application, or blocked by a privacy setting."
            )
            return

        unconfirmed = [c for c in self._cameras if not c.index_confirmed]
        if unconfirmed:
            self._device_note.setText(
                "An index marked assumed has not been opened, so it is this "
                "machine's enumeration order rather than a fact. Press Detect to "
                "confirm it — or add the camera and press Start, and check the "
                "picture. Two identical cameras cannot be told apart any other way."
            )
        else:
            self._device_note.setText(
                f"{len(self._cameras)} camera(s), each opened and confirmed."
            )

    # ---------------------------------------------------------- network camera

    def _build_network(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        layout.addWidget(_hint("An RTSP URL, including the credential if the camera needs one."))

        self._url = QLineEdit()
        self._url.setPlaceholderText(
            "rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101"
        )
        self._url.textChanged.connect(self._on_url_changed)
        layout.addWidget(self._url)

        self._url_preview = _hint("")
        layout.addWidget(self._url_preview)

        layout.addWidget(
            _hint(
                "The address must be on this network — loopback or a private "
                "range. A camera name that resolves to a public address is "
                "refused, because this system never reaches the Internet and "
                "will not resolve a name to find out whether it is allowed to."
            )
        )
        layout.addWidget(
            _hint(
                "The password is used once, at the moment of connection, and is "
                "held in memory only. It is never written to the database, the "
                "log, an export or this screen — and it is not saved, so it has "
                "to be entered again after a restart. Keychain storage is "
                "designed and not built; see STATUS.md."
            )
        )
        layout.addStretch(1)
        return page

    def _on_url_changed(self, text: str) -> None:
        # Shows the operator exactly what everything downstream will see, so the
        # redaction is something they can verify rather than take on trust.
        self._url_preview.setText(
            f"Stored and displayed as:  {redact_url(text.strip())}" if text.strip() else ""
        )
        self._refresh_ok()

    # ------------------------------------------------------------- video file

    def _build_file(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        layout.addWidget(
            _hint(
                "A file is the only source that is evidence: every frame is "
                "processed in order, so replaying it reproduces the original "
                "result exactly."
            )
        )

        row = QHBoxLayout()
        self._file = QLineEdit()
        self._file.setPlaceholderText("No file chosen")
        self._file.textChanged.connect(lambda _: self._refresh_ok())
        row.addWidget(self._file, 1)

        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row.addWidget(browse)
        layout.addLayout(row)

        layout.addStretch(1)
        return page

    def _browse(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add one or more cameras",
            str(Path.home()),
            "Video (*.mp4 *.mkv *.avi *.mov *.m4v);;All files (*)",
        )
        if paths:
            self._file.setText("; ".join(paths))

    # -------------------------------------------------------------- the answer

    def _refresh_ok(self) -> None:
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(
            bool(self._current_choices())
        )

    def _current_choices(self) -> list[ChosenSource]:
        page = self._tabs.currentIndex()

        if page == 0:
            chosen = []
            for item in self._device_list.selectedItems():
                camera: devices.LocalCamera = item.data(Qt.ItemDataRole.UserRole)
                chosen.append(
                    ChosenSource(
                        source=camera.source,
                        display=camera.source,
                        # The operating system's name, not "device:0". An
                        # operator picked "Logitech C920" and should see it.
                        suggested_id=_identifier_from(camera.name) or camera.source,
                    )
                )
            return chosen

        if page == 1:
            url = self._url.text().strip()
            if "://" not in url:
                return []
            display = redact_url(url)
            return [
                ChosenSource(
                    source=url,
                    display=display,
                    suggested_id=_identifier_from(_host_of(display)) or "camera",
                )
            ]

        text = self._file.text().strip()
        if not text:
            return []
        return [
            ChosenSource(source=part, display=part, suggested_id=Path(part).stem or "clip")
            for part in (piece.strip() for piece in text.split(";"))
            if part
        ]

    def _accept(self) -> None:
        self._chosen = self._current_choices()
        if self._chosen:
            self.accept()

    @property
    def chosen(self) -> list[ChosenSource]:
        return list(self._chosen)


def _identifier_from(name: str) -> str:
    """A camera id an operator would recognise, from a device or host name."""
    cleaned = "".join(character if character.isalnum() else "-" for character in name.lower())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-")[:32]


def _host_of(display_url: str) -> str:
    netloc = display_url.partition("://")[2].partition("/")[0]
    netloc = netloc.rpartition("@")[2] or netloc
    return netloc.partition(":")[0] if netloc.count(":") == 1 else netloc
