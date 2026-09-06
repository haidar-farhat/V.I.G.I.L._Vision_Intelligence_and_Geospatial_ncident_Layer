"""The console window: layout, the lock, the repaint timer, and nothing else.

It holds no domain state. Every change goes through `Commands`; every draw
comes from `Runtime.poll()` and the service's own readers. Three rules from
v1 are structural here and are asserted by tests:

* a connection is a **bound method**, never a lambda closing over ``self`` —
  such a lambda is a reference cycle holding a `QWidget`, and the widget is
  then destroyed after the `QApplication`, which is a shutdown crash;
* a dialog is opened only through `dialogs.ask`, which reads it and *then*
  deletes it;
* a greyed control still answers a click, with the reason it is greyed —
  the operator's report of v1 was "the buttons do nothing".
"""

from __future__ import annotations

import weakref
from pathlib import Path
from typing import Sequence

from PySide6.QtCore import QEvent, QTimer, Qt
from PySide6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton, QSplitter, QTabWidget,
    QVBoxLayout, QWidget,
)

from ...domain.geo import CameraPose
from ...service.auth import ANALYSIS_CONTROL, INCIDENT_EXPORT, SITE_CONFIGURE
from ...logs import get as _get_logger
from . import dialogs, theme
from .commands import Commands
from .plan import PlanView
from .video import VideoView
from .widgets import AuditView, CameraList, ElidingLabel, IncidentDetail, IncidentList, TrackTable

_log = _get_logger(__name__)

POLL_MILLIS = 100
#: How long the console stays unlocked with nobody touching it. A lock that
#: never re-arms is a lock somebody props open on the first day.
IDLE_RELOCK_MILLIS = 10 * 60 * 1000
LOCKED_REASON = "Locked. Press Configure to change the site — cameras, placement and zones."
#: The share of the status bar one permanent label may take, so the message
#: it sits beside keeps at least a third of the bar.
STATUS_LABEL_SHARE = 0.13


def _panel(title: str, body: QWidget) -> QFrame:
    frame = QFrame()
    frame.setObjectName("Panel")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(1, 1, 1, 1)
    layout.setSpacing(0)
    label = QLabel(title)
    label.setObjectName("PanelTitle")
    layout.addWidget(label)
    layout.addWidget(body, 1)
    return frame


class ConsoleWindow(QMainWindow):
    def __init__(self, commands: Commands, *, parent: QWidget | None = None):
        super().__init__(parent)
        self.commands = commands
        self._views: dict[str, VideoView] = {}
        self._configuring = False
        self._guarded: set[QWidget] = set()
        self._selected_camera: str | None = None
        self._beeped: set[tuple[str, str]] = set()
        self.setWindowTitle("Sentinel Vision")
        self.resize(1440, 900)
        self.setStyleSheet(theme.stylesheet())

        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)
        outer.addLayout(self._build_toolbar())

        self.camera_list = CameraList()
        self.camera_list.selected.connect(self._camera_selected)
        self.camera_list.record_toggled.connect(self._record_toggled)
        self.wall = QWidget()
        self._wall_grid = QGridLayout(self.wall)
        self._wall_grid.setContentsMargins(2, 2, 2, 2)
        self._wall_grid.setSpacing(4)
        self.plan = PlanView()

        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(_panel("CAMERAS", self.camera_list))
        top.addWidget(_panel("CAMERA WALL", self.wall))
        top.addWidget(_panel("GROUND — NO EXTERNAL TILES", self.plan))
        top.setStretchFactor(0, 1)
        top.setStretchFactor(1, 3)
        top.setStretchFactor(2, 2)
        # Not collapsible: the list is the only place a dark camera announces
        # itself, and a handle dragged shut would hide that for good.
        top.setCollapsible(0, False)
        self._top = top
        self._dealt = False

        self.incidents = IncidentList()
        self.incidents.selected.connect(self._incident_selected)
        self.tracks = TrackTable()
        self.detail = IncidentDetail()
        self.audit = AuditView()
        self.tabs = QTabWidget()
        self.tabs.addTab(self.detail, "Why")
        self.tabs.addTab(self.tracks, "Tracked objects")
        self.tabs.addTab(self.audit, "Audit trail")
        self.tabs.currentChanged.connect(self._tab_changed)

        lower = QSplitter(Qt.Orientation.Horizontal)
        lower.addWidget(_panel("INCIDENTS", self.incidents))
        lower.addWidget(self.tabs)
        lower.setStretchFactor(0, 3)
        lower.setStretchFactor(1, 2)
        lower.setMinimumHeight(240)

        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(top)
        split.addWidget(lower)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        # The conclusions must never be squeezed out of sight.
        split.setCollapsible(1, False)
        outer.addWidget(split, 1)
        self.setCentralWidget(central)

        self.status = self.statusBar()
        self.user_label = ElidingLabel("")
        self.lock_label = ElidingLabel("")
        self.placement_label = ElidingLabel("")
        self.detector_label = ElidingLabel("")
        self.alert_label = ElidingLabel("")
        self.alert_label.setStyleSheet(f"color: {theme.FAULT.name()}; font-weight: 600;")
        for label in self._permanent_labels():
            self.status.addPermanentWidget(label)

        self._timer = QTimer(self)
        self._timer.setInterval(POLL_MILLIS)
        self._timer.timeout.connect(self._collect)
        self._idle = QTimer(self)
        self._idle.setSingleShot(True)
        self._idle.setInterval(IDLE_RELOCK_MILLIS)
        self._idle.timeout.connect(self._relock)

        self._show_principal()
        self._set_configuring(False)
        self.refresh_site()
        self._say("Ready." if self.commands.cameras() else "Ready. Add a camera to begin.")

    # ------------------------------------------------------------- building

    def _build_toolbar(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.configure_button = QPushButton("Configure")
        self.configure_button.setCheckable(True)
        self.configure_button.setToolTip("Unlock the controls that change the site.")
        self.configure_button.toggled.connect(self._set_configuring)
        row.addWidget(self.configure_button)
        row.addSpacing(10)

        self.add_button = QPushButton("Add camera…")
        self.add_button.clicked.connect(self._add_camera)
        self.place_button = QPushButton("Place…")
        self.place_button.clicked.connect(self._place_camera)
        self.password_button = QPushButton("Password…")
        self.password_button.clicked.connect(self._set_password)
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self._remove_camera)
        self.zone_button = QPushButton("Draw zone")
        self.zone_button.setCheckable(True)
        self.zone_button.toggled.connect(self._draw_zone)
        self.drop_zone_button = QPushButton("Delete zone")
        self.drop_zone_button.clicked.connect(self._remove_zone)
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self._start)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self._stop)
        self.export_button = QPushButton("Export evidence…")
        self.export_button.clicked.connect(self._export)
        for button in (self.add_button, self.place_button, self.password_button, self.remove_button,
                       self.zone_button, self.drop_zone_button):
            row.addWidget(button)
        row.addSpacing(10)
        for button in (self.start_button, self.stop_button, self.export_button):
            row.addWidget(button)
        row.addStretch(1)
        about = QPushButton("About")
        about.clicked.connect(self._about)
        row.addWidget(about)
        return row

    def _configure_only(self) -> Sequence[QWidget]:
        return (self.add_button, self.place_button, self.password_button, self.remove_button,
                self.zone_button, self.drop_zone_button)

    # ----------------------------------------------------------- the lock

    def _set_configuring(self, on: bool) -> None:
        if on and not self.commands.may(SITE_CONFIGURE):
            # Permission, not the lock. The refusal is audited under the name.
            self.configure_button.setChecked(False)
            self._say(self.commands.refusal(SITE_CONFIGURE))
            return
        was, self._configuring = self._configuring, bool(on)
        for control in self._configure_only():
            control.setEnabled(self._configuring)
            control.setToolTip("" if self._configuring else LOCKED_REASON)
            if control not in self._guarded:
                # A disabled widget still runs its event filters, so a click on
                # a greyed control can be answered. See `eventFilter`.
                control.installEventFilter(self)
                self._guarded.add(control)
        self.camera_list.set_recording_editable(self._configuring)
        self.lock_label.setText("CONFIGURE — the site can be changed" if self._configuring
                                else "MONITOR — site locked; press Configure to change it")
        self.lock_label.setStyleSheet(f"color: {theme.STALE.name()}; font-weight: 600;" if self._configuring
                                      else f"color: {theme.TEXT_MUTED.name()};")
        if self._configuring:
            self._idle.start()
        else:
            self._idle.stop()
            if self.zone_button.isChecked():
                self.zone_button.setChecked(False)
        if was != self._configuring:
            self.commands.note("console.configure.entered" if self._configuring else "console.configure.left",
                               "unlocked the controls that change the site" if self._configuring else "re-locked")
        self._fit_labels()

    def _relock(self) -> None:
        if self._configuring:
            self.configure_button.setChecked(False)
            self._say("Re-locked after ten minutes with nothing touched.")

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt's name
        if event.type() == QEvent.Type.MouseButtonPress and isinstance(watched, QWidget) and not watched.isEnabled():
            self._say(LOCKED_REASON)
            return True
        return super().eventFilter(watched, event)

    # ------------------------------------------------------------ commands

    def _add_camera(self) -> None:
        ok, value = dialogs.ask(dialogs.AddCameraDialog(self))
        if not ok or value is None:
            return
        source = value["source"]
        outcome = self.commands.add_camera(value["id"], source, record=value["record"])
        if outcome and value["password"]:
            outcome = self.commands.set_password(value["id"], value["password"])
        self._say(outcome.message)
        if outcome:
            self.refresh_site()

    def _place_camera(self) -> None:
        camera = self._current_camera()
        if camera is None:
            self._say("Select a camera to place.")
            return
        ok, pose = dialogs.ask(dialogs.PlaceCameraDialog(camera.id, camera.pose, self))
        if not ok or pose is None:
            return
        self._say(self.commands.place_camera(camera.id, pose).message)
        self.refresh_site()

    def _set_password(self) -> None:
        camera = self._current_camera()
        if camera is None:
            self._say("Select a camera.")
            return
        ok, password = dialogs.ask(dialogs.PasswordDialog(camera.id, self))
        if ok and password:
            self._say(self.commands.set_password(camera.id, password).message)

    def _remove_camera(self) -> None:
        camera = self._current_camera()
        if camera is None:
            self._say("Select a camera to remove.")
            return
        answer = QMessageBox.question(self, f"Remove {camera.id}?",
                                      "The camera is forgotten and its password removed from the keychain. "
                                      "What it saw — its events and incidents — is kept.")
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._say(self.commands.remove_camera(camera.id).message)
        self.refresh_site()

    def _draw_zone(self, on: bool) -> None:
        if on:
            self.plan.begin_zone()
            self.zone_button.setText("Finish zone")
            self._say("Click the plan to place the corners; three or more, then Finish zone.")
            return
        self.zone_button.setText("Draw zone")
        ring = self.plan.end_zone()
        if len(ring) < 3:
            if ring:
                self._say("A zone needs at least three points; nothing was added.")
            return
        labels = self._detector_labels()
        ok, value = dialogs.ask(dialogs.ZoneDialog(ring, labels, self))
        if not ok or value is None:
            return
        outcome = self.commands.add_zone(value["id"], value["name"], value["kind"], value["ring"],
                                         watch=value["watch"], schedule=value["schedule"])
        self._say(outcome.message)
        self.refresh_site()

    def _remove_zone(self) -> None:
        zones = self.commands.zones()
        if not zones:
            self._say("There is no zone to delete.")
            return
        from PySide6.QtWidgets import QInputDialog

        names = [f"{z.id} — {z.name}" for z in zones]
        chosen, ok = QInputDialog.getItem(self, "Delete a zone", "Zone", names, 0, False)
        if not ok or not chosen:
            return
        self._say(self.commands.remove_zone(chosen.split(" — ")[0]).message)
        self.refresh_site()

    def _start(self) -> None:
        outcome = self.commands.start()
        self._say(outcome.message)
        if not outcome:
            return
        self._rebuild_wall()
        self._timer.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.add_button.setEnabled(False)

    def _stop(self) -> None:
        self._timer.stop()
        outcome = self.commands.stop()
        self._collect(final=True)
        self._say(outcome.message)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.add_button.setEnabled(self._configuring)
        for view in self._views.values():
            view.set_caption("stopped")

    def _export(self) -> None:
        incident = self.incidents.selected_incident()
        if incident is None:
            self._say("Select an incident to export.")
            return
        outcome = self.commands.export(incident)
        self._say(outcome.message)
        if outcome:
            QMessageBox.information(self, "Evidence exported", outcome.message)

    def _about(self) -> None:
        from ...version import describe

        QMessageBox.information(self, "About this build",
                                f"{describe()}\n\nNative Qt widgets — no embedded browser.\n"
                                "No Internet access at any point: no tiles, no telemetry, no model downloads.\n\n"
                                f"Signed in as {self.commands.principal.actor}.")

    # ------------------------------------------------------------ refresh

    def refresh_site(self) -> None:
        cameras = self.commands.cameras()
        self.camera_list.show_cameras(cameras, self.commands.health())
        self.plan.set_cameras({c.id: c.pose for c in cameras if c.pose is not None})
        self.plan.set_zones(self.commands.zones())
        placed = [c for c in cameras if c.placed]
        if not cameras:
            self.placement_label.setText("No camera yet")
        elif not placed:
            self.placement_label.setText("No camera placed — nothing can be located")
        else:
            self.placement_label.setText(f"{len(placed)} of {len(cameras)} camera(s) placed")
        self._show_detector()
        self.start_button.setEnabled(bool(cameras) and not self.commands.runtime.running)
        self.stop_button.setEnabled(self.commands.runtime.running)
        self._fit_labels()

    def _rebuild_wall(self) -> None:
        for view in self._views.values():
            view.setParent(None)
            view.deleteLater()
        self._views.clear()
        cameras = [c for c in self.commands.cameras()]
        columns = max(1, int(len(cameras) ** 0.5 + 0.999))
        for index, camera in enumerate(cameras):
            view = VideoView(camera.id)
            self._views[camera.id] = view
            self._wall_grid.addWidget(view, index // columns, index % columns)

    def _collect(self, final: bool = False) -> None:
        results = self.commands.poll()
        health = self.commands.health()
        rows = []
        for result in results:
            view = self._views.get(result.camera_id)
            state = health.get(result.camera_id)
            if view is not None:
                view.show_result(result, state.analysis_fps if state else 0.0)
            self.plan.set_tracks(result.camera_id, result.tracks)
            info = self._detector_info(result.camera_id)
            if view is not None:
                view.set_detector_info(info)
            rows.extend((result.camera_id, track, info) for track in result.tracks)
        if rows or results:
            self.tracks.show_tracks(rows)
        if results:
            self._show_detector()
        self.camera_list.show_cameras(self.commands.cameras(), health)
        self.incidents.show_incidents(self.commands.incidents())
        self._show_alerts()
        self.export_button.setEnabled(bool(self.incidents.selected_incident()) and self.commands.may(INCIDENT_EXPORT))
        if self.tabs.currentWidget() is self.audit:
            self.audit.show_rows(self.commands.audit_rows(limit=200))
        if not final and self._timer.isActive() and not self.commands.runtime.running:
            # Every camera ended on its own: for files that is completion. One
            # more drain first, so the last clip and events are never lost.
            self._timer.stop()
            self._collect(final=True)
            self._say("Finished.")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)

    def _show_alerts(self) -> None:
        alerts = self.commands.alerts()
        fresh = alerts.take_new()
        if fresh:
            from PySide6.QtWidgets import QApplication

            QApplication.beep()
        active = alerts.active()
        if not active:
            self.alert_label.setText("")
            return
        newest = active[-1]
        more = f" (+{len(active) - 1} more)" if len(active) > 1 else ""
        self.alert_label.setText(f"ALERT {newest.kind} {newest.subject}: {newest.detail}{more}")

    def _detector_info(self, camera_id: str):
        return self.commands.detector_info(camera_id)

    def _show_detector(self) -> None:
        """What is drawing the conclusions, and what it cannot do.

        An operator reading "does not classify" beside a track with no class
        learns something; one reading a model name beside the same track
        would assume the model looked and found nothing recognisable, which
        is the opposite of true. Before the run it says what *will* be used,
        because the choice is made before Start and a silent fall-back to
        motion detection is the defect this line exists to prevent.
        """
        info = next((i for i in (self._detector_info(c) for c in self._views) if i is not None), None)
        if info is None:
            model = self.commands.model
            self.detector_label.setText(f"Will use {model.name}" if model is not None
                                        else "No model found — motion only, which cannot classify")
            self.detector_label.setStyleSheet("" if model is not None else f"color: {theme.STALE.name()};")
            self.detector_label.setToolTip("" if model is not None else
                                           "Looked in: " + ", ".join(str(p) for p in self.commands.model_places))
            return
        if not info.classifies:
            self.detector_label.setText(f"{info.name} — does not classify, and cannot see a stationary object")
            self.detector_label.setStyleSheet(f"color: {theme.STALE.name()};")
            return
        names = sorted(set(info.class_names.values()))
        watching = ", ".join(names) if len(names) <= 8 else f"{len(names)} classes"
        digest = f" · {info.model_sha256[:12]}" if info.model_sha256 else ""
        masks = " with masks" if info.kind.endswith("segment") else ", boxes only"
        self.detector_label.setText(f"{info.name} — watching {watching}{masks}{digest}")
        self.detector_label.setStyleSheet("")

    def _detector_labels(self) -> list[str]:
        for camera_id in self._views:
            info = self._detector_info(camera_id)
            if info is not None and info.classifies:
                return sorted(set(info.class_names.values()))
        return []

    # ------------------------------------------------------------ plumbing

    def _current_camera(self):
        chosen = self.camera_list.selected_camera()
        if chosen is None:
            return None
        return next((c for c in self.commands.cameras() if c.id == chosen), None)

    def _camera_selected(self, camera_id) -> None:
        self._selected_camera = camera_id
        self.plan.select(camera_id)

    def _incident_selected(self, incident) -> None:
        self.export_button.setEnabled(incident is not None and self.commands.may(INCIDENT_EXPORT))
        self.detail.show_incident(incident)
        if incident is not None:
            self.tabs.setCurrentWidget(self.detail)

    def _tab_changed(self, index: int) -> None:
        if self.tabs.widget(index) is self.audit:
            self.audit.show_rows(self.commands.audit_rows(limit=200))

    def _record_toggled(self, camera_id: str, on: bool) -> None:
        self._say(self.commands.set_recording(camera_id, on).message)
        self._idle.start()

    def _show_principal(self) -> None:
        principal = self.commands.principal
        if principal.origin == "user":
            self.user_label.setText(f"{principal.name} · {str(principal.role).lower()}")
            self.user_label.setStyleSheet("")
        else:
            self.user_label.setText("no account — the audit trail names nobody")
            self.user_label.setStyleSheet(f"color: {theme.STALE.name()};")
            self.user_label.setToolTip("Create the first administrator with `vigil users add NAME --role ADMIN`.")

    def _say(self, text: str) -> None:
        if text:
            self.status.showMessage(text, 12_000)

    def _permanent_labels(self):
        return (self.user_label, self.lock_label, self.placement_label, self.detector_label, self.alert_label)

    def _fit_labels(self) -> None:
        cap = int((self.status.width() or self.width()) * STATUS_LABEL_SHARE)
        for label in self._permanent_labels():
            label.set_cap(cap)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt's name
        super().resizeEvent(event)
        if hasattr(self, "lock_label"):
            self._fit_labels()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt's name
        super().showEvent(event)
        if not self._dealt:
            self._dealt = True
            # Stretch factors divide only what is left beyond the size hints,
            # which at 1280 px left the camera list too narrow for its columns.
            width = max(self._top.width(), self.width())
            first = int(width * 0.30)
            rest = width - first
            self._top.setSizes([first, int(rest * 0.6), rest - int(rest * 0.6)])
        self._fit_labels()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt's name
        self._timer.stop()
        try:
            if self.commands.runtime.running:
                self.commands.stop()
            else:
                self.commands.poll()
        except Exception:  # noqa: BLE001 - closing must not fail
            _log.exception("console: stopping on close failed")
        super().closeEvent(event)
