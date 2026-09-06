"""The console window: layout, the lock, the repaint timer, and nothing else.

It holds no domain state. Every change goes through `Commands`; every draw
comes from one `Commands.snapshot()` and `Runtime.poll()`. Three rules from v1
are structural here and are asserted by tests:

* a connection is a **bound method**, never a lambda closing over ``self`` —
  such a lambda is a reference cycle holding a `QWidget`, and the widget is
  then destroyed after the `QApplication`, which is a shutdown crash;
* a dialog is opened only through `dialogs.ask`, which reads it and *then*
  deletes it;
* a greyed control still answers a click, with the reason it is greyed —
  the operator's report of v1 was "the buttons do nothing".

# The layout, and what was wrong with the one before it

**Every verb sits under the noun it acts on.** Camera actions are beneath the
camera list, zone actions beneath the plan, incident actions beneath the
incident list. Before, all seventeen were in one wrapping row across the top,
which cost two things:

* an operator had to read the whole row to find one control, because a camera
  action, a zone action, run control and *About* were all the same size and
  colour; and
* five of them acted on "the selected camera" from the far side of the
  window, so the most common outcome of pressing one was the sentence
  **"Select a camera to place."** — the interface asking for something it
  could see. With the button beside the list, that error is unreachable.

**One control is loud.** Start, and Stop when it is running. Everything else
is grey: a window where three things shout is a window where nothing does.

**The status bar carries three things, not seven.** It held site, principal,
lock, placement, ground, detector and alert, each capped at eleven per cent of
its width, which rendered them *"MONITOR — site locked; press…"* and
*"yolov8n-seg — watching 80 cl…"*. A sentence nobody can read is not on
screen. Those lines now sit in the heading of the panel they describe, where
there is room for them, and the bar keeps the message, the principal and the
alert.

**Rare things are in menus.** About, the identity switch and the detection
settings are opened once a month and were competing with Acknowledge.
"""

from __future__ import annotations

from typing import Sequence

from PySide6.QtCore import QEvent, QTimer, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox, QGridLayout, QMainWindow, QPushButton, QSplitter,
    QTabWidget, QVBoxLayout, QWidget,
)

from ...service.auth import INCIDENT_EXPORT, INCIDENT_REVIEW, SITE_CONFIGURE
from ...logs import get as _get_logger
from . import theme
from .actions import ConsoleActions
from .commands import Commands
from .plan import PlanView
from .video import VideoView
from .widgets import (
    ActionBar, AuditView, CameraList, ElidingLabel, FlowLayout, IncidentDetail, IncidentList,
    Panel, TrackTable,
)

_log = _get_logger(__name__)

POLL_MILLIS = 100
#: How long the console stays unlocked with nobody touching it. A lock that
#: never re-arms is a lock somebody props open on the first day.
IDLE_RELOCK_MILLIS = 10 * 60 * 1000
LOCKED_REASON = "Locked. Press Configure to change the site — cameras, placement and zones."
#: The share of the status bar one permanent label may take. Three labels
#: rather than seven, so each gets room to be read.
STATUS_LABEL_SHARE = 0.22


class ConsoleWindow(ConsoleActions, QMainWindow):
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

        self._build_actions()
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)
        outer.addLayout(self._build_toolbar())
        outer.addWidget(self._build_body(), 1)
        self.setCentralWidget(central)
        self._build_menus()
        self._build_status()

        self._timer = QTimer(self)
        self._timer.setInterval(POLL_MILLIS)
        self._timer.timeout.connect(self._collect)
        self._idle = QTimer(self)
        self._idle.setSingleShot(True)
        self._idle.setInterval(IDLE_RELOCK_MILLIS)
        self._idle.timeout.connect(self._relock)

        self._show_site()
        self._show_principal()
        self._set_configuring(False)
        self.refresh_site()
        self._say("Ready." if self.commands.cameras() else "Ready. Add a camera to begin.")

    # ------------------------------------------------------------- building

    #: The narrowest window this layout is designed to stay readable in. A
    #: 1366-wide laptop is the machine this runs on in the places it runs.
    NARROWEST_WINDOW = 1280

    def _button(self, text: str, slot, *, tip: str = "", shortcut: str = "",
                checkable: bool = False, name: str = "") -> QPushButton:
        """One button, with its tooltip and its key.

        The shortcut is appended to the tooltip rather than left for somebody
        to discover: a key nobody is told about is a key nobody presses.
        """
        button = QPushButton(text)
        if name:
            button.setObjectName(name)
        button.setCheckable(checkable)
        (button.toggled if checkable else button.clicked).connect(slot)
        if shortcut:
            button.setShortcut(QKeySequence(shortcut))
            tip = f"{tip}  ({shortcut})" if tip else shortcut
        if tip:
            button.setToolTip(tip)
        return button

    def _build_actions(self) -> None:
        """Every control, built once, before anything is laid out."""
        # Run control and the lock: the whole-window verbs.
        self.start_button = self._button("▶  Start", self._start, name="Primary",
                                         tip="Begin analysing every camera.", shortcut="F5")
        self.stop_button = self._button("■  Stop", self._stop, name="Stop",
                                        tip="Stop analysing. Recordings are closed cleanly.",
                                        shortcut="F6")
        self.configure_button = self._button(
            "Configure", self._set_configuring, checkable=True, shortcut="Ctrl+K",
            tip="Unlock the controls that change the site.")

        # Cameras: under the camera list.
        self.add_button = self._button("Add…", self._add_camera, shortcut="Ctrl+N",
                                       tip="Add a camera by file, device or address.")
        self.place_button = self._button("Place…", self._place_camera,
                                         tip="Where this camera is and where it looks.")
        self.calibrate_button = self._button(
            "Measure pose…", self._calibrate_camera,
            tip="Replace this camera's assumed accuracy with a measured one, by marking points "
                "you can find both in the picture and on the plan. At 40 m, the two degrees a "
                "placed camera assumes is 1.4 m of sideways error before the detector has "
                "contributed anything.")
        self.edit_camera_button = self._button("Edit…", self._edit_camera,
                                               tip="Rename this camera, or point it at a new address.")
        self.password_button = self._button("Password…", self._set_password,
                                            tip="Store this camera's password in the keychain.")
        self.remove_button = self._button("Remove", self._remove_camera,
                                          tip="Forget this camera. What it saw is kept.")

        # Zones: under the plan they are drawn on.
        self.zone_button = self._button("Draw zone", self._draw_zone, checkable=True,
                                        tip="Click the plan to place the corners.")
        self.edit_zone_button = self._button("Edit zone…", self._edit_zone)
        self.drop_zone_button = self._button("Delete zone", self._remove_zone)

        # Incidents: under the incident list.
        self.acknowledge_button = self._button("Acknowledge", self._acknowledge, shortcut="Ctrl+A",
                                               tip="Somebody has seen this and it is real.")
        self.dismiss_button = self._button("Dismiss…", self._dismiss, shortcut="Ctrl+D",
                                           tip="Seen, and not worth acting on. A reason is required.")
        self.export_button = self._button(
            "Export…", self._export, shortcut="Ctrl+E",
            tip="Write the evidence package for this incident: report, clips and a manifest "
                "that verifies.")
        self.dismissed_box = QCheckBox("Show dismissed")
        self.dismissed_box.toggled.connect(self._show_dismissed)

        # Reachable from the menu, and kept as attributes because the tests
        # and `_configure_only` name them.
        self.detection_button = self._button(
            "Watch for…", self._set_detection,
            tip="What this site looks for, and how sure the detector must be.")
        self.about_button = self._button("About", self._about)

    def _build_toolbar(self) -> "FlowLayout":
        """The top strip: run control, the lock, and the run's own state.

        Four controls, not seventeen. A wrapping layout still, because Qt's
        answer to a row that does not fit is to shrink the buttons and elide
        their labels — the shipped window once read "dd camera.", "elete zone"
        and "ort evidenc" on a 1280-wide screen, and only a photograph showed
        it.
        """
        row = FlowLayout(spacing=8)
        row.addWidget(self.start_button)
        row.addWidget(self.stop_button)
        row.addWidget(self.configure_button)
        self.toolbar = row
        return row

    def _build_body(self) -> QWidget:
        self.camera_list = CameraList()
        self.camera_list.selected.connect(self._camera_selected)
        self.camera_list.record_toggled.connect(self._record_toggled)
        camera_actions = ActionBar()
        camera_actions.add(self.add_button, self.place_button, self.calibrate_button,
                           self.edit_camera_button, self.password_button, self.remove_button)
        self.camera_panel = Panel("CAMERAS", self.camera_list, detail=True, actions=camera_actions)

        self.wall = QWidget()
        self._wall_grid = QGridLayout(self.wall)
        self._wall_grid.setContentsMargins(2, 2, 2, 2)
        self._wall_grid.setSpacing(4)
        self.wall_panel = Panel("CAMERA WALL", self.wall, detail=True)

        self.plan = PlanView()
        zone_actions = ActionBar()
        zone_actions.add(self.zone_button, self.edit_zone_button, self.drop_zone_button)
        # "NO TILES", not "NO EXTERNAL TILES". The claim is the product's and
        # stays on screen, but the longer form took 349 px of a 408 px heading
        # and left the ground's live state — what the map is worth, what the
        # solved slope is — 59 px to be elided into. A constant should not
        # crowd out a measurement.
        self.plan_panel = Panel("GROUND · NO TILES", self.plan, detail=True, actions=zone_actions)
        self.plan_panel.title.setToolTip(
            "Drawn entirely from this site's own geometry and its cameras' own frames. "
            "No map tiles are fetched, ever — that is the promise a site loses silently the "
            "first time it is air-gapped.")

        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(self.camera_panel)
        top.addWidget(self.wall_panel)
        top.addWidget(self.plan_panel)
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
        self.incidents.filtered.connect(self._filter_incidents)
        incident_actions = ActionBar()
        incident_actions.add(self.acknowledge_button, self.dismiss_button, self.export_button,
                             self.dismissed_box)
        self.incident_panel = Panel("INCIDENTS", self.incidents, detail=True,
                                    actions=incident_actions)

        self.tracks = TrackTable()
        self.detail = IncidentDetail()
        self.audit = AuditView()
        self.tabs = QTabWidget()
        self.tabs.addTab(self.detail, "Why")
        self.tabs.addTab(self.tracks, "Tracked objects")
        self.tabs.addTab(self.audit, "Audit trail")
        self.tabs.currentChanged.connect(self._tab_changed)

        lower = QSplitter(Qt.Orientation.Horizontal)
        lower.addWidget(self.incident_panel)
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
        return split

    def _build_menus(self) -> None:
        """The things done once a month, out of the way of the ones done hourly."""
        bar = self.menuBar()

        site = bar.addMenu("&Site")
        self._menu_item(site, "&Add camera…", self._add_camera, "Ctrl+N")
        self._menu_item(site, "&Watch for…", self._set_detection)
        site.addSeparator()
        self._menu_item(site, "Faces and plates…", self._identity)
        site.addSeparator()
        self._menu_item(site, "&Configure", self.configure_button.click, "Ctrl+K")

        analysis = bar.addMenu("&Analysis")
        self._menu_item(analysis, "&Start", self._start, "F5")
        self._menu_item(analysis, "S&top", self._stop, "F6")

        incident = bar.addMenu("&Incident")
        self._menu_item(incident, "&Acknowledge", self._acknowledge, "Ctrl+A")
        self._menu_item(incident, "&Dismiss…", self._dismiss, "Ctrl+D")
        self._menu_item(incident, "&Export…", self._export, "Ctrl+E")

        help_menu = bar.addMenu("&Help")
        self._menu_item(help_menu, "Keyboard shortcuts", self._shortcuts, "F1")
        self._menu_item(help_menu, "&About this build", self._about)

    def _menu_item(self, menu, text: str, slot, shortcut: str = "") -> QAction:
        action = QAction(text, self)
        action.triggered.connect(slot)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        menu.addAction(action)
        return action

    def _build_status(self) -> None:
        """Three labels, each with room to be read. See the module note."""
        self.status = self.statusBar()
        self.user_label = ElidingLabel("")
        self.lock_label = ElidingLabel("")
        self.alert_label = ElidingLabel("")
        self.alert_label.setStyleSheet(f"color: {theme.FAULT.name()}; font-weight: 600;")
        # These say what a panel is worth and live in that panel's heading.
        # They are kept as attributes because they are what the tests and the
        # older call sites name.
        self.site_label = ElidingLabel("")
        self.placement_label = ElidingLabel("")
        self.ground_label = ElidingLabel("")
        self.detector_label = ElidingLabel("")
        for label in self._permanent_labels():
            self.status.addPermanentWidget(label)

    def _configure_only(self) -> Sequence[QWidget]:
        return (self.add_button, self.place_button, self.calibrate_button, self.edit_camera_button,
                self.password_button, self.remove_button, self.detection_button, self.zone_button,
                self.edit_zone_button, self.drop_zone_button)

    def action_widgets(self) -> list[QWidget]:
        """Every control an operator can press, wherever it now lives.

        The old test walked the toolbar and asserted that no button was built
        and left out of it. The buttons are spread over four bars now, so the
        same guarantee needs one list of them — a control that exists in no
        layout is worse than one that is missing, because it is unreachable
        and nothing says so.
        """
        found: list[QWidget] = []
        for index in range(self.toolbar.count()):
            item = self.toolbar.itemAt(index)
            if item is not None and item.widget() is not None:
                found.append(item.widget())
        for panel in (self.camera_panel, self.plan_panel, self.incident_panel):
            if panel.actions is not None:
                found.extend(panel.actions.widgets())
        return found

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
                                else "MONITOR — locked")
        self.lock_label.setToolTip("" if self._configuring else LOCKED_REASON)
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
        # Unlocking enables the camera verbs, but only for a camera that is
        # actually selected. Without this, pressing Configure with no row
        # chosen lit six buttons whose only possible answer was "Select a
        # camera" — which is the error this layout exists to remove.
        self._show_camera_actions(self.camera_list.selected_camera())
        self._fit_labels()

    def _relock(self) -> None:
        if self._configuring:
            self.configure_button.setChecked(False)
            self._say("Re-locked after ten minutes with nothing touched.")

    #: Camera verbs: greyed by the lock *and* by there being a row to act on.
    #: Two different reasons, and a click on one must be answered with the
    #: right one — "Locked" beside an unlocked window sends somebody to press
    #: Configure, which is already pressed.
    def _needs_a_camera(self) -> Sequence[QWidget]:
        return (self.place_button, self.calibrate_button, self.edit_camera_button,
                self.password_button, self.remove_button)

    def _why_greyed(self, widget: QWidget) -> str:
        if not self._configuring:
            return LOCKED_REASON
        if widget in self._needs_a_camera():
            if self.camera_list.selected_camera() is None:
                return "Select a camera in the list; this acts on one camera."
            if widget is self.calibrate_button:
                return ("Place this camera first, roughly. Measuring refines a placement; it "
                        "cannot invent one.")
        return ""

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt's name
        if event.type() == QEvent.Type.MouseButtonPress and isinstance(watched, QWidget) and not watched.isEnabled():
            self._say(self._why_greyed(watched) or LOCKED_REASON)
            return True
        return super().eventFilter(watched, event)

    # ------------------------------------------------------------ commands
    # ------------------------------------------------------------ refresh

    def refresh_site(self) -> None:
        """Redraw everything that changes when the *site* changes.

        One `snapshot()`, so every panel is drawn from the same moment. Before
        this the window asked the service five separate questions here and six
        more per tick, and a camera removed between two of them appeared in
        one panel and not the next.
        """
        site = self.commands.snapshot()
        self.camera_list.show_cameras(site.cameras, site.health)
        self.incidents.set_cameras([c.id for c in site.cameras])
        self.plan.set_cameras({c.id: c.pose for c in site.cameras if c.pose is not None})
        self.plan.set_zones(site.zones)
        self.plan.set_ground(site.ground)
        self._show_placement(site)
        self._show_ground(site)
        self._show_detector()
        self._show_running(self.commands.runtime.running, site)
        # The selected camera may have just been removed or placed, and both
        # change what can be done to it.
        self._show_camera_actions(self.camera_list.selected_camera(), site)
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
        site = self.commands.snapshot()
        # The ground is refreshed every tick, not only when a camera is
        # edited. It is being built from these very frames, and a plan that
        # only redrew on a configuration change would show an empty yard for
        # the whole of the run that was filling it in.
        self.plan.set_ground(site.ground)
        self._show_ground(site)
        self._show_placement(site)
        poses = {c.id: c.pose for c in site.cameras}
        rows = []
        for result in results:
            view = self._views.get(result.camera_id)
            state = site.health.get(result.camera_id)
            if view is not None:
                view.show_result(result, state.analysis_fps if state else 0.0)
            self.plan.set_tracks(result.camera_id, result.tracks, result.relations)
            info = self._detector_info(result.camera_id)
            if view is not None:
                view.set_detector_info(info)
            pose = poses.get(result.camera_id)
            rows.extend((result.camera_id, track, info, pose,
                         tuple(r for r in result.relations
                               if r.subject == track.id or r.object == track.id))
                        for track in result.tracks)
        if rows or results:
            self.tracks.show_tracks(rows)
        if results:
            self._show_detector()
        self.camera_list.show_cameras(site.cameras, site.health)
        self._show_camera_actions(self.camera_list.selected_camera(), site)
        self.incidents.show_incidents(self.commands.incidents())
        self._incident_selected(self.incidents.selected_incident(), site)
        self._show_alerts()
        if self.tabs.currentWidget() is self.audit:
            self.audit.show_rows(self.commands.audit_rows(limit=200))
        if not final and self._timer.isActive() and not self.commands.runtime.running:
            # Every camera ended on its own: for files that is completion. One
            # more drain first, so the last clip and events are never lost.
            self._timer.stop()
            self._collect(final=True)
            self._say("Finished.")
            self._show_running(False, site)

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

    def _show_placement(self, site) -> None:
        """How much of this site can locate anything, in the cameras' heading."""
        self.placement_label.setText(site.placement())
        self.camera_panel.say(site.placement())

    def _show_detector(self) -> None:
        """What is drawing the conclusions, and what it cannot do.

        In the wall's heading, because that is what it describes. An operator
        reading "does not classify" beside a track with no class learns
        something; one reading a model name beside the same track would assume
        the model looked and found nothing recognisable, which is the opposite
        of true. Before the run it says what *will* be used, because the
        choice is made before Start and a silent fall-back to motion detection
        is the defect this line exists to prevent.
        """
        info = self.commands.detector()
        if info is None:
            model = self.commands.model
            self.detector_label.setText(f"Will use {model.name}" if model is not None
                                        else "No model found — motion only, which cannot classify")
            self.detector_label.setStyleSheet("" if model is not None else f"color: {theme.STALE.name()};")
            self.detector_label.setToolTip("" if model is not None else
                                           "Looked in: " + ", ".join(str(p) for p in self.commands.model_places))
        elif not info.classifies:
            self.detector_label.setText(f"{info.name} — does not classify, and cannot see a stationary object")
            self.detector_label.setStyleSheet(f"color: {theme.STALE.name()};")
        else:
            names = sorted(set(info.class_names.values()))
            watching = ", ".join(names) if len(names) <= 8 else f"{len(names)} classes"
            digest = f" · {info.model_sha256[:12]}" if info.model_sha256 else ""
            masks = " with masks" if info.kind.endswith("segment") else ", boxes only"
            self.detector_label.setText(f"{info.name} — watching {watching}{masks}{digest}")
            self.detector_label.setStyleSheet("")
        self.wall_panel.say(self.detector_label.text())
        self.wall_panel.detail.setToolTip(self.detector_label.toolTip() or self.detector_label.text())

    def _detector_labels(self) -> list[str]:
        return self.commands.labels()

    # ------------------------------------------------------------ plumbing

    def _current_camera(self):
        return self.commands.camera(self.camera_list.selected_camera())

    def _camera_selected(self, camera_id) -> None:
        self._selected_camera = camera_id
        self.plan.select(camera_id)
        self._show_camera_actions(camera_id)

    def _show_camera_actions(self, camera_id, site=None) -> None:
        """Grey what cannot apply to the row that is selected.

        A button that is live for a camera it cannot act on is a button whose
        only answer is a sentence explaining why not — which is the interface
        asking for something it can already see.
        """
        camera = site.camera(camera_id) if site is not None else self.commands.camera(camera_id)
        chosen = camera is not None
        for control in self._needs_a_camera():
            control.setEnabled(self._configuring and chosen)
        # Measuring a pose refines a placement, so it needs one to refine.
        self.calibrate_button.setEnabled(self._configuring and chosen and camera.pose is not None)

    def _incident_selected(self, incident, site=None) -> None:
        chosen = incident is not None
        self.export_button.setEnabled(chosen and self.commands.may(INCIDENT_EXPORT))
        may_review = chosen and self.commands.may(INCIDENT_REVIEW)
        self.acknowledge_button.setEnabled(may_review)
        self.dismiss_button.setEnabled(may_review)
        cameras = site.cameras if site is not None else self.commands.cameras()
        self.detail.show_incident(incident, {c.id: c.pose for c in cameras})
        if chosen:
            self.tabs.setCurrentWidget(self.detail)

    def _tab_changed(self, index: int) -> None:
        if self.tabs.widget(index) is self.audit:
            self.audit.show_rows(self.commands.audit_rows(limit=200))

    def _record_toggled(self, camera_id: str, on: bool) -> None:
        self._say(self.commands.set_recording(camera_id, on).message)
        self._idle.start()

    def _show_site(self) -> None:
        """The site's name and the clock its schedules are read in.

        In the title bar and the window's own heading, because a zone that
        closes at 22:00 closes in *this* clock and an operator reading a wall
        clock has to be able to check that.
        """
        name, zone = self.commands.site()
        self.setWindowTitle(f"Sentinel Vision — {name} · {zone}")
        self.site_label.setText(f"{name} · {zone}")
        self.site_label.setToolTip("Schedules on zones are read in this clock. Change it with `vigil site name`.")

    def _show_principal(self) -> None:
        principal = self.commands.principal
        if principal.origin == "user":
            self.user_label.setText(f"{principal.name} · {str(principal.role).lower()}")
            self.user_label.setStyleSheet("")
        else:
            self.user_label.setText("no account — nobody is named")
            self.user_label.setStyleSheet(f"color: {theme.STALE.name()};")
            self.user_label.setToolTip("The audit trail names the OS account. Create the first "
                                       "administrator with `vigil users add NAME --role ADMIN`.")

    def _say(self, text: str) -> None:
        if text:
            self.status.showMessage(text, 12_000)

    def _show_ground(self, site=None) -> None:
        """The solved ground and the live map, in the plan's own heading.

        It belongs beside the plan because it changes what every position on
        it means: a yard with a 3% fall was being projected onto a level plane
        until this appeared, and an operator comparing today's positions with
        last week's should be able to see that the ground under them stopped
        being an assumption.
        """
        plane = site.plane if site is not None else self.commands.ground_plane()
        state = site.map_state if site is not None else self.commands.map_state()
        parts = [] if plane is None else [plane.describe()]
        if state != "no map":
            parts.append(state)
        text = "  ·  ".join(parts) or "level ground assumed"
        self.ground_label.setText(text)
        self.plan_panel.say(text)
        self.plan_panel.detail.setToolTip(text)

    def _permanent_labels(self):
        """What the status bar carries. Three, so each can be read."""
        return (self.user_label, self.lock_label, self.alert_label)

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
            # The list asks for what its five columns need and no more; the
            # rest is split between the pictures and the ground, slightly in
            # the pictures' favour because a wall of four cameras is four
            # images and the plan is one.
            width = max(self._top.width(), self.width())
            first = max(360, int(width * 0.26))
            rest = max(0, width - first)
            wall = int(rest * 0.55)
            self._top.setSizes([first, wall, rest - wall])
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
