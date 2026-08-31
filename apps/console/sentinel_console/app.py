"""The console window.

A native Qt application. No embedded browser, no web view, no HTML — the widgets
paint directly. That is not stylistic: a control-room console that ships a
browser engine inherits its update cadence, its memory profile and its network
assumptions, and this system is meant to run for months on a machine with no
route to the Internet.

The layout answers four questions, in the order an operator asks them:

1. *What is happening?* — the camera wall, top left.
2. *Where is it happening?* — the plan view beside it, with every camera's
   footprint on one piece of ground.
3. *What does the system claim, and why?* — the incident list.
4. *What is it claiming that on?* — the track table beside it.

Two structural decisions carry the whole design.

**Each camera runs its own pipeline, knowing nothing of the others.** A session
is exactly what a worker node runs in a distributed deployment, so the
single-machine and multi-machine cases are the same code with a different
transport.

**Correlation happens above them, never inside them.** A camera that correlated
its own events in isolation would raise one incident per camera for one
intrusion, which is the alert duplication the whole system exists to prevent.
Three cameras seeing one person must produce one row.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent, QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from sentinel.core import (
    CameraPose,
    destination_point,
    field_of_view,
    haversine_distance,
)
from sentinel.decode import DecodeError, VideoSource
from sentinel.detect import MotionDetector
from sentinel.events import (
    AfterHoursRule,
    LoiteringRule,
    RapidMovementRule,
    ZoneEntryRule,
)
from sentinel.incidents import Correlator
from sentinel.zones import Zone, ZoneKind

from . import theme
from .incident_view import IncidentView
from .map_view import MapView
from .placement import PlacementDialog
from .session import CameraSession
from .video_view import VideoView
from .worker import AnalysisWorker

#: How often the interface collects results. 30 Hz is smooth to the eye and
#: leaves the analysis threads the rest of the machine.
REPAINT_INTERVAL_MILLIS = 33

#: How often events are correlated into incidents. Far slower than the repaint,
#: because correlation is a batch operation over a window and running it at
#: frame rate would cost far more than it tells anyone.
CORRELATE_INTERVAL_MILLIS = 1500

# There is deliberately no default pose. A camera nobody has placed cannot say
# where anything is, and a nominal origin would produce coordinates that look
# exactly like measured ones — which someone would then be sent to. Until an
# operator fills in the placement dialog, objects are tracked and reported as
# "not placed".


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
    """The main window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sentinel Vision — Console")
        self.resize(1500, 920)
        self.setStyleSheet(theme.STYLESHEET)

        self._sessions: dict[str, CameraSession] = {}
        self._zones: list[Zone] = []
        self._incidents: list = []

        self._build()

        self._timer = QTimer(self)
        self._timer.setInterval(REPAINT_INTERVAL_MILLIS)
        self._timer.timeout.connect(self._collect)

        self._correlate_timer = QTimer(self)
        self._correlate_timer.setInterval(CORRELATE_INTERVAL_MILLIS)
        self._correlate_timer.timeout.connect(self._correlate)

    # -------------------------------------------------------- the primary camera
    #
    # A single-camera console is a multi-camera console with one camera, so the
    # simple case keeps simple accessors rather than a separate code path.

    @property
    def _selected(self) -> CameraSession | None:
        chosen = self.camera_picker.currentData()
        if chosen in self._sessions:
            return self._sessions[chosen]
        return next(iter(self._sessions.values()), None)

    @property
    def _pose(self) -> CameraPose | None:
        session = self._selected
        return session.pose if session else None

    @property
    def _worker(self) -> AnalysisWorker | None:
        session = self._selected
        return session.worker if session else None

    @property
    def _running(self) -> bool:
        return any(session.is_running for session in self._sessions.values())

    # ------------------------------------------------------------------ layout

    def _build(self) -> None:
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)

        # The views are built before the toolbar, because the toolbar wires its
        # controls to them.
        self.wall = QWidget()
        self.wall_layout = QGridLayout(self.wall)
        self.wall_layout.setContentsMargins(0, 0, 0, 0)
        self.wall_layout.setSpacing(4)
        self.empty_wall = QLabel("No cameras. Add one to begin.")
        self.empty_wall.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_wall.setObjectName("Caption")
        self.wall_layout.addWidget(self.empty_wall, 0, 0)

        self.map = MapView()
        self.tracks = self._build_track_table()
        self.incidents = IncidentView()

        outer.addLayout(self._build_toolbar())

        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(_panel("CAMERAS", self.wall))
        top.addWidget(_panel("GROUND — NO EXTERNAL TILES", self.map))
        top.setStretchFactor(0, 3)
        top.setStretchFactor(1, 2)

        # Incidents beside tracks, and larger. Tracks are how the system reached
        # its conclusions; incidents are the conclusions, and they are what an
        # operator is here to read.
        lower = QSplitter(Qt.Orientation.Horizontal)
        lower.addWidget(_panel("INCIDENTS", self.incidents))
        lower.addWidget(_panel("TRACKED OBJECTS", self.tracks))
        lower.setStretchFactor(0, 3)
        lower.setStretchFactor(1, 2)

        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(top)
        split.addWidget(lower)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)

        outer.addWidget(split, 1)
        self.setCentralWidget(central)

        self.status = self.statusBar()
        self._set_status("Ready. Add a camera to begin.")
        self._build_menu()

    def _build_toolbar(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)

        self.open_button = QPushButton("Add camera…")
        self.open_button.clicked.connect(self._choose_source)
        row.addWidget(self.open_button)

        self.start_button = QPushButton("Start")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self._start)
        row.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop)
        row.addWidget(self.stop_button)

        row.addSpacing(12)

        row.addWidget(QLabel("camera"))
        self.camera_picker = QComboBox()
        self.camera_picker.setMinimumWidth(140)
        self.camera_picker.setToolTip(
            "Which camera the placement dialog applies to. Every camera is "
            "analysed whichever is selected here."
        )
        row.addWidget(self.camera_picker)

        self.place_button = QPushButton("Place…")
        self.place_button.clicked.connect(self._place_camera)
        row.addWidget(self.place_button)

        self.zone_button = QPushButton("Add zone")
        self.zone_button.setToolTip(
            "Adds a restricted area on the ground in front of the selected "
            "camera. Requires that camera to be placed first: a zone without a "
            "placed camera has nothing to be measured against."
        )
        self.zone_button.clicked.connect(self._add_zone)
        row.addWidget(self.zone_button)

        self.zone_radius = QDoubleSpinBox()
        self.zone_radius.setRange(2.0, 200.0)
        self.zone_radius.setValue(10.0)
        self.zone_radius.setSuffix(" m")
        self.zone_radius.setFixedWidth(80)
        row.addWidget(self.zone_radius)

        row.addSpacing(12)

        self.show_detections = QCheckBox("Show raw detections")
        self.show_detections.setChecked(True)
        self.show_detections.toggled.connect(self._toggle_detections)
        row.addWidget(self.show_detections)

        # Hidden until something breaks. A permanently visible status area that
        # says "OK" trains an operator to stop reading it.
        self.fault_label = QLabel("")
        self.fault_label.setStyleSheet(f"color: {theme.FAULT.name()}; font-weight: 600;")
        self.fault_label.setVisible(False)
        row.addWidget(self.fault_label)

        row.addStretch(1)

        self.placement_label = QLabel("No camera placed — objects will not be located")
        self.placement_label.setObjectName("Caption")
        row.addWidget(self.placement_label)

        row.addSpacing(14)
        self.detector_label = QLabel("")
        self.detector_label.setObjectName("Caption")
        row.addWidget(self.detector_label)

        return row

    def _build_track_table(self) -> QTreeWidget:
        tree = QTreeWidget()
        tree.setRootIsDecorated(False)
        tree.setAlternatingRowColors(True)
        tree.setUniformRowHeights(True)
        tree.setHeaderLabels(
            ["Camera", "ID", "Class", "Confidence", "Seen", "Duration", "Speed",
             "Heading", "Position", "Uncertainty", "Source"]
        )
        header = tree.header()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        for index, width in enumerate((92, 52, 110, 88, 56, 78, 82, 76, 186, 92)):
            tree.setColumnWidth(index, width)

        font = QFont("Consolas, monospace")
        font.setPointSize(11)
        tree.setFont(font)
        return tree

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        open_action = QAction("&Add camera…", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._choose_source)
        file_menu.addAction(open_action)

        place_action = QAction("&Camera placement…", self)
        place_action.setShortcut("Ctrl+P")
        place_action.triggered.connect(self._place_camera)
        file_menu.addAction(place_action)

        file_menu.addSeparator()
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        help_menu = self.menuBar().addMenu("&Help")
        about = QAction("&About this build", self)
        about.triggered.connect(self._show_about)
        help_menu.addAction(about)

    # ----------------------------------------------------------------- cameras

    def add_camera(self, path: Path, camera_id: str | None = None) -> CameraSession:
        """Register a source as a camera. Does not start it."""
        identifier = camera_id or path.stem or f"cam-{len(self._sessions) + 1:02d}"
        if identifier in self._sessions:
            identifier = f"{identifier}-{len(self._sessions) + 1}"

        view = VideoView()
        view.set_placeholder(f"{identifier} — not started")
        view.set_show_detections(self.show_detections.isChecked())

        session = CameraSession(camera_id=identifier, source_path=path, view=view)
        self._sessions[identifier] = session

        self.camera_picker.addItem(identifier, identifier)
        self.camera_picker.setCurrentIndex(self.camera_picker.count() - 1)
        self._relayout_wall()
        self.start_button.setEnabled(True)
        return session

    def _relayout_wall(self) -> None:
        """Arrange the camera views in as square a grid as fits.

        A wall of unequal panes makes one camera look more important than the
        others, which is a claim the layout should not be making.
        """
        while self.wall_layout.count():
            item = self.wall_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

        if not self._sessions:
            self.wall_layout.addWidget(self.empty_wall, 0, 0)
            self.empty_wall.setVisible(True)
            return

        self.empty_wall.setVisible(False)
        count = len(self._sessions)
        columns = 1 if count == 1 else 2 if count <= 4 else 3

        for index, session in enumerate(self._sessions.values()):
            self.wall_layout.addWidget(session.view, index // columns, index % columns)
            session.view.setVisible(True)

    def _choose_source(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add one or more cameras",
            str(Path.home()),
            "Video (*.mp4 *.mkv *.avi *.mov *.m4v);;All files (*)",
        )
        for path in paths:
            self.add_camera(Path(path))

        if paths:
            self._set_status(
                f"{len(self._sessions)} camera(s). Place them, then press Start."
            )

    def _toggle_detections(self, show: bool) -> None:
        for session in self._sessions.values():
            session.view.set_show_detections(show)

    # ---------------------------------------------------------------- placement

    def _place_camera(self) -> None:
        """Set where the selected camera is and where it points.

        Takes effect immediately on a running pipeline. Existing tracks keep
        their identity — a camera being placed does not make the people it was
        already following into different people.
        """
        session = self._selected
        if session is None:
            QMessageBox.information(self, "No camera", "Add a camera first.")
            return

        dialog = PlacementDialog(session.pose, self)
        if dialog.exec() != PlacementDialog.DialogCode.Accepted:
            return

        session.pose = dialog.pose()
        if session.worker is not None:
            session.worker.set_pose(session.pose)

        self._refresh_placement()

    def _refresh_placement(self) -> None:
        placed = {
            session.camera_id: session.pose
            for session in self._sessions.values()
            if session.pose is not None
        }
        self.map.set_cameras(placed)
        self.map.set_zones(self._zones)

        if not placed:
            self.placement_label.setText("No camera placed — objects will not be located")
        elif len(placed) == len(self._sessions):
            self.placement_label.setText(f"{len(placed)} camera(s) placed")
        else:
            self.placement_label.setText(
                f"{len(placed)} of {len(self._sessions)} cameras placed — "
                "the rest will not locate anything"
            )

    def _add_zone(self) -> None:
        """Put a restricted area on the ground in front of the selected camera.

        A placeholder for drawing one on the plan view, and honest about being
        one. What it is not is a default: a zone exists only because an operator
        asked for it, because a zone nobody drew is a source of alerts nobody
        expects.
        """
        pose = self._pose
        if pose is None:
            QMessageBox.information(
                self,
                "Place the camera first",
                "A zone is an area on the ground. Until the camera is placed "
                "there is nothing to measure it against, and objects are tracked "
                "but not located.",
            )
            return

        radius = self.zone_radius.value()

        # Placed just beyond the near edge of what this camera can *actually*
        # see, which is not the range it claims: a 6 m mast tilted 22 degrees
        # with a 36 degree vertical field covers 7 m to 86 m however large the
        # stated range is. The near end is also where positions are most
        # accurate, because uncertainty grows super-linearly with distance — so
        # a zone there is one the system can genuinely adjudicate rather than
        # one it will mostly report as UNCERTAIN.
        footprint = field_of_view(pose, arc_segments=16)
        near = (
            min(haversine_distance(pose.position, point) for point in footprint)
            if footprint
            else 10.0
        )
        centre = destination_point(pose.position, pose.heading, near + radius)
        ring = tuple(
            destination_point(centre, bearing, radius)
            for bearing in (0.0, 90.0, 180.0, 270.0)
        )

        index = len(self._zones) + 1
        self._zones.append(
            Zone(
                id=f"zone-{index}",
                name=f"Restricted Area {chr(64 + index)}",
                kind=ZoneKind.RESTRICTED,
                ring=ring,
                enter_after_millis=600,
            )
        )
        self.map.set_zones(self._zones)
        self._set_status(
            f"{len(self._zones)} zone(s). Rules apply to cameras started from now."
        )

    # ------------------------------------------------------------------ running

    def _rules(self) -> list:
        if not self._zones:
            # Without a zone there is nothing to be inside, so only the rules
            # that watch a track on its own can fire.
            return [RapidMovementRule(speed_mps=6.0)]
        return [
            ZoneEntryRule(),
            AfterHoursRule(),
            LoiteringRule(dwell_millis=8000),
            RapidMovementRule(speed_mps=6.0),
        ]

    def _start(self) -> None:
        if self._running or not self._sessions:
            return

        self.fault_label.setVisible(False)
        started = 0

        for session in self._sessions.values():
            session.fault = None
            session.events.clear()

            try:
                source = VideoSource(session.source_path, source_id=session.camera_id)
                source.open()
            except DecodeError as error:
                # A modal is right here: the operator asked for this, just now,
                # and is waiting for it.
                QMessageBox.warning(self, f"Cannot open {session.camera_id}", str(error))
                continue

            detector = MotionDetector()
            session.view.set_detector_info(detector.info)

            worker = AnalysisWorker(
                source,
                detector,
                session.pose,
                realtime=True,
                zones=self._zones,
                rules=self._rules(),
                node_id="local",
                parent=self,
            )
            worker.finished_run.connect(
                lambda reason, s=session: self._on_finished(reason, s)
            )
            worker.failed.connect(lambda message, s=session: self._on_failed(message, s))

            session.worker = worker
            worker.start()
            started += 1

        if started == 0:
            return

        self.detector_label.setText("MOG2 background subtraction — does not classify")
        self._timer.start()
        self._correlate_timer.start()

        self.open_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._set_status(f"Running {started} camera(s).")

    def _stop(self) -> None:
        for session in self._sessions.values():
            session.stop()
        self._teardown()
        self._set_status("Stopped.")

    def _teardown(self) -> None:
        self._timer.stop()
        self._correlate_timer.stop()
        self.open_button.setEnabled(True)
        self.start_button.setEnabled(bool(self._sessions))
        self.stop_button.setEnabled(False)

    # ------------------------------------------------------------------ updates

    def _collect(self) -> None:
        """Pull the newest result from every camera. Runs on the repaint timer."""
        changed = False

        for session in self._sessions.values():
            if session.worker is None:
                continue
            update = session.worker.take_latest()
            if update is None:
                continue

            session.absorb(update)
            session.view.show_update(update)
            self.map.set_tracks(update.result.tracks, session.camera_id)
            changed = True

        if changed:
            self._refresh_tracks()
            self._refresh_status()

    def _correlate(self) -> None:
        """Group every camera's events into incidents.

        Across all cameras, deliberately. A camera correlating its own events
        would raise one incident per camera for one intrusion, which is exactly
        the duplication this stage exists to remove.
        """
        events = [event for session in self._sessions.values() for event in session.events]
        if not events:
            self.incidents.show_incidents([])
            return

        correlator = Correlator(zone_kinds={zone.id: zone.kind for zone in self._zones})
        self._incidents = correlator.correlate(events)
        self.incidents.show_incidents(self._incidents)

    def _refresh_tracks(self) -> None:
        # Rebuilt rather than diffed. At the handful of objects a site sees this
        # costs nothing, and a table that reuses rows can show a stale value in
        # a column that failed to update — worse than a flicker in an evidence
        # view.
        self.tracks.clear()

        for session in self._sessions.values():
            update = session.last
            if update is None:
                continue

            info = session.worker.detector_info if session.worker else None
            for track in sorted(update.result.tracks, key=lambda t: t.id):
                self.tracks.addTopLevelItem(
                    self._track_row(session.camera_id, track, update, info)
                )

    def _track_row(self, camera_id: str, track, update, info) -> QTreeWidgetItem:
        duration = (track.last_seen_millis - track.first_seen_millis) / 1000.0
        gap = update.result.timestamp_millis - track.last_seen_millis

        if track.position is not None:
            where = f"{track.position.point.lat:+.6f}, {track.position.point.lon:+.6f}"
            radius = f"±{track.position.radius_meters:.1f} m"
            origin = (
                "projected"
                if track.position.source == "GROUND_PROJECTION"
                else "fallback"
            )
        else:
            # Never blank: a blank cell reads as zero. This is a statement.
            where, radius, origin = "not placed", "—", "no pose"

        if track.speed_mps is None:
            speed = "unknown"
        elif track.speed_mps < 0.3:
            speed = "still"
        else:
            speed = f"{track.speed_mps:.2f} m/s"

        heading = f"{track.heading_degrees:.0f}°" if track.heading_degrees is not None else "—"

        item = QTreeWidgetItem([
            camera_id,
            f"#{track.id}",
            info.label_for(track.class_id) if info else str(track.class_id),
            f"{track.confidence:.2f}",
            str(track.hits),
            f"{duration:.1f} s",
            speed,
            heading,
            where,
            radius,
            origin,
        ])
        item.setForeground(
            1, theme.TRACK_COASTING if gap > theme.COASTING_AFTER_MILLIS else theme.TRACK
        )
        return item

    def _refresh_status(self) -> None:
        running = [s for s in self._sessions.values() if s.is_running]
        tracked = sum(
            len(s.last.result.tracks) for s in running if s.last is not None
        )
        events = sum(len(s.events) for s in self._sessions.values())

        self._set_status(
            f"{len(running)} camera(s)   {tracked} tracked now   "
            f"{events} events -> {len(self._incidents)} incidents"
        )

    def _on_finished(self, reason: str, session: CameraSession | None = None) -> None:
        if session is not None:
            session.worker = None
        # One last collection before the timers stop. Without it the final
        # frames — and the incidents correlated from them — are produced and
        # then thrown away, so a file ending on an intrusion shows nothing.
        self._collect()
        self._correlate()

        if not self._running:
            self._teardown()
            self._set_status(reason)

    def _on_failed(self, message: str, session: CameraSession | None = None) -> None:
        """A running camera failed.

        Reported in place rather than as a modal. A modal is right for something
        the operator just asked for and which did not work; it is wrong for a
        camera dropping on its own, because that happens to twenty cameras at
        once when a switch loses power, and the operator would face a stack of
        dialogs each of which must be dismissed before anything else can be
        done — including looking at the cameras that are still working.

        The message came from DecodeError, which is redacted by construction, so
        it is safe to put in front of a person.
        """
        if session is not None:
            session.fault = message
            session.worker = None
            session.view.set_placeholder(f"{session.camera_id} — {message}")

        self.fault_label.setText(message)
        self.fault_label.setVisible(True)
        self._set_status(message)

        if not self._running:
            self._teardown()

    def _set_status(self, text: str) -> None:
        self.status.showMessage(text)

    def _show_about(self) -> None:
        QMessageBox.information(
            self,
            "About this build",
            "Sentinel Vision console.\n\n"
            "Native Qt widgets — no embedded browser.\n"
            "No Internet access at any point: no tiles, no telemetry, no model "
            "downloads.\n\n"
            "See STATUS.md for what is implemented and what is not.",
        )

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt naming
        for session in self._sessions.values():
            session.stop()
        event.accept()


def run() -> int:
    """Start the console. The console-script entry point."""
    import sys

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setApplicationName("Sentinel Vision Console")
    app.setOrganizationName("Sentinel Vision")

    window = ConsoleWindow()
    window.show()
    return app.exec()
