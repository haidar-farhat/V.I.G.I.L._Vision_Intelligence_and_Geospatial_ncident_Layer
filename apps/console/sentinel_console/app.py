"""The console window.

A native Qt application. No embedded browser, no web view, no HTML — the widgets
paint directly. That is not stylistic: a control-room console that ships a
browser engine inherits its update cadence, its memory profile and its network
assumptions, and this system is meant to run for months on a machine with no
route to the Internet.

The layout answers three questions in the order an operator asks them:

1. *What is happening?* — the camera view, centre, largest.
2. *Where is it happening?* — the plan view, beside it.
3. *What exactly does the system claim, and on what basis?* — the track table
   below, one row per object with its evidence.

The interface pulls from the analysis thread on a repaint timer rather than
being pushed to. See ``worker.py`` for why.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent, QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
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

from sentinel.core import CameraPose, LatLon
from sentinel.decode import DecodeError, VideoSource
from sentinel.core import destination_point, field_of_view, haversine_distance
from sentinel.detect import MotionDetector
from sentinel.events import AfterHoursRule, LoiteringRule, RapidMovementRule, ZoneEntryRule
from sentinel.zones import Zone, ZoneKind

from . import theme
from .incident_view import IncidentView
from .map_view import MapView
from .placement import PlacementDialog
from .video_view import VideoView
from .worker import AnalysisWorker, Update

#: How often the interface collects results. 30 Hz is smooth to the eye and
#: leaves the analysis thread the rest of the machine.
REPAINT_INTERVAL_MILLIS = 33

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
        self.resize(1360, 860)
        self.setStyleSheet(theme.STYLESHEET)

        self._worker: AnalysisWorker | None = None
        self._pose: CameraPose | None = None
        self._source_path: Path | None = None
        self._zones: list[Zone] = []

        self._build()

        self._timer = QTimer(self)
        self._timer.setInterval(REPAINT_INTERVAL_MILLIS)
        self._timer.timeout.connect(self._collect)

    # ------------------------------------------------------------------ layout

    def _build(self) -> None:
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)

        # The views are built before the toolbar, because the toolbar wires its
        # controls to them.
        self.video = VideoView()
        self.map = MapView()
        self.tracks = self._build_track_table()
        self.incidents = IncidentView()

        outer.addLayout(self._build_toolbar())

        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(_panel("CAMERA", self.video))
        top.addWidget(_panel("GROUND — NO EXTERNAL TILES", self.map))
        top.setStretchFactor(0, 3)
        top.setStretchFactor(1, 2)

        # Incidents above tracks, and larger. Tracks are how the system reached
        # its conclusions; incidents are the conclusions, and they are what an
        # operator is actually here to read.
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
        self._set_status("Ready. Open a video file to begin.")

        self._build_menu()

    def _build_toolbar(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)

        self.open_button = QPushButton("Open video…")
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

        self.place_button = QPushButton("Place camera…")
        self.place_button.clicked.connect(self._place_camera)
        row.addWidget(self.place_button)

        self.zone_button = QPushButton("Add restricted zone")
        self.zone_button.setToolTip(
            "Adds a restricted area on the ground in front of this camera. "
            "Requires the camera to be placed first: a zone without a placed "
            "camera has nothing to be measured against."
        )
        self.zone_button.clicked.connect(self._add_zone)
        row.addWidget(self.zone_button)

        row.addWidget(QLabel("radius"))
        self.zone_radius = QDoubleSpinBox()
        self.zone_radius.setRange(2.0, 200.0)
        self.zone_radius.setValue(9.0)
        self.zone_radius.setSuffix(" m")
        self.zone_radius.setFixedWidth(84)
        row.addWidget(self.zone_radius)

        row.addSpacing(12)

        self.show_detections = QCheckBox("Show raw detections")
        self.show_detections.setChecked(True)
        self.show_detections.toggled.connect(self.video.set_show_detections)
        row.addWidget(self.show_detections)

        # Hidden until something breaks. A permanently visible status area that
        # says "OK" trains an operator to stop reading it.
        self.fault_label = QLabel("")
        self.fault_label.setStyleSheet(f"color: {theme.FAULT.name()}; font-weight: 600;")
        self.fault_label.setVisible(False)
        row.addWidget(self.fault_label)

        row.addStretch(1)

        self.placement_label = QLabel("Camera not placed — objects will not be located")
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
            ["ID", "Class", "Confidence", "Seen", "Duration", "Speed", "Heading",
             "Position", "Uncertainty", "Source"]
        )
        header = tree.header()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        for index, width in enumerate((56, 110, 92, 60, 84, 84, 84, 190, 96)):
            tree.setColumnWidth(index, width)

        font = QFont("Consolas, monospace")
        font.setPointSize(11)
        tree.setFont(font)
        return tree

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        open_action = QAction("&Open video…", self)
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

    # ------------------------------------------------------------------ actions

    def _choose_source(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open a video file",
            str(Path.home()),
            "Video (*.mp4 *.mkv *.avi *.mov *.m4v);;All files (*)",
        )
        if not path:
            return

        self._source_path = Path(path)
        self.start_button.setEnabled(True)
        self._set_status(f"Ready: {self._source_path.name}")

    def _place_camera(self) -> None:
        """Set where this camera is and where it points.

        Takes effect immediately on a running pipeline. Existing tracks keep
        their identity — a camera being placed does not make the people it was
        already following into different people.
        """
        dialog = PlacementDialog(self._pose, self)
        if dialog.exec() != PlacementDialog.DialogCode.Accepted:
            return

        self._pose = dialog.pose()
        self.map.set_pose(self._pose)
        if self._worker is not None:
            self._worker.set_pose(self._pose)

        self.placement_label.setText(
            f"Placed at {self._pose.position.lat:.5f}, {self._pose.position.lon:.5f} — "
            f"{self._pose.mount_height:.1f} m, bearing {self._pose.heading:.0f}°"
        )

    def _add_zone(self) -> None:
        """Put a restricted area on the ground in front of the camera.

        A placeholder for drawing one on the plan view, and honest about being
        one. What it is not is a default: a zone is only ever created because an
        operator asked for it, because a zone nobody drew is a source of alerts
        nobody expects.
        """
        if self._pose is None:
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
        footprint = field_of_view(self._pose, arc_segments=16)
        near = (
            min(haversine_distance(self._pose.position, point) for point in footprint)
            if footprint
            else 10.0
        )
        centre = destination_point(self._pose.position, self._pose.heading, near + radius)
        ring = tuple(destination_point(centre, bearing, radius) for bearing in (0.0, 90.0, 180.0, 270.0))

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
            f"{len(self._zones)} zone(s). Rules apply from the next run."
        )

    def _start(self) -> None:
        if self._worker is not None or self._source_path is None:
            return

        self.fault_label.setVisible(False)

        try:
            source = VideoSource(self._source_path, source_id=self._source_path.stem)
            source.open()
        except DecodeError as error:
            # A modal is right here: the operator asked for this, just now, and
            # is waiting for it.
            QMessageBox.warning(self, "Cannot open source", str(error))
            return

        self.map.set_pose(self._pose)

        detector = MotionDetector()
        self.video.set_detector_info(detector.info)
        self.detector_label.setText(
            f"{detector.info.name} — does not classify"
            if not detector.info.classifies
            else detector.info.name
        )

        rules = [
            ZoneEntryRule(),
            AfterHoursRule(),
            LoiteringRule(dwell_millis=8000),
            RapidMovementRule(speed_mps=6.0),
        ]
        worker = AnalysisWorker(
            source,
            detector,
            self._pose,
            realtime=True,
            zones=self._zones,
            rules=rules if self._zones else [RapidMovementRule(speed_mps=6.0)],
            node_id="local",
            parent=self,
        )
        worker.finished_run.connect(self._on_finished)
        worker.failed.connect(self._on_failed)

        self._worker = worker
        worker.start()
        self._timer.start()

        self.open_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._set_status(f"Running: {source.display_url}")

    def _stop(self) -> None:
        if self._worker is None:
            return
        self._worker.stop()
        self._worker.wait(3000)
        self._teardown()
        self._set_status("Stopped.")

    def _teardown(self) -> None:
        self._timer.stop()
        self._worker = None
        self.open_button.setEnabled(True)
        self.start_button.setEnabled(self._source_path is not None)
        self.stop_button.setEnabled(False)

    # ------------------------------------------------------------------ updates

    def _collect(self) -> None:
        """Pull the newest result. Called on the repaint timer."""
        if self._worker is None:
            return

        update = self._worker.take_latest()
        if update is None:
            return

        self.video.show_update(update)
        self.map.set_tracks(update.result.tracks)
        self._refresh_tracks(update)
        self.incidents.show_incidents(list(update.incidents))

    def _refresh_tracks(self, update: Update) -> None:
        info = self._worker.detector_info if self._worker else None
        tracks = sorted(update.result.tracks, key=lambda t: t.id)

        # Rebuilt rather than diffed. At the handful of objects one camera sees
        # this costs nothing, and a table that reuses rows can show a stale value
        # in a column that failed to update — which in an evidence view is worse
        # than a flicker.
        self.tracks.clear()
        for track in tracks:
            duration = (track.last_seen_millis - track.first_seen_millis) / 1000.0
            gap = update.result.timestamp_millis - track.last_seen_millis

            if track.position is not None:
                where = f"{track.position.point.lat:+.6f}, {track.position.point.lon:+.6f}"
                radius = f"±{track.position.radius_meters:.1f} m"
                origin = "projected" if track.position.source == "GROUND_PROJECTION" else "fallback"
            else:
                # Never blank: a blank cell reads as zero. This is a statement.
                where, radius, origin = "not placed", "—", "no pose"

            if track.speed_mps is None:
                speed = "unknown"
            elif track.speed_mps < 0.3:
                speed = "still"
            else:
                speed = f"{track.speed_mps:.2f} m/s"

            heading = (
                f"{track.heading_degrees:.0f}°" if track.heading_degrees is not None else "—"
            )

            item = QTreeWidgetItem([
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

            colour = (
                theme.TRACK_COASTING if gap > theme.COASTING_AFTER_MILLIS else theme.TRACK
            )
            item.setForeground(0, colour)
            self.tracks.addTopLevelItem(item)

        stats = update.stats
        self._set_status(
            f"frame {update.result.index}   "
            f"{update.analysis_fps:.0f} fps analysed   "
            f"{len(tracks)} tracked now   "
            f"{stats.distinct_objects} distinct objects so far   "
            f"{stats.events} events -> {len(update.incidents)} incidents"
        )

    def _on_finished(self, reason: str) -> None:
        # One last collection before the timer stops. Without it the final
        # frames — and the incidents correlated from them — are produced and
        # then thrown away, so a file that ends on an intrusion shows nothing.
        self._collect()
        self._teardown()
        self._set_status(reason)

    def _on_failed(self, message: str) -> None:
        """A running source failed.

        Reported in place rather than as a modal dialog. A modal is right for
        something the operator just asked for and which did not work; it is
        wrong for a camera dropping on its own, because that happens to twenty
        cameras at once when a switch loses power, and the operator would face a
        stack of dialogs each of which must be dismissed before anything else
        can be done — including looking at the cameras that are still working.

        The message came from DecodeError, which is redacted by construction, so
        it is safe to put in front of a person.
        """
        self._teardown()
        self.fault_label.setText(message)
        self.fault_label.setVisible(True)
        self._set_status(message)

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
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait(3000)
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
