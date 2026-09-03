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

from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent, QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
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
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from sentinel.core import (
    CameraPose,
    LatLon,
    destination_point,
    field_of_view,
    haversine_distance,
)
from sentinel.decode import DecodeError, VideoSource
from sentinel.detect import DetectionError, detector_for
from sentinel.events import (
    AfterHoursRule,
    LoiteringRule,
    RapidMovementRule,
    ZoneEntryRule,
)
from sentinel.evidence import ExportError, export_incident
from sentinel.node import Node, NodeError, Update
from sentinel.paths import default_model_path
from sentinel.store import default_database_path
from sentinel import devices, logs, telemetry
from sentinel.zones import Zone, ZoneKind

from . import theme
from .incident_view import IncidentView
from .map_view import MapView
from .placement import PlacementDialog
from .session import CameraSession
from .add_camera import AddCameraDialog
from .video_view import VideoView
from .zones_view import ZoneDialog, ZonePropertiesPanel, ZonesView

_log = logs.get(__name__)

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


def _detector_summary(info) -> str:
    """One line describing what is drawing the conclusions.

    Says what the detector *cannot* do as plainly as what it can. An operator
    reading "does not classify" beside a track labelled `unclassified` learns
    something; one reading a model name beside the same track would assume the
    model looked and found nothing recognisable, which is the opposite of true.
    """
    if info is None:
        return "No detector running"
    if not info.classifies:
        return f"{info.name} — does not classify, and cannot see a stationary object"
    masks = " with masks" if info.kind.endswith("segment") else ", boxes only"
    digest = f" · {info.model_sha256[:12]}" if info.model_sha256 else ""
    return f"{info.name} — {len(info.class_names)} classes{masks}{digest}"


#: The least height the incident and track panels are ever given. See `_build`.
LOWER_PANEL_MINIMUM_HEIGHT = 210


class ConsoleWindow(QMainWindow):
    """The main window."""

    def __init__(
        self, database: str | Path | None = None, model: str | Path | None = None
    ):
        """
        ``database`` is the path to persist to. ``":memory:"`` runs the console
        without keeping anything, which is right for a test and wrong for a
        deployment — a system whose output is evidence that forgets on restart
        has not really produced evidence at all.
        """
        super().__init__()
        self.setWindowTitle("Sentinel Vision — Console")
        self.resize(1500, 920)
        self.setStyleSheet(theme.STYLESHEET)

        self._sessions: dict[str, CameraSession] = {}

        # `None` means motion detection, and it means it *explicitly*. Finding a
        # model on disk and using it is a decision about what the system can
        # conclude, so it is made once, out loud, at the entry point in `run()`
        # — not here, where a test constructing a window would silently acquire
        # a different detector than the one it was written against.
        self._model = Path(model) if model is not None else None
        # Bound to the *value*, never to `self`. A factory that closed over the
        # window put the window in a reference cycle with its own node, so it
        # was no longer freed when its last reference went but whenever the
        # cyclic collector got to it — for the last few windows a test run
        # creates, that is interpreter shutdown, after PySide has already torn
        # down the QApplication. Destroying a QMainWindow at that point
        # corrupted the heap (0xC0000374) at exit, in a run where every test had
        # passed. A widget's lifetime has to be the plain refcount.
        model_for_detector = self._model

        # The console is a *client* of this. It owns no store, no zones, no
        # rule set and no analysis thread; it owns widgets, and it calls
        # `poll()` on a repaint timer. The same object runs a worker node with
        # no display, which is the point: one analysis loop, not two that drift.
        self.node = Node(
            database if database is not None else default_database_path(),
            actor="console",
            # A viewer needs the frame the conclusions were drawn from, and
            # needs a file paced to its own timeline rather than flashing past.
            keep_images=True,
            realtime=True,
            correlate_every_millis=CORRELATE_INTERVAL_MILLIS,
            # One detector per camera, never shared, built at start.
            detector_factory=lambda: detector_for(model_for_detector),
        )

        self._build()
        self._restore_cameras()

        self._timer = QTimer(self)
        self._timer.setInterval(REPAINT_INTERVAL_MILLIS)
        self._timer.timeout.connect(self._collect)

    @property
    def store(self):
        """The node's store. The console does not own one."""
        return self.node.store

    @property
    def _zones(self) -> list:
        return list(self.node.zones)

    @property
    def _incidents(self) -> list:
        return list(self.node.incidents)

    def _restore_cameras(self) -> None:
        """Show the cameras this node already had.

        Placements used to be written and never read, so every restart brought
        the cameras back unplaced — or rather, did not bring them back at all.
        The node restores them; this gives each one a pane.
        """
        for record in self.node.cameras:
            self._attach(record)
        if self._sessions:
            self._relayout_wall()
            self.start_button.setEnabled(True)
            self._refresh_placement()

    def _attach(self, record) -> CameraSession:
        """Give a node camera a pane, a row in the picker and a place on the wall."""
        view = VideoView()
        view.set_placeholder(f"{record.camera_id} — not started")
        view.set_show_detections(self.show_detections.isChecked())

        session = CameraSession(record=record, view=view, node=self.node)
        self._sessions[record.camera_id] = session
        self.camera_picker.addItem(record.camera_id, record.camera_id)
        self.camera_picker.setCurrentIndex(self.camera_picker.count() - 1)
        return session

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
        self.map.picked.connect(self._map_picked)
        self.map.drawn.connect(self._zone_drawn)
        self.map.edited.connect(self._zone_outline_edited)
        self.map.zone_clicked.connect(self._zone_clicked_on_map)
        #: What the next picked map point is for: ("zone", (name, kind, radius))
        #: or ("camera", camera_id). Nothing, when nobody is picking.
        self._pick_action: tuple | None = None
        self.tracks = self._build_track_table()
        self.incidents = IncidentView()
        self.zones_view = ZonesView()
        self.zones_view.itemSelectionChanged.connect(self._zone_selection_changed)
        self.zone_properties = ZonePropertiesPanel()
        self.zone_properties.changed.connect(self._zone_properties_applied)
        self.zone_properties.set_clock(self.node.site_clock_label)

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
        # Tracks and zones share the right-hand panel as tabs: both are how the
        # system reached its conclusions, one observed and one configured, and
        # neither deserves a third of the screen all the time.
        self.detail_tabs = QTabWidget()
        self.detail_tabs.addTab(self.tracks, "Tracked objects")
        self.detail_tabs.addTab(self._build_zones_panel(), "Zones")
        lower.addWidget(_panel("TRACKED OBJECTS · ZONES", self.detail_tabs))
        lower.setStretchFactor(0, 3)
        lower.setStretchFactor(1, 2)

        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(top)
        split.addWidget(lower)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        # The conclusions must never be squeezed out of sight. A live screenshot
        # on a short window showed the cameras taking every pixel the video's
        # own minimum size demanded, and the incident and track panels reduced
        # to a header row with no rows under it — "2 tracked now" in the status
        # bar and an empty table above it. Enough for a title, a header and four
        # rows, and the splitter cannot collapse it.
        lower.setMinimumHeight(LOWER_PANEL_MINIMUM_HEIGHT)
        split.setCollapsible(1, False)

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

        self.map_place_button = QPushButton("Move on map")
        self.map_place_button.setToolTip(
            "Move the selected camera to a point you click on the plan view. "
            "Its height, heading and optics are kept; place it once with Place… "
            "first, because a click cannot say which way it faces."
        )
        self.map_place_button.clicked.connect(self._place_camera_on_map)
        row.addWidget(self.map_place_button)

        self.remove_button = QPushButton("Remove camera")
        self.remove_button.setToolTip(
            "Forget the selected camera. It is stopped first if it is running. "
            "What it saw — its events and incidents — is kept."
        )
        self.remove_button.clicked.connect(self._remove_camera)
        row.addWidget(self.remove_button)

        self.zone_button = QPushButton("Add zone…")
        self.zone_button.setToolTip(
            "Adds a zone on the ground: a restricted area, the perimeter, an "
            "entry, an exclusion, or an area of interest — in front of the "
            "selected camera or at a point you click on the plan view. Needs a "
            "placed camera: a zone without one has nothing to be measured against."
        )
        self.zone_button.clicked.connect(self._add_zone_dialog)
        row.addWidget(self.zone_button)

        self.export_button = QPushButton("Export incident…")
        self.export_button.setEnabled(False)
        self.export_button.setToolTip(
            "Writes the selected incident, its evidence and a readable report "
            "to a folder, with a SHA-256 for every file so any later alteration "
            "is detectable."
        )
        self.export_button.clicked.connect(self._export_incident)
        row.addWidget(self.export_button)

        self.zone_radius = QDoubleSpinBox()
        self.zone_radius.setRange(2.0, 200.0)
        self.zone_radius.setValue(10.0)
        self.zone_radius.setSuffix(" m")
        # Wide enough for "200.00 m" plus the arrows. At 80 px a screenshot of
        # the real thing showed "12.0(" — the value cut off mid-number, which
        # reads as a broken control rather than a narrow one.
        self.zone_radius.setMinimumWidth(104)
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

    def _build_zones_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        row = QHBoxLayout()
        add = QPushButton("Add zone…")
        add.setToolTip("A square of a chosen size, in front of the camera or at a clicked point.")
        add.clicked.connect(self._add_zone_dialog)
        row.addWidget(add)
        self.draw_zone_button = QPushButton("Draw zone")
        self.draw_zone_button.setToolTip(
            "Draw any outline on the plan view: click each corner, double-click "
            "or Enter to close, right-click to undo a corner, Esc to abandon."
        )
        self.draw_zone_button.clicked.connect(self._draw_zone)
        row.addWidget(self.draw_zone_button)
        self.reshape_zone_button = QPushButton("Reshape")
        self.reshape_zone_button.setToolTip(
            "Edit the selected zone's outline on the plan view: drag a corner, "
            "click an edge to add one, right-click a corner to remove it, drag "
            "inside to move the whole zone. Enter applies, Esc reverts."
        )
        self.reshape_zone_button.clicked.connect(self._edit_outline)
        row.addWidget(self.reshape_zone_button)
        self.remove_zone_button = QPushButton("Remove")
        self.remove_zone_button.setToolTip(
            "Forget the selected zone. Events it raised are kept and still name it."
        )
        self.remove_zone_button.clicked.connect(self._remove_zone)
        row.addWidget(self.remove_zone_button)
        row.addStretch(1)
        layout.addLayout(row)

        # The list and the properties of whichever row is selected, side by
        # side: an operator reading a zone's schedule should not lose sight of
        # which zone it is.
        body = QSplitter(Qt.Orientation.Horizontal)
        body.addWidget(self.zones_view)
        body.addWidget(self.zone_properties)
        body.setStretchFactor(0, 3)
        body.setStretchFactor(1, 2)
        body.setCollapsible(1, False)
        layout.addWidget(body, 1)
        return panel

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

    def add_camera(self, source: str | Path, camera_id: str | None = None) -> CameraSession:
        """Register a source as a camera. Does not start it, and opens nothing.

        ``source`` is a file path, an RTSP URL, or ``device:N`` for a camera
        attached to this machine. Nothing here distinguishes between them —
        that is `VideoSource`'s job — so adding a webcam is the same operation
        as adding a clip.
        """
        text = str(source)
        default = (
            Path(text).stem
            if "://" not in text and not text.startswith("device:")
            else text
        )
        identifier = camera_id or default or f"cam-{len(self._sessions) + 1:02d}"
        if identifier in self._sessions:
            identifier = f"{identifier}-{len(self._sessions) + 1}"

        record = self.node.add_camera(text, camera_id=identifier)
        session = self._attach(record)

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
        """Add a camera: one attached to this machine, one on the network, or a file.

        The dialog enumerates local cameras through the operating system's own
        device interface and opens none of them to do it — so this does not
        light a webcam, and on macOS does not raise a permission prompt for a
        camera nobody asked to use.
        """
        dialog = AddCameraDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        added = []
        for choice in dialog.chosen:
            try:
                session = self.add_camera(choice.source, camera_id=choice.suggested_id)
            except NodeError as error:
                # The node's message names the existing camera and carries the
                # redacted source only. Shown, not logged and swallowed: the
                # operator just asked for this and needs to know why not.
                QMessageBox.information(self, "Already a camera", str(error))
                continue
            added.append(session)
            # The display form, never the raw one: this line goes to a log file.
            _log.info("added camera %s from %s", session.camera_id, choice.display)

        if not added:
            return

        if any(devices.is_device_source(session.source) for session in added):
            self._set_status(
                f"{len(self._sessions)} camera(s). Press Start and check the "
                "picture is the camera you meant, then place it."
            )
        else:
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
        # Parented to the window, so without this every placement leaves another
        # dialog alive for the life of the console.
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        accepted = dialog.exec() == PlacementDialog.DialogCode.Accepted
        pose = dialog.pose() if accepted else None
        if not accepted:
            return

        # One call. It assigns the pose, pushes it to the running analysis so
        # existing tracks keep their identity, persists it with the *redacted*
        # source, and audits it — and it is the same call a daemon makes, so
        # the two cannot drift about what placing a camera means.
        self.node.place_camera(session.camera_id, pose)
        self._refresh_placement()

    def _refresh_placement(self) -> None:
        placed = {
            session.camera_id: session.pose
            for session in self._sessions.values()
            if session.pose is not None
        }
        self.map.set_cameras(placed)
        self.map.set_zones(self._zones)
        self.zones_view.show_zones(self._zones)
        self._sync_zone_properties()

        if not placed:
            self.placement_label.setText("No camera placed — objects will not be located")
        elif len(placed) == len(self._sessions):
            self.placement_label.setText(f"{len(placed)} camera(s) placed")
        else:
            self.placement_label.setText(
                f"{len(placed)} of {len(self._sessions)} cameras placed — "
                "the rest will not locate anything"
            )

    def _remove_camera(self) -> None:
        """Forget the selected camera, stopping it first if it is running.

        Asked before doing it, because the pane, the row and the placement go
        and there is no undo — though what the camera saw is kept, and the
        confirmation says so.
        """
        session = self._selected
        if session is None:
            QMessageBox.information(self, "No camera", "There is no camera to remove.")
            return
        answer = QMessageBox.question(
            self,
            "Remove camera",
            f"Remove {session.camera_id} ({session.display_source})?\n\n"
            + ("It is running and will be stopped first. " if session.is_running else "")
            + "Its events and incidents are kept.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.node.remove_camera(session.camera_id)
        except NodeError as error:
            QMessageBox.warning(self, "Not removed", str(error))
            return

        self._sessions.pop(session.camera_id, None)
        index = self.camera_picker.findData(session.camera_id)
        if index >= 0:
            self.camera_picker.removeItem(index)
        session.view.setParent(None)
        session.view.deleteLater()
        self._relayout_wall()
        self._refresh_placement()
        self.start_button.setEnabled(bool(self._sessions) and not self._running)
        if not self._running:
            self._teardown()
        self._set_status(f"Removed {session.camera_id}. {len(self._sessions)} camera(s).")

    def _place_camera_on_map(self) -> None:
        """Move the selected camera to a point clicked on the plan view."""
        session = self._selected
        if session is None:
            QMessageBox.information(self, "No camera", "Add a camera first.")
            return
        if session.pose is None:
            QMessageBox.information(
                self,
                "Place it once first",
                "A click on the map gives a position, not a height or a heading, "
                "and both decide where this camera's objects land on the ground. "
                "Use Place… once; after that the camera can be moved by clicking.",
            )
            return
        self._pick_action = ("camera", session.camera_id)
        if not self.map.begin_pick(f"Move {session.camera_id} to"):
            self._pick_action = None
            QMessageBox.information(
                self, "The map has no origin",
                "Place a camera with Place… first; the map cannot measure a click "
                "until it knows where one camera is.",
            )
            return
        self._set_status(f"Click the plan view where {session.camera_id} is.")

    def _map_picked(self, point: LatLon) -> None:
        """A ground point the operator clicked, for whatever asked for it."""
        action, self._pick_action = self._pick_action, None
        if action is None:
            return
        what, payload = action
        if what == "camera":
            session = self._sessions.get(payload)
            if session is None or session.pose is None:
                return
            self.node.place_camera(payload, replace(session.pose, position=point))
            self._refresh_placement()
            self._set_status(f"Moved {payload}.")
        elif what == "zone":
            name, kind, radius = payload
            self._add_zone(name=name, kind=kind, radius=radius, centre=point)

    def _add_zone_dialog(self) -> None:
        """Ask what kind of zone, how big, and where, then create it."""
        if self._pose is None and not any(s.pose for s in self._sessions.values()):
            QMessageBox.information(
                self,
                "Place the camera first",
                "A zone is an area on the ground. Until a camera is placed there "
                "is nothing to measure it against, and objects are tracked but "
                "not located.",
            )
            return
        dialog = ZoneDialog(
            default_radius=self.zone_radius.value(),
            can_pick=any(s.pose is not None for s in self._sessions.values()),
            parent=self,
        )
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        name, kind, radius = dialog.name() or None, dialog.kind(), dialog.radius()
        if dialog.pick_on_map():
            self._pick_action = ("zone", (name, kind, radius))
            self.map.begin_pick(f"Centre of {name or kind.value.title()}")
            self._set_status("Click the plan view where the zone's centre is.")
            return
        self._add_zone(name=name, kind=kind, radius=radius)

    # ------------------------------------------------------- zones: outlines

    def _placed_anywhere(self) -> bool:
        return any(s.pose is not None for s in self._sessions.values())

    def _draw_zone(self) -> None:
        """Draw a zone of any shape on the plan view."""
        if not self._placed_anywhere():
            QMessageBox.information(
                self,
                "Place the camera first",
                "A zone is an area on the ground. Until a camera is placed the map "
                "has no origin to draw against.",
            )
            return
        self.map.begin_draw("Draw a zone")
        self._set_status(
            "Drawing a zone: click each corner on the plan view, double-click or "
            "Enter to close it, right-click to undo, Esc to abandon."
        )

    def _zone_drawn(self, ring) -> None:
        """An outline was closed on the map; ask what it is, then create it."""
        dialog = ZoneDialog(ring_given=True, parent=self)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            self._set_status("Zone abandoned.")
            return
        self._create_zone(tuple(ring), name=dialog.name() or None, kind=dialog.kind())

    def _edit_outline(self) -> None:
        zone_id = self.zones_view.selected_zone_id()
        if zone_id is None or not self.map.begin_edit(zone_id):
            QMessageBox.information(self, "No zone selected", "Select a zone to reshape.")
            return
        self.detail_tabs.setCurrentIndex(1)
        self._set_status(
            "Reshaping: drag a corner, click an edge to add one, right-click a "
            "corner to remove it, drag inside to move. Enter applies, Esc reverts."
        )

    def _zone_outline_edited(self, zone_id: str, ring) -> None:
        zone = next((z for z in self._zones if z.id == zone_id), None)
        if zone is None:
            return
        try:
            reshaped = replace(zone, ring=tuple(ring))
        except ValueError as error:
            # The engine refused the outline (a figure of eight, a line). The
            # stored zone is untouched; say why, in the engine's words.
            QMessageBox.warning(self, "Outline not usable", str(error))
            return
        self.node.replace_zone(reshaped)
        self._refresh_placement()
        self.zones_view.select(zone_id)
        self._set_status(f"{zone.name} reshaped: {len(ring)} corners.")

    def _zone_selection_changed(self) -> None:
        zone_id = self.zones_view.selected_zone_id()
        self.map.select_zone(zone_id)
        self._sync_zone_properties()

    def _sync_zone_properties(self) -> None:
        zone_id = self.zones_view.selected_zone_id()
        zone = next((z for z in self._zones if z.id == zone_id), None)
        self.zone_properties.show_zone(zone)

    def _zone_clicked_on_map(self, zone_id: str) -> None:
        self.zones_view.select(zone_id)
        self.detail_tabs.setCurrentIndex(1)

    def _zone_properties_applied(self, zone: Zone) -> None:
        self.node.replace_zone(zone)
        self._refresh_placement()
        self.zones_view.select(zone.id)
        self._set_status(f"{zone.name} updated.")

    def _change_zone(self, zone_id: str, *, name: str, kind: ZoneKind) -> None:
        zone = next((z for z in self._zones if z.id == zone_id), None)
        if zone is None:
            raise ValueError(f"no zone {zone_id!r}")
        self.node.replace_zone(replace(zone, name=name, kind=kind))
        self._refresh_placement()
        self.zones_view.select(zone_id)
        self._set_status(f"{name} is now {kind.value.lower()}.")

    def _remove_zone(self) -> None:
        zone_id = self.zones_view.selected_zone_id()
        zone = next((z for z in self._zones if z.id == zone_id), None)
        if zone is None:
            QMessageBox.information(self, "No zone selected", "Select a zone to remove.")
            return
        answer = QMessageBox.question(
            self, "Remove zone",
            f"Remove {zone.name} ({zone.kind.value.lower()})? Events it raised are kept.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.node.remove_zone(zone.id)
        self._refresh_placement()
        self._set_status(f"Removed {zone.name}. {len(self._zones)} zone(s).")

    def _add_zone(
        self,
        name: str | None = None,
        kind: ZoneKind = ZoneKind.RESTRICTED,
        radius: float | None = None,
        centre: LatLon | None = None,
    ) -> None:
        """Put a zone on the ground: in front of the selected camera, or at
        ``centre``.

        What it is not is a default: a zone exists only because an operator
        asked for it, because a zone nobody drew is a source of alerts nobody
        expects.
        """
        pose = self._pose
        if pose is None and centre is None:
            QMessageBox.information(
                self,
                "Place the camera first",
                "A zone is an area on the ground. Until the camera is placed "
                "there is nothing to measure it against, and objects are tracked "
                "but not located.",
            )
            return

        radius = self.zone_radius.value() if radius is None else radius

        # Placed just beyond the near edge of what this camera can *actually*
        # see, which is not the range it claims: a 6 m mast tilted 22 degrees
        # with a 36 degree vertical field covers 7 m to 86 m however large the
        # stated range is. The near end is also where positions are most
        # accurate, because uncertainty grows super-linearly with distance — so
        # a zone there is one the system can genuinely adjudicate rather than
        # one it will mostly report as UNCERTAIN.
        if centre is None:
            assert pose is not None
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
        self._create_zone(ring, name=name, kind=kind)

    def _create_zone(
        self, ring: tuple, *, name: str | None, kind: ZoneKind
    ) -> Zone | None:
        """Make a zone from an outline. Returns it, or ``None`` if refused."""
        # The first id not in use, not "count plus one": after zone-1 is
        # removed, count-plus-one names zone-2 again and the upsert silently
        # overwrites the zone that is still there.
        taken = {z.id for z in self.node.zones}
        index = 1
        while f"zone-{index}" in taken:
            index += 1
        letter = chr(64 + min(index, 26))
        if name is None:
            name = (
                f"Restricted Area {letter}"
                if kind is ZoneKind.RESTRICTED
                else f"{kind.value.title()} zone {letter}"
            )
        try:
            zone = Zone(
                id=f"zone-{index}",
                name=name,
                kind=kind,
                ring=tuple(ring),
                enter_after_millis=600,
                # A zone meant to be ignored, or merely watched, may accept an
                # uncertain position; one that raises an alarm must not.
                accept_uncertain=kind in (ZoneKind.EXCLUSION, ZoneKind.INTEREST),
            )
        except ValueError as error:
            # A figure of eight, or points on a line. The engine's own words,
            # because they name the problem.
            QMessageBox.warning(self, "Outline not usable", str(error))
            return None
        # The node persists it, audits it, and — if this is the first zone —
        # rebuilds the rule set, because rules that need a zone are dead weight
        # until there is one and must not stay dead once there is.
        self.node.add_zone(zone)

        self.map.set_zones(self._zones)
        self.zones_view.show_zones(self._zones)
        self.zones_view.select(zone.id)
        self._set_status(
            f"{len(self._zones)} zone(s). Rules apply to cameras started from now."
        )
        return zone

    # ------------------------------------------------------------------ running

    def _export_incident(self) -> None:
        """Write the selected incident out as an evidence package.

        Audited, because who exported what and when is part of the chain of
        custody and is exactly the question asked when a package turns up
        somewhere it should not have.
        """
        incident = self._selected_incident()
        if incident is None:
            QMessageBox.information(
                self, "No incident selected", "Select an incident to export."
            )
            return

        destination = QFileDialog.getExistingDirectory(
            self, "Export evidence to", str(Path.home())
        )
        if not destination:
            return

        try:
            export = export_incident(
                incident,
                Path(destination),
                # No authentication yet, so there is nobody to name. Recording
                # "console" is the truth; inventing an operator name would be a
                # false entry in a chain of custody.
                exported_by="console (unauthenticated)",
            )
        except ExportError as error:
            QMessageBox.warning(self, "Export failed", str(error))
            return

        self.store.audit(
            "console", "incident.exported", incident.id, str(export.directory)
        )
        QMessageBox.information(
            self,
            "Evidence exported",
            "\n".join([
                incident.id,
                "",
                f"{len(export.files)} files written to",
                str(export.directory),
                "",
                "Manifest SHA-256:",
                export.manifest_sha256,
                "",
            ])
            + "Record that digest separately. It is what makes the package "
            "checkable later.",
        )

    def _selected_incident(self):
        item = self.incidents.currentItem()
        while item is not None and item.parent() is not None:
            item = item.parent()
        if item is None:
            return self._incidents[0] if self._incidents else None

        chosen = item.data(0, Qt.ItemDataRole.UserRole)
        return next((i for i in self._incidents if i.id == chosen), None)

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

        # Probed here rather than inside the node, because a *daemon* must not
        # block start-up on an unreachable camera while an *interface* should
        # say so at once: the operator asked for this, just now, and is waiting.
        for session in list(self._sessions.values()):
            if self.node.needs_credentials(session.camera_id):
                QMessageBox.warning(
                    self, f"{session.camera_id} needs its password",
                    "This camera was restored from the database, which never "
                    "stores a password — that is deliberate. Remove it and add "
                    "it again with its credentials.",
                )
                return
            try:
                self.node.probe(session.camera_id)
            except DecodeError as error:
                QMessageBox.warning(
                    self, f"Cannot open {session.camera_id}", str(error)
                )
                return

        # The model is loaded once here for the same reason the cameras are
        # probed here: the factory runs inside each camera's own thread, so a
        # broken model file would otherwise fail sixteen times somewhere the
        # operator cannot see, leaving a window that started and shows nothing.
        if self._model is not None:
            try:
                detector_for(self._model)
            except DetectionError as error:
                QMessageBox.warning(self, "Cannot load the detection model", str(error))
                return

        started = self.node.start()
        if started == 0:
            return

        # What is actually running, asked of the running thing. This used to be
        # a constant naming MOG2, which was true only for as long as MOG2 was
        # the only option — and a capability label that can be wrong is worse
        # than none, because it is the line an operator reads to decide whether
        # a classification means anything.
        info = None
        for session in self._sessions.values():
            runner = session.record.runner
            info = runner.detector_info if runner is not None else None
            session.view.set_detector_info(info)

        self.detector_label.setText(_detector_summary(info))
        self._timer.start()

        self.open_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._set_status(f"Running {started} camera(s).")

    def _stop(self) -> None:
        """Stop every camera.

        Signalled first, waited on second. Doing both per camera in one pass
        blocks the interface for up to the full timeout *per camera*, so a wall
        of sixteen could freeze for the better part of a minute — while the
        operator watches a window that has stopped responding.
        """
        ended = self.node.stop()
        stubborn = [] if ended else [
            s.camera_id for s in self._sessions.values() if s.is_running
        ]

        # One last collection, so the final frames and whatever the node
        # concluded from them reach the screen. Releasing first is how the last
        # seconds of a run get discarded — a file ending on an intrusion used to
        # show nothing.
        self._collect()
        self._teardown()
        if stubborn:
            self._set_status(
                f"Stopped. {', '.join(stubborn)} did not stop cleanly and is "
                "still running."
            )
        else:
            self._set_status("Stopped.")

    def _teardown(self) -> None:
        self._timer.stop()
        self.open_button.setEnabled(True)
        self.start_button.setEnabled(bool(self._sessions))
        self.stop_button.setEnabled(False)

    # ------------------------------------------------------------------ updates

    def _collect(self) -> None:
        """Move the node forward and draw what came back. On the repaint timer.

        `poll` is the whole engine step: it drains each camera's events,
        persists them, indexes any recorded segment, notices faults and
        correlates when due. The console does none of that any more; it draws.
        """
        updates = self.node.poll()

        for update in updates:
            session = self._sessions.get(update.result.source_id)
            if session is None:
                continue
            session.absorb(update)
            session.view.show_update(update)
            self.map.set_tracks(update.result.tracks, session.camera_id)

        self._show_faults()
        self.incidents.show_incidents(self._incidents)
        self.export_button.setEnabled(bool(self.node.incidents))

        if updates:
            self._refresh_tracks()
        self._refresh_status()

        if self._running and not any(s.is_running for s in self._sessions.values()):
            # Every camera has ended on its own. For files that is completion.
            self._teardown()
            self._set_status("Finished.")

    def _show_faults(self) -> None:
        """Report which camera is in trouble, in place and never modally.

        Twenty cameras drop together when a switch loses power, and twenty
        dialogs is not a user interface — it is a wall between the operator and
        the cameras that still work.
        """
        faulted = [s for s in self._sessions.values() if s.fault]
        if not faulted:
            self.fault_label.setVisible(False)
            return

        first = faulted[0]
        more = f" (+{len(faulted) - 1} more)" if len(faulted) > 1 else ""
        self.fault_label.setText(f"{first.camera_id}: {first.fault}{more}")
        self.fault_label.setVisible(True)
        for session in faulted:
            session.view.set_placeholder(f"{session.camera_id} — {session.fault}")

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

            runner = session.record.runner
            info = runner.detector_info if runner is not None else None
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
        """One line describing the state, which must not contradict the screen.

        It used to say "0 camera(s)   0 tracked now" beside a table listing four
        tracked objects — both statements true, of different instants, and
        together they read as a broken interface. A run that has ended says so
        instead, and the counts it then reports are the ones still on screen.
        """
        running = [s for s in self._sessions.values() if s.is_running]
        events = sum(len(s.events) for s in self._sessions.values())
        tail = f"{events} events -> {len(self._incidents)} incidents"

        if running:
            tracked = sum(
                len(s.last.result.tracks) for s in running if s.last is not None
            )
            self._set_status(f"{len(running)} camera(s)   {tracked} tracked now   {tail}")
            return

        if not self._sessions:
            self._set_status("No cameras. Add one to begin.")
            return

        faulted = [s for s in self._sessions.values() if s.fault]
        state = (
            f"{len(faulted)} of {len(self._sessions)} camera(s) faulted"
            if faulted
            else f"{len(self._sessions)} camera(s) stopped"
        )
        self._set_status(f"{state}   {tail}")

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
        """Shut down in the order that cannot leave something writing.

        An earlier version closed the database while both timers were still
        armed and worker signals were still queued, so the next 33 ms tick ran
        `_correlate()` against a closed connection and raised inside an event
        handler. That whole class of bug is now gone by construction: there are
        no queued signals to arrive late, because collection is a direct call.

        The timer stops first so nothing polls a closing node. `Node.close`
        does the rest — signal every camera, wait for all of them, correlate
        once more so the last seconds of a run are not discarded, and close the
        database.
        """
        self._timer.stop()
        self.node.close()
        event.accept()


def run(argv: list[str] | None = None) -> int:
    telemetry.silence()

    """Start the console. The console-script and packaged entry point.

    Two flags only, because everything else an operator sets belongs in the
    window rather than on a command line they will not see:

    ``--verbose`` turns on developer logging — DEBUG, with thread, file and
    line. It is what the ``-dev`` executable passes, and it is why that
    executable exists: a packaged operator build has no terminal to read.

    ``--database`` points at a specific database, which is how two deployments
    share a machine.
    """
    import argparse
    import sys

    from PySide6.QtWidgets import QApplication
    from sentinel import logs

    parser = argparse.ArgumentParser(
        prog="sentinel-console", description="Sentinel Vision operator console."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="developer logging: DEBUG, with thread, file and line",
    )
    parser.add_argument("--database", default=None, help="database to open")
    parser.add_argument(
        "--model", default=None, metavar="FILE",
        help=(
            "an ONNX model to detect with. Defaults to a *-seg.onnx in the "
            "models directory if one is there. Operator-supplied — nothing is "
            "ever downloaded."
        ),
    )
    parser.add_argument(
        "--no-model", action="store_true",
        help="ignore any installed model and detect motion only",
    )
    arguments, unknown = parser.parse_known_args(sys.argv[1:] if argv is None else argv)

    logs.configure(
        level="DEBUG" if arguments.verbose else None, developer=arguments.verbose
    )
    log = logs.get(__name__)
    if unknown:
        # Qt takes its own arguments (-platform, -style). Passing them through
        # rather than refusing them keeps `-platform offscreen` working, which
        # is what the tests and any headless check rely on.
        log.debug("passing %d argument(s) through to Qt", len(unknown))

    log.info("console starting")

    app = QApplication([sys.argv[0], *unknown] if unknown else sys.argv[:1])
    app.setApplicationName("Sentinel Vision Console")
    app.setOrganizationName("Sentinel Vision")

    try:
        # Resolved here, and logged, because "which detector am I running"
        # must never be something an operator has to infer.
        model = None
        if not arguments.no_model:
            model = Path(arguments.model) if arguments.model else default_model_path()

        if model is None:
            log.info(
                "no detection model: running on motion detection, which does "
                "not classify and cannot see a stationary object"
            )
        else:
            log.info("detection model: %s", model)

        window = ConsoleWindow(database=arguments.database, model=model)
        window.show()
        code = app.exec()
    except Exception:
        # A packaged build has no terminal, so an unhandled exception would
        # otherwise close the window with no trace of why. The log survives it.
        log.critical("the console failed to start", exc_info=True)
        raise

    log.info("console exited with %d", code)
    return code
