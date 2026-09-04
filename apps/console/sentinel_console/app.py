"""The console window.

A native Qt application. No embedded browser, no web view, no HTML — the widgets
paint directly. That is not stylistic: a control-room console that ships a
browser engine inherits its update cadence, its memory profile and its network
assumptions, and this system is meant to run for months on a machine with no
route to the Internet.

The layout answers four questions, in the order an operator asks them:

1. *What is happening?* — the camera list on the far left, one row per camera
   with what that camera is actually doing, and the wall of pictures beside it.
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
from PySide6.QtGui import QKeySequence, QShortcut, QAction, QCloseEvent, QFont
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
    bearing_degrees,
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
from sentinel.coverage import sigma_bands, zone_report
from sentinel.node import Node, NodeError, Update
from sentinel.paths import default_model_path
from sentinel.store import default_database_path
from sentinel import devices, logs, telemetry
from sentinel.zones import Zone, ZoneKind, zone_warnings

from . import theme
from .camera_list import CameraListPanel
from .incident_view import IncidentView
from .map_view import MODE_DRAW, MODE_MEASURE, MODE_SELECT, MapView
from .placement import PlacementDialog
from .session import CameraSession
from .add_camera import AddCameraDialog
from .video_view import VideoView
from .selection import CAMERA as CAMERA_KIND, Selection, SelectionBus
from .investigation import InvestigationPanel
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


#: Who the console is, in the audit log. Not `node.ACTOR`: the node acts on
#: its own behalf when it restores or correlates, and an operator unlocking the
#: site is a different actor doing a different thing. A trail that cannot tell
#: them apart cannot answer "who changed this".
CONSOLE_ACTOR = "console"

#: How long the console stays in Configure with nobody touching it. A lock
#: that never re-arms is a lock somebody props open on the first day.
CONFIGURE_IDLE_MILLIS = 10 * 60 * 1000

def _plate_cell(plate) -> str:
    """The plate column for one track: the reading with its evidence.

    Empty for a track that is not a vehicle or has not been read — an empty
    cell is honest here because the column is named, and "—" would suggest a
    read was attempted. A resolved-but-unconfident reading is shown with the
    agreement count, so an operator sees "B7X4921 (3)" and knows three frames
    is thin; a confident one is shown plainly.
    """
    if plate is None:
        return ""
    if plate.is_confident:
        return plate.display
    return f"{plate.display} ({plate.agreement})"


#: The least height the incident and track panels are ever given. See `_build`.
LOWER_PANEL_MINIMUM_HEIGHT = 260


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
        view.camera_id = record.camera_id
        view.clicked.connect(self.selection.select)
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

        # The site is locked until somebody says otherwise. Set before the
        # toolbar is built, because the buttons read it as they are created.
        self._configuring = False
        self._idle_timer = QTimer(self)
        self._idle_timer.setSingleShot(True)
        self._idle_timer.timeout.connect(self._relock)

        # One selected thing, shared by every panel. Built before the views
        # so each can be wired to it as it is created.
        self.selection = SelectionBus(self)
        self.selection.changed.connect(self._selection_changed)
        #: Guards the round trip: a panel told to show a selection emits its own
        #: "the user picked a row" signal, which would come straight back here.
        self._syncing = False

        # Every camera, one row each, with the status strip that says which of
        # them is delivering frames. Built here rather than in the toolbar
        # because it is a view, not a control: the combo box beside Place… is
        # still where "which camera do these buttons act on" is *stored*, and
        # the two are kept saying the same thing in `_selection_changed`.
        self.camera_list = CameraListPanel()
        self.camera_list.selected.connect(self._camera_row_selected)

        self.map = MapView()
        self.map.picked.connect(self._map_picked)
        self.map.selected.connect(self.selection.select)
        self.map.ground_moved.connect(self._ground_moved)
        self.map.mode_changed.connect(self._mode_changed)
        self.map.drawn.connect(self._zone_drawn)
        self.map.edited.connect(self._zone_outline_edited)
        self.map.zone_clicked.connect(self._zone_clicked_on_map)
        # Dragging a mast on the plan view is a placement, so it commits through
        # the node like every other one. Wired to bound methods, never to a
        # lambda closing over `self`: a closure cell holding this window is the
        # reference cycle that keeps it alive until interpreter shutdown.
        self.map.camera_moved.connect(self._camera_dragged)
        self.map.camera_aimed.connect(self._camera_turned)
        #: What the next picked map point is for: ("zone", (name, kind, radius))
        #: or ("camera", camera_id). Nothing, when nobody is picking.
        self._pick_action: tuple | None = None
        self.tracks = self._build_track_table()
        self.tracks.itemSelectionChanged.connect(self._track_row_selected)
        self.tracks.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tracks.customContextMenuRequested.connect(self._track_menu)
        self.incidents = IncidentView()
        self.incidents.itemSelectionChanged.connect(self._incident_row_selected)
        self.zones_view = ZonesView()
        self.zones_view.itemSelectionChanged.connect(self._zone_selection_changed)
        self.zone_properties = ZonePropertiesPanel()
        self.zone_properties.changed.connect(self._zone_properties_applied)
        self.zone_properties.set_clock(self.node.site_clock_label)
        # The class picker offers only the labels this site's detector can
        # produce. A filter for "person" on a motion-only site would silence
        # the zone for ever, so with no vocabulary the picker is disabled and
        # says why (the panel does that; this only hands it the words).
        self.zone_properties.set_classes(self._detector_labels())

        outer.addLayout(self._build_toolbar())

        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(self.camera_list)
        top.addWidget(_panel("CAMERA WALL", self.wall))
        top.addWidget(_panel("GROUND — NO EXTERNAL TILES", self.map))
        top.setStretchFactor(0, 1)
        top.setStretchFactor(1, 3)
        top.setStretchFactor(2, 2)
        # Not collapsible. The list is the only place a dark camera announces
        # itself, and a splitter handle dragged shut on the first day would put
        # the console back to reporting eight cameras identically whether all
        # eight were delivering frames or seven had been dark since midnight.
        top.setCollapsible(0, False)

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
        # The investigation surface was built, tested, and placed in no window:
        # the one blocker the wiring pass could not close from outside this
        # file. It searches the node's own store, never a database of its own.
        self.investigation = InvestigationPanel(self.node.store)
        self.investigation.selected.connect(self.selection.select)
        self.detail_tabs.addTab(self.investigation, "Investigation")
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
        # Permanent, on the right: it answers "where is that?" continuously, and
        # a message that scrolled it away would make the plan view unreadable
        # for the one purpose it exists for — sending somebody to a place.
        self.ground_label = QLabel("")
        self.ground_label.setObjectName("Caption")
        self.ground_label.setToolTip(
            "The ground point under the pointer. Ctrl+C copies it."
        )
        # Bound, because the tooltip above promises it. It was written before
        # the shortcut and the promise went unkept for as long as it took a
        # review to read the two lines together.
        copy = QShortcut(QKeySequence.StandardKey.Copy, self)
        copy.setContext(Qt.ShortcutContext.WindowShortcut)
        copy.activated.connect(self._copy_ground)
        self.status.addPermanentWidget(self.ground_label)
        self._ground_text = ""

        self._set_status("Ready. Add a camera to begin.")
        self._build_menu()

        # Locked to start with. Last, and both halves of that matter: after the
        # status bar, because this puts the map into Select, which reports its
        # mode, which writes to the status bar — called earlier it raised inside
        # a Qt slot and the swallowed exception's traceback held the whole
        # window alive. And after the menu, because the menu's actions are
        # among the things being locked.
        self._set_configuring(False)

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
        # The picker and the camera list are two views of one choice. This is
        # the half that carries a change made here out to the bus; the other
        # half is in `_selection_changed`, which brings a change made anywhere
        # else back to the picker.
        self.camera_picker.currentIndexChanged.connect(self._picker_changed)
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

        row.addSpacing(12)

        # What the next click on the map does, as three buttons that are
        # visibly one choice. Before this the same click panned, moved a camera
        # or dropped a zone corner depending on state nothing on screen showed.
        self.mode_buttons: dict[str, QPushButton] = {}
        for mode, label, tip in (
            (MODE_SELECT, "Select", "Click to select, drag to pan, wheel to zoom."),
            (MODE_DRAW, "Draw", "Click each corner of a zone on the plan view."),
            (MODE_MEASURE, "Measure",
             "Click two points to measure the ground between them. Changes nothing."),
        ):
            button = QPushButton(label)
            button.setCheckable(True)
            button.setToolTip(tip)
            button.setChecked(mode == MODE_SELECT)
            button.clicked.connect(self._mode_button_clicked)
            self.mode_buttons[mode] = button
            row.addWidget(button)

        self.configure_button = QPushButton("Configure")
        self.configure_button.setCheckable(True)
        self.configure_button.setToolTip(
            "Unlock the controls that change the site: adding and removing "
            "cameras, placing them, and drawing, reshaping or removing zones. "
            f"Returns to Monitor on Escape, or after "
            f"{CONFIGURE_IDLE_MILLIS // 60000} idle minutes."
        )
        self.configure_button.toggled.connect(self._set_configuring)
        row.addWidget(self.configure_button)

        row.addSpacing(12)

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
        # Kept on the instance so the lock can reach it: this is a second
        # door to the same room as the toolbar's Add zone, and a lock that
        # covers one door is not a lock.
        self.add_zone_button = QPushButton("Add zone…")
        self.add_zone_button.setToolTip(
            "A square of a chosen size, in front of the camera or at a clicked point."
        )
        self.add_zone_button.clicked.connect(self._add_zone_dialog)
        row.addWidget(self.add_zone_button)
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
        body.setStretchFactor(1, 4)
        body.setCollapsible(1, False)
        body.setCollapsible(0, False)
        layout.addWidget(body, 1)
        return panel

    def _build_track_table(self) -> QTreeWidget:
        tree = QTreeWidget()
        tree.setRootIsDecorated(False)
        tree.setAlternatingRowColors(True)
        tree.setUniformRowHeights(True)
        tree.setHeaderLabels(
            ["Camera", "ID", "Class", "Confidence", "Seen", "Duration", "Speed",
             "Heading", "Position", "Uncertainty", "Source", "Plate"]
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

        # Kept, because these are the same two actions as the toolbar buttons
        # and the lock has to cover them: a shortcut that changes the site while
        # the console says it is locked is not a lock. Ctrl+O and Ctrl+P did
        # exactly that until an audit of every path to `node.*` found them.
        self.add_camera_action = QAction("&Add camera…", self)
        self.add_camera_action.setShortcut("Ctrl+O")
        self.add_camera_action.triggered.connect(self._choose_source)
        file_menu.addAction(self.add_camera_action)

        self.place_action = QAction("&Camera placement…", self)
        self.place_action.setShortcut("Ctrl+P")
        self.place_action.triggered.connect(self._place_camera)
        file_menu.addAction(self.place_action)

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
        self._refresh_cameras()
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
        self._touch()
        dialog = AddCameraDialog(self)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
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

    # ------------------------------------------------------------- modes

    def _mode_button_clicked(self) -> None:
        """Turn "a mode button was pressed" into "which mode".

        Resolved from the sender rather than captured in a lambda: a lambda
        holding the mode would also close over `self`, and a closure cell
        holding the window is the reference cycle that kept every console alive
        until interpreter shutdown and corrupted the heap on the way out. The
        freeing test caught this one the moment it was written.
        """
        button = self.sender()
        for mode, candidate in self.mode_buttons.items():
            if candidate is button:
                self._choose_mode(mode)
                return

    def _choose_mode(self, mode: str) -> None:
        """Put the map into a mode from the toolbar."""
        if mode == MODE_DRAW:
            if not self._configuring:
                self._set_status("Drawing needs Configure. Press Configure first.")
                self._mode_changed(self.map.mode)
                return
            self._draw_zone()
        elif mode == MODE_MEASURE:
            if not self.map.begin_measure():
                self._set_status(
                    "Place a camera first: the map cannot measure without an origin."
                )
        else:
            self.map.to_select()
        self._mode_changed(self.map.mode)

    def _mode_changed(self, mode: str) -> None:
        """Keep the buttons showing what the map is actually doing.

        Driven from the map rather than from the click, because a mode also
        ends on its own — a drawing closes, a pick is taken — and a button left
        checked after that is the ambiguity this was built to remove.
        """
        for name, button in self.mode_buttons.items():
            # Place is not Select. Lighting Select while the next click moves a
            # camera permanently is the exact lie these buttons exist to stop;
            # during a pick none of them is lit and the map band says what the
            # click will do.
            button.setChecked(name == mode)
        # Deliberately does not touch the status bar. It used to call
        # `_refresh_status`, which overwrote the very message that said why a
        # mode had been refused — the operator saw the refusal for no frames at
        # all. The mode is shown by the buttons and by the map's own band.

    # --------------------------------------------------- monitor / configure

    def _configure_only(self) -> list:
        """The controls that change the site rather than watch it."""
        return [
            self.open_button, self.place_button, self.map_place_button,
            self.remove_button, self.zone_button, self.draw_zone_button,
            self.reshape_zone_button, self.remove_zone_button,
            self.zone_properties.apply_button,
            # The menu duplicates of the first two buttons, with shortcuts.
            self.add_camera_action, self.place_action,
            # And the Zones tab's own copy of Add zone.
            self.add_zone_button,
        ]

    def _set_configuring(self, on: bool) -> None:
        """Unlock or relock the controls that change the site.

        Undo helps the operator who notices a mis-drag. A lock protects against
        the one who does not — and a console left in a control room is left in
        whatever state the last person walked away from, which is why this
        re-arms itself.
        """
        was, self._configuring = self._configuring, bool(on)
        for control in self._configure_only():
            control.setEnabled(self._configuring)
        # Dragging a mast on the plan view is a configuration change, so it is
        # behind the same lock and not a second, quieter one. Turning it off
        # mid-gesture reverts the gesture: the lock coming back is not the
        # operator saying yes.
        self.map.set_editable(self._configuring)
        if self.configure_button.isChecked() != self._configuring:
            self.configure_button.setChecked(self._configuring)

        if self._configuring:
            self._idle_timer.start(CONFIGURE_IDLE_MILLIS)
        else:
            self._idle_timer.stop()
            # Whatever was half-drawn goes with the mode: a zone corner placed
            # in Configure must not be completed after it has relocked.
            self.map.to_select()
            self._pick_action = None

        if was != self._configuring:
            self.store.audit(
                CONSOLE_ACTOR,
                "console.configure.entered" if self._configuring else "console.configure.left",
                self.node.node_id,
                "unlocked the controls that change the site"
                if self._configuring
                else "relocked",
            )
            self._set_status("Configure: the site can be changed." if self._configuring
                             else "Monitor: the site is locked.")

    def _relock(self) -> None:
        """The idle timeout fired."""
        if self._configuring:
            self._set_configuring(False)
            self._set_status(
                f"Monitor: relocked after {CONFIGURE_IDLE_MILLIS // 60000} idle minutes."
            )

    def _touch(self) -> None:
        """Any deliberate action restarts the idle countdown."""
        if self._configuring:
            self._idle_timer.start(CONFIGURE_IDLE_MILLIS)

    def _detector_labels(self) -> list[str]:
        """The labels the site's detector can actually produce, sorted.

        Read from the model once, here, rather than from a running camera:
        a zone is configured before anything runs, and an operator drawing a
        person-only zone on an idle console must see "person" in the list.
        Empty for motion detection, which labels nothing, and empty — rather
        than a guessed COCO list — when the model cannot be loaded, because a
        picker offering words the detector will never say is a filter that
        silently disarms every zone it is applied to.
        """
        if self._model is None:
            return []
        try:
            info = detector_for(self._model).info
        except DetectionError:
            return []
        return sorted(set(info.class_names.values())) if info.classifies else []

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
        self._touch()
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

    def _refresh_cameras(self) -> None:
        """Re-read what each camera is actually doing, and show it in both
        places that claim to know.

        The state is the node's — `camera_health()` decides LIVE/DARK/… from the
        last-frame clock — and it is read once here and given to both the list
        and the map. Asking twice, or letting either derive its own, is how the
        strip comes to call a camera dark while the map beside it paints the
        same camera's footprint as watched ground.
        """
        health = self.node.camera_health()
        self.camera_list.show_cameras(self.node.cameras, health)
        # A camera that is nominally up and delivering nothing keeps its pose,
        # and until this was connected it kept the full wedge that goes with
        # one: ground nobody is watching, drawn as covered.
        self.map.set_dark_cameras(
            [camera_id for camera_id, facts in health.items() if facts.is_dark]
        )

    def _refresh_placement(self) -> None:
        placed = {
            session.camera_id: session.pose
            for session in self._sessions.values()
            if session.pose is not None
        }
        # Zones first: `set_cameras` is what refits the view, and it can only
        # frame the zones the map already knows about. The other order left the
        # fit one zone behind, so a zone drawn behind the camera — exactly the
        # one whose warning says it can never fire — was framed out of the only
        # view that could show the operator why.
        self.map.set_zones(self._zones)
        self.map.set_cameras(placed)
        # What each camera can rule on, and what that means for every zone.
        # The bands are memoised per pose upstream, so this costs one Shapely
        # pass per zone and never runs from a paint.
        self.map.set_sigma_bands({camera_id: sigma_bands(pose) for camera_id, pose in placed.items()})
        self._zone_reports, self._zone_warnings = self._assess_zones(placed)
        self.zones_view.show_zones(self._zones, self._zone_reports, self._zone_warnings)
        # The search pickers follow the site, so a camera added or a zone drawn
        # a minute ago can be searched for without restarting the console.
        self.investigation.set_cameras(list(self._sessions))
        self.investigation.set_zones(self._zones)
        self._sync_zone_properties()
        # The "Placed" column is part of this same fact, so the list is rebuilt
        # on the same call rather than waiting for the next poll — which, with
        # nothing running, never comes.
        self._refresh_cameras()

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
        self._touch()
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
        self.map.forget_camera(session.camera_id)
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
        self._touch()
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

    def _camera_dragged(self, camera_id: str, point: LatLon) -> None:
        """A mast was dragged across the plan view and let go.

        The same call the placement dialog makes, on the same object, so there
        is one meaning of "this camera is here" however the operator said it.
        Built with `replace`, exactly as `_map_picked` does: a drag across the
        ground says nothing about the mast's height or which way it faces, and
        a placement that reset either would silently re-aim a camera the
        operator only meant to shift.

        The map is *asking*. It is not told the node agreed, so a refusal is
        answered by taking the drag back — otherwise the map goes on drawing
        the camera metres from where the node has it, indefinitely.
        """
        self._touch()
        session = self._sessions.get(camera_id)
        if session is None or session.pose is None:
            self.map.revert_uncommitted(camera_id)
            self._set_status(f"{camera_id} was not moved: it has no placement.")
            return
        self.node.place_camera(camera_id, replace(session.pose, position=point))
        self._refresh_placement()
        self._set_status(f"Moved {camera_id}.")

    def _camera_turned(self, camera_id: str, heading: float) -> None:
        """A heading grip was dragged and let go.

        Kept apart from `_camera_dragged` for the reason the map keeps the two
        signals apart: turning a camera and moving it are different mistakes,
        and an operator who reads "Moved gate" after aiming it goes looking for
        a move that never happened.
        """
        self._touch()
        session = self._sessions.get(camera_id)
        if session is None or session.pose is None:
            self.map.revert_uncommitted(camera_id)
            self._set_status(f"{camera_id} was not turned: it has no placement.")
            return
        self.node.place_camera(camera_id, replace(session.pose, heading=heading))
        self._refresh_placement()
        self._set_status(f"{camera_id} now faces {heading:.0f}°.")

    def _add_zone_dialog(self) -> None:
        """Ask what kind of zone, how big, and where, then create it."""
        self._touch()
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
        self._touch()
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
        self._touch()
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
        warned = getattr(self, "_zone_warnings", {}).get(zone_id, ())
        self._set_status(
            f"{zone.name} reshaped: {len(ring)} corners."
            + (f" ⚠ {warned[0]}" if warned else "")
        )

    def _zone_selection_changed(self) -> None:
        zone_id = self.zones_view.selected_zone_id()
        self.map.select_zone(zone_id)
        self._sync_zone_properties()
        if not self._syncing and zone_id is not None:
            self.selection.select(Selection.zone(zone_id))

    # ------------------------------------------------------------- selection

    def _selection_changed(self, selection) -> None:
        """One selected thing, shown by every panel that can show it.

        Guarded, because each panel answers by emitting its own "the user
        picked something" signal, and without the guard a click on the map
        would come back from the table as a second selection.
        """
        if self._syncing:
            return
        self._syncing = True
        try:
            self.map.set_selection(selection)
            self.camera_list.set_selection(selection)
            self._show_selected_camera(selection)
            for session in self._sessions.values():
                session.view.set_selection(selection)
            self.incidents.set_selection(selection)
            self.investigation.set_selection(selection)
            self._show_selected_track_row(selection)
            if selection is None:
                self.zones_view.clearSelection()
                self._sync_zone_properties()
            elif selection.kind == "zone":
                self.zones_view.select(selection.zone_id)
                self.detail_tabs.setCurrentIndex(1)
            elif selection is not None and selection.kind == "track":
                self.detail_tabs.setCurrentIndex(0)
        finally:
            self._syncing = False
        self._refresh_status()

    def _camera_row_selected(self, selection) -> None:
        """A row was clicked in the camera list. It is the same choice as the
        picker's, so it goes to the bus and comes back to the picker from there.

        Guarded like every other panel's answer: `_selection_changed` hands the
        list a selection, and a list that answered that with a selection of its
        own would push the bus round in a circle.
        """
        if self._syncing:
            return
        self.selection.select(selection)

    def _picker_changed(self, index: int) -> None:
        """The toolbar's combo box moved. Say so, so the list moves with it.

        The combo box is still where "which camera do Place…, Move on map and
        Remove act on" is *stored* — `_selected` reads it — so this does not
        replace it; it stops the two disagreeing. A console showing one camera
        highlighted in the list while the buttons acted on another is a worse
        interface than the combo box alone was.
        """
        if self._syncing:
            return
        camera_id = self.camera_picker.currentData()
        if camera_id is None:
            # The last camera has gone. Only a camera selection is cleared:
            # removing a camera must not take the operator's zone selection
            # with it.
            current = self.selection.current
            if current is not None and current.kind == CAMERA_KIND:
                self.selection.clear()
            return
        self.selection.select(Selection.camera(camera_id))

    def _show_selected_camera(self, selection) -> None:
        """Point the picker at the selected camera. Called under `_syncing`.

        Deliberately does nothing for a selection of any other kind. Clearing
        the picker when a zone is selected would leave Place… and Remove with
        no camera to act on, which is not what selecting a zone meant.
        """
        if selection is None or selection.kind != CAMERA_KIND:
            return
        index = self.camera_picker.findData(selection.camera_id)
        if index >= 0 and index != self.camera_picker.currentIndex():
            self.camera_picker.setCurrentIndex(index)

    def _show_selected_track_row(self, selection) -> None:
        if selection is None or selection.kind != "track":
            self.tracks.clearSelection()
            return
        for index in range(self.tracks.topLevelItemCount()):
            item = self.tracks.topLevelItem(index)
            if item.data(0, Qt.ItemDataRole.UserRole) == (
                selection.camera_id, selection.track_id
            ):
                self.tracks.setCurrentItem(item)
                item.setSelected(True)
                self.tracks.scrollToItem(item)
                return

    def _track_row_selected(self) -> None:
        if self._syncing:
            return
        item = self.tracks.currentItem()
        key = item.data(0, Qt.ItemDataRole.UserRole) if item is not None else None
        if key is not None:
            self.selection.select(Selection.track(key[0], key[1]))

    def _incident_row_selected(self) -> None:
        if self._syncing:
            return
        incident_id = self.incidents.selected_incident_id()
        if incident_id is not None:
            self.selection.select(Selection.incident(incident_id))

    # --------------------------------------------------------- ground readout

    def _ground_moved(self, point) -> None:
        """Where the pointer is on the ground, continuously.

        Distance and bearing are given from the selected camera when one is
        selected, otherwise from the first placed camera — an operator reading
        "37 m at 148°" needs to know what it is 37 m from, so the label always
        names it.
        """
        if point is None:
            self._ground_text = ""
            self.ground_label.setText("")
            return

        reference = None
        chosen = self.selection.current
        if chosen is not None and chosen.camera_id in self._sessions:
            reference = self._sessions[chosen.camera_id]
        if reference is None or reference.pose is None:
            reference = next(
                (s for s in self._sessions.values() if s.pose is not None), None
            )

        parts = [f"{point.lat:+.6f}, {point.lon:+.6f}"]
        if reference is not None and reference.pose is not None:
            metres = haversine_distance(reference.pose.position, point)
            bearing = bearing_degrees(reference.pose.position, point)
            parts.insert(0, f"{metres:.1f} m at {bearing:.0f}° from {reference.camera_id}")
        self._ground_text = "  ·  ".join(parts)
        self.ground_label.setText(self._ground_text)

    def _copy_ground(self) -> None:
        """Ctrl+C: put the readout on the clipboard, or say there is nothing."""
        from PySide6.QtGui import QGuiApplication

        if not self._ground_text:
            self._set_status("Nothing to copy: move the pointer over the plan view.")
            return
        QGuiApplication.clipboard().setText(self._ground_text)
        self._set_status(f"Copied: {self._ground_text}")

    def _track_menu(self, position) -> None:
        """Right-click a track row: copy where it is."""
        from PySide6.QtGui import QGuiApplication
        from PySide6.QtWidgets import QMenu

        item = self.tracks.itemAt(position)
        if item is None:
            return
        key = item.data(0, Qt.ItemDataRole.UserRole)
        where = item.text(8)
        # The coordinate never travels alone. A position pasted into a radio
        # call or a report without its uncertainty — or worse, without the fact
        # that it is a *fallback* and not a location at all — is exactly how
        # somebody gets sent to a place the system never claimed.
        copied = f"{where}  ±{item.text(9).lstrip('±')}  ({item.text(10)})"
        menu = QMenu(self)
        action = menu.addAction("Copy position")
        locatable = bool(where) and where != "not placed"
        action.setEnabled(locatable)
        if not locatable:
            action.setToolTip("This camera is not placed, so there is no position.")
        chosen = menu.exec(self.tracks.viewport().mapToGlobal(position))
        menu.deleteLater()
        if chosen is action and locatable:
            QGuiApplication.clipboard().setText(copied)
            self._set_status(f"Copied #{key[1]} on {key[0]}: {copied}")

    def _assess_zones(self, placed: dict) -> tuple[dict, dict]:
        """Coverage report and warnings for every zone, by id."""
        reports: dict = {}
        warnings: dict = {}
        if not placed:
            return reports, warnings
        zones = list(self._zones)
        for zone in zones:
            try:
                reports[zone.id] = zone_report(zone.ring, placed)
            except Exception:  # noqa: BLE001 - a report is advisory; a failure must not take the map down
                _log.exception("coverage report failed for %s", zone.id)
                continue
        for zone in zones:
            report = reports.get(zone.id)
            if report is None:
                continue
            warnings[zone.id] = zone_warnings(
                zone, report, [z for z in zones if z.id != zone.id],
                # The detector's vocabulary, so a person-only zone under a
                # detector that cannot say "person" is shown as unable to fire
                # instead of looking armed.
                labels=self._detector_labels(),
            )
        return reports, warnings

    def _sync_zone_properties(self) -> None:
        zone_id = self.zones_view.selected_zone_id()
        zone = next((z for z in self._zones if z.id == zone_id), None)
        self.zone_properties.show_zone(zone)
        reports = getattr(self, "_zone_reports", {})
        warnings = getattr(self, "_zone_warnings", {})
        self.zone_properties.show_report(
            reports.get(zone_id) if zone else None, warnings.get(zone_id, ()) if zone else ()
        )

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
        self._touch()
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

        # Refresh through the same path placement uses, so the new zone gets
        # its coverage report and warnings at once; then say the first warning
        # in the status bar rather than a modal — the zone is created either
        # way, and the operator can see the hatch on the map.
        self._refresh_placement()
        self.zones_view.select(zone.id)
        warned = getattr(self, "_zone_warnings", {}).get(zone.id, ())
        self._set_status(
            f"{len(self._zones)} zone(s). Rules apply to cameras started from now."
            + (f" ⚠ {zone.name}: {warned[0]}" if warned else "")
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

    def start_on_launch(self) -> None:
        """`--start`: run whatever the node restored, and say so if it is nothing.

        Silence here would be the worst outcome — a console opened with
        `--start` on a machine with no cameras would look exactly like one that
        was starting, for as long as the operator waited.
        """
        if not self._sessions:
            self._set_status("--start: no cameras to start. Add one to begin.")
            _log.warning("--start given but the node has no cameras")
            return
        _log.info("--start: starting %d restored camera(s)", len(self._sessions))
        self._start()

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
        # Only if the site is unlocked. This used to re-enable it flatly, so
        # stopping a camera silently undid the lock for Add camera.
        self.open_button.setEnabled(self._configuring)
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
        # On the poll timer, because "is this camera still delivering frames"
        # is only true of the instant it was asked. A strip refreshed on
        # placement alone would have gone on reading "live" for a camera whose
        # decoder wedged an hour ago.
        self._refresh_cameras()
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
            # What the pipeline has read off each vehicle so far, by track. The
            # pipeline published these and nothing displayed them; a plate the
            # operator cannot see is a plate reader that might as well be off.
            plates = {plate.track_id: plate for plate in getattr(update.result, "plates", ())}
            for track in sorted(update.result.tracks, key=lambda t: t.id):
                self.tracks.addTopLevelItem(
                    self._track_row(session.camera_id, track, update, info, plates.get(track.id))
                )

        # The table is rebuilt about thirty times a second. Without this the
        # selected row lost its highlight on the very next frame while the bus
        # still held the selection — the map stayed lit and the table did not.
        self._show_selected_track_row(self.selection.current)

    def _track_row(self, camera_id: str, track, update, info, plate=None) -> QTreeWidgetItem:
        # The row's key is (camera, id): a track id is only unique within one
        # camera, and a table holding two cameras' rows would otherwise select
        # the wrong object as soon as both ran.
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
            # `display`, never `text`: display carries "?" for every character
            # the frames have not agreed on, which is the honest form for a
            # screen. `text` is None until all of them have, and is for rules.
            _plate_cell(plate),
        ])
        item.setData(0, Qt.ItemDataRole.UserRole, (camera_id, track.id))
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

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Escape means "never mind", everywhere.

        The map handles it first when it is mid-gesture — abandoning a drawing
        matters more than clearing a selection — and this catches the rest.
        """
        if event.key() == Qt.Key.Key_Escape:
            if self.map.drawing or self.map.editing or self.map.picking:
                self.map.cancel_draw()
                self.map.cancel_edit()
                self.map.cancel_pick()
                self._pick_action = None
                self._set_status("Cancelled.")
            else:
                self.selection.clear()
            event.accept()
            return
        super().keyPressEvent(event)

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
        "--start", action="store_true",
        help=(
            "start every restored camera as soon as the window is up. For a "
            "control room, where a console left idle until somebody finds the "
            "Start button is a site unwatched for that long"
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
        if arguments.start:
            # After the event loop is running, not before: starting opens
            # cameras on worker threads whose first frames arrive through
            # signals, and a window that has not shown yet has nowhere to put
            # them. A bound method, never a lambda — see the freeing test.
            QTimer.singleShot(0, window.start_on_launch)
        code = app.exec()
    except Exception:
        # A packaged build has no terminal, so an unhandled exception would
        # otherwise close the window with no trace of why. The log survives it.
        log.critical("the console failed to start", exc_info=True)
        raise

    log.info("console exited with %d", code)
    return code
