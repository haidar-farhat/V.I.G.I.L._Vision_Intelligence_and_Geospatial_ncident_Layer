"""Zones: the rules an operator wrote about the ground, listed and edited.

A zone is a claim about a place — nobody should be here, this is the boundary,
ignore this pavement — and until this panel existed the console could make
exactly one kind of claim (restricted), in exactly one place (in front of the
selected camera), and could never take it back. An operator who put a zone in
the wrong place had to delete the database.

The panel lists every zone with its kind and size, and the dialog beside it
creates or changes one. Placement is either in front of the selected camera —
the near edge of what it can actually see, where positions are most accurate —
or a point the operator clicks on the plan view.
"""

from __future__ import annotations

from PySide6.QtCore import QTime, Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QScrollArea,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHeaderView,
    QLabel,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTimeEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from sentinel.core import LatLon, haversine_distance
from collections.abc import Iterable
from dataclasses import fields, is_dataclass, replace
from datetime import time as clock

from sentinel.zones import Schedule, Zone, ZoneKind

from . import theme

#: What each kind means, in the operator's terms. Shown beside the choice,
#: because "PERIMETER" and "ENTRY" are not self-explanatory and a zone of the
#: wrong kind is a rule that fires for the wrong reason.
KIND_DESCRIPTIONS: dict[ZoneKind, str] = {
    ZoneKind.RESTRICTED: "Nobody should be here. Presence alone is an event.",
    ZoneKind.PERIMETER: "The site boundary. Crossing it inbound matters.",
    ZoneKind.ENTRY: "A door, gate or lane where presence is expected.",
    ZoneKind.EXCLUSION: "Ignore this: a public pavement, a tree that moves.",
    ZoneKind.INTEREST: "Worth recording presence in, without implying anything is wrong.",
}

#: Placement choices offered by the dialog, in order.
IN_FRONT_OF_CAMERA = "In front of the selected camera"
PICK_ON_MAP = "Pick the centre on the map"

#: The column of :class:`ZonesView` that says which object classes a zone
#: watches. Last, so the columns tests and operators already count on keep
#: their positions.
WATCHES_COLUMN = 5

#: How the empty filter reads. A zone with no class filter watches every object
#: the detector can name, and the word for that is "any", not a blank cell —
#: a blank cell beside "restricted" reads as a zone watching nothing.
WATCHES_ANY = "any"

#: How many class names a list cell spells out before it says "+N".
WATCHES_SHOWN = 3

#: What the panel says instead of a class picker on a site whose detector
#: labels nothing — a motion detector, or a model shipped without class names
#: (``DetectorInfo.class_names`` is empty for both). Such a detector cannot
#: name what it saw, so every observation reaches the zone with no label, and
#: a filter would silence the zone entirely while the panel looked as though
#: it were narrowing it.
MOTION_ONLY_EXPLANATION = (
    "This site's detector labels nothing — it sees motion, or runs a model "
    "without class names — and cannot name what it saw, so a class filter "
    "would silence the zone entirely. Every zone here watches any object it "
    "detects."
)

#: What the panel says while nobody has told it what the detector can name.
#: This is a third state, not a kind of motion-only: the engine's own
#: ``zone_warnings`` refuses to judge a filter when ``labels`` is None, and the
#: panel must refuse too. Before this state existed, a zone narrowed to
#: "person" on a panel the orchestrator had not yet briefed was presented as
#: "not named by this detector … watches nothing" — false the moment a
#: segmenter was running — and an operator who believed it and pressed
#: "Watch any object" reinstated exactly the couch incident.
VOCABULARY_UNKNOWN_CAPTION = (
    "The detector's labels are not known yet; the filter is shown as stored "
    "and not judged."
)

#: How many rows the picker shows before it scrolls inside itself. The panel
#: as a whole already scrolls, so this is not a limit on what can be ticked —
#: it stops an 80-class vocabulary from pushing the schedule and the dwell
#: times off the bottom of every window. The row height itself is measured
#: from the picker's own font at fill time, never assumed.
PICKER_ROWS = 8

#: The summary shown above the picker when nothing is ticked. Spelled out,
#: because an empty list of tick boxes looks like a zone watching nothing when
#: it is the zone watching everything.
ANY_OBJECT_SUMMARY = "Any object the detector names"


def zone_has_classes(zone) -> bool:
    """Whether this zone object carries a ``classes`` field at all.

    The class filter is a field of :class:`sentinel.zones.Zone` that landed
    separately from this panel. Until the engine has it — and for any older
    zone object handed in — the picker is hidden and ``zone_from_fields``
    does not pass ``classes``, because :func:`dataclasses.replace` with an
    unknown field raises, and a properties panel that raises on *Apply* loses
    every other edit the operator made with it.
    """
    if is_dataclass(zone) and not isinstance(zone, type):
        return any(f.name == "classes" for f in fields(zone))
    return hasattr(zone, "classes")


def zone_classes(zone) -> frozenset[str]:
    """The zone's class filter, read as the empty set when it is missing or None.

    Empty means *any*, and a zone that predates the field, or was stored with
    None by an older writer, watched everything: reading it as anything else
    would make that zone go quiet on upgrade — the silent failure that looks
    exactly like protection. So the absent field and the empty field are the
    same answer, and only a real filter is ever narrower than "any".
    """
    return frozenset(getattr(zone, "classes", ()) or ())


def watches_label(classes: Iterable[str], shown: int = WATCHES_SHOWN) -> str:
    """A list cell for the class filter: "any", or the first few names and a
    count for the rest. Sorted, so the same filter reads the same on every
    row instead of in whatever order a set happened to iterate."""
    names = sorted(classes)
    if not names:
        return WATCHES_ANY
    if len(names) <= shown:
        return ", ".join(names)
    return f"{', '.join(names[:shown])} +{len(names) - shown}"


def _floor_percent(fraction: float) -> str:
    """A percentage that never rounds *up* to a reassuring number.

    `f"{0.996:.0%}"` is "100%", and a zone with four square metres of blind
    ground reading "Covered 100%" is the console telling an operator there is
    nothing to check. Only a genuinely complete fraction may say 100%.
    """
    percent = fraction * 100.0
    if percent >= 99.95 and fraction < 1.0:
        return "99%"
    return f"{int(percent) if percent < 100 else 100}%"


def zone_extent_meters(zone: Zone) -> float:
    """How far across a zone is, roughly: twice the furthest vertex from the
    centroid. Enough to tell a 6 m doorway from a 60 m yard in a list."""
    lat = sum(p.lat for p in zone.ring) / len(zone.ring)
    lon = sum(p.lon for p in zone.ring) / len(zone.ring)
    centre = LatLon(lat, lon)
    return 2.0 * max(haversine_distance(centre, p) for p in zone.ring)


class ZonesView(QTreeWidget):
    """Every zone on the node, one row each."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setColumnCount(6)
        self.setHeaderLabels(["Zone", "Kind", "Across", "Covered", "Points", "Watches"])
        self.setRootIsDecorated(False)
        self.setAlternatingRowColors(True)
        header = self.header()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        self.setMinimumWidth(330)
        for index, width in enumerate((150, 90, 62, 74)):
            self.setColumnWidth(index, width)
        # Points is a two-digit column and gets exactly what its own header
        # needs in the running font — measured, not a literal — so that the
        # stretched Watches column beside it gets the rest. A guessed width
        # elides "Points" on a large font and wastes the list on a small one.
        self.setColumnWidth(4, header.sectionSizeHint(4))

    def show_zones(self, zones, reports: dict | None = None, warnings: dict | None = None) -> None:
        """Replace the list, keeping the selection by id.

        ``reports`` and ``warnings`` (by zone id) fill the Covered column: how
        much of the zone any camera can see, and a warning glyph with every
        warning as the tooltip. A zone nothing can see reads "⚠ 0%" — the one
        number an operator needs before trusting a zone with an alarm.

        The Watches column is the class filter: "any", or the classes it is
        narrowed to. It is what tells an operator, before the next couch is
        reported, that a restricted zone fires on everything the segmenter
        can name.
        """
        reports = reports or {}
        warnings = warnings or {}
        selected = self.selected_zone_id()
        self.clear()
        for zone in zones:
            report = reports.get(zone.id)
            warned = tuple(warnings.get(zone.id, ()))
            covered = "—" if report is None else _floor_percent(report.covered_fraction)
            if warned:
                covered = f"⚠ {covered}"
            item = QTreeWidgetItem([
                zone.name,
                zone.kind.value.lower(),
                f"{zone_extent_meters(zone):.0f} m",
                covered,
                str(len(zone.ring)),
                watches_label(zone_classes(zone)),
            ])
            item.setData(0, Qt.ItemDataRole.UserRole, zone.id)
            watched = zone_classes(zone)
            item.setToolTip(
                WATCHES_COLUMN,
                "Any object the detector names."
                if not watched
                else "Only: " + ", ".join(sorted(watched)),
            )
            item.setForeground(1, QBrush(theme.zone_colour(zone.kind)))
            item.setToolTip(1, KIND_DESCRIPTIONS.get(zone.kind, ""))
            if warned:
                item.setForeground(3, QBrush(theme.WARNING))
                item.setToolTip(3, "\n".join(warned))
            self.addTopLevelItem(item)
            if zone.id == selected:
                item.setSelected(True)
                self.setCurrentItem(item)

    def selected_zone_id(self) -> str | None:
        item = self.currentItem()
        if item is None or not item.isSelected():
            selected = self.selectedItems()
            item = selected[0] if selected else None
        return None if item is None else item.data(0, Qt.ItemDataRole.UserRole)

    def select(self, zone_id: str) -> None:
        for index in range(self.topLevelItemCount()):
            item = self.topLevelItem(index)
            if item.data(0, Qt.ItemDataRole.UserRole) == zone_id:
                self.setCurrentItem(item)
                item.setSelected(True)
                return


class ZoneDialog(QDialog):
    """Create a zone, or change the name and kind of one that exists.

    With ``ring_given`` the outline was just drawn on the map, so size and
    placement do not apply and are not shown. Reshaping an existing zone is done
    on the map, and every change to an outline is audited with the vertex count
    before and after — a zone quietly shrinking to exclude the door it was drawn
    around is exactly what the audit log exists to record.
    """

    def __init__(
        self,
        *,
        default_radius: float = 10.0,
        existing: Zone | None = None,
        can_pick: bool = False,
        ring_given: bool = False,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._existing = existing
        self._ring_given = ring_given
        self.setWindowTitle("Change zone" if existing else "Add zone")
        self.setMinimumWidth(460)

        form = QFormLayout()
        self._name = QLineEdit(existing.name if existing else "")
        self._name.setPlaceholderText("Loading bay, north gate, public pavement…")
        form.addRow("Name", self._name)

        self._kind = QComboBox()
        for kind in ZoneKind:
            # The value, not the enum: Qt hands a str-valued enum back as a
            # plain str, and `ZoneKind(value)` is the honest way round.
            self._kind.addItem(kind.value.title(), kind.value)
            index = self._kind.count() - 1
            self._kind.setItemData(index, KIND_DESCRIPTIONS[kind], Qt.ItemDataRole.ToolTipRole)
            self._kind.setItemData(
                index, QBrush(theme.zone_colour(kind)), Qt.ItemDataRole.ForegroundRole
            )
        if existing:
            self._kind.setCurrentIndex(list(ZoneKind).index(existing.kind))
        self._kind.currentIndexChanged.connect(self._describe)
        form.addRow("Kind", self._kind)

        self._description = QLabel("")
        self._description.setObjectName("Caption")
        self._description.setWordWrap(True)
        form.addRow("", self._description)

        self._radius = QDoubleSpinBox()
        self._radius.setRange(1.0, 500.0)
        self._radius.setSuffix(" m")
        self._radius.setValue(zone_extent_meters(existing) / 2 if existing else default_radius)
        self._radius.setToolTip("Half the width of the square drawn on the ground.")
        self._radius.setEnabled(existing is None and not ring_given)
        if not ring_given:
            form.addRow("Half-width", self._radius)

        self._placement = QComboBox()
        self._placement.addItem(IN_FRONT_OF_CAMERA)
        if can_pick:
            self._placement.addItem(PICK_ON_MAP)
        else:
            self._placement.setToolTip(
                "Picking on the map needs a placed camera: until one is placed "
                "the map has no origin to measure a click against."
            )
        self._placement.setEnabled(existing is None and not ring_given)
        if not ring_given:
            form.addRow("Where", self._placement)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)
        self._describe()

    def _describe(self) -> None:
        kind = self.kind()
        self._description.setText(KIND_DESCRIPTIONS.get(kind, ""))

    # ------------------------------------------------------------------ result

    def name(self) -> str:
        return self._name.text().strip()

    def kind(self) -> ZoneKind:
        return ZoneKind(self._kind.currentData())

    def radius(self) -> float:
        return float(self._radius.value())

    def pick_on_map(self) -> bool:
        return not self._ring_given and self._placement.currentText() == PICK_ON_MAP


WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


class ZonePropertiesPanel(QWidget):
    """Everything about the selected zone that is not its outline.

    Name, kind, which object classes it watches, when its rules apply, how long
    an object must be inside before it counts, how long it must be gone before
    the presence ends, and whether an uncertain position may count. Nothing is
    written until *Apply*; *Revert* puts the fields back to the zone as stored.

    The class picker exists because a restricted zone with no filter fires on
    every class the segmenter names, and "1 couch in Room (HIGH)" is the
    incident that stops an operator believing incidents. Its vocabulary is
    whatever the site's detector can actually say (:meth:`set_classes`), and
    it has three states, the same three the engine's ``zone_warnings`` has:
    unknown (nobody has said — the filter is shown as stored and not judged),
    empty (the detector labels nothing — any filter silences the zone), and a
    real list (a stored class outside it is flagged rather than silently
    kept, because such a zone watches nothing). Collapsing the first into the
    second is what made the panel call a working "person" filter dead.
    """

    #: The zone as it should now be. The owner persists it through the node.
    changed = Signal(object)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._zone: Zone | None = None

        form = QFormLayout()
        form.setContentsMargins(6, 4, 6, 4)

        self._name = QLineEdit()
        form.addRow("Name", self._name)

        self._kind = QComboBox()
        for kind in ZoneKind:
            self._kind.addItem(kind.value.title(), kind.value)
            index = self._kind.count() - 1
            self._kind.setItemData(index, KIND_DESCRIPTIONS[kind], Qt.ItemDataRole.ToolTipRole)
        form.addRow("Kind", self._kind)

        self._classes_supported = False
        #: None until :meth:`set_classes` is called: "not known", which is
        #: not the same as "labels nothing" and must never be shown as it.
        self._vocabulary: tuple[str, ...] | None = None
        self._classes_box = QGroupBox("Watches")
        classes_form = QVBoxLayout(self._classes_box)
        classes_form.setContentsMargins(8, 4, 8, 6)
        self.class_caption = QLabel("")
        self.class_caption.setObjectName("Caption")
        self.class_caption.setWordWrap(True)
        classes_form.addWidget(self.class_caption)
        self.class_summary = QLabel(ANY_OBJECT_SUMMARY)
        self.class_summary.setWordWrap(True)
        classes_form.addWidget(self.class_summary)
        self.class_picker = QListWidget()
        self.class_picker.setToolTip(
            "Tick the classes this zone should react to. Nothing ticked means any "
            "object the detector names."
        )
        # Ticks refresh the summary only. They must not emit ``changed``: the
        # owner persists every emission through the node, and a persisted
        # write per tick would audit six edits for one decision.
        self.class_picker.itemChanged.connect(self._classes_edited)
        classes_form.addWidget(self.class_picker)
        self.class_note = QLabel("")
        self.class_note.setWordWrap(True)
        self.class_note.setStyleSheet(f"color: {theme.WARNING.name()};")
        self.class_note.setVisible(False)
        classes_form.addWidget(self.class_note)
        self.any_object_button = QPushButton("Watch any object")
        self.any_object_button.setToolTip(
            "Clear the filter: react to any object the detector names."
        )
        self.any_object_button.clicked.connect(self.watch_any_object)
        classes_form.addWidget(self.any_object_button, 0, Qt.AlignmentFlag.AlignLeft)
        form.addRow(self._classes_box)

        self._scheduled = QCheckBox("Only between")
        self._scheduled.setToolTip(
            "Off: the zone's rules apply at all times. On: only inside this window, "
            "on these days. A window that ends before it starts wraps midnight, "
            "which is what every real after-hours schedule does."
        )
        self._start = QTimeEdit(QTime(18, 0))
        self._end = QTimeEdit(QTime(6, 0))
        for edit in (self._start, self._end):
            edit.setDisplayFormat("HH:mm")
        when = QHBoxLayout()
        when.addWidget(self._scheduled)
        when.addWidget(self._start)
        when.addWidget(QLabel("and"))
        when.addWidget(self._end)
        when.addStretch(1)
        form.addRow("Schedule", when)

        days = QHBoxLayout()
        self._days: list[QCheckBox] = []
        for name in WEEKDAYS:
            box = QCheckBox(name)
            box.setChecked(True)
            self._days.append(box)
            days.addWidget(box)
        days.addStretch(1)
        form.addRow("", days)
        self._scheduled.toggled.connect(self._enable_schedule)

        # Which clock the window is read against. Until a site record declares
        # a zone this is the machine's, and saying so is the difference between
        # an operator trusting 18:00 and an operator in Beirut being armed at
        # 21:00 without knowing.
        self._clock = QLabel("")
        self._clock.setObjectName("Caption")
        form.addRow("", self._clock)

        self._dwell = QDoubleSpinBox()
        self._dwell.setRange(0.0, 600.0)
        self._dwell.setDecimals(1)
        self._dwell.setSuffix(" s")
        self._dwell.setToolTip(
            "How long an object must be inside before it counts as present. Short "
            "for a doorway, long for a yard where people pass through."
        )
        form.addRow("Count after", self._dwell)

        self._exit = QDoubleSpinBox()
        self._exit.setRange(0.0, 600.0)
        self._exit.setDecimals(1)
        self._exit.setSuffix(" s")
        self._exit.setToolTip(
            "How long an object must be gone before its presence ends. Longer than "
            "the entry delay, or a one-frame dropout ends a presence and starts a "
            "second one."
        )
        form.addRow("Release after", self._exit)

        self._report_box = QGroupBox("What the cameras can rule on")
        report_form = QFormLayout(self._report_box)
        report_form.setContentsMargins(8, 4, 8, 6)
        self._report_caption = QLabel("A geometric upper bound — nothing here models occlusion.")
        self._report_caption.setObjectName("Caption")
        self._report_caption.setWordWrap(True)
        report_form.addRow(self._report_caption)
        self._covered = QLabel("—")
        self._covered.setToolTip(
            "How much of the zone any camera can reach, and of that, how much it "
            "can locate well enough to adjudicate — where the position error is "
            "smaller than half the zone's narrowest width. The rest reports "
            "UNCERTAIN, which a restricted area does not act on."
        )
        report_form.addRow("Covered", self._covered)
        self._seen_by = QLabel("—")
        report_form.addRow("Seen by", self._seen_by)
        self._sigma = QLabel("—")
        report_form.addRow("Position error", self._sigma)
        self._warnings = QLabel("")
        self._warnings.setWordWrap(True)
        self._warnings.setStyleSheet(f"color: {theme.WARNING.name()};")
        report_form.addRow(self._warnings)

        # Directly under the name and kind, above everything an operator tunes:
        # this is the part they read rather than edit, and it decides whether
        # the rest is worth setting at all. Below the schedule it fell off the
        # bottom of the panel on any window shorter than about 900 px.
        form.insertRow(2, self._report_box)

        self._uncertain = QCheckBox("An uncertain position may count as inside")
        self._uncertain.setToolTip(
            "A position whose uncertainty disc straddles the boundary. Right for an "
            "exclusion or interest zone; wrong for a restricted area, which must not "
            "raise an alarm on a maybe."
        )
        form.addRow("", self._uncertain)

        buttons = QHBoxLayout()
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply)
        self.revert_button = QPushButton("Revert")
        self.revert_button.clicked.connect(self.revert)
        buttons.addStretch(1)
        buttons.addWidget(self.revert_button)
        buttons.addWidget(self.apply_button)

        self._empty = QLabel("Select a zone to see its properties.")
        self._empty.setObjectName("Caption")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._empty)
        self._form_host = QWidget()
        host = QVBoxLayout(self._form_host)
        host.setContentsMargins(0, 0, 0, 0)
        host.addLayout(form)
        host.addLayout(buttons)
        host.addStretch(1)

        # Scrolled, because it will not always fit. A QFormLayout given less
        # height than it needs does not clip — it compresses every row into
        # every other, and the first photograph of this panel was six rows of
        # overlapping text. Scrolling degrades honestly.
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setWidget(self._form_host)
        self._scroll.setMinimumWidth(380)
        layout.addWidget(self._scroll)
        self.show_zone(None)

    # ------------------------------------------------------------------ state

    @property
    def zone(self) -> Zone | None:
        return self._zone

    def show_zone(self, zone: Zone | None) -> None:
        self._zone = zone
        self._scroll.setVisible(zone is not None)
        self._empty.setVisible(zone is None)
        if zone is None:
            return
        self._name.setText(zone.name)
        self._kind.setCurrentIndex(list(ZoneKind).index(zone.kind))
        self._classes_supported = zone_has_classes(zone)
        self._classes_box.setVisible(self._classes_supported)
        if self._classes_supported:
            self._fill_classes(zone_classes(zone))
        schedule = zone.schedule
        self._scheduled.setChecked(schedule is not None)
        if schedule is not None:
            self._start.setTime(QTime(schedule.start.hour, schedule.start.minute))
            self._end.setTime(QTime(schedule.end.hour, schedule.end.minute))
            for index, box in enumerate(self._days, start=1):
                box.setChecked(not schedule.days or index in schedule.days)
        else:
            for box in self._days:
                box.setChecked(True)
        self._enable_schedule(schedule is not None)
        self._dwell.setValue(zone.enter_after_millis / 1000.0)
        self._exit.setValue(zone.exit_after_millis / 1000.0)
        self._uncertain.setChecked(zone.accept_uncertain)

    def _enable_schedule(self, on: bool) -> None:
        for widget in (self._start, self._end, *self._days):
            widget.setEnabled(on)

    def set_clock(self, label: str) -> None:
        self._clock.setText(f"Times are read against {label}." if label else "")

    # ---------------------------------------------------------------- classes

    def set_classes(self, labels: Iterable[str] | None) -> None:
        """The vocabulary the site's detector actually has, or None for "not known".

        The picker offers exactly these, in this order, because a filter the
        detector can never satisfy is a zone that watches nothing. Ticks
        already made survive the vocabulary changing — the orchestrator may
        learn the detector's labels after the zone was shown — and a tick the
        new vocabulary cannot name stays visible, flagged, rather than being
        dropped on the floor and applied as a narrower filter than shown.

        None is accepted, and means what it means to ``zone_warnings``: the
        caller does not know what detector will run (``runner.detector_info``
        is None before a session starts), so stored filters are shown and not
        judged. An empty sequence is the other thing — a detector that labels
        nothing — and only that reads as motion-only.
        """
        if labels is None:
            self._vocabulary = None
        else:
            self._vocabulary = tuple(dict.fromkeys(str(label) for label in labels))
        self._fill_classes(self.selected_classes())

    def selected_classes(self) -> frozenset[str]:
        """The ticked classes, by the label each item carries — never its text.

        A flagged item's text is "forklift  (not named by this detector)";
        reading text would apply a filter for a class that does not exist and
        make the flagged zone quieter still. Empty means any object.
        """
        ticked = set()
        for index in range(self.class_picker.count()):
            item = self.class_picker.item(index)
            if item.checkState() == Qt.CheckState.Checked:
                ticked.add(item.data(Qt.ItemDataRole.UserRole))
        return frozenset(ticked)

    def watch_any_object(self) -> None:
        """Untick everything in one move, so a silenced zone has a one-click cure.

        A zone whose only ticks the detector cannot name watches nothing; the
        fix is the empty filter, and asking the operator to find and untick
        each flagged row is how such a zone stays silent for a week. Signals
        are blocked so the summary is redrawn once, not per row.
        """
        self.class_picker.blockSignals(True)
        try:
            for index in range(self.class_picker.count()):
                self.class_picker.item(index).setCheckState(Qt.CheckState.Unchecked)
        finally:
            self.class_picker.blockSignals(False)
        self._describe_classes()

    def _fill_classes(self, ticked: frozenset[str]) -> None:
        """Rebuild the picker from the vocabulary plus whatever is ticked.

        Flagged rows come first. They are the one thing an operator must see
        — a tick the detector cannot honour — and appended after an 80-class
        vocabulary they sat below the fold of a scrolled list, which is the
        same as not showing them. With the vocabulary unknown nothing is
        flagged: the stored ticks are offered as they are, because the panel
        has no grounds to call any of them dead.

        Signals are blocked while filling: ``itemChanged`` fires for every
        programmatic tick, and the summary is refreshed once at the end.
        """
        known = None if self._vocabulary is None else set(self._vocabulary)
        if known is None:
            flagged, offered = (), sorted(ticked)
        else:
            flagged, offered = tuple(sorted(ticked - known)), self._vocabulary
        self.class_picker.blockSignals(True)
        try:
            self.class_picker.clear()
            for label in (*flagged, *offered):
                item = QListWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, label)
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    Qt.CheckState.Checked if label in ticked else Qt.CheckState.Unchecked
                )
                if label in flagged:
                    item.setText(f"{label}  (not named by this detector)")
                    item.setForeground(QBrush(theme.WARNING))
                    item.setToolTip(
                        "This site's detector never produces this label, so the zone "
                        "cannot fire on it. Untick it, or change the detector."
                    )
                self.class_picker.addItem(item)
        finally:
            self.class_picker.blockSignals(False)
        self._size_picker()
        self._describe_classes()

    def _size_picker(self) -> None:
        """Give the picker the height of its rows, up to :data:`PICKER_ROWS`.

        The row height is asked of the picker itself, in its running font,
        because a literal pixel height is a guess about someone else's
        display: the first version shipped 110 px, which on a 5-class
        vocabulary plus one flagged row put the flagged row — the one the
        operator needs — behind a scrollbar. An empty picker is hidden rather
        than shown as an empty box; the caption above it says why it is empty.
        """
        count = self.class_picker.count()
        self.class_picker.setVisible(count > 0)
        if count == 0:
            return
        rows = min(count, PICKER_ROWS)
        height = rows * self.class_picker.sizeHintForRow(0) + 2 * self.class_picker.frameWidth()
        self.class_picker.setMinimumHeight(height)
        self.class_picker.setMaximumHeight(height)

    def _classes_edited(self, _item=None) -> None:
        # A Qt slot: nothing here may raise, and a summary label is not worth
        # taking the panel down for.
        try:
            self._describe_classes()
        except Exception:  # noqa: BLE001
            pass

    def _describe_classes(self) -> None:
        """Compose every claim the group makes to the operator, from the ticks.

        Three vocabularies, three captions: unknown is "shown as stored, not
        judged", empty is the motion-only explanation, a real list is the
        plain rule. The warning note is written only against a real list —
        the panel has no grounds to call a tick dead when nobody has said what
        the detector names, and doing so was how a live "person" filter was
        presented as watching nothing.

        Enabled follows "is there anything to untick", not "is there a
        vocabulary": a filter stored by another site's detector must stay
        reachable on a site that labels nothing, or the operator can see the
        zone is silenced and cannot un-silence it.
        """
        ticked = self.selected_classes()
        unknown = self._vocabulary is None
        motion_only = self._vocabulary == ()
        unnamed = [] if unknown else sorted(ticked - set(self._vocabulary))
        self.class_picker.setEnabled(bool(self._vocabulary) or bool(ticked))
        self.any_object_button.setEnabled(bool(ticked))
        if unknown:
            caption = VOCABULARY_UNKNOWN_CAPTION
        elif motion_only:
            caption = MOTION_ONLY_EXPLANATION
        else:
            caption = "Nothing ticked means any object the detector names."
        self.class_caption.setText(caption)
        if not ticked:
            self.class_summary.setText(ANY_OBJECT_SUMMARY)
        else:
            self.class_summary.setText("Only: " + ", ".join(sorted(ticked)))
        if unnamed:
            self.class_note.setText(
                "⚠ Not named by this detector: " + ", ".join(unnamed) + ". "
                + (
                    "The zone watches nothing here until the filter is cleared."
                    if len(unnamed) == len(ticked)
                    else "The zone cannot fire on these."
                )
            )
        self.class_note.setVisible(bool(unnamed))

    def show_report(self, report, warnings=()) -> None:
        """Fill the 'What the cameras can rule on' group for the shown zone."""
        warnings = tuple(warnings)
        if report is None:
            for label in (self._covered, self._seen_by, self._sigma):
                label.setText("—")
        else:
            self._covered.setText(
                f"{_floor_percent(report.covered_fraction)} · "
                f"{_floor_percent(report.confident_fraction)} confidently"
            )
            seen = ", ".join(report.cameras) if report.cameras else "no camera"
            self._seen_by.setText(f"{seen} · {report.area_m2:.0f} m²")
            if not report.cameras:
                # Not a large error — no error at all, because nothing is
                # looking. "Beyond 5 m" invited the operator to move the zone
                # closer, when the answer is to point a camera at it.
                self._sigma.setText("—  no camera sees this")
            elif report.best_sigma_m is None and report.worst_sigma_m is None:
                self._sigma.setText("beyond 5 m everywhere")
            else:
                best = "—" if report.best_sigma_m is None else f"≤ {report.best_sigma_m:g} m"
                worst = "beyond 5 m" if report.worst_sigma_m is None else f"≤ {report.worst_sigma_m:g} m"
                self._sigma.setText(f"{best} best · {worst} worst")
        self._warnings.setText("\n".join(f"⚠ {w}" for w in warnings))
        self._warnings.setVisible(bool(warnings))

    def zone_from_fields(self) -> Zone:
        """The selected zone with the fields as they are on screen."""
        assert self._zone is not None
        schedule = None
        if self._scheduled.isChecked():
            days = frozenset(
                index for index, box in enumerate(self._days, start=1) if box.isChecked()
            )
            # Every day checked means no day restriction, which is how the
            # engine spells "every day".
            if len(days) == len(WEEKDAYS):
                days = frozenset()
            start, end = self._start.time(), self._end.time()
            schedule = Schedule(
                start=clock(start.hour(), start.minute()),
                end=clock(end.hour(), end.minute()),
                days=days,
            )
        changes = dict(
            name=self._name.text().strip() or self._zone.name,
            kind=ZoneKind(self._kind.currentData()),
            schedule=schedule,
            enter_after_millis=int(round(self._dwell.value() * 1000)),
            exit_after_millis=int(round(self._exit.value() * 1000)),
            accept_uncertain=self._uncertain.isChecked(),
        )
        # Only for a zone that has the field: see `zone_has_classes`.
        if self._classes_supported:
            changes["classes"] = frozenset(self.selected_classes())
        return replace(self._zone, **changes)

    def apply(self) -> None:
        if self._zone is None:
            return
        self.changed.emit(self.zone_from_fields())

    def revert(self) -> None:
        self.show_zone(self._zone)
