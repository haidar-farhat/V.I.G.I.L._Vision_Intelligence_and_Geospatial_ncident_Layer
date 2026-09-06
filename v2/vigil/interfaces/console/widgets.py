"""The panels: cameras, incidents, tracks, the audit trail, and a label that elides.

Each is a plain view: it is given data and shows it. None of them reaches the
service; the window does that through `Commands`.
"""

from __future__ import annotations

import time
from typing import Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QHeaderView, QLabel, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from . import theme

CAMERA_COLUMN, SOURCE_COLUMN, PLACED_COLUMN, STATUS_COLUMN, RECORD_COLUMN = range(5)


class ElidingLabel(QLabel):
    """A permanent status label that shortens itself rather than push the message off.

    `text()` still returns the whole text — code and tests read it — and the
    whole text is in the tooltip. v1's status bar clipped its own message to
    `0 tracked no` because five permanent labels took a 2,000-px window.
    """

    def __init__(self, text: str = "", parent: QWidget | None = None):
        super().__init__("", parent)
        self._full = ""
        self._cap: int | None = None
        self.setObjectName("Caption")
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt's name
        self._full = text or ""
        self._repaint()

    def text(self) -> str:
        return self._full

    def set_cap(self, pixels: int | None) -> None:
        self._cap = None if pixels is None else max(24, int(pixels))
        self._repaint()

    @property
    def elided(self) -> bool:
        return super().text() != self._full

    def _repaint(self) -> None:
        shown = self._full
        if self._cap is not None and self._full:
            shown = self.fontMetrics().elidedText(self._full, Qt.TextElideMode.ElideRight, self._cap)
        super().setText(shown)
        if shown != self._full:
            self.setToolTip(self._full)


class CameraList(QWidget):
    """Every camera, what it is doing, and whether it records.

    The Record box is editable only while the site is unlocked *and* the
    person may configure; the flag is the operator's, not a live state, so it
    stays readable when the box is greyed.
    """

    selected = Signal(object)
    record_toggled = Signal(str, bool)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._quiet = False
        self._editable = False
        self._selected: str | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(5)
        self.tree.setHeaderLabels(["Camera", "Source", "Placed", "Status", "Rec"])
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.setTextElideMode(Qt.TextElideMode.ElideRight)
        header = self.tree.header()
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(44)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        # Source and Status share what is left; Source is the one an operator
        # can afford to lose the end of. v1 showed "Stat" and hid the Record
        # box behind a scrollbar at the default split.
        header.setSectionResizeMode(SOURCE_COLUMN, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(STATUS_COLUMN, QHeaderView.ResizeMode.Stretch)
        self.tree.setColumnWidth(CAMERA_COLUMN, 96)
        self.tree.setColumnWidth(PLACED_COLUMN, 62)
        self.tree.setColumnWidth(RECORD_COLUMN, 44)
        self.tree.setMinimumWidth(340)
        self.tree.itemSelectionChanged.connect(self._selection_changed)
        self.tree.itemChanged.connect(self._item_changed)
        layout.addWidget(self.tree)

    def set_recording_editable(self, editable: bool) -> None:
        self._editable = editable
        self._quiet = True
        try:
            for index in range(self.tree.topLevelItemCount()):
                self._apply_flags(self.tree.topLevelItem(index))
        finally:
            self._quiet = False

    def _apply_flags(self, item: QTreeWidgetItem) -> None:
        flags = item.flags()
        if self._editable:
            item.setFlags(flags | Qt.ItemFlag.ItemIsUserCheckable)
        else:
            item.setFlags(flags & ~Qt.ItemFlag.ItemIsUserCheckable)

    def show_cameras(self, cameras: Sequence, health: dict) -> None:
        self._quiet = True
        try:
            self.tree.clear()
            for camera in cameras:
                state = health.get(camera.id)
                item = QTreeWidgetItem([
                    camera.id, camera.source, "yes" if camera.placed else "—",
                    state.describe() if state is not None else "STOPPED", "",
                ])
                item.setData(0, Qt.ItemDataRole.UserRole, camera.id)
                item.setToolTip(SOURCE_COLUMN, camera.source)
                item.setCheckState(RECORD_COLUMN, Qt.CheckState.Checked if camera.record else Qt.CheckState.Unchecked)
                if state is not None:
                    item.setForeground(STATUS_COLUMN, _state_colour(state.state))
                    # The column stretches but can still be narrower than the
                    # sentence; an elided status must stay readable on hover.
                    item.setToolTip(STATUS_COLUMN, state.describe())
                if not camera.placed:
                    item.setToolTip(PLACED_COLUMN, "Unplaced: this camera cannot locate anything on the ground.")
                self._apply_flags(item)
                self.tree.addTopLevelItem(item)
                if camera.id == self._selected:
                    item.setSelected(True)
        finally:
            self._quiet = False

    def selected_camera(self) -> str | None:
        items = self.tree.selectedItems()
        return items[0].data(0, Qt.ItemDataRole.UserRole) if items else None

    def _selection_changed(self) -> None:
        if self._quiet:
            return
        self._selected = self.selected_camera()
        self.selected.emit(self._selected)

    def _item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._quiet or column != RECORD_COLUMN or not self._editable:
            return
        self.record_toggled.emit(item.data(0, Qt.ItemDataRole.UserRole),
                                 item.checkState(RECORD_COLUMN) == Qt.CheckState.Checked)


def _review_text(review) -> str:
    if str(review.state) == "NEW":
        return "new"
    who = (review.by or "").split(":")[-1]
    return f"{str(review.state).lower()} · {who}"


def _review_colour(state: str):
    return {"NEW": theme.STALE, "ACKNOWLEDGED": theme.LIVE, "DISMISSED": theme.TEXT_FAINT}.get(state, theme.TEXT_MUTED)


def _state_colour(state: str):
    return {"LIVE": theme.LIVE, "STARTING": theme.STALE, "DARK": theme.STALE, "FAULTED": theme.FAULT}.get(state, theme.TEXT_MUTED)


class IncidentList(QWidget):
    """The conclusions, worst first, with the filters to find one among many."""

    selected = Signal(object)
    #: The filter changed: (camera or "", severity or "", text). The window
    #: turns it into a query; this widget knows nothing about the store.
    filtered = Signal(str, str, str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._incidents: list = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._build_filters())
        self.tree = QTreeWidget()
        self.tree.setColumnCount(6)
        self.tree.setHeaderLabels(["Opened", "Severity", "Risk", "Summary", "Cameras", "State"])
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        header = self.tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        for column, width in ((0, 78), (1, 74), (2, 52), (4, 110), (5, 108)):
            self.tree.setColumnWidth(column, width)
        self.tree.itemSelectionChanged.connect(self._selection_changed)
        layout.addWidget(self.tree)

    def _build_filters(self) -> QWidget:
        from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLineEdit

        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(4, 2, 4, 2)
        row.setSpacing(6)
        self.camera_filter = QComboBox()
        self.camera_filter.addItem("every camera", "")
        self.camera_filter.currentIndexChanged.connect(self._filters_changed)
        self.severity_filter = QComboBox()
        self.severity_filter.addItem("any severity", "")
        for severity in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
            self.severity_filter.addItem(f"{severity.lower()} and worse", severity)
        self.severity_filter.currentIndexChanged.connect(self._filters_changed)
        self.text_filter = QLineEdit()
        self.text_filter.setPlaceholderText("text in the summary or a note")
        self.text_filter.setClearButtonEnabled(True)
        self.text_filter.textChanged.connect(self._filters_changed)
        row.addWidget(self.camera_filter)
        row.addWidget(self.severity_filter)
        row.addWidget(self.text_filter, 1)
        return bar

    def set_cameras(self, camera_ids: Sequence[str]) -> None:
        """Keep the picker in step with the site without losing the choice."""
        chosen = self.camera_filter.currentData()
        self.camera_filter.blockSignals(True)
        try:
            self.camera_filter.clear()
            self.camera_filter.addItem("every camera", "")
            for camera_id in camera_ids:
                self.camera_filter.addItem(camera_id, camera_id)
            index = self.camera_filter.findData(chosen)
            self.camera_filter.setCurrentIndex(max(0, index))
        finally:
            self.camera_filter.blockSignals(False)

    def filters(self) -> tuple[str, str, str]:
        return (self.camera_filter.currentData() or "", self.severity_filter.currentData() or "",
                self.text_filter.text().strip())

    def _filters_changed(self, *_args) -> None:
        self.filtered.emit(*self.filters())

    def show_incidents(self, incidents: Sequence) -> None:
        chosen = self.selected_incident()
        self._incidents = list(incidents)
        self.tree.clear()
        for incident in self._incidents:
            item = QTreeWidgetItem([
                time.strftime("%H:%M:%S", time.gmtime(incident.opened_at_millis / 1000)),
                str(incident.severity), f"{incident.risk.score:.2f}", incident.summary,
                ", ".join(incident.cameras), _review_text(incident.review),
            ])
            item.setData(0, Qt.ItemDataRole.UserRole, incident.id)
            item.setForeground(1, theme.severity_colour(str(incident.severity)))
            item.setForeground(5, _review_colour(str(incident.review.state)))
            item.setToolTip(5, incident.review.describe())
            item.setToolTip(3, "\n".join(f"{e.occurred_at:%H:%M:%S} [{e.severity}] {e.summary}" for e in incident.events))
            self.tree.addTopLevelItem(item)
            if chosen is not None and incident.id == chosen.id:
                item.setSelected(True)

    def selected_incident(self):
        items = self.tree.selectedItems()
        if not items:
            return None
        chosen = items[0].data(0, Qt.ItemDataRole.UserRole)
        return next((i for i in self._incidents if i.id == chosen), None)

    def _selection_changed(self) -> None:
        self.selected.emit(self.selected_incident())


class TrackTable(QWidget):
    """What is being tracked right now, and where it is on the ground."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(7)
        self.tree.setHeaderLabels(["Camera", "Track", "Class", "Conf.", "Speed", "From camera", "Doing"])
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        self.tree.setTextElideMode(Qt.TextElideMode.ElideRight)
        header = self.tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        for column, width in ((0, 96), (1, 52), (2, 92), (3, 52), (4, 68), (5, 110)):
            self.tree.setColumnWidth(column, width)
        layout.addWidget(self.tree)

    def show_tracks(self, rows: Sequence[tuple]) -> None:
        """``rows`` is (camera_id, track, detector_info, pose, relations)."""
        from ...domain.geo import distance_from_camera

        self.tree.clear()
        for camera_id, track, info, pose, relations in rows:
            label = info.label_for(track.class_id) if info is not None else None
            away, position = "—", "not placed on the ground"
            if track.position is not None:
                if track.position.is_projected:
                    position = (f"{track.position.point.lat:.6f}, {track.position.point.lon:.6f} "
                                f"±{track.position.radius_meters:.1f} m")
                    if pose is not None:
                        # Always with its error: a bare "11 m" invites somebody
                        # to act on a precision nobody measured.
                        away = distance_from_camera(pose, track.position).describe()
                else:
                    # Never a number pretending to be a fix: the camera's own
                    # position with the whole field of view as its error.
                    position = f"at the camera (not projected, ±{track.position.radius_meters:.0f} m)"
            doing = "; ".join(r.describe(_namer(rows, info)) for r in relations)
            item = QTreeWidgetItem([
                camera_id, str(track.id), label or "unclassified" if info and info.classifies else label or "—",
                f"{track.confidence:.2f}", "—" if track.speed_mps is None else f"{track.speed_mps:.1f} m/s",
                away, doing,
            ])
            item.setForeground(1, theme.track_colour(track.id))
            item.setToolTip(5, position)
            if relations:
                # Every relation is inferred; the reasons must be one hover away.
                item.setToolTip(6, "\n\n".join(
                    r.describe(_namer(rows, info)) + "\n  " + "\n  ".join(r.conditions) for r in relations))
                item.setForeground(6, theme.STALE)
            if track.coasting:
                item.setToolTip(1, "Coasting: the detector cannot see it and the tracker is extrapolating.")
                item.setForeground(0, theme.TEXT_FAINT)
            self.tree.addTopLevelItem(item)


def _namer(rows: Sequence[tuple], info):
    """How a track reads in a relation's sentence: its label, or its id."""
    labels = {}
    for row in rows:
        track, row_info = row[1], row[2]
        name = row_info.label_for(track.class_id) if row_info is not None else None
        labels[track.id] = name or f"track {track.id}"

    def name_of(track_id: int) -> str:
        return labels.get(track_id, f"track {track_id}")

    return name_of


class AuditView(QWidget):
    """The chain of custody, readable. Without this it is written for nobody."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(5)
        self.tree.setHeaderLabels(["When (UTC)", "Who", "Action", "Subject", "Detail"])
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        header = self.tree.header()
        header.setStretchLastSection(True)
        for column, width in ((0, 148), (1, 130), (2, 168), (3, 130)):
            self.tree.setColumnWidth(column, width)
        layout.addWidget(self.tree)

    def show_rows(self, rows: Sequence) -> None:
        self.tree.clear()
        for row in rows:
            stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(row["at"] / 1000))
            item = QTreeWidgetItem([stamp, row["principal"], row["action"], row["subject"] or "", row["detail"] or ""])
            if row["before"] or row["after"]:
                item.setToolTip(4, f"before {row['before']}\nafter  {row['after']}")
                item.setForeground(2, theme.ACCENT)
            self.tree.addTopLevelItem(item)


class IncidentDetail(QWidget):
    """Why the system said what it said, for the incident that is selected.

    Every claim with the evidence under it: the risk factors and their
    weights, each event with the conditions the rule actually checked, and
    each association with the reasons it rests on. An operator who cannot
    see this has to take the conclusion on faith, and a conclusion taken on
    faith is one nobody can defend afterwards.
    """

    def __init__(self, parent: QWidget | None = None):
        from PySide6.QtWidgets import QTextBrowser

        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.text = QTextBrowser()
        self.text.setOpenExternalLinks(False)
        layout.addWidget(self.text)
        self.show_incident(None)

    def show_incident(self, incident) -> None:
        if incident is None:
            self.text.setHtml(f"<p style='color:{theme.TEXT_FAINT.name()}'>Select an incident to see why it was raised.</p>")
            return
        colour = theme.severity_colour(str(incident.severity)).name()
        rows = [f"<h3 style='margin:0'>{_escape(incident.summary)}</h3>",
                f"<p style='margin:2px 0'><b style='color:{colour}'>{incident.severity}</b>"
                f" &nbsp; risk {incident.risk.score:.2f} &nbsp; "
                f"<span style='color:{theme.TEXT_MUTED.name()}'>{incident.id}</span></p>",
                f"<p style='color:{theme.TEXT_MUTED.name()};margin:2px 0'>"
                f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(incident.opened_at_millis / 1000))} UTC, "
                f"lasting {incident.duration_millis / 1000:.0f} s &nbsp;·&nbsp; "
                f"{incident.distinct_objects} distinct object(s) &nbsp;·&nbsp; "
                f"cameras {_escape(', '.join(incident.cameras))}</p>"]

        rows.append(f"<p style='margin:2px 0;color:{theme.TEXT_MUTED.name()}'>Review: "
                    f"{_escape(incident.review.describe())}</p>")
        rows.append("<h4>Risk</h4><ul>")
        for factor in incident.risk.factors:
            rows.append(f"<li>{_escape(factor.name)} <b>+{factor.weight:.2f}</b> — {_escape(factor.reason)}</li>")
        rows.append("</ul>")

        rows.append(f"<h4>Events ({len(incident.events)})</h4>")
        for event in incident.events:
            severity = theme.severity_colour(str(event.severity)).name()
            place = "not projected"
            if event.evidence.latitude is not None:
                place = (f"{event.evidence.latitude:.6f}, {event.evidence.longitude:.6f} "
                         f"±{event.evidence.position_uncertainty_meters:.1f} m")
            detector = event.evidence.detector
            what = detector.name if detector.classifies else f"{detector.name} (does not classify)"
            rows.append(
                f"<p style='margin:6px 0 0 0'><b style='color:{severity}'>{event.severity}</b> "
                f"{event.occurred_at:%H:%M:%S} — {_escape(event.summary)}</p>"
                f"<p style='margin:0;color:{theme.TEXT_MUTED.name()}'>rule <code>{_escape(event.rule_id)}</code>, "
                f"confidence {event.confidence:.2f}, {event.evidence.observations} observation(s), "
                f"camera {_escape(event.evidence.camera_id)} track {event.evidence.track_id}, {place}<br>"
                f"drawn by {_escape(what)}</p><ul style='margin:2px 0'>")
            for condition in event.evidence.conditions:
                rows.append(f"<li>{_escape(condition)}</li>")
            rows.append("</ul>")

        if incident.associations:
            rows.append(f"<h4>Why these were treated as the same object ({len(incident.associations)})</h4>")
            for link in incident.associations:
                rows.append(f"<p style='margin:4px 0 0 0'>{_escape(str(link.a))} ↔ {_escape(str(link.b))}"
                            f" &nbsp; score {link.score:.2f}</p><ul style='margin:2px 0'>")
                for reason in link.reasons:
                    rows.append(f"<li>{_escape(reason)}</li>")
                rows.append("</ul>")
        else:
            rows.append(f"<p style='color:{theme.TEXT_MUTED.name()}'>No association: nothing was joined to anything else.</p>")
        self.text.setHtml("".join(rows))


def _escape(text) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
