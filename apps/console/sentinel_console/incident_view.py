"""The incident list: what actually deserves the operator's attention.

This panel is the product. Everything else on screen — the frame, the tracks, the
plan view — is how the system arrived at what is written here, and an operator
who is doing their job well spends most of their time not reading any of it.

Which makes restraint the design constraint. A list that fills up is a list
nobody reads, and the seventh entry that mattered is lost among six that did not.
So the panel shows incidents, never events: twelve seconds of people crossing a
restricted area is one row, expandable into the sixteen events that evidence it.

Two things are always visible without expanding, because they are what an
operator triages on: **how many distinct objects**, and **why the system thinks
this is serious**. The risk score is never shown without its reasons — a number
an operator cannot interrogate is a number they eventually learn to ignore.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import QHeaderView, QTreeWidget, QTreeWidgetItem, QWidget

from sentinel.events import Severity
from sentinel.incidents import Incident

from . import theme

#: Colour per severity. An operator reads colour before text.
SEVERITY_COLOUR = {
    Severity.CRITICAL: theme.FAULT,
    Severity.HIGH: QColor(251, 146, 60),
    Severity.MEDIUM: theme.STALE,
    Severity.LOW: theme.DETECTION,
    Severity.INFO: theme.TEXT_MUTED,
}


class IncidentView(QTreeWidget):
    """A list of incidents, each expandable into its evidence."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setColumnCount(6)
        self.setHeaderLabels(["Incident", "Severity", "Objects", "Cameras", "Risk", "When"])
        self.setAlternatingRowColors(True)
        self.setUniformRowHeights(False)
        self.setExpandsOnDoubleClick(True)

        header = self.header()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        for index, width in enumerate((420, 90, 70, 110, 80)):
            self.setColumnWidth(index, width)

        font = QFont("Consolas, monospace")
        font.setPointSize(11)
        self.setFont(font)

        self._expanded: set[str] = set()
        self._selected: str | None = None

    def show_incidents(self, incidents: list[Incident]) -> None:
        """Replace the list, preserving which rows the operator had open.

        Rebuilt rather than diffed: at the handful of open incidents a site
        produces this costs nothing, and a tree that reuses rows can leave a
        stale value in a column that failed to update — which in an evidence view
        is worse than a flicker. Expansion is preserved by id, because an
        operator reading an incident should not have it collapse under them when
        a new event arrives.
        """
        self._remember_expansion()
        self.clear()

        # Most serious first, then most recent. Triage order, not arrival order.
        ordered = sorted(
            incidents,
            key=lambda i: (-_rank(i.severity), -i.opened_at_millis),
        )

        for incident in ordered:
            self.addTopLevelItem(self._row_for(incident))

        for index in range(self.topLevelItemCount()):
            item = self.topLevelItem(index)
            if item.data(0, Qt.ItemDataRole.UserRole) in self._expanded:
                item.setExpanded(True)
            # A rebuild must not silently drop the selection: this panel is
            # rebuilt on every collection tick.
            if item.data(0, Qt.ItemDataRole.UserRole) == self._selected:
                item.setSelected(True)
                self.setCurrentItem(item)

    def set_selection(self, selection) -> None:
        """Bring the selected incident's row forward, without stealing focus.

        `setCurrentItem` rather than `scrollToItem` alone: an operator who
        selected the incident somewhere else needs to see which row it is, and
        a row highlighted but off-screen is not an answer.
        """
        incident_id = getattr(selection, "incident_id", None) if selection else None
        self._selected = incident_id
        if incident_id is None:
            self.clearSelection()
            return
        for index in range(self.topLevelItemCount()):
            item = self.topLevelItem(index)
            if item.data(0, Qt.ItemDataRole.UserRole) == incident_id:
                self.setCurrentItem(item)
                item.setSelected(True)
                self.scrollToItem(item)
                return

    def selected_incident_id(self) -> str | None:
        """The incident whose row is current, following a child up to its parent."""
        item = self.currentItem()
        while item is not None and item.parent() is not None:
            item = item.parent()
        return None if item is None else item.data(0, Qt.ItemDataRole.UserRole)

    def _remember_expansion(self) -> None:
        for index in range(self.topLevelItemCount()):
            item = self.topLevelItem(index)
            incident_id = item.data(0, Qt.ItemDataRole.UserRole)
            if item.isExpanded():
                self._expanded.add(incident_id)
            else:
                self._expanded.discard(incident_id)

    def _row_for(self, incident: Incident) -> QTreeWidgetItem:
        colour = SEVERITY_COLOUR.get(incident.severity, theme.TEXT)

        item = QTreeWidgetItem([
            incident.summary,
            incident.severity.value,
            str(incident.distinct_objects),
            ", ".join(incident.cameras),
            f"{incident.risk.score:.0f}",
            f"t+{incident.opened_at_millis / 1000:.1f}s"
            f"  ({incident.duration_millis / 1000:.0f}s)",
        ])
        item.setData(0, Qt.ItemDataRole.UserRole, incident.id)
        item.setForeground(1, QBrush(colour))

        bold = QFont(self.font())
        bold.setBold(True)
        item.setFont(0, bold)

        item.addChild(_heading("why", incident.risk.describe().splitlines()[0]))
        for factor in incident.risk.factors:
            item.addChild(
                _detail(f"{factor.points:+.0f}", f"{factor.name} — {factor.because}")
            )

        # Cross-camera associations, with the reasoning. An operator must be able
        # to see why two cameras were treated as one object, and disagree.
        for association in incident.associations:
            item.addChild(
                _detail(
                    "linked",
                    f"{association.a[0]}#{association.a[1]} = "
                    f"{association.b[0]}#{association.b[1]} "
                    f"({association.score:.2f}): {association.reasons[0]}",
                )
            )

        item.addChild(_heading("timeline", f"{len(incident.events)} events"))
        for entry in incident.timeline():
            child = _detail(
                f"t+{entry.at_millis / 1000:.1f}s",
                f"{entry.camera_id}  {entry.summary}",
            )
            child.setForeground(
                1, QBrush(SEVERITY_COLOUR.get(entry.severity, theme.TEXT_MUTED))
            )
            item.addChild(child)

        return item


def _rank(severity: Severity) -> int:
    order = list(Severity)
    return order.index(severity)


def _heading(label: str, text: str) -> QTreeWidgetItem:
    item = QTreeWidgetItem(["", label, text])
    item.setForeground(1, QBrush(theme.TEXT_MUTED))
    return item


def _detail(label: str, text: str) -> QTreeWidgetItem:
    item = QTreeWidgetItem(["", label, text])
    item.setForeground(2, QBrush(theme.TEXT_MUTED))
    return item
