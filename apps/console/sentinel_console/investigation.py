"""Asking the record what happened, from the console.

The store has always been able to answer this. `sentinel.search` takes a camera,
a span of wall-clock time, a severity floor, a zone and a phrase, and hands back
the matching events or incidents together with *how many there were*. Nothing in
the interface ever asked it. An operator's only way into the evidence was the
live incident list — the most recent hundred, newest first — so the question an
investigation actually starts from, *what happened at the north gate between two
and four on Tuesday, and was any of it after hours*, could not be put to this
console at all. Evidence nobody can reach is evidence nobody has.

Three things this panel is careful about, each of them a way a search screen
quietly lies to the person using it:

**A page always says how big the answer was.** The line under the results is
built from `Results.total`, never from the number of rows on screen, and when
the page is short it says "showing 20 of 314" rather than showing twenty. A
silently truncated search in a security product is how an operator concludes
nothing happened on a night when a great deal did. This is the panel's whole
promise, and `test_a_truncated_page_says_how_many_there_really_were` is what
holds it.

**Nothing found says so.** An empty result renders as a sentence explaining that
the record was searched and holds nothing that fits, because an empty list and a
broken panel look identical, and the operator has no way to tell which they are
looking at.

**A refused query is reported, not swallowed.** A window whose end precedes its
start is a swapped pair of fields, not an absence of evidence, and the engine
raises rather than answering with an empty page. That message reaches the same
line the counts do — never a traceback out of a Qt slot, which the interpreter
retains and which pins this widget past the QApplication's own destruction.

Times are UTC throughout, stated on screen. The store keeps one clock and the
window is expressed in it; a filter bar reading "02:00" against the machine's
local time would search the wrong hours for half the year in half the world, and
would not look wrong while doing it.
"""

from __future__ import annotations

from contextlib import contextmanager

from PySide6.QtCore import QDateTime, QTimeZone, Qt, Signal
from PySide6.QtGui import QBrush, QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDateTimeEdit,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from sentinel.events import Severity
from sentinel.search import (
    DEFAULT_LIMIT,
    Query,
    Results,
    SearchError,
    Window,
    at_least,
    search_events,
    search_incidents,
)

from . import theme

# The severity palette the incident list already uses, imported rather than
# rebuilt: two tables would drift, and an incident shown orange in one panel and
# red in the one beside it teaches an operator to stop reading the colour.
from .incident_view import SEVERITY_COLOUR
from .selection import INCIDENT as INCIDENT_KIND, TRACK as TRACK_KIND, Selection

#: What a search can be about. Incidents are the conclusions and events are the
#: grounds for them, and an investigation needs both: "which incidents mention
#: the loading bay" and "every after-hours event on cam-02, incident or not" are
#: different questions, and the second one has no answer in the incident list.
SUBJECT_INCIDENTS = "incidents"
SUBJECT_EVENTS = "events"

#: Plural and singular, so a single result does not read "1 incidents".
SUBJECT_NOUNS = {
    SUBJECT_INCIDENTS: ("incidents", "incident"),
    SUBJECT_EVENTS: ("events", "event"),
}

#: The largest page this panel will ever ask for. The search is a direct call on
#: the interface thread — see `InvestigationPanel.search` — so the page size is
#: also the bound on how much row-building can hold the window still. Five
#: hundred rows is more than anyone reads and builds in well under a frame;
#: beyond it the honest instrument is the total beside the page, which says how
#: much more there was and does not cost anything to produce.
MAX_RESULTS = 500

#: Columns of the result list. Named, because a search that put the summary in
#: the "where" column would still pass a test that indexed by number.
WHEN_COLUMN = 0
SEVERITY_COLUMN = 1
WHAT_COLUMN = 2
WHERE_COLUMN = 3
CAMERAS_COLUMN = 4

#: How long a window the panel opens with: the last day, ending now. A window is
#: off until the operator turns it on, so this is only ever a starting point —
#: but it is a defensible one, and an unset pair of date fields showing the
#: start of the epoch invites an operator to search a span they did not mean.
DEFAULT_WINDOW_HOURS = 24

#: Shown for a zone-less event. Not an empty cell: a blank column reads as a
#: value that failed to load, and "this happened in no zone" is a fact.
NO_ZONE = "—"


def _utc(millis: int) -> QDateTime:
    """A QDateTime in UTC, which is the only clock the store keeps."""
    return QDateTime.fromMSecsSinceEpoch(int(millis), QTimeZone.utc())


class InvestigationPanel(QWidget):
    """Search the record: camera, time, severity, zone and phrase.

    Fed a `sentinel.store.Store` — the node's, never one of its own; the console
    owns no database — and, so the pickers can offer real choices rather than
    free text, the cameras and zones the node knows about.

    Selection is emitted, never assumed. Clicking an incident selects that
    incident; clicking an event selects the *track* it is about, keyed by camera
    as well as by id, because a track number is only unique within one camera.
    `set_selection` is what the console's selection bus calls and is therefore
    deliberately silent: a setter that re-emitted would push the bus and the
    panel round in a loop, which is a frozen window rather than a wrong pixel.
    """

    #: A `Selection` for the row the operator picked, or ``None`` when the list
    #: was cleared. The console's selection bus is on the other end of this.
    selected = Signal(object)

    def __init__(self, store=None, parent: QWidget | None = None):
        super().__init__(parent)
        self._store = store
        #: How many nested reasons there are to ignore Qt's signals — the panel
        #: changing its own highlight, or filling its own pickers. A depth count
        #: rather than a flag: `_fill` inside `set_selection` inside a rebuild is
        #: a shape this panel does not have today but is one edit away from, and
        #: with a flag the inner `finally` would clear the outer guard and turn a
        #: rebuild into a `selected` emit the operator never asked for.
        self._quiet = 0
        self._selected: Selection | None = None
        #: The last exception `_row_selected` swallowed, kept for the tests and
        #: for anyone reading a bug report. A click that fails silently leaves an
        #: operator prodding rows that do nothing, with nothing anywhere to read.
        self.last_selection_failure: Exception | None = None

        self.subject = QComboBox()
        self.subject.addItem("Incidents", SUBJECT_INCIDENTS)
        self.subject.addItem("Events", SUBJECT_EVENTS)
        self.subject.setToolTip(
            "Incidents are the conclusions; events are the observations they "
            "rest on. An event can belong to no incident and still be evidence."
        )

        self.camera = QComboBox()
        self.camera.addItem("Any camera", None)
        self.camera.setMinimumWidth(120)

        self.zone = QComboBox()
        self.zone.addItem("Any zone", None)
        self.zone.setMinimumWidth(140)
        self.zone.setToolTip(
            "Matched on the zone's id, not its name, so renaming a zone does not "
            "lose the evidence raised in it."
        )

        self.severity = QComboBox()
        self.severity.addItem("Any severity", None)
        # Most serious first: an operator narrowing a search is nearly always
        # cutting the noise off the bottom, and "and above" rather than "exactly"
        # because that is the filter they mean nine times in ten. The list comes
        # from the engine's own ordering through `at_least`, so a severity added
        # between two others cannot leave this combo behind.
        for member in reversed(list(Severity)):
            self.severity.addItem(f"{member.value.title()} and above", member.value)

        self.term = QLineEdit()
        self.term.setPlaceholderText("Phrase — a summary, a zone name, a plate fragment…")
        self.term.setClearButtonEnabled(True)
        self.term.setToolTip(
            "Matched literally, as a substring, ignoring case. Wildcards and "
            "quotes are searched for rather than obeyed."
        )
        # A bound method, never a lambda closing over `self`: a lambda in a
        # signal connection is a reference cycle holding a QWidget, and the
        # widget is then destroyed at interpreter shutdown, after PySide has torn
        # the QApplication down, which corrupts the heap on the way out.
        self.term.returnPressed.connect(self.search)

        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self._search_clicked)

        self.windowed = QCheckBox("Only between")
        self.windowed.setToolTip(
            "Off: all of recorded time. On: this span, in UTC. An incident that "
            "was already running when the window opened is included — an "
            "investigator scrubbing to 03:00 must see the incident that began at "
            "02:58 and had not finished."
        )
        now = QDateTime.currentDateTimeUtc().toMSecsSinceEpoch()
        self.start = QDateTimeEdit()
        self.end = QDateTimeEdit()
        for edit in (self.start, self.end):
            edit.setTimeZone(QTimeZone.utc())
            edit.setDisplayFormat("yyyy-MM-dd HH:mm")
            edit.setCalendarPopup(True)
            edit.setEnabled(False)
        self.start.setDateTime(_utc(now - DEFAULT_WINDOW_HOURS * 3_600_000))
        self.end.setDateTime(_utc(now))
        self.windowed.toggled.connect(self._window_toggled)

        self.limit = QSpinBox()
        self.limit.setRange(1, MAX_RESULTS)
        self.limit.setValue(min(DEFAULT_LIMIT, MAX_RESULTS))
        self.limit.setToolTip(
            "How many rows one page holds. The count under the list always says "
            "how many matched, whether or not they fit."
        )

        self.results = QTreeWidget()
        self.results.setColumnCount(5)
        self.results.setHeaderLabels(["When (UTC)", "Severity", "What", "Where", "Cameras"])
        self.results.setRootIsDecorated(False)
        self.results.setAlternatingRowColors(True)
        self.results.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        header = self.results.header()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        for index, width in enumerate((140, 80, 380, 150)):
            self.results.setColumnWidth(index, width)
        self.results.itemSelectionChanged.connect(self._row_selected)

        #: The line that keeps a page from reading as the whole answer. Never
        #: built from the number of rows on screen.
        self.summary = QLabel("")
        self.summary.setObjectName("Caption")
        self.summary.setWordWrap(True)

        filters = QHBoxLayout()
        filters.setContentsMargins(6, 4, 6, 0)
        filters.addWidget(self.subject)
        filters.addWidget(self.camera)
        filters.addWidget(self.zone)
        filters.addWidget(self.severity)
        filters.addWidget(self.term, 1)
        filters.addWidget(self.search_button)

        when = QHBoxLayout()
        when.setContentsMargins(6, 0, 6, 0)
        when.addWidget(self.windowed)
        when.addWidget(self.start)
        when.addWidget(QLabel("and"))
        when.addWidget(self.end)
        when.addSpacing(12)
        when.addWidget(QLabel("Show at most"))
        when.addWidget(self.limit)
        when.addStretch(1)

        clock = QLabel("Times are UTC — the clock the record is kept on.")
        clock.setObjectName("Caption")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addLayout(filters)
        layout.addLayout(when)
        layout.addWidget(clock)
        layout.addWidget(self.results, 1)
        layout.addWidget(self.summary)

        # Picking from a list is a decision, so it re-runs the search. Typing and
        # date entry are not: a search per keystroke would hit the database four
        # times while somebody spells "loading", and each of those is a full scan
        # in the worst case. Those wait for Enter or the button.
        self.subject.currentIndexChanged.connect(self._filters_changed)
        self.camera.currentIndexChanged.connect(self._filters_changed)
        self.zone.currentIndexChanged.connect(self._filters_changed)
        self.severity.currentIndexChanged.connect(self._filters_changed)
        self.limit.valueChanged.connect(self._filters_changed)

        self.search()

    # --------------------------------------------------------------- silence

    @contextmanager
    def _silent(self):
        """Ignore Qt's signals for the duration, however deeply nested.

        Every one of these blocks is the panel acting on itself — clearing a
        tree, filling a picker, moving a highlight — and each of those fires the
        same signals an operator's click does. Counted rather than flagged so a
        block inside a block cannot unmute its caller half way through: that
        would emit `selected` for a row nobody chose, and the console's selection
        bus would push every other panel to a thing the operator never clicked.
        """
        self._quiet += 1
        try:
            yield
        finally:
            self._quiet -= 1

    # ---------------------------------------------------------------- the node

    def set_store(self, store) -> None:
        """Point the panel at the node's store and search it.

        The console owns no database. This is the node's, handed over once the
        node exists; until then the panel says it has nothing to search rather
        than showing an empty list, which would be indistinguishable from a site
        on which nothing has ever happened.
        """
        self._store = store
        self.search()

    def set_cameras(self, camera_ids) -> None:
        """Fill the camera picker, keeping the operator's choice if it survives.

        Ids rather than free text: a camera filter that silently matches nothing
        because of a typo returns an empty page, and an empty page is read as an
        absence of evidence.
        """
        self._refill(self.camera, "Any camera", [(str(cid), str(cid)) for cid in camera_ids])

    def set_zones(self, zones) -> None:
        """Fill the zone picker from the node's zones — name shown, id matched."""
        self._refill(
            self.zone,
            "Any zone",
            [(getattr(z, "name", str(z)), getattr(z, "id", None)) for z in zones],
        )

    def _refill(self, combo: QComboBox, any_label: str, entries) -> None:
        """Rebuild a picker without re-running the search for each row added.

        Quiet throughout: every `addItem` fires ``currentIndexChanged``, and a
        node with eight cameras would otherwise run eight searches to end up
        showing what it already showed.
        """
        was = combo.currentData()
        with self._silent():
            combo.clear()
            combo.addItem(any_label, None)
            for label, value in entries:
                if value is None:
                    continue
                combo.addItem(str(label), str(value))
            index = combo.findData(was)
            combo.setCurrentIndex(index if index >= 0 else 0)

    # ------------------------------------------------------------- the query

    @property
    def subject_kind(self) -> str:
        return self.subject.currentData() or SUBJECT_INCIDENTS

    def time_window(self) -> Window | None:
        """The span the operator asked for, or ``None`` for all of recorded time.

        Read out of the fields in UTC, because that is what they display and what
        the store holds. A window whose end precedes its start is refused by the
        engine, and the refusal is shown rather than raised — the operator
        swapped two fields, which is a thing to say, not an empty page.
        """
        if not self.windowed.isChecked():
            return None
        return Window(
            self.start.dateTime().toMSecsSinceEpoch(),
            self.end.dateTime().toMSecsSinceEpoch(),
        )

    def query(self) -> Query:
        """The filter bar as the engine's `Query`. Unset controls are not filters."""
        camera = self.camera.currentData()
        zone = self.zone.currentData()
        floor = self.severity.currentData()
        return Query(
            cameras=(camera,) if camera else (),
            window=self.time_window(),
            severities=at_least(Severity(floor)) if floor else (),
            zones=(zone,) if zone else (),
            term=self.term.text(),
            limit=int(self.limit.value()),
        )

    # ------------------------------------------------------------ the search

    def search(self) -> None:
        """Run the current filters against the store and show what came back.

        A direct, synchronous call, on the interface thread, deliberately: this
        is one indexed SELECT and one COUNT inside a single transaction, and a
        worker thread would buy a few milliseconds at the price of a second
        SQLite connection and a class of ordering bug that is much harder to see
        than a brief pause.

        What that costs on a large database, said plainly rather than wished
        away: a phrase search is a substring match, which no index can serve, so
        it reads every event row. On a node holding a season of evidence — call
        it a few million events — that is seconds, and for those seconds this
        window does not repaint. The page size bounds the rows *built*, never the
        rows *examined*, so it does not help with this; a time window does, which
        is why the window sits at the front of the filter bar. If that pause ever
        becomes the complaint, the fix is a full-text index in the store, not a
        thread here.

        Nothing raises out of this method. It is reached from a Qt slot, and an
        exception escaping one is retained by the interpreter, which pins this
        widget past the QApplication's own destruction and corrupts the heap at
        exit. A refused query and a failed one are both reported on the line
        under the results, where the counts are — and so is a failure to *draw*
        the answer, which is the half that was left outside the guard once and
        put a `ValueError` straight out through `returnPressed`. Building a row
        touches a summary, a severity and a set of zones on every item that came
        back; a column this panel does not expect is a live possibility, not a
        theoretical one.
        """
        if self._store is None:
            self._fill(())
            self.summary.setText("No database is open, so there is nothing to search.")
            return

        try:
            query = self.query()
            if self.subject_kind == SUBJECT_EVENTS:
                results = search_events(self._store, query)
            else:
                results = search_incidents(self._store, query)
        except SearchError as refused:
            self._fill(())
            self.summary.setText(f"That search cannot be run: {refused}")
            return
        except Exception as failure:  # pragma: no cover - defensive, see above
            self._fill(())
            self.summary.setText(f"The search failed: {failure}")
            return

        try:
            self._fill(results.items)
            self.summary.setText(self._summarise(results, query))
        except Exception as failure:
            # The query came back; this panel could not draw it. Reported the
            # same way a refusal is, and the half-built list is cleared, because
            # rows from an answer that was never finished are worse than none:
            # they read as the whole of what matched.
            self._fill(())
            self.summary.setText(f"The search failed: {failure}")

    def _search_clicked(self, _checked: bool = False) -> None:
        """The button. Its ``clicked(bool)`` argument is not a filter."""
        self.search()

    def _filters_changed(self, *_ignored) -> None:
        """A picker moved. Silent while the pickers are being filled."""
        if self._quiet:
            return
        self.search()

    def _window_toggled(self, on: bool) -> None:
        for edit in (self.start, self.end):
            edit.setEnabled(on)
        self._filters_changed()

    # ------------------------------------------------------------- the answer

    def _summarise(self, results: Results, query: Query) -> str:
        """The line under the list, built from `Results.total` and nothing else.

        The truncated wording comes from `Results.describe` rather than being
        retyped here, so the number an operator reads and the number the engine
        counted cannot come to differ.

        The advice that follows it is conditional for the same reason. The page
        size stops at `MAX_RESULTS`, so telling an operator already at the bound
        to raise it is advice they cannot take — and a panel that hands out an
        impossible instruction teaches them to stop reading this line, which is
        the one line here that matters.
        """
        plural, singular = SUBJECT_NOUNS[self.subject_kind]
        if results.total == 0:
            if query.is_empty:
                return f"No {plural} have been recorded. Nothing matched because nothing is there."
            return (
                f"No {plural} matched {query.describe()}. The record was searched "
                "and holds nothing that fits."
            )
        noun = plural if results.total != 1 else singular
        if results.truncated:
            if query.limit < MAX_RESULTS:
                advice = "narrow the search, or raise the page size, to see the rest"
            else:
                advice = (
                    f"the page is already at its bound of {MAX_RESULTS:,}, so "
                    "narrowing the search — a camera, a zone or a time window — is "
                    "the only way to see the rest"
                )
            return f"{results.describe()} {noun} matching {query.describe()} — {advice}."
        return f"{results.total:,} {noun} matching {query.describe()}."

    def _fill(self, items) -> None:
        """Replace every row, keeping the operator's selection if it is still here.

        Rebuilt rather than diffed, like the incident and camera panels: a reused
        row can keep a stale value in a column that failed to update, and in an
        evidence view a stale cell is worse than a flicker.
        """
        with self._silent():
            self.results.clear()
            for item in items:
                row = (
                    self._event_row(item)
                    if self.subject_kind == SUBJECT_EVENTS
                    else self._incident_row(item)
                )
                self.results.addTopLevelItem(row)
                if row.data(WHEN_COLUMN, Qt.ItemDataRole.UserRole) == self._selected:
                    row.setSelected(True)
                    self.results.setCurrentItem(row)

    def _incident_row(self, incident) -> QTreeWidgetItem:
        item = QTreeWidgetItem([
            f"{incident.opened_at:%Y-%m-%d %H:%M:%S}",
            incident.severity.value,
            incident.summary,
            ", ".join(incident.zones) if incident.zones else NO_ZONE,
            ", ".join(incident.cameras),
        ])
        item.setData(WHEN_COLUMN, Qt.ItemDataRole.UserRole, Selection.incident(incident.id))
        item.setForeground(
            SEVERITY_COLUMN, QBrush(SEVERITY_COLOUR.get(incident.severity, theme.TEXT))
        )
        bold = QFont(self.results.font())
        bold.setBold(True)
        item.setFont(WHAT_COLUMN, bold)
        item.setToolTip(
            WHAT_COLUMN,
            f"{incident.distinct_objects} object(s), {len(incident.events)} event(s), "
            f"lasting {incident.duration_millis / 1000:.0f}s. Risk "
            f"{incident.risk.score:.0f}.",
        )
        return item

    def _event_row(self, event) -> QTreeWidgetItem:
        evidence = event.evidence
        item = QTreeWidgetItem([
            f"{event.occurred_at:%Y-%m-%d %H:%M:%S}",
            event.severity.value,
            event.summary,
            event.zone_name or NO_ZONE,
            f"{evidence.camera_id} #{evidence.track_id}",
        ])
        # A track, not the event: a selection is a thing on the ground that every
        # other panel can point at, and the wall, the map and the track table all
        # know what a track is. Keyed by camera as well as by number, because #3
        # on the gate and #3 on the yard are different people.
        item.setData(
            WHEN_COLUMN,
            Qt.ItemDataRole.UserRole,
            Selection.track(evidence.camera_id, evidence.track_id),
        )
        item.setForeground(
            SEVERITY_COLUMN, QBrush(SEVERITY_COLOUR.get(event.severity, theme.TEXT))
        )
        item.setToolTip(
            WHAT_COLUMN,
            f"{event.type.value} · rule {event.rule_id} · "
            f"confidence {event.confidence:.2f}",
        )
        return item

    # ------------------------------------------------------------- selection

    def _row_selected(self) -> None:
        """Qt's selection changed. Tell the bus, unless we caused it ourselves.

        Nothing may raise out of here: a traceback escaping a Qt slot is retained
        by the interpreter and pins the widget it came from past the
        QApplication's own destruction.

        Caught is not the same as hidden. A selection that never reaches the bus
        leaves an operator clicking rows while the map, the wall and the track
        table sit on somebody else's choice — and if this handler says nothing,
        there is no sign of it anywhere, on screen or in a bug report. So the
        failure goes on the same line the counts and the refusals go on, and is
        kept where a test can read it.
        """
        if self._quiet:
            return
        try:
            selection = self.selected_row()
            self._selected = selection
            self.selected.emit(selection)
        except Exception as failure:
            self.last_selection_failure = failure
            self.summary.setText(
                f"That row could not be selected: {failure}. The other panels are "
                "still showing whatever was selected before."
            )

    def selected_row(self) -> Selection | None:
        """What the highlighted row is about, or ``None`` when none is."""
        items = self.results.selectedItems()
        if not items:
            return None
        return items[0].data(WHEN_COLUMN, Qt.ItemDataRole.UserRole)

    def set_selection(self, selection) -> None:
        """Highlight what was selected elsewhere, emitting nothing.

        Silent by construction rather than by luck: the console's selection bus
        calls this from its own ``changed`` signal, so a re-emit here would go
        straight back into the bus and the window would stop answering.

        A selection this list cannot show — a camera, a zone, or an incident that
        is not on the current page — clears the highlight rather than guessing at
        a row, because a highlighted row is a claim that the operator picked it.
        The search is deliberately not re-run to go and find it: a selection made
        in another panel must not silently rewrite the filters the operator set
        here.
        """
        kind = getattr(selection, "kind", None) if selection is not None else None
        wanted = selection if kind in (INCIDENT_KIND, TRACK_KIND) else None
        self._selected = wanted
        with self._silent():
            if wanted is None:
                self.results.clearSelection()
                self.results.setCurrentItem(None)
                return
            for index in range(self.results.topLevelItemCount()):
                item = self.results.topLevelItem(index)
                if item.data(WHEN_COLUMN, Qt.ItemDataRole.UserRole) == wanted:
                    self.results.setCurrentItem(item)
                    item.setSelected(True)
                    self.results.scrollToItem(item)
                    return
            self.results.clearSelection()
            self.results.setCurrentItem(None)
