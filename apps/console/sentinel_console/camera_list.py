"""The cameras this node has, one row each, with what each is actually doing.

This replaces a combo box. A combo box shows one camera and hides the rest, so
the console could run eight cameras and look identical whether all eight were
delivering frames or seven had been dark since midnight — the operator would
find out from the gap in the evidence, weeks later.

The point of the list is therefore not the names. It is the status strip on the
right, and one distinction inside it: **a camera that is nominally running and
delivering nothing does not look like a camera that is running.** "Running" is
the runner's opinion of itself; frames arriving is the only evidence of it. A
green dot on a camera whose decoder wedged an hour ago is the most expensive lie
this window can tell, because the absence of events then reads as quiet.

The state itself is the engine's, not this panel's. `Node.camera_health()`
already decides LIVE/DARK/STARTING/STOPPED/FAULTED from the same last-frame
clock the map hatches ground from, and an interface that re-derived it with its
own thresholds would eventually paint a camera as a dark row in this list while
the map beside it painted its footprint as live coverage. So `camera_state`
translates the engine's word into the operator's and decides nothing; the
frame-clock path below it exists only for a health object that carries no state
at all.

Health facts are still read by name through `getattr` — see `HEALTH_FACTS` — so
the panel can be built and tested without a live node, but the names are the
engine's real ones. A spelling that drifts reads as "unknown" silently, which is
how the first version of this file rendered LIVE, DARK and STOPPED identically.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (
    QHeaderView,
    QLabel,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from sentinel.decode import redact_url

# The threshold, never the type: one number decides dark here and in the map's
# hatching, and a second copy of it in the console is a guarantee that the two
# windows will one day disagree about the same camera. `sentinel.node` is
# already imported by `app.py`, so this costs nothing at start-up.
from sentinel.node import DARK_AFTER_SECONDS

from . import theme
from .selection import CAMERA as CAMERA_KIND, Selection

#: The attribute names read off each health object — the field names of the
#: engine's `sentinel.node.CameraHealth`, read through `getattr` rather than by
#: importing the type so the panel can still be built and tested without a live
#: node. Anything carrying these names will do; anything carrying *other* names
#: reads as unknown in silence, so this tuple is the contract and drifting from
#: it is the one failure mode of duck typing here.
#:
#: * ``camera_id`` — which camera these facts are about.
#: * ``state`` — the engine's `CameraState`: LIVE, DARK, STARTING, STOPPED or
#:   FAULTED. **The authority.** Compared by its string value, so nothing of the
#:   engine's type system is imported. See `camera_state`.
#: * ``is_running`` — its thread is alive. Deliberately not the same as working,
#:   which is the whole reason `state` exists and is preferred over this.
#: * ``is_placed`` — it has a pose, so what it sees has a position.
#: * ``analysis_fps`` — frames analysed per second over the last second, or 0.0
#:   when no measurement is current enough to quote. Shown, never used to decide
#:   the state; a slow camera is not a dark one.
#: * ``frames`` — frames analysed for the whole run.
#: * ``frames_dropped`` — frames the live decoder discarded to stay current.
#: * ``frames_not_drawn`` — results published and never collected, lost from the
#:   screen but not from the analysis.
#: * ``reconnects`` — times the source has been reopened during this run.
#: * ``fault`` — why it stopped, or ``None``. Redacted by the engine.
#: * ``seconds_since_frame`` — since the newest frame arrived, or ``None`` when
#:   none ever has. The fallback state decision, and the tooltip's age.
#: * ``seconds_since_started`` — since its thread started, or ``None``. What
#:   makes "dark" defensible for a camera that has produced nothing at all.
HEALTH_FACTS = (
    "camera_id",
    "state",
    "is_running",
    "is_placed",
    "analysis_fps",
    "frames",
    "frames_dropped",
    "frames_not_drawn",
    "reconnects",
    "fault",
    "seconds_since_frame",
    "seconds_since_started",
)

#: The five things a camera can be, in the operator's terms rather than the
#: runner's. Strings, so a caller can compare without importing this namespace.
STATE_LIVE = "live"
STATE_STARTING = "starting"
STATE_DARK = "dark"
STATE_FAULT = "fault"
STATE_OFF = "off"

#: The engine's `CameraState` values, by name, in this panel's words. Keyed by
#: string rather than by the enum so the console imports no engine type, and
#: one-to-one so no engine state can be folded into a neighbour: STARTING is
#: emphatically not DARK — an RTSP stream takes seconds to open, and calling
#: those seconds dark teaches an operator to ignore the word by the end of the
#: first shift.
ENGINE_STATES: dict[str, str] = {
    "LIVE": STATE_LIVE,
    "DARK": STATE_DARK,
    "STARTING": STATE_STARTING,
    "STOPPED": STATE_OFF,
    "FAULTED": STATE_FAULT,
}

#: A camera that is nominally up and delivering nothing. The one colour this
#: module adds to `theme`, because none of the reserved state colours means this:
#: it is not live, it is not merely late, and it is not a fault — nothing has
#: reported anything wrong. It is the failure that reports itself as health.
#: Magenta because no other part of the console uses magenta for anything, so it
#: can never be mistaken for evidence, for a zone kind, or for a selection.
DARK = QColor(232, 121, 249)

#: Colour per state. `theme`'s meanings are reserved, so they are used exactly as
#: they are defined there and are nowhere reinterpreted.
STATE_COLOURS: dict[str, QColor] = {
    STATE_LIVE: theme.LIVE,
    STATE_STARTING: theme.STALE,
    STATE_DARK: DARK,
    STATE_FAULT: theme.FAULT,
    STATE_OFF: theme.IDLE,
}

#: Colour is not enough on its own: about one operator in twelve cannot separate
#: these hues, and a control-room monitor at a glancing angle washes them all
#: out. Filled means frames are arriving; hollow means they are not.
STATE_GLYPHS: dict[str, str] = {
    STATE_LIVE: "●",
    STATE_STARTING: "◐",
    STATE_DARK: "○",
    STATE_FAULT: "✕",
    STATE_OFF: "·",
}

CAMERA_COLUMN = 0
SOURCE_COLUMN = 1
PLACED_COLUMN = 2
STATUS_COLUMN = 3


def _fact(health, name: str, default=None):
    """One health fact, or ``default`` when this object does not carry it.

    Reading through `getattr` rather than typing the argument is what lets the
    console be built and tested without the engine's health type, and lets that
    type gain a field without this file changing. A health object that raised
    from a property would take a Qt slot down with it, so that is caught here
    too: a missing fact must degrade to "unknown", never to a dead panel.
    """
    if health is None:
        return default
    try:
        value = getattr(health, name, default)
    except Exception:
        return default
    return default if value is None else value


def camera_state(health) -> str:
    """Which of the five words this panel shows for a camera. Translation, not
    judgement.

    The engine decided this already, from the last-frame clock, and the map
    hatches ground from that same decision; deciding it again here with a second
    threshold is how one window ends up calling a camera dark while the window
    beside it paints its footprint as watched ground. So `state` wins whenever it
    is there, compared by its string value so no engine type is imported.

    The frame-clock path below is the fallback for a health object that carries
    no state — a stub, or an older engine. It is deliberately the *engine's*
    rule, not a friendlier one: silence is measured from the newest frame, or
    from the thread's start when no frame has ever arrived, so a camera that
    never connects goes dark instead of sitting at "starting" all night.

    Deliberately *not* decided on detections, in either path. A yard with nobody
    in it produces no detections all night and is working perfectly; dark is
    about frames.
    """
    if health is None:
        return STATE_OFF
    state = _fact(health, "state")
    if state is not None:
        # `.value` for the engine's StrEnum, the object itself for a plain
        # string. Never `str(state)`: on this interpreter that renders
        # "CameraState.LIVE" for the enum, which matches nothing.
        word = str(getattr(state, "value", state)).upper()
        if word in ENGINE_STATES:
            return ENGINE_STATES[word]
    if _fact(health, "fault"):
        return STATE_FAULT
    if not _fact(health, "is_running", False):
        return STATE_OFF
    age = _fact(health, "seconds_since_frame")
    silent_for = age if age is not None else _fact(health, "seconds_since_started")
    if silent_for is not None and float(silent_for) >= DARK_AFTER_SECONDS:
        return STATE_DARK
    if age is None:
        return STATE_STARTING
    return STATE_LIVE


def _fps_text(fps: float) -> str:
    """Frame rate at the precision it is actually known to.

    Below ten, one decimal: the difference between 0.4 fps and 4 fps is the
    difference between a camera in trouble and a camera working, and both render
    as "0 fps" and "4 fps" once rounded. Above ten nobody cares about the tenth.
    """
    return f"{fps:.1f} fps" if fps < 10 else f"{fps:.0f} fps"


def _age_text(seconds: float | None) -> str:
    """How stale the newest frame is, in the units an operator reads at speed."""
    if seconds is None:
        return "no frame yet"
    if seconds < 90:
        return f"no frame for {seconds:.0f}s"
    if seconds < 5400:
        return f"no frame for {seconds / 60:.0f} min"
    return f"no frame for {seconds / 3600:.0f} h"


def _newest_text(age, frames) -> str:
    """When the newest frame arrived, phrased so it cannot deny the frame count.

    "never" is a positive claim about the whole past, and the panel is only
    entitled to make it while holding a frame count that agrees. Printing
    "frames 1480" and "newest never" side by side — which is exactly what a
    misspelled field name produced here — is worse than admitting ignorance,
    because the operator cannot tell which of the two lines to believe.
    """
    if age is not None:
        return f"{float(age):.1f}s ago"
    if frames is not None and int(frames) > 0:
        return "unknown"
    return "never"


def status_text(health) -> str:
    """The status strip's words for one camera.

    Words as well as a colour and a glyph, because the difference between "live"
    and "dark" has to survive a photograph of the screen in an incident report,
    where the colour is whatever the phone's camera decided it was.
    """
    state = camera_state(health)
    if state == STATE_OFF:
        return "not started"
    if state == STATE_FAULT:
        return f"failed — {_fact(health, 'fault', '')}"
    age = _fact(health, "seconds_since_frame")
    age = None if age is None else float(age)
    if state == STATE_DARK:
        return f"dark — running, {_age_text(age)}"
    if state == STATE_STARTING:
        # Its own word, and never "dark": opening a stream takes seconds, and a
        # panel that cried dark for those seconds would train an operator to
        # scroll past the row that means it.
        started = _fact(health, "seconds_since_started")
        waited = "" if started is None else f", {float(started):.0f}s so far"
        return f"starting — opening the source{waited}"
    # Not `default=0.0`: a health object that does not carry a frame rate would
    # then read "live · 0.0 fps", which is a measurement nobody took and reads
    # as a camera in trouble. Unknown says unknown.
    fps = _fact(health, "analysis_fps")
    if fps is not None and float(fps) > 0:
        return "live · " + _fps_text(float(fps))
    # The engine quotes no rate unless a frame arrived within the last second,
    # so a live camera without one is one whose frames have slowed rather than
    # stopped — say how stale, rather than print a zero that reads as stalled.
    if age is not None and age >= 1.0:
        return f"live · {_age_text(age)}"
    return "live · fps unknown"


def health_tooltip(camera_id: str, health) -> str:
    """Every measured fact behind the strip, for the operator who does not
    believe it.

    The strip is a summary, and a summary is a claim; this is the evidence
    behind it. An unknown fact says "unknown" rather than showing a plausible
    zero, because a fabricated zero is indistinguishable from a measured one —
    and, harder, it must not print two facts that contradict each other. "1480
    frames" beside "newest never" is not an admission of ignorance, it is the
    panel disagreeing with itself in front of the person who came here to check.
    """
    lines = [camera_id]
    if health is None:
        lines.append("No health facts: this camera has never been started.")
        return "\n".join(lines)
    fps = _fact(health, "analysis_fps")
    frames = _fact(health, "frames")
    age = _fact(health, "seconds_since_frame")
    dropped = _fact(health, "frames_dropped")
    not_drawn = _fact(health, "frames_not_drawn")
    reconnects = _fact(health, "reconnects")
    lines.append(f"analysed   {'unknown' if fps is None else _fps_text(float(fps))}")
    lines.append(f"frames     {'unknown' if frames is None else int(frames)}")
    lines.append(f"newest     {_newest_text(age, frames)}")
    started = _fact(health, "seconds_since_started")
    if started is not None:
        lines.append(f"running    {float(started):.0f}s")
    if reconnects is not None:
        lines.append(f"reconnects {int(reconnects)}")
    if dropped is not None:
        lines.append(f"dropped    {int(dropped)} frames, to stay current")
    if not_drawn is not None:
        # "not shown", not "never shown": "never" belongs to the frame clock
        # alone in this tooltip, so a test can hold the whole text to the rule
        # that nothing claims never while counting frames.
        lines.append(f"not drawn  {int(not_drawn)} analysed but not shown")
    fault = _fact(health, "fault")
    if fault:
        lines.append(f"fault      {fault}")
    return "\n".join(lines)


def display_source(record) -> str:
    """The source of a camera in the only form allowed near a widget.

    `CameraRecord.display_source` is already redacted, so it is preferred; the
    fallback redacts here rather than trusting the caller, because the raw
    ``source`` may carry ``rtsp://admin:hunter2@…`` and a password that reaches a
    QTreeWidgetItem is in the accessibility tree, in every screenshot of the
    console, and in the tooltip an operator hovers in front of a visitor. There
    is deliberately no path through this panel that reads ``record.source`` and
    shows it.
    """
    shown = getattr(record, "display_source", None)
    if isinstance(shown, str) and shown:
        return shown
    return redact_url(getattr(record, "source", "") or "")


class CameraListPanel(QWidget):
    """Every camera on the node, one row each, with a live status strip.

    Selection is emitted, never assumed: the console has a selection bus, and
    this panel both feeds it and is fed by it. `set_selection` is therefore
    deliberately silent — a setter that re-emitted would make the bus and the
    panel push each other round in a loop, which is a frozen window rather than
    a wrong pixel.
    """

    #: A `Selection` for the camera the operator picked, or ``None`` when the
    #: list was cleared.
    selected = Signal(object)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        #: True while the panel is changing its own highlight on somebody else's
        #: instruction, so Qt's selection signal is not mistaken for a click.
        self._quiet = False
        self._selected_id: str | None = None

        title = QLabel("CAMERAS")
        title.setObjectName("PanelTitle")

        self.tree = QTreeWidget()
        self.tree.setColumnCount(4)
        self.tree.setHeaderLabels(["Camera", "Source", "Placed", "Status"])
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        header = self.tree.header()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        self.tree.setMinimumWidth(360)
        for index, width in enumerate((90, 170, 76)):
            self.tree.setColumnWidth(index, width)
        # A bound method, never a lambda closing over `self`: a lambda in this
        # connection is a reference cycle holding a QWidget, and the widget is
        # then destroyed at interpreter shutdown — after the QApplication has
        # gone — corrupting the heap on the way out.
        self.tree.itemSelectionChanged.connect(self._row_selected)

        #: One line an operator can read from across the room. The count of dark
        #: cameras belongs here because it is the number nobody goes looking for.
        self.summary = QLabel("")
        self.summary.setObjectName("Caption")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(title)
        layout.addWidget(self.tree)
        layout.addWidget(self.summary)
        self.show_cameras((), {})

    # ------------------------------------------------------------------- rows

    def show_cameras(self, records, health: dict | None = None) -> None:
        """Replace every row, keeping the operator's selection by camera id.

        ``records`` are the node's camera records — anything with ``camera_id``,
        ``pose`` and either ``display_source`` or ``source``. ``health`` maps a
        camera id to a health object as described in `HEALTH_FACTS`; a camera
        missing from it has never run, which is shown as "not started" rather
        than guessed at.

        Rebuilt rather than diffed, like the incident panel: at the handful of
        cameras a site has this costs nothing, and a reused row can keep a stale
        value in a column that failed to update — a strip still reading "live"
        for a camera that went dark is the one defect this panel must not have.
        """
        health = health or {}
        self._quiet = True
        try:
            self.tree.clear()
            counts: dict[str, int] = {}
            unplaced = 0
            for record in records:
                camera_id = str(getattr(record, "camera_id", ""))
                facts = health.get(camera_id)
                state = camera_state(facts)
                counts[state] = counts.get(state, 0) + 1
                placed = getattr(record, "pose", None) is not None
                if not placed:
                    unplaced += 1
                item = QTreeWidgetItem([
                    camera_id,
                    display_source(record),
                    "placed" if placed else "not placed",
                    f"{STATE_GLYPHS[state]}  {status_text(facts)}",
                ])
                item.setData(CAMERA_COLUMN, Qt.ItemDataRole.UserRole, camera_id)
                colour = STATE_COLOURS[state]
                item.setForeground(STATUS_COLUMN, QBrush(colour))
                item.setToolTip(STATUS_COLUMN, health_tooltip(camera_id, facts))
                item.setToolTip(
                    SOURCE_COLUMN, "Shown redacted. A credential is never displayed."
                )
                if not placed:
                    item.setForeground(PLACED_COLUMN, QBrush(theme.TEXT_FAINT))
                    item.setToolTip(
                        PLACED_COLUMN,
                        "Not placed: nothing this camera sees can be given a "
                        "position, and it covers no zone.",
                    )
                if state in (STATE_DARK, STATE_FAULT):
                    # The id too, and in bold: an operator scanning the left-hand
                    # column must not have to read the far side of the row to
                    # find out which camera is not working.
                    item.setForeground(CAMERA_COLUMN, QBrush(colour))
                    font = item.font(CAMERA_COLUMN)
                    font.setWeight(QFont.Weight.Bold)
                    item.setFont(CAMERA_COLUMN, font)
                    item.setFont(STATUS_COLUMN, font)
                self.tree.addTopLevelItem(item)
                if camera_id == self._selected_id:
                    item.setSelected(True)
                    self.tree.setCurrentItem(item)
            self.summary.setText(self._summarise(counts, unplaced))
        finally:
            self._quiet = False

    def _summarise(self, counts: dict[str, int], unplaced: int) -> str:
        """The one line under the list. Dark and failed cameras are named in it
        because a list long enough to scroll can hide the row that matters."""
        total = sum(counts.values())
        if not total:
            return "No cameras. Add one to begin."
        parts = [f"{total} camera{'s' if total != 1 else ''}"]
        dark = counts.get(STATE_DARK, 0)
        if dark:
            parts.append(f"{dark} dark")
        failed = counts.get(STATE_FAULT, 0)
        if failed:
            parts.append(f"{failed} failed")
        if unplaced:
            parts.append(f"{unplaced} not placed")
        return " · ".join(parts)

    # -------------------------------------------------------------- selection

    def _row_selected(self) -> None:
        """Qt's selection changed. Tell the bus, unless we caused it ourselves.

        Nothing may raise out of here: a traceback escaping a Qt slot is retained
        by the interpreter and pins the widget it came from past the
        QApplication's own destruction.
        """
        if self._quiet:
            return
        try:
            camera_id = self.selected_camera_id()
            self._selected_id = camera_id
            self.selected.emit(
                None if camera_id is None else Selection.camera(camera_id)
            )
        except Exception:  # pragma: no cover - defensive, see the docstring
            pass

    def selected_camera_id(self) -> str | None:
        """Which camera the placement, start and remove actions apply to.

        The combo box's ``currentData()`` in the shape the rest of the console
        already expects, so the toolbar keeps working once the picker is gone.
        """
        items = self.tree.selectedItems()
        if not items:
            return None
        return items[0].data(CAMERA_COLUMN, Qt.ItemDataRole.UserRole)

    def set_selection(self, selection) -> None:
        """Highlight what was selected elsewhere, emitting nothing.

        Silent by construction rather than by luck: the console's selection bus
        calls this from its own ``changed`` signal, so a re-emit here would go
        straight back into the bus. A selection of any other kind — a zone, a
        track on some camera, an incident — clears this list rather than guessing
        which camera it implies, because a highlighted row is a claim that the
        operator picked that camera.
        """
        camera_id = None
        if selection is not None and getattr(selection, "kind", None) == CAMERA_KIND:
            camera_id = getattr(selection, "camera_id", None)
        self._selected_id = camera_id
        self._quiet = True
        try:
            if camera_id is None:
                self.tree.clearSelection()
                self.tree.setCurrentItem(None)
                return
            for index in range(self.tree.topLevelItemCount()):
                item = self.tree.topLevelItem(index)
                if item.data(CAMERA_COLUMN, Qt.ItemDataRole.UserRole) == camera_id:
                    self.tree.setCurrentItem(item)
                    item.setSelected(True)
                    self.tree.scrollToItem(item)
                    return
            self.tree.clearSelection()
        finally:
            self._quiet = False
