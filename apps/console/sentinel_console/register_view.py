"""The People and Vehicles register, from the console.

The engine has held a register since `sentinel.registry` landed: who was
deliberately named, on what basis, by whom, and every track they were later
recognised on. Nothing in the interface has ever shown a row of it. An operator
who enrolled a member of staff through the command line had no screen on which
to see that it happened, no way to read where that person had been since, and —
the half that matters — no button that forgets them. A register nobody can read
is a register nobody can review, and a register nobody can delete from is not
one this product is allowed to hold.

One panel serves both kinds, constructed once per kind, because the safeguards
are the same safeguards. A face and a plate carry different weights of claim,
but "who enrolled this, under what basis, and how do I take it back" is the
same question for each, and two panels would answer it two ways. The differences
are the noun in the sentences and the dialog's plate field.

Four things this panel is careful about, each of them a way a screen over an
identity register quietly becomes indefensible:

**The capability is stated where the enrolment would happen.** The top line is
the node's own `identity_status` — "off", "faces: on (stand-in)", "faces: on
but no models in … — expecting yunet.onnx and sface.onnx". An operator who
switched faces on and installed nothing must read that here, beside the enrol
button that will refuse, not in a log file after an afternoon of wondering why
the button never enables. When identity is off, nothing runs and the button says
so; the panel does not hide itself, because a hidden panel is indistinguishable
from a feature that does not exist.

**Enrolment is an act on a track, never on a frame region the panel chose.**
The button is "Enrol from selected track…" and it enables only when the
console's selection bus has handed this panel a track — and, for a person, only
when the node has face templates for that live track. The panel never touches an
image. What reaches the register is what the node computed inside that track's
own box, and the node refuses when there is nothing.

**Forgetting names what it will delete, and reports what it did.** The
confirmation says the subject's name and the counts — identifiers, sightings —
so an irreversible act is reviewable at the moment it is taken; and the status
line afterwards carries the `Forgotten` counts the register returned, not the
counts the panel expected. The audit row the node writes carries ids and counts
only; the name is shown on this screen and nowhere that outlives the erasure.

**Nothing raises out of a slot.** A refused enrolment (blank name, no templates,
a plate already somebody else's), a refused pin, a node that has gone away — all
of them reach the status line as a sentence. A traceback escaping a Qt slot is
retained by the interpreter, which pins this widget past the QApplication's own
destruction and corrupts the heap at exit; the console found that out the hard
way and the rule is now the same in every panel.

The three buttons sit behind the console's Configure lock (`set_editable`), like
moving a camera or reshaping a zone, and for the same reason: a console left in
a control room is left in whatever state the last person walked away from, and
a sleeve across a touchscreen must not be able to enrol or erase anybody. The
list and the history stay readable in Monitor; reading is not the act the lock
exists to protect.

Times in the history are shown in the site's own clock when the node has one,
and say which clock on screen. A movement history read against the wrong zone
puts somebody at the gate an hour before they were there.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Protocol

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from sentinel.registry import (
    Confidence,
    Forgotten,
    Identifier,
    IdentifierKind,
    Register,
    RegistryError,
    Sighting,
    Subject,
    SubjectKind,
)

from . import theme
from .selection import TRACK as TRACK_KIND, Selection

if TYPE_CHECKING:  # pragma: no cover - typing only; nothing here is imported at runtime
    from sentinel.faces import FaceTemplate, TrackIdentity
    from sentinel.site import Identity, Site


class IdentityActions(Protocol):
    """What the panel needs from the node, and nothing more.

    A protocol rather than `sentinel.node.Node` itself so the panel can be
    tested against the real register with a small stand-in for the node, and
    so the console can hand it a node that has not been built yet without the
    panel reaching into one. Every method here mirrors the node's identity
    surface exactly; a method added to one and not the other is a wire that
    looks connected and is not.

    Reads go through `register` — the store's own `Register`, never a second
    one over a second connection, for the reasons `Store.register` gives.
    Writes go through the node, because the node is what writes the audit row
    beside each of them; a panel that wrote to the register directly would
    produce enrolments and erasures the audit log never heard about.
    """

    @property
    def register(self) -> Register: ...

    @property
    def identity_status(self) -> str: ...

    def site(self) -> Site: ...

    def set_identity(self, identity: Identity, *, reason: str) -> Site: ...

    def identity_of(self, camera_id: str, track_id: int) -> TrackIdentity | None: ...

    def templates_for(self, camera_id: str, track_id: int) -> tuple[FaceTemplate, ...]: ...

    def enrol_person(
        self, name: str, camera_id: str, track_id: int, *, basis: str, notes: str | None = None
    ) -> str: ...

    def enrol_vehicle(
        self,
        name: str,
        plate: str,
        *,
        basis: str,
        notes: str | None = None,
        camera_id: str | None = None,
        track_id: int | None = None,
    ) -> str: ...

    def forget_subject(self, subject_id: str) -> Forgotten: ...

    def pin_subject(self, subject_id: str, pinned: bool) -> None: ...


#: Columns of the subjects table. Named, because a panel that put the sighting
#: count in the pinned column would still pass a test that indexed by number.
NAME_COLUMN = 0
IDENTIFIERS_COLUMN = 1
LAST_SEEN_COLUMN = 2
SIGHTINGS_COLUMN = 3
PINNED_COLUMN = 4

#: Columns of the history list.
HISTORY_CAMERA_COLUMN = 0
HISTORY_TRACK_COLUMN = 1
HISTORY_FIRST_COLUMN = 2
HISTORY_LAST_COLUMN = 3
HISTORY_CONFIDENCE_COLUMN = 4
HISTORY_SCORE_COLUMN = 5

#: Shown where there is nothing: a subject never sighted, a declared sighting's
#: score. Not an empty cell — a blank column reads as a value that failed to
#: load, and "never seen" and "no score, because a person said so" are facts.
NOTHING = "—"

#: The lawful bases the dialog offers. Free text is allowed beside them because
#: the basis is the site's own list and this panel does not know a
#: jurisdiction's vocabulary; the register refuses a blank one and so does the
#: dialog, because a template held under no stated basis is one that has to be
#: deleted rather than defended.
LAWFUL_BASES = (
    "consent",
    "employment",
    "contract",
    "legal obligation",
    "legitimate interest",
)

#: The nouns each kind of panel speaks in. A line reading "no person is
#: enrolled" on the vehicles tab would be read as the feature being broken.
_NOUNS: dict[SubjectKind, tuple[str, str]] = {
    SubjectKind.PERSON: ("person", "people"),
    SubjectKind.VEHICLE: ("vehicle", "vehicles"),
}

#: How each confidence is drawn. POSSIBLE is in the warning colour because it
#: is the hedged claim `sentinel.faces` insists on drawing differently — a
#: history that showed a possible match in the same colour as a match would let
#: the hedge be read past. DECLARED is muted: an operator said so and there is
#: no score behind it, which is a different kind of fact rather than a weaker
#: one.
_CONFIDENCE_COLOUR = {
    Confidence.MATCH: theme.TEXT,
    Confidence.POSSIBLE: theme.WARNING,
    Confidence.DECLARED: theme.TEXT_MUTED,
}


def _when(millis: int, clock) -> str:
    """A stored instant in the given clock, to the second."""
    return datetime.fromtimestamp(int(millis) / 1000, clock).strftime("%Y-%m-%d %H:%M:%S")


def _describe_identifiers(kind: SubjectKind, identifiers: tuple[Identifier, ...]) -> str:
    """The Identifiers cell: plates are shown, templates are counted.

    A plate is what the operator typed or the recogniser read, and showing it
    is how they check the right van was enrolled. A face template is 128
    floats that cannot be rendered — that is the whole reason storing one is
    defensible — so all that can honestly be shown is how many there are.
    """
    if not identifiers:
        return "none (swept or never enrolled)"
    if kind is SubjectKind.VEHICLE:
        plates = [i.raw_text or i.plate or "?" for i in identifiers if i.kind is IdentifierKind.PLATE]
        return ", ".join(plates) if plates else NOTHING
    return f"{len(identifiers)} face template(s)"


class EnrolDialog(QDialog):
    """Name a track: the one place an identifier is deliberately attached.

    Asks for the three things the register demands before it will store
    anything — a name, a lawful basis and (for a vehicle) the plate — plus
    optional notes. It refuses a blank name and a blank basis at the door
    rather than passing them to the node to be refused there: the refusal
    reads the same, but here the operator's typing survives it.

    ``templates_count`` is shown for a person, because "enrol from 1 template"
    and "enrol from 12 templates" are different acts and the operator should
    know which they are making. ``plate_hint`` pre-fills the plate field with
    what a recogniser read, if the caller has one; it is a hint, not a fact,
    and the field stays editable because a read with a confusable character is
    exactly the kind of thing a person corrects here.
    """

    def __init__(
        self,
        kind: SubjectKind,
        *,
        templates_count: int | None = None,
        plate_hint: str | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._kind = kind
        noun, _ = _NOUNS[kind]
        self.setWindowTitle(f"Enrol a {noun} from the selected track")
        self.setMinimumWidth(460)

        form = QFormLayout()

        self.name_field = QLineEdit()
        self.name_field.setPlaceholderText(
            "A name the register can be reviewed by"
            if kind is SubjectKind.PERSON
            else "Contractor van, site manager's car…"
        )
        form.addRow("Name", self.name_field)

        self.plate_field: QLineEdit | None = None
        if kind is SubjectKind.VEHICLE:
            self.plate_field = QLineEdit(plate_hint or "")
            self.plate_field.setPlaceholderText("As printed on the plate")
            self.plate_field.setToolTip(
                "Stored normalised — spacing and punctuation are ignored — with "
                "the characters as typed kept beside it. A read with an "
                "unresolved character (?) is refused rather than completed."
            )
            form.addRow("Plate", self.plate_field)

        self.basis_box = QComboBox()
        self.basis_box.setEditable(True)
        self.basis_box.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        for basis in LAWFUL_BASES:
            self.basis_box.addItem(basis)
        self.basis_box.setToolTip(
            "Under what basis this identifier is held. Pick one, or type the "
            "site's own wording. It is recorded with the enrolment and shown "
            "to anybody who asks why this entry exists."
        )
        form.addRow("Lawful basis", self.basis_box)

        self.notes_field = QLineEdit()
        self.notes_field.setPlaceholderText("Optional — role, contractor, expiry of access…")
        form.addRow("Notes", self.notes_field)

        caption = QLabel("")
        caption.setObjectName("Caption")
        caption.setWordWrap(True)
        if kind is SubjectKind.PERSON:
            count = templates_count or 0
            caption.setText(
                f"The best of {count} face template(s) the node computed inside "
                "this track's own box will be stored — a vector, not an image. "
                "Nothing is enrolled by being seen; this dialog is the act."
            )
        else:
            caption.setText(
                "The plate is stored with who enrolled it, when and under what "
                "basis. The selected track is recorded as where it came from."
            )
        form.addRow("", caption)

        #: Why the dialog refused, shown on the dialog rather than raised.
        self.refusal = QLabel("")
        self.refusal.setWordWrap(True)
        self.refusal.setStyleSheet(f"color: {theme.WARNING.name()};")
        form.addRow("", self.refusal)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Enrol")
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self._buttons)

    # ------------------------------------------------------------------ result

    def name(self) -> str:
        return self.name_field.text().strip()

    def plate(self) -> str | None:
        """The plate as typed, or ``None`` for a person."""
        return None if self.plate_field is None else self.plate_field.text().strip()

    def basis(self) -> str:
        return self.basis_box.currentText().strip()

    def notes(self) -> str | None:
        """The notes, or ``None`` when the field was left empty."""
        text = self.notes_field.text().strip()
        return text or None

    def why_refused(self) -> str | None:
        """Why the dialog will not accept as it stands, or ``None``."""
        if not self.name():
            return "A name is required: an entry nobody can recognise cannot be reviewed or forgotten on purpose."
        if not self.basis():
            return "A lawful basis is required: an identifier held under none is one that has to be deleted."
        if self._kind is SubjectKind.VEHICLE and not self.plate():
            return "A plate is required: a vehicle with no registration is not something the register can match."
        return None

    def accept(self) -> None:  # type: ignore[override]
        """Accept only when the register would. The refusal stays on screen.

        Overridden rather than gating the OK button alone, because the button
        is not the only way in — Enter in a line edit triggers the default
        button — and a dialog that could be accepted blank would hand the node
        a name it will refuse, throwing the operator's basis and notes away
        with the refusal.
        """
        refusal = self.why_refused()
        if refusal is not None:
            self.refusal.setText(refusal)
            return
        super().accept()


class RegisterPanel(QWidget):
    """Who is enrolled, of one kind; where they have been; and the way out.

    Built once for people and once for vehicles. Fed the node through
    `set_actions` — the console's node, never a register of its own — and told
    the current selection through `set_selection`, which the console's
    selection bus calls and which is therefore deliberately silent.

    Clicking a sighting emits `selected` with the track it names, keyed by
    camera as well as id, so the wall, the map and the track table can point
    at where the subject was. Clicking a subject fills the history and emits
    nothing: a subject is not a thing on the ground.
    """

    #: A `Selection.track` for the sighting the operator clicked. The console's
    #: selection bus is on the other end of this.
    selected = Signal(object)

    def __init__(self, kind: SubjectKind, parent: QWidget | None = None):
        super().__init__(parent)
        self._kind = SubjectKind(kind)
        self._actions: IdentityActions | None = None
        #: Off by default, and off whenever the console is in Monitor — the
        #: same default as dragging a camera on the map, for the same reason.
        self._editable = False
        #: The track the console's selection bus last handed over, if it was
        #: a track. Only a track can be enrolled from.
        self._selection: Selection | None = None
        self._plate_hint: str | None = None
        #: The subjects on screen, by id, as read at the last refresh.
        self._subjects: dict[str, Subject] = {}
        #: Nested reasons to ignore Qt's selection signals — the panel moving
        #: its own highlight. A depth count rather than a flag, for the reason
        #: the investigation panel gives: a rebuild inside a highlight inside
        #: a refresh is one edit away, and a flag would unmute the outer block
        #: half way through.
        self._quiet = 0
        #: The last exception a slot swallowed, kept for tests and bug reports.
        self.last_failure: Exception | None = None

        noun, plural = _NOUNS[self._kind]

        #: The node's own account of whether identity runs, refreshed with the
        #: list. Sits at the top, beside the enrol button, because that is the
        #: button whose refusal it explains.
        self.capability = QLabel("")
        self.capability.setObjectName("Caption")
        self.capability.setWordWrap(True)

        self.subjects = QTreeWidget()
        self.subjects.setColumnCount(5)
        self.subjects.setHeaderLabels(["Name", "Identifiers", "Last seen", "Sightings", "Pinned"])
        self.subjects.setRootIsDecorated(False)
        self.subjects.setAlternatingRowColors(True)
        self.subjects.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        header = self.subjects.header()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        for index, width in enumerate((180, 170, 150, 70)):
            self.subjects.setColumnWidth(index, width)
        # Bound methods, never lambdas closing over `self`: a lambda in a
        # signal connection is a reference cycle holding a QWidget, which is
        # then destroyed at interpreter shutdown after PySide has torn the
        # QApplication down, and that corrupts the heap on the way out.
        self.subjects.itemSelectionChanged.connect(self._subject_selected)

        self.enrol_button = QPushButton("Enrol from selected track…")
        self.enrol_button.clicked.connect(self._enrol_clicked)
        self.forget_button = QPushButton("Forget…")
        self.forget_button.clicked.connect(self._forget_clicked)
        self.pin_button = QPushButton("Pin")
        self.pin_button.clicked.connect(self._pin_clicked)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setToolTip("Read the register again.")
        self.refresh_button.clicked.connect(self._refresh_clicked)

        #: What the last act did, or why it was refused. Left alone by a
        #: refresh, so the sentence about an erasure survives the list being
        #: rebuilt underneath it.
        self.status = QLabel("")
        self.status.setObjectName("Caption")
        self.status.setWordWrap(True)

        self.history_title = QLabel("")
        self.history_title.setObjectName("PanelTitle")

        self.history = QTreeWidget()
        self.history.setColumnCount(6)
        self.history.setHeaderLabels(
            ["Camera", "Track", "First seen", "Last seen", "Confidence", "Score"]
        )
        self.history.setRootIsDecorated(False)
        self.history.setAlternatingRowColors(True)
        self.history.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        history_header = self.history.header()
        history_header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        history_header.setStretchLastSection(True)
        for index, width in enumerate((110, 60, 150, 150, 100)):
            self.history.setColumnWidth(index, width)
        self.history.itemSelectionChanged.connect(self._sighting_selected)

        #: Which clock the history is read in, stated on screen.
        self.clock_caption = QLabel("")
        self.clock_caption.setObjectName("Caption")
        self.clock_caption.setWordWrap(True)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(6, 0, 6, 0)
        buttons.addWidget(self.enrol_button)
        buttons.addWidget(self.forget_button)
        buttons.addWidget(self.pin_button)
        buttons.addStretch(1)
        buttons.addWidget(self.refresh_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.capability)
        layout.addWidget(self.subjects, 2)
        layout.addLayout(buttons)
        layout.addWidget(self.status)
        layout.addWidget(self.history_title)
        layout.addWidget(self.history, 1)
        layout.addWidget(self.clock_caption)

        self.setToolTip(
            f"The {plural} somebody deliberately enrolled. Nothing is enrolled by "
            f"being seen; a {noun} is named here, on purpose, from a track."
        )
        self.refresh()

    # --------------------------------------------------------------- silence

    @contextmanager
    def _silent(self):
        """Ignore Qt's selection signals for the duration, however nested."""
        self._quiet += 1
        try:
            yield
        finally:
            self._quiet -= 1

    # ---------------------------------------------------------------- the node

    @property
    def kind(self) -> SubjectKind:
        return self._kind

    def set_actions(self, actions: IdentityActions | None) -> None:
        """Point the panel at the node and read its register.

        The console owns no database. Until the node exists the panel says it
        has nothing to read rather than showing an empty list, which is
        indistinguishable from a site on which nobody has ever been enrolled.
        The status line is cleared: it was about another node's register.
        """
        self._actions = actions
        self.status.setText("")
        self.refresh()

    def set_editable(self, editable: bool) -> None:
        """Allow, or forbid, enrolling, forgetting and pinning.

        Off by default, and off whenever the console is in Monitor: naming
        somebody and erasing somebody are the two acts on this screen that a
        sleeve across a touchscreen must not be able to perform. Reading is
        not locked — the list and the history stay readable — because the lock
        exists to protect the register from being changed, not from being
        reviewed, and a review is what makes holding it defensible.
        """
        self._editable = bool(editable)
        self._update_buttons()

    def set_selection(self, selection) -> None:
        """Note what was selected elsewhere, emitting nothing.

        Called from the console's selection bus, so a re-emit here would go
        straight back into the bus and the window would stop answering. Only a
        track is remembered: a camera, a zone or an incident is not a thing an
        identifier can be taken from, and selecting one after a track means the
        track is no longer what the operator has picked, so the enrol button
        disables.

        A sighting of the selected track already on screen is highlighted,
        silently, so the operator clicking a disc on the map sees which row of
        the history it is.
        """
        kind = getattr(selection, "kind", None) if selection is not None else None
        self._selection = selection if kind == TRACK_KIND else None
        self._update_buttons()
        with self._silent():
            self.history.clearSelection()
            self.history.setCurrentItem(None)
            if self._selection is None:
                return
            for index in range(self.history.topLevelItemCount()):
                item = self.history.topLevelItem(index)
                if item.data(HISTORY_CAMERA_COLUMN, Qt.ItemDataRole.UserRole) == self._selection:
                    self.history.setCurrentItem(item)
                    item.setSelected(True)
                    self.history.scrollToItem(item)
                    return

    def set_plate_hint(self, text: str | None) -> None:
        """What a recogniser read for the selected track, to pre-fill the dialog.

        A hint, never an enrolment: it goes into an editable field for the
        operator to confirm or correct. Ignored by a people panel.
        """
        self._plate_hint = (text or "").strip() or None

    # ---------------------------------------------------------------- reading

    def refresh(self) -> None:
        """Read the register and show it, keeping the selected subject.

        Every subject costs two further reads — its identifiers and its
        history — which is a query per row rather than a join. The register is
        a site's staff list, tens of rows, not its event log; the day it is
        thousands the honest fix is a summary query in the register, not a
        cache here that would show yesterday's sighting count beside today's
        name.

        Nothing raises out of here: it is reached from a Qt slot, and the read
        and the draw are both inside the guard. A failure clears the list —
        rows from a page that was never finished read as the whole register.
        """
        wanted = self.selected_subject_id()
        if self._actions is None:
            self._subjects = {}
            with self._silent():
                self.subjects.clear()
                self.history.clear()
            self.capability.setText("No node is connected, so the register cannot be read.")
            self.history_title.setText("")
            self.clock_caption.setText("")
            self._update_buttons()
            return

        self.capability.setText(self._capability_line())
        clock, clock_label = self._clock()
        self.clock_caption.setText(clock_label)
        try:
            register = self._actions.register
            subjects = register.subjects(kind=self._kind)
            rows = [
                (subject, register.identifiers(subject.id), register.history(subject.id))
                for subject in subjects
            ]
            self._subjects = {subject.id: subject for subject in subjects}
            with self._silent():
                self.subjects.clear()
                for subject, identifiers, history in rows:
                    item = self._subject_row(subject, identifiers, history, clock)
                    self.subjects.addTopLevelItem(item)
                    if subject.id == wanted:
                        self.subjects.setCurrentItem(item)
                        item.setSelected(True)
            self._fill_history()
        except Exception as failure:
            self.last_failure = failure
            self._subjects = {}
            with self._silent():
                self.subjects.clear()
                self.history.clear()
            self.status.setText(f"The register could not be read: {failure}")
        self._update_buttons()

    def _refresh_clicked(self, _checked: bool = False) -> None:
        """Swallows ``clicked(bool)`` so `refresh` never receives it as an argument."""
        self.refresh()

    def _capability_line(self) -> str:
        """The node's own identity status, or why it could not be asked."""
        assert self._actions is not None
        try:
            status = str(self._actions.identity_status)
        except Exception as failure:  # a node mid-shutdown, say
            self.last_failure = failure
            return f"The node could not say whether identity is on: {failure}"
        _noun, plural = _NOUNS[self._kind]
        return f"Identity: {status}. Listing enrolled {plural}."

    def _clock(self):
        """The site's clock and a caption naming it, or UTC and the reason.

        The site's IANA zone through `Site.clock`, which raises rather than
        falling back when the zone cannot be resolved on this machine. That
        refusal is honoured here: the caption says the site's clock could not
        be read and that the times are UTC, rather than printing UTC under a
        heading that promises local time.
        """
        assert self._actions is not None
        try:
            site = self._actions.site()
            clock = site.clock()
        except Exception as failure:
            self.last_failure = failure
            return timezone.utc, (
                f"Times are UTC: the site's clock could not be read ({failure})."
            )
        name = getattr(site, "timezone", "UTC")
        if name in ("UTC", "Etc/UTC"):
            return clock, "Times are UTC — the clock the record is kept on."
        return clock, f"Times are in the site's clock, {name}."

    def _subject_row(
        self,
        subject: Subject,
        identifiers: tuple[Identifier, ...],
        history: tuple[Sighting, ...],
        clock,
    ) -> QTreeWidgetItem:
        last_seen = max((s.last_seen_millis for s in history), default=None)
        item = QTreeWidgetItem([
            subject.display_name,
            _describe_identifiers(self._kind, identifiers),
            NOTHING if last_seen is None else _when(last_seen, clock),
            str(len(history)),
            "pinned" if subject.pinned else NOTHING,
        ])
        item.setData(NAME_COLUMN, Qt.ItemDataRole.UserRole, subject.id)
        item.setToolTip(
            NAME_COLUMN,
            f"{subject.id}" + (f" — {subject.notes}" if subject.notes else ""),
        )
        if identifiers:
            provenance = "; ".join(
                f"{i.kind.value.lower().replace('_', ' ')} enrolled by {i.enrolled_by} "
                f"on {_when(i.enrolled_at_millis, clock)}, basis: {i.basis}"
                + (f", from {i.source_camera} #{i.source_track}" if i.source_camera else "")
                for i in identifiers
            )
            item.setToolTip(IDENTIFIERS_COLUMN, provenance)
        else:
            item.setForeground(IDENTIFIERS_COLUMN, QBrush(theme.TEXT_MUTED))
        if subject.pinned:
            item.setToolTip(
                PINNED_COLUMN,
                "Exempt from the retention sweep until unpinned.",
            )
        return item

    # ------------------------------------------------------------- the history

    def _subject_selected(self) -> None:
        """Qt's subject selection changed. Fill the history; emit nothing."""
        if self._quiet:
            return
        try:
            self._fill_history()
        except Exception as failure:
            self.last_failure = failure
            self.status.setText(f"That {_NOUNS[self._kind][0]}'s history could not be shown: {failure}")
        self._update_buttons()

    def _fill_history(self) -> None:
        """Replace the history with the selected subject's, oldest first.

        Rebuilt rather than diffed, like every list over the store: a reused
        row keeping a stale confidence would show the certain form of a name
        the register has since hedged.
        """
        subject = self._selected_subject()
        noun, _ = _NOUNS[self._kind]
        with self._silent():
            self.history.clear()
            if subject is None or self._actions is None:
                self.history_title.setText(
                    f"MOVEMENT HISTORY — select a {noun} to see where they were recognised"
                )
                return
            clock, _label = self._clock()
            sightings = self._actions.register.history(subject.id)
            for sighting in sightings:
                self.history.addTopLevelItem(self._history_row(sighting, clock))
            count = len(sightings)
            self.history_title.setText(
                f"MOVEMENT HISTORY — {subject.display_name}: {count} sighting(s)"
                if count
                else f"MOVEMENT HISTORY — {subject.display_name}: never recognised on any track"
            )
            # The bus's track, if it is one of these rows, is highlighted
            # exactly as `set_selection` would have — silently.
            if self._selection is not None:
                for index in range(self.history.topLevelItemCount()):
                    item = self.history.topLevelItem(index)
                    if item.data(HISTORY_CAMERA_COLUMN, Qt.ItemDataRole.UserRole) == self._selection:
                        self.history.setCurrentItem(item)
                        item.setSelected(True)
                        break

    def _history_row(self, sighting: Sighting, clock) -> QTreeWidgetItem:
        confidence = sighting.confidence
        if confidence is Confidence.DECLARED:
            word, score = "declared", NOTHING
        else:
            word = "match" if confidence is Confidence.MATCH else "possible match"
            score = NOTHING if sighting.score is None else f"{sighting.score:.2f}"
        item = QTreeWidgetItem([
            sighting.camera_id,
            f"#{sighting.track_id}",
            _when(sighting.first_seen_millis, clock),
            _when(sighting.last_seen_millis, clock),
            word,
            score,
        ])
        # A track, keyed by camera as well as by number: #3 on the gate and #3
        # on the yard are different people, and the selection bus keys on both.
        item.setData(
            HISTORY_CAMERA_COLUMN,
            Qt.ItemDataRole.UserRole,
            Selection.track(sighting.camera_id, sighting.track_id),
        )
        item.setForeground(
            HISTORY_CONFIDENCE_COLUMN, QBrush(_CONFIDENCE_COLOUR.get(confidence, theme.TEXT))
        )
        item.setToolTip(
            HISTORY_CONFIDENCE_COLUMN,
            "An operator said so; there is no score behind a declaration."
            if confidence is Confidence.DECLARED
            else (
                f"{confidence.value.lower()} at {sighting.score:.2f}"
                + (
                    f", on enrolment {sighting.identifier_id}"
                    if sighting.identifier_id
                    else ", the enrolment behind it has since been removed"
                )
            ),
        )
        item.setToolTip(
            HISTORY_LAST_COLUMN, f"{sighting.duration_millis / 1000:.0f} s on this track"
        )
        return item

    def _sighting_selected(self) -> None:
        """Qt's history selection changed. Tell the bus, unless we caused it.

        Nothing may raise out of here. And caught is not hidden: a selection
        that never reaches the bus leaves the operator clicking rows while the
        map and the wall sit on somebody else's choice, so the failure goes on
        the status line and is kept where a test can read it.
        """
        if self._quiet:
            return
        try:
            selection = self.selected_sighting()
            if selection is None:
                return
            self.selected.emit(selection)
        except Exception as failure:
            self.last_failure = failure
            self.status.setText(f"That sighting could not be selected: {failure}")

    def selected_sighting(self) -> Selection | None:
        """The track the highlighted sighting names, or ``None``."""
        items = self.history.selectedItems()
        if not items:
            return None
        return items[0].data(HISTORY_CAMERA_COLUMN, Qt.ItemDataRole.UserRole)

    # ------------------------------------------------------------- the subject

    def selected_subject_id(self) -> str | None:
        items = self.subjects.selectedItems()
        if not items:
            return None
        return items[0].data(NAME_COLUMN, Qt.ItemDataRole.UserRole)

    def _selected_subject(self) -> Subject | None:
        subject_id = self.selected_subject_id()
        return None if subject_id is None else self._subjects.get(subject_id)

    def select_subject(self, subject_id: str | None) -> bool:
        """Highlight a subject and show their history. True when it is listed.

        Silent on the bus — a subject is not a thing on the ground — but the
        history is filled, because that is what selecting a subject is for.
        """
        with self._silent():
            self.subjects.clearSelection()
            self.subjects.setCurrentItem(None)
            found = False
            if subject_id is not None:
                for index in range(self.subjects.topLevelItemCount()):
                    item = self.subjects.topLevelItem(index)
                    if item.data(NAME_COLUMN, Qt.ItemDataRole.UserRole) == subject_id:
                        self.subjects.setCurrentItem(item)
                        item.setSelected(True)
                        self.subjects.scrollToItem(item)
                        found = True
                        break
        try:
            self._fill_history()
        except Exception as failure:
            self.last_failure = failure
            self.status.setText(f"That {_NOUNS[self._kind][0]}'s history could not be shown: {failure}")
        self._update_buttons()
        return found

    # ------------------------------------------------------------- the buttons

    def _template_count(self) -> int:
        """How many face templates the node holds for the selected track.

        Zero when there is no node, no track, or the node cannot be asked —
        and in the last case the reason goes on the status line, because a
        button that stays disabled for a reason nobody can read teaches the
        operator the feature is broken.
        """
        if self._actions is None or self._selection is None:
            return 0
        try:
            return len(
                self._actions.templates_for(self._selection.camera_id, self._selection.track_id)
            )
        except Exception as failure:
            self.last_failure = failure
            self.status.setText(
                f"The node could not be asked for that track's face templates: {failure}"
            )
            return 0

    def _update_buttons(self) -> None:
        """Enable what can be done now, and say why the rest cannot.

        The reason a button is disabled is its tooltip. A disabled "Enrol"
        with no explanation is the same picture whether Configure is locked,
        nothing is selected, faces are off, or the person has not faced a
        camera yet, and those need four different things done about them.
        """
        noun, _ = _NOUNS[self._kind]
        subject = self._selected_subject()

        reason: str | None = None
        if self._actions is None:
            reason = "No node is connected."
        elif not self._editable:
            reason = "Configure is locked. Press Configure to enrol, forget or pin."
        elif self._selection is None:
            reason = "Select a track first — on the wall, the map or the track table."
        elif self._kind is SubjectKind.PERSON and self._template_count() == 0:
            reason = (
                "That track has no face templates yet. Faces may be off, the "
                "models may be missing, or the person has not faced a camera "
                "clearly enough — the line at the top says which."
            )
        self.enrol_button.setEnabled(reason is None)
        self.enrol_button.setToolTip(
            reason
            or f"Name the selected track ({self._selection.describe()}) as a {noun}. "
            "Asks for a name and a lawful basis; nothing is stored until both are given."
        )

        can_change = self._actions is not None and self._editable and subject is not None
        if self._actions is None:
            why_not = "No node is connected."
        elif not self._editable:
            why_not = "Configure is locked. Press Configure to enrol, forget or pin."
        elif subject is None:
            why_not = f"Select a {noun} in the list first."
        else:
            why_not = ""
        self.forget_button.setEnabled(can_change)
        self.forget_button.setToolTip(
            why_not
            or f"Delete every identifier and sighting held for {subject.display_name!r}. "
            "Asks first, and says what it will delete."
        )
        self.pin_button.setEnabled(can_change)
        self.pin_button.setText("Unpin" if subject is not None and subject.pinned else "Pin")
        self.pin_button.setToolTip(
            why_not
            or (
                "Put this entry back under the retention sweep."
                if subject.pinned
                else "Exempt this entry from the retention sweep. A pin is an explicit, "
                "audited act; it is the only thing that keeps an identifier past its retention."
            )
        )

    # --------------------------------------------------------------- enrolling

    def _enrol_clicked(self, _checked: bool = False) -> None:
        """The button: ask, then enrol. Nothing raises out of here."""
        try:
            self._enrol_via_dialog()
        except Exception as failure:
            self.last_failure = failure
            self.status.setText(f"Enrolment failed: {failure}")

    def _enrol_via_dialog(self) -> None:
        if self._actions is None or self._selection is None:
            self.status.setText("Select a track first; there is nothing to enrol from.")
            return
        templates = self._template_count() if self._kind is SubjectKind.PERSON else None
        dialog = EnrolDialog(
            self._kind,
            templates_count=templates,
            plate_hint=self._plate_hint if self._kind is SubjectKind.VEHICLE else None,
            parent=self,
        )
        try:
            if dialog.exec() != QDialog.DialogCode.Accepted:
                self.status.setText("Enrolment cancelled; nothing was stored.")
                return
            name, basis, notes, plate = dialog.name(), dialog.basis(), dialog.notes(), dialog.plate()
        finally:
            # Owned by this panel for modality; released now rather than when
            # the panel dies, or every enrolment would leave a dialog behind.
            dialog.deleteLater()
        self.enrol(name, basis, notes, plate=plate)

    def enrol(
        self, name: str, basis: str, notes: str | None = None, *, plate: str | None = None
    ) -> str | None:
        """Enrol the selected track under this name. The subject id, or ``None``.

        The act the dialog performs, callable without it. Refusals — from this
        panel's own lock, or from the node and the register behind it — are
        sentences on the status line, never exceptions: a blank name, a track
        with no templates, a plate already somebody else's, faces switched off.
        The node writes the audit row; this panel writes nothing anywhere but
        the screen.
        """
        noun, _ = _NOUNS[self._kind]
        if self._actions is None:
            self.status.setText("No node is connected, so nothing can be enrolled.")
            return None
        if not self._editable:
            self.status.setText("Configure is locked; nothing was enrolled.")
            return None
        selection = self._selection
        if selection is None:
            self.status.setText("Select a track first; there is nothing to enrol from.")
            return None
        try:
            if self._kind is SubjectKind.PERSON:
                subject_id = self._actions.enrol_person(
                    name, selection.camera_id, selection.track_id, basis=basis, notes=notes or None
                )
            else:
                subject_id = self._actions.enrol_vehicle(
                    name,
                    plate or "",
                    basis=basis,
                    notes=notes or None,
                    camera_id=selection.camera_id,
                    track_id=selection.track_id,
                )
        except (RegistryError, ValueError) as refused:
            self.last_failure = refused
            self.status.setText(f"Not enrolled: {refused}")
            return None
        except Exception as failure:
            self.last_failure = failure
            self.status.setText(f"Enrolment failed: {failure}")
            return None
        self.refresh()
        self.select_subject(subject_id)
        self.status.setText(
            f"Enrolled {name!r} as {noun} {subject_id} from {selection.describe()}."
        )
        return subject_id

    # -------------------------------------------------------------- forgetting

    def _forget_clicked(self, _checked: bool = False) -> None:
        """The button: name what will go, ask, then forget. Nothing raises."""
        try:
            subject = self._selected_subject()
            if subject is None:
                self.status.setText(f"Select a {_NOUNS[self._kind][0]} to forget first.")
                return
            if not self._confirm_forget(subject):
                self.status.setText(f"Nothing was forgotten; {subject.display_name!r} is still enrolled.")
                return
            self.forget(subject.id)
        except Exception as failure:
            self.last_failure = failure
            self.status.setText(f"Forgetting failed: {failure}")

    def _confirm_forget(self, subject: Subject) -> bool:
        """Ask, naming the subject and the counts. The counts are read now.

        Read from the register at the moment of asking rather than from the
        table, which may be a refresh old: "delete 3 sightings" must be the
        number that will actually go.
        """
        assert self._actions is not None
        register = self._actions.register
        identifiers = len(register.identifiers(subject.id))
        sightings = len(register.history(subject.id))
        noun, _ = _NOUNS[self._kind]
        answer = QMessageBox.question(
            self,
            f"Forget this {noun}?",
            (
                f"Forget {subject.display_name!r}? This deletes {identifiers} "
                f"identifier(s) and removes {sightings} sighting(s) from the movement "
                "history. It cannot be undone. The audit log will record the counts "
                "and the id, never the name."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def forget(self, subject_id: str) -> Forgotten | None:
        """Forget a subject through the node, and say what went.

        The sentence carries the `Forgotten` counts the register returned —
        what was actually deleted — and the name, which is allowed on this
        screen for the operator's own confirmation and is deliberately absent
        from the audit row the node writes.
        """
        if self._actions is None:
            self.status.setText("No node is connected, so nothing can be forgotten.")
            return None
        if not self._editable:
            self.status.setText("Configure is locked; nothing was forgotten.")
            return None
        try:
            forgotten = self._actions.forget_subject(subject_id)
        except (RegistryError, ValueError) as refused:
            self.last_failure = refused
            self.status.setText(f"Not forgotten: {refused}")
            return None
        except Exception as failure:
            self.last_failure = failure
            self.status.setText(f"Forgetting failed: {failure}")
            return None
        self.refresh()
        if not forgotten.found:
            self.status.setText(
                f"Nothing to forget: no subject {subject_id} is enrolled — it may already "
                "have been forgotten."
            )
        else:
            self.status.setText(
                f"Forgot {forgotten.display_name!r}: {forgotten.identifiers_deleted} "
                f"identifier(s) deleted, {forgotten.sightings_unlinked} sighting(s) "
                "removed from the movement history."
            )
        return forgotten

    # ----------------------------------------------------------------- pinning

    def _pin_clicked(self, _checked: bool = False) -> None:
        """The button: pin the selected subject, or unpin. Nothing raises."""
        try:
            subject = self._selected_subject()
            if subject is None:
                self.status.setText(f"Select a {_NOUNS[self._kind][0]} to pin first.")
                return
            self.set_pinned(subject.id, not subject.pinned)
        except Exception as failure:
            self.last_failure = failure
            self.status.setText(f"Pinning failed: {failure}")

    def set_pinned(self, subject_id: str, pinned: bool) -> bool:
        """Pin or unpin through the node. True when the node accepted it."""
        if self._actions is None:
            self.status.setText("No node is connected, so nothing can be pinned.")
            return False
        if not self._editable:
            self.status.setText("Configure is locked; the pin was not changed.")
            return False
        try:
            self._actions.pin_subject(subject_id, bool(pinned))
        except (RegistryError, ValueError) as refused:
            self.last_failure = refused
            self.status.setText(f"Pin not changed: {refused}")
            return False
        except Exception as failure:
            self.last_failure = failure
            self.status.setText(f"Pinning failed: {failure}")
            return False
        self.refresh()
        subject = self._subjects.get(subject_id)
        name = subject.display_name if subject is not None else subject_id
        self.status.setText(
            f"Pinned {name!r}: exempt from the retention sweep until unpinned."
            if pinned
            else f"Unpinned {name!r}: back under the retention sweep."
        )
        return True

    # --------------------------------------------------------------- for tests

    def listed_subject_ids(self) -> list[str]:
        """The ids of the subjects on screen, top to bottom."""
        return [
            self.subjects.topLevelItem(index).data(NAME_COLUMN, Qt.ItemDataRole.UserRole)
            for index in range(self.subjects.topLevelItemCount())
        ]

    def subject_row_texts(self, subject_id: str) -> list[str]:
        """The cells of one subject's row, as the operator reads them."""
        for index in range(self.subjects.topLevelItemCount()):
            item = self.subjects.topLevelItem(index)
            if item.data(NAME_COLUMN, Qt.ItemDataRole.UserRole) == subject_id:
                return [item.text(column) for column in range(self.subjects.columnCount())]
        raise KeyError(f"subject {subject_id!r} is not on screen")

    def history_rows(self) -> list[tuple[str, ...]]:
        """The history's cells, top to bottom, as the operator reads them."""
        return [
            tuple(
                self.history.topLevelItem(index).text(column)
                for column in range(self.history.columnCount())
            )
            for index in range(self.history.topLevelItemCount())
        ]

    def history_selections(self) -> list[Selection]:
        """The track each history row would select, top to bottom."""
        return [
            self.history.topLevelItem(index).data(HISTORY_CAMERA_COLUMN, Qt.ItemDataRole.UserRole)
            for index in range(self.history.topLevelItemCount())
        ]

    def status_text(self) -> str:
        """The outcome line: what the last act did, or why it was refused."""
        return self.status.text()

    def capability_text(self) -> str:
        """The top line: the node's own identity status."""
        return self.capability.text()
