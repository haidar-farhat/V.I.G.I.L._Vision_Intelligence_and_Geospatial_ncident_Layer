"""Reading the audit log, from the console.

Every edit an operator makes — a zone reshaped, a camera moved, Configure taken
and given back — has been written to `audit_logs` since the first migration, and
since migration 6 a structured edit also carries the two states it went between
and a SHA-256 chained onto the row before it. Nothing in the interface has ever
read any of it. The log existed for a reader that did not exist, so the question
an audit actually starts from — *who changed the north gate zone, when, and from
what to what* — could only be put to a database browser, which is also the tool
that could quietly rewrite the answer.

Three things this panel is careful about, each a way an audit screen can say
more than the record supports:

**A page always says how big the log is.** The line under the list is built from
`Store.audit_totals`, never from the number of rows on screen, and a filter that
narrows the page says "showing 3 of 212" rather than showing three. The page is
the newest rows only, and when the log is longer than the page the line says the
older rows were neither shown nor filtered — a filter that silently searched a
fifth of the log would read as "nothing else matched".

**The chain covers what it covers.** Rows written through `Store.audit` carry no
hash: a node starting, analysis stopping, a camera added. They sit between the
chained rows and the chain says nothing about them. "Verify chain" reports the
chained count *and* the unprotected count on the same line, because a screen
reading "212 records verify" over a log of which forty are chained is the claim
that turns an integrity check into a false alibi. And on a chained row the hash
covers the two states and the who, what and when — `AuditRecord.canonical_bytes`
never sees the ``detail`` column — so the sentence in the Detail column can be
rewritten in a database browser and the chain still verifies. Measured: two
chained rows, ``UPDATE audit_logs SET detail = ...`` on the first, and "2
chained record(s) verify" over a sentence that never happened. The success line
and the row's tooltip both say so, and point at the expansion, which is the
diff over the two protected states and the only thing on the screen the hash
vouches for.

**A break is a row to examine, not yet proof of tampering.** The chain is
rebuilt from the stored rows — the row keeps the two JSON states, the actor, the
action, the subject, the node and the moment — and :func:`record_from_row` says
exactly how. Measured before this was written: a single-field edit, a moved ring
corner, a creation and a plain mapping all re-hash to the stored hash; an edit to
two fields at once, a set-valued field, and a timestamp finer than the
millisecond the row keeps do not, and `Node._audit` writes microsecond
timestamps today. So the break line names the row and says, in the same
sentence, that a row written in one of those forms fails here too. A panel that
called every such row "altered" would teach an operator to ignore the one line
that exists to be believed. The engine-side fix — a record reproducible from its
own row — is named in the handoff, not papered over here.

Nothing raises out of a slot. A malformed state — a row edited by hand into text
that is not JSON — is shown as exactly that, on the row, because a traceback out
of a Qt slot is retained by the interpreter and pins this widget past the
QApplication's own destruction.

Times are UTC throughout, stated on screen: the store keeps one clock.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QFont
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from sentinel.auditing import MISSING, AuditRecord, FieldChange, diff, verify_chain

from . import theme

#: Columns of the list. Named, because a panel that put the actor in the action
#: column would still pass a test that indexed by number.
WHEN_COLUMN = 0
ACTOR_COLUMN = 1
ACTION_COLUMN = 2
SUBJECT_COLUMN = 3
DETAIL_COLUMN = 4

#: Where an expanded row puts each half of a change. The field path takes the
#: tree column so it indents under its row; the two values take the detail
#: column, which is the only one wide enough to hold a coordinate pair.
FIELD_COLUMN = WHEN_COLUMN
CHANGE_COLUMN = DETAIL_COLUMN

#: How many of the newest rows one refresh loads. The read is synchronous on the
#: interface thread, and each row is diffed and built into an item when the list
#: is filled, so this is the bound on how long a refresh holds the window still.
#: Measured over 500 structured zone.changed rows, offscreen, one laptop, three
#: runs: 23–35 ms for a refresh, 21–31 ms per filter keystroke, 33–37 ms to
#: construct the panel — a short stall of about two frames, not under one, and
#: the item building is where it goes, not the diff. Beyond the page the honest
#: instrument is the total beside it, which says how much more there was.
PAGE_ROWS = 500

#: Shown where a row has no subject. Not an empty cell: a blank column reads as
#: a value that failed to load, and "this action was about nothing in
#: particular" — analysis stopping — is a fact.
NO_SUBJECT = "—"

#: The one child a prose-only row gets. It says why there is no diff rather
#: than showing an empty one, because an empty diff under a row that changed a
#: zone reads as "nothing changed", which is the opposite of what happened.
NO_STATES = (
    "This row carries no states: only the line above was written, and it has "
    "no chain hash, so the chain does not protect it."
)

#: What a chain hash does and does not vouch for, on the success line and in
#: every chained row's tooltip. Stated wherever the hash is mentioned, because
#: "hash 9dce…" beside a sentence reads as the sentence being verified, and the
#: sentence is the one column of the row the hash never sees: `Store.audit_record`
#: can be handed a ``detail`` that overrides the rendered line, and nothing
#: covers it. Expanding the row shows the diff over the two states, which is
#: covered.
HASH_COVERS = (
    "The hash covers each row's two states and its who, what and when — not "
    "the Detail sentence, which can be edited without breaking the chain; "
    "expand a row to see the protected diff."
)

#: Why a rebuilt row can fail to match its hash without having been altered.
#: Measured, not supposed — see the module docstring — and stated on the same
#: line as the break so the break cannot be read on its own.
REBUILD_LIMITS = (
    "A row also fails here when it was written in a form this rebuild cannot "
    "reproduce: a timestamp finer than the millisecond the row keeps, an edit "
    "to several fields at once, or a set-valued field. A break is a row to "
    "examine, not yet proof of tampering."
)

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class AuditRowError(ValueError):
    """A stored row that cannot be turned back into the record it came from.

    Raised with the row's id in the message, because "invalid JSON" on its own
    sends an operator through two hundred rows looking for the one that broke.
    """


def _at(millis: int) -> datetime:
    """The row's moment as an aware UTC datetime, to the millisecond, exactly.

    Integer arithmetic rather than ``fromtimestamp(millis / 1000)``: the division
    is a float, and a moment that comes back a microsecond off re-hashes to a
    different chain hash and reports a break on a row nobody touched.
    """
    return _EPOCH + timedelta(milliseconds=int(millis))


def _when(millis: int) -> str:
    return _at(millis).strftime("%Y-%m-%d %H:%M:%S")


def _parse_state(text: str | None, side: str, row_id: int):
    """One stored state as Python, ``MISSING`` when the row has none.

    A state that does not parse is a row that was not written by the store —
    `canonical` cannot produce it — and the error names the row and the side so
    the operator is sent to one cell rather than to the whole log.
    """
    if text is None:
        return MISSING
    try:
        return json.loads(text)
    except ValueError as error:
        raise AuditRowError(
            f"row #{row_id}'s {side} state is not JSON ({error}), which the store "
            "never writes"
        ) from None


def record_from_row(row) -> AuditRecord:
    """The `AuditRecord` a chained row was written from, rebuilt from the row.

    Everything the hash covers is on the row except the change list, which is
    recomputed with `auditing.diff` over the two stored states. That recovers
    the original list when the diff over the JSON walks the same paths in the
    same order as the diff over the objects did — one field, a ring corner, a
    creation — and not otherwise; :data:`REBUILD_LIMITS` names the cases. A
    prose-only row has no hash and nothing to rebuild; the caller is expected
    to skip it, since a record built from it would verify nothing.
    """
    row_id = row["id"]
    before = _parse_state(row["before_json"], "before", row_id)
    after = _parse_state(row["after_json"], "after", row_id)
    changes: tuple[FieldChange, ...] = ()
    if before is not MISSING and after is not MISSING:
        changes = tuple(diff(before, after))
    return AuditRecord(
        actor=row["actor"],
        action=row["action"],
        subject=row["subject"],
        node_id=row["node_id"],
        at=_at(row["at"]),
        before_json=row["before_json"],
        after_json=row["after_json"],
        changes=changes,
    )


def _render(value) -> str:
    """One side of a change as text. ``MISSING`` is named, never blanked.

    A corner that was appended and a corner that was set to null must read
    differently, because they are different facts about the outline.
    """
    if value is MISSING:
        return "(absent)"
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _row_id(row) -> int:
    return int(row["id"])


class AuditPanel(QWidget):
    """The audit log: who did what, when, from what to what — and does it hold.

    Fed a `sentinel.store.Store` — the node's, never one of its own; the console
    owns no database. Reads only, through `audit_trail` and `audit_totals`:
    there is no method on the store that changes an audit row and this panel
    wants none.
    """

    def __init__(self, store=None, parent: QWidget | None = None):
        super().__init__(parent)
        self._store = store
        #: The newest page, as the store returned it — newest first. Filtered
        #: in memory, so typing in the filter boxes never touches the database.
        self._rows: list = []
        #: `Store.audit_totals` at the last refresh: rows written, rows chained.
        self._written = 0
        self._chained = 0

        self.action_filter = QLineEdit()
        self.action_filter.setPlaceholderText("Action — zone.changed, camera.placed…")
        self.action_filter.setClearButtonEnabled(True)
        self.action_filter.setToolTip(
            "Matched literally, as a substring, ignoring case, against the "
            "rows already loaded."
        )
        self.subject_filter = QLineEdit()
        self.subject_filter.setPlaceholderText("Subject — a zone id, a camera id…")
        self.subject_filter.setClearButtonEnabled(True)
        self.subject_filter.setToolTip(self.action_filter.toolTip())
        # Bound methods, never lambdas closing over `self`: a lambda in a signal
        # connection is a reference cycle holding a QWidget, which is then
        # destroyed at interpreter shutdown after PySide has torn the
        # QApplication down, and that corrupts the heap on the way out.
        self.action_filter.textChanged.connect(self._filters_changed)
        self.subject_filter.textChanged.connect(self._filters_changed)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setToolTip("Read the newest rows from the store again.")
        self.refresh_button.clicked.connect(self._refresh_clicked)

        self.verify_button = QPushButton("Verify chain")
        self.verify_button.setToolTip(
            "Rebuild every chained record from its stored row and check each "
            "hash against the one before it. Prose-only rows carry no hash and "
            "are not covered."
        )
        self.verify_button.clicked.connect(self._verify_clicked)

        self.rows = QTreeWidget()
        self.rows.setColumnCount(5)
        self.rows.setHeaderLabels(["When (UTC)", "Actor", "Action", "Subject", "Detail"])
        self.rows.setRootIsDecorated(True)
        self.rows.setAlternatingRowColors(True)
        self.rows.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        header = self.rows.header()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        for index, width in enumerate((160, 130, 140, 140)):
            self.rows.setColumnWidth(index, width)

        #: The line that keeps a page from reading as the whole log. Never
        #: built from the number of rows on screen.
        self.summary = QLabel("")
        self.summary.setObjectName("Caption")
        self.summary.setWordWrap(True)

        #: What the last "Verify chain" found. Empty until it has been run:
        #: a verdict shown before anything was checked is a verdict about
        #: nothing.
        self.verdict = QLabel("")
        self.verdict.setObjectName("Caption")
        self.verdict.setWordWrap(True)
        # Never squeezed out: the verdict is the one sentence this tab exists
        # to produce, and in a short window the list took every pixel and left
        # it half a line tall — the first photograph of the tab cut the result
        # of "Verify chain" off mid-sentence. Room for three wrapped lines.
        self.verdict.setMinimumHeight(3 * self.verdict.fontMetrics().lineSpacing() + 6)

        filters = QHBoxLayout()
        filters.setContentsMargins(6, 4, 6, 0)
        filters.addWidget(self.action_filter, 1)
        filters.addWidget(self.subject_filter, 1)
        filters.addWidget(self.refresh_button)
        filters.addWidget(self.verify_button)

        clock = QLabel("Times are UTC — the clock the record is kept on.")
        clock.setObjectName("Caption")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addLayout(filters)
        layout.addWidget(clock)
        layout.addWidget(self.rows, 1)
        layout.addWidget(self.summary)
        layout.addWidget(self.verdict)

        self.refresh()

    # ---------------------------------------------------------------- the node

    def set_store(self, store) -> None:
        """Point the panel at the node's store and read it.

        Until the node exists the panel says it has nothing to read rather than
        showing an empty list, which is indistinguishable from a site on which
        nobody has ever done anything. The verdict is cleared too: it was about
        another database.
        """
        self._store = store
        self.verdict.setText("")
        self.refresh()

    # ---------------------------------------------------------------- reading

    def refresh(self) -> None:
        """Read the newest page and the totals, and show them.

        Nothing raises out of here — it is reached from a Qt slot — and the read
        and the draw are both inside the guard, because the half that builds a
        row parses JSON that a database browser may have edited into anything.
        A failure clears the list: rows from a page that was never finished
        read as the whole of the log.
        """
        if self._store is None:
            self._rows = []
            self._written = self._chained = 0
            self._fill()
            self.summary.setText("No database is open, so there is no audit log to read.")
            return
        try:
            self._written, self._chained = self._store.audit_totals()
            self._rows = list(self._store.audit_trail(limit=PAGE_ROWS))
            self._fill()
        except Exception as failure:
            self._rows = []
            self.rows.clear()
            self.summary.setText(f"The audit log could not be read: {failure}")

    def _refresh_clicked(self, _checked: bool = False) -> None:
        """Swallows ``clicked(bool)``: connected straight to `refresh`, the
        checked flag would be its first positional argument, and a later
        `refresh` that takes a parameter would silently receive ``False``."""
        self.refresh()

    def _filters_changed(self, *_ignored) -> None:
        """A filter box changed. In-memory, so no read; but still guarded."""
        try:
            self._fill()
        except Exception as failure:
            self.rows.clear()
            self.summary.setText(f"The audit log could not be shown: {failure}")

    # ---------------------------------------------------------------- the page

    def _matches(self, row) -> bool:
        action = self.action_filter.text().strip().casefold()
        subject = self.subject_filter.text().strip().casefold()
        if action and action not in (row["action"] or "").casefold():
            return False
        if subject and subject not in (row["subject"] or "").casefold():
            return False
        return True

    def _fill(self) -> None:
        """Replace every row with the ones that pass the filters, newest first.

        Rebuilt rather than diffed, like the other panels over the store: a
        reused row can keep a stale value in a column that failed to update,
        and in an audit view a stale cell is worse than a flicker.
        """
        self.rows.clear()
        shown = 0
        for row in self._rows:
            if not self._matches(row):
                continue
            self.rows.addTopLevelItem(self._item(row))
            shown += 1
        self.summary.setText(self._summarise(shown))

    def _summarise(self, shown: int) -> str:
        """The line under the list, from `audit_totals` and nothing else.

        The loaded-page caveat is conditional on the log actually being longer
        than the page, so the line never warns about rows that do not exist —
        a caveat that is always there is a caveat nobody reads.
        """
        if self._written == 0:
            return "No audit rows have been written. Nothing is shown because nothing is there."
        filters = []
        if self.action_filter.text().strip():
            filters.append(f"action containing {self.action_filter.text().strip()!r}")
        if self.subject_filter.text().strip():
            filters.append(f"subject containing {self.subject_filter.text().strip()!r}")
        text = f"Showing {shown:,} of {self._written:,} audit rows"
        if filters:
            text += " matching " + " and ".join(filters)
        loaded = len(self._rows)
        if loaded < self._written:
            text += (
                f"; only the newest {loaded:,} are loaded, so older rows are "
                "neither shown nor filtered"
            )
        text += f". {self._chained:,} of the {self._written:,} carry a chain hash."
        return text

    def _item(self, row) -> QTreeWidgetItem:
        item = QTreeWidgetItem([
            _when(row["at"]),
            row["actor"],
            row["action"],
            row["subject"] or NO_SUBJECT,
            row["detail"] or "",
        ])
        item.setData(WHEN_COLUMN, Qt.ItemDataRole.UserRole, _row_id(row))
        bold = QFont(self.rows.font())
        bold.setBold(True)
        item.setFont(ACTION_COLUMN, bold)
        if row["chain_hash"] is None:
            item.setToolTip(WHEN_COLUMN, f"Row #{row['id']} — prose only, no chain hash.")
            item.setForeground(WHEN_COLUMN, QBrush(theme.TEXT_MUTED))
        else:
            item.setToolTip(
                WHEN_COLUMN,
                f"Row #{row['id']} — written by node {row['node_id'] or '?'}; "
                f"chain hash {row['chain_hash'][:12]}…. {HASH_COVERS}",
            )
        for child in self._explain(row):
            item.addChild(child)
        return item

    def _explain(self, row) -> list[QTreeWidgetItem]:
        """The children a row expands into: a diff, or the reason there is none.

        Each state is parsed on its own, so a row whose after-state was edited
        into rubbish still shows its before-state, and the rubbish is named as
        rubbish on that row rather than raised out of the fill.
        """
        row_id = row["id"]
        if row["before_json"] is None and row["after_json"] is None:
            return [self._note(NO_STATES)]
        children: list[QTreeWidgetItem] = []
        states = {}
        for side in ("before", "after"):
            try:
                states[side] = _parse_state(row[f"{side}_json"], side, row_id)
            except AuditRowError as damaged:
                children.append(self._note(f"Cannot show the {side} state: {damaged}."))
        if len(states) < 2:
            return children
        before, after = states["before"], states["after"]
        if before is MISSING:
            child = self._note("No before-state: this row records a creation. After:")
            child.setText(CHANGE_COLUMN, _render(after))
            return [child]
        if after is MISSING:
            child = self._note("No after-state: this row records a removal. Before:")
            child.setText(CHANGE_COLUMN, _render(before))
            return [child]
        changes = diff(before, after)
        if not changes:
            return [self._note("Both states were recorded and they do not differ.")]
        for change in changes:
            child = QTreeWidgetItem()
            child.setText(FIELD_COLUMN, change.path or "(subject)")
            if change.added:
                child.setText(CHANGE_COLUMN, f"added {_render(change.after)}")
            elif change.removed:
                child.setText(CHANGE_COLUMN, f"removed {_render(change.before)}")
            else:
                child.setText(
                    CHANGE_COLUMN, f"{_render(change.before)} -> {_render(change.after)}"
                )
            child.setToolTip(CHANGE_COLUMN, child.text(CHANGE_COLUMN))
            children.append(child)
        return children

    def _note(self, text: str) -> QTreeWidgetItem:
        child = QTreeWidgetItem()
        child.setText(FIELD_COLUMN, text)
        child.setForeground(FIELD_COLUMN, QBrush(theme.TEXT_MUTED))
        return child

    # ------------------------------------------------------------- the chain

    def verify(self) -> None:
        """Rebuild the chain from every chained row and check it.

        Every chained row, not the page: a chain checked over its newest five
        hundred links says nothing about the row edited last year, which is
        the row an audit chain exists to catch. Read in insertion order,
        because that is the order `Store.audit_record` folded them in; the
        list is by time, and two rows written in one millisecond would be
        chained one way and checked the other.

        `auditing.verify_chain` does the checking. This method only rebuilds
        the records and words the answer; a second implementation of the
        check here would be a second place for it to be wrong.

        The read is sized from the count but not bounded by it. `Store.audit_record`
        says a `sentinel` command can commit beside the console, and `audit_trail`
        is newest-first with a LIMIT: a row committed between the count and the
        read would push the *oldest* row past the limit, the first chained
        record would then be checked against no predecessor, and the panel would
        report a break at record 1 on a log nobody touched. So the limit carries
        a page of headroom, and both numbers on the line — the chained count and
        the unprotected count — come from the rows actually read, never from the
        count taken a moment earlier. A store method returning every chained
        row in id order would remove the count altogether; until it exists this
        is the read that does not lose the oldest row.

        Nothing raises out of here. A row that cannot be rebuilt — a state that
        is no longer JSON — is reported as the row where the check stops, with
        the rows before it checked as far as they go.
        """
        if self._store is None:
            self.verdict.setText("No database is open, so there is no chain to verify.")
            return
        try:
            written, _chained = self._store.audit_totals()
            rows = self._store.audit_trail(limit=written + PAGE_ROWS)
            chained = sorted((r for r in rows if r["chain_hash"] is not None), key=_row_id)
            self.verdict.setText(self._check(chained, len(rows) - len(chained)))
        except Exception as failure:
            self.verdict.setText(f"The chain could not be checked: {failure}")

    def _verify_clicked(self, _checked: bool = False) -> None:
        """Swallows ``clicked(bool)``: connected straight to `verify`, the
        checked flag would be its first positional argument, and a later
        `verify` that takes a parameter would silently receive ``False``."""
        self.verify()

    def _check(self, chained: list, unprotected: int) -> str:
        """Run the chain and word what it found.

        The wording is the product. A success names the count and the
        unprotected count; a break names the position, the row, the actor and
        the action, says that every later record fails only because it follows,
        and — on the same line — that a row this rebuild cannot reproduce
        fails here too. Split across two places, the second half is the one
        nobody reads.
        """
        aside = (
            f"{unprotected:,} prose-only row(s) carry no hash and are not "
            "protected by the chain."
        )
        if not chained:
            if unprotected == 0:
                return "The audit log is empty: there is nothing to verify."
            return f"No row carries a chain hash, so nothing can be verified. {aside}"

        records: list[AuditRecord] = []
        hashes: list[str] = []
        stopped: str | None = None
        broken: int | None = None
        for index, row in enumerate(chained):
            try:
                records.append(record_from_row(row))
            except AuditRowError as damaged:
                # The rows before it are checked as far as they go; if they
                # hold, the unrebuildable row is where the chain breaks.
                earlier = verify_chain(records, hashes)
                broken = index if earlier is None else earlier
                stopped = str(damaged) if earlier is None else None
                break
            hashes.append(row["chain_hash"])
        else:
            broken = verify_chain(records, hashes)

        if broken is None:
            first, last = chained[0]["id"], chained[-1]["id"]
            return (
                f"{len(chained):,} chained record(s) verify, row #{first} to "
                f"#{last}, head {hashes[-1][:12]}…. {aside} A chain that verifies "
                f"shows its rows agree with one another. {HASH_COVERS} Only a "
                "head hash kept outside this database can show the whole chain "
                "was not rewritten."
            )

        row = chained[min(broken, len(chained) - 1)]
        reason = stopped or (
            "the hash stored on it is not the hash of its stored fields "
            "chained onto the record before it"
        )
        return (
            f"The chain breaks at chained record {broken + 1} of {len(chained):,} — "
            f"row #{row['id']}, {row['action']} by {row['actor']} at "
            f"{_when(row['at'])} UTC: {reason}. Every chained record after it "
            f"fails only because it follows. {aside} {REBUILD_LIMITS}"
        )

    # ------------------------------------------------------------- for tests

    def listed_row_ids(self) -> list[int]:
        """The ids of the rows on screen, top to bottom."""
        return [
            self.rows.topLevelItem(index).data(WHEN_COLUMN, Qt.ItemDataRole.UserRole)
            for index in range(self.rows.topLevelItemCount())
        ]

    def expansion_of(self, row_id: int) -> list[tuple[str, str]]:
        """What a row expands into: (field, change) pairs, or one note.

        Read off the tree rather than recomputed, so a test that asks for a
        row's diff is asking what the operator would see.
        """
        item = self._item_for(row_id)
        return [
            (item.child(k).text(FIELD_COLUMN), item.child(k).text(CHANGE_COLUMN))
            for k in range(item.childCount())
        ]

    def detail_of(self, row_id: int) -> tuple[str, str]:
        """A row's Detail cell and its tooltip, as the operator reads them.

        Exists so a test can show the Detail sentence *is* on screen after it
        was rewritten under a verifying chain — the exposure documented rather
        than hidden — and that the tooltip beside the hash says the hash does
        not cover it.
        """
        item = self._item_for(row_id)
        return item.text(DETAIL_COLUMN), item.toolTip(WHEN_COLUMN)

    def _item_for(self, row_id: int) -> QTreeWidgetItem:
        for index in range(self.rows.topLevelItemCount()):
            item = self.rows.topLevelItem(index)
            if item.data(WHEN_COLUMN, Qt.ItemDataRole.UserRole) == row_id:
                return item
        raise KeyError(f"row #{row_id} is not on screen")
