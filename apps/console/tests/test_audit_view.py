"""Tests for the audit panel.

The panel is the console's only reader of the audit log, so the failures it
must not have are the ones that make the log look like it says more than it
does:

- **A page must never read as the whole log.** Every assertion about the rows
  on screen also asserts the line under them, and that line carries the totals
  from the store, not the count of rows drawn. A filter that narrows the page
  says "3 of 5"; a log longer than the page says its older rows were not loaded.
- **The chain must say what it covers.** A verdict of "3 records verify" over a
  log of five is only honest with "2 carry no hash" on the same line — and the
  hash never sees the Detail column, so a chain that verifies over a rewritten
  sentence must say the sentence is not what it verified.
- **A count read a moment ago must not size the read.** A row committed
  between the count and the read pushes the oldest row past the limit and the
  first chained record is checked against nothing; the panel would then report
  a break on a log nobody touched.
- **A break must name the row.** "The log was altered" is not actionable and
  "record 3 of 3, row #3" is. And a break must not be called tampering outright:
  the rebuild has measured limits, and the line says so.
- **The affordances the panel advertises must be the ones under test.** The
  Verify chain and Refresh buttons are clicked here, through the widgets, and
  `set_store` is called the way the console calls it, because a button wired to
  nothing looks exactly like a button that worked.
- **Nothing reaches a Qt slot's caller.** A state edited into text that is not
  JSON is shown on its row as exactly that.

The store underneath is the real one: three structured rows written through
`audit_record` — one field changed, a creation, another field changed — and two
prose rows through `audit`. A stub store would let every one of these pass
against a panel that reads nothing.
"""

from __future__ import annotations

import gc
import os
import weakref
from datetime import datetime, timezone

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from sentinel.auditing import MISSING, AuditRecord, verify_chain  # noqa: E402
from sentinel.core import LatLon, destination_point  # noqa: E402
from sentinel.store import Store  # noqa: E402
from sentinel.zones import Zone, ZoneKind  # noqa: E402

from sentinel_console.audit_view import (  # noqa: E402
    HASH_COVERS,
    NO_STATES,
    PAGE_ROWS,
    REBUILD_LIMITS,
    AuditPanel,
    record_from_row,
)

SITE = LatLon(33.8938, 35.5018)
RING = tuple(destination_point(SITE, bearing, 30.0) for bearing in (0.0, 90.0, 180.0))

#: A fixed instant with no sub-millisecond part: the row keeps milliseconds and
#: the hash covers the whole timestamp, so a finer one cannot be rebuilt. That
#: limit is pinned by its own test below rather than hidden in the fixture.
AT = datetime(2026, 3, 1, 9, 30, 0, 250_000, tzinfo=timezone.utc)

#: How many rows the fixture writes, and how many carry a hash. Measured from
#: the store in the fixture before anything asserts on them.
ROWS_IN_STORE = 5
CHAINED_IN_STORE = 3
PROSE_IN_STORE = ROWS_IN_STORE - CHAINED_IN_STORE


@pytest.fixture(scope="session")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def zone(**overrides) -> Zone:
    fields = dict(id="zone-a", name="Yard", kind=ZoneKind.RESTRICTED, ring=RING)
    fields.update(overrides)
    return Zone(**fields)


def record(*, before=MISSING, after=MISSING, at=AT, **overrides) -> AuditRecord:
    fields = dict(
        actor="operator:alice",
        action="zone.changed",
        subject="zone-a",
        node_id="gatehouse",
        at=at,
        before=before,
        after=after,
    )
    fields.update(overrides)
    return AuditRecord.of(**fields)


@pytest.fixture
def store() -> Store:
    """Three chained rows, then two prose rows — so the prose rows are newest.

    `audit` stamps its rows with the wall clock, which is later than the fixed
    instant the chained rows carry, and the panel orders by time; writing the
    prose rows last makes "newest first" the same order as "last written",
    which is the order every assertion below can be read against.
    """
    with Store(":memory:") as db:
        db.audit_record(record(before=zone(), after=zone(kind=ZoneKind.EXCLUSION)))
        db.audit_record(
            record(action="zone.created", subject="zone-b", after=zone(id="zone-b"))
        )
        db.audit_record(record(before=zone(), after=zone(name="North yard")))
        db.audit("node", "analysis.started", detail="2 camera(s)")
        db.audit("operator:alice", "node.stopped", "gatehouse")
        assert db.audit_totals() == (ROWS_IN_STORE, CHAINED_IN_STORE)
        yield db


@pytest.fixture
def panel(qt_app, store: Store) -> AuditPanel:
    widget = AuditPanel(store)
    yield widget
    widget.deleteLater()


def tamper(store: Store, row_id: int, after_json: str) -> None:
    """Edit a row the way a database browser would: straight into the table."""
    store._connection.execute(
        "UPDATE audit_logs SET after_json = ? WHERE id = ?", (after_json, row_id)
    )


def rewrite_detail(store: Store, row_id: int, detail: str) -> None:
    """Rewrite the sentence in the Detail column and nothing else on the row."""
    store._connection.execute(
        "UPDATE audit_logs SET detail = ? WHERE id = ?", (detail, row_id)
    )


# ------------------------------------------------------------------ the list


def test_the_log_is_listed_newest_first(panel):
    assert panel.listed_row_ids() == [5, 4, 3, 2, 1]
    assert panel.summary.text().startswith(
        f"Showing {ROWS_IN_STORE} of {ROWS_IN_STORE} audit rows"
    )
    assert f"{CHAINED_IN_STORE} of the {ROWS_IN_STORE} carry a chain hash" in panel.summary.text()


def test_a_panel_with_no_database_says_so_instead_of_showing_nothing(qt_app):
    widget = AuditPanel()
    assert widget.listed_row_ids() == []
    assert "No database is open" in widget.summary.text()
    widget.verify()
    assert "No database is open" in widget.verdict.text()


def test_a_panel_built_before_the_node_reads_the_store_it_is_handed(qt_app, store):
    # `set_store` is how the console hands over the node's database once the
    # node exists; a panel built earlier must show the log then, not stay empty.
    widget = AuditPanel()
    assert widget.listed_row_ids() == []
    widget.set_store(store)
    assert widget.listed_row_ids() == [5, 4, 3, 2, 1]


def test_an_empty_log_says_nothing_has_been_written(qt_app):
    with Store(":memory:") as empty:
        widget = AuditPanel(empty)
        assert widget.listed_row_ids() == []
        assert "No audit rows have been written" in widget.summary.text()


def test_the_refresh_button_reads_rows_written_since(panel, store: Store):
    store.audit("operator:bob", "camera.added", "cam-07")
    assert panel.listed_row_ids() == [5, 4, 3, 2, 1], "the panel read the store unasked"
    panel.refresh_button.click()
    assert panel.listed_row_ids() == [6, 5, 4, 3, 2, 1]
    assert panel.summary.text().startswith("Showing 6 of 6 audit rows")


def test_a_log_longer_than_the_page_says_its_older_rows_are_not_loaded(qt_app):
    extra = 3
    with Store(":memory:") as long_log:
        for index in range(PAGE_ROWS + extra):
            long_log.audit("node", "tick", str(index))
        widget = AuditPanel(long_log)
        assert len(widget.listed_row_ids()) == PAGE_ROWS
        text = widget.summary.text()
        assert text.startswith(f"Showing {PAGE_ROWS} of {PAGE_ROWS + extra:,} audit rows")
        assert f"only the newest {PAGE_ROWS} are loaded" in text


# --------------------------------------------------------------- the filters


def test_the_action_filter_narrows_to_actions_that_say_it(panel):
    panel.action_filter.setText("zone")
    assert panel.listed_row_ids() == [3, 2, 1]
    assert panel.summary.text().startswith(
        f"Showing 3 of {ROWS_IN_STORE} audit rows matching action containing 'zone'"
    )


def test_the_subject_filter_narrows_to_that_subject(panel):
    panel.subject_filter.setText("GATEHOUSE")
    assert panel.listed_row_ids() == [5], "the match was case-sensitive"
    assert "subject containing 'GATEHOUSE'" in panel.summary.text()


def test_two_filters_narrow_further_than_either_alone(panel):
    panel.action_filter.setText("zone")
    panel.subject_filter.setText("zone-b")
    assert panel.listed_row_ids() == [2]
    assert "action containing 'zone' and subject containing 'zone-b'" in panel.summary.text()


def test_a_filter_can_be_taken_off_again(panel):
    panel.action_filter.setText("node")
    assert panel.listed_row_ids() == [5]
    panel.action_filter.clear()
    assert panel.listed_row_ids() == [5, 4, 3, 2, 1]
    assert "matching" not in panel.summary.text()


def test_a_filter_that_matches_nothing_still_says_how_big_the_log_is(panel):
    panel.action_filter.setText("no such action")
    assert panel.listed_row_ids() == []
    assert panel.summary.text().startswith(f"Showing 0 of {ROWS_IN_STORE} audit rows")


# ------------------------------------------------------------ the expansion


def test_an_expanded_structured_row_shows_exactly_the_changed_field(panel):
    # One field changed between the two states, so one child and no more: a
    # diff that listed every field would bury the edit among the unchanged.
    assert panel.expansion_of(1) == [("kind", '"RESTRICTED" -> "EXCLUSION"')]
    assert panel.expansion_of(3) == [("name", '"Yard" -> "North yard"')]


def test_a_creation_row_says_it_has_no_before_state(panel):
    (note, state), = panel.expansion_of(2)
    assert note.startswith("No before-state: this row records a creation")
    assert '"id": "zone-b"' in state


def test_a_prose_only_row_explains_it_carries_no_states(panel):
    assert panel.expansion_of(5) == [(NO_STATES, "")]
    assert panel.expansion_of(4) == [(NO_STATES, "")]
    assert "no chain hash" in NO_STATES


def test_a_state_that_is_not_json_is_shown_as_such_not_raised(panel, store: Store):
    tamper(store, 1, "{not json")
    panel.refresh_button.click()  # a Qt slot: an exception here would be retained
    expansion = panel.expansion_of(1)
    assert len(expansion) == 1
    note, _ = expansion[0]
    assert note.startswith("Cannot show the after state: row #1's after state is not JSON")
    assert panel.listed_row_ids() == [5, 4, 3, 2, 1], "one bad row emptied the page"


# ---------------------------------------------------------------- the chain


def test_a_row_rebuilt_from_the_store_is_the_record_that_was_written(store: Store):
    # The whole verification rests on this: the record the row is turned back
    # into must hash to the hash the store wrote from the original. Checked
    # here with the engine's own `chain`, not the panel's wording.
    rows = sorted(store.audit_trail(), key=lambda r: r["id"])
    chained = [row for row in rows if row["chain_hash"] is not None]
    records = [record_from_row(row) for row in chained]
    assert records[0].chain(None) == chained[0]["chain_hash"]
    assert verify_chain(records, [row["chain_hash"] for row in chained]) is None


def test_verify_reports_the_chained_count_and_the_unprotected_count(panel):
    assert panel.verdict.text() == "", "a verdict was shown before anything was checked"
    panel.verify_button.click()
    text = panel.verdict.text()
    assert text.startswith(f"{CHAINED_IN_STORE} chained record(s) verify, row #1 to #3")
    assert f"{PROSE_IN_STORE} prose-only row(s) carry no hash and are not protected" in text
    assert "Only a head hash kept outside this database" in text


def test_a_rewritten_detail_sentence_still_verifies_and_the_line_says_the_sentence_is_not_covered(
    panel, store: Store
):
    """The exposure, documented rather than hidden.

    `AuditRecord.canonical_bytes` hashes the actor, action, subject, node, moment
    and the two states; the ``detail`` column is outside it, by the store's own
    docstring. So a sentence rewritten in a database browser sits on screen
    beside a hash that still verifies. The panel cannot detect that — nothing
    can — and the one thing it can do is say, on the success line and in the
    tooltip beside the hash, that the sentence is not what verified. A line
    reading "3 records verify" over an invented sentence, with nothing else,
    is an alibi the record does not support.
    """
    invented = "kind RESTRICTED -> PUBLIC (never happened)"
    rewrite_detail(store, 1, invented)
    panel.refresh_button.click()
    panel.verify_button.click()
    text = panel.verdict.text()
    assert text.startswith(f"{CHAINED_IN_STORE} chained record(s) verify"), (
        "the chain does not cover the sentence, so it must still verify"
    )
    assert HASH_COVERS in text
    assert "not the Detail sentence, which can be edited without breaking the chain" in HASH_COVERS
    detail, tooltip = panel.detail_of(1)
    assert detail == invented, "the exposure is real: the rewritten sentence is what is shown"
    assert "chain hash" in tooltip and HASH_COVERS in tooltip
    # The expansion is the protected part, and it still shows what happened.
    assert panel.expansion_of(1) == [("kind", '"RESTRICTED" -> "EXCLUSION"')]


def test_a_prose_only_row_tooltip_does_not_mention_a_hash(panel):
    _, tooltip = panel.detail_of(5)
    assert "prose only, no chain hash" in tooltip
    assert HASH_COVERS not in tooltip


def test_verify_names_the_row_where_the_chain_breaks(panel, store: Store):
    # The after-state of the last chained row is rewritten to a state the row
    # never held. The hash covers that text, so the rebuilt record no longer
    # matches, and the line has to say which record and which row.
    tamper(store, 3, '{"kind":"EXCLUSION","name":"South yard"}')
    panel.verify_button.click()
    text = panel.verdict.text()
    assert text.startswith(
        f"The chain breaks at chained record 3 of {CHAINED_IN_STORE} — row #3, "
        "zone.changed by operator:alice at 2026-03-01 09:30:00 UTC"
    )
    assert "not the hash of its stored fields" in text
    assert "fails only because it follows" in text
    assert REBUILD_LIMITS in text, "a break was reported as tampering outright"


def test_a_break_in_the_first_row_is_reported_at_the_first_row(panel, store: Store):
    tamper(store, 1, '{"kind":"EXCLUSION","name":"Yard"}')
    panel.verify_button.click()
    assert panel.verdict.text().startswith(
        f"The chain breaks at chained record 1 of {CHAINED_IN_STORE} — row #1,"
    )


def test_a_state_that_is_not_json_stops_the_check_at_that_row(panel, store: Store):
    tamper(store, 2, "not json at all")
    panel.verify_button.click()
    text = panel.verdict.text()
    assert text.startswith(f"The chain breaks at chained record 2 of {CHAINED_IN_STORE} — row #2,")
    assert "row #2's after state is not JSON" in text


def test_verify_checks_rows_older_than_the_page(qt_app):
    # The page is the newest rows; the chain is every chained row. A check that
    # stopped at the page would never see the row edited last year.
    with Store(":memory:") as long_log:
        long_log.audit_record(record(before=zone(), after=zone(kind=ZoneKind.EXCLUSION)))
        for index in range(PAGE_ROWS + 1):
            long_log.audit("node", "tick", str(index))
        widget = AuditPanel(long_log)
        assert 1 not in widget.listed_row_ids(), "the chained row is on the page"
        widget.verify_button.click()
        assert widget.verdict.text().startswith("1 chained record(s) verify, row #1 to #1")
        tamper(long_log, 1, '{"kind":"EXCLUSION"}')
        widget.verify_button.click()
        assert widget.verdict.text().startswith(
            "The chain breaks at chained record 1 of 1 — row #1,"
        )


class CountThenCommitBeside:
    """A store whose count is stale by one row the moment it is returned.

    `Store` has ``__slots__``, so the interleaving is forced by delegation
    rather than by patching: the panel reads through this object, the count
    comes from the real store, and a prose row is committed to the real store
    before the count is handed back — the worst case of a `sentinel` command
    committing beside the console.
    """

    def __init__(self, store: Store):
        self._store = store
        #: Off while the panel is handed the store, because `set_store` reads
        #: the totals too and the race under test is the one inside `verify`.
        self.armed = False

    def audit_totals(self) -> tuple[int, int]:
        totals = self._store.audit_totals()
        if self.armed:
            self._store.audit("sentinel", "cameras.remove", "cam-09")
        return totals

    def audit_trail(self, *, limit: int):
        return self._store.audit_trail(limit=limit)


def test_a_row_committed_between_the_count_and_the_read_is_still_checked(panel, store: Store):
    """The race `Store.audit_record` warns about, forced.

    `audit_trail` is newest-first under a LIMIT. Sized exactly to a count taken
    a moment earlier, one row committed in between pushes the *oldest* row out
    of the read; the first chained record is then checked against no
    predecessor and reported as a break on a log nobody touched. The verdict
    must still verify, and the unprotected count must be the one from the rows
    actually read, not the count.
    """
    beside = CountThenCommitBeside(store)
    panel.set_store(beside)
    beside.armed = True
    panel.verify_button.click()
    text = panel.verdict.text()
    assert text.startswith(f"{CHAINED_IN_STORE} chained record(s) verify, row #1 to #3"), text
    assert f"{PROSE_IN_STORE + 1} prose-only row(s) carry no hash" in text, (
        "the unprotected count came from the count, not from the rows read"
    )


def test_a_prose_only_log_says_nothing_can_be_verified(qt_app):
    with Store(":memory:") as prose:
        prose.audit("node", "node.started", "gatehouse")
        prose.audit("node", "node.stopped", "gatehouse")
        widget = AuditPanel(prose)
        widget.verify_button.click()
        text = widget.verdict.text()
        assert text.startswith("No row carries a chain hash, so nothing can be verified")
        assert "2 prose-only row(s)" in text


def test_a_two_field_edit_is_reported_as_a_break_and_the_line_says_why(qt_app):
    """The measured limit, pinned so that lifting it is seen.

    A record's hash covers its change list in the order `diff` produced it over
    the zone objects — declaration order — and the row does not keep that list.
    Rebuilt over the stored JSON the list comes out in key order, so an edit to
    `name` and `kind` together re-hashes differently. A row nobody touched then
    reads as a break, and the only honest thing the panel can do is say on the
    same line that this is one of the ways an unaltered row fails. When the
    engine makes a record reproducible from its row, this test fails and comes
    out; until then it is the reason `REBUILD_LIMITS` exists.
    """
    with Store(":memory:") as db:
        db.audit_record(
            record(before=zone(), after=zone(name="North yard", kind=ZoneKind.EXCLUSION))
        )
        widget = AuditPanel(db)
        widget.verify_button.click()
        text = widget.verdict.text()
        assert text.startswith("The chain breaks at chained record 1 of 1 — row #1,")
        assert "an edit to several fields at once" in text


def test_a_timestamp_finer_than_a_millisecond_is_stored_and_verifies_to_the_millisecond(qt_app):
    """The store hashes what it writes.

    `Node._audit` stamps records with `datetime.now(timezone.utc)`, which has
    microseconds; the row keeps milliseconds. The store used to hash the
    microseconds and write the milliseconds, so every real row failed its own
    verification — this test pinned that as a "measured limit" until the first
    photograph of the tab showed it on a fresh database. The store now
    truncates before hashing, and a microsecond stamp verifies.
    """
    with Store(":memory:") as db:
        db.audit_record(
            record(
                before=zone(),
                after=zone(kind=ZoneKind.EXCLUSION),
                at=AT.replace(microsecond=250_123),
            )
        )
        widget = AuditPanel(db)
        widget.verify_button.click()
        text = widget.verdict.text()
        assert text.startswith("1 chained record(s) verify"), text


def test_a_new_store_clears_the_old_verdict(panel, store: Store):
    panel.verify_button.click()
    assert panel.verdict.text() != ""
    with Store(":memory:") as other:
        panel.set_store(other)
        assert panel.verdict.text() == "", "a verdict about another database survived"
        panel.set_store(store)


# --------------------------------------------------------------- lifetime


def test_the_panel_is_freed_when_its_last_reference_goes(qt_app, store: Store):
    # A reference cycle holding a QWidget means the widget is destroyed at
    # interpreter shutdown, after PySide has torn the QApplication down, which
    # corrupts the heap and kills the process at exit — after every test has
    # passed. A lambda closing over `self` in a signal connection is how that
    # happens, so every connection inside the panel is a bound method.
    widget = AuditPanel(store)
    widget.action_filter.setText("zone")
    widget.verify()
    ref = weakref.ref(widget)
    del widget
    gc.collect()
    survivor = ref()
    if survivor is not None:
        holders = sorted(
            {type(r).__name__ for r in gc.get_referrers(survivor)} - {"frame", "list"}
        )
        del survivor
        raise AssertionError(f"AuditPanel outlived its last reference (held by {holders}).")


def test_the_verdict_is_never_squeezed_to_less_than_three_lines(qt_app):
    # The first photograph of this tab cut the result of "Verify chain" off
    # mid-sentence: the list took every pixel. The verdict is the sentence the
    # tab exists to produce and keeps room for three wrapped lines.
    from sentinel_console.audit_view import AuditPanel

    panel = AuditPanel()
    panel.resize(900, 300)
    panel.show()
    qt_app.processEvents()
    assert panel.verdict.height() >= 3 * panel.verdict.fontMetrics().lineSpacing()
