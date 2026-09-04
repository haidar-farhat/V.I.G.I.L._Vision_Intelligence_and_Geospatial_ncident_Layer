"""Tests for the investigation panel.

The panel is the console's only way into the stored record, so the failures it
must not have are the ones that make a search look like an answer when it is not:

- **A page must never read as the whole answer.** Every assertion about the rows
  on screen also asserts the line under them, and that line has to carry the real
  total. A panel that shows twenty of three hundred and says nothing is how an
  operator concludes a quiet night.
- **Nothing found must say so.** An empty list and a panel that failed to run are
  the same picture, and the operator cannot tell them apart.
- **Every control must narrow.** A filter that is wired to nothing looks exactly
  like a filter that matched everything.
- **The affordances the panel advertises must be the ones under test.** The two
  ways an operator runs a search are the Search button and Enter in the phrase
  field, and both were once wired to nothing that any test touched: cutting both
  connections left all twenty-four tests green, because every one of them drove a
  combo box or called `search()` directly. So the button is clicked and
  `returnPressed` is emitted here, through the widgets, exactly as a person does
  it. The same goes for `set_store`, which is how the console hands over the
  node's database once the node exists.
- **`set_selection` must not emit.** It is called from the console's selection
  bus, and a re-emit goes straight back into the bus — a frozen window.
- **Nothing reaches a Qt slot's caller.** Not a refused query, not a failed one,
  and not a row the panel cannot draw — that last one escaped `returnPressed` as
  a live `ValueError` until the render half of `search()` was brought inside the
  guard its own docstring promised.

The store underneath is the real one, built the way `engine/tests/test_search.py`
builds it: five events across three cameras, four severities and two zones, two
of them correlated into one incident and two into another. A stub store would
let every one of these tests pass against a panel that queries nothing.
"""

from __future__ import annotations

import gc
import os
import weakref
from datetime import datetime, timezone

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QDateTime, Qt, QTimeZone  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from sentinel.core import LatLon  # noqa: E402
from sentinel.events import (  # noqa: E402
    Event,
    Evidence,
    EventType,
    Severity,
    utc_from_millis,
)
from sentinel.incidents import Correlator  # noqa: E402
from sentinel.store import Store  # noqa: E402

from sentinel_console.investigation import (  # noqa: E402
    MAX_RESULTS,
    SUBJECT_EVENTS,
    SUBJECT_INCIDENTS,
    WHEN_COLUMN,
    InvestigationPanel,
)
from sentinel_console.selection import Selection, SelectionBus  # noqa: E402

SITE = LatLon(33.8938, 35.5018)

#: A fixed instant, in whole seconds so the wall clock survives the float trip
#: through `datetime.timestamp()` that the store makes on the way in.
BASE = int(datetime(2026, 8, 30, 3, 0, tzinfo=timezone.utc).timestamp() * 1000)


@pytest.fixture(scope="session")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def make_event(
    *,
    camera: str,
    track: int,
    offset_millis: int,
    event_type: EventType = EventType.ZONE_ENTRY,
    severity: Severity = Severity.MEDIUM,
    zone_id: str | None = "zone-a",
    zone_name: str | None = "Restricted Area A",
    summary: str = "An object entered Restricted Area A",
    class_label: str = "person",
    rule_id: str = "zone-entry",
) -> Event:
    """One event, built the way the engine's own search tests build them."""
    evidence = Evidence(
        camera_id=camera,
        track_id=track,
        first_seen_millis=offset_millis,
        last_seen_millis=offset_millis + 3000,
        observations=17,
        detector="MOG2 background subtraction",
        detector_classifies=False,
        model_digest=None,
        class_label=class_label,
        latitude=SITE.lat,
        longitude=SITE.lon,
        position_uncertainty_meters=1.75,
        position_source="GROUND_PROJECTION",
        speed_mps=1.35,
        heading_degrees=88.5,
        frame_indices=(12, 13, 14),
    )
    return Event(
        id=f"ev_{camera}_{track}_{offset_millis}",
        type=event_type,
        severity=severity,
        summary=summary,
        occurred_at_millis=offset_millis,
        occurred_at=utc_from_millis(BASE + offset_millis),
        zone_id=zone_id,
        zone_name=zone_name,
        rule_id=rule_id,
        evidence=evidence,
        triggering_conditions=("membership held for 600 ms",),
        confidence=0.875,
    )


ENTRY = make_event(camera="cam-01", track=1, offset_millis=0, severity=Severity.HIGH)
LOITER = make_event(
    camera="cam-01",
    track=2,
    offset_millis=60_000,
    event_type=EventType.LOITERING,
    severity=Severity.MEDIUM,
    summary="An object loitered in Restricted Area A",
    rule_id="loitering",
)
AFTER_HOURS = make_event(
    camera="cam-02",
    track=3,
    offset_millis=120_000,
    event_type=EventType.AFTER_HOURS_PRESENCE,
    severity=Severity.CRITICAL,
    zone_id="zone-b",
    zone_name="Loading Bay",
    summary="Presence in the Loading Bay after hours",
    rule_id="after-hours",
)
RAPID = make_event(
    camera="cam-02",
    track=4,
    offset_millis=180_000,
    event_type=EventType.RAPID_MOVEMENT,
    severity=Severity.LOW,
    zone_id=None,
    zone_name=None,
    summary="A vehicle crossed the yard quickly",
    class_label="vehicle",
    rule_id="rapid-movement",
)
UNZONED = make_event(
    camera="cam-03",
    track=5,
    offset_millis=240_000,
    severity=Severity.INFO,
    zone_id=None,
    zone_name=None,
    summary="An object entered the view",
)

#: How many events and incidents the fixture holds. Measured from the store
#: below before anything asserts on them, so a fixture that stopped correlating
#: the way it is described fails here rather than somewhere confusing.
EVENTS_IN_STORE = 5
INCIDENTS_IN_STORE = 2


@pytest.fixture
def store() -> Store:
    """The five events, in two incidents plus one event belonging to neither.

    Correlated in two batches on purpose: the correlator merges anything close
    in time and space, and these five share a position, so one batch would
    produce a single incident and there would be nothing for an incident filter
    to choose between.
    """
    with Store(":memory:") as db:
        first = Correlator().correlate([ENTRY, LOITER])
        second = Correlator().correlate([AFTER_HOURS, RAPID])
        assert len(first) == 1 and len(second) == 1, "the fixture wanted two incidents"
        for incident in first + second:
            db.save_incident(incident)
        db.save_events([UNZONED])
        assert db.event_count() == EVENTS_IN_STORE
        assert db.incident_count() == INCIDENTS_IN_STORE
        yield db


@pytest.fixture
def crowded_store() -> Store:
    """One event more than the largest page this panel will ever ask for.

    Real rows rather than a fabricated `Results`, because the sentence under test
    is about what an operator can actually do next, and only a store that really
    holds more than `MAX_RESULTS` puts the page size at its bound with the answer
    still short.
    """
    with Store(":memory:") as db:
        db.save_events([
            make_event(camera="cam-01", track=n, offset_millis=n * 1000)
            for n in range(MAX_RESULTS + 1)
        ])
        assert db.event_count() == MAX_RESULTS + 1
        yield db


@pytest.fixture
def panel(qt_app, store: Store) -> InvestigationPanel:
    widget = InvestigationPanel(store)
    widget.set_cameras(["cam-01", "cam-02", "cam-03"])
    widget.set_zones([_zone("zone-a", "Restricted Area A"), _zone("zone-b", "Loading Bay")])
    return widget


def _zone(zone_id: str, name: str):
    """The two attributes the picker reads off a zone, and nothing else."""
    from types import SimpleNamespace

    return SimpleNamespace(id=zone_id, name=name)


def rows(widget: InvestigationPanel) -> int:
    return widget.results.topLevelItemCount()


def selections(widget: InvestigationPanel) -> list:
    return [
        widget.results.topLevelItem(index).data(WHEN_COLUMN, Qt.ItemDataRole.UserRole)
        for index in range(widget.results.topLevelItemCount())
    ]


def show_events(widget: InvestigationPanel) -> None:
    widget.subject.setCurrentIndex(widget.subject.findData(SUBJECT_EVENTS))


# ----------------------------------------------------- the record, untouched


def test_an_untouched_panel_shows_the_record_rather_than_an_empty_list(panel):
    # An empty filter bar is not a filter. A panel that showed nothing until
    # somebody typed would teach an operator that the record is empty.
    print(f"untouched: {rows(panel)} row(s), summary {panel.summary.text()!r}")
    assert rows(panel) == INCIDENTS_IN_STORE
    assert str(INCIDENTS_IN_STORE) in panel.summary.text()

    show_events(panel)
    print(f"events: {rows(panel)} row(s), summary {panel.summary.text()!r}")
    assert rows(panel) == EVENTS_IN_STORE
    assert str(EVENTS_IN_STORE) in panel.summary.text()


def test_a_panel_with_no_database_says_so_instead_of_showing_nothing(qt_app):
    widget = InvestigationPanel()
    assert rows(widget) == 0
    assert "no database" in widget.summary.text().lower(), widget.summary.text()


def test_a_panel_built_before_the_node_searches_the_store_it_is_handed(qt_app, store):
    # The console very plausibly builds this panel before the node's store
    # exists. If `set_store` only remembered the database and did not go and read
    # it, the panel would sit in front of a live record still saying there is
    # nothing to search, and nothing on screen would ever contradict it.
    widget = InvestigationPanel()
    assert rows(widget) == 0
    assert "no database" in widget.summary.text().lower()

    widget.set_store(store)
    print(f"after set_store: {rows(widget)} row(s), summary {widget.summary.text()!r}")
    assert rows(widget) == INCIDENTS_IN_STORE
    assert str(INCIDENTS_IN_STORE) in widget.summary.text()
    assert "no database" not in widget.summary.text().lower()


# ------------------------------------------------- the two ways to run one


def test_the_search_button_is_what_it_says_it_is(panel):
    """The button the panel's own promise names, pressed the way a person does.

    Not `panel.search()`: this asserts the connection, which is the part that was
    missing. It also pins the other half of that decision — typing is not a
    search, because a query per keystroke is four full scans while somebody
    spells "loading".
    """
    show_events(panel)
    before = rows(panel)
    panel.term.setText("Loading Bay")
    assert rows(panel) == before, "typing alone must not re-run the search"

    panel.search_button.click()
    print(f'button pressed: {rows(panel)} of {before}, summary {panel.summary.text()!r}')
    assert rows(panel) == 1, "only the after-hours event mentions the loading bay"
    assert rows(panel) < before


def test_enter_in_the_phrase_field_runs_the_search(panel):
    # The other advertised affordance, and the one that reaches `search` as a
    # direct slot — so it is also the path an exception would escape through.
    show_events(panel)
    before = rows(panel)
    panel.term.setText("Loading Bay")
    assert rows(panel) == before

    panel.term.returnPressed.emit()
    print(f"Enter pressed: {rows(panel)} of {before}")
    assert rows(panel) == 1
    assert rows(panel) < before


# ------------------------------------------------------------- every filter


def test_the_camera_filter_narrows_to_that_camera(panel):
    show_events(panel)
    before = rows(panel)
    panel.camera.setCurrentIndex(panel.camera.findData("cam-01"))
    print(f"cam-01: {rows(panel)} of {before}")

    assert rows(panel) == 2, "cam-01 raised exactly the entry and the loitering event"
    assert rows(panel) < before


def test_the_zone_filter_narrows_to_that_zone(panel):
    show_events(panel)
    before = rows(panel)
    panel.zone.setCurrentIndex(panel.zone.findData("zone-b"))
    print(f"zone-b: {rows(panel)} of {before}")

    assert rows(panel) == 1, "only the after-hours event was raised in the loading bay"
    assert rows(panel) < before


def test_the_severity_filter_is_a_floor_and_not_an_exact_match(panel):
    # "HIGH and above" must include the CRITICAL one. A filter that matched HIGH
    # exactly would hide the most serious thing in the database behind the word
    # an operator reaches for first.
    show_events(panel)
    panel.severity.setCurrentIndex(panel.severity.findData(Severity.HIGH.value))
    print(f"HIGH and above: {rows(panel)} row(s)")

    assert rows(panel) == 2, "the HIGH entry and the CRITICAL after-hours event"

    panel.severity.setCurrentIndex(panel.severity.findData(Severity.CRITICAL.value))
    assert rows(panel) == 1


def test_the_phrase_narrows_to_what_actually_says_it(panel):
    show_events(panel)
    before = rows(panel)
    panel.term.setText("Loading Bay")
    panel.search()
    print(f'"Loading Bay": {rows(panel)} of {before}')

    assert rows(panel) == 1
    assert rows(panel) < before


def test_the_time_window_narrows_to_what_happened_inside_it(panel):
    show_events(panel)
    before = rows(panel)
    panel.windowed.setChecked(True)
    panel.start.setDateTime(QDateTime.fromMSecsSinceEpoch(BASE - 1000, QTimeZone.utc()))
    panel.end.setDateTime(QDateTime.fromMSecsSinceEpoch(BASE + 61_000, QTimeZone.utc()))
    panel.search()
    print(f"first minute: {rows(panel)} of {before}")

    assert rows(panel) == 2, "the entry at t+0 and the loitering event at t+60s"
    assert rows(panel) < before


def test_two_filters_narrow_further_than_either_alone(panel):
    show_events(panel)
    panel.camera.setCurrentIndex(panel.camera.findData("cam-02"))
    both_cameras = rows(panel)
    panel.severity.setCurrentIndex(panel.severity.findData(Severity.CRITICAL.value))
    print(f"cam-02: {both_cameras}; cam-02 and CRITICAL: {rows(panel)}")

    assert both_cameras == 2
    assert rows(panel) == 1


def test_a_filter_can_be_taken_off_again(panel):
    # A narrowing that cannot be undone is a panel an operator has to restart.
    show_events(panel)
    panel.camera.setCurrentIndex(panel.camera.findData("cam-01"))
    assert rows(panel) == 2
    panel.camera.setCurrentIndex(panel.camera.findData(None))
    assert rows(panel) == EVENTS_IN_STORE


# -------------------------------------------------- the page is not the answer


def test_a_truncated_page_says_how_many_there_really_were(panel):
    """The central promise: a short page must never read as a complete one.

    This is the assertion that fails if the panel is ever changed to report
    ``len(results.items)`` — which is the number it can see on screen — instead
    of ``Results.total``, which is the number that matched.
    """
    show_events(panel)
    panel.limit.setValue(2)
    text = panel.summary.text()
    print(f"limit 2 of {EVENTS_IN_STORE}: {rows(panel)} row(s), summary {text!r}")

    assert rows(panel) == 2, "the page itself must honour the limit"
    assert "showing 2 of 5" in text, text
    assert str(EVENTS_IN_STORE) in text, "the true total is missing from the count line"


def test_the_page_size_cannot_be_raised_past_the_bound(panel):
    # The search runs on the interface thread, so the page size is also the
    # bound on how long row-building can hold the window still.
    panel.limit.setValue(10_000)
    print(f"asked for 10000, got {panel.limit.value()} (bound {MAX_RESULTS})")
    assert panel.limit.value() == MAX_RESULTS


def test_a_page_already_at_the_bound_does_not_advise_raising_it(qt_app, crowded_store):
    """Honest UI: never offer an operator a move they are not allowed to make.

    Below the bound, "raise the page size" is real advice. At it, the spin box
    refuses to go further, and a line that keeps suggesting it is teaching the
    operator that this line can be ignored — the one line on the panel that says
    the answer is bigger than the screen.
    """
    widget = InvestigationPanel(crowded_store)
    widget.subject.setCurrentIndex(widget.subject.findData(SUBJECT_EVENTS))

    widget.limit.setValue(MAX_RESULTS // 2)
    below = widget.summary.text()
    print(f"below the bound: {below!r}")
    assert "raise the page size" in below

    widget.limit.setValue(MAX_RESULTS)
    at_bound = widget.summary.text()
    print(f"at the bound: {at_bound!r}")
    assert widget.limit.value() == MAX_RESULTS
    assert f"showing {MAX_RESULTS:,} of {MAX_RESULTS + 1:,}" in at_bound, at_bound
    assert "raise the page size" not in at_bound, "advice the spin box will refuse"
    assert "narrowing the search" in at_bound, at_bound


def test_a_complete_page_does_not_claim_to_be_truncated(panel):
    show_events(panel)
    text = panel.summary.text()
    print(f"complete page summary {text!r}")
    assert "showing" not in text.lower(), text
    assert str(EVENTS_IN_STORE) in text


def test_a_search_that_matches_nothing_says_so_rather_than_looking_broken(panel):
    show_events(panel)
    panel.term.setText("a phrase nothing in the record contains")
    panel.search()
    text = panel.summary.text()
    print(f"no match summary {text!r}")

    assert rows(panel) == 0
    assert "no events matched" in text.lower(), text
    assert "searched" in text.lower(), "an empty list must say the record was read"


def test_a_window_that_ends_before_it_starts_is_reported_not_raised(panel):
    # The engine refuses this rather than answering with an empty page, and the
    # refusal must reach the operator: two swapped fields is a thing to say, not
    # an absence of evidence. It must also not escape the slot it was raised in.
    show_events(panel)
    panel.windowed.setChecked(True)
    panel.start.setDateTime(QDateTime.fromMSecsSinceEpoch(BASE + 60_000, QTimeZone.utc()))
    panel.end.setDateTime(QDateTime.fromMSecsSinceEpoch(BASE, QTimeZone.utc()))
    panel.search()
    text = panel.summary.text()
    print(f"reversed window summary {text!r}")

    assert rows(panel) == 0
    assert "cannot be run" in text.lower(), text


def _unbuildable_row(_item):
    """A row this panel cannot draw — a column changed under it, say."""
    raise ValueError("a row this panel cannot draw")


def test_a_row_the_panel_cannot_draw_is_reported_rather_than_raised(panel):
    """The query came back; the drawing failed. That must not leave the slot.

    `search` is the direct target of ``term.returnPressed``, and an exception
    escaping a Qt slot is retained by the interpreter, which pins this widget
    past the QApplication's own destruction and corrupts the heap on the way out.
    Building a row touches a summary, a severity and a set of zones on every item
    that came back, so this is one column change away at any time.
    """
    panel._incident_row = _unbuildable_row

    panel.term.returnPressed.emit()  # through the slot, the way it would happen
    text = panel.summary.text()
    print(f"undrawable rows: {rows(panel)} row(s), summary {text!r}")

    assert rows(panel) == 0, "a half-built list reads as the whole of what matched"
    assert "search failed" in text.lower(), text
    assert "a row this panel cannot draw" in text, "the reason has to reach the line"

    panel.search()  # and again, directly: still returns rather than raising


# ---------------------------------------------------------------- selection


def test_clicking_an_incident_selects_that_incident(panel, store: Store):
    seen: list = []
    panel.selected.connect(seen.append)
    panel.results.topLevelItem(0).setSelected(True)

    expected = selections(panel)[0]
    print(f"clicked row 0, emitted {seen}")
    assert expected.kind == "incident"
    assert store.incident(expected.incident_id) is not None, "an id no store knows"
    assert seen == [expected]
    assert panel.selected_row() == expected


def test_clicking_an_event_selects_the_track_it_is_about(panel):
    # An event is not a selectable thing anywhere else in the console; the track
    # it is about is, and every other panel can point at one. Keyed by camera as
    # well as by number, because #3 on the gate is not #3 on the yard.
    show_events(panel)
    seen: list = []
    panel.selected.connect(seen.append)
    panel.results.topLevelItem(0).setSelected(True)

    print(f"clicked newest event, emitted {seen}")
    assert seen == [Selection.track("cam-03", 5)], "newest first: the cam-03 event"


def test_set_selection_highlights_the_row_without_emitting(panel):
    # The console's selection bus calls this from its own `changed` signal. A
    # re-emit here goes straight back into the bus, and the window stops
    # answering.
    wanted = selections(panel)[1]
    seen: list = []
    panel.selected.connect(seen.append)
    panel.set_selection(wanted)

    assert seen == []
    assert panel.selected_row() == wanted


def test_set_selection_for_something_this_list_cannot_show_clears_it(panel):
    panel.set_selection(selections(panel)[0])
    seen: list = []
    panel.selected.connect(seen.append)
    panel.set_selection(Selection.camera("cam-01"))
    assert panel.selected_row() is None
    panel.set_selection(None)
    assert panel.selected_row() is None
    assert seen == []


def test_a_selection_made_elsewhere_does_not_rewrite_the_filters(panel):
    # A search an operator set up must survive somebody clicking a disc on the
    # map. Re-running it to go and find the selected thing would silently widen
    # the filters they chose.
    show_events(panel)
    panel.camera.setCurrentIndex(panel.camera.findData("cam-01"))
    narrowed = rows(panel)
    panel.set_selection(Selection.track("cam-03", 5))

    assert rows(panel) == narrowed
    assert panel.camera.currentData() == "cam-01"


def test_the_highlighted_row_survives_the_same_search_being_run_again(panel):
    wanted = selections(panel)[0]
    panel.set_selection(wanted)
    seen: list = []
    panel.selected.connect(seen.append)
    panel.search()

    assert panel.selected_row() == wanted
    assert seen == [], "a rebuild is not the operator clicking"


def test_a_clicked_row_reaches_a_real_selection_bus(panel):
    """The connection the console has to make, made here and exercised.

    `selected` is only worth having if what comes out of it is something
    `SelectionBus.select` accepts and holds. This asserts the whole hop —
    click, emit, bus — against the real bus rather than a list, so a Selection
    this panel builds that no other panel could use would fail here.
    """
    bus = SelectionBus()
    panel.selected.connect(bus.select)
    panel.results.topLevelItem(0).setSelected(True)

    expected = selections(panel)[0]
    print(f"bus now holds {bus.current}")
    assert bus.current == expected
    assert bus.current.kind == "incident"


def _undeliverable_selection():
    """A selection that cannot be worked out — a row whose payload is gone."""
    raise RuntimeError("this row is about nothing the console can select")


def test_a_selection_that_cannot_be_delivered_is_reported_rather_than_silent(panel):
    # Swallowing the traceback is right — it must not escape a Qt slot — but
    # swallowing it *silently* leaves an operator clicking rows while every other
    # panel sits on somebody else's choice, with no sign of it anywhere.
    panel.selected_row = _undeliverable_selection
    panel.results.topLevelItem(0).setSelected(True)
    text = panel.summary.text()
    print(f"undeliverable selection: {text!r}")

    assert isinstance(panel.last_selection_failure, RuntimeError)
    assert "could not be selected" in text.lower(), text
    assert "nothing the console can select" in text


def test_silence_nests_so_an_inner_rebuild_cannot_unmute_the_outer_one(panel):
    """The guard is a depth count, not a flag.

    No path in this panel nests these blocks today. One edit makes one — a
    `_fill` inside a `set_selection` inside a rebuild — and with a flag the inner
    `finally` would clear the outer guard half way through, so the rest of the
    outer block would emit `selected` for a row nobody clicked and push every
    other panel onto it.
    """
    seen: list = []
    panel.selected.connect(seen.append)

    with panel._silent():
        panel.set_selection(None)  # an inner block, using the same guard
        panel.results.topLevelItem(0).setSelected(True)

    print(f"emitted from inside nested silence: {seen}")
    assert seen == [], "the inner block unmuted its caller"
    assert panel._quiet == 0, "the guard did not unwind"


# ------------------------------------------------------------------ pickers


def test_filling_the_pickers_neither_emits_nor_loses_the_current_choice(panel):
    show_events(panel)
    panel.camera.setCurrentIndex(panel.camera.findData("cam-02"))
    seen: list = []
    panel.selected.connect(seen.append)
    panel.set_cameras(["cam-01", "cam-02", "cam-03", "cam-04"])

    assert panel.camera.currentData() == "cam-02"
    assert rows(panel) == 2
    assert seen == []


def test_the_zone_picker_shows_names_and_searches_ids(panel):
    # A zone can be renamed. A filter that matched the stored name would stop
    # finding the incidents it found yesterday.
    index = panel.zone.findData("zone-b")
    assert panel.zone.itemText(index) == "Loading Bay"
    show_events(panel)
    panel.zone.setCurrentIndex(index)
    assert panel.query().zones == ("zone-b",)


# --------------------------------------------------------------- lifetime


def test_the_panel_is_freed_when_its_last_reference_goes(qt_app, store: Store):
    # A reference cycle holding a QWidget means the widget is destroyed at
    # interpreter shutdown, after PySide has torn the QApplication down, which
    # corrupts the heap and kills the process with 0xC0000374 at exit — after
    # every test has passed. A lambda closing over `self` in a signal connection
    # is how that happens, so every connection inside the panel is a bound
    # method.
    widget = InvestigationPanel(store)
    widget.set_cameras(["cam-01"])
    widget.search()
    ref = weakref.ref(widget)
    del widget
    gc.collect()
    survivor = ref()
    if survivor is not None:
        holders = sorted(
            {type(r).__name__ for r in gc.get_referrers(survivor)} - {"frame", "list"}
        )
        del survivor
        raise AssertionError(
            f"InvestigationPanel outlived its last reference (held by {holders})."
        )


def test_the_subject_switch_asks_a_different_question_of_the_same_store(panel):
    # Incidents are the conclusions and events are the grounds for them; a panel
    # that could only show one of them cannot answer "was that event ever
    # concluded to be anything".
    assert panel.subject_kind == SUBJECT_INCIDENTS
    assert rows(panel) == INCIDENTS_IN_STORE
    show_events(panel)
    assert panel.subject_kind == SUBJECT_EVENTS
    assert rows(panel) == EVENTS_IN_STORE
