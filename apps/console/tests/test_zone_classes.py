"""Tests for the class filter on the zone properties panel and the zones list.

The failure this filter exists to prevent was photographed on a real laptop
camera: a RESTRICTED zone raised "1 couch in Room (HIGH, risk 55)", because a
restricted zone with no filter fires on every class the segmenter can name.
What is checked hardest here is what the operator *reads*:

- An empty filter must read as "any object", never as an empty box. A blank
  beside "restricted" looks like a zone watching nothing, when it is the zone
  watching everything, couch included.
- A motion-only site must not offer a filter at all, and must say why in one
  sentence: a motion detector cannot name what it saw, so a filter would
  silence the zone entirely.
- Ticks must not emit ``changed``. The console persists every emission through
  the node, and the first version of the schedule fields taught this lesson:
  one emission per keystroke is one audited edit per keystroke.
- "Not known" is not "labels nothing". A panel nobody has briefed about the
  detector must show a stored filter as stored and not judge it; the first
  version judged it, called a live "person" filter dead, and offered a button
  that would have reinstated the couch.
- The row an operator most needs to see — a tick the detector cannot honour
  — must be inside the picker's viewport, not below the fold of a scrolled
  list. Layout is measured here, offscreen, not assumed.

The panel is built with a stand-in zone rather than the engine's ``Zone``: the
``classes`` field lands on the engine separately, and the panel is required to
work both before and after it does.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from sentinel.core import LatLon  # noqa: E402
from sentinel.zones import ZoneKind  # noqa: E402

from sentinel_console.zones_view import (  # noqa: E402
    ANY_OBJECT_SUMMARY,
    MOTION_ONLY_EXPLANATION,
    PICKER_ROWS,
    VOCABULARY_UNKNOWN_CAPTION,
    WATCHES_ANY,
    WATCHES_COLUMN,
    ZonePropertiesPanel,
    ZonesView,
    watches_label,
    zone_has_classes,
)


@pytest.fixture(scope="session")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


#: Three corners about 10 m apart: enough for the list's "Across" column to
#: have something to measure, which is all the ring is for here.
RING = (
    LatLon(51.5000, -0.1000),
    LatLon(51.5001, -0.1000),
    LatLon(51.5001, -0.0999),
)

#: What a segmenter on the laptop actually said it could name, couch included.
VOCABULARY = ("person", "car", "truck", "bottle", "couch")


@dataclass(frozen=True)
class ZoneWithClasses:
    """A stand-in for ``sentinel.zones.Zone`` once it carries ``classes``.

    Frozen and a dataclass, because ``zone_from_fields`` goes through
    :func:`dataclasses.replace` and the test must exercise that path, not a
    friendlier one.
    """

    id: str
    name: str
    kind: ZoneKind
    ring: tuple = RING
    schedule: object = None
    enter_after_millis: int = 600
    exit_after_millis: int = 2000
    accept_uncertain: bool = False
    classes: frozenset = field(default_factory=frozenset)


@dataclass(frozen=True)
class ZoneWithoutClasses:
    """The engine's zone as it is until the ``classes`` field lands."""

    id: str
    name: str
    kind: ZoneKind
    ring: tuple = RING
    schedule: object = None
    enter_after_millis: int = 600
    exit_after_millis: int = 2000
    accept_uncertain: bool = False


def panel_showing(zone, vocabulary=VOCABULARY) -> ZonePropertiesPanel:
    panel = ZonePropertiesPanel()
    panel.set_classes(vocabulary)
    panel.show_zone(zone)
    return panel


def tick(panel: ZonePropertiesPanel, *labels: str) -> None:
    """Tick classes the way an operator does: through the item, so the
    picker's own ``itemChanged`` signal fires."""
    wanted = set(labels)
    for index in range(panel.class_picker.count()):
        item = panel.class_picker.item(index)
        if item.data(Qt.ItemDataRole.UserRole) in wanted:
            item.setCheckState(Qt.CheckState.Checked)


def offered(panel: ZonePropertiesPanel) -> list[str]:
    """The labels the picker offers, top to bottom, by the data each carries."""
    return [
        panel.class_picker.item(i).data(Qt.ItemDataRole.UserRole)
        for i in range(panel.class_picker.count())
    ]


def item_for(panel: ZonePropertiesPanel, label: str):
    for index in range(panel.class_picker.count()):
        item = panel.class_picker.item(index)
        if item.data(Qt.ItemDataRole.UserRole) == label:
            return item
    raise AssertionError(f"{label!r} is not offered")


def laid_out(qt_app, panel: ZonePropertiesPanel) -> None:
    """Show the panel and let Qt lay it out, so viewport rectangles are real."""
    panel.resize(480, 640)
    panel.show()
    qt_app.processEvents()


def within_viewport(panel: ZonePropertiesPanel, item) -> bool:
    return panel.class_picker.viewport().rect().contains(
        panel.class_picker.visualItemRect(item)
    )


# --------------------------------------------------------------- the default


def test_the_default_filter_reads_as_any_object_not_as_an_empty_box(qt_app):
    panel = panel_showing(ZoneWithClasses("z1", "Room", ZoneKind.RESTRICTED))

    assert panel.selected_classes() == frozenset()
    assert panel.class_summary.text() == ANY_OBJECT_SUMMARY
    assert "any object" in panel.class_summary.text().lower()
    assert not panel._classes_box.isHidden()
    assert panel.class_picker.isEnabled()
    # Every class the detector names is offered, in the detector's order.
    offered = [
        panel.class_picker.item(i).data(Qt.ItemDataRole.UserRole)
        for i in range(panel.class_picker.count())
    ]
    assert offered == list(VOCABULARY)
    assert not panel.class_note.isVisibleTo(panel), "nothing to warn about yet"


def test_zone_from_fields_carries_an_empty_frozenset_for_the_default(qt_app):
    panel = panel_showing(ZoneWithClasses("z1", "Room", ZoneKind.RESTRICTED))
    zone = panel.zone_from_fields()
    assert zone.classes == frozenset()
    assert isinstance(zone.classes, frozenset)


# ------------------------------------------------------------ round-tripping


def test_two_ticked_classes_round_trip_through_zone_from_fields(qt_app):
    panel = panel_showing(ZoneWithClasses("z1", "Room", ZoneKind.RESTRICTED))
    tick(panel, "person", "car")

    zone = panel.zone_from_fields()
    assert zone.classes == frozenset({"person", "car"})
    assert isinstance(zone.classes, frozenset)
    # Everything else survives untouched.
    assert (zone.id, zone.name, zone.kind) == ("z1", "Room", ZoneKind.RESTRICTED)
    assert zone.enter_after_millis == 600 and zone.exit_after_millis == 2000

    # And the summary says so, in a stable order.
    assert panel.class_summary.text() == "Only: car, person"


def test_a_stored_filter_is_shown_ticked_and_survives_apply_unchanged(qt_app):
    stored = ZoneWithClasses("z1", "Room", ZoneKind.RESTRICTED, classes=frozenset({"truck"}))
    panel = panel_showing(stored)

    assert panel.selected_classes() == frozenset({"truck"})
    assert panel.zone_from_fields().classes == frozenset({"truck"})


def test_watch_any_object_clears_the_filter(qt_app):
    stored = ZoneWithClasses("z1", "Room", ZoneKind.RESTRICTED, classes=frozenset({"truck"}))
    panel = panel_showing(stored)
    assert panel.any_object_button.isEnabled()

    panel.watch_any_object()

    assert panel.selected_classes() == frozenset()
    assert panel.zone_from_fields().classes == frozenset()
    assert panel.class_summary.text() == ANY_OBJECT_SUMMARY
    assert not panel.any_object_button.isEnabled(), "nothing left to clear"


def test_revert_puts_the_ticks_back_to_the_zone_as_stored(qt_app):
    stored = ZoneWithClasses("z1", "Room", ZoneKind.RESTRICTED, classes=frozenset({"person"}))
    panel = panel_showing(stored)
    tick(panel, "couch")
    assert panel.selected_classes() == frozenset({"person", "couch"})

    panel.revert()

    assert panel.selected_classes() == frozenset({"person"})


def test_the_vocabulary_arriving_after_the_zone_keeps_the_ticks(qt_app):
    """The orchestrator may learn the detector's labels after a zone is shown."""
    panel = ZonePropertiesPanel()
    panel.show_zone(ZoneWithClasses("z1", "Room", ZoneKind.RESTRICTED, classes=frozenset({"car"})))
    panel.set_classes(VOCABULARY)

    assert panel.selected_classes() == frozenset({"car"})
    assert panel.class_picker.isEnabled()
    assert not panel.class_note.isVisibleTo(panel), "car is in the vocabulary"


# ------------------------------------------------------------------ the list


def test_the_list_column_shows_the_filter(qt_app):
    view = ZonesView()
    view.show_zones([
        ZoneWithClasses("any", "Yard", ZoneKind.RESTRICTED),
        ZoneWithClasses("two", "Gate", ZoneKind.RESTRICTED, classes=frozenset({"person", "car"})),
        ZoneWithClasses("five", "Lot", ZoneKind.INTEREST, classes=frozenset(VOCABULARY)),
    ])
    rows = {view.topLevelItem(i).text(0): view.topLevelItem(i) for i in range(view.topLevelItemCount())}

    assert view.headerItem().text(WATCHES_COLUMN) == "Watches"
    assert rows["Yard"].text(WATCHES_COLUMN) == WATCHES_ANY == "any"
    assert rows["Gate"].text(WATCHES_COLUMN) == "car, person"
    # Five names truncate to three and a count; the tooltip carries them all.
    assert rows["Lot"].text(WATCHES_COLUMN) == "bottle, car, couch +2"
    for label in VOCABULARY:
        assert label in rows["Lot"].toolTip(WATCHES_COLUMN)

    # The columns other tests count on are where they were.
    assert rows["Gate"].text(1) == "restricted"
    assert rows["Gate"].text(3) == "—"


def test_the_list_says_any_for_a_zone_that_has_no_classes_field_yet(qt_app):
    view = ZonesView()
    view.show_zones([ZoneWithoutClasses("old", "Yard", ZoneKind.RESTRICTED)])
    assert view.topLevelItem(0).text(WATCHES_COLUMN) == WATCHES_ANY


def test_watches_label_truncates_with_a_count_and_never_reorders(qt_app):
    assert watches_label(()) == "any"
    assert watches_label({"person"}) == "person"
    assert watches_label({"truck", "car", "person"}) == "car, person, truck"
    assert watches_label({"truck", "car", "person", "bus"}) == "bus, car, person +1"
    assert watches_label(["b", "a"], shown=1) == "a +1"


# ------------------------------------------------------------- motion only


def test_a_motion_only_site_disables_the_picker_and_says_why(qt_app):
    panel = panel_showing(ZoneWithClasses("z1", "Room", ZoneKind.RESTRICTED), vocabulary=())

    assert not panel.class_picker.isEnabled()
    assert panel.class_picker.count() == 0
    assert panel.class_caption.text() == MOTION_ONLY_EXPLANATION
    # The one sentence an operator needs, and the reason inside it.
    assert "cannot name what it saw" in MOTION_ONLY_EXPLANATION
    assert "silence the zone entirely" in MOTION_ONLY_EXPLANATION
    # The zone still reads as watching any object — it is, and honestly so.
    assert panel.class_summary.text() == ANY_OBJECT_SUMMARY
    assert panel.zone_from_fields().classes == frozenset()


def test_a_filter_the_detector_cannot_name_is_flagged_not_silently_kept(qt_app):
    """A zone narrowed to "forklift" on a segmenter that never says forklift
    watches nothing, and the panel must say so rather than tick a box."""
    stored = ZoneWithClasses("z1", "Bay", ZoneKind.RESTRICTED, classes=frozenset({"forklift"}))
    panel = panel_showing(stored, vocabulary=("person", "car"))

    # Flagged first: the row the operator must see is never below the fold.
    assert offered(panel) == ["forklift", "person", "car"]
    flagged = panel.class_picker.item(0)
    assert flagged.checkState() == Qt.CheckState.Checked
    assert "not named by this detector" in flagged.text()
    assert panel.class_note.isVisibleTo(panel)
    assert "forklift" in panel.class_note.text()
    assert "watches nothing" in panel.class_note.text()
    # The panel does not rewrite what it was not asked to: Apply keeps it.
    assert panel.zone_from_fields().classes == frozenset({"forklift"})


def test_a_silencing_filter_stays_reachable_on_a_motion_only_site(qt_app):
    """Disabled would mean the operator can see the zone is silenced and
    cannot un-silence it."""
    stored = ZoneWithClasses("z1", "Bay", ZoneKind.RESTRICTED, classes=frozenset({"person"}))
    panel = panel_showing(stored, vocabulary=())

    assert panel.class_picker.isEnabled()
    assert panel.class_caption.text() == MOTION_ONLY_EXPLANATION
    assert panel.class_note.isVisibleTo(panel)
    panel.watch_any_object()
    assert panel.zone_from_fields().classes == frozenset()
    assert not panel.class_picker.isEnabled(), "nothing left to untick"


# ------------------------------------------------------ vocabulary not known


def test_a_stored_filter_on_an_unbriefed_panel_is_shown_as_stored_and_not_judged(qt_app):
    """The console as shipped never told the panel what the detector names.
    Reading that silence as "labels nothing" presented a zone that watched
    persons as watching nothing, with a button to make it watch the couch."""
    panel = ZonePropertiesPanel()  # set_classes never called
    panel.show_zone(ZoneWithClasses("z1", "Room", ZoneKind.RESTRICTED, classes=frozenset({"person"})))

    assert offered(panel) == ["person"]
    assert "not named" not in item_for(panel, "person").text()
    assert not panel.class_note.isVisibleTo(panel), "nothing has been judged"
    assert panel.class_caption.text() != MOTION_ONLY_EXPLANATION
    assert panel.class_caption.text() == VOCABULARY_UNKNOWN_CAPTION
    assert "not known" in VOCABULARY_UNKNOWN_CAPTION
    assert panel.class_summary.text() == "Only: person"
    # The tick can still be removed — that is a decision the operator may
    # take without the vocabulary — and Apply keeps what was stored.
    assert panel.class_picker.isEnabled()
    assert panel.any_object_button.isEnabled()
    assert panel.zone_from_fields().classes == frozenset({"person"})


def test_the_same_filter_is_flagged_once_the_panel_learns_the_detector_labels_nothing(qt_app):
    panel = ZonePropertiesPanel()
    panel.show_zone(ZoneWithClasses("z1", "Room", ZoneKind.RESTRICTED, classes=frozenset({"person"})))
    assert not panel.class_note.isVisibleTo(panel)

    panel.set_classes(())

    assert "not named by this detector" in item_for(panel, "person").text()
    assert panel.class_note.isVisibleTo(panel)
    assert "watches nothing" in panel.class_note.text()
    assert panel.class_caption.text() == MOTION_ONLY_EXPLANATION
    assert panel.selected_classes() == frozenset({"person"}), "still not rewritten"


def test_the_engine_zone_on_an_unbriefed_panel_is_not_called_dead(qt_app):
    """The finding was reproduced with the real Zone; so is the fix."""
    from sentinel.zones import Zone

    if not zone_has_classes(_real_zone()):
        pytest.skip("the engine's Zone has no classes field yet")
    zone = Zone(id="room", name="Room", kind=ZoneKind.RESTRICTED, ring=RING, classes=frozenset({"person"}))
    panel = ZonePropertiesPanel()
    panel.show_zone(zone)

    assert item_for(panel, "person").text() == "person"
    assert not panel.class_note.isVisibleTo(panel)
    assert panel.class_caption.text() == VOCABULARY_UNKNOWN_CAPTION
    assert panel.zone_from_fields().watches("person")


def test_set_classes_none_means_not_known_and_does_not_raise(qt_app):
    """``runner.detector_info`` is None before a session starts; the
    orchestrator must be able to say so without a TypeError."""
    panel = panel_showing(ZoneWithClasses("z1", "Bay", ZoneKind.RESTRICTED, classes=frozenset({"forklift"})),
                          vocabulary=("person", "car"))
    assert panel.class_note.isVisibleTo(panel), "judged against a real vocabulary"

    panel.set_classes(None)

    assert panel.selected_classes() == frozenset({"forklift"})
    assert offered(panel) == ["forklift"], "only what is stored: nothing to offer, nothing to judge"
    assert "not named" not in item_for(panel, "forklift").text()
    assert not panel.class_note.isVisibleTo(panel)
    assert panel.class_caption.text() == VOCABULARY_UNKNOWN_CAPTION


def test_an_unbriefed_panel_with_no_filter_reads_as_any_object_and_hides_the_empty_box(qt_app):
    panel = ZonePropertiesPanel()
    panel.show_zone(ZoneWithClasses("z1", "Room", ZoneKind.RESTRICTED))

    assert panel.class_summary.text() == ANY_OBJECT_SUMMARY
    assert panel.class_caption.text() == VOCABULARY_UNKNOWN_CAPTION
    assert panel.class_picker.count() == 0
    assert panel.class_picker.isHidden(), "an empty box reads as a zone watching nothing"
    assert not panel.any_object_button.isEnabled()
    assert panel.zone_from_fields().classes == frozenset()


# ---------------------------------------------------------------- layout


def test_the_flagged_row_is_inside_the_viewport_even_below_a_long_vocabulary(qt_app):
    """An 80-class segmenter is the normal case. Appended after it, the one
    row the operator must see sat below the fold of a scrolled list."""
    long_vocabulary = tuple(f"class_{i}" for i in range(80))
    stored = ZoneWithClasses("z1", "Bay", ZoneKind.RESTRICTED, classes=frozenset({"forklift"}))
    panel = panel_showing(stored, vocabulary=long_vocabulary)
    laid_out(qt_app, panel)

    flagged = item_for(panel, "forklift")
    assert panel.class_picker.row(flagged) == 0
    assert within_viewport(panel, flagged)
    # The picker scrolls inside itself past PICKER_ROWS rows rather than
    # pushing the schedule and dwell fields off the panel.
    assert panel.class_picker.verticalScrollBar().maximum() > 0
    assert not within_viewport(panel, item_for(panel, long_vocabulary[-1]))


def test_a_vocabulary_up_to_the_row_cap_shows_every_row_without_scrolling(qt_app):
    """Sized from the picker's own row height, not a literal pixel count."""
    vocabulary = tuple(f"class_{i}" for i in range(PICKER_ROWS))
    panel = panel_showing(ZoneWithClasses("z1", "Room", ZoneKind.RESTRICTED), vocabulary=vocabulary)
    laid_out(qt_app, panel)

    assert panel.class_picker.count() == PICKER_ROWS
    assert panel.class_picker.verticalScrollBar().maximum() == 0
    for label in vocabulary:
        assert within_viewport(panel, item_for(panel, label)), label
    row = panel.class_picker.sizeHintForRow(0)
    assert row > 0
    assert panel.class_picker.height() == PICKER_ROWS * row + 2 * panel.class_picker.frameWidth()

    # And a shorter vocabulary gets a shorter picker, not the cap's height.
    panel.set_classes(VOCABULARY)
    qt_app.processEvents()
    assert panel.class_picker.height() == len(VOCABULARY) * row + 2 * panel.class_picker.frameWidth()


def test_the_points_column_is_as_wide_as_its_own_header_needs(qt_app):
    """Measured from the header's font, so "Points" is never elided on a
    large font nor given room the Watches column needed on a small one."""
    view = ZonesView()
    header = view.header()
    assert view.columnWidth(4) == header.sectionSizeHint(4)
    assert view.columnWidth(4) >= header.fontMetrics().horizontalAdvance("Points")


# -------------------------------------------------------- before the field lands


def test_a_zone_without_a_classes_field_hides_the_picker_and_is_not_handed_one(qt_app):
    panel = panel_showing(ZoneWithoutClasses("old", "Yard", ZoneKind.RESTRICTED))

    assert panel._classes_box.isHidden()
    assert not zone_has_classes(ZoneWithoutClasses("old", "Yard", ZoneKind.RESTRICTED))
    assert zone_has_classes(ZoneWithClasses("new", "Yard", ZoneKind.RESTRICTED))
    # `replace` with an unknown field raises; the panel must not try.
    zone = panel.zone_from_fields()
    assert not hasattr(zone, "classes")
    assert zone.name == "Yard"


def test_showing_a_newer_zone_after_an_older_one_brings_the_picker_back(qt_app):
    panel = panel_showing(ZoneWithoutClasses("old", "Yard", ZoneKind.RESTRICTED))
    assert panel._classes_box.isHidden()
    panel.show_zone(ZoneWithClasses("new", "Gate", ZoneKind.RESTRICTED))
    assert not panel._classes_box.isHidden()
    assert panel.zone_from_fields().classes == frozenset()


# --------------------------------------------------------------- silence rules


def test_apply_emits_once_and_show_zone_and_ticks_emit_nothing(qt_app):
    panel = ZonePropertiesPanel()
    emitted: list = []

    def record(zone):
        emitted.append(zone)

    panel.changed.connect(record)
    panel.set_classes(VOCABULARY)
    panel.show_zone(ZoneWithClasses("z1", "Room", ZoneKind.RESTRICTED, classes=frozenset({"car"})))
    assert emitted == [], "showing a zone is not an edit"

    tick(panel, "person", "truck")
    panel.watch_any_object()
    tick(panel, "person")
    panel.set_classes(("person", "car"))
    panel.revert()
    assert emitted == [], "ticking, clearing and reverting are not edits either"

    tick(panel, "person")
    panel.apply()
    assert len(emitted) == 1
    assert emitted[0].classes == frozenset({"car", "person"})

    panel.show_zone(None)
    panel.apply()
    assert len(emitted) == 1, "with nothing shown there is nothing to apply"


# ------------------------------------------------------------ the real Zone


def _real_zone():
    from sentinel.zones import Zone

    return Zone(id="room", name="Room", kind=ZoneKind.RESTRICTED, ring=RING)


@pytest.mark.skipif(
    not zone_has_classes(_real_zone()), reason="the engine's Zone has no classes field yet"
)
def test_the_engine_zone_narrowed_to_person_no_longer_watches_a_couch(qt_app):
    """The incident this run exists for, closed end to end: "1 couch in Room
    (HIGH)" came from a restricted zone that watched anything the segmenter
    named. Narrowed through the panel, the engine's own ``watches`` says no."""
    panel = panel_showing(_real_zone(), vocabulary=("person", "couch", "bottle"))
    assert panel.zone.watches("couch"), "unfiltered, the zone watches everything named"

    tick(panel, "person")
    zone = panel.zone_from_fields()

    assert zone.classes == frozenset({"person"})
    assert zone.watches("person")
    assert not zone.watches("couch")
    assert not zone.watches("bottle")
    view = ZonesView()
    view.show_zones([zone])
    assert view.topLevelItem(0).text(WATCHES_COLUMN) == "person"
