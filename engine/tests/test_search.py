"""Tests for the investigation surface.

Four groups, in descending order of how badly a failure would matter:

**The search box must not be a way into the database.** A term is the only place
in this system where a person's typing reaches SQL, so a term shaped like an
injection is searched for literally and changes nothing. A term containing a LIKE
wildcard is the quieter half of the same failure and is tested beside it.

**A page must not read as the whole answer.** Every assertion about a page also
asserts the total beside it. A result that silently truncates is a wrong answer
wearing the clothes of a complete one.

**A filter must narrow and never widen**, alone and in combination, and an unset
filter must not narrow at all — a panel nobody has touched shows the record.

**An answer must not lose what the store kept.** A search that quietly drops a
field looks complete and is not.

Every count here is printed before it is asserted, so a failure says what the
number actually was rather than only that it was wrong.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from sentinel.core import LatLon
from sentinel.events import Event, Evidence, EventType, Severity, utc_from_millis
from sentinel.incidents import Correlator
from sentinel.search import (
    DEFAULT_LIMIT,
    MomentKind,
    Query,
    SearchError,
    Window,
    at_least,
    search_events,
    search_incidents,
    timeline,
)
from sentinel.store import Store

SITE = LatLon(33.8938, 35.5018)

#: A fixed instant, in whole seconds so the wall clock survives the float trip
#: through `datetime.timestamp()` that the store makes on the way in.
BASE = int(datetime(2026, 8, 30, 3, 0, tzinfo=timezone.utc).timestamp() * 1000)


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
    position: LatLon | None = SITE,
) -> Event:
    """One event, built the way `test_store.py` builds them.

    ``offset_millis`` is both the media time and the offset of the wall clock
    from `BASE`, which is what a source running at real time produces — and it
    is what lets a window assertion below name an instant rather than reciting a
    stored column back at itself.
    """
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
        latitude=position.lat if position else None,
        longitude=position.lon if position else None,
        position_uncertainty_meters=1.75 if position else None,
        position_source="GROUND_PROJECTION" if position else None,
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


#: Five events across three cameras, four types, four severities and two zones,
#: so that every filter has something it must include and something it must
#: leave out. Ordered oldest first; the searches must hand them back reversed.
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

ALL_EVENTS = (ENTRY, LOITER, AFTER_HOURS, RAPID, UNZONED)


@pytest.fixture
def blank() -> Store:
    """An empty database, for the tests that supply their own odd rows."""
    with Store(":memory:") as db:
        yield db


@pytest.fixture
def store(blank: Store) -> Store:
    """The five events, in two incidents plus one event belonging to neither.

    Correlated in two batches on purpose. The correlator merges anything close
    in time and space, and these five share a position, so one batch would
    produce a single incident and there would be nothing for an incident filter
    to choose between.
    """
    first = Correlator().correlate([ENTRY, LOITER])
    second = Correlator().correlate([AFTER_HOURS, RAPID])
    assert len(first) == 1 and len(second) == 1, "the fixture wanted two incidents"

    for incident in first + second:
        blank.save_incident(incident)
    blank.save_events([UNZONED])
    return blank


@pytest.fixture
def incidents(store: Store):
    return search_incidents(store).items


def ids(results) -> list[str]:
    return [item.id for item in results]


# ------------------------------------------------------- the term is not SQL


def test_a_term_containing_a_quote_and_a_semicolon_is_matched_literally(blank: Store):
    """The search box is an injection surface, so it is tested like one.

    Two halves. The term must not execute — the table it names is still there
    afterwards, with every row in it — and it must still *work*: the event whose
    summary really does contain that text is found, which is what proves the term
    reached the database as data rather than being stripped on the way.
    """
    nasty = "Robert'); DROP TABLE events;--"
    blank.save_events(
        [
            make_event(camera="cam-01", track=1, offset_millis=0, summary=nasty),
            make_event(camera="cam-01", track=2, offset_millis=1000),
        ]
    )
    before = blank.event_count()

    found = search_events(blank, Query(term=nasty))
    print(f"term {nasty!r} matched {found.total} of {before} event(s)")

    assert found.total == 1, "the literal term matched the wrong number of rows"
    assert found.items[0].summary == nasty
    assert "events" in blank.table_names(), "the term reached the database as SQL"
    assert blank.event_count() == before, "the term changed the record"


def test_an_injection_shaped_term_nobody_wrote_down_simply_matches_nothing(
    store: Store,
):
    """The other half: a term matching no row is an empty page, not an error.

    And still not a statement. Both searches are run, because the incident query
    puts the same term into five bound parameters across two subqueries and a
    mistake in any one of them would be a hole nobody looked at.
    """
    nasty = "'; DELETE FROM incidents WHERE '1'='1"
    events_before, incidents_before = store.event_count(), store.incident_count()

    matched_events = search_events(store, Query(term=nasty))
    matched_incidents = search_incidents(store, Query(term=nasty))
    print(
        f"events {matched_events.total}, incidents {matched_incidents.total}, "
        f"record {store.event_count()}/{store.incident_count()} "
        f"(was {events_before}/{incidents_before})"
    )

    assert matched_events.total == 0
    assert matched_incidents.total == 0
    assert store.event_count() == events_before
    assert store.incident_count() == incidents_before


def test_a_term_containing_a_wildcard_matches_the_wildcard_itself(blank: Store):
    """`_` and `%` are LIKE wildcards, and a term is not a pattern.

    Unescaped, a search for the fragment ``AB_12`` also returns ``AB712`` — a
    result the operator cannot tell from a real one — and a lone ``%`` returns
    the entire table while looking like a filter that narrowed it.
    """
    blank.save_events(
        [
            make_event(camera="cam-01", track=1, offset_millis=0, summary="plate AB_12"),
            make_event(camera="cam-01", track=2, offset_millis=1000, summary="plate AB712"),
        ]
    )

    underscore = search_events(blank, Query(term="AB_12"))
    everything = search_events(blank, Query(term="%"))
    print(f"AB_12 matched {underscore.total}; a bare % matched {everything.total}")

    assert underscore.total == 1, "the underscore was treated as a wildcard"
    assert underscore.items[0].summary == "plate AB_12"
    assert everything.total == 0, "a bare % returned rows it does not appear in"


# ---------------------------------------------------- a page is not the answer


def test_an_empty_query_returns_everything_up_to_the_limit(store: Store):
    everything = search_events(store)
    print(f"{len(everything)} item(s), total {everything.total}, {everything.describe()}")

    assert len(everything) == len(ALL_EVENTS)
    assert everything.total == len(ALL_EVENTS)
    assert not everything.truncated
    assert Query().is_empty, "an untouched panel must not be filtering"


def test_a_page_says_how_many_it_left_behind(store: Store):
    """"Showing 2 of 5" rather than five rows silently becoming two."""
    page = search_events(store, Query(limit=2))
    print(f"{page.describe()} — items {len(page)}, total {page.total}")

    assert len(page) == 2
    assert page.total == len(ALL_EVENTS)
    assert page.truncated
    assert page.describe() == "showing 2 of 5"


def test_the_total_counts_what_matched_not_what_was_returned(store: Store):
    """The count must be of the filtered set, not of the table."""
    page = search_events(store, Query(cameras="cam-01", limit=1))
    print(f"cam-01: {page.describe()} of {store.event_count()} stored")

    assert len(page) == 1
    assert page.total == 2, "the total counted rows the filter excluded"
    assert page.total < store.event_count()


def test_events_come_back_newest_first(store: Store):
    """A list is read from the top, so the recent thing belongs there."""
    found = search_events(store)
    times = [event.occurred_at_millis for event in found]
    print(f"media times, in returned order: {times}")

    assert times == sorted(times, reverse=True)
    assert ids(found)[0] == UNZONED.id


# ----------------------------------------------------------- one filter alone


def test_filtering_by_camera_returns_only_that_cameras_events(store: Store):
    found = search_events(store, Query(cameras=["cam-01"]))
    print(f"cam-01 matched {found.total}: {ids(found)}")

    assert found.total == 2
    assert {e.evidence.camera_id for e in found} == {"cam-01"}


def test_filtering_by_event_type_returns_only_that_type(store: Store):
    found = search_events(store, Query(types=[EventType.LOITERING]))
    print(f"LOITERING matched {found.total}: {ids(found)}")

    assert found.total == 1
    assert found.items[0].id == LOITER.id


def test_filtering_by_severity_returns_only_those_severities(store: Store):
    """`at_least` is the filter an operator actually means by "and above"."""
    serious = at_least(Severity.HIGH)
    found = search_events(store, Query(severities=serious))
    print(f"{[s.value for s in serious]} matched {found.total}: {ids(found)}")

    assert serious == (Severity.HIGH, Severity.CRITICAL)
    assert found.total == 2
    assert {e.id for e in found} == {ENTRY.id, AFTER_HOURS.id}


def test_filtering_by_zone_returns_only_events_raised_inside_it(store: Store):
    found = search_events(store, Query(zones="zone-b"))
    print(f"zone-b matched {found.total}: {ids(found)}")

    assert found.total == 1
    assert found.items[0].zone_name == "Loading Bay"


def test_filtering_by_a_window_returns_only_what_happened_inside_it(store: Store):
    """Cut on the wall clock, which is also what the results are ordered by.

    The window is named in absolute time and the assertion is about which events
    fall in it — not about a column read back out of the row that produced it,
    which would pass even if the store and this module disagreed about which
    clock they were using.
    """
    window = Window(BASE + 60_000, BASE + 180_000)
    found = search_events(store, Query(window=window))
    print(f"{window.duration_millis / 1000:.0f}s window matched {found.total}: {ids(found)}")

    assert found.total == 3
    assert {e.id for e in found} == {LOITER.id, AFTER_HOURS.id, RAPID.id}
    assert window.contains(BASE + 60_000), "both ends of a window are inside it"


def test_filtering_by_a_term_matches_any_of_the_columns_it_names(store: Store):
    """Summary, zone name, class label, camera id and rule id, and no others."""
    by_summary = search_events(store, Query(term="loitered"))
    by_zone_name = search_events(store, Query(term="loading bay"))
    by_label = search_events(store, Query(term="vehicle"))
    by_camera = search_events(store, Query(term="cam-03"))
    print(
        f"summary {by_summary.total}, zone name {by_zone_name.total}, "
        f"label {by_label.total}, camera {by_camera.total}"
    )

    assert by_summary.total == 1 and by_summary.items[0].id == LOITER.id
    # Case-insensitive: an operator does not type a zone's capitalisation.
    assert by_zone_name.total == 1 and by_zone_name.items[0].id == AFTER_HOURS.id
    assert by_label.total == 1 and by_label.items[0].id == RAPID.id
    assert by_camera.total == 1 and by_camera.items[0].id == UNZONED.id


def test_a_blank_search_box_is_not_a_filter(store: Store):
    """Whitespace is what a search box holds after somebody clears it."""
    found = search_events(store, Query(term="   "))
    print(f"a whitespace term matched {found.total}")

    assert found.total == len(ALL_EVENTS)
    assert Query(term="   ").term is None


# ------------------------------------------------------------ filters combine


def test_filters_combine_by_and_and_can_only_narrow(store: Store):
    camera_only = search_events(store, Query(cameras="cam-01"))
    type_only = search_events(store, Query(types=EventType.ZONE_ENTRY))
    both = search_events(store, Query(cameras="cam-01", types=EventType.ZONE_ENTRY))
    print(f"camera {camera_only.total}, type {type_only.total}, both {both.total}")

    assert camera_only.total == 2
    assert type_only.total == 2
    assert both.total == 1, "combining two filters did not narrow"
    assert both.items[0].id == ENTRY.id


def test_every_filter_at_once_still_finds_the_event_that_satisfies_all_of_them(
    store: Store,
):
    """The whole panel set at once, which is the state a real search ends in."""
    query = Query(
        cameras="cam-02",
        window=Window(BASE + 100_000, BASE + 140_000),
        types=[EventType.AFTER_HOURS_PRESENCE],
        severities=at_least(Severity.HIGH),
        zones="zone-b",
        term="after hours",
    )
    found = search_events(store, query)
    print(f"{query.describe()} -> {found.total}")

    assert found.total == 1
    assert found.items[0].id == AFTER_HOURS.id
    assert not query.is_empty


def test_a_filter_that_excludes_everything_returns_an_empty_page_not_an_error(
    store: Store,
):
    found = search_events(store, Query(cameras="cam-01", zones="zone-b"))
    print(f"cam-01 in zone-b matched {found.total}")

    assert found.total == 0
    assert found.items == ()
    assert not found.truncated, "an empty page is not a truncated one"


# --------------------------------------------------------------- the incidents


def test_an_incident_is_found_by_a_camera_only_its_events_name(store: Store):
    """The incident row stores camera names as JSON; the events store the ids.

    Matching through the events is what makes this exact rather than a substring
    hunt through a JSON array, where a camera called ``cam-0`` would match
    ``cam-01``.
    """
    found = search_incidents(store, Query(cameras="cam-02"))
    print(f"cam-02 matched {found.total} incident(s) of {store.incident_count()}")

    assert found.total == 1
    assert set(found.items[0].cameras) == {"cam-02"}


def test_an_incident_matches_only_when_one_event_satisfies_every_filter(store: Store):
    """Not "contains a cam-01 event, and separately contains rapid movement"."""
    apart = search_incidents(store, Query(types=EventType.RAPID_MOVEMENT))
    together = search_incidents(
        store, Query(cameras="cam-01", types=EventType.RAPID_MOVEMENT)
    )
    print(f"rapid movement anywhere {apart.total}, on cam-01 {together.total}")

    assert apart.total == 1
    assert together.total == 0


def test_an_incident_that_was_already_running_when_the_window_opened_is_found(
    store: Store,
):
    """Overlap, not containment — the reason this module does not reuse
    "opened inside the window".

    The window below sits between two events of one incident, so nothing that
    happened is inside it and the incident spans the whole of it. An investigator
    scrubbing there is looking at the middle of something, and a search that
    returned nothing would be telling them the site was quiet.
    """
    quiet = Window(BASE + 140_000, BASE + 160_000)
    events_inside = search_events(store, Query(window=quiet))
    incidents_across = search_incidents(store, Query(window=quiet))
    print(
        f"between events: {events_inside.total} event(s), "
        f"{incidents_across.total} incident(s)"
    )

    assert events_inside.total == 0, "the window was supposed to fall between events"
    assert incidents_across.total == 1
    assert incidents_across.items[0].opened_at_millis < 140_000
    assert incidents_across.items[0].closed_at_millis > 160_000


def test_incidents_come_back_newest_first_with_a_total_beside_them(store: Store):
    page = search_incidents(store, Query(limit=1))
    everything = search_incidents(store)
    opened = [incident.opened_at_millis for incident in everything]
    print(f"{page.describe()}; opened at {opened} in returned order")

    assert everything.total == 2
    assert opened == sorted(opened, reverse=True)
    assert len(page) == 1 and page.total == 2 and page.truncated
    assert page.items[0].id == everything.items[0].id


def test_a_found_incident_still_carries_its_events_and_its_reasoning(
    incidents,
):
    """Rebuilt through the store, so nothing is recomputed from newer rules."""
    first = incidents[0]
    print(
        f"{first.id}: {len(first.events)} event(s), risk {first.risk.score:.3f} "
        f"from {len(first.risk.factors)} factor(s)"
    )

    assert len(first.events) >= 1
    assert first.risk.factors, "the risk came back without the reasoning behind it"
    assert first.severity in tuple(Severity)


# ------------------------------------------------------------------- timeline


def test_the_timeline_merges_events_and_incidents_oldest_first(store: Store):
    """One axis, both kinds. A scrub bar runs forwards."""
    whole = timeline(store, Window(BASE - 1000, BASE + 300_000))
    at = [moment.at_millis for moment in whole]
    kinds = {moment.kind for moment in whole}
    print(f"{whole.total} moment(s), kinds {sorted(k.value for k in kinds)}")

    assert whole.total == len(ALL_EVENTS) + 2
    assert at == sorted(at), "the timeline was not in the order things happened"
    assert kinds == {MomentKind.EVENT, MomentKind.INCIDENT}
    assert all(moment.at == utc_from_millis(moment.at_millis) for moment in whole)


def test_a_truncated_timeline_returns_the_earliest_and_says_how_many_there_were(
    store: Store,
):
    """A short bar with a count, never a bar with invisible holes in it."""
    window = Window(BASE - 1000, BASE + 300_000)
    whole = timeline(store, window)
    short = timeline(store, window, limit=3)
    print(f"{short.describe()} — first three of {whole.total}")

    assert len(short) == 3
    assert short.total == whole.total
    assert short.truncated
    assert [m.id for m in short] == [m.id for m in whole][:3]


def test_a_timeline_moment_names_the_cameras_and_zones_it_covers(store: Store):
    whole = timeline(store, Window(BASE - 1000, BASE + 300_000))
    incident_moments = [m for m in whole if m.kind is MomentKind.INCIDENT]
    unzoned = [m for m in whole if m.id == UNZONED.id]
    print(
        f"incident cameras {[m.cameras for m in incident_moments]}, "
        f"unzoned event zones {[m.zones for m in unzoned]}"
    )

    assert all(moment.cameras for moment in incident_moments)
    assert unzoned and unzoned[0].zones == (), (
        "an event in no zone must have no zone name, not an empty one"
    )


def test_a_timeline_of_a_quiet_window_is_empty_rather_than_wrong(store: Store):
    quiet = timeline(store, Window(BASE + 3_600_000, BASE + 7_200_000))
    print(f"an hour later: {quiet.total} moment(s)")

    assert quiet.total == 0
    assert quiet.items == ()


# ------------------------------------------------------- refusals, not silence


def test_a_window_that_ends_before_it_starts_is_refused(store: Store):
    """Two swapped arguments, which otherwise return an empty page forever."""
    with pytest.raises(SearchError) as refusal:
        Window(BASE + 1000, BASE)

    print(refusal.value)
    assert "before it starts" in str(refusal.value)


def test_an_unknown_event_type_is_refused_rather_than_matching_nothing(store: Store):
    """"Nothing matched" and "you asked for something impossible" are the same
    screen, and only one of them is the operator's fault."""
    with pytest.raises(SearchError) as refusal:
        Query(types=["LOTERING"])

    print(refusal.value)
    assert "LOITERING" in str(refusal.value), "the refusal must list what is valid"


def test_a_bare_camera_id_is_one_camera_not_a_sequence_of_characters(store: Store):
    """The silent version of this searches for cameras named c, a, m."""
    query = Query(cameras="cam-01")
    found = search_events(store, query)
    print(f"cameras={query.cameras} matched {found.total}")

    assert query.cameras == ("cam-01",)
    assert found.total == 2


def test_a_single_severity_is_one_filter_not_five(store: Store):
    """`Severity` is a string enum, so a lone member is also a string."""
    query = Query(severities=Severity.CRITICAL)
    found = search_events(store, query)
    print(f"severities={[s.value for s in query.severities]} matched {found.total}")

    assert query.severities == (Severity.CRITICAL,)
    assert found.total == 1


def test_a_page_of_no_rows_is_refused(store: Store):
    with pytest.raises(SearchError):
        Query(limit=0)
    with pytest.raises(SearchError):
        timeline(store, Window(BASE, BASE + 1000), limit=0)


# --------------------------------------------------------- nothing is dropped


def test_a_found_event_keeps_everything_the_store_kept(store: Store):
    """A read that quietly drops a field looks complete and is not.

    The same rule the store holds itself to, asserted again here because this
    module is a second door onto the same rows and a second reader is a second
    chance to lose something.
    """
    found = search_events(store, Query(term="quickly"))
    assert found.total == 1
    event = found.items[0]
    print(f"{event.id}: {event.evidence.describe()}")

    assert event.type is RAPID.type
    assert event.severity is RAPID.severity
    assert event.summary == RAPID.summary
    assert event.confidence == pytest.approx(RAPID.confidence)
    assert event.triggering_conditions == RAPID.triggering_conditions
    assert event.evidence == RAPID.evidence
    assert event.occurred_at_millis == RAPID.occurred_at_millis
    assert event.occurred_at == RAPID.occurred_at


def test_the_wall_clock_a_search_filters_on_is_the_one_the_store_wrote(store: Store):
    """One clock, end to end.

    A window is expressed in wall-clock milliseconds and the store keeps media
    time in a different column. If these two ever came apart, every window in
    this file would still pass — they are all built from the same offsets — so
    the identity is asserted directly, once.
    """
    found = search_events(store, Query(window=Window(BASE + 240_000, BASE + 240_000)))
    print(f"a one-instant window at BASE+240s matched {found.total}: {ids(found)}")

    assert found.total == 1
    assert found.items[0].id == UNZONED.id
    assert int(UNZONED.occurred_at.timestamp() * 1000) == BASE + 240_000


def test_the_default_page_is_the_documented_one(store: Store):
    assert Query().limit == DEFAULT_LIMIT
    assert search_events(store).limit == DEFAULT_LIMIT
