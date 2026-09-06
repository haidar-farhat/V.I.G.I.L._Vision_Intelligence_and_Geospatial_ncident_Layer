"""Finding what happened, without reading a month of history to do it."""

from __future__ import annotations

import ast
import inspect

import pytest

from test_incidents import event as make_event
from vigil.domain.events import Severity
from vigil.domain.incidents import Correlator
from vigil.service.auth import Forbidden, Principal, Role
from vigil.service.review import IncidentReview
from vigil.service.search import Query, Search, SearchError, moment
from vigil.storage.store import Store

OPERATOR = Principal("alice", Role.OPERATOR, "user")
NOBODY = Principal("nobody", Role.VIEWER, "user", active=False)
NOW = 1_800_000_000_000


def test_a_time_a_person_typed_is_understood_or_refused_by_name():
    assert moment(None) is None and moment("  ") is None
    assert moment("2h", now_millis=NOW) == NOW - 2 * 3600_000
    assert moment("90m", now_millis=NOW) == NOW - 90 * 60_000
    assert moment("3d", now_millis=NOW) == NOW - 3 * 86_400_000
    assert moment("1w", now_millis=NOW) == NOW - 7 * 86_400_000
    assert moment("2026-09-01") == 1788220800000  # midnight UTC
    assert moment("2026-09-01T12:00:00+00:00") == 1788264000000
    assert moment("1s", now_millis=NOW) == NOW - 1000
    with pytest.raises(SearchError, match="is not a time"):
        moment("last tuesday")


@pytest.fixture
def searchable():
    with Store(":memory:") as store:
        events = [
            make_event("north-gate", 1, 10_000, severity=Severity.HIGH, zone="yard"),
            make_event("north-gate", 2, 20_000, severity=Severity.LOW, zone="yard"),
            make_event("loading-bay", 3, 500_000, severity=Severity.MEDIUM, zone="bay"),
        ]
        store.save_events(events)
        store.save_incidents(Correlator().correlate(events))
        yield store, Search(store), events


def test_incidents_are_found_by_camera_zone_severity_time_and_text(searchable):
    store, search, _events = searchable
    everything = search.incidents(Query(), by=OPERATOR)
    assert len(everything) == 2, "two situations, ten minutes apart"

    at_the_gate = search.incidents(Query(camera="north-gate"), by=OPERATOR)
    assert len(at_the_gate) == 1 and at_the_gate[0].cameras == ("north-gate",)
    assert search.incidents(Query(camera="nothing-here"), by=OPERATOR) == []

    # A severity means "that and worse", which is what a person means by it.
    assert len(search.incidents(Query(severity="HIGH"), by=OPERATOR)) == 1
    assert len(search.incidents(Query(severity="LOW"), by=OPERATOR)) == 2
    with pytest.raises(SearchError, match="not a severity"):
        search.incidents(Query(severity="URGENT"), by=OPERATOR)

    assert len(search.incidents(Query(zone="Yard"), by=OPERATOR)) == 2, "both are in a zone named Yard"
    # An incident's own words: its summary and any note somebody left on it.
    assert len(search.incidents(Query(contains="zone entry"), by=OPERATOR)) == 2
    assert len(search.incidents(Query(contains="person"), by=OPERATOR)) == 2
    assert search.incidents(Query(contains="nothing like this"), by=OPERATOR) == []

    later = search.incidents(Query(since="1s", until=None), by=OPERATOR, now_millis=400_000)
    assert [i.cameras for i in later] == [("loading-bay",)], "only what closed after the cut"


def test_the_queue_state_narrows_a_search_and_a_bad_state_is_refused(searchable):
    store, search, _events = searchable
    review = IncidentReview(store)
    first = search.incidents(Query(), by=OPERATOR)[0]
    review.dismiss(first.id, by=OPERATOR, note="a delivery")
    assert first.id not in [i.id for i in search.incidents(Query(state="queue"), by=OPERATOR)]
    assert first.id in [i.id for i in search.incidents(Query(state="dismissed"), by=OPERATOR)]
    assert first.id in [i.id for i in search.incidents(Query(state="all"), by=OPERATOR)]
    with pytest.raises(SearchError, match="not a review state"):
        search.incidents(Query(state="pondered"), by=OPERATOR)


def test_events_are_found_the_same_way_and_a_disabled_account_may_not_look(searchable):
    _store, search, events = searchable
    assert len(search.events(Query(), by=OPERATOR)) == 3
    assert len(search.events(Query(contains="entered"), by=OPERATOR)) == 3, "the events keep their own words"
    assert len(search.events(Query(camera="loading-bay"), by=OPERATOR)) == 1
    assert len(search.events(Query(severity="MEDIUM"), by=OPERATOR)) == 2
    assert len(search.events(Query(zone="yard"), by=OPERATOR)) == 2
    assert len(search.events(Query(until="1s"), by=OPERATOR, now_millis=100_000)) == 2
    assert search.events(Query(limit=1), by=OPERATOR) != []
    with pytest.raises(SearchError, match="positive"):
        search.events(Query(limit=0), by=OPERATOR)
    with pytest.raises(Forbidden):
        search.events(Query(), by=NOBODY)


def test_the_filtering_happens_in_the_database_not_in_python(searchable):
    """Pulling a month of events into memory to throw most away does not scale."""
    _store, search, _events = searchable
    source = inspect.getsource(Search)
    tree = ast.parse(source.replace("class Search:", "class Search:", 1))
    for node in ast.walk(tree):
        assert not isinstance(node, (ast.ListComp, ast.GeneratorExp)), "a comprehension here is a filter in Python"
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") in ("filter", "sorted"):
            raise AssertionError("Search filters or sorts in Python")


def test_a_query_says_what_it_asked_for():
    assert Query().describe() == "everything"
    said = Query(camera="north-gate", since="2d", severity="HIGH").describe()
    assert "camera north-gate" in said and "since 2d" in said and "severity HIGH" in said
