"""Working the incident queue: a judgement that names a person, or none at all."""

from __future__ import annotations

import pytest

from test_incidents import event
from vigil.domain.incidents import Correlator, ReviewState
from vigil.service.auth import Forbidden, Principal, Role
from vigil.service.review import IncidentReview, ReviewError
from vigil.storage.store import Store

OPERATOR = Principal("alice", Role.OPERATOR, "user")
VIEWER = Principal("vic", Role.VIEWER, "user")


@pytest.fixture
def reviewed():
    with Store(":memory:") as store:
        events = [event("a", 1, 10_000), event("a", 2, 40_000)]
        store.save_events(events)
        incidents = Correlator().correlate(events)
        store.save_incidents(incidents)
        yield store, IncidentReview(store), incidents


def test_an_incident_starts_new_and_a_judgement_names_the_person(reviewed):
    store, review, incidents = reviewed
    first = incidents[0]
    assert store.incident(first.id).review.state is ReviewState.NEW
    assert store.incident(first.id).review.describe() == "not yet reviewed"

    acknowledged = review.acknowledge(first.id, by=OPERATOR, note="the gate was open")
    assert acknowledged.review.state is ReviewState.ACKNOWLEDGED
    assert acknowledged.review.by == "user:alice" and acknowledged.review.at_millis
    assert "acknowledged by user:alice — the gate was open" == acknowledged.review.describe()

    rows = [r for r in store.audit_trail() if r["action"].startswith("incident.")]
    assert rows and rows[0]["principal"] == "user:alice"
    assert '"state": "NEW"' in rows[0]["before"] and '"state": "ACKNOWLEDGED"' in rows[0]["after"]


def test_a_dismissal_without_a_reason_is_refused(reviewed):
    _store, review, incidents = reviewed
    with pytest.raises(ReviewError, match="reason"):
        review.dismiss(incidents[0].id, by=OPERATOR, note="   ")
    dismissed = review.dismiss(incidents[0].id, by=OPERATOR, note="the cat again")
    assert dismissed.review.state is ReviewState.DISMISSED and dismissed.review.note == "the cat again"
    with pytest.raises(ReviewError, match="at most"):
        review.reopen(incidents[0].id, by=OPERATOR, note="x" * 501)


def test_the_queue_hides_what_was_dismissed_and_reopening_brings_it_back(reviewed):
    _store, review, incidents = reviewed
    assert len(review.queue()) == len(incidents)
    review.dismiss(incidents[0].id, by=OPERATOR, note="a delivery")
    assert incidents[0].id not in [i.id for i in review.queue()]
    assert incidents[0].id in [i.id for i in review.queue(include_dismissed=True)]
    review.reopen(incidents[0].id, by=OPERATOR)
    assert incidents[0].id in [i.id for i in review.queue()]


def test_a_viewer_may_not_judge_and_an_unknown_incident_is_named(reviewed):
    _store, review, incidents = reviewed
    with pytest.raises(Forbidden):
        review.acknowledge(incidents[0].id, by=VIEWER)
    with pytest.raises(ReviewError, match="no incident"):
        review.acknowledge("inc-nothing", by=OPERATOR)


def test_re_correlation_refines_the_conclusion_and_never_undoes_the_judgement(reviewed):
    """The system may learn more about an incident; it may not overrule a person."""
    store, review, incidents = reviewed
    review.acknowledge(incidents[0].id, by=OPERATOR, note="checked on camera")
    later = Correlator().correlate(store.events())
    store.save_incidents(later)
    kept = store.incident(incidents[0].id)
    assert kept.review.state is ReviewState.ACKNOWLEDGED and kept.review.note == "checked on camera"


def test_the_review_migration_carries_a_way_back(tmp_path):
    from vigil.storage.schema import MIGRATIONS

    with Store(tmp_path / "r.db") as store:
        assert store.applied_versions() == [m.version for m in MIGRATIONS]
        undone = store.rollback()
        assert undone is not None and undone.name == "incident_review"
        assert "state" not in store.column_names("incidents"), "the column survived its own down"
        assert "incidents" in store.table_names(), "the incidents themselves must survive"
        store.migrate()
        assert "state" in store.column_names("incidents")
