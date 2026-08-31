"""Tests for correlation, object identity and risk.

The specification's central claim is one sentence: *three cameras seeing one
person is one incident, not three alerts.* This file is where that is either
true or it is not.

The hardest property here is transitivity. If camera 7 and camera 8 saw the same
person, and 8 and 9 saw the same person, then all three saw one person — even
though 7 and 9 never overlapped. A pairwise implementation gets that wrong
silently, and the symptom is an incident that says "three people" about one.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from sentinel.core import LatLon, destination_point
from sentinel.events import Event, Evidence, EventType, Severity
from sentinel.incidents import (
    Correlator,
    ObjectIdentity,
    associate,
    incident_id,
    score_risk,
)
from sentinel.zones import ZoneKind

SITE = LatLon(33.8938, 35.5018)
START = datetime(2026, 8, 30, 3, 0, tzinfo=timezone.utc)


def make_event(
    *,
    camera: str,
    track: int,
    at_millis: int,
    point: LatLon | None = SITE,
    uncertainty: float = 2.0,
    severity: Severity = Severity.HIGH,
    event_type: EventType = EventType.ZONE_ENTRY,
    zone_id: str | None = "zone-a",
    zone_name: str | None = "Restricted Area A",
    summary: str = "An object entered Restricted Area A",
    confidence: float = 1.0,
    classifies: bool = False,
    class_label: str = "unclassified",
) -> Event:
    evidence = Evidence(
        camera_id=camera,
        track_id=track,
        first_seen_millis=at_millis,
        last_seen_millis=at_millis + 1000,
        observations=12,
        detector="MOG2 background subtraction" if not classifies else "yolo-test",
        detector_classifies=classifies,
        model_digest=None if not classifies else "a" * 64,
        class_label=class_label,
        latitude=point.lat if point else None,
        longitude=point.lon if point else None,
        position_uncertainty_meters=uncertainty if point else None,
        position_source="GROUND_PROJECTION" if point else None,
        speed_mps=1.2,
        heading_degrees=90.0,
        frame_indices=(at_millis // 200,),
    )
    return Event(
        id=f"ev_{camera}_{track}_{at_millis}_{event_type.value}",
        type=event_type,
        severity=severity,
        summary=summary,
        occurred_at_millis=at_millis,
        occurred_at=START,
        zone_id=zone_id,
        zone_name=zone_name,
        rule_id="zone-entry",
        evidence=evidence,
        triggering_conditions=("test",),
        confidence=confidence,
    )


# ------------------------------------------------------------------ union-find


def test_object_identity_is_transitive():
    # The property that decides whether an incident says "one person" or "three".
    identity = ObjectIdentity()
    identity.union(("cam-07", 1), ("cam-08", 4))
    identity.union(("cam-08", 4), ("cam-09", 2))

    assert identity.distinct_count() == 1
    assert identity.find("cam-07", 1) == identity.find("cam-09", 2)


def test_unrelated_tracks_stay_distinct():
    identity = ObjectIdentity()
    identity.add("cam-07", 1)
    identity.add("cam-07", 2)
    identity.union(("cam-08", 1), ("cam-09", 1))

    assert identity.distinct_count() == 3


def test_identity_survives_a_long_chain():
    # Path compression is iterative on purpose: a deep chain on a long-running
    # node would otherwise recurse further than Python allows.
    identity = ObjectIdentity()
    for index in range(5000):
        identity.union(("cam", index), ("cam", index + 1))

    assert identity.distinct_count() == 1


# ---------------------------------------------------------------- association


def test_two_cameras_seeing_the_same_place_are_associated():
    events = [
        make_event(camera="cam-07", track=1, at_millis=0),
        make_event(camera="cam-08", track=4, at_millis=1500,
                   point=destination_point(SITE, 90.0, 8.0)),
    ]
    links = associate(events)

    assert len(links) == 1
    assert links[0].score > 0.5
    assert "apart" in links[0].reasons[0]


def test_uncertainty_widens_the_association_gate():
    # Two positions 50 m apart, each known to ±20 m, are consistent with one
    # object. Comparing raw distance against a fixed threshold makes the answer
    # depend on how far each camera was from the subject, which is not a
    # property of the subject.
    far = destination_point(SITE, 90.0, 50.0)

    confident = associate([
        make_event(camera="cam-07", track=1, at_millis=0, uncertainty=1.0),
        make_event(camera="cam-08", track=4, at_millis=500, point=far, uncertainty=1.0),
    ])
    vague = associate([
        make_event(camera="cam-07", track=1, at_millis=0, uncertainty=20.0),
        make_event(camera="cam-08", track=4, at_millis=500, point=far, uncertainty=20.0),
    ])

    assert confident == []
    assert len(vague) == 1


def test_the_uncertainty_allowance_is_capped():
    # Otherwise one badly-placed camera reporting ±60 m associates everything on
    # the site into a single incident.
    far = destination_point(SITE, 90.0, 200.0)
    links = associate([
        make_event(camera="cam-07", track=1, at_millis=0, uncertainty=90.0),
        make_event(camera="cam-08", track=4, at_millis=500, point=far, uncertainty=90.0),
    ])

    assert links == []


def test_events_far_apart_in_time_are_not_associated():
    links = associate([
        make_event(camera="cam-07", track=1, at_millis=0),
        make_event(camera="cam-08", track=4, at_millis=600_000),
    ])
    assert links == []


def test_tracks_on_one_camera_are_never_associated_here():
    # Within a camera, identity is the tracker's job. Second-guessing it from
    # positions would undo its own evidence.
    links = associate([
        make_event(camera="cam-07", track=1, at_millis=0),
        make_event(camera="cam-07", track=2, at_millis=500),
    ])
    assert links == []


def test_unplaced_cameras_are_not_associated_on_timing_alone():
    # Two cameras that cannot say where they saw anything cannot support the
    # claim that they saw the same thing.
    links = associate([
        make_event(camera="cam-07", track=1, at_millis=0, point=None),
        make_event(camera="cam-08", track=4, at_millis=500, point=None),
    ])
    assert links == []


# ------------------------------------------------------- the central claim


def test_three_cameras_seeing_one_person_produce_one_incident():
    """The specification's headline requirement.

    One person walks past three cameras. Each raises its own event. The operator
    must receive one incident describing one object — not three alerts, and not
    one incident claiming three people.
    """
    events = [
        make_event(camera="cam-07", track=1, at_millis=0),
        make_event(camera="cam-08", track=4, at_millis=4000,
                   point=destination_point(SITE, 90.0, 12.0)),
        make_event(camera="cam-09", track=2, at_millis=8000,
                   point=destination_point(SITE, 90.0, 24.0)),
    ]

    incidents = Correlator().correlate(events)

    assert len(incidents) == 1, "three alerts reached the operator"
    assert incidents[0].distinct_objects == 1, (
        f"reported {incidents[0].distinct_objects} objects for one person"
    )
    assert incidents[0].cameras == ("cam-07", "cam-08", "cam-09")
    assert "1 object" in incidents[0].summary


def test_the_middle_camera_carries_the_chain():
    # 07 and 09 never overlap; only 08 links them. Pairwise comparison reports
    # two objects here, and the failure is silent.
    events = [
        make_event(camera="cam-07", track=1, at_millis=0),
        make_event(camera="cam-08", track=4, at_millis=3000,
                   point=destination_point(SITE, 90.0, 30.0)),
        make_event(camera="cam-09", track=2, at_millis=6000,
                   point=destination_point(SITE, 90.0, 60.0)),
    ]

    incidents = Correlator().correlate(events)

    assert len(incidents) == 1
    assert incidents[0].distinct_objects == 1


def test_three_people_at_one_place_are_one_incident_with_three_objects():
    # One breach, three people in it. The incident count and the object count
    # answer different questions and must not be conflated.
    events = [
        make_event(camera="cam-07", track=track, at_millis=track * 500)
        for track in (1, 2, 3)
    ]

    incidents = Correlator().correlate(events)

    assert len(incidents) == 1
    assert incidents[0].distinct_objects == 3
    assert "3 objects" in incidents[0].summary


def test_two_separate_intrusions_stay_separate():
    # An hour apart, at opposite ends of the site. Collapsing these would hide
    # one of them.
    events = [
        make_event(camera="cam-07", track=1, at_millis=0),
        make_event(camera="cam-09", track=1, at_millis=3_600_000,
                   point=destination_point(SITE, 0.0, 500.0)),
    ]

    assert len(Correlator().correlate(events)) == 2


def test_correlation_reduces_what_a_person_has_to_read():
    # The primary output of this stage.
    events = [
        make_event(camera="cam-07", track=1, at_millis=step * 400,
                   event_type=EventType.ZONE_ENTRY if step % 2 else EventType.LOITERING)
        for step in range(20)
    ]

    correlator = Correlator()
    incidents = correlator.correlate(events)

    assert len(incidents) == 1
    assert correlator.stats.reduction > 0.9


def test_an_empty_batch_produces_nothing():
    assert Correlator().correlate([]) == []


# ------------------------------------------------------------------ restraint


def test_an_incident_never_says_people_from_motion_blobs():
    # A blob is not a person. The summary is what an operator reads, and this is
    # the fabrication the whole system is built to avoid.
    events = [make_event(camera="cam-07", track=t, at_millis=t * 500) for t in (1, 2, 3)]
    incident = Correlator().correlate(events)[0]

    assert "object" in incident.summary
    assert "person" not in incident.summary and "people" not in incident.summary


def test_an_incident_says_people_when_the_detector_classified_them():
    events = [
        make_event(camera="cam-07", track=t, at_millis=t * 500,
                   classifies=True, class_label="person")
        for t in (1, 2)
    ]
    incident = Correlator().correlate(events)[0]

    assert "2 persons" in incident.summary or "2 people" in incident.summary


def test_a_mixed_batch_falls_back_to_object():
    # One classifying detector and one that cannot does not license the claim.
    events = [
        make_event(camera="cam-07", track=1, at_millis=0, classifies=True, class_label="person"),
        make_event(camera="cam-07", track=2, at_millis=500, classifies=False),
    ]
    incident = Correlator().correlate(events)[0]

    assert "object" in incident.summary


# ---------------------------------------------------------------------- risk


def test_risk_is_explained_rather_than_asserted():
    events = [make_event(camera="cam-07", track=1, at_millis=0)]
    risk = score_risk(events, 1, ["cam-07"], 0, [ZoneKind.RESTRICTED])

    assert risk.factors
    assert all(factor.because for factor in risk.factors)
    assert "severity" in risk.describe()


def test_corroboration_raises_risk():
    events = [
        make_event(camera="cam-07", track=1, at_millis=0),
        make_event(camera="cam-08", track=1, at_millis=1000),
    ]
    one = score_risk(events[:1], 1, ["cam-07"], 0)
    two = score_risk(events, 1, ["cam-07", "cam-08"], 0)

    assert two.score > one.score


def test_a_group_raises_risk():
    events = [make_event(camera="cam-07", track=1, at_millis=0)]
    alone = score_risk(events, 1, ["cam-07"], 0)
    group = score_risk(events, 4, ["cam-07"], 0)

    assert group.score > alone.score


def test_weak_evidence_cannot_reach_the_top_band_by_accumulation():
    # Confidence scales the total rather than adding to it, so an incident built
    # from uncertain observations has to be carried by its evidence.
    strong = [make_event(camera=f"cam-0{i}", track=1, at_millis=i * 900, confidence=1.0)
              for i in range(1, 5)]
    weak = [make_event(camera=f"cam-0{i}", track=1, at_millis=i * 900, confidence=0.2)
            for i in range(1, 5)]

    high = score_risk(strong, 4, ["a", "b", "c", "d"], 300_000, [ZoneKind.RESTRICTED])
    low = score_risk(weak, 4, ["a", "b", "c", "d"], 300_000, [ZoneKind.RESTRICTED])

    assert high.score > low.score
    assert any(f.name == "confidence" for f in low.factors)


def test_risk_bands_are_ordered():
    assert score_risk([], 0, [], 0).band is Severity.INFO


def test_severity_never_falls_below_the_worst_event():
    # Risk may raise the band; it must never quietly lower a CRITICAL event.
    events = [make_event(camera="cam-07", track=1, at_millis=0, severity=Severity.CRITICAL,
                         confidence=0.1)]
    incident = Correlator().correlate(events)[0]

    assert incident.severity is Severity.CRITICAL


# -------------------------------------------------------------- identity, ids


def test_incident_ids_are_deterministic():
    events = [make_event(camera="cam-07", track=1, at_millis=0)]

    first = Correlator().correlate(events)[0]
    second = Correlator().correlate(events)[0]

    assert first.id == second.id
    assert first.id == incident_id(events[0])
    assert first.id.startswith("inc_")


def test_replaying_the_same_events_produces_the_same_incident():
    # Re-examining an incident must show the same incident, not a second one
    # beside it.
    events = [
        make_event(camera="cam-07", track=1, at_millis=0),
        make_event(camera="cam-08", track=4, at_millis=2000,
                   point=destination_point(SITE, 90.0, 10.0)),
    ]

    first = Correlator().correlate(events)
    second = Correlator().correlate(list(reversed(events)))

    assert [i.id for i in first] == [i.id for i in second]
    assert first[0].distinct_objects == second[0].distinct_objects


# ------------------------------------------------------------------ timeline


def test_the_timeline_is_ordered_and_complete():
    events = [
        make_event(camera="cam-07", track=1, at_millis=4000),
        make_event(camera="cam-07", track=1, at_millis=1000),
        make_event(camera="cam-07", track=1, at_millis=2500),
    ]
    incident = Correlator().correlate(events)[0]
    timeline = incident.timeline()

    assert [entry.at_millis for entry in timeline] == [1000, 2500, 4000]
    assert len(timeline) == len(events)
    assert all(entry.event_id for entry in timeline)


def test_an_incident_describes_itself_for_review():
    events = [
        make_event(camera="cam-07", track=1, at_millis=0),
        make_event(camera="cam-08", track=4, at_millis=3000,
                   point=destination_point(SITE, 90.0, 10.0)),
    ]
    text = Correlator(zone_kinds={"zone-a": ZoneKind.RESTRICTED}).correlate(events)[0].describe()

    for expected in ("inc_", "objects", "cameras", "risk", "timeline", "Restricted Area A"):
        assert expected in text


def test_associations_are_attached_to_the_incident_they_justify():
    # The operator has to be able to see *why* two cameras were treated as one
    # object, and disagree if it is wrong.
    events = [
        make_event(camera="cam-07", track=1, at_millis=0),
        make_event(camera="cam-08", track=4, at_millis=2000,
                   point=destination_point(SITE, 90.0, 10.0)),
    ]
    incident = Correlator().correlate(events)[0]

    assert incident.associations
    assert incident.associations[0].reasons
    assert incident.associations[0].separation_meters == pytest.approx(10.0, abs=1.0)
