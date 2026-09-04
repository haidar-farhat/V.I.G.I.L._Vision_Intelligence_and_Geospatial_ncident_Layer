"""Tests for correlation, object identity and risk.

The specification's central claim is one sentence: *three cameras seeing one
person is one incident, not three alerts.* This file is where that is either
true or it is not.

The hardest property here is transitivity. If camera 7 and camera 8 saw the same
person, and 8 and 9 saw the same person, then all three saw one person — even
though 7 and 9 never overlapped. A pairwise implementation gets that wrong
silently, and the symptom is an incident that says "three people" about one.

The same claim has a within-camera half. The laptop camera gave one person seven
track ids in fifteen seconds; an incident that says "7 objects in Room" about
one colleague is the same failure on one camera, and the fragment tests at the
end are where that is either reconciled or it is not.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

import numpy as np
import pytest

from sentinel.core import LatLon, destination_point
from sentinel.events import Event, Evidence, EventType, Severity
from sentinel.incidents import (
    BLIND_FRAGMENT_MAX_GAP_MILLIS,
    Correlator,
    ObjectIdentity,
    associate,
    incident_id,
    link_same_camera_fragments,
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
    first_seen_millis: int | None = None,
    last_seen_millis: int | None = None,
) -> Event:
    # By default the evidence says the track lived one second past the event.
    # The real pipeline copies the track's last_seen *at the moment the event
    # is built*, so an entry event's last_seen is its own start; tests about
    # that shape pass ``last_seen_millis=at_millis`` explicitly.
    evidence = Evidence(
        camera_id=camera,
        track_id=track,
        first_seen_millis=at_millis if first_seen_millis is None else first_seen_millis,
        last_seen_millis=at_millis + 1000 if last_seen_millis is None else last_seen_millis,
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


# ------------------------------------------------- same-camera fragments


def fragments(
    *,
    gap_millis: int = 400,
    step_meters: float = 0.6,
    tracks: tuple[int, ...] = (3, 5, 7),
    camera: str = "webcam",
    placed: bool = True,
    **overrides,
) -> list[Event]:
    """One person as the tracker reported them: consecutive ids on one camera.

    Each fragment lives one second, the next begins ``gap_millis`` after it was
    last seen, and the person drifts ``step_meters`` between them — the shape
    the laptop camera produced when it handed out seven ids in fifteen seconds.
    ``placed=False`` is a camera that could not say where any of it was.
    """
    events, at = [], 0
    for index, track in enumerate(tracks):
        events.append(
            make_event(
                camera=camera, track=track, at_millis=at,
                point=destination_point(SITE, 90.0, step_meters * index) if placed else None,
                **overrides,
            )
        )
        at += 1000 + gap_millis
    return events


@dataclasses.dataclass(frozen=True, slots=True)
class SeenEvidence(Evidence):
    """Evidence that carries a look, which today's ``Evidence`` does not.

    The correlator reads ``appearance`` by name so it is live the day the real
    evidence grows the field; this is that day, for the test.
    """

    appearance: object = None


def seen(event: Event, histogram: np.ndarray) -> Event:
    fields = {f.name: getattr(event.evidence, f.name) for f in dataclasses.fields(Evidence)}
    return dataclasses.replace(event, evidence=SeenEvidence(**fields, appearance=histogram))


RED_COAT = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
BLUE_COAT = np.array([0.0, 0.0, 0.1, 0.9], dtype=np.float32)


def test_one_person_split_into_three_fragments_counts_as_one_object():
    """The within-camera half of the headline claim.

    Three consecutive ids, 0.4 s and 0.6 m apart — one person the detector
    lost twice. The operator must read "1 object", not "3 objects", and the
    two joins must be on the record.
    """
    incident = Correlator().correlate(fragments())[0]

    assert incident.distinct_objects == 1, (
        f"reported {incident.distinct_objects} objects for one person"
    )
    assert "1 object" in incident.summary
    assert len(incident.associations) == 2
    assert all(link.same_camera for link in incident.associations)


def test_two_tracks_whose_evidence_shows_them_overlapping_never_merge():
    # What the time condition proves, exactly: when the evidence itself says
    # the first track was still seen after the second began, no gate downstream
    # may join them. Overlapping spans, and the edge case: a track first seen
    # on the very frame another was last seen shared that frame with it. This
    # is *not* "two people present at once never merge" — see the next test
    # for the case the evidence cannot show.
    overlapping = [
        make_event(camera="webcam", track=1, at_millis=0),
        make_event(camera="webcam", track=2, at_millis=500),
    ]
    same_frame = [
        make_event(camera="webcam", track=1, at_millis=0),
        make_event(camera="webcam", track=2, at_millis=1000),
    ]

    assert link_same_camera_fragments(overlapping) == []
    assert link_same_camera_fragments(same_frame) == []
    assert Correlator().correlate(overlapping)[0].distinct_objects == 2


def test_a_track_that_lived_on_unseen_is_joined_to_a_newcomer_until_its_next_event():
    """A documented limit, pinned so the flicker is on the record.

    The pipeline copies a track's last_seen when it builds the event, and an
    entry event is built on first sight — so a track with one event has an end
    equal to its own start. Person A enters at t=0 and stays; person B enters
    1.5 s later, 1.0 m away. On that evidence A ended at t+0.0 s, B began
    inside the hold and within place, and the two are joined: "1 object".
    Measured: gap 1.5 s, 1.0 m within 6.0 m, score 0.6292. The link's own
    reasons say the end is an estimate and that place alone stood between
    them. A later event about A (t=5 s) shows the overlap, and the count
    becomes 2 — on a live window it goes 1 -> 2. Lifting this needs the
    tracker's live track ends; nothing in the evidence can.
    """
    a_enters = make_event(camera="webcam", track=1, at_millis=0, last_seen_millis=0)
    b_enters = make_event(
        camera="webcam", track=2, at_millis=1500, last_seen_millis=1500,
        point=destination_point(SITE, 90.0, 1.0),
    )
    a_later = make_event(
        camera="webcam", track=1, at_millis=5000,
        event_type=EventType.LOITERING, first_seen_millis=0, last_seen_millis=5000,
    )

    links = link_same_camera_fragments([a_enters, b_enters])
    assert len(links) == 1
    assert links[0].describe() == "#2 = #1 on webcam (gap 1.5 s, 1.0 m apart)"
    assert links[0].score == pytest.approx(0.6292, abs=0.0005)
    assert any(
        "as of its last event, t+0.0 s" in reason and "only place stood between them" in reason
        for reason in links[0].reasons
    )
    assert Correlator().correlate([a_enters, b_enters])[0].distinct_objects == 1

    assert link_same_camera_fragments([a_enters, b_enters, a_later]) == []
    assert Correlator().correlate([a_enters, b_enters, a_later])[0].distinct_objects == 2


def test_a_gap_beyond_the_hold_does_not_merge_without_appearance():
    # 2.5 s: inside what reid allows with a look, beyond the 2 s hold that is
    # all time and place alone may vouch for.
    assert BLIND_FRAGMENT_MAX_GAP_MILLIS == 2000
    assert link_same_camera_fragments(fragments(gap_millis=2500)) == []
    assert Correlator().correlate(fragments(gap_millis=2500))[0].distinct_objects == 3


def test_fragments_too_far_apart_do_not_merge():
    # 5 m in 0.4 s, positions known to ±2 m: within the 7.6 m reid would allow
    # a pair whose colours agree, beyond the 3.8 m allowed blind.
    assert link_same_camera_fragments(fragments(step_meters=5.0)) == []
    assert len(link_same_camera_fragments(fragments(step_meters=3.5))) == 2


def test_a_fragment_link_carries_its_reasons():
    # "#5 = #3 on webcam (gap 0.4 s, 0.6 m apart)": the count is auditable
    # only if every join says what it rests on — and, blind, that it rests on
    # time and place alone.
    link = link_same_camera_fragments(fragments())[0]

    assert link.a == ("webcam", 3) and link.b == ("webcam", 5)
    assert link.describe() == "#5 = #3 on webcam (gap 0.4 s, 0.6 m apart)"
    assert link.time_gap_millis == 400
    assert link.separation_meters == pytest.approx(0.6, abs=0.05)
    assert link.allowance_meters == pytest.approx(3.8, abs=0.05)
    assert link.score > 0.8  # measured 0.8274: 0.65 * (1 - 0.6/3.8) + 0.35 * (1 - 0.4/2)
    assert any("2 s hold" in reason for reason in link.reasons)
    assert any("half" in reason and "no appearance was carried" in reason for reason in link.reasons)
    # The fourth reason: #3's end is an estimate (its last event, t+1.0 s in
    # this fixture), so the operator can see the link rests on it.
    assert any(
        "#3's end is as of its last event, t+1.0 s" in reason and "only place" in reason
        for reason in link.reasons
    )
    assert any("does not classify" in reason for reason in link.reasons)


def test_a_one_sided_appearance_names_the_side_that_lacked_one():
    # Only the middle fragment carries a look. The pair falls to the blind
    # path — a comparison needs two — and the reason must not say "no
    # appearance was carried" when one was; it names which fragment had none.
    events = fragments()
    events[1] = seen(events[1], RED_COAT)

    links = link_same_camera_fragments(events)

    assert len(links) == 2
    first, second = links  # #5 = #3, then #7 = #5; #5 is the one with the look
    assert any("#3 carried no appearance" in reason for reason in first.reasons)
    assert any("#7 carried no appearance" in reason for reason in second.reasons)
    assert not any("no appearance was carried" in reason for link in links for reason in link.reasons)


def test_linking_fragments_removes_the_group_factor():
    # Seven ids for one person was worth a "group" factor and a HIGH band about
    # nobody. Measured: 59.0 counted as three, 45.0 counted as one.
    events = fragments()
    incident = Correlator().correlate(events)[0]
    unlinked = score_risk(events, 3, ["webcam"], incident.duration_millis)

    assert all(factor.name != "group" for factor in incident.risk.factors)
    assert any(factor.name == "group" for factor in unlinked.factors)
    assert unlinked.score - incident.risk.score >= 10.0


def test_unplaced_fragments_are_never_linked():
    # A CAMERA_FALLBACK position is the camera's own location, shared by every
    # track on it: zero metres between anything and everything. Time alone
    # would then merge whoever walked past next, so an unplaced camera keeps
    # its inflated count rather than being handed a flattering one.
    fallback = [
        dataclasses.replace(
            event,
            evidence=dataclasses.replace(event.evidence, position_source="CAMERA_FALLBACK"),
        )
        for event in fragments()
    ]
    unplaced = fragments(placed=False)

    assert link_same_camera_fragments(fallback) == []
    assert link_same_camera_fragments(unplaced) == []
    assert Correlator().correlate(fallback)[0].distinct_objects == 3


def test_a_person_fragment_never_continues_a_bottle():
    # A bottle appears between two fragments of a person, in the right place
    # at the right time for either. Class rules it out of both joins; the
    # person's fragments rejoin around it (1.8 s, 1.2 m), and the incident
    # has two objects — the person and the bottle — not one and not three.
    person, bottle, person_again = fragments(classifies=True, class_label="person")
    bottle = dataclasses.replace(
        bottle, evidence=dataclasses.replace(bottle.evidence, class_label="bottle")
    )

    links = link_same_camera_fragments([person, bottle, person_again])
    incident = Correlator().correlate([person, bottle, person_again])[0]

    assert len(links) == 1
    assert links[0].a == ("webcam", 3) and links[0].b == ("webcam", 7)
    assert "both classified person" in links[0].reasons[-1]
    assert incident.distinct_objects == 2


def test_appearance_when_carried_extends_the_gap_and_a_different_look_refuses_it():
    # Same 2.5 s gap the blind gate refuses. With a look on the evidence, reid's
    # own 5 s gap applies — and only when the looks agree.
    same_coat = [seen(event, RED_COAT) for event in fragments(gap_millis=2500)]
    different_coats = [
        seen(event, coat)
        for event, coat in zip(fragments(gap_millis=2500), (RED_COAT, BLUE_COAT, RED_COAT))
    ]

    linked = link_same_camera_fragments(same_coat)
    assert len(linked) == 2
    assert linked[0].time_gap_millis == 2500
    assert any("appearance" in reason for reason in linked[0].reasons)
    # A look does not make the earlier end known; the seen link says so too,
    # and names both defences.
    assert any(
        "as of its last event" in reason and "place and appearance stood between them" in reason
        for reason in linked[0].reasons
    )
    assert link_same_camera_fragments(different_coats) == []


def test_a_fragment_is_continued_by_at_most_one_other():
    # Two tracks appear after one vanishes, both close, both in time. At most
    # one can be its continuation; the other is somebody else, and the two
    # newcomers — held at once — are never each other.
    vanished = make_event(camera="webcam", track=1, at_millis=0)
    left = make_event(camera="webcam", track=2, at_millis=1400,
                      point=destination_point(SITE, 90.0, 0.6))
    right = make_event(camera="webcam", track=3, at_millis=1400,
                       point=destination_point(SITE, 270.0, 0.6))

    links = link_same_camera_fragments([vanished, left, right])
    incident = Correlator().correlate([vanished, left, right])[0]

    assert len(links) == 1
    assert incident.distinct_objects == 2


def test_the_incident_report_lists_its_fragment_links():
    correlator = Correlator()
    text = correlator.correlate(fragments())[0].describe()

    assert "links" in text
    assert "#5 = #3 on webcam (gap 0.4 s, 0.6 m apart)" in text
    assert "#7 = #5 on webcam" in text
    assert correlator.stats.fragment_links == 2
    assert correlator.stats.associations == 0


def test_fragment_links_never_cross_cameras():
    # The correlator's cross-camera gate has its own evidence and its own
    # window; the fragment gate must not quietly become a second, looser one.
    events = [
        make_event(camera="webcam", track=3, at_millis=0),
        make_event(camera="cam-08", track=5, at_millis=1400,
                   point=destination_point(SITE, 90.0, 0.6)),
    ]

    assert link_same_camera_fragments(events) == []
    assert len(associate(events)) == 1  # that pair is the other gate's to judge
