from datetime import datetime, timezone

from vigil.domain.detection import DetectorInfo
from vigil.domain.events import Event, EventType, Evidence, Severity
from vigil.domain.geo import LatLon, destination_point
from vigil.domain.incidents import Correlator, ObjectIdentity, associate, score_risk, time_and_place_score
from vigil.domain.zones import ZoneKind

ORIGIN = LatLon(33.8938, 35.5018)
CLASSIFYING = DetectorInfo("onnx-detect", "test", class_names={0: "person"}, classifies=True)
MOTION = DetectorInfo("motion", "MOG2", classifies=False)


def event(camera: str, track: int, at: int, point: LatLon | None = ORIGIN, *, detector=CLASSIFYING, label="person",
          severity=Severity.MEDIUM, zone="z1") -> Event:
    evidence = Evidence(camera, track, at // 66, detector, label if detector.classifies else None,
                        point.lat if point else None, point.lon if point else None, 1.5 if point else None, 5, ("c",))
    return Event(f"evt-{camera}-{track}-{at}", EventType.ZONE_ENTRY, severity, "entered", at,
                 datetime.fromtimestamp(at / 1000, tz=timezone.utc), "node", "zone-entry", 0.8, evidence, zone, "Yard")


def test_association_is_conservative_and_explains_itself():
    near = destination_point(ORIGIN, 90, 5)
    far = destination_point(ORIGIN, 90, 80)
    links = associate([event("a", 1, 0), event("b", 7, 3000, near), event("c", 2, 4000, far), event("a", 3, 200_000)])
    assert [(l.a, l.b) for l in links] == [(("a", 1), ("b", 7))], "same camera never; far never"
    assert links[0].reasons and 0 < links[0].score <= 1
    assert time_and_place_score(1.0, 0.0) == 0.65


def test_unplaced_cameras_never_associate_across_cameras():
    assert associate([event("a", 1, 0, None), event("b", 2, 100, None)]) == []


def test_identity_union_find_counts_distinct_objects():
    identity = ObjectIdentity()
    for key in (("a", 1), ("b", 2), ("c", 3)):
        identity.add(*key)
    identity.union(("a", 1), ("b", 2))
    assert identity.distinct_count() == 2


def test_the_correlator_groups_one_situation_and_separates_another():
    events = [event("a", 1, 0), event("b", 5, 2000, destination_point(ORIGIN, 45, 4)), event("a", 9, 600_000, destination_point(ORIGIN, 0, 300))]
    incidents = Correlator(zone_kinds={"z1": ZoneKind.RESTRICTED}).correlate(events)
    assert len(incidents) == 2
    first = incidents[0]
    assert first.distinct_objects == 1 and first.cameras == ("a", "b")
    assert first.summary.startswith("1 person in Yard, seen by 2 cameras")
    assert first.risk.score > 0 and any(f.name == "restricted" for f in first.risk.factors)
    assert first.associations and first.severity in (Severity.MEDIUM, Severity.HIGH)


def test_a_motion_only_incident_says_object_not_person():
    incidents = Correlator().correlate([event("a", 1, 0, detector=MOTION), event("a", 2, 1000, detector=MOTION)])
    assert len(incidents) == 1 and incidents[0].summary.startswith("2 objects")


def test_risk_bands_rise_with_severity_and_group_and_are_bounded():
    low = score_risk([event("a", 1, 0, severity=Severity.LOW)], 1, ("a",), 0, [])
    high = score_risk([event("a", 1, 0, severity=Severity.CRITICAL), event("b", 2, 1000, severity=Severity.CRITICAL)], 2, ("a", "b"), 120_000, [ZoneKind.RESTRICTED])
    assert low.score < high.score <= 1.0
    assert high.band in (Severity.HIGH, Severity.CRITICAL)
