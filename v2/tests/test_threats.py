"""The threat vocabulary: empty until a site fills it, and never a guess."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from test_zones_events import CLASSIFYING, MOTION, square, track
from vigil.domain.events import RuleContext, Severity
from vigil.domain.geo import LatLon
from vigil.domain.relations import Relation, RelationKind
from vigil.domain.threats import SUGGESTED, Threat, ThreatRule, ThreatVocabulary
from vigil.domain.zones import PresenceTracker, Zone, ZoneKind

CENTRE = LatLon(33.8938, 35.5018)
ARMED = DetectorInfo = None  # replaced below


def _detector():
    from vigil.domain.detection import DetectorInfo

    return DetectorInfo("onnx-detect", "site-weights", model_sha256="ab" * 32,
                        class_names={0: "person", 5: "knife"}, classifies=True)


def _entered(zone: Zone, class_id: int = 0, confidence: float = 0.9):
    presence = PresenceTracker([zone])
    subject = track(1, CENTRE, class_id=class_id)
    subject.confidence = confidence
    presence.update([subject], 0)
    change = presence.update([subject], 100)[0]
    change.presence.observations = 5
    return subject, change


def test_nothing_is_a_threat_until_a_site_says_so():
    """The shipped model names `knife`; a kitchen must not raise a critical alert."""
    empty = ThreatVocabulary()
    assert not empty and len(empty) == 0
    assert empty.of("knife") is None
    assert empty.describe() == "no label is treated as a threat on this site"

    zone = Zone("z", "Yard", ZoneKind.RESTRICTED, square(CENTRE), enter_after_millis=0)
    subject, change = _entered(zone, class_id=5)
    context = RuleContext("node", "cam", zone, subject, change.presence, 100, datetime.now(timezone.utc),
                          _detector(), 1)
    assert ThreatRule(empty).on_presence_change(change, context) == [], "an unconfigured site claimed a threat"


def test_a_configured_label_is_claimed_with_the_detectors_own_words():
    vocabulary = ThreatVocabulary.from_labels(["knife"])
    assert vocabulary.of("Knife").severity is Severity.HIGH
    zone = Zone("z", "Yard", ZoneKind.RESTRICTED, square(CENTRE), enter_after_millis=0)
    subject, change = _entered(zone, class_id=5)
    context = RuleContext("node", "cam", zone, subject, change.presence, 100, datetime.now(timezone.utc),
                          _detector(), 1)
    events = ThreatRule(vocabulary).on_presence_change(change, context)
    assert len(events) == 1
    assert events[0].summary == "A knife was detected in Yard"
    assert events[0].severity is Severity.HIGH
    assert any("labelled it 'knife' at 0.90" in c for c in events[0].evidence.conditions)
    assert any("this site treats 'knife' as" in c for c in events[0].evidence.conditions)
    assert events[0].evidence.detector.model_sha256, "the claim must name the weights that made it"


def test_a_threat_needs_more_confidence_and_more_frames_than_an_ordinary_detection():
    vocabulary = ThreatVocabulary.from_labels(["knife"])
    zone = Zone("z", "Yard", ZoneKind.RESTRICTED, square(CENTRE), enter_after_millis=0)
    subject, change = _entered(zone, class_id=5, confidence=0.4)
    context = RuleContext("node", "cam", zone, subject, change.presence, 100, datetime.now(timezone.utc),
                          _detector(), 1)
    assert ThreatRule(vocabulary).on_presence_change(change, context) == [], "a weak label was believed"

    subject, change = _entered(zone, class_id=5, confidence=0.9)
    change.presence.observations = 1
    brief = RuleContext("node", "cam", zone, subject, change.presence, 100, datetime.now(timezone.utc),
                        _detector(), 1)
    assert ThreatRule(vocabulary).on_presence_change(change, brief) == [], "one frame was believed"


def test_a_detector_that_cannot_classify_can_never_have_said_knife():
    vocabulary = ThreatVocabulary.from_labels(["knife"])
    zone = Zone("z", "Yard", ZoneKind.RESTRICTED, square(CENTRE), enter_after_millis=0)
    subject, change = _entered(zone, class_id=5)
    context = RuleContext("node", "cam", zone, subject, change.presence, 100, datetime.now(timezone.utc), MOTION, 1)
    assert ThreatRule(vocabulary).on_presence_change(change, context) == []


def test_something_carried_is_said_so_and_raises_the_severity_one_step():
    """A knife on a bench and a knife in a hand are not the same fact."""
    vocabulary = ThreatVocabulary.from_labels(["knife"])
    zone = Zone("z", "Yard", ZoneKind.RESTRICTED, square(CENTRE), enter_after_millis=0)
    subject, change = _entered(zone, class_id=0)  # the person entered
    carried = Relation(RelationKind.CARRIED, 1, 9, confidence=0.7, observations=6,
                       conditions=("58% of the knife's box lay within the person's",))
    context = RuleContext("node", "cam", zone, subject, change.presence, 100, datetime.now(timezone.utc),
                          _detector(), 1, None, (carried,), {1: "person", 9: "knife"})
    events = ThreatRule(vocabulary).on_presence_change(change, context)
    assert len(events) == 1
    assert events[0].summary == "Somebody is carrying a knife in Yard"
    assert events[0].severity is Severity.CRITICAL, "a knife in a hand outranks a knife on a bench"
    assert any("58% of the knife's box" in c for c in events[0].evidence.conditions)
    assert any("inferred from one camera and may be wrong" in c for c in events[0].evidence.conditions)


def test_a_site_is_told_when_its_model_cannot_name_what_it_configured():
    """Configuring `knife` against a vehicle model protects nothing, silently."""
    vocabulary = ThreatVocabulary.from_labels(["knife", "gun"])
    assert vocabulary.unknown_to(["person", "car"]) == ("gun", "knife")
    assert vocabulary.unknown_to(["person", "knife", "gun"]) == ()
    assert ThreatVocabulary().unknown_to([]) == ()


def test_the_suggested_set_is_an_offer_and_carries_sensible_severities():
    suggested = ThreatVocabulary.suggested()
    assert len(suggested) == len(SUGGESTED)
    assert suggested.of("gun").severity is Severity.CRITICAL
    assert suggested.of("crowbar").severity is Severity.MEDIUM
    assert "a gun (CRITICAL)" in suggested.describe()
    # A label nobody suggested still works, at a stated default.
    made_up = ThreatVocabulary.from_labels(["blowtorch"])
    assert made_up.of("blowtorch") == Threat("blowtorch", "a blowtorch", Severity.HIGH)
