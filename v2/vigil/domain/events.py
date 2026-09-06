"""Events: assertions built from observations, each carrying its evidence."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone, tzinfo
from enum import StrEnum
from typing import Sequence

from .detection import DetectorInfo
from .tracking import Track
from .zones import Presence, PresenceChange, Zone, ZoneKind


class EventType(StrEnum):
    ZONE_ENTRY = "ZONE_ENTRY"
    LOITERING = "LOITERING"
    AFTER_HOURS_PRESENCE = "AFTER_HOURS_PRESENCE"


class Severity(StrEnum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


_SEVERITY_ORDER = {s: i for i, s in enumerate(Severity)}


def severity_rank(severity: Severity) -> int:
    return _SEVERITY_ORDER[severity]


@dataclass(frozen=True, slots=True)
class Evidence:
    camera_id: str
    track_id: int
    frame_index: int
    detector: DetectorInfo
    class_label: str | None
    latitude: float | None
    longitude: float | None
    position_uncertainty_meters: float | None
    observations: int
    conditions: tuple[str, ...]

    @property
    def detector_classifies(self) -> bool:
        return self.detector.classifies


@dataclass(frozen=True, slots=True)
class Event:
    id: str
    type: EventType
    severity: Severity
    summary: str
    occurred_at_millis: int
    occurred_at: datetime
    node_id: str
    rule_id: str
    confidence: float
    evidence: Evidence
    zone_id: str | None = None
    zone_name: str | None = None

    def describe(self) -> str:
        return f"[{self.severity}] {self.summary} ({self.evidence.camera_id} track {self.evidence.track_id})"


def event_id(node_id: str, camera_id: str, track_id: int, rule_id: str, at_millis: int) -> str:
    digest = hashlib.sha256(f"{node_id}|{camera_id}|{track_id}|{rule_id}|{at_millis}".encode()).hexdigest()
    return f"evt-{digest[:16]}"


@dataclass(frozen=True, slots=True)
class RuleContext:
    """Everything a rule may look at. A rule that reaches past this is not reviewable."""

    node_id: str
    camera_id: str
    zone: Zone | None
    track: Track | None
    presence: Presence | None
    at_millis: int
    moment: datetime
    detector: DetectorInfo
    frame_index: int
    site_tz: tzinfo | None = None

    @property
    def local_moment(self) -> datetime:
        return self.moment.astimezone(self.site_tz) if self.site_tz else self.moment

    @property
    def class_label(self) -> str | None:
        if self.track is None:
            return None
        return self.detector.label_for(self.track.class_id)


class Rule:
    id: str = "rule"
    description: str = ""
    severity: Severity = Severity.INFO
    event_type: EventType = EventType.ZONE_ENTRY

    def on_presence_change(self, change: PresenceChange, context: RuleContext) -> list[Event]:
        return []

    def on_frame(self, context: RuleContext) -> list[Event]:
        return []

    def _build(self, context: RuleContext, *, summary: str, conditions: Sequence[str],
               confidence: float, observations: int, severity: Severity | None = None) -> Event:
        track = context.track
        position = track.position if track is not None else None
        projected = position is not None and position.is_projected
        evidence = Evidence(
            camera_id=context.camera_id,
            track_id=track.id if track is not None else -1,
            frame_index=context.frame_index,
            detector=context.detector,
            class_label=context.class_label,
            latitude=position.point.lat if projected else None,
            longitude=position.point.lon if projected else None,
            position_uncertainty_meters=position.radius_meters if projected else None,
            observations=observations,
            conditions=tuple(conditions),
        )
        return Event(
            id=event_id(context.node_id, context.camera_id, evidence.track_id, self.id, context.at_millis),
            type=self.event_type,
            severity=severity or self.severity,
            summary=summary,
            occurred_at_millis=context.at_millis,
            occurred_at=context.moment,
            node_id=context.node_id,
            rule_id=self.id,
            confidence=round(min(1.0, max(0.0, confidence)), 3),
            evidence=evidence,
            zone_id=context.zone.id if context.zone else None,
            zone_name=context.zone.name if context.zone else None,
        )


def _subject(context: RuleContext) -> str:
    label = context.class_label
    return "An object" if label is None else f"A {label.replace('_', ' ')}"


class ZoneEntryRule(Rule):
    id = "zone-entry"
    description = "An object entered a restricted or perimeter zone."
    event_type = EventType.ZONE_ENTRY
    severity = Severity.MEDIUM

    def __init__(self, kinds=(ZoneKind.RESTRICTED, ZoneKind.PERIMETER)):
        self._kinds = frozenset(kinds)

    def on_presence_change(self, change: PresenceChange, context: RuleContext) -> list[Event]:
        zone = context.zone
        if change.kind != "ENTERED" or zone is None or zone.kind not in self._kinds:
            return []
        severity = Severity.HIGH if zone.kind is ZoneKind.RESTRICTED else Severity.MEDIUM
        return [self._build(
            context, summary=f"{_subject(context)} entered {zone.name}",
            conditions=(f"membership held for {zone.enter_after_millis} ms", f"zone kind is {zone.kind}"),
            confidence=change.presence.confidence, observations=change.presence.observations, severity=severity,
        )]


class LoiteringRule(Rule):
    id = "loitering"
    description = "An object stayed in a zone, nearly still, longer than the dwell."
    event_type = EventType.LOITERING
    severity = Severity.MEDIUM

    def __init__(self, dwell_millis: int = 30_000, still_speed_mps: float = 0.5):
        self.dwell_millis = dwell_millis
        self.still_speed_mps = still_speed_mps
        self._raised: set[tuple[str, int]] = set()

    def on_frame(self, context: RuleContext) -> list[Event]:
        presence, zone, track = context.presence, context.zone, context.track
        if presence is None or zone is None or track is None or not presence.confirmed:
            return []
        key = (zone.id, track.id)
        if key in self._raised:
            return []
        dwell = context.at_millis - presence.entered_millis
        if dwell < self.dwell_millis:
            return []
        if track.speed_mps is not None and track.speed_mps > self.still_speed_mps:
            return []
        self._raised.add(key)
        return [self._build(
            context, summary=f"{_subject(context)} loitered in {zone.name} for {dwell // 1000} s",
            conditions=(f"dwell {dwell} ms ≥ {self.dwell_millis} ms",
                        f"speed {'unknown' if track.speed_mps is None else f'{track.speed_mps:.1f} m/s'} ≤ {self.still_speed_mps} m/s"),
            confidence=presence.confidence, observations=presence.observations,
        )]

    def forget(self, track_id: int) -> None:
        self._raised = {k for k in self._raised if k[1] != track_id}


class AfterHoursRule(Rule):
    id = "after-hours"
    description = "An object was present during a scheduled closed period."
    event_type = EventType.AFTER_HOURS_PRESENCE
    severity = Severity.HIGH

    def on_presence_change(self, change: PresenceChange, context: RuleContext) -> list[Event]:
        zone = context.zone
        if change.kind != "ENTERED" or zone is None or zone.schedule is None:
            return []
        local = context.local_moment
        if not zone.schedule.is_closed_at(local.hour + local.minute / 60):
            return []
        return [self._build(
            context, summary=f"{_subject(context)} was in {zone.name} outside permitted hours",
            conditions=(f"{local:%H:%M %Z} falls within {zone.schedule.describe()}",
                        f"presence confirmed after {zone.enter_after_millis} ms"),
            confidence=change.presence.confidence, observations=change.presence.observations,
        )]


def default_rules() -> list[Rule]:
    return [ZoneEntryRule(), LoiteringRule(), AfterHoursRule()]


def utc(millis: int) -> datetime:
    return datetime.fromtimestamp(millis / 1000, tz=timezone.utc)
