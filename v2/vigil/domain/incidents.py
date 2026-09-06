"""From events to a small number of incidents, conservatively and visibly.

Ported from v1: time-and-place association with a stated 65/35 weighting,
uncertainty-widened gates capped so one badly placed camera cannot swallow
the site, identity by union-find, and summaries that never say "person" on
the strength of a motion blob.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

from .events import Event, Severity, severity_rank
from .geo import LatLon, haversine_distance
from .zones import ZoneKind

DEFAULT_WINDOW_MILLIS = 120_000
DEFAULT_RADIUS_METERS = 15.0
MAX_UNCERTAINTY_ALLOWANCE_METERS = 20.0


class ObjectIdentity:
    """Union-find over (camera, track)."""

    def __init__(self) -> None:
        self._parent: dict[tuple[str, int], tuple[str, int]] = {}

    def add(self, camera_id: str, track_id: int) -> None:
        self._parent.setdefault((camera_id, track_id), (camera_id, track_id))

    def find(self, camera_id: str, track_id: int) -> tuple[str, int]:
        key = (camera_id, track_id)
        self.add(*key)
        while self._parent[key] != key:
            self._parent[key] = self._parent[self._parent[key]]
            key = self._parent[key]
        return key

    def union(self, a: tuple[str, int], b: tuple[str, int]) -> None:
        ra, rb = self.find(*a), self.find(*b)
        if ra != rb:
            self._parent[max(ra, rb)] = min(ra, rb)

    def distinct_count(self) -> int:
        return len({self.find(*k) for k in self._parent})


@dataclass(frozen=True, slots=True)
class Association:
    a: tuple[str, int]
    b: tuple[str, int]
    score: float
    separation_meters: float
    allowance_meters: float
    time_gap_millis: int
    reasons: tuple[str, ...]


def _position_of(event: Event) -> LatLon | None:
    if event.evidence.latitude is None or event.evidence.longitude is None:
        return None
    return LatLon(event.evidence.latitude, event.evidence.longitude)


def time_and_place_score(proximity: float, recency: float) -> float:
    """Place outweighs time, 65/35. Stated here so it can be argued with."""
    return round(0.65 * proximity + 0.35 * recency, 4)


def associate(events: Sequence[Event], *, window_millis: int = DEFAULT_WINDOW_MILLIS,
              radius_meters: float = DEFAULT_RADIUS_METERS) -> list[Association]:
    out: list[Association] = []
    for i, first in enumerate(events):
        for second in events[i + 1:]:
            ka = (first.evidence.camera_id, first.evidence.track_id)
            kb = (second.evidence.camera_id, second.evidence.track_id)
            if ka == kb or ka[0] == kb[0]:
                continue
            gap = abs(second.occurred_at_millis - first.occurred_at_millis)
            if gap > window_millis:
                continue
            pa, pb = _position_of(first), _position_of(second)
            if pa is None or pb is None:
                continue
            separation = haversine_distance(pa, pb)
            slack = min(MAX_UNCERTAINTY_ALLOWANCE_METERS,
                        (first.evidence.position_uncertainty_meters or 0.0) + (second.evidence.position_uncertainty_meters or 0.0))
            allowance = radius_meters + slack
            if separation > allowance:
                continue
            score = time_and_place_score(1 - separation / allowance, 1 - gap / window_millis)
            out.append(Association(ka, kb, score, round(separation, 2), round(allowance, 2), gap, (
                f"{separation:.1f} m apart, within {allowance:.1f} m allowed by the two position uncertainties",
                f"{gap / 1000:.1f} s apart, within {window_millis / 1000:.0f} s",
            )))
    return out


@dataclass(frozen=True, slots=True)
class RiskFactor:
    name: str
    weight: float
    reason: str


@dataclass(frozen=True, slots=True)
class Risk:
    score: float
    factors: tuple[RiskFactor, ...]

    @property
    def band(self) -> Severity:
        if self.score >= 0.8:
            return Severity.CRITICAL
        if self.score >= 0.6:
            return Severity.HIGH
        if self.score >= 0.35:
            return Severity.MEDIUM
        if self.score >= 0.15:
            return Severity.LOW
        return Severity.INFO


def score_risk(events: Sequence[Event], distinct: int, cameras: Sequence[str], duration_millis: int,
               kinds: Sequence[ZoneKind]) -> Risk:
    factors: list[RiskFactor] = []
    worst = max(severity_rank(e.severity) for e in events) / max(1, len(Severity) - 1)
    factors.append(RiskFactor("severity", 0.45 * worst, f"worst event severity {max(events, key=lambda e: severity_rank(e.severity)).severity}"))
    if distinct > 1:
        factors.append(RiskFactor("group", min(0.2, 0.08 * (distinct - 1)), f"{distinct} distinct objects"))
    if len(cameras) > 1:
        factors.append(RiskFactor("multi-camera", 0.1, f"seen by {len(cameras)} cameras"))
    if duration_millis >= 60_000:
        factors.append(RiskFactor("duration", min(0.15, 0.05 * (duration_millis // 60_000)), f"lasted {duration_millis // 1000} s"))
    if ZoneKind.RESTRICTED in kinds:
        factors.append(RiskFactor("restricted", 0.15, "a restricted zone is involved"))
    mean_confidence = sum(e.confidence for e in events) / len(events)
    score = min(1.0, sum(f.weight for f in factors)) * (0.6 + 0.4 * mean_confidence)
    return Risk(round(score, 3), tuple(factors))


@dataclass(frozen=True, slots=True)
class Incident:
    id: str
    severity: Severity
    summary: str
    opened_at_millis: int
    closed_at_millis: int
    opened_at: datetime
    distinct_objects: int
    cameras: tuple[str, ...]
    zones: tuple[str, ...]
    events: tuple[Event, ...]
    associations: tuple[Association, ...]
    risk: Risk

    @property
    def duration_millis(self) -> int:
        return self.closed_at_millis - self.opened_at_millis

    def describe(self) -> str:
        return f"[{self.severity}] {self.summary} — {len(self.events)} event(s), risk {self.risk.score:.2f}"


def incident_id(opening: Event) -> str:
    return "inc-" + hashlib.sha256(f"{opening.id}|{opening.occurred_at_millis}".encode()).hexdigest()[:16]


@dataclass
class CorrelationStats:
    events_in: int = 0
    associations: int = 0
    incidents_out: int = 0


class Correlator:
    """Batch: an incident is a statement about a span, decided once nothing more is coming."""

    def __init__(self, *, window_millis: int = DEFAULT_WINDOW_MILLIS, radius_meters: float = DEFAULT_RADIUS_METERS,
                 zone_kinds: dict[str, ZoneKind] | None = None):
        self._window = window_millis
        self._radius = radius_meters
        self._zone_kinds = zone_kinds or {}
        self.stats = CorrelationStats()

    def correlate(self, events: Sequence[Event]) -> list[Incident]:
        if not events:
            return []
        ordered = sorted(events, key=lambda e: (e.occurred_at_millis, e.id))
        self.stats.events_in += len(ordered)
        links = associate(ordered, window_millis=self._window, radius_meters=self._radius)
        self.stats.associations += len(links)
        identity = ObjectIdentity()
        for event in ordered:
            identity.add(event.evidence.camera_id, event.evidence.track_id)
        for link in links:
            identity.union(link.a, link.b)
        groups = self._group(ordered, identity)
        incidents = [self._build(group, links, identity) for group in groups]
        self.stats.incidents_out += len(incidents)
        return incidents

    def _group(self, events: Sequence[Event], identity: ObjectIdentity) -> list[list[Event]]:
        groups: list[list[Event]] = []
        roots: list[set[tuple[str, int]]] = []
        for event in events:
            root = identity.find(event.evidence.camera_id, event.evidence.track_id)
            position = _position_of(event)
            target = None
            for index, group in enumerate(groups):
                if root in roots[index] or self._is_near(event, position, group):
                    target = index
                    break
            if target is None:
                groups.append([event])
                roots.append({root})
            else:
                groups[target].append(event)
                roots[target].add(root)
        return groups

    def _is_near(self, event: Event, position: LatLon | None, group: Sequence[Event]) -> bool:
        for other in group:
            if abs(event.occurred_at_millis - other.occurred_at_millis) > self._window:
                continue
            other_position = _position_of(other)
            if position is None or other_position is None:
                if event.evidence.camera_id == other.evidence.camera_id:
                    return True
                continue
            slack = min(MAX_UNCERTAINTY_ALLOWANCE_METERS,
                        (event.evidence.position_uncertainty_meters or 0.0) + (other.evidence.position_uncertainty_meters or 0.0))
            if haversine_distance(position, other_position) <= self._radius + slack:
                return True
        return False

    def _build(self, group: Sequence[Event], links: Sequence[Association], identity: ObjectIdentity) -> Incident:
        ordered = sorted(group, key=lambda e: e.occurred_at_millis)
        opening = ordered[0]
        members = {(e.evidence.camera_id, e.evidence.track_id) for e in ordered}
        distinct = len({identity.find(*k) for k in members})
        cameras = tuple(sorted({e.evidence.camera_id for e in ordered}))
        zone_names = tuple(sorted({e.zone_name for e in ordered if e.zone_name}))
        kinds = [self._zone_kinds[z] for z in {e.zone_id for e in ordered if e.zone_id} if z in self._zone_kinds]
        duration = ordered[-1].occurred_at_millis - opening.occurred_at_millis
        risk = score_risk(ordered, distinct, cameras, duration, kinds)
        worst = max(ordered, key=lambda e: severity_rank(e.severity)).severity
        severity = worst if severity_rank(worst) >= severity_rank(risk.band) else risk.band
        relevant = tuple(l for l in links if l.a in members and l.b in members)
        return Incident(
            id=incident_id(opening), severity=severity,
            summary=self._summarise(ordered, distinct, cameras, zone_names),
            opened_at_millis=opening.occurred_at_millis, closed_at_millis=ordered[-1].occurred_at_millis,
            opened_at=opening.occurred_at, distinct_objects=distinct, cameras=cameras, zones=zone_names,
            events=tuple(ordered), associations=relevant, risk=risk,
        )

    @staticmethod
    def _summarise(events: Sequence[Event], distinct: int, cameras: Sequence[str], zones: Sequence[str]) -> str:
        labels = {e.evidence.class_label for e in events if e.evidence.detector_classifies}
        if labels and None not in labels and len(labels) == 1:
            noun = next(iter(labels))
            noun = noun if distinct == 1 else (noun + "s" if not noun.endswith("s") else noun)
        else:
            noun = "object" if distinct == 1 else "objects"
        where = f" in {', '.join(zones)}" if zones else ""
        seen = f"seen by {len(cameras)} cameras" if len(cameras) > 1 else f"on {cameras[0]}"
        kinds = sorted({e.type.lower().replace('_', ' ') for e in events})
        return f"{distinct} {noun}{where}, {seen}: {', '.join(kinds)}"
