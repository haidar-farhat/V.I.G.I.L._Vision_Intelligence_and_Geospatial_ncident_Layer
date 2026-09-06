"""From events to a small number of incidents, conservatively and visibly.

Ported from v1: time-and-place association with a stated 65/35 weighting,
uncertainty-widened gates capped so one badly placed camera cannot swallow
the site, identity by union-find, and summaries that never say "person" on
the strength of a motion blob.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Sequence

from .events import Event, Severity, severity_rank
from .geo import LatLon, distance_meters
from .zones import ZoneKind

DEFAULT_WINDOW_MILLIS = 120_000
DEFAULT_RADIUS_METERS = 15.0
MAX_UNCERTAINTY_ALLOWANCE_METERS = 20.0


def _allowance(first_sigma: float | None, second_sigma: float | None, radius: float) -> float:
    """How far apart two sightings may be and still be one object.

    The two position errors combine **in quadrature**, not by addition.

    This module used to add them. `geo.Distance` has always said why that is
    wrong — "adding them would claim the errors always conspire" — and the two
    modules disagreed about the same two numbers while this one was the one
    deciding whether two sightings are the same person. Two errors of 3 m
    allowed 6 m of separation when what they justify is 4.24.

    The cap stays: one badly placed camera must not be able to swallow the
    site by claiming a huge uncertainty.
    """
    a = first_sigma or 0.0
    b = second_sigma or 0.0
    return radius + min(MAX_UNCERTAINTY_ALLOWANCE_METERS, math.hypot(a, b))


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
            separation = distance_meters(pa, pb)
            allowance = _allowance(first.evidence.position_uncertainty_meters,
                                   second.evidence.position_uncertainty_meters, radius_meters)
            if separation > allowance:
                continue
            score = time_and_place_score(1 - separation / allowance, 1 - gap / window_millis)
            out.append(Association(ka, kb, score, round(separation, 2), round(allowance, 2), gap, (
                f"{separation:.1f} m apart, within {allowance:.1f} m allowed by the two position uncertainties",
                f"{gap / 1000:.1f} s apart, within {window_millis / 1000:.0f} s",
            )))
    return out


# ------------------------------------------------------ same-camera fragments

#: Without appearance, the longest a fragment may be missing and still be
#: rejoined. Inside this the detector blinked — a person turned, or went half
#: behind a chair — and place can vouch for that. Beyond it the object was
#: genuinely gone, and "the same one came back" is a claim only a look could
#: support, which this build does not have.
FRAGMENT_MAX_GAP_MILLIS = 2000

#: The fraction of the cross-camera allowance a fragment must fall within.
#: Halved, not merely trimmed: the cross-camera gate is sized for a pair that
#: two cameras agree about, and on one camera with time and place alone the
#: same gate would join two people through one doorway a second apart.
FRAGMENT_ALLOWANCE_FRACTION = 0.5


def link_same_camera_fragments(events: Sequence[Event], *, radius_meters: float = DEFAULT_RADIUS_METERS) -> list[Association]:
    """Rejoin one camera's tracks that a blinking detector split in two.

    Deliberately narrow. Without appearance features this can only say "the
    detector lost it for under two seconds and it came back within a metre or
    two of where it was", and the association's reasons say exactly that, so
    a reader can disagree with it. Without this a person the detector dropped
    for one second is counted twice, and the summary reads "2 persons" for
    one — which is the kind of inflation this system exists not to do.
    """
    by_track: dict[tuple[str, int], list[Event]] = {}
    for event in events:
        by_track.setdefault((event.evidence.camera_id, event.evidence.track_id), []).append(event)
    for span in by_track.values():
        span.sort(key=lambda e: e.occurred_at_millis)

    links: list[Association] = []
    keys = sorted(by_track)
    for index, earlier_key in enumerate(keys):
        for later_key in keys[index + 1:]:
            if earlier_key[0] != later_key[0] or earlier_key[1] == later_key[1]:
                continue
            earlier, later = by_track[earlier_key], by_track[later_key]
            last, first = earlier[-1], later[0]
            if first.occurred_at_millis < last.occurred_at_millis:
                last, first = later[-1], earlier[0]
            gap = first.occurred_at_millis - last.occurred_at_millis
            if gap < 0 or gap > FRAGMENT_MAX_GAP_MILLIS:
                continue
            if last.evidence.class_label != first.evidence.class_label:
                # A person track must never absorb a vehicle, however close.
                continue
            point_a, point_b = _position_of(last), _position_of(first)
            if point_a is None or point_b is None:
                continue
            separation = distance_meters(point_a, point_b)
            allowance = _allowance(last.evidence.position_uncertainty_meters,
                                   first.evidence.position_uncertainty_meters,
                                   radius_meters) * FRAGMENT_ALLOWANCE_FRACTION
            if separation > allowance:
                continue
            score = time_and_place_score(1 - separation / allowance, 1 - gap / FRAGMENT_MAX_GAP_MILLIS)
            links.append(Association(earlier_key, later_key, score, round(separation, 2), round(allowance, 2), gap, (
                f"one camera lost it for {gap / 1000:.1f} s, within {FRAGMENT_MAX_GAP_MILLIS / 1000:.0f} s",
                f"it came back {separation:.1f} m away, within {allowance:.1f} m",
                "time and place only: this build has no appearance features to check a look",
            )))
    return links


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


class ReviewState(StrEnum):
    """Where an incident stands with the people who have to act on it."""

    NEW = "NEW"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    DISMISSED = "DISMISSED"


@dataclass(frozen=True, slots=True)
class Review:
    """Somebody's judgement, and who made it. Never anonymous."""

    state: ReviewState = ReviewState.NEW
    by: str | None = None
    at_millis: int | None = None
    note: str | None = None

    def describe(self) -> str:
        if self.state is ReviewState.NEW:
            return "not yet reviewed"
        note = f" — {self.note}" if self.note else ""
        return f"{str(self.state).lower()} by {self.by}{note}"


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
    review: Review = Review()

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
        # Same-camera fragments join the same identity as cross-camera pairs,
        # so the distinct count is of objects and not of the ids a flickering
        # detector handed out.
        links = links + link_same_camera_fragments(ordered, radius_meters=self._radius)
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
            if distance_meters(position, other_position) <= _allowance(
                event.evidence.position_uncertainty_meters,
                other.evidence.position_uncertainty_meters, self._radius,
            ):
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
