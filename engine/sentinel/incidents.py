"""Incidents: many events, one thing that happened.

... -> EVENT ANALYSIS -> MULTI-CAMERA CORRELATION -> RISK SCORING -> INCIDENT

This is the module the whole system exists for, and the one whose success is
measured by how *little* it produces. Twelve seconds of one person walking
through a restricted area generates an entry event, an after-hours event, a
loitering event, and the same again from the neighbouring camera. Six events. One
thing happened.

An operator who receives six alerts for one intrusion learns to skim, and the
skimming is what loses the seventh alert that mattered. Alert fatigue is not an
annoyance to be traded against sensitivity; it is a failure mode of the system,
and reducing event count is the primary output of this stage.

Three mechanisms, in order of how much they matter:

**Object identity is transitive.** If camera 7 and camera 8 saw the same person,
and camera 8 and camera 9 saw the same person, then all three saw one person —
even if 7 and 9 never overlapped. Union-find over associated tracks gets this
right; pairwise comparison does not, and the failure is silent. This is also
where the count comes from: an incident's "three people" must be the number of
distinct *objects*, not of track segments, or a tracker that fragments turns
three people into six.

**Association accounts for uncertainty.** Two positions 5 m apart, each known to
±4 m, are entirely consistent with being one object. Comparing raw distance
against a fixed threshold makes the answer depend on how far each camera was from
the subject, which is not a property of the subject at all.

**Risk is explained, never asserted.** The score comes with the contributions
that produced it. A number an operator cannot interrogate is a number they will
eventually learn to ignore.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Sequence

from .core import LatLon, haversine_distance
from .events import Event, Severity, severity_rank
from .zones import ZoneKind

#: How far apart in time two events may be and still belong to one incident.
#: Long enough to hold a walk across a site, short enough that two unrelated
#: visits an hour apart stay separate.
DEFAULT_WINDOW_MILLIS = 120_000

#: How far apart on the ground, before each position's own uncertainty is added.
DEFAULT_RADIUS_METERS = 40.0

#: The most an uncertainty may widen the association gate. Without a cap, one
#: badly-placed camera reporting ±60 m would associate everything on the site
#: into a single incident.
MAX_UNCERTAINTY_ALLOWANCE_METERS = 25.0


# ------------------------------------------------------------------ union-find


class ObjectIdentity:
    """Union-find over track identities.

    Two tracks that the association step decided are the same object are merged.
    The transitive closure is the point: 7↔8 and 8↔9 must yield one object, and
    doing this pairwise instead produces a count that is wrong in a way nobody
    notices until an incident says "six people" about three.
    """

    __slots__ = ("_parent", "_rank")

    def __init__(self) -> None:
        self._parent: dict[tuple[str, int], tuple[str, int]] = {}
        self._rank: dict[tuple[str, int], int] = {}

    def add(self, camera_id: str, track_id: int) -> None:
        key = (camera_id, track_id)
        if key not in self._parent:
            self._parent[key] = key
            self._rank[key] = 0

    def find(self, camera_id: str, track_id: int) -> tuple[str, int]:
        key = (camera_id, track_id)
        self.add(camera_id, track_id)

        # Path compression, iterative: a deep chain on a long-running node would
        # otherwise recurse further than Python allows.
        root = key
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[key] != root:
            self._parent[key], key = root, self._parent[key]
        return root

    def union(self, a: tuple[str, int], b: tuple[str, int]) -> None:
        root_a = self.find(*a)
        root_b = self.find(*b)
        if root_a == root_b:
            return

        if self._rank[root_a] < self._rank[root_b]:
            root_a, root_b = root_b, root_a
        self._parent[root_b] = root_a
        if self._rank[root_a] == self._rank[root_b]:
            self._rank[root_a] += 1

    def groups(self) -> list[set[tuple[str, int]]]:
        buckets: dict[tuple[str, int], set[tuple[str, int]]] = {}
        for key in self._parent:
            buckets.setdefault(self.find(*key), set()).add(key)
        return list(buckets.values())

    def distinct_count(self) -> int:
        return len(self.groups())


# ----------------------------------------------------------------- association


@dataclass(frozen=True, slots=True)
class Association:
    """A judgement that two tracks are the same object.

    ``score`` is not a probability. It is how well the observation supports the
    claim, given the uncertainty in both positions, and it is reported alongside
    the reasons so an operator can disagree with it.
    """

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


def associate(
    events: Sequence[Event],
    *,
    window_millis: int = DEFAULT_WINDOW_MILLIS,
    radius_meters: float = DEFAULT_RADIUS_METERS,
) -> list[Association]:
    """Decide which tracks across cameras are the same object.

    Deliberately conservative, and deliberately weak. Without appearance
    features, position and time are all there is, so this will merge two people
    who walked the same spot ten seconds apart and will fail to merge one person
    whose two cameras disagree about where they were. Both failures are visible
    in the association's own ``score`` and ``reasons`` rather than hidden inside
    a threshold.

    A same-camera pair is never associated here: within one camera, identity is
    the tracker's job, and second-guessing it from positions would undo the
    tracker's own evidence.
    """
    associations: list[Association] = []

    for index, first in enumerate(events):
        for second in events[index + 1 :]:
            key_a = (first.evidence.camera_id, first.evidence.track_id)
            key_b = (second.evidence.camera_id, second.evidence.track_id)

            if key_a == key_b or key_a[0] == key_b[0]:
                continue

            gap = abs(second.occurred_at_millis - first.occurred_at_millis)
            if gap > window_millis:
                continue

            point_a, point_b = _position_of(first), _position_of(second)
            if point_a is None or point_b is None:
                # Two unplaced cameras cannot support a claim that they saw the
                # same object. Merging them on timing alone would collapse an
                # entire site into one incident.
                continue

            separation = haversine_distance(point_a, point_b)

            # Each position's own uncertainty widens the gate, capped so one
            # badly-placed camera cannot swallow the site.
            slack = min(
                MAX_UNCERTAINTY_ALLOWANCE_METERS,
                (first.evidence.position_uncertainty_meters or 0.0)
                + (second.evidence.position_uncertainty_meters or 0.0),
            )
            allowance = radius_meters + slack

            if separation > allowance:
                continue

            proximity = 1.0 - separation / allowance
            recency = 1.0 - gap / window_millis
            score = round(0.65 * proximity + 0.35 * recency, 4)

            associations.append(
                Association(
                    a=key_a,
                    b=key_b,
                    score=score,
                    separation_meters=round(separation, 2),
                    allowance_meters=round(allowance, 2),
                    time_gap_millis=gap,
                    reasons=(
                        f"{separation:.1f} m apart, within {allowance:.1f} m "
                        f"allowed by the two position uncertainties",
                        f"{gap / 1000:.1f} s apart, within {window_millis / 1000:.0f} s",
                    ),
                )
            )

    return associations


# ---------------------------------------------------------------- risk scoring


@dataclass(frozen=True, slots=True)
class RiskFactor:
    name: str
    points: float
    because: str


@dataclass(frozen=True, slots=True)
class Risk:
    """A score with the reasoning that produced it.

    0–100, and always accompanied by its factors. A bare number invites an
    operator to calibrate against it without understanding it, and then to
    ignore it when it is wrong once.
    """

    score: float
    factors: tuple[RiskFactor, ...]

    @property
    def band(self) -> Severity:
        if self.score >= 80:
            return Severity.CRITICAL
        if self.score >= 60:
            return Severity.HIGH
        if self.score >= 35:
            return Severity.MEDIUM
        if self.score >= 15:
            return Severity.LOW
        return Severity.INFO

    def describe(self) -> str:
        lines = [f"risk {self.score:.0f}/100 ({self.band.value})"]
        for factor in self.factors:
            lines.append(f"  {factor.points:+5.0f}  {factor.name}: {factor.because}")
        return "\n".join(lines)


#: Points contributed by the most severe event in an incident.
_SEVERITY_POINTS = {
    Severity.INFO: 5.0,
    Severity.LOW: 15.0,
    Severity.MEDIUM: 30.0,
    Severity.HIGH: 45.0,
    Severity.CRITICAL: 60.0,
}


def score_risk(
    events: Sequence[Event],
    distinct_objects: int,
    cameras: Sequence[str],
    duration_millis: int,
    zone_kinds: Iterable[ZoneKind] = (),
) -> Risk:
    """Combine an incident's properties into a score, showing the working.

    The weights are judgements, not measurements, and they are stated here in one
    place so they can be argued with. What is *not* a judgement is the shape: the
    score rises with corroboration (more cameras, more objects, longer duration)
    because each of those makes the underlying observation harder to explain away.
    """
    factors: list[RiskFactor] = []

    worst = max(events, key=lambda e: severity_rank(e.severity)) if events else None
    if worst is not None:
        factors.append(
            RiskFactor(
                "severity",
                _SEVERITY_POINTS[worst.severity],
                f"most serious event is {worst.severity.value} ({worst.type.value})",
            )
        )

    if distinct_objects > 1:
        # Capped: the difference between one intruder and three matters; the
        # difference between nine and eleven does not.
        points = min(20.0, 7.0 * (distinct_objects - 1))
        factors.append(
            RiskFactor("group", points, f"{distinct_objects} distinct objects involved")
        )

    if len(cameras) > 1:
        points = min(15.0, 7.5 * (len(cameras) - 1))
        factors.append(
            RiskFactor(
                "corroboration",
                points,
                f"seen by {len(cameras)} cameras, which is harder to explain as a "
                "single camera's error",
            )
        )

    minutes = duration_millis / 60_000
    if minutes >= 1.0:
        points = min(15.0, 5.0 * minutes)
        factors.append(
            RiskFactor("duration", points, f"sustained for {minutes:.1f} minutes")
        )

    kinds = set(zone_kinds)
    if ZoneKind.RESTRICTED in kinds:
        factors.append(
            RiskFactor("location", 10.0, "occurred in a restricted zone")
        )
    elif ZoneKind.PERIMETER in kinds:
        factors.append(RiskFactor("location", 6.0, "occurred at the perimeter"))

    # Confidence scales the total rather than adding to it. An incident built
    # from weak observations should not reach the top band by accumulating
    # circumstances; the evidence has to carry it.
    if events:
        confidence = sum(e.confidence for e in events) / len(events)
        raw = sum(f.points for f in factors)
        scaled = raw * (0.55 + 0.45 * confidence)
        if confidence < 0.999:
            factors.append(
                RiskFactor(
                    "confidence",
                    round(scaled - raw, 1),
                    f"mean event confidence {confidence:.2f} scales the total",
                )
            )
        total = scaled
    else:
        total = 0.0

    return Risk(score=round(min(100.0, max(0.0, total)), 1), factors=tuple(factors))


# -------------------------------------------------------------------- incident


@dataclass(frozen=True, slots=True)
class TimelineEntry:
    at_millis: int
    at: datetime
    camera_id: str
    summary: str
    severity: Severity
    event_id: str


@dataclass(frozen=True, slots=True)
class Incident:
    """One thing that happened, assembled from the events that evidence it."""

    id: str
    #: The most serious severity among its events, unless risk says worse.
    severity: Severity
    summary: str
    opened_at_millis: int
    closed_at_millis: int
    opened_at: datetime
    #: Distinct objects, from union-find roots — never a count of track segments.
    distinct_objects: int
    cameras: tuple[str, ...]
    zones: tuple[str, ...]
    events: tuple[Event, ...]
    associations: tuple[Association, ...]
    risk: Risk

    @property
    def duration_millis(self) -> int:
        return self.closed_at_millis - self.opened_at_millis

    def timeline(self) -> tuple[TimelineEntry, ...]:
        return tuple(
            TimelineEntry(
                at_millis=event.occurred_at_millis,
                at=event.occurred_at,
                camera_id=event.evidence.camera_id,
                summary=event.summary,
                severity=event.severity,
                event_id=event.id,
            )
            for event in sorted(self.events, key=lambda e: e.occurred_at_millis)
        )

    def describe(self) -> str:
        lines = [
            f"{self.id}  [{self.severity.value}]  {self.summary}",
            f"  when       {self.opened_at:%Y-%m-%d %H:%M:%S}, "
            f"lasting {self.duration_millis / 1000:.1f}s",
            f"  objects    {self.distinct_objects}",
            f"  cameras    {', '.join(self.cameras)}",
            f"  zones      {', '.join(self.zones) if self.zones else '—'}",
            f"  events     {len(self.events)}",
        ]
        lines.append("  " + self.risk.describe().replace("\n", "\n  "))
        lines.append("  timeline")
        for entry in self.timeline():
            lines.append(
                f"    t+{entry.at_millis / 1000:6.1f}s  [{entry.severity.value:8}] "
                f"{entry.camera_id}  {entry.summary}"
            )
        return "\n".join(lines)


def incident_id(opening_event: Event) -> str:
    """Deterministic, derived from the event that opened the incident.

    The same reasoning as event ids, one level up: replaying footage must produce
    the same incident rather than a second one beside the first.
    """
    material = f"{opening_event.id}|{opening_event.evidence.camera_id}"
    return "inc_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:20].upper()


@dataclass
class CorrelationStats:
    events_in: int = 0
    incidents_out: int = 0
    associations: int = 0

    @property
    def reduction(self) -> float:
        """How much less there is for a person to read. The primary output."""
        if self.events_in == 0:
            return 0.0
        return round(1.0 - self.incidents_out / self.events_in, 4)


class Correlator:
    """Groups events into incidents.

    Batch rather than streaming, deliberately. An incident is a statement about a
    span of time, and deciding it is closed requires knowing nothing more is
    coming — which a streaming correlator can only guess at. A live deployment
    runs this over a sliding window; the semantics stay the same.
    """

    def __init__(
        self,
        *,
        window_millis: int = DEFAULT_WINDOW_MILLIS,
        radius_meters: float = DEFAULT_RADIUS_METERS,
        zone_kinds: dict[str, ZoneKind] | None = None,
    ):
        self._window = window_millis
        self._radius = radius_meters
        self._zone_kinds = zone_kinds or {}
        self.stats = CorrelationStats()

    def correlate(self, events: Sequence[Event]) -> list[Incident]:
        if not events:
            return []

        ordered = sorted(events, key=lambda e: (e.occurred_at_millis, e.id))
        self.stats.events_in += len(ordered)

        links = associate(
            ordered, window_millis=self._window, radius_meters=self._radius
        )
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

    def _group(
        self, events: Sequence[Event], identity: ObjectIdentity
    ) -> list[list[Event]]:
        """Partition events into incidents.

        Two events belong together when they concern the same object, or when
        they are close in both time and space. The second condition is what puts
        two different people who breached the same fence together in one
        incident, which is what an operator wants: it is one breach, with two
        people in it.
        """
        groups: list[list[Event]] = []
        roots: list[set[tuple[str, int]]] = []

        for event in events:
            root = identity.find(event.evidence.camera_id, event.evidence.track_id)
            position = _position_of(event)

            target = None
            for index, group in enumerate(groups):
                if root in roots[index]:
                    target = index
                    break
                if self._is_near(event, position, group):
                    target = index
                    break

            if target is None:
                groups.append([event])
                roots.append({root})
            else:
                groups[target].append(event)
                roots[target].add(root)

        return groups

    def _is_near(
        self, event: Event, position: LatLon | None, group: Sequence[Event]
    ) -> bool:
        for other in group:
            if abs(event.occurred_at_millis - other.occurred_at_millis) > self._window:
                continue

            other_position = _position_of(other)
            if position is None or other_position is None:
                # Same camera and same window is enough when neither is placed:
                # one unplaced camera's events over two minutes are far more
                # likely to be one situation than several.
                if event.evidence.camera_id == other.evidence.camera_id:
                    return True
                continue

            slack = min(
                MAX_UNCERTAINTY_ALLOWANCE_METERS,
                (event.evidence.position_uncertainty_meters or 0.0)
                + (other.evidence.position_uncertainty_meters or 0.0),
            )
            if haversine_distance(position, other_position) <= self._radius + slack:
                return True

        return False

    def _build(
        self,
        group: Sequence[Event],
        links: Sequence[Association],
        identity: ObjectIdentity,
    ) -> Incident:
        ordered = sorted(group, key=lambda e: e.occurred_at_millis)
        opening = ordered[0]

        members = {
            (e.evidence.camera_id, e.evidence.track_id) for e in ordered
        }
        distinct = len({identity.find(*key) for key in members})

        cameras = tuple(sorted({e.evidence.camera_id for e in ordered}))
        zone_names = tuple(
            sorted({e.zone_name for e in ordered if e.zone_name is not None})
        )
        zone_ids = {e.zone_id for e in ordered if e.zone_id is not None}
        kinds = [self._zone_kinds[z] for z in zone_ids if z in self._zone_kinds]

        duration = ordered[-1].occurred_at_millis - opening.occurred_at_millis
        risk = score_risk(ordered, distinct, cameras, duration, kinds)

        worst = max(ordered, key=lambda e: severity_rank(e.severity)).severity
        severity = worst if severity_rank(worst) >= severity_rank(risk.band) else risk.band

        relevant = tuple(
            link for link in links if link.a in members and link.b in members
        )

        return Incident(
            id=incident_id(opening),
            severity=severity,
            summary=self._summarise(ordered, distinct, cameras, zone_names),
            opened_at_millis=opening.occurred_at_millis,
            closed_at_millis=ordered[-1].occurred_at_millis,
            opened_at=opening.occurred_at,
            distinct_objects=distinct,
            cameras=cameras,
            zones=zone_names,
            events=tuple(ordered),
            associations=relevant,
            risk=risk,
        )

    def _summarise(
        self,
        events: Sequence[Event],
        distinct: int,
        cameras: Sequence[str],
        zones: Sequence[str],
    ) -> str:
        """One line, saying only what the evidence supports.

        The count is of distinct objects, and the noun is "object" unless every
        contributing detector actually classified. A summary that says "3 people"
        on the strength of motion blobs is the fabrication this system is built
        to avoid.
        """
        classified = {
            e.evidence.class_label
            for e in events
            if e.evidence.detector_classifies
        }
        if classified and len(classified) == 1 and all(
            e.evidence.detector_classifies for e in events
        ):
            noun = next(iter(classified)).replace("_", " ")
            subject = f"{distinct} {noun}" if distinct == 1 else f"{distinct} {noun}s"
        else:
            subject = "1 object" if distinct == 1 else f"{distinct} objects"

        where = f" in {zones[0]}" if len(zones) == 1 else (
            f" across {len(zones)} zones" if zones else ""
        )
        seen = f" ({len(cameras)} cameras)" if len(cameras) > 1 else ""

        return f"{subject}{where}{seen}"
