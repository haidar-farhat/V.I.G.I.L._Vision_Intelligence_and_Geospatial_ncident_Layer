"""Events: the point where the system starts making claims.

... -> TEMPORAL CONTEXT -> EVENT ANALYSIS -> ...

Everything before this module reports observations. An event is different: it is
an assertion that something *happened*, and it is what eventually interrupts a
person. That difference sets the rules this module lives by.

**Every event carries its own evidence.** Not a reference to evidence stored
somewhere — the actual grounds, attached, so the claim can be judged months later
by someone who was not there. What the detector was, which model with which
digest, which camera, which track, how long, how confident, and which conditions
fired. An event that cannot answer "why did you say that" should never have been
raised.

**Event identity is deterministic.** The id is derived from what the event *is*,
not from when it was recorded, so replaying the same footage produces the same
ids. Persisting a replayed event is an idempotent upsert rather than a duplicate,
and after a multi-node outage delivers the same batch twice the result converges.
This is the property that makes at-least-once delivery safe, and it has to be
designed in rather than added later.

**Fewer events is the goal.** The measure of this system is how little it says.
One person loitering for four minutes is one event, not two hundred and forty.
Rules debounce, presences persist across dropouts, and a rule that fires
repeatedly for an ongoing condition is a defect rather than a feature.

**A zone's class filter is honoured by every rule that acts on a presence.**
A restricted area used to fire on whatever the detector named — a couch, a
bottle — and an operator who sees a sofa raise a HIGH incident stops believing
incidents. Each such rule asks :meth:`~sentinel.zones.Zone.watches` with
:attr:`RuleContext.class_label`, the one place the detector's word for a track
is read, so a rule cannot reach the label by a second route and disagree with
the first about what a motion detector can say (nothing).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone, tzinfo
from enum import Enum
from typing import Iterable, Sequence

from .core import Track
from .detect import DetectorInfo
from .zones import Presence, PresenceChange, Zone, ZoneKind


class EventType(str, Enum):
    """What kind of thing happened.

    Not every member is produced. A stored event names its type as a string, so
    removing a member would make an old database unreadable and re-using one for
    something else would silently reinterpret history — which is why the unused
    ones stay and are labelled instead of being deleted.

    Which is which is asserted in `test_events.py`, against the rules that
    actually exist, so this comment cannot quietly stop being true.
    """

    #: Produced today, each by the rule named beside it.
    ZONE_ENTRY = "ZONE_ENTRY"  # ZoneEntryRule
    LOITERING = "LOITERING"  # LoiteringRule
    AFTER_HOURS_PRESENCE = "AFTER_HOURS_PRESENCE"  # AfterHoursRule
    RAPID_MOVEMENT = "RAPID_MOVEMENT"  # RapidMovementRule

    #: Reserved. No rule raises these yet. `ZONE_EXIT` waits on a rule that is
    #: worth having — an exit is only interesting in context, and one per
    #: departure is exactly the alert fatigue this system exists to avoid.
    #: `PERIMETER_BREACH` waits on a line-crossing test, which is a different
    #: predicate from polygon containment and is not written.
    ZONE_EXIT = "ZONE_EXIT"
    PERIMETER_BREACH = "PERIMETER_BREACH"


class Severity(str, Enum):
    """How much of a person's attention this deserves.

    Deliberately coarse. A ten-point scale invites arguments about whether
    something is a 6 or a 7, and an operator reads the colour anyway.
    """

    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


#: Ordering, for picking the most serious of a set.
_SEVERITY_ORDER = {s: index for index, s in enumerate(Severity)}


def severity_rank(severity: Severity) -> int:
    return _SEVERITY_ORDER[severity]


@dataclass(frozen=True, slots=True)
class Evidence:
    """The grounds for an event.

    Every field here answers a question an operator or an auditor will actually
    ask. Nothing is optional decoration: an event without a detector name cannot
    be re-examined when that detector turns out to be miscalibrated, and one
    without a model digest cannot be re-examined when the weights are replaced.
    """

    camera_id: str
    track_id: int
    #: Milliseconds, in the source's own timeline.
    first_seen_millis: int
    last_seen_millis: int
    #: How many frames the detector actually confirmed this object in.
    observations: int
    #: What produced the detections.
    detector: str
    detector_classifies: bool
    model_digest: str | None
    class_label: str
    #: Where, and how well that is known.
    latitude: float | None
    longitude: float | None
    position_uncertainty_meters: float | None
    position_source: str | None
    #: Motion, where it is known. ``None`` means unknown, ``0.0`` means still.
    speed_mps: float | None
    heading_degrees: float | None
    #: Frame indices an operator can jump to. Bounded — this is a pointer into
    #: the recording, not a copy of it.
    frame_indices: tuple[int, ...] = ()

    def describe(self) -> str:
        parts = [
            f"camera {self.camera_id}",
            f"track #{self.track_id}",
            f"{self.observations} observations",
            f"detector {self.detector}",
        ]
        if self.model_digest:
            parts.append(f"model {self.model_digest[:12]}")
        if self.latitude is not None and self.position_uncertainty_meters is not None:
            parts.append(
                f"at {self.latitude:.6f}, {self.longitude:.6f} "
                f"±{self.position_uncertainty_meters:.1f} m"
            )
        else:
            parts.append("position unknown")
        return "; ".join(parts)


@dataclass(frozen=True, slots=True)
class Event:
    """Something the system asserts happened."""

    id: str
    type: EventType
    severity: Severity
    #: Human-readable, and never more specific than the evidence supports.
    summary: str
    occurred_at_millis: int
    #: Wall-clock, kept separately from the media timeline because the
    #: difference between them is itself evidence about the deployment.
    occurred_at: datetime
    zone_id: str | None
    zone_name: str | None
    rule_id: str
    evidence: Evidence
    #: What in the rule actually fired, in the rule's own terms.
    triggering_conditions: tuple[str, ...]
    #: 0..1. How much this rests on confident observation rather than inference.
    confidence: float

    def describe(self) -> str:
        return (
            f"[{self.severity.value}] {self.summary}\n"
            f"  when       {self.occurred_at:%Y-%m-%d %H:%M:%S} "
            f"(t+{self.occurred_at_millis / 1000:.1f}s)\n"
            f"  because    {'; '.join(self.triggering_conditions)}\n"
            f"  evidence   {self.evidence.describe()}\n"
            f"  confidence {self.confidence:.2f}\n"
            f"  id         {self.id}"
        )


def event_id(
    node_id: str,
    camera_id: str,
    rule_id: str,
    event_type: EventType,
    track_id: int,
    occurred_at_millis: int,
    bucket_millis: int = 1000,
) -> str:
    """A deterministic id for an event.

    Derived from what the event *is*, never from when it was recorded or from a
    counter. Two consequences, both of which the distributed design depends on:

    - **Replay is idempotent.** Re-running the same footage produces the same
      ids, so persisting a replayed event is an upsert rather than a duplicate
      row, and an incident built from replayed events is the same incident.
    - **At-least-once delivery is safe.** A worker that reconnects after an
      outage resends everything after the last acknowledged sequence. The control
      node sees duplicates, and they collapse.

    The timestamp is bucketed because two nodes observing the same event will not
    agree on the millisecond. A one-second bucket is coarse enough to absorb that
    and fine enough that two genuinely separate events a second apart stay
    separate.
    """
    bucket = occurred_at_millis // max(1, bucket_millis)
    material = "|".join(
        [node_id, camera_id, rule_id, event_type.value, str(track_id), str(bucket)]
    )
    return "ev_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


# --------------------------------------------------------------------- rules


@dataclass(frozen=True, slots=True)
class RuleContext:
    """Everything a rule may look at.

    Passed as one object so a rule cannot reach for state nobody gave it. A rule
    that consults the wall clock, a file, or a network is not reviewable, and an
    unreviewable rule is one nobody can defend after it fires wrongly.
    """

    node_id: str
    camera_id: str
    zone: Zone | None
    track: Track | None
    presence: Presence | None
    at_millis: int
    #: When, in UTC. What `Event.occurred_at` records.
    moment: datetime
    detector: DetectorInfo
    frame_index: int
    #: The clock schedules are written in, for anything a rule *says* about
    #: the time. ``None`` means UTC.
    site_tz: tzinfo | None = None

    @property
    def local_moment(self) -> datetime:
        """The moment in the site's clock, for text a person reads."""
        return self.moment.astimezone(self.site_tz) if self.site_tz else self.moment

    @property
    def class_label(self) -> str | None:
        """What the detector calls this track, or ``None`` when it cannot say.

        ``None`` rather than ``"unclassified"`` under a motion detector, and
        that difference is load-bearing: a zone filtered to ``person`` is asked
        with this value, and a detector that classifies nothing must never
        satisfy it — a blob is not a person, however confidently it moved. The
        string ``"unclassified"`` is reserved for a classifying detector that
        looked and could not decide, which a filter may legitimately name.
        """
        if self.track is None or not self.detector.classifies:
            return None
        return self.detector.label_for(self.track.class_id)


class Rule:
    """A condition that turns observations into an assertion.

    Subclasses implement :meth:`evaluate`. A rule returns events or nothing; it
    never mutates anything, never raises for ordinary input, and never looks
    outside its context.
    """

    id: str = "rule"
    description: str = ""
    severity: Severity = Severity.INFO
    event_type: EventType = EventType.ZONE_ENTRY

    def on_presence_change(
        self, change: PresenceChange, context: RuleContext
    ) -> list[Event]:
        return []

    def on_frame(self, context: RuleContext) -> list[Event]:
        return []

    # -------------------------------------------------------------- helpers

    @staticmethod
    def _watched(context: RuleContext) -> bool:
        """Whether the zone in this context acts on this track's class at all.

        The check every rule about a presence makes before anything else, so
        that a filter set on a zone silences all of them together. A rule that
        skipped it would raise "loitering" for the couch the entry rule had
        just declined to announce, and the operator would be back to
        disbelieving the screen. A context with no zone is not filtered: there
        is no filter to consult.
        """
        return context.zone is None or context.zone.watches(context.class_label)

    def _build(
        self,
        context: RuleContext,
        *,
        summary: str,
        conditions: Sequence[str],
        confidence: float,
        severity: Severity | None = None,
        observations: int | None = None,
    ) -> Event:
        track = context.track
        assert track is not None, "a rule built an event with no track to attribute it to"

        position = track.position
        evidence = Evidence(
            camera_id=context.camera_id,
            track_id=track.id,
            first_seen_millis=track.first_seen_millis,
            last_seen_millis=track.last_seen_millis,
            observations=observations if observations is not None else track.hits,
            detector=context.detector.name,
            detector_classifies=context.detector.classifies,
            model_digest=context.detector.model_sha256,
            class_label=context.detector.label_for(track.class_id),
            latitude=position.point.lat if position else None,
            longitude=position.point.lon if position else None,
            position_uncertainty_meters=position.radius_meters if position else None,
            position_source=position.source if position else None,
            speed_mps=track.speed_mps,
            heading_degrees=track.heading_degrees,
            frame_indices=(context.frame_index,),
        )

        chosen = severity or self.severity
        return Event(
            id=event_id(
                context.node_id,
                context.camera_id,
                self.id,
                self.event_type,
                track.id,
                context.at_millis,
            ),
            type=self.event_type,
            severity=chosen,
            summary=summary,
            occurred_at_millis=context.at_millis,
            occurred_at=context.moment,
            zone_id=context.zone.id if context.zone else None,
            zone_name=context.zone.name if context.zone else None,
            rule_id=self.id,
            evidence=evidence,
            triggering_conditions=tuple(conditions),
            confidence=round(min(1.0, max(0.0, confidence)), 4),
        )


class ZoneEntryRule(Rule):
    """Something entered a zone that should not have anything in it."""

    id = "zone-entry"
    description = "An object entered a restricted or perimeter zone."
    event_type = EventType.ZONE_ENTRY
    severity = Severity.MEDIUM

    def __init__(self, kinds: Iterable[ZoneKind] = (ZoneKind.RESTRICTED,)):
        self._kinds = frozenset(kinds)

    def on_presence_change(
        self, change: PresenceChange, context: RuleContext
    ) -> list[Event]:
        zone = context.zone
        if change.kind != "ENTERED" or zone is None or zone.kind not in self._kinds:
            return []
        if not self._watched(context):
            # A couch in a person-only zone. Nothing to say, and saying it
            # anyway is the incident this filter exists to stop.
            return []

        presence = change.presence
        severity = (
            Severity.HIGH if zone.kind is ZoneKind.RESTRICTED else Severity.MEDIUM
        )

        # The summary says what the detector can support and no more. Under a
        # motion detector this reads "An object entered", not "A person entered".
        label = context.class_label
        subject = "An object" if label is None else f"A {label.replace('_', ' ')}"

        return [
            self._build(
                context,
                summary=f"{subject} entered {zone.name}",
                conditions=[
                    f"membership held for {zone.enter_after_millis} ms",
                    f"zone kind is {zone.kind.value}",
                ],
                confidence=presence.confidence,
                severity=severity,
                observations=presence.observations,
            )
        ]


class LoiteringRule(Rule):
    """Something stayed in one place longer than it should have.

    Fires **once** per presence. An ongoing condition reported every frame is the
    alert-fatigue failure mode: the operator stops reading, and the one that
    mattered is in the middle of two hundred identical lines.
    """

    id = "loitering"
    description = "An object remained in a zone beyond the dwell threshold."
    event_type = EventType.LOITERING
    severity = Severity.MEDIUM

    def __init__(self, dwell_millis: int = 30_000, still_speed_mps: float = 0.5):
        self._dwell = dwell_millis
        self._still = still_speed_mps
        self._fired: set[tuple[str, int]] = set()

    def on_frame(self, context: RuleContext) -> list[Event]:
        presence, zone, track = context.presence, context.zone, context.track
        if presence is None or zone is None or track is None:
            return []
        if not self._watched(context):
            # Before the dwell test and before `_fired`, so an unwatched class
            # neither raises nor uses up the one firing this presence gets.
            return []

        key = (zone.id, track.id)
        if key in self._fired:
            return []
        if presence.duration_millis < self._dwell:
            return []

        self._fired.add(key)

        conditions = [
            f"present for {presence.duration_millis / 1000:.0f}s, "
            f"threshold {self._dwell / 1000:.0f}s"
        ]
        # Standing still is a stronger signal than merely being present, and the
        # distinction only exists because motion has three states rather than
        # two: "not moving" is a fact, "motion unknown" is not.
        if track.speed_mps is not None and track.speed_mps < self._still:
            conditions.append(f"stationary at {track.speed_mps:.2f} m/s")
        elif track.speed_mps is None:
            conditions.append("motion could not be determined")

        return [
            self._build(
                context,
                summary=(
                    f"An object remained in {zone.name} for "
                    f"{presence.duration_millis / 1000:.0f} seconds"
                ),
                conditions=conditions,
                confidence=presence.confidence,
                observations=presence.observations,
            )
        ]

    def forget(self, zone_id: str, track_id: int) -> None:
        self._fired.discard((zone_id, track_id))


class AfterHoursRule(Rule):
    """Presence during a window when the site should be empty.

    Separate from :class:`ZoneEntryRule` because the zone and the schedule are
    different facts: the same area is unremarkable at 14:00 and worth waking
    somebody for at 03:00, and an operator reading the event needs to see that
    the *time* is why.
    """

    id = "after-hours"
    description = "An object was present during a scheduled closed period."
    event_type = EventType.AFTER_HOURS_PRESENCE
    severity = Severity.HIGH

    def on_presence_change(
        self, change: PresenceChange, context: RuleContext
    ) -> list[Event]:
        zone = context.zone
        if change.kind != "ENTERED" or zone is None or zone.schedule is None:
            return []
        if not self._watched(context):
            # The hour makes a person worth waking somebody for; it does not
            # make a couch one.
            return []

        return [
            self._build(
                context,
                summary=f"An object was in {zone.name} outside permitted hours",
                conditions=[
                    # The site's clock, with its offset, because the schedule
                    # was written in it and the reader will check it against a
                    # wall clock — and `occurred_at` stays UTC beside it.
                    f"{context.local_moment:%H:%M UTC%z} falls within {zone.schedule.describe()}",
                    f"presence confirmed after {zone.enter_after_millis} ms",
                ],
                confidence=change.presence.confidence,
                observations=change.presence.observations,
            )
        ]


class RapidMovementRule(Rule):
    """Something moved faster than the setting explains.

    Speeds derived from ground projection are noisy near the horizon, where a
    pixel is metres. So this rule requires a *confident* position as well as a
    high speed — otherwise it fires on projection error rather than on anything
    that happened, which is the failure mode that makes speed rules distrusted.

    Deliberately not subject to a zone's class filter. This rule is about the
    track, not about where it is: the zone in its context, when there is one,
    is incidental, and a thing moving at 12 m/s is worth a LOW event whatever
    the detector calls it. Asserted in `test_events.py` so the omission cannot
    be mistaken for an oversight.
    """

    id = "rapid-movement"
    description = "An object moved faster than the threshold for this site."
    event_type = EventType.RAPID_MOVEMENT
    severity = Severity.LOW

    def __init__(self, speed_mps: float = 6.0, max_uncertainty_meters: float = 3.0):
        self._speed = speed_mps
        self._max_uncertainty = max_uncertainty_meters
        self._fired: set[int] = set()

    def on_frame(self, context: RuleContext) -> list[Event]:
        track = context.track
        if track is None or track.speed_mps is None or track.position is None:
            return []
        if track.id in self._fired:
            return []
        if track.speed_mps < self._speed:
            return []

        if track.position.radius_meters > self._max_uncertainty:
            # Deliberately silent. A 12 m/s reading from a position known to
            # ±9 m is a measurement of the projection, not of the object.
            return []

        self._fired.add(track.id)
        return [
            self._build(
                context,
                summary=f"An object moved at {track.speed_mps:.1f} m/s",
                conditions=[
                    f"speed {track.speed_mps:.1f} m/s exceeds {self._speed:.1f} m/s",
                    f"position confident to ±{track.position.radius_meters:.1f} m",
                ],
                confidence=1.0,
            )
        ]


def default_rules(zones: Sequence[Zone] = ()) -> list[Rule]:
    """The rule set a caller gets when it does not choose one.

    Matched to what is actually configured. Without a zone there is nothing to
    be inside, so the zone rules would be dead weight — and worse, a run would
    report "0 events" for a reason that has nothing to do with the footage.

    One definition, because there were two: the console and the CLI each had
    their own copy, and a third was about to appear in the node. Rule sets that
    drift produce two deployments that disagree about what an incident is.
    """
    if not zones:
        return [RapidMovementRule(speed_mps=6.0)]
    return [
        ZoneEntryRule(),
        AfterHoursRule(),
        LoiteringRule(dwell_millis=8000),
        RapidMovementRule(speed_mps=6.0),
    ]


# -------------------------------------------------------------------- engine


@dataclass
class EventEngineStats:
    events: int = 0
    suppressed_duplicates: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    by_severity: dict[str, int] = field(default_factory=dict)


class EventEngine:
    """Runs rules over zone activity and emits events.

    Holds a set of ids it has already emitted, so a rule that fires twice for the
    same deterministic event produces one. That is belt and braces on top of the
    rules' own debouncing: rules are the part most likely to be edited by someone
    who does not know the whole system, and a duplicate reaching an operator is
    the thing this codebase is least willing to allow.
    """

    __slots__ = ("_rules", "_node_id", "_camera_id", "_seen", "stats", "_site_tz",)

    def __init__(
        self, rules: Sequence[Rule], *, node_id: str, camera_id: str,
        site_tz: tzinfo | None = None,
    ):
        self._rules = list(rules)
        self._node_id = node_id
        self._camera_id = camera_id
        self._site_tz = site_tz
        self._seen: set[str] = set()
        self.stats = EventEngineStats()

    @property
    def rules(self) -> tuple[Rule, ...]:
        return tuple(self._rules)

    def on_presence_changes(
        self,
        changes: Sequence[PresenceChange],
        zones: dict[str, Zone],
        tracks: dict[int, Track],
        *,
        at_millis: int,
        moment: datetime,
        detector: DetectorInfo,
        frame_index: int,
    ) -> list[Event]:
        produced: list[Event] = []
        for change in changes:
            track = tracks.get(change.presence.track_id)
            if track is None:
                # The presence closed because the track ended. There is nothing
                # to attribute an event to, and inventing one would be worse.
                continue

            context = RuleContext(
                node_id=self._node_id,
                camera_id=self._camera_id,
                zone=zones.get(change.presence.zone_id),
                track=track,
                presence=change.presence,
                at_millis=at_millis,
                moment=moment,
                site_tz=self._site_tz,
                detector=detector,
                frame_index=frame_index,
            )
            for rule in self._rules:
                produced.extend(rule.on_presence_change(change, context))

        return self._accept(produced)

    def on_frame(
        self,
        presences: Sequence[Presence],
        zones: dict[str, Zone],
        tracks: dict[int, Track],
        *,
        at_millis: int,
        moment: datetime,
        detector: DetectorInfo,
        frame_index: int,
    ) -> list[Event]:
        produced: list[Event] = []

        # Rules that watch a presence.
        for presence in presences:
            track = tracks.get(presence.track_id)
            if track is None:
                continue
            context = RuleContext(
                node_id=self._node_id,
                camera_id=self._camera_id,
                zone=zones.get(presence.zone_id),
                track=track,
                presence=presence,
                at_millis=at_millis,
                moment=moment,
                site_tz=self._site_tz,
                detector=detector,
                frame_index=frame_index,
            )
            for rule in self._rules:
                produced.extend(rule.on_frame(context))

        # Rules that watch a track regardless of any zone.
        in_a_zone = {p.track_id for p in presences}
        for track in tracks.values():
            if track.id in in_a_zone:
                continue
            context = RuleContext(
                node_id=self._node_id,
                camera_id=self._camera_id,
                zone=None,
                track=track,
                presence=None,
                at_millis=at_millis,
                moment=moment,
                site_tz=self._site_tz,
                detector=detector,
                frame_index=frame_index,
            )
            for rule in self._rules:
                produced.extend(rule.on_frame(context))

        return self._accept(produced)

    def _accept(self, events: Sequence[Event]) -> list[Event]:
        kept: list[Event] = []
        for event in events:
            if event.id in self._seen:
                self.stats.suppressed_duplicates += 1
                continue
            self._seen.add(event.id)
            kept.append(event)

            self.stats.events += 1
            self.stats.by_type[event.type.value] = (
                self.stats.by_type.get(event.type.value, 0) + 1
            )
            self.stats.by_severity[event.severity.value] = (
                self.stats.by_severity.get(event.severity.value, 0) + 1
            )
        return kept


def utc_from_millis(millis: int) -> datetime:
    return datetime.fromtimestamp(millis / 1000.0, tz=timezone.utc)
