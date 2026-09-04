"""Zones, schedules, and what a track is doing in them.

... -> SPATIAL CONTEXT -> TEMPORAL CONTEXT -> ...

A zone turns a position into a *meaning*. "An object at 33.8938, 35.5018" is
data; "an object inside Restricted Area A" is the beginning of a reason to wake
somebody up. This module is where the first becomes the second, and where the
qualifications that must travel with it are attached.

Three things it is careful about, each of which is a way real systems generate
false alarms:

**Uncertainty is not discarded at the boundary.** A track's position has a 1σ
radius that grows toward the horizon. An object 3 m outside a fence, known to
±8 m, is neither in nor out, and this module says so rather than picking one.
See :class:`~sentinel.core.ZoneMembership`.

**A boundary is not a trigger.** An object walking along a fence line crosses it
repeatedly as its estimate jitters. Membership must persist for a configured time
before it counts as a presence, and must be absent for a configured time before
the presence ends. Without that hysteresis one person produces forty events.

**Time is part of the condition.** The same person in the same place is
unremarkable at 14:00 and worth waking somebody for at 03:00. A zone carries a
schedule, and the schedule is evaluated in the site's clock — today the machine's
own zone, passed in as ``site_tz``; UTC when none is given — because that is
what "after hours" means to the person being woken.

**What the thing is, is part of the condition.** A restricted area used to fire
on any class the detector named — on a real camera, "1 couch in Room (HIGH,
risk 55)" and "A bottle entered Room" — and an operator who has seen a sofa
raise a HIGH incident stops believing incidents. A zone therefore carries a
class filter, :attr:`Zone.classes`, and the rules ask :meth:`Zone.watches`
before acting on a presence. The filter is by the detector's own label string,
because that is the only vocabulary a site has: the classes are whatever the
operator's model file names, and nothing here can know them in advance. Empty
means any, which is what every zone meant before the filter existed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timezone, tzinfo
from enum import Enum
from typing import Iterable, Sequence

from .core import LatLon, Track, ZoneMembership, zone_membership


class ZoneKind(str, Enum):
    """What a zone means, which decides what its presence implies.

    The kind is deliberately separate from the rules that act on it. A site has
    one restricted area and may have five rules about it, and the rules change
    far more often than the geography does.
    """

    #: Nobody should be here. Presence alone is the event.
    RESTRICTED = "RESTRICTED"
    #: The site boundary. Crossing it inbound matters; loitering outside it may.
    PERIMETER = "PERIMETER"
    #: A door, gate or lane where presence is expected and *absence* of an
    #: expected pattern (tailgating, wrong direction) is what matters.
    ENTRY = "ENTRY"
    #: Somewhere the system should deliberately ignore — a public pavement
    #: inside the camera's view, a tree that moves. Suppresses events.
    EXCLUSION = "EXCLUSION"
    #: Worth recording presence in, without implying anything is wrong.
    INTEREST = "INTEREST"


@dataclass(frozen=True, slots=True)
class Schedule:
    """When a zone's rules apply, in the site's local time.

    ``start`` after ``end`` means the window wraps midnight, which is what almost
    every real "after hours" schedule does. Getting that wrong silently disarms a
    site every night, so it is the case the tests lead with.

    ``days`` are ISO weekday numbers, Monday = 1. Empty means every day.
    """

    start: time
    end: time
    days: frozenset[int] = field(default_factory=frozenset)

    def covers(self, moment: datetime) -> bool:
        if self.days and moment.isoweekday() not in self.days:
            return False

        current = moment.time()
        if self.start <= self.end:
            return self.start <= current < self.end
        # Wraps midnight: 18:00 to 06:00 is "at or after 18:00, or before 06:00".
        return current >= self.start or current < self.end

    def describe(self) -> str:
        span = f"{self.start:%H:%M}–{self.end:%H:%M}"
        if not self.days:
            return span
        names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        return f"{span} on {', '.join(names[day - 1] for day in sorted(self.days))}"


#: Always active. A zone with no schedule applies at all times.
ALWAYS = None


def ring_problem(ring) -> str | None:
    """Why a ring is not a usable area, or ``None`` if it is.

    A self-intersecting outline (a figure of eight drawn by a slip of the
    mouse) has no inside — point-in-polygon gives a different answer depending
    on which lobe the point is in and which way the test happens to count —
    so a zone built on one would raise or suppress events at random. Shapely
    decides validity; that geometry is hard and its bugs are invisible until
    one arrangement of vertices produces a wrong answer.
    """
    if len(ring) < 3:
        return f"{len(ring)} points: two points are a line, not an area"
    from shapely.geometry import Polygon
    from shapely.validation import explain_validity

    polygon = Polygon([(p.lon, p.lat) for p in ring])
    # The hull, not the polygon: a figure of eight has two lobes whose signed
    # areas cancel to nothing, and would otherwise be reported as a line.
    # Collinear points have a hull that is a line, whose area is zero up to
    # rounding — 1e-13 square degrees is about a hand's breadth squared.
    if polygon.convex_hull.area < 1e-13:
        return "the points lie on a line and enclose no area"
    if not polygon.is_valid:
        # Shapely's text names the problem and roughly where, e.g.
        # "Self-intersection[35.5018 33.8938]". Plain enough to show an operator.
        return explain_validity(polygon).split("[")[0].strip().lower() or "invalid outline"
    return None


def zone_warnings(
    zone: "Zone",
    report,
    others: Sequence["Zone"] = (),
    *,
    labels: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Everything wrong with this zone that the geometry can prove.

    Warnings, never refusals. An operator who draws a zone somewhere no camera
    looks has made a mistake worth telling them about immediately — a zone that
    can never fire is the most dangerous object in the system, because it looks
    exactly like protection — but they may be about to place the camera that
    fixes it, and a tool that refuses the zone makes that impossible.

    ``report`` is anything carrying ``covered_fraction``, ``confident_fraction``
    and ``area_m2``; :class:`sentinel.coverage.ZoneReport` is what the console
    passes. Duck-typed on purpose, so that neither the tests nor a future
    caller has to build a full coverage report to ask this question.

    ``labels`` is the vocabulary of the detector watching this zone — the
    values of ``DetectorInfo.class_names`` — and it is the same kind of
    question as coverage: a filter naming a class the detector never emits is
    a zone that can never fire, and it reads on screen as a zone that filters.
    An empty vocabulary is a detector that labels nothing, under which any
    filter at all is dead. ``None`` means the caller does not know what
    detector will run, and the filter is not judged.
    """
    from shapely.geometry import Polygon

    messages: list[str] = []

    if zone.classes and labels is not None:
        known = frozenset(labels)
        if not known:
            messages.append(
                f"watches only {', '.join(sorted(zone.classes))}, but the detector "
                "labels nothing — it can never fire"
            )
        else:
            unknown = sorted(zone.classes - known)
            if len(unknown) == len(zone.classes):
                messages.append(
                    f"watches only {', '.join(unknown)}, which the detector never "
                    "names — it can never fire"
                )
            elif unknown:
                messages.append(
                    f"watches {', '.join(unknown)}, which the detector never names"
                )

    if report.covered_fraction <= 0.0:
        messages.append("no camera can see this zone — it can never fire")
    elif report.confident_fraction < 0.5 and not zone.accept_uncertain:
        # A zone whose own width is smaller than the error over it cannot say
        # which side of its line somebody is on. It will still raise
        # memberships; they will be UNCERTAIN, and a restricted area does not
        # act on those — so it is armed and silent, which is the worst state.
        beyond = round((1.0 - report.confident_fraction) * 100)
        messages.append(
            f"{beyond}% of this zone is beyond confident range "
            "(σ larger than half its width) — it will mostly report UNCERTAIN"
        )

    frame = _metric_frame(zone.ring[0])
    mine = Polygon([frame(point) for point in zone.ring])
    if not mine.is_valid:
        mine = mine.buffer(0)

    for other in others:
        if other.id == zone.id or len(other.ring) < 3:
            continue
        theirs = Polygon([frame(point) for point in other.ring])
        if not theirs.is_valid:
            theirs = theirs.buffer(0)
        shared = mine.intersection(theirs)
        if shared.is_empty or shared.area <= 0.0:
            continue

        if other.kind is zone.kind:
            # Two zones of one kind over the same ground raise two events for
            # one person, and the correlator has no way to know they were the
            # same fence drawn twice.
            messages.append(
                f"overlaps {other.name} ({other.kind.value}), {shared.area:.0f} m²"
            )
        elif other.kind is ZoneKind.EXCLUSION and zone.kind in (
            ZoneKind.RESTRICTED, ZoneKind.PERIMETER
        ):
            # An exclusion zone is "deliberately ignore this". Laid over an
            # alarm zone it silences it there, and nothing else on screen says
            # so: both are drawn, both look armed.
            messages.append(f"inside exclusion {other.name}: silenced there")

    schedule = zone.schedule
    if schedule is not None and schedule.start == schedule.end:
        # Not "all day": `covers` asks ``start <= now < end``, which no moment
        # satisfies when they are equal. The zone is disarmed permanently, and
        # it reads on screen as a zone with a schedule.
        messages.append(
            f"schedule {schedule.start:%H:%M}–{schedule.end:%H:%M} covers no time"
        )

    if report.area_m2 < 1.0:
        # Smaller than the ground a person stands on, and far smaller than the
        # position error anywhere on a real site.
        messages.append(f"area {report.area_m2:.1f} m² is under 1 m²")

    return tuple(messages)


def _metric_frame(origin: LatLon):
    """Metres east and north of ``origin``, for planar geometry.

    Areas in degrees are wrong by the cosine of the latitude, and an overlap
    reported as "0 m²" because of that is a warning nobody sees. Borrowed from
    `sentinel.coverage`, which owns this conversion, rather than copied —
    two tangent planes that came to differ would be a bug nobody could find.
    """
    from .coverage import _Frame

    frame = _Frame(origin)
    return frame.to_xy


@dataclass(frozen=True, slots=True)
class Zone:
    """A named area on the ground.

    ``ring`` is an open ring of at least three points; the closing edge is
    implied. ``min_confidence`` is the membership standard a presence must meet:
    a restricted area should demand :data:`ZoneMembership.INSIDE`, while a
    coverage or interest zone can accept :data:`ZoneMembership.UNCERTAIN`.
    """

    id: str
    name: str
    kind: ZoneKind
    ring: tuple[LatLon, ...]
    schedule: Schedule | None = ALWAYS
    #: How long membership must hold before it counts as a presence.
    enter_after_millis: int = 600
    #: How long absence must hold before the presence ends. Longer than the
    #: entry delay on purpose: a track that flickers out for one frame has not
    #: left, and treating it as if it had produces a second event when it
    #: reappears.
    exit_after_millis: int = 2000
    #: Whether an uncertain position may count as being in the zone.
    accept_uncertain: bool = False
    #: Which detector labels this zone acts on. EMPTY MEANS ANY.
    #:
    #: The labels are the detector's own strings — ``"person"`` here is
    #: whatever the operator's model calls ``"person"``, matched exactly —
    #: because a model file's names are the only vocabulary a site has, and a
    #: list fixed in this code would be a guess about weights it has never
    #: seen. Empty is the default because it is what every zone written before
    #: this field existed meant, and an outline that went quiet on upgrade
    #: would read on screen as protection.
    classes: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if len(self.ring) < 3:
            raise ValueError(
                f"Zone {self.id!r} has {len(self.ring)} points. Two points are a "
                "line, not an area, and a half-drawn zone must not start "
                "producing intrusion events."
            )
        problem = ring_problem(self.ring)
        if problem is not None:
            raise ValueError(f"Zone {self.id!r}: {problem}")
        if not isinstance(self.classes, frozenset):
            # A caller that hands over a list or a set has said what it meant.
            # Held as a frozenset so the zone stays hashable and so two zones
            # with the same filter written in a different order compare equal
            # — the audit diff compares sets whole, and would otherwise report
            # an edit that changed nothing.
            object.__setattr__(self, "classes", frozenset(self.classes))

    def is_active(self, moment: datetime) -> bool:
        return self.schedule is None or self.schedule.covers(moment)

    def watches(self, label: str | None) -> bool:
        """Whether a presence the detector calls ``label`` is this zone's business.

        ``label`` is the detector's word for the object, or ``None`` when the
        detector cannot say — a motion detector finds movement and names
        nothing. The two answers for ``None`` are the whole point:

        - An **empty** filter watches everything, so it fires for a blob the
          detector could not name. That is what every zone did before the
          filter existed and what a motion-only site relies on.
        - A **non-empty** filter never fires for ``None``. A blob that cannot
          be named cannot be said to be a person, and a person-only zone that
          fired on it anyway would be the sofa incident again, this time with
          the word "person" on it.
        """
        if not self.classes:
            return True
        return label is not None and label in self.classes

    def accepts(self, membership: ZoneMembership) -> bool:
        if membership is ZoneMembership.INSIDE:
            return True
        return self.accept_uncertain and membership is ZoneMembership.UNCERTAIN

    def membership_of(self, track: Track) -> ZoneMembership:
        """Where a track sits relative to this zone.

        A track with no position is :data:`OUTSIDE` — not because it is, but
        because an unplaced camera cannot support any claim about where its
        objects are, and a zone rule must not fire on one.
        """
        if track.position is None:
            return ZoneMembership.OUTSIDE
        return zone_membership(
            self.ring, track.position.point, track.position.radius_meters
        )


@dataclass(slots=True)
class Presence:
    """One track's continuous stay in one zone.

    Held open across brief losses, so a detector dropout does not end a presence
    and start a new one — which would turn one person loitering for four minutes
    into eight separate two-minute intrusions.
    """

    zone_id: str
    track_id: int
    started_millis: int
    last_present_millis: int
    #: Set once the presence has held long enough to be reported.
    confirmed: bool = False
    #: How many updates supported it, and how many were merely uncertain.
    observations: int = 0
    uncertain_observations: int = 0
    #: Set when the presence closes.
    ended_millis: int | None = None

    @property
    def duration_millis(self) -> int:
        end = self.ended_millis if self.ended_millis is not None else self.last_present_millis
        return end - self.started_millis

    @property
    def confidence(self) -> float:
        """How much of this presence was actually observed, not inferred.

        The fraction of supporting observations that were confidently inside
        rather than uncertain. A presence built entirely from uncertain positions
        is a weak claim, and the number that says so must travel with it.
        """
        if self.observations == 0:
            return 0.0
        confident = self.observations - self.uncertain_observations
        return round(confident / self.observations, 4)


@dataclass(frozen=True, slots=True)
class PresenceChange:
    """A presence starting or ending, reported once."""

    kind: str  # "ENTERED" or "LEFT"
    presence: Presence
    at_millis: int


class ZoneEvaluator:
    """Tracks which objects are in which zones, over time.

    Stateful and per-camera, because a presence is a fact about a continuous
    observation. Feed it every frame's tracks; it reports only the transitions.
    """

    __slots__ = ("_zones", "_open", "_last_seen", "_site_tz")

    def __init__(self, zones: Iterable[Zone], *, site_tz: tzinfo | None = None):
        """
        ``site_tz`` is the clock schedules are written in. Without it the
        moment is used as given, which for the pipeline means UTC — and a
        schedule of 18:00–06:00 typed by someone in Beirut would arm at 21:00
        their time. The pipeline and the node pass the machine's zone.
        """
        self._zones = {zone.id: zone for zone in zones}
        self._site_tz = site_tz
        #: (zone_id, track_id) -> Presence
        self._open: dict[tuple[str, int], Presence] = {}
        self._last_seen: dict[tuple[str, int], int] = {}

    @property
    def zones(self) -> tuple[Zone, ...]:
        return tuple(self._zones.values())

    def open_presences(self) -> tuple[Presence, ...]:
        return tuple(p for p in self._open.values() if p.confirmed)

    def update(
        self, tracks: Sequence[Track], at_millis: int, moment: datetime | None = None
    ) -> list[PresenceChange]:
        """Feed one frame's tracks. Returns presences that started or ended."""
        when = moment or datetime.now(timezone.utc)
        if self._site_tz is not None:
            when = when.astimezone(self._site_tz)
        changes: list[PresenceChange] = []
        live = {track.id for track in tracks}

        for zone in self._zones.values():
            if not zone.is_active(when):
                # An inactive zone closes anything it was holding, rather than
                # leaving a presence to reopen hours later with a duration
                # spanning the whole night.
                changes.extend(self._close_all(zone.id, at_millis))
                continue

            for track in tracks:
                key = (zone.id, track.id)
                membership = zone.membership_of(track)

                if zone.accepts(membership):
                    self._last_seen[key] = at_millis
                    change = self._observe(zone, track, membership, at_millis)
                    if change is not None:
                        changes.append(change)

            changes.extend(self._expire(zone, live, at_millis))

        return changes

    def _observe(
        self, zone: Zone, track: Track, membership: ZoneMembership, at_millis: int
    ) -> PresenceChange | None:
        key = (zone.id, track.id)
        presence = self._open.get(key)

        if presence is None:
            presence = Presence(
                zone_id=zone.id,
                track_id=track.id,
                started_millis=at_millis,
                last_present_millis=at_millis,
            )
            self._open[key] = presence

        presence.last_present_millis = at_millis
        presence.observations += 1
        if membership is ZoneMembership.UNCERTAIN:
            presence.uncertain_observations += 1

        if not presence.confirmed:
            held = at_millis - presence.started_millis
            if held >= zone.enter_after_millis:
                presence.confirmed = True
                return PresenceChange("ENTERED", presence, at_millis)

        return None

    def _expire(
        self, zone: Zone, live: set[int], at_millis: int
    ) -> list[PresenceChange]:
        """Close presences whose absence has lasted long enough to be real."""
        closed: list[PresenceChange] = []

        for key in [k for k in self._open if k[0] == zone.id]:
            presence = self._open[key]
            gone_for = at_millis - self._last_seen.get(key, presence.last_present_millis)

            # A track the tracker has dropped entirely cannot come back under the
            # same id, so there is nothing to wait for.
            track_gone = presence.track_id not in live
            if gone_for < zone.exit_after_millis and not track_gone:
                continue

            del self._open[key]
            self._last_seen.pop(key, None)
            presence.ended_millis = presence.last_present_millis
            if presence.confirmed:
                closed.append(PresenceChange("LEFT", presence, at_millis))

        return closed

    def _close_all(self, zone_id: str, at_millis: int) -> list[PresenceChange]:
        closed: list[PresenceChange] = []
        for key in [k for k in self._open if k[0] == zone_id]:
            presence = self._open.pop(key)
            self._last_seen.pop(key, None)
            presence.ended_millis = presence.last_present_millis
            if presence.confirmed:
                closed.append(PresenceChange("LEFT", presence, at_millis))
        return closed
