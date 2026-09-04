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

**A fragmented track is still one object.** The laptop camera gave one person
seven track ids in fifteen seconds (measured, `tools/measure_fragmentation.py`).
Counted as seven objects that one person becomes "7 objects in Room", a "group"
risk factor worth 20 points, and a HIGH incident about nobody. So the same
union-find that joins tracks across cameras also joins consecutive fragments on
one camera — using the time and place gates :mod:`sentinel.reid` exposes, with
appearance when the evidence carries it and a stricter gate when it does not —
and each join is an :class:`Association` with its reasons, so the count that
reaches the operator can be argued with link by link. What the evidence can
prove is narrower than what the tracker knew: a fragment's end is known only
as of its last event, so two tracks the *evidence* shows overlapping stay two,
and a track that lived on unseen can still be joined to a newcomer. Each such
link says so.

**Risk is explained, never asserted.** The score comes with the contributions
that produced it. A number an operator cannot interrogate is a number they will
eventually learn to ignore.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Iterable, Sequence

from .core import BoundingBox, LatLon, PositionEstimate, haversine_distance
from .events import Event, Severity, severity_rank
from .zones import ZoneKind

if TYPE_CHECKING:  # pragma: no cover
    # ``reid`` imports this module for its union-find; the runtime import goes
    # the other way inside :func:`link_same_camera_fragments` so neither module
    # sees the other half-initialised.
    from .reid import TrackAppearance

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

    @property
    def same_camera(self) -> bool:
        """Whether this joins two fragments of one track rather than two cameras.

        Derived from the keys rather than stored, so a link read back from the
        store — which persists only the fields above — is classified the same
        way as one just made. For a same-camera link ``a`` is the earlier
        fragment and ``b`` the one that continues it.
        """
        return self.a[0] == self.b[0]

    def describe(self) -> str:
        """``#7 = #3 on webcam (gap 0.4 s, 0.6 m apart)`` — what an operator reads."""
        gap = f"{self.time_gap_millis / 1000:.1f} s"
        if self.same_camera:
            return (
                f"#{self.b[1]} = #{self.a[1]} on {self.a[0]} "
                f"(gap {gap}, {self.separation_meters:.1f} m apart)"
            )
        return (
            f"{self.a[0]}#{self.a[1]} = {self.b[0]}#{self.b[1]} "
            f"({self.separation_meters:.1f} m apart, {gap} apart)"
        )


def _position_of(event: Event) -> LatLon | None:
    if event.evidence.latitude is None or event.evidence.longitude is None:
        return None
    return LatLon(event.evidence.latitude, event.evidence.longitude)


def _time_and_place_score(proximity: float, recency: float) -> float:
    """Order candidates that only time and place can speak for.

    Read by the cross-camera gate and by the blind fragment gate, so a link of
    either kind sorts the same way and the split lives in one place. Place
    outweighs time because the allowance is a bound on the *pair* — built from
    the two positions' own uncertainties — while the window or hold is a bound
    on the site: a gap short by that standard is short for everyone who was
    there, and says less about who this was. The 65/35 is a judgement, stated
    here so it can be argued with; :mod:`sentinel.reid` gives appearance half
    its score and splits the rest evenly, because there a look has already
    done the separating that place has to do alone here.
    """
    return round(0.65 * proximity + 0.35 * recency, 4)


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
            score = _time_and_place_score(proximity, recency)

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


# ------------------------------------------------------ same-camera fragments

#: Without appearance, the longest a fragment may be missing and still be
#: rejoined: the tracker's own hold. Inside the hold a detection that came back
#: and was *not* matched is a box that jumped — the person turned, or half of
#: them went behind a chair — and place can vouch for that. Beyond it the object
#: was genuinely gone, and "the same one came back" is a claim only a look could
#: support; :mod:`sentinel.reid` allows 5 s when it has one, this allows less
#: than half of that without.
BLIND_FRAGMENT_MAX_GAP_MILLIS = 2000

#: Without appearance, the fraction of reid's spatial allowance a fragment must
#: fall within. Halved rather than merely trimmed, because the allowance reid
#: builds is sized for a pair whose colours already agree; on time and place
#: alone the same gate would join two people who walked through one doorway a
#: second apart. Halving does not make that impossible — nothing without a look
#: can — it makes the second person have to appear within a metre or two of
#: where the first was last placed, and the link's reasons say that is all it
#: rests on.
BLIND_FRAGMENT_ALLOWANCE_FRACTION = 0.5

#: Evidence carries a place, not a box. A fragment built from events has no
#: image position for reid's frame fallback, and this stands in so the gate has
#: a field to read; the fallback's result is refused below, never measured.
_NO_BOX = BoundingBox(0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class _Fragment:
    """One track as its events describe it, in the shape reid's gates read."""

    track: "TrackAppearance"
    #: ``None`` when the detector could not say what it saw.
    class_label: str | None


def _ground_position(event: Event) -> PositionEstimate | None:
    """The event's position, only when it measures the *object*.

    A ``CAMERA_FALLBACK`` position is the camera's own location, which every
    track on that camera shares. A place gate fed two of them finds zero metres
    between anything and everything, and time alone would then merge whoever
    walked past next. So it is dropped here, and a fragment without a ground
    position is never linked — an unplaced camera keeps its inflated count
    rather than being given a flattering one.
    """
    evidence = event.evidence
    if (
        evidence.latitude is None
        or evidence.longitude is None
        or evidence.position_source != "GROUND_PROJECTION"
    ):
        return None
    return PositionEstimate(
        point=LatLon(evidence.latitude, evidence.longitude),
        radius_meters=evidence.position_uncertainty_meters or 0.0,
        source=evidence.position_source,
    )


def _carried_appearance(event: Event) -> tuple[object | None, int]:
    """A descriptor on the evidence, if the evidence carries one.

    :class:`~sentinel.events.Evidence` has no such field today. This reads
    ``appearance`` by name — a :class:`~sentinel.reid.Appearance`-shaped
    object with a ``histogram``, or the array itself — so the seen path is
    live the day the evidence grows one, and returns ``(None, 0)`` until then.
    """
    carried = getattr(event.evidence, "appearance", None)
    if carried is None:
        return None, 0
    histogram = getattr(carried, "histogram", carried)
    return histogram, int(getattr(carried, "samples", 1))


def _fragments_of(events: Sequence[Event]) -> dict[tuple[str, int], _Fragment]:
    """Fold every event about a track into one fragment.

    A track's start is known to every event about it. Its end is known only as
    of the *last* event about it — the track may have lived on without raising
    anything more — so the end folded here is a lower bound on the true end and
    the gap measured downstream an upper bound on the true gap. That cuts both
    ways. Against the hold it errs towards refusing, which is the side to err
    on. Against the overlap check it errs towards *joining*: a track that was
    still there when the next one began shows a positive gap, not a negative
    one, and nothing here can tell the two apart. That is the limit
    :func:`link_same_camera_fragments` states and every blind link repeats.
    """
    import numpy as np

    from .reid import TrackAppearance

    by_track: dict[tuple[str, int], list[Event]] = {}
    for event in events:
        key = (event.evidence.camera_id, event.evidence.track_id)
        by_track.setdefault(key, []).append(event)

    fragments: dict[tuple[str, int], _Fragment] = {}
    for key, concerning in by_track.items():
        concerning.sort(key=lambda e: (e.occurred_at_millis, e.id))
        first, last = concerning[0], concerning[-1]
        first_seen = min(e.evidence.first_seen_millis for e in concerning)
        last_seen = max(e.evidence.last_seen_millis for e in concerning)

        # The newest descriptor is the running one: the most frames behind it.
        histogram, samples = None, 0
        for event in reversed(concerning):
            histogram, samples = _carried_appearance(event)
            if histogram is not None:
                break

        track = TrackAppearance(
            camera_id=key[0],
            track_id=key[1],
            first_seen_millis=first_seen,
            last_seen_millis=last_seen,
            last_confirmed_millis=last_seen,
            first_box=_NO_BOX,
            last_box=_NO_BOX,
            first_position=_ground_position(first),
            last_position=_ground_position(last),
            histogram=None if histogram is None else np.asarray(histogram, dtype=np.float32),
            samples=samples if histogram is not None else 0,
        )
        label = last.evidence.class_label if last.evidence.detector_classifies else None
        fragments[key] = _Fragment(track=track, class_label=label)
    return fragments


def _class_reason(earlier: _Fragment, later: _Fragment) -> str:
    if earlier.class_label is not None and later.class_label is not None:
        return f"both classified {earlier.class_label}"
    if earlier.class_label is None and later.class_label is None:
        return "the detector does not classify, so class could not separate them"
    label = earlier.class_label if earlier.class_label is not None else later.class_label
    return f"one fragment classified {label}, the other unclassified: class could not separate them"


def _estimated_end_reason(a: "TrackAppearance", b: "TrackAppearance", defence: str) -> str:
    """Say that the earlier end is an estimate, and what stood in for it.

    Without this line a link reads as though the two tracks were known to be
    consecutive. They were not: ``a``'s end is as of its last event, and an
    operator weighing "1 object" against a possible second person needs to see
    that the only evidence of order is a last event and the place gate.
    """
    return (
        f"#{a.track_id}'s end is as of its last event, t+{a.last_confirmed_millis / 1000:.1f} s, "
        f"not the tracker's; had it lived on unseen it overlapped #{b.track_id} and "
        f"only {defence} stood between them"
    )


def _consider_fragment(earlier: _Fragment, later: _Fragment) -> Association | None:
    """Decide whether ``later`` continues ``earlier`` on one camera.

    The time condition keeps apart two tracks the evidence shows overlapping:
    one first seen before the other's last event, or on that very frame —
    they shared it. It cannot see past the last event, so a track that lived
    on unseen is not protected by it (see :func:`link_same_camera_fragments`).
    Place and, when carried, appearance are then reid's gates unchanged; blind,
    the gap is the tracker's hold and the allowance is halved (see the
    constants), because time and place alone must not merge two people passing
    one spot, and the link's reasons state what it rests on — including that
    the earlier end is an estimate.
    """
    from . import reid

    a, b = earlier.track, later.track
    gap = b.first_seen_millis - a.last_confirmed_millis
    if gap <= 0:
        return None
    if (
        earlier.class_label is not None
        and later.class_label is not None
        and earlier.class_label != later.class_label
    ):
        # A person's fragment cannot continue a bottle's.
        return None

    if a.has_appearance and b.has_appearance:
        link = reid._consider(a, b, reid.DEFAULT_MAX_GAP_MILLIS, reid.DEFAULT_MIN_SIMILARITY)
        if link is None or link.separation_unit != "m":
            # "frame" here means an end with no ground position: the stand-in
            # box measured nothing, and nothing is not evidence of place.
            return None
        return Association(
            a=a.key,
            b=b.key,
            score=link.score,
            separation_meters=round(link.separation, 2),
            allowance_meters=round(link.allowance, 2),
            time_gap_millis=gap,
            reasons=link.reasons + (
                _estimated_end_reason(a, b, "place and appearance"),
                _class_reason(earlier, later),
            ),
        )

    if gap > BLIND_FRAGMENT_MAX_GAP_MILLIS:
        return None
    separation, allowance, unit = reid._spatial_gate(a, b, gap / 1000.0)
    if unit != "m":
        return None
    allowance *= BLIND_FRAGMENT_ALLOWANCE_FRACTION
    if separation > allowance:
        return None

    if a.has_appearance != b.has_appearance:
        # One side had a look and the other did not. Saying "no appearance was
        # carried" here would name a failure that did not happen; the failure
        # is that a comparison needs two.
        lacking = b if a.has_appearance else a
        why_blind = (
            f"because #{lacking.track_id} carried no appearance, so nothing "
            "could say whether they look alike"
        )
    else:
        why_blind = "because no appearance was carried to say they look alike"

    proximity = 1.0 - separation / allowance if allowance > 0 else 1.0
    recency = 1.0 - gap / BLIND_FRAGMENT_MAX_GAP_MILLIS
    score = _time_and_place_score(proximity, recency)
    return Association(
        a=a.key,
        b=b.key,
        score=score,
        separation_meters=round(separation, 2),
        allowance_meters=round(allowance, 2),
        time_gap_millis=gap,
        reasons=(
            f"#{b.track_id} began {gap / 1000:.1f} s after #{a.track_id} was last "
            f"seen, within the tracker's {BLIND_FRAGMENT_MAX_GAP_MILLIS / 1000:.0f} s hold",
            f"{separation:.2f} m apart, within {allowance:.2f} m — half of what the "
            f"gap and the two position uncertainties would allow, {why_blind}",
            _estimated_end_reason(a, b, "place"),
            _class_reason(earlier, later),
        ),
    )


def link_same_camera_fragments(events: Sequence[Event]) -> list[Association]:
    """Decide which tracks on one camera are fragments of one object.

    The counterpart of :func:`associate`, which refuses same-camera pairs
    because identity within a camera is the tracker's job. It is — and the
    tracker gave one person seven ids in fifteen seconds. Second-guessing it on
    the evidence it did not have (a longer memory, and appearance when carried)
    is what this does; second-guessing it on the evidence it *did* have is what
    the time condition forbids: two tracks whose *evidence* shows them
    overlapping never merge.

    That is weaker than "two tracks the tracker held at once", and the gap
    between the two is this function's honest limit. A fragment's end is known
    only as of its last event — an entry event is raised on first sight, so a
    track with one event has an end equal to its own start — and a track that
    lived on silently after it has an end that is too early. A newcomer within
    the hold can therefore be joined to a track that was in fact still there,
    and the place gate is the only thing between them; every such link says so
    in its reasons. Until the pipeline gives the correlator the tracker's live
    track ends, the count can flicker on a live window: "1 object" while the
    first track's evidence ends early, "2 objects" once a later event about it
    arrives.

    Each fragment continues at most one other and is continued by at most one.
    When two tracks appear after one vanishes, only the better-supported one
    can be its continuation, and the other is somebody else.
    """
    fragments = _fragments_of(events)
    by_camera: dict[str, list[_Fragment]] = {}
    for key, fragment in fragments.items():
        by_camera.setdefault(key[0], []).append(fragment)

    accepted: list[Association] = []
    for on_camera in by_camera.values():
        candidates: list[Association] = []
        for earlier in on_camera:
            for later in on_camera:
                if later is earlier:
                    continue
                link = _consider_fragment(earlier, later)
                if link is not None:
                    candidates.append(link)

        candidates.sort(key=lambda link: (-link.score, link.time_gap_millis, link.b[1]))
        continued: set[tuple[str, int]] = set()
        continues: set[tuple[str, int]] = set()
        for link in candidates:
            if link.a in continued or link.b in continues:
                continue
            continued.add(link.a)
            continues.add(link.b)
            accepted.append(link)

    accepted.sort(key=lambda link: (link.a[0], fragments[link.b].track.first_seen_millis, link.b[1]))
    return accepted


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
        if self.associations:
            # The object count above is only as good as these; an operator
            # who cannot see them cannot tell "1 object" from a lucky merge.
            lines.append("  links")
            for link in self.associations:
                lines.append(f"    {link.describe()}")
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
    #: Cross-camera joins.
    associations: int = 0
    #: Same-camera fragment joins, counted apart because they measure a
    #: different thing: how badly the tracker fragmented, not how many cameras
    #: agreed.
    fragment_links: int = 0

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

        # Same-camera fragments join the same identity as cross-camera pairs,
        # so the distinct-object count — and the "group" factor built on it —
        # is of objects, not of the ids a flickering detector handed out.
        fragments = link_same_camera_fragments(ordered)
        self.stats.fragment_links += len(fragments)
        links = links + fragments

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
