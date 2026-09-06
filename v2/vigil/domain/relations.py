"""What tracked things are doing with each other, and how sure that can be.

A list of boxes is not a situation. *Three people in the car*, *somebody
carrying something*, *two of them together*, *walking towards the gate* — that
is what an operator says, and each of those is a relation between tracks that
this module measures.

**Every relation here is inferred, never observed.** One camera cannot tell
*inside* from *in front of*: a person walking past a car occludes it exactly
as a person sitting in it does. So each relation carries the numbers it was
drawn from — the overlap, the distance, the frames it held — and its wording
hedges. "Probably in the vehicle", never "in the vehicle". Two cameras or a
depth sensor would let it say more; until then it says less.

Nothing here reads a clock or touches the outside world: it is given tracks
and a moment and returns what it can support.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable, Iterable, Sequence

from .detection import BoundingBox
from .geo import Distance, separation
from .tracking import Track

#: Labels treated as something a person can be inside. Named here so a site
#: with a different model's vocabulary can see what the rule actually keys on.
VEHICLES = frozenset({"car", "truck", "bus", "motorcycle", "train", "boat", "airplane", "van", "lorry"})
PEOPLE = frozenset({"person"})


class RelationKind(StrEnum):
    INSIDE = "INSIDE"
    CARRIED = "CARRIED"
    NEAR = "NEAR"
    APPROACHING = "APPROACHING"


@dataclass(frozen=True, slots=True)
class RelationRules:
    """Every threshold, named and in one place so a site can argue with them."""

    #: Fraction of a person's box that must lie within a vehicle's.
    inside_overlap: float = 0.6
    #: Fraction of a *carried* object's own box that must lie within a person's.
    carried_overlap: float = 0.5
    #: The most of a person's box area a carried object may take. Above this
    #: it is not something held; it is something stood in front of.
    carried_max_area: float = 0.25
    #: How far above the person's feet the object must sit, as a fraction of
    #: their height. A bag on the ground beside somebody is not carried.
    carried_lift: float = 0.12
    #: Ground distance for "together", before either position's error.
    near_meters: float = 3.0
    #: How long a candidate must hold before it is reported. One frame of
    #: overlap is a coincidence; a second of it is a situation.
    hold_millis: int = 700
    #: The window over which a closing gap is measured.
    approach_window_millis: int = 2500
    #: The gap must close by more than this *and* by more than the combined
    #: position error, so noise cannot look like approach.
    approach_min_meters: float = 1.0


@dataclass(frozen=True, slots=True)
class Relation:
    """One inferred relation, with what it was drawn from."""

    kind: RelationKind
    subject: int
    object: int | None = None
    zone_id: str | None = None
    confidence: float = 0.0
    conditions: tuple[str, ...] = ()
    held_millis: int = 0
    observations: int = 0

    @property
    def key(self) -> tuple:
        return (self.kind, self.subject, self.object, self.zone_id)

    def describe(self, name_of: Callable[[int], str] | None = None) -> str:
        """Hedged on purpose. This is what one camera can support, and no more."""
        who = name_of(self.subject) if name_of else f"track {self.subject}"
        what = (name_of(self.object) if name_of else f"track {self.object}") if self.object is not None else None
        if self.kind is RelationKind.INSIDE:
            return f"{who} is probably in {what}"
        if self.kind is RelationKind.CARRIED:
            return f"{who} appears to be carrying {what}"
        if self.kind is RelationKind.NEAR:
            return f"{who} is with {what}"
        target = what if what is not None else f"zone {self.zone_id}"
        return f"{who} is moving towards {target}"


def overlap_fraction(inner: BoundingBox, outer: BoundingBox) -> float:
    """How much of ``inner`` lies within ``outer``, by area. 0 when it does not."""
    if inner.area <= 0:
        return 0.0
    ix = max(0.0, min(inner.right, outer.right) - max(inner.x, outer.x))
    iy = max(0.0, min(inner.bottom, outer.bottom) - max(inner.y, outer.y))
    return (ix * iy) / inner.area


@dataclass
class _Candidate:
    since: int
    last: int
    observations: int = 0
    strength: float = 0.0


class RelationTracker:
    """Turns per-frame geometry into relations that have held long enough.

    Stateful in exactly the way `PresenceTracker` is, and for the same
    reason: one frame of anything is noise.
    """

    def __init__(self, rules: RelationRules | None = None):
        self.rules = rules or RelationRules()
        self._candidates: dict[tuple, _Candidate] = {}
        #: Gap history per pair, for "approaching". Bounded by the window.
        self._gaps: dict[tuple, deque] = {}

    def update(self, tracks: Sequence[Track], at_millis: int, *,
               label_of: Callable[[int], str | None] | None = None,
               zones: Iterable = ()) -> list[Relation]:
        """Every relation that holds right now, with what it rests on."""
        label = label_of or (lambda class_id: None)
        found: list[Relation] = []
        seen: set[tuple] = set()

        for subject in tracks:
            for other in tracks:
                if subject.id == other.id:
                    continue
                for relation in self._between(subject, other, at_millis, label):
                    seen.add(relation.key)
                    found.append(relation)
            for zone in zones:
                relation = self._towards_zone(subject, zone, at_millis)
                if relation is not None:
                    seen.add(relation.key)
                    found.append(relation)

        # A candidate nobody saw this frame is forgotten at once: unlike a
        # zone presence, a relation has no exit hold, because "they are no
        # longer in the car" is not an event anybody acts on.
        for key in [k for k in self._candidates if k not in seen and self._candidates[k].last < at_millis]:
            del self._candidates[key]
        return found

    # ------------------------------------------------------------ one pair

    def _between(self, subject: Track, other: Track, at_millis: int, label) -> list[Relation]:
        out = []
        subject_label = (label(subject.class_id) or "").lower()
        other_label = (label(other.class_id) or "").lower()

        if subject_label in PEOPLE and other_label in VEHICLES:
            fraction = overlap_fraction(subject.bbox, other.bbox)
            if fraction >= self.rules.inside_overlap:
                out.append(self._hold(
                    RelationKind.INSIDE, subject.id, other.id, None, at_millis, fraction,
                    (f"{fraction:.0%} of the person's box lay within the {other_label}'s",
                     "one camera cannot tell being inside from passing in front"),
                ))

        if subject_label in PEOPLE and other_label not in PEOPLE and other_label not in VEHICLES and other_label:
            fraction = overlap_fraction(other.bbox, subject.bbox)
            area_ratio = other.bbox.area / subject.bbox.area if subject.bbox.area > 0 else 1.0
            lift = subject.bbox.bottom - other.bbox.bottom
            if (fraction >= self.rules.carried_overlap and area_ratio <= self.rules.carried_max_area
                    and lift >= self.rules.carried_lift * subject.bbox.height):
                out.append(self._hold(
                    RelationKind.CARRIED, subject.id, other.id, None, at_millis, fraction,
                    (f"{fraction:.0%} of the {other_label}'s box lay within the person's",
                     f"it is {area_ratio:.0%} of their size and {lift / max(subject.bbox.height, 1e-6):.0%} of their "
                     "height above their feet, so it is not on the ground"),
                ))

        if subject.id < other.id and subject.position and other.position:
            if subject.position.is_projected and other.position.is_projected:
                gap = separation(subject.position, other.position)
                if gap.within(self.rules.near_meters):
                    out.append(self._hold(
                        RelationKind.NEAR, subject.id, other.id, None, at_millis,
                        1.0 - min(1.0, gap.meters / max(self.rules.near_meters, 1e-6)),
                        (f"{gap.describe()} apart on the ground, within {self.rules.near_meters:.0f} m "
                         "even allowing for both position errors",),
                    ))
                approaching = self._approach(("t", subject.id, other.id), gap, at_millis)
                if approaching is not None:
                    closed, seconds = approaching
                    out.append(self._hold(
                        RelationKind.APPROACHING, subject.id, other.id, None, at_millis,
                        min(1.0, closed / max(self.rules.approach_min_meters, 1e-6) / 3),
                        (f"the gap closed {closed:.1f} m in {seconds:.1f} s, which is more than either "
                         f"position's error ({gap.describe()} now)",),
                    ))
        return [r for r in out if r is not None]

    def _towards_zone(self, subject: Track, zone, at_millis: int) -> Relation | None:
        if subject.position is None or not subject.position.is_projected:
            return None
        gap = zone.distance_from(subject.position)
        if gap.meters <= 0:
            self._gaps.pop(("z", subject.id, zone.id), None)
            return None  # already inside; presence, not approach, is the fact
        approaching = self._approach(("z", subject.id, zone.id), gap, at_millis)
        if approaching is None:
            return None
        closed, seconds = approaching
        return self._hold(
            RelationKind.APPROACHING, subject.id, None, zone.id, at_millis,
            min(1.0, closed / max(self.rules.approach_min_meters, 1e-6) / 3),
            (f"the gap to {zone.name} closed {closed:.1f} m in {seconds:.1f} s",
             f"it is {gap.describe()} from the edge now"),
        )

    # -------------------------------------------------------------- state

    def _approach(self, key: tuple, gap: Distance, at_millis: int) -> tuple[float, float] | None:
        """How much a gap has closed over the window, or ``None`` if it has not.

        Refuses to call noise approach: the closing must beat both a stated
        minimum and the combined position error.
        """
        history = self._gaps.setdefault(key, deque())
        history.append((at_millis, gap.meters))
        while history and at_millis - history[0][0] > self.rules.approach_window_millis:
            history.popleft()
        if len(history) < 2:
            return None
        first_at, first_gap = history[0]
        seconds = (at_millis - first_at) / 1000
        closed = first_gap - gap.meters
        if seconds <= 0 or closed <= max(self.rules.approach_min_meters, gap.error_meters):
            return None
        return closed, seconds

    def _hold(self, kind: RelationKind, subject: int, object_id: int | None, zone_id: str | None,
              at_millis: int, strength: float, conditions: tuple[str, ...]) -> Relation | None:
        """Report a relation only once it has held for `hold_millis`."""
        key = (kind, subject, object_id, zone_id)
        candidate = self._candidates.get(key)
        if candidate is None:
            self._candidates[key] = _Candidate(since=at_millis, last=at_millis, observations=1, strength=strength)
            return None
        candidate.last = at_millis
        candidate.observations += 1
        candidate.strength = candidate.strength * 0.7 + strength * 0.3
        held = at_millis - candidate.since
        if held < self.rules.hold_millis:
            return None
        # Confidence is the strength of the geometry, tempered by how long it
        # has held — stated here rather than tuned into a magic number.
        steadiness = min(1.0, held / max(1, self.rules.hold_millis * 3))
        confidence = round(min(0.95, candidate.strength * (0.6 + 0.4 * steadiness)), 3)
        return Relation(kind, subject, object_id, zone_id, confidence,
                        conditions + (f"held for {held} ms over {candidate.observations} frames",),
                        held, candidate.observations)

    def forget_track(self, track_id: int) -> None:
        for key in [k for k in self._candidates if track_id in (k[1], k[2])]:
            del self._candidates[key]
        for key in [k for k in self._gaps if track_id == k[1] or track_id == k[2]]:
            del self._gaps[key]
