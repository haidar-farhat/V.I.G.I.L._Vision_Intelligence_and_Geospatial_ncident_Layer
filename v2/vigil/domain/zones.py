"""Zones and presence, with hysteresis on every state a person will read."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable, Sequence

from .geo import LatLon, distance_to_ring_edge, point_in_ring
from .tracking import Track


class ZoneKind(StrEnum):
    RESTRICTED = "RESTRICTED"
    PERIMETER = "PERIMETER"
    INTEREST = "INTEREST"


class Membership(StrEnum):
    INSIDE = "INSIDE"
    UNCERTAIN = "UNCERTAIN"
    OUTSIDE = "OUTSIDE"


@dataclass(frozen=True, slots=True)
class Schedule:
    """Hours during which a zone's after-hours rule applies, in the site's clock.

    ``closed_from`` and ``closed_until`` are hours 0–24; a window that wraps
    midnight (22 → 6) is allowed. ``None`` means the zone has no schedule.
    """

    closed_from: int
    closed_until: int

    def is_closed_at(self, hour: float) -> bool:
        if self.closed_from <= self.closed_until:
            return self.closed_from <= hour < self.closed_until
        return hour >= self.closed_from or hour < self.closed_until

    def describe(self) -> str:
        return f"closed {self.closed_from:02d}:00–{self.closed_until:02d}:00"


@dataclass(frozen=True, slots=True)
class Zone:
    id: str
    name: str
    kind: ZoneKind
    ring: tuple[LatLon, ...]
    #: Labels the zone acts on. Empty means every label; a motion detector's
    #: unlabelled blob satisfies an empty list and nothing else.
    watch: frozenset[str] = frozenset()
    enter_after_millis: int = 600
    exit_after_millis: int = 2000
    #: The membership a presence must reach. A restricted area demands INSIDE.
    min_membership: Membership = Membership.INSIDE
    schedule: Schedule | None = None

    def validate(self) -> None:
        if len(self.ring) < 3:
            raise ValueError(f"zone {self.name!r} needs at least three points")
        if self.enter_after_millis < 0 or self.exit_after_millis < 0:
            raise ValueError("holds cannot be negative")

    def watches(self, label: str | None) -> bool:
        if not self.watch:
            return True
        return label is not None and label.lower() in self.watch

    def membership(self, point: LatLon, uncertainty_meters: float) -> Membership:
        inside = point_in_ring(self.ring, point)
        edge = distance_to_ring_edge(self.ring, point)
        if edge <= uncertainty_meters:
            return Membership.UNCERTAIN
        return Membership.INSIDE if inside else Membership.OUTSIDE

    def distance_from(self, position) -> "Distance":
        """From a position to this zone's nearest edge, negative when inside.

        Signed on purpose: "two metres inside" and "two metres outside" are
        different situations and a bare magnitude cannot tell them apart.
        """
        from .geo import Distance

        edge = distance_to_ring_edge(self.ring, position.point)
        inside = point_in_ring(self.ring, position.point)
        return Distance(-edge if inside else edge, position.radius_meters)

    def admits(self, membership: Membership) -> bool:
        order = {Membership.INSIDE: 2, Membership.UNCERTAIN: 1, Membership.OUTSIDE: 0}
        return order[membership] >= order[self.min_membership]


@dataclass(slots=True)
class Presence:
    """One track's standing in one zone."""

    zone_id: str
    track_id: int
    entered_millis: int
    last_inside_millis: int
    observations: int = 1
    confidence: float = 0.0
    confirmed: bool = False
    #: Where the track was when it was last inside; the evidence position.
    last_point: LatLon | None = None
    last_radius: float = 0.0


@dataclass(frozen=True, slots=True)
class PresenceChange:
    kind: str  # ENTERED | LEFT
    presence: Presence
    at_millis: int


class PresenceTracker:
    """Turns per-frame memberships into entered/left with holds.

    A track is *present* after it has been admitted for `enter_after_millis`;
    it has *left* after it has failed to be admitted for `exit_after_millis`.
    One flickering frame is not a departure.
    """

    def __init__(self, zones: Iterable[Zone]):
        self._zones = {z.id: z for z in zones}
        self._presences: dict[tuple[str, int], Presence] = {}

    @property
    def zones(self) -> dict[str, Zone]:
        return dict(self._zones)

    def presences(self) -> list[Presence]:
        return [p for p in self._presences.values() if p.confirmed]

    def presence_of(self, zone_id: str, track_id: int) -> Presence | None:
        return self._presences.get((zone_id, track_id))

    def update(self, tracks: Sequence[Track], at_millis: int, *, label_for=None) -> list[PresenceChange]:
        changes: list[PresenceChange] = []
        seen: set[tuple[str, int]] = set()
        for track in tracks:
            if track.position is None or not track.position.is_projected:
                continue
            label = label_for(track.class_id) if label_for else None
            for zone in self._zones.values():
                if not zone.watches(label):
                    continue
                membership = zone.membership(track.position.point, track.position.radius_meters)
                key = (zone.id, track.id)
                if not zone.admits(membership):
                    continue
                seen.add(key)
                presence = self._presences.get(key)
                if presence is None:
                    presence = Presence(zone.id, track.id, at_millis, at_millis, 1, track.confidence,
                                        last_point=track.position.point, last_radius=track.position.radius_meters)
                    self._presences[key] = presence
                    continue
                presence.observations += 1
                presence.last_inside_millis = at_millis
                presence.last_point = track.position.point
                presence.last_radius = track.position.radius_meters
                presence.confidence = presence.confidence * 0.7 + track.confidence * 0.3
                if not presence.confirmed and at_millis - presence.entered_millis >= zone.enter_after_millis:
                    presence.confirmed = True
                    changes.append(PresenceChange("ENTERED", presence, at_millis))

        for key, presence in list(self._presences.items()):
            if key in seen:
                continue
            zone = self._zones[key[0]]
            # Absent from the frame counts as outside and nothing more: the
            # exit hold decides, and `forget_track` is the definitive end.
            if at_millis - presence.last_inside_millis >= zone.exit_after_millis:
                del self._presences[key]
                if presence.confirmed:
                    changes.append(PresenceChange("LEFT", presence, at_millis))
        return changes

    def forget_track(self, track_id: int, at_millis: int) -> list[PresenceChange]:
        changes = []
        for key in [k for k in self._presences if k[1] == track_id]:
            presence = self._presences.pop(key)
            if presence.confirmed:
                changes.append(PresenceChange("LEFT", presence, at_millis))
        return changes
