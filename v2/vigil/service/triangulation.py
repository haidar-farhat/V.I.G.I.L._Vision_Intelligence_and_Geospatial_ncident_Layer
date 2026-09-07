"""Pairing what two cameras see, and solving the ground they both stand on.

# What this does with a frame

Each camera's worker hands over the tracks it is holding and when it saw them.
This keeps the most recent sighting per camera and, for every pair of cameras
that overlap, offers every plausible pair of tracks to
`domain.triangulation`. A pair survives when the rays actually converge —
close enough together to be one object, far enough apart in angle to say where
it is — and is dropped otherwise.

# Why the geometry does the associating

There is no appearance model in this file and it does not need one. The
question "are these two tracks the same object" already has a sharp geometric
answer: two rays aimed at one object pass within centimetres of each other,
and two rays aimed at two people three metres apart do not come within three
metres. `Refusal.TOO_FAR_APART` **is** the association test, and it is a
stronger one than a colour histogram, because it is a statement about where
things are rather than about what they look like.

Appearance narrows the candidate list before geometry decides, which is worth
doing when a camera pair sees a crowd; that is `domain.appearance`'s job and
it is passed in rather than reached for, so this module keeps working when it
is not.

The honest limit: when two people stand one behind the other along the line
joining the cameras, the rays converge for the wrong pairing as readily as for
the right one. Geometry cannot break that tie and neither can appearance if
they are dressed alike. Such a pair is reported with its `gap_m`, and a small
gap on an ambiguous pair is still ambiguous — the number is there so nothing
downstream has to pretend otherwise.

# The ground

Every triangulated point is a sample of where *something* was. The ones near
the fitted plane are the ground; the ones consistently above it are people
standing on something, or on nothing this system models. `fit_ground` is
RANSAC precisely so the second group does not tilt the answer, and the height
above the fitted plane is then the signal that says which group a point is in.

Solved tilt is written back to each camera, so even a camera with no overlap
stops assuming a level yard — that is the part that improves positions the
product was already producing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Iterable

import numpy as np

from ..domain.appearance import Appearance, CrossCameraSeparation
from ..domain.geo import CameraPose, LatLon, LocalFrame, Vec2
from ..domain.triangulation import (
    MIN_PARALLAX_DEG, GroundPlane, Refusal, Sighting, TriangulatedPoint, fit_ground, triangulate,
)
from ..logs import get as _get_logger

_log = _get_logger(__name__)

#: How far apart in time two sightings may be and still be one moment.
#:
#: A person walks 1.4 m/s, so 200 ms is 28 cm of real movement — under the
#: `MAX_GAP_M` the geometry allows, which is what makes the pairing sound.
#: Cameras are not synchronised and never will be on this hardware, so this is
#: a tolerance rather than an assumption of simultaneity.
MAX_SKEW_MILLIS = 200

#: Points held for the ground fit. Enough to cover a yard at the sampling
#: rate; older ones fall out because ground that was resurfaced is not ground
#: that is still there.
GROUND_SAMPLES = 4000

#: Below this many inlying observations, a solved plane is not written back.
#: Three points define a plane and say nothing about a site; this is the point
#: where the tilt is better than the assumption it replaces.
MIN_GROUND_SAMPLES = 200

#: Height above the fitted plane past which a point is not ground.
#: A person's ground contact is at their feet, so a contact that is a metre up
#: is standing on something.
NOT_GROUND_M = 0.5

#: The worst a triangulated position may be and still be worth having, metres.
#:
#: Three, because that is roughly what a *ground projection* is worth at the
#: far end of a 60 m camera with an assumed pose, and a triangulated point
#: that cannot beat it has nothing to offer but the appearance of rigour.
#: See `Geometry._parallax_for`, which turns this into an angle per pair.
WORTH_HAVING_M = 3.0

#: Pixel count attached to a descriptor that arrives already normalised.
#: `Appearance.usable` gates on it, and the worker has already applied that
#: gate — an unusable look never reaches here — so this only has to be above
#: the floor rather than to mean anything.
MIN_APPEARANCE_PIXELS = 10_000


@dataclass(frozen=True, slots=True)
class Pairing:
    """One object, seen by two cameras, and the two tracks it came from."""

    camera_a: str
    track_a: int
    camera_b: str
    track_b: int
    point: TriangulatedPoint
    at_millis: int

    def describe(self) -> str:
        return (f"{self.camera_a}#{self.track_a} and {self.camera_b}#{self.track_b}: "
                f"{self.point.describe()}")


@dataclass
class _Seen:
    """The last thing one camera reported, and when."""

    at_millis: int
    pose: CameraPose
    #: `(track_id, class_id, contact, appearance)`. The appearance is already
    #: normalised for this camera, so two cameras' descriptors are comparable.
    tracks: tuple[tuple[int, int, Vec2, object], ...] = ()


@dataclass
class Geometry:
    """Cross-camera positions and the site's own ground, for one site.

    Not a background thread. It is fed from whichever thread already has the
    frame, holds a few hundred bytes per camera, and does its work when it is
    asked — a second thread here would need a lock around the pose table that
    the console also edits, for no gain.
    """

    origin: LatLon
    #: Angular tolerance handed to the triangulator. Read from each pose's
    #: measured uncertainty; this is only the floor for an unplaced one.
    min_parallax_deg: float | None = None
    _frame: LocalFrame = field(init=False)
    _seen: dict[str, _Seen] = field(default_factory=dict, init=False)
    _ground: list[tuple[float, float, float]] = field(default_factory=list, init=False)
    _plane: GroundPlane | None = field(default=None, init=False)
    _pairings: tuple[Pairing, ...] = field(default=(), init=False)
    #: Whether appearance can separate objects between these cameras at all,
    #: measured from pairs geometry has already decided. See
    #: `domain.appearance.CrossCameraSeparation`.
    separation: CrossCameraSeparation = field(default_factory=CrossCameraSeparation, init=False)

    def __post_init__(self) -> None:
        self._frame = LocalFrame(self.origin)

    # ------------------------------------------------------------- feeding

    def observe(self, camera_id: str, pose: CameraPose | None, at_millis: int,
                tracks: Iterable, looks: dict | None = None) -> None:
        """Record what one camera is holding. Pairing happens in `resolve`.

        Recording and pairing are separate calls, and the separation is not
        tidiness. Pairing on arrival triangulates every object **twice** per
        round of frames — once when the first camera reports, against the
        other camera's previous frame, and again when the second reports
        against the first's current one. Both pairings converge, both look
        right, and the stale one puts a person where they were a tenth of a
        second ago. Measured on a two-camera walk: 119 pairings from 60
        moments, with half of them fed into the ground fit as if they were
        independent observations.
        """
        if pose is None:
            return
        # The descriptors arrive already normalised, from the worker that owns
        # this camera's colour statistics. Not recomputed here: two running
        # averages of the same pictures are two answers waiting to differ, and
        # the correlator reads the worker's copy off the evidence.
        looks = looks or {}
        rows = []
        for track in tracks:
            if not _usable(track):
                continue
            vector = looks.get(track.id)
            look = Appearance(np.asarray(vector, dtype=np.float64), MIN_APPEARANCE_PIXELS) if vector else None
            rows.append((track.id, track.class_id, track.contact, look))
        self._seen[camera_id] = _Seen(at_millis, pose, tuple(rows))
        self._observe_strangers(rows)

    def _observe_strangers(self, rows) -> None:
        """Two tracks in one frame are different objects, so how far apart
        they look is what a stranger scores — measured after normalisation,
        which is the space the cross-camera comparison happens in."""
        looks = [look for _id, _class, _contact, look in rows if isinstance(look, Appearance)]
        for i, first in enumerate(looks):
            for second in looks[i + 1:]:
                self.separation.observe_different(first.distance(second))

    def resolve(self) -> tuple[Pairing, ...]:
        """Triangulate every overlapping camera pair, once, on their freshest
        sightings. Called once per cycle by whatever is polling the cameras."""
        out: list[Pairing] = []
        names = sorted(self._seen)
        for i, a_id in enumerate(names):
            a = self._seen[a_id]
            if not a.tracks:
                continue
            for b_id in names[i + 1:]:
                b = self._seen[b_id]
                if not b.tracks:
                    continue
                if abs(a.at_millis - b.at_millis) > MAX_SKEW_MILLIS:
                    # Not one moment. Pairing across it triangulates a person
                    # against where they were a second ago, which converges
                    # perfectly well and is wrong.
                    continue
                out.extend(self._pair(a_id, a, b_id, b, max(a.at_millis, b.at_millis)))
        self._pairings = tuple(out)
        for pairing in self._pairings:
            self._remember_ground(pairing.point)
        return self._pairings

    def _pair(self, a_id: str, a: _Seen, b_id: str, b: _Seen, at_millis: int) -> list[Pairing]:
        """Every same-class pair whose rays actually converge.

        Best-gap-wins per track rather than every pair that passes: one track
        cannot be two objects, and a crowd otherwise produces a pairing for
        each combination and a position for each pairing.
        """
        found: list[Pairing] = []
        taken: set[int] = set()
        for track_a, class_a, contact_a, look_a in a.tracks:
            best: tuple[float, Pairing] | None = None
            for track_b, class_b, contact_b, look_b in b.tracks:
                if class_a != class_b or track_b in taken:
                    continue
                result = triangulate(
                    Sighting(a.pose, contact_a), Sighting(b.pose, contact_b),
                    frame=self._frame,
                    min_parallax_deg=self._parallax_for(a.pose, b.pose),
                )
                if isinstance(result, Refusal):
                    continue
                if best is None or result.gap_m < best[0]:
                    best = (result.gap_m, Pairing(a_id, track_a, b_id, track_b,
                                                  result, at_millis),
                            look_a, look_b)
            if best is not None:
                taken.add(best[1].track_b)
                found.append(best[1])
                # Geometry decided this pair without consulting appearance, so
                # how far apart the two *look* is free ground truth for what
                # the same object scores across these two cameras. This is the
                # evidence a labelled dataset would otherwise be needed for.
                if isinstance(best[2], Appearance) and isinstance(best[3], Appearance):
                    self.separation.observe_same(best[2].distance(best[3]))
        return found

    def _parallax_for(self, a: CameraPose, b: CameraPose) -> float:
        """How much parallax *these two cameras* need, from how well they are
        known.

        A constant floor is wrong in both directions. The error of a two-view
        point is `range * sigma / sin(parallax)`, so a calibrated pair at
        +/-0.06 degrees clears `WORTH_HAVING_M` at almost any angle, while an
        uncalibrated pair at +/-2 degrees and 60 m needs about 44 degrees
        before it beats the projection it would replace. Requiring five of
        both lets the second produce confident nonsense; requiring forty-four
        of both throws away the first entirely.

        Solved rather than tabulated: `sin(p) >= range * sigma / tolerance`.
        When the right-hand side exceeds one, no geometry saves this pair and
        the returned 90 degrees refuses every one of them, which is the
        correct answer rather than an awkward one.
        """
        if self.min_parallax_deg is not None:
            return self.min_parallax_deg
        sigma = math.radians(max(Sighting(a, Vec2(0.5, 0.5)).angular_sigma_deg,
                                 Sighting(b, Vec2(0.5, 0.5)).angular_sigma_deg))
        reach = max(a.range_meters, b.range_meters)
        needed = reach * sigma / WORTH_HAVING_M
        if needed >= 1.0:
            return 90.0
        return max(MIN_PARALLAX_DEG, math.degrees(math.asin(needed)))

    def _remember_ground(self, point: TriangulatedPoint) -> None:
        local = self._frame.to_local(point.position)
        self._ground.append((local.x, local.y, point.height_m))
        if len(self._ground) > GROUND_SAMPLES:
            del self._ground[: len(self._ground) - GROUND_SAMPLES]

    # ------------------------------------------------------------ the ground

    def solve_ground(self, *, threshold_m: float = 0.3, seed: int = 1) -> GroundPlane | None:
        """Fit the ground to everything triangulated so far.

        `None` when there is not enough, which is the common case early in a
        run and is not a failure.
        """
        if len(self._ground) < 3:
            return None
        plane = fit_ground(self._ground, self.origin, threshold_m=threshold_m, seed=seed)
        if plane is not None and plane.inliers >= MIN_GROUND_SAMPLES:
            self._plane = plane
            _log.info("ground solved: %s", plane.describe())
        return plane

    @property
    def ground(self) -> GroundPlane | None:
        """The last solved plane, or `None` while the site is still assuming
        a level one."""
        return self._plane

    @property
    def samples(self) -> int:
        return len(self._ground)

    def height_above_ground(self, point: TriangulatedPoint) -> float:
        """Metres above the *fitted* ground, falling back to the nominal one.

        This is the number that tells a person on a dock from a person on the
        yard, and a wall from a floor.
        """
        if self._plane is None:
            return point.height_m
        local = self._frame.to_local(point.position)
        return self._plane.height_above(local.x, local.y, point.height_m)

    def stands_on_the_ground(self, point: TriangulatedPoint) -> bool:
        return abs(self.height_above_ground(point)) <= NOT_GROUND_M

    def pairings(self) -> tuple[Pairing, ...]:
        return self._pairings

    def appearance_ceiling(self) -> float | None:
        """How far apart two sightings on different cameras may look and still
        be called one object, or `None` when appearance cannot say."""
        return self.separation.ceiling()


def tilted(pose: CameraPose, plane: GroundPlane | None) -> CameraPose:
    """The pose with its terrain assumption replaced by the measured slope.

    The slope is not the same quantity as `terrain_slope`, which is an
    uncertainty — a claim about how far the ground *might* depart from level.
    Once it is measured, what remains uncertain is the residual about the
    fitted plane, which is what `rms` over the projected range gives. A site
    that measures a 3% fall should not keep carrying a 2% error bar for the
    possibility of one.
    """
    if plane is None or plane.inliers < MIN_GROUND_SAMPLES:
        return pose
    # The residual, as a fraction of a typical range. Floored well above zero:
    # a plane fitted to a flat yard has a tiny RMS and the ground is still not
    # a mathematical plane.
    residual = max(0.005, plane.rms / max(1.0, pose.range_meters))
    return replace(pose, uncertainty=replace(pose.uncertainty, terrain_slope=residual))


def _usable(track) -> bool:
    """A track worth offering to the triangulator.

    Coasted tracks are excluded. A coasted box is the filter's prediction, not
    a measurement, and two predictions triangulate into a confident position
    that nothing observed.
    """
    from ..domain.tracking import TrackState

    return (getattr(track, "state", None) is TrackState.CONFIRMED
            and track.last_detected_millis == track.last_seen_millis)
