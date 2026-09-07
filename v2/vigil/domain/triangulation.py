"""Two cameras, one object, and the ground under it — without assuming a plane.

# The assumption this removes

`geo.project_to_ground` intersects one ray with an *assumed* plane: level
ground at the camera's mount height. It is the right answer when there is only
one camera, and it is wrong in three specific ways that this module measures
instead of assuming.

- **Anything not standing on the ground is placed long.** A person on a
  loading dock 1.4 m up, seen from 35 m, is put roughly 8 m past themselves,
  because the ray is followed until it reaches a plane the person is not on.
- **A sloped yard biases every position along the line of sight.**
  `PoseUncertainty.terrain_slope` exists only to widen the error bar over that
  bias; it never removes it.
- **Nothing can tell a wall from a floor.** The map smears a wall radially and
  the confidence layer marks it unusable because the samples *disagreed*, not
  because it is a wall.

Two cameras that can both see one object need none of that. The object is
where their rays come closest, its height above the fitted ground is a
measurement, and a thing consistently a metre and a half above the ground is
standing on something.

# What is not claimed

The pairing is somebody else's job. This module takes two sightings that a
caller believes are the same object and says where that object is — and, when
they cannot be, refuses. `service.triangulation` does the pairing, and the
[`Refusal.TOO_FAR_APART`] answer here is what catches it when the pairing was
wrong.

# The frame, and the one assumption that remains

Everything is computed in a local ENU frame in metres, with `z = 0` at the
site's nominal ground level and each camera at `z = mount_height`. That last
part is the assumption that survives: two cameras whose mount heights are
measured from ground at *different* levels put their origins wrong by the
difference. On a yard that is centimetres. On a site with a camera on a roof
and another in a basement, it is not, and `fit_ground` is what would have to
be run per region rather than per site.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence

import numpy as np

from ..kernel import native
from .geo import (
    CameraPose, LatLon, LocalFrame, PositionEstimate, PositionSource, Vec2, basis_for,
)

#: Angle between two rays below which a pair is refused. See
#: `core/src/triangulate.rs`, which derives it from the error rather than
#: choosing it: at a quarter-degree of pose error, 40 m and 5 degrees is
#: about 2 m, which is where triangulating stops beating projecting.
MIN_PARALLAX_DEG = native.MIN_PARALLAX_DEG

#: Closest approach beyond which the two rays are not looking at one object.
MAX_GAP_M = native.MAX_GAP_M


class Refusal(StrEnum):
    """Why a pair could not be triangulated, in words a caller can report."""

    PARALLEL = "PARALLEL"
    TOO_LITTLE_PARALLAX = "TOO_LITTLE_PARALLAX"
    BEHIND = "BEHIND"
    TOO_FAR_APART = "TOO_FAR_APART"
    #: One of the cameras is not placed, so it has no ray to contribute.
    NOT_PLACED = "NOT_PLACED"

    def describe(self) -> str:
        return {
            Refusal.PARALLEL: "the two cameras look along the same line, so they see one ray "
                              "between them and cannot say how far along it anything is",
            Refusal.TOO_LITTLE_PARALLAX: f"the cameras are too close together for this range: "
                                         f"under {MIN_PARALLAX_DEG:.0f} degrees of parallax the "
                                         f"answer is worse than projecting one camera onto the "
                                         f"ground",
            Refusal.BEHIND: "the rays meet behind one of the cameras, which means these are not "
                            "the same object",
            Refusal.TOO_FAR_APART: f"the rays passed more than {MAX_GAP_M:.0f} m apart, so they "
                                   f"are looking at two different things",
            Refusal.NOT_PLACED: "one of these cameras has no placement, so it has no ray",
        }[self]


_CODES = {
    native.PARALLEL: Refusal.PARALLEL,
    native.TOO_LITTLE_PARALLAX: Refusal.TOO_LITTLE_PARALLAX,
    native.BEHIND: Refusal.BEHIND,
    native.TOO_FAR_APART: Refusal.TOO_FAR_APART,
}


@dataclass(frozen=True, slots=True)
class Sighting:
    """One camera's view of one object: where it is looking, and how well it
    knows where it is looking."""

    pose: CameraPose
    #: Normalised image point — for a track, its ground contact.
    image: Vec2

    @property
    def angular_sigma_deg(self) -> float:
        """The pose's own pointing error, which is what scales the result.

        The worst of heading, pitch and roll rather than a combination: the
        rays are being intersected in three dimensions and there is no single
        axis the error lies along, so the honest scalar is the largest of them.
        """
        u = self.pose.uncertainty
        return max(u.heading_deg, u.pitch_deg, u.roll_deg)


@dataclass(frozen=True, slots=True)
class TriangulatedPoint:
    """Where two cameras agree an object is, and every number to disbelieve it."""

    position: LatLon
    #: Metres above the frame's nominal ground level. The signal that says an
    #: object is not standing on the ground.
    height_m: float
    #: 1-sigma horizontal, metres, in its worst direction — along the baseline.
    sigma_m: float
    parallax_deg: float
    #: How far apart the rays passed. Near zero is two views of one object.
    gap_m: float
    range_a: float
    range_b: float

    def estimate(self) -> PositionEstimate:
        return PositionEstimate(self.position, self.sigma_m, PositionSource.TRIANGULATED)

    def describe(self) -> str:
        return (f"{self.position.lat:.6f},{self.position.lon:.6f} "
                f"+/-{self.sigma_m:.2f} m, {self.height_m:+.2f} m above the ground, "
                f"{self.parallax_deg:.0f} deg parallax, rays {self.gap_m:.2f} m apart")


def _origin_of(pose: CameraPose, frame: LocalFrame) -> np.ndarray:
    local = frame.to_local(pose.position)
    return np.array([local.x, local.y, pose.mount_height], dtype=np.float64)


def triangulate(a: Sighting, b: Sighting, *,
                frame: LocalFrame | None = None,
                min_parallax_deg: float = MIN_PARALLAX_DEG) -> TriangulatedPoint | Refusal:
    """Where two sightings of one object put it, or why they cannot.

    `frame` fixes the local ENU origin. Pass the site's own frame when the
    result will be compared with anything else computed in one; two frames
    disagree by the convergence of meridians, which is millimetres over a
    yard and metres over a country, and is how two screens come to disagree
    about where something is.
    """
    if a.pose is None or b.pose is None:
        return Refusal.NOT_PLACED
    frame = frame if frame is not None else LocalFrame(a.pose.position)
    a_origin, b_origin = _origin_of(a.pose, frame), _origin_of(b.pose, frame)
    a_direction = np.array(basis_for(a.pose).ray(a.image.x, a.image.y), dtype=np.float64)
    b_direction = np.array(basis_for(b.pose).ray(b.image.x, b.image.y), dtype=np.float64)
    sigma = max(a.angular_sigma_deg, b.angular_sigma_deg)

    code, values = native.triangulate(a_origin, a_direction, b_origin, b_direction,
                                      sigma, min_parallax_deg)
    if code != 0:
        return _CODES.get(code, Refusal.PARALLEL)
    east, north, up = values[0], values[1], values[2]
    return TriangulatedPoint(
        position=frame.to_lat_lon(Vec2(east, north)),
        height_m=float(up),
        sigma_m=float(values[7]),
        parallax_deg=float(values[3]),
        gap_m=float(values[4]),
        range_a=float(values[5]),
        range_b=float(values[6]),
    )


@dataclass(frozen=True, slots=True)
class GroundPlane:
    """The ground a site's own observations say it has, rather than the level
    plane every projection assumed."""

    #: Unit normal in ENU, kept pointing up.
    normal: tuple[float, float, float]
    offset: float
    #: Rise per metre east and per metre north. What a pose needs to stop
    #: assuming a level yard.
    tilt_east: float
    tilt_north: float
    #: How many of the observations this plane explains. The number that
    #: decides whether to believe it at all.
    inliers: int
    #: RMS height of those about the plane, metres.
    rms: float
    #: The frame the numbers are in. Two planes from two frames are two
    #: different claims, and mixing them is the meridian-convergence bug
    #: `LocalFrame` exists to prevent.
    origin: LatLon

    @property
    def slope(self) -> float:
        """Fractional steepest slope — what `PoseUncertainty.terrain_slope`
        assumes, now measured."""
        return math.hypot(self.tilt_east, self.tilt_north)

    def height_above(self, east: float, north: float, up: float) -> float:
        nx, ny, nz = self.normal
        return nx * east + ny * north + nz * up - self.offset

    def describe(self) -> str:
        return (f"ground falls {self.slope * 100:.1f}% "
                f"({self.tilt_east * 100:+.1f}% east, {self.tilt_north * 100:+.1f}% north) "
                f"from {self.inliers} observations, {self.rms:.2f} m RMS")


def fit_ground(points: Sequence[tuple[float, float, float]], origin: LatLon, *,
               threshold_m: float = 0.3, iterations: int = 200,
               seed: int = 1) -> GroundPlane | None:
    """The ground plane through observed ENU contact points.

    `threshold_m` is how far off the plane a point may be and still count. It
    is a property of the data, not a knob: at 40 m with a quarter-degree pose,
    a ground contact is worth about 0.2 m vertically, so 0.3 m suits a
    calibrated camera and 1.0 m an assumed one.

    `None` when there is nothing a plane explains — which is what a car park
    full of parked cars looks like, and is a better answer than a plane fitted
    to their roofs.
    """
    cloud = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(cloud) < 3:
        return None
    fit = native.fit_plane(cloud, threshold_m, iterations, seed)
    if fit is None:
        return None
    return GroundPlane(
        normal=(float(fit[0]), float(fit[1]), float(fit[2])),
        offset=float(fit[3]),
        tilt_east=float(fit[6]),
        tilt_north=float(fit[7]),
        inliers=int(fit[4]),
        rms=float(fit[5]),
        origin=origin,
    )
