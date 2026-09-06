"""The camera model: where a thing in a picture is on the ground.

Geodesy — latitude, longitude, metres and rings — moved to `geodesy` when this
module outgrew its line budget, along the same seam the Rust core already
uses: `core/src/geodesy.rs` and `core/src/camera.rs`. Every name from there is
re-exported below, so `from .geo import LatLon` still works and no caller had
to change.

Everything here is a pure function of its arguments. A position is never
returned without a statement of how it was obtained and how well it is known:
a map that implies precision it does not have sends somebody to the wrong
place.

This module and `core/src/camera.rs` are the same model. The Rust copy exists
because the map builder projects a thousand points per frame and cannot cross
a language boundary to do it; `tests/test_native.py` drives both over a grid
of poses and holds them to 1e-9. This is the copy to read.

# The defect this replaced

**The camera model was not a pinhole.** v1's Rust core and v2's port of it
both computed `yaw = atan(dx*tan(hfov/2))` and `elevation = pitch +
atan(dy*tan(vfov/2))`, and the docstring called it rectilinear. Treating the
two axes as independent describes a sensor bent into a cylinder; a rectilinear
lens puts the scene on a *plane*, where the axes are coupled through the tilt.
The error is zero along the centre row and grows into the corners with the
pitch. At this product's own reference pose — 4 m mast, 25 degrees down, 62 by
36 — the bottom corner came out 4.8 degrees wrong in elevation, which is a
**21% error in the distance** to whatever was standing there.

`roll` made it worse by being ignored: the field has existed since v1 and
nothing read it, so a camera clamped a few degrees off level folded that error
into every position it produced.

**There were two Earths**, 0.248% apart. That one is described where it now
lives, in `geodesy`.

# Conventions, stated once

- World frame is **ENU**: east, north, up. Metres.
- `heading` is degrees **clockwise from true north**, the compass sense.
- `pitch` is the **elevation of the optical axis**: negative looks down.
- `roll` is degrees **clockwise about the optical axis seen from behind the
  camera** — the way a horizon tips when you tilt your head right.
- Image coordinates are normalised `[0, 1]`, `(0, 0)` **top-left**.
- The ground is the plane `z = 0`; the camera is at `z = mount_height`.

# What is still assumed

A flat ground plane and a distortion-free lens. Both are stated rather than
hidden, and both widen the error rather than being corrected for:
`PoseUncertainty.terrain_slope` makes a slope show up as uncertainty instead
of as a confident wrong answer, and the field-of-view intrinsics are
documented as the fallback for a camera nobody has calibrated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Sequence

from .lens import Distortion
from .measure import (  # noqa: F401 - re-exported; callers import these from `geo`
    Bearing, Distance, ProjectionUncertainty, Speed, motion_of, principal_axes,
)
from .geodesy import (  # noqa: F401 - re-exported so no caller had to change
    EARTH_RADIUS_M, LatLon, LocalFrame, Vec2, angle_difference, bearing_degrees,
    destination_point, distance_meters, distance_to_ring_edge, meters_per_degree_latitude,
    meters_per_degree_longitude, normalize_degrees, normalize_longitude, point_in_polygon,
    point_in_ring, spherical_distance,
)

#: Below this depression angle a ray is refused rather than projected. At 2
#: degrees a 6 m mast reaches 172 m and one pixel of contact error is worth
#: 3 m of range: the answer is not wrong so much as meaningless.
MIN_DEPRESSION_ANGLE_DEG = 2.0

#: 1-sigma angular error of a detection's ground-contact point. Three quarters
#: of a degree is about 13 px on a 1080-line frame at 36 degrees vertical:
#: roughly what a box's bottom edge is worth against a real foot.
DEFAULT_ANGULAR_UNCERTAINTY_DEG = 0.75

# ------------------------------------------------------------------- camera


@dataclass(frozen=True, slots=True)
class PoseUncertainty:
    """How well each input to a projection is known, 1-sigma.

    Separate from the pose because a pose is a claim about the world and this
    is a claim about the claim; they have different lifetimes. A camera that
    gets surveyed keeps its position and gains a smaller `heading_deg`.

    The defaults are what an operator clicking a map and sighting along a
    compass is worth. They are stated assumptions, not measurements, and they
    are deliberately not small: a system that reports a metre of error it
    cannot justify is worse than one that reports three it can.
    """

    #: A camera aimed by eye is not better than this.
    heading_deg: float = 2.0
    pitch_deg: float = 2.0
    roll_deg: float = 2.0
    #: A tape-measured mast.
    mount_height_m: float = 0.15
    #: Ground-plane departure as a fraction of distance. 2% is a 1-in-50
    #: slope: a yard that drains. The term that says "the ground is not
    #: actually flat" out loud instead of assuming it away.
    terrain_slope: float = 0.02

    @classmethod
    def exact(cls) -> "PoseUncertainty":
        """Everything known perfectly. For tests that want the contact
        point's own error and nothing else; never for a real camera."""
        return cls(0.0, 0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class CameraPose:
    """Where a camera is, where it looks, and how far its answers are worth
    having. Angles in degrees, heights in metres."""

    position: LatLon
    mount_height: float
    heading: float
    pitch: float
    roll: float = 0.0
    horizontal_fov: float = 62.0
    vertical_fov: float = 36.0
    #: Beyond this ground distance a projection is refused, not clamped.
    range_meters: float = 60.0
    #: How well the eight numbers above are known.
    uncertainty: PoseUncertainty = field(default_factory=PoseUncertainty)
    #: What the lens does to a straight line. All-zero — the default — is an
    #: uncalibrated camera assumed rectilinear, which is what every camera was
    #: before `vigil cameras calibrate` existed.
    lens: Distortion = field(default_factory=Distortion)

    def validate(self) -> None:
        if not self.position.is_valid():
            raise ValueError("position is off the planet")
        if self.mount_height <= 0:
            raise ValueError("mount height must be above the ground")
        if not (1 <= self.horizontal_fov < 180 and 1 <= self.vertical_fov < 180):
            raise ValueError("fields of view must be between 1° and 180°")
        if not -90 <= self.pitch <= 90:
            raise ValueError("pitch is an elevation, so it is between -90° and 90°")
        if not -180 <= self.roll <= 180:
            raise ValueError("roll is between -180° and 180°")
        if self.range_meters <= 0:
            raise ValueError("range must be positive")
        if not self.lens.is_identity:
            # A lens is only usable if it can be *inverted* over the frame it
            # covers, and that is not guaranteed: Brown-Conrady is a
            # polynomial with no closed-form inverse, solved by iteration, and
            # at a wide enough angle with a strong enough barrel it diverges.
            # Measured: a 110-degree lens with k1=-0.28 fails outright at any
            # iteration count. Checked at the **raw image corner**, because
            # that is the coordinate `ray` actually undistorts — checking a
            # distorted-then-undistorted point instead tests a smaller radius
            # than the one that fails, and passes a lens that does not work.
            corner_x = math.tan(math.radians(self.horizontal_fov / 2))
            corner_y = math.tan(math.radians(self.vertical_fov / 2))
            if not self.lens.converges(corner_x, corner_y):
                raise ValueError(
                    f"this lens correction cannot be inverted at the corner of a "
                    f"{self.horizontal_fov:.0f}x{self.vertical_fov:.0f} frame, so a detection there "
                    f"would be placed anywhere. Re-calibrate, or use a fisheye model this product "
                    f"does not have"
                )


@dataclass(frozen=True, slots=True)
class _Basis:
    """The orthonormal camera axes in world ENU.

    `(right, up, forward)` is *left*-handed, because a compass heading turns
    clockwise while ENU turns anticlockwise; the identity that holds is
    `up x right = forward`. It is the same triad OpenCV writes as
    `(right, down, forward)` and calls right-handed.
    """

    right: tuple[float, float, float]
    up: tuple[float, float, float]
    forward: tuple[float, float, float]
    tan_half_h: float
    tan_half_v: float
    mount_height: float
    lens: Distortion = Distortion()

    def ray(self, u: float, v: float) -> tuple[float, float, float]:
        """The unnormalised world direction through a normalised image point.
        Its length is arbitrary and is never used as a distance.

        The pixel handed in has already been bent by the lens, so it is
        **undistorted** first: the ray belongs to the ideal coordinate, not to
        where the sensor recorded it. With no calibration this is the identity
        and costs one comparison.

        The correction happens in **normalised camera coordinates** — `x/z`,
        `y/z` — which is the space Brown-Conrady is defined in and which
        `(2u-1)*tan(hfov/2)` already is. Applying it in the `[-1, 1]` frame
        coordinate instead is a different function of a differently scaled
        argument, and the two directions then stop inverting each other.
        """
        x = (2.0 * u - 1.0) * self.tan_half_h
        y = (1.0 - 2.0 * v) * self.tan_half_v
        if not self.lens.is_identity:
            x, y = self.lens.undistort(x, y)
        return (
            self.right[0] * x + self.up[0] * y + self.forward[0],
            self.right[1] * x + self.up[1] * y + self.forward[1],
            self.right[2] * x + self.up[2] * y + self.forward[2],
        )

    def ground_offset(self, u: float, v: float) -> Vec2 | None:
        """Where that ray meets the ground, as (east, north) metres from the
        camera. `None` at or above the horizon.

        This is the whole projection: `t = h / -d_up`, and the horizontal part
        of `t*d`. No `tan`, no case analysis, no decoupled axes.
        """
        dx, dy, dz = self.ray(u, v)
        if not dz < 0.0 or not math.isfinite(dz):
            return None
        t = self.mount_height / -dz
        if not math.isfinite(t) or t <= 0.0:
            return None
        east, north = t * dx, t * dy
        if not (math.isfinite(east) and math.isfinite(north)):
            return None
        return Vec2(east, north)


def _basis(heading: float, pitch: float, roll: float, tan_half_h: float,
           tan_half_v: float, mount_height: float,
           lens: Distortion = Distortion()) -> _Basis:
    psi, theta, phi = math.radians(heading), math.radians(pitch), math.radians(roll)
    sin_psi, cos_psi = math.sin(psi), math.cos(psi)
    sin_th, cos_th = math.sin(theta), math.cos(theta)
    sin_ph, cos_ph = math.sin(phi), math.cos(phi)
    forward = (sin_psi * cos_th, cos_psi * cos_th, sin_th)
    # Right is horizontal by construction: roll tips the image, it does not
    # move where the lens points.
    right0 = (cos_psi, -sin_psi, 0.0)
    up0 = (-sin_th * sin_psi, -sin_th * cos_psi, cos_th)
    right = tuple(right0[i] * cos_ph - up0[i] * sin_ph for i in range(3))
    up = tuple(right0[i] * sin_ph + up0[i] * cos_ph for i in range(3))
    return _Basis(right, up, forward, tan_half_h, tan_half_v, mount_height, lens)  # type: ignore[arg-type]


def basis_for(pose: CameraPose) -> _Basis:
    """The camera's axes. Build once and reuse across many rays: the six trig
    calls are the whole cost of a projection."""
    return _basis(
        pose.heading, pose.pitch, pose.roll,
        math.tan(math.radians(pose.horizontal_fov / 2)),
        math.tan(math.radians(pose.vertical_fov / 2)),
        pose.mount_height, pose.lens,
    )


class PositionSource(StrEnum):
    GROUND_PROJECTION = "GROUND_PROJECTION"
    CAMERA_FALLBACK = "CAMERA_FALLBACK"


class ProjectionFailure(StrEnum):
    """Why a projection was refused.

    Named rather than collapsed into `None`, because the two that matter are
    different problems for whoever has to fix them: `TOO_SHALLOW` is a camera
    aimed too flat and `OUT_OF_RANGE` is a camera sited too far from what it
    is meant to watch. v1 and v2 both returned `None` and told an operator
    neither.
    """

    ABOVE_HORIZON = "ABOVE_HORIZON"
    TOO_SHALLOW = "TOO_SHALLOW"
    OUT_OF_RANGE = "OUT_OF_RANGE"
    BAD_POSE = "BAD_POSE"


@dataclass(frozen=True, slots=True)
class PositionEstimate:
    point: LatLon
    #: 1-sigma horizontal uncertainty, metres: the ellipse's semi-major axis.
    radius_meters: float
    source: PositionSource
    #: The full ellipse, when there is one. `None` for a camera fallback,
    #: where there is no line of sight to be along or across.
    ellipse: ProjectionUncertainty | None = None

    @property
    def is_projected(self) -> bool:
        return self.source is PositionSource.GROUND_PROJECTION


def separation(a: PositionEstimate, b: PositionEstimate) -> Distance:
    """How far apart two estimated positions are, with the errors combined.

    In quadrature, because the two projections are independent measurements:
    adding them would claim the errors always conspire, and ignoring one would
    claim it does not exist.
    """
    return Distance(distance_meters(a.point, b.point), math.hypot(a.radius_meters, b.radius_meters))


def distance_from_camera(pose: CameraPose, position: PositionEstimate) -> Distance:
    """From the mast to the object.

    The camera's own position is taken as given — an operator typed it — so
    the error here is the projection's alone. If the placement is wrong, every
    distance from this camera is wrong by the same amount, which is a
    different problem and a visible one.
    """
    return Distance(distance_meters(pose.position, position.point), position.radius_meters)


@dataclass(frozen=True, slots=True)
class GroundProjection:
    position: LatLon
    ground_distance_meters: float
    bearing_deg: float
    uncertainty: ProjectionUncertainty

    @property
    def uncertainty_meters(self) -> float:
        return self.uncertainty.radius_meters


def ray_angles(pose: CameraPose, u: float, v: float) -> tuple[float, float]:
    """(bearing, elevation) of the ray through a normalised image point.

    A true pinhole: the two angles are coupled through the tilt, which is what
    the decoupled formula this replaced got wrong by degrees at the corners of
    a tilted frame.
    """
    dx, dy, dz = basis_for(pose).ray(min(1.0, max(0.0, u)), min(1.0, max(0.0, v)))
    return (normalize_degrees(math.degrees(math.atan2(dx, dy))),
            math.degrees(math.atan2(dz, math.hypot(dx, dy))))


# Steps for the finite-difference Jacobian, per parameter, in its own units.
# Central differences, so truncation is O(step^2) and round-off O(eps/step);
# these sit near the minimum of the sum for a projection whose scale is
# metres. The alternative — six hand-derived partials through a rotation
# composition — is the kind of algebra that stays wrong for a year without
# anybody noticing. The height column has a closed form and a test compares
# against it.
_STEP_ANGLE_DEG = 1e-4
_STEP_HEIGHT_M = 1e-4
_STEP_IMAGE = 1e-5


def project_to_ground_detail(
    pose: CameraPose, u: float, v: float,
    angular_uncertainty_deg: float = DEFAULT_ANGULAR_UNCERTAINTY_DEG, *,
    enforce_range: bool = True,
) -> tuple[GroundProjection | None, ProjectionFailure | None]:
    """The projection, or the reason there is not one. Exactly one is not
    `None`."""
    if not pose.position.is_valid() or not (pose.mount_height > 0) or not math.isfinite(pose.mount_height):
        return None, ProjectionFailure.BAD_POSE
    if not (0 < pose.horizontal_fov < 180 and 0 < pose.vertical_fov < 180) or not pose.range_meters > 0:
        return None, ProjectionFailure.BAD_POSE
    basis = basis_for(pose)
    dx, dy, dz = basis.ray(u, v)
    elevation = math.degrees(math.atan2(dz, math.hypot(dx, dy)))
    if elevation >= 0:
        return None, ProjectionFailure.ABOVE_HORIZON
    if -elevation < MIN_DEPRESSION_ANGLE_DEG:
        return None, ProjectionFailure.TOO_SHALLOW
    offset = basis.ground_offset(u, v)
    if offset is None:
        return None, ProjectionFailure.ABOVE_HORIZON
    distance = math.hypot(offset.x, offset.y)
    if enforce_range and distance > pose.range_meters:
        return None, ProjectionFailure.OUT_OF_RANGE
    bearing = normalize_degrees(math.degrees(math.atan2(offset.x, offset.y)))
    return GroundProjection(
        position=destination_point(pose.position, bearing, distance),
        ground_distance_meters=distance,
        bearing_deg=bearing,
        uncertainty=_ground_uncertainty(pose, u, v, angular_uncertainty_deg, offset),
    ), None


def project_to_ground(
    pose: CameraPose, u: float, v: float,
    angular_uncertainty_deg: float = DEFAULT_ANGULAR_UNCERTAINTY_DEG, *,
    enforce_range: bool = True,
) -> GroundProjection | None:
    """Where a normalised image point meets the ground, or `None`.

    Nothing rather than a clamped guess: a position the system cannot
    determine must not appear on a map. `project_to_ground_detail` says which
    of the four reasons applied.
    """
    projection, _ = project_to_ground_detail(
        pose, u, v, angular_uncertainty_deg, enforce_range=enforce_range
    )
    return projection


def _ground_uncertainty(pose: CameraPose, u: float, v: float, contact_sigma_deg: float,
                        offset: Vec2) -> ProjectionUncertainty:
    """Propagate every stated input error through the projection.

    `Sigma_g = J Sigma_p J'` with `J` by central differences over (heading,
    pitch, roll, mount height, u, v), then a terrain term along the line of
    sight — terrain tilts the plane the ray lands on, which moves the hit
    towards or away from the camera and barely sideways.

    The result is rotated into (along, across) the line of sight, which for a
    mast camera is within a percent of the true eigenvectors. Reporting a
    rotated ellipse whose axes are 1% off is far more honest than reporting a
    circle that is 300% off.
    """
    tan_half_h = math.tan(math.radians(pose.horizontal_fov / 2))
    tan_half_v = math.tan(math.radians(pose.vertical_fov / 2))

    def at(heading, pitch, roll, height, du, dv) -> Vec2 | None:
        return _basis(heading, pitch, roll, tan_half_h, tan_half_v, height,
                      pose.lens).ground_offset(u + du, v + dv)

    # The contact point's angular error, in the image coordinates it is
    # measured in. The vertical axis carries the range error and the two
    # half-angles differ, so one shared value would be wrong on one axis.
    sigma_u = math.radians(contact_sigma_deg) / (2 * math.atan(tan_half_h))
    sigma_v = math.radians(contact_sigma_deg) / (2 * math.atan(tan_half_v))
    sigma = pose.uncertainty
    h, p, r, m = pose.heading, pose.pitch, pose.roll, pose.mount_height

    columns = (
        (sigma.heading_deg, _STEP_ANGLE_DEG,
         at(h + _STEP_ANGLE_DEG, p, r, m, 0, 0), at(h - _STEP_ANGLE_DEG, p, r, m, 0, 0)),
        (sigma.pitch_deg, _STEP_ANGLE_DEG,
         at(h, p + _STEP_ANGLE_DEG, r, m, 0, 0), at(h, p - _STEP_ANGLE_DEG, r, m, 0, 0)),
        (sigma.roll_deg, _STEP_ANGLE_DEG,
         at(h, p, r + _STEP_ANGLE_DEG, m, 0, 0), at(h, p, r - _STEP_ANGLE_DEG, m, 0, 0)),
        (sigma.mount_height_m, _STEP_HEIGHT_M,
         at(h, p, r, m + _STEP_HEIGHT_M, 0, 0), at(h, p, r, m - _STEP_HEIGHT_M, 0, 0)),
        (sigma_u, _STEP_IMAGE, at(h, p, r, m, _STEP_IMAGE, 0), at(h, p, r, m, -_STEP_IMAGE, 0)),
        (sigma_v, _STEP_IMAGE, at(h, p, r, m, 0, _STEP_IMAGE), at(h, p, r, m, 0, -_STEP_IMAGE)),
    )

    cov = [[0.0, 0.0], [0.0, 0.0]]
    for param_sigma, step, plus, minus in columns:
        # A perturbation that pushes the ray over the horizon has no
        # derivative worth taking; the depression floor has already refused
        # the rays where that can happen at any distance from the horizon.
        if not param_sigma > 0 or plus is None or minus is None:
            continue
        de = (plus.x - minus.x) / (2 * step) * param_sigma
        dn = (plus.y - minus.y) / (2 * step) * param_sigma
        cov[0][0] += de * de
        cov[0][1] += de * dn
        cov[1][0] += de * dn
        cov[1][1] += dn * dn

    distance = math.hypot(offset.x, offset.y)
    if distance > 1e-9:
        ux, uy = offset.x / distance, offset.y / distance
    else:
        ux, uy = 0.0, 1.0
    px, py = -uy, ux
    along_var = ux * (cov[0][0] * ux + cov[0][1] * uy) + uy * (cov[1][0] * ux + cov[1][1] * uy)
    across_var = px * (cov[0][0] * px + cov[0][1] * py) + py * (cov[1][0] * px + cov[1][1] * py)

    # Terrain: a plane tilted by `slope` moves the hit along the ray by about
    # `distance * slope / tan(depression)`. Capped at the distance itself,
    # because "it might be twice as far as it looks" is the most a slope can
    # honestly say before the answer should simply be refused.
    terrain = 0.0
    if sigma.terrain_slope > 0 and distance > 1e-9:
        depression = math.atan2(pose.mount_height, distance)
        terrain = min(distance, distance * sigma.terrain_slope / max(1e-6, math.tan(depression)))

    # The terrain term acts along the line of sight, so it is added to the
    # covariance in that frame before the eigendecomposition rather than to
    # one axis afterwards — otherwise the principal axes would be of a
    # different matrix than the one reported.
    if terrain > 0:
        cov[0][0] += (terrain * ux) ** 2
        cov[1][1] += (terrain * uy) ** 2
        cov[0][1] += terrain * terrain * ux * uy
        cov[1][0] = cov[0][1]
    major, minor, bearing = principal_axes(cov)
    return ProjectionUncertainty(
        along_meters=math.sqrt(max(0.0, along_var) + terrain * terrain),
        across_meters=math.sqrt(max(0.0, across_var)),
        orientation_deg=normalize_degrees(math.degrees(math.atan2(offset.x, offset.y))),
        semi_major_meters=major, semi_minor_meters=minor, major_bearing_deg=bearing,
    )


def project_point(pose: CameraPose, contact: Vec2) -> PositionEstimate:
    """A contact point to a map position, degrading honestly to the camera.

    The fallback is the camera's own position with the range as its error,
    which is the truthful statement "somewhere this camera can see". It is not
    a guess dressed as a measurement: `is_projected` is False and every
    consumer that matters checks it.
    """
    projection = project_to_ground(pose, contact.x, contact.y)
    if projection is None:
        return PositionEstimate(pose.position, pose.range_meters, PositionSource.CAMERA_FALLBACK)
    return PositionEstimate(projection.position, projection.uncertainty.radius_meters,
                            PositionSource.GROUND_PROJECTION, projection.uncertainty)


def _edge_distance(pose: CameraPose, v: float, pick) -> float | None:
    """The extreme ground distance along one horizontal edge of the frame.

    Sampled across the edge rather than read off its centre: with roll or a
    wide lens the nearest ground in view is at a *corner*. v1 read the centre
    column and drew a footprint whose near edge cut through ground the camera
    could see.
    """
    best: float | None = None
    for i in range(17):
        projection = project_to_ground(pose, i / 16, v, 0.0, enforce_range=False)
        if projection is None:
            continue
        best = projection.ground_distance_meters if best is None else pick(best, projection.ground_distance_meters)
    return best


def near_ground_distance(pose: CameraPose) -> float | None:
    return _edge_distance(pose, 1.0, min)


def far_ground_distance(pose: CameraPose) -> float | None:
    return _edge_distance(pose, 0.0, max)


def field_of_view(pose: CameraPose, arc_segments: int = 24) -> list[LatLon]:
    """The ground footprint, as a closed ring.

    An annular sector, never a pie slice: a downward-tilted camera does not
    see the ground at its own feet, and drawing the slice tells an operator
    the camera covers ground it is blind to. Empty when the camera sees no
    ground at all.

    Traced by walking the frame's border in image space, so a rolled camera
    produces the rotated footprint it actually has rather than a symmetric
    wedge that is right only when roll is zero.
    """
    segments = max(4, arc_segments)
    ring: list[LatLon] = []

    def clamped(projection: GroundProjection) -> LatLon:
        if projection.ground_distance_meters <= pose.range_meters:
            return projection.position
        return destination_point(pose.position, projection.bearing_deg, pose.range_meters)

    for v, order in ((0.0, range(segments + 1)), (1.0, range(segments, -1, -1))):
        for i in order:
            projection = project_to_ground(pose, i / segments, v, 0.0, enforce_range=False)
            if projection is not None:
                ring.append(clamped(projection))
    return ring if len(ring) >= 3 else []


def image_coordinates(pose: CameraPose, point: LatLon) -> Vec2 | None:
    """Where a ground point appears in the image; the exact inverse of the
    projection.

    Unclipped: a value outside 0..1 is meaningful and means "off the side of
    the frame by this much". `None` only when the point is behind the image
    plane.
    """
    basis = basis_for(pose)
    local = LocalFrame(pose.position).to_local(point)
    # The ground is `mount_height` below the camera, and the offset has to be
    # taken in the camera's own frame or the tilt gets applied twice.
    offset = (local.x, local.y, -pose.mount_height)
    depth = sum(offset[i] * basis.forward[i] for i in range(3))
    if not depth > 1e-9:
        return None
    x = sum(offset[i] * basis.right[i] for i in range(3))
    y = sum(offset[i] * basis.up[i] for i in range(3))
    # Normalised camera coordinates, which is the space the lens model is
    # defined in. Bend the ideal point to where the sensor actually records
    # it, so this stays the exact inverse of `ray`, which unbends it.
    camera_x, camera_y = x / depth, y / depth
    if not basis.lens.is_identity:
        camera_x, camera_y = basis.lens.distort(camera_x, camera_y)
    return Vec2((camera_x / basis.tan_half_h + 1) / 2,
                (1 - camera_y / basis.tan_half_v) / 2)


def camera_sees(pose: CameraPose, point: LatLon) -> bool:
    image = image_coordinates(pose, point)
    if image is None or not (0 <= image.x <= 1 and 0 <= image.y <= 1):
        return False
    return distance_meters(pose.position, point) <= pose.range_meters

