"""Geodesy and camera geometry.

Everything here is a pure function of its arguments. A position is never
returned without a statement of how it was obtained and how well it is known:
a map that implies precision it does not have sends somebody to the wrong
place.

This module and `core/src/camera.rs` are the same model. The Rust copy exists
because the map builder projects a thousand points per frame and cannot cross
a language boundary to do it; `tests/test_native.py` drives both over a grid
of poses and holds them to 1e-9. This is the copy to read.

# Two defects this replaced

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

**There were two Earths.** Distances between tracks went through a haversine
on a sphere; zone tests and the ground grid went through a tangent plane from
the WGS84 latitude series. They disagree by 0.248% — 25 cm per 100 m. A zone
edge and a track measured from the same camera were on different planets.
Everything is the tangent plane now, so `distance_meters`, `bearing_degrees`
and `destination_point` are exact inverses of each other instead of being
approximately consistent. `spherical_distance` is kept, called by nothing, and
tested as the reference that says how far apart the two models are.

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

#: Below this depression angle a ray is refused rather than projected. At 2
#: degrees a 6 m mast reaches 172 m and one pixel of contact error is worth
#: 3 m of range: the answer is not wrong so much as meaningless.
MIN_DEPRESSION_ANGLE_DEG = 2.0

#: 1-sigma angular error of a detection's ground-contact point. Three quarters
#: of a degree is about 13 px on a 1080-line frame at 36 degrees vertical:
#: roughly what a box's bottom edge is worth against a real foot.
DEFAULT_ANGULAR_UNCERTAINTY_DEG = 0.75

#: Mean Earth radius, IUGG. Used only by `spherical_distance`.
EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True, slots=True)
class LatLon:
    lat: float
    lon: float

    def is_valid(self) -> bool:
        return (
            math.isfinite(self.lat) and math.isfinite(self.lon)
            and -90.0 <= self.lat <= 90.0 and -180.0 <= self.lon <= 180.0
        )


@dataclass(frozen=True, slots=True)
class Vec2:
    x: float
    y: float


# ------------------------------------------------------------------ geodesy


def meters_per_degree_latitude(latitude_deg: float) -> float:
    lat = math.radians(latitude_deg)
    return 111132.92 - 559.82 * math.cos(2 * lat) + 1.175 * math.cos(4 * lat)


def meters_per_degree_longitude(latitude_deg: float) -> float:
    lat = math.radians(latitude_deg)
    return 111412.84 * math.cos(lat) - 93.5 * math.cos(3 * lat) + 0.118 * math.cos(5 * lat)


def normalize_degrees(deg: float) -> float:
    return deg % 360.0


def normalize_longitude(deg: float) -> float:
    return ((deg + 180.0) % 360.0) - 180.0


def angle_difference(a: float, b: float) -> float:
    """Signed smallest difference a - b in (-180, 180]."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d


class LocalFrame:
    """East/north metres around an origin.

    Construct once and reuse. Two *different* frames disagree by the
    convergence of meridians, `D*d*tan(lat)/R` for origins `D` apart and a
    point `d` from one of them: 0.6 mm for frames 60 m apart over a 100 m
    span here, and metres across a country. Building a second frame for a job
    that already has one is how two screens come to disagree about where a
    zone is.
    """

    __slots__ = ("origin", "_m_lat", "_m_lon")

    def __init__(self, origin: LatLon):
        self.origin = origin
        # A frame at a pole has no east. Clamped rather than left to divide by
        # zero: a NaN reaching a polygon test is worse than a useless frame.
        self._m_lat = max(1.0, meters_per_degree_latitude(origin.lat))
        self._m_lon = max(1.0, meters_per_degree_longitude(origin.lat))

    def to_local(self, point: LatLon) -> Vec2:
        return Vec2(normalize_longitude(point.lon - self.origin.lon) * self._m_lon,
                    (point.lat - self.origin.lat) * self._m_lat)

    def to_lat_lon(self, local: Vec2) -> LatLon:
        return LatLon(self.origin.lat + local.y / self._m_lat,
                      normalize_longitude(self.origin.lon + local.x / self._m_lon))


def distance_meters(a: LatLon, b: LatLon) -> float:
    """Metres between two points, on the tangent plane at `a`.

    This is the product's distance. It is not a haversine; see the module
    docstring for the 0.248% that cost.
    """
    local = LocalFrame(a).to_local(b)
    return math.hypot(local.x, local.y)


def bearing_degrees(a: LatLon, b: LatLon) -> float:
    """Degrees clockwise from true north, on the tangent plane at `a`."""
    local = LocalFrame(a).to_local(b)
    if local.x == 0.0 and local.y == 0.0:
        return 0.0
    return normalize_degrees(math.degrees(math.atan2(local.x, local.y)))


def destination_point(origin: LatLon, bearing_deg: float, distance_meters: float) -> LatLon:
    """The exact inverse of `distance_meters` and `bearing_degrees` from the
    same origin."""
    theta = math.radians(bearing_deg)
    return LocalFrame(origin).to_lat_lon(
        Vec2(distance_meters * math.sin(theta), distance_meters * math.cos(theta))
    )


def spherical_distance(a: LatLon, b: LatLon) -> float:
    """Great-circle distance on a sphere.

    Kept for reference and for the test that measures how far it is from
    `distance_meters`. The product does not use it: mixing the two is the
    defect this module was rewritten to remove.
    """
    phi1, phi2 = math.radians(a.lat), math.radians(b.lat)
    dphi = phi2 - phi1
    dlambda = math.radians(normalize_longitude(b.lon - a.lon))
    h = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(h)))


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

    def ray(self, u: float, v: float) -> tuple[float, float, float]:
        """The unnormalised world direction through a normalised image point.
        Its length is arbitrary and is never used as a distance."""
        x = (2.0 * u - 1.0) * self.tan_half_h
        y = (1.0 - 2.0 * v) * self.tan_half_v
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
           tan_half_v: float, mount_height: float) -> _Basis:
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
    return _Basis(right, up, forward, tan_half_h, tan_half_v, mount_height)  # type: ignore[arg-type]


def basis_for(pose: CameraPose) -> _Basis:
    """The camera's axes. Build once and reuse across many rays: the six trig
    calls are the whole cost of a projection."""
    return _basis(
        pose.heading, pose.pitch, pose.roll,
        math.tan(math.radians(pose.horizontal_fov / 2)),
        math.tan(math.radians(pose.vertical_fov / 2)),
        pose.mount_height,
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
class ProjectionUncertainty:
    """The error ellipse of a projected position, on the ground, in metres.

    An ellipse and not a radius because the two are genuinely different: a
    shallow ray is sharp across its own direction and vague along it, and a
    camera 40 m away reporting "plus or minus 6 m" as a circle claims a
    sideways error it does not have.
    """

    #: 1-sigma along the line of sight.
    along_meters: float
    #: 1-sigma across it.
    across_meters: float
    #: Bearing of the along-axis, degrees.
    orientation_deg: float

    @property
    def radius_meters(self) -> float:
        """The conservative single number: the semi-major axis. A circle
        fitted inside the ellipse would understate the error in the direction
        it actually points."""
        return max(self.along_meters, self.across_meters)

    @property
    def rms_meters(self) -> float:
        """For combining errors rather than for drawing them."""
        return math.sqrt((self.along_meters ** 2 + self.across_meters ** 2) / 2)


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


@dataclass(frozen=True, slots=True)
class Distance:
    """A distance and how well it is known. Never one without the other.

    Two positions each known to ±1.4 m are eleven metres apart *give or take
    about two*, and a plain "11 m" invites somebody to act on a precision
    nobody measured.
    """

    meters: float
    error_meters: float

    def describe(self) -> str:
        return f"{self.meters:.1f} ± {self.error_meters:.1f} m"

    @property
    def at_most(self) -> float:
        return self.meters + self.error_meters

    @property
    def at_least(self) -> float:
        return max(0.0, self.meters - self.error_meters)

    def within(self, limit: float) -> bool:
        """True only when it is within `limit` even at its worst."""
        return self.at_most <= limit

    def beyond(self, limit: float) -> bool:
        """True only when it is past `limit` even at its best."""
        return self.at_least > limit


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
        return _basis(heading, pitch, roll, tan_half_h, tan_half_v, height).ground_offset(u + du, v + dv)

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

    return ProjectionUncertainty(
        along_meters=math.sqrt(max(0.0, along_var) + terrain * terrain),
        across_meters=math.sqrt(max(0.0, across_var)),
        orientation_deg=normalize_degrees(math.degrees(math.atan2(offset.x, offset.y))),
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
    return Vec2((x / depth / basis.tan_half_h + 1) / 2, (1 - y / depth / basis.tan_half_v) / 2)


def camera_sees(pose: CameraPose, point: LatLon) -> bool:
    image = image_coordinates(pose, point)
    if image is None or not (0 <= image.x <= 1 and 0 <= image.y <= 1):
        return False
    return distance_meters(pose.position, point) <= pose.range_meters


# ---------------------------------------------------------------- polygons


def point_in_polygon(point: Vec2, ring: Sequence[Vec2]) -> bool:
    """Even-odd rule. A ring with fewer than three points contains nothing."""
    if len(ring) < 3:
        return False
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        a, b = ring[i], ring[j]
        if (a.y > point.y) != (b.y > point.y):
            x = (b.x - a.x) * (point.y - a.y) / (b.y - a.y) + a.x
            if point.x < x:
                inside = not inside
        j = i
    return inside


def point_in_ring(ring: Sequence[LatLon], point: LatLon) -> bool:
    if len(ring) < 3:
        return False
    frame = LocalFrame(ring[0])
    return point_in_polygon(frame.to_local(point), [frame.to_local(p) for p in ring])


def distance_to_ring_edge(ring: Sequence[LatLon], point: LatLon) -> float:
    """Metres from a point to the nearest edge of the ring."""
    if len(ring) < 2:
        return math.inf
    frame = LocalFrame(ring[0])
    p = frame.to_local(point)
    local = [frame.to_local(v) for v in ring]
    best = math.inf
    for i in range(len(local)):
        a, b = local[i], local[(i + 1) % len(local)]
        best = min(best, _segment_distance(p, a, b))
    return best


def _segment_distance(p: Vec2, a: Vec2, b: Vec2) -> float:
    abx, aby = b.x - a.x, b.y - a.y
    length_sq = abx * abx + aby * aby
    if length_sq == 0:
        return math.hypot(p.x - a.x, p.y - a.y)
    t = max(0.0, min(1.0, ((p.x - a.x) * abx + (p.y - a.y) * aby) / length_sq))
    return math.hypot(p.x - (a.x + t * abx), p.y - (a.y + t * aby))
