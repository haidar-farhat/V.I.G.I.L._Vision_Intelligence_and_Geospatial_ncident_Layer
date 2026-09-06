"""Geodesy and camera geometry. Ported from v1's Rust core, measured there.

Everything is a pure function of its arguments. A position is never returned
without a statement of how it was obtained and how well it is known: a map
that implies precision it does not have sends somebody to the wrong place.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence

#: Below this depression angle a ray is treated as missing the ground. At 2°
#: a 6 m mast projects to 170 m with an uncertainty larger than the distance.
MIN_DEPRESSION_ANGLE_DEG = 2.0
#: The angular error assumed for a detection's contact point, 1-σ.
DEFAULT_ANGULAR_UNCERTAINTY_DEG = 0.75


@dataclass(frozen=True, slots=True)
class LatLon:
    lat: float
    lon: float


@dataclass(frozen=True, slots=True)
class Vec2:
    x: float
    y: float


def meters_per_degree_latitude(latitude_deg: float) -> float:
    lat = math.radians(latitude_deg)
    return 111132.92 - 559.82 * math.cos(2 * lat) + 1.175 * math.cos(4 * lat)


def meters_per_degree_longitude(latitude_deg: float) -> float:
    lat = math.radians(latitude_deg)
    return 111412.84 * math.cos(lat) - 93.5 * math.cos(3 * lat) + 0.118 * math.cos(5 * lat)


def haversine_distance(a: LatLon, b: LatLon) -> float:
    """Metres between two points on the WGS84 sphere approximation."""
    radius = 6371008.8
    phi1, phi2 = math.radians(a.lat), math.radians(b.lat)
    dphi = phi2 - phi1
    dlambda = math.radians(b.lon - a.lon)
    h = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(min(1.0, math.sqrt(h)))


def bearing_degrees(a: LatLon, b: LatLon) -> float:
    phi1, phi2 = math.radians(a.lat), math.radians(b.lat)
    dlambda = math.radians(b.lon - a.lon)
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return normalize_degrees(math.degrees(math.atan2(y, x)))


def destination_point(origin: LatLon, bearing_deg: float, distance_meters: float) -> LatLon:
    radius = 6371008.8
    delta = distance_meters / radius
    theta = math.radians(bearing_deg)
    phi1, lambda1 = math.radians(origin.lat), math.radians(origin.lon)
    phi2 = math.asin(math.sin(phi1) * math.cos(delta) + math.cos(phi1) * math.sin(delta) * math.cos(theta))
    lambda2 = lambda1 + math.atan2(
        math.sin(theta) * math.sin(delta) * math.cos(phi1),
        math.cos(delta) - math.sin(phi1) * math.sin(phi2),
    )
    return LatLon(math.degrees(phi2), normalize_longitude(math.degrees(lambda2)))


def normalize_degrees(deg: float) -> float:
    return deg % 360.0


def normalize_longitude(deg: float) -> float:
    return ((deg + 180.0) % 360.0) - 180.0


def angle_difference(a: float, b: float) -> float:
    """Signed smallest difference a − b in (−180, 180]."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d


class LocalFrame:
    """East/north metres around an origin. Sub-centimetre across a site."""

    def __init__(self, origin: LatLon):
        self.origin = origin
        self._m_lat = meters_per_degree_latitude(origin.lat)
        self._m_lon = meters_per_degree_longitude(origin.lat)

    def to_local(self, point: LatLon) -> Vec2:
        return Vec2((point.lon - self.origin.lon) * self._m_lon, (point.lat - self.origin.lat) * self._m_lat)

    def to_lat_lon(self, local: Vec2) -> LatLon:
        return LatLon(self.origin.lat + local.y / self._m_lat, self.origin.lon + local.x / self._m_lon)


# ------------------------------------------------------------------ camera


@dataclass(frozen=True, slots=True)
class CameraPose:
    """Where a camera is and where it looks. Angles in degrees, heights in metres."""

    position: LatLon
    mount_height: float
    heading: float
    pitch: float
    roll: float = 0.0
    horizontal_fov: float = 62.0
    vertical_fov: float = 36.0
    #: Beyond this ground distance a projection is refused, not clamped.
    range_meters: float = 60.0

    def validate(self) -> None:
        if not (-90 <= self.position.lat <= 90 and -180 <= self.position.lon <= 180):
            raise ValueError("position is off the planet")
        if self.mount_height <= 0:
            raise ValueError("mount height must be above the ground")
        if not (1 <= self.horizontal_fov < 180 and 1 <= self.vertical_fov < 180):
            raise ValueError("fields of view must be between 1° and 180°")
        if self.range_meters <= 0:
            raise ValueError("range must be positive")


class PositionSource(StrEnum):
    GROUND_PROJECTION = "GROUND_PROJECTION"
    CAMERA_FALLBACK = "CAMERA_FALLBACK"


@dataclass(frozen=True, slots=True)
class PositionEstimate:
    point: LatLon
    #: 1-σ horizontal uncertainty, metres.
    radius_meters: float
    source: PositionSource

    @property
    def is_projected(self) -> bool:
        return self.source is PositionSource.GROUND_PROJECTION


@dataclass(frozen=True, slots=True)
class GroundProjection:
    position: LatLon
    ground_distance_meters: float
    bearing_deg: float
    uncertainty_meters: float


def ray_angles(pose: CameraPose, u: float, v: float) -> tuple[float, float]:
    """(bearing, elevation) of the ray through a normalised image point.

    Rectilinear model: a lens maps angle to sensor position through tan, not
    linearly, and the difference at the frame edge of a 90° lens is 20 m at
    40 m.
    """
    half_h = math.radians(pose.horizontal_fov / 2)
    half_v = math.radians(pose.vertical_fov / 2)
    dx = min(1.0, max(0.0, u)) * 2 - 1
    dy = 1 - min(1.0, max(0.0, v)) * 2
    yaw = math.atan(dx * math.tan(half_h))
    pitch_offset = math.atan(dy * math.tan(half_v))
    return normalize_degrees(pose.heading + math.degrees(yaw)), pose.pitch + math.degrees(pitch_offset)


def project_to_ground(
    pose: CameraPose, u: float, v: float,
    angular_uncertainty_deg: float = DEFAULT_ANGULAR_UNCERTAINTY_DEG, *, enforce_range: bool = True,
) -> GroundProjection | None:
    """Where a normalised image point meets the ground, or ``None``.

    ``None`` when the ray points at or above the horizon or beyond the useful
    range. Nothing rather than a clamped guess: a position the system cannot
    determine must not appear on a map.
    """
    bearing, elevation = ray_angles(pose, u, v)
    depression_deg = -elevation
    if depression_deg < MIN_DEPRESSION_ANGLE_DEG:
        return None
    depression = math.radians(depression_deg)
    distance = pose.mount_height / math.tan(depression)
    if not math.isfinite(distance) or distance <= 0:
        return None
    if enforce_range and distance > pose.range_meters:
        return None
    sigma = math.radians(angular_uncertainty_deg)
    sin_d = math.sin(depression)
    range_sigma = pose.mount_height * sigma / (sin_d * sin_d)
    lateral_sigma = distance * sigma
    return GroundProjection(
        position=destination_point(pose.position, bearing, distance),
        ground_distance_meters=distance,
        bearing_deg=bearing,
        uncertainty_meters=math.hypot(range_sigma, lateral_sigma),
    )


def project_point(pose: CameraPose, contact: Vec2) -> PositionEstimate:
    """A contact point to a map position, degrading honestly to the camera."""
    projection = project_to_ground(pose, contact.x, contact.y)
    if projection is None:
        return PositionEstimate(pose.position, pose.range_meters, PositionSource.CAMERA_FALLBACK)
    return PositionEstimate(projection.position, projection.uncertainty_meters, PositionSource.GROUND_PROJECTION)


def near_ground_distance(pose: CameraPose) -> float | None:
    projection = project_to_ground(pose, 0.5, 1.0, enforce_range=False)
    return projection.ground_distance_meters if projection else None


def far_ground_distance(pose: CameraPose) -> float | None:
    projection = project_to_ground(pose, 0.5, 0.0, enforce_range=False)
    return projection.ground_distance_meters if projection else None


def field_of_view(pose: CameraPose, arc_segments: int = 24) -> list[LatLon]:
    """The ground footprint: an annular sector, never a pie slice.

    A downward-tilted camera does not see the ground at its own feet; drawing
    the slice tells an operator the camera covers ground it is blind to.
    Empty when the camera sees no ground at all.
    """
    segments = max(2, arc_segments)
    near = near_ground_distance(pose)
    if near is None:
        return []
    far = far_ground_distance(pose)
    far_range = min(far, pose.range_meters) if far is not None else pose.range_meters
    near_range = min(near, far_range)
    half = pose.horizontal_fov / 2

    def bearing_at(t: float) -> float:
        return normalize_degrees(pose.heading - half + t * pose.horizontal_fov)

    points = [destination_point(pose.position, bearing_at(i / segments), far_range) for i in range(segments + 1)]
    if near_range > 0.5:
        points.extend(
            destination_point(pose.position, bearing_at(i / segments), near_range)
            for i in range(segments, -1, -1)
        )
    else:
        points.append(pose.position)
    return points


def image_coordinates(pose: CameraPose, point: LatLon) -> Vec2 | None:
    """Where a ground point appears in the image; the inverse of the projection.

    Unclipped: a point outside 0..1 is meaningful. ``None`` when the point is
    behind the camera or at the mast.
    """
    distance = haversine_distance(pose.position, point)
    if distance < 1e-6:
        return None
    bearing = bearing_degrees(pose.position, point)
    yaw = angle_difference(bearing, pose.heading)
    if abs(yaw) >= 90:
        return None
    elevation = -math.degrees(math.atan2(pose.mount_height, distance))
    pitch_offset = elevation - pose.pitch
    half_h = math.tan(math.radians(pose.horizontal_fov / 2))
    half_v = math.tan(math.radians(pose.vertical_fov / 2))
    dx = math.tan(math.radians(yaw)) / half_h
    dy = math.tan(math.radians(pitch_offset)) / half_v
    return Vec2((dx + 1) / 2, (1 - dy) / 2)


def camera_sees(pose: CameraPose, point: LatLon) -> bool:
    image = image_coordinates(pose, point)
    if image is None or not (0 <= image.x <= 1 and 0 <= image.y <= 1):
        return False
    return haversine_distance(pose.position, point) <= pose.range_meters


# ------------------------------------------------------------------ polygons


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
