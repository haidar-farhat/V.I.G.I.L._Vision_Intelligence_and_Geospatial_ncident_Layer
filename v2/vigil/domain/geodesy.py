"""Latitude, longitude, metres between them, and rings drawn on the ground.

Split out of `geo` when that module outgrew its line budget, and split along
the same seam the Rust core already uses: `core/src/geodesy.rs` holds exactly
these functions and `core/src/camera.rs` holds the camera model. Two languages
with the same shape are easier to hold in one head than two languages with two
shapes.

# The defect this replaced

v1 and v2 both used two different Earths at once. Distances between tracks
went through a haversine on a sphere of radius 6 371 008.8 m; zone tests and
the ground grid went through a tangent plane from the WGS84 latitude series.
Those disagree by **0.248%** at this product's reference latitude — 25 cm per
100 m, 1.24 m across a 500 m site. It is not a rounding difference, it is two
models: the sphere's 111 195 metres per degree of latitude is a global
average and the real meridian at 34 degrees is 110 920.

Everything is the tangent plane now, because that is the model the rest of
the system already had to use — there is no spherical version of "is this
point in this ring". `distance_meters`, `bearing_degrees` and
`destination_point` are therefore exact inverses of each other rather than
approximately consistent. `spherical_distance` is kept, called by nothing, and
tested as the reference that says how far apart the two models are.

`geo` re-exports every name here, so `from .geo import LatLon` keeps working
and no caller had to change when this file appeared.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

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



# ---------------------------------------------------------------- polygons


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
