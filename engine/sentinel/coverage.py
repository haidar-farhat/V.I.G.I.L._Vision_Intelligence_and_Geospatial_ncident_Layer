"""Which parts of a site are watched, and — the useful half — which are not.

Every camera already knows the ground it can see: `field_of_view` returns an
annular sector, correctly excluding the blind foreground under a tilted camera's
own mast. What nobody could ask until now is the question an installer actually
has: **given these cameras, what is left uncovered?**

That is a polygon union subtracted from a site boundary, and it is worth having
for a reason beyond tidiness. Coverage gaps are invisible on a plan view. Six
cameras drawn as six overlapping wedges look like thorough coverage, and the
four-metre corridor between two of them looks like nothing at all — until
somebody walks down it. A number and a shape are what make it arguable.

**What this measures, stated precisely.** A camera's footprint is a *geometric*
claim: the ground its optics can reach, given where it is and where it points.
It is not a claim about what it can usefully see. Nothing here models occlusion,
so a wall inside a footprint is covered as far as this is concerned; nor
resolution, so the far edge of a 90 m range counts the same as the near edge
even though a person there is a handful of pixels. Both would make coverage
*smaller*, never larger, so every number this produces is an **upper bound** —
the most that could be covered, not the least. An upper bound is the honest
direction for a security tool to err in only if it is labelled as one, which is
why every report here says so.

`shapely` does the geometry. Union and difference over polygons with holes,
degenerate rings and near-tangent edges is genuinely difficult, it is the kind
of code whose bugs are invisible until a specific arrangement of cameras
produces a wrong answer, and a maintained computational-geometry library is
better at it than anything written here would be. It is offline: no sockets, no
telemetry, three compiled files carrying no endpoint — checked with this
project's own binary audit before it was adopted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .core import (
    CameraPose,
    LatLon,
    bearing_degrees,
    destination_point,
    field_of_view,
    haversine_distance,
)
from .logs import get as _get_logger

_log = _get_logger(__name__)

#: Arc resolution when turning a camera's field of view into a polygon. Twenty
#: four segments puts the chord error under a tenth of a percent of the radius,
#: which is far below the uncertainty in the pose that produced it.
ARC_SEGMENTS = 24

#: Gaps below this are not reported. A sliver between two overlapping footprints
#: is an artefact of the arc approximation, not a place somebody can stand, and
#: reporting one teaches an operator to ignore the field that reports real ones.
MINIMUM_GAP_M2 = 4.0


class CoverageError(RuntimeError):
    """Coverage could not be computed from what was given."""


@dataclass(frozen=True, slots=True)
class Gap:
    """One uncovered region.

    `area_m2` is the polygon's own area, holes subtracted — not the area of
    `ring`. The distinction is the whole difference between a useful number and
    a wrong one: an uncovered region is very often the site boundary with a
    camera-shaped hole punched in it, so its *outline* is the whole site.
    Measuring the outline reported a 10,446 m² gap on a 14,400 m² site as
    14,400 m², which is both alarming and false.
    """

    #: The outer boundary of the gap.
    ring: tuple[LatLon, ...]
    #: Covered islands inside it. A renderer needs these to fill correctly, and
    #: without them a gap is drawn over ground a camera can see.
    holes: tuple[tuple[LatLon, ...], ...]
    area_m2: float


@dataclass(frozen=True, slots=True)
class Coverage:
    """What a set of cameras covers of a site, and what they miss."""

    site_area_m2: float
    covered_area_m2: float
    #: Uncovered regions, largest first.
    gaps: tuple[Gap, ...]
    #: Cameras that contributed nothing: they see no ground at all, or all of
    #: the ground they see is outside the site.
    blind_cameras: tuple[str, ...]

    @property
    def uncovered_area_m2(self) -> float:
        return max(0.0, self.site_area_m2 - self.covered_area_m2)

    @property
    def covered_fraction(self) -> float:
        return self.covered_area_m2 / self.site_area_m2 if self.site_area_m2 else 0.0

    @property
    def largest_gap_m2(self) -> float:
        return max((gap.area_m2 for gap in self.gaps), default=0.0)

    def describe(self) -> str:
        lines = [
            f"site            {self.site_area_m2:,.0f} m²",
            f"covered         {self.covered_area_m2:,.0f} m²  "
            f"({self.covered_fraction:.0%})",
            f"uncovered       {self.uncovered_area_m2:,.0f} m²  "
            f"in {len(self.gaps)} area(s)",
        ]
        if self.gaps:
            lines.append(f"largest gap     {self.largest_gap_m2:,.0f} m²")
        if self.blind_cameras:
            lines.append(
                f"contributing nothing   {', '.join(self.blind_cameras)}"
            )
        lines.append("")
        lines.append(
            "This is a geometric upper bound. Nothing occludes anything here, "
            "and range is counted at full weight to its far edge, so real "
            "coverage is smaller than this — never larger."
        )
        return "\n".join(lines)


class _Frame:
    """Metres east and north of a site origin, for planar geometry.

    Coverage is a local question — a site is hundreds of metres across, not
    hundreds of kilometres — so a local tangent plane is exact enough and lets
    a planar geometry library do the work. Doing the union in degrees would
    make every area wrong by the cosine of the latitude.

    Built from `haversine_distance`, `bearing_degrees` and `destination_point`
    rather than from metres-per-degree constants of its own. Those constants
    exist, in Rust, and are tested there; a second copy in Python would be a
    second answer, and the two would come to differ for reasons nobody would
    find.
    """

    __slots__ = ("origin",)

    def __init__(self, origin: LatLon):
        self.origin = origin

    def to_xy(self, point: LatLon) -> tuple[float, float]:
        distance = haversine_distance(self.origin, point)
        if distance == 0.0:
            return (0.0, 0.0)
        bearing = math.radians(bearing_degrees(self.origin, point))
        return (distance * math.sin(bearing), distance * math.cos(bearing))

    def to_latlon(self, x: float, y: float) -> LatLon:
        distance = math.hypot(x, y)
        if distance == 0.0:
            return self.origin
        return destination_point(
            self.origin, math.degrees(math.atan2(x, y)) % 360.0, distance
        )


def _ring_area(ring: Sequence[LatLon]) -> float:
    """Area of a lat/lon ring in square metres, by the shoelace formula."""
    if len(ring) < 3:
        return 0.0
    frame = _Frame(ring[0])
    points = [frame.to_xy(point) for point in ring]
    total = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def analyse(
    site: Sequence[LatLon],
    cameras: dict[str, CameraPose],
    *,
    minimum_gap_m2: float = MINIMUM_GAP_M2,
) -> Coverage:
    """What these cameras cover of this site, and where the holes are.

    ``site`` is the boundary being protected. ``cameras`` maps a camera id to
    its pose; an unplaced camera has none and simply cannot be included, which
    is itself worth knowing and is why the caller passes only placed ones.
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    if len(site) < 3:
        raise CoverageError(
            "a site boundary needs at least three points; "
            f"{len(site)} were given"
        )

    frame = _Frame(site[0])
    boundary = Polygon([frame.to_xy(point) for point in site])
    if not boundary.is_valid:
        # A self-intersecting boundary — a figure of eight, usually a typo in a
        # vertex — has no meaningful area. Repaired rather than refused, because
        # refusing gives an operator nothing to act on.
        boundary = boundary.buffer(0)
    if boundary.is_empty or boundary.area <= 0:
        raise CoverageError("the site boundary encloses no area")

    footprints = []
    blind: list[str] = []

    for camera_id, pose in sorted(cameras.items()):
        ring = field_of_view(pose, arc_segments=ARC_SEGMENTS)
        if len(ring) < 3:
            # A camera whose bottom-of-frame is above the horizon sees no ground
            # at all. It is not a broken camera; it is a camera pointed at the
            # sky, and the honest report says so rather than skipping it.
            blind.append(camera_id)
            continue

        polygon = Polygon([frame.to_xy(point) for point in ring])
        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        inside = polygon.intersection(boundary)
        if inside.is_empty or inside.area <= 0:
            # It sees ground, but none of it is on this site.
            blind.append(camera_id)
            continue
        footprints.append(inside)

    covered = unary_union(footprints) if footprints else None
    covered_area = float(covered.area) if covered is not None else 0.0

    uncovered = boundary if covered is None else boundary.difference(covered)
    gaps: list[Gap] = []

    for piece in _polygons(uncovered):
        # `piece.area` already has the holes subtracted, which is what makes it
        # the number worth reporting.
        if piece.area < minimum_gap_m2:
            continue
        gaps.append(
            Gap(
                ring=tuple(frame.to_latlon(x, y) for x, y in piece.exterior.coords),
                holes=tuple(
                    tuple(frame.to_latlon(x, y) for x, y in interior.coords)
                    for interior in piece.interiors
                ),
                area_m2=float(piece.area),
            )
        )

    gaps.sort(key=lambda gap: gap.area_m2, reverse=True)

    result = Coverage(
        site_area_m2=float(boundary.area),
        covered_area_m2=covered_area,
        gaps=tuple(gaps),
        blind_cameras=tuple(blind),
    )
    _log.info(
        "coverage: %.0f m² of %.0f m² (%.0f%%), %d gap(s), %d camera(s) contributing nothing",
        result.covered_area_m2, result.site_area_m2,
        result.covered_fraction * 100, len(result.gaps), len(blind),
    )
    return result


def _polygons(geometry):
    """Every polygon in a result that may be one, several, or none."""
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    if geometry.geom_type in ("MultiPolygon", "GeometryCollection"):
        return [part for part in geometry.geoms if part.geom_type == "Polygon"]
    return []
