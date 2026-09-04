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
from functools import lru_cache
from typing import Sequence

from .core import (
    CameraPose,
    LatLon,
    bearing_degrees,
    destination_point,
    field_of_view,
    haversine_distance,
    project_to_ground,
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

#: Position-error contours worth drawing, tightest first, in metres of 1σ
#: horizontal error. Chosen against what a zone is *for*: half a metre
#: distinguishes one side of a doorway from the other, five metres does not
#: distinguish one side of a small yard from the other, and a zone whose own
#: half-width is smaller than the error over it cannot be adjudicated at all.
SIGMA_THRESHOLDS_M: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0)

#: How loosely a threshold may exceed the zone's half-width and still count.
#: A four-metre square built from geodesic destination points measures
#: 3.999990 m, so an exact ``<=`` dropped the two-metre band from a zone sized
#: for exactly that band — a wrong answer produced by five microns of rounding,
#: and invisible in every zone but the one whose width matches a threshold.
_WIDTH_TOLERANCE = 1e-4

#: Bisection depth when finding a contour in image space. Fourteen halvings of
#: the [0, 1] image axis resolve a row to one part in sixteen thousand, which on
#: any real sensor is far inside one pixel — and the pose that produced it is
#: known to nothing like that precision.
_BISECTION_STEPS = 14


class CoverageError(RuntimeError):
    """Coverage could not be computed from what was given."""


@dataclass(frozen=True, slots=True)
class SigmaBand:
    """The ground over which a camera's position error stays within a bound.

    ``ring`` is the region where 1σ horizontal error is at most
    ``threshold_m``. Bands nest: the half-metre band lies inside the one-metre
    band, both hugging the near edge of the footprint, because error grows with
    distance. What is in no band at all is not unseen — it is seen, and its
    position is known to worse than the loosest threshold.
    """

    threshold_m: float
    ring: tuple[LatLon, ...]


@dataclass(frozen=True, slots=True)
class ZoneReport:
    """What the placed cameras could rule on, for one zone.

    An **upper bound**, for the reasons in this module's docstring: nothing here
    models occlusion or resolution, so a wall or a shrub inside a footprint
    counts as covered. Both would make these numbers smaller, never larger.
    """

    area_m2: float
    #: Share of the zone inside at least one camera's footprint.
    covered_fraction: float
    #: The rest. Named separately because it is the number worth alarming on.
    outside_fraction: float
    #: Share of the zone where the position error is smaller than half the
    #: zone's own narrowest width — where a presence can actually be
    #: adjudicated rather than reported as UNCERTAIN.
    confident_fraction: float
    #: Which cameras contribute any of it.
    cameras: tuple[str, ...]
    #: Tightest band the zone touches at all, and the loosest band that
    #: contains the whole of it. ``None`` means "beyond the loosest threshold":
    #: for ``best`` the zone touches no band, for ``worst`` some part of it is
    #: outside every band.
    best_sigma_m: float | None
    worst_sigma_m: float | None


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


@lru_cache(maxsize=64)
def sigma_bands(
    pose: CameraPose, *, columns: int = 25, angular_uncertainty_deg: float = 1.5
) -> tuple[SigmaBand, ...]:
    """Where this camera's position error stays inside each threshold.

    Walks the image: for each of ``columns`` verticals, the first row that
    lands on the ground at all is found by bisection, then the row where the
    error crosses each threshold. Error falls monotonically down the image — a
    lower row is nearer the camera — which is what makes bisection valid and a
    scan unnecessary.

    Memoised on the pose, and that is not an optimisation detail: this costs
    on the order of two thousand calls across the FFI, which is nothing once
    per placement and ruinous once per repaint. `CameraPose` is a frozen
    dataclass, so it hashes; the cache is what lets a paint path ask for bands
    freely. Never call this from `paintEvent`.

    Returns bands tightest first. A camera pointed above the horizon returns
    none, which is the same thing `field_of_view` says about it.
    """
    footprint = field_of_view(pose, arc_segments=ARC_SEGMENTS)
    if len(footprint) < 3:
        return ()

    def project(u: float, v: float):
        return project_to_ground(pose, u, v, angular_uncertainty_deg)

    # Per column: the row where ground first appears, and the near row's error.
    horizons: dict[float, float] = {}
    for index in range(columns):
        u = index / (columns - 1)
        if project(u, 1.0) is None:
            # Not even the bottom of the frame reaches the ground within range.
            continue
        low, high = 0.0, 1.0  # low invalid, high valid
        if project(u, 0.0) is not None:
            low = 0.0
            high = 0.0
        else:
            for _ in range(_BISECTION_STEPS):
                middle = (low + high) / 2
                if project(u, middle) is None:
                    low = middle
                else:
                    high = middle
        horizons[u] = high

    if len(horizons) < 2:
        return ()

    bands: list[SigmaBand] = []
    for threshold in SIGMA_THRESHOLDS_M:
        far: list[LatLon] = []
        near: list[LatLon] = []
        for u, horizon in sorted(horizons.items()):
            nearest = project(u, 1.0)
            furthest = project(u, horizon)
            if nearest is None or furthest is None:
                continue
            if nearest.uncertainty_meters > threshold:
                # Even the closest ground this column sees is known worse than
                # this. The column contributes nothing to this band.
                continue
            if furthest.uncertainty_meters <= threshold:
                edge = furthest
            else:
                low, high = horizon, 1.0  # low worse than threshold, high better
                for _ in range(_BISECTION_STEPS):
                    middle = (low + high) / 2
                    projection = project(u, middle)
                    if projection is None or projection.uncertainty_meters > threshold:
                        low = middle
                    else:
                        high = middle
                edge = project(u, high)
                if edge is None:
                    continue
            far.append(edge.point)
            near.append(nearest.point)

        if len(far) < 2:
            continue

        # Out along the far contour, back along the near edge of the footprint.
        ring = _clip_to_footprint(tuple(far) + tuple(reversed(near)), footprint)
        if len(ring) >= 3:
            bands.append(SigmaBand(threshold_m=threshold, ring=ring))

    return tuple(bands)


def _clip_to_footprint(
    ring: Sequence[LatLon], footprint: Sequence[LatLon]
) -> tuple[LatLon, ...]:
    """Trim a band to the ground the camera actually reaches.

    The contour is built from projections that already honour the range, so
    this rarely changes anything — but the arc approximation and the contour
    are two different discretisations of the same edge, and a band poking a
    few centimetres outside the footprint it belongs to would be a claim about
    ground the same function says is not covered.
    """
    from shapely.geometry import Polygon

    frame = _Frame(ring[0])
    band = Polygon([frame.to_xy(point) for point in ring])
    if not band.is_valid:
        band = band.buffer(0)
    reach = Polygon([frame.to_xy(point) for point in footprint])
    if not reach.is_valid:
        reach = reach.buffer(0)

    clipped = band.intersection(reach)
    pieces = sorted(_polygons(clipped), key=lambda piece: piece.area, reverse=True)
    if not pieces:
        return ()
    return tuple(frame.to_latlon(x, y) for x, y in pieces[0].exterior.coords)


def _footprint_polygons(frame: "_Frame", cameras: dict[str, CameraPose]) -> tuple[dict, list[str]]:
    """Every camera's reachable ground, as polygons in one metric frame.

    One code path, because there were about to be two: `analyse` builds these
    to subtract them from a site, `zone_report` builds them to intersect them
    with a zone, and two unions that came to disagree would have one screen
    saying a zone was covered while another said the site had a hole in exactly
    that place. Returns the polygons and the cameras that see no ground at all.
    """
    from shapely.geometry import Polygon

    polygons: dict = {}
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
        polygons[camera_id] = polygon
    return polygons, blind


def zone_report(ring: Sequence[LatLon], cameras: dict[str, CameraPose]) -> ZoneReport:
    """What these cameras could rule on inside this zone.

    The number that matters is ``confident_fraction``: not "can a camera see
    this ground" but "if somebody stands there, can the system say which side
    of the line they are on". A zone drawn at 70 m from a mast is covered and
    unadjudicable at the same time, and until this existed nothing said so —
    the zone simply produced UNCERTAIN memberships that no rule acted on.

    The standard is the zone's own narrowest width: an error of two metres is
    fine for a car park and useless for a doorway, so the threshold is half the
    shorter side of the zone's minimum bounding rectangle. Raises
    `CoverageError` on a ring that is not an area.
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    if len(ring) < 3:
        raise CoverageError(
            f"a zone needs at least three points; {len(ring)} were given"
        )

    frame = _Frame(ring[0])
    zone = Polygon([frame.to_xy(point) for point in ring])
    if not zone.is_valid:
        # Repaired for *reporting* only. Nothing here is stored, and `Zone`
        # itself refuses a ring that needs this — but a half-drawn outline is
        # reported on while the operator is still dragging it.
        zone = zone.buffer(0)
    if zone.is_empty or zone.area <= 0:
        raise CoverageError("the zone encloses no area")

    area = float(zone.area)
    polygons, _ = _footprint_polygons(frame, cameras)

    seen: list[str] = []
    covering = []
    for camera_id, polygon in polygons.items():
        inside = polygon.intersection(zone)
        if inside.is_empty or inside.area <= 0:
            continue
        seen.append(camera_id)
        covering.append(polygon)

    reachable = unary_union(covering).intersection(zone) if covering else None
    covered = float(reachable.area) if reachable is not None else 0.0
    covered_fraction = covered / area

    # Half the shorter side of the tightest rectangle around the zone: the
    # error at which "inside or outside?" stops having an answer.
    rectangle = zone.minimum_rotated_rectangle
    corners = list(getattr(rectangle, "exterior", zone.exterior).coords)
    sides = [math.dist(corners[i], corners[i + 1]) for i in range(len(corners) - 1)]
    half_width = (min(sides) / 2.0) if sides else 0.0

    confident = 0.0
    best: float | None = None
    worst: float | None = None
    for threshold in SIGMA_THRESHOLDS_M:
        rings = [
            Polygon([frame.to_xy(point) for point in band.ring])
            for pose in cameras.values()
            for band in sigma_bands(pose)
            if band.threshold_m == threshold and len(band.ring) >= 3
        ]
        if not rings:
            continue
        within = unary_union(rings).intersection(zone)
        if within.is_empty:
            continue
        if best is None:
            best = threshold
        if threshold <= half_width * (1.0 + _WIDTH_TOLERANCE):
            confident = max(confident, float(within.area))
        # The loosest error bound that covers everything the cameras reach —
        # measured against the *reachable* part, not the whole zone. A zone
        # with one corner in the blind foreground under its own mast is
        # contained by no band at all, and reporting that as "beyond 5 m"
        # would blame the position error for a hole that `covered_fraction`
        # has already reported.
        if worst is None and reachable is not None and within.area >= covered * 0.999:
            worst = threshold

    return ZoneReport(
        area_m2=area,
        covered_fraction=covered_fraction,
        outside_fraction=1.0 - covered_fraction,
        confident_fraction=confident / area,
        cameras=tuple(sorted(seen)),
        best_sigma_m=best,
        worst_sigma_m=worst,
    )


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
    polygons, blind = _footprint_polygons(frame, cameras)

    for camera_id, polygon in polygons.items():
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
