"""Which parts of a site are watched, and — the useful half — which are not.

Migrated from v1's `coverage.py`, which v2 dropped. Every camera already knows
the ground it can see; what nobody could ask until this existed is the
question an installer actually has: **given these cameras, what is left
uncovered?**

That is a polygon union subtracted from a site boundary, and it is worth
having for a reason beyond tidiness. Coverage gaps are invisible on a plan
view. Six cameras drawn as six overlapping wedges look like thorough
coverage, and the four-metre corridor between two of them looks like nothing
at all — until somebody walks down it. A number and a shape are what make it
arguable.

# What this measures, stated precisely

A camera's footprint is a *geometric* claim: the ground its optics reach,
given where it is and where it points. It is not a claim about what it can
usefully see. Nothing here models occlusion, so a wall inside a footprint is
covered as far as this is concerned; nor resolution, so the far edge of a 60 m
range counts the same as the near edge even though a person there is a handful
of pixels.

Both of those would make coverage *smaller*, never larger, so every number
here is an **upper bound** — the most that could be covered, not the least.
An upper bound is the honest direction for a security tool to err in only if
it is labelled as one, which is why every report says so.

# What is new since v1

v1's footprints came from the decoupled camera model, so its coverage numbers
inherited that model's error — degrees at the corners of a tilted frame, which
is metres of ground. These footprints come from the pinhole model and are
traced around the frame's border, so a rolled camera produces the rotated
footprint it actually has.

And coverage is now reported in **bands of position error** as well as in
area. "Covered" and "covered well enough to say which side of a line somebody
was on" are different questions, and only the second one decides whether a
zone can be adjudicated at all.

`shapely` does the geometry. Union and difference over polygons with holes,
degenerate rings and near-tangent edges is genuinely difficult, its bugs are
invisible until one specific arrangement of cameras produces a wrong answer,
and a maintained computational-geometry library is better at it than anything
written here. It is offline: no sockets, no telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..domain.geo import (
    CameraPose, LatLon, LocalFrame, Vec2, distance_meters, field_of_view, project_to_ground,
)
from ..logs import get as _get_logger

_log = _get_logger(__name__)

#: Arc resolution when turning a field of view into a polygon. Twenty-four
#: segments puts the chord error under a tenth of a percent of the radius,
#: far below the uncertainty in the pose that produced it.
ARC_SEGMENTS = 24

#: Gaps below this are not reported. A sliver between two overlapping
#: footprints is an artefact of the arc approximation, not a place somebody
#: can stand, and reporting one teaches an operator to ignore the field that
#: reports real ones.
MINIMUM_GAP_M2 = 4.0

#: Position-error contours worth drawing, tightest first, in metres of 1-sigma
#: horizontal error. Chosen against what a zone is *for*: half a metre
#: distinguishes one side of a doorway from the other, five metres does not
#: distinguish one side of a small yard from the other, and a zone whose own
#: half-width is smaller than the error over it cannot be adjudicated at all.
SIGMA_BANDS_M: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0)


class CoverageError(RuntimeError):
    pass


def _shapely():
    try:
        from shapely.geometry import Polygon  # noqa: F401
        from shapely.ops import unary_union  # noqa: F401
    except ImportError as error:  # pragma: no cover - stated, not hidden
        raise CoverageError(
            "coverage analysis needs shapely, which is not installed. Every other part of the "
            "product works without it."
        ) from error
    import shapely.geometry as geometry
    import shapely.ops as ops

    return geometry, ops


@dataclass(frozen=True, slots=True)
class Gap:
    """A patch of ground inside the boundary that no camera reaches."""

    area_m2: float
    ring: tuple[LatLon, ...]
    #: Longest straight line that fits inside it: how wide the corridor is.
    span_m: float

    def describe(self) -> str:
        return f"{self.area_m2:.0f} m² gap, {self.span_m:.0f} m across"


@dataclass(frozen=True, slots=True)
class Band:
    """How much ground is covered to at least this position accuracy."""

    sigma_m: float
    area_m2: float

    def describe(self) -> str:
        return f"±{self.sigma_m:.1f} m or better: {self.area_m2:.0f} m²"


@dataclass(frozen=True, slots=True)
class Coverage:
    boundary_m2: float
    covered_m2: float
    gaps: tuple[Gap, ...]
    bands: tuple[Band, ...]
    cameras: tuple[str, ...]
    #: Cameras that were skipped, and why.
    skipped: tuple[tuple[str, str], ...] = ()

    @property
    def fraction(self) -> float:
        return self.covered_m2 / self.boundary_m2 if self.boundary_m2 > 0 else 0.0

    def describe(self) -> str:
        lines = [
            f"{self.covered_m2:.0f} m² of {self.boundary_m2:.0f} m² reachable by "
            f"{len(self.cameras)} camera(s) — {self.fraction:.0%}",
            "  (an upper bound: nothing here models occlusion or resolution, so the real "
            "figure is lower)",
        ]
        for band in self.bands:
            lines.append("  " + band.describe())
        if self.gaps:
            lines.append(f"  {len(self.gaps)} gap(s) over {MINIMUM_GAP_M2:.0f} m²:")
            for gap in self.gaps[:5]:
                lines.append("    " + gap.describe())
            if len(self.gaps) > 5:
                lines.append(f"    … and {len(self.gaps) - 5} more")
        else:
            lines.append("  no gap over " f"{MINIMUM_GAP_M2:.0f} m²")
        for name, why in self.skipped:
            lines.append(f"  {name}: not counted — {why}")
        return "\n".join(lines)


def _polygon(frame: LocalFrame, ring: Sequence[LatLon], geometry):
    points = [frame.to_local(p) for p in ring]
    if len(points) < 3:
        return None
    shape = geometry.Polygon([(p.x, p.y) for p in points])
    if not shape.is_valid:
        # A self-intersecting ring — a footprint whose near arc crossed its
        # far one at a steep tilt. `buffer(0)` is shapely's documented repair;
        # an invalid polygon poisons every union it takes part in.
        shape = shape.buffer(0)
    return shape if not shape.is_empty else None


def _ring(frame: LocalFrame, shape) -> tuple[LatLon, ...]:
    return tuple(frame.to_lat_lon(Vec2(x, y)) for x, y in shape.exterior.coords)


def _sigma_footprint(pose: CameraPose, sigma_m: float, frame, geometry):
    """The part of a camera's footprint placed to within `sigma_m`.

    Traced by walking the frame's border and keeping the rows whose projection
    is accurate enough — which is a band, because error grows with range, so
    "accurate to half a metre" is the near part of the footprint and nothing
    else.
    """
    rows: list[float] = []
    for i in range(33):
        v = 1.0 - i / 32.0
        projection = project_to_ground(pose, 0.5, v, enforce_range=True)
        if projection is None or projection.uncertainty.radius_meters > sigma_m:
            continue
        rows.append(v)
    if len(rows) < 2:
        return None
    near, far = max(rows), min(rows)
    ring: list[LatLon] = []
    for v, order in ((far, range(ARC_SEGMENTS + 1)), (near, range(ARC_SEGMENTS, -1, -1))):
        for i in order:
            projection = project_to_ground(pose, i / ARC_SEGMENTS, v, enforce_range=True)
            if projection is not None:
                ring.append(projection.position)
    return _polygon(frame, ring, geometry) if len(ring) >= 3 else None


def analyse(boundary: Sequence[LatLon], cameras: dict[str, CameraPose]) -> Coverage:
    """What these cameras reach inside this boundary, and what they miss."""
    geometry, ops = _shapely()
    if len(boundary) < 3:
        raise CoverageError("a site boundary is at least three points")
    frame = LocalFrame(boundary[0])
    site = _polygon(frame, boundary, geometry)
    if site is None or site.area <= 0:
        raise CoverageError("that boundary encloses no ground")

    shapes = []
    counted: list[str] = []
    skipped: list[tuple[str, str]] = []
    for name in sorted(cameras):
        pose = cameras[name]
        ring = field_of_view(pose, ARC_SEGMENTS)
        if not ring:
            skipped.append((name, "it is aimed at or above the horizon, so it sees no ground"))
            continue
        shape = _polygon(frame, ring, geometry)
        if shape is None:
            skipped.append((name, "its footprint is degenerate"))
            continue
        clipped = shape.intersection(site)
        if clipped.is_empty:
            skipped.append((name, "its footprint falls entirely outside the boundary"))
            continue
        shapes.append(clipped)
        counted.append(name)

    covered = ops.unary_union(shapes) if shapes else geometry.Polygon()
    uncovered = site.difference(covered)

    gaps: list[Gap] = []
    for piece in _pieces(uncovered):
        if piece.area < MINIMUM_GAP_M2:
            continue
        ring = _ring(frame, piece)
        span = 0.0
        coords = list(piece.exterior.coords)
        # The longest chord: an O(n^2) over a simplified ring, because a
        # 4 m corridor and a 4 m square are the same area and completely
        # different problems.
        simple = list(piece.simplify(0.5).exterior.coords) or coords
        for i, (ax, ay) in enumerate(simple):
            for bx, by in simple[i + 1:]:
                span = max(span, ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)
        gaps.append(Gap(piece.area, ring, span))
    gaps.sort(key=lambda g: -g.area_m2)

    bands: list[Band] = []
    for sigma in SIGMA_BANDS_M:
        pieces = []
        for name in counted:
            shape = _sigma_footprint(cameras[name], sigma, frame, geometry)
            if shape is not None:
                pieces.append(shape.intersection(site))
        area = ops.unary_union(pieces).area if pieces else 0.0
        bands.append(Band(sigma, area))

    return Coverage(site.area, covered.area, tuple(gaps), tuple(bands), tuple(counted),
                    tuple(skipped))


def _pieces(shape):
    if shape.is_empty:
        return []
    if shape.geom_type == "Polygon":
        return [shape]
    return [p for p in shape.geoms if p.geom_type == "Polygon"]


def boundary_from_cameras(cameras: dict[str, CameraPose], margin_m: float = 10.0) -> list[LatLon]:
    """A rectangle around every camera's footprint, for a site with no boundary.

    A stand-in and labelled as one: a real boundary is a fence somebody
    surveyed, and coverage measured against a rectangle drawn around the
    cameras will always look better than coverage measured against the ground
    that actually has to be watched — the rectangle is *defined* by where the
    cameras point.
    """
    if not cameras:
        raise CoverageError("no camera to build a boundary around")
    first = next(iter(sorted(cameras)))
    frame = LocalFrame(cameras[first].position)
    xs: list[float] = []
    ys: list[float] = []
    for pose in cameras.values():
        for point in field_of_view(pose, 12) or [pose.position]:
            local = frame.to_local(point)
            xs.append(local.x)
            ys.append(local.y)
    west, east = min(xs) - margin_m, max(xs) + margin_m
    south, north = min(ys) - margin_m, max(ys) + margin_m
    return [frame.to_lat_lon(Vec2(x, y))
            for x, y in ((west, south), (east, south), (east, north), (west, north))]


def unwatched_share(boundary: Sequence[LatLon], cameras: dict[str, CameraPose]) -> float:
    """Fraction of the boundary no camera reaches. For a one-line check."""
    return 1.0 - analyse(boundary, cameras).fraction


__all__ = [
    "ARC_SEGMENTS", "MINIMUM_GAP_M2", "SIGMA_BANDS_M", "Band", "Coverage", "CoverageError", "Gap",
    "analyse", "boundary_from_cameras", "unwatched_share",
]
