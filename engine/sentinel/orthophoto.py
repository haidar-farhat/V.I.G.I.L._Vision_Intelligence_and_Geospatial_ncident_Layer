"""The basemap the cameras draw themselves.

A plan view needs ground under it. Every other way of getting that ground is
unavailable here: a tile server is a network call, an aerial photograph is a
purchase and a download, and a scanned site plan is a drawing of the site as
somebody once intended it rather than as it is. So the ground is taken from the
only imagery this system is guaranteed to have — its own cameras — by inverting
the projection that already turns an image point into a map position.

**Only the ground plane is right, and that is not a caveat to bury.** Every
pixel is placed where it would be *if it lay on the ground*. A wall, a parked
van, a person: each has height, so each smears radially away from the camera
that saw it, out along the ray. Anyone reading a single-frame patch as a
photograph will read those smears as ground markings. That is the failure
:class:`MedianAccumulator` exists to prevent, and it is why a single patch is
called a sample and not a map.

**Empty is a value.** A cell no camera has seen stays empty, and nothing here
will interpolate one from its neighbours. Under a security overlay — where an
operator will judge "inside the fence" against what is drawn — inventing ground
nobody has looked at is the one thing this must not do. Because black is a
perfectly ordinary colour for asphalt at dusk, emptiness is carried in a
separate ``valid`` mask and never inferred from the pixel value; a renderer that
tests the colour will paint tarmac as a hole and a hole as tarmac.

**How the inverse is done, stated plainly.** :func:`~sentinel.core.image_coordinates`
is the exact inverse and would be the obvious tool, but it is one FFI call per
ground cell — measured here at 2.7 µs, so roughly 0.2 s for a 200×200 grid once
the per-cell :class:`~sentinel.core.LatLon` is built too, per camera per frame,
against a temporal median that wants hundreds of frames. So this module goes the
other way: it projects a coarse lattice of image points onto the ground with
:func:`~sentinel.core.project_to_ground` (1,089 calls at the default lattice,
about 3 ms), which gives an irregular ground mesh whose corners carry known
image coordinates, and then inverts that mesh by barycentric interpolation
across each of its triangles in numpy. The inverse is therefore exact at the
lattice nodes and piecewise-linear between them. A ground cell is only filled if
it falls inside a triangle *all* of whose corners projected — so the ragged edge
at the horizon and at maximum range falls inward, leaving cells empty rather
than claiming ground the pose does not reach.

The tangent-plane conversion is :class:`coverage._Frame`, imported rather than
rewritten. A second metres-from-a-lat-lon of this module's own would eventually
disagree with the one the coverage report uses, and then one screen would draw a
zone over ground another screen said was somewhere else.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping, Sequence

import numpy as np

from .core import CameraPose, LatLon, project_to_ground
from .coverage import _Frame
from .logs import get as _get_logger

_log = _get_logger(__name__)

#: Points per axis in the image lattice whose forward projection is inverted.
#: Thirty-three squared is 1,089 projections — about 3 ms — and the residual
#: against the exact inverse is well under a tenth of a pixel over the near
#: two-thirds of a frame, where a mast-mounted camera's ground resolution is
#: worth having at all. Raising it costs projections quadratically and buys
#: accuracy only near the horizon, where one image row already spans metres.
DEFAULT_LATTICE = 33

#: How many observations a cell needs before the median is reported at all.
#: Two samples make a mean, not a median, and one sample is simply whatever
#: happened to be standing there — which is exactly the thing the median is for
#: removing. Below this the cell stays empty.
MINIMUM_SAMPLES = 3

#: Default depth of the per-cell sample ring. See :class:`MedianAccumulator` for
#: the memory this bounds and why fifteen is enough to outvote a walker.
DEFAULT_CAPACITY = 15

#: Barycentric slack when rasterising a triangle. A cell centre landing exactly
#: on the shared edge of two lattice triangles must belong to one of them; with
#: an exact test, floating point drops a thin diagonal of cells that both
#: triangles believe belongs to the other, and the patch comes out pinstriped.
_BARYCENTRIC_EPSILON = 1e-9


class OrthophotoError(RuntimeError):
    """A patch or mosaic could not be built from what was given."""


@lru_cache(maxsize=32)
def _frame_for(origin: LatLon) -> _Frame:
    """The metric frame for a grid origin, built once.

    ``_Frame`` construction is free but its use is not — every conversion is two
    or three FFI calls — and grids are converted repeatedly from paint paths.
    Cached on the origin, which is hashable because ``LatLon`` is frozen.
    """
    return _Frame(origin)


@dataclass(frozen=True, slots=True)
class GroundGrid:
    """A metric raster over the site: where each cell is, in metres and degrees.

    ``origin`` is the **north-west corner**, and ``row`` increases southward, so
    the arrays this addresses are already north-up and can be handed to an image
    renderer without a flip. Getting that convention wrong mirrors the site
    about its own east-west axis, which is not obvious on a symmetrical yard and
    is catastrophic on any other.

    Cells are square and metric, not angular: a raster in degrees is a raster
    whose cells are a third narrower than they are tall at this latitude, and
    every area measured on it would be wrong by the cosine of the latitude.
    """

    origin: LatLon
    cell_size_m: float
    #: Cells east of the origin.
    columns: int
    #: Cells south of the origin.
    rows: int

    def __post_init__(self) -> None:
        if self.cell_size_m <= 0:
            raise OrthophotoError(
                f"a grid cell must have a positive size; {self.cell_size_m} m was given"
            )
        if self.columns < 1 or self.rows < 1:
            raise OrthophotoError(
                "a grid needs at least one cell in each direction; "
                f"{self.columns} by {self.rows} was given"
            )

    @property
    def shape(self) -> tuple[int, int]:
        """``(rows, columns)`` — numpy order, so it can size an array directly."""
        return (self.rows, self.columns)

    @property
    def cell_count(self) -> int:
        return self.rows * self.columns

    @property
    def width_m(self) -> float:
        return self.columns * self.cell_size_m

    @property
    def height_m(self) -> float:
        return self.rows * self.cell_size_m

    def contains(self, row: int, column: int) -> bool:
        return 0 <= row < self.rows and 0 <= column < self.columns

    def cell_xy(self, row: int, column: int) -> tuple[float, float]:
        """Centre of a cell, in metres east and north of the origin."""
        return (
            (column + 0.5) * self.cell_size_m,
            -(row + 0.5) * self.cell_size_m,
        )

    def cell_centre(self, row: int, column: int) -> LatLon:
        """Where a cell's centre is on the map.

        The centre, never a corner. A cell's colour is a claim about the ground
        it covers, and reporting the corner shifts every such claim half a cell
        north-west — half a metre on a metre grid, which is the difference
        between one side of a doorway and the other.
        """
        if not self.contains(row, column):
            raise OrthophotoError(
                f"cell ({row}, {column}) is outside a "
                f"{self.rows} by {self.columns} grid"
            )
        x, y = self.cell_xy(row, column)
        return _frame_for(self.origin).to_latlon(x, y)

    def cell_of(self, point: LatLon) -> tuple[int, int] | None:
        """Which cell a map position falls in, or ``None`` if it is off the grid.

        ``None`` rather than a clamped edge cell: a position off the raster is
        not a position at its border, and clamping would pile everything beyond
        the site onto its boundary row.
        """
        x, y = _frame_for(self.origin).to_xy(point)
        column = math.floor(x / self.cell_size_m)
        row = math.floor(-y / self.cell_size_m)
        if not self.contains(row, column):
            return None
        return (row, column)

    def centres_xy(self) -> tuple[np.ndarray, np.ndarray]:
        """Every cell centre in metres, as ``(x, y)`` arrays shaped like the grid.

        Metres, not degrees, and computed by arithmetic rather than by a
        geodesic call per cell — 40,000 cells would otherwise be 40,000 trips
        across the FFI for a frame that has to be sampled at video rate.
        """
        columns = (np.arange(self.columns, dtype=np.float64) + 0.5) * self.cell_size_m
        rows = -(np.arange(self.rows, dtype=np.float64) + 0.5) * self.cell_size_m
        return np.meshgrid(columns, rows)

    @classmethod
    def covering(
        cls,
        points: Sequence[LatLon],
        *,
        cell_size_m: float,
        margin_m: float = 0.0,
    ) -> "GroundGrid":
        """The smallest grid holding every one of these points, plus a margin.

        Built from the footprints the cameras actually reach, so a site's raster
        is sized by what can be seen rather than by a bounding box somebody
        typed. ``margin_m`` widens it; a zero margin puts the extreme points on
        the outermost cells, where a half-cell of rounding can push one off.
        """
        if not points:
            raise OrthophotoError("a grid needs at least one point to cover")
        frame = _frame_for(points[0])
        xs, ys = zip(*(frame.to_xy(point) for point in points))
        west = min(xs) - margin_m
        east = max(xs) + margin_m
        south = min(ys) - margin_m
        north = max(ys) + margin_m
        columns = max(1, math.ceil((east - west) / cell_size_m))
        rows = max(1, math.ceil((north - south) / cell_size_m))
        return cls(
            origin=frame.to_latlon(west, north),
            cell_size_m=cell_size_m,
            columns=columns,
            rows=rows,
        )


@dataclass(frozen=True, slots=True, eq=False)
class GroundPatch:
    """One camera's view of the ground, resampled top-down onto a grid.

    ``valid`` is the authority on which cells mean anything. ``colour`` is zero
    where a cell is empty, and zero is also what a black car reads as, so
    nothing may infer emptiness from the pixels.

    ``sigma_m`` is the 1σ horizontal position error of the ground point each
    cell was sampled from — the same number
    :func:`~sentinel.core.project_to_ground` attaches to any position — and it
    travels with the colour because it is what makes one camera's version of a
    cell better than another's. It grows toward the horizon, so the far half of
    a patch is both blurrier and less certainly *where* it says it is.

    It is finite on every valid cell and NaN on every invalid one, with nothing
    in between: a cell this patch offers as ground whose error nobody measured
    is refused where it enters :class:`MedianAccumulator` or :func:`mosaic`,
    because there is no number to put in its place that is not a lie. Zero would
    outrank a surveyed camera and NaN reads as "no ground here" to whatever
    draws the result.

    Equality is off: two patches are megabytes of pixels, and a dataclass
    ``__eq__`` over numpy arrays raises rather than answers.
    """

    camera_id: str
    grid: GroundGrid
    #: ``(rows, columns, channels)``, the image's own dtype.
    colour: np.ndarray
    #: ``(rows, columns)`` bool. The only honest test for "is there ground here".
    valid: np.ndarray
    #: ``(rows, columns)`` float32, NaN where invalid.
    sigma_m: np.ndarray
    #: ``(rows, columns)`` float64 unix seconds, NaN where never observed.
    updated_at: np.ndarray
    #: ``(rows, columns)`` uint32 — how many frames stand behind each cell. One
    #: for a single sample; the count that survived retention for a median.
    samples: np.ndarray

    @property
    def channels(self) -> int:
        return int(self.colour.shape[2])

    @property
    def covered_cells(self) -> int:
        return int(np.count_nonzero(self.valid))

    @property
    def covered_fraction(self) -> float:
        return self.covered_cells / self.grid.cell_count

    def age_seconds(self, now: float | None = None) -> np.ndarray:
        """Per-cell staleness, NaN where the cell was never observed.

        NaN and not a large number: "never seen" and "seen an hour ago" are
        different facts, and a renderer that fades by age must draw the first as
        absent rather than as very old.
        """
        moment = time.time() if now is None else now
        return moment - self.updated_at


@dataclass(frozen=True, slots=True, eq=False)
class Mosaic:
    """Every camera's ground, composited — each cell from whoever knows it best.

    ``source`` names which camera won each cell as an index into ``cameras``,
    and is ``-1`` where nobody covered it. That index is worth keeping: when an
    operator disputes what the basemap shows under a fence line, the answer is a
    specific camera and a specific pose, not "the mosaic".
    """

    grid: GroundGrid
    colour: np.ndarray
    valid: np.ndarray
    sigma_m: np.ndarray
    updated_at: np.ndarray
    #: ``(rows, columns)`` int32 index into :attr:`cameras`; ``-1`` where empty.
    source: np.ndarray
    cameras: tuple[str, ...]

    @property
    def covered_cells(self) -> int:
        return int(np.count_nonzero(self.valid))

    @property
    def covered_fraction(self) -> float:
        return self.covered_cells / self.grid.cell_count

    def age_seconds(self, now: float | None = None) -> np.ndarray:
        moment = time.time() if now is None else now
        return moment - self.updated_at

    def cells_from(self, camera_id: str) -> int:
        """How many cells this camera won. Zero for a camera nobody outranked."""
        if camera_id not in self.cameras:
            return 0
        return int(np.count_nonzero(self.source == self.cameras.index(camera_id)))

    def describe(self) -> str:
        lines = [
            f"grid            {self.grid.rows}×{self.grid.columns} cells "
            f"at {self.grid.cell_size_m:g} m "
            f"({self.grid.height_m:,.0f} × {self.grid.width_m:,.0f} m)",
            f"mapped          {self.covered_cells:,} cells "
            f"({self.covered_fraction:.0%})",
            f"empty           {self.grid.cell_count - self.covered_cells:,} cells "
            "— left empty, never interpolated",
        ]
        for index, camera_id in enumerate(self.cameras):
            won = int(np.count_nonzero(self.source == index))
            lines.append(f"  {camera_id:<12} {won:,} cell(s)")
        lines.append("")
        lines.append(
            "Only the ground plane is correct. Anything with height is smeared "
            "along the ray from the camera that saw it, so this is a basemap, "
            "not a photograph."
        )
        return "\n".join(lines)


def _refuse_unknown_cell_error(patch: GroundPatch) -> None:
    """Refuse a patch that offers ground whose position error nobody measured.

    ``sigma_m`` is NaN exactly where ``valid`` is false, and both producers here
    keep that. A NaN on a *valid* cell is a cell whose error is unknown, and no
    substitute for it is honest: zero — the obvious one — makes an unmeasured
    cell beat every surveyed camera in :func:`mosaic`, which is the same failure
    that function already refuses a missing pose error for, and NaN carried
    through means a renderer that tests for it erases ground a camera really
    saw. So the patch is refused at the door, naming the camera, rather than
    averaged over somewhere the operator will read the result as surveyed.
    """
    unknown = int(np.count_nonzero(patch.valid & np.isnan(patch.sigma_m)))
    if unknown:
        raise OrthophotoError(
            f"patch {patch.camera_id!r} offers {unknown:,} cell(s) as ground "
            "whose position error is unknown (sigma_m is NaN where valid is "
            "true). There is no safe stand-in: zero would let those cells "
            "outrank a surveyed camera, and NaN reads as 'no ground here' to "
            "anything that draws the result."
        )


def _as_three_dimensional(image: np.ndarray) -> np.ndarray:
    """Treat a greyscale frame as a one-channel one, so one code path samples both."""
    if image.ndim == 2:
        return image[:, :, np.newaxis]
    if image.ndim == 3:
        return image
    raise OrthophotoError(
        f"an image must be (height, width) or (height, width, channels); "
        f"an array of shape {image.shape} was given"
    )


def _forward_lattice(
    pose: CameraPose,
    grid: GroundGrid,
    lattice: int,
    angular_uncertainty_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project a lattice of image points onto the grid's metric frame.

    Returns cell-space coordinates, per-node uncertainty, and which nodes landed
    on the ground at all. A node that returned ``None`` is above the horizon or
    beyond the pose's range; it is dropped rather than clamped, because a
    clamped horizon ray lands at infinity and would stretch one triangle across
    the whole site.
    """
    frame = _frame_for(grid.origin)
    axis = np.linspace(0.0, 1.0, lattice)

    cell_x = np.full((lattice, lattice), np.nan)
    cell_y = np.full((lattice, lattice), np.nan)
    sigma = np.full((lattice, lattice), np.nan)

    for row, v in enumerate(axis):
        for column, u in enumerate(axis):
            projection = project_to_ground(
                pose, float(u), float(v), angular_uncertainty_deg
            )
            if projection is None:
                continue
            x, y = frame.to_xy(projection.point)
            # Cell space: the centre of cell (r, c) sits at (c + 0.5, r + 0.5),
            # which makes the rasteriser below index arithmetic rather than
            # metres, and keeps the north-up row convention in one place.
            cell_x[row, column] = x / grid.cell_size_m
            cell_y[row, column] = -y / grid.cell_size_m
            sigma[row, column] = projection.uncertainty_meters

    return cell_x, cell_y, sigma, np.isfinite(cell_x)


def _rasterise_triangle(
    corners: tuple[tuple[float, float], ...],
    attributes: np.ndarray,
    grid: GroundGrid,
    out_u: np.ndarray,
    out_v: np.ndarray,
    out_sigma: np.ndarray,
    out_covered: np.ndarray,
) -> None:
    """Fill every grid cell whose centre lies in this triangle, by interpolation.

    One numpy pass over the triangle's bounding box of cells, not a Python loop
    over cells. The whole reason the inverse is built this way is that a real
    grid has tens of thousands of cells and a per-cell loop across the FFI takes
    a fifth of a second per camera per frame — unusable against a median that
    wants hundreds of frames.
    """
    (x0, y0), (x1, y1), (x2, y2) = corners
    determinant = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(determinant) < 1e-12:
        # Three collinear projections — normal near the horizon, where a whole
        # lattice row lands on one line. It covers no area, so it fills nothing.
        return

    low_column = max(0, math.floor(min(x0, x1, x2) - 0.5))
    high_column = min(grid.columns - 1, math.ceil(max(x0, x1, x2) - 0.5))
    low_row = max(0, math.floor(min(y0, y1, y2) - 0.5))
    high_row = min(grid.rows - 1, math.ceil(max(y0, y1, y2) - 0.5))
    if low_column > high_column or low_row > high_row:
        return

    columns = np.arange(low_column, high_column + 1) + 0.5
    rows = np.arange(low_row, high_row + 1) + 0.5
    px, py = np.meshgrid(columns, rows)

    first = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / determinant
    second = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / determinant
    third = 1.0 - first - second
    inside = (
        (first >= -_BARYCENTRIC_EPSILON)
        & (second >= -_BARYCENTRIC_EPSILON)
        & (third >= -_BARYCENTRIC_EPSILON)
    )
    if not inside.any():
        return

    window = (slice(low_row, high_row + 1), slice(low_column, high_column + 1))
    for target, values in zip(
        (out_u, out_v, out_sigma),
        (
            first * attributes[0, 0] + second * attributes[1, 0] + third * attributes[2, 0],
            first * attributes[0, 1] + second * attributes[1, 1] + third * attributes[2, 1],
            first * attributes[0, 2] + second * attributes[1, 2] + third * attributes[2, 2],
        ),
    ):
        target[window] = np.where(inside, values, target[window])
    out_covered[window] |= inside


def sample_frame(
    pose: CameraPose,
    image: np.ndarray,
    grid: GroundGrid,
    *,
    camera_id: str,
    captured_at: float | None = None,
    lattice: int = DEFAULT_LATTICE,
    angular_uncertainty_deg: float = 1.5,
) -> GroundPatch:
    """Resample one frame into a top-down patch on the grid.

    For each ground cell inside the camera's footprint, the image pixel that
    cell would appear in is found and its colour taken. Cells outside the
    footprint — behind the camera, above the horizon, past its range, or simply
    off the edge of the frame — stay empty, and ``valid`` says which is which.

    Sampling is nearest-neighbour. Bilinear would average across the boundary
    between a wall and the ground behind it and produce a colour no pixel of the
    frame ever held; on an image where anything with height is already smeared
    along its ray, inventing intermediate colour adds error that looks like
    detail.

    The inverse map is the lattice-and-interpolate one described in this
    module's docstring: exact at the lattice nodes, piecewise-linear between,
    and biased inward at the footprint edge so no cell is claimed that the pose
    does not reach.

    ``camera_id`` is required and has no default. Every cell of the mosaic this
    feeds names the camera it came from, and a default would name two different
    masts the same thing — which is the one question a disputed cell has to be
    able to answer.
    """
    picture = _as_three_dimensional(np.asarray(image))
    height, width = picture.shape[:2]
    if height < 1 or width < 1:
        raise OrthophotoError("an image with no pixels cannot be sampled")
    if lattice < 2:
        raise OrthophotoError(
            f"the projection lattice needs at least 2 points per axis; {lattice} was given"
        )

    moment = time.time() if captured_at is None else float(captured_at)

    cell_x, cell_y, node_sigma, ok = _forward_lattice(
        pose, grid, lattice, angular_uncertainty_deg
    )

    out_u = np.zeros(grid.shape)
    out_v = np.zeros(grid.shape)
    out_sigma = np.full(grid.shape, np.nan)
    covered = np.zeros(grid.shape, dtype=bool)

    axis = np.linspace(0.0, 1.0, lattice)
    for row in range(lattice - 1):
        for column in range(lattice - 1):
            # Each lattice quad, split into the two triangles that share its
            # diagonal. Triangles, not quads, because a projected quad is not
            # planar in image space and an inverse bilinear over it has two
            # roots — one of which is the wrong side of the camera.
            corner_rows = (row, row, row + 1, row + 1)
            corner_columns = (column, column + 1, column + 1, column)
            for triangle in ((0, 1, 2), (0, 2, 3)):
                indices = [(corner_rows[i], corner_columns[i]) for i in triangle]
                if not all(ok[r, c] for r, c in indices):
                    # One corner did not reach the ground. Dropping the whole
                    # triangle leaves cells empty at the horizon and at maximum
                    # range; filling them would claim ground the pose refused.
                    continue
                corners = tuple((cell_x[r, c], cell_y[r, c]) for r, c in indices)
                attributes = np.array(
                    [
                        (axis[c], axis[r], node_sigma[r, c])
                        for r, c in indices
                    ]
                )
                _rasterise_triangle(
                    corners, attributes, grid, out_u, out_v, out_sigma, covered
                )

    # The range is a circle, and the lattice edge is a chain of straight
    # triangle edges that can bulge a little past it. Enforced exactly here so
    # no cell is coloured from ground the pose says is out of reach.
    camera_x, camera_y = _frame_for(grid.origin).to_xy(pose.position)
    grid_x, grid_y = grid.centres_xy()
    within_range = np.hypot(grid_x - camera_x, grid_y - camera_y) <= pose.range_meters
    valid = covered & within_range & (out_u >= 0.0) & (out_u <= 1.0)
    valid &= (out_v >= 0.0) & (out_v <= 1.0)

    pixel_x = np.clip(np.rint(out_u * (width - 1)).astype(np.intp), 0, width - 1)
    pixel_y = np.clip(np.rint(out_v * (height - 1)).astype(np.intp), 0, height - 1)

    colour = np.zeros((*grid.shape, picture.shape[2]), dtype=picture.dtype)
    colour[valid] = picture[pixel_y[valid], pixel_x[valid]]

    sigma = np.where(valid, out_sigma, np.nan).astype(np.float32)
    updated = np.where(valid, moment, np.nan)
    samples = valid.astype(np.uint32)

    return GroundPatch(
        camera_id=camera_id,
        grid=grid,
        colour=colour,
        valid=valid,
        sigma_m=sigma,
        updated_at=updated,
        samples=samples,
    )


class MedianAccumulator:
    """A running per-cell median, which is what turns footage into a basemap.

    A single patch is a photograph of the site *including whatever was on it*:
    a van in the loading bay, a person crossing the yard, a shadow at the angle
    the sun happened to be. Composite those and the basemap has people printed
    into the ground.

    The median removes them, and the reason is exact rather than statistical.
    Changing fewer than half of a cell's samples cannot move their median
    outside the range spanned by the samples that were not changed. A person
    walking across a cell occupies it for a second or two out of the minutes
    this accumulates, so their colour is a small minority of that cell's stack
    and contributes nothing to the value that survives. What is left is the
    ground that was there the whole time — the empty site, which is what a
    basemap should be. The corollary is honest too: something parked in the same
    cell for more than half of the observation window *is* the ground as far as
    this is concerned, and will be printed into the map.

    **Memory is bounded per cell, not per frame.** Each cell keeps a ring of the
    last ``capacity`` samples and nothing else, so a thousand frames cost
    exactly what fifteen do: ``capacity × (channels + 4)`` bytes per cell —
    ``channels`` bytes of colour plus a float32 sample time. At the default
    capacity, three channels and a 200×200 grid that is 4.2 MB, and it does not
    grow. Without that bound an hour of footage at 15 fps would be 54,000
    samples per cell, which is not a basemap, it is a video file.

    The median is taken per channel. The result is therefore a colour that no
    single frame necessarily contained; for ground under a map overlay that is
    the right trade, and it is stated here rather than discovered later.
    """

    __slots__ = (
        "_grid",
        "_channels",
        "_capacity",
        "_dtype",
        "_minimum_samples",
        "_retain_seconds",
        "_samples",
        "_times",
        "_written",
        "_sigma",
        "_epoch",
        "_frames",
    )

    def __init__(
        self,
        grid: GroundGrid,
        *,
        channels: int = 3,
        capacity: int = DEFAULT_CAPACITY,
        dtype: np.dtype | type = np.uint8,
        minimum_samples: int = MINIMUM_SAMPLES,
        retain_seconds: float | None = None,
    ) -> None:
        if capacity < 1:
            raise OrthophotoError(
                f"a sample ring needs at least one slot; {capacity} was given"
            )
        if minimum_samples < 1 or minimum_samples > capacity:
            raise OrthophotoError(
                f"minimum_samples must be between 1 and the capacity ({capacity}); "
                f"{minimum_samples} was given"
            )
        self._grid = grid
        self._channels = channels
        self._capacity = capacity
        self._dtype = np.dtype(dtype)
        self._minimum_samples = minimum_samples
        self._retain_seconds = retain_seconds
        self._samples = np.zeros((capacity, *grid.shape, channels), dtype=self._dtype)
        # Seconds since the first observation, as float32. Storing unix seconds
        # in float32 would quantise them to about two minutes; storing an offset
        # keeps sub-centisecond resolution over a day at half the memory of
        # float64.
        self._times = np.zeros((capacity, *grid.shape), dtype=np.float32)
        self._written = np.zeros(grid.shape, dtype=np.uint32)
        self._sigma = np.full(grid.shape, np.nan, dtype=np.float32)
        self._epoch: float | None = None
        self._frames = 0

    @property
    def grid(self) -> GroundGrid:
        return self._grid

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def frames(self) -> int:
        """How many patches have been folded in."""
        return self._frames

    @property
    def bytes_per_cell(self) -> int:
        """The memory bound this class exists to keep, per cell.

        The ring — colour plus a sample time, times its depth — and the two
        per-cell tallies beside it: how many samples have been written, and the
        worst position error behind them. Every term is constant in the number
        of frames observed, which is the whole point. Multiplied by the cell
        count this is exactly :attr:`memory_bytes`, and the test says so, because
        a stated bound that quietly omits a term is not a bound.
        """
        ring = self._capacity * (self._channels * self._dtype.itemsize + 4)
        return ring + self._written.itemsize + self._sigma.itemsize

    @property
    def memory_bytes(self) -> int:
        return int(
            self._samples.nbytes
            + self._times.nbytes
            + self._written.nbytes
            + self._sigma.nbytes
        )

    def update(self, patch: GroundPatch) -> None:
        """Fold one patch in, writing only where it has ground.

        A cell outside this patch's footprint is untouched, not zeroed: an empty
        sample is not an observation of black ground, and counting it would drag
        every edge cell's median toward the colour of nothing.
        """
        if patch.grid != self._grid:
            raise OrthophotoError(
                "this patch is on a different grid from the accumulator; "
                "compositing two rasters that do not share an origin, cell size "
                "and extent silently shifts the map"
            )
        if patch.channels != self._channels:
            raise OrthophotoError(
                f"the accumulator holds {self._channels}-channel colour; "
                f"this patch has {patch.channels}"
            )
        # Caught here rather than at the mosaic, where the camera named in the
        # refusal would be this accumulator's own id and not the frame that
        # arrived without an error to its name.
        _refuse_unknown_cell_error(patch)

        rows, columns = np.nonzero(patch.valid)
        self._frames += 1
        if rows.size == 0:
            return

        moment = float(np.nanmax(patch.updated_at))
        if self._epoch is None:
            self._epoch = moment
        offset = np.float32(moment - self._epoch)

        slot = self._written[rows, columns] % self._capacity
        self._samples[slot, rows, columns, :] = patch.colour[rows, columns, :]
        self._times[slot, rows, columns] = offset
        self._written[rows, columns] += 1

        # The worst position error any contributing frame had, never the best.
        # For a fixed camera it is the same number every frame; if the pose has
        # moved, keeping the maximum means the mosaic never ranks this cell on
        # a certainty that only one of its samples had.
        self._sigma[rows, columns] = np.fmax(
            self._sigma[rows, columns], patch.sigma_m[rows, columns]
        )

    def result(self, *, camera_id: str, now: float | None = None) -> GroundPatch:
        """The median patch: the empty site, as far as the samples can show it.

        Cells with fewer than ``minimum_samples`` surviving observations come
        back empty. That is deliberately strict at the edges of a footprint,
        where a cell may have been seen twice by a wobbling mast — two samples
        cannot outvote a passer-by, and a cell that cannot outvote one has no
        business being drawn as ground.

        ``updated_at`` is the newest sample behind each cell, so a renderer can
        fade stale ground without being told which frames went in.

        ``camera_id`` is required and deliberately has no default. It used to
        default to ``"median"``, which named every accumulator's result the same
        thing — so the ordinary pipeline, one accumulator per camera, produced a
        set of patches that :func:`mosaic` could not tell apart and a ``source``
        index that resolved back to nobody.
        """
        filled = np.arange(self._capacity, dtype=np.uint32)[:, None, None] < np.minimum(
            self._written, self._capacity
        )
        usable = filled
        if self._retain_seconds is not None and self._epoch is not None:
            moment = time.time() if now is None else float(now)
            oldest = np.float32(moment - self._epoch - self._retain_seconds)
            usable = filled & (self._times >= oldest)

        counts = usable.sum(axis=0, dtype=np.uint32)
        valid = counts >= self._minimum_samples

        colour = np.zeros((*self._grid.shape, self._channels), dtype=self._dtype)
        updated = np.full(self._grid.shape, np.nan)
        rows, columns = np.nonzero(valid)
        if rows.size:
            # Only the covered cells are stacked. An all-empty cell would be an
            # all-NaN median — a warning and a wasted allocation over ground
            # nobody has looked at.
            stack = self._samples[:, rows, columns, :].astype(np.float32)
            mask = usable[:, rows, columns]
            stack[~mask] = np.nan
            median = np.nanmedian(stack, axis=0)
            if self._dtype.kind in "ui":
                median = np.rint(median)
            colour[rows, columns, :] = median.astype(self._dtype)

            times = np.where(mask, self._times[:, rows, columns], -np.inf)
            updated[rows, columns] = times.max(axis=0) + (self._epoch or 0.0)

        return GroundPatch(
            camera_id=camera_id,
            grid=self._grid,
            colour=colour,
            valid=valid,
            sigma_m=np.where(valid, self._sigma, np.nan).astype(np.float32),
            updated_at=updated,
            samples=counts,
        )


def mosaic(
    patches: Sequence[GroundPatch],
    sigma_by_camera: Mapping[str, float],
) -> Mosaic:
    """Composite patches, each cell taken from whoever knows that cell best.

    "Best" is the smallest position error *at that cell*, not the best camera
    overall: a camera 15 m away wins the near ground and loses the far corner to
    a camera that has that corner under its own mast. The per-cell error is the
    projection uncertainty the patch already carries, combined in quadrature
    with the camera's own pose error from ``sigma_by_camera`` — two independent
    contributions, one from where the ray lands and one from how well the mast
    is surveyed.

    Both halves of that rank must be evidence, so both are refused when they
    are not. A camera missing from ``sigma_by_camera`` is an error rather than a
    zero — defaulting an unsurveyed camera to a perfect pose would let it win
    every cell it touches, which is precisely backwards — and so is a patch that
    offers a cell as ground without an error for it, for exactly the same
    reason one cell lower down.

    Two patches claiming the same ``camera_id`` are refused too. A cell's
    ``source`` has to resolve to one camera and one pose to be worth keeping,
    and a repeated id makes it resolve to whichever patch sorted first.

    Cells no patch covers stay empty. Nothing is interpolated across them, ever.
    """
    if not patches:
        raise OrthophotoError("a mosaic needs at least one patch")

    grid = patches[0].grid
    channels = patches[0].channels
    claimed: set[str] = set()
    for patch in patches:
        if patch.grid != grid:
            raise OrthophotoError(
                f"patch {patch.camera_id!r} is on a different grid; patches must "
                "share an origin, cell size and extent to be composited"
            )
        if patch.channels != channels:
            raise OrthophotoError(
                f"patch {patch.camera_id!r} has {patch.channels} channel(s) "
                f"where the first has {channels}"
            )
        if patch.camera_id not in sigma_by_camera:
            raise OrthophotoError(
                f"no pose error was given for camera {patch.camera_id!r}. Every "
                "camera in a mosaic needs one: a camera whose pose error is "
                "unknown cannot be ranked against one whose is, and assuming "
                "zero would make it win every cell it can see."
            )
        if patch.camera_id in claimed:
            raise OrthophotoError(
                f"two patches both claim to come from camera "
                f"{patch.camera_id!r}. A cell's source must resolve to one "
                "camera and one pose: with a repeated id the winner is "
                "whichever patch sorted first, `cells_from` under-reports by "
                "however many patches share the name, and an operator "
                "disputing what is drawn under a fence line gets an index "
                "that names nobody in particular."
            )
        claimed.add(patch.camera_id)
        _refuse_unknown_cell_error(patch)

    ordered = sorted(patches, key=lambda patch: patch.camera_id)
    cameras = tuple(patch.camera_id for patch in ordered)

    colour = np.zeros((*grid.shape, channels), dtype=ordered[0].colour.dtype)
    best = np.full(grid.shape, np.inf)
    source = np.full(grid.shape, -1, dtype=np.int32)
    updated = np.full(grid.shape, np.nan)

    for index, patch in enumerate(ordered):
        pose_sigma = float(sigma_by_camera[patch.camera_id])
        if not math.isfinite(pose_sigma) or pose_sigma < 0.0:
            raise OrthophotoError(
                f"the pose error for camera {patch.camera_id!r} is {pose_sigma}; "
                "it must be a non-negative number of metres"
            )
        # Every valid cell carries its own measured error — a patch that did
        # not was refused above — so what ranks the cameras here is evidence the
        # whole way down: where the ray landed, and how well the mast it came
        # from is surveyed, in quadrature. Substituting zero for an unmeasured
        # cell was the bug this replaced: it handed the least-known camera the
        # smallest number and printed its guess as a surveyed one.
        effective = np.hypot(patch.sigma_m.astype(np.float64), pose_sigma)
        # Strictly better, over patches in camera-id order: a tie goes to the
        # first camera by name, so the same inputs always give the same mosaic.
        wins = patch.valid & (effective < best)
        best = np.where(wins, effective, best)
        source = np.where(wins, index, source)
        colour[wins] = patch.colour[wins]
        updated = np.where(wins, patch.updated_at, updated)

    valid = source >= 0
    sigma = np.where(valid, best, np.nan).astype(np.float32)

    result = Mosaic(
        grid=grid,
        colour=colour,
        valid=valid,
        sigma_m=sigma,
        updated_at=updated,
        source=source,
        cameras=cameras,
    )
    _log.info(
        "mosaic: %d of %d cells from %d camera(s); %d left empty",
        result.covered_cells,
        grid.cell_count,
        len(cameras),
        grid.cell_count - result.covered_cells,
    )
    return result
