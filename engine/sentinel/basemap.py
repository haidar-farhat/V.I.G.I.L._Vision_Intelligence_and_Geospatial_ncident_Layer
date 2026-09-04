"""The site's own basemap: built from its cameras, fingerprinted, kept on disk.

:mod:`sentinel.orthophoto` can already turn one frame into a top-down patch,
run a median over many of them, and composite the medians from several cameras.
Until this module existed nothing called any of it — the sixth instance of the
repository's recurring defect, correct and tested code that no product path
reaches. This is the product path: a :class:`BasemapBuilder` that is fed frames
and owns one grid and one accumulator per camera, a :class:`BasemapAsset` that
is the finished raster plus everything needed to argue about it later, and a
pair of files under the data directory that survive the process.

**What is stored, and what deliberately is not.** The asset is the ground
raster and its provenance: the grid, the poses it was built from, which camera
won each cell, how old each cell's ground was when the map was built, and the
position error behind it. No frame is ever written. A basemap is the *empty*
site by construction — the median has removed whoever walked through — and
keeping the frames that fed it would keep the people the median removed. The
source of each camera is not written either, not even redacted: a camera URL
carries a credential, and the basemap is the one artefact that is meant to be
copied about freely.

**The fingerprint is the claim of identity.** An incident judged "inside the
fence" against this map has to be re-checkable against *this* map, not against
whatever is in the folder by then. So the asset carries the SHA-256 of the PNG
bytes and the canonical JSON of its metadata, and :func:`load_basemap` refuses
to hand back an asset whose files no longer hash to it. A tampered file and a
half-written one look identical to that check, and both must not be drawn.

**The pose error is an assumption, and it says so.** Ranking two cameras over
a shared cell needs each mast's own position error, and today nothing measures
one: an operator places a camera by clicking a map. :data:`DEFAULT_POSE_SIGMA_M`
is what that click is worth — a metre — and it is a stated default, not a
measurement. When a surveyed value exists it replaces this; until then every
asset is built on the same honest guess for every camera, which ranks them by
the only evidence that differs between them, the per-cell projection error.

**One lattice, one grid per camera.** Each camera samples onto a grid of its
own, sized once from its footprint plus a margin. Every such grid lies on one
shared cell lattice — anchored where the first camera's grid was placed — so a
later camera's cells are a whole number of cells from the first camera's, and
the build lays every median onto the union grid by an integer shift and
nothing else. No ring is ever moved or copied: a second camera joining the
build leaves the first camera's samples exactly where they are. An earlier
shape of this module sized one grid from the first footprint and *extended* it
when a later footprint fell outside, moving every ring by an index shift — and
re-measuring the first footprint in the grid's own frame found its edge a
fraction of a millimetre past where it had been placed, which ``ceil`` made a
whole row: feeding the same camera a *second frame* grew the grid and copied
thirteen megabytes for nothing. The two measurements disagreed by the
convergence of meridians between two local frames, ``D·d·tan(lat)/R`` — 0.6 mm
for frames 60 m apart over a 100 m footprint at this latitude — which no
tolerance a millimetre wide would have survived at a larger site. A footprint
is now measured once, in one frame, and never again.

**Memory.** Each camera's ring keeps ``capacity`` samples per cell of *its own*
grid and nothing else: ``capacity × (3 + 4) + 8`` bytes per cell, 113 at the
default depth of fifteen. The reference pose at the CLI's default quarter-metre
cell is 328×362 = 118,736 cells, so 13.4 MB per camera — measured — and it
grows neither with the minutes fed nor with the cameras that join later. The
one approximation in the lattice is that each grid is a local frame at its own
origin, and two such frames disagree by the convergence of meridians,
``D·d·tan(lat)/R`` for origins ``D`` apart east-west and a cell ``d`` away:
0.18 mm measured for a camera 60 m east of the first, a few centimetres across
a half-kilometre site, and never a resampling. Sampling one frame
onto that grid takes 79 ms, which is why the command line feeds four frames a
second and not every frame the camera decodes: the median needs *seconds* of
separation between samples to see a walker leave a cell, and a hundred frames
of the same half-second would fill the ring with one moment.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import time
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

from . import paths
from .core import CameraPose, LatLon, field_of_view
from .coverage import ARC_SEGMENTS
from .logs import get as _get_logger
from .orthophoto import (
    DEFAULT_CAPACITY,
    MINIMUM_SAMPLES,
    GroundGrid,
    GroundPatch,
    MedianAccumulator,
    OrthophotoError,
    _frame_for,
    mosaic,
    sample_frame,
)

_log = _get_logger(__name__)

#: How well an operator placed a camera by clicking a map, in metres of 1σ
#: horizontal error. An honest assumption rather than a measurement: nothing in
#: this system surveys a mast yet, and a metre is what a careful click on a
#: plan at site scale is worth. It is the same for every camera, so it ranks
#: none of them above another; what decides a shared cell is the per-cell
#: projection error, which *is* measured. Replace it with a surveyed value the
#: moment one exists.
DEFAULT_POSE_SIGMA_M = 1.0

#: Ground added around every footprint when its grid is placed. One metre
#: covers the chord bulge of the arc approximation and the rounding that
#: :meth:`GroundGrid.covering` warns can push an extreme point off the raster;
#: the placement here rounds *outward* to whole cells on top of it.
GRID_MARGIN_M = 1.0

#: The two files an asset is kept as. Fixed names, one asset per directory:
#: the basemap is a site's, not a run's, and a second one beside it would be a
#: second answer to "what is the ground here".
PNG_NAME = "basemap.png"
JSON_NAME = "basemap.json"

#: Written into the JSON so a reader can refuse a shape it does not understand
#: rather than guess at one.
FORMAT = "sentinel-basemap/1"

#: Layer dtypes on disk, little-endian and explicit, so the fingerprint means
#: the same bytes on every machine.
_SIGMA_DTYPE = "<f4"
_AGE_DTYPE = "<f4"
_SOURCE_DTYPE = "<i2"


class BasemapError(RuntimeError):
    """A basemap could not be built, saved or loaded from what was given."""


@dataclass(frozen=True, slots=True, eq=False)
class BasemapAsset:
    """The finished basemap and everything needed to argue about it later.

    ``valid`` is the only honest test for "is there ground here". ``colour`` is
    zero where a cell is empty, and zero is also what asphalt at dusk reads as,
    so a renderer must test the mask and never the pixel. Everything per-cell
    beside the colour is ``inf`` or ``-1`` where the cell is empty, so a
    renderer that fades by age or ranks by error reads "absent", never "very
    old" or "perfect".

    ``cameras`` lists every camera the asset was built from, including one
    that saw no ground at all — it is part of the provenance that a camera
    contributed nothing — and ``source`` indexes into it. ``poses`` are the
    poses given at feed time, unchanged, because a cell's source has to resolve
    to the pose that produced it.

    Equality is off, as on the patches this is built from: the arrays are
    megabytes and a dataclass ``__eq__`` over numpy raises rather than answers.
    The fingerprint is the identity.
    """

    grid: GroundGrid
    #: ``(rows, columns, 3)`` uint8 BGR.
    colour: np.ndarray
    #: ``(rows, columns)`` bool — cells some camera actually covered.
    valid: np.ndarray
    #: ``(rows, columns)`` float32 — seconds between the cell's newest sample
    #: and ``built_at``; ``inf`` where not valid.
    age_seconds: np.ndarray
    #: ``(rows, columns)`` float32 — the cell's position error; ``inf`` where
    #: not valid.
    sigma_m: np.ndarray
    #: ``(rows, columns)`` int16 index into :attr:`cameras`; ``-1`` where empty.
    source: np.ndarray
    cameras: tuple[str, ...]
    poses: dict[str, CameraPose]
    #: Wall clock, UTC epoch milliseconds.
    built_at_millis: int
    #: Frames the medians were built from, across every camera.
    frames_used: int
    #: SHA-256 hex of the PNG bytes and the canonical JSON metadata.
    fingerprint: str

    @property
    def covered_cells(self) -> int:
        return int(np.count_nonzero(self.valid))

    @property
    def covered_fraction(self) -> float:
        return self.covered_cells / self.grid.cell_count

    def cells_from(self, camera_id: str) -> int:
        """How many cells this camera won. Zero for one that saw no ground.

        A camera the asset was not built from is refused, not counted as zero:
        "contributed nothing" is a fact about a camera in :attr:`cameras`, and
        a mistyped id that read as it would be reported as a camera that saw no
        ground rather than as a camera nobody asked about.
        """
        if camera_id not in self.cameras:
            raise BasemapError(
                f"camera {camera_id!r} is not one this basemap was built from "
                f"({', '.join(self.cameras)})"
            )
        return int(np.count_nonzero(self.source == self.cameras.index(camera_id)))

    def describe(self) -> str:
        """One line, and an honest one: what it is, how much of it, and how old."""
        ages = self.age_seconds[self.valid]
        if ages.size:
            freshness = f"ground {ages.min():.0f}–{ages.max():.0f} s old at build"
        else:
            freshness = "no ground mapped"
        when = datetime.fromtimestamp(
            self.built_at_millis / 1000.0, tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S")
        return (
            f"basemap {self.grid.rows}×{self.grid.columns} cells at "
            f"{self.grid.cell_size_m:g} m: {self.covered_cells:,} mapped "
            f"({self.covered_fraction:.0%}), {len(self.cameras)} camera(s) "
            f"{', '.join(self.cameras)}; {freshness}; built {when} UTC from "
            f"{self.frames_used} frame(s)"
        )


@dataclass(frozen=True, slots=True)
class _Placement:
    """Where one camera's grid sits on the shared lattice.

    ``row`` and ``column`` are the lattice coordinates of the grid's north-west
    corner — whole cells south and east of the lattice origin, negative for a
    grid that reaches north or west of it. Integers by construction, which is
    what lets the build lay this grid onto the union with an index shift and
    nothing else.
    """

    grid: GroundGrid
    row: int
    column: int


class BasemapBuilder:
    """Feeds frames into one median per camera and builds the composite.

    Each camera gets a grid of its own the first time it is fed, sized from
    its footprint plus a margin and placed on the lattice the first camera's
    grid anchored, so that every grid here is a whole number of cells from
    every other. That footprint is measured once, in one frame, and never
    again: a camera's pose is fixed for the build (a changed one is refused),
    so its grid is fixed too, and feeding it a thousandth frame costs the same
    as feeding it a second. The grid a later camera brings can reach outside
    the union of the grids before it; the union grows, and no ring moves —
    the union exists only at :meth:`build`, where each camera's median is laid
    onto it at its lattice offset and the mosaic is taken there.

    A camera pointed at the sky feeds nothing: its footprint is empty, so no
    grid can be placed for it and no patch sampled. It is still counted — in
    :meth:`frames_fed`, in :meth:`blind_cameras`, and in the asset's
    ``cameras`` with zero cells — because "this camera contributed nothing" is
    provenance, not an absence.

    Not thread-safe. One thread feeds; a console that samples on a worker and
    builds on the main thread puts a lock around both.
    """

    __slots__ = (
        "_cell_size_m",
        "_capacity",
        "_pose_sigma_m",
        "_lattice",
        "_placements",
        "_grid",
        "_grid_at",
        "_accumulators",
        "_poses",
        "_fed",
        "_blind",
    )

    def __init__(
        self,
        *,
        cell_size_m: float = 0.25,
        capacity: int = DEFAULT_CAPACITY,
        pose_sigma_m: float = DEFAULT_POSE_SIGMA_M,
    ) -> None:
        if not (cell_size_m > 0.0 and math.isfinite(cell_size_m)):
            raise BasemapError(
                f"a basemap cell must have a positive size; {cell_size_m} m was given"
            )
        if capacity < MINIMUM_SAMPLES:
            # Refused here rather than at the first feed, where the accumulator
            # would refuse it anyway: a builder that cannot ever report a cell
            # should say so before a camera is opened for it.
            raise BasemapError(
                f"the sample ring must hold at least {MINIMUM_SAMPLES} samples "
                f"for a median to be reported; {capacity} was given"
            )
        if not (math.isfinite(pose_sigma_m) and pose_sigma_m >= 0.0):
            raise BasemapError(
                f"the pose error must be a non-negative number of metres; "
                f"{pose_sigma_m} was given"
            )
        self._cell_size_m = float(cell_size_m)
        self._capacity = int(capacity)
        self._pose_sigma_m = float(pose_sigma_m)
        #: The lattice origin: the first camera's grid origin, once there is one.
        self._lattice: LatLon | None = None
        self._placements: dict[str, _Placement] = {}
        #: The union of every placed grid, and its own lattice coordinates.
        #: Replaced only when a camera joins outside it, so a caller holding
        #: the grid holds the current one until then.
        self._grid: GroundGrid | None = None
        self._grid_at: tuple[int, int] = (0, 0)
        self._accumulators: dict[str, MedianAccumulator] = {}
        self._poses: dict[str, CameraPose] = {}
        self._fed: dict[str, int] = {}
        self._blind: set[str] = set()

    @property
    def grid(self) -> GroundGrid | None:
        """The union of every camera's grid, or ``None`` before any ground was fed.

        The same object until a camera joins outside it: feeding a camera that
        is already placed never changes it, whatever the frame.
        """
        return self._grid

    @property
    def cell_size_m(self) -> float:
        return self._cell_size_m

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def pose_sigma_m(self) -> float:
        return self._pose_sigma_m

    @property
    def memory_bytes(self) -> int:
        """What the accumulators hold, which is what this builder costs."""
        return sum(accumulator.memory_bytes for accumulator in self._accumulators.values())

    def grid_of(self, camera_id: str) -> GroundGrid | None:
        """The grid this camera's ring is on, or ``None`` when it has no ring.

        ``None`` for a camera never fed and for one that sees no ground —
        :meth:`blind_cameras` tells the two apart.
        """
        placement = self._placements.get(camera_id)
        return None if placement is None else placement.grid

    def frames_fed(self) -> dict[str, int]:
        """Frames taken per camera — a sky-pointing camera's counted, a frame
        that could not be sampled not."""
        return dict(self._fed)

    def frames_sampled(self) -> dict[str, int]:
        """Frames that actually reached a median, per camera."""
        return {
            camera_id: accumulator.frames
            for camera_id, accumulator in self._accumulators.items()
        }

    def blind_cameras(self) -> tuple[str, ...]:
        """Cameras fed here that see no ground at all, in id order."""
        return tuple(sorted(self._blind))

    def feed(
        self,
        camera_id: str,
        pose: CameraPose,
        image: np.ndarray,
        captured_at: float,
    ) -> None:
        """Sample one frame into this camera's median.

        ``captured_at`` is unix seconds and becomes the cell's sample time, so
        the asset can say how old each cell's ground was. A camera whose pose
        differs from the one it was first fed with is refused: the samples
        already in its ring were placed by the old pose, and a stack mixing two
        placements has no single pose a disputed cell could resolve to. Start a
        new builder for a camera that moved.

        A frame that cannot be sampled is refused whole: it is not counted, its
        pose is not recorded, and a grid placed for it is dropped. Nothing
        about a frame that was not fed survives the refusal.
        """
        if not camera_id:
            raise BasemapError("a camera needs an id; every cell of the basemap names its source")
        if not math.isfinite(captured_at):
            raise BasemapError(f"captured_at must be a finite time in seconds; {captured_at} was given")
        known = self._poses.get(camera_id)
        if known is not None and known != pose:
            raise BasemapError(
                f"camera {camera_id!r} was fed with a different pose earlier in this "
                "build. A cell's samples must all come from one placement; start a "
                "new builder for a camera that has moved."
            )

        picture = _as_bgr(image)
        if camera_id in self._blind:
            # Sees no ground, as established on its first frame. Counted, never
            # sampled; the footprint is not measured again.
            self._fed[camera_id] += 1
            return

        placement = self._placements.get(camera_id)
        if placement is None:
            # The first frame from this camera. Its pose is fixed for the
            # build, so this is the one time its footprint is measured.
            footprint = field_of_view(pose, arc_segments=ARC_SEGMENTS)
            if len(footprint) < 3:
                _log.warning(
                    "basemap: camera %s sees no ground (the bottom of its frame is "
                    "above the horizon); it contributes nothing", camera_id,
                )
                self._blind.add(camera_id)
                self._poses[camera_id] = pose
                self._fed[camera_id] = 1
                return
            placement = _place(
                footprint, cell_size_m=self._cell_size_m, lattice=self._lattice,
                camera=pose.position,
            )
            accumulator = MedianAccumulator(placement.grid, channels=3, capacity=self._capacity)
        else:
            accumulator = self._accumulators[camera_id]

        try:
            patch = sample_frame(
                pose, picture, placement.grid, camera_id=camera_id, captured_at=captured_at
            )
            accumulator.update(patch)
        except OrthophotoError as error:
            raise BasemapError(f"camera {camera_id!r}: {error}") from error

        # Only now, with the sample in the ring, does the frame count and the
        # camera exist here. Counting first meant a frame the sampler refused
        # was reported as fed, under a pose that was never used.
        self._fed[camera_id] = self._fed.get(camera_id, 0) + 1
        self._poses[camera_id] = pose
        if camera_id not in self._placements:
            self._join(camera_id, placement, accumulator)

    def _join(self, camera_id: str, placement: _Placement, accumulator: MedianAccumulator) -> None:
        """Register a placed camera and grow the union to hold it."""
        self._placements[camera_id] = placement
        self._accumulators[camera_id] = accumulator
        grid = placement.grid
        if self._lattice is None:
            self._lattice = grid.origin
            self._grid = grid
            self._grid_at = (placement.row, placement.column)
            _log.info(
                "basemap: grid placed %d×%d cells at %g m for camera %s; the "
                "lattice is anchored at its origin",
                grid.rows, grid.columns, self._cell_size_m, camera_id,
            )
            return

        union, at = self._union()
        if union == self._grid:
            _log.info(
                "basemap: camera %s joins with a %d×%d grid at lattice (%d, %d), "
                "inside the %d×%d union",
                camera_id, grid.rows, grid.columns, placement.row, placement.column,
                union.rows, union.columns,
            )
            return
        assert self._grid is not None
        _log.info(
            "basemap: camera %s joins with a %d×%d grid at lattice (%d, %d); the "
            "union grows from %d×%d to %d×%d cells, and no ring moves",
            camera_id, grid.rows, grid.columns, placement.row, placement.column,
            self._grid.rows, self._grid.columns, union.rows, union.columns,
        )
        self._grid = union
        self._grid_at = at

    def _union(self) -> tuple[GroundGrid, tuple[int, int]]:
        """The smallest grid on the lattice holding every placed grid, and where it sits."""
        assert self._lattice is not None and self._placements
        placements = self._placements.values()
        row = min(placement.row for placement in placements)
        column = min(placement.column for placement in placements)
        rows = max(placement.row + placement.grid.rows for placement in placements) - row
        columns = max(placement.column + placement.grid.columns for placement in placements) - column
        cell = self._cell_size_m
        grid = GroundGrid(
            origin=_frame_for(self._lattice).to_latlon(column * cell, -row * cell),
            cell_size_m=cell,
            columns=columns,
            rows=rows,
        )
        return grid, (row, column)

    def build(self, *, now: float | None = None) -> BasemapAsset:
        """The composite of every camera's median, as an asset.

        Raises :class:`BasemapError` when nothing was fed, when every camera
        fed sees no ground, or when no cell was seen the minimum number of
        times — an asset with no ground in it is not a basemap, and saving one
        would let the plan view draw "nothing here" over a site that simply
        had not been watched long enough.
        """
        if not self._fed:
            raise BasemapError("nothing was fed to this builder, so there is no ground to map")
        if not self._accumulators:
            raise BasemapError(
                "no camera fed to this builder sees any ground: "
                f"{', '.join(self.blind_cameras())}. A basemap needs at least one "
                "camera whose frame reaches the ground."
            )
        grid = self._grid
        assert grid is not None
        union_row, union_column = self._grid_at

        moment = time.time() if now is None else float(now)
        patches = []
        for camera_id, accumulator in sorted(self._accumulators.items()):
            placement = self._placements[camera_id]
            patches.append(_placed(
                accumulator.result(camera_id=camera_id, now=moment), grid,
                row_offset=placement.row - union_row,
                column_offset=placement.column - union_column,
            ))
        composite = mosaic(
            patches, {camera_id: self._pose_sigma_m for camera_id in self._accumulators}
        )
        if composite.covered_cells == 0:
            sampled = ", ".join(
                f"{camera_id} {frames}" for camera_id, frames in sorted(self.frames_sampled().items())
            )
            raise BasemapError(
                f"no cell has been seen {MINIMUM_SAMPLES} times yet (frames sampled: "
                f"{sampled}). The median reports nothing it cannot outvote a "
                "passer-by on; feed more frames before building."
            )

        # Every camera fed, the blind ones included, in id order — and the
        # mosaic's own indices mapped onto that list, so ``source`` resolves
        # against the full provenance rather than against the subset that won.
        cameras = tuple(sorted(self._poses))
        remap = np.full(len(composite.cameras) + 1, -1, dtype=np.int16)
        for index, camera_id in enumerate(composite.cameras):
            remap[index] = cameras.index(camera_id)
        source = np.where(composite.source >= 0, remap[composite.source], np.int16(-1)).astype(np.int16)

        valid = composite.valid.copy()
        colour = np.ascontiguousarray(composite.colour, dtype=np.uint8)
        age = np.where(valid, moment - composite.updated_at, np.inf).astype(np.float32)
        sigma = np.where(valid, composite.sigma_m, np.inf).astype(np.float32)
        built_at_millis = int(round(moment * 1000.0))
        frames_used = sum(accumulator.frames for accumulator in self._accumulators.values())
        poses = dict(self._poses)

        document = _document(
            grid=grid, cameras=cameras, poses=poses, built_at_millis=built_at_millis,
            frames_used=frames_used, valid=valid, source=source, sigma_m=sigma,
            age_seconds=age,
        )
        fingerprint = _fingerprint(_encode_png(colour, valid), document)

        asset = BasemapAsset(
            grid=grid, colour=colour, valid=valid, age_seconds=age, sigma_m=sigma,
            source=source, cameras=cameras, poses=poses, built_at_millis=built_at_millis,
            frames_used=frames_used, fingerprint=fingerprint,
        )
        _log.info("basemap: built — %s", asset.describe())
        return asset


def _place(
    footprint: list[LatLon],
    *,
    cell_size_m: float,
    lattice: LatLon | None,
    camera: LatLon,
) -> _Placement:
    """A grid for this footprint on the lattice, or the grid that anchors one.

    The footprint is measured in one frame — the lattice origin's, or the
    camera's own position's when there is no lattice yet — and its bounds,
    widened by :data:`GRID_MARGIN_M`, are rounded *outward* to whole cells.
    Outward, so the margin is never less than stated; whole cells, so the
    grid's corner is on the lattice and every other grid placed here is a
    whole number of cells away from it. Nothing here is measured twice: the
    frame a footprint is measured in is the frame its grid is placed in, so
    no rounding can act on the disagreement between two frames.

    The frame for the first camera is deliberately not one of its footprint's
    own points, and :meth:`GroundGrid.covering` is not used for the same
    reason. Both put a footprint edge, plus a margin that is a whole number
    of cells, *exactly* on a lattice line — and a second camera on the same
    mast, measured from the lattice origin sixty metres away, found that edge
    half a millimetre to one side of the line (the convergence of meridians
    between the two frames) and was given a grid one cell wider than the
    first: four poses in ten, measured. From the camera's position the
    extremes sit at irrational distances and land mid-cell.

    The first grid placed is at lattice ``(0, 0)`` and its origin becomes the
    lattice origin, which is what a later placement is measured from.
    """
    frame = _frame_for(camera if lattice is None else lattice)
    xs, ys = zip(*(frame.to_xy(point) for point in footprint))
    cell = cell_size_m
    west = math.floor((min(xs) - GRID_MARGIN_M) / cell)
    east = math.ceil((max(xs) + GRID_MARGIN_M) / cell)
    north = math.ceil((max(ys) + GRID_MARGIN_M) / cell)
    south = math.floor((min(ys) - GRID_MARGIN_M) / cell)
    grid = GroundGrid(
        origin=frame.to_latlon(west * cell, north * cell),
        cell_size_m=cell,
        columns=max(1, east - west),
        rows=max(1, north - south),
    )
    if lattice is None:
        return _Placement(grid=grid, row=0, column=0)
    return _Placement(grid=grid, row=-north, column=west)


def _placed(
    patch: GroundPatch,
    grid: GroundGrid,
    *,
    row_offset: int,
    column_offset: int,
) -> GroundPatch:
    """The same patch on the union grid, its arrays copied in at an offset.

    A pure copy through the patch's public fields: no cell is resampled,
    averaged or dropped, and everything outside the window is what an empty
    cell is — ``False``, zero colour, NaN error, NaN time, zero samples — so
    the mosaic reads "no ground" there and not "black ground". The patch's
    own grid is not consulted for position; the lattice offset is the whole
    claim, and the equivalence test in ``test_basemap`` holds it cell for cell.
    """
    if patch.grid == grid:
        # Already there: the single-camera case, where the union is the grid.
        return patch
    rows = slice(row_offset, row_offset + patch.grid.rows)
    columns = slice(column_offset, column_offset + patch.grid.columns)
    if row_offset < 0 or column_offset < 0 or rows.stop > grid.rows or columns.stop > grid.columns:
        raise BasemapError(
            f"a {patch.grid.rows}×{patch.grid.columns} patch at ({row_offset}, "
            f"{column_offset}) does not fit a {grid.rows}×{grid.columns} grid"
        )
    colour = np.zeros((*grid.shape, patch.channels), dtype=patch.colour.dtype)
    valid = np.zeros(grid.shape, dtype=bool)
    sigma = np.full(grid.shape, np.nan, dtype=np.float32)
    updated = np.full(grid.shape, np.nan)
    samples = np.zeros(grid.shape, dtype=np.uint32)
    colour[rows, columns] = patch.colour
    valid[rows, columns] = patch.valid
    sigma[rows, columns] = patch.sigma_m
    updated[rows, columns] = patch.updated_at
    samples[rows, columns] = patch.samples
    return GroundPatch(
        camera_id=patch.camera_id, grid=grid, colour=colour, valid=valid,
        sigma_m=sigma, updated_at=updated, samples=samples,
    )


def _as_bgr(image: np.ndarray) -> np.ndarray:
    """The frame as three-channel uint8 BGR, which is what the ring holds.

    Greyscale and BGRA are converted; anything else is refused rather than
    scaled, because a float frame quietly cast to bytes is a black basemap.
    """
    picture = np.asarray(image)
    if picture.dtype != np.uint8:
        raise BasemapError(
            f"a basemap is built from uint8 frames as the decoder produces them; "
            f"a {picture.dtype} array was given"
        )
    if picture.ndim == 2:
        return cv2.cvtColor(picture, cv2.COLOR_GRAY2BGR)
    if picture.ndim == 3 and picture.shape[2] == 4:
        return cv2.cvtColor(picture, cv2.COLOR_BGRA2BGR)
    if picture.ndim == 3 and picture.shape[2] == 3:
        return picture
    raise BasemapError(
        f"a frame must be (height, width) or (height, width, 3 or 4); "
        f"an array of shape {picture.shape} was given"
    )


# ------------------------------------------------------------------ the files


def basemap_directory(data_dir: Path | None = None) -> Path:
    """Where the site's basemap lives: ``basemap/`` under the data directory.

    Under the data directory like every other artefact this system keeps, and
    overridden with it, so "where is my data" has one answer.
    """
    base = paths.data_directory() if data_dir is None else Path(data_dir)
    return base / "basemap"


def _pose_document(pose: CameraPose) -> dict[str, float]:
    """Every field of a pose, ``roll`` included.

    The store once dropped ``roll`` on the way in and defaulted it on the way
    out, handing back a pose that differed from the one saved. All nine
    fields, always.
    """
    return {
        "lat": float(pose.position.lat),
        "lon": float(pose.position.lon),
        "mount_height": float(pose.mount_height),
        "heading": float(pose.heading),
        "pitch": float(pose.pitch),
        "roll": float(pose.roll),
        "horizontal_fov": float(pose.horizontal_fov),
        "vertical_fov": float(pose.vertical_fov),
        "range_meters": float(pose.range_meters),
    }


def _pose_from(document: Mapping[str, object]) -> CameraPose:
    return CameraPose(
        position=LatLon(float(document["lat"]), float(document["lon"])),
        mount_height=float(document["mount_height"]),
        heading=float(document["heading"]),
        pitch=float(document["pitch"]),
        roll=float(document["roll"]),
        horizontal_fov=float(document["horizontal_fov"]),
        vertical_fov=float(document["vertical_fov"]),
        range_meters=float(document["range_meters"]),
    )


def _pack(array: np.ndarray, dtype: str) -> dict[str, object]:
    """A per-cell layer as bytes the JSON can carry.

    Binary rather than a list of numbers: a quarter-metre raster over the
    reference footprint is 118,374 cells, and three layers of them as JSON
    numbers is several megabytes of text that ``inf`` cannot even be written
    into. Little-endian, compressed, base64 — and every one of those choices is
    named in the document, so a reader that does not recognise them refuses
    rather than decodes noise.
    """
    data = np.ascontiguousarray(array, dtype=np.dtype(dtype))
    return {
        "dtype": dtype,
        "shape": [int(size) for size in data.shape],
        "encoding": "zlib+base64",
        "data": base64.b64encode(zlib.compress(data.tobytes(), 6)).decode("ascii"),
    }


def _unpack(document: Mapping[str, object], dtype: str, shape: tuple[int, int], name: str) -> np.ndarray:
    if document.get("dtype") != dtype or document.get("encoding") != "zlib+base64":
        raise BasemapError(
            f"layer {name!r} is stored as {document.get('dtype')!r}/"
            f"{document.get('encoding')!r}, not {dtype!r}/'zlib+base64'"
        )
    if list(document.get("shape", ())) != list(shape):
        raise BasemapError(
            f"layer {name!r} is shaped {document.get('shape')}, not {list(shape)} like the grid"
        )
    try:
        raw = zlib.decompress(base64.b64decode(str(document["data"])))
    except (ValueError, zlib.error, KeyError) as error:
        raise BasemapError(f"layer {name!r} could not be decoded: {error}") from error
    expected = shape[0] * shape[1] * np.dtype(dtype).itemsize
    if len(raw) != expected:
        raise BasemapError(
            f"layer {name!r} holds {len(raw)} bytes where {expected} were expected"
        )
    return np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(shape).copy()


def _document(
    *,
    grid: GroundGrid,
    cameras: tuple[str, ...],
    poses: Mapping[str, CameraPose],
    built_at_millis: int,
    frames_used: int,
    valid: np.ndarray,
    source: np.ndarray,
    sigma_m: np.ndarray,
    age_seconds: np.ndarray,
) -> dict[str, object]:
    """Everything about the asset but the colour, as plain JSON types.

    Plain Python ints, floats and strings only — a numpy scalar would refuse to
    serialise, and one that slipped through as a float would canonicalise
    differently on load and break the fingerprint of a file nobody touched.
    """
    covered = int(np.count_nonzero(valid))
    return {
        "format": FORMAT,
        "grid": {
            "origin": {"lat": float(grid.origin.lat), "lon": float(grid.origin.lon)},
            "cell_size_m": float(grid.cell_size_m),
            "rows": int(grid.rows),
            "columns": int(grid.columns),
        },
        "cameras": list(cameras),
        "poses": {camera_id: _pose_document(pose) for camera_id, pose in poses.items()},
        "built_at_millis": int(built_at_millis),
        "frames_used": int(frames_used),
        "coverage": {
            "covered_cells": covered,
            "covered_fraction": covered / grid.cell_count,
            "cells_per_camera": {
                camera_id: int(np.count_nonzero(source == index))
                for index, camera_id in enumerate(cameras)
            },
        },
        "layers": {
            "sigma_m": _pack(sigma_m, _SIGMA_DTYPE),
            "age_seconds": _pack(age_seconds, _AGE_DTYPE),
            "source": _pack(source, _SOURCE_DTYPE),
        },
    }


def _document_of(asset: BasemapAsset) -> dict[str, object]:
    return _document(
        grid=asset.grid, cameras=asset.cameras, poses=asset.poses,
        built_at_millis=asset.built_at_millis, frames_used=asset.frames_used,
        valid=asset.valid, source=asset.source, sigma_m=asset.sigma_m,
        age_seconds=asset.age_seconds,
    )


def _canonical(document: Mapping[str, object]) -> bytes:
    """One byte sequence per document, whatever order it was built in."""
    return json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _fingerprint(png: bytes, document: Mapping[str, object]) -> str:
    """SHA-256 over the PNG bytes, a newline, and the canonical JSON.

    The PNG is hashed as bytes rather than as decoded pixels so that the check
    on load is a check of the file that is actually there. The encoder is
    deterministic for the same pixels on the same library, and an asset moved
    to a machine with a different encoder still verifies — it carries its own
    bytes and its own hash of them.
    """
    digest = hashlib.sha256()
    digest.update(png)
    digest.update(b"\n")
    digest.update(_canonical(document))
    return digest.hexdigest()


def _encode_png(colour: np.ndarray, valid: np.ndarray) -> bytes:
    """The colour as a PNG whose alpha channel is the valid mask.

    Alpha carries emptiness so a plain image viewer shows the holes as holes.
    The colour bytes under an empty cell are written as they are — zero, from
    the mosaic — and read back as they are, so the round trip is exact rather
    than "exact where valid".
    """
    if colour.ndim != 3 or colour.shape[2] != 3 or colour.dtype != np.uint8:
        raise BasemapError(
            f"the basemap colour must be (rows, columns, 3) uint8; "
            f"{colour.shape} {colour.dtype} was given"
        )
    if valid.shape != colour.shape[:2]:
        raise BasemapError("the valid mask and the colour are not the same shape")
    alpha = np.where(valid, np.uint8(255), np.uint8(0)).astype(np.uint8)
    bgra = np.dstack([colour, alpha])
    ok, buffer = cv2.imencode(".png", bgra, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise BasemapError("the basemap could not be encoded as PNG")
    return buffer.tobytes()


def _replace_atomically(path: Path, data: bytes) -> None:
    """Write beside the target and rename over it, so a reader sees old or new.

    A crash half-way through a direct write leaves a file that is neither, and
    a PNG cut short decodes as a basemap with its bottom half missing — which
    the plan view would draw as ground nobody covered. The temporary is
    removed on failure so a retry does not find it.
    """
    temporary = path.with_name(path.name + ".tmp")
    try:
        with open(temporary, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def save_basemap(asset: BasemapAsset, directory: Path) -> tuple[Path, Path]:
    """Write ``basemap.png`` and ``basemap.json`` under ``directory``.

    The PNG first, then the JSON, each written whole and renamed into place. The
    pair is not one atomic write — no filesystem here offers that — and the
    fingerprint is what covers the gap: a crash between the two renames leaves
    a new PNG beside the old JSON, which :func:`load_basemap` refuses. The
    asset's own fingerprint is checked against what is about to be written, so
    an asset whose arrays were altered after it was built is refused rather
    than saved under a hash it no longer matches.

    Returns the two paths, PNG first.
    """
    directory = Path(directory)
    png = _encode_png(asset.colour, asset.valid)
    document = _document_of(asset)
    expected = _fingerprint(png, document)
    if expected != asset.fingerprint:
        raise BasemapError(
            "the asset's arrays no longer match its fingerprint; it was altered "
            "after it was built and cannot be saved under that identity"
        )
    document["fingerprint"] = asset.fingerprint

    directory.mkdir(parents=True, exist_ok=True)
    png_path = directory / PNG_NAME
    json_path = directory / JSON_NAME
    _replace_atomically(png_path, png)
    _replace_atomically(
        json_path,
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8") + b"\n",
    )
    _log.info(
        "basemap: saved %s and %s (%d bytes of PNG, fingerprint %s…)",
        png_path.name, json_path.name, len(png), asset.fingerprint[:12],
    )
    return png_path, json_path


def load_basemap(directory: Path) -> BasemapAsset | None:
    """The asset under ``directory``, ``None`` when there is none.

    Raises :class:`BasemapError` when the files are there but do not hash to
    the fingerprint they carry — a PNG edited after the fact, a JSON with a
    number changed, or the half-written pair a crash leaves — or when they are
    not in a shape this reader understands. A basemap that cannot be tied to
    its fingerprint must not be drawn, because the whole point of the
    fingerprint is that "inside the fence" was judged against *this* map.
    """
    directory = Path(directory)
    json_path = directory / JSON_NAME
    png_path = directory / PNG_NAME
    if not json_path.is_file():
        if png_path.exists():
            # No record, so nothing to draw; the orphan is worth a line in the
            # log because it is usually the trace of an interrupted save.
            _log.warning("basemap: %s is present with no %s beside it", png_path, JSON_NAME)
        return None
    if not png_path.is_file():
        raise BasemapError(
            f"{json_path} is present but {PNG_NAME} is missing beside it; the "
            "basemap cannot be drawn. Rebuild it."
        )

    try:
        document = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise BasemapError(f"{json_path} could not be read: {error}") from error
    if not isinstance(document, dict) or document.get("format") != FORMAT:
        raise BasemapError(
            f"{json_path} is not a {FORMAT} document; this build cannot read it"
        )
    fingerprint = document.pop("fingerprint", None)
    if not isinstance(fingerprint, str):
        raise BasemapError(f"{json_path} carries no fingerprint, so it cannot be verified")

    png = png_path.read_bytes()
    actual = _fingerprint(png, document)
    if actual != fingerprint:
        raise BasemapError(
            f"the basemap in {directory} does not match its fingerprint "
            f"({actual[:12]}… on disk, {fingerprint[:12]}… recorded). It was "
            "altered after it was saved, or the save was interrupted. It must "
            "not be drawn; rebuild it."
        )

    try:
        grid_document = document["grid"]
        grid = GroundGrid(
            origin=LatLon(float(grid_document["origin"]["lat"]), float(grid_document["origin"]["lon"])),
            cell_size_m=float(grid_document["cell_size_m"]),
            columns=int(grid_document["columns"]),
            rows=int(grid_document["rows"]),
        )
        cameras = tuple(str(camera_id) for camera_id in document["cameras"])
        poses = {
            str(camera_id): _pose_from(pose) for camera_id, pose in document["poses"].items()
        }
        built_at_millis = int(document["built_at_millis"])
        frames_used = int(document["frames_used"])
        layers = document["layers"]
        sigma = _unpack(layers["sigma_m"], _SIGMA_DTYPE, grid.shape, "sigma_m")
        age = _unpack(layers["age_seconds"], _AGE_DTYPE, grid.shape, "age_seconds")
        source = _unpack(layers["source"], _SOURCE_DTYPE, grid.shape, "source")
    except (KeyError, TypeError, ValueError, AttributeError, OrthophotoError) as error:
        raise BasemapError(f"{json_path} is not in the expected shape: {error}") from error

    if set(poses) != set(cameras):
        raise BasemapError(f"{json_path} lists cameras and poses that do not name the same set")

    bgra = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if bgra is None or bgra.ndim != 3 or bgra.shape[2] != 4 or bgra.dtype != np.uint8:
        raise BasemapError(f"{png_path} is not an 8-bit PNG with an alpha channel")
    if bgra.shape[:2] != grid.shape:
        raise BasemapError(
            f"{png_path} is {bgra.shape[0]}×{bgra.shape[1]} where the grid says "
            f"{grid.rows}×{grid.columns}"
        )
    colour = np.ascontiguousarray(bgra[:, :, :3])
    valid = bgra[:, :, 3] > 0
    if not np.array_equal(valid, source >= 0):
        raise BasemapError(
            f"the alpha channel of {png_path} and the source layer disagree about "
            "which cells are ground"
        )

    return BasemapAsset(
        grid=grid, colour=colour, valid=valid, age_seconds=age, sigma_m=sigma,
        source=source, cameras=cameras, poses=poses, built_at_millis=built_at_millis,
        frames_used=frames_used, fingerprint=fingerprint,
    )
