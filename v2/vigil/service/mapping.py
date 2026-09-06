"""The site's own ground map, built by the cameras that watch it.

# What this is, and what v2 had instead

v1 had 1 974 lines of this — `orthophoto.py` and `basemap.py` — and v2 dropped
all of it. A plan view needs ground under it, and every other way of getting
that ground is unavailable to an offline appliance: a tile server is a network
call, an aerial photograph is a purchase and a download, and a site plan is a
drawing of the site as somebody once intended it rather than as it is. The only
imagery this system is guaranteed to have is its own cameras', and the
projection that turns an image point into a map position inverts.

# Only the ground plane is right, and that is not a caveat to bury

Every pixel is placed **where it would be if it lay on the ground**. A wall, a
parked van, a person: each has height, so each smears radially away from the
camera along the ray that saw it. Anyone reading a single frame's patch as a
photograph reads those smears as ground markings.

Two things stop that being a lie. The **median over many frames** removes
whatever walked through, because a person occupies a cell for a second or two
out of a minute. And the **confidence layer** says, per cell, how much of what
is drawn there is a measurement: a cell whose samples disagreed, or that was
only ever seen down a grazing ray, is marked unusable rather than merely being
blurry.

# Empty is a value

A cell no camera has seen stays empty, and nothing here interpolates one from
its neighbours. Under a security overlay — where an operator judges "inside
the fence" against what is drawn — inventing ground nobody has looked at is
the one thing this must not do. Emptiness lives in the `valid` mask and is
never inferred from the colour, because black is an ordinary colour for
asphalt at dusk.

# One lattice

Every camera samples onto a grid of its own, sized from its own footprint, so
memory grows with what a camera can see rather than with the site. All those
grids are snapped to **one shared lattice** anchored at the site origin, so
compositing is an integer shift and no ring is ever resampled. v1 learned this
the expensive way: an earlier shape of it sized one grid from the first
footprint and *extended* it when a later one fell outside, and re-measuring a
footprint in a second local frame found its edge a fraction of a millimetre
past where it had been placed — which `ceil` made a whole row, so feeding the
same camera a second frame grew the grid and copied thirteen megabytes for
nothing.

# The Rust core is required

`vigil.kernel.native.ortho_sample` has no NumPy path and this module does not
provide one. v1 measured the NumPy version at 79 ms per frame per camera and
sampled four frames a second because of it. Substituting that here would turn
a one-minute build into eighty without saying so, which is exactly the kind of
fallback that makes a system look functional while it is not.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..domain.geo import (
    CameraPose, LatLon, LocalFrame, distance_meters, field_of_view, project_to_ground,
)
from ..kernel import native
from ..logs import get as _get_logger
from ..perception.quality import FrameQualityMonitor

_log = _get_logger(__name__)

#: Default ground cell, metres. A quarter of a metre is finer than the
#: projection error over most of a camera's range and coarse enough that a
#: 120 m footprint is 230 000 cells rather than millions.
DEFAULT_CELL_SIZE_M = 0.25

#: Samples kept per cell. The median needs enough of them to outvote whatever
#: walked through, and the memory is `cells * depth * 3` bytes: 10 MB per
#: camera at the default cell and a 120 m footprint.
DEFAULT_DEPTH = 15

#: Fewest samples before a cell is drawn at all. One sample is not a median —
#: it is one frame, with whoever was walking through it still in it.
MIN_SAMPLES = 5

#: Seconds between samples fed to one camera's accumulator.
#:
#: The median needs *time* between samples, not frames: a hundred frames of
#: the same half-second fill the ring with one moment and the person standing
#: in it survives the median. A quarter of a second is fast enough to fill a
#: ring in four seconds and slow enough that a walker has left the cell.
DEFAULT_SAMPLE_INTERVAL_S = 0.25

#: Ground metres per source pixel past which a cell's colour is a smear of a
#: handful of pixels stretched across the ground rather than a measurement of
#: it. A quarter-metre cell built from pixels each covering a fifth of a metre
#: is at the edge of meaning something.
MAX_USABLE_RESOLUTION_M = 0.20

#: Share of samples that may disagree with a cell's median before the cell is
#: not ground. A doorway, a parking space or a wall smearing differently as
#: the light moves all show up here.
MAX_USABLE_DISTURBANCE = 0.35

#: Position error past which a cell cannot be adjudicated against a zone. Half
#: a metre distinguishes one side of a doorway from the other; two metres does
#: not distinguish one side of a small yard from the other.
MAX_USABLE_UNCERTAINTY_M = 2.0

#: Confidence below which a cell is reported as unusable.
USABLE_CONFIDENCE = 0.35


class MappingError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Grid:
    """A rectangle of ground cells on one shared lattice.

    `origin` is the south-west corner. Row 0 is the **north** edge, so the
    raster draws the right way up without anybody flipping it on the way to a
    screen — a flip that gets forgotten once and puts a whole site's map upside
    down under its zones.
    """

    origin: LatLon
    cell_size_m: float
    rows: int
    cols: int
    #: Cell offset of this grid's origin from the site lattice origin, so two
    #: grids composite by an integer shift and never by a resample.
    east_cells: int = 0
    north_cells: int = 0

    @property
    def cells(self) -> int:
        return self.rows * self.cols

    def values(self) -> np.ndarray:
        return native.grid_values(self.origin.lat, self.origin.lon, self.cell_size_m, self.rows, self.cols)

    def centre_of(self, row: int, col: int) -> LatLon:
        frame = LocalFrame(self.origin)
        from ..domain.geo import Vec2

        return frame.to_lat_lon(Vec2((col + 0.5) * self.cell_size_m,
                                     (self.rows - 0.5 - row) * self.cell_size_m))

    def distances_from(self, point: LatLon) -> np.ndarray:
        """Ground distance from `point` to every cell centre, metres.

        Vectorised in the grid's own local frame rather than one geodesic per
        cell: at 230 000 cells the loop is the whole cost of the confidence
        layer.
        """
        frame = LocalFrame(self.origin)
        local = frame.to_local(point)
        east = (np.arange(self.cols) + 0.5) * self.cell_size_m - local.x
        north = (self.rows - 0.5 - np.arange(self.rows)) * self.cell_size_m - local.y
        return np.hypot(east[None, :], north[:, None])


def site_lattice(origin: LatLon, cell_size_m: float, pose: CameraPose, *,
                 margin_m: float = 2.0) -> Grid:
    """A grid covering one camera's footprint, snapped to the site lattice.

    Snapped, not merely sized: two grids whose origins are an integer number
    of cells apart composite by a shift, and two that are not have to be
    resampled — which blurs a map that is already only as sharp as its
    projection.
    """
    ring = field_of_view(pose, 48)
    if not ring:
        raise MappingError(f"a camera looking at {pose.pitch:.0f}° sees no ground, so it maps none of it")
    frame = LocalFrame(origin)
    locals_ = [frame.to_local(p) for p in ring]
    east = [p.x for p in locals_]
    north = [p.y for p in locals_]
    west_cell = math.floor((min(east) - margin_m) / cell_size_m)
    south_cell = math.floor((min(north) - margin_m) / cell_size_m)
    east_cell = math.ceil((max(east) + margin_m) / cell_size_m)
    north_cell = math.ceil((max(north) + margin_m) / cell_size_m)
    cols = max(1, east_cell - west_cell)
    rows = max(1, north_cell - south_cell)
    from ..domain.geo import Vec2

    corner = frame.to_lat_lon(Vec2(west_cell * cell_size_m, south_cell * cell_size_m))
    return Grid(corner, cell_size_m, rows, cols, west_cell, south_cell)


def uncertainty_over(pose: CameraPose, distances: np.ndarray) -> np.ndarray:
    """1-sigma position error at each ground distance, metres.

    Sampled down the frame's centre column and interpolated by distance, which
    is an approximation and worth naming as one: the error also varies
    *across* the frame, by up to about 15% at the corners of a wide lens. The
    range term dominates it by an order of magnitude — error grows as the
    square of range through the depression angle — and the alternative is a
    finite-difference Jacobian per cell, which is 230 000 of them.
    """
    samples: list[tuple[float, float]] = []
    for i in range(65):
        v = 1.0 - i / 64.0
        projection = project_to_ground(pose, 0.5, v, enforce_range=False)
        if projection is not None:
            samples.append((projection.ground_distance_meters, projection.uncertainty.radius_meters))
    if not samples:
        return np.full(distances.shape, np.inf)
    samples.sort()
    xs = np.array([s[0] for s in samples])
    ys = np.array([s[1] for s in samples])
    # Outside the sampled span the answer is "worse than the worst measured",
    # not the nearest value: `np.interp` clamps, and a clamp here would report
    # the far edge's error for ground twice as far away.
    out = np.interp(distances, xs, ys)
    out[distances > xs[-1]] = np.inf
    return out


@dataclass(frozen=True, slots=True)
class GroundMap:
    """A finished map and everything needed to argue about it."""

    grid: Grid
    #: BGR, `(rows, cols, 3)`. Undefined where `valid` is 0.
    colour: np.ndarray
    #: 1 where a camera saw the cell often enough to take a median.
    valid: np.ndarray
    #: 0..1 per cell. See `confidence_of` for what goes into it.
    confidence: np.ndarray
    #: Ground metres per source pixel; how sharp the texture there is.
    resolution: np.ndarray
    #: 1-sigma position error of the projection that placed the cell.
    uncertainty: np.ndarray
    #: Share of samples that disagreed with the median: how often something
    #: was in the way.
    disturbance: np.ndarray
    #: Which camera won each cell, as an index into `cameras`. -1 for empty.
    source: np.ndarray
    cameras: tuple[str, ...] = ()
    poses: dict[str, CameraPose] = field(default_factory=dict)

    @property
    def usable(self) -> np.ndarray:
        """Cells whose colour is a measurement of the ground rather than a
        picture of something that happened to be over it."""
        return (self.valid > 0) & (self.confidence >= USABLE_CONFIDENCE)

    def summary(self) -> dict[str, float | int]:
        seen = int((self.valid > 0).sum())
        usable = int(self.usable.sum())
        return {
            "cells": self.grid.cells,
            "seen": seen,
            "usable": usable,
            "covered_m2": round(seen * self.grid.cell_size_m ** 2, 1),
            "usable_m2": round(usable * self.grid.cell_size_m ** 2, 1),
            "mean_confidence": round(float(self.confidence[self.valid > 0].mean()), 3) if seen else 0.0,
            "median_resolution_m": (round(float(np.median(self.resolution[self.valid > 0])), 3)
                                    if seen else 0.0),
        }

    def describe(self) -> str:
        s = self.summary()
        if not s["seen"]:
            return "no ground was mapped"
        return (f"{s['usable_m2']:.0f} m² usable of {s['covered_m2']:.0f} m² seen "
                f"({s['usable'] / max(1, s['seen']):.0%}), "
                f"median {s['median_resolution_m']:.2f} m per source pixel")


def confidence_of(samples: np.ndarray, disturbance: np.ndarray, resolution: np.ndarray,
                  uncertainty: np.ndarray, depth: int) -> np.ndarray:
    """How much of a cell's colour is a measurement, 0..1.

    Four independent ways a cell can be worthless, multiplied rather than
    averaged: a cell seen sharply, from a well-known pose, thirty times, with
    a lorry parked over it half of them, is *not* two-thirds trustworthy. Each
    term is a probability that this particular objection does not apply, and
    they apply independently.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        # Enough looks to outvote a passer-by. Saturating: the tenth sample
        # adds much less than the third.
        support = np.clip(samples / max(1.0, min(depth, MIN_SAMPLES * 2)), 0.0, 1.0)
        # How often something was over it.
        undisturbed = np.clip(1.0 - disturbance / MAX_USABLE_DISTURBANCE, 0.0, 1.0)
        # How sharp the texture is.
        sharp = np.clip(MAX_USABLE_RESOLUTION_M / np.maximum(resolution, 1e-6), 0.0, 1.0)
        # How well the cell's *position* is known, which is a different
        # question from how it looks and is the one a zone judgement needs.
        placed = np.clip(MAX_USABLE_UNCERTAINTY_M / np.maximum(uncertainty, 1e-6), 0.0, 1.0)
    out = support * undisturbed * sharp * placed
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class _CameraMap:
    """One camera's grid, accumulator and scratch buffers."""

    def __init__(self, camera_id: str, pose: CameraPose, grid: Grid, depth: int):
        self.camera_id = camera_id
        self.pose = pose
        self.grid = grid
        self.depth = depth
        self.frames = 0
        self.last_sample_at: float | None = None
        cells = grid.cells
        self._grid_values = grid.values()
        self._colour = np.zeros(cells * 3, dtype=np.uint8)
        self._valid = np.zeros(cells, dtype=np.uint8)
        self._resolution = np.full(cells, np.inf, dtype=np.float32)
        #: The best resolution ever seen for each cell, kept across frames: a
        #: camera does not move, so a cell's sharpness is a property of the
        #: geometry rather than of the frame, and one frame where a lattice
        #: triangle fell differently should not make the map claim less than
        #: it knows.
        self._best_resolution = np.full(cells, np.inf, dtype=np.float32)
        self._accumulator = native.MedianAccumulator(cells, depth)

    def observe(self, image: np.ndarray) -> int:
        filled = native.ortho_sample(self.pose, self._grid_values, image, self.grid.rows,
                                     self.grid.cols, native_lattice(), self._colour, self._valid,
                                     self._resolution)
        if filled:
            self._accumulator.add(self._colour, self._valid)
            np.minimum(self._best_resolution, self._resolution, out=self._best_resolution)
            self.frames += 1
        return filled

    def result(self, minimum_samples: int):
        cells = self.grid.cells
        colour = np.zeros(cells * 3, dtype=np.uint8)
        valid = np.zeros(cells, dtype=np.uint8)
        samples = np.zeros(cells, dtype=np.uint16)
        deviation = np.zeros(cells, dtype=np.uint8)
        disturbed = np.zeros(cells, dtype=np.uint8)
        self._accumulator.result(minimum_samples, colour, valid, samples, deviation, disturbed)
        shape = (self.grid.rows, self.grid.cols)
        return (
            colour.reshape(*shape, 3),
            valid.reshape(shape),
            samples.reshape(shape).astype(np.float32),
            disturbed.reshape(shape).astype(np.float32) / 100.0,
            self._best_resolution.reshape(shape).copy(),
        )

    def close(self) -> None:
        self._accumulator.close()


def native_lattice() -> int:
    """Points per axis in the image lattice the core inverts. See
    `core/src/ortho.rs` for why 33."""
    return 33


class MapBuilder:
    """Feed it frames; ask it for a map.

    One accumulator per camera, all on one lattice. Frames are *rate limited*
    per camera rather than taken as they arrive: see `DEFAULT_SAMPLE_INTERVAL_S`
    for why a hundred frames of the same half-second is one sample's worth of
    information.
    """

    def __init__(self, origin: LatLon, *, cell_size_m: float = DEFAULT_CELL_SIZE_M,
                 depth: int = DEFAULT_DEPTH, sample_interval_s: float = DEFAULT_SAMPLE_INTERVAL_S):
        if not native.available():
            raise MappingError(
                f"building a map needs the engine core, which is not loaded: {native.fault()}. "
                "There is no NumPy path for this on purpose — v1 measured one at 79 ms per frame "
                "per camera, and quietly running eighty times slower is not a fallback."
            )
        self.origin = origin
        self.cell_size_m = cell_size_m
        self.depth = depth
        self.sample_interval_s = sample_interval_s
        self._cameras: dict[str, _CameraMap] = {}

    def __enter__(self) -> "MapBuilder":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def cameras(self) -> tuple[str, ...]:
        return tuple(sorted(self._cameras))

    def frames_for(self, camera_id: str) -> int:
        entry = self._cameras.get(camera_id)
        return entry.frames if entry else 0

    def observe(self, camera_id: str, pose: CameraPose, image: np.ndarray,
                at_seconds: float | None = None) -> bool:
        """Fold one frame in. False when it was skipped by the rate limit."""
        entry = self._cameras.get(camera_id)
        if entry is None:
            entry = _CameraMap(camera_id, pose, site_lattice(self.origin, self.cell_size_m, pose), self.depth)
            self._cameras[camera_id] = entry
            _log.info("mapping %s: %d x %d cells at %.2f m", camera_id, entry.grid.rows,
                      entry.grid.cols, self.cell_size_m)
        elif entry.pose != pose:
            # A camera that has been re-placed is a different observer. Its
            # accumulated samples were projected through the old pose and
            # blending them with the new ones would draw the ground twice.
            entry.close()
            entry = _CameraMap(camera_id, pose, site_lattice(self.origin, self.cell_size_m, pose), self.depth)
            self._cameras[camera_id] = entry
            _log.info("mapping %s: the pose changed, so its samples were discarded", camera_id)
        if at_seconds is not None and entry.last_sample_at is not None:
            if at_seconds - entry.last_sample_at < self.sample_interval_s:
                return False
        entry.last_sample_at = at_seconds
        entry.observe(image)
        return True

    def build(self, *, minimum_samples: int = MIN_SAMPLES) -> GroundMap:
        """Composite every camera onto one grid.

        Where two cameras see a cell, the one whose claim is better wins
        outright rather than the two being averaged. Averaging two projections
        of the same wall from different angles produces a smear that is a
        measurement of neither, and there is no way to tell afterwards that it
        happened.
        """
        if not self._cameras:
            raise MappingError("no camera has contributed a frame")
        union = self._union_grid()
        shape = (union.rows, union.cols)
        colour = np.zeros((*shape, 3), dtype=np.uint8)
        valid = np.zeros(shape, dtype=np.uint8)
        confidence = np.zeros(shape, dtype=np.float32)
        resolution = np.full(shape, np.inf, dtype=np.float32)
        uncertainty = np.full(shape, np.inf, dtype=np.float32)
        disturbance = np.zeros(shape, dtype=np.float32)
        source = np.full(shape, -1, dtype=np.int16)

        names = self.cameras
        for index, camera_id in enumerate(names):
            entry = self._cameras[camera_id]
            patch_colour, patch_valid, samples, disturbed, patch_resolution = entry.result(minimum_samples)
            distances = entry.grid.distances_from(entry.pose.position)
            patch_uncertainty = uncertainty_over(entry.pose, distances).astype(np.float32)
            patch_confidence = confidence_of(samples, disturbed, patch_resolution,
                                             patch_uncertainty, entry.depth)
            rows, cols = self._offset(union, entry.grid)
            window = (slice(rows, rows + entry.grid.rows), slice(cols, cols + entry.grid.cols))
            better = (patch_valid > 0) & (patch_confidence > confidence[window])
            colour[window][better] = patch_colour[better]
            valid[window][better] = 1
            confidence[window][better] = patch_confidence[better]
            resolution[window][better] = patch_resolution[better]
            uncertainty[window][better] = patch_uncertainty[better]
            disturbance[window][better] = disturbed[better]
            source[window][better] = index

        return GroundMap(union, colour, valid, confidence, resolution, uncertainty, disturbance,
                         source, names, {c: self._cameras[c].pose for c in names})

    def close(self) -> None:
        for entry in self._cameras.values():
            entry.close()
        self._cameras.clear()

    # ------------------------------------------------------------ internals

    def _union_grid(self) -> Grid:
        grids = [e.grid for e in self._cameras.values()]
        west = min(g.east_cells for g in grids)
        south = min(g.north_cells for g in grids)
        east = max(g.east_cells + g.cols for g in grids)
        north = max(g.north_cells + g.rows for g in grids)
        from ..domain.geo import Vec2

        corner = LocalFrame(self.origin).to_lat_lon(
            Vec2(west * self.cell_size_m, south * self.cell_size_m))
        return Grid(corner, self.cell_size_m, north - south, east - west, west, south)

    @staticmethod
    def _offset(union: Grid, patch: Grid) -> tuple[int, int]:
        """Where a camera's grid sits in the union, in whole cells.

        Whole by construction: both were snapped to the same lattice, so this
        is a subtraction and never a rounding. If it ever needed rounding the
        two grids would be misaligned and the composite would be blurred by
        half a cell everywhere.
        """
        cols = patch.east_cells - union.east_cells
        # Row 0 is north, so a patch whose north edge is below the union's
        # starts further down.
        rows = (union.north_cells + union.rows) - (patch.north_cells + patch.rows)
        return rows, cols


# ------------------------------------------------------- from live cameras


@dataclass(frozen=True, slots=True)
class BuildReport:
    """What a build actually managed to do, per camera."""

    ground: GroundMap
    samples: dict[str, int]
    skipped: dict[str, int]
    faults: dict[str, str]


def build_from_cameras(sources: dict[str, tuple[str, CameraPose]], origin: LatLon, *,
                       seconds: float, cell_size_m: float = DEFAULT_CELL_SIZE_M,
                       depth: int = DEFAULT_DEPTH, minimum_samples: int = MIN_SAMPLES,
                       progress=None) -> BuildReport:
    """Watch some cameras for a while and build the map they can see.

    Here rather than in the command because an interface does not open a
    camera — the layering rule says so, and the reason it says so is that a
    capture loop in a CLI is a capture loop nothing else can reuse and no test
    can run without a camera.

    `sources` maps a camera id to `(url with credentials, pose)`. The url
    never leaves this function; nothing about it is logged, printed or
    written into the map.
    """
    from ..adapters.decode import DecodeError, LiveReader, VideoSource

    samples: dict[str, int] = {}
    skipped: dict[str, int] = {}
    faults: dict[str, str] = {}
    opened: list[tuple[str, VideoSource, LiveReader | None, FrameQualityMonitor, CameraPose]] = []
    started = time.monotonic()
    try:
        with MapBuilder(origin, cell_size_m=cell_size_m, depth=depth) as builder:
            for camera_id, (url, pose) in sorted(sources.items()):
                source = VideoSource(url, source_id=camera_id)
                try:
                    source.open()
                except DecodeError as error:
                    faults[camera_id] = str(error)
                    continue
                reader = None
                if source.live:
                    reader = LiveReader(source)
                    reader.start()
                opened.append((camera_id, source, reader, FrameQualityMonitor(), pose))
            if not opened:
                raise MappingError("no camera could be opened: "
                                   + "; ".join(f"{k}: {v}" for k, v in faults.items()))
            while time.monotonic() - started < seconds:
                for camera_id, source, reader, quality, pose in opened:
                    frame = reader.read(timeout=0.2) if reader is not None else source.read()
                    if frame is None:
                        continue
                    # A frame nothing can be detected in is a frame nothing
                    # should be mapped from either: it passes the median's own
                    # vote and leaves a blurred smear on the ground. A frame
                    # identical to the last one is skipped too, for the
                    # opposite reason — it adds nothing a median did not have.
                    if not quality.measure(frame.image).worth_sampling:
                        skipped[camera_id] = skipped.get(camera_id, 0) + 1
                        continue
                    builder.observe(camera_id, pose, frame.image, at_seconds=time.monotonic())
                if progress is not None:
                    progress(time.monotonic() - started, seconds)
                time.sleep(0.02)
            for camera_id, *_rest in opened:
                samples[camera_id] = builder.frames_for(camera_id)
            ground = builder.build(minimum_samples=minimum_samples)
    finally:
        for _id, source, reader, _q, _p in opened:
            if reader is not None:
                reader.stop()
            source.close()
    return BuildReport(ground, samples, skipped, faults)


# ---------------------------------------------------------------- on disk


def _canonical(document: dict) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")


def fingerprint(colour_png: bytes, document: dict) -> str:
    """The identity of a map, over its pixels *and* its metadata.

    An incident judged "inside the fence" against a map has to be re-checkable
    against *that* map, not against whatever is in the folder by then. A
    tampered file and a half-written one look identical to this check, and
    neither should be drawn.
    """
    digest = hashlib.sha256()
    digest.update(colour_png)
    digest.update(_canonical(document))
    return digest.hexdigest()


def save_map(ground: GroundMap, directory: Path) -> Path:
    """Write the map and its provenance. No frame is ever written.

    A map is the *empty* site by construction — the median has removed whoever
    walked through — and keeping the frames that fed it would keep the people
    the median removed. The camera sources are not written either, not even
    redacted: a camera URL carries a credential, and the map is the one
    artefact meant to be copied about freely.
    """
    import cv2

    directory.mkdir(parents=True, exist_ok=True)
    layers = np.dstack([
        ground.colour,
        (ground.valid * 255).astype(np.uint8),
    ])
    ok, encoded = cv2.imencode(".png", layers)
    if not ok:
        raise MappingError("the map could not be encoded")
    png = encoded.tobytes()
    document = {
        "cell_size_m": ground.grid.cell_size_m,
        "rows": ground.grid.rows,
        "cols": ground.grid.cols,
        "origin": [ground.grid.origin.lat, ground.grid.origin.lon],
        "east_cells": ground.grid.east_cells,
        "north_cells": ground.grid.north_cells,
        "cameras": list(ground.cameras),
        "poses": {
            name: {
                "lat": pose.position.lat, "lon": pose.position.lon,
                "mount_height": pose.mount_height, "heading": pose.heading,
                "pitch": pose.pitch, "roll": pose.roll,
                "horizontal_fov": pose.horizontal_fov, "vertical_fov": pose.vertical_fov,
                "range_meters": pose.range_meters,
            }
            for name, pose in ground.poses.items()
        },
        "summary": ground.summary(),
    }
    document["fingerprint"] = fingerprint(png, document)
    (directory / "ground.png").write_bytes(png)
    # The measured layers travel as a compressed array rather than in the PNG:
    # they are float and int16, and squeezing them into 8-bit channels to keep
    # one file would throw away the precision the confidence is made of.
    np.savez_compressed(
        directory / "ground.npz",
        confidence=ground.confidence, resolution=ground.resolution,
        uncertainty=ground.uncertainty, disturbance=ground.disturbance, source=ground.source,
    )
    (directory / "ground.json").write_text(json.dumps(document, indent=2), encoding="utf-8")
    _log.info("map written to %s: %s", directory, ground.describe())
    return directory / "ground.json"


def load_map(directory: Path) -> GroundMap | None:
    """The map, or `None` when there is not one that hashes to its own claim."""
    import cv2

    manifest = directory / "ground.json"
    if not manifest.is_file():
        return None
    document = json.loads(manifest.read_text(encoding="utf-8"))
    claimed = document.pop("fingerprint", None)
    png = (directory / "ground.png").read_bytes()
    if claimed != fingerprint(png, document):
        _log.error("the map at %s does not hash to its own fingerprint and will not be drawn; "
                   "it has been edited, truncated, or written by a different build", directory)
        return None
    layers = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if layers is None or layers.ndim != 3 or layers.shape[2] != 4:
        return None
    measured = np.load(directory / "ground.npz")
    grid = Grid(LatLon(*document["origin"]), document["cell_size_m"], document["rows"],
                document["cols"], document["east_cells"], document["north_cells"])
    poses = {
        name: CameraPose(LatLon(p["lat"], p["lon"]), p["mount_height"], p["heading"], p["pitch"],
                         p["roll"], p["horizontal_fov"], p["vertical_fov"], p["range_meters"])
        for name, p in document.get("poses", {}).items()
    }
    return GroundMap(
        grid, np.ascontiguousarray(layers[:, :, :3]), (layers[:, :, 3] > 0).astype(np.uint8),
        measured["confidence"], measured["resolution"], measured["uncertainty"],
        measured["disturbance"], measured["source"], tuple(document.get("cameras", ())), poses,
    )


def visualise(ground: GroundMap) -> np.ndarray:
    """The map as a picture, with the unusable ground shown as unusable.

    Cells nobody has seen are a dark checker rather than black, because black
    is a perfectly ordinary colour for asphalt at dusk and a renderer that
    paints emptiness as a colour teaches an operator to read holes as tarmac.
    Cells that were seen but cannot be trusted are desaturated and dimmed:
    visible enough to give context, obviously not a measurement.
    """
    rows, cols = ground.valid.shape
    checker = (((np.arange(rows)[:, None] // 4) + (np.arange(cols)[None, :] // 4)) % 2)
    out = np.where(checker[..., None] > 0, 28, 20).astype(np.uint8).repeat(3, axis=2)

    seen = ground.valid > 0
    usable = ground.usable
    doubtful = seen & ~usable
    if doubtful.any():
        grey = ground.colour[doubtful].mean(axis=1, keepdims=True)
        out[doubtful] = (grey * 0.45).astype(np.uint8)
    out[usable] = ground.colour[usable]
    return out
