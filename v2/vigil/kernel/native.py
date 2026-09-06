"""The Rust core, through ctypes.

This is the only module in Python that knows the core is a shared library.
Everything above it works with NumPy arrays and dataclasses and never sees a
pointer.

# What is in Rust and what is not

The split is by measurement, not by taste. Three things are here because
Python is genuinely too slow for them at the rate they run:

- **Ground rasterisation** (`ortho_sample`). A quarter-metre grid over a
  60 m footprint is 230 400 cells and 1 089 lattice projections *per frame per
  camera*. v1 did it in NumPy and measured 79 ms, which is why v1 sampled four
  frames a second and called that a design. Measured here at 0.62 ms.
- **Rectangular assignment** (`assign`). Optimal association is O(n^3), SciPy
  is not a dependency of an offline appliance, and a Python triple loop at
  15 fps across sixteen cameras is not one either.
- **The Kalman filter** (`kalman_*`). Eight-state predict and update per track
  per frame. NumPy pays more in call overhead on an 8x8 than the arithmetic
  costs; a fixed-size Rust routine does not.

Per-detection geometry is *not* here. `vigil.domain.geo` is the readable
implementation of the same model and it costs microseconds, so having one
implementation everybody can read beats saving them. Where the two do overlap
— `camera.rs` has to have its own copy of the projection because
`ortho_sample` cannot cross the boundary a thousand times per frame —
`tests/test_native.py` drives both over a grid of poses and holds them to
1e-9. Two implementations are safe exactly as long as something proves they
agree.

# When the library is missing

`available()` says so and `vigil doctor` reports it. Nothing silently
substitutes a slower path and then reports the same numbers: the mapping
service refuses to build a basemap without it, because a Python
implementation of that loop would take eighty times as long and quietly turn a
one-minute build into eighty. The tracker and the assignment do have NumPy
fallbacks, they are the same algorithm, and `tests/test_native.py` holds them
to the Rust results — those are chosen deliberately per function and named
here rather than being a blanket "try native, except: pass".
"""

from __future__ import annotations

import math
import ctypes
import os
import sys
import threading
from pathlib import Path

import numpy as np

from ..logs import get as _get_logger
from . import filtering as _filtering
# Re-exported: these are `native`'s own fallbacks, and the tests that hold
# them to the core address them here.
from .fallbacks import (  # noqa: F401
    MAX_GAP_M, MIN_PARALLAX_DEG, _assign_numpy, _fit_plane_numpy, _nms_numpy,
    _soft_nms_numpy, _suppress_numpy, _triangulate_numpy,
)

_log = _get_logger(__name__)

#: The ABI this binding was written against. A library reporting anything else
#: is refused rather than called: a signature that moved underneath produces
#: plausible, wrong geometry rather than a crash.
#:
#: **2** was the pose growing its lens. A core built for ABI 1 would read
#: fourteen values where nine were sent, and the five it invented would be
#: whatever was next in memory — a lens made of stack garbage applied to every
#: ray. **3** added triangulation and the ground-plane fit; against an older
#: core those symbols are simply absent, and one refusal at load beats an
#: `AttributeError` from inside a frame loop. **4** grew the pose again by the
#: two ground tilts, which is the dangerous kind of change: fourteen values
#: sent where sixteen are read gives the core a ground plane tilted by
#: whatever was next in memory. **5** adds detection suppression.
ABI_VERSION = 5

#: Array lengths the ABI promises,
#: `[pose, sigma, projection, grid, kalman, triangulation, plane]`.
EXPECTED_LAYOUT = (16, 5, 6, 5, 72, 8, 8)

(POSE_VALUES, SIGMA_VALUES, PROJECTION_VALUES, GRID_VALUES, KALMAN_VALUES,
 TRIANGULATION_VALUES, PLANE_VALUES) = EXPECTED_LAYOUT

#: Override for a deployment that keeps the library somewhere of its own.
LIBRARY_VARIABLE = "VIGIL_CORE_PATH"


class NativeError(RuntimeError):
    """The core is present but refused the call, or is not the right core."""


def _library_name() -> str:
    if sys.platform == "win32":
        return "vigil_core.dll"
    if sys.platform == "darwin":
        return "libvigil_core.dylib"
    return "libvigil_core.so"


def library_candidates() -> list[Path]:
    """Every place the core may be, in order.

    More than one on purpose, and it is the same lesson `Settings.model_directories`
    learned: a packaged build ships it beside the executable, a checkout has it
    under `core/target/release` where cargo puts it, and a deployment may put
    it with its data. Looking in only one of those is how a run silently loses
    its fast path.
    """
    name = _library_name()
    places: list[Path] = []
    override = os.environ.get(LIBRARY_VARIABLE, "").strip()
    if override:
        places.append(Path(override))
    if getattr(sys, "frozen", False):
        places.append(Path(sys.executable).resolve().parent / name)
        places.append(Path(sys.executable).resolve().parent / "_internal" / name)
    here = Path(__file__).resolve().parent.parent.parent  # v2/
    places.append(here / "core" / "target" / "release" / name)
    places.append(here / name)
    places.append(here / "vigil" / name)
    seen: set[Path] = set()
    ordered: list[Path] = []
    for place in places:
        if place not in seen:
            seen.add(place)
            ordered.append(place)
    return ordered


_LOCK = threading.Lock()
_LIBRARY: ctypes.CDLL | None = None
_LOAD_FAULT: str | None = None
_LOADED_FROM: Path | None = None


def _declare(library: ctypes.CDLL) -> None:
    """Argument and return types for every entry point.

    Declared rather than left to ctypes' defaults, because the default return
    type is `int` and a function that returns a `double` then hands back the
    low half of a register as an integer — silently, and only on some
    platforms.
    """
    f64 = ctypes.c_double
    p64 = ctypes.POINTER(ctypes.c_double)
    pu8 = ctypes.POINTER(ctypes.c_uint8)
    pf32 = ctypes.POINTER(ctypes.c_float)
    pu16 = ctypes.POINTER(ctypes.c_uint16)
    pi32 = ctypes.POINTER(ctypes.c_int32)
    pi64 = ctypes.POINTER(ctypes.c_int64)
    u32 = ctypes.c_uint32

    library.vigil_abi_version.restype = ctypes.c_uint32
    library.vigil_abi_version.argtypes = []
    library.vigil_layout.restype = ctypes.c_int32
    library.vigil_layout.argtypes = [ctypes.POINTER(ctypes.c_uint32), u32]

    library.vigil_distance_meters.restype = f64
    library.vigil_distance_meters.argtypes = [f64, f64, f64, f64]
    library.vigil_bearing_degrees.restype = f64
    library.vigil_bearing_degrees.argtypes = [f64, f64, f64, f64]
    library.vigil_destination_point.restype = ctypes.c_int32
    library.vigil_destination_point.argtypes = [f64, f64, f64, f64, p64]

    library.vigil_project_batch.restype = ctypes.c_int32
    library.vigil_project_batch.argtypes = [p64, p64, u32, f64, p64, u32, p64, pi32]
    library.vigil_image_coordinates_batch.restype = ctypes.c_int32
    library.vigil_image_coordinates_batch.argtypes = [p64, p64, u32, p64, pi32]
    library.vigil_footprint.restype = ctypes.c_int32
    library.vigil_footprint.argtypes = [p64, u32, p64, u32]
    library.vigil_ground_span.restype = ctypes.c_int32
    library.vigil_ground_span.argtypes = [p64, p64]

    library.vigil_assign.restype = ctypes.c_int32
    library.vigil_assign.argtypes = [p64, u32, u32, pi64]

    library.vigil_triangulate.restype = ctypes.c_int32
    library.vigil_triangulate.argtypes = [p64, p64, p64, p64, f64, f64, p64]
    library.vigil_fit_plane.restype = ctypes.c_int32
    library.vigil_fit_plane.argtypes = [p64, u32, f64, u32, ctypes.c_uint64, p64]
    library.vigil_suppress.restype = ctypes.c_int32
    library.vigil_suppress.argtypes = [p64, p64, pi64, u32, f64, u32, f64, f64, pi64]

    library.vigil_kalman_initiate.restype = ctypes.c_int32
    library.vigil_kalman_initiate.argtypes = [p64, f64, f64, f64, f64]
    library.vigil_kalman_predict.restype = ctypes.c_int32
    library.vigil_kalman_predict.argtypes = [p64, f64]
    library.vigil_kalman_update.restype = ctypes.c_int32
    library.vigil_kalman_update.argtypes = [p64, f64, f64, f64, f64]
    library.vigil_kalman_gate.restype = f64
    library.vigil_kalman_gate.argtypes = [p64, f64, f64, f64, f64, u32]
    library.vigil_kalman_gate_batch.restype = ctypes.c_int32
    library.vigil_kalman_gate_batch.argtypes = [p64, p64, u32, u32, p64]
    library.vigil_kalman_warp.restype = ctypes.c_int32
    library.vigil_kalman_warp.argtypes = [p64, p64]

    library.vigil_ortho_sample.restype = ctypes.c_int64
    library.vigil_ortho_sample.argtypes = [p64, p64, pu8, u32, u32, u32, u32, pu8, pu8, pf32]
    library.vigil_median_create.restype = ctypes.c_void_p
    library.vigil_median_create.argtypes = [u32, u32]
    library.vigil_median_destroy.restype = None
    library.vigil_median_destroy.argtypes = [ctypes.c_void_p]
    library.vigil_median_add.restype = ctypes.c_int32
    library.vigil_median_add.argtypes = [ctypes.c_void_p, pu8, pu8, u32]
    library.vigil_median_result.restype = ctypes.c_int64
    library.vigil_median_result.argtypes = [
        ctypes.c_void_p, u32, u32, pu8, pu8, pu16, pu8, pu8,
    ]


def load() -> ctypes.CDLL | None:
    """The core, or `None` with the reason in `fault()`. Loaded once."""
    global _LIBRARY, _LOAD_FAULT, _LOADED_FROM
    with _LOCK:
        if _LIBRARY is not None or _LOAD_FAULT is not None:
            return _LIBRARY
        tried: list[str] = []
        for candidate in library_candidates():
            if not candidate.is_file():
                tried.append(f"{candidate} (absent)")
                continue
            try:
                library = ctypes.CDLL(str(candidate))
                _declare(library)
            except (OSError, AttributeError) as error:
                tried.append(f"{candidate} ({error})")
                continue
            version = library.vigil_abi_version()
            if version != ABI_VERSION:
                tried.append(f"{candidate} (ABI {version}, this build speaks {ABI_VERSION})")
                continue
            layout = (ctypes.c_uint32 * len(EXPECTED_LAYOUT))()
            if library.vigil_layout(layout, len(EXPECTED_LAYOUT)) != len(EXPECTED_LAYOUT):
                tried.append(f"{candidate} (would not report its layout)")
                continue
            if tuple(layout) != EXPECTED_LAYOUT:
                tried.append(f"{candidate} (layout {tuple(layout)}, expected {EXPECTED_LAYOUT})")
                continue
            _LIBRARY = library
            _LOADED_FROM = candidate
            _log.info("engine core loaded: %s, ABI %d", candidate, version)
            return _LIBRARY
        _LOAD_FAULT = "; ".join(tried) or "no candidate paths"
        _log.warning("engine core not loaded, so the mapping build is unavailable and tracking "
                     "runs on the NumPy path: %s", _LOAD_FAULT)
        return None


def available() -> bool:
    return load() is not None


def fault() -> str | None:
    """Why the core is not loaded, or `None` when it is."""
    load()
    return _LOAD_FAULT if _LIBRARY is None else None


def loaded_from() -> Path | None:
    load()
    return _LOADED_FROM


def require() -> ctypes.CDLL:
    library = load()
    if library is None:
        raise NativeError(
            f"the engine core ({_library_name()}) is not loaded, and this operation needs it. "
            f"Build it with `python tasks.py core`, or set {LIBRARY_VARIABLE}. Tried: {_LOAD_FAULT}"
        )
    return library


def forget() -> None:
    """Drop the cached handle. For tests that move the library about."""
    global _LIBRARY, _LOAD_FAULT, _LOADED_FROM
    with _LOCK:
        _LIBRARY = None
        _LOAD_FAULT = None
        _LOADED_FROM = None


# ---------------------------------------------------------------- conversions


def _f64(array) -> np.ndarray:
    """A contiguous float64 array whose buffer the core can read directly."""
    return np.ascontiguousarray(array, dtype=np.float64)


def _ptr(array: np.ndarray, kind=ctypes.c_double):
    return array.ctypes.data_as(ctypes.POINTER(kind))


def pose_values(pose) -> np.ndarray:
    """A `CameraPose` as the sixteen numbers the ABI takes.

    The five lens coefficients are all-zero for an uncalibrated camera, which
    is every camera until `vigil cameras calibrate` has been run on it, and
    the two ground tilts are zero until the site has solved a plane — and zero
    is exactly the identity for both, on both sides of the boundary.
    """
    lens = getattr(pose, "lens", None)
    return _f64([
        pose.position.lat, pose.position.lon, pose.mount_height, pose.heading,
        pose.pitch, pose.roll, pose.horizontal_fov, pose.vertical_fov, pose.range_meters,
        0.0 if lens is None else lens.k1, 0.0 if lens is None else lens.k2,
        0.0 if lens is None else lens.p1, 0.0 if lens is None else lens.p2,
        0.0 if lens is None else lens.k3,
        getattr(pose, "ground_tilt_east", 0.0), getattr(pose, "ground_tilt_north", 0.0),
    ])


def sigma_values(sigma) -> np.ndarray:
    return _f64([
        sigma.heading_deg, sigma.pitch_deg, sigma.roll_deg,
        sigma.mount_height_m, sigma.terrain_slope,
    ])


# ------------------------------------------------------------------ geometry


def project_batch(pose, points, contact_sigma_deg: float, sigma, enforce_range: bool = True):
    """Project many normalised image points at once.

    Returns `(results, status)`: an `(n, 6)` array of
    `[lat, lon, distance, bearing, along_sigma, across_sigma]` and an `(n,)`
    array of status codes, 0 for a projection and 1-4 for a refusal.
    """
    library = require()
    uv = _f64(np.reshape(points, (-1, 2)))
    count = len(uv)
    results = np.zeros((count, PROJECTION_VALUES), dtype=np.float64)
    status = np.zeros(count, dtype=np.int32)
    if count == 0:
        return results, status
    pose_array = pose_values(pose)
    sigma_array = sigma_values(sigma)
    written = library.vigil_project_batch(
        _ptr(pose_array), _ptr(uv), count, ctypes.c_double(contact_sigma_deg),
        _ptr(sigma_array), 1 if enforce_range else 0,
        _ptr(results), _ptr(status, ctypes.c_int32),
    )
    if written < 0:
        raise NativeError("the core refused the pose or the buffers")
    return results, status


def image_coordinates_batch(pose, points):
    """Where many ground positions appear in the image.

    Returns `(uv, status)`, status 0 in front of the camera and 1 behind it.
    """
    library = require()
    latlon = _f64(np.reshape(points, (-1, 2)))
    count = len(latlon)
    out = np.zeros((count, 2), dtype=np.float64)
    status = np.zeros(count, dtype=np.int32)
    if count == 0:
        return out, status
    pose_array = pose_values(pose)
    if library.vigil_image_coordinates_batch(
        _ptr(pose_array), _ptr(latlon), count, _ptr(out), _ptr(status, ctypes.c_int32)
    ) < 0:
        raise NativeError("the core refused the pose or the buffers")
    return out, status


# ---------------------------------------------------------------- assignment


def assign(cost: np.ndarray) -> np.ndarray:
    """Optimal rectangular assignment. One entry per row, -1 for unassigned.

    Falls back to a NumPy implementation of the same algorithm when the core
    is absent. `tests/test_native.py` holds the two to identical answers, so
    the fallback is the same matching rather than a different one that happens
    to run.
    """
    matrix = _f64(cost)
    if matrix.ndim != 2:
        raise ValueError("a cost matrix is two-dimensional")
    rows, cols = matrix.shape
    if rows == 0 or cols == 0:
        return np.full(rows, -1, dtype=np.int64)
    library = load()
    if library is None:
        return _assign_numpy(matrix)
    out = np.zeros(rows, dtype=np.int64)
    if library.vigil_assign(_ptr(matrix), rows, cols, _ptr(out, ctypes.c_int64)) < 0:
        raise NativeError(f"the core refused a {rows}x{cols} assignment")
    return out


# -------------------------------------------------------------------- kalman


def kalman_initiate(cx: float, cy: float, aspect: float, height: float) -> np.ndarray:
    """A fresh filter state: 8 mean values then a flattened 8x8 covariance."""
    state = np.zeros(KALMAN_VALUES, dtype=np.float64)
    library = load()
    if library is None:
        return _kalman_numpy_initiate(state, cx, cy, aspect, height)
    library.vigil_kalman_initiate(_ptr(state), cx, cy, aspect, height)
    return state


def kalman_predict(state: np.ndarray, dt_seconds: float) -> None:
    library = load()
    if library is None:
        _kalman_numpy_predict(state, dt_seconds)
        return
    library.vigil_kalman_predict(_ptr(state), dt_seconds)


def kalman_update(state: np.ndarray, cx: float, cy: float, aspect: float, height: float) -> bool:
    """Fold in a measurement. False when the filter refused it as degenerate."""
    library = load()
    if library is None:
        return _kalman_numpy_update(state, cx, cy, aspect, height)
    return library.vigil_kalman_update(_ptr(state), cx, cy, aspect, height) == 0


def kalman_gate(state: np.ndarray, boxes: np.ndarray, position_only: bool = False) -> np.ndarray:
    """Squared Mahalanobis distance from one track to many measured boxes."""
    measurements = _f64(np.reshape(boxes, (-1, 4)))
    out = np.zeros(len(measurements), dtype=np.float64)
    if len(measurements) == 0:
        return out
    library = load()
    if library is None:
        return _kalman_numpy_gate(state, measurements, position_only)
    library.vigil_kalman_gate_batch(
        _ptr(state), _ptr(measurements), len(measurements), 1 if position_only else 0, _ptr(out)
    )
    return out


def kalman_warp(state: np.ndarray, warp: np.ndarray) -> bool:
    """Apply a 2x3 affine. False when the warp was not usable."""
    values = _f64(np.reshape(warp, -1)[:6])
    if not np.isfinite(values).all():
        return False
    library = load()
    if library is None:
        return _kalman_numpy_warp(state, values)
    return library.vigil_kalman_warp(_ptr(state), _ptr(values)) == 0


# The NumPy mirrors of the filter. Same model, same constants, named once so a
# change to the model cannot land on only one of them.
_kalman_numpy_initiate = _filtering.initiate
_kalman_numpy_predict = _filtering.predict
_kalman_numpy_update = _filtering.update
_kalman_numpy_gate = _filtering.gate
_kalman_numpy_warp = _filtering.warp


# --------------------------------------------------------------------- ortho


def grid_values(origin_lat: float, origin_lon: float, cell_size_m: float, rows: int, cols: int) -> np.ndarray:
    return _f64([origin_lat, origin_lon, cell_size_m, float(rows), float(cols)])


def ortho_sample(pose, grid: np.ndarray, image: np.ndarray, rows: int, cols: int,
                 lattice: int, colour: np.ndarray, valid: np.ndarray, resolution: np.ndarray) -> int:
    """Resample one BGR frame onto a ground grid, into caller-owned buffers.

    No NumPy fallback, and that is deliberate. The Python version of this loop
    is v1's, it was measured at 79 ms per frame per camera, and substituting
    it here would turn a one-minute map build into eighty without saying so.
    A caller that needs a map on a machine with no core is told.
    """
    library = require()
    frame = np.ascontiguousarray(image, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("a frame is a BGR array of shape (height, width, 3)")
    height, width = frame.shape[:2]
    pose_array = pose_values(pose)
    filled = library.vigil_ortho_sample(
        _ptr(pose_array), _ptr(grid), _ptr(frame, ctypes.c_uint8),
        width, height, width * 3, lattice,
        _ptr(colour, ctypes.c_uint8), _ptr(valid, ctypes.c_uint8), _ptr(resolution, ctypes.c_float),
    )
    if filled < 0:
        raise NativeError("the core refused the pose, the grid or the frame")
    return int(filled)


class MedianAccumulator:
    """A bounded per-cell history in the core, released on `close`.

    A context manager because it owns memory across the boundary and a leaked
    accumulator on a long-running node is tens of megabytes per camera per
    build. `__del__` is a backstop, not the mechanism: a finaliser that runs
    at interpreter shutdown may find the library already unloaded.
    """

    def __init__(self, cells: int, capacity: int):
        library = require()
        self._library = library
        self._cells = int(cells)
        handle = library.vigil_median_create(int(cells), int(capacity))
        if not handle:
            raise NativeError(f"the core refused an accumulator of {cells} cells x {capacity}")
        self._handle = ctypes.c_void_p(handle)

    def __enter__(self) -> "MedianAccumulator":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def closed(self) -> bool:
        return self._handle is None

    def add(self, colour: np.ndarray, valid: np.ndarray) -> None:
        if self._handle is None:
            raise NativeError("this accumulator has been closed")
        self._library.vigil_median_add(
            self._handle, _ptr(colour, ctypes.c_uint8), _ptr(valid, ctypes.c_uint8), self._cells
        )

    def result(self, minimum_samples: int, colour: np.ndarray, valid: np.ndarray,
               samples: np.ndarray, deviation: np.ndarray, disturbed: np.ndarray) -> int:
        if self._handle is None:
            raise NativeError("this accumulator has been closed")
        filled = self._library.vigil_median_result(
            self._handle, int(minimum_samples), self._cells,
            _ptr(colour, ctypes.c_uint8), _ptr(valid, ctypes.c_uint8),
            _ptr(samples, ctypes.c_uint16), _ptr(deviation, ctypes.c_uint8),
            _ptr(disturbed, ctypes.c_uint8),
        )
        if filled < 0:
            raise NativeError("the core refused the output buffers")
        return int(filled)

    def close(self) -> None:
        if self._handle is not None:
            self._library.vigil_median_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001 - a finaliser must not raise
            pass


# ------------------------------------------------------------- triangulation

#: Refusal codes the core returns, mirrored so callers can name them. The
#: numbers are the ABI's; `domain.triangulation` turns them into an enum.
#: Bound from `fallbacks` rather than repeated, so the code a fallback returns
#: and the code a caller compares against cannot drift apart.
from .fallbacks import BEHIND, PARALLEL, TOO_FAR_APART, TOO_LITTLE_PARALLAX  # noqa: E402,F401


def triangulate(a_origin, a_direction, b_origin, b_direction,
                angular_sigma_deg: float, min_parallax_deg: float) -> tuple[int, np.ndarray]:
    """Two ENU rays intersected. Returns `(code, values)`.

    `code` is 0 or one of the refusals above; `values` is
    `[x, y, z, parallax_deg, gap_m, range_a, range_b, sigma_m]` and is only
    meaningful when the code is 0.

    A code rather than an exception because a refused pair is the *expected*
    outcome for most pairs a correlator offers — two cameras looking at
    different people, or at the same one from nearly the same angle — and
    raising on the common case would make the caller's normal path a
    try/except.
    """
    a_o, a_d = _f64(a_origin), _f64(a_direction)
    b_o, b_d = _f64(b_origin), _f64(b_direction)
    for name, v in (("a_origin", a_o), ("a_direction", a_d),
                    ("b_origin", b_o), ("b_direction", b_d)):
        if v.shape != (3,):
            raise ValueError(f"{name} is three numbers, east/north/up in metres")
    out = np.zeros(TRIANGULATION_VALUES, dtype=np.float64)
    library = load()
    if library is None:
        return _triangulate_numpy(a_o, a_d, b_o, b_d, angular_sigma_deg, min_parallax_deg, out)
    code = library.vigil_triangulate(_ptr(a_o), _ptr(a_d), _ptr(b_o), _ptr(b_d),
                                     float(angular_sigma_deg), float(min_parallax_deg), _ptr(out))
    if code == -1:
        raise NativeError("the core refused the rays: a direction of no length, or a NaN")
    return int(code), out


def fit_plane(points: np.ndarray, threshold_m: float = 0.3, iterations: int = 200,
              seed: int = 1) -> np.ndarray | None:
    """RANSAC ground plane through `(n, 3)` ENU points.

    Returns `[nx, ny, nz, offset, inliers, rms, tilt_east, tilt_north]`, or
    `None` when no plane could be fitted.
    """
    cloud = _f64(points)
    if cloud.ndim != 2 or cloud.shape[1] != 3:
        raise ValueError("points are (n, 3): east, north, up in metres")
    if len(cloud) < 3:
        return None
    out = np.zeros(PLANE_VALUES, dtype=np.float64)
    library = load()
    if library is None:
        return _fit_plane_numpy(cloud, threshold_m, iterations, seed, out)
    code = library.vigil_fit_plane(_ptr(cloud), len(cloud), float(threshold_m),
                                   int(iterations), ctypes.c_uint64(int(seed) & 0xFFFFFFFFFFFFFFFF),
                                   _ptr(out))
    if code == -1:
        raise NativeError("the core refused the point cloud: it holds a NaN")
    return None if code < 0 else out


# --------------------------------------------------- detection post-processing


def suppress(xyxy: np.ndarray, scores: np.ndarray, classes: np.ndarray,
             iou_threshold: float, *, soft: bool = False,
             sigma: float = 0.5, floor: float = 0.2) -> np.ndarray:
    """Class-aware NMS, hard or soft. Returns kept indices in score order.

    Measured before it was written, which is this repository's rule for a
    Rust kernel: NumPy took **3.14 ms hard and 4.92 ms soft** on 300
    proposals over 8 classes, against about 12.5 ms for the detection itself
    — so suppression was a quarter to a third of detection, and tiling runs
    it once per tile. The assignment cost matrix, which was on the same list
    of candidates, measured 5 microseconds and stayed in Python.
    """
    boxes = _f64(xyxy).reshape(-1, 4)
    values = _f64(scores).reshape(-1)
    labels = np.ascontiguousarray(classes, dtype=np.int64).reshape(-1)
    if not (len(boxes) == len(values) == len(labels)):
        raise ValueError("boxes, scores and classes must be the same length")
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)
    library = load()
    if library is None:
        return _suppress_numpy(boxes, values, labels, iou_threshold, soft, sigma, floor)
    out = np.zeros(len(boxes), dtype=np.int64)
    kept = library.vigil_suppress(_ptr(boxes), _ptr(values), _ptr(labels, ctypes.c_int64),
                                  len(boxes), float(iou_threshold), 1 if soft else 0,
                                  float(sigma), float(floor), _ptr(out, ctypes.c_int64))
    if kept < 0:
        raise NativeError("the core refused the boxes: a NaN, or a null buffer")
    return out[:kept]


