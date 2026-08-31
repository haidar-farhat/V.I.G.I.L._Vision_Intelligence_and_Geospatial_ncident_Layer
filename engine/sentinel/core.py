"""Bindings to the Rust engine core.

The hot path — projection, zones, tracking — lives in Rust behind a plain C ABI.
This module is the only place in Python that knows that, so everything above it
works with ordinary dataclasses and never sees a pointer.

Why ctypes rather than PyO3: the core is built with the GNU toolchain while
CPython on Windows is built with MSVC, and PyO3 across that boundary is an ABI
hazard. The C ABI is the C ABI. It also means the core is loadable from anything,
so the engine is not welded to Python.

The struct layouts below mirror ``core/src/ffi.rs`` field for field. That
duplication is the price of a C boundary, so it is guarded: the library reports
an ABI version and this module refuses to load a mismatch rather than calling
functions whose signatures may have moved underneath it.
"""

from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence

ABI_VERSION = 4

# --------------------------------------------------------------------- structs


class CPose(ctypes.Structure):
    """Camera pose. Field order is part of the ABI."""

    _fields_ = [
        ("lat", ctypes.c_double),
        ("lon", ctypes.c_double),
        ("mount_height", ctypes.c_double),
        ("heading", ctypes.c_double),
        ("pitch", ctypes.c_double),
        ("roll", ctypes.c_double),
        ("horizontal_fov", ctypes.c_double),
        ("vertical_fov", ctypes.c_double),
        ("range_meters", ctypes.c_double),
    ]


class CDetection(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_double),
        ("y", ctypes.c_double),
        ("w", ctypes.c_double),
        ("h", ctypes.c_double),
        ("confidence", ctypes.c_double),
        ("class_id", ctypes.c_uint32),
        ("_pad", ctypes.c_uint32),
    ]


class CTrack(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_uint64),
        ("class_id", ctypes.c_uint32),
        ("confirmed", ctypes.c_uint32),
        ("x", ctypes.c_double),
        ("y", ctypes.c_double),
        ("w", ctypes.c_double),
        ("h", ctypes.c_double),
        ("confidence", ctypes.c_double),
        ("first_seen_millis", ctypes.c_int64),
        ("last_seen_millis", ctypes.c_int64),
        ("hits", ctypes.c_uint32),
        ("has_position", ctypes.c_uint32),
        ("lat", ctypes.c_double),
        ("lon", ctypes.c_double),
        ("uncertainty_meters", ctypes.c_double),
        ("position_source", ctypes.c_uint32),
        ("has_speed", ctypes.c_uint32),
        ("has_heading", ctypes.c_uint32),
        ("_pad2", ctypes.c_uint32),
        ("speed_mps", ctypes.c_double),
        ("heading_degrees", ctypes.c_double),
    ]


class CProjection(ctypes.Structure):
    _fields_ = [
        ("valid", ctypes.c_uint32),
        ("_pad", ctypes.c_uint32),
        ("lat", ctypes.c_double),
        ("lon", ctypes.c_double),
        ("ground_distance_meters", ctypes.c_double),
        ("bearing_deg", ctypes.c_double),
        ("uncertainty_meters", ctypes.c_double),
    ]


class CPoint(ctypes.Structure):
    _fields_ = [("lat", ctypes.c_double), ("lon", ctypes.c_double)]


# ---------------------------------------------------------------- python types


@dataclass(frozen=True, slots=True)
class LatLon:
    lat: float
    lon: float


@dataclass(frozen=True, slots=True)
class CameraPose:
    """Camera placement and optics.

    ``pitch`` is negative looking down, which is the normal mounting. ``heading``
    is a compass bearing with 0 = north.
    """

    position: LatLon
    mount_height: float
    heading: float
    pitch: float
    roll: float = 0.0
    horizontal_fov: float = 70.0
    vertical_fov: float = 40.0
    range_meters: float = 90.0

    def to_c(self) -> CPose:
        return CPose(
            self.position.lat,
            self.position.lon,
            self.mount_height,
            self.heading,
            self.pitch,
            self.roll,
            self.horizontal_fov,
            self.vertical_fov,
            self.range_meters,
        )


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Normalised image space: 0..1, origin top-left.

    Normalised rather than pixels so a box survives a resolution change, a
    sub-stream switch, or a model input-size change.
    """

    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True, slots=True)
class Detection:
    bbox: BoundingBox
    confidence: float
    class_id: int


@dataclass(frozen=True, slots=True)
class PositionEstimate:
    """A map position and how well it is actually known.

    ``radius_meters`` is a 1-sigma horizontal uncertainty. It is part of the
    position, never separated from it: rendering a horizon detection with the
    same confidence as one at the camera's feet is a lie the geometry does not
    support.
    """

    point: LatLon
    radius_meters: float
    #: "GROUND_PROJECTION" or "CAMERA_FALLBACK".
    source: str


@dataclass(frozen=True, slots=True)
class Track:
    id: int
    class_id: int
    bbox: BoundingBox
    confidence: float
    hits: int
    first_seen_millis: int
    last_seen_millis: int
    position: PositionEstimate | None
    #: ``None`` means too few observations to say. ``0.0`` means standing
    #: still — a different and often more interesting fact.
    speed_mps: float | None
    #: ``None`` when the object is not moving: a heading derived from jitter
    #: would be worse than admitting there is none.
    heading_degrees: float | None


@dataclass(frozen=True, slots=True)
class GroundProjection:
    point: LatLon
    ground_distance_meters: float
    bearing_deg: float
    uncertainty_meters: float


class CoreError(RuntimeError):
    """The core reported a failure, or could not be loaded."""


# ---------------------------------------------------------------------- loading


def _library_name() -> str:
    if sys.platform == "win32":
        return "sentinel_core.dll"
    if sys.platform == "darwin":
        return "libsentinel_core.dylib"
    return "libsentinel_core.so"


def _candidate_paths() -> Iterable[Path]:
    """Where the core might be, most specific first.

    An explicit environment variable wins so a packaged build can place the
    library anywhere; otherwise the cargo output directories, so a developer who
    has just run ``cargo build`` finds it without a step to remember.
    """
    override = os.environ.get("SENTINEL_CORE_LIB")
    if override:
        yield Path(override)

    here = Path(__file__).resolve()
    root = here.parents[2]
    name = _library_name()

    yield here.parent / name
    yield root / "core" / "target" / "release" / name
    yield root / "core" / "target" / "debug" / name


def _bind(lib: ctypes.CDLL) -> None:
    """Declare every signature.

    Without this ctypes assumes ``int`` for arguments and returns, which silently
    truncates every float that crosses the boundary — a failure that looks like
    bad geometry rather than a binding mistake.
    """
    lib.sentinel_abi_version.argtypes = []
    lib.sentinel_abi_version.restype = ctypes.c_uint32

    lib.sentinel_struct_sizes.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32]
    lib.sentinel_struct_sizes.restype = ctypes.c_int32

    lib.sentinel_project_to_ground.argtypes = [
        ctypes.POINTER(CPose), ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_uint32, ctypes.POINTER(CProjection),
    ]
    lib.sentinel_project_to_ground.restype = ctypes.c_int32

    lib.sentinel_field_of_view.argtypes = [
        ctypes.POINTER(CPose), ctypes.c_uint32, ctypes.POINTER(CPoint),
        ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.sentinel_field_of_view.restype = ctypes.c_int32

    lib.sentinel_camera_sees.argtypes = [ctypes.POINTER(CPose), ctypes.c_double, ctypes.c_double]
    lib.sentinel_camera_sees.restype = ctypes.c_int32

    lib.sentinel_point_in_zone.argtypes = [
        ctypes.POINTER(CPoint), ctypes.c_uint32, ctypes.c_double, ctypes.c_double,
    ]
    lib.sentinel_point_in_zone.restype = ctypes.c_int32

    lib.sentinel_zone_membership.argtypes = [
        ctypes.POINTER(CPoint), ctypes.c_uint32,
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ]
    lib.sentinel_zone_membership.restype = ctypes.c_int32

    lib.sentinel_haversine_distance.argtypes = [ctypes.c_double] * 4
    lib.sentinel_haversine_distance.restype = ctypes.c_double

    lib.sentinel_bearing_degrees.argtypes = [ctypes.c_double] * 4
    lib.sentinel_bearing_degrees.restype = ctypes.c_double

    lib.sentinel_destination_point.argtypes = [
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.POINTER(CPoint),
    ]
    lib.sentinel_destination_point.restype = ctypes.c_int32

    lib.sentinel_tracker_create.argtypes = [
        ctypes.POINTER(CPose), ctypes.c_double, ctypes.c_double,
        ctypes.c_int64, ctypes.c_uint32,
    ]
    lib.sentinel_tracker_create.restype = ctypes.c_void_p

    lib.sentinel_tracker_destroy.argtypes = [ctypes.c_void_p]
    lib.sentinel_tracker_destroy.restype = None

    lib.sentinel_tracker_set_pose.argtypes = [ctypes.c_void_p, ctypes.POINTER(CPose)]
    lib.sentinel_tracker_set_pose.restype = ctypes.c_int32

    lib.sentinel_tracker_update.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(CDetection), ctypes.c_uint32, ctypes.c_int64,
    ]
    lib.sentinel_tracker_update.restype = ctypes.c_int32

    lib.sentinel_tracker_tracks.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(CTrack), ctypes.c_uint32,
    ]
    lib.sentinel_tracker_tracks.restype = ctypes.c_int32

    lib.sentinel_tracker_ended.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32,
    ]
    lib.sentinel_tracker_ended.restype = ctypes.c_int32

    lib.sentinel_tracker_reset.argtypes = [ctypes.c_void_p]
    lib.sentinel_tracker_reset.restype = ctypes.c_int32


_lib: ctypes.CDLL | None = None


#: Names in the order ``sentinel_struct_sizes`` reports them.
_BOUNDARY_STRUCTS = (CPose, CDetection, CTrack, CProjection, CPoint)


def _check_struct_layout(lib: ctypes.CDLL, path: Path) -> None:
    """Compare every struct size against the Rust definition.

    The layouts above are maintained by hand. When one drifts out of step the
    failure is not a crash — it is reading the wrong bytes, which yields
    plausible and wrong geometry. Cheap to check once at load; nearly impossible
    to diagnose later.
    """
    count = len(_BOUNDARY_STRUCTS)
    buffer = (ctypes.c_uint32 * count)()
    written = lib.sentinel_struct_sizes(buffer, count)
    if written != count:
        raise CoreError(
            f"{path} reports {written} boundary structs; this build knows {count}."
        )

    mismatched = [
        f"{struct.__name__}: core says {buffer[index]} bytes, Python says "
        f"{ctypes.sizeof(struct)}"
        for index, struct in enumerate(_BOUNDARY_STRUCTS)
        if buffer[index] != ctypes.sizeof(struct)
    ]
    if mismatched:
        raise CoreError(
            "The Python struct layouts disagree with the core. Refusing to load "
            "rather than misread every value that crosses the boundary:\n  "
            + "\n  ".join(mismatched)
        )


def load_core() -> ctypes.CDLL:
    """Load the core, once.

    An ABI mismatch is refused rather than tolerated: calling a function whose
    signature has moved produces corrupted geometry, which is far harder to
    diagnose than a refusal at start-up.
    """
    global _lib
    if _lib is not None:
        return _lib

    tried: list[str] = []
    for path in _candidate_paths():
        tried.append(str(path))
        if not path.exists():
            continue
        try:
            lib = ctypes.CDLL(str(path))
        except OSError as error:
            raise CoreError(f"Found {path} but could not load it: {error}") from error

        _bind(lib)
        version = lib.sentinel_abi_version()
        if version != ABI_VERSION:
            raise CoreError(
                f"{path} reports ABI version {version}; this build expects "
                f"{ABI_VERSION}. Rebuild the core with 'cargo build --release'."
            )

        _check_struct_layout(lib, path)
        _lib = lib
        return lib

    raise CoreError(
        "The Rust engine core was not found. Build it with 'cargo build --release' "
        "in core/, or set SENTINEL_CORE_LIB. Looked in:\n  " + "\n  ".join(tried)
    )


# ------------------------------------------------------------------- geometry


def project_to_ground(
    pose: CameraPose,
    u: float,
    v: float,
    angular_uncertainty_deg: float = 1.5,
    enforce_range: bool = True,
) -> GroundProjection | None:
    """Project a normalised image point onto the ground plane.

    ``None`` when the ray is at or above the horizon, or lands beyond the pose's
    range. A position the system cannot determine must not appear on a map at
    all, so nothing is returned rather than a clamped guess.
    """
    lib = load_core()
    c_pose = pose.to_c()
    out = CProjection()

    status = lib.sentinel_project_to_ground(
        ctypes.byref(c_pose), u, v, angular_uncertainty_deg,
        1 if enforce_range else 0, ctypes.byref(out),
    )
    if status != 0:
        raise CoreError(f"projection failed with status {status}")
    if out.valid == 0:
        return None

    return GroundProjection(
        point=LatLon(out.lat, out.lon),
        ground_distance_meters=out.ground_distance_meters,
        bearing_deg=out.bearing_deg,
        uncertainty_meters=out.uncertainty_meters,
    )


def field_of_view(pose: CameraPose, arc_segments: int = 24) -> list[LatLon]:
    """Ground footprint of a camera's field of view.

    An annular sector, not a pie slice: a downward-tilted camera cannot see the
    ground at its own mast, and drawing the slice would claim coverage it does
    not have.
    """
    lib = load_core()
    c_pose = pose.to_c()

    capacity = arc_segments * 2 + 4
    buffer = (CPoint * capacity)()
    written = ctypes.c_uint32(0)

    status = lib.sentinel_field_of_view(
        ctypes.byref(c_pose), arc_segments, buffer, capacity, ctypes.byref(written)
    )
    if status < 0:
        raise CoreError(f"field of view failed with status {status}")

    return [LatLon(buffer[i].lat, buffer[i].lon) for i in range(written.value)]


def camera_sees(pose: CameraPose, point: LatLon) -> bool:
    """Whether a camera can geometrically see a ground point.

    Geometric reach only: no occlusion, no detectability at range. An upper
    bound, never a guarantee that something there would be detected.
    """
    lib = load_core()
    c_pose = pose.to_c()
    result = lib.sentinel_camera_sees(ctypes.byref(c_pose), point.lat, point.lon)
    if result < 0:
        raise CoreError("camera_sees failed")
    return result == 1


def point_in_zone(ring: Sequence[LatLon], point: LatLon) -> bool:
    if len(ring) < 3:
        return False

    lib = load_core()
    buffer = (CPoint * len(ring))(*[CPoint(p.lat, p.lon) for p in ring])
    return lib.sentinel_point_in_zone(buffer, len(ring), point.lat, point.lon) == 1


class ZoneMembership(str, Enum):
    """Where an object sits relative to a zone, given how well it is known.

    Three states, not two, and the third is the important one. A position
    estimate carries a 1-sigma radius that grows toward the horizon — metres
    across at 40 m from a mast — so "is this point inside" and "is this object
    inside" are different questions. Collapsing them produces intrusion alerts
    for objects that were never in the zone.

    A rule that raises an alarm must require :data:`INSIDE`. A rule that reports
    coverage should treat :data:`UNCERTAIN` as a gap.
    """

    OUTSIDE = "OUTSIDE"
    INSIDE = "INSIDE"
    UNCERTAIN = "UNCERTAIN"


_MEMBERSHIP = {0: ZoneMembership.OUTSIDE, 1: ZoneMembership.INSIDE, 2: ZoneMembership.UNCERTAIN}


def zone_membership(
    ring: Sequence[LatLon], point: LatLon, uncertainty_meters: float = 0.0
) -> ZoneMembership:
    """Whether an object is in a zone, accounting for its position uncertainty."""
    if len(ring) < 3:
        return ZoneMembership.OUTSIDE

    lib = load_core()
    buffer = (CPoint * len(ring))(*[CPoint(p.lat, p.lon) for p in ring])
    result = lib.sentinel_zone_membership(
        buffer, len(ring), point.lat, point.lon, uncertainty_meters
    )
    if result < 0:
        raise CoreError("zone membership failed")
    return _MEMBERSHIP[result]


def haversine_distance(a: LatLon, b: LatLon) -> float:
    return load_core().sentinel_haversine_distance(a.lat, a.lon, b.lat, b.lon)


def bearing_degrees(a: LatLon, b: LatLon) -> float:
    return load_core().sentinel_bearing_degrees(a.lat, a.lon, b.lat, b.lon)


def destination_point(origin: LatLon, bearing_deg: float, distance_meters: float) -> LatLon:
    lib = load_core()
    out = CPoint()
    lib.sentinel_destination_point(
        origin.lat, origin.lon, bearing_deg, distance_meters, ctypes.byref(out)
    )
    return LatLon(out.lat, out.lon)


# -------------------------------------------------------------------- tracking

_MAX_TRACKS = 256


class Tracker:
    """A single camera's tracker, living in Rust.

    Use as a context manager, or call ``close()``. The underlying handle is a
    Rust allocation, and leaking one leaks for the process lifetime — which for a
    long-running worker cycling cameras is a slow leak nobody notices.
    """

    __slots__ = ("_handle", "_lib", "_track_buffer", "_ended_buffer", "_closed")

    def __init__(
        self,
        pose: CameraPose | None = None,
        *,
        iou_threshold: float = 0.2,
        gate_factor: float = 2.5,
        max_gap_millis: int = 2000,
        min_hits_to_confirm: int = 2,
    ) -> None:
        self._lib = load_core()
        self._closed = False

        c_pose = ctypes.byref(pose.to_c()) if pose is not None else None
        handle = self._lib.sentinel_tracker_create(
            c_pose, iou_threshold, gate_factor, max_gap_millis, min_hits_to_confirm
        )
        if not handle:
            raise CoreError("could not create a tracker")

        self._handle = handle
        # Allocated once and reused: this is called per frame per camera.
        self._track_buffer = (CTrack * _MAX_TRACKS)()
        self._ended_buffer = (ctypes.c_uint64 * _MAX_TRACKS)()

    def __enter__(self) -> "Tracker":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self._closed or not self._handle:
            return
        self._lib.sentinel_tracker_destroy(self._handle)
        self._handle = None
        self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # Interpreter shutdown can pull the library out from under us. A
            # failure here would only produce noise on exit.
            pass

    def set_pose(self, pose: CameraPose | None) -> None:
        self._check()
        c_pose = ctypes.byref(pose.to_c()) if pose is not None else None
        self._lib.sentinel_tracker_set_pose(self._handle, c_pose)

    def update(self, detections: Sequence[Detection], at_millis: int) -> list[Track]:
        """Feed one frame and receive the confirmed tracks."""
        self._check()

        count = len(detections)
        if count:
            buffer = (CDetection * count)()
            for index, detection in enumerate(detections):
                buffer[index] = CDetection(
                    detection.bbox.x, detection.bbox.y, detection.bbox.w, detection.bbox.h,
                    detection.confidence, detection.class_id, 0,
                )
            pointer = buffer
        else:
            pointer = None

        status = self._lib.sentinel_tracker_update(self._handle, pointer, count, at_millis)
        if status < 0:
            raise CoreError(f"tracker update failed with status {status}")

        written = self._lib.sentinel_tracker_tracks(
            self._handle, self._track_buffer, _MAX_TRACKS
        )
        if written < 0:
            raise CoreError("could not read tracks")

        return [_to_track(self._track_buffer[i]) for i in range(written)]

    def ended(self) -> list[int]:
        """Track ids closed by the most recent update."""
        self._check()
        written = self._lib.sentinel_tracker_ended(
            self._handle, self._ended_buffer, _MAX_TRACKS
        )
        return [int(self._ended_buffer[i]) for i in range(max(0, written))]

    def reset(self) -> int:
        self._check()
        return max(0, self._lib.sentinel_tracker_reset(self._handle))

    def _check(self) -> None:
        if self._closed or not self._handle:
            raise CoreError("this tracker has been closed")


def _to_track(c: CTrack) -> Track:
    position = (
        PositionEstimate(
            point=LatLon(c.lat, c.lon),
            radius_meters=c.uncertainty_meters,
            source="CAMERA_FALLBACK" if c.position_source == 1 else "GROUND_PROJECTION",
        )
        if c.has_position
        else None
    )

    return Track(
        id=int(c.id),
        class_id=int(c.class_id),
        bbox=BoundingBox(c.x, c.y, c.w, c.h),
        confidence=c.confidence,
        hits=int(c.hits),
        first_seen_millis=int(c.first_seen_millis),
        last_seen_millis=int(c.last_seen_millis),
        position=position,
        speed_mps=c.speed_mps if c.has_speed else None,
        heading_degrees=c.heading_degrees if c.has_heading else None,
    )
