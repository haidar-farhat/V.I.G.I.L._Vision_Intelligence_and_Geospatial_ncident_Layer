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

import numpy as np
import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence

ABI_VERSION = 6

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
    """A detection crossing into the core.

    ``has_contact`` is 1 when ``contact_x`` / ``contact_y`` carry a measured
    ground-contact point (ABI 6). With it 0 the core uses the box's
    bottom-centre, which is what every position was projected from before
    segmentation existed.
    """

    _fields_ = [
        ("x", ctypes.c_double),
        ("y", ctypes.c_double),
        ("w", ctypes.c_double),
        ("h", ctypes.c_double),
        ("confidence", ctypes.c_double),
        ("class_id", ctypes.c_uint32),
        ("has_contact", ctypes.c_uint32),
        ("contact_x", ctypes.c_double),
        ("contact_y", ctypes.c_double),
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
        ("contact_x", ctypes.c_double),
        ("contact_y", ctypes.c_double),
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
class ContactPoint:
    """A point in normalised image coordinates: 0..1 across, 0..1 down."""

    x: float
    y: float

    def __iter__(self):
        yield self.x
        yield self.y


@dataclass(frozen=True, slots=True)
class Detection:
    bbox: BoundingBox
    confidence: float
    class_id: int
    #: This instance's shape, cropped to `bbox`, as 0/1 `uint8`. `None` from any
    #: detector that produces boxes only.
    #:
    #: Cropped rather than full-frame because a full-frame mask per detection is
    #: megabytes per frame at video rate, and every consumer already has the box.
    #: It is what makes a truthful ground-contact point possible: see
    #: :func:`ground_contact`, which takes the lowest row that has any of the
    #: object in it rather than assuming a rectangle's bottom edge.
    mask: "np.ndarray | None" = None


def ground_contact(detection: Detection) -> ContactPoint:
    """Where this object meets the ground, in normalised frame coordinates.

    **This is the reason segmentation is worth having.** Everything downstream —
    the projection to a map position, the zone test, the distance between two
    cameras' observations — rests on one point per object, and until now that
    point was the bottom-centre of a rectangle. That is correct only for a
    tight box around an upright, unoccluded person. For anybody leaning,
    carrying something, or half behind a car, the bottom-centre of the box is in
    the air or inside the obstacle, and the position it produces is confidently
    wrong.

    With a mask the answer is measurable: the lowest row that has any of this
    object in it, and the horizontal centre *of that row* — not of the whole
    mask, because a person mid-stride has their feet somewhere other than under
    their centre of mass.

    Falls back to the box's bottom-centre when there is no mask, which is what
    every detector without one has always produced.

    Lives here rather than beside the segmenter because the tracker calls it for
    every detection: it is the one place the mask influences the position, and
    it has to run whether or not a model is installed.
    """
    box = detection.bbox
    if detection.mask is None or detection.mask.size == 0:
        return ContactPoint(box.x + box.w / 2.0, box.y + box.h)

    rows = np.flatnonzero(detection.mask.any(axis=1))
    if rows.size == 0:
        return ContactPoint(box.x + box.w / 2.0, box.y + box.h)

    lowest = int(rows[-1])
    columns = np.flatnonzero(detection.mask[lowest])
    centre = float(columns.mean()) if columns.size else detection.mask.shape[1] / 2.0

    height, width = detection.mask.shape
    return ContactPoint(
        box.x + box.w * (centre + 0.5) / width,
        box.y + box.h * (lowest + 1) / height,
    )


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
    #: "GROUND_PROJECTION" (the contact, projected), "FRAME_EDGE" (the frame
    #: cut the contact off — the object is somewhere between the camera and
    #: the point the edge projects to; see `FRAME_EDGE_TOLERANCE`) or
    #: "CAMERA_FALLBACK" (the projection failed; this is the camera).
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
    #: Where the object last met the ground — the point its map position was
    #: projected from. Measured from the silhouette when the detector could see
    #: one, the box's bottom-centre when it could not. The core always reports
    #: it; ``None`` only for a track built by hand without one.
    contact: ContactPoint | None = None


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

    lib.sentinel_image_coordinates.argtypes = [
        ctypes.POINTER(CPose), ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.sentinel_image_coordinates.restype = ctypes.c_int32

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
    # Sizes only. This catches a struct that grew or shrank, which is the
    # common drift, and it does NOT catch two fields of the same width being
    # swapped — that keeps the total identical and reads plausible, wrong
    # values. Ordering is held instead by the round-trip tests in
    # `test_core.py`, which push a known value through each field and read it
    # back: a swap moves the value and the assertion fails. Both are needed and
    # neither is sufficient alone.
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

        # The version is read before anything else is bound. `_bind` touches
        # every exported symbol, so binding first meant a core older than this
        # build died on a bare AttributeError about a missing symbol instead of
        # the CoreError that explains what to do about it.
        try:
            version = lib.sentinel_abi_version
        except AttributeError as error:
            raise CoreError(
                f"{path} does not export sentinel_abi_version, so it is either "
                "not the engine core or is far older than this build. Rebuild "
                "it with 'cargo build --release'."
            ) from error

        version.restype = ctypes.c_uint32
        version = version()

        if version != ABI_VERSION:
            raise CoreError(
                f"{path} reports ABI version {version}; this build expects "
                f"{ABI_VERSION}. Rebuild the core with 'cargo build --release'."
            )

        _bind(lib)
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

    # The core clamps arc_segments to a minimum of 2, so the buffer has to be
    # sized for what the core will actually produce rather than for what was
    # asked. Sizing it from the raw argument under-allocated for anything below
    # 2 and then discarded the truncation status, silently returning a partial
    # footprint that would have been drawn as real coverage.
    segments = max(2, int(arc_segments))
    capacity = segments * 2 + 4
    buffer = (CPoint * capacity)()
    written = ctypes.c_uint32(0)

    status = lib.sentinel_field_of_view(
        ctypes.byref(c_pose), arc_segments, buffer, capacity, ctypes.byref(written)
    )
    if status < 0:
        raise CoreError(f"field of view failed with status {status}")
    if status == 1:
        raise CoreError(
            "the field-of-view buffer was too small and the footprint was "
            "truncated. A partial footprint drawn as a complete one claims "
            "coverage that does not exist."
        )

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


@dataclass(frozen=True, slots=True)
class ImagePoint:
    """Where a world point appears in a camera's image.

    ``u`` and ``v`` are normalised and **not clipped**: a value outside 0..1
    means the point is off frame in that direction, which is a meaningful answer
    rather than an error. ``in_frame`` is the clipped question.
    """

    u: float
    v: float
    distance_meters: float
    in_frame: bool


def image_coordinates(
    pose: CameraPose, point: LatLon, height_meters: float = 0.0
) -> ImagePoint | None:
    """Where a point at a given height appears in this camera's image.

    The exact inverse of :func:`project_to_ground`. ``None`` when the point is
    behind the camera, where no image position exists — which is different from
    being off the edge of the frame, and must not be confused with it.
    """
    lib = load_core()
    c_pose = pose.to_c()
    u = ctypes.c_double(0.0)
    v = ctypes.c_double(0.0)
    distance = ctypes.c_double(0.0)
    in_frame = ctypes.c_uint32(0)

    result = lib.sentinel_image_coordinates(
        ctypes.byref(c_pose), point.lat, point.lon, height_meters,
        ctypes.byref(u), ctypes.byref(v), ctypes.byref(distance), ctypes.byref(in_frame),
    )
    if result < 0:
        raise CoreError("image_coordinates failed")
    if result == 0:
        return None

    return ImagePoint(
        u=u.value, v=v.value, distance_meters=distance.value, in_frame=bool(in_frame.value)
    )


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

#: How close to the frame's bottom edge a box's lower side may sit before its
#: contact is taken as cut off by the frame rather than measured. Four per cent
#: of the frame height — about twenty rows at 480p — because a detector
#: regresses a truncated box a little short of the last row rather than onto
#: it. Measured on the laptop camera: a person seated at the desk, half a metre
#: from the lens with their feet below the picture, had box bottoms between
#: 0.975 and 1.0 across three probes and was projected to 2.16 m ± 0.13 m,
#: whatever their true distance; at two per cent the packaged build still
#: reported three confident entries into a zone that began 2 m out.
FRAME_EDGE_TOLERANCE = 0.04


class Tracker:
    """A single camera's tracker, living in Rust.

    Use as a context manager, or call ``close()``. The underlying handle is a
    Rust allocation, and leaking one leaks for the process lifetime — which for a
    long-running worker cycling cameras is a slow leak nobody notices.
    """

    __slots__ = ("_handle", "_lib", "_track_buffer", "_ended_buffer", "_closed", "_pose")

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
        #: Kept so a position can be judged against the camera's own location:
        #: see `_bounded_by_the_frame_edge`.
        self._pose = pose

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
        self._pose = pose

    def update(self, detections: Sequence[Detection], at_millis: int) -> list[Track]:
        """Feed one frame and receive the confirmed tracks."""
        self._check()

        count = len(detections)
        if count:
            buffer = (CDetection * count)()
            for index, detection in enumerate(detections):
                # The one place a mask changes a position. For a detection
                # without one this is the box's bottom-centre, so a detector
                # that produces boxes only gets exactly the answer it always
                # did — and a foreign caller that passes the flag clear does
                # too.
                contact = ground_contact(detection)
                buffer[index] = CDetection(
                    detection.bbox.x, detection.bbox.y, detection.bbox.w, detection.bbox.h,
                    detection.confidence, detection.class_id, 1,
                    contact.x, contact.y,
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

        return [_to_track(self._track_buffer[i], self._pose) for i in range(written)]

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


def _bounded_by_the_frame_edge(
    c: CTrack, position: PositionEstimate, pose: CameraPose
) -> PositionEstimate:
    """Widen a projection whose contact the frame cut off.

    A box whose lower side sits on the bottom edge of the frame is a box the
    frame truncated: the feet are below the picture, and the lowest visible row
    is the frame's, not the object's. Projecting that row gives the nearest
    ground the camera sees — the same distance for everybody it happens to,
    with the small uncertainty of a nearby point. The laptop camera showed it:
    a person seated half a metre from the lens was placed at 2.16 m ± 0.13 m,
    inside a zone that began at 2 m, and "entered" it without leaving their
    chair.

    What the geometry supports is a bound: the object is somewhere between the
    camera and that point. So the estimate becomes the middle of that stretch
    with a radius reaching both ends, tagged ``FRAME_EDGE``, and a zone that
    begins inside the stretch sees an uncertain membership rather than a
    confident one. The core is not changed — it reports what it measured — and
    this is the one place the frame's edge is known to be the reason.
    """
    if position.source != "GROUND_PROJECTION":
        return position
    if c.y + c.h < 1.0 - FRAME_EDGE_TOLERANCE:
        return position
    far = haversine_distance(pose.position, position.point)
    if far <= 0.0:
        return position
    bearing = bearing_degrees(pose.position, position.point)
    return PositionEstimate(
        point=destination_point(pose.position, bearing, far / 2.0),
        radius_meters=far / 2.0 + position.radius_meters,
        source="FRAME_EDGE",
    )


def _to_track(c: CTrack, pose: CameraPose | None = None) -> Track:
    position = (
        PositionEstimate(
            point=LatLon(c.lat, c.lon),
            radius_meters=c.uncertainty_meters,
            source="CAMERA_FALLBACK" if c.position_source == 1 else "GROUND_PROJECTION",
        )
        if c.has_position
        else None
    )
    if position is not None and pose is not None:
        position = _bounded_by_the_frame_edge(c, position, pose)

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
        contact=ContactPoint(c.contact_x, c.contact_y),
    )
