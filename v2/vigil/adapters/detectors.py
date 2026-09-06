"""Detectors: motion (free, blind to a stationary object) and ONNX (boxes or masks).

Nothing is downloaded. The model is a file the operator placed; its digest
travels with every event. The model is read once per process (v1 loaded it
four times per Start), and onnxruntime's telemetry is disarmed before the
native library initialises.

# Three defects fixed here

**Non-maximum suppression was class-agnostic.** Every detection in the frame
went into one suppression pass, so a person standing in front of a car with
70% overlap deleted whichever of the two scored lower. On a security camera
that is not an edge case — it is a car park. NMS is per class now, which is
what every YOLO implementation does and what the model was trained against.

**The execution provider was pinned to the CPU.** `providers=["CPUExecutionProvider"]`
was hard-coded, so a machine with onnxruntime-gpu installed ran on the CPU
anyway and nothing said so. The provider is chosen from what the installed
runtime actually offers, and it is reported in `DetectorInfo` so `vigil doctor`
can tell an operator which one they got.

**Thread count was left to the default**, which is "every core, per session".
With one session per camera thread, eight cameras on eight cores means eight
sessions each trying to use eight cores: the threads spend their time
descheduling each other. The session is configured for the way this product
actually runs.

# The mask, and what it is for

A segmentation model's mask was used for one thing — finding where an object
meets the ground — and thrown away. It is kept now, cropped to the box and
downsampled, because an appearance descriptor taken over a whole bounding box
is mostly a descriptor of the background. See `vigil.domain.appearance`.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Iterable, Protocol

import cv2
import numpy as np

from ..domain.detection import UNCLASSIFIED, BoundingBox, Detection, DetectorInfo
from ..kernel import native as _native
from ..kernel import onnx as _onnx
from ..domain.geo import Vec2
from ..logs import get as _get_logger

_log = _get_logger(__name__)

#: The classes a security site watches unless told otherwise. A bottle on a
#: shelf is not an intruder; v1 learned that from an operator's screenshot.
WATCHED_LABELS = frozenset({"person", "bicycle", "car", "motorcycle", "bus", "truck"})

#: The floor below which a single frame's detection is not even proposed.
#:
#: 0.25, down from 0.5. The tracker this feeds was rebuilt around cumulative
#: confirmation, a weak-detection recovery pass and re-identification across a
#: gap, so a thing seen once at 0.3 and never again is never confirmed and
#: never reaches a rule. Lowering the floor therefore buys recall — a person
#: at 40 m, or side-on, or half behind a van, scores in the thirties — without
#: buying false alarms, because what an operator sees is decided by the
#: *track*, not by one frame. Raising it back is a site setting.
DEFAULT_CONFIDENCE = 0.25

#: The old floor, kept because it is what the reporting threshold used to be
#: and the comparison is worth being able to make.
LEGACY_CONFIDENCE = 0.5

#: Gaussian width for soft-NMS. At 0.5, a box overlapping 0.9 keeps 6% of its
#: score and a genuine neighbour at 0.55 keeps 45%.
SOFT_NMS_SIGMA = 0.5

#: Score below which a soft-decayed box is dropped. Same as the detection
#: floor: a box the decay has pushed under what a fresh detection needs is not
#: worth proposing.
SOFT_NMS_FLOOR = 0.2

#: The providers, and the thread knob, both live in `kernel.onnx` now so that
#: faces and plates use the same ones. Bound here rather than re-declared:
#: `vigil doctor`, the console and the tests all read them from this module,
#: and a second copy of a preference order is a second answer waiting to
#: differ from the one the sessions are actually opened with.
PREFERRED_PROVIDERS = _onnx.PREFERRED_PROVIDERS
DEFAULT_INTRA_OP_THREADS = _onnx.DEFAULT_INTRA_OP_THREADS
THREADS_VARIABLE = _onnx.THREADS_VARIABLE

#: Threads per session for the CPU provider.
#:
#: One session per camera thread means the default — all cores, per session —
#: has eight cameras each asking for eight cores on an eight-core machine, and
#: they spend their time descheduling each other. Two is enough to use the
#: model's own parallelism without the sessions fighting; a single-camera
#: deployment gets more from `VIGIL_ORT_THREADS`.

#: Size the kept mask is downsampled to before it leaves the detector, in
#: (width, height). An appearance descriptor bins a few hundred pixels; a
#: full-resolution mask crop is kilobytes per detection per frame for no gain.
MASK_KEEP_SIZE = (24, 48)


class DetectionError(RuntimeError):
    pass


class Detector(Protocol):
    @property
    def info(self) -> DetectorInfo: ...
    def detect(self, image: np.ndarray) -> list[Detection]: ...


# ------------------------------------------------------------------ motion


class MotionDetector:
    """Background subtraction. Answers "what changed", which includes a curtain."""

    def __init__(self, *, min_area_fraction: float = 0.002, history: int = 200, detect_scale: float = 0.5):
        self._subtractor = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=32, detectShadows=True)
        self._min_area = min_area_fraction
        self._scale = detect_scale
        self._info = DetectorInfo(kind="motion", name="MOG2 background subtraction", classifies=False)

    @property
    def info(self) -> DetectorInfo:
        return self._info

    def detect(self, image: np.ndarray) -> list[Detection]:
        small = cv2.resize(image, None, fx=self._scale, fy=self._scale) if self._scale != 1.0 else image
        mask = self._subtractor.apply(small)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = small.shape[:2]
        out = []
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            fraction = (bw * bh) / float(w * h)
            if fraction < self._min_area:
                continue
            region = mask[y:y + bh, x:x + bw]
            moved = cv2.countNonZero(region) / float(max(1, bw * bh))
            # The changed pixels inside the box *are* the object's silhouette,
            # as far as a subtractor can tell. Keeping them gives a motion
            # detector the same masked appearance a segmentation model gets,
            # which is what lets the tracker re-identify without a model.
            kept = cv2.resize(region, MASK_KEEP_SIZE, interpolation=cv2.INTER_NEAREST)
            out.append(Detection(BoundingBox(x / w, y / h, bw / w, bh / h), round(moved, 3),
                                 UNCLASSIFIED, mask=kept))
        return out


# -------------------------------------------------------------------- onnx

_MODEL_LOCK = threading.Lock()
_MODEL_INFO: dict[tuple[str, int, int], DetectorInfo] = {}
_OUTPUT_COUNTS: dict[tuple[str, int, int], int] = {}


# Loading, provider selection, telemetry and digests live in `kernel.onnx`,
# because `perception`'s face and plate models need exactly the same rules and
# `perception` may not import `adapters`. Duplicating them was the alternative,
# and these are rules that rot when duplicated: the one that matters most —
# read the provider back rather than assuming it — is invisible when wrong.


def forget_models() -> None:
    _onnx.forget()
    with _MODEL_LOCK:
        _MODEL_INFO.clear()
        _OUTPUT_COUNTS.clear()


def available_providers() -> list[str]:
    """The providers the installed runtime offers, best first.

    Re-exported rather than moved: `vigil doctor` and the console both read it
    from here, and "why is this slow" is answered by this list far more often
    than by anything else — an operator who installed `onnxruntime` rather
    than `onnxruntime-directml` has a CPU-only runtime and no sign of it.
    """
    return _onnx.available_providers()


def _model_key(path: str | Path) -> tuple[str, int, int]:
    try:
        return _onnx.model_key(path)
    except _onnx.ModelError as error:
        raise DetectionError(str(error)) from error


def _session(path: Path) -> tuple[object, str]:
    try:
        return _onnx.open_session(path)
    except _onnx.ModelError as error:
        raise DetectionError(str(error)) from error


def _sha256(path: Path) -> str:
    return _onnx.digest(path)


def _names_from_metadata(session) -> dict[int, str]:
    import ast

    meta = session.get_modelmeta().custom_metadata_map or {}
    raw = meta.get("names")
    if not raw:
        return {}
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return {}
    if isinstance(parsed, dict):
        return {int(k): str(v) for k, v in parsed.items()}
    if isinstance(parsed, (list, tuple)):
        return {i: str(v) for i, v in enumerate(parsed)}
    return {}


def restrict_vocabulary(names: dict[int, str], classes: Iterable[str] | None) -> tuple[dict[int, str], set[int] | None]:
    if classes is None:
        return dict(names), None
    wanted = {str(c).strip().lower() for c in classes if str(c).strip()}
    if not wanted:
        raise DetectionError("asked to watch no class at all; leave the watch list unset to watch everything")
    if not names:
        raise DetectionError(f"asked to watch {', '.join(sorted(wanted))}, but this model names no classes")
    known = {n.strip().lower(): i for i, n in names.items()}
    unknown = sorted(wanted - set(known))
    if unknown:
        raise DetectionError(f"asked to watch {', '.join(unknown)}, which this model does not name; it knows {', '.join(sorted(known))}")
    kept = {known[l]: names[known[l]] for l in sorted(wanted)}
    return kept, set(kept)


def model_info(path: str | Path, *, classes: Iterable[str] | None = None) -> DetectorInfo:
    """What a model can do, read once per process and per file."""
    key = _model_key(path)
    with _MODEL_LOCK:
        info = _MODEL_INFO.get(key)
    if info is None:
        info = OnnxDetector(path).info
        with _MODEL_LOCK:
            _MODEL_INFO[key] = info
    if classes is None:
        return info
    names, _ = restrict_vocabulary(info.class_names, classes)
    return DetectorInfo(info.kind, info.name, info.model_path, info.model_sha256, info.input_size,
                        names, info.classifies, info.provider)


class OnnxDetector:
    """YOLO-family ONNX: one input NCHW, one output (boxes) or two (boxes + mask protos)."""

    def __init__(self, model_path: str | Path, *, confidence: float = DEFAULT_CONFIDENCE, iou: float = 0.45,
                 classes: Iterable[str] | None = None, soft_nms: bool = True):
        path = Path(model_path).resolve()
        key = _model_key(path)
        self._session, provider = _session(path)
        inputs = self._session.get_inputs()
        if len(inputs) != 1:
            raise DetectionError(f"expected one input; {path.name} has {len(inputs)}")
        shape = inputs[0].shape
        if len(shape) != 4:
            raise DetectionError(f"expected an NCHW input; {path.name} has shape {shape}")
        self._input_name = inputs[0].name
        h = int(shape[2]) if isinstance(shape[2], int) else 640
        w = int(shape[3]) if isinstance(shape[3], int) else 640
        self._size = (w, h)
        self._outputs = [o.name for o in self._session.get_outputs()]
        with _MODEL_LOCK:
            _OUTPUT_COUNTS[key] = len(self._outputs)
        self._segments = len(self._outputs) >= 2
        names = _names_from_metadata(self._session)
        names, self._watched = restrict_vocabulary(names, classes)
        self._confidence = confidence
        self._iou = iou
        self._soft = soft_nms
        self._info = DetectorInfo(
            kind="onnx-segment" if self._segments else "onnx-detect", name=path.stem, model_path=str(path),
            model_sha256=_sha256(path), input_size=self._size, class_names=names, classifies=True,
            provider=provider,
        )
        _log.info("model ready: %s, %dx%d, %d class name(s)%s, on %s", path.name, w, h, len(names),
                  ", masks" if self._segments else "", provider)

    @property
    def info(self) -> DetectorInfo:
        return self._info

    def detect(self, image: np.ndarray) -> list[Detection]:
        blob, scale, pad = _letterbox(image, self._size)
        outputs = self._session.run(self._outputs, {self._input_name: blob})
        predictions = outputs[0]
        if predictions.ndim == 3:
            predictions = predictions[0]
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T  # (N, 4 + classes [+ mask coeffs])
        protos = outputs[1][0] if self._segments and len(outputs) > 1 else None
        n_mask = protos.shape[0] if protos is not None else 0
        class_count = predictions.shape[1] - 4 - n_mask
        if class_count <= 0:
            return []
        boxes = predictions[:, :4]
        scores = predictions[:, 4:4 + class_count]
        class_ids = scores.argmax(axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]
        keep = confidences >= self._confidence
        if self._watched is not None:
            keep &= np.isin(class_ids, list(self._watched))
        boxes, confidences, class_ids = boxes[keep], confidences[keep], class_ids[keep]
        coeffs = predictions[keep, 4 + class_count:] if protos is not None else None
        if len(boxes) == 0:
            return []
        xyxy = np.stack([boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
                         boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2], axis=1)
        order = _nms_per_class(xyxy, confidences, class_ids, self._iou, soft=self._soft)
        h, w = image.shape[:2]
        out = []
        for i in order:
            x1, y1, x2, y2 = xyxy[i]
            x1 = (x1 - pad[0]) / scale / w
            x2 = (x2 - pad[0]) / scale / w
            y1 = (y1 - pad[1]) / scale / h
            y2 = (y2 - pad[1]) / scale / h
            bbox = BoundingBox(float(x1), float(y1), float(x2 - x1), float(y2 - y1)).clamped()
            if bbox.area <= 0:
                continue
            contact, mask = None, None
            if protos is not None and coeffs is not None:
                contact, mask = _mask_for(protos, coeffs[i], xyxy[i], self._size, scale, pad, (w, h))
            out.append(Detection(bbox, float(confidences[i]), int(class_ids[i]), contact, mask))
        return out


def _letterbox(image: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, float, tuple[float, float]]:
    w, h = size
    ih, iw = image.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = int(round(iw * scale)), int(round(ih * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((h, w, 3), 114, dtype=np.uint8)
    px, py = (w - nw) / 2, (h - nh) / 2
    canvas[int(py):int(py) + nh, int(px):int(px) + nw] = resized
    blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[None]
    return np.ascontiguousarray(blob), scale, (px, py)


def _nms(xyxy: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    order = scores.argsort()[::-1]
    keep: list[int] = []
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    while len(order):
        i = int(order[0])
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(xyxy[i, 0], xyxy[rest, 0])
        yy1 = np.maximum(xyxy[i, 1], xyxy[rest, 1])
        xx2 = np.minimum(xyxy[i, 2], xyxy[rest, 2])
        yy2 = np.minimum(xyxy[i, 3], xyxy[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


def _soft_nms(xyxy: np.ndarray, scores: np.ndarray, iou_threshold: float,
              sigma: float = SOFT_NMS_SIGMA, floor: float = SOFT_NMS_FLOOR) -> list[int]:
    """Decay an overlapping box's score instead of deleting it.

    Hard NMS assumes that two boxes overlapping past a threshold are two
    proposals for one object. In a queue of people or a row of parked cars
    that is false, and the second real object is deleted with no trace
    anywhere — the failure this whole module is most careful about, because a
    suppressed detection leaves nothing for anybody to notice.

    Gaussian soft-NMS multiplies the score by `exp(-iou^2 / sigma)` instead.
    A genuine duplicate at 0.9 overlap keeps 6% of its score and falls under
    the floor; a real neighbour at 0.55 keeps 45% and survives. The score it
    survives with is honestly lower, and the tracker's cumulative confirmation
    is what turns "seen weakly, repeatedly, in the same place" into a track.
    """
    order = list(scores.argsort()[::-1])
    working = scores.astype(np.float64).copy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    keep: list[int] = []
    while order:
        order.sort(key=lambda idx: -working[idx])
        i = order.pop(0)
        if working[i] < floor:
            break
        keep.append(int(i))
        if not order:
            break
        rest = np.asarray(order, dtype=np.int64)
        xx1 = np.maximum(xyxy[i, 0], xyxy[rest, 0])
        yy1 = np.maximum(xyxy[i, 1], xyxy[rest, 1])
        xx2 = np.minimum(xyxy[i, 2], xyxy[rest, 2])
        yy2 = np.minimum(xyxy[i, 3], xyxy[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        # Only boxes that actually overlap are touched, so a distant one keeps
        # its score exactly rather than being multiplied by something near one.
        decayed = iou > iou_threshold
        working[rest[decayed]] *= np.exp(-(iou[decayed] ** 2) / sigma)
        order = [int(j) for j in rest if working[j] >= floor]
    return keep


def _nms_per_class(xyxy: np.ndarray, scores: np.ndarray, class_ids: np.ndarray,
                   iou_threshold: float, *, soft: bool = False) -> list[int]:
    """Suppress within each class, never across them.

    One pass over every box in the frame lets a person standing in front of a
    car delete the car — a 70% overlap between two *different* things is not
    a duplicate detection, it is a car park. Class-agnostic NMS was what this
    module did, and it is the more expensive mistake: a suppressed detection
    leaves no trace anywhere for anybody to notice.
    """
    # In the core when it is loaded: measured at 3.14 ms hard and 4.92 ms soft
    # on 300 proposals in NumPy, against about 12.5 ms for the detection
    # itself — and tiling runs this once per tile. The NumPy path below is the
    # same algorithm and `tests/test_native.py` holds the two to identical
    # answers, ties included.
    return [int(i) for i in _native.suppress(
        xyxy, scores, class_ids, iou_threshold, soft=soft,
        sigma=SOFT_NMS_SIGMA, floor=SOFT_NMS_FLOOR)]


def _mask_for(protos, coeff, box, size, scale, pad, image_size):
    """The object's silhouette and where it meets the ground.

    Only the box's own region of the prototype stack is multiplied out. The
    obvious form — `coeff @ protos.reshape(c, -1)` — evaluates the mask over
    the whole 160x160 field for every detection, and then reads a box that is
    typically a twentieth of it. Cropping first is the same arithmetic over
    twenty times less of it.
    """
    c, mh, mw = protos.shape
    sx, sy = mw / size[0], mh / size[1]
    x1, y1 = max(0, int(box[0] * sx)), max(0, int(box[1] * sy))
    x2, y2 = min(mw, int(np.ceil(box[2] * sx))), min(mh, int(np.ceil(box[3] * sy)))
    if x2 <= x1 or y2 <= y1:
        return None, None
    window = protos[:, y1:y2, x1:x2].reshape(c, -1)
    values = (coeff @ window).reshape(y2 - y1, x2 - x1)
    inside = values > 0.0  # sigmoid(x) > 0.5 is x > 0; the sigmoid is not needed
    rows = np.flatnonzero(inside.any(axis=1))
    kept = cv2.resize(inside.astype(np.uint8), MASK_KEEP_SIZE, interpolation=cv2.INTER_NEAREST)
    if len(rows) == 0:
        return None, kept
    lowest = int(rows[-1])
    columns = np.flatnonzero(inside[lowest])
    cx = (x1 + float(columns.mean())) / sx
    cy = (y1 + lowest + 1) / sy
    contact = Vec2(float((cx - pad[0]) / scale / image_size[0]),
                   float((cy - pad[1]) / scale / image_size[1]))
    return contact, kept


def detector_for(model: str | Path | None, *, classes: Iterable[str] | None = None,
                 confidence: float | None = None) -> Detector:
    """Motion without a model; ONNX with one. A motion detector cannot watch a class, so the list is dropped."""
    if model is None:
        return MotionDetector()
    return OnnxDetector(model, classes=classes, confidence=DEFAULT_CONFIDENCE if confidence is None else confidence)
