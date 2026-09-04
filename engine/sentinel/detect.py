"""Detection.

Two detectors, because a local-first appliance must not be useless without a
model file.

:class:`MotionDetector` needs nothing. It is background subtraction — the
baseline a great many deployed systems still run on. It finds *that something
moved*, and it is scrupulously clear that it cannot say *what*: every detection
it emits is :data:`UNCLASSIFIED`. A blob is not a person, and a system that
labels it one has fabricated evidence.

:class:`OnnxDetector` runs a model the operator supplied. It never downloads
anything, ever — a security appliance that fetches executable weights from the
Internet has a supply chain, and this one deliberately does not. The model path
comes from configuration, the file is read from disk, and if it is absent the
system says so and falls back rather than reaching out.

Both emit the same :class:`Detection` in normalised coordinates, so everything
downstream is indifferent to which one produced it — except that the class label
travels with the detection, so a downstream rule can require a classified object
and refuse to fire on a blob.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import cv2
import numpy as np

from . import telemetry
from .core import BoundingBox, Detection

#: Class id meaning "something moved and we do not know what it is".
#:
#: Deliberately not 0: 0 is "person" in most detection models, and a motion blob
#: silently inheriting that id is precisely the confusion this constant exists to
#: prevent.
UNCLASSIFIED = 9999


class DetectionError(RuntimeError):
    """A detector could not be constructed or run."""


@dataclass(frozen=True, slots=True)
class DetectorInfo:
    """What a detector is, for the record.

    Every conclusion the system draws has to be attributable to the thing that
    drew it. This travels with events into the evidence bundle, so "why did it
    say that" is answerable months later.
    """

    kind: str
    name: str
    #: Present only when a model file is involved.
    model_path: str | None = None
    model_sha256: str | None = None
    input_size: tuple[int, int] | None = None
    #: Class id to label. Empty when the model carries no names — reporting a
    #: guessed label would be an invention.
    class_names: dict[int, str] = field(default_factory=dict)
    #: True when the detector classifies. False for motion, which does not.
    classifies: bool = False

    def label_for(self, class_id: int) -> str:
        if class_id == UNCLASSIFIED:
            return "unclassified"
        return self.class_names.get(class_id, f"class_{class_id}")


class Detector(Protocol):
    """Anything that turns a frame into detections."""

    @property
    def info(self) -> DetectorInfo: ...

    def detect(self, image: np.ndarray) -> list[Detection]: ...


# ------------------------------------------------------------------- motion


class MotionDetector:
    """Background subtraction. Finds movement; never claims to classify it.

    Tuned for a fixed camera watching a scene. It will produce nothing useful on
    a moving camera, and says so rather than emitting the whole frame as one
    detection when the view pans.
    """

    __slots__ = ("_subtractor", "_open_kernel", "_close_kernel", "_kernel_shape",
                 "_close_height", "_min_area", "_max_area", "_warmup", "_seen",
                 "_info", "_scale")

    def __init__(
        self,
        *,
        history: int = 200,
        variance_threshold: float = 16.0,
        detect_shadows: bool = True,
        min_area_fraction: float = 0.0006,
        max_area_fraction: float = 0.35,
        close_height_fraction: float = 0.065,
        warmup_frames: int = 12,
        detect_scale: float = 0.75,
    ):
        """
        ``min_area_fraction`` is the smallest blob taken seriously, as a fraction
        of the frame. Too low and every leaf is an intruder; too high and a
        person at 60 m is invisible. It is a fraction rather than a pixel count
        so it survives a resolution change.

        ``max_area_fraction`` rejects blobs covering most of the frame. Those are
        not objects — they are an illumination change, an auto-exposure step, or
        a camera that just moved.

        ``close_height_fraction`` sets how far apart two pieces of foreground can
        be and still be joined vertically. See :meth:`_build_kernels` for why it
        is vertical and why it is a fraction.

        ``detect_scale`` shrinks the frame before the background model sees it,
        and it is the single most consequential number here for capacity.

        MOG2 keeps a mixture of Gaussians *per pixel*, read and written every
        frame, and that working set — not the GIL, not Python, not the FFI — is
        what stops this system scaling across cameras. Measured on this machine
        with eight detectors running side by side: a Gaussian blur scales 6.8x
        and a memory-only loop scales 14.5x, but MOG2 plateaus at 2.1x. Shrinking
        the frame it models shrinks that state quadratically:

        | scale | 1 worker | 8 workers | recall | mean IoU |
        |-------|----------|-----------|--------|----------|
        | 1.00  |  236 fps |   455 fps |  0.690 |    0.503 |
        | 0.75  |  349 fps |   784 fps |  0.707 |    0.511 |
        | 0.50  | 1018 fps |  1927 fps |  0.652 |    0.460 |
        | 0.35  |  900 fps |  4189 fps |  0.616 |    0.419 |

        0.75 is the default because it is better on **both** axes — 1.7x the
        throughput and slightly *better* detection, because the downscale is a
        mild denoise. 0.5 buys 4.2x for a real cost in recall, and is the right
        choice for a node carrying more cameras than cores.

        Nothing downstream needs to know. Boxes are normalised and every
        threshold here is a fraction of the frame, so the detector's output is
        identical in meaning at any scale.
        """
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=variance_threshold, detectShadows=detect_shadows
        )
        self._close_height = close_height_fraction
        self._scale = float(detect_scale)
        if not 0.1 <= self._scale <= 1.0:
            raise DetectionError(
                f"detect_scale must be between 0.1 and 1.0, not {detect_scale}. "
                "Below 0.1 a person is a handful of pixels and the detector is "
                "measuring noise."
            )
        # Built on the first frame, once the real resolution is known.
        self._open_kernel: np.ndarray | None = None
        self._close_kernel: np.ndarray | None = None
        self._kernel_shape: tuple[int, int] | None = None
        self._min_area = min_area_fraction
        self._max_area = max_area_fraction
        self._warmup = warmup_frames
        self._seen = 0
        self._info = DetectorInfo(
            kind="motion",
            # The scale is part of the detector's identity: two runs at
            # different scales are not the same detector, and an event's
            # provenance should say which one produced it.
            name=(
                "MOG2 background subtraction"
                if self._scale == 1.0
                else f"MOG2 background subtraction at {self._scale:g} scale"
            ),
            classifies=False,
        )

    @property
    def info(self) -> DetectorInfo:
        return self._info

    @property
    def is_warm(self) -> bool:
        """Whether the background model has seen enough to be trusted.

        Before this, every frame is mostly foreground. Emitting detections during
        warm-up would open an incident every time a camera reconnects.
        """
        return self._seen >= self._warmup

    def _build_kernels(self, shape: tuple[int, int]) -> None:
        """Size the morphology to the frame, on the first frame.

        Two decisions, both measured on the reference scene rather than assumed:

        **The closing kernel is tall and narrow, not square.** People, vehicles
        and animals are upright, and a background model splits them along their
        length — a head separated from a torso, legs separated from a body. A
        vertical kernel rejoins those pieces while leaving two people standing
        side by side as two objects, which a square kernel of the same reach
        would merge into one. Every spurious detection measured on the reference
        scene was a fragment of a real object rather than noise, so this is the
        failure that was actually happening. Against a square 9x9: mean overlap
        with ground truth rises from 0.40 to 0.50, fragments per frame fall from
        1.20 to 0.42, and recall rises from 0.64 to 0.69.

        **It is a fraction of frame height, not a pixel count.** A person is
        roughly the same fraction of the frame at any resolution; 31 pixels on
        576p is a third of the reach on 1080p, so a fixed kernel silently stops
        working when someone switches to the main stream.
        """
        if self._kernel_shape == shape:
            return

        height = shape[0]
        reach = max(5, int(round(height * self._close_height)) | 1)
        self._open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, reach))
        self._kernel_shape = shape

    def detect(self, image: np.ndarray) -> list[Detection]:
        self._seen += 1

        # Shrunk before the background model sees it. INTER_AREA because it
        # averages the pixels it discards rather than sampling one of them,
        # which is what keeps a small distant object from disappearing between
        # sample points.
        if self._scale != 1.0:
            image = cv2.resize(
                image, None, fx=self._scale, fy=self._scale, interpolation=cv2.INTER_AREA
            )

        mask = self._subtractor.apply(image)
        # Sized before the warm-up check, so a detector that has seen a frame is
        # fully configured whether or not it is ready to report anything yet.
        self._build_kernels(mask.shape[:2])

        if not self.is_warm:
            return []

        # MOG2 marks shadows 127 and foreground 255. Keeping shadows would make
        # every object twice its real width and drag its ground contact point
        # sideways, which lands it in the wrong place on the map.
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open_kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_kernel, iterations=3)

        height, width = mask.shape[:2]
        frame_area = float(height * width)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: list[Detection] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = float(w * h)
            fraction = area / frame_area
            if fraction < self._min_area or fraction > self._max_area:
                continue

            # Confidence here is the fraction of the box that is actually moving,
            # not a probability of anything. A hollow box — a lighting edge —
            # scores low; a solid one scores high. It is an honest measure of
            # "how much of this rectangle moved", and nothing more.
            filled = float(cv2.countNonZero(mask[y : y + h, x : x + w])) / max(area, 1.0)

            detections.append(
                Detection(
                    bbox=BoundingBox(x / width, y / height, w / width, h / height),
                    confidence=round(min(1.0, filled), 4),
                    class_id=UNCLASSIFIED,
                )
            )

        detections.sort(key=lambda d: d.bbox.w * d.bbox.h, reverse=True)
        return detections


# --------------------------------------------------------------------- ONNX


#: What a security console watches by default when a model can name classes:
#: people and the vehicles they arrive in. Measured on the laptop camera with
#: every COCO class tracked: a jar on a shelf and a phone on the desk became
#: "bottle" and "cell phone" tracks at 0.43–0.51 confidence, labelled in the
#: same green as the person, and the operator's word for them was
#: "hallucinations". They were not — the model saw a jar — but they were noise,
#: and a system measured by how little it says must not track what nobody asked
#: it to watch. Nothing is lost: the whole vocabulary stays available, and a
#: site that wants "dog" adds it.
WATCHED_LABELS = frozenset({"person", "bicycle", "car", "motorcycle", "bus", "truck"})


def _restrict_vocabulary(
    names: dict[int, str], classes: "Iterable[str] | None"
) -> "tuple[dict[int, str], np.ndarray | None]":
    """The part of a model's vocabulary an operator asked to watch.

    Returns the names to report and the class ids to keep — ``None`` for the
    ids when nothing was asked, which keeps everything, as before. The reported
    names shrink with the filter on purpose: a zone's class picker reads them,
    and offering "bottle" to a zone on a site whose detector drops bottles would
    be a filter that silently disarms that zone.

    A name the model does not know is refused rather than ignored: ignoring it
    would let a typo in "person" watch nothing and say nothing about it. So is
    an empty list, because "watch nothing" is never what anybody meant.
    """
    if classes is None:
        return dict(names), None
    wanted = {str(label).strip().lower() for label in classes if str(label).strip()}
    if not wanted:
        raise DetectionError(
            "asked to watch no class at all; leave the watch list unset to watch everything"
        )
    if not names:
        raise DetectionError(
            f"asked to watch {', '.join(sorted(wanted))}, but this model names no classes"
        )
    known = {name.strip().lower(): class_id for class_id, name in names.items()}
    unknown = sorted(wanted - set(known))
    if unknown:
        raise DetectionError(
            f"asked to watch {', '.join(unknown)}, which this model does not name; "
            f"it knows {', '.join(sorted(known))}"
        )
    kept = {known[label]: names[known[label]] for label in sorted(wanted)}
    return kept, np.asarray(sorted(kept), dtype=np.int64)


def detector_for(
    model_path: "str | Path | None" = None, **options
) -> "Detector":
    """The right detector for what the operator supplied.

    One place that decides, because there are now three and the difference
    between them is not a preference — it is what the system is capable of
    concluding:

    - **No model** gives the motion detector. It answers "what changed", which
      includes a curtain, and it cannot see anything that has stopped moving.
      Free, fast, and the reason a real webcam produced twenty tracks for one
      seated person.
    - **A model with two outputs** is instance segmentation: a mask per object,
      a class, and a ground-contact point taken from the object's own lowest
      pixel rather than from a rectangle's bottom edge.
    - **A model with one output** is a detector: boxes and classes, no masks.

    The choice is made by *reading the model*, not by a flag, because a flag can
    disagree with the file and the operator would have no way to tell which won.
    """
    if model_path is None:
        # A motion detector cannot name a class, so it cannot watch one. The
        # list is dropped rather than refused: the console applies its watch
        # list to whatever detector it has, and a motion-only site is not an
        # error.
        options.pop("classes", None)
        # Nor is its confidence a probability: it is the fraction of a box
        # that actually moved (see `MotionDetector.detect`), so a floor meant
        # for a classifier's score would silently throw away solid, real
        # movement below it. The console hands one set of options to whatever
        # detector it has; motion takes the ones that apply to it.
        options.pop("confidence_threshold", None)
        return MotionDetector(**options)

    path = Path(model_path)
    outputs = _output_count(path)

    if outputs == 2:
        from .segment import Segmenter

        return Segmenter(path, **options)
    return OnnxDetector(path, **options)


def _output_count(path: "str | Path") -> int:
    """How many tensors the model produces, without loading it for inference.

    Reads the graph only. Building a full session to ask a structural question
    would pay the optimisation cost twice.
    """
    from . import telemetry

    telemetry.silence()
    import onnxruntime as ort

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise DetectionError(
            f"No model at {resolved}. Models are supplied by the operator; "
            "nothing is ever downloaded."
        )
    try:
        session = ort.InferenceSession(
            str(resolved), providers=["CPUExecutionProvider"]
        )
    except Exception as error:
        raise DetectionError(f"Could not load the model at {resolved}: {error}") from error
    return len(session.get_outputs())


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _Layout:
    """How to read a model's output tensor.

    Detection models disagree about this, and guessing wrong does not fail — it
    produces detections in the wrong places. So the layout is determined from the
    tensor shape, and an unrecognised shape is refused.
    """

    #: True for ``[1, 4+nc, N]`` (YOLOv8 and similar), False for ``[1, N, 4+1+nc]``.
    channels_first: bool
    #: Number of classes.
    class_count: int
    #: True when the model emits a separate objectness score (YOLOv5 family).
    has_objectness: bool


class OnnxDetector:
    """Runs an operator-supplied ONNX detection model.

    Nothing is downloaded. The model is read from a path under the models
    directory, and its digest is recorded so an event can name the exact weights
    that produced it.
    """

    __slots__ = ("_watched_ids", "_session", "_input_name", "_input_size", "_layout", "_info",
                 "_confidence", "_iou", "_letterbox")

    def __init__(
        self,
        model_path: str | Path,
        *,
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        class_names: dict[int, str] | None = None,
        providers: Sequence[str] | None = None,
        letterbox: bool = True,
        classes: "Iterable[str] | None" = None,
    ):
        import onnxruntime as ort

        path = Path(model_path).resolve()
        if not path.is_file():
            raise DetectionError(
                f"No model at {path}. Models are supplied by the operator and "
                "placed in the models directory; nothing is ever downloaded."
            )

        options = ort.SessionOptions()
        # A security appliance should not be writing optimised model copies next
        # to the operator's files.
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Telemetry off. The runtime API is the *second* half of this — the
        # first is `ORT_DISABLE_TELEMETRY`, set by `telemetry.silence()` at every
        # entry point, because the native library reads it when it initialises
        # and Microsoft's own documentation says an initialisation event may
        # already have been sent before any Python call can reach the switch.
        #
        # This is not hypothetical. The manylinux wheel of the version pinned
        # here contains a Microsoft 1DS collector endpoint with an ingestion
        # token, a statically linked mbedTLS stack, a persistent device-id
        # database, and the payload fields `osDescription`, `cpuModel` and
        # `totalMemoryMB` — verified by scanning the shipped `.so`, not by
        # reading documentation. Telemetry is ON by default in the official
        # builds. The host is deliberately not written here: the offline audit
        # refuses shipped source that names a destination, and a comment is
        # exactly how one gets in. `tools/binary_audit.py` holds the string,
        # because finding it is that file's job.
        telemetry.silence()
        telemetry.silence_runtime_apis()

        try:
            session = ort.InferenceSession(
                str(path), sess_options=options,
                providers=list(providers) if providers else ["CPUExecutionProvider"],
            )
        except Exception as error:
            raise DetectionError(f"Could not load the model at {path}: {error}") from error

        inputs = session.get_inputs()
        if len(inputs) != 1:
            raise DetectionError(
                f"Expected a model with one input; {path.name} has {len(inputs)}."
            )

        shape = inputs[0].shape
        if len(shape) != 4:
            raise DetectionError(
                f"Expected a 4-dimensional image input; {path.name} declares {shape}."
            )

        # Dynamic axes come back as strings. Fall back to the common 640 only
        # when the model genuinely does not state a size.
        height = shape[2] if isinstance(shape[2], int) and shape[2] > 0 else 640
        width = shape[3] if isinstance(shape[3], int) and shape[3] > 0 else 640

        self._session = session
        self._input_name = inputs[0].name
        self._input_size = (int(width), int(height))
        self._confidence = confidence_threshold
        self._iou = iou_threshold
        self._letterbox = letterbox
        self._layout = _infer_layout(session.get_outputs()[0].shape, path.name)

        names = dict(class_names) if class_names else _names_from_metadata(session)
        # `classes` is what the operator asked to watch; everything else the
        # model can see is dropped below, before it can become a track.
        names, self._watched_ids = _restrict_vocabulary(names, classes)
        self._info = DetectorInfo(
            kind="onnx",
            name=path.stem,
            model_path=str(path),
            model_sha256=_sha256(path),
            input_size=self._input_size,
            class_names=names,
            classifies=True,
        )

    @property
    def info(self) -> DetectorInfo:
        return self._info

    def detect(self, image: np.ndarray) -> list[Detection]:
        tensor, scale, pad = self._preprocess(image)
        outputs = self._session.run(None, {self._input_name: tensor})
        return self._postprocess(outputs[0], image.shape[1], image.shape[0], scale, pad)

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, tuple[float, float]]:
        target_w, target_h = self._input_size
        h, w = image.shape[:2]

        if self._letterbox:
            # Preserve aspect ratio. Stretching a 16:9 frame into a square makes
            # every person short and wide, and a model trained on letterboxed
            # input then misses them.
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
            pad_x = (target_w - new_w) / 2.0
            pad_y = (target_h - new_h) / 2.0
            canvas[int(pad_y) : int(pad_y) + new_h, int(pad_x) : int(pad_x) + new_w] = resized
            prepared, padding = canvas, (pad_x, pad_y)
        else:
            prepared = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            scale, padding = 1.0, (0.0, 0.0)

        rgb = cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        return np.ascontiguousarray(tensor.transpose(2, 0, 1)[None]), scale, padding

    def _postprocess(
        self,
        raw: np.ndarray,
        frame_w: int,
        frame_h: int,
        scale: float,
        pad: tuple[float, float],
    ) -> list[Detection]:
        predictions = raw[0]
        if self._layout.channels_first:
            predictions = predictions.T

        boxes = predictions[:, :4]
        if self._layout.has_objectness:
            objectness = predictions[:, 4]
            class_scores = predictions[:, 5:]
            scores = objectness[:, None] * class_scores
        else:
            scores = predictions[:, 4:]

        if scores.size == 0:
            return []

        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(scores.shape[0]), class_ids]

        keep = confidences >= self._confidence
        if self._watched_ids is not None:
            keep &= np.isin(class_ids, self._watched_ids)
        if not np.any(keep):
            return []

        boxes = boxes[keep]
        class_ids = class_ids[keep]
        confidences = confidences[keep]

        # Models emit centre/width/height in input-tensor pixels. Undo the
        # letterbox before anything else looks at these numbers.
        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = (cx - bw / 2 - pad[0]) / scale
        y1 = (cy - bh / 2 - pad[1]) / scale
        x2 = (cx + bw / 2 - pad[0]) / scale
        y2 = (cy + bh / 2 - pad[1]) / scale

        xyxy = np.stack([x1, y1, x2, y2], axis=1)
        kept = _non_max_suppression(xyxy, confidences, class_ids, self._iou)

        detections: list[Detection] = []
        for index in kept:
            left = float(np.clip(xyxy[index, 0], 0, frame_w))
            top = float(np.clip(xyxy[index, 1], 0, frame_h))
            right = float(np.clip(xyxy[index, 2], 0, frame_w))
            bottom = float(np.clip(xyxy[index, 3], 0, frame_h))
            if right <= left or bottom <= top:
                continue

            detections.append(
                Detection(
                    bbox=BoundingBox(
                        left / frame_w,
                        top / frame_h,
                        (right - left) / frame_w,
                        (bottom - top) / frame_h,
                    ),
                    confidence=float(confidences[index]),
                    class_id=int(class_ids[index]),
                )
            )

        return detections


def _infer_layout(shape: Sequence[object], model_name: str) -> _Layout:
    """Work out how to read the output tensor, or refuse.

    Guessing wrong here yields detections at plausible but wrong coordinates,
    which is worse than not running at all.
    """
    if len(shape) != 3:
        raise DetectionError(
            f"{model_name} produces a rank-{len(shape)} output; this reader "
            "understands [1, channels, anchors] and [1, anchors, channels]."
        )

    dims = [d if isinstance(d, int) and d > 0 else None for d in shape]
    _, a, b = dims
    if a is None or b is None:
        raise DetectionError(
            f"{model_name} declares a dynamic output shape {list(shape)}. The "
            "layout cannot be determined without running it, and guessing "
            "produces boxes in the wrong places."
        )

    # The anchor count is always far larger than the channel count.
    channels_first = a < b
    channels = a if channels_first else b

    if channels < 5:
        raise DetectionError(
            f"{model_name} emits {channels} channels; a detection head needs at "
            "least 4 box values plus one score."
        )

    # v5-family heads carry an objectness column; v8-family do not. The two are
    # indistinguishable from shape alone, so assume the v8 layout and let the
    # caller override — but only when the channel count is consistent with it.
    return _Layout(channels_first=channels_first, class_count=channels - 4, has_objectness=False)


def _names_from_metadata(session: object) -> dict[int, str]:
    """Class names carried inside the model, if it carries any.

    Never falls back to a standard class list. A model whose classes are unknown
    reports numeric ids, because attaching "person" to an output that might mean
    something else is fabricated evidence.
    """
    try:
        metadata = session.get_modelmeta().custom_metadata_map  # type: ignore[attr-defined]
    except Exception:
        return {}

    raw = metadata.get("names")
    if not raw:
        return {}

    try:
        import ast

        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return {}

    if isinstance(parsed, dict):
        return {int(k): str(v) for k, v in parsed.items()}
    if isinstance(parsed, (list, tuple)):
        return {index: str(value) for index, value in enumerate(parsed)}
    return {}


def _non_max_suppression(
    boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, threshold: float
) -> list[int]:
    """Per-class NMS.

    Per-class, not global: a person standing in front of a car should not
    suppress the car. Suppressing across classes is how detections quietly go
    missing in crowded scenes.
    """
    kept: list[int] = []
    for class_id in np.unique(class_ids):
        indices = np.flatnonzero(class_ids == class_id)
        order = indices[np.argsort(-scores[indices])]

        while order.size:
            best = int(order[0])
            kept.append(best)
            if order.size == 1:
                break

            rest = order[1:]
            xx1 = np.maximum(boxes[best, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[best, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[best, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[best, 3], boxes[rest, 3])

            overlap = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            area_best = (boxes[best, 2] - boxes[best, 0]) * (boxes[best, 3] - boxes[best, 1])
            area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
            union = area_best + area_rest - overlap
            iou = np.where(union > 0, overlap / union, 0.0)

            order = rest[iou < threshold]

    return sorted(kept)


# ------------------------------------------------------------------ timing


@dataclass
class DetectorTiming:
    """Measured throughput. Reported, never estimated.

    A claim about frame rate that was not measured on this machine with this
    model is marketing, not engineering.
    """

    frames: int = 0
    total_seconds: float = 0.0
    slowest_seconds: float = 0.0

    def record(self, seconds: float) -> None:
        self.frames += 1
        self.total_seconds += seconds
        self.slowest_seconds = max(self.slowest_seconds, seconds)

    @property
    def fps(self) -> float:
        return self.frames / self.total_seconds if self.total_seconds > 0 else 0.0

    @property
    def mean_millis(self) -> float:
        return 1000.0 * self.total_seconds / self.frames if self.frames else 0.0


class TimedDetector:
    """Wraps a detector and measures it."""

    __slots__ = ("_inner", "timing")

    def __init__(self, inner: Detector):
        self._inner = inner
        self.timing = DetectorTiming()

    @property
    def info(self) -> DetectorInfo:
        return self._inner.info

    def detect(self, image: np.ndarray) -> list[Detection]:
        started = time.perf_counter()
        try:
            return self._inner.detect(image)
        finally:
            self.timing.record(time.perf_counter() - started)
