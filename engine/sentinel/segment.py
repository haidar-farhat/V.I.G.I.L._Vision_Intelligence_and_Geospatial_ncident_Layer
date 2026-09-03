"""Instance segmentation: a mask per object, not a box.

A box is a claim about a rectangle. A person is not a rectangle, and the
difference is not cosmetic — it decides where the system thinks somebody is
standing. The bottom-centre of a bounding box is the ground contact point only
when the box is tight and the person is upright and unoccluded; for anybody
leaning, carrying something, or half behind a car, it is somewhere in the air or
inside the obstacle. A mask's lowest pixel is where they actually meet the
ground.

**Why this exists at all.** Pointed at a real webcam for twelve seconds, the
motion detector reported twenty tracks for one seated person — fragments of a
face, plus curtains and a wall — with track ids already in the eighties. The
synthetic reference scene had flattered it enormously; four tracks for three
people reads as a mild over-count, and on real video it was not an over-count,
it was noise. Background subtraction answers "what changed", and what changed
includes a curtain. A segmentation model answers "what is this", which is the
question the rest of the system has been assuming an answer to.

The model is **operator-supplied and never downloaded** — the same rule as every
other model here. `devtools/export_model.py` produces one on a connected
machine; this loads whatever `.onnx` it is pointed at.

**What it produces.** One `Detection` per instance, carrying a boolean mask
cropped to its own box. Cropped rather than full-frame because a full-frame mask
per detection is megabytes per frame at video rate, and every consumer already
knows the box.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from . import telemetry
from .core import BoundingBox, Detection
from .detect import (
    DetectionError,
    DetectorInfo,
    _names_from_metadata,
    _non_max_suppression,
    _sha256,
)
from .logs import get as _get_logger

_log = _get_logger(__name__)

#: Mask prototypes are combined and squashed through a sigmoid, so this is a
#: probability. Half is the conventional cut and is what the exported model was
#: calibrated against; moving it trades missed limbs for haloes.
MASK_THRESHOLD = 0.5

#: A mask smaller than this is not an object, it is a speck of activation. The
#: box that survives NMS can still be a few pixels across at the edge of the
#: frame, and a two-pixel mask has no meaningful lowest point.
MINIMUM_MASK_PIXELS = 12


class Segmenter:
    """Runs an operator-supplied instance-segmentation model through onnxruntime.

    Deliberately built on the same session, letterboxing and NMS conventions as
    `OnnxDetector`. Two implementations of "resize, pad, un-pad" is two chances
    to be off by half a pixel in different directions, and the resulting error
    looks like bad geometry rather than a bug.
    """

    __slots__ = (
        "_session", "_input_name", "_input_size", "_info", "_confidence",
        "_iou", "_mask_threshold", "_outputs", "_proto_index", "_pred_index",
    )

    def __init__(
        self,
        model_path: str | Path,
        *,
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        mask_threshold: float = MASK_THRESHOLD,
        class_names: dict[int, str] | None = None,
        providers: Sequence[str] | None = None,
    ):
        # Before onnxruntime loads. See `sentinel/telemetry.py`: the manylinux
        # wheels carry a collector that is on by default, and the environment
        # variable is read when the native library initialises.
        telemetry.silence()

        import onnxruntime as ort

        telemetry.silence_runtime_apis()

        path = Path(model_path).resolve()
        if not path.is_file():
            raise DetectionError(
                f"No model at {path}. Models are supplied by the operator and "
                "placed in the models directory; nothing is ever downloaded. "
                "Produce one with devtools/export_model.py on a connected machine."
            )

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

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
                f"Expected a 4-dimensional input; {path.name} declares {shape}."
            )

        height = shape[2] if isinstance(shape[2], int) else 640
        width = shape[3] if isinstance(shape[3], int) else 640

        outputs = session.get_outputs()
        if len(outputs) != 2:
            raise DetectionError(
                f"{path.name} has {len(outputs)} outputs. An instance-segmentation "
                "model has two: predictions, and mask prototypes. A model with one "
                "is a detector — load it with OnnxDetector, which will not pretend "
                "to produce masks."
            )

        # Which output is which is decided by rank, not by name. Exporters
        # disagree about names and agree about shape: prototypes are
        # (1, protos, h, w) and predictions are (1, channels, anchors).
        first, second = outputs[0].shape, outputs[1].shape
        if len(first) == 4:
            self._proto_index, self._pred_index = 0, 1
        elif len(second) == 4:
            self._proto_index, self._pred_index = 1, 0
        else:
            raise DetectionError(
                f"{path.name} has two outputs but neither is a 4-dimensional "
                f"prototype tensor ({first}, {second}), so masks cannot be built."
            )

        self._session = session
        self._input_name = inputs[0].name
        self._input_size = (int(width), int(height))
        self._confidence = confidence_threshold
        self._iou = iou_threshold
        self._mask_threshold = mask_threshold
        self._outputs = [output.name for output in outputs]

        names = dict(class_names) if class_names else _names_from_metadata(session)
        self._info = DetectorInfo(
            kind="onnx-segment",
            name=f"{path.stem} instance segmentation",
            model_path=str(path),
            model_sha256=_sha256(path),
            input_size=(int(width), int(height)),
            class_names=names,
            # It classifies only if the model actually carries names. A model
            # with none produces class ids nothing can interpret, and inventing
            # labels for them is the failure this flag exists to prevent.
            classifies=bool(names),
        )
        _log.info(
            "segmenter ready: %s, %dx%d, %d class name(s)",
            path.name, width, height, len(names),
        )

    @property
    def info(self) -> DetectorInfo:
        return self._info

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detections with masks. Satisfies the same protocol as every detector."""
        tensor, scale, pad = self._preprocess(image)
        predictions, prototypes = self._run(tensor)
        return self._postprocess(
            predictions, prototypes, image.shape[1], image.shape[0], scale, pad
        )

    # ------------------------------------------------------------------ inside

    def _run(self, tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        results = self._session.run(self._outputs, {self._input_name: tensor})
        return results[self._pred_index], results[self._proto_index]

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, tuple[float, float]]:
        """Letterbox to the model's input. Identical in effect to OnnxDetector's.

        Aspect ratio is preserved: stretching a 16:9 frame into a square makes
        every person short and wide, and a model trained on letterboxed input
        then misses them.
        """
        target_w, target_h = self._input_size
        h, w = image.shape[:2]

        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        pad_x = (target_w - new_w) / 2.0
        pad_y = (target_h - new_h) / 2.0
        canvas[int(pad_y) : int(pad_y) + new_h, int(pad_x) : int(pad_x) + new_w] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        return np.ascontiguousarray(tensor.transpose(2, 0, 1)[None]), scale, (pad_x, pad_y)

    def _postprocess(
        self,
        raw: np.ndarray,
        prototypes: np.ndarray,
        frame_w: int,
        frame_h: int,
        scale: float,
        pad: tuple[float, float],
    ) -> list[Detection]:
        predictions = raw[0]
        if predictions.shape[0] < predictions.shape[1]:
            # (channels, anchors) -> (anchors, channels).
            predictions = predictions.T

        protos = prototypes[0]
        proto_count = protos.shape[0]
        channels = predictions.shape[1]
        class_count = channels - 4 - proto_count
        if class_count <= 0:
            raise DetectionError(
                f"the model's {channels} prediction channels do not decompose "
                f"into 4 box + classes + {proto_count} mask coefficients"
            )

        boxes = predictions[:, :4]
        scores = predictions[:, 4 : 4 + class_count]
        coefficients = predictions[:, 4 + class_count :]

        if scores.size == 0:
            return []

        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(scores.shape[0]), class_ids]

        keep = confidences >= self._confidence
        if not np.any(keep):
            return []

        boxes = boxes[keep]
        class_ids = class_ids[keep]
        confidences = confidences[keep]
        coefficients = coefficients[keep]

        # Centre-width-height to corners, in letterboxed pixels.
        corners = np.empty_like(boxes)
        corners[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        corners[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        corners[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        corners[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

        survivors = _non_max_suppression(corners, confidences, class_ids, self._iou)
        if not survivors:
            return []

        pad_x, pad_y = pad
        proto_h, proto_w = protos.shape[1], protos.shape[2]
        flat = protos.reshape(proto_count, -1).astype(np.float32)
        target_w, target_h = self._input_size

        detections: list[Detection] = []
        for index in survivors:
            # The mask, at prototype resolution, for this instance alone.
            combined = coefficients[index].astype(np.float32) @ flat
            mask = 1.0 / (1.0 + np.exp(-combined.reshape(proto_h, proto_w)))

            box = corners[index]
            # Prototype space is the letterboxed input scaled down, so the box
            # maps into it by the same ratio in both axes.
            gain_x, gain_y = proto_w / target_w, proto_h / target_h
            px1 = int(max(0, np.floor(box[0] * gain_x)))
            py1 = int(max(0, np.floor(box[1] * gain_y)))
            px2 = int(min(proto_w, np.ceil(box[2] * gain_x)))
            py2 = int(min(proto_h, np.ceil(box[3] * gain_y)))
            if px2 <= px1 or py2 <= py1:
                continue

            # Cropped to the box *before* thresholding. A prototype activation
            # is global, so a person's coefficients light up faintly on other
            # people too; cropping is what makes the mask this instance's.
            crop = mask[py1:py2, px1:px2]

            # Un-letterbox the box into frame pixels.
            x1 = (box[0] - pad_x) / scale
            y1 = (box[1] - pad_y) / scale
            x2 = (box[2] - pad_x) / scale
            y2 = (box[3] - pad_y) / scale

            x1 = float(np.clip(x1, 0, frame_w))
            y1 = float(np.clip(y1, 0, frame_h))
            x2 = float(np.clip(x2, 0, frame_w))
            y2 = float(np.clip(y2, 0, frame_h))
            if x2 <= x1 or y2 <= y1:
                continue

            width_px = int(round(x2 - x1))
            height_px = int(round(y2 - y1))
            if width_px < 1 or height_px < 1:
                continue

            resized = cv2.resize(
                crop, (width_px, height_px), interpolation=cv2.INTER_LINEAR
            )
            binary = (resized >= self._mask_threshold).astype(np.uint8)
            if int(binary.sum()) < MINIMUM_MASK_PIXELS:
                # A handful of lit pixels has no meaningful lowest point, and a
                # ground contact derived from one would be noise presented as a
                # position.
                continue

            detections.append(
                Detection(
                    bbox=BoundingBox(
                        x=x1 / frame_w, y=y1 / frame_h,
                        w=(x2 - x1) / frame_w, h=(y2 - y1) / frame_h,
                    ),
                    confidence=float(confidences[index]),
                    class_id=int(class_ids[index]),
                    mask=binary,
                )
            )

        return detections


def ground_contact(detection: Detection, frame_w: int, frame_h: int) -> tuple[float, float]:
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
    """
    box = detection.bbox
    if detection.mask is None or detection.mask.size == 0:
        return (box.x + box.w / 2.0, box.y + box.h)

    rows = np.flatnonzero(detection.mask.any(axis=1))
    if rows.size == 0:
        return (box.x + box.w / 2.0, box.y + box.h)

    lowest = int(rows[-1])
    columns = np.flatnonzero(detection.mask[lowest])
    centre = float(columns.mean()) if columns.size else detection.mask.shape[1] / 2.0

    height, width = detection.mask.shape
    return (
        box.x + box.w * (centre + 0.5) / width,
        box.y + box.h * (lowest + 1) / height,
    )
