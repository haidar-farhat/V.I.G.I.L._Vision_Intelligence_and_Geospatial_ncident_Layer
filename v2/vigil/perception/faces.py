"""Finding a face inside a person, and turning it into a vector.

# What this does and does not decide

It produces embeddings. It does not decide who anybody is — `domain.identity`
does that, and it is structurally incapable of asserting a name from one
frame. The split is deliberate: the part that touches pixels and the part that
makes a claim about a person should not be the same file, because the second
needs to be readable by somebody who does not care how the first works.

# The person box is required

`embed_faces` takes a person box and searches **inside it only**. There is no
default that means "the whole frame", and that absence is the single most
important privacy property here: a face pipeline pointed at the frame is a
face pipeline pointed at whoever is walking past outside the fence, and a
default argument is all it would take.

# Quality is a detector score, and is not called a confidence

v1 clipped its detector's raw score into `[0, 1]`, stored it as `quality`,
showed it to operators, and used it to choose which biometric to keep. A
detector's box score is not a probability of anything; clipping it makes the
fabrication well-formed rather than true. Here the number is carried under the
name `detector_score`, its docstring says what it is not, and nothing
thresholds a *person* on it.

# Nothing is downloaded, and nothing runs when the switch is off

The models are operator-supplied, opened through `kernel.onnx` with the same
digest-and-provider discipline as the object detector. `FaceReader` is only
constructed when the identity switch is on; there is no `enabled` flag inside
it to be checked in twelve places and missed in one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..kernel import onnx as _onnx
from ..logs import get as _get_logger

_log = _get_logger(__name__)

#: Fewest pixels on a side of a *face* box for the crop to be worth embedding.
#:
#: Thirty-two. Recognition models take 112x112, so this is under a third of
#: their input in each axis and everything below it is being invented by the
#: resize. Distinct from the person-crop minimum below, which v1 conflated:
#: a 32x32 person cannot contain a 32px face, so one number could not be right
#: for both.
MIN_FACE_PIXELS = 32

#: Fewest pixels on a side of the *person* crop that is searched.
MIN_PERSON_PIXELS = 64

#: Detector score below which a proposed face box is not embedded.
#:
#: A stated default with no measurement behind it — no face model ships here.
#: A low-scoring box from a face detector is as often a hand, a poster or a
#: wheel arch, and embedding one puts a vector of nothing into the register.
MIN_DETECTOR_SCORE = 0.6


class FaceError(RuntimeError):
    """A face model that is missing or cannot be used."""


@dataclass(frozen=True, slots=True)
class Face:
    """One face found inside one person, and its embedding.

    The embedding is unit length on construction, because everything
    downstream compares by cosine distance and a cosine distance is only a
    distance for unit vectors.
    """

    #: Box in frame fractions, so it is in the same coordinates as everything
    #: else and can be redacted without a second conversion.
    x: float
    y: float
    width: float
    height: float
    embedding: np.ndarray
    #: The face detector's own box score.
    #:
    #: **Not a confidence and not a probability.** It is an uncalibrated
    #: number a detector emits, useful only for ordering candidate boxes from
    #: the same detector on the same frame. It is never compared against a
    #: threshold that decides anything about a person.
    detector_score: float
    model_sha256: str

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height


@dataclass(frozen=True, slots=True)
class FaceModels:
    """The detector and the embedder, which are useless apart."""

    detector: Path
    embedder: Path

    def missing(self) -> list[Path]:
        return [p for p in (self.detector, self.embedder) if not Path(p).is_file()]

    def require_present(self) -> None:
        absent = self.missing()
        if absent:
            raise FaceError(
                "these face models are missing and are supplied by the operator — nothing is "
                "downloaded:\n  " + "\n  ".join(str(p) for p in absent))


class FaceReader:
    """Faces inside a person box, as embeddings.

    Constructed only when the identity switch is on. There is deliberately no
    `enabled` flag on it: a feature that is off should be a `None` where the
    object would be, not an object that remembers to do nothing.
    """

    def __init__(self, models: FaceModels, *, min_detector_score: float = MIN_DETECTOR_SCORE,
                 min_face_pixels: int = MIN_FACE_PIXELS):
        models.require_present()
        self.models = models
        self.min_detector_score = min_detector_score
        self.min_face_pixels = min_face_pixels
        self._detector, self.provider = _onnx.open_session(models.detector)
        self._embedder, _ = _onnx.open_session(models.embedder)
        self._detector_input = self._detector.get_inputs()[0].name
        self._detector_outputs = [o.name for o in self._detector.get_outputs()]
        self._embedder_input = self._embedder.get_inputs()[0].name
        self._embedder_outputs = [o.name for o in self._embedder.get_outputs()]
        shape = self._embedder.get_inputs()[0].shape
        self._embed_size = (
            int(shape[3]) if isinstance(shape[3], int) else 112,
            int(shape[2]) if isinstance(shape[2], int) else 112,
        )
        #: The **embedder's** digest identifies the vector space. Two
        #: embedders' vectors are not comparable at all, and this is the value
        #: `domain.identity` refuses a cross-model comparison on.
        self.model_sha256 = _onnx.digest(models.embedder)
        _log.info("face reader ready: %s + %s on %s",
                  Path(models.detector).name, Path(models.embedder).name, self.provider)

    def embed_faces(self, image: np.ndarray, person_box) -> list[Face]:
        """Every face inside `person_box`, embedded.

        `person_box` is required and is a `BoundingBox` in frame fractions.
        The search happens inside the crop and nowhere else.
        """
        crop, origin = _crop(image, person_box, MIN_PERSON_PIXELS)
        if crop is None:
            return []
        boxes = self._detect(crop)
        out: list[Face] = []
        height, width = image.shape[:2]
        for x1, y1, x2, y2, score in boxes:
            if score < self.min_detector_score:
                continue
            if min(x2 - x1, y2 - y1) < self.min_face_pixels:
                continue
            face = crop[int(y1):int(y2), int(x1):int(x2)]
            if face.size == 0:
                continue
            embedding = self._embed(face)
            if embedding is None:
                continue
            out.append(Face(
                x=(origin[0] + x1) / width, y=(origin[1] + y1) / height,
                width=(x2 - x1) / width, height=(y2 - y1) / height,
                embedding=embedding, detector_score=float(score),
                model_sha256=self.model_sha256,
            ))
        return out

    # ------------------------------------------------------------- the models

    def _detect(self, crop: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        """Face boxes in the crop's own pixels.

        Output shapes differ between face detectors, so this reads the common
        `(n, >=5)` form of `x1, y1, x2, y2, score` and **refuses** anything
        else rather than guessing at an axis. A guessed axis decodes boxes
        from scores, which produces faces in plausible places.
        """
        import cv2

        blob = cv2.resize(crop, (320, 320), interpolation=cv2.INTER_LINEAR)
        blob = blob[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        try:
            outputs = self._detector.run(self._detector_outputs, {self._detector_input: blob})
        except Exception as error:  # noqa: BLE001 - onnxruntime raises many types
            _log.debug("face detection failed: %s", error)
            return []
        raw = np.asarray(outputs[0])
        while raw.ndim > 2:
            raw = raw[0]
        if raw.ndim != 2 or raw.shape[1] < 5:
            _log.error("the face detector returned shape %s, which this build cannot read; "
                       "expected (n, >=5) of x1,y1,x2,y2,score", raw.shape)
            return []
        h, w = crop.shape[:2]
        sx, sy = w / 320.0, h / 320.0
        return [(float(r[0]) * sx, float(r[1]) * sy, float(r[2]) * sx, float(r[3]) * sy,
                 float(r[4])) for r in raw]

    def _embed(self, face: np.ndarray) -> np.ndarray | None:
        import cv2

        resized = cv2.resize(face, self._embed_size, interpolation=cv2.INTER_LINEAR)
        blob = resized[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        try:
            outputs = self._embedder.run(self._embedder_outputs, {self._embedder_input: blob})
        except Exception as error:  # noqa: BLE001
            _log.debug("face embedding failed: %s", error)
            return None
        vector = np.asarray(outputs[0], dtype=np.float64).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm < 1e-9:
            # A zero vector has a cosine distance of exactly 1.0 from
            # everything, which sits outside every sensible threshold but does
            # not look wrong on a screen. Refused rather than stored.
            return None
        return vector / norm


#: How far outside `[0, 1]` a box may stray and still be treated as a box.
#:
#: A thousandth. Real boxes overshoot by an ulp or two after a clamp or a
#: rescale, and refusing those would reject legitimate detections at the frame
#: edge. Anything further out is not a normalised box at all, and the
#: important part is that it is **refused rather than clamped**: clamping
#: `BoundingBox(0, 0, 200.0, 150.0)` produces the whole frame, which is
#: exactly the thing requiring a person box was meant to prevent. v1 clamped,
#: and shipped that hole for a year.
BOX_TOLERANCE = 1e-3


def _sane(box) -> bool:
    return (box.width > 0 and box.height > 0
            and box.x >= -BOX_TOLERANCE and box.y >= -BOX_TOLERANCE
            and box.right <= 1.0 + BOX_TOLERANCE and box.bottom <= 1.0 + BOX_TOLERANCE)


def _crop(image: np.ndarray, box, minimum: int) -> tuple[np.ndarray | None, tuple[int, int]]:
    """The box's pixels and where they came from, or `(None, ...)`.

    The **size** is bounds-checked as well as the offset. v1 checked x and y
    only, and a box of width 200.0 in a normalised frame passed that check and
    handed the face detector the entire picture — which is precisely the
    failure the person box exists to prevent.
    """
    if image is None or image.ndim != 3 or not _sane(box):
        return None, (0, 0)
    height, width = image.shape[:2]
    x1 = int(max(0.0, min(1.0, box.x)) * width)
    y1 = int(max(0.0, min(1.0, box.y)) * height)
    x2 = int(max(0.0, min(1.0, box.right)) * width)
    y2 = int(max(0.0, min(1.0, box.bottom)) * height)
    if x2 - x1 < minimum or y2 - y1 < minimum:
        return None, (0, 0)
    return image[y1:y2, x1:x2], (x1, y1)


def blur(image: np.ndarray, boxes, *, strength: int = 31) -> np.ndarray:
    """The frame with each box blurred beyond recognition.

    Used by the export path. A blur rather than a black rectangle so that an
    exported clip still shows that somebody was there and roughly what they
    did, which is usually the point of the clip; a black box removes the
    evidence along with the face.

    The kernel is a fraction of the box rather than a fixed size, because a
    fixed 31-pixel kernel over a 400-pixel face is a soft-focus portrait.
    """
    import cv2

    if image is None or image.ndim != 3 or not boxes:
        return image
    out = image.copy()
    height, width = image.shape[:2]
    for box in boxes:
        x1 = int(max(0.0, min(1.0, box.x)) * width)
        y1 = int(max(0.0, min(1.0, box.y)) * height)
        x2 = int(max(0.0, min(1.0, box.right)) * width)
        y2 = int(max(0.0, min(1.0, box.bottom)) * height)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        region = out[y1:y2, x1:x2]
        kernel = max(strength, (min(x2 - x1, y2 - y1) // 4) | 1)
        kernel = kernel if kernel % 2 else kernel + 1
        out[y1:y2, x1:x2] = cv2.GaussianBlur(region, (kernel, kernel), 0)
    return out
