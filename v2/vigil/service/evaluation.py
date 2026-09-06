"""Scoring the detector against labels a person corrected.

# What this can and cannot tell you

It computes precision, recall and average precision per class against
**corrected** labels — YOLO `.txt` files a person has fixed. Everything it
reports is a statement about the frames it was given and about nothing else.
No dataset ships with this product, so until somebody labels one this module
has nothing to score, and it says so rather than producing a number.

That is the whole point of it existing before there is data: the harness has
to be here, and honest, so that the day labels appear there is no temptation
to write a quick script that scores the training set.

# The refusal that matters

**A validation set sharing a day with training is refused.** Consecutive
frames of security footage are nearly identical, so a random split leaks
almost perfectly: the model is tested on frame 101 having trained on frame
100, scores 0.98, and means nothing by it. `dataset.export` already splits by
whole days for this reason; this checks that whatever it is handed kept that
property, because a corpus can be re-split by hand between the two.

`vigil eval` on a leaking split does not warn and continue. It stops, because
the number it would print is worse than no number: a warning above a 0.98 is
read as a 0.98.

# Matching, and the choices inside it

A prediction matches a label when they are the same class and overlap by at
least `MIN_IOU`. Greedy by descending confidence, each label claimed once —
the standard protocol, stated because there are several and they disagree.

Average precision is the area under the precision-recall curve by the
**all-points** interpolation, not the 11-point one: the 11-point version
quantises to steps of 0.1 recall, which on a class with nine instances is
noise dressed as a metric.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from ..logs import get as _get_logger

_log = _get_logger(__name__)

#: Overlap at which a prediction is counted as finding a labelled object.
#:
#: 0.5, the convention, so a figure from here is comparable with a published
#: one. Stated rather than assumed: at 0.5 a box half the size of the object,
#: in the right place, counts as a find, which is right for "did it see the
#: person" and wrong for "did it place them accurately". The second question
#: is the projection's, and `PoseUncertainty` answers it in metres.
MIN_IOU = 0.5


class EvaluationError(ValueError):
    """A corpus that cannot honestly be scored."""


@dataclass(frozen=True, slots=True)
class ClassScore:
    label: str
    labelled: int
    predicted: int
    true_positives: int
    average_precision: float

    @property
    def precision(self) -> float:
        return self.true_positives / self.predicted if self.predicted else 0.0

    @property
    def recall(self) -> float:
        return self.true_positives / self.labelled if self.labelled else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def describe(self) -> str:
        return (f"{self.label:<14} P {self.precision:.3f}  R {self.recall:.3f}  "
                f"F1 {self.f1:.3f}  AP {self.average_precision:.3f}  "
                f"({self.true_positives}/{self.labelled} found, {self.predicted} proposed)")


@dataclass(frozen=True, slots=True)
class Evaluation:
    """What the detector scored, and on what."""

    classes: tuple[ClassScore, ...]
    frames: int
    days: tuple[str, ...]
    #: Frames that had a label file but no image, or the reverse. Reported
    #: rather than skipped silently: a corpus quietly missing half its labels
    #: scores beautifully on recall.
    incomplete: int = 0
    per_camera: dict = field(default_factory=dict)

    @property
    def mean_average_precision(self) -> float:
        """mAP over classes that actually appear in the labels.

        Classes with nothing labelled are excluded rather than scored zero: a
        model is not wrong about a class the validation set never contained,
        and averaging in a zero for each of COCO's eighty would bury every
        real figure.
        """
        present = [c.average_precision for c in self.classes if c.labelled]
        return float(np.mean(present)) if present else 0.0

    def describe(self) -> str:
        lines = [f"{self.frames} frame(s) over {len(self.days)} day(s): "
                 f"{', '.join(self.days)}"]
        if self.incomplete:
            lines.append(f"WARNING: {self.incomplete} frame(s) had an image or a label but not "
                         f"both, and were skipped")
        for score in sorted(self.classes, key=lambda c: -c.labelled):
            if score.labelled or score.predicted:
                lines.append("  " + score.describe())
        lines.append(f"mAP@{MIN_IOU:.2f} {self.mean_average_precision:.3f} over "
                     f"{sum(1 for c in self.classes if c.labelled)} class(es) present")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class Box:
    class_id: int
    x: float
    y: float
    width: float
    height: float
    confidence: float = 1.0

    def iou(self, other: "Box") -> float:
        x1, y1 = max(self.x, other.x), max(self.y, other.y)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        union = self.width * self.height + other.width * other.height - inter
        return inter / union if union > 0 else 0.0


def read_yolo(path: Path, *, with_confidence: bool = False) -> list[Box]:
    """A YOLO label file as boxes. `class cx cy w h [conf]`, all normalised."""
    out: list[Box] = []
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:5])
            confidence = float(parts[5]) if with_confidence and len(parts) > 5 else 1.0
        except ValueError:
            continue
        out.append(Box(class_id, cx - w / 2, cy - h / 2, w, h, confidence))
    return out


def average_precision(matched: Sequence[bool], confidences: Sequence[float],
                      labelled: int) -> float:
    """Area under the precision-recall curve, all-points interpolation.

    Zero when nothing was labelled: a class the validation set never contained
    has no precision-recall curve, and inventing one would put a number where
    there is no measurement.
    """
    if not labelled or not matched:
        return 0.0
    order = np.argsort(-np.asarray(confidences, dtype=np.float64), kind="stable")
    hits = np.asarray(matched, dtype=np.float64)[order]
    true_positives = np.cumsum(hits)
    false_positives = np.cumsum(1.0 - hits)
    recall = true_positives / labelled
    precision = true_positives / np.maximum(true_positives + false_positives, 1e-12)
    # Make precision monotonically decreasing from the right, then integrate.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[precision[0] if len(precision) else 0.0], precision])
    return float(np.sum(np.diff(recall) * precision[1:]))


def check_split(train_days: Sequence[str], validation_days: Sequence[str]) -> None:
    """Refuse a split that leaks. See the module note."""
    shared = sorted(set(train_days) & set(validation_days))
    if shared:
        raise EvaluationError(
            f"this split shares {len(shared)} day(s) between training and validation: "
            f"{', '.join(shared)}. Consecutive frames of security footage are nearly identical, so "
            f"the model would be tested on the frame after the one it trained on and would score "
            f"beautifully for no reason. Split by whole days -- `vigil dataset export` does"
        )
    if not validation_days:
        raise EvaluationError(
            "there is no validation set: everything came from one day, and there is no honest "
            "split of one day's footage. Record on more days before asking what the detector is "
            "worth"
        )


def evaluate(corpus: Path, detector, names: dict[int, str] | None = None, *,
             min_iou: float = MIN_IOU, limit: int | None = None) -> Evaluation:
    """Score `detector` over a corpus's **validation** frames.

    The corpus is what `vigil dataset export` wrote, with its labels corrected
    by a person. Its `manifest.json` says which days are validation, and the
    split is checked before a single frame is read.
    """
    import cv2

    corpus = Path(corpus)
    manifest_path = corpus / "manifest.json"
    if not manifest_path.is_file():
        raise EvaluationError(
            f"no manifest at {manifest_path}. This scores a corpus written by "
            f"`vigil dataset export`, whose manifest says which days are held out"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = manifest.get("split") or {}
    train_days = tuple(split.get("train_days") or ())
    validation_days = tuple(split.get("validation_days") or ())
    check_split(train_days, validation_days)
    # The classes are the corpus's own, in its own order. A model's class ids
    # and a corpus's are two different numbering schemes, and scoring one
    # against the other silently reports every class as wrong.
    vocabulary = {int(k): v for k, v in (manifest.get("classes") or {}).items()}

    wanted = set(validation_days)
    images = sorted((corpus / "images").glob("*.jpg")) + sorted((corpus / "images").glob("*.png"))
    labels_dir = corpus / "labels"
    by_day = {Path(entry["image"]).name: entry.get("day")
              for entry in manifest.get("samples", [])}

    frames = 0
    incomplete = 0
    labelled_per_class: defaultdict[int, int] = defaultdict(int)
    predicted: defaultdict[int, list[tuple[float, bool]]] = defaultdict(list)

    for image_path in images:
        day = by_day.get(image_path.name)
        if wanted and day not in wanted:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            incomplete += 1
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            incomplete += 1
            continue
        frames += 1
        truth = read_yolo(label_path)
        for box in truth:
            labelled_per_class[box.class_id] += 1
        found = [Box(d.class_id, d.bbox.x, d.bbox.y, d.bbox.width, d.bbox.height, d.confidence)
                 for d in detector.detect(image)]
        _score_frame(truth, found, min_iou, predicted)
        if limit is not None and frames >= limit:
            break

    if not frames:
        raise EvaluationError(
            f"no validation frames were found in {corpus}. The manifest names "
            f"{len(validation_days)} validation day(s) and none of their images had a label file "
            f"beside them -- there is nothing here to score"
        )

    vocabulary = dict(vocabulary)
    vocabulary.update(names or {})
    classes = []
    for class_id in sorted(set(labelled_per_class) | set(predicted)):
        entries = predicted.get(class_id, [])
        matched = [hit for _c, hit in entries]
        confidences = [c for c, _h in entries]
        classes.append(ClassScore(
            label=vocabulary.get(class_id, str(class_id)),
            labelled=labelled_per_class.get(class_id, 0),
            predicted=len(entries),
            true_positives=sum(matched),
            average_precision=average_precision(matched, confidences,
                                                labelled_per_class.get(class_id, 0)),
        ))
    return Evaluation(tuple(classes), frames, validation_days, incomplete)


def _score_frame(truth: list[Box], found: list[Box], min_iou: float,
                 predicted: defaultdict) -> None:
    """Greedy matching by descending confidence, each label claimed once."""
    claimed = [False] * len(truth)
    for box in sorted(found, key=lambda b: -b.confidence):
        best, best_iou = -1, min_iou
        for index, label in enumerate(truth):
            if claimed[index] or label.class_id != box.class_id:
                continue
            overlap = label.iou(box)
            if overlap >= best_iou:
                best, best_iou = index, overlap
        if best >= 0:
            claimed[best] = True
        predicted[box.class_id].append((box.confidence, best >= 0))
