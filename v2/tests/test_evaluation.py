"""Scoring the detector, and refusing to score it dishonestly.

No labelled dataset exists for this site, so what is tested here is the
harness: that the arithmetic is right on cases whose answers can be worked out
by hand, and — the part that matters — that a corpus which would produce a
flattering meaningless number is refused rather than scored.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from vigil.domain.detection import BoundingBox, Detection, DetectorInfo
from vigil.service.evaluation import (
    Box, EvaluationError, average_precision, check_split, evaluate, read_yolo,
)

NAMES = {0: "person", 1: "car"}
INFO = DetectorInfo("onnx-detect", "test", "/x", "sha", (640, 640), NAMES, True, "CPU")


class _Detector:
    """Returns whatever it was told to, per image, by filename."""

    def __init__(self, answers):
        self._answers = answers
        self.info = INFO

    def detect(self, image):
        # The frame carries its own index in the top-left pixel, so this can
        # answer differently per image without being told which it is.
        key = int(image[0, 0, 0])
        return [Detection(BoundingBox(*box), confidence, class_id, None)
                for class_id, confidence, box in self._answers.get(key, [])]


def _corpus(tmp_path: Path, *, frames, train_days, validation_days, labels) -> Path:
    """A corpus in the shape `vigil dataset export` writes."""
    import cv2

    root = tmp_path / "corpus"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    samples = []
    for index, day in frames:
        name = f"frame-{index:03d}.png"
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        image[0, 0, 0] = index
        cv2.imwrite(str(root / "images" / name), image)
        lines = [f"{c} {x + w / 2:.6f} {y + h / 2:.6f} {w:.6f} {h:.6f}"
                 for c, (x, y, w, h) in labels.get(index, [])]
        (root / "labels" / f"frame-{index:03d}.txt").write_text("\n".join(lines), encoding="utf-8")
        samples.append({"image": f"images/{name}", "day": day, "camera": "gate"})
    (root / "manifest.json").write_text(json.dumps({
        "split": {"train_days": list(train_days), "validation_days": list(validation_days)},
        "classes": {str(k): v for k, v in NAMES.items()},
        "samples": samples,
    }), encoding="utf-8")
    return root


# ------------------------------------------------------------- the refusals


def test_a_split_that_shares_a_day_is_refused():
    """Consecutive frames of security footage are near-duplicates, so the model
    would be tested on the frame after the one it trained on."""
    with pytest.raises(EvaluationError, match="shares 1 day"):
        check_split(["2026-09-01", "2026-09-02"], ["2026-09-02"])
    check_split(["2026-09-01"], ["2026-09-02"])


def test_a_corpus_from_one_day_has_no_honest_validation_set():
    with pytest.raises(EvaluationError, match="no honest split"):
        check_split(["2026-09-01"], [])


def test_a_leaking_corpus_is_refused_before_a_single_frame_is_read(tmp_path):
    """It stops rather than warning: a warning above a 0.98 is read as a 0.98."""
    corpus = _corpus(tmp_path, frames=[(1, "2026-09-01"), (2, "2026-09-01")],
                     train_days=["2026-09-01"], validation_days=["2026-09-01"],
                     labels={1: [(0, (0.1, 0.1, 0.2, 0.4))]})
    with pytest.raises(EvaluationError, match="shares"):
        evaluate(corpus, _Detector({}), NAMES)


def test_a_corpus_with_no_manifest_says_what_it_wanted(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(EvaluationError, match="dataset export"):
        evaluate(tmp_path / "empty", _Detector({}), NAMES)


def test_a_corpus_whose_validation_day_has_no_frames_is_refused(tmp_path):
    corpus = _corpus(tmp_path, frames=[(1, "2026-09-01")],
                     train_days=["2026-09-01"], validation_days=["2026-09-02"],
                     labels={1: []})
    with pytest.raises(EvaluationError, match="nothing here to score"):
        evaluate(corpus, _Detector({}), NAMES)


# ------------------------------------------------------------ the arithmetic


def test_a_perfect_detector_scores_one_and_a_blind_one_scores_zero(tmp_path):
    box = (0.1, 0.1, 0.2, 0.4)
    corpus = _corpus(tmp_path, frames=[(1, "2026-09-01"), (2, "2026-09-02")],
                     train_days=["2026-09-01"], validation_days=["2026-09-02"],
                     labels={1: [(0, box)], 2: [(0, box)]})

    perfect = evaluate(corpus, _Detector({2: [(0, 0.9, box)]}), NAMES)
    assert perfect.frames == 1, "only the validation day should be scored"
    person = next(c for c in perfect.classes if c.label == "person")
    assert person.precision == 1.0 and person.recall == 1.0
    assert perfect.mean_average_precision == pytest.approx(1.0)

    blind = evaluate(corpus, _Detector({}), NAMES)
    person = next(c for c in blind.classes if c.label == "person")
    assert person.recall == 0.0 and person.average_precision == 0.0


def test_a_box_in_the_wrong_place_is_a_false_positive_and_a_miss(tmp_path):
    corpus = _corpus(tmp_path, frames=[(1, "2026-09-01"), (2, "2026-09-02")],
                     train_days=["2026-09-01"], validation_days=["2026-09-02"],
                     labels={2: [(0, (0.1, 0.1, 0.2, 0.4))]})
    result = evaluate(corpus, _Detector({2: [(0, 0.9, (0.6, 0.6, 0.2, 0.4))]}), NAMES)
    person = next(c for c in result.classes if c.label == "person")
    assert person.precision == 0.0 and person.recall == 0.0
    assert person.predicted == 1 and person.labelled == 1


def test_the_right_box_with_the_wrong_class_does_not_count(tmp_path):
    """A car found where a person is, is not a person found."""
    box = (0.1, 0.1, 0.2, 0.4)
    corpus = _corpus(tmp_path, frames=[(1, "2026-09-01"), (2, "2026-09-02")],
                     train_days=["2026-09-01"], validation_days=["2026-09-02"],
                     labels={2: [(0, box)]})
    result = evaluate(corpus, _Detector({2: [(1, 0.9, box)]}), NAMES)
    person = next(c for c in result.classes if c.label == "person")
    car = next(c for c in result.classes if c.label == "car")
    assert person.recall == 0.0
    assert car.predicted == 1 and car.true_positives == 0


def test_two_predictions_on_one_label_count_one_hit_and_one_false_positive(tmp_path):
    """Each label is claimed once. A detector that proposes the same object
    twice has made one find and one mistake."""
    box = (0.1, 0.1, 0.2, 0.4)
    nearly = (0.105, 0.105, 0.2, 0.4)
    corpus = _corpus(tmp_path, frames=[(1, "2026-09-01"), (2, "2026-09-02")],
                     train_days=["2026-09-01"], validation_days=["2026-09-02"],
                     labels={2: [(0, box)]})
    result = evaluate(corpus, _Detector({2: [(0, 0.9, box), (0, 0.8, nearly)]}), NAMES)
    person = next(c for c in result.classes if c.label == "person")
    assert person.true_positives == 1 and person.predicted == 2
    assert person.precision == 0.5 and person.recall == 1.0


def test_a_class_the_validation_set_never_contained_does_not_drag_the_map_down(tmp_path):
    """A model is not wrong about a class nobody labelled, and averaging in a
    zero for each of COCO's eighty would bury every real figure."""
    box = (0.1, 0.1, 0.2, 0.4)
    corpus = _corpus(tmp_path, frames=[(1, "2026-09-01"), (2, "2026-09-02")],
                     train_days=["2026-09-01"], validation_days=["2026-09-02"],
                     labels={2: [(0, box)]})
    result = evaluate(corpus, _Detector({2: [(0, 0.9, box)]}), NAMES)
    assert result.mean_average_precision == pytest.approx(1.0)
    assert "1 class(es) present" in result.describe()


def test_average_precision_rewards_ranking_the_true_positives_first():
    """Two detectors that find the same objects are not equally good if one
    ranks its mistakes above its finds."""
    good = average_precision([True, True, False, False], [0.9, 0.8, 0.4, 0.3], labelled=2)
    bad = average_precision([False, False, True, True], [0.9, 0.8, 0.4, 0.3], labelled=2)
    assert good == pytest.approx(1.0)
    assert bad < good
    assert average_precision([], [], labelled=0) == 0.0
    assert average_precision([True], [0.9], labelled=0) == 0.0


def test_a_label_file_is_read_as_centres_and_returned_as_corners(tmp_path):
    path = tmp_path / "one.txt"
    path.write_text("0 0.5 0.5 0.2 0.4\n1 0.25 0.25 0.1 0.1\nrubbish\n", encoding="utf-8")
    boxes = read_yolo(path)
    assert len(boxes) == 2, "the unparseable line should be skipped, not fatal"
    assert boxes[0].x == pytest.approx(0.4) and boxes[0].y == pytest.approx(0.3)
    assert read_yolo(tmp_path / "missing.txt") == []


def test_overlap_is_symmetric_and_zero_when_they_miss():
    a = Box(0, 0.0, 0.0, 0.2, 0.2)
    b = Box(0, 0.1, 0.1, 0.2, 0.2)
    assert a.iou(b) == pytest.approx(b.iou(a))
    assert 0.0 < a.iou(b) < 1.0
    assert a.iou(Box(0, 0.8, 0.8, 0.1, 0.1)) == 0.0
    assert a.iou(a) == pytest.approx(1.0)


def test_frames_missing_a_label_file_are_counted_rather_than_ignored(tmp_path):
    """A corpus quietly missing half its labels scores beautifully on recall."""
    corpus = _corpus(tmp_path, frames=[(1, "2026-09-01"), (2, "2026-09-02"), (3, "2026-09-02")],
                     train_days=["2026-09-01"], validation_days=["2026-09-02"],
                     labels={2: [(0, (0.1, 0.1, 0.2, 0.4))], 3: []})
    (corpus / "labels" / "frame-003.txt").unlink()
    result = evaluate(corpus, _Detector({2: [(0, 0.9, (0.1, 0.1, 0.2, 0.4))]}), NAMES)
    assert result.frames == 1 and result.incomplete == 1
    assert "had an image or a label but not both" in result.describe()
