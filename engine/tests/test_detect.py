"""Tests for detection.

Half of these are about what the detector finds. The other half are about what it
must never claim, which for a system that produces evidence is the more important
half: a motion blob is not a person, and a detector that says otherwise has
fabricated the basis for an alert.

The numbers asserted here were measured on the reference scene, not chosen. They
are deliberately stated as floors well below what was measured, so they catch a
regression without failing on the ordinary variation between machines.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import scene
from sentinel.decode import VideoSource
from sentinel.detect import (
    UNCLASSIFIED,
    DetectionError,
    DetectorTiming,
    MotionDetector,
    OnnxDetector,
    TimedDetector,
    _infer_layout,
    _non_max_suppression,
)


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    intersection = ix * iy
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


@pytest.fixture(scope="module")
def detections_over_scene(reference_video: Path):
    """Every detection from the reference scene, with the truth beside it."""
    detector = TimedDetector(MotionDetector())
    per_frame = []
    with VideoSource(reference_video) as source:
        for frame in source:
            boxes = [
                (d.bbox.x * frame.width, d.bbox.y * frame.height,
                 d.bbox.w * frame.width, d.bbox.h * frame.height)
                for d in detector.detect(frame.image)
            ]
            per_frame.append((frame.index, boxes, scene.ground_truth(frame.index)))
    return per_frame, detector.timing


# --------------------------------------------------------------- what it finds


def test_the_motion_detector_finds_the_objects_that_are_there(detections_over_scene):
    per_frame, _ = detections_over_scene

    hits = total = 0
    for index, boxes, truth in per_frame:
        if index < 12:
            continue
        for actual in truth.values():
            best = max((iou(box, actual) for box in boxes), default=0.0)
            total += 1
            hits += best > 0.3

    recall = hits / total
    # Measured at 0.71. The floor is well below that: this guards against a
    # regression, not against machine-to-machine variation.
    assert recall > 0.60, f"recall fell to {recall:.2f}"


def test_the_boxes_it_produces_are_roughly_the_right_boxes(detections_over_scene):
    per_frame, _ = detections_over_scene

    overlaps = []
    for index, boxes, truth in per_frame:
        if index < 12:
            continue
        for actual in truth.values():
            overlaps.append(max((iou(box, actual) for box in boxes), default=0.0))

    # Measured at 0.51. A detector whose recall holds while its boxes degrade
    # still ruins ground projection, because the position comes from the bottom
    # edge of the box.
    assert float(np.mean(overlaps)) > 0.40


def test_it_does_not_invent_objects_where_there_are_none(detections_over_scene):
    per_frame, _ = detections_over_scene

    stray = 0
    frames = 0
    for index, boxes, truth in per_frame:
        if index < 12:
            continue
        frames += 1
        for box in boxes:
            if all(iou(box, actual) <= 0.01 for actual in truth.values()):
                stray += 1

    # Measured at zero: every extra box on the reference scene was a fragment of
    # a real object rather than noise. Allowing a small budget rather than
    # asserting zero, because sensor noise is stochastic.
    assert stray / frames < 0.05, f"{stray} boxes appeared where nothing was"


def test_it_runs_faster_than_the_video_it_is_watching(detections_over_scene):
    _, timing = detections_over_scene

    # Measured at ~430 fps on 640x480 at the default 0.75 detection scale. The
    # assertion is only that it keeps up with the 15 fps source with room to
    # spare, because a detector that cannot is not a detector, it is a backlog.
    assert timing.fps > scene.FPS * 2, f"only {timing.fps:.0f} fps"


# ------------------------------------------------------------- what it refuses


def test_every_motion_detection_is_explicitly_unclassified(detections_over_scene):
    detector = MotionDetector()
    image = np.zeros((240, 320, 3), dtype=np.uint8)

    seen_any = False
    for step in range(40):
        image[:] = 40
        image[100 : 160, 40 + step * 4 : 90 + step * 4] = 200
        for detection in detector.detect(image):
            seen_any = True
            assert detection.class_id == UNCLASSIFIED

    assert seen_any, "the fixture produced no detections, so nothing was checked"


def test_the_unclassified_id_cannot_be_mistaken_for_a_real_class():
    # 0 is "person" in essentially every detection model. A motion blob that
    # inherited it would silently become a person everywhere downstream.
    assert UNCLASSIFIED != 0
    assert UNCLASSIFIED > 1000


def test_the_motion_detector_says_it_does_not_classify():
    info = MotionDetector().info

    assert info.classifies is False
    assert info.model_path is None
    assert info.label_for(UNCLASSIFIED) == "unclassified"


def test_it_produces_nothing_while_the_background_model_is_cold():
    # Every frame is foreground to an empty model. Emitting detections here would
    # open an incident every time a camera reconnects.
    detector = MotionDetector(warmup_frames=10)
    noise = np.random.default_rng(1).integers(0, 255, (240, 320, 3), dtype=np.uint8)

    for _ in range(9):
        assert detector.detect(noise) == []
    assert detector.is_warm is False


def test_a_whole_frame_changing_is_not_an_object():
    # An auto-exposure step or a light being switched on changes everything at
    # once. Reporting that as a detection is how a system cries wolf at dusk.
    detector = MotionDetector(warmup_frames=5)
    dark = np.full((240, 320, 3), 30, dtype=np.uint8)
    for _ in range(30):
        detector.detect(dark)

    bright = np.full((240, 320, 3), 220, dtype=np.uint8)
    detections = detector.detect(bright)

    assert all(d.bbox.w * d.bbox.h < 0.35 for d in detections)


def test_the_morphology_scales_with_resolution():
    # A kernel measured on 480p must not silently become a third as effective on
    # 1080p, where a person occupies the same fraction of the frame.
    detector = MotionDetector()
    detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
    small = detector._close_kernel.shape

    detector.detect(np.zeros((1080, 1920, 3), dtype=np.uint8))
    large = detector._close_kernel.shape

    assert large[0] > small[0] * 2


# ---------------------------------------------------------------- ONNX loading


def test_a_missing_model_is_refused_and_says_nothing_is_downloaded(tmp_path: Path):
    with pytest.raises(DetectionError) as caught:
        OnnxDetector(tmp_path / "absent.onnx")

    message = str(caught.value)
    assert "No model at" in message
    assert "downloaded" in message


def test_a_file_that_is_not_a_model_is_refused(tmp_path: Path):
    fake = tmp_path / "not-a-model.onnx"
    fake.write_bytes(b"this is not a protobuf")

    with pytest.raises(DetectionError, match="Could not load"):
        OnnxDetector(fake)


def test_a_dynamic_output_shape_is_refused_rather_than_guessed():
    # Guessing the layout does not fail loudly; it produces detections at
    # plausible but wrong coordinates, which is worse than not running.
    with pytest.raises(DetectionError, match="dynamic output shape"):
        _infer_layout([1, "anchors", "channels"], "model.onnx")


def test_an_output_rank_that_is_not_understood_is_refused():
    with pytest.raises(DetectionError, match="rank-2"):
        _infer_layout([1, 100], "model.onnx")


def test_too_few_channels_for_a_detection_head_is_refused():
    with pytest.raises(DetectionError, match="at least 4 box values"):
        _infer_layout([1, 3, 8400], "model.onnx")


def test_both_common_output_layouts_are_recognised():
    v8 = _infer_layout([1, 84, 8400], "yolov8.onnx")
    assert v8.channels_first is True
    assert v8.class_count == 80

    v5 = _infer_layout([1, 25200, 85], "yolov5.onnx")
    assert v5.channels_first is False


# ------------------------------------------------------------------------ NMS


def test_overlapping_boxes_of_one_class_collapse_to_the_best():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [2, 2, 12, 12]], dtype=float)
    scores = np.array([0.9, 0.8, 0.7])
    classes = np.array([0, 0, 0])

    assert _non_max_suppression(boxes, scores, classes, 0.45) == [0]


def test_suppression_never_crosses_class_boundaries():
    # A person standing in front of a car must not delete the car.
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=float)
    scores = np.array([0.9, 0.85])
    classes = np.array([0, 2])

    assert _non_max_suppression(boxes, scores, classes, 0.45) == [0, 1]


def test_distant_boxes_are_all_kept():
    boxes = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], dtype=float)
    scores = np.array([0.9, 0.8])
    classes = np.array([0, 0])

    assert _non_max_suppression(boxes, scores, classes, 0.45) == [0, 1]


# --------------------------------------------------------------------- timing


def test_timing_is_measured_rather_than_estimated():
    timing = DetectorTiming()
    timing.record(0.010)
    timing.record(0.030)

    assert timing.frames == 2
    assert timing.fps == pytest.approx(50.0)
    assert timing.mean_millis == pytest.approx(20.0)
    assert timing.slowest_seconds == pytest.approx(0.030)


def test_an_unmeasured_detector_reports_no_rate_rather_than_a_default():
    assert DetectorTiming().fps == 0.0


def test_timing_is_recorded_even_when_detection_fails():
    class Broken:
        info = MotionDetector().info

        def detect(self, image):
            raise RuntimeError("boom")

    timed = TimedDetector(Broken())
    with pytest.raises(RuntimeError):
        timed.detect(np.zeros((10, 10, 3), dtype=np.uint8))

    assert timed.timing.frames == 1, "a detector that fails slowly must still show as slow"


# ------------------------------------------------------------ detection scale


def test_the_detection_scale_does_not_change_what_a_box_means():
    """Shrinking the frame the model sees must not move anything downstream.

    Every threshold in the detector is a fraction of the frame and every box it
    emits is normalised, so a detection at 0.5 scale has to mean the same thing
    as one at full scale. If it did not, `detect_scale` would be a capacity
    knob that silently moved every object on the map.
    """
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    boxes_by_scale = {}
    for scale in (1.0, 0.75, 0.5):
        detector = MotionDetector(detect_scale=scale, warmup_frames=4)
        found = []
        for step in range(24):
            image[:] = 40
            top = 150
            left = 200 + step * 6
            image[top : top + 120, left : left + 40] = 210
            found = detector.detect(image)
        boxes_by_scale[scale] = found

    full = boxes_by_scale[1.0]
    assert full, "the fixture produced no detections at full scale"

    for scale, boxes in boxes_by_scale.items():
        assert boxes, f"nothing detected at {scale} scale"
        # Same object, same normalised place, within the coarser grid's own
        # resolution.
        assert abs(boxes[0].bbox.x - full[0].bbox.x) < 0.08, (
            f"the box moved at {scale} scale"
        )


def test_an_impossible_detection_scale_is_refused():
    with pytest.raises(DetectionError, match="detect_scale"):
        MotionDetector(detect_scale=0.01)


def test_the_scale_is_part_of_the_detector_identity():
    # Two runs at different scales are not the same detector, and an event's
    # provenance has to be able to say which one produced it.
    assert MotionDetector(detect_scale=1.0).info.name != MotionDetector(detect_scale=0.5).info.name
    assert "0.5" in MotionDetector(detect_scale=0.5).info.name


def test_a_motion_detector_ignores_a_watch_list_rather_than_refusing_it():
    # It cannot name a class, so it cannot watch one; the console applies its
    # watch list to whatever detector it has, and motion-only is not an error.
    from sentinel.detect import WATCHED_LABELS, MotionDetector, detector_for

    assert isinstance(detector_for(None, classes=WATCHED_LABELS), MotionDetector)
    assert "person" in WATCHED_LABELS and "car" in WATCHED_LABELS
    assert "bottle" not in WATCHED_LABELS


def test_a_motion_detector_ignores_a_confidence_floor_rather_than_refusing_it():
    # Its confidence is the fraction of a box that moved, not a probability, so
    # a floor meant for a classifier is dropped the way the watch list is. The
    # console sets one for every detector it builds, and a motion-only site
    # must still start.
    from sentinel.detect import MotionDetector, detector_for

    detector = detector_for(None, confidence_threshold=0.5, classes={"person"})
    assert isinstance(detector, MotionDetector)
    assert detector.info.classifies is False
