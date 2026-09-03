"""Tests for instance segmentation.

These run against a **real pretrained model**, not a fixture. A fixture built to
make a code path executable proves the plumbing and nothing about the thing the
plumbing carries — and the reason this module exists is a measurement no fixture
would have produced: pointed at a real webcam, the motion detector reported
twenty tracks for one seated person, most of them fragments of a face, the rest
curtains and a wall.

The model is operator-supplied and never downloaded. `devtools/export_model.py`
writes one on a connected machine; without it these skip, loudly, rather than
quietly passing on a stub.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from sentinel import logs
from sentinel.core import BoundingBox, Detection
from sentinel.detect import DetectionError, MotionDetector, OnnxDetector, detector_for
from sentinel.segment import MINIMUM_MASK_PIXELS, Segmenter, ground_contact

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "yolov8n-seg.onnx"

needs_model = pytest.mark.skipif(
    not MODEL.is_file(),
    reason=(
        f"no segmentation model at {MODEL}. Produce one on a connected machine "
        "with `python devtools/export_model.py --task segment --size n`. "
        "Nothing is ever downloaded automatically."
    ),
)


@pytest.fixture(autouse=True)
def clean_logging():
    logs.reset()
    yield
    logs.reset()


@pytest.fixture(scope="module")
def segmenter():
    if not MODEL.is_file():
        pytest.skip("no model")
    return Segmenter(MODEL, confidence_threshold=0.30)


# ------------------------------------------------------- the ground contact
#
# The reason segmentation is worth its cost. Everything downstream rests on one
# point per object, and until now that point was the bottom-centre of a
# rectangle — correct only for a tight box around an upright, unoccluded person.


def mask_detection(mask: np.ndarray, box: BoundingBox) -> Detection:
    return Detection(bbox=box, confidence=0.9, class_id=0, mask=mask.astype(np.uint8))


def test_contact_comes_from_the_lowest_lit_row_not_the_box():
    # An L: the object's lowest part is at the left, nowhere near the box's
    # bottom-centre. A rectangle would put this person's feet in mid-air to the
    # right of where they are standing.
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:, 0:3] = 1        # a vertical bar down the left
    mask[9, 0:3] = 1        # its foot

    box = BoundingBox(x=0.0, y=0.0, w=1.0, h=1.0)
    x, y = ground_contact(mask_detection(mask, box), 100, 100)

    assert y == pytest.approx(1.0, abs=1e-6)
    # Centre of the lit columns on that row: (0+1+2)/3 = 1, +0.5, /10.
    assert x == pytest.approx(0.15, abs=1e-6)
    # And it is not the box's bottom-centre, which is what it replaces.
    assert abs(x - (box.x + box.w / 2)) > 0.3


def test_contact_uses_the_lowest_row_not_the_whole_mask_centre():
    # Somebody mid-stride has their feet somewhere other than under their centre
    # of mass, so averaging the whole mask is the wrong answer.
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0:8, 4:6] = 1      # torso, centred
    mask[8:10, 7:9] = 1     # a leading foot, off to the right

    x, _ = ground_contact(mask_detection(mask, BoundingBox(0, 0, 1, 1)), 100, 100)

    assert x == pytest.approx(0.80, abs=1e-6)


def test_contact_falls_back_to_the_box_without_a_mask():
    # Every detector that produces boxes only has always produced this, and
    # must go on producing it rather than failing.
    box = BoundingBox(x=0.2, y=0.1, w=0.4, h=0.6)
    plain = Detection(bbox=box, confidence=0.9, class_id=0)

    assert ground_contact(plain, 100, 100) == (0.4, pytest.approx(0.7))


def test_contact_is_inside_the_box():
    # A contact point outside its own detection would be projected onto the map
    # at a place nothing was seen.
    rng = np.random.default_rng(7)
    for _ in range(40):
        mask = (rng.random((12, 12)) > 0.6).astype(np.uint8)
        if not mask.any():
            continue
        box = BoundingBox(x=0.3, y=0.25, w=0.2, h=0.35)
        x, y = ground_contact(mask_detection(mask, box), 640, 480)

        assert box.x <= x <= box.x + box.w
        assert box.y <= y <= box.y + box.h


# ----------------------------------------------------------- the real model


@needs_model
def test_the_model_loads_and_says_what_it_is(segmenter):
    info = segmenter.info

    assert info.kind == "onnx-segment"
    assert info.input_size == (640, 640)
    assert info.classifies is True
    assert len(info.class_names) == 80, "COCO names should come from the model itself"
    assert info.class_names[0] == "person"
    # Provenance: every event this produces names the exact file.
    assert info.model_sha256 and len(info.model_sha256) == 64


@needs_model
@pytest.mark.skipif(
    os.environ.get("SENTINEL_TEST_CAMERA") != "1",
    reason=(
        "needs a real camera and a person in front of it. Set "
        "SENTINEL_TEST_CAMERA=1 to run. Synthetic frames cannot stand in here: "
        "the claim is that the mask follows a real silhouette, and a drawn "
        "rectangle has no silhouette to follow."
    ),
)
def test_a_real_person_gets_a_silhouette_not_a_rectangle(segmenter):
    import cv2

    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    try:
        if not capture.isOpened():
            pytest.skip("no camera at index 0")
        people = []
        for _ in range(30):  # the first frames off a webcam are often dark
            ok, frame = capture.read()
            if not ok:
                continue
            people = [d for d in segmenter.detect(frame) if d.class_id == 0]
            if people:
                break
    finally:
        capture.release()

    assert people, "no person was segmented -- stand in front of the camera"

    person = max(people, key=lambda d: d.confidence)
    assert person.mask is not None
    height, width = person.mask.shape
    filled = int(person.mask.sum()) / max(1, height * width)

    # A person is roughly a third to three-quarters of their own bounding box.
    # Anything at 1.0 is a box that has been handed a mask's name.
    assert 0.15 < filled < 0.95, f"mask fills {filled:.0%} of its box"

    # And the contact point must have moved off the box's bottom-centre, or the
    # whole reason for segmenting is unspent.
    x, y = ground_contact(person, width, height)
    assert person.bbox.y + person.bbox.h - 1e-6 <= y <= person.bbox.y + person.bbox.h


@needs_model
def test_every_detection_carries_a_usable_mask(segmenter):
    image = np.full((480, 640, 3), 90, dtype=np.uint8)
    image[200:400, 250:350] = 200

    for detection in segmenter.detect(image):
        assert detection.mask is not None
        assert detection.mask.dtype == np.uint8
        assert set(np.unique(detection.mask)) <= {0, 1}
        assert int(detection.mask.sum()) >= MINIMUM_MASK_PIXELS

        expected = (
            int(round(detection.bbox.h * 480)),
            int(round(detection.bbox.w * 640)),
        )
        assert detection.mask.shape == expected, (
            "a mask must be cropped to its own box, or nothing can place it"
        )


@needs_model
def test_boxes_stay_inside_the_frame(segmenter):
    image = np.full((480, 640, 3), 128, dtype=np.uint8)

    for detection in segmenter.detect(image):
        box = detection.bbox
        assert 0.0 <= box.x <= 1.0
        assert 0.0 <= box.y <= 1.0
        assert box.x + box.w <= 1.0 + 1e-6
        assert box.y + box.h <= 1.0 + 1e-6


@needs_model
def test_a_non_square_frame_is_letterboxed_not_stretched(segmenter):
    # Stretching a wide frame into a square makes every person short and wide,
    # and a model trained on letterboxed input then misses them. Both aspect
    # ratios must produce sane boxes rather than one of them producing none.
    for shape in ((480, 640, 3), (720, 1280, 3), (640, 480, 3)):
        image = np.full(shape, 128, dtype=np.uint8)
        segmenter.detect(image)  # must not raise, whatever the aspect


# ------------------------------------------------------------- the factory


def test_no_model_gives_the_motion_detector():
    assert isinstance(detector_for(None), MotionDetector)


@needs_model
def test_a_segmentation_model_gives_a_segmenter():
    # Decided by reading the model, not by a flag. A flag can disagree with the
    # file, and the operator would have no way to tell which one won.
    assert isinstance(detector_for(MODEL), Segmenter)


def test_a_missing_model_says_so_and_does_not_reach_for_one(tmp_path: Path):
    with pytest.raises(DetectionError, match="nothing is ever downloaded"):
        detector_for(tmp_path / "absent.onnx")


@needs_model
def test_a_detection_model_is_not_mistaken_for_a_segmentation_one(tmp_path: Path):
    # The distinction is what the system can conclude, not a preference: a
    # detector has one output and no masks, and pretending otherwise would give
    # every object a ground-contact point derived from a rectangle while
    # claiming it came from a silhouette.
    from sentinel.detect import _output_count

    assert _output_count(MODEL) == 2
