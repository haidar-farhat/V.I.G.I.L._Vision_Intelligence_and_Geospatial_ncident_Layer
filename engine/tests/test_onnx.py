"""Tests for the ONNX detection path, running a real model.

These exist because `OnnxDetector` was previously code that had never executed.
Every line of it — letterboxing, normalisation, session execution, output-layout
inference, coordinate un-letterboxing, per-class NMS — now runs against an actual
ONNX graph built by `onnx_fixture`.

The fixture model is a brightness detector: crude, but its output is a genuine
function of its input, which is what makes these tests capable of failing. A stub
returning constant boxes would exercise the same code paths and catch none of the
mistakes that actually happen here — a transposed output, a forgotten letterbox
offset, a normalisation against the wrong dimension.

What is **not** tested: detection quality. No real weights have been run. See
STATUS.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import onnx_fixture
from onnx_fixture import CELL, INPUT_SIZE
from sentinel.detect import DetectionError, OnnxDetector


@pytest.fixture(scope="module")
def model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return onnx_fixture.build_model(tmp_path_factory.mktemp("models") / "brightness.onnx")


@pytest.fixture(scope="module")
def detector(model_path: Path) -> OnnxDetector:
    return OnnxDetector(model_path, confidence_threshold=0.5, iou_threshold=0.45)


def square(width: int, height: int, left: int, top: int, size: int) -> np.ndarray:
    """A black image with one white square."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[top : top + size, left : left + size] = 255
    return image


def centre_of(detection, width: int, height: int) -> tuple[float, float]:
    box = detection.bbox
    return (box.x + box.w / 2) * width, (box.y + box.h / 2) * height


# --------------------------------------------------------------------- loading


def test_the_model_loads_and_reports_what_it_is(detector: OnnxDetector):
    info = detector.info

    assert info.kind == "onnx"
    assert info.classifies is True
    assert info.input_size == (INPUT_SIZE, INPUT_SIZE)
    assert info.model_path is not None


def test_the_model_digest_is_recorded(detector: OnnxDetector):
    # An event must be able to name the exact weights that produced it, months
    # later, when three model versions have been through the site.
    assert detector.info.model_sha256 is not None
    assert len(detector.info.model_sha256) == 64


def test_class_names_come_from_the_model_rather_than_a_standard_list(
    detector: OnnxDetector,
):
    # Attaching "person" to an output that might mean something else is
    # fabricated evidence. This model says what its one class is, and that is
    # what must be reported.
    assert detector.info.class_names == {0: "bright_region"}
    assert detector.info.label_for(0) == "bright_region"


def test_an_unknown_class_reports_its_id_rather_than_a_guess(detector: OnnxDetector):
    assert detector.info.label_for(7) == "class_7"


def test_the_output_layout_is_inferred_correctly(detector: OnnxDetector):
    layout = detector._layout

    assert layout.channels_first is True, "[1, 5, 400] is channels-first"
    assert layout.class_count == 1


# ------------------------------------------------------------------- inference


def test_it_finds_an_object_and_puts_the_box_on_it(detector: OnnxDetector):
    image = square(INPUT_SIZE, INPUT_SIZE, left=320, top=200, size=CELL * 2)
    detections = detector.detect(image)

    assert detections, "the model found nothing in an image with an obvious object"
    for detection in detections:
        x, y = centre_of(detection, INPUT_SIZE, INPUT_SIZE)
        assert 320 - CELL <= x <= 320 + CELL * 3
        assert 200 - CELL <= y <= 200 + CELL * 3


def test_moving_the_object_moves_the_box(detector: OnnxDetector):
    # The test that catches a transposed or constant output. A stub fixture would
    # pass every other test in this file and fail this one.
    first = detector.detect(square(INPUT_SIZE, INPUT_SIZE, 320, 200, CELL * 2))
    second = detector.detect(square(INPUT_SIZE, INPUT_SIZE, 96, 416, CELL * 2))

    assert first and second
    x1, y1 = centre_of(first[0], INPUT_SIZE, INPUT_SIZE)
    x2, y2 = centre_of(second[0], INPUT_SIZE, INPUT_SIZE)

    assert abs(x2 - x1) > 150
    assert abs(y2 - y1) > 150


def test_an_empty_image_produces_nothing(detector: OnnxDetector):
    assert detector.detect(np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)) == []


def test_confidence_reflects_the_evidence(detector: OnnxDetector):
    # This model's confidence is the cell's mean brightness, so a brighter object
    # must score higher. The point is that confidence is computed rather than
    # constant — a detector reporting a fixed 0.9 tells an operator nothing.
    bright = detector.detect(square(INPUT_SIZE, INPUT_SIZE, 320, 320, CELL * 2))

    dim = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    dim[320 : 320 + CELL * 2, 320 : 320 + CELL * 2] = 160
    dimmer = detector.detect(dim)

    assert bright and dimmer
    assert max(d.confidence for d in bright) > max(d.confidence for d in dimmer)


def test_every_box_is_normalised_into_the_frame(detector: OnnxDetector):
    detections = detector.detect(square(INPUT_SIZE, INPUT_SIZE, 0, 0, CELL * 3))

    assert detections
    for detection in detections:
        box = detection.bbox
        assert 0.0 <= box.x <= 1.0
        assert 0.0 <= box.y <= 1.0
        assert 0.0 < box.w <= 1.0
        assert 0.0 < box.h <= 1.0
        assert box.x + box.w <= 1.0 + 1e-9
        assert box.y + box.h <= 1.0 + 1e-9


# ----------------------------------------------------------------- letterboxing


def test_a_non_square_frame_comes_back_in_its_own_coordinates(model_path: Path):
    """The case that catches an un-letterboxing mistake.

    A 640x480 frame is padded to 640x640 with 80 rows above and below. A model
    reporting a box at tensor row 296 means row 216 of the original frame, and
    216/480 normalised — not 296/640, and not 296/480. Getting this wrong puts
    every object in the wrong place by a fixed offset, which looks like a
    calibration problem rather than a bug.
    """
    detector = OnnxDetector(model_path, confidence_threshold=0.5)

    width, height = 640, 480
    left, top, size = 320, 200, CELL * 2
    detections = detector.detect(square(width, height, left, top, size))

    assert detections
    x, y = centre_of(detections[0], width, height)

    assert abs(x - (left + size / 2)) < CELL * 1.5
    assert abs(y - (top + size / 2)) < CELL * 1.5, (
        f"vertical position is off by {y - (top + size / 2):.0f}px; the letterbox "
        "padding was probably not subtracted"
    )


def test_letterboxing_and_stretching_agree_horizontally(model_path: Path):
    # Only the vertical axis is padded for a landscape frame, so a horizontal
    # position must survive either preprocessing. If it does not, the scale
    # factor is being applied to the wrong axis.
    boxed = OnnxDetector(model_path, confidence_threshold=0.5, letterbox=True)
    stretched = OnnxDetector(model_path, confidence_threshold=0.5, letterbox=False)

    image = square(640, 480, left=320, top=200, size=CELL * 2)
    a = boxed.detect(image)
    b = stretched.detect(image)

    assert a and b
    assert abs(centre_of(a[0], 640, 480)[0] - centre_of(b[0], 640, 480)[0]) < CELL * 2


# ------------------------------------------------------------------------ NMS


def test_overlapping_candidates_collapse(model_path: Path):
    # With a low IoU threshold the adjacent cells covering one object should
    # reduce; with a high one they should not. This is NMS running on real
    # model output rather than on a hand-built array.
    strict = OnnxDetector(model_path, confidence_threshold=0.4, iou_threshold=0.01)
    loose = OnnxDetector(model_path, confidence_threshold=0.4, iou_threshold=0.99)

    image = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    image[200:300, 300:400] = 255

    assert len(strict.detect(image)) <= len(loose.detect(image))


def test_the_detector_is_deterministic(detector: OnnxDetector):
    image = square(INPUT_SIZE, INPUT_SIZE, 288, 288, CELL * 2)

    first = [(d.bbox, round(d.confidence, 9)) for d in detector.detect(image)]
    second = [(d.bbox, round(d.confidence, 9)) for d in detector.detect(image)]

    assert first == second


# ------------------------------------------------------------- refusing to run


def test_a_model_that_is_not_a_detector_is_refused(tmp_path: Path):
    """A model whose output is not a detection head must be refused.

    Guessing a layout does not fail loudly — it produces detections at plausible
    but wrong coordinates, which reads as a calibration problem and can survive
    for a long time. Refusing at load is the only honest option.

    The unit-level cases (dynamic dimensions, too few channels, each output
    layout) are in ``test_detect.py`` against ``_infer_layout`` directly. This one
    exists because it goes through a real session: onnxruntime performs its own
    shape inference and can report something quite different from what the model
    file declares, which is a way a guard can pass its unit test and still be
    bypassed in practice.
    """
    import onnx
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "not-a-detector",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 640, 640])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, "n", "c"])],
    )
    path = tmp_path / "not-a-detector.onnx"
    onnx.save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), str(path)
    )

    with pytest.raises(DetectionError) as caught:
        OnnxDetector(path)

    message = str(caught.value)
    assert "this reader understands" in message, "the refusal must say what it wanted"
    assert path.name in message, "and which model it refused"


def test_the_detector_reports_it_classifies_unlike_the_motion_detector(
    detector: OnnxDetector,
):
    from sentinel.detect import MotionDetector

    assert detector.info.classifies is True
    assert MotionDetector().info.classifies is False
