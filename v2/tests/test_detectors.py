from pathlib import Path

import numpy as np
import pytest

from vigil.adapters.decode import VideoSource
from vigil.adapters.detectors import DetectionError, MotionDetector, OnnxDetector, detector_for, forget_models, model_info, restrict_vocabulary
from vigil.domain.detection import UNCLASSIFIED


def test_motion_finds_the_walking_block_and_says_it_cannot_classify(reference_video):
    detector = MotionDetector()
    assert not detector.info.classifies and detector.info.label_for(0) is None
    hits = 0
    with VideoSource(reference_video) as source:
        for frame in source:
            detections = detector.detect(frame.image)
            if frame.index > 10 and frame.index < 50:
                hits += bool(detections)
                assert all(d.class_id == UNCLASSIFIED for d in detections)
    assert hits >= 20, f"motion saw the block in only {hits} frames"


def test_a_motion_detector_drops_a_watch_list_rather_than_refusing_it():
    assert isinstance(detector_for(None, classes={"person"}, confidence=0.9), MotionDetector)


def test_restrict_vocabulary_refuses_unknown_names_and_keeps_known_ones():
    names = {0: "person", 1: "car"}
    kept, ids = restrict_vocabulary(names, ["Person"])
    assert kept == {0: "person"} and ids == {0}
    with pytest.raises(DetectionError, match="does not name"):
        restrict_vocabulary(names, ["unicorn"])
    with pytest.raises(DetectionError, match="no class"):
        restrict_vocabulary(names, [])


def _tiny_model(path: Path) -> Path:
    """A one-layer ONNX 'detector': constant output with one confident box of class 0 at the frame centre."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    boxes = np.zeros((1, 6, 8), dtype=np.float32)  # (batch, 4 + 2 classes, 8 candidates)
    boxes[0, :, 0] = [320, 320, 100, 200, 0.9, 0.05]  # cx, cy, w, h, class scores
    boxes[0, :, 1] = [100, 100, 20, 20, 0.2, 0.1]  # below the floor
    constant = helper.make_node("Constant", [], ["prediction"], value=numpy_helper.from_array(boxes, "prediction_value"))
    graph = helper.make_graph([constant], "tiny", [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
                              [helper.make_tensor_value_info("prediction", TensorProto.FLOAT, [1, 6, 8])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    model.metadata_props.append(onnx.StringStringEntryProto(key="names", value="{0: 'person', 1: 'car'}"))
    onnx.save(model, str(path))
    return path


def test_the_onnx_path_reads_names_scales_boxes_and_applies_the_watch_list(tmp_path):
    model = _tiny_model(tmp_path / "tiny.onnx")
    forget_models()
    detector = OnnxDetector(model, confidence=0.5)
    assert detector.info.classifies and detector.info.class_names == {0: "person", 1: "car"}
    assert detector.info.model_sha256 and detector.info.input_size == (640, 640)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(image)
    assert len(detections) == 1
    box = detections[0].bbox
    assert abs(box.center.x - 0.5) < 0.02 and 0.3 < box.center.y < 0.7 and detections[0].class_id == 0
    assert OnnxDetector(model, confidence=0.5, classes=["car"]).detect(image) == []
    with pytest.raises(DetectionError):
        OnnxDetector(model, classes=["unicorn"])


def test_model_info_reads_the_file_once_per_process(tmp_path, monkeypatch):
    model = _tiny_model(tmp_path / "tiny.onnx")
    forget_models()
    from vigil.adapters import detectors

    loads = []
    real = detectors.OnnxDetector.__init__

    def counting(self, *a, **k):
        loads.append(1)
        real(self, *a, **k)

    monkeypatch.setattr(detectors.OnnxDetector, "__init__", counting)
    for _ in range(3):
        assert sorted(model_info(model).class_names.values()) == ["car", "person"]
    assert model_info(model, classes=["person"]).class_names == {0: "person"}
    assert len(loads) == 1
    forget_models()
    with pytest.raises(DetectionError, match="no model"):
        model_info(tmp_path / "missing.onnx")
