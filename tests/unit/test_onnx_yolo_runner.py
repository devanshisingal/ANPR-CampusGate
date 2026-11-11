"""Unit tests for pure ONNX Runtime YOLO runner and ONNX detector integration."""

from __future__ import annotations

import sys
from unittest.mock import patch

import numpy as np

from src.detection._onnx_yolo import OnnxYoloRunner, _letterbox, _parse_names
from src.detection.plate_detector import PlateDetector
from src.detection.vehicle_detector import VehicleDetector


def test_parse_names_handles_dict_list_and_invalid():
    assert _parse_names({"names": "{0: 'car', 1: 'truck'}"}) == {0: "car", 1: "truck"}
    assert _parse_names({"names": "['license_plate']"}) == {0: "license_plate"}
    assert _parse_names({}) == {}
    assert _parse_names({"names": "not valid python ***"}) == {}


def test_letterbox_fixed_and_auto_rect_preserve_bounds():
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    fixed, r_fixed, pad_fixed = _letterbox(img, (480, 480), auto=False, stride=32)
    assert fixed.shape == (480, 480, 3)
    assert abs(r_fixed - 0.75) < 1e-6
    assert pad_fixed == (0.0, 105.0)

    rect, r_rect, _ = _letterbox(img, (320, 320), auto=True, stride=32)
    assert rect.shape[0] % 32 == 0 and rect.shape[1] % 32 == 0
    assert rect.shape == (192, 320, 3)
    assert abs(r_rect - 0.5) < 1e-6


def test_runner_decodes_multiclass_and_applies_class_filter_and_nms():
    runner = OnnxYoloRunner.__new__(OnnxYoloRunner)
    runner.imgsz = 480
    runner.device = "cpu"
    runner.names = {0: "person", 2: "car", 5: "bus"}
    runner.stride = 32
    runner._end2end = False
    runner._target_hw = (480, 480)
    runner._auto_rect = False
    runner._is_fp16 = False
    runner._input_name = "images"
    runner._output_names = ["output0"]

    # Two overlapping 'car' boxes, one 'person' box above threshold, one overlapping 'bus'
    raw = np.zeros((1, 4 + 6, 4), dtype=np.float32)
    # anchor 0: car high conf centered at (240, 240)
    raw[0, :4, 0] = [240.0, 240.0, 100.0, 80.0]
    raw[0, 4 + 2, 0] = 0.90
    # anchor 1: duplicate car slightly shifted
    raw[0, :4, 1] = [242.0, 242.0, 100.0, 80.0]
    raw[0, 4 + 2, 1] = 0.75
    # anchor 2: person high conf (filtered by allowed_classes)
    raw[0, :4, 2] = [100.0, 100.0, 40.0, 90.0]
    raw[0, 4 + 0, 2] = 0.95
    # anchor 3: bus overlapping car (kept due to class-aware NMS)
    raw[0, :4, 3] = [241.0, 241.0, 100.0, 80.0]
    raw[0, 4 + 5, 3] = 0.85

    class _FakeSession:
        def run(self, _outs, _inputs):
            return [raw]

    runner._session = _FakeSession()
    frame = np.zeros((480, 480, 3), dtype=np.uint8)
    boxes = runner.predict(
        frame,
        conf_threshold=0.5,
        iou_threshold=0.5,
        allowed_classes={"car", "bus"},
    )
    labels = sorted(b.class_label for b in boxes)
    assert labels == ["bus", "car"]
    assert abs(boxes[0].confidence - 0.90) < 1e-5


def test_onnx_detectors_avoid_importing_torch_or_ultralytics():
    fake_runner_calls = []

    class _StubRunner:
        def __init__(self, model_path, *, imgsz, device):
            self.active_provider = "CPUExecutionProvider"
            fake_runner_calls.append(("init", model_path, imgsz, device))

        def warmup(self):
            fake_runner_calls.append(("warmup",))

        def close(self):
            fake_runner_calls.append(("close",))

    with (
        patch("src.detection.vehicle_detector.OnnxYoloRunner", _StubRunner),
        patch("src.detection.plate_detector.OnnxYoloRunner", _StubRunner),
    ):
        vd = VehicleDetector("dummy_vehicle.onnx", imgsz=480, device="cpu", half=False)
        vd._load_model()
        pd = PlateDetector("dummy_plate.onnx", imgsz=320, device="cpu", half=False)
        pd._load_model()
        vd.close()
        pd.close()

    assert ("init", "dummy_vehicle.onnx", 480, "cpu") in fake_runner_calls
    assert ("init", "dummy_plate.onnx", 320, "cpu") in fake_runner_calls
    assert fake_runner_calls.count(("warmup",)) == 2
    assert fake_runner_calls.count(("close",)) == 2

