import numpy as np

from src.ocr.rapidocr_engine import RapidOCREngine


def test_recognize_handles_detections_list_and_time():
    engine = RapidOCREngine.__new__(RapidOCREngine)
    # Simulate RapidOCR returning (detections_list, time_float)
    # detections_list: list of [box, text, score]
    dets = [
        ([[1, 2], [3, 4]], 'KA19TR0234', 0.87),
    ]
    engine._engine = lambda image, **kwargs: (dets, 0.123)
    engine._init_failed = False
    engine.text_score = 0.5
    engine.use_det = True
    engine.use_cls = True
    engine.use_rec = True

    text, confidence = engine.recognize(np.zeros((32, 128, 3), dtype=np.uint8))

    assert text == "KA19TR0234"
    assert abs(confidence - 0.87) < 1e-6


def test_recognize_handles_detections_list_and_elapse_list():
    engine = RapidOCREngine.__new__(RapidOCREngine)
    # Actual rapidocr_onnxruntime use_det=True return shape: (items, [det_s, cls_s, rec_s])
    dets = [
        ([[61.0, 8.0], [120.0, 9.0], [120.0, 38.0], [61.0, 37.0]], "PSS0L5318", 0.8539),
    ]
    engine._engine = lambda image, **kwargs: (dets, [0.35, 0.0, 0.0346])
    engine._init_failed = False
    engine.text_score = 0.5
    engine.use_det = True
    engine.use_cls = False
    engine.use_rec = True

    text, confidence = engine.recognize(np.zeros((32, 128, 3), dtype=np.uint8))

    assert text == "PSS0L5318"
    assert abs(confidence - 0.8539) < 1e-6


def test_recognize_handles_det_false_rec_only_tuple():
    engine = RapidOCREngine.__new__(RapidOCREngine)
    # Actual rapidocr_onnxruntime use_det=False return shape: ([[text, score]], [rec_s])
    dets = [["UP65DV7672", 0.9549]]
    engine._engine = lambda image, **kwargs: (dets, [0.0346])
    engine._init_failed = False
    engine.text_score = 0.5
    engine.use_det = False
    engine.use_cls = False
    engine.use_rec = True

    text, confidence = engine.recognize(np.zeros((32, 128, 3), dtype=np.uint8))

    assert text == "UP65DV7672"
    assert abs(confidence - 0.9549) < 1e-6


def test_recognize_handles_legacy_two_tuple_txts_scores():
    engine = RapidOCREngine.__new__(RapidOCREngine)
    engine._engine = lambda image, **kwargs: (["DL01AB1234"], [0.92])
    engine._init_failed = False
    engine.text_score = 0.5
    engine.use_det = True
    engine.use_cls = True
    engine.use_rec = True

    text, confidence = engine.recognize(np.zeros((32, 128, 3), dtype=np.uint8))

    assert text == "DL01AB1234"
    assert abs(confidence - 0.92) < 1e-6


def test_recognize_pads_border_only_when_use_det_enabled():
    seen_shapes = []

    def fake_engine(image, **kwargs):
        seen_shapes.append(image.shape[:2])
        return ([([[1, 2], [3, 4]], "UP65DL5318", 0.90)], [0.02, 0.0, 0.01])

    engine = RapidOCREngine.__new__(RapidOCREngine)
    engine._engine = fake_engine
    engine._init_failed = False
    engine.text_score = 0.5
    engine.use_det = True
    engine.use_cls = False
    engine.use_rec = True
    engine.det_border_pad = 16

    crop = np.zeros((40, 120, 3), dtype=np.uint8)
    text, conf = engine.recognize(crop)
    assert text == "UP65DL5318"
    assert abs(conf - 0.90) < 1e-6
    assert seen_shapes[-1] == (72, 152)

    engine.use_det = False
    engine.recognize(crop)
    assert seen_shapes[-1] == (40, 120)
