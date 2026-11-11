"""Targeted mocked-ML pipeline wiring test + short-clip E2E smoke test.

Demonstrates two test-speed practices without dropping coverage:
1. Mock expensive YOLO/OCR models when testing tracker, direction, fusion,
   blacklist alerts, and SQLite persistence together (< 1s runtime).
2. Reuse a single module-scoped ``PipelineModels`` bundle across real-model
   checks and run the real E2E smoke pass over a 15-second (375-frame) CCTV
   clip rather than reprocessing the full 60-second video.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from src.alerts import blacklist as bl_mod
from src.cameras.manager import ProgressReporter
from src.cameras.models import CameraProgress
from src.cameras.pipeline_models import PipelineModels
from src.database import db as database
from src.detection.vehicle_detector import Detection
from src.utils.config import load_config

_TEST_VIDEO = Path(__file__).resolve().parents[2] / "data" / "uploads" / "test_video.mp4"
_CLIP_SECONDS = 15


def _isolated_config(tmp_path: Path, video_path: Path, blacklist_plate: str | None = None) -> tuple[dict, Path]:
    """Build an isolated config writing all DB and image outputs into *tmp_path*."""
    cfg = load_config("config/config.yaml")
    db_path = tmp_path / "smoke.db"
    os.environ["ALPR_DB_PATH"] = str(db_path)
    cfg["video"]["source"] = str(video_path)
    cfg["video"]["frame_skip"] = 1
    cfg.setdefault("database", {})["path"] = f"sqlite:///{db_path}"
    cfg.setdefault("paths", {})["images"] = str(tmp_path / "plates")
    cfg["paths"]["vehicle_images"] = str(tmp_path / "vehicles")
    cfg["paths"]["thumbnails"] = str(tmp_path / "thumbs")
    cfg["direction"]["min_confirm_frames"] = 2
    if blacklist_plate is not None:
        bl_file = tmp_path / "blacklist.yaml"
        bl_file.write_text(
            yaml.safe_dump({"plates": [{"plate": blacklist_plate, "reason": "smoke"}]}),
            encoding="utf-8",
        )
        return cfg, bl_file
    return cfg, tmp_path / "unused_blacklist.yaml"


def _write_synthetic_video(path: Path, num_frames: int = 18, width: int = 640, height: int = 480) -> None:
    """Create a tiny synthetic MP4 video for mocked-model pipeline tests."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (width, height)
    )
    assert writer.isOpened(), f"Could not create synthetic video at {path}"
    try:
        frame = np.full((height, width, 3), 120, dtype=np.uint8)
        for _ in range(num_frames):
            writer.write(frame)
    finally:
        writer.release()


class _FakeVehicleDetector:
    """Deterministic lower-right (+x, +y) moving vehicle stub matching ch01 IN vector."""

    def __init__(self) -> None:
        self._frame_idx = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        # Step (+14, +14) on a 240x160 box -> consecutive IoU (~0.76 > 0.50)
        # and ~39.6px displacement (>= 25px) by the 3rd tracked frame.
        x1 = 40 + self._frame_idx * 14
        y1 = 40 + self._frame_idx * 14
        self._frame_idx += 1
        return [
            Detection(
                bbox=(float(x1), float(y1), float(x1 + 240), float(y1 + 160)),
                class_label="car",
                confidence=0.92,
            )
        ]


class _FakePlateDetector:
    def detect(self, vehicle_crop: np.ndarray) -> list[np.ndarray]:
        return [np.full((36, 130, 3), 240, dtype=np.uint8)]


class _FakeSrEnhancer:
    def enhance(self, crop: np.ndarray) -> np.ndarray:
        return crop


class _FakeOcrEngine:
    def recognize(self, image: np.ndarray) -> tuple[str, float]:
        return ("DL01AB1234", 0.93)


@pytest.fixture(autouse=True)
def _restore_db_and_blacklist():
    old_bl = bl_mod._default
    yield
    bl_mod._default = old_bl
    database._engine = None
    database._SessionFactory = None
    database._db_type = None


@pytest.fixture(scope="module")
def reused_models() -> PipelineModels:
    """Load ONNX + RapidOCR models ONCE for all real-model tests in this module."""
    cfg = load_config("config/config.yaml")
    models = PipelineModels.load(cfg)
    try:
        yield models
    finally:
        models.close()


@pytest.fixture(scope="module")
def short_15s_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Slice a 15-second CCTV clip from ``data/uploads/test_video.mp4`` once per module."""
    if not _TEST_VIDEO.exists():
        pytest.fail(f"Required sample CCTV video missing: {_TEST_VIDEO}")

    clip_dir = tmp_path_factory.mktemp("short_cctv_clip")
    clip_path = clip_dir / "clip_15s.mp4"

    cap = cv2.VideoCapture(str(_TEST_VIDEO))
    assert cap.isOpened(), f"Failed to open {_TEST_VIDEO}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(round(fps * _CLIP_SECONDS))

    writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened(), f"Failed to open writer for {clip_path}"
    written = 0
    try:
        while written < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
    finally:
        cap.release()
        writer.release()

    assert 250 <= written <= 500, f"Expected a 10-20s clip (~250-500 frames @ 25fps), wrote {written}"
    return clip_path


def test_pipeline_wiring_with_mocked_models(tmp_path: Path):
    """Exercise tracker, IN/OUT direction, OCR fusion, DB store, and blacklist alert with mocked ML."""
    from scripts.run_pipeline import run_pipeline

    video_path = tmp_path / "synthetic.mp4"
    _write_synthetic_video(video_path, num_frames=18)
    cfg, bl_file = _isolated_config(tmp_path, video_path, blacklist_plate="DL01AB1234")
    bl_mod._default = bl_mod.Blacklist(bl_file)

    alerts: list[str] = []

    class _AlertSpy(ProgressReporter):
        def on_alert(self, plate_number: str) -> bool:
            fired = super().on_alert(plate_number)
            if fired:
                alerts.append(plate_number)
            return fired

    spy = _AlertSpy(
        CameraProgress(camera_id="GATE-01", camera_name="Main Gate", order=1),
        publish=lambda: None,
        stop_event=threading.Event(),
        log=lambda _msg: None,
    )
    fake_models = PipelineModels(
        _FakeVehicleDetector(), _FakePlateDetector(), _FakeOcrEngine(), _FakeSrEnhancer()
    )

    t0 = time.perf_counter()
    summary = run_pipeline(cfg, benchmark=False, progress=spy, models=fake_models.shared())
    elapsed = time.perf_counter() - t0

    assert elapsed < 10.0, f"Mocked pipeline test exceeded 10s budget ({elapsed:.2f}s)"
    assert summary["frames_processed"] == 18
    assert summary["events_stored"] == 1
    assert alerts == ["DL01AB1234"]

    with database.get_session() as session:
        events = database.get_all_events(session)
        assert len(events) == 1
        assert events[0]["plate_number"] == "DL01AB1234"
        assert events[0]["direction"] == "IN"


def test_reused_models_single_frame_detection_and_ocr(
    reused_models: PipelineModels, short_15s_clip: Path
):
    """Reuse already-loaded ONNX + RapidOCR models on a single real CCTV frame (< 2s)."""
    cap = cv2.VideoCapture(str(short_15s_clip))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 42)
    ok, frame = cap.read()
    cap.release()
    assert ok and frame is not None

    t0 = time.perf_counter()
    vehicles = reused_models.vehicle_detector.detect(frame)
    plates_found = 0
    ocr_texts: list[str] = []
    for det in vehicles:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        for plate_crop in reused_models.plate_detector.detect(crop):
            plates_found += 1
            text, _conf = reused_models.ocr_engine.recognize(plate_crop)
            if text:
                ocr_texts.append(text)
    elapsed = time.perf_counter() - t0

    assert elapsed < 2.0, f"Reused single-frame inference took too long ({elapsed:.2f}s)"
    assert len(vehicles) >= 1
    assert plates_found >= 1


@pytest.mark.e2e
def test_short_15s_cctv_clip_e2e_smoke(
    tmp_path: Path, reused_models: PipelineModels, short_15s_clip: Path
):
    """Run one fast E2E smoke test over a 15-second CCTV clip using reused models."""
    from scripts.run_pipeline import run_pipeline

    cfg, _ = _isolated_config(tmp_path, short_15s_clip)
    stop_event = threading.Event()
    reporter = ProgressReporter(
        CameraProgress(camera_id="GATE-01", camera_name="Main Gate", order=1),
        publish=lambda: None,
        stop_event=stop_event,
        log=lambda _msg: None,
    )
    # Enforce a hard 55s watchdog timeout so stalled runs fail immediately.
    watchdog = threading.Timer(55.0, stop_event.set)
    watchdog.daemon = True
    watchdog.start()
    try:
        t0 = time.perf_counter()
        summary = run_pipeline(
            cfg,
            benchmark=False,
            progress=reporter,
            models=reused_models.shared(),
        )
        elapsed = time.perf_counter() - t0
    finally:
        watchdog.cancel()

    assert not stop_event.is_set(), f"15s clip E2E smoke exceeded 55s timeout (frames={summary.get('frames_processed')})"
    assert summary["frames_processed"] == 375
    assert summary["events_stored"] >= 1
    assert elapsed < 55.0
    assert "torch" not in sys.modules
    assert "ultralytics" not in sys.modules

    with database.get_session() as session:
        events = database.get_all_events(session)
        assert len(events) >= 1
        assert all(ev["plate_number"] for ev in events)

