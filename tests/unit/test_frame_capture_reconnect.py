"""Unit tests for FrameCapture live-stream buffering, reconnect, and stall recovery."""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import numpy as np

from src.capture.frame_capture import FrameCapture


class _FakeCapture:
    """Thread-safe configurable cv2.VideoCapture stub for live reconnect tests."""

    def __init__(self, script):
        self._script = list(script)
        self.released = False
        self.props = {}

    def isOpened(self) -> bool:
        return True

    def set(self, prop, value) -> bool:
        self.props[prop] = value
        return True

    def read(self):
        if not self._script:
            time.sleep(0.02)
            return False, None
        item = self._script.pop(0)
        if callable(item):
            return item()
        return item

    def release(self) -> None:
        self.released = True


def test_live_stream_reconnects_after_read_failure(monkeypatch):
    """Verify _reader_loop invokes _reconnect on read failure and resumes frames."""
    frame_a = np.full((32, 32, 3), 10, dtype=np.uint8)
    frame_b = np.full((32, 32, 3), 200, dtype=np.uint8)
    trigger_drop = threading.Event()

    pause = threading.Event()

    def _cap1_read():
        if not trigger_drop.is_set():
            pause.wait(0.005)
            return True, frame_a
        return False, None

    cap1 = _FakeCapture([_cap1_read] * 500)
    cap2 = _FakeCapture([(True, frame_b)] * 50)
    caps = [cap1, cap2]

    monkeypatch.setattr("src.capture.frame_capture.time.sleep", lambda _s: None)
    with patch("src.capture.frame_capture.cv2.VideoCapture", side_effect=lambda _src: caps.pop(0)):
        fc = FrameCapture("rtsp://camera.local/stream", max_retries=2, stall_timeout=1.5)
        assert fc.open() is True
        try:
            ok1, got1 = fc.read_frame()
            assert ok1 and int(got1[0, 0, 0]) == 10

            # Force stream failure on cap1 and verify _reader_loop reconnects to cap2.
            trigger_drop.set()
            got2 = None
            for _ in range(15):
                ok2, frame = fc.read_frame()
                if ok2 and int(frame[0, 0, 0]) == 200:
                    got2 = frame
                    break
            assert got2 is not None, "Expected recovered frame after RTSP reconnect"
        finally:
            fc.release()
    assert cap1.released is True


def test_live_stream_abandons_stalled_reader_and_opens_new_epoch(monkeypatch):
    """Verify stall_timeout abandons a hung read() epoch and prevents stale clobber."""
    unblock_stall = threading.Event()
    stale_frame = np.full((32, 32, 3), 55, dtype=np.uint8)
    fresh_frame = np.full((32, 32, 3), 240, dtype=np.uint8)

    def _hung_read():
        unblock_stall.wait(timeout=2.0)
        return True, stale_frame

    stalled_cap = _FakeCapture([_hung_read])
    recovered_cap = _FakeCapture([(True, fresh_frame)] * 20)
    caps = [stalled_cap, recovered_cap]

    monkeypatch.setattr("src.capture.frame_capture.time.sleep", lambda _s: None)
    with patch("src.capture.frame_capture.cv2.VideoCapture", side_effect=lambda _src: caps.pop(0)):
        fc = FrameCapture("rtsp://camera.local/stalled", max_retries=2, stall_timeout=0.15)
        assert fc.open() is True
        try:
            assert fc._epoch == 1
            ok_stalled, f_stalled = fc.read_frame()
            assert (ok_stalled, f_stalled) == (False, None)
            assert fc._epoch == 2  # Abandoned stalled epoch 1, started fresh epoch 2.

            # Unblock old reader thread: epoch check must keep epoch 2 frames clean.
            unblock_stall.set()
            ok_fresh, got_fresh = fc.read_frame()
            assert ok_fresh and int(got_fresh[0, 0, 0]) == 240
        finally:
            unblock_stall.set()
            fc.release()
