"""Unit tests for early-exit consensus in OCRFusion and early blacklist alerts."""

from __future__ import annotations

import threading

from src.alerts import blacklist as blacklist_mod
from src.alerts.blacklist import Blacklist
from src.cameras.manager import ProgressReporter
from src.cameras.models import CameraProgress
from src.validation.ocr_fusion import OCRFusion


def test_has_consensus_triggers_early_on_uncontested_high_conf_reads():
    fusion = OCRFusion(window_size=5, min_exact_votes=2)
    fusion.add_result(1, "BR06PO7564", 0.86)
    assert not fusion.has_consensus(1, early_exact_votes=2, min_consensus_conf=0.80)

    fusion.add_result(1, "BR06PO7564", 0.84)
    assert fusion.has_consensus(1, early_exact_votes=2, min_consensus_conf=0.80)
    assert fusion.flush(1) == ("BR06PO7564", 0.86)


def test_has_consensus_waits_when_confidence_is_below_early_threshold():
    fusion = OCRFusion(window_size=5, min_exact_votes=2)
    fusion.add_result(1, "BR06PO7564", 0.65)
    fusion.add_result(1, "BR06PO7564", 0.68)
    assert not fusion.has_consensus(1, early_exact_votes=2, min_consensus_conf=0.80)

    # Full window still triggers consensus flush as before
    fusion.add_result(1, "BR06PO7564", 0.69)
    fusion.add_result(1, "BR06PO7564", 0.67)
    fusion.add_result(1, "BR06PO7564", 0.66)
    assert fusion.has_consensus(1, early_exact_votes=2, min_consensus_conf=0.80)


def test_has_consensus_waits_on_tied_candidates():
    fusion = OCRFusion(window_size=5, min_exact_votes=2)
    fusion.add_result(1, "BR06PO7564", 0.85)
    fusion.add_result(1, "BR06P07564", 0.85)
    assert not fusion.has_consensus(1, early_exact_votes=2, min_consensus_conf=0.80)


def test_early_blacklist_alert_does_not_duplicate_on_detection(monkeypatch, tmp_path):
    bl_path = tmp_path / "blacklist.yaml"
    bl_path.write_text('plates:\n  - plate: "DL7CD5017"\n    reason: "Flagged"\n')
    monkeypatch.setattr(blacklist_mod, "_default", Blacklist(bl_path))

    logs: list[str] = []
    publishes = 0

    def _pub():
        nonlocal publishes
        publishes += 1

    reporter = ProgressReporter(
        CameraProgress("CAM001", "Gate 1", 1),
        publish=_pub,
        stop_event=threading.Event(),
        log=logs.append,
    )
    assert reporter.on_alert("DL7CD5017") is True
    assert publishes == 1
    assert reporter.on_alert("KA01AA1111") is False

    reporter.on_detection(plate_number="DL7CD5017", alert_already_sent=True)
    alerts = [line for line in logs if "BLACKLISTED" in line]
    assert alerts == ["ALERT: BLACKLISTED vehicle DL7CD5017 (Flagged) detected at Gate 1"]

