"""
Unit tests for campus-specific vehicle direction detection (ch01 Main Gate 2 and ch02 Gate 2).

Validates:
- CAMERA 1 (ch01): IN = lower-right (+x, +y), OUT = upper-left (-x, -y)
- CAMERA 2 (ch02): IN = lower-left (-x, +y), OUT = upper-right (+x, -y)
- Multi-frame confirmation over consecutive frames
- Suppression of stationary vehicles (< min_displacement_px)
- Suppression of vehicles merely appearing (history length < min_confirm_frames)
- Suppression of ambiguous perpendicular cross-traffic -> UNKNOWN
- Single-crossing direction lock to prevent flip-flops
- Runtime profile switching and configuration persistence
- Backwards compatibility with VirtualLine
"""

from __future__ import annotations

import unittest
from src.utils.direction_detector import DirectionDetector, VirtualLine, CameraDirectionConfig


class TestCampusDirectionDetection(unittest.TestCase):
    """Verify camera-specific vehicle direction detection for ch01 and ch02."""

    def setUp(self):
        self.config_ch01 = {
            "direction": {
                "active_camera": "ch01",
                "min_displacement_px": 25.0,
                "min_confirm_frames": 3,
                "history_frames": 15,
                "cone_angle_deg": 55.0,
            }
        }
        self.config_ch02 = {
            "direction": {
                "active_camera": "ch02",
                "min_displacement_px": 25.0,
                "min_confirm_frames": 3,
                "history_frames": 15,
                "cone_angle_deg": 55.0,
            }
        }

    # -------------------------------------------------------------
    # CAMERA 1 — ch01 (Main Gate 2)
    # IN: lower-right (+x, +y)
    # OUT: upper-left (-x, -y)
    # -------------------------------------------------------------
    def test_ch01_in_direction(self):
        """ch01: Vehicle moving towards lower-right (+x, +y) must detect IN."""
        dd = DirectionDetector.from_config(self.config_ch01)
        self.assertEqual(dd.camera_profile, "ch01")

        track_id = 101
        coords = [(400.0, 200.0), (430.0, 230.0), (460.0, 260.0), (490.0, 290.0)]
        results = [dd.update(track_id, pt) for pt in coords]

        self.assertIsNone(results[0])
        self.assertIsNone(results[1])
        self.assertIsNone(results[2])
        self.assertEqual(results[3], "IN")
        self.assertEqual(dd.get_direction(track_id), "IN")
        self.assertTrue(dd.is_confirmed(track_id))

    def test_ch01_out_direction(self):
        """ch01: Vehicle moving towards upper-left (-x, -y) must detect OUT."""
        dd = DirectionDetector.from_config(self.config_ch01)

        track_id = 102
        coords = [(500.0, 500.0), (470.0, 470.0), (440.0, 440.0), (410.0, 410.0)]
        results = [dd.update(track_id, pt) for pt in coords]

        self.assertIsNone(results[0])
        self.assertIsNone(results[1])
        self.assertIsNone(results[2])
        self.assertEqual(results[3], "OUT")
        self.assertEqual(dd.get_direction(track_id), "OUT")
        self.assertTrue(dd.is_confirmed(track_id))

    # -------------------------------------------------------------
    # CAMERA 2 — ch02 (Gate 2)
    # IN: lower-left (-x, +y)
    # OUT: upper-right (+x, -y)
    # -------------------------------------------------------------
    def test_ch02_in_direction(self):
        """ch02: Vehicle moving towards lower-left (-x, +y) must detect IN."""
        dd = DirectionDetector.from_config(self.config_ch02)
        self.assertEqual(dd.camera_profile, "ch02")

        track_id = 201
        coords = [(600.0, 200.0), (570.0, 230.0), (540.0, 260.0), (510.0, 290.0)]
        results = [dd.update(track_id, pt) for pt in coords]

        self.assertIsNone(results[0])
        self.assertIsNone(results[1])
        self.assertIsNone(results[2])
        self.assertEqual(results[3], "IN")
        self.assertEqual(dd.get_direction(track_id), "IN")
        self.assertTrue(dd.is_confirmed(track_id))

    def test_ch02_out_direction(self):
        """ch02: Vehicle moving towards upper-right (+x, -y) must detect OUT."""
        dd = DirectionDetector.from_config(self.config_ch02)

        track_id = 202
        coords = [(400.0, 500.0), (430.0, 470.0), (460.0, 440.0), (490.0, 410.0)]
        results = [dd.update(track_id, pt) for pt in coords]

        self.assertIsNone(results[0])
        self.assertIsNone(results[1])
        self.assertIsNone(results[2])
        self.assertEqual(results[3], "OUT")
        self.assertEqual(dd.get_direction(track_id), "OUT")
        self.assertTrue(dd.is_confirmed(track_id))

    # -------------------------------------------------------------
    # Constraint & Edge-case Tests
    # -------------------------------------------------------------
    def test_stationary_vehicle_suppression(self):
        """Vehicles parked or trembling (< 25px displacement) should NOT trigger IN/OUT."""
        dd = DirectionDetector.from_config(self.config_ch01)
        track_id = 301

        jitter = [(300.0, 300.0), (301.0, 300.5), (300.5, 301.2), (300.0, 300.0), (302.0, 301.0)]
        for pt in jitter:
            res = dd.update(track_id, pt)
            self.assertIsNone(res)

        self.assertEqual(dd.get_direction(track_id), "UNKNOWN")
        self.assertFalse(dd.is_confirmed(track_id))

    def test_vehicle_merely_appearing(self):
        """Vehicle appearing for only 1 or 2 frames cannot confirm direction."""
        dd = DirectionDetector.from_config(self.config_ch01)
        track_id = 302

        res1 = dd.update(track_id, (100.0, 100.0))
        res2 = dd.update(track_id, (150.0, 150.0))

        self.assertIsNone(res1)
        self.assertIsNone(res2)
        self.assertFalse(dd.is_confirmed(track_id))
        self.assertEqual(dd.get_direction(track_id), "UNKNOWN")

    def test_ambiguous_cross_traffic(self):
        """Cross-traffic moving perpendicularly must evaluate to UNKNOWN."""
        dd = DirectionDetector.from_config(self.config_ch01)
        track_id = 303

        coords = [(500.0, 200.0), (470.0, 230.0), (440.0, 260.0), (410.0, 290.0)]
        for pt in coords:
            res = dd.update(track_id, pt)
            self.assertIsNone(res)

        self.assertEqual(dd.get_direction(track_id), "UNKNOWN")
        self.assertFalse(dd.is_confirmed(track_id))

    def test_direction_lock_prevents_duplicate_flip_flop(self):
        """Once confirmed, direction remains locked to prevent flip-flops on erratic tracking."""
        dd = DirectionDetector.from_config(self.config_ch01)
        track_id = 304

        for pt in [(400.0, 200.0), (430.0, 230.0), (460.0, 260.0), (490.0, 290.0)]:
            dd.update(track_id, pt)

        self.assertEqual(dd.get_direction(track_id), "IN")
        self.assertTrue(dd.is_confirmed(track_id))

        res = dd.update(track_id, (450.0, 250.0))
        self.assertEqual(res, "IN")
        self.assertEqual(dd.get_direction(track_id), "IN")

    def test_runtime_profile_switching(self):
        """Verify dynamic profile switching between ch01, ch02, and default."""
        dd = DirectionDetector()
        self.assertEqual(dd.camera_profile, "ch01")

        dd.set_camera_profile("ch02")
        self.assertEqual(dd.camera_profile, "ch02")
        self.assertEqual(dd._active_config.camera_id, "ch02")

        for pt in [(600.0, 200.0), (570.0, 230.0), (540.0, 260.0), (510.0, 290.0)]:
            dd.update(401, pt)
        self.assertEqual(dd.get_direction(401), "IN")

    def test_virtual_line_backward_compatibility(self):
        """Verify traditional virtual line crossing still works when line is crossed."""
        line = VirtualLine(x1=0, y1=360, x2=1280, y2=360)
        dd = DirectionDetector(line=line)
        dd.update(501, (640.0, 300.0))
        res = dd.update(501, (640.0, 420.0))
        self.assertEqual(res, "IN")


if __name__ == "__main__":
    unittest.main()
