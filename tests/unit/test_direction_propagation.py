"""
Unit tests verifying vehicle direction estimation, line crossing, and database event upgrading.
"""

import unittest
from src.database import db
from src.database.models import VehicleEvent
from src.utils.direction_detector import DirectionDetector, VirtualLine


class TestDirectionPropagation(unittest.TestCase):
    """Verify direction detector integration and DB upgrade logic."""

    @classmethod
    def setUpClass(cls):
        db.init_db("sqlite:///:memory:")

    def setUp(self):
        # Create fresh tables for each test
        with db.get_session() as session:
            session.query(VehicleEvent).delete()
            session.commit()

    def test_camera_direction_confirmation_sequence(self):
        """Track moving in ch01 (+x, +y) correctly transitions from UNKNOWN to IN."""
        config = {
            "direction": {
                "active_camera": "ch01",
                "min_displacement_px": 25.0,
                "min_confirm_frames": 3,
                "history_frames": 15,
                "cone_angle_deg": 55.0,
            }
        }
        dd = DirectionDetector.from_config(config)
        track_id = 99

        # Jitter / minor displacement -> UNKNOWN
        dd.update(track_id, (500.0, 200.0))
        dd.update(track_id, (505.0, 205.0))
        self.assertFalse(dd.is_confirmed(track_id))
        self.assertEqual(dd.get_direction(track_id), "UNKNOWN")

        # Significant movement along (+x, +y) vector -> confirmed IN (needs 3 confirmed frame votes)
        dd.update(track_id, (540.0, 240.0))
        dd.update(track_id, (570.0, 270.0))
        dd.update(track_id, (600.0, 300.0))
        self.assertTrue(dd.is_confirmed(track_id))
        self.assertEqual(dd.get_direction(track_id), "IN")

    def test_upgrade_recent_event_direction_and_plate(self):
        """When an event was saved with UNKNOWN direction or low conf, upgrade_recent_event

        updates it in-place to confirmed IN/OUT and better plate read.
        """
        with db.get_session() as session:
            ev = db.insert_event(
                session,
                {
                    "plate_number": "UP70PI5101",
                    "vehicle_type": "Private",
                    "plate_color": "White",
                    "series_type": "normal",
                    "direction": "UNKNOWN",
                    "image_path": "path/to/early.jpg",
                    "confidence": 0.85,
                    "ocr_text": "UP70PI5101",
                },
            )
            self.assertIsNotNone(ev)
            event_id = ev.id

        # Now simulate upgrade when subsequent observations confirm IN and 'UP70PT5101'
        with db.get_session() as session:
            upgraded = db.upgrade_recent_event(
                session=session,
                plate_number="UP70PT5101",
                direction="IN",
                confidence=0.99,
                ocr_text="UP70PT5101",
            )
            self.assertTrue(upgraded)

            # Query updated event
            updated_ev = session.query(VehicleEvent).filter_by(id=event_id).first()
            self.assertEqual(updated_ev.plate_number, "UP70PT5101")
            self.assertEqual(updated_ev.direction, "IN")
            self.assertAlmostEqual(updated_ev.confidence, 0.99)


if __name__ == "__main__":
    unittest.main()
