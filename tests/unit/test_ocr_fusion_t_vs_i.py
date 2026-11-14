"""
Unit tests verifying OCR character disambiguation (T vs I) and multi-frame consensus
for Indian number plates in the ALPR University Gate system.
"""

import unittest
from src.validation.ocr_fusion import OCRFusion
from src.ocr.ocr_postprocessor import correct_ocr_text
from src.validation.plate_validator import PlateValidator


class TestOCRFusionTvsI(unittest.TestCase):
    """Verify that multi-frame positional voting and area weighting correctly resolve T vs I."""

    def setUp(self):
        self.validator = PlateValidator()

    def test_postprocessor_maps_digit_7_to_T_at_letter_position(self):
        """Digit '7' at letter positions 4-5 in XX00XX0000 format maps to 'T'."""
        corrected = correct_ocr_text("UP70P75101")
        self.assertEqual(corrected, "UP70PT5101")

    def test_postprocessor_preserves_valid_letters_I_and_T(self):
        """Neither valid letter 'I' nor 'T' is blindly mutated at position 5."""
        self.assertEqual(correct_ocr_text("UP70PT5101"), "UP70PT5101")
        self.assertEqual(correct_ocr_text("UP70PI5101"), "UP70PI5101")

    def test_character_level_voting_resolves_t_vs_i_majority(self):
        """When sharper/close-up frames show 'T' and distant frames show 'I',

        character-level consensus resolves position 5 to 'T'.
        """
        fusion = OCRFusion(window_size=7, min_exact_votes=2)
        track_id = 8

        # 1 distant frame reading 'I'
        fusion.add_result(track_id, "UP70PI5101", 0.94, crop_area=2500)

        # 3 closer frames reading 'T' with high confidence and larger area
        fusion.add_result(track_id, "UP70PT5101", 0.98, crop_area=6000)
        fusion.add_result(track_id, "UP70PT5101", 0.99, crop_area=6200)
        fusion.add_result(track_id, "UP70PT5101", 0.99, crop_area=6400)

        fused_plate, fused_conf = fusion.get_result(track_id)
        self.assertEqual(fused_plate, "UP70PT5101")
        self.assertGreaterEqual(fused_conf, 0.98)

    def test_direction_gated_consensus_suppresses_premature_exit(self):
        """has_consensus returns False when direction is unconfirmed even if minimum votes exist."""
        fusion = OCRFusion(window_size=7, min_exact_votes=2)
        track_id = 8

        # 3 votes of the same plate
        fusion.add_result(track_id, "UP70PI5101", 0.94)
        fusion.add_result(track_id, "UP70PI5101", 0.94)
        fusion.add_result(track_id, "UP70PI5101", 0.94)

        # When direction is explicitly unconfirmed, early exit is suppressed
        self.assertFalse(
            fusion.has_consensus(track_id, early_exact_votes=3, direction_confirmed=False)
        )

        # When direction is confirmed, consensus triggers
        self.assertTrue(
            fusion.has_consensus(track_id, early_exact_votes=3, direction_confirmed=True)
        )


if __name__ == "__main__":
    unittest.main()

