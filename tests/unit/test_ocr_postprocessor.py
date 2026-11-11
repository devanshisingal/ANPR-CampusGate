"""Unit tests for ocr.ocr_postprocessor positional character corrections."""

from src.ocr.ocr_postprocessor import correct_ocr_text


def test_correct_ocr_text_preserves_valid_plates():
    assert correct_ocr_text("BR01WT3877") == "BR01WT3877"
    assert correct_ocr_text("DL01AB1234") == "DL01AB1234"
    assert correct_ocr_text("KA19TR0234") == "KA19TR0234"
    assert correct_ocr_text("BH01AB1234") == "BH01AB1234"


def test_correct_ocr_text_fixes_digits_in_letter_slots_and_letters_in_digit_slots():
    # Positions 0-1 and 4-5 convert digits->letters; 2-3 and 6-9 convert letters->digits
    assert correct_ocr_text("8R018T3877") == "BR01BT3877"
    assert correct_ocr_text("BRO1WT3B77") == "BR01WT3877"
    assert correct_ocr_text("0L015B1Z3G") == "OL01SB1236"
    assert correct_ocr_text("BR06P97564") == "BR06PO7564"


def test_correct_ocr_text_leaves_non_10_char_lengths_alone():
    assert correct_ocr_text("DL1CAB1234") == "DL1CAB1234"
    assert correct_ocr_text("UP65A1234") == "UP65A1234"
    assert correct_ocr_text("") == ""

