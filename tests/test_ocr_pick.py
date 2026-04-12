"""OCR parse selection helpers."""

from __future__ import annotations

from roulette_predict.vision.ocr_spin import _pick_best_parse


def test_prefers_more_digits_when_both_valid() -> None:
    attempts = [
        ("7\n", 7),
        ("17", 17),
    ]
    assert _pick_best_parse(attempts) == 17


def test_prefers_22_over_2() -> None:
    attempts = [("2", 2), ("22", 22)]
    assert _pick_best_parse(attempts) == 22


def test_single_digit_when_only_option() -> None:
    assert _pick_best_parse([("7", 7)]) == 7


def test_ignores_invalid() -> None:
    assert _pick_best_parse([("99", None), ("12", 12)]) == 12


def test_empty_none() -> None:
    assert _pick_best_parse([]) is None
    assert _pick_best_parse([("", 5)]) is None
