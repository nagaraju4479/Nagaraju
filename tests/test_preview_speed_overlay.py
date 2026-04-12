"""Speed overlay must not treat the dot in (exp.) as the numeric value."""

from __future__ import annotations

from roulette_predict.ui.preview_frame import extract_speed_number_for_overlay


def test_extracts_value_after_label_not_exp_dot() -> None:
    assert extract_speed_number_for_overlay("Ball speed (exp.): 1.23 rad/s") == "1.23"


def test_ellipsis_means_no_value() -> None:
    assert extract_speed_number_for_overlay("Ball speed (exp.): …") is None


def test_empty_means_no_value() -> None:
    assert extract_speed_number_for_overlay("") is None
