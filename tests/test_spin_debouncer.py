"""SpinDebouncer fast history behavior."""

from __future__ import annotations

from roulette_predict.vision.ocr_spin import SpinDebouncer


def test_emit_first_value_on_first_clean_read() -> None:
    d = SpinDebouncer()
    assert d.feed("x") is None
    assert d.feed("12") == 12


def test_teen_two_digit_raw_emits_one_read() -> None:
    """Two-digit OCR string → one frame (reduces lag vs waiting for duplicate parse)."""
    d = SpinDebouncer()
    assert d.feed("12") == 12
    assert d.feed("12") is None
    assert d.feed("26") == 26
    assert d.feed("26") is None


def test_single_digit_still_requires_two_consecutive_reads() -> None:
    d = SpinDebouncer()
    assert d.feed("12") == 12
    assert d.feed("2") is None
    assert d.feed("2") == 2


def test_same_value_twice_only_after_cooldown() -> None:
    """Brief OCR miss + same digit no longer duplicates; repeat same value needs a long gap."""
    d = SpinDebouncer()
    t0 = 1000.0
    assert d.feed("5", now=t0) == 5
    assert d.feed("5", now=t0 + 0.05) is None
    assert d.feed("", now=t0 + 0.1) is None
    assert d.feed("5", now=t0 + 0.2) is None
    assert d.feed("5", now=t0 + 0.25) is None  # two reads but < cooldown since last emit
    assert d.feed("5", now=t0 + 13.0) == 5


def test_no_spam_while_display_stable() -> None:
    d = SpinDebouncer()
    assert d.feed("3") == 3
    for _ in range(10):
        assert d.feed("3") is None


def test_teen_transition_one_read_each_when_raw_has_two_digits() -> None:
    d = SpinDebouncer()
    d.last_emitted = 11
    assert d.feed("34") == 34
    assert d.feed("26") == 26


def test_single_frame_glitch_between_values_ignored() -> None:
    d = SpinDebouncer()
    d.last_emitted = 6
    assert d.feed("8") is None
    assert d.feed("4") is None
    assert d.feed("4") == 4
