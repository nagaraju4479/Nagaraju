"""SpinDebouncer fast history behavior."""

from __future__ import annotations

from roulette_predict.vision.ocr_spin import SpinDebouncer


def test_emit_first_value_on_first_clean_read() -> None:
    d = SpinDebouncer()
    assert d.feed("x") is None
    assert d.feed("12") == 12


def test_two_digit_requires_two_consecutive_reads() -> None:
    """Two-digit OCR values now need 2 consecutive agreeing reads to prevent misreads."""
    d = SpinDebouncer()
    assert d.feed("12") == 12
    assert d.feed("12") is None
    assert d.feed("26") is None  # first read of 26
    assert d.feed("26") == 26   # second read confirms


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
    assert d.feed("5", now=t0 + 28.0) == 5     # cooldown is 25 s


def test_no_spam_while_display_stable() -> None:
    d = SpinDebouncer()
    assert d.feed("3") == 3
    for _ in range(10):
        assert d.feed("3") is None


def test_two_digit_transition_needs_confirmation() -> None:
    """Even 2-digit values need 2 reads — prevents one-off misreads like 36→35."""
    d = SpinDebouncer()
    d.last_emitted = 11
    assert d.feed("34") is None  # first read
    assert d.feed("34") == 34   # confirmed
    assert d.feed("26") is None  # first read
    assert d.feed("26") == 26   # confirmed


def test_single_frame_glitch_between_values_ignored() -> None:
    d = SpinDebouncer()
    d.last_emitted = 6
    assert d.feed("8") is None
    assert d.feed("4") is None
    assert d.feed("4") == 4


def test_one_off_misread_rejected() -> None:
    """If OCR briefly misreads 36 as 35, the glitch is ignored because 35 only appears once."""
    d = SpinDebouncer()
    d.last_emitted = 10
    assert d.feed("35") is None  # misread (only once)
    assert d.feed("36") is None  # correct value, first read
    assert d.feed("36") == 36   # confirmed
