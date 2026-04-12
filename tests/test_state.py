"""State machine transitions."""

from __future__ import annotations

import pytest

from roulette_predict.state import AppState, StateModel


def test_idle_to_setup_to_collecting() -> None:
    m = StateModel()
    assert m.state == AppState.IDLE
    assert m.begin_setup()
    assert m.state == AppState.SETUP_STEP1_WHEEL
    assert m.advance_after_step1()
    assert m.state == AppState.SETUP_STEP2_PATH
    assert m.advance_after_step2()
    assert m.state == AppState.SETUP_STEP3_OCR
    assert m.finish_setup()
    assert m.state == AppState.COLLECTING_SPINS
    assert m.spin.count == 0


def test_cancel_setup() -> None:
    m = StateModel()
    m.begin_setup()
    m.advance_after_step1()
    m.cancel_setup()
    assert m.state == AppState.IDLE


def test_thirty_spins_to_trained() -> None:
    m = StateModel()
    m.begin_setup()
    m.advance_after_step1()
    m.advance_after_step2()
    m.finish_setup()
    for _ in range(29):
        assert not m.on_spin_recorded()
        assert m.state == AppState.COLLECTING_SPINS
    assert m.on_spin_recorded()
    assert m.state == AppState.TRAINED


def test_double_begin_setup_from_idle_only() -> None:
    m = StateModel()
    assert m.begin_setup()
    assert not m.begin_setup()
    m.cancel_setup()
    assert m.begin_setup()


def test_finish_setup_wrong_state() -> None:
    m = StateModel()
    assert not m.finish_setup()


def test_complete_calibration_from_overlay_after_step1() -> None:
    """Overlay completes all steps in one flow; main window never advances SETUP_STEP2/3."""
    m = StateModel()
    m.begin_setup()
    assert m.state == AppState.SETUP_STEP1_WHEEL
    m.complete_calibration_from_overlay()
    assert m.state == AppState.COLLECTING_SPINS
    assert m.spin.count == 0
