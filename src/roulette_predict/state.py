"""Application state machine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class AppState(Enum):
    IDLE = auto()
    SETUP_STEP1_WHEEL = auto()
    SETUP_STEP2_PATH = auto()
    SETUP_STEP3_OCR = auto()
    COLLECTING_SPINS = auto()
    TRAINED = auto()


@dataclass
class SpinProgress:
    count: int = 0
    max_spins: int = 30


@dataclass
class StateModel:
    state: AppState = AppState.IDLE
    spin: SpinProgress = field(default_factory=SpinProgress)
    last_error: Optional[str] = None

    def begin_setup(self) -> bool:
        if self.state not in (AppState.IDLE, AppState.TRAINED):
            return False
        self.state = AppState.SETUP_STEP1_WHEEL
        self.last_error = None
        return True

    def advance_after_step1(self) -> bool:
        if self.state != AppState.SETUP_STEP1_WHEEL:
            return False
        self.state = AppState.SETUP_STEP2_PATH
        return True

    def advance_after_step2(self) -> bool:
        if self.state != AppState.SETUP_STEP2_PATH:
            return False
        self.state = AppState.SETUP_STEP3_OCR
        return True

    def finish_setup(self) -> bool:
        if self.state != AppState.SETUP_STEP3_OCR:
            return False
        self.state = AppState.COLLECTING_SPINS
        self.spin = SpinProgress(count=0, max_spins=30)
        return True

    def complete_calibration_from_overlay(self) -> None:
        """Overlay finishes all steps internally; main window never advances SETUP_STEP2/3 in this model."""
        if self.state in (
            AppState.SETUP_STEP1_WHEEL,
            AppState.SETUP_STEP2_PATH,
            AppState.SETUP_STEP3_OCR,
        ):
            self.state = AppState.COLLECTING_SPINS
            self.spin = SpinProgress(count=0, max_spins=30)
        self.last_error = None

    def cancel_setup(self) -> None:
        self.state = AppState.IDLE
        self.last_error = None

    def on_spin_recorded(self) -> bool:
        """Returns True if transitioned to TRAINED."""
        if self.state != AppState.COLLECTING_SPINS:
            return False
        self.spin.count += 1
        if self.spin.count >= self.spin.max_spins:
            self.state = AppState.TRAINED
            return True
        return False

    def reset_spins(self) -> None:
        self.spin.count = 0
        if self.state == AppState.TRAINED:
            self.state = AppState.COLLECTING_SPINS
