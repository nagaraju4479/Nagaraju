"""Main widget: tabs, controls, history, opacity/topmost, red border."""

from __future__ import annotations

import copy
import random
import threading
import time
from typing import Optional

from PySide6.QtCore import QCoreApplication, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QGuiApplication, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from roulette_predict.capture.worker import OcrSpinWorker, VisionWorker, ball_path_roi, wheel_roi
from roulette_predict.config_model import CalibrationData, HsvSettings, normalize_tesseract_cmd
from roulette_predict.persistence import load_config, save_config
from roulette_predict.state import AppState, StateModel
from roulette_predict.ui.ball_pick_overlay import BallPickScreenOverlay
from roulette_predict.ui.calibration_overlay import CalibrationOverlay
from roulette_predict.ui.preview_frame import RoulettePreviewFrame
from roulette_predict.ui.theme import build_app_stylesheet
from roulette_predict.vision.ocr_spin import is_tesseract_available, read_roulette_number_from_roi


def _mss_monitor_index_for_screen(screen) -> int:
    screens = QGuiApplication.screens()
    try:
        return screens.index(screen) + 1
    except ValueError:
        return 1


_STYLE_PRED_ACTIVE = "color: #00FF00; font-weight: bold; font-size: 10pt; font-family: 'Segoe UI', sans-serif;"
_STYLE_PRED_IDLE = "color: #CCCCCC; font-size: 10pt; font-family: 'Segoe UI', sans-serif;"


class MainWindow(QMainWindow):
    _seed_ocr_done = Signal(object, object)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RoulettePredict")
        self.resize(960, 640)
        self.setMinimumSize(10, 10)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)

        self._state = StateModel()
        self._cal = CalibrationData()
        self._hsv = HsvSettings()
        self._overlay: Optional[CalibrationOverlay] = None
        self._ball_pick_overlay: Optional[BallPickScreenOverlay] = None
        self._instruction_popup: Optional[QDialog] = None
        self._seed_history_attempt: int = 0
        self._history_numbers: list[int] = []
        self._tesseract_cmd: Optional[str] = None
        self._omega_was_low: bool = True
        self._HIGH_BALL_SPEED_RAD_S: float = 2.0
        self._LOW_BALL_SPEED_RAD_S: float = 1.0
        self._shutdown_in_progress: bool = False
        self._shutdown_deadline: float = 0.0

        raw = load_config()
        from roulette_predict.persistence import parse_loaded

        self._cal, self._hsv, red_border, opacity, self._tesseract_cmd = parse_loaded(raw)
        self._build_ui()
        self._seed_ocr_done.connect(self._on_seed_ocr_done)
        self._red_border_check.setChecked(red_border)
        self._opacity_slider.setValue(int(opacity * 100))
        self._apply_opacity(opacity)
        self._apply_red_border(red_border)
        self._connect_sliders()

        self._worker = VisionWorker(self)
        self._ocr_worker = OcrSpinWorker(self)
        self._worker.frame_wheel.connect(self._on_wheel_frame)
        self._worker.frame_color.connect(self._on_color_frame)
        self._worker.speed_text.connect(self._roulette_preview.set_speed_text)
        self._worker.status_text.connect(self._capture_hint.setText)
        self._worker.ball_omega.connect(self._on_ball_omega)
        self._ocr_worker.spin_detected.connect(
            self._on_spin,
            Qt.ConnectionType.QueuedConnection,
        )
        self._ocr_worker.ocr_debug.connect(
            self._ocr_debug_label.setText,
            Qt.ConnectionType.QueuedConnection,
        )
        # Keep preview blank until setup flow is completed.
        self._worker.set_calibration(None)
        self._ocr_worker.set_calibration(None)
        self._worker.set_hsv(self._hsv)
        self._ocr_worker.set_tesseract_cmd(self._tesseract_cmd)
        self._worker.start(QThread.Priority.HighPriority)
        self._ocr_worker.start(QThread.Priority.HighPriority)
        # Windows often throttles the GUI thread when our window is not focused; QueuedConnection
        # slots (OCR → history) then pile up. Pump posted events so capture keeps updating in background.
        self._event_pump = QTimer(self)
        self._event_pump.setTimerType(Qt.TimerType.PreciseTimer)
        self._event_pump.timeout.connect(self._pump_cross_thread_events)
        self._event_pump.start(33)
        self._set_previews_blank()

        esc_pick = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        esc_pick.setContext(Qt.ShortcutContext.WindowShortcut)
        esc_pick.activated.connect(self._cancel_ball_pick)

    def _cancel_ball_pick(self) -> None:
        if self._ball_pick_btn.isChecked():
            self._ball_pick_btn.setChecked(False)

    def _pump_cross_thread_events(self) -> None:
        QCoreApplication.sendPostedEvents()

    def _open_ball_screen_pick_overlay(self) -> None:
        screens = QGuiApplication.screens()
        if not screens:
            return
        mid = int(self._cal.monitor_index) - 1
        mid = max(0, min(mid, len(screens) - 1))
        geo = screens[mid].geometry()
        self._ball_pick_overlay = BallPickScreenOverlay(geo, self)
        self._ball_pick_overlay.picked.connect(self._on_ball_pick_screen)
        self._ball_pick_overlay.cancelled.connect(self._on_ball_pick_screen_cancel)
        self._ball_pick_overlay.destroyed.connect(self._on_ball_pick_overlay_destroyed)
        self.showMinimized()
        self._ball_pick_overlay.showFullScreen()

    def _on_ball_pick_overlay_destroyed(self) -> None:
        self._ball_pick_overlay = None

    def _on_ball_pick_screen_cancel(self) -> None:
        self._ball_pick_btn.blockSignals(True)
        self._ball_pick_btn.setChecked(False)
        self._ball_pick_btn.blockSignals(False)
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _on_ball_pick_screen(self, gx: int, gy: int) -> None:
        import mss
        import numpy as np

        broi = ball_path_roi(self._cal)
        if not broi:
            QMessageBox.warning(
                self,
                "Ball pick",
                "No Step-2 ball path in calibration. Complete Setup first.",
            )
            self._on_ball_pick_screen_cancel()
            return
        bl, bt, bw, bh = broi
        if not (bl <= gx < bl + bw and bt <= gy < bt + bh):
            QMessageBox.warning(
                self,
                "Ball pick",
                "Click inside the Step-2 ball-path area on screen (the rectangle around your painted path).",
            )
            self._open_ball_screen_pick_overlay()
            return
        rad = 7
        left = gx - rad
        top = gy - rad
        w = 2 * rad + 1
        h = 2 * rad + 1
        try:
            with mss.mss() as sct:
                shot = sct.grab({"left": left, "top": top, "width": w, "height": h})
                arr = np.asarray(shot)[:, :, :3].copy()
        except Exception:
            QMessageBox.warning(
                self,
                "Ball pick",
                "Could not read that screen region. Try again slightly away from the screen edge.",
            )
            self._open_ball_screen_pick_overlay()
            return
        self._apply_ball_color_lock_from_small_bgr_patch(arr)
        self._set_ball_pick_highlight_for_screen_xy(gx, gy)
        self._ball_pick_btn.blockSignals(True)
        self._ball_pick_btn.setChecked(False)
        self._ball_pick_btn.blockSignals(False)
        self._ball_pick_overlay = None
        self.showNormal()
        self.raise_()
        self.activateWindow()
        self._capture_hint.setText(
            "Ball color locked — tracker uses Step-2 path + motion on the spinning ball only. "
            "White disk + green ring on Roulette Detection; Color tab = ball-path mask."
        )

    def _set_ball_pick_highlight_for_screen_xy(self, gx: int, gy: int) -> None:
        wroi = wheel_roi(self._cal)
        if not wroi:
            self._worker.set_pick_highlight(None)
            return
        wl, wt, ww, wh = wroi
        img = self._roulette_preview.last_image()
        if img is None or img.isNull():
            self._worker.set_pick_highlight(None)
            return
        pw, ph = img.width(), img.height()
        u = (gx - wl) / max(1.0, float(ww))
        v = (gy - wt) / max(1.0, float(wh))
        hx = int(max(0, min(pw - 1, round(u * pw))))
        hy = int(max(0, min(ph - 1, round(v * ph))))
        self._worker.set_pick_highlight((hx, hy))

    def _apply_ball_color_lock_from_small_bgr_patch(self, arr) -> None:
        h, w = arr.shape[:2]
        if h < 2 or w < 2:
            return
        us_before = self._sliders["U-S"].value()
        self._sample_hsv_from_wheel_bgr_roi(arr, 0, 0, w, h, tight=True, for_ball_lock=True)
        picked = copy.copy(self._hsv)
        self._worker.set_ball_hsv_override(picked)
        self._sliders["U-S"].blockSignals(True)
        self._sliders["U-S"].setValue(us_before)
        self._sliders["U-S"].blockSignals(False)
        self._on_hsv_changed()

    def _on_ball_pick_toggled(self, on: bool) -> None:
        self._roulette_preview.set_ball_pick_cursor(False)
        if on:
            if not self._cal or len(self._cal.ball_path_points) < 2:
                self._ball_pick_btn.blockSignals(True)
                self._ball_pick_btn.setChecked(False)
                self._ball_pick_btn.blockSignals(False)
                self._capture_hint.setText("Complete **Step 2** (ball path on the wheel) in Setup first.")
                return
            self._capture_hint.setText(
                "The window minimizes — click the ball on the **live** wheel inside your Step-2 path. **Esc** cancels."
            )
            self._open_ball_screen_pick_overlay()
        else:
            if self._ball_pick_overlay is not None:
                self._ball_pick_overlay.close()
                self._ball_pick_overlay = None
            self.showNormal()
            self.raise_()
            self.activateWindow()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(self._splitter)

        left = QWidget()
        left.setMinimumWidth(140)
        left.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(0, 0, 0, 0)
        self._tabs = QTabWidget()
        self._roulette_preview = RoulettePreviewFrame()
        self._color_preview = QLabel()
        self._color_preview.setMinimumSize(10, 10)
        self._color_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._color_preview.setStyleSheet("background-color: #000000; color: #888888; border: none;")
        self._color_preview.setText("HSV masked preview")
        self._tabs.addTab(self._roulette_preview, "Roulette Detection")
        self._tabs.addTab(self._color_preview, "Color Detection")
        left_l.addWidget(self._tabs)

        right_inner = QWidget()
        right_inner.setMinimumWidth(0)
        right_inner.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        rl = QVBoxLayout(right_inner)
        rl.setSpacing(8)
        rl.setContentsMargins(4, 4, 8, 8)

        hdr_controls = QLabel("CONTROLS & INFO")
        hdr_controls.setObjectName("sectionHeader")
        rl.addWidget(hdr_controls)

        self._start_setup_btn = QPushButton("Start Setup")
        self._start_setup_btn.clicked.connect(self._start_setup)
        rl.addWidget(self._start_setup_btn)

        pred_key = QLabel("PREDICTION:")
        pred_key.setObjectName("predictionKey")
        self._prediction_value = QLabel("Waiting for Setup")
        self._prediction_value.setStyleSheet(_STYLE_PRED_IDLE)
        rl.addWidget(pred_key)
        rl.addWidget(self._prediction_value)

        self._prediction_result = QLabel("")
        self._prediction_result.setStyleSheet(_STYLE_PRED_ACTIVE)
        self._prediction_result.hide()
        rl.addWidget(self._prediction_result)

        self._capture_hint = QLabel("")
        self._capture_hint.setObjectName("hintMuted")
        self._capture_hint.setWordWrap(True)
        rl.addWidget(self._capture_hint)

        self._ocr_debug_label = QLabel("")
        self._ocr_debug_label.setObjectName("hintMuted")
        self._ocr_debug_label.setWordWrap(True)
        rl.addWidget(self._ocr_debug_label)

        tess_lbl = QLabel("Tesseract executable (optional)")
        tess_lbl.setObjectName("hintMuted")
        rl.addWidget(tess_lbl)
        self._tesseract_edit = QLineEdit()
        self._tesseract_edit.setPlaceholderText(
            r"e.g. C:\Program Files\Tesseract-OCR\tesseract.exe — leave empty to use PATH"
        )
        self._tesseract_edit.setText(self._tesseract_cmd or "")
        self._tesseract_edit.editingFinished.connect(self._on_tesseract_path_changed)
        rl.addWidget(self._tesseract_edit)

        history_frame = QFrame()
        history_frame.setObjectName("historyPanel")
        hf_l = QVBoxLayout(history_frame)
        hf_l.setContentsMargins(0, 0, 0, 0)
        hf_l.setSpacing(8)
        hdr_hist = QLabel("NUMBER HISTORY")
        hdr_hist.setObjectName("sectionHeader")
        hdr_hist.setStyleSheet("margin-top: 0; margin-bottom: 2px;")
        hf_l.addWidget(hdr_hist)
        hist_where = QLabel(
            "Source: Tesseract reads a screen capture of the rectangle you drew in Step 3 "
            "(those exact screen pixels each time). It does not read the web page’s code or DOM."
        )
        hist_where.setObjectName("hintMuted")
        hist_where.setWordWrap(True)
        hf_l.addWidget(hist_where)
        hist_hint = QLabel(
            "Tip: in Step 3, draw a tight box around only the newest number cell (often the leftmost). "
            "Shown here: newest → left, up to 30 numbers."
        )
        hist_hint.setObjectName("hintMuted")
        hist_hint.setWordWrap(True)
        hf_l.addWidget(hist_hint)
        self._history_label = QLabel("—")
        self._history_label.setObjectName("historyStrip")
        self._history_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        hist_scroll = QScrollArea()
        hist_scroll.setWidget(self._history_label)
        hist_scroll.setWidgetResizable(True)
        hist_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        hist_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        hist_scroll.setMinimumHeight(52)
        hist_scroll.setMaximumHeight(72)
        hist_scroll.setFrameShape(QFrame.Shape.NoFrame)
        hf_l.addWidget(hist_scroll)
        rl.addWidget(history_frame)

        hdr_color = QLabel("COLOR DETECTION — mask (U-S + more)")
        hdr_color.setObjectName("sectionHeader")
        rl.addWidget(hdr_color)
        color_hint = QLabel(
            "**1)** On Color tab, tune **U-S** / **L-V** — preview uses your **Step-2 painted path** as a narrow "
            "tube so the whole golden ring does not turn white (only the channel you brushed). "
            "**2)** **Pick white ball** (window minimizes) — click the spinning ball on the **live** game window, "
            "inside your painted path. After pick, this tab shows **only** the picked color on **one** blob; "
            "the green ring marks the tracked center."
        )
        color_hint.setObjectName("hintMuted")
        color_hint.setWordWrap(True)
        rl.addWidget(color_hint)

        self._ball_pick_btn = QPushButton("Pick white ball (live wheel — fullscreen click)")
        self._ball_pick_btn.setCheckable(True)
        self._ball_pick_btn.toggled.connect(self._on_ball_pick_toggled)
        rl.addWidget(self._ball_pick_btn)

        self._sliders = {}
        us_row = QHBoxLayout()
        us_lab = QLabel("U-S")
        us_lab.setObjectName("sliderLabel")
        self._sliders["U-S"] = QSlider(Qt.Orientation.Horizontal)
        self._sliders["U-S"].setRange(0, 255)
        us_row.addWidget(us_lab)
        us_row.addWidget(self._sliders["U-S"])
        rl.addLayout(us_row)

        self._advanced_hsv = QWidget()
        adv_l = QVBoxLayout(self._advanced_hsv)
        adv_l.setContentsMargins(0, 0, 0, 0)
        for name, lo, hi in [
            ("L-H", 0, 179),
            ("L-S", 0, 255),
            ("L-V", 0, 255),
            ("U-H", 0, 179),
            ("U-V", 0, 255),
        ]:
            row = QHBoxLayout()
            lab = QLabel(name)
            lab.setObjectName("sliderLabel")
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(lo, hi)
            row.addWidget(lab)
            row.addWidget(s)
            adv_l.addLayout(row)
            self._sliders[name] = s
        self._advanced_hsv.setVisible(False)
        rl.addWidget(self._advanced_hsv)
        self._advanced_hsv_toggle = QCheckBox("Show all HSV sliders")
        self._advanced_hsv_toggle.setChecked(False)
        self._advanced_hsv_toggle.toggled.connect(self._on_advanced_hsv_toggle)
        rl.addWidget(self._advanced_hsv_toggle)

        self._sliders["L-H"].setValue(self._hsv.l_h)
        self._sliders["U-H"].setValue(self._hsv.u_h)
        self._sliders["L-S"].setValue(self._hsv.l_s)
        self._sliders["U-S"].setValue(self._hsv.u_s)
        self._sliders["L-V"].setValue(self._hsv.l_v)
        self._sliders["U-V"].setValue(self._hsv.u_v)

        opts = QHBoxLayout()
        self._topmost_check = QCheckBox("Always on top")
        self._topmost_check.setChecked(True)
        self._topmost_check.toggled.connect(self._on_topmost)
        opts.addWidget(self._topmost_check)

        self._red_border_check = QCheckBox("Red border (highlight)")
        self._red_border_check.toggled.connect(self._apply_red_border)
        opts.addWidget(self._red_border_check)

        rl.addLayout(opts)

        op_row = QHBoxLayout()
        op_row.addWidget(QLabel("Opacity %"))
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(10, 100)
        self._opacity_slider.valueChanged.connect(self._on_opacity_slider)
        op_row.addWidget(self._opacity_slider)
        rl.addLayout(op_row)

        self._close_app_btn = QPushButton("Close application")
        self._close_app_btn.setToolTip(
            "Exit the app immediately; background capture threads stop without freezing the UI."
        )
        self._close_app_btn.clicked.connect(self.close)
        self._close_app_btn.setStyleSheet(
            "QPushButton { padding: 8px 12px; font-weight: bold; }"
        )
        rl.addWidget(self._close_app_btn)

        right_scroll = QScrollArea()
        right_scroll.setWidget(right_inner)
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        right_scroll.setMinimumWidth(180)
        right_scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        self._splitter.addWidget(left)
        self._splitter.addWidget(right_scroll)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.setHandleWidth(8)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setSizes([520, 420])

        self._update_status_label()

    def _on_tesseract_path_changed(self) -> None:
        self._tesseract_cmd = normalize_tesseract_cmd(self._tesseract_edit.text())
        self._ocr_worker.set_tesseract_cmd(self._tesseract_cmd)
        self._persist()
        if self._tesseract_cmd:
            self._capture_hint.setText(f"Tesseract path set: {self._tesseract_cmd}")
        else:
            self._capture_hint.setText("Tesseract: using default PATH lookup.")

    def _connect_sliders(self) -> None:
        for k, s in self._sliders.items():
            s.valueChanged.connect(self._on_hsv_changed)

    def _on_hsv_changed(self) -> None:
        self._hsv.l_h = self._sliders["L-H"].value()
        self._hsv.u_h = self._sliders["U-H"].value()
        self._hsv.l_s = self._sliders["L-S"].value()
        self._hsv.u_s = self._sliders["U-S"].value()
        self._hsv.l_v = self._sliders["L-V"].value()
        self._hsv.u_v = self._sliders["U-V"].value()
        self._worker.set_hsv(self._hsv)
        self._persist()

    def _on_wheel_frame(self, img: QImage) -> None:
        self._roulette_preview.set_frame_image(img)

    def _on_color_frame(self, img: QImage) -> None:
        if img.isNull():
            return
        self._color_preview.setText("")
        self._color_preview.setPixmap(QPixmap.fromImage(img).scaled(
            self._color_preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def _prepend_history(self, value: int) -> None:
        self._history_numbers.insert(0, value)
        if len(self._history_numbers) > 30:
            self._history_numbers.pop()
        self._history_label.setText("  ".join(str(n) for n in self._history_numbers))

    def _clear_history_display(self) -> None:
        self._history_numbers.clear()
        self._history_label.setText("—")

    def _on_advanced_hsv_toggle(self, on: bool) -> None:
        self._advanced_hsv.setVisible(on)

    def _on_spin(self, value: int) -> None:
        if self._state.state != AppState.COLLECTING_SPINS:
            return
        self._prepend_history(value)
        trained = self._state.on_spin_recorded()
        self._update_status_label()
        if trained:
            self._omega_was_low = True
            self._prediction_result.setText(
                "Prediction: — (shown when ball speed is high after training)"
            )
            self._prediction_result.show()

    def _on_ball_omega(self, omega: float) -> None:
        if self._state.state != AppState.TRAINED:
            return
        hi = omega >= self._HIGH_BALL_SPEED_RAD_S
        if hi and self._omega_was_low:
            self._omega_was_low = False
            sectors = list(range(37))
            guess = random.choice(sectors)
            self._prediction_result.setText(
                f"Prediction: {guess} (placeholder; not a guarantee)"
            )
            self._prediction_result.show()
        elif omega < self._LOW_BALL_SPEED_RAD_S:
            self._omega_was_low = True

    def _update_status_label(self) -> None:
        s = self._state.state
        if s != AppState.TRAINED:
            self._prediction_result.hide()
        if s == AppState.IDLE:
            self._prediction_value.setStyleSheet(_STYLE_PRED_IDLE)
            self._prediction_value.setText("Waiting for Setup")
        elif s in (AppState.SETUP_STEP1_WHEEL, AppState.SETUP_STEP2_PATH, AppState.SETUP_STEP3_OCR):
            self._prediction_value.setStyleSheet(_STYLE_PRED_IDLE)
            self._prediction_value.setText("Calibration in progress…")
        elif s == AppState.COLLECTING_SPINS:
            n = self._state.spin.count
            self._prediction_value.setStyleSheet(_STYLE_PRED_ACTIVE)
            self._prediction_value.setText(f"Analyzing… ({n}/{self._state.spin.max_spins} spins)")
        elif s == AppState.TRAINED:
            self._prediction_value.setStyleSheet(_STYLE_PRED_ACTIVE)
            self._prediction_value.setText("Analysis complete (placeholder prediction)")

    def _start_setup(self) -> None:
        if not self._state.begin_setup():
            QMessageBox.information(self, "Setup", "Finish current mode first.")
            return
        self._update_status_label()
        # Spec: dim main window during wizard so the rotor is visible through the overlay.
        self._apply_opacity(0.3)
        self._worker.set_calibration(None)
        self._worker.set_pick_highlight(None)
        self._ocr_worker.set_calibration(None)
        self._set_previews_blank()
        screen = QGuiApplication.primaryScreen()
        if not screen:
            return
        geo = screen.geometry()
        mss_idx = _mss_monitor_index_for_screen(screen)
        self._overlay = CalibrationOverlay(geo, mss_idx, self)
        self._overlay.committed.connect(self._on_calib_done)
        self._overlay.cancelled.connect(self._on_calib_cancel)
        self._overlay.step_changed.connect(self._show_setup_instruction)
        self._overlay.showFullScreen()

    def _on_calib_done(self, cal: CalibrationData) -> None:
        self._close_setup_instruction()
        self._overlay = None
        screen = QGuiApplication.primaryScreen()
        if screen:
            cal.screen_scale = float(screen.devicePixelRatio())
        self._cal = cal
        self._worker.set_calibration(self._cal)
        self._worker.set_pick_highlight(None)
        self._ball_pick_btn.blockSignals(True)
        self._ball_pick_btn.setChecked(False)
        self._ball_pick_btn.blockSignals(False)
        self._roulette_preview.set_ball_pick_cursor(False)
        self._roulette_preview.end_pick_snapshot()
        self._ocr_worker.set_calibration(self._cal)
        self._ocr_worker.reset_spin_debounce()
        self._state.complete_calibration_from_overlay()
        self._apply_opacity(1.0)
        self._opacity_slider.setValue(100)
        self._persist()
        self._clear_history_display()
        self._prediction_result.hide()
        self._prediction_result.clear()
        self._set_hsv_ball_tune_default()
        self._start_seed_history_retries()
        self._update_status_label()

    def _on_calib_cancel(self) -> None:
        self._close_setup_instruction()
        self._overlay = None
        self._state.cancel_setup()
        self._apply_opacity(self._opacity_slider.value() / 100.0)
        self._update_status_label()

    def _show_setup_instruction(self, text: str) -> None:
        self._close_setup_instruction()
        popup = QDialog(self)
        popup.setWindowTitle("Setup Instruction")
        popup.setWindowModality(Qt.WindowModality.NonModal)
        popup.setMinimumSize(280, 110)
        popup.resize(320, 120)
        layout = QVBoxLayout(popup)
        layout.setContentsMargins(10, 12, 10, 10)
        layout.setSpacing(8)
        body = QFrame(popup)
        body.setStyleSheet("background-color:#101010; border:1px solid #3b3b3b;")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(8, 6, 8, 6)
        msg = QLabel(text, body)
        msg.setWordWrap(True)
        msg.setStyleSheet("color:#f0f0f0; font-size:9pt;")
        body_layout.addWidget(msg)
        layout.addWidget(body)
        popup.show()
        popup.raise_()
        popup.activateWindow()
        center = self.geometry().center()
        popup.move(center.x() - popup.width() // 2, center.y() - popup.height() // 2)
        self._instruction_popup = popup

    def _close_setup_instruction(self) -> None:
        if self._instruction_popup is not None:
            self._instruction_popup.close()
            self._instruction_popup = None

    def _sample_hsv_from_wheel_bgr_roi(
        self,
        arr,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        *,
        tight: bool,
        for_ball_lock: bool = False,
    ) -> None:
        import cv2
        import numpy as np

        roi = arr[y0:y1, x0:x1]
        if roi.size < 12:
            return
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape((-1, 3))
        # Wheel click sample: narrow percentiles + margins so neighbor pocket numbers do not widen the range.
        if for_ball_lock:
            p_lo, p_hi = 35, 65
            h_margin, sv_margin = 4, 10
        elif tight:
            p_lo, p_hi = (12, 88)
            h_margin = 5
            sv_margin = 22
        else:
            p_lo, p_hi = (20, 80)
            h_margin = 8
            sv_margin = 35
        h_low, s_low, v_low = np.percentile(hsv_roi, p_lo, axis=0)
        h_high, s_high, v_high = np.percentile(hsv_roi, p_hi, axis=0)
        self._sliders["L-H"].setValue(max(0, int(h_low) - h_margin))
        self._sliders["U-H"].setValue(min(179, int(h_high) + h_margin))
        self._sliders["L-S"].setValue(max(0, int(s_low) - sv_margin))
        self._sliders["U-S"].setValue(min(255, int(s_high) + sv_margin))
        self._sliders["L-V"].setValue(max(0, int(v_low) - sv_margin))
        self._sliders["U-V"].setValue(min(255, int(v_high) + sv_margin))

    def _start_seed_history_retries(self) -> None:
        self._seed_history_attempt = 0
        if not is_tesseract_available(self._tesseract_cmd):
            self._capture_hint.setText(
                "Tesseract OCR not found. Install it, add to PATH, or set Tesseract executable path above, "
                "then complete setup again."
            )
            return
        self._try_seed_history_once()

    def _try_seed_history_once(self) -> None:
        if not self._cal.ocr_rect:
            return
        cal = self._cal
        tess = self._tesseract_cmd

        def work() -> None:
            num: Optional[int] = None
            err: Optional[str] = None
            try:
                import mss
                import numpy as np

                with mss.mss() as sct:
                    mons = sct.monitors
                    idx = cal.monitor_index if 1 <= cal.monitor_index < len(mons) else 1
                    mon = mons[idx]
                    r = cal.ocr_rect
                    assert r is not None
                    s = cal.pixel_scale()
                    left = int(mon["left"] + r["x"] * s)
                    top = int(mon["top"] + r["y"] * s)
                    w = max(1, int(round(r["w"] * s)))
                    h = max(1, int(round(r["h"] * s)))
                    region = {"left": left, "top": top, "width": w, "height": h}
                    shot = sct.grab(region)
                    bgr = np.asarray(shot)[:, :, :3].copy()
                num = read_roulette_number_from_roi(bgr, tess)
            except Exception as e:
                err = str(e)
            self._seed_ocr_done.emit(num, err)

        threading.Thread(target=work, daemon=True).start()

    def _on_seed_ocr_done(self, num: object, err: object) -> None:
        max_attempts = 40
        if isinstance(num, int) and 0 <= num <= 36:
            self._ocr_worker.seed_spin_debouncer_last_recorded(num)
            self._capture_hint.setText(
                f"Latest number from Step 3 ROI (OCR seed): {num} — will log to history on next spin change."
            )
            return
        if isinstance(err, str) and err:
            self._ocr_debug_label.setText(f"Seed OCR: {err}")
        self._seed_history_attempt += 1
        if self._seed_history_attempt < max_attempts:
            QTimer.singleShot(350, self._try_seed_history_once)
        else:
            self._capture_hint.setText(
                "Could not OCR the latest number into history. Widen Step 3 rectangle or check Tesseract install."
            )

    def _set_previews_blank(self) -> None:
        self._roulette_preview.set_frame_image(QImage())
        self._roulette_preview.set_speed_text("")
        self._color_preview.setPixmap(QPixmap())
        self._color_preview.setText("Waiting for setup completion…")

    def _set_hsv_ball_tune_default(self) -> None:
        """Defaults tuned so the track-ring mask is not one huge blob (L-V rejects dark felt/ink)."""
        self._sliders["L-H"].setValue(0)
        self._sliders["U-H"].setValue(179)
        self._sliders["L-S"].setValue(0)
        self._sliders["U-S"].setValue(200)
        self._sliders["L-V"].setValue(130)
        self._sliders["U-V"].setValue(255)
        self._capture_hint.setText(
            "Open **Color** tab: lower **U-S** so only the ball path shows white (mask the track, not the whole ring). "
            "If numbers still appear, use **Show all HSV** and tighten **L-V** / **U-V**. "
            "Then **Pick white ball** — the window minimizes; click the ball on the **live** wheel inside your path. "
            "White disk + green ring on Roulette Detection."
        )

    def _on_topmost(self, on: bool) -> None:
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, on)
        self.show()

    def _on_opacity_slider(self, v: int) -> None:
        self._apply_opacity(v / 100.0)
        self._persist()

    def _apply_opacity(self, a: float) -> None:
        self.setWindowOpacity(max(0.1, min(1.0, a)))

    def _apply_red_border(self, on: bool) -> None:
        self.setStyleSheet(build_app_stylesheet(red_border=on))
        self._persist()

    def _persist(self) -> None:
        save_config(
            self._cal,
            self._hsv,
            self._red_border_check.isChecked(),
            self._opacity_slider.value() / 100.0,
            self._tesseract_cmd,
        )

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._roulette_preview.rescale_frame()
        pm2 = self._color_preview.pixmap()
        if pm2 is not None and not pm2.isNull():
            self._color_preview.setPixmap(pm2.scaled(
                self._color_preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))

    def _begin_graceful_shutdown(self) -> None:
        """Stop workers without blocking the GUI thread (fixes unresponsive title-bar close)."""
        self._close_setup_instruction()
        if self._ball_pick_overlay is not None:
            self._ball_pick_overlay.close()
            self._ball_pick_overlay = None
        if self._overlay is not None:
            self._overlay.close()
            self._overlay = None
        self._event_pump.stop()
        self._persist()
        self._worker.request_stop()
        self._ocr_worker.request_stop()
        self.hide()
        self._shutdown_deadline = time.monotonic() + 10.0
        QTimer.singleShot(0, self._poll_threads_then_quit)

    def _poll_threads_then_quit(self) -> None:
        if not self._worker.isRunning() and not self._ocr_worker.isRunning():
            QCoreApplication.quit()
            return
        if time.monotonic() >= self._shutdown_deadline:
            QCoreApplication.quit()
            return
        QTimer.singleShot(40, self._poll_threads_then_quit)

    def closeEvent(self, e) -> None:
        if self._shutdown_in_progress:
            e.accept()
            super().closeEvent(e)
            return
        self._shutdown_in_progress = True
        e.ignore()
        self._begin_graceful_shutdown()
