"""Background capture and vision processing (QThread)."""

from __future__ import annotations

import math
import time
from typing import Optional, Tuple

import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from roulette_predict.config_model import CalibrationData, HsvSettings
from roulette_predict.vision.ball_flow import (
    extrapolate_ball_on_wheel_orbit,
    preprocess_ball_gray,
    track_centroid_farneback_path_tube,
)
from roulette_predict.vision.ball_track import (
    BALL_PATH_ROI_MARGIN,
    annulus_path_tube_mask_ball_roi,
    ball_path_hsv_preview_mask,
    ball_track_mask_and_centroid,
    mask_keep_blob_at_tracked_centroid,
    path_tube_mask_ball_roi,
    wheel_center_radius_in_ball_roi,
)
from roulette_predict.vision.hsv_mask import apply_hsv_mask_bgr
from roulette_predict.vision.ocr_spin import SpinDebouncer, read_roulette_number_live_fast
from roulette_predict.vision.speed import SpeedTracker


def _bgr_to_qimage(bgr: np.ndarray) -> QImage:
    if bgr.size == 0:
        return QImage()
    h, w = bgr.shape[:2]
    bgr = np.ascontiguousarray(bgr)
    return QImage(bgr.data, w, h, 3 * w, QImage.Format.Format_BGR888).copy()


def _mss_monitor(monitor_index: int) -> dict:
    import mss

    with mss.mss() as sct:
        mons = sct.monitors
        # mss: 0 = virtual full desktop; 1..N = physical monitors
        idx = monitor_index if 1 <= monitor_index < len(mons) else 1
        return dict(mons[idx])


def _grab_region(left: int, top: int, width: int, height: int) -> np.ndarray:
    import mss

    if width < 1 or height < 1:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    with mss.mss() as sct:
        region = {"left": left, "top": top, "width": width, "height": height}
        shot = sct.grab(region)
        arr = np.asarray(shot)
        return arr[:, :, :3].copy()


def _monitor_for_cal(cal: CalibrationData, sct) -> dict:
    mons = sct.monitors
    idx = cal.monitor_index if 1 <= cal.monitor_index < len(mons) else 1
    return dict(mons[idx])


def _wheel_roi_mon(cal: CalibrationData, mon: dict) -> Optional[Tuple[int, int, int, int]]:
    if not cal.wheel_circle:
        return None
    s = cal.pixel_scale()
    c = cal.wheel_circle
    cx, cy, r = c["cx"], c["cy"], c["r"]
    pad = 6
    left = int(mon["left"] + (cx - r - pad) * s)
    top = int(mon["top"] + (cy - r - pad) * s)
    w = int((2 * r + 2 * pad) * s)
    h = int((2 * r + 2 * pad) * s)
    return left, top, w, h


def _ball_path_roi_mon(cal: CalibrationData, mon: dict) -> Optional[Tuple[int, int, int, int]]:
    if len(cal.ball_path_points) < 2:
        return None
    s = cal.pixel_scale()
    xs = [p[0] for p in cal.ball_path_points]
    ys = [p[1] for p in cal.ball_path_points]
    margin = BALL_PATH_ROI_MARGIN  # include brush-width margin from calibration path
    min_x = (min(xs) - margin) * s
    min_y = (min(ys) - margin) * s
    max_x = (max(xs) + margin) * s
    max_y = (max(ys) + margin) * s
    left = int(mon["left"] + min_x)
    top = int(mon["top"] + min_y)
    w = int(max(1, max_x - min_x))
    h = int(max(1, max_y - min_y))
    return left, top, w, h


def _ocr_roi_mon(cal: CalibrationData, mon: dict) -> Optional[Tuple[int, int, int, int]]:
    if not cal.ocr_rect:
        return None
    s = cal.pixel_scale()
    r = cal.ocr_rect
    left = int(mon["left"] + r["x"] * s)
    top = int(mon["top"] + r["y"] * s)
    w = int(max(1, round(r["w"] * s)))
    h = int(max(1, round(r["h"] * s)))
    return left, top, w, h


def wheel_roi(cal: CalibrationData) -> Optional[Tuple[int, int, int, int]]:
    mon = _mss_monitor(cal.monitor_index)
    return _wheel_roi_mon(cal, mon)


def ball_path_roi(cal: CalibrationData) -> Optional[Tuple[int, int, int, int]]:
    mon = _mss_monitor(cal.monitor_index)
    return _ball_path_roi_mon(cal, mon)


def ocr_roi(cal: CalibrationData) -> Optional[Tuple[int, int, int, int]]:
    mon = _mss_monitor(cal.monitor_index)
    return _ocr_roi_mon(cal, mon)


def _wheel_preview_xy_to_ball_roi_xy(
    hx: int,
    hy: int,
    wheel_pw: int,
    wheel_ph: int,
    ww: int,
    wh: int,
    wl: int,
    wt: int,
    bl: int,
    bt: int,
    bw: int,
    bh: int,
) -> Optional[Tuple[int, int]]:
    """Map Roulette tab pixel (hx,hy) to ball-path grab coords, or None if outside ball ROI."""
    wx = int(round(hx * ww / max(1, wheel_pw)))
    wy = int(round(hy * wh / max(1, wheel_ph)))
    bx = wx - (bl - wl)
    by = wy - (bt - wt)
    if 0 <= bx < bw and 0 <= by < bh:
        return bx, by
    return None


class VisionWorker(QThread):
    """Capture wheel + ball path + speed + color mask. OCR runs on a separate thread."""

    frame_wheel = Signal(QImage)
    frame_color = Signal(QImage)
    speed_text = Signal(str)
    status_text = Signal(str)
    ball_omega = Signal(float)

    _WHEEL_PREVIEW_MAX_W = 960

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._running = False
        self._cal: Optional[CalibrationData] = None
        self._hsv = HsvSettings()
        self._speed_tracker: Optional[SpeedTracker] = None
        self._wheel_center_in_roi: Optional[Tuple[float, float]] = None
        self._last_wheel_emit_t: float = 0.0
        self._pick_highlight_xy: Optional[Tuple[int, int]] = None
        # Ball-path tracking uses this when set; wheel mask still uses `_hsv` (so U-S can stay low).
        self._ball_hsv_override: Optional[HsvSettings] = None
        # Last good ball centroid in ball-path ROI — continuity vs static rim highlights (ball ROI px).
        self._last_ball_cent_roi: Optional[Tuple[float, float]] = None
        # Preprocessed (blurred) grayscale for consecutive Farneback frames.
        self._prev_ball_prep: Optional[np.ndarray] = None
        # Last trusted (flow or HSV) centroid for ω extrapolation when flow mass is weak (not extrapolated).
        self._trusted_ball_cent: Optional[Tuple[float, float]] = None
        self._extrap_frames_used: int = 0
        self._last_track_time: float = 0.0

    def set_ball_hsv_override(self, hsv: Optional[HsvSettings]) -> None:
        """Lock ball-track color range (e.g. from wheel click sample). None = use same as wheel sliders."""
        self._ball_hsv_override = hsv
        if hsv is None:
            self._last_ball_cent_roi = None
            self._prev_ball_prep = None
            self._trusted_ball_cent = None
            self._extrap_frames_used = 0
            self._last_track_time = 0.0

    def set_pick_highlight(self, xy: Optional[Tuple[int, int]]) -> None:
        """Wheel-preview pixel coords (same as emitted QImage). Cyan ring on wheel + mask."""
        self._pick_highlight_xy = xy

    def set_calibration(self, cal: Optional[CalibrationData]) -> None:
        self._cal = cal
        self._speed_tracker = None
        self._wheel_center_in_roi = None
        self._ball_hsv_override = None
        self._last_ball_cent_roi = None
        self._prev_ball_prep = None
        self._trusted_ball_cent = None
        self._extrap_frames_used = 0
        self._last_track_time = 0.0

    def set_hsv(self, hsv: HsvSettings) -> None:
        self._hsv = hsv

    def run(self) -> None:
        self._running = True
        while self._running:
            t0 = time.perf_counter()
            cal = self._cal
            if cal and cal.wheel_circle and len(cal.ball_path_points) >= 2:
                self._process_frame(cal)
            else:
                self.status_text.emit("Complete calibration to start capture.")
            elapsed = time.perf_counter() - t0
            sleep_ms = max(0, int(33 - elapsed * 1000))  # ~30 fps (OCR not on this thread)
            self.msleep(sleep_ms)

    def request_stop(self) -> None:
        """Signal the capture loop to exit without blocking the GUI thread."""
        self._running = False

    def stop(self) -> None:
        self.request_stop()
        self.wait(5000)

    def _ensure_trackers(self, cal: CalibrationData, mon: dict, bl: int, bt: int) -> None:
        if self._speed_tracker is None and cal.wheel_circle:
            s = cal.pixel_scale()
            wc = cal.wheel_circle
            wcx = mon["left"] + wc["cx"] * s
            wcy = mon["top"] + wc["cy"] * s
            self._wheel_center_in_roi = (wcx - bl, wcy - bt)
            self._speed_tracker = SpeedTracker.new(self._wheel_center_in_roi[0], self._wheel_center_in_roi[1])

    def _process_frame(self, cal: CalibrationData) -> None:
        import cv2
        import mss

        def grab(sct, left: int, top: int, width: int, height: int) -> np.ndarray:
            if width < 1 or height < 1:
                return np.zeros((1, 1, 3), dtype=np.uint8)
            region = {"left": left, "top": top, "width": width, "height": height}
            shot = sct.grab(region)
            return np.asarray(shot)[:, :, :3].copy()

        with mss.mss() as sct:
            mon = _monitor_for_cal(cal, sct)
            wroi = _wheel_roi_mon(cal, mon)
            broi = _ball_path_roi_mon(cal, mon)
            if not wroi or not broi:
                return
            wl, wt, ww, wh = wroi
            bl, bt, bw, bh = broi
            self._ensure_trackers(cal, mon, bl, bt)

            wheel_bgr = grab(sct, wl, wt, ww, wh)
            if ww > self._WHEEL_PREVIEW_MAX_W:
                nh = max(1, int(wh * (self._WHEEL_PREVIEW_MAX_W / ww)))
                wheel_bgr = cv2.resize(
                    wheel_bgr,
                    (self._WHEEL_PREVIEW_MAX_W, nh),
                    interpolation=cv2.INTER_AREA,
                )
            mask, masked = apply_hsv_mask_bgr(
                wheel_bgr,
                self._hsv.l_h,
                self._hsv.u_h,
                self._hsv.l_s,
                self._hsv.u_s,
                self._hsv.l_v,
                self._hsv.u_v,
            )
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            ball_bgr = grab(sct, bl, bt, bw, bh)

        cx_b, cy_b, r_px = wheel_center_radius_in_ball_roi(cal, mon, bl, bt, bw, bh)
        tune_lower = np.array([self._hsv.l_h, self._hsv.l_s, self._hsv.l_v], dtype=np.uint8)
        tune_upper = np.array([self._hsv.u_h, self._hsv.u_s, self._hsv.u_v], dtype=np.uint8)
        ball_locked = self._ball_hsv_override is not None
        bh, bw = ball_bgr.shape[0], ball_bgr.shape[1]
        tube_mask = path_tube_mask_ball_roi(bh, bw, cal)
        anchor_ball_roi: Optional[Tuple[float, float]] = None

        if not ball_locked:
            # Tuning: darken the track with sliders only — no centroid, no auto-white speckles.
            bmask_display = ball_path_hsv_preview_mask(
                ball_bgr, cx_b, cy_b, r_px, tune_lower, tune_upper, tube_mask=tube_mask
            )
            cent = None
            self.speed_text.emit("Tune U-S / HSV on Color tab, then Pick white ball — tracking starts after pick.")
            self.ball_omega.emit(0.0)
        else:
            ball_cfg = self._ball_hsv_override
            assert ball_cfg is not None
            lower = np.array([ball_cfg.l_h, ball_cfg.l_s, ball_cfg.l_v], dtype=np.uint8)
            upper = np.array([ball_cfg.u_h, ball_cfg.u_s, ball_cfg.u_v], dtype=np.uint8)
            if self._pick_highlight_xy is not None:
                hx, hy = self._pick_highlight_xy
                mapped = _wheel_preview_xy_to_ball_roi_xy(
                    hx,
                    hy,
                    wheel_bgr.shape[1],
                    wheel_bgr.shape[0],
                    ww,
                    wh,
                    wl,
                    wt,
                    bl,
                    bt,
                    bw,
                    bh,
                )
                if mapped is not None:
                    anchor_ball_roi = (float(mapped[0]), float(mapped[1]))

            eff_anchor: Optional[Tuple[float, float]] = None
            eff_anchor_w = 0.0
            if self._last_ball_cent_roi is not None:
                eff_anchor = self._last_ball_cent_roi
                eff_anchor_w = 0.62
            elif anchor_ball_roi is not None:
                eff_anchor = anchor_ball_roi
                eff_anchor_w = 0.78

            # Locked HSV mask — baseline centroid; dense optical flow fuses for fast spin.
            bmask_track, cent_hsv, _auto_ball = ball_track_mask_and_centroid(
                ball_bgr,
                cx_b,
                cy_b,
                r_px,
                lower,
                upper,
                prefer_manual=True,
                anchor_ball_xy=eff_anchor,
                anchor_weight=eff_anchor_w,
                allow_auto_fallback=False,
                tube_mask=tube_mask,
            )
            t_now = time.perf_counter()
            cent: Optional[Tuple[float, float]] = cent_hsv
            gray = cv2.cvtColor(ball_bgr, cv2.COLOR_BGR2GRAY)
            prep = preprocess_ball_gray(gray)
            flow_roi = annulus_path_tube_mask_ball_roi(bh, bw, cx_b, cy_b, r_px, tube_mask)
            flow_cent: Optional[Tuple[float, float]] = None
            if self._prev_ball_prep is not None and self._prev_ball_prep.shape == prep.shape:
                flow_cent = track_centroid_farneback_path_tube(
                    self._prev_ball_prep,
                    prep,
                    cx_b,
                    cy_b,
                    flow_roi,
                )
            self._prev_ball_prep = prep.copy()

            if flow_cent is not None:
                cent = flow_cent
                self._trusted_ball_cent = flow_cent
                self._extrap_frames_used = 0
            elif (
                self._extrap_frames_used < 3
                and self._trusted_ball_cent is not None
                and self._speed_tracker is not None
                and abs(self._speed_tracker.last_signed_omega_rad_s) > 1e-5
            ):
                if self._last_track_time <= 0:
                    dt = 1.0 / 30.0
                else:
                    dt = min(max(t_now - self._last_track_time, 1e-4), 0.25)
                cent = extrapolate_ball_on_wheel_orbit(
                    cx_b,
                    cy_b,
                    self._trusted_ball_cent[0],
                    self._trusted_ball_cent[1],
                    self._speed_tracker.last_signed_omega_rad_s,
                    dt,
                )
                self._extrap_frames_used += 1
            else:
                cent = cent_hsv
                self._extrap_frames_used = 0
                if cent_hsv is not None:
                    self._trusted_ball_cent = cent_hsv
            self._last_track_time = t_now

            if cent is not None:
                self._last_ball_cent_roi = cent
                ix = int(np.clip(round(cent[0]), 0, bw - 1))
                iy = int(np.clip(round(cent[1]), 0, bh - 1))
                mask_seed = cent
                if int(bmask_track[iy, ix]) < 128 and cent_hsv is not None:
                    mask_seed = cent_hsv
                bmask_display = mask_keep_blob_at_tracked_centroid(bmask_track, mask_seed)
            else:
                bmask_display = np.zeros((bh, bw), dtype=np.uint8)
            # Loose guard: ball orbits inside ~r; strict 1.02*r rejected valid rim positions when r was tight.
            if cent is not None and r_px > 1e-3:
                radial = math.hypot(cent[0] - cx_b, cent[1] - cy_b)
                if radial > float(r_px) * 1.12:
                    cent = None
            omega_out = 0.0
            if cent and self._speed_tracker:
                omega = self._speed_tracker.update(time.perf_counter(), cent[0], cent[1])
                if omega is not None:
                    omega_out = float(omega)
                    self.speed_text.emit(f"Ball speed (exp.): {omega:.2f} rad/s")
                else:
                    self.speed_text.emit("Ball speed (exp.): …")
            else:
                self.speed_text.emit("Ball speed (exp.): …")
            self.ball_omega.emit(omega_out)

        # Show white+green dot at tracked centroid, or at pick projection if mask has not locked yet.
        cent_draw = cent
        if ball_locked and cent_draw is None and anchor_ball_roi is not None:
            cent_draw = anchor_ball_roi

        sx = wheel_bgr.shape[1] / float(ww)
        sy = wheel_bgr.shape[0] / float(wh)
        wheel_out = wheel_bgr.copy()
        # Wheel-tab mask preview (path + ball in range); Color tab uses ball-path ROI only below.
        mask_wheel = mask_bgr.copy()
        if cent_draw is not None:
            bx = int((bl + cent_draw[0] - wl) * sx)
            by = int((bt + cent_draw[1] - wt) * sy)
            if 0 <= bx < wheel_out.shape[1] and 0 <= by < wheel_out.shape[0]:
                r_ball = max(8, min(18, wheel_out.shape[0] // 45))
                fill = (220, 230, 255) if cent is None else (255, 255, 255)
                edge = (0, 200, 255) if cent is None else (0, 255, 0)
                cv2.circle(wheel_out, (bx, by), r_ball, fill, -1)
                cv2.circle(wheel_out, (bx, by), r_ball, edge, 2)
                cv2.circle(mask_wheel, (bx, by), 16, edge, 2)
        if self._pick_highlight_xy is not None:
            hx, hy = self._pick_highlight_xy
            if 0 <= hx < wheel_out.shape[1] and 0 <= hy < wheel_out.shape[0]:
                cv2.circle(wheel_out, (hx, hy), 12, (0, 255, 255), 2)
                cv2.circle(mask_wheel, (hx, hy), 12, (0, 255, 255), 2)

        # Color Detection: tuning mask before pick; after pick, locked HSV with only the tracked blob.
        ball_vis = cv2.cvtColor(bmask_display, cv2.COLOR_GRAY2BGR)
        color_h = wheel_out.shape[0]
        color_w = max(1, int(bw * color_h / max(1, bh)))
        color_out = cv2.resize(ball_vis, (color_w, color_h), interpolation=cv2.INTER_NEAREST)
        if cent_draw is not None and bw > 0 and bh > 0:
            cx = int(cent_draw[0] * color_w / bw)
            cy = int(cent_draw[1] * color_h / bh)
            ccol = (0, 200, 255) if cent is None else (0, 255, 0)
            cv2.circle(color_out, (cx, cy), max(8, min(20, color_h // 12)), ccol, 2)
        if self._pick_highlight_xy is not None:
            hx, hy = self._pick_highlight_xy
            mapped = _wheel_preview_xy_to_ball_roi_xy(
                hx, hy, wheel_out.shape[1], wheel_out.shape[0], ww, wh, wl, wt, bl, bt, bw, bh
            )
            if mapped is not None:
                px = int(mapped[0] * color_w / bw)
                py = int(mapped[1] * color_h / bh)
                cv2.circle(color_out, (px, py), max(6, min(16, color_h // 14)), (0, 255, 255), 2)

        now = time.perf_counter()
        if now - self._last_wheel_emit_t >= 1.0 / 30.0:
            self._last_wheel_emit_t = now
            self.frame_color.emit(_bgr_to_qimage(color_out))
            self.frame_wheel.emit(_bgr_to_qimage(wheel_out))


class OcrSpinWorker(QThread):
    """Step-3 ROI only: Tesseract runs here so wheel preview stays smooth."""

    ocr_debug = Signal(str)
    spin_detected = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._running = False
        self._cal: Optional[CalibrationData] = None
        self._debouncer = SpinDebouncer()
        self._tesseract_cmd: Optional[str] = None

    def set_tesseract_cmd(self, cmd: Optional[str]) -> None:
        if not cmd or not str(cmd).strip():
            self._tesseract_cmd = None
        else:
            self._tesseract_cmd = str(cmd).strip().strip('"').strip("'")

    def set_calibration(self, cal: Optional[CalibrationData]) -> None:
        self._cal = cal

    def reset_spin_debounce(self) -> None:
        self._debouncer = SpinDebouncer()

    def seed_spin_debouncer_last_recorded(self, value: int) -> None:
        if 0 <= value <= 36:
            self._debouncer.last_recorded = value

    def run(self) -> None:
        import mss

        self._running = True
        while self._running:
            t0 = time.perf_counter()
            cal = self._cal
            if cal and cal.ocr_rect:
                try:
                    with mss.mss() as sct:
                        mon = _monitor_for_cal(cal, sct)
                        oroi = _ocr_roi_mon(cal, mon)
                        if oroi:
                            ol, ot, ow, oh = oroi
                            region = {"left": ol, "top": ot, "width": ow, "height": oh}
                            shot = sct.grab(region)
                            ocr_bgr = np.asarray(shot)[:, :, :3].copy()
                        else:
                            ocr_bgr = None
                    if ocr_bgr is not None and ocr_bgr.size > 0:
                        raw_txt, _ = read_roulette_number_live_fast(ocr_bgr, self._tesseract_cmd)
                        self.ocr_debug.emit(raw_txt)
                        spin = self._debouncer.feed(raw_txt)
                        if spin is not None:
                            self.spin_detected.emit(spin)
                except Exception as e:
                    self.ocr_debug.emit(f"OCR error: {e}")
            else:
                self.msleep(80)
                continue

            elapsed = time.perf_counter() - t0
            # Poll Step-3 ROI as fast as Tesseract allows so debouncer can confirm each new digit twice.
            sleep_ms = max(0, int(8 - elapsed * 1000))
            self.msleep(sleep_ms)

    def request_stop(self) -> None:
        """Signal the OCR loop to exit without blocking the GUI thread."""
        self._running = False

    def stop(self) -> None:
        self.request_stop()
        self.wait(5000)
