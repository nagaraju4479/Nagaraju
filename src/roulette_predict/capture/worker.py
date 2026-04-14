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
    centroid_in_track_annulus,
    centroid_near_step2_track_mask,
    detect_white_ball,
    path_tube_mask_ball_roi,
    track_region_mask_ball_roi,
    wheel_center_radius_in_ball_roi,
)
from roulette_predict.vision.ocr_spin import SpinDebouncer, read_roulette_number_live_fast
from roulette_predict.vision.speed import SpeedTracker


def _bgr_to_qimage(bgr: np.ndarray) -> QImage:
    if bgr.size == 0:
        return QImage()
    h, w = bgr.shape[:2]
    bgr = np.ascontiguousarray(bgr)
    return QImage(bgr.data, w, h, 3 * w, QImage.Format.Format_BGR888).copy()


def _ocr_frame_fingerprint(bgr: np.ndarray) -> np.ndarray:
    """Tiny grayscale thumbnail for fast frame-to-frame comparison."""
    import cv2

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (24, 24), interpolation=cv2.INTER_AREA)


def _ocr_frame_changed(
    prev_fp: Optional[np.ndarray], curr_fp: np.ndarray, threshold: float = 8.0
) -> bool:
    """True when the OCR rectangle content has visibly changed since last emission."""
    if prev_fp is None:
        return True
    if prev_fp.shape != curr_fp.shape:
        return True
    import cv2

    diff = cv2.absdiff(prev_fp, curr_fp)
    return float(np.mean(diff)) > threshold


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


def _screen_union_rect(
    a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
) -> Tuple[int, int, int, int]:
    """Smallest axis-aligned rect containing both ``(left, top, width, height)`` screen ROIs."""
    al, at, aw, ah = a
    bl, bt, bw, bh = b
    ar, ab = al + aw, at + ah
    br, bb = bl + bw, bt + bh
    ul = min(al, bl)
    ut = min(at, bt)
    ur = max(ar, br)
    ub = max(ab, bb)
    return ul, ut, max(1, ur - ul), max(1, ub - ut)


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


def snapshot_ball_path_bgr_and_geometry(
    cal: CalibrationData,
) -> Optional[Tuple[np.ndarray, int, int, int, int, float, float, float]]:
    """
    Single grab of the full ball-path ROI plus wheel center/radius in ball-ROI pixels.
    Used for live color pick (white-ball search + HSV patch) without racing the vision thread.
    """
    import mss

    with mss.mss() as sct:
        mon = _monitor_for_cal(cal, sct)
        broi = _ball_path_roi_mon(cal, mon)
        if not broi:
            return None
        bl, bt, bw, bh = broi
        ball_bgr = _grab_region(bl, bt, bw, bh)
        cx_b, cy_b, r_px = wheel_center_radius_in_ball_roi(cal, mon, bl, bt, bw, bh)
        return ball_bgr, bl, bt, bw, bh, cx_b, cy_b, r_px


# Max width of wheel image for UI + mask preview (full-res ball ROI still used for tracking).
# Higher = sharper Roulette tab but more CPU per frame.
WHEEL_PREVIEW_MAX_W = 1280


def ball_roi_xy_to_wheel_preview_xy(
    bx: float,
    by: float,
    wheel_pw: int,
    wheel_ph: int,
    ww: int,
    wh: int,
    wl: int,
    wt: int,
    bl: int,
    bt: int,
) -> Tuple[int, int]:
    """Inverse of ``_wheel_preview_xy_to_ball_roi_xy``: ball-path ROI → wheel preview pixels."""
    wx = float(bx) + (bl - wl)
    wy = float(by) + (bt - wt)
    hx = int(max(0, min(wheel_pw - 1, round(wx * wheel_pw / max(1, ww)))))
    hy = int(max(0, min(wheel_ph - 1, round(wy * wheel_ph / max(1, wh)))))
    return hx, hy


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

    _WHEEL_PREVIEW_MAX_W = WHEEL_PREVIEW_MAX_W
    # Target loop period (ms) — ~60 Hz cap; actual FPS depends on CPU + grab size.
    _TARGET_LOOP_MS = 16
    # Run dense Farneback every Nth locked frame to stay near real-time when CPU-bound.
    _FLOW_DECIMATE_STRIDE = 2
    # Flow/HSV centroids can sit a few px off the painted stroke; wider = fewer false drops.
    _TRACK_MASK_NEAR_RADIUS_PX = 8
    # After a lost centroid, keep drawing the ring at the last good position for a few frames.
    _RING_COAST_FRAMES = 32
    # Require this many consecutive misses before switching green → orange (reduces rapid color flicker).
    _ORANGE_RING_AFTER_MISS_FRAMES = 6

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._running = False
        self._cal: Optional[CalibrationData] = None
        self._hsv = HsvSettings()
        self._speed_tracker: Optional[SpeedTracker] = None
        self._wheel_center_in_roi: Optional[Tuple[float, float]] = None
        # Last good ball centroid in ball-path ROI — continuity vs static rim highlights (ball ROI px).
        self._last_ball_cent_roi: Optional[Tuple[float, float]] = None
        # Preprocessed (blurred) grayscale for consecutive Farneback frames.
        self._prev_ball_prep: Optional[np.ndarray] = None
        # Last trusted (flow or HSV) centroid for ω extrapolation when flow mass is weak (not extrapolated).
        self._trusted_ball_cent: Optional[Tuple[float, float]] = None
        self._extrap_frames_used: int = 0
        self._last_track_time: float = 0.0
        self._flow_decimate_tick: int = 0
        self._viz_cent_miss_streak: int = 0
        self._ring_coast_frames_left: int = 0
        self._last_speed_overlay_text: str = ""

    def _emit_speed_overlay(self, text: str) -> None:
        if text != self._last_speed_overlay_text:
            self._last_speed_overlay_text = text
            self.speed_text.emit(text)

    def set_calibration(self, cal: Optional[CalibrationData]) -> None:
        self._cal = cal
        self._speed_tracker = None
        self._wheel_center_in_roi = None
        self._last_ball_cent_roi = None
        self._prev_ball_prep = None
        self._trusted_ball_cent = None
        self._extrap_frames_used = 0
        self._last_track_time = 0.0
        self._flow_decimate_tick = 0
        self._viz_cent_miss_streak = 0
        self._ring_coast_frames_left = 0
        self._last_speed_overlay_text = ""

    def set_hsv(self, hsv: HsvSettings) -> None:
        self._hsv = hsv

    def run(self) -> None:
        import mss

        self._running = True
        sct = mss.mss()
        try:
            while self._running:
                t0 = time.perf_counter()
                cal = self._cal
                if cal and cal.wheel_circle and len(cal.ball_path_points) >= 2:
                    self._process_frame(cal, sct)
                else:
                    self.status_text.emit("Complete calibration to start capture.")
                elapsed = time.perf_counter() - t0
                sleep_ms = max(0, int(self._TARGET_LOOP_MS - elapsed * 1000))
                self.msleep(sleep_ms)
        finally:
            try:
                sct.close()
            except Exception:
                pass

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

    def _process_frame(self, cal: CalibrationData, sct) -> None:
        import cv2

        mon = _monitor_for_cal(cal, sct)
        wroi = _wheel_roi_mon(cal, mon)
        broi = _ball_path_roi_mon(cal, mon)
        if not wroi or not broi:
            return
        wl, wt, ww, wh = wroi
        bl, bt, bw, bh = broi
        self._ensure_trackers(cal, mon, bl, bt)

        # One screen grab for wheel + ball path — same video instant, half the mss overhead vs two grabs.
        ul, ut, uuw, uuh = _screen_union_rect(wroi, broi)
        region = {"left": ul, "top": ut, "width": uuw, "height": uuh}
        shot = sct.grab(region)
        full = np.asarray(shot)
        full_bgr = full[:, :, :3]
        rwl, rwt = wl - ul, wt - ut
        rbl, rbt = bl - ul, bt - ut
        wheel_bgr = np.ascontiguousarray(full_bgr[rwt : rwt + wh, rwl : rwl + ww])
        ball_bgr = np.ascontiguousarray(full_bgr[rbt : rbt + bh, rbl : rbl + bw])

        if ww > self._WHEEL_PREVIEW_MAX_W:
            nh = max(1, int(wh * (self._WHEEL_PREVIEW_MAX_W / ww)))
            wheel_bgr = cv2.resize(
                wheel_bgr,
                (self._WHEEL_PREVIEW_MAX_W, nh),
                interpolation=cv2.INTER_AREA,
            )
        cx_b, cy_b, r_px = wheel_center_radius_in_ball_roi(cal, mon, bl, bt, bw, bh)
        bh, bw = ball_bgr.shape[0], ball_bgr.shape[1]
        tube_mask = path_tube_mask_ball_roi(bh, bw, cal)
        track_geom_mask = track_region_mask_ball_roi(bh, bw, cx_b, cy_b, r_px, tube_mask)

        eff_anchor: Optional[Tuple[float, float]] = None
        eff_anchor_w = 0.0
        if self._last_ball_cent_roi is not None:
            eff_anchor = self._last_ball_cent_roi
            eff_anchor_w = 0.82

        # Fixed-HSV white-ball detector — no slider dependency; ring mask + morphology + circularity.
        cent_hsv, mask_dbg, roi_dbg, det_dbg = detect_white_ball(
            ball_bgr,
            cx_b,
            cy_b,
            r_px,
            tube_mask=tube_mask,
            anchor_xy=eff_anchor,
            anchor_weight=eff_anchor_w,
        )
        t_now = time.perf_counter()
        cent: Optional[Tuple[float, float]] = cent_hsv
        gray = cv2.cvtColor(ball_bgr, cv2.COLOR_BGR2GRAY)
        prep = preprocess_ball_gray(gray)
        flow_roi = annulus_path_tube_mask_ball_roi(bh, bw, cx_b, cy_b, r_px, tube_mask)
        flow_cent: Optional[Tuple[float, float]] = None
        self._flow_decimate_tick += 1
        run_farneback = (
            self._flow_decimate_tick % self._FLOW_DECIMATE_STRIDE == 0
            and self._prev_ball_prep is not None
            and self._prev_ball_prep.shape == prep.shape
        )
        if run_farneback:
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
            self._extrap_frames_used = 0
        elif (
            self._extrap_frames_used < 7
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
        self._last_track_time = t_now

        def _track_point_ok(c: Optional[Tuple[float, float]]) -> bool:
            if c is None or r_px <= 1e-3:
                return False
            if not centroid_in_track_annulus(c[0], c[1], cx_b, cy_b, r_px):
                return False
            if track_geom_mask is not None:
                if not centroid_near_step2_track_mask(
                    c[0],
                    c[1],
                    track_geom_mask,
                    radius_px=self._TRACK_MASK_NEAR_RADIUS_PX,
                ):
                    return False
            return True

        # Flow can land slightly off the painted stroke or fail masks; fall back to HSV centroid.
        if not _track_point_ok(cent):
            cent = cent_hsv if _track_point_ok(cent_hsv) else None

        if cent is not None:
            self._trusted_ball_cent = cent
            self._last_ball_cent_roi = cent
            self._viz_cent_miss_streak = 0
            self._ring_coast_frames_left = self._RING_COAST_FRAMES
        else:
            self._viz_cent_miss_streak += 1
            if self._ring_coast_frames_left > 0:
                self._ring_coast_frames_left -= 1
        omega_out = 0.0
        if cent and self._speed_tracker:
            omega = self._speed_tracker.update(time.perf_counter(), cent[0], cent[1])
            if omega is not None:
                omega_out = float(omega)
                self._emit_speed_overlay(f"Ball speed (exp.): {omega:.2f} rad/s")
            else:
                self._emit_speed_overlay("Ball speed (exp.): …")
        else:
            self._emit_speed_overlay("Ball speed (exp.): …")
        self.ball_omega.emit(omega_out)

        # Ring position: follow centroid; coast at last good position. Debounce green vs orange.
        cent_draw: Optional[Tuple[float, float]] = None
        ring_use_green = True
        if cent is not None:
            cent_draw = cent
        elif self._trusted_ball_cent is not None and self._ring_coast_frames_left > 0:
            cent_draw = self._trusted_ball_cent
            ring_use_green = self._viz_cent_miss_streak < self._ORANGE_RING_AFTER_MISS_FRAMES

        sx = wheel_bgr.shape[1] / float(ww)
        sy = wheel_bgr.shape[0] / float(wh)
        wheel_out = wheel_bgr.copy()
        if cent_draw is not None:
            bx = int((bl + cent_draw[0] - wl) * sx)
            by = int((bt + cent_draw[1] - wt) * sy)
            if 0 <= bx < wheel_out.shape[1] and 0 <= by < wheel_out.shape[0]:
                r_ball = max(8, min(18, wheel_out.shape[0] // 45))
                edge = (0, 255, 0) if ring_use_green else (0, 200, 255)
                cv2.circle(wheel_out, (bx, by), r_ball, edge, 3)

        # Color Detection tab: mask (left) | ROI (right) side-by-side debug composite.
        target_h = wheel_out.shape[0]
        mask_h, mask_w = mask_dbg.shape[:2]
        roi_h, roi_w = roi_dbg.shape[:2]
        mw = max(1, int(mask_w * target_h / max(1, mask_h)))
        rw = max(1, int(roi_w * target_h / max(1, roi_h)))
        m_resized = cv2.resize(mask_dbg, (mw, target_h), interpolation=cv2.INTER_NEAREST)
        r_resized = cv2.resize(roi_dbg, (rw, target_h), interpolation=cv2.INTER_AREA)
        color_out = np.hstack([m_resized, r_resized])

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
        self._last_emit_fp: Optional[np.ndarray] = None

    def set_tesseract_cmd(self, cmd: Optional[str]) -> None:
        if not cmd or not str(cmd).strip():
            self._tesseract_cmd = None
        else:
            self._tesseract_cmd = str(cmd).strip().strip('"').strip("'")

    def set_calibration(self, cal: Optional[CalibrationData]) -> None:
        self._cal = cal

    def reset_spin_debounce(self) -> None:
        self._debouncer = SpinDebouncer()
        self._last_emit_fp = None

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
                        fp = _ocr_frame_fingerprint(ocr_bgr)
                        if _ocr_frame_changed(self._last_emit_fp, fp):
                            raw_txt, _ = read_roulette_number_live_fast(ocr_bgr, self._tesseract_cmd)
                            self.ocr_debug.emit(raw_txt)
                            spin = self._debouncer.feed(raw_txt)
                            if spin is not None:
                                self._last_emit_fp = fp
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
