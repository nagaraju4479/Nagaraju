"""Automatic white-ball detection in the wheel track (annulus around calibrated wheel circle)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from roulette_predict.config_model import CalibrationData

# Padding around painted path bbox when building ball-path ROI grab — must match capture/worker.
BALL_PATH_ROI_MARGIN = 28


def _hsv_lo_hi_for_inrange(
    hsv_lower: np.ndarray, hsv_upper: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV ``inRange`` is empty if any lower > upper; sampling can invert a channel."""
    lo = np.minimum(hsv_lower.astype(np.int32), hsv_upper.astype(np.int32))
    hi = np.maximum(hsv_lower.astype(np.int32), hsv_upper.astype(np.int32))
    lo = np.clip(lo, [0, 0, 0], [179, 255, 255])
    hi = np.clip(hi, [0, 0, 0], [179, 255, 255])
    return lo.astype(np.uint8), hi.astype(np.uint8)


def ball_radius_px_min_max(wheel_r_px: float, h: int, w: int) -> tuple[float, float]:
    """
    Expected ball radius in pixels from calibrated wheel radius (ball is small vs track radius).

    Used to reject rim streaks / specular arcs (large aspect ratio, wrong area) in locked tracking.
    """
    dim = float(min(h, w))
    r_lo = float(max(2.5, min(wheel_r_px * 0.011, dim * 0.017)))
    r_hi = float(min(dim * 0.125, max(r_lo + 5.0, wheel_r_px * 0.095)))
    if r_hi < r_lo + 2.0:
        r_hi = r_lo + 8.0
    return r_lo, r_hi


def _contour_passes_ball_geometry(
    *,
    a: float,
    circ: float,
    extent: float,
    solidity: float,
    bw_: int,
    bh_: int,
    r_enc: float,
    wheel_r_px: float,
    h: int,
    w: int,
) -> bool:
    """Reject elongated rim highlights / arcs; keep compact round blobs near expected ball size."""
    r_lo, r_hi = ball_radius_px_min_max(wheel_r_px, h, w)
    r_eq = float(np.sqrt(max(a, 1e-6) / np.pi))
    if r_eq < r_lo * 0.82 or r_eq > r_hi * 1.18:
        return False
    if r_enc < r_lo * 0.78 or r_enc > r_hi * 1.22:
        return False
    asp = max(bw_, bh_) / float(max(1, min(bw_, bh_)))
    if asp > 1.58:
        return False
    if circ < 0.45:
        return False
    if solidity < 0.68:
        return False
    if extent < 0.30 or extent > 0.99:
        return False
    fill = a / (np.pi * r_enc * r_enc + 1e-6)
    if fill < 0.50:
        return False
    return True


def path_tube_mask_ball_roi(h: int, w: int, cal: CalibrationData) -> Optional[np.ndarray]:
    """
    Pixels within a thick stroke along **Step 2** painted ball path (ball ROI coords).

    Intersecting the annulus with this mask drops most of the broad "golden circle" wood that
    shares ball-like brightness — only the channel you painted (where the ball rolls) stays eligible.
    """
    import cv2

    pts = cal.ball_path_points
    if len(pts) < 2 or h < 2 or w < 2:
        return None
    s = cal.pixel_scale()
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    min_x = (min(xs) - BALL_PATH_ROI_MARGIN) * s
    min_y = (min(ys) - BALL_PATH_ROI_MARGIN) * s
    roi_pts: list[list[int]] = []
    for p in pts:
        bx = int(round(p[0] * s - min_x))
        by = int(round(p[1] * s - min_y))
        roi_pts.append([max(0, min(w - 1, bx)), max(0, min(h - 1, by))])
    m = np.zeros((h, w), dtype=np.uint8)
    arr = np.array(roi_pts, dtype=np.int32).reshape((-1, 1, 2))
    # Stroke width ~ painted brush + margin so the ball stays inside the tube as it moves.
    thickness = max(16, min(52, int(min(h, w) * 0.052)))
    cv2.polylines(m, [arr], isClosed=False, color=255, thickness=thickness, lineType=cv2.LINE_AA)
    return m


def _combine_annulus_and_path_tube(ann: np.ndarray, tube: Optional[np.ndarray]) -> np.ndarray:
    """Annulus ∩ tube; if intersection would be empty (misaligned calibration), keep annulus."""
    import cv2

    if tube is None or tube.shape != ann.shape:
        return ann
    combo = cv2.bitwise_and(ann, tube)
    if np.count_nonzero(combo) == 0:
        return ann
    return combo


def step1_wheel_disk_mask(h: int, w: int, wheel_cx: float, wheel_cy: float, wheel_r_px: float) -> np.ndarray:
    """Filled disk from Step 1 (cx, cy, r) — drops path/tube pixels drawn outside the green wheel circle."""
    import cv2

    m = np.zeros((h, w), dtype=np.uint8)
    # Use calibrated radius; only shrink so the filled circle stays inside the ROI.
    max_r = max(
        0.0,
        min(
            float(wheel_r_px),
            float(wheel_cx),
            float(wheel_cy),
            float(w - 1) - float(wheel_cx),
            float(h - 1) - float(wheel_cy),
        ),
    )
    rr = int(max(4, round(max_r)))
    cv2.circle(m, (int(round(wheel_cx)), int(round(wheel_cy))), rr, 255, thickness=-1)
    return m


def _region_annulus_tube_disk(
    h: int,
    w: int,
    wheel_cx: float,
    wheel_cy: float,
    wheel_r_px: float,
    tube_mask: Optional[np.ndarray],
) -> np.ndarray:
    """Annulus ∩ tube ∩ Step-1 disk — tracking and mask stay inside the calibrated wheel."""
    import cv2

    ann = track_annulus_mask_from_wheel_radius(h, w, wheel_cx, wheel_cy, wheel_r_px)
    region = _combine_annulus_and_path_tube(ann, tube_mask)
    disk = step1_wheel_disk_mask(h, w, wheel_cx, wheel_cy, wheel_r_px)
    region = cv2.bitwise_and(region, disk)
    if np.count_nonzero(region) == 0:
        return cv2.bitwise_and(ann, disk)
    return region


def track_region_mask_ball_roi(
    h: int,
    w: int,
    wheel_cx: float,
    wheel_cy: float,
    wheel_r_px: float,
    tube_mask: Optional[np.ndarray],
) -> np.ndarray:
    """Annulus ∩ path tube ∩ Step-1 disk — same geometry as ``ball_track_mask_and_centroid``."""
    return _region_annulus_tube_disk(h, w, wheel_cx, wheel_cy, wheel_r_px, tube_mask)


def annulus_path_tube_mask_ball_roi(
    h: int,
    w: int,
    wheel_cx: float,
    wheel_cy: float,
    wheel_r_px: float,
    tube_mask: Optional[np.ndarray],
) -> np.ndarray:
    """Annulus ∩ Step-2 path tube only (no extra disk) — region for dense optical flow."""
    ann = track_annulus_mask_from_wheel_radius(h, w, wheel_cx, wheel_cy, wheel_r_px)
    return _combine_annulus_and_path_tube(ann, tube_mask)


def wheel_center_radius_in_ball_roi(
    cal: CalibrationData,
    mon: dict,
    bl: int,
    bt: int,
    bw: int,
    bh: int,
) -> Tuple[float, float, float]:
    """Wheel center and radius in **ball-path grab** pixel coordinates."""
    s = cal.pixel_scale()
    wc = cal.wheel_circle
    if not wc:
        return bw * 0.5, bh * 0.5, min(bw, bh) * 0.45
    cx = mon["left"] + wc["cx"] * s - bl
    cy = mon["top"] + wc["cy"] * s - bt
    r = float(wc["r"]) * s
    if not (0 <= cx < bw and 0 <= cy < bh):
        cx = float(bw) * 0.5
        cy = float(bh) * 0.5
        r = min(bw, bh) * 0.45
    return cx, cy, r


def track_annulus_mask_from_wheel_radius(h: int, w: int, wheel_cx: float, wheel_cy: float, wheel_r_px: float) -> np.ndarray:
    """Ball-orbit band in ball-path ROI pixels (same geometry as tracking)."""
    r = max(float(wheel_r_px), 8.0)
    r_in = max(5.0, r * 0.24)
    r_out = r * 0.995
    return _annulus_mask(h, w, wheel_cx, wheel_cy, r_in, r_out)


def _annulus_mask(h: int, w: int, cx: float, cy: float, r_in: float, r_out: float) -> np.ndarray:
    import cv2

    r_out_i = int(max(1, min(r_out, float(np.hypot(w, h)))))
    r_in_i = int(max(0, min(r_in, r_out_i - 2)))
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(m, (int(round(cx)), int(round(cy))), r_out_i, 255, thickness=-1)
    if r_in_i > 0:
        inner = np.zeros_like(m)
        cv2.circle(inner, (int(round(cx)), int(round(cy))), r_in_i, 255, thickness=-1)
        m = cv2.subtract(m, inner)
    return m


def _white_ball_mask_bgr(bgr: np.ndarray) -> np.ndarray:
    """Ivory / white ball: high value, limited saturation (avoids golden speculars a bit)."""
    import cv2

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array([0, 0, 175], dtype=np.uint8), np.array([179, 105, 255], dtype=np.uint8))


def ball_path_hsv_preview_mask(
    ball_bgr: np.ndarray,
    wheel_cx: float,
    wheel_cy: float,
    wheel_r_px: float,
    hsv_lower: np.ndarray,
    hsv_upper: np.ndarray,
    tube_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    (Annulus ∩ optional path tube) ∩ HSV for **tuning only** (before ball color pick).

    ``tube_mask`` comes from Step 2 paint — it removes the broad golden ring outside the ball channel.
    """
    import cv2

    if ball_bgr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    h, w = ball_bgr.shape[:2]
    region = _region_annulus_tube_disk(h, w, wheel_cx, wheel_cy, wheel_r_px, tube_mask)
    hsv = cv2.cvtColor(ball_bgr, cv2.COLOR_BGR2HSV)
    lo, hi = _hsv_lo_hi_for_inrange(hsv_lower, hsv_upper)
    full = cv2.inRange(hsv, lo, hi)
    out = cv2.bitwise_and(full, region)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k, iterations=1)
    return out


def mask_keep_blob_at_tracked_centroid(
    mask: np.ndarray,
    cent: Tuple[float, float],
) -> np.ndarray:
    """
    Keep only the connected component that contains the tracked centroid (or nearest blob if the
    point falls on background). Used for Color-tab display after pick so reflections / extra HSV
    hits elsewhere show as black — only the ball blob stays white.
    """
    import cv2

    if mask.size == 0:
        return mask
    h, w = mask.shape[:2]
    m = ((mask > 127).astype(np.uint8)) * 255
    if not np.any(m):
        return np.zeros_like(mask)
    num, labels, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num < 2:
        return np.zeros_like(mask)
    ix = int(round(cent[0]))
    iy = int(round(cent[1]))
    ix = max(0, min(w - 1, ix))
    iy = max(0, min(h - 1, iy))
    lbl = int(labels[iy, ix])
    if lbl != 0:
        out = np.zeros_like(mask)
        out[labels == lbl] = 255
        return out
    best_lbl = -1
    best_d = 1e18
    for i in range(1, num):
        if int(stats[i, cv2.CC_STAT_AREA]) < 8:
            continue
        ccx = float(cents[i, 0])
        ccy = float(cents[i, 1])
        d = (ccx - cent[0]) ** 2 + (ccy - cent[1]) ** 2
        if d < best_d:
            best_d = d
            best_lbl = i
    if best_lbl < 1:
        return np.zeros_like(mask)
    out = np.zeros_like(mask)
    out[labels == best_lbl] = 255
    return out


def ball_track_mask_and_centroid(
    ball_bgr: np.ndarray,
    wheel_cx: float,
    wheel_cy: float,
    wheel_r_px: float,
    hsv_lower: np.ndarray,
    hsv_upper: np.ndarray,
    *,
    prefer_manual: bool = False,
    anchor_ball_xy: Optional[Tuple[float, float]] = None,
    anchor_weight: Optional[float] = None,
    allow_auto_fallback: bool = True,
    tube_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, Optional[Tuple[float, float]], bool]:
    """
    Slider ``inRange`` mask is intersected with the **track annulus** (same band as auto detection).

    Returns ``(mask_bgr_or_gray_for_vis, centroid, used_auto)``.
    If ``prefer_manual`` is True and that annulus mask has a blob in the area window, it wins before
    auto white — so lowering **U-S** / tightening HSV darkens the Color tab and changes tracking together.

    ``anchor_ball_xy`` (ball-ROI pixels): optional hint (pick point or prior frame); ``anchor_weight``
    scales distance bonus (default 0.65 if unset). Locked mode uses stricter **round** + radius gates.

    If ``allow_auto_fallback`` is False (ball color locked after pick), never use the internal
    auto-white detector — avoids random white speckles when manual mask is empty.

    ``tube_mask``: Step-2 path tube; intersect with annulus so the golden track ring does not all match HSV.
    """
    import cv2

    if ball_bgr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8), None, False

    h, w = ball_bgr.shape[:2]
    region = _region_annulus_tube_disk(h, w, wheel_cx, wheel_cy, wheel_r_px, tube_mask)
    white = _white_ball_mask_bgr(ball_bgr)
    auto = cv2.bitwise_and(white, region)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    auto = cv2.morphologyEx(auto, cv2.MORPH_OPEN, k, iterations=1)
    auto = cv2.morphologyEx(auto, cv2.MORPH_CLOSE, k, iterations=1)

    img_area = float(h * w)
    min_a = max(12.0, img_area * 0.00005)
    # Pocket text / merged ink often forms a much larger blob than the ball; cap harder for manual HSV.
    max_a_default = img_area * 0.12
    max_a_manual = min(max_a_default, img_area * 0.065)

    def best_ball_centroid(
        mask: np.ndarray,
        *,
        max_area: float,
        anchor_xy: Optional[Tuple[float, float]] = None,
        anchor_weight: float = 0.0,
        strict_shape: bool = False,
    ) -> tuple[Optional[Tuple[float, float]], float]:
        """
        Prefer a *round* blob (ball), not the largest white region (wheel numbers / paint).

        ``strict_shape`` (locked tracking): skip blobs that fail radius / roundness gates so rim
        streaks and specular arcs do not win over the ball.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0.0
        diag = float(np.hypot(w, h))
        best_c: Optional[Tuple[float, float]] = None
        best_score = -1e18
        best_area = 0.0
        for c in contours:
            a = float(cv2.contourArea(c))
            if a < min_a or a > max_area:
                continue
            perim = cv2.arcLength(c, True)
            if perim < 1e-3:
                continue
            circ = 4.0 * np.pi * a / (perim * perim)
            x, y, bw_, bh_ = cv2.boundingRect(c)
            extent = a / float(max(1, bw_ * bh_))
            (_, _), r_enc = cv2.minEnclosingCircle(c)
            m = cv2.moments(c)
            if m["m00"] < 1e-6:
                continue
            cx = float(m["m10"] / m["m00"])
            cy = float(m["m01"] / m["m00"])
            hull = cv2.convexHull(c)
            hull_a = float(cv2.contourArea(hull))
            solidity = a / (hull_a + 1e-6)
            if strict_shape and not _contour_passes_ball_geometry(
                a=a,
                circ=circ,
                extent=extent,
                solidity=solidity,
                bw_=bw_,
                bh_=bh_,
                r_enc=float(r_enc),
                wheel_r_px=wheel_r_px,
                h=h,
                w=w,
            ):
                continue
            # Typical ball: high circularity & extent (~pi/4); strokes / digits: lower.
            shape = 1.15 * circ + 0.45 * extent + 0.35 * solidity
            size_bonus = 0.35 * min(a / (img_area * 0.012 + 1e-6), 1.0)
            score = shape + size_bonus
            if anchor_xy is not None and anchor_weight > 0.0:
                ax, ay = anchor_xy
                d = float(np.hypot(cx - ax, cy - ay)) / (diag + 1e-6)
                score += anchor_weight * (1.0 - min(d * 2.8, 1.0))
            if score > best_score:
                best_score = score
                best_c = (cx, cy)
                best_area = a
        return best_c, best_area

    hsv = cv2.cvtColor(ball_bgr, cv2.COLOR_BGR2HSV)
    lo, hi = _hsv_lo_hi_for_inrange(hsv_lower, hsv_upper)
    manual_full = cv2.inRange(hsv, lo, hi)
    # Restrict to path tube + annulus — drops hub, outer golden ring, and off-path highlights.
    manual_raw = cv2.bitwise_and(manual_full, region)
    # Locked mode: avoid opening away the whole ball blob; tune phase still uses light cleanup.
    open_iters = 0 if allow_auto_fallback is False else 1
    manual = manual_raw
    if open_iters > 0:
        manual = cv2.morphologyEx(manual, cv2.MORPH_OPEN, k, iterations=open_iters)
    manual = cv2.morphologyEx(manual, cv2.MORPH_CLOSE, k, iterations=1)
    if allow_auto_fallback is False and np.count_nonzero(manual) < 30 and np.count_nonzero(manual_raw) > np.count_nonzero(
        manual
    ):
        manual = manual_raw

    anchor_ball: Optional[Tuple[float, float]] = None
    anchor_w = 0.0
    if anchor_ball_xy is not None:
        anchor_ball = (float(anchor_ball_xy[0]), float(anchor_ball_xy[1]))
        anchor_w = 0.65 if anchor_weight is None else float(anchor_weight)

    strict = not allow_auto_fallback
    cent_m, area_m = best_ball_centroid(
        manual,
        max_area=max_a_manual,
        anchor_xy=anchor_ball,
        anchor_weight=anchor_w,
        strict_shape=strict,
    )

    cent_a, area_a = best_ball_centroid(
        auto, max_area=max_a_default, anchor_xy=None, anchor_weight=0.0, strict_shape=False
    )

    def manual_blob_ok(a: float) -> bool:
        return min_a <= a <= max_a_manual

    if not allow_auto_fallback:
        if cent_m is not None:
            return manual, cent_m, False
        return manual, None, False

    if prefer_manual and cent_m is not None and manual_blob_ok(area_m):
        return manual, cent_m, False

    if cent_a is not None and min_a <= area_a <= max_a_default:
        return auto, cent_a, True

    if cent_m is not None and manual_blob_ok(area_m):
        return manual, cent_m, False

    if cent_a is not None:
        return auto, cent_a, True
    return manual, cent_m, False
