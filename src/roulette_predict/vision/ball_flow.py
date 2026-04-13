"""Dense optical flow (Farneback) for ball tracking: masked ROI, tangential motion, weighted centroid."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

# High flow magnitude in the path ROI ⇒ motion blur; use a lower mag percentile to keep streak pixels.
MOTION_BLUR_MAX_MAG_TRIGGER = 6.5
MAG_PERCENTILE_NORMAL = 58.0
MAG_PERCENTILE_BLUR = 40.0

# Real-time budget: default OpenCV Farneback( levels=5, winsize=41 ) is heavy on large ball ROIs.
# Slightly fewer levels + smaller window cuts CPU with modest impact on fast-spin tracking.
FARNEBACK_PYR_LEVELS = 4
FARNEBACK_WINSIZE = 35


def preprocess_ball_gray(gray: np.ndarray) -> np.ndarray:
    """Grayscale frame with light Gaussian blur to suppress sensor noise before Farneback."""
    import cv2

    if gray.size == 0:
        return gray
    return cv2.GaussianBlur(gray, (5, 5), 0.85)


def extrapolate_ball_on_wheel_orbit(
    wheel_cx: float,
    wheel_cy: float,
    last_x: float,
    last_y: float,
    omega_signed_rad_s: float,
    dt: float,
) -> Tuple[float, float]:
    """Advance a point on a circular arc around the wheel center by ``omega * dt`` (rad)."""
    dx = last_x - wheel_cx
    dy = last_y - wheel_cy
    r = math.hypot(dx, dy)
    if r < 1e-4:
        return last_x, last_y
    theta = math.atan2(dy, dx)
    theta2 = theta + omega_signed_rad_s * dt
    return wheel_cx + r * math.cos(theta2), wheel_cy + r * math.sin(theta2)


def track_centroid_farneback_path_tube(
    prev_prep: np.ndarray,
    curr_prep: np.ndarray,
    wheel_cx: float,
    wheel_cy: float,
    path_tube_mask: np.ndarray,
    *,
    min_weight_sum: float = 130.0,
    min_pixels: int = 28,
) -> Optional[Tuple[float, float]]:
    """
    OpenCV Farneback dense flow between preprocessed frames; motion is analyzed **only** inside
    ``path_tube_mask`` (annulus ∩ path tube).

    Uses ``pyr_scale=0.5``, ``levels>=5``, ``winsize=41`` for large inter-frame displacement.

    Steps:
      1. ``calcOpticalFlowFarneback`` on blurred grayscale pair.
      2. ``cartToPolar`` → magnitude + angle of each flow vector.
      3. **Motion blur**: if max ``mag`` inside ROI is high, ``mag`` percentile floor is **40**;
         otherwise **58** (rejects jitter while keeping blurred streaks).
      4. Decompose flow into radial vs tangential relative to ``(wheel_cx, wheel_cy)``; keep
         tangential energy.
      5. Direction weights from CCW/CW tangent projections.
      6. Weighted centroid of surviving pixels.

    Returns ``None`` if motion mass is too weak (caller may extrapolate or use HSV).
    """
    import cv2

    if prev_prep.shape != curr_prep.shape or prev_prep.ndim != 2:
        return None
    h, wi = curr_prep.shape
    if path_tube_mask.shape != (h, wi):
        return None

    flow = cv2.calcOpticalFlowFarneback(
        prev_prep,
        curr_prep,
        None,
        0.5,
        FARNEBACK_PYR_LEVELS,
        FARNEBACK_WINSIZE,
        3,
        7,
        1.5,
        0,
    )
    if not np.isfinite(flow).all():
        return None

    fx = flow[:, :, 0].astype(np.float64)
    fy = flow[:, :, 1].astype(np.float64)
    mag, _ang = cv2.cartToPolar(fx.astype(np.float32), fy.astype(np.float32))

    roi = path_tube_mask > 127
    if np.count_nonzero(roi) < min_pixels:
        return None

    mag_f = mag.astype(np.float64)
    inside = roi & (mag_f > 1e-6)
    if np.count_nonzero(inside) < min_pixels:
        return None

    max_roi_mag = float(np.max(mag_f[roi]))
    # Near-static ROI: no meaningful motion → do not lock onto texture/noise outside the moving ball.
    # Too strict a floor drops flow on dim video / 60 Hz timing; HSV fallback still applies in worker.
    if max_roi_mag < 0.26:
        return None
    mag_percentile = (
        MAG_PERCENTILE_BLUR if max_roi_mag >= MOTION_BLUR_MAX_MAG_TRIGGER else MAG_PERCENTILE_NORMAL
    )

    mags = mag_f[inside]
    mag_thresh = float(np.percentile(mags, mag_percentile))
    mag_thresh = max(0.28, mag_thresh)

    motion = roi & (mag_f >= mag_thresh)
    if np.count_nonzero(motion) < min_pixels:
        return None

    y_idx, x_idx = np.indices((h, wi))
    dx = x_idx.astype(np.float64) - float(wheel_cx)
    dy = y_idx.astype(np.float64) - float(wheel_cy)
    rlen = np.sqrt(dx * dx + dy * dy) + 1e-5
    ux = dx / rlen
    uy = dy / rlen

    radial = fx * ux + fy * uy
    tang_sq = fx * fx + fy * fy - radial * radial
    tangential_mag = np.sqrt(np.maximum(tang_sq, 0.0))

    tccw_x, tccw_y = -uy, ux
    tcw_x, tcw_y = uy, -ux
    dot_ccw = fx * tccw_x + fy * tccw_y
    dot_cw = fx * tcw_x + fy * tcw_y
    align = np.maximum(np.abs(dot_ccw), np.abs(dot_cw))
    dir_w = np.where(align > 1e-6, align / (tangential_mag + 1e-4), 0.0)

    diff = cv2.absdiff(prev_prep, curr_prep).astype(np.float64) / 255.0
    weights = (
        motion.astype(np.float64)
        * tangential_mag
        * (0.55 + 0.45 * np.minimum(dir_w, 2.5))
        * (1.0 + 0.55 * diff)
    )
    sw = float(np.sum(weights))
    if sw < min_weight_sum or not np.isfinite(sw):
        return None

    cx = float(np.sum(x_idx.astype(np.float64) * weights) / sw)
    cy = float(np.sum(y_idx.astype(np.float64) * weights) / sw)
    if not np.isfinite(cx) or not np.isfinite(cy):
        return None
    return (cx, cy)
