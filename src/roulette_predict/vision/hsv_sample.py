"""Derive HSV slider ranges from a BGR patch (e.g. user click on the ball)."""

from __future__ import annotations

import numpy as np

from roulette_predict.config_model import HsvSettings


def hsv_settings_from_bgr_patch_median(
    bgr_patch: np.ndarray,
    *,
    margin_h: int = 14,
    margin_s: int = 42,
    margin_v: int = 38,
) -> HsvSettings:
    """
    Median color in ``bgr_patch`` → OpenCV HSV, then axis-aligned box with margins.

    Use after the user has darkened the mask (L-V / L-S) and clicks **on the ball** so only
    that pixel neighborhood defines the new in-range cone.
    """
    import cv2

    if bgr_patch.size == 0 or bgr_patch.shape[0] < 1 or bgr_patch.shape[1] < 1:
        return HsvSettings()
    hsv = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
    flat = hsv.reshape(-1, 3).astype(np.float32)
    mh = float(np.median(flat[:, 0]))
    ms = float(np.median(flat[:, 1]))
    mv = float(np.median(flat[:, 2]))

    l_h = int(np.clip(round(mh - margin_h), 0, 179))
    u_h = int(np.clip(round(mh + margin_h), 0, 179))
    l_s = int(np.clip(round(ms - margin_s), 0, 255))
    u_s = int(np.clip(round(ms + margin_s), 0, 255))
    l_v = int(np.clip(round(mv - margin_v), 0, 255))
    u_v = int(np.clip(round(mv + margin_v), 0, 255))

    if l_h > u_h:
        l_h, u_h = u_h, l_h
    if l_s > u_s:
        l_s, u_s = u_s, l_s
    if l_v > u_v:
        l_v, u_v = u_v, l_v

    return HsvSettings(l_h=l_h, u_h=u_h, l_s=l_s, u_s=u_s, l_v=l_v, u_v=u_v)
