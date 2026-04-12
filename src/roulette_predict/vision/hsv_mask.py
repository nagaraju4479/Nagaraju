"""HSV masking for wheel crop preview."""

from __future__ import annotations

import numpy as np


def apply_hsv_mask_bgr(
    bgr: np.ndarray,
    l_h: int,
    u_h: int,
    l_s: int,
    u_s: int,
    l_v: int,
    u_v: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mask uint8 0/255, masked_bgr)."""
    import cv2

    if bgr.size == 0:
        empty = np.zeros((1, 1), dtype=np.uint8)
        return empty, bgr.copy() if bgr.size else bgr

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([l_h, l_s, l_v], dtype=np.uint8)
    upper = np.array([u_h, u_s, u_v], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    masked = cv2.bitwise_and(bgr, bgr, mask=mask)
    return mask, masked
