"""Digit template matching (OpenCV matchTemplate) as fallback when OCR is empty."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np


@lru_cache(maxsize=128)
def _digit_patch(label: str, box_h: int) -> np.ndarray:
    import cv2

    img = np.zeros((40, 140), dtype=np.uint8)
    cv2.putText(img, label, (4, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2, cv2.LINE_AA)
    th = max(14, min(52, box_h))
    scale = th / 40.0
    nw = max(8, int(img.shape[1] * scale))
    nh = max(8, int(img.shape[0] * scale))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)


def match_roulette_number_template(bgr: np.ndarray) -> Optional[int]:
    """Best-effort 0–36 via normalized cross-correlation on binarized ROI."""
    import cv2

    if bgr.size == 0:
        return None
    h0, w0 = bgr.shape[:2]
    if h0 < 6 or w0 < 6:
        return None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hi, wi = bin_img.shape[:2]
    best_n: Optional[int] = None
    best_score = -1.0
    for n in range(37):
        label = str(n)
        tpl = _digit_patch(label, hi)
        th, tw = tpl.shape[:2]
        if th > hi or tw > wi:
            tpl = cv2.resize(tpl, (min(wi, tw), min(hi, th)), interpolation=cv2.INTER_AREA)
            th, tw = tpl.shape[:2]
        if th < 4 or tw < 4 or th > hi or tw > wi:
            continue
        res = cv2.matchTemplate(bin_img, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > best_score:
            best_score = float(max_val)
            best_n = n
    if best_n is None or best_score < 0.22:
        return None
    return best_n
