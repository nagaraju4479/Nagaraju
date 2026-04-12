"""HSV mask on synthetic image."""

from __future__ import annotations

import numpy as np

from roulette_predict.vision.hsv_mask import apply_hsv_mask_bgr


def test_hsv_mask_white_circle_on_black() -> None:
    import cv2

    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(img, (100, 100), 40, (255, 255, 255), thickness=-1)
    mask, _ = apply_hsv_mask_bgr(img, 0, 179, 0, 30, 200, 255)
    ratio = float(np.count_nonzero(mask)) / mask.size
    assert 0.01 < ratio < 0.5
