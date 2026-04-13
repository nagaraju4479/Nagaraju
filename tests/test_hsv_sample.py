"""HSV range from BGR patch median."""

from __future__ import annotations

import numpy as np

from roulette_predict.vision.hsv_sample import hsv_settings_from_bgr_patch_median


def test_hsv_from_uniform_blue_patch() -> None:
    patch = np.full((7, 7, 3), (255, 0, 0), dtype=np.uint8)  # BGR blue
    h = hsv_settings_from_bgr_patch_median(patch, margin_h=10, margin_s=20, margin_v=20)
    assert 0 <= h.l_h <= h.u_h <= 179
    assert 0 <= h.l_s <= h.u_s <= 255
    assert 0 <= h.l_v <= h.u_v <= 255
    assert h.u_h - h.l_h <= 22
    assert h.u_s - h.l_s <= 42
    assert h.u_v - h.l_v <= 42
