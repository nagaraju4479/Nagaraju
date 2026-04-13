"""Automatic ball detection in wheel annulus."""

from __future__ import annotations

import numpy as np

from roulette_predict.config_model import CalibrationData


def test_centroid_in_track_annulus_rejects_hub() -> None:
    from roulette_predict.vision.ball_track import centroid_in_track_annulus

    cx, cy, r = 100.0, 100.0, 80.0
    assert not centroid_in_track_annulus(cx + 5, cy, cx, cy, r)
    assert centroid_in_track_annulus(cx + 0.65 * r, cy, cx, cy, r)


def test_centroid_on_step2_track_mask_requires_white_pixel() -> None:
    from roulette_predict.vision.ball_track import centroid_on_step2_track_mask

    m = np.zeros((40, 40), dtype=np.uint8)
    m[10:30, 10:30] = 255
    assert centroid_on_step2_track_mask(15.0, 15.0, m)
    assert not centroid_on_step2_track_mask(5.0, 5.0, m)


def test_find_moving_ball_requires_frame_change() -> None:
    """Identical frames → no motion mask → None (caller falls back to white-blob pick)."""
    from roulette_predict.config_model import CalibrationData
    from roulette_predict.vision.ball_track import (
        find_moving_ball_centroid_near_click_for_pick,
        path_tube_mask_ball_roi,
    )

    h, w = 120, 180
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cal = CalibrationData(ball_path_points=[[30.0, 30.0], [90.0, 90.0]])
    tube = path_tube_mask_ball_roi(h, w, cal)
    assert tube is not None
    out = find_moving_ball_centroid_near_click_for_pick(
        img, img, 60.0, 60.0, 55.0, 60.0, 60.0, tube
    )
    assert out is None


def test_centroid_near_step2_track_mask_allows_small_offset() -> None:
    from roulette_predict.vision.ball_track import centroid_near_step2_track_mask

    m = np.zeros((40, 40), dtype=np.uint8)
    m[20, 20] = 255
    assert centroid_near_step2_track_mask(20.0, 20.0, m, radius_px=5)
    assert centroid_near_step2_track_mask(23.0, 18.0, m, radius_px=5)
    assert not centroid_near_step2_track_mask(5.0, 5.0, m, radius_px=2)


def test_path_tube_mask_covers_painted_channel() -> None:
    from roulette_predict.vision.ball_track import path_tube_mask_ball_roi

    cal = CalibrationData(
        ball_path_points=[[100.0, 100.0], [180.0, 100.0]],
    )
    m = path_tube_mask_ball_roi(120, 220, cal)
    assert m is not None
    assert m.shape == (120, 220)
    # Horizontal stroke ~y=28 in ROI (path y=100, min_y=(100-28))
    assert np.count_nonzero(m) > 500


def test_white_ball_found_in_track_annulus() -> None:
    import cv2

    from roulette_predict.vision.ball_track import ball_track_mask_and_centroid

    h, w = 200, 200
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cx, cy, r_wheel = 100.0, 100.0, 90.0
    bx = int(cx + 0.72 * r_wheel)
    by = int(cy)
    cv2.circle(img, (bx, by), 9, (255, 255, 255), thickness=-1)
    # High L-V so the annulus is not one huge inRange blob (would exceed max area and fall back to auto).
    lo = np.array([0, 0, 200], dtype=np.uint8)
    hi = np.array([179, 255, 255], dtype=np.uint8)
    mask, cent, used_auto = ball_track_mask_and_centroid(img, cx, cy, r_wheel, lo, hi, prefer_manual=True)
    assert used_auto is False
    assert cent is not None
    assert abs(cent[0] - bx) < 4
    assert abs(cent[1] - by) < 4
    assert mask.shape == (h, w)


def test_locked_tracking_prefers_round_ball_over_elongated_streak() -> None:
    """Rim streaks fail roundness / aspect gates; compact ball blob wins."""
    import cv2

    from roulette_predict.vision.ball_track import ball_track_mask_and_centroid

    h, w = 200, 200
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cx, cy, rw = 100.0, 100.0, 90.0
    bx = int(cx + 0.72 * rw)
    by = int(cy)
    cv2.circle(img, (bx, by), 9, (255, 255, 255), -1)
    sx = int(cx - 0.72 * rw)
    for dx in range(-18, 19):
        for dy in range(-2, 3):
            img[by + dy, sx + dx] = (255, 255, 255)
    lo = np.array([0, 0, 200], dtype=np.uint8)
    hi = np.array([179, 255, 255], dtype=np.uint8)
    _mask, cent, used_auto = ball_track_mask_and_centroid(
        img, cx, cy, rw, lo, hi, prefer_manual=True, allow_auto_fallback=False
    )
    assert used_auto is False
    assert cent is not None
    assert abs(cent[0] - float(bx)) < 5.0
    assert abs(cent[1] - float(by)) < 5.0


def test_mask_keep_blob_keeps_only_tracked_component() -> None:
    import cv2

    from roulette_predict.vision.ball_track import mask_keep_blob_at_tracked_centroid

    m = np.zeros((80, 80), dtype=np.uint8)
    cv2.circle(m, (20, 40), 8, 255, -1)
    cv2.circle(m, (60, 40), 8, 255, -1)
    out = mask_keep_blob_at_tracked_centroid(m, (20.0, 40.0))
    assert np.count_nonzero(out) > 0
    assert np.count_nonzero(out) < np.count_nonzero(m)
    # Centroid sits on left blob only
    assert out[40, 20] == 255
    assert out[40, 60] == 0


def test_prefer_manual_uses_inrange_first() -> None:
    import cv2

    from roulette_predict.vision.ball_track import ball_track_mask_and_centroid

    h, w = 200, 200
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cx, cy, r_wheel = 100.0, 100.0, 90.0
    bx = int(cx + 0.72 * r_wheel)
    by = int(cy)
    cv2.circle(img, (bx, by), 9, (255, 255, 255), thickness=-1)
    # Tight range so manual mask is the ball only (full 0–255 range would threshold the whole frame).
    lo = np.array([0, 0, 200], dtype=np.uint8)
    hi = np.array([179, 120, 255], dtype=np.uint8)
    mask, cent, used_auto = ball_track_mask_and_centroid(
        img, cx, cy, r_wheel, lo, hi, prefer_manual=True
    )
    assert used_auto is False
    assert cent is not None
    assert abs(cent[0] - bx) < 4
    assert abs(cent[1] - by) < 4
