"""Automatic ball detection in wheel annulus."""

from __future__ import annotations

import numpy as np

from roulette_predict.config_model import CalibrationData


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
