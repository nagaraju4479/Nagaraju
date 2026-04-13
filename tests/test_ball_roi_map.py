"""Wheel preview ↔ ball-path ROI coordinate mapping."""

from __future__ import annotations

from roulette_predict.capture.worker import (
    ball_roi_xy_to_wheel_preview_xy,
    _wheel_preview_xy_to_ball_roi_xy,
)


def test_pick_inside_ball_roi_maps() -> None:
    # Wheel grab 100x100; ball 50x50 at bl=10. Preview 50x50. Wheel-local (20,10) → preview (10,5).
    # ball-local = (20-10, 10-0) = (10,10).
    m = _wheel_preview_xy_to_ball_roi_xy(10, 5, 50, 50, 100, 100, 0, 0, 10, 0, 50, 50)
    assert m == (10, 10)


def test_pick_outside_ball_roi_none() -> None:
    m = _wheel_preview_xy_to_ball_roi_xy(0, 0, 50, 50, 100, 100, 0, 0, 80, 0, 50, 50)
    assert m is None


def test_roi_to_preview_roundtrips_forward_map() -> None:
    hx, hy = 10, 5
    m = _wheel_preview_xy_to_ball_roi_xy(hx, hy, 50, 50, 100, 100, 0, 0, 10, 0, 50, 50)
    assert m == (10, 10)
    bx, by = ball_roi_xy_to_wheel_preview_xy(float(m[0]), float(m[1]), 50, 50, 100, 100, 0, 0, 10, 0)
    assert (bx, by) == (hx, hy)
