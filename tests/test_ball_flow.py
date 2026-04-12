"""Dense optical flow centroid + orbit extrapolation."""

from __future__ import annotations

import math

import numpy as np


def test_preprocess_blur_does_not_crash() -> None:
    from roulette_predict.vision.ball_flow import preprocess_ball_gray

    g = np.zeros((32, 40), dtype=np.uint8)
    out = preprocess_ball_gray(g)
    assert out.shape == g.shape


def test_farneback_returns_none_when_no_motion() -> None:
    from roulette_predict.vision.ball_flow import track_centroid_farneback_path_tube

    h, wi = 48, 48
    prev = np.ones((h, wi), dtype=np.uint8) * 100
    curr = prev.copy()
    mask = np.zeros((h, wi), dtype=np.uint8)
    mask[10:40, 10:40] = 255
    fb = track_centroid_farneback_path_tube(prev, curr, 24.0, 24.0, mask)
    assert fb is None


def test_extrapolate_advances_on_orbit() -> None:
    from roulette_predict.vision.ball_flow import extrapolate_ball_on_wheel_orbit

    cx, cy = 100.0, 100.0
    lx, ly = 150.0, 100.0
    r = 50.0
    dt = 1.0 / 30.0
    omega = 2.0  # rad/s
    nx, ny = extrapolate_ball_on_wheel_orbit(cx, cy, lx, ly, omega, dt)
    theta0 = math.atan2(ly - cy, lx - cx)
    theta1 = theta0 + omega * dt
    assert abs(nx - (cx + r * math.cos(theta1))) < 1e-6
    assert abs(ny - (cy + r * math.sin(theta1))) < 1e-6
