"""Unit tests: circle, rect, polyline."""

from __future__ import annotations

import pytest

from roulette_predict.geometry import (
    Circle,
    circle_from_drag,
    circle_from_three_points,
    normalize_rect,
    polyline_length,
    resample_polyline,
)


def test_circle_from_drag() -> None:
    c = circle_from_drag(100.0, 100.0, 100.0, 150.0)
    assert c.cx == 100.0 and c.cy == 100.0
    assert abs(c.r - 50.0) < 1e-6


def test_circle_from_three_points() -> None:
    c = circle_from_three_points(1.0, 0.0, 0.0, 1.0, -1.0, 0.0)
    assert abs(c.cx) < 1e-5 and abs(c.cy) < 1e-5
    assert abs(c.r - 1.0) < 1e-5


def test_circle_from_three_collinear_raises() -> None:
    with pytest.raises(ValueError):
        circle_from_three_points(0.0, 0.0, 1.0, 1.0, 2.0, 2.0)


def test_normalize_rect_negative_size() -> None:
    r = normalize_rect(10.0, 10.0, 5.0, 20.0)
    assert r.x == 5.0 and r.y == 10.0 and r.w == 5.0 and r.h == 10.0


def test_polyline_length() -> None:
    pts = [(0.0, 0.0), (3.0, 4.0), (3.0, 0.0)]
    assert abs(polyline_length(pts) - 9.0) < 1e-6


def test_resample_polyline() -> None:
    pts = [(0.0, 0.0), (10.0, 0.0)]
    out = resample_polyline(pts, 3)
    assert len(out) == 3
    assert out[0][0] == 0.0 and out[-1][0] == 10.0


def test_circle_invalid_radius() -> None:
    with pytest.raises(ValueError):
        Circle(0.0, 0.0, 0.0)
