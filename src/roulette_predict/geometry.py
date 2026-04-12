"""Screen-coordinate geometry helpers (circle, rect, polyline)."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Circle:
    cx: float
    cy: float
    r: float

    def __post_init__(self) -> None:
        if self.r <= 0:
            raise ValueError("radius must be positive")


def circle_from_drag(x0: float, y0: float, x1: float, y1: float) -> Circle:
    """Circle from drag: center at start, radius = distance to end."""
    cx, cy = x0, y0
    r = math.hypot(x1 - x0, y1 - y0)
    if r <= 0:
        r = 1.0
    return Circle(cx, cy, r)


def circle_from_three_points(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> Circle:
    """Circumcircle of three non-collinear points."""
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-9:
        raise ValueError("points are collinear")
    ax2_ay2 = ax * ax + ay * ay
    bx2_by2 = bx * bx + by * by
    cx2_cy2 = cx * cx + cy * cy
    ux = (ax2_ay2 * (by - cy) + bx2_by2 * (cy - ay) + cx2_cy2 * (ay - by)) / d
    uy = (ax2_ay2 * (cx - bx) + bx2_by2 * (ax - cx) + cx2_cy2 * (bx - ax)) / d
    r = math.hypot(ax - ux, ay - uy)
    if r <= 0:
        r = 1.0
    return Circle(ux, uy, r)


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    @property
    def left(self) -> float:
        return self.x

    @property
    def top(self) -> float:
        return self.y

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h


def normalize_rect(x0: float, y0: float, x1: float, y1: float) -> Rect:
    """Axis-aligned rect from drag corners (handles negative width/height)."""
    left = min(x0, x1)
    top = min(y0, y1)
    w = abs(x1 - x0)
    h = abs(y1 - y0)
    if w < 1:
        w = 1.0
    if h < 1:
        h = 1.0
    return Rect(left, top, w, h)


def polyline_length(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        total += math.hypot(x1 - x0, y1 - y0)
    return total


def resample_polyline(points: list[tuple[float, float]], n: int) -> list[tuple[float, float]]:
    """Evenly sample along polyline length (at least 2 points if n>=2)."""
    if len(points) < 2 or n < 2:
        return list(points)
    length = polyline_length(points)
    if length <= 0:
        return [points[0], points[-1]]
    out: list[tuple[float, float]] = []
    seg_lens: list[float] = []
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        seg_lens.append(math.hypot(x1 - x0, y1 - y0))
    for k in range(n):
        t = k / (n - 1) if n > 1 else 0.0
        target = t * length
        acc = 0.0
        for i, sl in enumerate(seg_lens):
            if acc + sl >= target or i == len(seg_lens) - 1:
                seg_t = 0.0 if sl <= 0 else (target - acc) / sl
                seg_t = max(0.0, min(1.0, seg_t))
                x0, y0 = points[i]
                x1, y1 = points[i + 1]
                out.append((x0 + seg_t * (x1 - x0), y0 + seg_t * (y1 - y0)))
                break
            acc += sl
    return out
