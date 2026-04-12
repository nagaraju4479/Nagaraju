"""Experimental angular speed from ball path ROI (centroid tracking)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np


@dataclass
class SpeedTracker:
    wheel_cx: float
    wheel_cy: float
    history: Deque[Tuple[float, float, float]]  # (t_sec, angle_rad, |omega| approx)
    # Last signed angular rate (rad/s) for orbit extrapolation when flow drops out.
    last_signed_omega_rad_s: float = 0.0

    @classmethod
    def new(cls, wheel_cx: float, wheel_cy: float, maxlen: int = 30) -> "SpeedTracker":
        return cls(wheel_cx, wheel_cy, deque(maxlen=maxlen))

    def update(self, t_sec: float, cx: float, cy: float) -> Optional[float]:
        """Append centroid; return finite-difference |dtheta/dt| (rad/s) or None."""
        dx = cx - self.wheel_cx
        dy = cy - self.wheel_cy
        ang = float(np.arctan2(dy, dx))
        if self.history:
            t0, a0, _ = self.history[-1]
            dt = t_sec - t0
            if dt > 1e-4:
                da = ang - a0
                while da > np.pi:
                    da -= 2 * np.pi
                while da < -np.pi:
                    da += 2 * np.pi
                self.last_signed_omega_rad_s = float(da / dt)
                omega = abs(self.last_signed_omega_rad_s)
            else:
                omega = 0.0
        else:
            omega = 0.0
        self.history.append((t_sec, ang, omega))
        if len(self.history) < 2:
            return None
        return float(self.history[-1][2])


def largest_contour_centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    import cv2

    if mask.size == 0:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    m = cv2.moments(c)
    if m["m00"] < 1e-6:
        return None
    return float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])
