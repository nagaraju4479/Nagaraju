"""Serializable calibration and HSV settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from roulette_predict.geometry import Circle, Rect


@dataclass
class CalibrationData:
    monitor_index: int = 0
    # Qt mouse/geometry coords are logical pixels; mss uses physical — scale at calibration time.
    screen_scale: float = 1.0
    wheel_circle: Optional[dict[str, float]] = None  # cx, cy, r
    ball_path_points: list[list[float]] = field(default_factory=list)  # [[x,y], ...]
    ocr_rect: Optional[dict[str, float]] = None  # x, y, w, h

    def pixel_scale(self) -> float:
        s = float(self.screen_scale)
        return s if s > 0.25 else 1.0

    def bounding_wheel_rect(self) -> Optional[tuple[int, int, int, int]]:
        if not self.wheel_circle:
            return None
        c = self.wheel_circle
        cx, cy, r = c["cx"], c["cy"], c["r"]
        pad = 4
        return (
            int(cx - r - pad),
            int(cy - r - pad),
            int(2 * r + 2 * pad),
            int(2 * r + 2 * pad),
        )


def circle_to_dict(circle: Circle) -> dict[str, float]:
    return {"cx": circle.cx, "cy": circle.cy, "r": circle.r}


def dict_to_circle(d: dict[str, float]) -> Circle:
    return Circle(float(d["cx"]), float(d["cy"]), float(d["r"]))


def rect_to_dict(r: Rect) -> dict[str, float]:
    return {"x": r.x, "y": r.y, "w": r.w, "h": r.h}


def dict_to_rect(d: dict[str, float]) -> Rect:
    return Rect(float(d["x"]), float(d["y"]), float(d["w"]), float(d["h"]))


@dataclass
class HsvSettings:
    l_h: int = 0
    u_h: int = 179
    l_s: int = 0
    u_s: int = 255
    l_v: int = 0
    u_v: int = 255


def default_config_dict() -> dict[str, Any]:
    cal = CalibrationData()
    hsv = HsvSettings()
    return {
        "calibration": asdict(cal),
        "hsv": asdict(hsv),
        "red_border": False,
        "window_opacity": 1.0,
        "tesseract_cmd": "",
    }


def normalize_tesseract_cmd(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip().strip('"').strip("'")
    return s if s else None


def merge_config(raw: dict[str, Any]) -> tuple[CalibrationData, HsvSettings, bool, float, Optional[str]]:
    cal = CalibrationData(**{**asdict(CalibrationData()), **raw.get("calibration", {})})
    hsv = HsvSettings(**{**asdict(HsvSettings()), **raw.get("hsv", {})})
    red = bool(raw.get("red_border", False))
    opacity = float(raw.get("window_opacity", 1.0))
    tess = normalize_tesseract_cmd(raw.get("tesseract_cmd"))
    return cal, hsv, red, max(0.1, min(1.0, opacity)), tess
