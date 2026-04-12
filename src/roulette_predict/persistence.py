"""Load/save JSON config to disk."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from roulette_predict.config_model import CalibrationData, HsvSettings, default_config_dict, merge_config


def config_path() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    d = Path(base) / "RoulettePredict"
    d.mkdir(parents=True, exist_ok=True)
    return d / "config.json"


def load_config(path: Path | None = None) -> dict[str, Any]:
    p = path or config_path()
    if not p.is_file():
        return default_config_dict()
    try:
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return default_config_dict()
        merged = default_config_dict()
        merged.update(raw)
        return merged
    except (OSError, json.JSONDecodeError):
        return default_config_dict()


def save_config(
    cal: CalibrationData,
    hsv: HsvSettings,
    red_border: bool,
    window_opacity: float,
    tesseract_cmd: Optional[str] = None,
    path: Path | None = None,
) -> None:
    from dataclasses import asdict

    p = path or config_path()
    data = {
        "calibration": asdict(cal),
        "hsv": asdict(hsv),
        "red_border": red_border,
        "window_opacity": window_opacity,
        "tesseract_cmd": tesseract_cmd or "",
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def parse_loaded(raw: dict[str, Any]) -> tuple[CalibrationData, HsvSettings, bool, float, Optional[str]]:
    return merge_config(raw)
