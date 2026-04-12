"""mss-based BGR capture for monitor subregions."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def grab_region_bgr(
    left: int,
    top: int,
    width: int,
    height: int,
    monitor_index: int = 0,
) -> np.ndarray:
    import mss

    if width < 1 or height < 1:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    with mss.mss() as sct:
        mons = sct.monitors
        if monitor_index < 0 or monitor_index >= len(mons):
            monitor_index = 0
        base = mons[monitor_index]
        region = {
            "left": base["left"] + left,
            "top": base["top"] + top,
            "width": width,
            "height": height,
        }
        shot = sct.grab(region)
        arr = np.asarray(shot)  # BGRA
        bgr = arr[:, :, :3].copy()
        return bgr


def grab_monitor_full_bgr(monitor_index: int = 1) -> Tuple[np.ndarray, dict]:
    """monitor_index 1 = first real monitor in mss (0 is virtual full desktop)."""
    import mss

    with mss.mss() as sct:
        mons = sct.monitors
        idx = monitor_index if 0 <= monitor_index < len(mons) else 1
        mon = mons[idx]
        shot = sct.grab(mon)
        arr = np.asarray(shot)
        bgr = arr[:, :, :3].copy()
        return bgr, dict(mon)
