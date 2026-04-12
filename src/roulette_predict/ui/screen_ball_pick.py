"""Fullscreen click overlay to sample ball color from the live game (not the preview tab)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import QWidget


class ScreenBallPickOverlay(QWidget):
    """Semi-transparent fullscreen layer; one left-click samples a screen patch and emits mean BGR."""

    picked_bgr = Signal(float, float, float)  # B, G, R means
    cancelled = Signal()

    def __init__(self, screen_geometry: QRect, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setGeometry(screen_geometry)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.setFocus(Qt.FocusReason.PopupFocusReason)
        self.activateWindow()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 40, 90))
        painter.setPen(QPen(QColor(0, 255, 255), 2))
        painter.drawText(24, 48, "Click the ball on the live game window. Esc cancels.")

    def mousePressEvent(self, e: QMouseEvent) -> None:
        if e.button() != Qt.MouseButton.LeftButton:
            return
        g = e.globalPosition().toPoint()
        bgr = _grab_mean_bgr_around(g.x(), g.y(), radius=14)
        if bgr is not None:
            b, gr, r = bgr
            self.picked_bgr.emit(float(b), float(gr), float(r))
        self.close()

    def keyPressEvent(self, e) -> None:
        from PySide6.QtGui import QKeyEvent

        if isinstance(e, QKeyEvent) and e.key() == Qt.Key.Key_Escape:
            self.cancelled.emit()
            self.close()
            return
        super().keyPressEvent(e)


def _grab_mean_bgr_around(cx: int, cy: int, radius: int = 14) -> Optional[Tuple[float, float, float]]:
    import mss

    side = 2 * radius + 1
    with mss.mss() as sct:
        region = {"left": int(cx - radius), "top": int(cy - radius), "width": side, "height": side}
        try:
            shot = sct.grab(region)
        except Exception:
            return None
        arr = np.asarray(shot, dtype=np.float64)[:, :, :3]
    m = arr.reshape(-1, 3).mean(axis=0)
    return float(m[0]), float(m[1]), float(m[2])
