"""Fullscreen dim overlay: click the ball on the live wheel (screen coordinates)."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import QWidget


class BallPickScreenOverlay(QWidget):
    """Covers one monitor with a light dim; one click emits global screen coordinates."""

    picked = Signal(int, int)
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
        self._origin = QPoint(screen_geometry.topLeft())

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.setFocus(Qt.FocusReason.PopupFocusReason)
        self.activateWindow()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 100))
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawText(
            24,
            36,
            "Click the ball on the LIVE wheel (inside your Step-2 path).  Esc = cancel.",
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            g = event.globalPosition().toPoint()
            self.picked.emit(g.x(), g.y())
            self.close()
        super().mousePressEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.cancelled.emit()
            self.close()
        else:
            super().keyPressEvent(event)
