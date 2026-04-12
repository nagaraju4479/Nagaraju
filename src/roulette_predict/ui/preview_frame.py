"""Roulette tab: video preview with top-left yellow speed overlay (reference UI)."""

from __future__ import annotations

import re
from typing import Optional

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QImage, QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel, QWidget

from roulette_predict.ui.theme import COLOR_BG_PREVIEW


def extract_speed_number_for_overlay(raw: str) -> Optional[str]:
    """Parse numeric ball speed from worker text like ``Ball speed (exp.): 1.23 rad/s``."""
    if "…" in raw or "..." in raw or not raw.strip():
        return None
    m = re.search(r":\s*([\d]+(?:\.\d+)?)\s*", raw)
    return m.group(1) if m else None


class RoulettePreviewFrame(QWidget):
    """Black preview area with bold yellow Speed: … text in the top-left."""
    image_clicked = Signal(int, int)
    image_rect_selected = Signal(int, int, int, int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._last_image: Optional[QImage] = None
        self._last_scaled: Optional[QPixmap] = None
        self._last_pixmap_pos = QPoint(0, 0)
        self._selection_active = False
        self._selection_start = QPoint(0, 0)
        self._selection_end = QPoint(0, 0)
        self._selected_rect: Optional[QRect] = None
        self.setMinimumSize(10, 10)
        self.setStyleSheet(f"background-color: {COLOR_BG_PREVIEW};")
        self._image = QLabel(self)
        self._image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image.setStyleSheet(f"background-color: {COLOR_BG_PREVIEW}; border: none;")
        # Otherwise QLabel steals clicks — ball pick never reaches RoulettePreviewFrame.
        self._image.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._speed = QLabel(self)
        self._speed.setText("Speed: —")
        self._speed.setStyleSheet(
            "color: #FFFF00; font-weight: bold; font-size: 15px; "
            "font-family: 'Segoe UI', 'Arial', sans-serif; "
            "background: transparent;"
        )
        self._speed.raise_()
        self._speed.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def set_ball_pick_cursor(self, _enabled: bool) -> None:
        """Normal arrow cursor only (pick mode uses the same as the rest of the UI)."""
        cur = Qt.CursorShape.ArrowCursor
        self.setCursor(cur)
        self._image.setCursor(cur)

    def begin_pick_snapshot(self) -> bool:
        """Ready for ball pick: require a frame. Preview stays **live** so the ball matches the click."""
        if self._last_image is None or self._last_image.isNull():
            return False
        return True

    def end_pick_snapshot(self) -> None:
        """Called after pick or cancel (no-op; preview was never frozen)."""
        return

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._image.setGeometry(self.rect())
        self._speed.move(12, 10)
        self.rescale_frame()

    def rescale_frame(self) -> None:
        if self._last_image is None or self._last_image.isNull():
            return
        pm = QPixmap.fromImage(self._last_image).scaled(
            self._image.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = max(0, (self._image.width() - pm.width()) // 2)
        y = max(0, (self._image.height() - pm.height()) // 2)
        self._last_scaled = pm
        self._last_pixmap_pos = QPoint(x, y)
        self._image.setPixmap(pm)

    def set_frame_image(self, img: QImage) -> None:
        if img.isNull():
            self._last_image = None
            self._last_scaled = None
            self._image.setPixmap(QPixmap())
            return
        self._last_image = img.copy()
        self.rescale_frame()

    def last_image(self) -> Optional[QImage]:
        if self._last_image is None:
            return None
        return self._last_image

    def set_speed_text(self, raw: str) -> None:
        num = extract_speed_number_for_overlay(raw)
        self._speed.setText(f"Speed: {num}" if num is not None else "Speed: —")

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        if self._last_scaled is None or self._last_image is None:
            return
        px = int(event.position().x() - self._last_pixmap_pos.x())
        py = int(event.position().y() - self._last_pixmap_pos.y())
        if px < 0 or py < 0 or px >= self._last_scaled.width() or py >= self._last_scaled.height():
            return
        iw = self._last_image.width()
        ih = self._last_image.height()
        if iw < 1 or ih < 1:
            return
        self._selection_active = True
        self._selection_start = QPoint(px, py)
        self._selection_end = QPoint(px, py)
        self.update()

    def mouseMoveEvent(self, event) -> None:
        if not self._selection_active or self._last_scaled is None:
            super().mouseMoveEvent(event)
            return
        px = int(event.position().x() - self._last_pixmap_pos.x())
        py = int(event.position().y() - self._last_pixmap_pos.y())
        px = max(0, min(self._last_scaled.width() - 1, px))
        py = max(0, min(self._last_scaled.height() - 1, py))
        self._selection_end = QPoint(px, py)
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self._last_scaled is None or self._last_image is None:
            super().mouseReleaseEvent(event)
            return
        px = int(event.position().x() - self._last_pixmap_pos.x())
        py = int(event.position().y() - self._last_pixmap_pos.y())
        px = max(0, min(self._last_scaled.width() - 1, px))
        py = max(0, min(self._last_scaled.height() - 1, py))
        self._selection_end = QPoint(px, py)
        self._selection_active = False
        sel = QRect(self._selection_start, self._selection_end).normalized()
        if sel.width() < 4 or sel.height() < 4:
            # Treat tiny drag as click pick.
            iw = self._last_image.width()
            ih = self._last_image.height()
            ix = int(sel.left() * iw / self._last_scaled.width())
            iy = int(sel.top() * ih / self._last_scaled.height())
            self._selected_rect = QRect(sel.left(), sel.top(), 6, 6)
            self.image_clicked.emit(ix, iy)
            self.update()
            return
        self._selected_rect = sel
        iw = self._last_image.width()
        ih = self._last_image.height()
        x0 = int(sel.left() * iw / self._last_scaled.width())
        y0 = int(sel.top() * ih / self._last_scaled.height())
        x1 = int(sel.right() * iw / self._last_scaled.width())
        y1 = int(sel.bottom() * ih / self._last_scaled.height())
        self.image_rect_selected.emit(x0, y0, x1, y1)
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        from PySide6.QtGui import QColor, QPainter, QPen

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self._selection_active:
            r = QRect(self._selection_start, self._selection_end).normalized()
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            painter.drawRect(
                self._last_pixmap_pos.x() + r.x(),
                self._last_pixmap_pos.y() + r.y(),
                r.width(),
                r.height(),
            )
        elif self._selected_rect is not None:
            r = self._selected_rect
            painter.setPen(QPen(QColor(0, 255, 0), 2))
            painter.drawRect(
                self._last_pixmap_pos.x() + r.x(),
                self._last_pixmap_pos.y() + r.y(),
                r.width(),
                r.height(),
            )
