"""Fullscreen 3-step calibration: wheel circle, ball path (brush), OCR rectangle."""

from __future__ import annotations

import math
from enum import Enum, auto
from typing import List, Optional, Tuple

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import QWidget

from roulette_predict.config_model import CalibrationData, circle_to_dict, rect_to_dict
from roulette_predict.geometry import Circle, Rect, circle_from_drag, normalize_rect, polyline_length


# Brush stroke width in screen pixels (visual + approximate capture width).
_PATH_BRUSH_WIDTH = 14
_PATH_SAMPLE_MIN_DIST = 4.0


class CalibStep(Enum):
    WHEEL = auto()
    PATH = auto()
    OCR = auto()


class CalibrationOverlay(QWidget):
    """Frameless fullscreen overlay on one screen; draws in monitor-relative coords."""

    committed = Signal(object)  # CalibrationData
    cancelled = Signal()
    step_changed = Signal(str)

    def __init__(self, screen_geometry: QRect, monitor_index: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setGeometry(screen_geometry)
        self._monitor_index = monitor_index
        self._origin = QPoint(screen_geometry.topLeft())
        self._step = CalibStep.WHEEL
        self._wheel_drag_start: Optional[QPoint] = None
        self._wheel_drag_end: Optional[QPoint] = None
        self._wheel_circle: Optional[Circle] = None
        self._path_points: List[Tuple[float, float]] = []
        self._path_brush_down = False
        self._rect_drag_start: Optional[QPoint] = None
        self._rect_drag_end: Optional[QPoint] = None
        self._ocr_rect: Optional[Rect] = None

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.setFocus(Qt.FocusReason.PopupFocusReason)
        self.activateWindow()
        self.step_changed.emit("Step 1: Draw the wheel circle and press Enter.")

    def step(self) -> CalibStep:
        return self._step

    def _to_rel(self, g: QPoint) -> Tuple[float, float]:
        return float(g.x() - self._origin.x()), float(g.y() - self._origin.y())

    def _append_path_point(self, x: float, y: float) -> None:
        if not self._path_points:
            self._path_points.append((x, y))
            return
        lx, ly = self._path_points[-1]
        if math.hypot(x - lx, y - ly) >= _PATH_SAMPLE_MIN_DIST:
            self._path_points.append((x, y))

    def _path_valid_for_commit(self) -> bool:
        if len(self._path_points) < 2:
            return False
        return polyline_length(self._path_points) >= 18.0 or len(self._path_points) >= 25

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 120))
        hint = "Step 1/3: Drag from wheel center outward (release to set radius)"
        if self._step == CalibStep.PATH:
            hint = "Step 2/3: Hold left button and paint ball path (brush); Enter when done"
        elif self._step == CalibStep.OCR:
            hint = (
                "Step 3/3: Tight rectangle around the newest number cell only (not the whole row); Enter to save"
            )
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawText(20, 30, hint)
        painter.setPen(QPen(QColor(0, 255, 0), 2))
        if self._wheel_drag_start and self._wheel_drag_end and self._step == CalibStep.WHEEL:
            c = circle_from_drag(
                self._wheel_drag_start.x() - self._origin.x(),
                self._wheel_drag_start.y() - self._origin.y(),
                self._wheel_drag_end.x() - self._origin.x(),
                self._wheel_drag_end.y() - self._origin.y(),
            )
            painter.drawEllipse(
                int(self._origin.x() + c.cx - c.r),
                int(self._origin.y() + c.cy - c.r),
                int(2 * c.r),
                int(2 * c.r),
            )
        elif self._wheel_circle:
            c = self._wheel_circle
            painter.drawEllipse(
                int(self._origin.x() + c.cx - c.r),
                int(self._origin.y() + c.cy - c.r),
                int(2 * c.r),
                int(2 * c.r),
            )

        if self._step == CalibStep.PATH:
            pen = QPen(QColor(255, 0, 0, 210))
            pen.setWidth(_PATH_BRUSH_WIDTH)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            if len(self._path_points) >= 2:
                for i in range(len(self._path_points) - 1):
                    x0, y0 = self._path_points[i]
                    x1, y1 = self._path_points[i + 1]
                    painter.drawLine(
                        int(self._origin.x() + x0),
                        int(self._origin.y() + y0),
                        int(self._origin.x() + x1),
                        int(self._origin.y() + y1),
                    )
        elif self._step == CalibStep.OCR:
            if self._rect_drag_start and self._rect_drag_end:
                r = normalize_rect(
                    self._rect_drag_start.x() - self._origin.x(),
                    self._rect_drag_start.y() - self._origin.y(),
                    self._rect_drag_end.x() - self._origin.x(),
                    self._rect_drag_end.y() - self._origin.y(),
                )
                painter.setPen(QPen(QColor(255, 255, 255), 2))
                painter.drawRect(
                    int(self._origin.x() + r.x),
                    int(self._origin.y() + r.y),
                    int(r.w),
                    int(r.h),
                )
            elif self._ocr_rect:
                r = self._ocr_rect
                painter.setPen(QPen(QColor(255, 255, 255), 2))
                painter.drawRect(int(self._origin.x() + r.x), int(self._origin.y() + r.y), int(r.w), int(r.h))

    def mousePressEvent(self, e: QMouseEvent) -> None:
        if e.button() != Qt.MouseButton.LeftButton:
            return
        g = e.globalPosition().toPoint()
        if self._step == CalibStep.WHEEL:
            self._wheel_drag_start = g
            self._wheel_drag_end = g
        elif self._step == CalibStep.PATH:
            self._path_brush_down = True
            x, y = self._to_rel(g)
            self._append_path_point(x, y)
        elif self._step == CalibStep.OCR:
            self._rect_drag_start = g
            self._rect_drag_end = g
        self.update()

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        g = e.globalPosition().toPoint()
        if self._step == CalibStep.WHEEL and self._wheel_drag_start:
            self._wheel_drag_end = g
            self.update()
        elif self._step == CalibStep.PATH and self._path_brush_down:
            x, y = self._to_rel(g)
            self._append_path_point(x, y)
            self.update()
        elif self._step == CalibStep.OCR and self._rect_drag_start:
            self._rect_drag_end = g
            self.update()

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        if e.button() != Qt.MouseButton.LeftButton:
            return
        g = e.globalPosition().toPoint()
        if self._step == CalibStep.WHEEL and self._wheel_drag_start:
            self._wheel_drag_end = g
            x0, y0 = self._wheel_drag_start.x() - self._origin.x(), self._wheel_drag_start.y() - self._origin.y()
            x1, y1 = g.x() - self._origin.x(), g.y() - self._origin.y()
            self._wheel_circle = circle_from_drag(float(x0), float(y0), float(x1), float(y1))
            self._wheel_drag_start = None
            self._wheel_drag_end = None
        elif self._step == CalibStep.PATH and self._path_brush_down:
            self._path_brush_down = False
            x, y = self._to_rel(g)
            self._append_path_point(x, y)
        elif self._step == CalibStep.OCR and self._rect_drag_start:
            self._rect_drag_end = g
            x0, y0 = self._rect_drag_start.x() - self._origin.x(), self._rect_drag_start.y() - self._origin.y()
            x1, y1 = g.x() - self._origin.x(), g.y() - self._origin.y()
            self._ocr_rect = normalize_rect(float(x0), float(y0), float(x1), float(y1))
            self._rect_drag_start = None
            self._rect_drag_end = None
        self.update()

    def keyPressEvent(self, e) -> None:
        from PySide6.QtGui import QKeyEvent

        if not isinstance(e, QKeyEvent):
            return
        if e.key() == Qt.Key.Key_Escape:
            self.cancelled.emit()
            self.close()
            return
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self._step == CalibStep.WHEEL:
                if self._wheel_circle:
                    self._step = CalibStep.PATH
                    self.step_changed.emit("Step 2: Paint the ball path with the brush and press Enter.")
                    self.update()
            elif self._step == CalibStep.PATH:
                if self._path_valid_for_commit():
                    self._step = CalibStep.OCR
                    self.step_changed.emit(
                        "Step 3: Tight box around only the newest history cell (usually leftmost), then Enter."
                    )
                    self.update()
            elif self._step == CalibStep.OCR:
                self._commit()
        super().keyPressEvent(e)

    def _commit(self) -> None:
        if self._rect_drag_start and self._rect_drag_end:
            x0, y0 = self._rect_drag_start.x() - self._origin.x(), self._rect_drag_start.y() - self._origin.y()
            x1, y1 = self._rect_drag_end.x() - self._origin.x(), self._rect_drag_end.y() - self._origin.y()
            self._ocr_rect = normalize_rect(float(x0), float(y0), float(x1), float(y1))
        if not self._wheel_circle or not self._path_valid_for_commit() or not self._ocr_rect:
            return
        cal = CalibrationData(
            monitor_index=self._monitor_index,
            wheel_circle=circle_to_dict(self._wheel_circle),
            ball_path_points=[[float(p[0]), float(p[1])] for p in self._path_points],
            ocr_rect=rect_to_dict(self._ocr_rect),
        )
        self.committed.emit(cal)
        self.close()
