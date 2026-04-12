"""Reference UI colors and Qt stylesheets (RoulettePredict-style dark theme)."""

from __future__ import annotations

# Palette (from reference screenshots)
COLOR_BG = "#2B2B2B"
COLOR_BG_DEEP = "#1E1E1E"
COLOR_BG_PREVIEW = "#000000"
COLOR_TEXT = "#FFFFFF"
COLOR_TEXT_MUTED = "#CCCCCC"
COLOR_ACCENT_BLUE = "#0078D7"
COLOR_ACCENT_BLUE_HOVER = "#1C8CE8"
COLOR_CYAN = "#00E5FF"
COLOR_STATUS_GREEN = "#00FF00"
COLOR_SPEED_YELLOW = "#FFFF00"
COLOR_BORDER_SUBTLE = "#555555"
COLOR_RED_BORDER = "#FF0000"


def build_app_stylesheet(*, red_border: bool) -> str:
    border = f"3px solid {COLOR_RED_BORDER}" if red_border else "none"
    return f"""
    QMainWindow {{
        background-color: {COLOR_BG};
        border: {border};
    }}
    QWidget {{
        color: {COLOR_TEXT};
        font-family: "Segoe UI", "Arial", sans-serif;
        font-size: 10pt;
    }}
    /* Horizontal splitter (preview | controls): vertical bar between panes */
    QSplitter::handle:vertical {{
        width: 8px;
        background-color: {COLOR_BORDER_SUBTLE};
    }}
    QSplitter::handle:vertical:hover {{
        background-color: #6A6A6A;
    }}
    /* Vertical splitter if used elsewhere */
    QSplitter::handle:horizontal {{
        height: 8px;
        background-color: {COLOR_BORDER_SUBTLE};
    }}
    QSplitter::handle:horizontal:hover {{
        background-color: #6A6A6A;
    }}
    QScrollArea {{
        background-color: {COLOR_BG};
        border: none;
    }}
    QScrollArea > QWidget > QWidget {{
        background-color: {COLOR_BG};
    }}
    QTabWidget::pane {{
        border: 1px solid {COLOR_BORDER_SUBTLE};
        background-color: {COLOR_BG_DEEP};
        top: -1px;
    }}
    QTabBar::tab {{
        background-color: #3C3C3C;
        color: {COLOR_TEXT_MUTED};
        padding: 8px 18px;
        margin-right: 2px;
        border-top-left-radius: 2px;
        border-top-right-radius: 2px;
    }}
    QTabBar::tab:selected {{
        background-color: {COLOR_ACCENT_BLUE};
        color: {COLOR_TEXT};
        font-weight: bold;
    }}
    QTabBar::tab:!selected:hover {{
        background-color: #4A4A4A;
        color: {COLOR_TEXT};
    }}
    QPushButton {{
        background-color: {COLOR_ACCENT_BLUE};
        color: {COLOR_TEXT};
        border: none;
        border-radius: 2px;
        padding: 10px 16px;
        font-weight: bold;
        min-height: 22px;
    }}
    QPushButton:hover {{
        background-color: {COLOR_ACCENT_BLUE_HOVER};
    }}
    QPushButton:pressed {{
        background-color: #006CBD;
    }}
    QLabel#sectionHeader {{
        color: {COLOR_CYAN};
        font-weight: bold;
        font-size: 11pt;
        margin-top: 8px;
        margin-bottom: 6px;
    }}
    QLabel#predictionKey {{
        color: {COLOR_CYAN};
        font-weight: bold;
        font-size: 10pt;
    }}
    QLabel#predictionValueIdle {{
        color: {COLOR_TEXT_MUTED};
        font-size: 10pt;
    }}
    QLabel#predictionValueActive {{
        color: {COLOR_STATUS_GREEN};
        font-weight: bold;
        font-size: 10pt;
    }}
    QLabel#hintMuted {{
        color: {COLOR_TEXT_MUTED};
        font-size: 9pt;
    }}
    QListWidget {{
        background-color: {COLOR_BG_DEEP};
        border: 1px solid {COLOR_BORDER_SUBTLE};
        color: {COLOR_TEXT};
        padding: 4px;
    }}
    QScrollBar:vertical {{
        background: {COLOR_BG_DEEP};
        width: 10px;
        border: none;
    }}
    QScrollBar::handle:vertical {{
        background: #666666;
        min-height: 24px;
        border-radius: 2px;
    }}
    QCheckBox {{
        color: {COLOR_TEXT};
        spacing: 8px;
    }}
    QCheckBox::indicator {{
        width: 16px;
        height: 16px;
    }}
    QSlider::groove:horizontal {{
        height: 6px;
        background: #1A1A1A;
        border: 1px solid #444444;
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        background: {COLOR_CYAN};
        width: 10px;
        height: 20px;
        margin: -8px 0;
        border-radius: 2px;
        border: 1px solid #00B8CC;
    }}
    QSlider::sub-page:horizontal {{
        background: #2D4A52;
        border-radius: 2px;
    }}
    QLabel#sliderLabel {{
        color: {COLOR_TEXT};
        min-width: 36px;
        font-size: 9pt;
    }}
    QFrame#historyPanel {{
        border: 1px solid {COLOR_BORDER_SUBTLE};
        border-radius: 4px;
        background-color: {COLOR_BG_DEEP};
        margin-top: 6px;
        margin-bottom: 6px;
        padding: 10px 10px 8px 10px;
    }}
    QLabel#historyStrip {{
        color: {COLOR_TEXT};
        font-size: 13pt;
        font-family: Consolas, "Cascadia Mono", "Segoe UI", monospace;
        font-weight: bold;
        padding: 8px 4px;
        min-height: 36px;
        background-color: transparent;
    }}
    QLineEdit {{
        background-color: {COLOR_BG_DEEP};
        color: {COLOR_TEXT};
        border: 1px solid {COLOR_BORDER_SUBTLE};
        border-radius: 2px;
        padding: 4px 6px;
        selection-background-color: {COLOR_ACCENT_BLUE};
    }}
    """
