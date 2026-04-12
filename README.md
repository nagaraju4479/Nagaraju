# Roulette Predict (widget prototype)

## Stack decisions (confirmed)

| Choice | Decision | Rationale |
|--------|----------|-----------|
| **Desktop UI** | **Python + PySide6** | Matches OpenCV, fast iteration, native transparency and always-on-top. **C#/WPF** was considered for a more native Windows shell but adds friction for OpenCV interop and packaging; this repo standardizes on Python. |
| **OCR** | **Tesseract via `pytesseract`** | No heavy ML runtime (EasyOCR pulls PyTorch). Fixed ROIs benefit from classic OCR after preprocessing. Install the [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) Windows build and ensure `tesseract.exe` is on `PATH`, or set the full path under **Tesseract executable** in the app (saved as `tesseract_cmd` in `%LOCALAPPDATA%\\RoulettePredict\\config.json`). **EasyOCR** remains an optional swap if you need neural OCR quality. |

**Disclaimer:** For certified online games, outcomes are not realistically predictable from video. This tool implements UI/state-machine behavior and experimental computer-vision metrics for educational purposes only—not financial or gambling advice.

## Setup

```bash
cd D:\RouleCursorProject
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

Install Tesseract OCR and verify `tesseract --version` in a terminal.

## Run

```bash
python -m roulette_predict
```

## Tests

```bash
pytest tests/ -v
```

## Manual test checklist (multi-monitor / DPI)

See `tests/MANUAL_CHECKLIST.md`.
