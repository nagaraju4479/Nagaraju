# Manual test checklist (multi-monitor / DPI)

Run `python -m roulette_predict` after installing dependencies and Tesseract.

## Multi-monitor

- [ ] Place the game window on the **secondary** monitor; run calibration on that monitor (fullscreen overlay should cover the monitor where you draw).
- [ ] After saving, confirm wheel and OCR crops match the intended regions (previews update).
- [ ] Move the game to another monitor without recalibrating; confirm crops are wrong until you recalibrate (expected).

## DPI scaling (Windows Display settings)

- [ ] **100%** — overlay drawing aligns with cursor; ROIs match game elements.
- [ ] **125%** or **150%** — same checks; if misaligned, note Qt/mss DPI handling may need per-monitor scaling work.

## Opacity and topmost

- [ ] During setup, lower opacity (slider) so the game remains visible through the widget; complete calibration.
- [ ] After step 3, opacity returns to 100% (main window behavior).
- [ ] With **Always on top** checked, the widget stays above the browser; uncheck and confirm it can go behind.

## Resize

- [ ] Drag the splitter and window edges; previews **letterbox** (aspect ratio preserved) without obvious corruption.

## Failure / cancel

- [ ] Start setup and press **Esc** on the overlay — calibration cancels and status returns to Idle.
- [ ] If Tesseract is not installed, OCR line stays empty and no history spam (worker should not crash).

## Performance

- [ ] CPU usage stays reasonable during ~15 FPS capture; UI remains responsive when dragging sliders.
