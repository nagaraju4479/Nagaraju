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

## Recent development (changelog for maintainers)

This section summarizes **implementation choices and behavior** added or changed so future work (or AI sessions) can pick up context quickly.

### HSV tuning vs ball visibility

- **Single source of truth:** `VisionWorker` uses the **Color tab HSV sliders** (`set_hsv`) for both the mask preview and ball tracking.
- **Manual ball color (after darkening the mask):** If you lower **L-S** / **L-V** until glints and the ball disappear from the mask, use **Pick ball color (click wheel)** on the **Roulette** tab: click on the ball. The app samples a small patch from the **ball-path screen grab**, derives an HSV box from the median color (see `vision/hsv_sample.py`), updates all sliders, and persists. That brings the ball back into range without re-opening the cone to every glint.
- **Why the ball might look “gone” in the mask:** HSV thresholds pixels by **hue / saturation / value**. If you tune so the **felt/background** is black and the **path** is white, a **white ball** often shares similar **V** (brightness) to the path — it should still appear **in-range** if **H** and **S** bands include the ball. If the ball disappears, **widen** `L-H`–`U-H` or raise **L-V** / **U-V** so the ball’s pixels stay inside the cone.
- **When color is ambiguous:** The tracker also uses **dense optical flow** (Farneback) on the ball-path ROI, so **motion** on the tube can still localize the ball even when static HSV would confuse ball vs wood.
- **Design tradeoff:** A pure “grayscale / white path only” mask is not a separate mode — the pipeline is **HSV + flow + tube geometry**, not ML segmentation.

**Lighting / specular (bright arcs and dots):** Glare is often **elongated** or **clipped to full white**; the ball stays more **round** and **stable frame-to-frame**. The tracker **biases toward the last good position** so random bright specks lose. On the **Color** tab, after a lock-on, the preview shows **only the tracked blob** (other HSV hits are blacked) to reduce visual clutter — tune **U-S**, **L-S** (try **~15–40** to drop desaturated glints), and **L-V / U-V** so the ball stays in-range without flooding the whole rim.

### Roulette tab preview quality / FPS

- Wheel preview is **downscaled** to `WHEEL_PREVIEW_MAX_W` (currently **1280** px width) before display and wheel HSV mask; the **ball-path crop** used for tracking stays **full resolution** from the screen grab.
- `RoulettePreviewFrame.rescale_frame()` uses **`FastTransformation`** (cheaper scaling than smooth) for snappier UI updates.
- Tracking runs **HSV centroid + optical flow + masks + speed** every frame when calibration is complete; effective FPS is limited by **single-thread** `_process_frame` cost.

### Tracker pipeline tuning (`VisionWorker`)

- **Farneback:** `vision/ball_flow.py` uses tuned `FARNEBACK_PYR_LEVELS` / `FARNEBACK_WINSIZE` (lighter than OpenCV defaults) for CPU budget.
- **Flow decimation:** dense flow runs every **`_FLOW_DECIMATE_STRIDE`** frames (2); other frames rely on HSV + orbit extrapolation when applicable.
- **Geometry:** `centroid_near_step2_track_mask` uses **`_TRACK_MASK_NEAR_RADIUS_PX`** (8) so centroids slightly off the painted stroke are not dropped as often.
- **Extrapolation:** when flow is weak, orbit extrapolation can run for up to **7** consecutive frames (see `_extrap_frames_used` cap in worker).

### GUI: fewer dropped-looking updates

- `main_window.py` **coalesces** `frame_wheel` / `frame_color` handling: latest image is stored and applied with `QTimer.singleShot(0, …)` so the UI thread does not run **many expensive rescales** when the worker queues ahead of paint.

### Overlay rings (flicker reduction)

- **Green** ring: **current track** when the centroid is trusted.
- **Orange**: coasting / uncertain — after several consecutive missed centroids, while still showing the ring at the **last good** position.
- **Coasting:** when `cent` is lost, the ring stays at **`_trusted_ball_cent`** for **`_RING_COAST_FRAMES`** (~32) frames before disappearing.
- **Color hysteresis:** green vs orange shifts after **`_ORANGE_RING_AFTER_MISS_FRAMES`** (6) consecutive misses.
- **Speed line:** `speed_text` uses **`_emit_speed_overlay`** (deduplicated emits).

### Tests

- `tests/test_ball_roi_map.py` — forward/inverse mapping between wheel preview pixels and ball-path ROI (`_wheel_preview_xy_to_ball_roi_xy` / `ball_roi_xy_to_wheel_preview_xy`).
- `tests/test_hsv_sample.py` — `hsv_settings_from_bgr_patch_median` clamps and range sanity on a solid-color patch.

### Not in scope (documented for clarity)

- **YOLO / detectors:** the stack remains classical CV + optional ML discussion; adding a small detector on the ball ROI would be a separate feature (dataset, ONNX/PyTorch, fusion with existing masks).

## Manual test checklist (multi-monitor / DPI)

See `tests/MANUAL_CHECKLIST.md`.
