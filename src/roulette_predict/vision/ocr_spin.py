"""Tesseract OCR on result ROI + debounced spin detection."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Same digit still visible in Step-3 ROI after a brief OCR miss used to re-log immediately.
# Require this many seconds between two history entries with the **same** value (blocks glitches;
# two real spins both e.g. 32 are usually farther apart than this on live/auto wheels).
SAME_NUMBER_REPEAT_MIN_SEC = 10.0


def preprocess_ocr_roi(bgr: np.ndarray) -> np.ndarray:
    import cv2

    if bgr.size == 0:
        return bgr
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def parse_roulette_int(text: str) -> Optional[int]:
    digits = re.sub(r"[^\d]", "", text)
    if not digits:
        return None
    try:
        v = int(digits)
    except ValueError:
        return None
    if 0 <= v <= 36:
        return v
    return None


def _digit_run_len(text: str) -> int:
    return len(re.sub(r"[^\d]", "", text))


def _pick_best_attempt(attempts: list[tuple[str, Optional[int]]]) -> Optional[tuple[str, int]]:
    """Prefer readings with more digit characters (e.g. 13 over 3). Returns winning raw OCR text + value."""
    valid: list[tuple[str, int]] = []
    for raw, v in attempts:
        if v is None:
            continue
        r = (raw or "").strip()
        if not r:
            continue
        valid.append((r, v))
    if not valid:
        return None

    def score(item: tuple[str, int]) -> tuple[int, int]:
        raw, _ = item
        return (_digit_run_len(raw), len(raw))

    return max(valid, key=score)


def _pick_best_parse(attempts: list[tuple[str, Optional[int]]]) -> Optional[int]:
    b = _pick_best_attempt(attempts)
    return b[1] if b else None


def tesseract_digits(bgr: np.ndarray, tesseract_cmd: Optional[str] = None) -> str:
    import pytesseract
    from pytesseract import TesseractNotFoundError

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    proc = preprocess_ocr_roi(bgr)
    try:
        cfg = "--psm 7 -c tessedit_char_whitelist=0123456789"
        return pytesseract.image_to_string(proc, config=cfg) or ""
    except TesseractNotFoundError:
        return ""


def is_tesseract_available(tesseract_cmd: Optional[str] = None) -> bool:
    import pytesseract
    from pytesseract import TesseractNotFoundError

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    try:
        _ = pytesseract.get_tesseract_version()
        return True
    except (TesseractNotFoundError, OSError):
        return False


def _pad_patch_for_ocr(bgr_patch: np.ndarray, px: int = 8) -> np.ndarray:
    """Border padding helps Tesseract on tight crops (white “1” at cell edge)."""
    import cv2

    if bgr_patch.size == 0 or px <= 0:
        return bgr_patch
    return cv2.copyMakeBorder(bgr_patch, px, px, px, px, cv2.BORDER_REPLICATE)


def _ocr_image_variants(bgr_patch: np.ndarray, *, quick: bool) -> list[np.ndarray]:
    """Binary / grayscale images for Tesseract.

    White-on-dark and red-on-dark often differ in luminance; max(R,G,B) keeps both bright vs background.
    HSV **V** boosts pale white glyphs; horizontal dilate thickens thin vertical strokes (e.g. “1” in 13).
    """
    import cv2

    if bgr_patch.size == 0:
        return []
    bgr_patch = _pad_patch_for_ocr(bgr_patch, 8)
    b, g, r = cv2.split(bgr_patch)
    mx = cv2.max(cv2.max(r, g), b)
    # Red digits on dark/red cells: R dominates G and B (white digits still have high mx).
    r_dom = np.clip(r.astype(np.int32) * 2 - g.astype(np.int32) - b.astype(np.int32), 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
    _, _, vchan = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    scale = 4.2 if quick else 4.5
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    h_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    variants: list[np.ndarray] = []

    def add_from_base(base: np.ndarray, *, use_plain_big: bool) -> None:
        big = cv2.resize(base, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        big = cv2.GaussianBlur(big, (3, 3), 0)
        bc = clahe.apply(big)
        gimgs = (bc, big) if use_plain_big else (bc,)
        for gimg in gimgs:
            _, th_ot = cv2.threshold(gimg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, th_oi = cv2.threshold(gimg, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            variants.append(th_ot)
            variants.append(th_oi)
            variants.append(cv2.morphologyEx(th_ot, cv2.MORPH_CLOSE, kern))
            variants.append(cv2.dilate(th_ot, h_dilate, iterations=1))

    # max(R,G,B) first — often best for white + red on dark pills
    add_from_base(mx, use_plain_big=not quick)
    add_from_base(r_dom, use_plain_big=False)
    add_from_base(gray, use_plain_big=not quick)
    add_from_base(vchan, use_plain_big=False)
    if not quick:
        add_from_base(255 - gray, use_plain_big=True)
        big = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        big = cv2.GaussianBlur(big, (3, 3), 0)
        bigc = clahe.apply(big)
        variants.append(
            cv2.adaptiveThreshold(
                bigc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 23, -2
            )
        )
    return variants


def _read_roulette_from_patch(
    bgr_patch: np.ndarray,
    tesseract_cmd: Optional[str],
    *,
    quick: bool = False,
) -> tuple[str, Optional[int]]:
    """Returns (best raw OCR text, value). Raw text is required for debouncing two-digit reads."""
    import pytesseract
    from pytesseract import TesseractNotFoundError

    if bgr_patch.size == 0:
        return "", None
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def ocr(img: np.ndarray, psm: int) -> str:
        try:
            cfg = f"--psm {psm} -c tessedit_char_whitelist=0123456789"
            return pytesseract.image_to_string(img, config=cfg) or ""
        except TesseractNotFoundError:
            return ""

    candidates = _ocr_image_variants(bgr_patch, quick=quick)
    psms = (7, 8, 6, 10, 13) if not quick else (7, 8)
    attempts: list[tuple[str, Optional[int]]] = []

    if quick:
        # Speed: try PSM 7 on all images first; stop early on a confident two-digit read.
        for img in candidates:
            raw = ocr(img, 7)
            attempts.append((raw, parse_roulette_int(raw)))
            best = _pick_best_attempt(attempts)
            if best is not None and _digit_run_len(best[0]) >= 2:
                return best[0], best[1]
        for img in candidates:
            raw = ocr(img, 8)
            attempts.append((raw, parse_roulette_int(raw)))
            best = _pick_best_attempt(attempts)
            if best is not None and _digit_run_len(best[0]) >= 2:
                return best[0], best[1]
        best = _pick_best_attempt(attempts)
        return (best[0], best[1]) if best else ("", None)

    for img in candidates:
        for psm in psms:
            raw = ocr(img, psm)
            attempts.append((raw, parse_roulette_int(raw)))
    best = _pick_best_attempt(attempts)
    return (best[0], best[1]) if best else ("", None)


def read_roulette_number_from_roi(bgr: np.ndarray, tesseract_cmd: Optional[str] = None) -> Optional[int]:
    """Read 0–36 from ROI. Wide strips (history row) bias to the **left** — newest number is usually first."""
    if bgr.size == 0:
        return None

    h, w = bgr.shape[:2]
    if w < 8 or h < 4:
        _, v = _read_roulette_from_patch(bgr, tesseract_cmd)
        return v

    # Tight box (single number): OCR the full crop only — avoid left-fraction tricks that pick wrong digits.
    strip_like = w > max(72, int(h * 1.35))
    if not strip_like:
        _, v = _read_roulette_from_patch(bgr, tesseract_cmd)
        return v

    # Horizontal history: crop from left only; avoid digits from older results on the right.
    if w > h * 1.15:
        for frac in (0.30, 0.28, 0.32, 0.26, 0.35, 0.38, 0.22, 0.40, 0.48, 0.55, 0.65, 1.0):
            cw = max(18, int(w * frac))
            patch = bgr[:, :cw]
            _, v = _read_roulette_from_patch(patch, tesseract_cmd)
            if v is not None:
                return v

        col_w = max(14, int(h * 0.9))
        x_step = max(6, col_w // 2)
        for x0 in range(0, min(w, col_w * 5), x_step):
            x1 = min(w, x0 + col_w + 8)
            patch = bgr[:, x0:x1]
            if patch.shape[1] < 10:
                continue
            _, v = _read_roulette_from_patch(patch, tesseract_cmd)
            if v is not None:
                return v

    _, v = _read_roulette_from_patch(bgr, tesseract_cmd)
    return v


def read_roulette_number_live_fast(bgr: np.ndarray, tesseract_cmd: Optional[str] = None) -> tuple[str, Optional[int]]:
    """OCR for the live worker. Returns **raw Tesseract text** for the chosen parse (not only str(v))."""
    if bgr.size == 0:
        return "", None
    h, w = bgr.shape[:2]
    strip_like = w > max(72, int(h * 1.35))
    patch = bgr
    # Left crop: wide enough for two digits + thin “1”; too narrow drops the leading stroke on white numbers.
    if strip_like and w > max(48, int(h * 1.25)):
        cw = max(22, min(int(w * 0.44), int(h * 3.0)))
        patch = bgr[:, :cw]
    return _read_roulette_from_patch(patch, tesseract_cmd, quick=True)


@dataclass
class SpinDebouncer:
    """Debounce OCR noise vs real new spins.

    • **First value ever** (``last_emitted is None``): emit on first clean parse (setup / cold start).
    • **New digit** after that: if raw OCR has **two** digit chars (10–36), emit on **one** read (less lag).
      Single-digit reads still need **two** consecutive parses (filters bogus “8” between 6 and 4).
    • **Same digit again** (cell still shows the last logged number): never spam duplicates; only emit
      the same value twice if ``SAME_NUMBER_REPEAT_MIN_SEC`` has passed **and** two reads agree
      (covers the next spin landing on the same number, rare).
    """

    pending_value: Optional[int] = None
    pending_count: int = 0
    last_emitted: Optional[int] = None
    last_emit_time: float = 0.0

    # Compatibility for seed path / older code
    @property
    def last_recorded(self) -> Optional[int]:
        return self.last_emitted

    @last_recorded.setter
    def last_recorded(self, v: Optional[int]) -> None:
        self.last_emitted = v

    def feed(self, raw_text: str, now: Optional[float] = None) -> Optional[int]:
        now = now or time.perf_counter()
        v = parse_roulette_int(raw_text)
        if v is None:
            self.pending_value = None
            self.pending_count = 0
            return None
        if v == self.pending_value:
            self.pending_count += 1
        else:
            self.pending_value = v
            self.pending_count = 1

        if v != self.last_emitted:
            if self.last_emitted is None:
                return self._emit(v, now)
            dlen = _digit_run_len(raw_text)
            if dlen >= 2:
                return self._emit(v, now)
            if self.pending_count >= 2:
                return self._emit(v, now)
            return None

        # Same number still showing — do not log again until cooldown (stops duplicate 32 after OCR glitch).
        if self.pending_count < 2:
            return None
        if now - self.last_emit_time < SAME_NUMBER_REPEAT_MIN_SEC:
            return None
        return self._emit(v, now)

    def _emit(self, v: int, now: float) -> int:
        self.last_emitted = v
        self.last_emit_time = now
        self.pending_value = None
        self.pending_count = 0
        return v
