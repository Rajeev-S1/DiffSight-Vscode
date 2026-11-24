"""Service 4 – OCR and text extraction for scanned pages.

Primary engine: Tesseract 5 (LSTM) - fast, word boxes + confidences, strong on punctuation.
Second opinion: RapidOCR (PP-OCR / PaddleOCR-family models on ONNX Runtime) re-reads only the
low-confidence lines; its reading is stored as an alternative so the text comparison can
ignore differences that are merely OCR disagreement."""
from __future__ import annotations

import os
import threading
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import Config
from .models import Line, Word
from .textutil import normalize_ocr_text, reading_order

os.environ.setdefault("OMP_THREAD_LIMIT", "1")  # one thread per Tesseract process; we parallelise pages

try:
    import pytesseract
except Exception:  # noqa: BLE001
    pytesseract = None


def _find_windows_tesseract() -> None:
    """Give pytesseract the exact executable path (PATH, common folders, the NexaScan-managed copy) instead of
    relying on PATH alone. See tesseract_locator.py."""
    if pytesseract is None:
        return
    try:
        from .tesseract_locator import configure
        configure(pytesseract)
    except Exception:  # noqa: BLE001 - never break OCR import because of the locator
        pass


_find_windows_tesseract()

_RAPID = None
_RAPID_LOCK = threading.Lock()
_RAPID_FAILED = False


def tesseract_available() -> bool:
    if pytesseract is None:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:  # noqa: BLE001
        return False


def _rapid(cfg: Config):
    """Lazily create the RapidOCR engine (None if not installed)."""
    global _RAPID, _RAPID_FAILED
    if _RAPID is not None or _RAPID_FAILED:
        return _RAPID
    with _RAPID_LOCK:
        if _RAPID is None and not _RAPID_FAILED:
            try:
                from rapidocr_onnxruntime import RapidOCR  # noqa: WPS433
                kw = {}
                if cfg.use_gpu:
                    kw = dict(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True)
                _RAPID = RapidOCR(**kw)
            except Exception:  # noqa: BLE001
                _RAPID_FAILED = True
    return _RAPID


def tesseract_words(img: np.ndarray, cfg: Config) -> List[dict]:
    if not tesseract_available():
        raise RuntimeError("Tesseract is not installed or not on PATH (apt install tesseract-ocr).")
    conf = f"--oem 1 --psm {cfg.ocr_psm} -c preserve_interword_spaces=1"
    d = pytesseract.image_to_data(img, lang=cfg.ocr_lang, config=conf, output_type=pytesseract.Output.DICT)
    out = []
    for i in range(len(d["text"])):
        txt = (d["text"][i] or "").strip()
        if not txt:
            continue
        try:
            c = float(d["conf"][i])
        except (TypeError, ValueError):
            c = -1.0
        if c < 0:
            continue
        x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
        if w <= 0 or h <= 0:
            continue
        out.append({"text": txt, "bbox": (float(x), float(y), float(x + w), float(y + h)), "conf": c,
                    "key": (d["block_num"][i], d["par_num"][i], d["line_num"][i])})
    return out


def _group_lines(raw: List[dict]) -> Tuple[List[Word], List[Line], float]:
    groups: dict = {}
    for r in raw:
        groups.setdefault(r["key"], []).append(r)
    pre: List[Tuple[tuple, List[Word]]] = []
    for key, ws in groups.items():
        ws.sort(key=lambda r: r["bbox"][0])
        words = []
        for r in ws:
            text = normalize_ocr_text(r["text"])
            if r["conf"] < 10 and len(text) <= 1:
                continue
            words.append(Word(text=text, bbox=r["bbox"], conf=r["conf"], source="ocr"))
        if not words:
            continue
        lb = (min(w.bbox[0] for w in words), min(w.bbox[1] for w in words),
              max(w.bbox[2] for w in words), max(w.bbox[3] for w in words))
        pre.append((lb, words))
    heights = [lb[3] - lb[1] for lb, _ in pre]
    glyph_h = float(np.median(heights)) if heights else 0.0
    ordered = reading_order(pre, lambda t: t[0], glyph_h)
    words_out: List[Word] = []
    lines: List[Line] = []
    for li, (lb, ws) in enumerate(ordered):
        ids = []
        for w in ws:
            w.line = li
            ids.append(len(words_out))
            words_out.append(w)
        wsum = sum(max(1, len(w.text)) for w in ws)
        conf = sum(w.conf * max(1, len(w.text)) for w in ws) / wsum
        lines.append(Line(idx=li, words=ids, bbox=lb, text=" ".join(w.text for w in ws), conf=float(conf)))
    return words_out, lines, glyph_h


def second_opinion(img: np.ndarray, lines: List[Line], cfg: Config) -> int:
    """Re-read low-confidence lines with RapidOCR; store differing readings as alternatives."""
    eng = _rapid(cfg)
    if eng is None:
        return 0
    n = 0
    H, W = img.shape[:2]
    for ln in lines:
        if ln.conf >= cfg.second_opinion_conf:
            continue
        x0, y0, x1, y1 = ln.bbox
        ph = max(4, int(0.25 * (y1 - y0)))
        pw = max(6, int(0.5 * (y1 - y0)))
        crop = img[max(0, int(y0) - ph):min(H, int(y1) + ph), max(0, int(x0) - pw):min(W, int(x1) + pw)]
        if crop.size == 0:
            continue
        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        try:
            res, _ = eng(crop, use_det=False, use_cls=False, use_rec=True)
        except Exception:  # noqa: BLE001
            continue
        if not res:
            continue
        try:
            txt = normalize_ocr_text(" ".join(str(r[0]) for r in res).strip())
        except Exception:  # noqa: BLE001
            continue
        if txt and txt != ln.text:
            ln.alt_texts.append(txt)
            n += 1
    return n


def ocr_image(img: np.ndarray, cfg: Config) -> Tuple[List[Word], List[Line], float]:
    """OCR one processed page image (gray or BGR). Returns words, lines (reading order), line height."""
    raw = tesseract_words(img, cfg)
    words, lines, glyph_h = _group_lines(raw)
    if cfg.second_opinion and lines:
        second_opinion(img, lines, cfg)
    return words, lines, glyph_h
