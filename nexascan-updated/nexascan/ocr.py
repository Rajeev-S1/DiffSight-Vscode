"""Service 4 – PaddleOCR Large text extraction for scanned pages.

Uses PP-OCRv6 medium detection and recognition ONNX models, matching the
detection/crop/recognition workflow in large.py."""
from __future__ import annotations

import threading
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import Config
from .models import Line, Word
from .textutil import normalize_ocr_text

PADDLE_LARGE_DET = "PP-OCRv6_medium_det"
PADDLE_LARGE_REC = "PP-OCRv6_medium_rec"

_DETECTOR = None
_RECOGNIZER = None
_MODEL_LOCK = threading.Lock()
_MODEL_ERROR = None


def paddle_available() -> bool:
    try:
        from paddleocr import TextDetection, TextRecognition  # noqa: F401
        return True
    except Exception:
        return False


def _get_models():
    global _DETECTOR, _RECOGNIZER, _MODEL_ERROR
    if _DETECTOR is not None and _RECOGNIZER is not None:
        return _DETECTOR, _RECOGNIZER
    if _MODEL_ERROR is not None:
        raise RuntimeError(_MODEL_ERROR)
    with _MODEL_LOCK:
        if _DETECTOR is not None and _RECOGNIZER is not None:
            return _DETECTOR, _RECOGNIZER
        try:
            from paddleocr import TextDetection, TextRecognition
            _DETECTOR = TextDetection(model_name=PADDLE_LARGE_DET, engine="onnxruntime")
            _RECOGNIZER = TextRecognition(model_name=PADDLE_LARGE_REC, engine="onnxruntime")
        except Exception as exc:
            _MODEL_ERROR = f"PaddleOCR Large initialization failed: {exc}"
            raise RuntimeError(_MODEL_ERROR) from exc
    return _DETECTOR, _RECOGNIZER


def _unwrap(result):
    if hasattr(result, "json"):
        result = result.json
    if callable(result):
        result = result()
    if isinstance(result, dict):
        return result.get("res", result)
    return result


def _detection_polygons(result) -> List[np.ndarray]:
    result = _unwrap(result)
    if isinstance(result, dict):
        return [np.asarray(p, dtype=np.float32) for p in result.get("dt_polys", [])]
    return []


def _sort_boxes(boxes: List[np.ndarray]) -> List[np.ndarray]:
    return sorted(boxes, key=lambda box: (
        int(np.min(np.asarray(box)[:, 1]) // 15),
        int(np.min(np.asarray(box)[:, 0]))
    ))


def _crop_text_region(image: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    x0 = max(0, int(np.min(polygon[:, 0])))
    y0 = max(0, int(np.min(polygon[:, 1])))
    x1 = min(image.shape[1], int(np.max(polygon[:, 0])) + 1)
    y1 = min(image.shape[0], int(np.max(polygon[:, 1])) + 1)
    return image[y0:y1, x0:x1]


def recognize_crop(crop: np.ndarray) -> Tuple[str, float]:
    if crop is None or crop.size == 0:
        return "", 0.0
    _, recognizer = _get_models()
    results = recognizer.predict(input=crop)
    if not results:
        return "", 0.0
    item = _unwrap(results[0])
    if isinstance(item, dict):
        text = str(item.get("rec_text", "")).strip()
        try:
            conf = float(item.get("rec_score", 0.0)) * 100.0
        except (TypeError, ValueError):
            conf = 0.0
        return normalize_ocr_text(text), conf
    return "", 0.0


def ocr_image(img: np.ndarray, cfg: Config) -> Tuple[List[Word], List[Line], float]:
    """OCR one processed page image and return the existing Word/Line output contract."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    detector, _ = _get_models()
    detection_results = detector.predict(input=img)
    regions = []
    for result in detection_results or []:
        regions.extend(_detection_polygons(result))
    regions = _sort_boxes(regions)

    words: List[Word] = []
    lines: List[Line] = []
    heights = []
    for polygon in regions:
        crop = _crop_text_region(img, polygon)
        text, conf = recognize_crop(crop)
        if not text:
            continue
        x0 = max(0.0, float(np.min(polygon[:, 0])))
        y0 = max(0.0, float(np.min(polygon[:, 1])))
        x1 = min(float(img.shape[1]), float(np.max(polygon[:, 0])) + 1)
        y1 = min(float(img.shape[0]), float(np.max(polygon[:, 1])) + 1)
        bbox = (x0, y0, x1, y1)
        heights.append(y1 - y0)
        line_idx = len(lines)
        word_idx = len(words)
        words.append(Word(text=text, bbox=bbox, conf=conf, source="ocr"))
        for word in words[word_idx:]:
            word.line = line_idx
        lines.append(Line(idx=line_idx, words=list(range(word_idx, len(words))),
                          bbox=bbox, text=text, conf=conf))

    glyph_h = float(np.median(heights)) if heights else 0.0
    return words, lines, glyph_h
