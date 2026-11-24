"""Cropped visual evidence for every finding (A crop, B crop, overlay, optional layer crops)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from . import align as alignmod
from .config import Config
from .models import BBox, Finding
from .overlap import separate_layers

RED = (40, 40, 230)       # BGR
GREEN = (60, 170, 40)
BLUE = (220, 120, 20)
MAX_SIDE = 900


def _crop_box(box: BBox, w: int, h: int, cfg: Config) -> Tuple[int, int, int, int]:
    mm = cfg.dpi / 25.4
    bw, bh = box[2] - box[0], box[3] - box[1]
    pad = max(4 * mm, 0.35 * max(bw, bh))
    x0, y0 = box[0] - pad, box[1] - pad
    x1, y1 = box[2] + pad, box[3] + pad
    # minimum context window of ~30 x 14 mm so tiny differences are legible
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if x1 - x0 < 30 * mm:
        x0, x1 = cx - 15 * mm, cx + 15 * mm
    if y1 - y0 < 14 * mm:
        y0, y1 = cy - 7 * mm, cy + 7 * mm
    return (int(max(0, x0)), int(max(0, y0)), int(min(w, x1)), int(min(h, y1)))


def _draw(img: np.ndarray, box: BBox, origin: Tuple[int, int], colour, dashed=False, scale=1.0, thick=2) -> None:
    x0, y0 = int((box[0] - origin[0]) * scale), int((box[1] - origin[1]) * scale)
    x1, y1 = int((box[2] - origin[0]) * scale), int((box[3] - origin[1]) * scale)
    if x1 - x0 < 4:
        x0, x1 = x0 - 2, x1 + 2
    if dashed:
        step = 10
        for x in range(x0, x1, step * 2):
            cv2.line(img, (x, y0), (min(x + step, x1), y0), colour, thick)
            cv2.line(img, (x, y1), (min(x + step, x1), y1), colour, thick)
        for y in range(y0, y1, step * 2):
            cv2.line(img, (x0, y), (x0, min(y + step, y1)), colour, thick)
            cv2.line(img, (x1, y), (x1, min(y + step, y1)), colour, thick)
    else:
        cv2.rectangle(img, (x0, y0), (x1, y1), colour, thick)


def _fit(img: np.ndarray) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    s = MAX_SIDE / float(max(h, w))
    if s < 1.0:
        return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), s
    s = min(2.0, s)
    return (cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC) if s > 1.05 else img), max(1.0, s) if s > 1.05 else 1.0


def _label(img: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((24, img.shape[1], 3), 245, np.uint8)
    cv2.putText(bar, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def make_evidence(f: Finding, key: str, out_dir: Path, bgr_a: Optional[np.ndarray], bgr_b: Optional[np.ndarray],
                  bgr_b_warped: Optional[np.ndarray], al: Optional[alignmod.Alignment], cfg: Config) -> Dict[str, str]:
    """Write evidence PNGs; returns {role: filename}. Coordinates: A-frame boxes for A, B-frame boxes for B."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files: Dict[str, str] = {}
    region = f.region or f.bbox_a
    if region is None and f.bbox_b is not None:
        region = alignmod.map_box(al.M, f.bbox_b) if al is not None else f.bbox_b
    if region is None:
        return files

    def save(role: str, img: np.ndarray) -> None:
        name = f"{key}_{role}.png"
        cv2.imwrite(str(out_dir / name), img, [cv2.IMWRITE_PNG_COMPRESSION, 4])
        files[role] = name

    # --- A crop (A frame)
    a_box = f.bbox_a if f.bbox_a is not None else region
    if bgr_a is not None:
        h, w = bgr_a.shape[:2]
        x0, y0, x1, y1 = _crop_box(region, w, h, cfg)
        crop = bgr_a[y0:y1, x0:x1].copy()
        crop, s = _fit(crop)
        if f.bbox_a is not None:
            _draw(crop, a_box, (x0, y0), RED, scale=s)
        else:
            _draw(crop, region, (x0, y0), RED, dashed=True, scale=s)
        save("a", _label(crop, "Document A" + ("" if f.bbox_a is not None else " (nothing here)")))
    # --- B crop (B frame)
    if bgr_b is not None:
        if f.bbox_b is not None:
            b_box, dashed = f.bbox_b, False
        else:
            b_box, dashed = (alignmod.map_box(al.inverse(), region) if al is not None else region), True
        union_b = alignmod.map_box(al.inverse(), region) if al is not None else region
        h, w = bgr_b.shape[:2]
        x0, y0, x1, y1 = _crop_box(union_b, w, h, cfg)
        crop = bgr_b[y0:y1, x0:x1].copy()
        crop, s = _fit(crop)
        _draw(crop, b_box, (x0, y0), GREEN, dashed=dashed, scale=s)
        save("b", _label(crop, "Document B" + (" (nothing here)" if dashed else "")))
    # --- overlay (A frame): A ink in red, B ink in green; unchanged stays dark
    if bgr_a is not None and bgr_b_warped is not None:
        h, w = bgr_a.shape[:2]
        x0, y0, x1, y1 = _crop_box(region, w, h, cfg)
        ga = cv2.cvtColor(bgr_a[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(bgr_b_warped[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        # BGR = [A, min(A,B), B]:  ink only in A -> red, ink only in B -> blue, shared ink -> black
        ov = np.dstack([ga, np.minimum(ga, gb), gb]).copy()
        ov, s = _fit(ov)
        _draw(ov, region, (x0, y0), (0, 0, 0), scale=s, thick=1)
        save("overlay", _label(ov, "Overlay: red = only in A, blue = only in B, black = both"))
    # --- layer separation crops for overlays recovered by colour
    if f.details.get("layer_separation") and bgr_b is not None and f.bbox_b is not None:
        h, w = bgr_b.shape[:2]
        x0, y0, x1, y1 = _crop_box(f.bbox_b, w, h, cfg)
        text_l, over_l = separate_layers(bgr_b[y0:y1, x0:x1])
        both = np.hstack([cv2.cvtColor(text_l, cv2.COLOR_GRAY2BGR), cv2.cvtColor(over_l, cv2.COLOR_GRAY2BGR)])
        both, _ = _fit(both)
        save("layers", _label(both, "B: text layer (left) / overlay layer (right)"))
    return files


def page_thumb(bgr: np.ndarray, out_dir: Path, name: str) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = bgr.shape[:2]
    s = 700.0 / max(h, w)
    th = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_dir / name), th)
    return name
