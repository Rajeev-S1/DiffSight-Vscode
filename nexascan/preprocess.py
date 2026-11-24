"""Service 3 – Image preprocessing: orientation, skew, background flattening, ink masks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from .config import Config

try:
    import pytesseract
except Exception:  # noqa: BLE001
    pytesseract = None


@dataclass
class Prepared:
    """Pixel layers of one processed page (all uint8, same size)."""
    bgr: np.ndarray
    gray: np.ndarray       # luminance of the page as rendered
    norm: np.ndarray       # illumination-flattened luminance (== gray for clean digital renders)
    ink: np.ndarray        # bool: dark OR coloured ink
    sat: np.ndarray        # HSV saturation
    noise: float


def to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


# ----------------------------------------------------------------------------- orientation


def estimate_orientation(gray: np.ndarray) -> Tuple[int, float]:
    """Return (clockwise degrees to apply so text is upright, confidence).
    Uses Tesseract OSD; returns (0, 0.0) when it cannot decide."""
    if pytesseract is None:
        return 0, 0.0
    h, w = gray.shape
    s = 1600.0 / max(h, w)
    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    try:
        out = pytesseract.image_to_osd(small, config="--psm 0 -c min_characters_to_try=30",
                                       output_type=pytesseract.Output.DICT)
    except Exception:  # noqa: BLE001 - OSD raises on pages with too little text
        return 0, 0.0
    rot = int(out.get("rotate", 0)) % 360
    conf = float(out.get("orientation_conf", 0.0))
    if rot == 0:
        return 0, conf
    return (rot, conf) if conf >= 2.5 else (0, conf)


# ----------------------------------------------------------------------------- skew


def estimate_skew(gray: np.ndarray, max_angle: float = 6.0) -> Tuple[float, float]:
    """Projection-profile skew estimate. Returns (degrees to rotate CCW to straighten, confidence)."""
    h, w = gray.shape
    s = 1000.0 / max(h, w)
    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    _, bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if (bw > 0).mean() < 0.004:
        return 0.0, 0.0
    h2, w2 = bw.shape
    centre = (w2 / 2.0, h2 / 2.0)

    def score(angle: float) -> float:
        M = cv2.getRotationMatrix2D(centre, angle, 1.0)
        r = cv2.warpAffine(bw, M, (w2, h2), flags=cv2.INTER_NEAREST)
        prof = r.sum(axis=1, dtype=np.float64)
        return float(np.sum(np.diff(prof) ** 2))

    base = score(0.0)
    if base <= 0:
        return 0.0, 0.0
    coarse = np.arange(-max_angle, max_angle + 1e-6, 0.5)
    best = max(coarse, key=score)
    fine = np.arange(best - 0.5, best + 0.5 + 1e-6, 0.05)
    best = max(fine, key=score)
    conf = (score(best) - base) / base
    return float(best), float(conf)


# ----------------------------------------------------------------------------- tone / noise


def flatten_background(gray: np.ndarray) -> np.ndarray:
    """Divide out slowly varying illumination (shadows, paper tint) of scans."""
    h, w = gray.shape
    s = 4
    small = cv2.resize(gray, (max(1, w // s), max(1, h // s)), interpolation=cv2.INTER_AREA)
    k = max(3, int(0.012 * max(h, w) / s)) | 1
    bg = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 2.0)
    paper = float(np.percentile(small, 90))
    bg = np.maximum(bg, 0.65 * paper)       # never "brighten" genuinely dark solid areas away
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    out = gray.astype(np.float32) / np.maximum(bg, 1.0) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def estimate_noise(gray: np.ndarray) -> float:
    """Std-dev (grey levels) of the paper background; ~0 for clean renders, 1-6 for scans."""
    h, w = gray.shape
    s = 1500.0 / max(h, w)
    g = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    med = cv2.medianBlur(g, 3)
    resid = g.astype(np.int16) - med.astype(np.int16)
    flat = (cv2.dilate(g, np.ones((5, 5), np.uint8)).astype(np.int16)
            - cv2.erode(g, np.ones((5, 5), np.uint8)).astype(np.int16)) < 30
    bright = g > np.percentile(g, 40)
    m = flat & bright
    if m.sum() < 500:
        return 0.0
    return float(np.std(resid[m]))


def estimate_contrast(gray: np.ndarray) -> float:
    """Ink-to-paper contrast in grey levels: paper (median of the brighter half) minus ink (1st percentile).
    ~180-230 for a good scan, below ~110 for a faded one. Page-relative: a pale but evenly inked page is not 'noisy'."""
    g = gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    paper = float(np.median(g[g >= np.median(g)]))
    ink = float(np.percentile(g, 1.0))
    return max(0.0, paper - ink)


def estimate_sharpness(gray: np.ndarray, max_side: int = 2800) -> float:
    """Edge sharpness of the page = 100 / (edge width in pixels), measured on the strongest text edges
    (gradient / ink-to-paper contrast, so it does not depend on contrast or font size).
    A crisp edge is 1-2 px wide (score ~50-100); blur spreads every edge (score falls towards 10)."""
    g = gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    if max(h, w) > max_side:                            # only ever shrink (cost); never up-sample (it would blur)
        sc = max_side / float(max(h, w))
        g = cv2.resize(g, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    contrast = estimate_contrast(g)
    if contrast < 20:
        return 0.0
    f = g.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    mag = np.sqrt(gx * gx + gy * gy)
    thr = float(np.percentile(mag, 97.0))
    strong = mag[mag >= max(thr, 1e-3)]
    if strong.size < 50:
        return 0.0
    return float(np.median(strong) / contrast * 100.0)


def ink_mask(norm: np.ndarray, sat: np.ndarray, scan: bool) -> np.ndarray:
    """Dark ink or strongly coloured ink (stamps, pens, coloured text)."""
    if scan:
        t, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        thr = float(np.clip(t, 120, 200))
    else:
        thr = 170.0
    ink = norm < thr
    ink |= (sat > 60) & (norm < 235)
    return ink


def prepare(bgr: np.ndarray, scan: bool) -> Prepared:
    gray = to_gray(bgr)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    noise = estimate_noise(gray) if scan else 0.0
    norm = flatten_background(gray) if scan else gray
    ink = ink_mask(norm, sat, scan)
    return Prepared(bgr=bgr, gray=gray, norm=norm, ink=ink, sat=sat, noise=noise)
