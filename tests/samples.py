"""Synthetic test-document factory with *known* injected differences (ground truth).

Everything is generated procedurally (PyMuPDF for digital PDFs, OpenCV/Pillow for scan
degradation, stamps and signatures), so the benchmark never relies on fabricated detections."""
from __future__ import annotations

import io
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pymupdf

A4 = (595.0, 842.0)
LETTER = (612.0, 792.0)

BASE_PAGE1 = [
    ("NexaScan Services Agreement", 16, True),
    ("", 11, False),
    ("This agreement is made on 12-11-2026 between Alpha Traders Pvt Ltd and Beta Logistics.", 11, False),
    ("Payment amount is Rs. 10,000. Approved.", 11, False),
    ("The supplier shall deliver the goods within thirty days of the order date.", 11, False),
    ("Invoice No. INV-2026/0045 dated 05-11-2026 for a total of 1,25,000.50 is attached.", 11, False),
    ("All disputes are subject to the jurisdiction of the courts at Bengaluru.", 11, False),
    ("Contact: accounts@alpha-traders.example; phone +91 98765 43210.", 11, False),
    ("", 11, False),
    ("Terms: payment within 15 days (net), late fee applies, no refunds after delivery.", 11, False),
    ("The buyer confirms receipt of the goods in good condition.", 11, False),
]
BASE_PAGE2 = [
    ("Schedule of Deliverables", 14, True),
    ("", 11, False),
    ("Item 1: Packaging material, quantity 500 units.", 11, False),
    ("Item 2: Labels, quantity 1,200 units.", 11, False),
    ("Item 3: Pallets, quantity 40 units.", 11, False),
    ("", 11, False),
    ("Authorised signatory for Alpha Traders Pvt Ltd:", 11, False),
]
BASE_PAGE3 = [
    ("Annexure A", 14, True),
    ("", 11, False),
    ("The delivery location is Warehouse 7, Peenya Industrial Area, Bengaluru 560058.", 11, False),
    ("Working hours are 9 am to 6 pm, Monday to Saturday.", 11, False),
]


def _draw_lines(page, lines, x=72, y=80, lh=22, font="helv"):
    pos = []
    for text, size, bold in lines:
        if text:
            fn = "hebo" if bold else font
            page.insert_text((x, y), text, fontsize=size, fontname=fn)
            w = pymupdf.get_text_length(text, fontname=fn, fontsize=size)
            pos.append((text, (x, y - size, x + w, y + size * 0.3)))
        y += lh if size <= 12 else lh + 6
    return pos


def apply_replacements(lines, repl: Dict[str, str]):
    out = []
    for text, size, bold in lines:
        for k, v in repl.items():
            if k in text:
                text = text.replace(k, v)
        out.append((text, size, bold))
    return out


def stamp_png(text="APPROVED", color=(30, 60, 200), size=170, angle=0.0, seed=1) -> np.ndarray:
    """RGBA stamp: double ring + text, slightly rough like ink."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 4), np.uint8)
    c = size // 2
    col = (*color, 255)
    cv2.circle(img, (c, c), c - 6, col, 4, cv2.LINE_AA)
    cv2.circle(img, (c, c), c - 18, col, 2, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
    sc = (size * 0.62) / tw
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.9 * sc, max(1, int(2 * sc)))
    cv2.putText(img, text, (c - tw // 2, c + th // 2), cv2.FONT_HERSHEY_DUPLEX, 0.9 * sc, col,
                max(1, int(2 * sc)), cv2.LINE_AA)
    cv2.putText(img, "* 2026 *", (c - 28, c + th + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    noise = rng.random((size, size)) < 0.06
    img[..., 3][noise] = (img[..., 3][noise] * 0.4).astype(np.uint8)
    if angle:
        M = cv2.getRotationMatrix2D((c, c), angle, 1.0)
        img = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    return img


def signature_png(seed=3, w=260, h=90, color=(20, 20, 90)) -> np.ndarray:
    """RGBA handwriting-like scribble from a random smooth curve (synthetic, not a real signature)."""
    rng = random.Random(seed)
    img = np.zeros((h, w, 4), np.uint8)
    pts = []
    x = 10.0
    t = 0.0
    while x < w - 20:
        y = h / 2 + math.sin(t * rng.uniform(1.6, 2.4)) * rng.uniform(14, 30) + rng.uniform(-6, 6)
        pts.append((x, y))
        x += rng.uniform(5, 11)
        t += 1.0
    pts_np = np.array(pts, np.float32)
    # smooth through Catmull-Rom-ish by resampling
    xs = np.interp(np.linspace(0, len(pts_np) - 1, len(pts_np) * 8), np.arange(len(pts_np)), pts_np[:, 0])
    ys = np.interp(np.linspace(0, len(pts_np) - 1, len(pts_np) * 8), np.arange(len(pts_np)), pts_np[:, 1])
    ys = cv2.GaussianBlur(ys.reshape(-1, 1), (0, 0), 3).ravel()
    poly = np.stack([xs, ys], 1).astype(np.int32)
    cv2.polylines(img, [poly], False, (*color, 255), 2, cv2.LINE_AA)
    cv2.line(img, (12, int(h * 0.82)), (w - 30, int(h * 0.74)), (*color, 255), 2, cv2.LINE_AA)
    return img


def blit_rgba(page_bgr: np.ndarray, rgba: np.ndarray, x: int, y: int, opacity: float = 1.0) -> Tuple[int, int, int, int]:
    h, w = rgba.shape[:2]
    H, W = page_bgr.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return (x0, y0, x0, y0)
    sub = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    alpha = (sub[..., 3:4].astype(np.float32) / 255.0) * opacity
    rgb = sub[..., :3][:, :, ::-1].astype(np.float32)  # RGBA(rgb) -> BGR
    dst = page_bgr[y0:y1, x0:x1].astype(np.float32)
    page_bgr[y0:y1, x0:x1] = (dst * (1 - alpha) + rgb * alpha).astype(np.uint8)
    return (x0, y0, x1, y1)


def render_pdf(path: Path, pages: List[List[tuple]], size=A4, rotate_pages: Optional[Dict[int, int]] = None,
               extra=None) -> List[List[Tuple[str, tuple]]]:
    """Write a digital PDF. ``extra(page_index, page)`` may draw additional vector content."""
    doc = pymupdf.open()
    positions = []
    for i, lines in enumerate(pages):
        page = doc.new_page(width=size[0], height=size[1])
        positions.append(_draw_lines(page, lines))
        if extra:
            extra(i, page)
        if rotate_pages and i in rotate_pages:
            page.set_rotation(rotate_pages[i])
    doc.save(str(path))
    doc.close()
    return positions


def pdf_to_images(path: Path, dpi=300) -> List[np.ndarray]:
    doc = pymupdf.open(str(path))
    out = []
    for p in doc:
        pix = p.get_pixmap(dpi=dpi, alpha=False, colorspace=pymupdf.csRGB)
        arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
        out.append(np.ascontiguousarray(arr[:, :, ::-1]))
    doc.close()
    return out


def degrade_scan(bgr: np.ndarray, angle=0.0, noise=6.0, blur=0.8, jpeg=55, shift=(0, 0), scale=1.0,
                 brightness=0.0, shadow=0.0, seed=0) -> np.ndarray:
    """Simulate a physical scan: skew, shift, blur, sensor noise, uneven lighting, JPEG artefacts."""
    rng = np.random.default_rng(seed)
    h, w = bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += shift[0]
    M[1, 2] += shift[1]
    img = cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(250, 250, 250))
    if blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    f = img.astype(np.float32)
    if shadow:
        xx = np.linspace(1 - shadow, 1.0, w, dtype=np.float32)[None, :, None]
        f = f * xx
    f = f + brightness + rng.normal(0, noise, f.shape).astype(np.float32)
    img = np.clip(f, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def images_to_pdf(path: Path, images: List[np.ndarray], dpi=300):
    """Wrap raster pages in a PDF (a 'scanned PDF', no text layer)."""
    doc = pymupdf.open()
    for im in images:
        h, w = im.shape[:2]
        page = doc.new_page(width=w * 72.0 / dpi, height=h * 72.0 / dpi)
        ok, enc = cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        page.insert_image(page.rect, stream=enc.tobytes())
    doc.save(str(path))
    doc.close()


def px(pt_box, dpi=300):
    z = dpi / 72.0
    return tuple(v * z for v in pt_box)
