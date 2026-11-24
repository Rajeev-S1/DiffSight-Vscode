"""Deterministic synthetic documents for the DPI / retry tests (no external data, fixed random seeds)."""
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image

A4_W, A4_H = 595.276, 841.89                      # points


def text_page_gray(dpi=300, lines=34, seed=0):
    """A clean, text-only A4 page rendered at ``dpi`` (uint8 grey)."""
    d = fitz.open()
    pg = d.new_page(width=A4_W, height=A4_H)
    rng = np.random.default_rng(seed)
    words = ["agreement", "payment", "invoice", "delivery", "service", "customer", "total", "amount", "terms", "support", "date", "reference"]
    for i in range(lines):
        pg.insert_text((56, 70 + i * 21), " ".join(rng.choice(words, 9)) + f" {i + 1}", fontsize=11)
    pix = pg.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()


def blur(g, sigma):
    return cv2.GaussianBlur(g, (0, 0), sigma)


def noise(g, sigma, seed=7):
    rng = np.random.default_rng(seed)
    return np.clip(g.astype(np.float32) + rng.normal(0, sigma, g.shape), 0, 255).astype(np.uint8)


def faded(g, k):
    return np.clip(255 - (255 - g.astype(np.float32)) * k, 0, 255).astype(np.uint8)


def at_resolution(g, source_dpi, from_dpi=300):
    """The same page as a scan of ``source_dpi`` (pixels = A4 inches x source_dpi)."""
    s = source_dpi / float(from_dpi)
    return cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s != 1 else g


def _png(g):
    ok, buf = cv2.imencode(".png", g)
    return buf.tobytes()


def scan_pdf(path, pages, source_dpis=None):
    """PDF whose pages are full-page scan images. ``pages`` = list of grey arrays (their pixel size sets the source DPI)."""
    d = fitz.open()
    for g in pages:
        pg = d.new_page(width=A4_W, height=A4_H)
        pg.insert_image(pg.rect, stream=_png(g))
    d.save(str(path))
    return str(path)


def digital_pdf(path, n_pages=1, text="Amount payable: 1000", logo=False):
    d = fitz.open()
    for i in range(n_pages):
        pg = d.new_page(width=A4_W, height=A4_H)
        pg.insert_text((72, 100), f"Page {i + 1} contract clause number {i}", fontsize=12)
        pg.insert_text((72, 130), text, fontsize=12)
        if logo:                                              # small decorative image: must NOT make this a scan
            pg.insert_image(fitz.Rect(450, 40, 540, 90), stream=_png(np.full((60, 120), 90, np.uint8)))
    d.save(str(path))
    return str(path)


def image_file(path, g, dpi=None):
    im = Image.fromarray(g)
    kw = {"dpi": dpi} if dpi else {}
    im.save(str(path), **kw)
    return str(path)
