"""Service 2 – PDF & image processing: rasterisation and native (text-layer) extraction.

Pixel arrays are never cached; they are re-rendered deterministically because rendering a
PDF page costs ~0.1-0.3 s whereas OCR / object detection (which *are* cached) cost seconds."""
from __future__ import annotations

import hashlib
import threading
import unicodedata
import warnings
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import pymupdf

from .config import Config
from .models import DocInfo, Line, Word
from .textutil import reading_order

Image.MAX_IMAGE_PIXELS = None  # we enforce our own limit in upload.validate_file

_LOCK = threading.RLock()  # MuPDF documents are not thread-safe
_DOCS: Dict[str, "pymupdf.Document"] = {}


def _open(path: str):
    d = _DOCS.get(path)
    if d is None:
        d = pymupdf.open(path)
        _DOCS[path] = d
    return d


def close_all() -> None:
    with _LOCK:
        for d in _DOCS.values():
            try:
                d.close()
            except Exception:  # noqa: BLE001
                pass
        _DOCS.clear()


# ----------------------------------------------------------------------------- rasterisation


def _image_scale(h: int, w: int, cfg: Config) -> float:
    long = max(h, w)
    if long < 2400:
        return min(2.5, cfg.scan_target_long_px / long)
    if long > cfg.scan_max_long_px:
        return 4200.0 / long
    return 1.0


def render_raw(info: DocInfo, index: int, cfg: Config) -> Tuple[np.ndarray, float]:
    """Render page ``index`` to a BGR uint8 array. Returns (image, scale_applied)."""
    if info.kind == "pdf":
        z = cfg.dpi / 72.0
        with _LOCK:
            page = _open(info.path)[index]
            pix = page.get_pixmap(matrix=pymupdf.Matrix(z, z), alpha=False, colorspace=pymupdf.csRGB)
            arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).copy()
        return np.ascontiguousarray(arr[:, :, ::-1]), 1.0

    with Image.open(info.path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            bg.alpha_composite(im)
            im = bg
        im = im.convert("RGB")
        arr = np.asarray(im).copy()
    bgr = np.ascontiguousarray(arr[:, :, ::-1])
    s = _image_scale(*bgr.shape[:2], cfg)
    if abs(s - 1.0) > 0.02:
        interp = cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA
        bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=interp)
    else:
        s = 1.0
    return bgr, s


def apply_orientation(bgr: np.ndarray, rot_cw: int) -> np.ndarray:
    rot_cw %= 360
    if rot_cw == 90:
        return cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    if rot_cw == 180:
        return cv2.rotate(bgr, cv2.ROTATE_180)
    if rot_cw == 270:
        return cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return bgr


def apply_skew(bgr: np.ndarray, deg: float) -> np.ndarray:
    """Rotate by ``deg`` (counter-clockwise positive) keeping the canvas size, white fill."""
    if abs(deg) < 1e-3:
        return bgr
    h, w = bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, 1.0)
    return cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(255, 255, 255))


def load_processed(info: DocInfo, index: int, cfg: Config, rotation: int, skew: float) -> np.ndarray:
    """Deterministic re-creation of the processed page frame used for all coordinates."""
    bgr, _ = render_raw(info, index, cfg)
    return apply_skew(apply_orientation(bgr, rotation), skew)


# ----------------------------------------------------------------------------- native extraction


def _is_bold(flags: int, font: str) -> bool:
    f = font.lower()
    return bool(flags & 16) or "bold" in f or "black" in f or "heavy" in f


def _is_italic(flags: int, font: str) -> bool:
    f = font.lower()
    return bool(flags & 2) or "italic" in f or "oblique" in f


def _rgb_tuple(c) -> Tuple[float, ...] | None:
    if c is None:
        return None
    return tuple(round(float(v), 2) for v in c)


def extract_native(info: DocInfo, index: int, cfg: Config) -> dict:
    """Words (with per-character boxes and font data), lines in reading order, embedded
    images (with content hashes) and vector drawings of one PDF page, in render-pixel coordinates."""
    z = cfg.dpi / 72.0
    with _LOCK:
        doc = _open(info.path)
        page = doc[index]
        rm = page.rotation_matrix
        raw = page.get_text("rawdict")
        rect = page.rect
        img_infos = page.get_image_info(xrefs=True)
        drawings = page.get_drawings()
        img_hashes: Dict[int, str] = {}
        for im in img_infos:
            x = im.get("xref", 0)
            if x and x not in img_hashes:
                try:
                    img_hashes[x] = hashlib.sha1(doc.extract_image(x)["image"]).hexdigest()
                except Exception:  # noqa: BLE001
                    img_hashes[x] = ""

    def tb(b):
        r = pymupdf.Rect(b).transform(rm)
        return (r.x0 * z, r.y0 * z, r.x1 * z, r.y1 * z)

    raw_lines: List[Tuple[tuple, List[Word]]] = []
    n_chars = 0
    for blk in raw.get("blocks", []):
        if blk.get("type") != 0:
            continue
        for ln in blk.get("lines", []):
            chars = []
            for sp in ln.get("spans", []):
                for ch in sp.get("chars", []):
                    chars.append((ch["c"], ch["bbox"], sp))
            words: List[Word] = []
            cur: list = []

            def flush():
                if not cur:
                    return
                text = "".join(c[0] for c in cur)
                t2 = unicodedata.normalize("NFC", text)
                if len(t2) == len(text):
                    text = t2
                boxes = [tb(c[1]) for c in cur]
                bb = (min(b[0] for b in boxes), min(b[1] for b in boxes),
                      max(b[2] for b in boxes), max(b[3] for b in boxes))
                sp0 = cur[0][2]
                font = sp0.get("font", "")
                words.append(Word(text=text, bbox=bb, conf=100.0, source="native", font=font,
                                  size=float(sp0.get("size", 0.0)), bold=_is_bold(sp0.get("flags", 0), font),
                                  italic=_is_italic(sp0.get("flags", 0), font), color=int(sp0.get("color", 0)),
                                  char_boxes=boxes))
                cur.clear()

            for c in chars:
                if c[0].isspace():
                    flush()
                else:
                    cur.append(c)
                    n_chars += 1
            flush()
            if words:
                lb = (min(w.bbox[0] for w in words), min(w.bbox[1] for w in words),
                      max(w.bbox[2] for w in words), max(w.bbox[3] for w in words))
                raw_lines.append((lb, words))

    heights = [lb[3] - lb[1] for lb, _ in raw_lines]
    glyph_h = float(np.median(heights)) if heights else 0.0
    ordered = reading_order(raw_lines, lambda t: t[0], glyph_h)
    all_words: List[Word] = []
    lines: List[Line] = []
    for li, (lb, ws) in enumerate(ordered):
        ids = []
        for w in ws:
            w.line = li
            ids.append(len(all_words))
            all_words.append(w)
        lines.append(Line(idx=li, words=ids, bbox=lb, text=" ".join(w.text for w in ws), conf=100.0))

    page_area = rect.width * rect.height * z * z
    images = []
    cover = 0.0
    for im in img_infos:
        bb = tb(im["bbox"])
        area = max(0.0, bb[2] - bb[0]) * max(0.0, bb[3] - bb[1])
        cover = max(cover, area / page_area if page_area else 0.0)
        h = img_hashes.get(im.get("xref", 0), "") or f"inline-{im.get('width')}x{im.get('height')}"
        images.append({"bbox": bb, "hash": h, "w": im.get("width"), "h": im.get("height")})

    draws = []
    for dr in drawings[:6000]:
        r = dr.get("rect")
        if r is None:
            continue
        draws.append({"bbox": tb(r), "type": dr.get("type"), "color": _rgb_tuple(dr.get("color")),
                      "fill": _rgb_tuple(dr.get("fill")), "width": round(float(dr.get("width") or 0), 2)})

    return {
        "words": all_words, "lines": lines, "images": images, "drawings": draws,
        "n_chars": n_chars, "glyph_h": glyph_h, "cover": cover,
        "page_mm": (rect.width * 25.4 / 72.0, rect.height * 25.4 / 72.0),
        "n_drawings_total": len(drawings),
    }


def is_scan_like(native: dict, cfg: Config) -> bool:
    """A page is treated as image-only (OCR) when one image covers most of it, or when it has almost no
    text *and* carries a large image. A short native page without a large image (a cover page, a
    near-empty continuation page) keeps its exact text layer."""
    if native["cover"] >= cfg.scan_cover_ratio:
        return True
    return native["n_chars"] < cfg.native_min_chars and native["cover"] >= cfg.small_page_image_cover
