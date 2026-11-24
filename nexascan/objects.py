"""Service 5 – Stamp, signature, handwriting and visual-object detection.

No pretrained detector can be bundled offline, so this is classical computer vision with an
explicit, inspectable score per object.  Types are deliberately conservative:

  stamp        ring / outline geometry + inner text-like components (+ ink colour)
  handwriting  long connected strokes, off-axis stroke orientation, irregular size (a
               *signature-like* hint is added only from nearby 'signature' wording / underline)
  text_unread  printed-looking glyph rows that OCR did not read (never treated as 'missing')
  image / block / line   embedded images (exact, from the PDF), solid cover blocks, ruled lines
  mark         any other ink the text layer does not explain

An object is never claimed to belong to a person; only its appearance is reported."""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional

import cv2
import numpy as np

from .config import Config
from .models import Word
from .preprocess import Prepared

SIGN_RE = re.compile(r"sign|signatory|authori[sz]ed|witness|seal|stamp|by\s*:|date\s*:", re.I)


def _px_per_mm(cfg: Config, ds: float) -> float:
    return cfg.dpi * ds / 25.4


def _down(mask: np.ndarray, ds: float) -> np.ndarray:
    if ds >= 0.999:
        return mask.copy()
    m = cv2.resize(mask.astype(np.uint8) * 255, None, fx=ds, fy=ds, interpolation=cv2.INTER_AREA)
    return m > 50


def _ellipse_kernel(w: int, h: Optional[int] = None) -> np.ndarray:
    h = w if h is None else h
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, int(w)) | 1, max(1, int(h)) | 1))


def _mean_color(bgr: np.ndarray, mask: np.ndarray) -> tuple:
    if mask.sum() == 0:
        return (0, 0, 0)
    m = bgr[mask].mean(axis=0)
    return (int(m[2]), int(m[1]), int(m[0]))  # RGB


def _hue_name(rgb: tuple) -> str:
    r, g, b = rgb
    if max(rgb) - min(rgb) < 40:
        return "black/grey"
    hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0, 0]
    h = int(hsv[0]) * 2
    if h < 20 or h >= 340:
        return "red"
    if h < 70:
        return "orange/yellow"
    if h < 170:
        return "green"
    if h < 255:
        return "blue"
    return "purple"


def _axis_alignment(ang: np.ndarray, mag: np.ndarray, mask: np.ndarray) -> float:
    """Share of gradient energy along 0/90 degrees (printed glyphs/rules ~0.7+, handwriting lower).
    ``ang`` / ``mag`` are the page-wide gradient direction (0..180) and magnitude, pre-sliced."""
    if mask.sum() < 20:
        return 1.0
    edge = mask & (mag > 20)
    if edge.sum() < 15:
        return 1.0
    a = ang[edge]
    w = mag[edge]
    near = ((a < 18) | (a > 162) | (np.abs(a - 90) < 18))
    return float((w * near).sum() / max(w.sum(), 1e-6))


def detect_objects(prep: Prepared, words: List[Word], glyph_h: float, native_images: List[dict],
                   native_drawings: List[dict], cfg: Config, scan: bool) -> List[dict]:
    H, W = prep.ink.shape
    ds = 0.5 if max(H, W) > 2400 else 1.0
    inv = 1.0 / ds
    ppm = _px_per_mm(cfg, ds)
    h_s, w_s = (int(round(H * ds)), int(round(W * ds)))
    ink_s = _down(prep.ink, ds)
    if ink_s.shape != (h_s, w_s):
        h_s, w_s = ink_s.shape
    bgr_s = cv2.resize(prep.bgr, (w_s, h_s), interpolation=cv2.INTER_AREA)
    gray_s = cv2.cvtColor(bgr_s, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr_s, cv2.COLOR_BGR2HSV)
    gx_ = cv2.Sobel(gray_s, cv2.CV_32F, 1, 0, ksize=3)
    gy_ = cv2.Sobel(gray_s, cv2.CV_32F, 0, 1, ksize=3)
    g_mag = np.hypot(gx_, gy_)
    g_ang = np.degrees(np.arctan2(gy_, gx_)) % 180.0
    del gx_, gy_
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    gh = max(glyph_h * ds, 6.0 * ppm / 2.0) if glyph_h else 4.0 * ppm

    objects: List[dict] = []

    def add(otype: str, box_s, score: float, **kw):
        x0, y0, x1, y1 = box_s
        objects.append({"type": otype, "bbox": (x0 * inv, y0 * inv, x1 * inv, y1 * inv),
                        "score": round(float(score), 3), **kw})

    # ---- text mask (what the text layer / OCR explains) ------------------------------
    text_mask = np.zeros((h_s, w_s), bool)
    for w in words:
        if w.source == "ocr" and w.conf < 40:
            continue
        x0, y0, x1, y1 = [v * ds for v in w.bbox]
        pad = 0.08 * (y1 - y0)
        text_mask[max(0, int(y0 - pad)):int(y1 + pad) + 1, max(0, int(x0 - pad)):int(x1 + pad) + 1] = True

    # ---- embedded images (exact for PDFs) ---------------------------------------------
    img_mask = np.zeros((h_s, w_s), bool)
    small_imgs: List[tuple] = []
    for im in native_images:
        x0, y0, x1, y1 = [v * ds for v in im["bbox"]]
        if (x1 - x0) * (y1 - y0) < 4 * ppm * ppm:
            continue
        # a page-covering background image is not an "object"
        if (x1 - x0) * (y1 - y0) > 0.8 * h_s * w_s:
            continue
        add("image", (x0, y0, x1, y1), 1.0, source="pdf-image", hash=im.get("hash", ""))
        # large pictures are opaque content; small ones (stamps, signatures, logos) are also
        # classified by their own ink (inside their own box only) so an embedded stamp is a stamp
        if (x1 - x0) * (y1 - y0) > 0.25 * h_s * w_s:
            img_mask[max(0, int(y0)):int(y1) + 1, max(0, int(x0)):int(x1) + 1] = True
        else:
            small_imgs.append((len(objects) - 1, (x0, y0, x1, y1)))

    # ---- ruled lines -------------------------------------------------------------------
    llen = int(max(8, 7.0 * ppm))
    hor = cv2.morphologyEx(ink_s.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (llen, 1)))
    ver = cv2.morphologyEx(ink_s.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, llen)))
    lines_mask = (hor > 0) | (ver > 0)
    lm = cv2.dilate(lines_mask.astype(np.uint8), _ellipse_kernel(max(3, int(0.6 * ppm)))) > 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(lm.astype(np.uint8), connectivity=8)
    for k in range(1, n):
        x, y, w, h, a = stats[k]
        if max(w, h) >= 10 * ppm and min(w, h) <= 2.5 * ppm:   # a filled area is not a ruled line
            add("line", (x, y, x + w, y + h), 0.95, orientation="horizontal" if w >= h else "vertical")

    # ---- solid blocks (redaction boxes, covers) ----------------------------------------
    dark = (gray_s < 90) & ink_s & ~img_mask
    kb = max(3, int(1.2 * ppm)) | 1
    core = cv2.erode(dark.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (kb, kb)))
    core = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_RECT, (kb, kb))) > 0
    blocks_mask = np.zeros_like(core)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), connectivity=8)
    for k in range(1, n):
        x, y, w, h, a = stats[k]
        if a < 12 * ppm * ppm:
            continue
        tm = text_mask[y:y + h, x:x + w].mean()
        if tm > 0.5 and h < 2.5 * gh:      # bold heading glyphs, not a cover block
            continue
        sub = lab[y:y + h, x:x + w] == k
        blocks_mask[y:y + h, x:x + w] |= sub
        add("block", (x, y, x + w, y + h), 0.9, fill=round(float(a) / (w * h), 3),
            dark_level=int(gray_s[y:y + h, x:x + w][sub].mean()))
    blocks_d = cv2.dilate(blocks_mask.astype(np.uint8), _ellipse_kernel(5)) > 0

    # ---- candidate ink: not text, not ruling, not image/block -----------------------------
    # coloured ink (stamps, coloured pens) is separated by colour so a stamp that touches printed text
    # lines does not merge those lines into one giant object; coloured *text* is excluded
    colour_raw = ink_s & (sat > 70) & (val < 245) & ~blocks_d & ~img_mask
    colour_ink = np.zeros_like(colour_raw)
    if colour_raw.any():
        cj = cv2.dilate(colour_raw.astype(np.uint8), _ellipse_kernel(int(1.5 * ppm))) > 0
        n, lab, stats, _ = cv2.connectedComponentsWithStats(cj.astype(np.uint8), connectivity=8)
        for k in range(1, n):
            x, y, w, h, a = stats[k]
            sel = (lab[y:y + h, x:x + w] == k) & colour_raw[y:y + h, x:x + w]
            if sel.sum() >= 8 and text_mask[y:y + h, x:x + w][sel].mean() < 0.6:
                colour_ink[y:y + h, x:x + w] |= sel
    cand = (ink_s & ~text_mask & ~lm & ~blocks_d & ~img_mask) | colour_ink
    # long dark strokes that cross printed text (handwriting over text) are candidates too
    black_ink = ink_s & ~lm & ~blocks_d & ~img_mask & ~cv2.dilate(colour_ink.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(black_ink.astype(np.uint8), connectivity=8)
    for k in range(1, n):
        x, y, w, h, a = stats[k]
        if (w > 9 * gh or h > 2.6 * gh) and a > 3 * ppm * ppm and text_mask[y:y + h, x:x + w].mean() > 0.05 \
                and h < 8 * gh and w < 110 * ppm:      # a page-wide web of text lines is not a mark
            cand[y:y + h, x:x + w] |= (lab[y:y + h, x:x + w] == k)
    min_px = max(3, int(0.6 * ppm * ppm))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_px
    cand = keep[lab]
    if not cand.any():
        return _finish(objects)

    def classify(reg, x, y, w, h):
        """Type one ink region (reg is a bool mask of its box x,y,w,h in downscaled coordinates)."""
        area = int(reg.sum())
        w_mm, h_mm = w / ppm, h / ppm
        sub_bgr = bgr_s[y:y + h, x:x + w]
        sub_gray = gray_s[y:y + h, x:x + w]
        rgb = _mean_color(sub_bgr, reg)
        colored = (sat[y:y + h, x:x + w][reg] > 70).mean() > 0.5 and _hue_name(rgb) != "black/grey"
        ncomp, clab, cst, _ = cv2.connectedComponentsWithStats(reg.astype(np.uint8), connectivity=8)
        comps = [(cst[i, 2], cst[i, 3], cst[i, 4], cst[i, 0], cst[i, 1]) for i in range(1, ncomp) if cst[i, 4] >= max(3, min_px // 2)]
        n_inner = max(0, len(comps) - 1)
        # ---- stamp score
        yy, xx = np.nonzero(reg)
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        a_, b_ = max(w / 2.0, 1.0), max(h / 2.0, 1.0)
        rr = np.sqrt(((xx - cx) / a_) ** 2 + ((yy - cy) / b_) ** 2)
        ring_circle = float(((rr > 0.80) & (rr < 1.10)).mean())
        edge_d = np.minimum.reduce([xx, yy, w - 1 - xx, h - 1 - yy]) / max(1.0, min(w, h))
        ring_rect = float((edge_d < 0.08).mean())
        outline = max(ring_circle, ring_rect)
        roundness = min(w, h) / float(max(w, h))
        size_ok = 8.0 <= min(w_mm, h_mm) and max(w_mm, h_mm) <= 90.0
        s_stamp = (0.25 * size_ok + 0.30 * min(1.0, outline / 0.35) + 0.15 * (roundness > 0.6)
                   + 0.15 * (n_inner >= 3) + 0.15 * colored)
        circular = ring_circle >= ring_rect
        feat: Dict = {"w_mm": round(w_mm, 1), "h_mm": round(h_mm, 1), "ink_px": area, "outline": round(outline, 3),
                      "roundness": round(roundness, 3), "inner_components": n_inner, "colour": rgb,
                      "colour_name": _hue_name(rgb), "circular": circular}
        # ---- handwriting score
        heights = np.array([c[1] for c in comps], np.float64) if comps else np.array([h], np.float64)
        ycent = np.array([c[4] + c[1] / 2.0 for c in comps], np.float64) if comps else np.array([cy])
        h_cv = float(heights.std() / max(heights.mean(), 1.0))
        aligned = float(ycent.std() / max(np.median(heights), 1.0)) < 0.25
        longest = max((c[0] for c in comps), default=w)
        axis = _axis_alignment(g_ang[y:y + h, x:x + w], g_mag[y:y + h, x:x + w], reg)
        hull = cv2.convexHull(np.column_stack([xx, yy]).astype(np.int32))
        hull_a = max(cv2.contourArea(hull), 1.0)
        fill = area / hull_a
        s_hand = (0.30 * min(1.0, longest / (6.0 * gh)) + 0.30 * min(1.0, max(0.0, (0.80 - axis) / 0.35))
                  + 0.20 * min(1.0, h_cv / 0.5) + 0.20 * (fill < 0.35))
        feat.update({"axis_alignment": round(axis, 3), "height_cv": round(h_cv, 3), "fill": round(fill, 3),
                     "longest_component_mm": round(longest / ppm, 1)})
        near_sign = False
        cx_mm_box = (x - 3 * gh, y - 2.5 * gh, x + w + 3 * gh, y + h + 2.5 * gh)
        for wd in words:
            bx = [v * ds for v in wd.bbox]
            if bx[2] > cx_mm_box[0] and bx[0] < cx_mm_box[2] and bx[3] > cx_mm_box[1] and bx[1] < cx_mm_box[3]:
                if SIGN_RE.search(wd.text):
                    near_sign = True
                    break
        under = bool(lines_mask[min(h_s - 1, y + h):min(h_s, y + h + int(1.5 * gh)), x:x + w].any())
        feat["near_signature_wording"] = near_sign
        feat["underline_below"] = under

        if s_stamp >= 0.6 and outline >= 0.25 and (circular or colored or roundness > 0.6):
            otype, score = "stamp", min(0.95, s_stamp)
            if not colored:
                score = min(score, 0.75)
        elif len(comps) >= 4 and h_cv < 0.30 and aligned and axis > 0.6:
            otype, score = "text_unread", 0.6
        elif s_hand >= 0.5:
            otype, score = "handwriting", min(0.9, s_hand)
        elif axis > 0.62 and h_cv < 0.45 and len(comps) >= 2 and not colored:
            otype, score = "text_unread", 0.5
        else:
            otype, score = "mark", min(0.6, 0.3 + 0.3 * s_hand)
        rect = cv2.minAreaRect(np.column_stack([xx, yy]).astype(np.float32))
        feat["rotation_deg"] = round(float(rect[2]), 1)
        return otype, score, feat, bool(colored), {"stamp": round(s_stamp, 3), "handwriting": round(s_hand, 3)}

    # ---- embedded images: classified inside their own box (never merged with neighbours) ------
    ink_obj = ink_s & ~(text_mask & (sat < 60))
    for oi, (bx0, by0, bx1, by1) in small_imgs:
        x, y = max(0, int(bx0)), max(0, int(by0))
        w, h = min(w_s, int(bx1)) - x, min(h_s, int(by1)) - y
        if w < 4 or h < 4:
            continue
        reg = ink_obj[y:y + h, x:x + w]
        if int(reg.sum()) < min_px or max(w, h) / ppm < 2.5:
            continue
        res = classify(reg, x, y, w, h)
        cand[y:y + h, x:x + w] &= ~reg          # these pixels belong to the image object now
        if res is None:
            continue
        otype, score, feat, colored, scores = res
        if otype in ("stamp", "handwriting") or (otype == "mark" and False):
            obj = objects[oi]
            obj.update({"type": otype, "score": round(float(score), 3), "colored": colored, "features": feat,
                        "scores": scores, "source": "pdf-image"})

    # ---- cluster candidates into objects -------------------------------------------------
    clus = cv2.dilate(cand.astype(np.uint8), _ellipse_kernel(int(2.6 * ppm), int(1.8 * ppm))) > 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(clus.astype(np.uint8), connectivity=8)
    for k in range(1, n):
        x, y, w, h, _a = stats[k]
        reg = (lab[y:y + h, x:x + w] == k) & cand[y:y + h, x:x + w]
        if int(reg.sum()) < min_px or max(w, h) / ppm < 2.5:
            continue
        res = classify(reg, x, y, w, h)
        if res is None:
            continue
        otype, score, feat, colored, scores = res
        add(otype, (x, y, x + w, y + h), score, colored=colored, features=feat, scores=scores)
    return _finish(objects)


def _jsonable(v):
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v] if not isinstance(v, tuple) or len(v) != 3 or not all(
            isinstance(i, (int, np.integer)) for i in v) else tuple(int(i) for i in v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def _absorb_children(objects: List[dict]) -> List[dict]:
    """Inner text / marks that lie inside a stamp belong to the stamp (not separate objects)."""
    stamps = [o for o in objects if o["type"] == "stamp"]
    out = []
    for o in objects:
        if o["type"] in ("line", "image", "block"):
            out.append(o)
            continue
        absorbed = False
        for s in stamps:
            if s is o:
                continue
            sb, ob = s["bbox"], o["bbox"]
            ix = max(0.0, min(sb[2], ob[2]) - max(sb[0], ob[0]))
            iy = max(0.0, min(sb[3], ob[3]) - max(sb[1], ob[1]))
            oa = max(1e-6, (ob[2] - ob[0]) * (ob[3] - ob[1]))
            sa = (sb[2] - sb[0]) * (sb[3] - sb[1])
            if ix * iy / oa >= 0.85 and (sa > oa * 1.5 or o["type"] != "stamp"):
                absorbed = True
                break
        if not absorbed:
            out.append(o)
    # lines that are really the rim of a stamp are dropped too
    final = []
    for o in out:
        if o["type"] == "line":
            ob = o["bbox"]
            if any(s["bbox"][0] - 2 <= ob[0] and s["bbox"][1] - 2 <= ob[1] and s["bbox"][2] + 2 >= ob[2]
                   and s["bbox"][3] + 2 >= ob[3] for s in stamps):
                continue
        final.append(o)
    return final


def _merge_pdf_images(objects: List[dict]) -> List[dict]:
    """An embedded image that was classified (stamp / handwriting) by its ink becomes that typed
    object and keeps the image hash; otherwise the plain 'image' object stays."""
    imgs = [o for o in objects if o["type"] == "image"]
    typed = [o for o in objects if o["type"] in ("stamp", "handwriting", "mark")]
    drop = set()
    for t in typed:
        tb = t["bbox"]
        ta = max(1e-6, (tb[2] - tb[0]) * (tb[3] - tb[1]))
        for im in imgs:
            ib = im["bbox"]
            ix = max(0.0, min(tb[2], ib[2]) - max(tb[0], ib[0]))
            iy = max(0.0, min(tb[3], ib[3]) - max(tb[1], ib[1]))
            if ix * iy / ta >= 0.6:
                t["hash"] = im.get("hash", "")
                t["source"] = "pdf-image"
                drop.add(id(im))
                break
    return [o for o in objects if id(o) not in drop]


def _finish(objects: List[dict]) -> List[dict]:
    objects = _merge_pdf_images(objects)
    objects = _absorb_children(objects)
    for i, o in enumerate(objects):
        o["id"] = i
        o["bbox"] = tuple(round(float(v), 1) for v in o["bbox"])
    return [_jsonable(o) for o in objects]
