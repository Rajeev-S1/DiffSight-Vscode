"""Service 8 – Overlap detection engine.

Finds differences where content is covered, hidden, or written over:
  1. objects (stamp / handwriting / block / image) that newly cover words, or stop covering them
  2. words that exist in the PDF text layer but are not visibly rendered (white box on top,
     invisible text) in one document and visible in the other
  3. text written over text (native text layer: words whose boxes intersect)
For every finding the *visible* and the *hidden* content are reported separately; when the
content under an overlay cannot be read it is reported as unreadable, never guessed."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import align as alignmod
from .align import Alignment
from .config import Config
from .models import BBox, Finding, Word, bbox_area, bbox_inter, bbox_union
from .textutil import normalize_ocr_text
from .visual_compare import VisualContext

COVER_TYPES = ("stamp", "handwriting", "mark", "block", "image")


# --------------------------------------------------------------------------- layer separation


def separate_layers(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split a crop into (dark text layer on white, coloured overlay layer on white).
    Colour = clearly saturated ink (blue/red stamps and pens); text = dark, low-saturation ink."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    colour = (sat > 70) & (val < 245)
    colour_d = cv2.dilate(colour.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    text = np.where(((val < 120) & (sat < 110)) & ~colour_d, gray, 255).astype(np.uint8)
    overlay = np.where(colour, gray, 255).astype(np.uint8)
    return text, overlay


def _ocr_crop(img: np.ndarray, psm: int = 7) -> Tuple[str, float]:
    if img.size == 0:
        return "", 0.0
    try:
        from . import ocr as ocrmod
        text, conf = ocrmod.recognize_crop(img)
    except Exception:  # noqa: BLE001
        return "", 0.0
    return normalize_ocr_text(text), conf


def _crop(img: np.ndarray, box: BBox, pad: int = 3) -> np.ndarray:
    h, w = img.shape[:2]
    return img[max(0, int(box[1]) - pad):min(h, int(box[3]) + pad), max(0, int(box[0]) - pad):min(w, int(box[2]) + pad)]


# --------------------------------------------------------------------------- helpers


def _words_af(words: List[Word], M: Optional[np.ndarray]) -> List[BBox]:
    return [alignmod.map_box(M, w.bbox) if M is not None else w.bbox for w in words]


def _count(mask: np.ndarray, box: BBox) -> int:
    h, w = mask.shape
    x0, y0, x1, y1 = max(0, int(box[0])), max(0, int(box[1])), min(w, int(box[2]) + 1), min(h, int(box[3]) + 1)
    return int(mask[y0:y1, x0:x1].sum()) if x1 > x0 and y1 > y0 else 0


def _ink_frac(ink: np.ndarray, box: BBox) -> float:
    h, w = ink.shape
    x0, y0, x1, y1 = max(0, int(box[0])), max(0, int(box[1])), min(w, int(box[2])), min(h, int(box[3]))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(ink[y0:y1, x0:x1].mean())


# --------------------------------------------------------------------------- engine


def analyze_overlaps(ctx: VisualContext, objs_a: List[dict], objs_b: List[dict], words_a: List[Word],
                     words_b: List[Word], al: Alignment, cfg: Config, native: bool, pa_no: int,
                     pb_no: int) -> List[Finding]:
    out: List[Finding] = []
    inv = al.inverse()
    min_px = max(6, int(cfg.mm2(cfg.min_ink_mm2) * 2))
    boxes_a = _words_af(words_a, None)
    boxes_b = _words_af(words_b, al.M)
    added = ctx.added if ctx.added is not None else np.zeros((ctx.h, ctx.w), bool)
    removed = ctx.removed if ctx.removed is not None else np.zeros((ctx.h, ctx.w), bool)

    def cover_objects(objs, M):
        res = []
        for o in objs:
            if o["type"] in COVER_TYPES:
                bb = alignmod.map_box(M, o["bbox"]) if M is not None else tuple(o["bbox"])
                res.append((o, bb))
        return res

    # ---- 1) objects that newly cover words (in B) or stopped covering words (in A) ---------------
    # native text layers are exact on both sides; for scans OCR under an overlay is garbage, so the
    # words are taken from the side that does NOT carry the overlay
    for side, objs, wboxes, words, pix, M_obj in (
            ("B", objs_b, boxes_b if native else boxes_a, words_b if native else words_a, added, al.M),
            ("A", objs_a, boxes_a if native else boxes_b, words_a if native else words_b, removed, None)):
        for o, ob in cover_objects(objs, M_obj):
            hit = [i for i, wb in enumerate(wboxes)
                   if bbox_inter(wb, ob) >= 0.15 * max(1.0, bbox_area(wb)) and _count(pix, wb) >= min_px]
            if not hit:
                continue
            # colour overlays: require overlay pixels actually inside the word box
            covered = [words[i] for i in hit]
            cbox = bbox_union([wboxes[i] for i in hit])
            region = bbox_union([cbox, ob])
            label = {"stamp": "stamp", "handwriting": "handwritten mark", "mark": "ink mark", "block": "solid block",
                     "image": "image"}[o["type"]]
            covered_txt = " ".join(w.text for w in covered)
            src_bgr = ctx.bgr_b if side == "B" else ctx.bgr_a
            other_bgr = ctx.bgr_a if side == "B" else ctx.bgr_b
            hidden = visible = None
            hidden_note = ""
            if native:
                hidden = covered_txt   # text layer content under the object
                vis_txt, vis_conf = _ocr_crop(_crop(cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY), cbox), 7)
                visible = vis_txt if vis_conf >= 60 else None
                if visible is not None and " ".join(visible.split()) != " ".join(covered_txt.split()):
                    hidden_note = (f"visible reading is '{visible}' while the text layer contains '{covered_txt}' "
                                   f"(text layer content is partially hidden by the {label})")
                else:
                    hidden_note = f"the text layer still contains '{covered_txt}' under the {label}"
            else:
                feats = o.get("features", {})
                if feats.get("colour_name", "black/grey") != "black/grey" or o.get("colored"):
                    txt_layer, _ = separate_layers(_crop(src_bgr, cbox))
                    rec, rc = _ocr_crop(txt_layer, 7)
                    if rec and rc >= 55:
                        hidden = rec
                        hidden_note = f"text beneath the coloured {label} was recovered by layer separation: '{rec}' (OCR confidence {rc:.0f}%)"
                    else:
                        hidden_note = f"text beneath the {label} could not be read reliably and is not guessed"
                else:
                    hidden_note = f"text beneath the dark {label} cannot be recovered"
            who = "added in B" if side == "B" else "present in A but not in B"
            cat_desc = (f"{label} {who} overlaps {len(covered)} word(s): '{covered_txt[:80]}'; {hidden_note}")
            box_a = region
            box_b = alignmod.map_box(inv, region)
            conf = float(np.clip(0.6 + 0.3 * o.get("score", 0.6), 0.5, 0.92))
            out.append(Finding(
                category="Overlapped", dtype="Overlap", page_a=pa_no, page_b=pb_no,
                a_content=(f"text '{covered_txt[:100]}' unobstructed" if side == "B" else f"text '{covered_txt[:100]}' covered by {label}"),
                b_content=(f"text '{covered_txt[:100]}' covered by {label}" if side == "B" else f"text '{covered_txt[:100]}' unobstructed"),
                description=cat_desc, bbox_a=tuple(map(float, box_a)), bbox_b=tuple(map(float, box_b)),
                confidence=round(conf, 3), engines=["overlap"],
                details={"overlay_type": o["type"], "overlay_in": side, "covered_words": [w.text for w in covered],
                         "hidden_content": hidden, "visible_reading": visible, "note": hidden_note,
                         "layer_separation": (not native and bool(hidden)), "region_a_frame": tuple(map(float, region))},
                region=tuple(map(float, region))))

    # ---- 2) text-layer words that are not visibly rendered in one document ---------------------
    if native:
        ia, ib = ctx.ink_a, ctx.ink_b
        hits = []   # (direction, box_a_frame_union, text, fa, fb)
        by_text: Dict[str, List[int]] = {}     # index A words by text: only same-text candidates are examined
        for i, wa in enumerate(words_a):
            by_text.setdefault(wa.text, []).append(i)
        tol_px = cfg.mm(1.0)
        for k, wb_ in enumerate(words_b):
            if len(wb_.text) < 2:
                continue
            bb = boxes_b[k]
            match = None
            for i in by_text.get(wb_.text, ()):
                if abs(boxes_a[i][0] - bb[0]) < tol_px and abs(boxes_a[i][1] - bb[1]) < tol_px:
                    match = i
                    break
            if match is None:
                continue
            fa, fb = _ink_frac(ia, boxes_a[match]), _ink_frac(ib, bb)
            if fa >= 0.05 and fb < 0.012:
                direction = "hidden"
            elif fb >= 0.05 and fa < 0.012:
                direction = "revealed"
            else:
                continue
            hits.append((direction, bbox_union([boxes_a[match], bb]), wb_.text, fa, fb))
        hits.sort(key=lambda h: (h[0], round(h[1][1] / max(1.0, cfg.mm(2))), h[1][0]))
        groups: List[list] = []
        for h in hits:   # merge neighbouring words of the same row into one phrase
            if groups and groups[-1][-1][0] == h[0] and abs(groups[-1][-1][1][1] - h[1][1]) < cfg.mm(1.5) \
                    and h[1][0] - groups[-1][-1][1][2] < cfg.mm(4.0):
                groups[-1].append(h)
            else:
                groups.append([h])
        for g in groups:
            direction = g[0][0]
            region = bbox_union([h[1] for h in g])
            if direction == "hidden" and any(bbox_inter(region, f.region) > 0 for f in out if f.region and f.details.get("overlay_type")):
                continue  # already explained by an overlay finding
            txt = " ".join(h[2] for h in g)
            fa = float(np.mean([h[3] for h in g]))
            fb = float(np.mean([h[4] for h in g]))
            out.append(Finding(
                category="Overlapped", dtype="Hidden text", page_a=pa_no, page_b=pb_no,
                a_content=(f"'{txt}' visible" if direction == "hidden" else f"'{txt}' in text layer, not visible"),
                b_content=(f"'{txt}' in text layer, not visible" if direction == "hidden" else f"'{txt}' visible"),
                description=(f"'{txt}' is visible in A but not visibly rendered in B (still present in the PDF text layer: covered or invisible)"
                             if direction == "hidden" else
                             f"'{txt}' is present but not visible in A and visible in B"),
                bbox_a=tuple(map(float, region)), bbox_b=tuple(map(float, alignmod.map_box(inv, region))),
                confidence=0.9, engines=["overlap-hidden"],
                details={"word": txt, "ink_fraction_a": round(fa, 3), "ink_fraction_b": round(fb, 3),
                         "region_a_frame": tuple(map(float, region))}, region=tuple(map(float, region))))

    # ---- 3) text written over text (native text layer), one finding per overlap cluster ----------
    def overlapping_pairs(words, boxes):
        pairs = {}
        order = sorted(range(len(words)), key=lambda i: boxes[i][0])
        for ii, i in enumerate(order):
            for j in order[ii + 1:]:
                if boxes[j][0] > boxes[i][2]:
                    break
                inter = bbox_inter(boxes[i], boxes[j])
                small = min(bbox_area(boxes[i]), bbox_area(boxes[j]))
                if small > 0 and inter / small >= 0.30:
                    pairs[(i, j)] = inter / small
        return pairs

    if native:
        pairs_a, pairs_b = overlapping_pairs(words_a, boxes_a), overlapping_pairs(words_b, boxes_b)
        keyset_a = {tuple(sorted([words_a[i].text, words_a[j].text])) + (round(boxes_a[i][0] / cfg.mm(2)), round(boxes_a[i][1] / cfg.mm(2)))
                    for (i, j) in pairs_a}
        new_pairs = {}
        for (i, j), frac in pairs_b.items():
            key = tuple(sorted([words_b[i].text, words_b[j].text])) + (round(boxes_b[i][0] / cfg.mm(2)), round(boxes_b[i][1] / cfg.mm(2)))
            if key not in keyset_a:
                new_pairs[(i, j)] = frac
        parent = {}

        def root(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for (i, j) in new_pairs:
            parent[root(i)] = root(j)
        comps: Dict[int, set] = {}
        for (i, j) in new_pairs:
            comps.setdefault(root(i), set()).update((i, j))
        for members in comps.values():
            idx = sorted(members, key=lambda k: boxes_b[k][0])
            region = bbox_union([boxes_b[k] for k in idx])
            names = [words_b[k].text for k in idx]
            frac = max(f for (i, j), f in new_pairs.items() if i in members)
            out.append(Finding(
                category="Overlapped", dtype="Overlap", page_a=pa_no, page_b=pb_no,
                a_content="words do not overlap here", b_content="overlapping words: " + ", ".join(f"'{n}'" for n in names),
                description=f"text written over text: {', '.join(repr(n) for n in names)} overlap each other (up to {frac * 100:.0f}% of the smaller word)",
                bbox_a=tuple(map(float, region)), bbox_b=tuple(map(float, alignmod.map_box(inv, region))),
                confidence=0.88, engines=["overlap-text"],
                details={"words": names, "overlap_fraction": round(frac, 3), "text_over_text": True,
                         "region_a_frame": tuple(map(float, region))}, region=tuple(map(float, region))))
    return out
