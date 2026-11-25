"""Service 9 – Difference classification engine.

Combines the findings of all engines for one matched page pair:
  1. cross-validation   OCR-derived findings need independent visual support, otherwise they are
                        suppressed (OCR failure is never proof of a difference)
  2. merging            the same modification seen by several engines becomes ONE finding
                        (generic visual regions are absorbed by the specific finding that explains
                        their pixels; 'stamp added' + 'stamp covers text' become one finding)
  3. moved text         removed + added identical text => Moved (identical text without any pixel
                        change is OCR disagreement and dropped)
  4. de-duplication     findings of the same type whose regions overlap >= 60 % become one
  5. scoring / split    confirmed / uncertain / suppressed, with reasons for suppression
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import Config
from .models import BBox, Finding, bbox_area, bbox_inter, bbox_pad, bbox_union
from .textutil import normalize_ocr_text
from .visual_compare import VisualContext, local_ncc

MOVE_MIN_KEY = 3          # shortest normalised text that may be paired as "moved" (was 6)
NO_CROSSCHECK_CAP = 0.6   # OCR-only findings on pages that could not be aligned stay "uncertain"
NCC_SUPPORT = 0.955   # identical scanned text correlates >= 0.98 after local re-alignment

PRIORITY = {"Signature": 90, "Stamp": 90, "Image": 88, "Handwriting": 85, "Block": 80, "Hidden text": 79,
            "Overlap": 78, "Number": 72, "Punctuation": 72, "Text": 70, "Font": 60, "Layout": 50,
            "Line": 45, "Colour": 40, "Visual": 10}
OBJECT_TYPES = ("Signature", "Stamp", "Image", "Handwriting", "Block", "Line")
TEXT_TYPES = ("Text", "Number", "Punctuation")


def _count_changed(ctx: VisualContext, box: BBox) -> int:
    if ctx.changed is None:
        return 0
    h, w = ctx.changed.shape
    x0, y0, x1, y1 = max(0, int(box[0])), max(0, int(box[1])), min(w, int(box[2]) + 1), min(h, int(box[3]) + 1)
    return int(ctx.changed[y0:y1, x0:x1].sum()) if x1 > x0 and y1 > y0 else 0


def _inter_frac(a: BBox, b: BBox) -> float:
    """Intersection area as a fraction of the smaller box."""
    s = min(bbox_area(a), bbox_area(b))
    return bbox_inter(a, b) / s if s > 0 else 0.0


def _ink_kept(ctx: VisualContext, box: BBox, pad: int, tol: int, a_in_b: bool, need: float) -> bool:
    """True if the ink of A (a_in_b) / of B (not a_in_b) inside the box is also present in the other aligned page,
    i.e. nothing was removed from / added at that place."""
    h, w = ctx.ink_a.shape
    x0, y0 = max(0, int(box[0]) - pad), max(0, int(box[1]) - pad)
    x1, y1 = min(w, int(box[2]) + pad + 1), min(h, int(box[3]) + pad + 1)
    if x1 <= x0 or y1 <= y0:
        return False
    src, dst = (ctx.ink_a, ctx.ink_b) if a_in_b else (ctx.ink_b, ctx.ink_a)
    s_ = src[y0:y1, x0:x1].astype(np.uint8)
    d_ = dst[y0:y1, x0:x1].astype(np.uint8)
    if s_.sum() < 20:
        return False
    import cv2
    k = np.ones((2 * max(1, tol) + 1, 2 * max(1, tol) + 1), np.uint8)
    return float((s_ & cv2.dilate(d_, k)).sum()) / float(s_.sum()) >= need


def _same_lines(layout_region: BBox, text_region: Optional[BBox], pad: float) -> bool:
    """True if the layout region is made of the lines the text finding is on: at least half of the layout
    region's height lies inside the text finding's vertical extent and the two overlap horizontally."""
    if text_region is None:
        return False
    t = bbox_pad(text_region, pad)
    h = layout_region[3] - layout_region[1]
    if h <= 0:
        return False
    vy = min(layout_region[3], t[3]) - max(layout_region[1], t[1])
    hx = min(layout_region[2], t[2]) - max(layout_region[0], t[0])
    return vy >= 0.5 * h and hx > 0


def _norm(s: str) -> str:
    return "".join(ch for ch in normalize_ocr_text(s).lower() if ch.isalnum())


def classify_pair(findings: List[Finding], ctx: Optional[VisualContext], cfg: Config
                  ) -> Tuple[List[Finding], List[dict]]:
    """Return (kept findings with status set, suppressed diagnostics)."""
    suppressed: List[dict] = []
    mm = cfg.dpi / 25.4
    min_px = max(4, int(cfg.mm2(cfg.min_ink_mm2)))
    pad = int(cfg.mm(1.5))
    work = [f for f in findings]

    # ---- 1) cross-validate text findings against the visual evidence -------------------------
    kept: List[Finding] = []
    for f in work:
        if f.dtype in TEXT_TYPES + ("Layout",) and f.region is not None and ctx is not None:
            n = _count_changed(ctx, bbox_pad(f.region, pad))
            supported = n >= min_px
            ncc_note = None
            if not supported and f.details.get("ocr_only") and f.bbox_a is not None and f.details.get("a_tokens") \
                    and f.details.get("b_tokens"):
                ncc = local_ncc(ctx, bbox_pad(f.bbox_a, 2), cfg)
                ncc_note = round(ncc, 3)
                if ncc < NCC_SUPPORT:
                    supported = True
                    f.details["local_ncc"] = ncc_note
            if f.details.get("ocr_only") or (f.engines == ["layout-text"] and ctx.scan):
                if not supported:
                    ocr_conf = f.confidence / 0.9 * 100 if f.details.get("ocr_only") else 0.0
                    if f.details.get("ocr_only") and ocr_conf >= 80:
                        # confident OCR on both sides but no independent visual evidence: keep, never confirm
                        f.confidence = round(min(f.confidence, 0.6), 3)
                        f.details["note"] = ("OCR-only difference: both readings are confident but the pixel comparison "
                                             "shows no clear change (local correlation %s); verify manually" % ncc_note)
                        f.engines.append("ocr-only")
                        kept.append(f)
                        continue
                    suppressed.append({"reason": "OCR-derived difference without independent visual support",
                                       "description": f.description, "page": f.page_a,
                                       "a": f.a_content, "b": f.b_content})
                    continue
                f.confidence = round(min(0.95, f.confidence + 0.08), 3)
                f.engines.append("visual")
            elif supported:
                if "visual" not in f.engines:
                    f.engines.append("visual")
            elif f.dtype in TEXT_TYPES:
                f.details["note"] = "text-layer difference without a visible change (hidden / invisible text)"
                f.confidence = round(min(f.confidence, 0.8), 3)
        elif ctx is None and (f.details.get("ocr_only") or f.details.get("ocr_derived")):
            # the pages could not be aligned, so there is no pixel cross-check: an OCR-only finding is never confirmed
            f.confidence = round(min(f.confidence, NO_CROSSCHECK_CAP), 3)
            f.details["uncertain_reason"] = "pages not aligned: OCR difference could not be cross-checked visually"
            if "no-visual-check" not in f.engines:
                f.engines.append("no-visual-check")
        kept.append(f)
    work = kept

    # ---- 2a) removed + added with identical content ------------------------------------------------
    #   * same normalised text and no real pixel change  => the two OCR readings disagree: drop both
    #   * same normalised text at another place          => the text moved
    #   Numbers pair too ("70 - 100" removed here, added there); very short keys (< MOVE_MIN_KEY) are ignored.
    PAIRABLE = ("Text", "Number")
    removed = [f for f in work if f.dtype in PAIRABLE and f.category == "Removed"]
    added = [f for f in work if f.dtype in PAIRABLE and f.category == "Added"]
    used = set()
    for r in removed:
        key = _norm(r.details.get("a_changed", ""))
        if len(key) < MOVE_MIN_KEY:
            continue
        for a in added:
            if id(a) in used:
                continue
            if _norm(a.details.get("b_changed", "")) != key:
                continue
            ra, rb = r.region, a.region
            if ctx is not None and ra is not None and rb is not None and \
                    _count_changed(ctx, bbox_pad(ra, pad)) + _count_changed(ctx, bbox_pad(rb, pad)) < min_px:
                used.add(id(a))
                used.add(id(r))
                suppressed.append({"reason": "OCR disagreement: identical text removed and added, pixels show no change",
                                   "kind": "ocr_disagreement", "description": r.description, "page": r.page_a,
                                   "a": r.a_content, "b": a.b_content})
                break
            used.add(id(a))
            used.add(id(r))
            dist = float(np.hypot((ra[0] + ra[2]) / 2 - (rb[0] + rb[2]) / 2, (ra[1] + ra[3]) / 2 - (rb[1] + rb[3]) / 2)) / mm
            if ctx is not None and ctx.scan and (dist < 3.0 or (
                    _ink_kept(ctx, ra, pad, int(max(1, ctx.tol_px)), True, 0.9) and _ink_kept(ctx, rb, pad, int(max(1, ctx.tol_px)), False, 0.75))):
                suppressed.append({"reason": "OCR false move: the pixels at both places are unchanged",
                                   "kind": "ocr_false_move", "description": r.description, "page": r.page_a,
                                   "a": r.a_content, "b": a.b_content})
                continue
            m = Finding(category="Moved", dtype=r.dtype, page_a=r.page_a, page_b=a.page_b,
                        a_content=r.details.get("a_changed", ""), b_content=a.details.get("b_changed", ""),
                        description=f"{'number' if r.dtype == 'Number' else 'text'} moved by about {dist:.0f} mm: '{r.details.get('a_changed', '')[:80]}'",
                        bbox_a=r.bbox_a, bbox_b=a.bbox_b, confidence=round(min(r.confidence, a.confidence), 3),
                        engines=sorted(set(r.engines + a.engines)),
                        details={**r.details, "moved_mm": round(dist, 1),
                                 "region_a_frame": bbox_union([ra, rb])},
                        region=bbox_union([ra, rb]))
            work.append(m)
            break
    work = [f for f in work if id(f) not in used]

    # ---- 2a2) a layout shift that sits on the lines of a text finding is part of that finding ----
    #   ("2 lines moved by 5 mm" next to "text added: ..." on the same lines is the consequence of the edit)
    texts = [f for f in work if f.dtype in TEXT_TYPES and f.category in ("Added", "Removed", "Modified", "Moved")
             and f.region is not None]
    lay_drop = set()
    for f in work:
        if f.dtype != "Layout" or f.category != "Moved" or f.region is None:
            continue
        for t in texts:
            if _same_lines(f.region, t.details.get("explain_region") or t.region, pad):
                t.details.setdefault("merged_layout", []).append(f.description)
                t.details["layout_shift_mm"] = (f.details.get("dx_mm"), f.details.get("dy_mm"))
                # the shifted pixels of those lines are explained by this finding now
                t.details["explain_region"] = bbox_union([t.details.get("explain_region") or t.region, f.region])
                t.description += f"; the same lines also shifted ({f.description})"
                for e in f.engines:
                    if e not in t.engines:
                        t.engines.append(e)
                lay_drop.add(id(f))
                break
    work = [f for f in work if id(f) not in lay_drop]

    # ---- 2a3) a layout shift directly below an ADDED line (or above a REMOVED one) is its consequence ----
    #   ("7 lines moved down by 6 mm" right under "Text added: Wireless mouse ..." is not a separate edit)
    lay_drop = set()
    for f in work:
        if f.dtype != "Layout" or f.category != "Moved" or f.region is None:
            continue
        dx_mm, dy_mm = f.details.get("dx_mm") or 0.0, f.details.get("dy_mm") or 0.0
        if abs(dx_mm) > 2.0 or abs(dy_mm) < 0.5:
            continue
        for t in work:
            if t.dtype not in TEXT_TYPES or t.region is None:
                continue
            if not ((t.category == "Added" and dy_mm > 0) or (t.category == "Removed" and dy_mm < 0)):
                continue
            th = t.region[3] - t.region[1]
            if abs(dy_mm) * mm > 2.0 * th + 4 * mm:           # shift far bigger than the edit: unrelated
                continue
            hx = min(f.region[2], t.region[2]) - max(f.region[0], t.region[0])
            if t.category == "Added":
                gap = f.region[1] - t.region[3]
            else:
                gap = t.region[1] - f.region[3]
            if hx > 0 and -2 * th <= gap <= 8 * mm:
                t.details.setdefault("merged_layout", []).append(f.description)
                t.details["layout_shift_mm"] = (dx_mm, dy_mm)
                t.details["explain_region"] = bbox_union([t.details.get("explain_region") or t.region, f.region])
                t.description += f"; the lines below also shifted ({f.description})"
                for e in f.engines:
                    if e not in t.engines:
                        t.engines.append(e)
                lay_drop.add(id(f))
                break
    work = [f for f in work if id(f) not in lay_drop]


    # ---- 2a3) 'block' findings lying (>= 70 %) inside a stamp finding are part of the stamp ---------
    stamps = [f for f in work if f.dtype == "Stamp" and f.region is not None]
    blk_drop = set()
    for b in work:
        if b.dtype != "Block" or b.region is None or bbox_area(b.region) <= 0:
            continue
        for s_ in stamps:
            if bbox_inter(b.region, s_.region) / bbox_area(b.region) >= 0.70:
                s_.details["merged_block"] = True
                s_.details.setdefault("merged_block_regions", []).append(tuple(map(float, b.region)))
                for e in b.engines:
                    if e not in s_.engines:
                        s_.engines.append(e)
                blk_drop.add(id(b))
                break
    work = [f for f in work if id(f) not in blk_drop]

    # ---- 2b) merge object finding + overlap finding of the same object -----------------------------
    objs = [f for f in work if f.dtype in OBJECT_TYPES and f.category in ("Added", "Removed", "Moved", "Modified")]
    ovs = [f for f in work if f.dtype == "Overlap" and f.details.get("overlay_type")]
    drop = set()
    for ov in ovs:
        want = {"B": ("Added", "Moved", "Modified"), "A": ("Removed", "Moved", "Modified")}[ov.details.get("overlay_in", "B")]
        best = None
        for o in objs:
            if o.category not in want or id(o) in drop:
                continue
            if _inter_frac(o.region, ov.region) >= 0.4:
                if best is None or PRIORITY.get(o.dtype, 0) > PRIORITY.get(best.dtype, 0):
                    best = o
        if best is not None:
            covered = ", ".join(f"'{w}'" for w in ov.details.get("covered_words", [])[:6])
            best.details["also_overlapped"] = True
            best.details["covered_words"] = ov.details.get("covered_words", [])
            best.details["hidden_content"] = ov.details.get("hidden_content")
            best.details["visible_reading"] = ov.details.get("visible_reading")
            best.details["layer_separation"] = ov.details.get("layer_separation", False)
            best.details["original_category"] = best.category
            best.category = "Overlapped"
            best.description += f"; covers text {covered}. {ov.details.get('note', '')}"
            best.engines = sorted(set(best.engines + ov.engines))
            best.confidence = round(max(best.confidence, ov.confidence), 3)
            best.region = bbox_union([best.region, ov.region])
            best.details["region_a_frame"] = best.region
            drop.add(id(ov))
    work = [f for f in work if id(f) not in drop]

    # ---- 2b2) text that was *added on top of* existing text is one finding, not two ---------------
    tot = [f for f in work if f.dtype == "Overlap" and f.details.get("text_over_text")]
    drop = set()
    for ov in tot:
        for f in work:
            if f.dtype in TEXT_TYPES and f.category in ("Added", "Modified") and f.region is not None \
                    and _inter_frac(bbox_pad(ov.region, pad), f.region) >= 0.6 and id(f) not in drop:
                ov.details["added_text"] = f.details.get("b_changed", "")
                ov.description += f"; the new text '{f.details.get('b_changed', '')[:60]}' is what was written over the existing text"
                ov.engines = sorted(set(ov.engines + f.engines))
                drop.add(id(f))
    work = [f for f in work if id(f) not in drop]

    # ---- 2b3) weak 'ink mark' objects touching a stronger object finding are fragments of it ------
    strong = [f for f in work if f.dtype in ("Stamp", "Signature", "Image", "Block") and f.region is not None]
    weak_drop = set()
    for f in work:
        if f.dtype == "Handwriting" and (f.details.get("a_type") == "mark" or f.details.get("b_type") == "mark") \
                and f.confidence < 0.85:
            if any(_inter_frac(bbox_pad(s_.region, int(cfg.mm(5))), f.region) > 0 for s_ in strong):
                weak_drop.add(id(f))
    work = [f for f in work if id(f) not in weak_drop]

    # ---- 2b4) underline / rule that belongs to a signature or other object finding ---------------
    holders = [f for f in work if f.dtype in ("Signature", "Handwriting", "Stamp") and f.region is not None]
    line_drop = set()
    for f in work:
        if f.dtype == "Line" and f.category in ("Added", "Removed"):
            for h in holders:
                if _inter_frac(bbox_pad(h.region, int(cfg.mm(3))), f.region) >= 0.8:
                    h.details["includes_line"] = True
                    h.region = bbox_union([h.region, f.region])
                    line_drop.add(id(f))
                    break
    work = [f for f in work if id(f) not in line_drop]

    # ---- 2b5) layout deltas that merely follow from a font change on the same line ---------------
    fonts = [f for f in work if f.dtype == "Font" and f.region is not None]
    work = [f for f in work if not (f.dtype == "Layout" and f.details.get("lines") == 1 and
                                    any(_inter_frac(g.region, f.region) >= 0.8 for g in fonts))]

    # ---- 2c) absorb generic visual / colour regions explained by more specific findings -----------
    specific = [f for f in work if f.dtype not in ("Visual", "Colour") and f.region is not None]

    def reg_of(s_: Finding) -> BBox:
        return s_.details.get("explain_region") or s_.region

    out: List[Finding] = []
    for f in work:
        if f.dtype not in ("Visual", "Colour"):
            out.append(f)
            continue
        cover = [s for s in specific if _inter_frac(bbox_pad(reg_of(s), pad), f.region) > 0]
        if not cover:
            out.append(f)
            continue
        if f.dtype == "Colour":
            if any(_inter_frac(bbox_pad(reg_of(s), pad), f.region) >= 0.6 for s in cover):
                for s in cover:
                    if "visual-colour" not in s.engines:
                        s.engines.append("visual-colour")
                continue
            out.append(f)
            continue
        # pixels of this visual region that the covering findings account for
        total = _count_changed(ctx, f.region) if ctx is not None else 0
        if total == 0:
            out.append(f)
            continue
        sub = np.zeros((int(f.region[3] - f.region[1]) + 1, int(f.region[2] - f.region[0]) + 1), bool)
        for s in cover:
            b = bbox_pad(reg_of(s), pad)
            x0, y0 = int(max(b[0], f.region[0]) - f.region[0]), int(max(b[1], f.region[1]) - f.region[1])
            x1, y1 = int(min(b[2], f.region[2]) - f.region[0]), int(min(b[3], f.region[3]) - f.region[1])
            if x1 > x0 and y1 > y0:
                sub[y0:y1 + 1, x0:x1 + 1] = True
        rx0, ry0 = int(f.region[0]), int(f.region[1])
        ch = ctx.changed[ry0:ry0 + sub.shape[0], rx0:rx0 + sub.shape[1]]
        sub = sub[:ch.shape[0], :ch.shape[1]]
        explained = float((ch & sub).sum()) / max(1, int(ch.sum()))
        if explained >= 0.65:
            for s in cover:
                if "visual" not in s.engines:
                    s.engines.append("visual")
                    if s.dtype not in TEXT_TYPES:   # two independent engines agree
                        s.confidence = round(min(0.95, max(s.confidence, f.confidence) + 0.05), 3)
            continue
        f.details["explained_fraction"] = round(explained, 3)
        f.description += " (not explained by any text / object finding)"
        out.append(f)
    work = out

    # ---- 2d) final de-duplication: same type, same category, regions overlapping >= 60 % ------------
    work = _dedupe(work)

    # ---- 3) status ---------------------------------------------------------------------------------
    final: List[Finding] = []
    for f in work:
        # a scanned number edit that the pixels confirm (NCC / ink check passed) is not left in the uncertain list
        if f.dtype == "Number" and f.category == "Modified" and f.details.get("ocr_only") \
                and "visual" in f.engines and 0.62 <= f.confidence < cfg.confirmed_min:
            f.confidence = cfg.confirmed_min
        if cfg.severe_poor_scan and f.details.get("ocr_only") and f.confidence >= cfg.confirmed_min:
            f.confidence = cfg.confirmed_min - 0.01       # severely degraded scan: fine-detail OCR findings need a manual look
            f.details["quality_note"] = "scan severely degraded (blur / noise / low contrast): please verify manually"
        if f.confidence < cfg.uncertain_min:
            suppressed.append({"reason": f"confidence {f.confidence:.2f} below {cfg.uncertain_min}",
                               "description": f.description, "page": f.page_a, "a": f.a_content, "b": f.b_content})
            continue
        if f.confidence >= cfg.confirmed_min:
            f.status = "confirmed"
        else:
            f.status = "uncertain"
            f.details["original_category"] = f.category
            f.category = "Uncertain"
        final.append(f)
    return final, suppressed


DEDUP_OVERLAP = 0.60


def _dedupe(work: List[Finding]) -> List[Finding]:
    """Merge findings of the same type (and category / page) whose regions overlap by >= 60 % of the
    smaller region, so the totals never count one difference twice. The stronger finding survives."""
    work = sorted(work, key=lambda f: -f.confidence)
    kept: List[Finding] = []
    for f in work:
        host = None
        if f.region is not None:
            for k in kept:
                if k.region is not None and k.dtype == f.dtype and k.category == f.category \
                        and k.page_a == f.page_a and k.page_b == f.page_b \
                        and _inter_frac(k.region, f.region) >= DEDUP_OVERLAP:
                    host = k
                    break
        if host is None:
            kept.append(f)
            continue
        host.region = bbox_union([host.region, f.region])
        host.details["region_a_frame"] = host.region
        host.details["merged_duplicates"] = host.details.get("merged_duplicates", 0) + 1 + f.details.get("merged_duplicates", 0)
        for e in f.engines:
            if e not in host.engines:
                host.engines.append(e)
    return kept


def sort_findings(findings: List[Finding]) -> List[Finding]:
    def key(f: Finding):
        pg = f.page_a if f.page_a is not None else (f.page_b or 0)
        r = f.region or f.bbox_a or f.bbox_b or (0, 0, 0, 0)
        return (pg, round(r[1] / 40.0), r[0])
    return sorted(findings, key=key)
