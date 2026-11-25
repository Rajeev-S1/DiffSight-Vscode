"""Compare detected objects (stamps, signature-like / handwritten marks, images, blocks, rules)
between A and B and classify each as added / removed / modified / moved.

Matching is first by position (same place), then by appearance anywhere on the page
(rotation-tolerant for stamps), so a stamp that merely moved or rotated is never reported as
'removed + added'.  Only appearance is compared; nothing is said about who made a mark."""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import align as alignmod
from .align import Alignment
from .config import Config
from .models import BBox, Finding, bbox_inter, bbox_iou
from .visual_compare import VisualContext

GROUPS = {"stamp": "stamp", "handwriting": "hand", "mark": "hand", "image": "image", "block": "block",
          "line": "line"}
LABEL = {"stamp": "Stamp", "handwriting": "Handwriting", "mark": "Handwritten / ink mark", "image": "Image",
         "block": "Solid block", "line": "Line"}
PATCH = 64


def _crop_mask(ink: np.ndarray, box: BBox, pad: int = 2) -> np.ndarray:
    h, w = ink.shape
    x0, y0, x1, y1 = int(max(0, box[0] - pad)), int(max(0, box[1] - pad)), int(min(w, box[2] + pad)), int(min(h, box[3] + pad))
    return ink[y0:y1, x0:x1]


def _patch(mask: np.ndarray) -> Optional[np.ndarray]:
    """Aspect-preserving 64x64 blurred ink patch."""
    if mask.size == 0 or not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    m = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
    h, w = m.shape
    s = (PATCH - 6) / float(max(h, w))
    m = cv2.resize(m, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA)
    out = np.zeros((PATCH, PATCH), np.float32)
    y0, x0 = (PATCH - m.shape[0]) // 2, (PATCH - m.shape[1]) // 2
    out[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = m
    return cv2.GaussianBlur(out, (0, 0), 1.2)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 1e-9 else 0.0


def _best_rot(pa: np.ndarray, pb: np.ndarray, rotate: bool) -> Tuple[float, float]:
    n0 = _ncc(pa, pb)
    if not rotate or n0 >= 0.93:
        return n0, 0.0
    best, ang = n0, 0.0
    c = (PATCH / 2.0, PATCH / 2.0)
    for a in list(range(-175, 180, 10)):
        if a == 0:
            continue
        M = cv2.getRotationMatrix2D(c, a, 1.0)
        r = cv2.warpAffine(pb, M, (PATCH, PATCH), flags=cv2.INTER_LINEAR)
        n = _ncc(pa, r)
        if n > best:
            best, ang = n, float(a)
    for a in np.arange(ang - 5, ang + 5.1, 2.5):
        M = cv2.getRotationMatrix2D(c, a, 1.0)
        n = _ncc(pa, cv2.warpAffine(pb, M, (PATCH, PATCH), flags=cv2.INTER_LINEAR))
        if n > best:
            best, ang = n, float(a)
    return best, ang


def _stamp_text(bgr: np.ndarray, ink: np.ndarray, box: BBox, cfg: Config) -> Tuple[str, float]:
    """Best-effort OCR of the lettering inside a stamp (rotation tolerant by trying 4 orientations)."""
    try:
        from . import ocr as ocrmod
    except Exception:  # noqa: BLE001
        return "", 0.0
    h, w = ink.shape
    x0, y0, x1, y1 = int(max(0, box[0] - 4)), int(max(0, box[1] - 4)), int(min(w, box[2] + 4)), int(min(h, box[3] + 4))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return "", 0.0
    crop = np.where(ink[y0:y1, x0:x1], 0, 255).astype(np.uint8)
    crop = cv2.copyMakeBorder(crop, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    best = ("", 0.0)
    for k in (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE):
        img = crop if k is None else cv2.rotate(crop, k)
        try:
            txt, conf = ocrmod.recognize_crop(img)
        except Exception:  # noqa: BLE001
            return "", 0.0
        if txt and len(txt.strip()) >= 3:
            score = conf * len(re.sub(r"\W", "", txt))
            if score > best[1]:
                best = (txt, score)
        if k is None and best[1] > 400:
            break
    txt = best[0]
    return txt, (best[1] / max(1, len(re.sub(r"\W", "", txt)))) if txt else 0.0


def _change_fraction(ctx: VisualContext, box: BBox) -> Tuple[float, float, float, int]:
    """(changed fraction, retained fraction of A ink, extra fraction of B ink) inside ``box``,
    using the same ink tolerance as the visual engine."""
    ia, ib = _crop_mask(ctx.ink_a, box, 0), _crop_mask(ctx.ink_b, box, 0)
    if ia.size == 0:
        return 0.0, 1.0, 0.0, 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(ctx.tol_px * 2) | 1,) * 2)
    da = cv2.dilate(ia.astype(np.uint8), k) > 0
    db = cv2.dilate(ib.astype(np.uint8), k) > 0
    removed = int((ia & ~db).sum())
    added = int((ib & ~da).sum())
    total = max(1, int(ia.sum()) + int(ib.sum()))
    keep = 1.0 - removed / max(1, int(ia.sum()))
    extra = added / max(1, int(ib.sum()))
    return (removed + added) / total, float(keep), float(extra), removed + added


def _colour_compatible(a: dict, b: dict) -> bool:
    ca = a.get("features", {}).get("colour")
    cb = b.get("features", {}).get("colour")
    if not ca or not cb:
        return True
    na, nb = a["features"].get("colour_name"), b["features"].get("colour_name")
    if na == "black/grey" and nb == "black/grey":
        return True
    if na != nb:
        return False
    return float(np.linalg.norm(np.array(ca, float) - np.array(cb, float))) < 90.0


def _type_label(o: dict) -> Tuple[str, str]:
    """(dtype, human label). 'Signature' only when the surroundings suggest one; appearance only."""
    t = o["type"]
    f = o.get("features", {})
    if t in ("handwriting", "mark") and (f.get("near_signature_wording") or f.get("underline_below")) and t == "handwriting":
        return "Signature", "signature-like handwritten mark"
    if t == "handwriting":
        return "Handwriting", "handwritten mark"
    if t == "mark":
        return "Handwriting", "unclassified ink mark"
    return {"stamp": ("Stamp", "stamp"), "image": ("Image", "image"), "block": ("Block", "solid block"),
            "line": ("Line", "ruled line")}[t]


def compare_objects(ctx: VisualContext, objs_a: List[dict], objs_b: List[dict], al: Alignment, cfg: Config,
                    pa_no: int, pb_no: int) -> List[Finding]:
    inv = al.inverse()
    A = [dict(o, bbox_a=tuple(o["bbox"])) for o in objs_a if o["type"] in GROUPS]
    B = []
    for o in objs_b:
        if o["type"] not in GROUPS:
            continue
        B.append(dict(o, bbox_a=alignmod.map_box(al.M, o["bbox"]), bbox_own=tuple(o["bbox"])))
    # objects whose mapped box falls outside the valid area are not comparable
    mm = cfg.dpi / 25.4
    out: List[Finding] = []
    ma, mb = set(), set()
    pairs: List[Tuple[int, int, str]] = []

    def pa_patch(o):
        return _patch(_crop_mask(ctx.ink_a, o["bbox_a"]))

    def pb_patch(o):
        return _patch(_crop_mask(ctx.ink_b, o["bbox_a"]))

    # ---- phase 1: same place ---------------------------------------------------------------
    cand = []
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            ga, gb = GROUPS[a["type"]], GROUPS[b["type"]]
            if ga != gb and not ({ga, gb} <= {"stamp", "hand"}):
                continue
            iou = bbox_iou(a["bbox_a"], b["bbox_a"])
            if iou >= 0.25:
                cand.append((iou, i, j))
    for iou, i, j in sorted(cand, reverse=True):
        if i in ma or j in mb:
            continue
        ma.add(i)
        mb.add(j)
        pairs.append((i, j, "place"))
    # ---- phase 2: same appearance elsewhere ------------------------------------------------
    cand = []
    for i, a in enumerate(A):
        if i in ma or a["type"] in ("line",):
            continue
        pa = pa_patch(a)
        if pa is None:
            continue
        for j, b in enumerate(B):
            if j in mb or GROUPS[a["type"]] != GROUPS[b["type"]] or not _colour_compatible(a, b):
                continue
            wa, ha = a["bbox_a"][2] - a["bbox_a"][0], a["bbox_a"][3] - a["bbox_a"][1]
            wb, hb = b["bbox_a"][2] - b["bbox_a"][0], b["bbox_a"][3] - b["bbox_a"][1]
            if not (0.5 <= max(wa, ha) / max(wb, hb, 1) <= 2.0):
                continue
            pb = pb_patch(b)
            if pb is None:
                continue
            n, ang = _best_rot(pa, pb, a["type"] == "stamp")
            if n >= (0.88 if a["type"] == "stamp" else 0.80):
                cand.append((n, i, j, ang))
    for n, i, j, ang in sorted(cand, reverse=True):
        if i in ma or j in mb:
            continue
        ma.add(i)
        mb.add(j)
        pairs.append((i, j, "appearance"))

    # ---- classify pairs --------------------------------------------------------------------
    for i, j, how in pairs:
        a, b = A[i], B[j]
        dtype, label = _type_label(a if a["score"] >= b["score"] else b)
        box_a, box_b_af = a["bbox_a"], b["bbox_a"]
        union = (min(box_a[0], box_b_af[0]), min(box_a[1], box_b_af[1]), max(box_a[2], box_b_af[2]), max(box_a[3], box_b_af[3]))
        cent_a = ((box_a[0] + box_a[2]) / 2, (box_a[1] + box_a[3]) / 2)
        cent_b = ((box_b_af[0] + box_b_af[2]) / 2, (box_b_af[1] + box_b_af[3]) / 2)
        shift_mm = float(np.hypot(cent_a[0] - cent_b[0], cent_a[1] - cent_b[1])) / mm
        size_a = (box_a[2] - box_a[0], box_a[3] - box_a[1])
        size_b = (box_b_af[2] - box_b_af[0], box_b_af[3] - box_b_af[1])
        resized = abs(size_b[0] / max(size_a[0], 1) - 1) > 0.15 or abs(size_b[1] / max(size_a[1], 1) - 1) > 0.15
        pa_, pb_ = pa_patch(a), pb_patch(b)
        ncc, ang = (0.0, 0.0) if pa_ is None or pb_ is None else _best_rot(pa_, pb_, a["type"] == "stamp")
        ncc0 = 0.0 if pa_ is None or pb_ is None else _ncc(pa_, pb_)
        base_conf = min(a["score"], b["score"])
        changes: List[str] = []
        cat = None
        details = {"a_type": a["type"], "b_type": b["type"], "appearance_ncc": round(ncc, 3),
                   "region_a_frame": union, "shift_mm": round(shift_mm, 2)}
        if a["type"] == "image" and b["type"] == "image" and a.get("hash") and b.get("hash"):
            same_img = a["hash"] == b["hash"]
            if not same_img:
                cat, conf = "Modified", 0.97
                changes.append("embedded image content differs")
            elif shift_mm >= cfg.move_threshold_mm:
                cat, conf = "Moved", 0.97
                changes.append(f"image moved by {shift_mm:.1f} mm")
            elif resized:
                cat, conf = "Modified", 0.95
                changes.append("image resized")
            else:
                continue
        else:
            if shift_mm >= cfg.move_threshold_mm and ncc >= 0.8:
                cat = "Moved"
                changes.append(f"moved by {shift_mm:.1f} mm")
            if abs(ang) >= 8 and ncc >= 0.78 and ncc0 < ncc - 0.05:
                cat = cat or "Modified"
                changes.append(f"rotated by about {abs(ang):.0f} degrees")
                details["rotation_deg"] = ang
            if resized and ncc >= 0.7:
                cat = cat or "Modified"
                changes.append(f"size changed ({size_a[0] / mm:.0f}x{size_a[1] / mm:.0f} mm to {size_b[0] / mm:.0f}x{size_b[1] / mm:.0f} mm)")
            cf, keep, extra, cpx = _change_fraction(ctx, union) if how == "place" else (0.0, 1.0, 0.0, 0)
            changed_mm2 = cpx / (mm * mm)
            details.update({"changed_ink_fraction": round(cf, 3), "retained_fraction": round(keep, 3),
                            "extra_fraction": round(extra, 3), "changed_ink_mm2": round(changed_mm2, 2)})
            thr = 0.30 if ctx.scan else 0.12
            abs_thr = 10.0 if ctx.scan else 3.0     # a large object can change a lot without changing much *relatively*
            if how == "place" and (cf >= thr or (changed_mm2 >= abs_thr and cf >= 0.03)) and not (abs(ang) >= 8 and ncc >= 0.78 and ncc0 < ncc - 0.05):
                if keep < 0.75 and extra < 0.2:
                    changes.append("part of the original is missing / not visible in B (partially visible or covered)")
                elif extra > 0.4 and keep > 0.8:
                    changes.append("additional ink present on the object in B")
                else:
                    changes.append(f"appearance changed ({cf * 100:.0f}% of the ink, {changed_mm2:.1f} mm2, differs)")
                cat = cat or "Modified"
            elif how == "appearance" and ncc < 0.88 and a["type"] != "stamp":
                changes.append(f"appearance differs (similarity {ncc:.2f})")
                cat = cat or "Modified"
            # stamp lettering
            if a["type"] == "stamp" and b["type"] == "stamp" and (cat is not None or ncc < 0.97):
                ta, ca = _stamp_text(ctx.bgr_a, ctx.ink_a, box_a, cfg)
                tb, cb = _stamp_text(ctx.bgr_b, ctx.ink_b, box_b_af, cfg)
                details.update({"stamp_text_a": ta, "stamp_text_b": tb})
                if ta and tb and ta != tb and min(ca, cb) >= 50:
                    changes.append(f"stamp text appears to differ (OCR A: '{ta}', B: '{tb}')")
                    cat = cat or "Modified"
            if cat is None:
                continue
            conf = float(np.clip(0.55 + 0.35 * base_conf * min(1.0, 0.4 + 2.0 * max(details["changed_ink_fraction"], 1 - ncc)), 0.3, 0.95))
            if cat == "Moved":
                conf = float(np.clip(0.6 + 0.3 * ncc, 0.3, 0.95))
        a_txt = f"{label} present at ({cent_a[0] / mm:.0f}, {cent_a[1] / mm:.0f}) mm"
        b_txt = f"{label} present at ({cent_b[0] / mm:.0f}, {cent_b[1] / mm:.0f}) mm"
        out.append(Finding(category=cat, dtype=dtype, page_a=pa_no, page_b=pb_no, a_content=a_txt, b_content=b_txt,
                           description=f"{label}: " + "; ".join(changes), bbox_a=tuple(map(float, box_a)),
                           bbox_b=tuple(map(float, b["bbox_own"])), confidence=round(conf, 3),
                           engines=["object"], details=details, region=tuple(map(float, union))))

    def backed_by_pixels(o: dict, ink: np.ndarray) -> bool:
        """An object that is really added/removed has (almost) all of its ink unmatched on the other side."""
        if ctx.changed is None or o["type"] == "image":
            return True
        box = o["bbox_a"]
        h, w = ink.shape
        x0, y0, x1, y1 = max(0, int(box[0])), max(0, int(box[1])), min(w, int(box[2]) + 1), min(h, int(box[3]) + 1)
        n_ink = int(ink[y0:y1, x0:x1].sum())
        n_chg = int(ctx.changed[y0:y1, x0:x1].sum())
        return n_ink > 0 and n_chg / n_ink >= 0.35

    # ---- unmatched objects: removed / added --------------------------------------------------
    for i, a in enumerate(A):
        if i in ma:
            continue
        dtype, label = _type_label(a)
        if a["type"] == "line" and a["score"] < 0.9:
            continue
        if not backed_by_pixels(a, ctx.ink_a):
            continue
        box = a["bbox_a"]
        box_b_own = alignmod.map_box(inv, box)
        conf = float(np.clip(0.5 + 0.4 * a["score"], 0.3, 0.95))
        out.append(Finding(category="Removed", dtype=dtype, page_a=pa_no, page_b=pb_no,
                           a_content=f"{label} present", b_content=f"no {label} at this position",
                           description=f"{label} removed (present in A, absent from B)",
                           bbox_a=tuple(map(float, box)), bbox_b=tuple(map(float, box_b_own)), confidence=round(conf, 3),
                           engines=["object"], details={"a_type": a["type"], "features": a.get("features", {}),
                                                        "region_a_frame": tuple(map(float, box))},
                           region=tuple(map(float, box))))
    for j, b in enumerate(B):
        if j in mb:
            continue
        dtype, label = _type_label(b)
        if b["type"] == "line" and b["score"] < 0.9:
            continue
        if not backed_by_pixels(b, ctx.ink_b):
            continue
        box = b["bbox_a"]
        conf = float(np.clip(0.5 + 0.4 * b["score"], 0.3, 0.95))
        out.append(Finding(category="Added", dtype=dtype, page_a=pa_no, page_b=pb_no,
                           a_content=f"no {label} at this position", b_content=f"{label} present",
                           description=f"{label} added (absent from A, present in B)",
                           bbox_a=tuple(map(float, box)), bbox_b=tuple(map(float, b["bbox_own"])), confidence=round(conf, 3),
                           engines=["object"], details={"b_type": b["type"], "features": b.get("features", {}),
                                                        "region_a_frame": tuple(map(float, box))},
                           region=tuple(map(float, box))))
    return out
