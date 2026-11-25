"""Service 7 – Visual comparison engine (independent of OCR).

Page B is warped into the frame of page A and compared as an image:

  * tolerance-based ink difference (ink that has no counterpart within `tol` mm on the other side).
    A tolerance instead of a plain XOR is what keeps scan jitter / anti-aliasing from being
    reported while still catching a 0.5 mm full stop.
  * clustering of changed pixels into regions
  * per-region local re-registration: if the region's content matches the other side after a small
    shift it is 'moved' (or scan jitter) instead of added/removed
  * SSIM per region as an extra evidence value
  * colour / tint changes (highlights, background fills) via Lab delta-E on ink-free areas
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from . import align as alignmod
from .align import Alignment
from .config import Config
from .models import BBox, Finding, PageData
from .preprocess import Prepared, ink_mask


SCAN_COLOUR_CAP = 0.85   # colour evidence on a scan never exceeds this (lighting / colour casts)


@dataclass
class VisualContext:
    """Everything the later engines need, all in the A frame."""
    h: int
    w: int
    bgr_a: np.ndarray
    bgr_b: np.ndarray          # B warped into A
    norm_a: np.ndarray
    norm_b: np.ndarray
    ink_a: np.ndarray
    ink_b: np.ndarray
    valid: np.ndarray          # bool: A pixels that B actually covers
    scan: bool
    tol_px: float
    stats: dict
    added: Optional[np.ndarray] = None     # B ink with no counterpart in A (filtered, A frame)
    removed: Optional[np.ndarray] = None   # A ink with no counterpart in B
    changed: Optional[np.ndarray] = None   # union


def build_context(pa: Prepared, pb: Prepared, al: Alignment, cfg: Config, scan: bool) -> VisualContext:
    h, w = pa.gray.shape
    M = al.M
    bgr_b = alignmod.warp_to_a(pb.bgr, M, (h, w), border=255, interp=cv2.INTER_LINEAR)
    norm_b = alignmod.warp_to_a(pb.norm, M, (h, w), border=255, interp=cv2.INTER_LINEAR)
    sat_b = alignmod.warp_to_a(pb.sat, M, (h, w), border=0, interp=cv2.INTER_LINEAR)
    ones = np.full(pb.gray.shape, 255, np.uint8)
    valid = alignmod.warp_to_a(ones, M, (h, w), border=0, interp=cv2.INTER_NEAREST) > 127
    ink_b = ink_mask(norm_b, sat_b, scan) & valid
    tol_mm = cfg.tol_mm_scan if scan else cfg.tol_mm_native
    tol_px = max(1.0, cfg.mm(tol_mm) + 0.5 * al.residual_px * (1 if scan else 0))
    return VisualContext(h=h, w=w, bgr_a=pa.bgr, bgr_b=bgr_b, norm_a=pa.norm, norm_b=norm_b,
                         ink_a=pa.ink & valid, ink_b=ink_b, valid=valid, scan=scan, tol_px=tol_px,
                         stats={"tol_px": round(tol_px, 2), "valid_frac": float(valid.mean())})


def _kernel(r: float) -> np.ndarray:
    d = int(round(2 * r)) + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))


def _ink_density(norm: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return cv2.GaussianBlur(255.0 - norm[y0:y1, x0:x1].astype(np.float32), (0, 0), 1.2)


def _local_shift(ctx: VisualContext, box: Tuple[int, int, int, int], cfg: Config,
                 a_to_b: bool, search_mm: Optional[float] = None) -> Tuple[float, float, float]:
    """Find (dx, dy, ncc) such that content of A inside ``box`` appears in B displaced by (dx, dy).
    With a_to_b False, the roles are swapped (content of B located in A). ``search_mm`` widens the
    search window (default cfg.local_shift_mm)."""
    x0, y0, x1, y1 = box
    m = int(cfg.mm(cfg.local_shift_mm if search_mm is None else search_mm))
    src, dst = (ctx.norm_a, ctx.norm_b) if a_to_b else (ctx.norm_b, ctx.norm_a)
    X0, Y0, X1, Y1 = max(0, x0 - 2), max(0, y0 - 2), min(ctx.w, x1 + 2), min(ctx.h, y1 + 2)
    if X1 - X0 < 6 or Y1 - Y0 < 6:
        return 0.0, 0.0, 0.0
    tpl = cv2.GaussianBlur(255.0 - src[Y0:Y1, X0:X1].astype(np.float32), (0, 0), 1.2)
    if tpl.std() < 1.0:
        return 0.0, 0.0, 0.0
    SX0, SY0, SX1, SY1 = max(0, X0 - m), max(0, Y0 - m), min(ctx.w, X1 + m), min(ctx.h, Y1 + m)
    win = cv2.GaussianBlur(255.0 - dst[SY0:SY1, SX0:SX1].astype(np.float32), (0, 0), 1.2)
    if win.shape[0] < tpl.shape[0] or win.shape[1] < tpl.shape[1]:
        return 0.0, 0.0, 0.0
    res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(res)
    return float(SX0 + loc[0] - X0), float(SY0 + loc[1] - Y0), float(mx)


def local_ncc(ctx: VisualContext, box: BBox, cfg: Config, margin_mm: float = 1.5) -> float:
    """Best correlation of A's crop inside ``box`` with B after a small local re-alignment.
    Identical content scores >= ~0.98 even on noisy scans; changed glyphs score lower."""
    dx, dy, ncc = _local_shift(ctx, tuple(int(v) for v in box), cfg, a_to_b=True)
    return ncc


def local_move(ctx: VisualContext, box: BBox, cfg: Config, search_mm: float) -> Tuple[float, float, float]:
    """Where does A's crop inside ``box`` appear in B? Returns (dx_mm, dy_mm, ncc), searching +-search_mm.
    Used to confirm that OCR-reported moved lines really sit at a different place in the pixels."""
    dx, dy, ncc = _local_shift(ctx, tuple(int(v) for v in box), cfg, a_to_b=True, search_mm=search_mm)
    mm = 25.4 / cfg.dpi
    return dx * mm, dy * mm, ncc


def _region_ssim(ctx: VisualContext, box) -> float:
    x0, y0, x1, y1 = box
    a = ctx.norm_a[y0:y1, x0:x1]
    b = ctx.norm_b[y0:y1, x0:x1]
    if min(a.shape) < 7:
        return 1.0 if np.array_equal(a, b) else 0.5
    try:
        return float(structural_similarity(a, b, data_range=255, win_size=7))
    except ValueError:
        return 1.0


def _components(mask: np.ndarray, min_area: int):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return n, lab, st


def compare_visual(ctx: VisualContext, page_a: PageData, page_b: PageData, al: Alignment, cfg: Config,
                   pa_no: int, pb_no: int) -> Tuple[List[Finding], dict]:
    """Return visual findings (A frame regions). All findings carry dtype 'Visual'; the
    classification engine later refines them using object / text context."""
    h, w = ctx.h, ctx.w
    scan = ctx.scan
    ctx.added = ctx.removed = np.zeros((h, w), bool)
    ctx.changed = ctx.added
    k = _kernel(ctx.tol_px)
    dil_a = cv2.dilate(ctx.ink_a.astype(np.uint8), k) > 0
    dil_b = cv2.dilate(ctx.ink_b.astype(np.uint8), k) > 0
    added = ctx.ink_b & ~dil_a
    removed = ctx.ink_a & ~dil_b

    min_px = max(2, int(cfg.mm2(cfg.min_ink_mm2) * (3.0 if scan else 1.0)))
    min_thick = 1.0 if scan else 0.0
    changed = []
    for name, m in (("added", added), ("removed", removed)):
        if not m.any():
            continue
        if min_thick > 0:  # drop thin slivers caused by edge misregistration
            dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
            n, lab, st = _components(m, 1)
            keep = np.zeros(n, bool)
            for i in range(1, n):
                x, y, ww, hh, a = st[i]
                if a >= min_px and dist[y:y + hh, x:x + ww][lab[y:y + hh, x:x + ww] == i].max() >= min_thick:
                    keep[i] = True
            m = keep[lab]
        else:
            n, lab, st = _components(m, 1)
            keep = np.zeros(n, bool)
            keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_px
            m = keep[lab]
        changed.append((name, m))
    stats = {"jitter_suppressed": 0, "regions": 0}
    findings: List[Finding] = []
    ctx.added = next((m for n_, m in changed if n_ == "added"), np.zeros((h, w), bool))
    ctx.removed = next((m for n_, m in changed if n_ == "removed"), np.zeros((h, w), bool))
    ctx.changed = ctx.added | ctx.removed
    if not changed:
        stats.update(_colour_pass(ctx, cfg, np.zeros((h, w), bool), findings, pa_no, pb_no, al))
        return findings, stats

    add_m = next((m for n, m in changed if n == "added"), np.zeros((h, w), bool))
    rem_m = next((m for n, m in changed if n == "removed"), np.zeros((h, w), bool))
    union = add_m | rem_m
    gap = cfg.mm(1.4)
    clus = cv2.dilate(union.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(gap * 2) | 1, int(gap) | 1))) > 0
    n, lab, st = _components(clus, 1)
    pad = int(cfg.mm(0.6))
    for i in range(1, n):
        x, y, ww, hh, _a = st[i]
        reg = lab[y:y + hh, x:x + ww] == i
        nA = int((rem_m[y:y + hh, x:x + ww] & reg).sum())
        nB = int((add_m[y:y + hh, x:x + ww] & reg).sum())
        if nA + nB < min_px:
            continue
        # tight box of the changed pixels (not of the dilated cluster)
        ys, xs = np.nonzero((union[y:y + hh, x:x + ww] & reg))
        bx = (x + int(xs.min()), y + int(ys.min()), x + int(xs.max()) + 1, y + int(ys.max()) + 1)
        box = (max(0, bx[0] - pad), max(0, bx[1] - pad), min(w, bx[2] + pad), min(h, bx[3] + pad))
        category = "Modified" if (nA and nB) else ("Added" if nB else "Removed")
        # --- local re-registration: same content at a slightly different place?
        moved_mm = 0.0
        if nA and nB and 0.4 <= nA / max(nB, 1) <= 2.5:
            dx, dy, ncc = _local_shift(ctx, box, cfg, a_to_b=True)
            if ncc >= 0.80:
                shift_mm = float(np.hypot(dx, dy)) * 25.4 / cfg.dpi
                if shift_mm >= cfg.move_threshold_mm:
                    category, moved_mm = "Moved", shift_mm
                elif scan:
                    stats["jitter_suppressed"] += 1
                    continue
        ssim = _region_ssim(ctx, box)
        area_mm2 = (nA + nB) / ((cfg.dpi / 25.4) ** 2)
        # confidence: bigger / more structural change => more certain; noisy scans lower it
        size_term = min(1.0, area_mm2 / (cfg.min_ink_mm2 * (12 if not scan else 25)))
        conf = 0.55 + 0.25 * size_term + 0.15 * min(1.0, max(0.0, (0.98 - ssim) / 0.3))
        if scan:
            conf -= min(0.2, 0.03 * ctx.stats.get("noise", 0.0))
            conf -= 0.1 if al.ncc < 0.6 else 0.0
        if not al.ok:
            conf -= 0.25
        conf = float(np.clip(conf, 0.05, 0.95))
        box_b = alignmod.map_box(al.inverse(), box)
        a_txt = "ink present" if nA else "nothing here"
        b_txt = "ink present" if nB else "nothing here"
        desc = {"Added": "additional ink in B with no counterpart in A",
                "Removed": "ink present in A is absent from B",
                "Modified": "ink differs between A and B",
                "Moved": f"same content shifted by {moved_mm:.1f} mm"}[category]
        findings.append(Finding(
            category=category, dtype="Visual", page_a=pa_no, page_b=pb_no, a_content=a_txt, b_content=b_txt,
            description=desc, bbox_a=tuple(map(float, box)), bbox_b=tuple(map(float, box_b)),
            confidence=round(conf, 3), engines=["visual"],
            details={"removed_px": nA, "added_px": nB, "area_mm2": round(area_mm2, 3), "ssim": round(ssim, 3),
                     "moved_mm": round(moved_mm, 2), "region_a_frame": tuple(map(float, box))},
            region=tuple(map(float, box))))
        stats["regions"] += 1

    stats.update(_colour_pass(ctx, cfg, clus, findings, pa_no, pb_no, al))
    return findings, stats


def _colour_pass(ctx: VisualContext, cfg: Config, exclude: np.ndarray, findings: List[Finding],
                 pa_no: int, pb_no: int, al: Alignment) -> dict:
    """Highlights, background fills and tints: large-ish areas whose colour differs."""
    h, w = ctx.h, ctx.w
    sc = 0.25
    a = cv2.resize(ctx.bgr_a, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    b = cv2.resize(ctx.bgr_b, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    a = cv2.GaussianBlur(a, (0, 0), 1.5)
    b = cv2.GaussianBlur(b, (0, 0), 1.5)
    la = cv2.cvtColor(a, cv2.COLOR_BGR2LAB).astype(np.float32)
    lb = cv2.cvtColor(b, cv2.COLOR_BGR2LAB).astype(np.float32)
    # OpenCV 8-bit Lab: L in 0..255 (scaled 255/100), a/b offset 128
    de = np.sqrt(((la[..., 0] - lb[..., 0]) * 100.0 / 255.0) ** 2 + (la[..., 1] - lb[..., 1]) ** 2 + (la[..., 2] - lb[..., 2]) ** 2)
    valid = cv2.resize(ctx.valid.astype(np.uint8), (de.shape[1], de.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    thr = cfg.color_delta * (1.6 if ctx.scan else 1.0)
    m = (de > thr) & valid
    ex = cv2.resize(cv2.dilate(exclude.astype(np.uint8), np.ones((5, 5), np.uint8)), (de.shape[1], de.shape[0]),
                    interpolation=cv2.INTER_NEAREST) > 0
    m &= ~ex
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    n, lab, st = _components(m, 1)
    min_a = cfg.mm2(cfg.min_color_area_mm2) * sc * sc
    px_mm2 = (cfg.dpi / 25.4) ** 2
    regs = []   # candidate regions at 1/4 scale: box, mask pixels, mean delta-E, mean A / B colour
    for i in range(1, n):
        x, y, ww, hh, area = st[i]
        if area < min_a:
            continue
        sel = (lab == i)
        regs.append({"box": (x, y, x + ww, y + hh), "n": int(area), "de": float(de[sel].mean()),
                     "ca": np.array(cv2.mean(a, sel.astype(np.uint8))[:3][::-1]),
                     "cb": np.array(cv2.mean(b, sel.astype(np.uint8))[:3][::-1])})

    # merge neighbouring regions that are the same recolouring (similar delta-E and similar A / B colours)
    gap = max(2, int(cfg.mm(3.0) * sc))
    parent = list(range(len(regs)))

    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def near(p, q):
        return (p[0] - gap <= q[2] and q[0] - gap <= p[2] and p[1] - gap <= q[3] and q[1] - gap <= p[3])

    for i_ in range(len(regs)):
        for j_ in range(i_ + 1, len(regs)):
            ri, rj = regs[i_], regs[j_]
            if near(ri["box"], rj["box"]) and abs(ri["de"] - rj["de"]) <= max(8.0, 0.25 * max(ri["de"], rj["de"])) \
                    and np.linalg.norm(ri["ca"] - rj["ca"]) <= 30 and np.linalg.norm(ri["cb"] - rj["cb"]) <= 30:
                parent[root(i_)] = root(j_)
    groups: dict = {}
    for k in range(len(regs)):
        groups.setdefault(root(k), []).append(regs[k])

    found = 0
    for g in groups.values():
        tot = sum(r["n"] for r in g)
        x0 = min(r["box"][0] for r in g); y0 = min(r["box"][1] for r in g)
        x1 = max(r["box"][2] for r in g); y1 = max(r["box"][3] for r in g)
        box = (int(x0 / sc), int(y0 / sc), int(x1 / sc), int(y1 / sc))
        mean_de = float(sum(r["de"] * r["n"] for r in g) / tot)
        ca = tuple(int(v) for v in sum(r["ca"] * r["n"] for r in g) / tot)
        cb = tuple(int(v) for v in sum(r["cb"] * r["n"] for r in g) / tot)
        area_mm2 = tot / (sc * sc) / px_mm2
        # confidence grows with both the colour distance and the changed area (a recoloured header band is
        # a certain finding, a few mm2 of tint is not); scans stay capped because of lighting / colour casts
        de_term = min(1.0, max(0.0, (mean_de - thr) / 40.0))
        area_term = min(1.0, max(0.0, np.log10(max(area_mm2, 1e-6) / cfg.min_color_area_mm2) / np.log10(30.0)))
        conf = 0.45 + 0.30 * de_term + 0.25 * area_term
        conf = float(np.clip(conf, 0.3, SCAN_COLOUR_CAP if ctx.scan else 0.95))
        findings.append(Finding(
            category="Modified", dtype="Colour", page_a=pa_no, page_b=pb_no,
            a_content=f"colour rgb{ca}", b_content=f"colour rgb{cb}",
            description=f"area colour / tint changed (mean delta-E {mean_de:.0f}, {area_mm2:.0f} mm2)",
            bbox_a=tuple(map(float, box)), bbox_b=alignmod.map_box(al.inverse(), box),
            confidence=round(conf, 3), engines=["visual-colour"],
            details={"delta_e": round(mean_de, 1), "a_rgb": ca, "b_rgb": cb, "area_mm2": round(area_mm2, 1),
                     "merged_regions": len(g), "region_a_frame": tuple(map(float, box))},
            region=tuple(map(float, box))))
        found += 1
    return {"colour_regions": found}
