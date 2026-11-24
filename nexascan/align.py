"""Registration of page B onto the frame of page A.

Strategy (most to least specific, best candidate chosen by ink-layer correlation):
  0. identity fast path (same size, near-identical layout, e.g. digital vs digital)
  1. SIFT features + RANSAC similarity transform (rotation, uniform scale, translation)
  2. ORB features (fallback)
  3. content bounding-box alignment (fallback for feature-poor pages)
then ECC (enhanced correlation coefficient) refinement with an affine model and a final
phase-correlation sub-pixel shift.  Different page sizes, rotated and skewed scans are all
handled by the same similarity/affine model."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Alignment:
    M: np.ndarray                  # 2x3, maps B pixel coords -> A pixel coords (full resolution)
    method: str
    inliers: int
    ncc: float                     # ink-layer correlation after alignment (0..1)
    ok: bool                       # alignment trustworthy enough for pixel comparison
    scale: float = 1.0
    rotation_deg: float = 0.0
    shift_px: Tuple[float, float] = (0.0, 0.0)
    residual_px: float = 0.0       # estimated remaining misregistration (full-res px)

    def inverse(self) -> np.ndarray:
        return cv2.invertAffineTransform(self.M)

    def map_box(self, b):
        return map_box(self.M, b)

    def describe(self) -> dict:
        return {"method": self.method, "inliers": self.inliers, "ncc": round(self.ncc, 3), "ok": self.ok,
                "scale": round(self.scale, 4), "rotation_deg": round(self.rotation_deg, 3),
                "shift_px": [round(self.shift_px[0], 1), round(self.shift_px[1], 1)],
                "residual_px": round(self.residual_px, 2)}


def map_box(M: np.ndarray, b) -> Tuple[float, float, float, float]:
    pts = np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]], np.float64)
    out = pts @ M[:, :2].T + M[:, 2]
    return (float(out[:, 0].min()), float(out[:, 1].min()), float(out[:, 0].max()), float(out[:, 1].max()))


def _to3(M: np.ndarray) -> np.ndarray:
    return np.vstack([M, [0, 0, 1]])


def _small(gray: np.ndarray, long_side: int) -> Tuple[np.ndarray, float]:
    h, w = gray.shape[:2]
    s = long_side / float(max(h, w))
    if s < 1.0:
        return cv2.resize(gray, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA), s
    return gray, 1.0


def _ink_density(g: np.ndarray) -> np.ndarray:
    d = 255.0 - g.astype(np.float32)
    return cv2.GaussianBlur(d, (0, 0), 2.0)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / den) if den > 1e-9 else 0.0


def _score(a_small: np.ndarray, b_small: np.ndarray, M_small: np.ndarray) -> float:
    h, w = a_small.shape
    wb = cv2.warpAffine(b_small, M_small, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                        borderValue=255)
    return _ncc(_ink_density(a_small), _ink_density(wb))


def _feature_candidate(a: np.ndarray, b: np.ndarray, kind: str) -> Tuple[Optional[np.ndarray], int]:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ga, gb = clahe.apply(a), clahe.apply(b)
    if kind == "sift":
        det = cv2.SIFT_create(nfeatures=5000)
        norm = cv2.NORM_L2
    else:
        det = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, edgeThreshold=15, patchSize=31)
        norm = cv2.NORM_HAMMING
    ka, da = det.detectAndCompute(ga, None)
    kb, db = det.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return None, 0
    bf = cv2.BFMatcher(norm)
    try:
        pairs = bf.knnMatch(db, da, k=2)
    except cv2.error:
        return None, 0
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.8 * n.distance]
    if len(good) < 8:
        return None, len(good)
    pb = np.float32([kb[m.queryIdx].pt for m in good])
    pa = np.float32([ka[m.trainIdx].pt for m in good])
    M, inl = cv2.estimateAffinePartial2D(pb, pa, method=cv2.RANSAC, ransacReprojThreshold=4.0,
                                         maxIters=5000, confidence=0.999, refineIters=20)
    if M is None or inl is None:
        return None, 0
    return M, int(inl.sum())


def _bbox_candidate(a: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    def bbox(g):
        m = g < 200
        if m.sum() < 50:
            return None
        ys, xs = np.where(m)
        return (np.percentile(xs, 0.5), np.percentile(ys, 0.5), np.percentile(xs, 99.5), np.percentile(ys, 99.5))
    ba, bb = bbox(a), bbox(b)
    if ba is None or bb is None:
        return None
    sx = (ba[2] - ba[0]) / max(1.0, bb[2] - bb[0])
    sy = (ba[3] - ba[1]) / max(1.0, bb[3] - bb[1])
    s = (sx + sy) / 2.0
    if not (0.5 < s < 2.0):
        s = 1.0
    cxa, cya = (ba[0] + ba[2]) / 2, (ba[1] + ba[3]) / 2
    cxb, cyb = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
    return np.array([[s, 0, cxa - s * cxb], [0, s, cya - s * cyb]], np.float64)


def _ecc_refine(a: np.ndarray, b: np.ndarray, M_init: np.ndarray) -> Optional[np.ndarray]:
    """Refine M (B->A, at the resolution of a/b) with ECC. Returns refined M or None."""
    try:
        A = cv2.GaussianBlur(a.astype(np.float32), (0, 0), 1.5) / 255.0
        B = cv2.GaussianBlur(b.astype(np.float32), (0, 0), 1.5) / 255.0
        W = cv2.invertAffineTransform(M_init).astype(np.float32)  # ECC wants template(A) -> input(B)
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
        _, W = cv2.findTransformECC(A, B, W, cv2.MOTION_AFFINE, crit, None, 5)
        return cv2.invertAffineTransform(W)
    except cv2.error:
        return None


def _decompose(M: np.ndarray) -> Tuple[float, float, Tuple[float, float]]:
    sc = math.sqrt(abs(np.linalg.det(M[:, :2])))
    rot = math.degrees(math.atan2(M[1, 0], M[0, 0]))
    return sc, rot, (float(M[0, 2]), float(M[1, 2]))


def register(gray_a: np.ndarray, gray_b: np.ndarray, same_size_digital: bool = False,
             ecc_long: int = 900) -> Alignment:
    """Estimate the transform mapping B pixel coordinates to A pixel coordinates."""
    # ---- candidate generation on a ~1400 px working copy --------------------------------
    a_s, sa = _small(gray_a, 1400)
    b_s, sb = _small(gray_b, 1400)
    Sa, Sb = np.diag([sa, sa, 1.0]), np.diag([sb, sb, 1.0])

    def full_from_small(Ms: np.ndarray) -> np.ndarray:
        return (np.linalg.inv(Sa) @ _to3(Ms) @ Sb)[:2]

    cands: List[Tuple[str, np.ndarray, int]] = []
    if same_size_digital and gray_a.shape == gray_b.shape:
        ident = np.array([[1, 0, 0], [0, 1, 0]], np.float64)
        s_id = _score(a_s, b_s, np.array([[1, 0, 0], [0, 1, 0]], np.float64) if sa == sb else
                      np.array([[sa / sb, 0, 0], [0, sa / sb, 0]], np.float64))
        if s_id >= 0.75:
            ncc_full = s_id
            return Alignment(M=ident, method="identity", inliers=0, ncc=ncc_full, ok=True,
                             scale=1.0, rotation_deg=0.0, shift_px=(0.0, 0.0), residual_px=0.0)
    for kind in ("sift", "orb"):
        Ms, inl = _feature_candidate(a_s, b_s, kind)
        if Ms is not None and inl >= 10:
            cands.append((kind, Ms, inl))
            if kind == "sift" and inl >= 40:
                break
    bb = _bbox_candidate(a_s, b_s)
    if bb is not None:
        cands.append(("bbox", bb, 0))
    cands.append(("identity", np.array([[sa / sb, 0, 0], [0, sa / sb, 0]], np.float64) if sa != sb
                  else np.array([[1, 0, 0], [0, 1, 0]], np.float64), 0))

    best = None
    for name, Ms, inl in cands:
        sc = _score(a_s, b_s, Ms)
        if best is None or sc > best[0] + 1e-6:
            best = (sc, name, Ms, inl)
    sc0, name, Ms, inl = best

    # ---- ECC refinement at higher resolution -----------------------------------------------
    a_e, ea = _small(gray_a, ecc_long)
    b_e, eb = _small(gray_b, ecc_long)
    Ea, Eb = np.diag([ea, ea, 1.0]), np.diag([eb, eb, 1.0])
    M_full0 = full_from_small(Ms)
    M_e0 = (Ea @ _to3(M_full0) @ np.linalg.inv(Eb))[:2]
    score_e0 = _score(a_e, b_e, M_e0)
    M_final_full = M_full0
    final_score = score_e0
    method = name
    # ECC is slow and only helps feature-poor pages: skip it when SIFT has plenty of inliers
    if 0.15 < sc0 < 0.995 and not (name == "sift" and inl >= 150):
        M_e1 = _ecc_refine(a_e, b_e, M_e0)
        if M_e1 is not None:
            s0, r0, _ = _decompose(M_e0)
            s1, r1, _ = _decompose(M_e1)
            if abs(s1 / s0 - 1) < 0.03 and abs(r1 - r0) < 1.5:
                score_e1 = _score(a_e, b_e, M_e1)
                # affine ECC can drift by a fraction of a degree without changing NCC visibly,
                # so it must buy a clear improvement before it replaces the feature estimate
                if score_e1 > score_e0 + 0.003:
                    M_final_full = (np.linalg.inv(Ea) @ _to3(M_e1) @ Eb)[:2]
                    final_score = score_e1
                    method = name + "+ecc"

    # ---- sub-pixel residual via phase correlation (accepted only if it measurably helps) -----
    h, w = gray_a.shape
    resid = 0.0
    try:
        sP = min(1.0, 1200.0 / max(h, w))
        Ms_p = (np.diag([sP, sP, 1.0]) @ _to3(M_final_full) @ np.diag([1 / sP, 1 / sP, 1.0]))[:2]
        ga = cv2.resize(gray_a, None, fx=sP, fy=sP, interpolation=cv2.INTER_AREA) if sP < 1 else gray_a
        gb = cv2.resize(gray_b, None, fx=sP, fy=sP, interpolation=cv2.INTER_AREA) if sP < 1 else gray_b
        wb = cv2.warpAffine(gb, Ms_p, (ga.shape[1], ga.shape[0]), flags=cv2.INTER_LINEAR, borderValue=255)
        win = cv2.createHanningWindow((ga.shape[1], ga.shape[0]), cv2.CV_32F)
        (dx, dy), resp = cv2.phaseCorrelate(_ink_density(ga), _ink_density(wb), win)
        # Diagnostic only: applying this correction was measured to make exact test transforms
        # *worse* (it works on a down-scaled copy), so it is reported but never applied.
        resid = math.hypot(dx, dy) / sP if resp > 0.05 else 0.0
    except cv2.error:
        pass

    # same-size digital pages whose estimate is within a pixel of identity: use exact identity so
    # that estimation noise can never create sub-pixel "differences"
    if same_size_digital and gray_a.shape == gray_b.shape:
        dev = np.abs(M_final_full - np.array([[1, 0, 0], [0, 1, 0]], np.float64))
        if dev[:, :2].max() < 2e-3 and dev[:, 2].max() < 1.5:
            M_final_full = np.array([[1, 0, 0], [0, 1, 0]], np.float64)
            method += "->identity"
    sc, rot, shift = _decompose(M_final_full)
    ok = final_score >= 0.30
    return Alignment(M=M_final_full, method=method, inliers=inl, ncc=float(final_score), ok=ok,
                     scale=sc, rotation_deg=rot, shift_px=shift, residual_px=float(resid))


def warp_to_a(img: np.ndarray, M: np.ndarray, size_a_hw: Tuple[int, int], border=255,
              interp=cv2.INTER_LINEAR) -> np.ndarray:
    h, w = size_a_hw
    bv = (border,) * (img.shape[2] if img.ndim == 3 else 1)
    return cv2.warpAffine(img, M, (w, h), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=bv)
