"""Identify corresponding pages of two documents (handles inserted, removed, moved and
replaced pages) with a similarity matrix + dynamic programming for the in-order alignment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import cv2
import numpy as np

from .models import PageData
from .textutil import tokenize

THUMB_W, THUMB_H = 24, 34


def make_thumb(ink: np.ndarray) -> List[float]:
    t = cv2.resize(ink.astype(np.float32), (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
    t = cv2.GaussianBlur(t, (0, 0), 0.8)
    return [float(v) for v in t.ravel()]


def _shingles(page: PageData, use_ocr: bool = False) -> set:
    words = page.ocr_words if (use_ocr and page.ocr_words is not None) else page.words
    toks: List[str] = []
    for w in words:
        toks += [t.lower() for t in tokenize(w.text) if any(c.isalnum() for c in t)]
    if len(toks) < 3:
        return set(toks)
    return {hash(tuple(toks[i:i + 3])) for i in range(len(toks) - 2)}


def _vis_sim(a: PageData, b: PageData) -> float:
    if not a.thumb or not b.thumb:
        return 0.0
    x = np.asarray(a.thumb, np.float32)
    y = np.asarray(b.thumb, np.float32)
    x -= x.mean()
    y -= y.mean()
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    return max(0.0, float((x * y).sum() / den)) if den > 1e-9 else 0.0


def page_similarity(a: PageData, b: PageData, sa: Optional[set] = None, sb: Optional[set] = None) -> float:
    if a.ink_frac < 0.0005 and b.ink_frac < 0.0005:
        return 1.0
    vis = _vis_sim(a, b)
    sa = _shingles(a) if sa is None else sa
    sb = _shingles(b) if sb is None else sb
    if len(sa) >= 8 and len(sb) >= 8:
        jac = len(sa & sb) / float(len(sa | sb))
        return 0.7 * jac + 0.3 * vis
    return vis


@dataclass
class PagePair:
    a: Optional[int]
    b: Optional[int]
    sim: float
    kind: str   # matched | moved | replaced | removed | added


def match_pages(pa: Sequence[PageData], pb: Sequence[PageData], min_sim: float = 0.30,
                moved_sim: float = 0.60) -> List[PagePair]:
    n, m = len(pa), len(pb)
    sh_a = [_shingles(p) for p in pa]
    sh_b = [_shingles(p) for p in pb]
    S = np.zeros((n, m), np.float64)
    for i in range(n):
        for j in range(m):
            S[i, j] = page_similarity(pa[i], pb[j], sh_a[i], sh_b[j])

    # in-order alignment maximising total similarity (matches below min_sim are not allowed)
    dp = np.zeros((n + 1, m + 1))
    back = np.zeros((n + 1, m + 1), np.int8)  # 0 diag, 1 up (skip A), 2 left (skip B)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best, bk = dp[i - 1, j], 1
            if dp[i, j - 1] > best:
                best, bk = dp[i, j - 1], 2
            s = S[i - 1, j - 1]
            if s >= min_sim and dp[i - 1, j - 1] + s > best:
                best, bk = dp[i - 1, j - 1] + s, 0
            dp[i, j], back[i, j] = best, bk
    pairs = {}
    i, j = n, m
    while i > 0 and j > 0:
        if back[i, j] == 0:
            pairs[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif back[i, j] == 1:
            i -= 1
        else:
            j -= 1

    out: List[PagePair] = [PagePair(a, b, float(S[a, b]), "matched") for a, b in sorted(pairs.items())]
    used_a, used_b = set(pairs.keys()), set(pairs.values())

    # pages that moved (out of order) but are clearly the same
    cand = sorted(((S[i, j], i, j) for i in range(n) if i not in used_a for j in range(m) if j not in used_b),
                  reverse=True)
    for s, i, j in cand:
        if s < moved_sim:
            break
        if i in used_a or j in used_b:
            continue
        out.append(PagePair(i, j, float(s), "moved"))
        used_a.add(i)
        used_b.add(j)

    # leftover pages that sit in the same gap between matched neighbours are "replaced" pairs
    anchors = sorted((p.a, p.b) for p in out if p.kind == "matched")

    def gap_key(idx: int, side: int) -> int:
        return sum(1 for pr in anchors if pr[side] < idx)

    rest_a = [i for i in range(n) if i not in used_a]
    rest_b = [j for j in range(m) if j not in used_b]
    by_gap_a, by_gap_b = {}, {}
    for i in rest_a:
        by_gap_a.setdefault(gap_key(i, 0), []).append(i)
    for j in rest_b:
        by_gap_b.setdefault(gap_key(j, 1), []).append(j)
    for g, ia in by_gap_a.items():
        jb = by_gap_b.get(g, [])
        k = min(len(ia), len(jb))
        for t in range(k):
            out.append(PagePair(ia[t], jb[t], float(S[ia[t], jb[t]]), "replaced"))
            used_a.add(ia[t])
            used_b.add(jb[t])
    for i in range(n):
        if i not in used_a:
            out.append(PagePair(i, None, 0.0, "removed"))
    for j in range(m):
        if j not in used_b:
            out.append(PagePair(None, j, 0.0, "added"))
    out.sort(key=lambda p: (p.a if p.a is not None else 10 ** 6, p.b if p.b is not None else 10 ** 6))
    return out
