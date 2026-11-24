"""Judging a higher-resolution retry: stable findings, honest selection, nothing silently lost.

A finding is STABLE when it appears in both runs with the same type and category, the same page(s), the same place
(overlap of the boxes in millimetres, so the DPI does not matter) and a compatible description. Confidence alone never
identifies a finding.

Selection (evaluate_retry):
  1. fewer uncertain findings in the retry                      -> keep the retry
  2. same number: higher average confidence over the stable pairs -> keep the retry
  3. otherwise                                                    -> keep the first result
Never silently dropped:
  * a CONFIRMED finding that only the retry has -> list "seen_only_at_higher_resolution" (review item, not a verified change)
  * a CONFIRMED finding that only the first run had, when the retry is kept -> moved to "uncertain" with
    details.seen_only_at_original_resolution
Confidence is never raised because the DPI is higher.
"""
from __future__ import annotations

import difflib
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

IOU_MIN = 0.5            # box overlap needed for "same place"
CENTER_MM = 2.0          # ...or centres closer than this (tiny findings such as a single dot)
TEXT_SIM_MIN = 0.6       # compatible description of what changed
FIRST_ONLY_CAP = 0.69    # confidence ceiling of a confirmed first-run finding that the retry did not reproduce


def _conf(f: dict) -> float:
    try:
        v = float(f.get("confidence"))
        return v if v == v else 0.0
    except (TypeError, ValueError):
        return 0.0


def _box(f: dict) -> Optional[Tuple[float, float, float, float]]:
    loc = f.get("location") or {}
    for side in ("a", "b"):
        b = (loc.get(side) or {}).get("bbox_mm")
        if b and len(b) == 4:
            return tuple(float(v) for v in b)          # type: ignore[return-value]
    return None


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _same_place(f: dict, g: dict) -> float:
    """Overlap score in [0,1]; 0 = different place."""
    a, b = _box(f), _box(g)
    if a is None or b is None:
        return 0.0
    iou = _iou(a, b)
    if iou >= IOU_MIN:
        return iou
    ca, cb = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2), ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    if ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5 <= CENTER_MM:
        return max(iou, 0.5)
    return 0.0


def _same_identity(f: dict, g: dict) -> bool:
    if f.get("dtype") != g.get("dtype") or f.get("category") != g.get("category"):
        return False
    if f.get("page_a") != g.get("page_a") or f.get("page_b") != g.get("page_b"):
        return False
    ta = f"{f.get('document_a') or ''} {f.get('document_b') or ''}".strip().lower()
    tb = f"{g.get('document_a') or ''} {g.get('document_b') or ''}".strip().lower()
    if ta and tb:
        return difflib.SequenceMatcher(None, ta, tb).ratio() >= TEXT_SIM_MIN
    return True


def stable_match(f: dict, g: dict) -> float:
    """0 when the two findings are not the same finding, else the overlap score."""
    return _same_place(f, g) if _same_identity(f, g) else 0.0


def match_runs(first: Sequence[dict], retry: Sequence[dict]) -> List[Tuple[int, int]]:
    """One-to-one pairing of stable findings, best overlap first."""
    cands = []
    for i, f in enumerate(first):
        for j, g in enumerate(retry):
            s = stable_match(f, g)
            if s > 0:
                cands.append((s, i, j))
    cands.sort(reverse=True)
    used_i, used_j, out = set(), set(), []
    for _, i, j in cands:
        if i not in used_i and j not in used_j:
            used_i.add(i)
            used_j.add(j)
            out.append((i, j))
    return out


def dedupe(items: Sequence[dict]) -> List[dict]:
    """Drop duplicates of the same finding inside one list (keeps the higher confidence)."""
    kept: List[dict] = []
    for f in sorted(items, key=_conf, reverse=True):
        if not any(stable_match(f, k) > 0 for k in kept):
            kept.append(f)
    return kept


def _all(rep: dict) -> List[dict]:
    return list(rep.get("differences") or []) + list(rep.get("uncertain") or [])


def evaluate_retry(first: dict, retry: dict) -> dict:
    """Compare the two reports. Pure function: returns the decision and the findings that need special handling."""
    f_all, r_all = _all(first), _all(retry)
    pairs = match_runs(f_all, r_all)
    fu, ru = len(first.get("uncertain") or []), len(retry.get("uncertain") or [])
    avg_f = sum(_conf(f_all[i]) for i, _ in pairs) / len(pairs) if pairs else 0.0
    avg_r = sum(_conf(r_all[j]) for _, j in pairs) / len(pairs) if pairs else 0.0
    if ru < fu:
        keep, why = "retry", f"fewer uncertain findings ({fu} -> {ru})"
    elif ru == fu and pairs and avg_r > avg_f:
        keep, why = "retry", f"same number of uncertain findings, higher average confidence of the stable findings ({avg_f:.2f} -> {avg_r:.2f})"
    else:
        keep, why = "first", ("no improvement: " + (f"{ru} uncertain findings vs {fu}" if ru > fu else
                                                    "same uncertain count and no higher confidence in the stable findings"))
    matched_f = {i for i, _ in pairs}
    matched_r = {j for _, j in pairs}
    hr_only = dedupe([g for j, g in enumerate(r_all) if j not in matched_r and g.get("status") == "confirmed"])
    first_only = dedupe([f for i, f in enumerate(f_all) if i not in matched_f and f.get("status") == "confirmed"])
    return {"keep": keep, "reason": why, "first_uncertain": fu, "retry_uncertain": ru, "stable_pairs": len(pairs),
            "avg_conf_first": round(avg_f, 3), "avg_conf_retry": round(avg_r, 3), "hr_only": hr_only, "first_only": first_only}


def _copy_evidence(item: dict, src_dir: Path, dst_dir: Path, prefix: str) -> dict:
    """Copy the evidence images of ``item`` from src_dir/evidence to dst_dir/evidence under a unique prefix."""
    item = dict(item)
    ev = dict(item.get("evidence") or {})
    out: Dict[str, str] = {}
    for k, rel in ev.items():
        name = Path(rel).name
        src = Path(src_dir) / "evidence" / name
        if src.is_file():
            (Path(dst_dir) / "evidence").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, Path(dst_dir) / "evidence" / f"{prefix}{name}")
            out[k] = f"evidence/{prefix}{name}"
    item["evidence"] = out
    return item


def finalize(first: dict, retry: dict, decision: dict, retry_dpi: int, first_dpi: int,
             first_dir: Path, retry_dir: Path, keep_dir: Path) -> dict:
    """Return the selected report with the special findings attached. Evidence files are copied into the
    selected result's evidence folder (``keep_dir`` = where the final files live; the caller moves the retry there)."""
    keep = decision["keep"]
    hr: List[dict] = []
    for f in decision["hr_only"]:
        item = _copy_evidence(f, retry_dir, keep_dir, "hr_")
        d = dict(item.get("details") or {})
        d.update({"seen_only_at_higher_resolution": True, "higher_dpi": retry_dpi, "first_dpi": first_dpi,
                  "note": f"Detected only in the {retry_dpi} DPI comparison, not in the {first_dpi} DPI run. "
                          f"It needs manual review; it is not counted as an independently verified change."})
        item["details"] = d
        item["status"] = "review"
        hr.append(item)
    if keep == "retry":
        sel = retry
        sel["differences"] = [f for f in sel["differences"] if not any(f is h for h in decision["hr_only"])]
        for f in decision["first_only"]:                          # confirmed at the original DPI only: keep for review
            item = _copy_evidence(f, first_dir, keep_dir, "or_")
            d = dict(item.get("details") or {})
            d.update({"seen_only_at_original_resolution": True, "original_dpi": first_dpi,
                      "note": f"Confirmed at {first_dpi} DPI but not reproduced at {retry_dpi} DPI; kept as uncertain."})
            item["details"] = d
            item["status"] = "uncertain"
            item["confidence"] = min(_conf(item), FIRST_ONLY_CAP)
            sel["uncertain"].append(item)
    else:
        sel = first
    sel["seen_only_at_higher_resolution"] = hr
    return sel
