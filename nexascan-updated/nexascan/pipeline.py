"""Orchestrates the full comparison: validate → analyse pages (cached, parallel) → match pages →
per pair: align, text / visual / object / overlap engines → classify → evidence → report."""
from __future__ import annotations

import gc
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import __version__
from . import align as alignmod
from . import evidence as evmod
from . import pagedata, upload
from .classify import classify_pair, sort_findings
from .config import Config
from .models import DocInfo, Finding, PageData
from .object_compare import compare_objects
from .overlap import analyze_overlaps
from .pagematch import PagePair, match_pages
from .report import build_report
from .text_compare import compare_text
from .visual_compare import build_context, compare_visual

Progress = Callable[[str, float, str], None]

try:  # `resource` does not exist on Windows
    import resource as _resource
except ImportError:  # pragma: no cover - platform dependent
    _resource = None


def _peak_rss_mb() -> float:
    """Peak resident set size of this process in MB (ru_maxrss is bytes on macOS, KB on Linux)."""
    if _resource is not None:
        try:
            v = float(_resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss)
            return v / (1024.0 * 1024.0) if sys.platform == "darwin" else v / 1024.0
        except Exception:  # noqa: BLE001
            pass
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return float(getattr(mi, "peak_wset", mi.rss)) / (1024.0 * 1024.0)   # peak on Windows, current elsewhere
    except Exception:  # noqa: BLE001
        return 0.0


def _noop(stage: str, frac: float, msg: str) -> None:
    pass


def compare_documents(path_a: str, name_a: str, path_b: str, name_b: str, out_dir: Path,
                      cfg: Optional[Config] = None, progress: Optional[Progress] = None,
                      info_a: Optional[DocInfo] = None, info_b: Optional[DocInfo] = None) -> dict:
    cfg = cfg or Config()
    progress = progress or _noop
    out_dir = Path(out_dir)
    ev_dir = out_dir / "evidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    timing: Dict[str, float] = {}

    progress("validate", 0.01, "Validating uploads")
    ia = info_a or upload.validate_file(path_a, name_a, cfg)
    ib = info_b or upload.validate_file(path_b, name_b, cfg)

    # ---------------------------------------------------------------- analyse pages
    t = time.time()
    jobs = [("A", ia, i) for i in range(ia.pages)] + [("B", ib, i) for i in range(ib.pages)]
    pages: Dict[Tuple[str, int], PageData] = {}
    done = 0

    def work(job):
        tag, info, idx = job
        return job, pagedata.build_page(info, idx, cfg)

    with ThreadPoolExecutor(max_workers=max(1, cfg.ocr_workers)) as ex:
        for job, page in ex.map(work, jobs):
            pages[(job[0], job[2])] = page
            done += 1
            progress("analyse", 0.03 + 0.42 * done / len(jobs), f"Analysed page {job[2] + 1} of document {job[0]}")
    pa_list = [pages[("A", i)] for i in range(ia.pages)]
    pb_list = [pages[("B", i)] for i in range(ib.pages)]
    timing["analyse_pages_s"] = round(time.time() - t, 2)

    # ---------------------------------------------------------------- page matching
    progress("match", 0.46, "Matching corresponding pages")
    pairs = match_pages(pa_list, pb_list, cfg.page_match_min)

    all_findings: List[Finding] = []
    suppressed: List[dict] = []
    page_diag: List[dict] = []
    t = time.time()
    n_pairs = max(1, len(pairs))
    for pi, pair in enumerate(pairs):
        base = 0.48 + 0.48 * pi / n_pairs
        span = 0.48 / n_pairs
        label = f"A{pair.a + 1 if pair.a is not None else '-'}↔B{pair.b + 1 if pair.b is not None else '-'}"
        progress("compare", base, f"Comparing pages {label}")
        try:
            fs, sup, diag = _compare_pair(ia, ib, pa_list, pb_list, pair, pi, cfg, ev_dir, progress, base, span)
        except Exception as e:  # noqa: BLE001 - one failing page must not lose the whole report
            fs, sup = [], []
            diag = {"pair": label, "error": f"{type(e).__name__}: {e}"}
            fs.append(Finding(category="Uncertain", dtype="Processing", page_a=(pair.a + 1) if pair.a is not None else None,
                              page_b=(pair.b + 1) if pair.b is not None else None, a_content="", b_content="",
                              description=f"pages {label} could not be fully compared ({type(e).__name__}: {e}); results for these pages are incomplete",
                              bbox_a=None, bbox_b=None, confidence=0.5, engines=["pipeline"], status="uncertain",
                              details={"original_category": "Uncertain", "error": str(e)}))
        all_findings += fs
        suppressed += sup
        page_diag.append(diag)
        gc.collect()
    timing["compare_pairs_s"] = round(time.time() - t, 2)

    # ---------------------------------------------------------------- finish
    progress("report", 0.97, "Generating report")
    all_findings = sort_findings(all_findings)
    for i, f in enumerate(all_findings, 1):
        f.id = i
    timing["total_s"] = round(time.time() - t_start, 2)
    peak_mb = _peak_rss_mb()
    report = build_report(ia, ib, pa_list, pb_list, pairs, all_findings, suppressed, page_diag, timing,
                          round(peak_mb, 1), cfg, out_dir)
    progress("done", 1.0, "Done")
    return report


# -------------------------------------------------------------------------------------------------


def _filter_words(words, lines, keep):
    """Return (words, lines) without the words for which keep(word) is False; lines are rebuilt."""
    from .models import Line, bbox_union
    new_words, new_lines = [], []
    for ln in lines:
        ids = [i for i in ln.words if keep(words[i])]
        if not ids:
            continue
        ws = []
        for i in ids:
            w = words[i]
            w.line = len(new_lines)
            ws.append(len(new_words))
            new_words.append(w)
        full = len(ids) == len(ln.words)
        new_lines.append(Line(idx=len(new_lines), words=ws, bbox=bbox_union([new_words[j].bbox for j in ws]),
                              text=" ".join(new_words[j].text for j in ws), conf=ln.conf,
                              alt_texts=list(ln.alt_texts) if full else []))
    return new_words, new_lines


def _exclude_under_objects(words_a, lines_a, words_b, lines_b, objs_a, objs_b, al, cfg):
    """OCR cannot read text under stamps / handwriting / blocks. Such words are excluded from the text
    diff (the overlap engine reports them) instead of being compared as if they were real text."""
    import copy as _copy
    pad = cfg.mm(1.0)
    zones = []
    for o in objs_a:
        if o["type"] in ("stamp", "handwriting", "block") or (o["type"] == "mark" and o.get("score", 0) >= 0.5):
            zones.append(tuple(o["bbox"]))
    for o in objs_b:
        if o["type"] in ("stamp", "handwriting", "block") or (o["type"] == "mark" and o.get("score", 0) >= 0.5):
            zones.append(alignmod.map_box(al.M, o["bbox"]))
    if not zones:
        return words_a, lines_a, words_b, lines_b, 0

    def inside(bb):
        area = max(1.0, (bb[2] - bb[0]) * (bb[3] - bb[1]))
        for z in zones:
            ix = min(bb[2], z[2] + pad) - max(bb[0], z[0] - pad)
            iy = min(bb[3], z[3] + pad) - max(bb[1], z[1] - pad)
            if ix > 0 and iy > 0 and ix * iy / area >= 0.2:
                return True
        return False

    wa, la = _filter_words(_copy.deepcopy(words_a), _copy.deepcopy(lines_a), lambda w: not inside(w.bbox))
    wb, lb = _filter_words(_copy.deepcopy(words_b), _copy.deepcopy(lines_b),
                           lambda w: not inside(alignmod.map_box(al.M, w.bbox)))
    return wa, la, wb, lb, (len(words_a) - len(wa)) + (len(words_b) - len(wb))


def _page_finding(category: str, dtype: str, pa: Optional[int], pb: Optional[int], desc: str, a: str, b: str,
                  conf: float, engines: List[str]) -> Finding:
    return Finding(category=category, dtype=dtype, page_a=pa, page_b=pb, a_content=a, b_content=b,
                   description=desc, bbox_a=None, bbox_b=None, confidence=conf, engines=engines,
                   status="confirmed" if conf >= 0.7 else "uncertain")


def _compare_pair(ia: DocInfo, ib: DocInfo, pa_list: List[PageData], pb_list: List[PageData], pair: PagePair,
                  pi: int, cfg: Config, ev_dir: Path, progress: Progress, base: float, span: float):
    suppressed: List[dict] = []
    ano = pair.a + 1 if pair.a is not None else None
    bno = pair.b + 1 if pair.b is not None else None
    diag = {"pair": f"A{ano or '-'}<->B{bno or '-'}", "kind": pair.kind, "similarity": round(pair.sim, 3)}

    # ---- whole-page additions / removals ---------------------------------------------------------
    if pair.kind in ("removed", "added"):
        info, idx, pg = (ia, pair.a, pa_list[pair.a]) if pair.kind == "removed" else (ib, pair.b, pb_list[pair.b])
        f = _page_finding("Removed" if pair.kind == "removed" else "Added", "Page", ano, bno,
                          f"Page {ano} of document A has no counterpart in B: the whole page was removed" if pair.kind == "removed"
                          else f"Page {bno} of document B has no counterpart in A: a new page was added",
                          f"page {ano} present" if pair.kind == "removed" else "no corresponding page",
                          "no corresponding page" if pair.kind == "removed" else f"page {bno} present", 0.97, ["page-match"])
        bgr = pagedata.get_prepared(info, pg, cfg).bgr
        name = evmod.page_thumb(bgr, ev_dir, f"p{pi}_0_{'a' if pair.kind == 'removed' else 'b'}.png")
        f.evidence = {("a" if pair.kind == "removed" else "b"): name}
        return [f], suppressed, diag

    pa, pb = pa_list[pair.a], pb_list[pair.b]
    findings: List[Finding] = []
    if pair.kind == "moved":
        findings.append(_page_finding("Moved", "Page", ano, bno, f"Page {ano} of A appears as page {bno} in B (page order changed)",
                                      f"page {ano}", f"page {bno}", 0.9, ["page-match"]))
    if pair.sim < 0.15 and pair.kind == "replaced":
        f = _page_finding("Modified", "Page", ano, bno,
                          f"Page {ano} of A and page {bno} of B share almost no content: the page was replaced", f"page {ano}",
                          f"page {bno}", 0.9, ["page-match"])
        f.evidence = {"a": evmod.page_thumb(pagedata.get_prepared(ia, pa, cfg).bgr, ev_dir, f"p{pi}_0_a.png"),
                      "b": evmod.page_thumb(pagedata.get_prepared(ib, pb, cfg).bgr, ev_dir, f"p{pi}_0_b.png")}
        diag["skipped"] = "replaced page, content entirely different"
        return findings + [f], suppressed, diag

    both_native = pa.source == "native" and pb.source == "native"
    ocr_mode = not both_native
    prep_a = pagedata.get_prepared(ia, pa, cfg)
    prep_b = pagedata.get_prepared(ib, pb, cfg)
    if ocr_mode:
        if pa.source == "native":
            pagedata.ensure_ocr(ia, pa, cfg, prep_a)
        if pb.source == "native":
            pagedata.ensure_ocr(ib, pb, cfg, prep_b)
    words_a, lines_a = (pa.words, pa.lines) if (pa.source == "scan" or not ocr_mode) else (pa.ocr_words, pa.ocr_lines)
    words_b, lines_b = (pb.words, pb.lines) if (pb.source == "scan" or not ocr_mode) else (pb.ocr_words, pb.ocr_lines)
    full_words_a, full_words_b = words_a, words_b
    objs_a, objs_b = pa.objects, pb.objects

    # ---- page geometry notes ---------------------------------------------------------------------------
    if pa.page_mm[0] and pb.page_mm[0] and (abs(pa.page_mm[0] - pb.page_mm[0]) > 1.5 or abs(pa.page_mm[1] - pb.page_mm[1]) > 1.5) \
            and pa.source == "native" and pb.source == "native":
        findings.append(_page_finding("Modified", "Layout", ano, bno,
                                      f"page size differs: A is {pa.page_mm[0]:.0f} x {pa.page_mm[1]:.0f} mm, B is {pb.page_mm[0]:.0f} x {pb.page_mm[1]:.0f} mm",
                                      f"{pa.page_mm[0]:.0f} x {pa.page_mm[1]:.0f} mm", f"{pb.page_mm[0]:.0f} x {pb.page_mm[1]:.0f} mm",
                                      0.99, ["page-geometry"]))
    notes = []
    for tag, pg in (("A", pa), ("B", pb)):
        notes += [f"{tag}: {n}" for n in pg.notes]
    diag["page_notes"] = notes
    diag["mode"] = "ocr" if ocr_mode else "native-text"
    diag["noise_sigma"] = [round(pa.noise_sigma, 2), round(pb.noise_sigma, 2)]

    # ---- alignment -----------------------------------------------------------------------------------------
    same_digital = both_native and prep_a.gray.shape == prep_b.gray.shape
    al = alignmod.register(prep_a.gray, prep_b.gray, same_size_digital=same_digital)
    diag["alignment"] = al.describe()
    if not al.ok:
        findings.append(_page_finding("Uncertain", "Alignment", ano, bno,
                                      f"pages could not be aligned reliably (correlation {al.ncc:.2f}); visual comparison skipped, only text compared",
                                      "", "", 0.5, ["align"]))
    # a failed alignment must never be used to map B boxes into the A frame: compare text without a transform
    al_for_text = al if al.ok else None
    progress("compare", base + 0.2 * span, f"Aligned pages {ano}/{bno} ({al.method})")

    # ---- engines -----------------------------------------------------------------------------------------------
    ctx = build_context(prep_a, prep_b, al, cfg, scan=ocr_mode)
    ctx.stats["noise"] = max(pa.noise_sigma, pb.noise_sigma)
    raw: List[Finding] = []
    n_excl = 0
    if ocr_mode and al.ok:
        words_a, lines_a, words_b, lines_b, n_excl = _exclude_under_objects(words_a, lines_a, words_b, lines_b,
                                                                           objs_a, objs_b, al, cfg)
    tf, tinfo = compare_text(pa, pb, words_a, lines_a, words_b, lines_b, al_for_text, cfg, ano, bno, ocr_mode,
                            ctx if al.ok else None)
    tinfo["words_excluded_under_objects"] = n_excl
    raw += tf
    diag["text"] = tinfo
    if al.ok:
        vf, vstats = compare_visual(ctx, pa, pb, al, cfg, ano, bno)
        raw += vf
        diag["visual"] = vstats
        raw += compare_objects(ctx, objs_a, objs_b, al, cfg, ano, bno)
        raw += analyze_overlaps(ctx, objs_a, objs_b, full_words_a, full_words_b, al, cfg, not ocr_mode, ano, bno)
    progress("compare", base + 0.6 * span, f"Classifying pages {ano}/{bno}")
    final, sup = classify_pair(raw, ctx if al.ok else None, cfg)
    suppressed += sup
    n_dis = sum(1 for s_ in sup if s_.get("kind") == "ocr_disagreement")
    if n_dis:    # identical text removed + added without any pixel change is OCR disagreement, not a difference
        tinfo["suppressed_ocr_disagreement"] += n_dis
    diag["engine_counts"] = {"raw": len(raw), "final": len(final), "suppressed": len(sup)}

    # ---- evidence -------------------------------------------------------------------------------------------------
    for k, f in enumerate(final):
        try:
            f.evidence = evmod.make_evidence(f, f"p{pi}_{k + 1}", ev_dir, prep_a.bgr, prep_b.bgr, ctx.bgr_b, al, cfg)
        except Exception as e:  # noqa: BLE001
            f.evidence = {}
            f.details["evidence_error"] = str(e)
    findings = [f for f in findings if f.status] + final
    del ctx, prep_a, prep_b
    return findings, suppressed, diag
