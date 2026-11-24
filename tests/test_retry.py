"""Retry evaluation: stable findings, ties, higher-resolution-only findings, failures. Deterministic, no real documents."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from nexascan import retry as rt  # noqa: E402
from nexascan import worker  # noqa: E402
from nexascan.report import refresh_summary, write_reports  # noqa: E402


def item(n, text="Amount", box=(20, 30, 60, 38), conf=0.9, status="confirmed", dtype="Text", cat="Modified", page=1, ev=None):
    loc = {"bbox_mm": list(box), "bbox_px": [int(v * 11.8) for v in box]}
    return {"number": n, "type": f"{dtype} {cat.lower()}", "dtype": dtype, "category": cat, "page_a": page, "page_b": page,
            "document_a": text, "document_b": text + " 2", "difference": f"{text} changed", "confidence": conf, "status": status,
            "engines": ["text"], "evidence": ev or {}, "details": {}, "location": {"dpi": 300, "a": loc, "b": loc}}


def report(confirmed=(), uncertain=()):
    c, u = [dict(i) for i in confirmed], [dict(i) for i in uncertain]
    return {"title": "t", "tagline": "g", "version": "x", "generated_utc": "now", "disclaimer": "d",
            "document_a": {"name": "a", "pages": 1, "kind": "pdf", "sha256": ""}, "document_b": {"name": "b", "pages": 1, "kind": "pdf", "sha256": ""},
            "summary": {"total_differences": len(c), "uncertain_findings": len(u), "by_category": {}, "by_type": {}, "page_pairs": []},
            "differences": c, "uncertain": u, "seen_only_at_higher_resolution": [], "verified_unchanged": [],
            "diagnostics": {"timing": {}, "dpi_selection": {"mode": "auto", "used": 300, "message": "Clean scan → analysed at 300 DPI."}}}


# ---------------------------------------------------------------- stable findings
def test_same_finding_is_stable_even_when_alignment_shifts_a_little():
    a, b = item(1), item(1, box=(21, 30.8, 61, 38.8))               # 1 mm shift between the two runs
    assert rt.stable_match(a, b) > 0


def test_different_place_type_or_page_is_not_stable():
    a = item(1)
    assert rt.stable_match(a, item(1, box=(120, 200, 160, 208))) == 0         # other place
    assert rt.stable_match(a, item(1, dtype="Stamp")) == 0                    # other type
    assert rt.stable_match(a, item(1, page=2)) == 0                           # other page
    assert rt.stable_match(a, item(1, text="Completely different words here")) == 0   # incompatible identity


def test_confidence_alone_never_identifies_a_finding():
    assert rt.stable_match(item(1, conf=0.9), item(2, box=(150, 150, 180, 160), conf=0.9)) == 0


def test_duplicates_are_removed():
    d = rt.dedupe([item(1, conf=0.8), item(2, conf=0.95), item(3, box=(100, 100, 120, 110))])
    assert len(d) == 2 and d[0]["confidence"] == 0.95


def test_missing_confidence_is_safe():
    a, b = item(1), item(1)
    a["confidence"] = None
    del b["confidence"]
    r = rt.evaluate_retry(report([a], [item(5, box=(1, 1, 5, 5), status="uncertain", cat="Uncertain")]), report([b], []))
    assert r["keep"] == "retry"                                               # fewer uncertain findings; no crash on None / missing


# ---------------------------------------------------------------- Test 4: retry tie
def test_tie_keeps_retry_only_if_stable_confidence_is_higher():
    unc = item(9, box=(100, 100, 130, 110), status="uncertain", cat="Uncertain", conf=0.55)
    first = report([item(1, conf=0.80)], [unc])
    better = report([item(1, conf=0.93)], [dict(unc)])
    worse = report([item(1, conf=0.70)], [dict(unc)])
    same = report([item(1, conf=0.80)], [dict(unc)])
    assert rt.evaluate_retry(first, better)["keep"] == "retry"
    assert rt.evaluate_retry(first, worse)["keep"] == "first"
    assert rt.evaluate_retry(first, same)["keep"] == "first"                  # no improvement: original is retained


def test_fewer_uncertain_wins_and_more_uncertain_loses():
    u1, u2 = item(8, box=(100, 100, 120, 110), status="uncertain", cat="Uncertain"), item(9, box=(150, 20, 170, 30), status="uncertain", cat="Uncertain")
    first = report([item(1)], [u1, u2])
    assert rt.evaluate_retry(first, report([item(1)], [u1]))["keep"] == "retry"
    assert rt.evaluate_retry(report([item(1)], [u1]), report([item(1)], [u1, u2]))["keep"] == "first"


def test_confidence_is_not_raised_by_the_retry():
    first = report([item(1, conf=0.8)], [item(9, box=(100, 100, 130, 110), status="uncertain", cat="Uncertain", conf=0.5)])
    retry_r = report([item(1, conf=0.8)], [])
    sel = rt.finalize(first, retry_r, rt.evaluate_retry(first, retry_r), 400, 300, Path("."), Path("."), Path("."))
    assert sel["differences"][0]["confidence"] == 0.8


# ---------------------------------------------------------------- Test 5: higher-resolution-only finding
def _dirs(tmp_path):
    first_dir, retry_dir = tmp_path / "first", tmp_path / "retry"
    for d in (first_dir, retry_dir):
        (d / "evidence").mkdir(parents=True)
    return first_dir, retry_dir


def test_retry_only_confirmed_finding_is_kept_separately_when_retry_wins(tmp_path):
    first_dir, retry_dir = _dirs(tmp_path)
    (retry_dir / "evidence" / "p0_2_a.png").write_bytes(b"png")
    only = item(2, text="Tiny dot", box=(120, 200, 124, 204), ev={"a": "evidence/p0_2_a.png"})
    first = report([item(1)], [item(9, box=(100, 100, 130, 110), status="uncertain", cat="Uncertain")])
    retry_r = report([item(1), only], [])
    dec = rt.evaluate_retry(first, retry_r)
    assert dec["keep"] == "retry" and len(dec["hr_only"]) == 1
    sel = refresh_summary(rt.finalize(first, retry_r, dec, 400, 300, first_dir, retry_dir, retry_dir))
    assert [i["document_a"] for i in sel["differences"]] == ["Amount"]                 # not counted as confirmed
    hr = sel["seen_only_at_higher_resolution"]
    assert len(hr) == 1 and hr[0]["details"]["seen_only_at_higher_resolution"] is True
    assert hr[0]["details"]["higher_dpi"] == 400 and hr[0]["details"]["first_dpi"] == 300
    assert "only in the 400 DPI" in hr[0]["details"]["note"] and hr[0]["page_a"] == 1 and hr[0]["location"]["a"]["bbox_mm"] == [120, 200, 124, 204]
    assert (retry_dir / "evidence" / "hr_p0_2_a.png").is_file() and hr[0]["evidence"]["a"] == "evidence/hr_p0_2_a.png"
    assert sel["summary"]["total_differences"] == 1 and sel["summary"]["seen_only_at_higher_resolution"] == 1


def test_retry_only_finding_is_kept_when_first_wins_too(tmp_path):
    first_dir, retry_dir = _dirs(tmp_path)
    (retry_dir / "evidence" / "p0_2_a.png").write_bytes(b"png")
    only = item(2, text="Tiny dot", box=(120, 200, 124, 204), ev={"a": "evidence/p0_2_a.png"})
    first = report([item(1, conf=0.9)], [item(9, box=(100, 100, 130, 110), status="uncertain", cat="Uncertain")])
    retry_r = report([item(1, conf=0.9), only], [item(9, box=(100, 100, 130, 110), status="uncertain", cat="Uncertain")])
    dec = rt.evaluate_retry(first, retry_r)
    assert dec["keep"] == "first"
    sel = rt.finalize(first, retry_r, dec, 400, 300, first_dir, retry_dir, first_dir)
    assert len(sel["seen_only_at_higher_resolution"]) == 1 and len(sel["differences"]) == 1
    assert (first_dir / "evidence" / "hr_p0_2_a.png").is_file()                       # evidence preserved in the selected result


def test_original_only_confirmed_finding_is_not_lost_when_retry_wins(tmp_path):
    first_dir, retry_dir = _dirs(tmp_path)
    (first_dir / "evidence" / "p0_3_a.png").write_bytes(b"png")
    lost = item(3, text="Only at 300", box=(150, 40, 170, 48), ev={"a": "evidence/p0_3_a.png"})
    first = report([item(1), lost], [item(9, box=(100, 100, 130, 110), status="uncertain", cat="Uncertain")])
    retry_r = report([item(1)], [])
    dec = rt.evaluate_retry(first, retry_r)
    sel = refresh_summary(rt.finalize(first, retry_r, dec, 400, 300, first_dir, retry_dir, retry_dir))
    kept = [i for i in sel["uncertain"] if i["details"].get("seen_only_at_original_resolution")]
    assert len(kept) == 1 and kept[0]["confidence"] <= rt.FIRST_ONLY_CAP and kept[0]["status"] == "uncertain"
    assert (retry_dir / "evidence" / "or_p0_3_a.png").is_file()


def test_duplicates_across_both_runs_are_counted_once(tmp_path):
    first_dir, retry_dir = _dirs(tmp_path)
    dup = [item(2, text="Tiny dot", box=(120, 200, 124, 204)), item(3, text="Tiny dot", box=(120.2, 200, 124.2, 204))]
    first = report([item(1)], [item(9, box=(100, 100, 130, 110), status="uncertain", cat="Uncertain")])
    retry_r = report([item(1)] + dup, [])
    dec = rt.evaluate_retry(first, retry_r)
    assert len(dec["hr_only"]) == 1


def test_no_uncertain_findings_means_no_retry_needed():
    first = report([item(1)], [])
    dec = rt.evaluate_retry(first, report([item(1)], []))
    assert dec["keep"] == "first" and dec["first_uncertain"] == 0 and not dec["hr_only"]


# ---------------------------------------------------------------- the worker stage: failure, evidence and files
def _job(tmp_path, first_rep):
    jd = tmp_path / "job"
    (jd / "evidence").mkdir(parents=True)
    (jd / "evidence" / "p0_1_a.png").write_bytes(b"first")
    write_reports(refresh_summary(first_rep), jd)
    return jd


def test_retry_failure_preserves_the_original_result(tmp_path):
    first = report([item(1, ev={"a": "evidence/p0_1_a.png"})], [item(9, box=(100, 100, 130, 110), status="uncertain", cat="Uncertain")])
    jd = _job(tmp_path, first)

    def boom(*a, **k):
        raise MemoryError("out of memory")
    out = worker.retry_stage(jd, first, 300, 400, boom, ("a", "a", "b", "b"), object(), lambda *a: None)
    sel = out["diagnostics"]["dpi_selection"]
    assert sel["retried"] and sel["retry_result"] == "failed" and sel["kept"] == "first" and sel["used"] == 300
    assert (jd / "evidence" / "p0_1_a.png").read_bytes() == b"first" and not (jd / "_retry").exists()
    saved = json.loads((jd / "report.json").read_text(encoding="utf-8"))
    assert saved["diagnostics"]["dpi_selection"]["retry_result"] == "failed" and len(saved["differences"]) == 1
    assert "could not be completed" in (jd / "report.txt").read_text(encoding="utf-8")


def test_retry_that_wins_replaces_files_and_keeps_special_evidence(tmp_path):
    first = report([item(1, ev={"a": "evidence/p0_1_a.png"}), item(3, text="Only at 300", box=(150, 40, 170, 48), ev={"a": "evidence/p0_1_a.png"})],
                   [item(9, box=(100, 100, 130, 110), status="uncertain", cat="Uncertain")])
    jd = _job(tmp_path, first)
    retry_r = report([item(1, ev={"a": "evidence/p0_5_a.png"}), item(2, text="Tiny dot", box=(120, 200, 124, 204), ev={"a": "evidence/p0_5_a.png"})], [])

    def fake_compare(pa, na, pb, nb, out_dir, cfg, cb):
        (out_dir / "evidence").mkdir(parents=True)
        (out_dir / "evidence" / "p0_5_a.png").write_bytes(b"retry")
        write_reports(refresh_summary(retry_r), out_dir)
        return retry_r
    out = worker.retry_stage(jd, first, 300, 400, fake_compare, ("a", "a", "b", "b"), object(), lambda *a: None)
    sel = out["diagnostics"]["dpi_selection"]
    assert sel["retry_result"] == "improved" and sel["kept"] == "retry" and sel["used"] == 400 and sel["higher_resolution_only"] == 1
    assert (jd / "evidence" / "p0_5_a.png").read_bytes() == b"retry"
    assert (jd / "evidence" / "hr_p0_5_a.png").is_file() and (jd / "evidence" / "or_p0_1_a.png").is_file()   # both kinds preserved
    saved = json.loads((jd / "report.json").read_text(encoding="utf-8"))
    assert len(saved["seen_only_at_higher_resolution"]) == 1 and saved["summary"]["total_differences"] == 1
    assert "SEEN ONLY AT HIGHER RESOLUTION" in (jd / "report.txt").read_text(encoding="utf-8")
    assert not (jd / "_retry").exists()


def test_retry_that_does_not_help_keeps_the_original(tmp_path):
    first = report([item(1, ev={"a": "evidence/p0_1_a.png"})], [item(9, box=(100, 100, 130, 110), status="uncertain", cat="Uncertain", conf=0.6)])
    jd = _job(tmp_path, first)
    retry_r = report([item(1)], [item(9, box=(100, 100, 130, 110), status="uncertain", cat="Uncertain", conf=0.6)])

    def fake_compare(pa, na, pb, nb, out_dir, cfg, cb):
        out_dir.mkdir(parents=True, exist_ok=True)
        return retry_r
    out = worker.retry_stage(jd, first, 300, 400, fake_compare, ("a", "a", "b", "b"), object(), lambda *a: None)
    sel = out["diagnostics"]["dpi_selection"]
    assert sel["retry_result"] == "not_improved" and sel["kept"] == "first" and sel["used"] == 300
    assert "original result is shown" in (jd / "report.txt").read_text(encoding="utf-8")
