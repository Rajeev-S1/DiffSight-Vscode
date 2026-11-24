"""Benchmark: runs NexaScan on every suite case and scores it against the injected ground truth.

Metrics: recall (confirmed, and confirmed+uncertain), precision, false positives, duplicates,
type accuracy, processing time, peak memory.   Usage:
    python tests/benchmark.py [suite_dir] [--only substr] [--json out.json]
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOL_MM = 5.0
MM = 300 / 25.4


def run_one(case_dir: Path, out_dir: Path) -> dict:
    """Run one case in a fresh process; returns report + time + peak RSS."""
    code = (
        "import sys,json,time,resource,warnings;warnings.filterwarnings('ignore');sys.path.insert(0,%r);"
        "from pathlib import Path;from nexascan.config import Config;from nexascan.pipeline import compare_documents;"
        "d=Path(%r);o=Path(%r);fa=[p for p in d.glob('A.*')][0];fb=[p for p in d.glob('B.*')][0];"
        "t=time.time();r=compare_documents(str(fa),fa.name,str(fb),fb.name,o,Config(use_cache=False));"
        "print('@@'+json.dumps({'seconds':time.time()-t,'rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}))"
    ) % (str(ROOT), str(case_dir), str(out_dir))
    shutil.rmtree(out_dir, ignore_errors=True)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=900)
    meta = None
    for line in p.stdout.splitlines():
        if line.startswith("@@"):
            meta = json.loads(line[2:])
    if meta is None:
        return {"error": (p.stderr or p.stdout)[-800:]}
    rep = json.loads((out_dir / "report.json").read_text())
    return {"report": rep, **meta}


def _boxes(item: dict):
    out = []
    for side in ("a", "b"):
        loc = item["location"].get(side)
        if loc:
            out.append((side, loc["bbox_px"]))
    return out


def _hit(item: dict, t: dict, page_area: float) -> bool:
    # page must agree (page-level items have no box)
    ok_page = (t["page_a"] is not None and item["page_a"] == t["page_a"]) or (t["page_b"] is not None and item["page_b"] == t["page_b"])
    if not ok_page:
        return False
    if t["bbox_a"] is None and t["bbox_b"] is None:
        return True      # page-level truth: same page is enough
    tol = TOL_MM * MM
    for side, b in _boxes(item):
        tb = t["bbox_a"] if side == "a" else t["bbox_b"]
        tb = tb or t["bbox_a"]
        if tb is None:
            continue
        if (b[2] - b[0]) * (b[3] - b[1]) > 0.4 * page_area:
            continue
        if b[0] < tb[2] + tol and b[2] > tb[0] - tol and b[1] < tb[3] + tol and b[3] > tb[1] - tol:
            return True
    return False


def score(report: dict, truth: list) -> dict:
    page_area = 2480 * 3508
    conf, unc = report["differences"], report["uncertain"]
    required = [t for t in truth if not t["optional"]]
    res = {"truth": len(required), "confirmed": len(conf), "uncertain": len(unc)}
    matched_conf, matched_any, dup, type_ok, fps, missed = set(), set(), 0, 0, [], []
    assigned: set = set()
    for it in conf:
        hits = [i for i, t in enumerate(truth) if _hit(it, t, page_area)]
        if not hits:
            fps.append(f"p{it['page']} {it['type']} {it['category']}: {it['difference'][:90]}")
            continue
        def dist(i):
            t = truth[i]
            tb = t["bbox_a"] or t["bbox_b"]
            fb = (it["location"]["a"] or it["location"]["b"])
            if not tb or not fb:
                return 0.0
            fb = fb["bbox_px"]
            return ((tb[0] + tb[2]) / 2 - (fb[0] + fb[2]) / 2) ** 2 + ((tb[1] + tb[3]) / 2 - (fb[1] + fb[3]) / 2) ** 2
        nearest = min(hits, key=dist)
        if nearest in assigned:
            dup += 1
        assigned.add(nearest)
        for i in hits:
            matched_conf.add(i)
            matched_any.add(i)
        if it["dtype"] in truth[nearest]["kinds"] or it["details"].get("original_category") in truth[nearest]["kinds"]:
            type_ok += 1
    for it in unc:
        for i, t in enumerate(truth):
            if _hit(it, t, page_area):
                matched_any.add(i)
    req_idx = [i for i, t in enumerate(truth) if not t["optional"]]
    tp = [i for i in req_idx if i in matched_conf]
    for i in req_idx:
        if i not in matched_any:
            missed.append(truth[i]["desc"])
    res.update({"tp": len(tp), "tp_incl_uncertain": len([i for i in req_idx if i in matched_any]),
                "fp": len(fps), "dups": dup, "type_ok": type_ok, "fp_list": fps, "missed": missed,
                "unmatched_confirmed_matches": len(conf) - len(fps)})
    return res


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("suite", nargs="?", default=str(ROOT / "data" / "suite"))
    ap.add_argument("--only", default="")
    ap.add_argument("--json", default=str(ROOT / "data" / "benchmark_results.json"))
    ns = ap.parse_args()
    suite = Path(ns.suite)
    only, out_json = ns.only, ns.json
    names = json.loads((suite / "cases.json").read_text())
    rows, tot = [], dict(truth=0, tp=0, tpu=0, fp=0, conf=0, dups=0, typeok=0, matched_conf=0)
    print(f"{'case':26s} {'truth':>5s} {'TP':>3s} {'+unc':>4s} {'FP':>3s} {'dup':>3s} {'conf':>4s} {'sec':>6s} {'MB':>6s}  notes")
    for n in names:
        if only and only not in n:
            continue
        r = run_one(suite / n, ROOT / "data" / "bench_out" / n)
        if "error" in r:
            print(f"{n:26s} ERROR {r['error'][-300:]}")
            rows.append({"case": n, "error": r["error"]})
            continue
        truth = json.loads((suite / n / "truth.json").read_text())["truth"]
        s = score(r["report"], truth)
        rows.append({"case": n, **{k: v for k, v in s.items()}, "seconds": r["seconds"], "rss_mb": r["rss_mb"]})
        tot["truth"] += s["truth"]; tot["tp"] += s["tp"]; tot["tpu"] += s["tp_incl_uncertain"]
        tot["fp"] += s["fp"]; tot["conf"] += s["confirmed"]; tot["dups"] += s["dups"]
        tot["typeok"] += s["type_ok"]; tot["matched_conf"] += s["unmatched_confirmed_matches"]
        note = ""
        if s["missed"]:
            note += " MISSED: " + "; ".join(s["missed"])
        if s["fp_list"]:
            note += " | FP: " + " ## ".join(s["fp_list"][:3])
        print(f"{n:26s} {s['truth']:5d} {s['tp']:3d} {s['tp_incl_uncertain']:4d} {s['fp']:3d} {s['dups']:3d} {s['confirmed']:4d} {r['seconds']:6.1f} {r['rss_mb']:6.0f} {note}")
    ok = [r for r in rows if "error" not in r]
    secs = sorted(r["seconds"] for r in ok)
    prec = tot["matched_conf"] / tot["conf"] if tot["conf"] else 1.0
    summary = {
        "cases": len(ok), "truth_items": tot["truth"], "recall_confirmed": round(tot["tp"] / max(1, tot["truth"]), 4),
        "recall_incl_uncertain": round(tot["tpu"] / max(1, tot["truth"]), 4), "precision": round(prec, 4),
        "false_positives": tot["fp"], "confirmed_findings": tot["conf"], "duplicate_findings": tot["dups"],
        "type_accuracy_of_matches": round(tot["typeok"] / max(1, tot["matched_conf"]), 4),
        "median_seconds": round(secs[len(secs) // 2], 2) if secs else None, "max_seconds": round(secs[-1], 2) if secs else None,
        "peak_rss_mb": round(max((r["rss_mb"] for r in ok), default=0), 0),
    }
    print("\nSUMMARY", json.dumps(summary, indent=1))
    Path(out_json).write_text(json.dumps({"summary": summary, "cases": rows}, indent=1))


if __name__ == "__main__":
    main()
