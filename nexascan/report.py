"""Service 10 – Final report generator (JSON, plain text, HTML)."""
from __future__ import annotations

import base64
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import __version__
from .config import Config
from .models import DocInfo, Finding, PageData
from .text_categories import CHECKLIST_CATEGORIES, normalized_category_counts

DISCLAIMER = ("DiffSight reports differences found by automated image and text analysis. It does not claim 100% "
              "accuracy. Findings marked 'uncertain' have weaker evidence and should be checked by a person. "
              "Stamps, signatures and handwriting are compared by appearance only; no statement is made about who "
              "made a mark. Content that could not be read is reported as unreadable, never guessed.")

TITLES = {"Text": "Text Difference", "Number": "Text Difference (number)", "Punctuation": "Punctuation Difference",
          "Paragraph": "Paragraph Difference",
          "Stamp": "Stamp Difference", "Signature": "Signature Difference", "Handwriting": "Handwriting Difference",
          "Image": "Image Difference", "Block": "Visual Difference (solid block)", "Line": "Visual Difference (line)",
          "Overlap": "Overlapping Content", "Hidden text": "Overlapping Content (hidden text)",
          "Font": "Font / Style Difference", "Layout": "Layout Difference", "Colour": "Visual Difference (colour)",
          "Visual": "Visual Difference", "Page": "Page Difference", "Alignment": "Alignment Warning",
          "Processing": "Processing Warning"}


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float):
        return round(o, 4)
    return o


def _box(b, dpi, dims) -> Optional[dict]:
    if b is None:
        return None
    mm = 25.4 / dpi
    d = {"bbox_px": [round(float(v), 1) for v in b], "bbox_mm": [round(float(v) * mm, 1) for v in b]}
    if dims:
        w, h = dims
        d["bbox_rel"] = [round(b[0] / w, 4), round(b[1] / h, 4), round(b[2] / w, 4), round(b[3] / h, 4)]
    return d


def finding_to_dict(f: Finding, cfg: Config, pa_list: List[PageData], pb_list: List[PageData]) -> dict:
    da = (pa_list[f.page_a - 1].width, pa_list[f.page_a - 1].height) if f.page_a else None
    db = (pb_list[f.page_b - 1].width, pb_list[f.page_b - 1].height) if f.page_b else None
    category = f.category
    d = {
        "number": f.id,
        "type": TITLES.get(f.dtype, f.dtype + " Difference"),
        "dtype": f.dtype,
        "category": category,
        "status": f.status,
        "page": f.page_a if f.page_a is not None else f.page_b,
        "page_a": f.page_a, "page_b": f.page_b,
        "document_a": f.a_content, "document_b": f.b_content,
        "difference": f.description[:1].upper() + f.description[1:] if f.description else "",
        "location": {"dpi": cfg.dpi, "a": _box(f.bbox_a, cfg.dpi, da), "b": _box(f.bbox_b, cfg.dpi, db)},
        "confidence": round(float(f.confidence), 3),
        "engines": sorted(set(f.engines)),
        "checklist_categories": list(dict.fromkeys(
            f.checklist_categories or f.details.get("checklist_categories") or []
        )),
        "detection_method": f.detection_method,
        "character_change": f.character_change,
        "whitespace_change": f.whitespace_change,
        "evidence": {k: f"evidence/{v}" for k, v in f.evidence.items()},
        "details": {k: v for k, v in f.details.items() if k not in ("region_a_frame",)},
    }
    return _clean(d)


def verified_unchanged(pairs, page_diag: List[dict]) -> List[dict]:
    """Per matched page pair: how much text was compared and found identical (the reassurance that
    "no difference" on a page means "checked and equal", not "not looked at")."""
    rows = []
    for p, d in zip(pairs, page_diag):
        t = d.get("text") if isinstance(d, dict) else None
        row = {"page_a": (p.a + 1 if p.a is not None else None), "page_b": (p.b + 1 if p.b is not None else None),
               "kind": p.kind, "compared": bool(t)}
        if t:
            row.update({"lines_a": t.get("lines_a", 0), "lines_b": t.get("lines_b", 0),
                        "matched_lines": t.get("matched_lines", 0),
                        "words_a": t.get("words_a", 0), "words_b": t.get("words_b", 0),
                        "matched_words": t.get("matched_words", 0),
                        "percent_identical": t.get("identical_pct", 0.0)})
        else:
            row.update({"lines_a": 0, "lines_b": 0, "matched_lines": 0, "words_a": 0, "words_b": 0,
                        "matched_words": 0, "percent_identical": 0.0})
        rows.append(row)
    return rows


def build_report(ia: DocInfo, ib: DocInfo, pa_list, pb_list, pairs, findings: List[Finding], suppressed: List[dict],
                 page_diag: List[dict], timing: dict, peak_mb: float, cfg: Config, out_dir: Path) -> dict:
    items = [finding_to_dict(f, cfg, pa_list, pb_list) for f in findings]
    confirmed = [i for i in items if i["status"] == "confirmed"]
    uncertain = [i for i in items if i["status"] == "uncertain"]
    # renumber so the two lists are each point-wise 1..N
    for n, i in enumerate(confirmed, 1):
        i["number"] = n
    for n, i in enumerate(uncertain, 1):
        i["number"] = n
    by_cat, by_type = count_confirmed(confirmed)
    text_comparison_summary = normalized_category_counts(confirmed)
    report = {
        "title": "DiffSight – Document Comparison Report",
        "tagline": "Every Detail. Every Difference.",
        "version": __version__,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "document_a": {"name": ia.display_name, "pages": ia.pages, "kind": ia.kind, "sha256": ia.sha256},
        "document_b": {"name": ib.display_name, "pages": ib.pages, "kind": ib.kind, "sha256": ib.sha256},
        "summary": {"total_differences": len(confirmed), "uncertain_findings": len(uncertain),
                    "by_category": by_cat, "by_type": by_type,
                    "text_comparison_summary": text_comparison_summary,
                    "page_pairs": [{"a": (p.a + 1 if p.a is not None else None), "b": (p.b + 1 if p.b is not None else None),
                                    "kind": p.kind, "similarity": round(p.sim, 3)} for p in pairs]},
        "differences": confirmed,
        "uncertain": uncertain,
        "seen_only_at_higher_resolution": [],
        "verified_unchanged": verified_unchanged(pairs, page_diag),
        "diagnostics": {"timing": timing, "peak_memory_mb": peak_mb, "pages": page_diag,
                        "suppressed_count": len(suppressed), "suppressed": suppressed[:200],
                        "dpi_selection": dict(cfg.dpi_info) if cfg.dpi_info else {"mode": "manual", "used": cfg.dpi},
                        "config": {"dpi": cfg.dpi, "tol_mm_native": cfg.tol_mm_native, "tol_mm_scan": cfg.tol_mm_scan,
                                   "confirmed_min": cfg.confirmed_min, "uncertain_min": cfg.uncertain_min}},
        "disclaimer": DISCLAIMER,
        "text_comparison_summary": text_comparison_summary,
    }
    report = _clean(report)
    write_reports(report, out_dir)
    return report


def count_confirmed(confirmed: list) -> tuple:
    by_cat: Dict[str, int] = {}
    by_type: Dict[str, int] = {}
    for i in confirmed:
        c = i["details"].get("original_category", i["category"]) if i["category"] == "Uncertain" else i["category"]
        by_cat[c] = by_cat.get(c, 0) + 1
        by_type[i["type"]] = by_type.get(i["type"], 0) + 1
    return by_cat, by_type


def refresh_summary(report: dict) -> dict:
    """Renumber the lists and recompute the counts after findings were moved between lists."""
    for key in ("differences", "uncertain", "seen_only_at_higher_resolution"):
        for n, i in enumerate(report.get(key) or [], 1):
            i["number"] = n
    by_cat, by_type = count_confirmed(report["differences"])
    text_comparison_summary = normalized_category_counts(report["differences"])
    s = report["summary"]
    s.update({"total_differences": len(report["differences"]), "uncertain_findings": len(report["uncertain"]),
              "by_category": by_cat, "by_type": by_type,
              "text_comparison_summary": text_comparison_summary,
              "seen_only_at_higher_resolution": len(report.get("seen_only_at_higher_resolution") or [])})
    report["text_comparison_summary"] = text_comparison_summary
    return report


def write_reports(report: dict, out_dir: Path) -> None:
    """(Re)write report.json / report.txt / report.html from the report dict."""
    out_dir = Path(out_dir)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "report.txt").write_text(to_text(report), encoding="utf-8")
    (out_dir / "report.html").write_text(to_html(report, out_dir, embed=False), encoding="utf-8")


def dpi_line(r: dict) -> str:
    """Human-readable: what was measured, which resolution NexaScan chose and why, and what the retry did."""
    d = (r.get("diagnostics") or {}).get("dpi_selection") or {}
    if not d:
        return ""
    txt = d.get("message") or f"Analysed at {d.get('used')} DPI."
    if d.get("retried"):
        res = d.get("retry_result")
        if res == "improved":
            txt += (f" Re-checked at {d.get('retry_dpi')} DPI because {d.get('first_uncertain')} finding(s) were uncertain at "
                    f"{d.get('first_dpi')} DPI; the higher-resolution result is shown ({d.get('retry_reason')}).")
        elif res == "failed":
            txt += f" A re-check at {d.get('retry_dpi')} DPI could not be completed; the original result is shown."
        else:
            txt += (f" Re-checked at {d.get('retry_dpi')} DPI because {d.get('first_uncertain')} finding(s) were uncertain at "
                    f"{d.get('first_dpi')} DPI; it was not clearer, so the original result is shown ({d.get('retry_reason')}).")
        if d.get("higher_resolution_only"):
            txt += f" {d['higher_resolution_only']} finding(s) appeared only at the higher resolution and are listed for review."
    return txt


# ----------------------------------------------------------------------------------------- text


def _loc(d: Optional[dict]) -> str:
    if not d:
        return "n/a"
    x0, y0, x1, y1 = d["bbox_mm"]
    return f"x {x0:.0f}-{x1:.0f} mm, y {y0:.0f}-{y1:.0f} mm from top-left (px {', '.join(str(int(v)) for v in d['bbox_px'])})"


def _item_text(i: dict, n: Optional[int] = None) -> List[str]:
    lines = [f"{i['number']}. {i['type']}"]
    page = i["page_a"] if i["page_a"] == i["page_b"] or i["page_b"] is None else f"{i['page_a']} (A) / {i['page_b']} (B)"
    if i["page_a"] is None:
        page = f"{i['page_b']} (B)"
    lines += [f"   - Page: {page}",
              f"   - Document A: {i['document_a']}",
              f"   - Document B: {i['document_b']}",
              f"   - Difference: {i['difference']}",
              f"   - Category: {i['category']}" + (f" (looks like: {i['details']['original_category']})" if i["category"] == "Uncertain" and "original_category" in i["details"] else ""),
              f"   - Location in A: {_loc(i['location']['a'])}",
              f"   - Location in B: {_loc(i['location']['b'])}",
              f"   - Confidence: {i['confidence'] * 100:.0f}%",
              f"   - Detected by: {', '.join(i['engines'])}"]
    if i["evidence"]:
        lines.append("   - Visual evidence: " + ", ".join(f"{k}: {v}" for k, v in i["evidence"].items()))
    return lines


def to_text(r: dict) -> str:
    s = r["summary"]
    out = [r["title"], r["tagline"], "",
           f"Document A: {r['document_a']['name']} ({r['document_a']['pages']} page(s))",
           f"Document B: {r['document_b']['name']} ({r['document_b']['pages']} page(s))",
           *([dpi_line(r)] if dpi_line(r) else []),
           f"Total Differences: {s['total_differences']}"
           + (f"   (+ {s['uncertain_findings']} uncertain finding(s) listed separately)" if s["uncertain_findings"] else ""),
           ""]
    out += ["TEXT COMPARISON SUMMARY"]
    for n, name in enumerate(CHECKLIST_CATEGORIES, 1):
        out.append(f"{n}. {name} - {r.get('text_comparison_summary', {}).get(name, 0)}")
    out.append("")
    for i in r["differences"]:
        out += _item_text(i) + [""]
    if not r["differences"]:
        out += ["No confirmed differences were found.", ""]
    if r["uncertain"]:
        out += ["-" * 60, "UNCERTAIN FINDINGS (weaker evidence – please verify manually)", "-" * 60, ""]
        for i in r["uncertain"]:
            out += _item_text(i) + [""]
    if r.get("seen_only_at_higher_resolution"):
        out += ["-" * 60, "SEEN ONLY AT HIGHER RESOLUTION (review manually - not independently verified)", "-" * 60, ""]
        for i in r["seen_only_at_higher_resolution"]:
            out += _item_text(i) + [f"   - Note: {i['details'].get('note', '')}", ""]
    out += ["-" * 60, "VERIFIED UNCHANGED (per page: text that was compared and found identical)", "-" * 60]
    for v in r.get("verified_unchanged", []):
        pg = f"A{v['page_a'] or '-'} / B{v['page_b'] or '-'}"
        if v["compared"]:
            out.append(f"  Page {pg}: {v['matched_lines']} of {max(v['lines_a'], v['lines_b'])} line(s) and "
                       f"{v['matched_words']} of {max(v['words_a'], v['words_b'])} word(s) identical "
                       f"({v['percent_identical']:.1f}% identical)")
        else:
            out.append(f"  Page {pg}: not compared line by line ({v['kind']} page)")
    out += ["", "-" * 60, "Note: " + r["disclaimer"]]
    return "\n".join(out)


# ----------------------------------------------------------------------------------------- html

_CSS = """
:root{--navy:#0a1f33;--navy2:#103a52;--teal:#0e9f8e;--teal2:#19d3b4;--ink:#10202d;--mut:#5b6b78;--line:#dbe4ea;--paper:#fff;--bg:#eef3f6;--ok:#0f7b5f;--warn:#b45309;--bad:#b42318}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 "Segoe UI",system-ui,-apple-system,Roboto,sans-serif}
.rhead{display:flex;align-items:center;gap:18px;padding:30px max(24px,calc((100% - 1100px)/2));background:linear-gradient(120deg,var(--navy),var(--navy2));color:#fff;border-bottom:4px solid var(--teal2)}
.mark{flex:none;width:52px;height:52px;border-radius:14px;background:linear-gradient(135deg,var(--teal2),var(--teal));color:#062b2a;display:grid;place-items:center;font:800 28px system-ui}
.rhead h1{margin:0;font-size:25px;letter-spacing:.2px}.rhead .sub{color:#a9c7d6;margin-top:2px}
main{max-width:1100px;margin:0 auto;padding:26px 16px 70px}
.meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px;margin:0 0 14px}
.meta span{background:var(--paper);border:1px solid var(--line);border-radius:10px;padding:12px 14px;color:var(--mut)}.meta b{display:block;color:var(--ink);font-size:16px;word-break:break-word}
.sum{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px}.pill{border:1px solid var(--line);background:var(--paper);border-radius:999px;padding:3px 12px;font-size:13px;font-weight:600}
.tag{background:#e8f6f4;border-left:4px solid var(--teal);padding:8px 12px;border-radius:6px;color:#0b4f48;margin:0 0 18px}
h2{margin:34px 0 12px;font-size:19px;color:var(--navy);display:flex;align-items:center;gap:10px}h2::before{content:"";width:6px;height:20px;border-radius:3px;background:var(--teal)}
.item{background:var(--paper);border:1px solid var(--line);border-left:5px solid var(--teal);border-radius:10px;margin:14px 0;padding:14px 18px;box-shadow:0 1px 3px rgba(16,32,45,.06)}
.c-Added{border-left-color:var(--ok)}.c-Removed,.c-Overlapped{border-left-color:var(--bad)}.c-Modified,.c-Moved{border-left-color:var(--warn)}.c-Uncertain{border-left-color:#94a3b8}
.item h3{margin:0 0 8px;font-size:16px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.cat{font-size:12px;font-weight:700;border-radius:999px;padding:2px 10px;background:#e0f2ef;color:#0b6b60}.cat.Added{background:#dcf5e9;color:var(--ok)}.cat.Removed,.cat.Overlapped{background:#fde4e1;color:var(--bad)}.cat.Modified,.cat.Moved{background:#fdecd2;color:var(--warn)}.cat.Uncertain{background:#e5e9ee;color:#475467}
dl{display:grid;grid-template-columns:140px 1fr;gap:5px 14px;margin:8px 0}dt{color:var(--mut);font-size:12.5px;text-transform:uppercase;letter-spacing:.04em;padding-top:2px}dd{margin:0;word-break:break-word}
.ev{display:flex;flex-wrap:wrap;gap:12px;margin-top:10px}.ev figure{margin:0;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:#fff;flex:1 1 280px;max-width:100%}
.ev img{display:block;max-width:100%;height:auto}figcaption{font-size:12px;color:var(--mut);padding:6px 10px;border-top:1px solid var(--line);background:#f6f9fb}
table.vu{border-collapse:collapse;width:100%;font-size:14px;background:var(--paper);border:1px solid var(--line);border-radius:8px}table.vu th{background:#f1f5f8;color:var(--mut)}
table.vu th,table.vu td{border-bottom:1px solid var(--line);padding:7px 12px;text-align:left}
.note{color:var(--mut);font-size:13px;margin-top:30px}
@media print{body{background:#fff}.item{break-inside:avoid;box-shadow:none}.rhead{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
"""


def _img(src: str, out_dir: Optional[Path], embed: bool) -> str:
    if embed and out_dir is not None:
        p = out_dir / src
        if p.is_file():
            return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()
    return html.escape(src)


_EV_LABEL = {"a": "Document A", "b": "Document B", "overlay": "Highlighted differences", "layers": "Layers"}


def _item_html(i: dict, out_dir: Optional[Path], embed: bool) -> str:
    e = html.escape
    page = f"{i['page_a']}" if i["page_a"] == i["page_b"] or i["page_b"] is None else f"{i['page_a']} (A) / {i['page_b']} (B)"
    if i["page_a"] is None:
        page = f"{i['page_b']} (B)"
    figs = "".join(f'<figure><img loading="lazy" src="{_img(v, out_dir, embed)}" alt="{e(k)} evidence"><figcaption>{e(_EV_LABEL.get(k, k))}</figcaption></figure>'
                   for k, v in sorted(i["evidence"].items(), key=lambda kv: ["a", "b", "overlay", "layers"].index(kv[0]) if kv[0] in ("a", "b", "overlay", "layers") else 9))
    orig = f" – looks like: {e(i['details']['original_category'])}" if i["category"] == "Uncertain" and "original_category" in i["details"] else ""
    tags = ", ".join(i.get("checklist_categories") or [])
    checklist = f'<dt>Checklist categories</dt><dd>{e(tags)}</dd>' if tags else ""
    return (f'<section class="item c-{e(i["category"])}"><h3>{i["number"]}. {e(i["type"])}<span class="cat {e(i["category"])}">{e(i["category"])}</span></h3>'
            f'<dl><dt>Page</dt><dd>{e(str(page))}</dd><dt>Document A</dt><dd>{e(i["document_a"])}</dd>'
            f'<dt>Document B</dt><dd>{e(i["document_b"])}</dd><dt>Difference</dt><dd>{e(i["difference"])}{orig}</dd>{checklist}'
            f'<dt>Location (A)</dt><dd>{e(_loc(i["location"]["a"]))}</dd><dt>Location (B)</dt><dd>{e(_loc(i["location"]["b"]))}</dd>'
            f'<dt>Confidence</dt><dd>{i["confidence"] * 100:.0f}% <span style="color:var(--mut)">({e(", ".join(i["engines"]))})</span></dd></dl>'
            f'<div class="ev">{figs}</div></section>')


def to_html(r: dict, out_dir: Optional[Path] = None, embed: bool = False) -> str:
    e = html.escape
    s = r["summary"]
    pills = "".join(f'<span class="pill">{e(k)}: {v}</span>' for k, v in sorted(s["by_category"].items()))
    head = (f'<header class="rhead"><span class="mark">D</span><div><h1>{e(r["title"])}</h1>'
            f'<div class="sub">{e(r["tagline"])}</div></div></header>')
    body = [f'<div class="meta"><span>Document A: <b>{e(r["document_a"]["name"])}</b> ({r["document_a"]["pages"]} p)</span>'
            f'<span>Document B: <b>{e(r["document_b"]["name"])}</b> ({r["document_b"]["pages"]} p)</span>'
            f'<span>Total Differences: <b>{s["total_differences"]}</b></span>'
            f'<span>Uncertain: <b>{s["uncertain_findings"]}</b></span></div><div class="sum">{pills}</div>'
            + (f'<div class="tag">{e(dpi_line(r))}</div>' if dpi_line(r) else "")]
    summary = r.get("text_comparison_summary") or {}
    body.append("<h2>Text comparison summary</h2><div class=\"sum\">")
    body += [f'<span class="pill">{n}. {e(name)}: {summary.get(name, 0)}</span>'
             for n, name in enumerate(CHECKLIST_CATEGORIES, 1)]
    body.append("</div>")
    body.append("<h2>Differences</h2>")
    body += [_item_html(i, out_dir, embed) for i in r["differences"]] or ["<p>No confirmed differences were found.</p>"]
    if r["uncertain"]:
        body.append("<h2>Uncertain findings (verify manually)</h2>")
        body += [_item_html(i, out_dir, embed) for i in r["uncertain"]]
    if r.get("seen_only_at_higher_resolution"):
        body.append("<h2>Seen only at higher resolution (review manually)</h2>")
        body += [_item_html(i, out_dir, embed) + f'<p class="note">{e(i["details"].get("note", ""))}</p>'
                 for i in r["seen_only_at_higher_resolution"]]
    vu = r.get("verified_unchanged", [])
    if vu:
        rows = "".join(
            f'<tr><td>A{v["page_a"] or "-"} / B{v["page_b"] or "-"}</td>'
            + (f'<td>{v["matched_lines"]} / {max(v["lines_a"], v["lines_b"])}</td><td>{v["matched_words"]} / {max(v["words_a"], v["words_b"])}</td>'
               f'<td>{v["percent_identical"]:.1f}%</td>' if v["compared"] else f'<td colspan="3">not compared line by line ({e(v["kind"])} page)</td>')
            + "</tr>" for v in vu)
        body.append('<h2>Verified unchanged</h2><table class="vu"><thead><tr><th>Page</th><th>Matched lines</th>'
                    f'<th>Matched words</th><th>Identical</th></tr></thead><tbody>{rows}</tbody></table>')
    body.append(f'<p class="note">{e(r["disclaimer"])}<br>Generated {e(r["generated_utc"])} UTC · DiffSight {e(r["version"])}</p>')
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>DiffSight report</title><style>{_CSS}</style></head><body>' + head + '<main>' + "".join(body) + "</main></body></html>")


def save_embedded_html(report: dict, out_dir: Path) -> Path:
    p = out_dir / "report_embedded.html"
    p.write_text(to_html(report, out_dir, embed=True), encoding="utf-8")
    return p
