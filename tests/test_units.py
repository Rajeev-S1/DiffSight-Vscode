import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nexascan.config import Config
from nexascan.textutil import tokenize, is_punct_token, punct_name
from nexascan.upload import UploadError, sniff_kind, safe_display_name, save_and_validate
from nexascan.text_categories import (
    CHECKLIST_CATEGORIES,
    classify_atomic_edit,
    normalized_category_counts,
)


def test_tokenize_splits_punctuation_individually():
    t = tokenize("Pay $1,200.50, now!")
    assert "now" in t and "!" in t
    assert sum(is_punct_token(x) for x in t) >= 2


def test_punct_name_known():
    assert "full stop" in punct_name(".").lower()


def test_sniff_kinds():
    assert sniff_kind(b"%PDF-1.7 ...") == "pdf"
    assert sniff_kind(b"\x89PNG\r\n\x1a\n" + b"0" * 8) == "png"
    assert sniff_kind(b"\xff\xd8\xff\xe0" + b"0" * 8) == "jpeg"
    assert sniff_kind(b"MZ\x90\x00") is None


def test_display_name_sanitised():
    n = safe_display_name("../../etc/passwd<script>.pdf")
    assert "/" not in n and "<" not in n


def test_upload_rejects_wrong_type(tmp_path):
    with pytest.raises(UploadError) as e:
        save_and_validate(io.BytesIO(b"MZ not a document"), "evil.pdf", tmp_path, "a", Config())
    assert str(tmp_path) not in str(e.value)


def test_upload_rejects_corrupt_pdf(tmp_path):
    with pytest.raises(UploadError):
        save_and_validate(io.BytesIO(b"%PDF-1.4 garbage"), "x.pdf", tmp_path, "a", Config())


# ------------------------------------------------------------------ additions: fixes and classify rules
import json
import time
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool

import numpy as np

from nexascan import classify, ingest, pagedata
from nexascan.models import Finding
from nexascan.textutil import reading_order
from nexascan.visual_compare import VisualContext
from nexascan.worker import JobDeleted, run_job, write_json_atomic


def _finding(dtype="Text", cat="Added", region=(0, 0, 100, 20), a="", b="", conf=0.98, **details):
    return Finding(category=cat, dtype=dtype, page_a=1, page_b=1, a_content=a, b_content=b, description=f"{cat} {dtype}",
                   bbox_a=region, bbox_b=region, confidence=conf, engines=["text-native"],
                   details={"a_changed": a, "b_changed": b, "region_a_frame": region, **details}, region=region)


def _ctx(changed_boxes=(), h=1000, w=1000):
    ch = np.zeros((h, w), bool)
    for x0, y0, x1, y1 in changed_boxes:
        ch[y0:y1, x0:x1] = True
    z = np.zeros((h, w), np.uint8)
    return VisualContext(h=h, w=w, bgr_a=z, bgr_b=z, norm_a=z, norm_b=z, ink_a=ch, ink_b=ch, valid=np.ones((h, w), bool),
                         scan=False, tol_px=2.0, stats={}, added=ch, removed=ch, changed=ch)


def test_reading_order_row_grouping_is_stable_across_bucket_boundaries():
    # three words of ONE row whose y jitters across what used to be a round(y/gh) bucket edge
    gh = 20.0
    boxes = {"c": (300, 29, 340, 49), "a": (10, 31, 50, 51), "b": (150, 30, 190, 50)}
    items = list(boxes)
    # force the single-row fallback (no XY cut possible: boxes overlap in y, gaps < 1.5 gh in x)
    out = reading_order(items, lambda k: boxes[k], gh)
    assert out == ["a", "b", "c"]
    # a lower row stays below
    boxes["d"] = (10, 80, 50, 100)
    assert reading_order(list(boxes), lambda k: boxes[k], gh)[-1] == "d"


def test_write_json_atomic_is_quiet_when_directory_is_gone(tmp_path):
    gone = tmp_path / "deleted"
    write_json_atomic(gone / "progress.json", {"a": 1})        # must not raise
    gone.mkdir()
    write_json_atomic(gone / "progress.json", {"a": 1})
    assert json.loads((gone / "progress.json").read_text()) == {"a": 1}


def test_run_job_on_deleted_job_returns_quietly(tmp_path):
    jd = tmp_path / "gone"          # never created: the first progress call finds no directory
    run_job(str(jd), "a.pdf", "b.pdf", {})                      # no exception, no files created
    assert not jd.exists()
    assert issubclass(JobDeleted, BaseException) and not issubclass(JobDeleted, Exception)


def test_run_job_releases_pdf_handles(tmp_path):
    ingest._DOCS["x"] = object()                                  # any leftover handle
    run_job(str(tmp_path / "nope"), "a", "b", {})
    assert ingest._DOCS == {}


def test_native_page_with_few_characters_and_no_large_image_stays_native():
    cfg = Config()
    assert ingest.is_scan_like({"n_chars": 3, "cover": 0.0}, cfg) is False        # short native page
    assert ingest.is_scan_like({"n_chars": 0, "cover": 0.1}, cfg) is False        # small logo only
    assert ingest.is_scan_like({"n_chars": 3, "cover": 0.5}, cfg) is True         # large image, no text: scan
    assert ingest.is_scan_like({"n_chars": 900, "cover": 0.95}, cfg) is True      # full-page image with text layer
    assert ingest.is_scan_like({"n_chars": 900, "cover": 0.1}, cfg) is False


def test_cache_purge_and_config_switch(tmp_path):
    sha = "ab" + "0" * 62
    d = tmp_path / sha[:2]
    d.mkdir()
    (d / f"{sha}_0_x.json").write_text("{}")
    (d / f"{sha}_1_x.json").write_text("{}")
    other = tmp_path / "cd" / ("cd" + "0" * 62 + "_0_x.json")      # another document: must survive
    other.parent.mkdir()
    other.write_text("{}")
    assert pagedata.purge_doc_cache(sha, str(tmp_path)) == 2
    assert not d.exists() and other.exists()
    assert pagedata.purge_doc_cache("../evil", str(tmp_path)) == 0
    assert Config().use_cache is True                              # default on; the API job runner passes False


def test_purge_old_cache_removes_only_old_files(tmp_path):
    old, new = tmp_path / "ab" / "old.json", tmp_path / "ab" / "new.json"
    old.parent.mkdir()
    old.write_text("{}")
    new.write_text("{}")
    t = time.time() - 7200
    import os
    os.utime(old, (t, t))
    assert pagedata.purge_old_cache(str(tmp_path), 3600) == 1
    assert new.exists() and not old.exists()


def test_api_done_callback_records_a_crashed_worker(tmp_path, monkeypatch):
    from nexascan import api
    jd = tmp_path / "job"
    jd.mkdir()
    write_json_atomic(jd / "progress.json", {"status": "running"})
    fut = Future()
    fut.set_exception(BrokenProcessPool("worker died"))
    pool = api._executor()
    api._on_job_done(jd, pool, fut)
    prog = json.loads((jd / "progress.json").read_text())
    assert prog["status"] == "error" and "crashed" in prog["message"]
    assert api._pool is None                                       # the broken pool was dropped, next submit recreates it
    # a job that already finished must not be overwritten
    write_json_atomic(jd / "progress.json", {"status": "done"})
    api._on_job_done(jd, api._executor(), fut)
    assert json.loads((jd / "progress.json").read_text())["status"] == "done"
    api._reset_pool(api._executor())


def test_api_pool_uses_spawn_and_recycles_workers():
    from nexascan import api
    pool = api._new_pool()
    try:
        assert pool._mp_context.get_start_method() == "spawn"
        assert pool._max_tasks_per_child == 1
    finally:
        pool.shutdown(wait=False)


def test_classify_pairs_short_keys_and_numbers_as_moved():
    ctx = _ctx([(0, 0, 100, 20), (500, 500, 600, 520)])
    rem = _finding("Number", "Removed", (0, 0, 100, 20), a="70 - 100")
    add = _finding("Number", "Added", (500, 500, 600, 520), b="70-100")
    out, _ = classify.classify_pair([rem, add], ctx, Config())
    assert [(f.category, f.dtype) for f in out] == [("Moved", "Number")]
    rem = _finding("Text", "Removed", (0, 0, 100, 20), a="Net")
    add = _finding("Text", "Added", (500, 500, 600, 520), b="net")
    out, _ = classify.classify_pair([rem, add], ctx, Config())
    assert [f.category for f in out] == ["Moved"]                  # 3-character keys are paired now (was >= 6)


def test_classify_identical_removed_added_without_pixel_change_is_ocr_disagreement():
    ctx = _ctx([])                                                 # pixels identical
    rem = _finding("Text", "Removed", (0, 0, 100, 20), a="Reference range")
    add = _finding("Text", "Added", (0, 30, 100, 50), b="reference  range")
    out, sup = classify.classify_pair([rem, add], ctx, Config())
    assert out == [] and sup[0]["kind"] == "ocr_disagreement"


def test_classify_block_inside_stamp_is_merged():
    stamp = _finding("Stamp", "Added", (100, 100, 300, 300))
    inside = _finding("Block", "Added", (120, 120, 200, 200))
    straddle = _finding("Block", "Added", (250, 250, 400, 400))   # < 70 % inside: stays
    out, _ = classify.classify_pair([stamp, inside, straddle], _ctx([(100, 100, 400, 400)]), Config())
    assert sorted(f.dtype for f in out) == ["Block", "Stamp"]
    assert next(f for f in out if f.dtype == "Stamp").details["merged_block"] is True


def test_classify_layout_row_on_the_lines_of_a_text_finding_is_merged():
    txt = _finding("Text", "Added", (200, 100, 260, 120), b="twice", explain_region=(200, 100, 500, 120))
    lay = Finding(category="Moved", dtype="Layout", page_a=1, page_b=1, a_content="", b_content="",
                  description="1 line(s) moved left by 9.9 mm", bbox_a=(150, 100, 500, 120), bbox_b=(150, 100, 500, 120),
                  confidence=0.9, engines=["layout-text"], details={"lines": 1, "dx_mm": -9.9, "dy_mm": 0.0},
                  region=(150, 100, 500, 120))
    far = Finding(category="Moved", dtype="Layout", page_a=1, page_b=1, a_content="", b_content="",
                  description="3 line(s) moved down by 4.0 mm", bbox_a=(100, 600, 500, 700), bbox_b=(100, 600, 500, 700),
                  confidence=0.9, engines=["layout-text"], details={"lines": 3}, region=(100, 600, 500, 700))
    out, _ = classify.classify_pair([txt, lay, far], _ctx([(150, 100, 500, 120), (100, 600, 500, 700)]), Config())
    assert sorted(f.dtype for f in out) == ["Layout", "Text"]       # only the unrelated layout row remains
    assert "shifted" in next(f for f in out if f.dtype == "Text").description


def test_classify_final_dedup_merges_same_type_overlapping_findings():
    a = _finding("Text", "Added", (0, 0, 100, 20), b="alpha beta", conf=0.9)
    b = _finding("Text", "Added", (10, 2, 100, 20), b="beta", conf=0.95)
    c = _finding("Text", "Added", (0, 200, 100, 220), b="gamma")
    d = _finding("Number", "Added", (0, 0, 100, 20), b="42")        # other type: not merged
    out, _ = classify.classify_pair([a, b, c, d], _ctx([(0, 0, 100, 20), (0, 200, 100, 220)]), Config())
    assert len(out) == 3
    assert max(f.details.get("merged_duplicates", 0) for f in out) == 1


def test_classify_without_visual_context_caps_ocr_only_findings():
    f = _finding("Text", "Modified", (0, 0, 100, 20), a="foo", b="fob", conf=0.88, ocr_only=True)
    out, _ = classify.classify_pair([f], None, Config())
    assert out[0].status == "uncertain" and out[0].confidence <= 0.6
    native = _finding("Text", "Modified", (0, 0, 100, 20), a="foo", b="fob", conf=0.98, ocr_only=False)
    out, _ = classify.classify_pair([native], None, Config())
    assert out[0].status == "confirmed"                             # exact native text needs no pixel cross-check


def test_checklist_classifies_character_spelling_and_casing_atomically():
    tags, details = classify_atomic_edit("sediment", "sedimant", "Modified")
    assert "Modified words or characters" in tags
    assert "Single-character changes" in tags
    assert "Spelling mistakes" in tags
    assert details["character_change"]["atomic"] is True
    assert details["spelling"]["incorrect_word"] == "sedimant"
    assert details["spelling"]["reference_word"] == "sediment"

    tags, _ = classify_atomic_edit("Vice", "vice", "Modified")
    assert "Uppercase and lowercase changes" in tags
    assert "Single-character changes" in tags


def test_checklist_classifies_punctuation_hyphen_whitespace_and_breaks():
    tags, details = classify_atomic_edit(".", ";", "Modified")
    assert "Punctuation changes" in tags
    assert details["punctuation"] == {"original": ".", "modified": ";"}

    tags, _ = classify_atomic_edit("mid-afternoon", "midafternoon", "Modified")
    assert "Hyphenation and abbreviations" in tags
    assert "Punctuation changes" not in tags

    tags, details = classify_atomic_edit("30 minutes", "30  minutes", "Modified")
    assert "Extra or missing spaces" in tags
    assert details["whitespace_change"]["modified_runs"][0]["count"] == 2

    tags, _ = classify_atomic_edit("Paragraph A", "Paragraph A\n\nParagraph B",
                                   "Added", sentence_or_block=True,
                                   line_break_changed=True, paragraph_break_changed=True)
    assert "Added sentences or paragraphs" in tags
    assert "Line breaks and paragraph breaks" in tags


def test_checklist_summary_has_all_categories_with_unique_finding_counts():
    f1 = {"checklist_categories": [
        "Modified words or characters",
        "Single-character changes",
        "Spelling mistakes",
    ]}
    f2 = {"checklist_categories": [
        "Punctuation changes",
        "Modified words or characters",
    ]}
    counts = normalized_category_counts([f1, f2])
    assert list(counts) == CHECKLIST_CATEGORIES
    assert counts["Modified words or characters"] == 2
    assert counts["Single-character changes"] == 1
    assert counts["Punctuation changes"] == 1
    assert counts["Paragraph order changes"] == 0


def test_multiword_replacement_is_one_token_change_block():
    from nexascan.text_compare import Tok, _token_changes
    ta = [Tok("BASELINE", 0, (0, 0, 20, 10)), Tok("VERSION", 1, (20, 0, 40, 10))]
    tb = [Tok("REVISED", 0, (0, 0, 20, 10)), Tok("EDITION", 1, (20, 0, 40, 10))]
    changes = list(_token_changes(ta, tb))
    assert len(changes) == 1
    assert changes[0][0] == [0, 1]
    assert changes[0][1] == [0, 1]


def _crash():
    import os
    os._exit(1)


def test_api_really_crashed_worker_is_reported_and_pool_recreated(tmp_path):
    """A worker that dies abruptly (OOM kill / segfault) must end as status 'error', and the next job must
    get a fresh pool. The callback runs inside the pool's management thread, so this also guards against
    calling pool.shutdown() on that thread (which deadlocks)."""
    from nexascan import api
    jd = tmp_path / "job"
    jd.mkdir()
    write_json_atomic(jd / "progress.json", {"status": "running"})
    old = api._executor()
    fut = old.submit(_crash)
    fut.add_done_callback(lambda f: api._on_job_done(jd, old, f))
    deadline = time.time() + 30
    status = None
    while time.time() < deadline:
        status = json.loads((jd / "progress.json").read_text())["status"]
        if status == "error":
            break
        time.sleep(0.2)
    assert status == "error"
    assert api._pool is None and api._executor() is not old
    api._reset_pool(api._executor())
