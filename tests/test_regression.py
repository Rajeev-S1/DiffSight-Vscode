"""Regression tests for Tests 01-04 (document pairs built by tests/make_master.py).

Each test runs the full pipeline on a pair of digital PDFs and checks the exact expected result."""
import sys
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from nexascan.config import Config  # noqa: E402
from nexascan.pipeline import compare_documents  # noqa: E402
import make_master  # noqa: E402


@pytest.fixture(scope="module")
def pairs(tmp_path_factory):
    return make_master.make_all(tmp_path_factory.mktemp("master"))


def _run(pairs, key, tmp_path_factory):
    a, b = pairs[key]
    out = tmp_path_factory.mktemp(f"out{key}")
    return compare_documents(str(a), a.name, str(b), b.name, out, Config(use_cache=False))


@pytest.fixture(scope="module")
def r01(pairs, tmp_path_factory):
    return _run(pairs, "01", tmp_path_factory)


@pytest.fixture(scope="module")
def r02(pairs, tmp_path_factory):
    return _run(pairs, "02", tmp_path_factory)


@pytest.fixture(scope="module")
def r03(pairs, tmp_path_factory):
    return _run(pairs, "03", tmp_path_factory)


@pytest.fixture(scope="module")
def r04(pairs, tmp_path_factory):
    return _run(pairs, "04", tmp_path_factory)


def _all_text(items):
    return " || ".join(f"{i['document_a']} {i['document_b']} {i['difference']}" for i in items)


# ------------------------------------------------------------------------------------------ Test 01
def test01_seven_content_changes_and_no_layout_rows(r01):
    diffs = r01["differences"]
    assert list(r01["text_comparison_summary"]) == [
        "Added sentences or paragraphs",
        "Deleted sentences or paragraphs",
        "Modified words or characters",
        "Single-character changes",
        "Spelling mistakes",
        "Punctuation changes",
        "Uppercase and lowercase changes",
        "Extra or missing spaces",
        "Line breaks and paragraph breaks",
        "Hyphenation and abbreviations",
        "Paragraph order changes",
    ]
    assert len(diffs) == 7, [i["difference"] for i in diffs]
    assert not r01["uncertain"]
    assert [i for i in diffs if i["dtype"] in ("Layout", "Visual")] == []     # 0 extra layout rows
    text = _all_text(diffs)
    for needle in ("10,000", "12,500", "thirty", "sixty", "Logistics", "Freight", "twice",
                   "The buyer confirms receipt", "Delivery is subject to inspection"):
        assert needle in text, needle
    twice = next(i for i in diffs if "twice" in i["difference"])
    assert "shifted" in twice["difference"]            # the layout shift was merged into this finding


# ------------------------------------------------------------------------------------------ Test 02
def test02_three_changes_no_reference_range_pair(r02):
    diffs = r02["differences"]
    assert len(diffs) == 3, [i["difference"] for i in diffs]
    assert not r02["uncertain"]
    text = _all_text(diffs)
    for old, new in (("92", "118"), ("181", "214"), ("0.9", "1.4")):
        assert old in text and new in text
    for rng in make_master.RANGES:
        assert rng not in text, f"reference range {rng!r} must not be reported"
    assert [i for i in diffs if i["category"] in ("Removed", "Added")] == []


# ------------------------------------------------------------------------------------------ Test 03
def test03_one_stamp_and_one_header_colour(r03):
    diffs = r03["differences"]
    assert len(diffs) == 2, [(i["dtype"], i["difference"]) for i in diffs]
    stamps = [i for i in diffs if i["dtype"] == "Stamp"]
    colours = [i for i in diffs if i["dtype"] == "Colour"]
    assert len(stamps) == 1 and stamps[0]["category"] == "Added"
    assert len(colours) == 1
    assert colours[0]["confidence"] >= 0.7             # a large header recolour is a confident finding
    assert not r03["uncertain"]


# ------------------------------------------------------------------------------------------ Test 04
def test04_agreement_pair(r04):
    assert (r04["document_a"]["pages"], r04["document_b"]["pages"]) == (2, 3)
    pairs = r04["summary"]["page_pairs"]
    assert {"a": None, "b": 3, "kind": "added"}.items() <= next(p for p in pairs if p["b"] == 3).items()
    assert all(p["a"] is None for p in pairs if p["b"] == 3)         # APPENDIX A is not matched to an existing page
    diffs = r04["differences"]
    pages = [i for i in diffs if i["dtype"] == "Page"]
    assert len(pages) == 1 and pages[0]["category"] == "Added" and pages[0]["page_b"] == 3   # reported once
    text = _all_text(diffs)
    for needle in ("l1-desk@orbitnet.example", "l2-duty@orbitnet.example", "l3-account@orbitnet.example",
                   "Appendix A", "Service Manager", "Operations Manager", "Rs. 1,500"):
        assert needle in text, needle
    escal = [i for i in diffs if i["category"] == "Added" and "@orbitnet.example" in i["difference"]]
    assert len(escal) == 3
    colour = [i for i in diffs if i["dtype"] == "Colour"]
    assert len(colour) == 1 and colour[0]["confidence"] >= 0.7
    assert len(diffs) == 8 and not r04["uncertain"]
    for value in make_master.UNCHANGED_04:                            # no flags on unchanged fields
        assert value not in text, f"unchanged field {value!r} was flagged"


def test04_verified_unchanged_summary(r04, tmp_path):
    rows = r04["verified_unchanged"]
    assert [(r["page_a"], r["page_b"]) for r in rows] == [(1, 1), (2, 2), (None, 3)]
    p1 = rows[0]
    assert p1["compared"] and p1["matched_lines"] > 0 and p1["matched_words"] > 0
    assert 0 < p1["percent_identical"] < 100
    assert rows[2]["compared"] is False
