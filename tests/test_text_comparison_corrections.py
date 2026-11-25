"""Regression coverage for comparison-only text classifications."""
from pathlib import Path

import pymupdf

from nexascan.config import Config
from nexascan.ingest import close_all
from nexascan.pipeline import compare_documents


def _write_pdf(path: Path, text: str) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), text, fontsize=12)
    doc.save(str(path))
    doc.close()


def _compare(tmp_path, original: str, modified: str) -> dict:
    path_a, path_b = tmp_path / "original.pdf", tmp_path / "modified.pdf"
    _write_pdf(path_a, original)
    _write_pdf(path_b, modified)
    try:
        return compare_documents(
            str(path_a), path_a.name, str(path_b), path_b.name,
            tmp_path / "report", Config(use_cache=False),
        )
    finally:
        close_all()


def test_spelling_like_substitution_is_only_a_direct_modified_word(tmp_path):
    report = _compare(tmp_path, "Recieve payment", "Receive payment")
    findings = report["differences"] + report["uncertain"]

    assert len(findings) == 1
    finding = findings[0]
    assert finding["category"] == "Modified"
    assert finding["document_a"] == "Recieve payment"
    assert finding["document_b"] == "Receive payment"
    assert "Modified words or characters" in finding["checklist_categories"]
    assert "Spelling mistakes" not in finding["checklist_categories"]
    assert "Single-character changes" not in finding["checklist_categories"]
    assert "Uppercase and lowercase changes" not in finding["checklist_categories"]
    assert "spelling" not in finding["details"]


def test_pure_spacing_change_is_reported_once_with_exact_text(tmp_path):
    report = _compare(tmp_path, "Alpha beta", "Alpha  beta")
    findings = report["differences"] + report["uncertain"]

    assert len(findings) == 1
    finding = findings[0]
    assert finding["dtype"] == "Text"
    assert finding["document_a"] == "Alpha beta"
    assert finding["document_b"] == "Alpha  beta"
    assert finding["checklist_categories"] == ["Extra or missing spaces"]
    assert finding["whitespace_change"]["modified_runs"][0]["count"] == 2


def test_word_and_punctuation_edits_are_not_cross_classified(tmp_path):
    report = _compare(tmp_path, "Recieve payment.", "Receive payment!")
    findings = report["differences"] + report["uncertain"]

    assert len(findings) == 2
    word_change = next(item for item in findings if item["dtype"] == "Text")
    punctuation_change = next(item for item in findings if item["dtype"] == "Punctuation")
    assert word_change["checklist_categories"] == ["Modified words or characters"]
    assert punctuation_change["checklist_categories"] == ["Punctuation changes"]
