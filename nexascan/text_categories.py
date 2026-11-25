"""Stable DiffSight text-comparison checklist categories and atomic-edit classification."""
from __future__ import annotations

import difflib
import re
from typing import Iterable, List

from .textutil import is_punct_token


CHECKLIST_CATEGORIES = [
    "Added sentences or paragraphs",
    "Deleted sentences or paragraphs",
    "Modified words or characters",
    "Punctuation changes",
    "Uppercase and lowercase changes",
    "Extra or missing spaces",
    "Line breaks and paragraph breaks",
    "Hyphenation and abbreviations",
    "Paragraph order changes",
]


def normalized_category_counts(findings: Iterable[dict]) -> dict:
    counts = {name: 0 for name in CHECKLIST_CATEGORIES}
    for finding in findings:
        tags = finding.get("checklist_categories") or finding.get("details", {}).get("checklist_categories") or []
        for tag in set(tags):
            if tag in counts:
                counts[tag] += 1
    return counts


def character_change(a: str, b: str) -> dict:
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    operations = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        operations.append({
            "operation": op,
            "original": a[i1:i2],
            "modified": b[j1:j2],
            "original_position": i1 + 1,
            "modified_position": j1 + 1,
        })
    return {"operations": operations, "atomic": len(operations) == 1}


def whitespace_change(a: str, b: str) -> dict:
    if re.sub(r"\s+", "", a) != re.sub(r"\s+", "", b):
        return {}
    a_runs = [(m.group(0), m.start()) for m in re.finditer(r"\s+", a)]
    b_runs = [(m.group(0), m.start()) for m in re.finditer(r"\s+", b)]
    if a_runs == b_runs:
        return {}
    return {
        "original_runs": [{"text": v, "position": p, "count": len(v)} for v, p in a_runs],
        "modified_runs": [{"text": v, "position": p, "count": len(v)} for v, p in b_runs],
    }


def classify_atomic_edit(original: str, modified: str, operation: str = "Modified",
                         sentence_or_block: bool = False,
                         line_break_changed: bool = False,
                         paragraph_break_changed: bool = False) -> tuple[List[str], dict]:
    tags: List[str] = []
    details: dict = {}
    if operation == "Added" and (sentence_or_block or len(modified.split()) >= 4):
        tags.append("Added sentences or paragraphs")
    if operation == "Removed" and (sentence_or_block or len(original.split()) >= 4):
        tags.append("Deleted sentences or paragraphs")

    content_a = "".join(ch for ch in original if not ch.isspace() and not is_punct_token(ch))
    content_b = "".join(ch for ch in modified if not ch.isspace() and not is_punct_token(ch))
    if content_a != content_b and original and modified and any(ch.isalpha() for ch in original + modified):
        tags.append("Modified words or characters")

    cc = character_change(original, modified)
    if cc["operations"]:
        details["character_change"] = cc

    punct_a = [ch for ch in original if not ch.isspace() and is_punct_token(ch)]
    punct_b = [ch for ch in modified if not ch.isspace() and is_punct_token(ch)]
    hyphenation_change = (
        original != modified
        and "-" in original + modified
        and original.replace("-", "") == modified.replace("-", "")
    )
    if punct_a != punct_b:
        if not hyphenation_change:
            tags.append("Punctuation changes")
        details["punctuation"] = {"original": "".join(punct_a), "modified": "".join(punct_b)}

    case_only_change = (
        len(original) == len(modified)
        and original != modified
        and all(a == b or (a.isalpha() and b.isalpha() and a.lower() == b.lower())
                for a, b in zip(original, modified))
    )
    if case_only_change:
        tags.append("Uppercase and lowercase changes")

    ws = whitespace_change(original, modified)
    if ws:
        tags.append("Extra or missing spaces")
        details["whitespace_change"] = ws

    if line_break_changed or paragraph_break_changed:
        tags.append("Line breaks and paragraph breaks")
        details["line_break_changed"] = line_break_changed
        details["paragraph_break_changed"] = paragraph_break_changed

    if hyphenation_change:
        tags.append("Hyphenation and abbreviations")

    return list(dict.fromkeys(tags)), details
