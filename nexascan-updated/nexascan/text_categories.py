"""Stable DiffSight text-comparison checklist categories and atomic-edit classification."""
from __future__ import annotations

import difflib
import re
from typing import Iterable, List


CHECKLIST_CATEGORIES = [
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

PUNCTUATION = set(".,;:!?-()[]{}'\"/")
ABBREVIATIONS = {
    "hrs", "hr", "lab", "labs", "noc", "noc",
}

COMMON_WORDS = {
    "maintenance", "sediment", "laboratory", "thirteen", "fourteen", "twenty",
    "president", "infrastructure", "below", "average", "network", "operations",
    "center", "centers", "ninety", "percent", "minutes", "data", "the",
    "and", "for", "from", "with", "about", "before", "after", "while",
    "sentence", "paragraph", "document", "version", "revised", "baseline",
}


def normalized_category_counts(findings: Iterable[dict]) -> dict:
    counts = {name: 0 for name in CHECKLIST_CATEGORIES}
    for finding in findings:
        tags = finding.get("checklist_categories") or finding.get("details", {}).get("checklist_categories") or []
        for tag in set(tags):
            if tag in counts:
                counts[tag] += 1
    return counts


def is_single_character_change(a: str, b: str) -> bool:
    if not a or not b:
        return abs(len(a) - len(b)) == 1
    return abs(len(a) - len(b)) <= 1 and difflib.SequenceMatcher(
        None, a, b, autojunk=False
    ).get_opcodes()[0][0] != "equal"


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
    if len(re.findall(r" ", a)) == len(re.findall(r" ", b)) and re.findall(r"\s+", a) == re.findall(r"\s+", b):
        return {}
    a_runs = [(m.group(0), m.start()) for m in re.finditer(r"\s+", a)]
    b_runs = [(m.group(0), m.start()) for m in re.finditer(r"\s+", b)]
    return {
        "original_runs": [{"text": v, "position": p, "count": len(v)} for v, p in a_runs],
        "modified_runs": [{"text": v, "position": p, "count": len(v)} for v, p in b_runs],
    }


def spelling_candidate(a: str, b: str) -> dict:
    pairs = []
    for old in re.findall(r"[A-Za-z][A-Za-z'-]*", a):
        for new in re.findall(r"[A-Za-z][A-Za-z'-]*", b):
            if old.lower() == new.lower():
                continue
            if old.lower() in ABBREVIATIONS or new.lower() in ABBREVIATIONS:
                continue
            if old.lower() in COMMON_WORDS or new.lower() in COMMON_WORDS:
                if old.isalpha() and new.isalpha():
                    dist = difflib.SequenceMatcher(None, old.lower(), new.lower(), autojunk=False).ratio()
                    if dist >= 0.72:
                        pairs.append((old, new, dist))
    if not pairs:
        return {}
    pairs.sort(key=lambda item: -item[2])
    old, new, dist = pairs[0]
    return {"incorrect_word": new, "reference_word": old, "confidence": round(dist, 3)}


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

    if operation == "Modified" or original != modified:
        if original and modified and (
            re.search(r"[A-Za-z]", original) or re.search(r"[A-Za-z]", modified)
        ):
            tags.append("Modified words or characters")

    cc = character_change(original, modified)
    operations = cc["operations"]
    changed_chars = sum(max(len(op["original"]), len(op["modified"])) for op in operations)
    if changed_chars == 1 and len(operations) == 1:
        tags.append("Single-character changes")
        details["character_change"] = cc

    spelling = spelling_candidate(original, modified)
    if spelling:
        tags.append("Spelling mistakes")
        details["spelling"] = spelling

    punct_a = sorted(set(ch for ch in original if ch in PUNCTUATION))
    punct_b = sorted(set(ch for ch in modified if ch in PUNCTUATION))
    hyphen_related = "-" in punct_a or "-" in punct_b
    if punct_a != punct_b:
        if not hyphen_related:
            tags.append("Punctuation changes")
        details["punctuation"] = {"original": "".join(punct_a), "modified": "".join(punct_b)}

    if original != modified and original.isalpha() and modified.isalpha() and original.lower() == modified.lower():
        tags.append("Uppercase and lowercase changes")
    elif re.search(r"[A-Z]", original) and re.search(r"[a-z]", modified):
        tags.append("Uppercase and lowercase changes")

    ws = whitespace_change(original, modified)
    if ws:
        tags.append("Extra or missing spaces")
        details["whitespace_change"] = ws

    if line_break_changed or paragraph_break_changed:
        tags.append("Line breaks and paragraph breaks")
        details["line_break_changed"] = line_break_changed
        details["paragraph_break_changed"] = paragraph_break_changed

    old_hyphen = "-" in original
    new_hyphen = "-" in modified
    old_words = re.findall(r"[A-Za-z]+", original)
    new_words = re.findall(r"[A-Za-z]+", modified)
    abbreviation_change = (
        len(old_words) == 1 and len(new_words) >= 2
        or len(new_words) == 1 and len(old_words) >= 2
        or any(w.lower() in ABBREVIATIONS for w in old_words + new_words)
    )
    if old_hyphen != new_hyphen or (
        old_hyphen and new_hyphen and original.replace("-", "").lower() == modified.replace("-", "").lower()
    ) or abbreviation_change:
        tags.append("Hyphenation and abbreviations")

    return list(dict.fromkeys(tags)), details
