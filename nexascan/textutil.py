"""Shared text/layout helpers: reading order, tokenisation, OCR text normalisation."""
from __future__ import annotations

import re
import unicodedata
from typing import Callable, List, Sequence

# --------------------------------------------------------------------------- reading order


def reading_order(items: Sequence, bbox_of: Callable, glyph_h: float, _depth: int = 0) -> list:
    """Order layout items (lines) with a recursive XY-cut.

    The same function orders native and OCR lines so that A and B are always compared in the
    same sequence, independent of the producer's content-stream order."""
    items = list(items)
    if len(items) <= 1 or _depth > 40:
        return items
    gh = max(glyph_h, 4.0)
    boxes = [bbox_of(it) for it in items]

    # 1) horizontal cut (stacked blocks)
    order = sorted(range(len(items)), key=lambda i: (boxes[i][1], boxes[i][0]))
    run_max = boxes[order[0]][3]
    for k in range(1, len(order)):
        i = order[k]
        if boxes[i][1] > run_max + 0.35 * gh:
            top = [items[j] for j in order[:k]]
            bottom = [items[j] for j in order[k:]]
            return (reading_order(top, bbox_of, glyph_h, _depth + 1)
                    + reading_order(bottom, bbox_of, glyph_h, _depth + 1))
        run_max = max(run_max, boxes[i][3])

    # 2) vertical cut (columns / side-by-side cells)
    order = sorted(range(len(items)), key=lambda i: (boxes[i][0], boxes[i][1]))
    run_max = boxes[order[0]][2]
    for k in range(1, len(order)):
        i = order[k]
        if boxes[i][0] > run_max + 1.5 * gh:
            left = [items[j] for j in order[:k]]
            right = [items[j] for j in order[k:]]
            return (reading_order(left, bbox_of, glyph_h, _depth + 1)
                    + reading_order(right, bbox_of, glyph_h, _depth + 1))
        run_max = max(run_max, boxes[i][2])

    # 3) a single visual row (or overlapping mess): stable sort by vertical centre, group items whose centres
    #    differ by less than half a glyph height into one row (no fixed-grid bucketing, so a row can never be
    #    split at an arbitrary bucket boundary), then left to right inside each row
    cy = [(boxes[i][1] + boxes[i][3]) / 2.0 for i in range(len(items))]
    by_y = sorted(range(len(items)), key=lambda i: cy[i])          # sorted() is stable
    rows: list = []
    for i in by_y:
        if rows and abs(cy[i] - rows[-1][1] / len(rows[-1][0])) < 0.5 * gh:
            rows[-1][0].append(i)
            rows[-1][1] += cy[i]
        else:
            rows.append([[i], cy[i]])
    out: list = []
    for members, _ in rows:
        out += [items[i] for i in sorted(members, key=lambda i: boxes[i][0])]
    return out


# --------------------------------------------------------------------------- OCR normalisation


def normalize_ocr_text(s: str) -> str:
    """Canonicalise Unicode composition without changing punctuation or whitespace."""
    return unicodedata.normalize("NFC", s)


def normalize_native_text(s: str) -> str:
    """Native text keeps its punctuation exactly; only Unicode normal form and ligatures."""
    return unicodedata.normalize("NFC", s)


# --------------------------------------------------------------------------- tokenisation

# letters (unicode) | numbers with inner separators | any single other char (punctuation, symbols)
TOKEN_RE = re.compile(r"[^\W\d_]+|\d+(?:[.,]\d+)*|[^\s]", re.UNICODE)


def tokenize(s: str) -> List[str]:
    return TOKEN_RE.findall(s)


PUNCT_NAMES = {
    ".": "full stop", ",": "comma", ":": "colon", ";": "semicolon", "?": "question mark",
    "!": "exclamation mark", "(": "opening bracket", ")": "closing bracket",
    "[": "opening square bracket", "]": "closing square bracket", "{": "opening brace",
    "}": "closing brace", '"': "double quote", "'": "apostrophe / single quote",
    "‘": "left single quote", "’": "right single quote / apostrophe",
    "“": "left double quote", "”": "right double quote",
    "-": "hyphen", "–": "en dash", "—": "em dash", "/": "slash", "\\": "backslash",
    "_": "underscore", "@": "at sign", "#": "hash", "%": "percent sign", "&": "ampersand",
    "*": "asterisk", "+": "plus sign", "=": "equals sign", "<": "less-than sign",
    ">": "greater-than sign", "|": "vertical bar", "~": "tilde", "$": "dollar sign",
    "₹": "rupee sign", "£": "pound sign", "€": "euro sign", "•": "bullet",
    "…": "ellipsis", "`": "backtick", "^": "caret", "°": "degree sign",
}


def is_punct_token(t: str) -> bool:
    return len(t) == 1 and not t.isalnum()


def punct_name(t: str) -> str:
    return PUNCT_NAMES.get(t, f"symbol '{t}'")
