"""Service 6 – Text comparison engine.

Three levels over reading-ordered lines of one matched page pair:
  1. line level   (SequenceMatcher over whole lines; fast anchor)
  2. token level  (words, numbers and *each punctuation mark* are separate tokens)
  3. character level (inside replaced tokens, to describe single-character edits)
Equal text is additionally checked for font, position, alignment and spacing changes.

All boxes are returned in the *A frame* (B words are mapped through the alignment)."""
from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .align import Alignment, map_box
from .config import Config
from .models import BBox, Finding, Line, PageData, Word, bbox_union
from .visual_compare import VisualContext, local_move
from .textutil import is_punct_token, punct_name, tokenize
from .text_categories import classify_atomic_edit, whitespace_change


@dataclass
class Tok:
    text: str
    word: int                 # index into the word list
    bbox: BBox                # A-frame box of this token (char-interpolated for sub-word tokens)


def _char_span_boxes(word: Word, M: Optional[np.ndarray]) -> List[BBox]:
    """Per-character boxes of a word (native: exact; OCR: proportional split of the word box)."""
    n = len(word.text)
    if word.char_boxes and len(word.char_boxes) == n:
        boxes = list(word.char_boxes)
    else:
        x0, y0, x1, y1 = word.bbox
        # weight narrow glyphs (. , ' : ;  i l 1 |) less so punctuation boxes are tighter
        wts = [0.4 if c in ".,;:'`!|il1()[]" else (1.4 if c in "mwMW@" else 1.0) for c in word.text]
        tot = sum(wts)
        boxes, acc = [], 0.0
        for wt in wts:
            boxes.append((x0 + (x1 - x0) * acc / tot, y0, x0 + (x1 - x0) * (acc + wt) / tot, y1))
            acc += wt
    if M is not None:
        boxes = [map_box(M, b) for b in boxes]
    return boxes


def _tokens(words: List[Word], lines: List[Line], line_ids: List[int], M: Optional[np.ndarray]) -> List[Tok]:
    toks: List[Tok] = []
    for li in line_ids:
        for wi in lines[li].words:
            w = words[wi]
            boxes = _char_span_boxes(w, M)
            pos = 0
            for t in tokenize(w.text):
                k = w.text.find(t, pos)
                if k < 0:
                    k = pos
                pos = k + len(t)
                bb = bbox_union(boxes[k:k + len(t)]) or (map_box(M, w.bbox) if M is not None else w.bbox)
                toks.append(Tok(t, wi, bb))
    return toks


def _line_key(l: Line) -> str:
    return " ".join(l.text.split())


def _mapped_line_bbox(l: Line, M: Optional[np.ndarray]) -> BBox:
    return map_box(M, l.bbox) if M is not None else l.bbox


def _kind_of_pair(a: str, b: str) -> Tuple[str, str]:
    """Return (dtype, human description) for a one-token replacement."""
    if is_punct_token(a) and is_punct_token(b):
        return "Punctuation", f"{punct_name(a)} ({a}) changed to {punct_name(b)} ({b})"
    if a.replace(",", "").replace(".", "").isdigit() and b.replace(",", "").replace(".", "").isdigit():
        return "Number", f"number changed from {a} to {b}"
    if (a.isalpha() and re.fullmatch(r"\d+(?:[.,]\d+)*", b)
            or re.fullmatch(r"\d+(?:[.,]\d+)*", a) and b.isalpha()):
        return "Number", f"word-to-number or number-to-word change: '{a}' changed to '{b}'"
    if a.isalpha() and b.isalpha():
        sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
        if sm.ratio() >= 0.6:
            ops = [(op, a[i1:i2], b[j1:j2]) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal"]
            det = "; ".join(f"'{x}' -> '{y}'" if op == "replace" else (f"'{y}' inserted" if op == "insert" else f"'{x}' deleted")
                            for op, x, y in ops)
            return "Text", f"word changed from '{a}' to '{b}' (character change: {det})"
    return "Text", f"changed from '{a}' to '{b}'"


def _describe_block(a_toks: List[str], b_toks: List[str]) -> Tuple[str, str, str]:
    """(dtype, category, description) for one non-equal token block."""
    a_join, b_join = _join(a_toks), _join(b_toks)
    a_numbers = re.findall(r"\d+(?:[.,]\d+)*", a_join)
    b_numbers = re.findall(r"\d+(?:[.,]\d+)*", b_join)
    if len(a_numbers) == 1 and len(b_numbers) == 1 and a_numbers[0] != b_numbers[0]:
        return "Number", "Modified", f"number changed from {a_numbers[0]} to {b_numbers[0]}"
    if len(a_numbers) == len(b_numbers) and len(a_numbers) > 0:
        differing = [(x, y) for x, y in zip(a_numbers, b_numbers) if x != y]
        if len(differing) == 1:
            old, new = differing[0]
            return "Number", "Modified", f"number changed from {old} to {new}"
    unmatched_a = list(a_numbers)
    unmatched_b = list(b_numbers)
    for value in list(unmatched_a):
        if value in unmatched_b:
            unmatched_a.remove(value)
            unmatched_b.remove(value)
    if len(unmatched_a) == 1 and len(unmatched_b) == 1:
        return "Number", "Modified", f"number changed from {unmatched_a[0]} to {unmatched_b[0]}"
    if len(a_numbers) >= 1 and len(b_numbers) >= 1 and a_numbers[0] != b_numbers[0]:
        return "Number", "Modified", f"number changed from {a_numbers[0]} to {b_numbers[0]}"
    def numeric_value(value: str) -> Optional[float]:
        try:
            return float(value.replace(",", "."))
        except ValueError:
            return None
    if len(a_numbers) == 1 and len(b_numbers) >= 1:
        av = numeric_value(a_numbers[0])
        candidates = [(numeric_value(v), v) for v in b_numbers]
        candidates = [(dv, v) for dv, v in candidates if dv is not None]
        if av is not None and candidates:
            _, bv = min(candidates, key=lambda item: abs(item[0] - av))
            if bv != a_numbers[0]:
                return "Number", "Modified", f"number changed from {a_numbers[0]} to {bv}"
    if not a_toks and b_toks:
        if all(is_punct_token(t) for t in b_toks):
            names = ", ".join(f"{punct_name(t)} ({t})" for t in b_toks)
            return "Punctuation", "Added", f"{names} added"
        return "Text", "Added", f"text added: '{_join(b_toks)}'"
    if a_toks and not b_toks:
        if all(is_punct_token(t) for t in a_toks):
            names = ", ".join(f"{punct_name(t)} ({t})" for t in a_toks)
            return "Punctuation", "Removed", f"{names} removed"
        return "Text", "Removed", f"text removed: '{_join(a_toks)}'"
    if len(a_toks) == 1 and len(b_toks) == 1:
        dtype, desc = _kind_of_pair(a_toks[0], b_toks[0])
        return dtype, "Modified", desc
    if all(is_punct_token(t) for t in a_toks + b_toks):
        return "Punctuation", "Modified", f"'{''.join(a_toks)}' changed to '{''.join(b_toks)}'"
    if all(t.replace(",", "").replace(".", "").isdigit() or is_punct_token(t) for t in a_toks + b_toks) and \
            any(t[0].isdigit() for t in a_toks + b_toks):
        return "Number", "Modified", f"'{_join(a_toks)}' changed to '{_join(b_toks)}'"
    return "Text", "Modified", f"text changed from '{_join(a_toks)}' to '{_join(b_toks)}'"


def _join(tokens: List[str]) -> str:
    """Re-join tokens for display without inventing spaces before punctuation."""
    out = ""
    for t in tokens:
        if out and not is_punct_token(t) and not out[-1] in "(\"'[{/-":
            out += " "
        elif out and is_punct_token(t) and t in "([{\"'" and out[-1] not in "(\"'[{":
            out += " "
        out += t
    return out


def _context(toks: List[Tok], i1: int, i2: int, words: List[Word], n: int = 4) -> str:
    """Words around a change, as written in the document (not re-assembled from tokens)."""
    seen: List[int] = []
    for t in toks[max(0, i1 - n):min(len(toks), i2 + n)]:
        if t.word not in seen:
            seen.append(t.word)
    return " ".join(words[w].text for w in seen)


def _span_text(toks: List[Tok], idxs: List[int], words: List[Word]) -> str:
    """Changed tokens as text: whole words when a word is entirely changed, else just the changed part."""
    per_word: Dict[int, List[Tok]] = {}
    order: List[int] = []
    for k in idxs:
        w = toks[k].word
        if w not in per_word:
            per_word[w] = []
            order.append(w)
        per_word[w].append(toks[k])
    parts = []
    for w in order:
        ch = per_word[w]
        full = "".join(t.text for t in ch)
        parts.append(words[w].text if full == words[w].text.replace(" ", "") else full)
    return " ".join(parts)


def _equal_line_pairs(la: List[Line], lb: List[Line], ops) -> List[Tuple[int, int]]:
    pairs = []
    for op, i1, i2, j1, j2 in ops:
        if op == "equal":
            pairs += list(zip(range(i1, i2), range(j1, j2)))
    return pairs


LOCAL_MATCH_NCC = 0.80   # a moved line must correlate at least this well with its counterpart in the other image
_NEG = -1e9


def _tok_score(x: str, y: str) -> float:
    if x == y:
        return 1.0
    px, py = is_punct_token(x), is_punct_token(y)
    if px and py:
        return 0.55
    if px != py:
        return _NEG
    nx = bool(re.fullmatch(r"\d+(?:[.,]\d+)*", x))
    ny = bool(re.fullmatch(r"\d+(?:[.,]\d+)*", y))
    if nx and ny:
        return 0.85
    if (x.isalpha() and ny) or (nx and y.isalpha()):
        return 0.70
    r = difflib.SequenceMatcher(None, x, y, autojunk=False).ratio()
    return r if r >= 0.5 else _NEG


def _pair_tokens(a: List[str], b: List[str]) -> List[Tuple[Optional[int], Optional[int]]]:
    """Max-weight monotone pairing of two short token lists (similar tokens are paired)."""
    n, m = len(a), len(b)
    dp = np.zeros((n + 1, m + 1))
    bk = np.zeros((n + 1, m + 1), np.int8)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best, k = dp[i - 1, j], 1
            if dp[i, j - 1] > best:
                best, k = dp[i, j - 1], 2
            s = _tok_score(a[i - 1], b[j - 1])
            if s > 0 and dp[i - 1, j - 1] + s > best:
                best, k = dp[i - 1, j - 1] + s, 0
            dp[i, j], bk[i, j] = best, k
    out: List[Tuple[Optional[int], Optional[int]]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and bk[i, j] == 0:
            out.append((i - 1, j - 1)); i -= 1; j -= 1
        elif i > 0 and (j == 0 or bk[i, j] == 1):
            out.append((i - 1, None)); i -= 1
        else:
            out.append((None, j - 1)); j -= 1
    out.reverse()
    return out


def _token_changes(ta: List[Tok], tb: List[Tok]):
    """Yield (a_idx, b_idx, a_pos, b_pos): indices of changed tokens (each list may be empty) and the
    number of tokens consumed on each side before the change (for locating pure insertions)."""
    a, b = [t.text for t in ta], [t.text for t in tb]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for op, a1, a2, b1, b2 in sm.get_opcodes():
        if op == "equal":
            continue
        ai, bi = list(range(a1, a2)), list(range(b1, b2))
        if len(ai) * len(bi) == 0 or len(ai) > 60 or len(bi) > 60 or (len(ai) == 1 and len(bi) == 1):
            yield ai, bi, a1, b1
            continue
        # Adjacent replacement words that align by position form one phrase-level edit.
        # This avoids splitting "BASELINE VERSION" -> "REVISED EDITION" into two findings.
        if 2 <= len(ai) == len(bi) <= 12:
            a_words = [a[i] for i in ai]
            b_words = [b[j] for j in bi]
            if all(t.isalpha() for t in a_words + b_words):
                yield ai, bi, a1, b1
                continue
        pairs = _pair_tokens([a[i] for i in ai], [b[j] for j in bi])
        pend_a: List[int] = []
        pend_b: List[int] = []

        def flush(pa_, pb_):
            nonlocal pend_a, pend_b
            if pend_a or pend_b:
                yield_list.append((list(pend_a), list(pend_b), pa_, pb_))
            pend_a, pend_b = [], []

        yield_list: List[tuple] = []
        for pi, pj in pairs:
            if pi is not None and pj is not None:
                gi, gj = ai[pi], bi[pj]
                if a[gi] == b[gj]:
                    flush(gi, gj)
                else:
                    flush(gi, gj)
                    yield_list.append(([gi], [gj], gi, gj))
            elif pi is not None:
                pend_a.append(ai[pi])
            else:
                pend_b.append(bi[pj])
        flush(a2, b2)
        for item in yield_list:
            yield item


def _fuzzy_line_blocks(lines_a: List[Line], lines_b: List[Line], i1: int, i2: int, j1: int, j2: int):
    """Align the lines of a line-level replace block by similarity so that a changed line is
    diffed against its counterpart instead of against its neighbours."""
    A, B = list(range(i1, i2)), list(range(j1, j2))
    if len(A) <= 1 or len(B) <= 1 or len(A) * len(B) > 900:
        return [(A, B)]
    ta = [tokenize(lines_a[i].text) for i in A]
    tb = [tokenize(lines_b[j].text) for j in B]
    n, m = len(A), len(B)
    S = np.full((n, m), _NEG)
    for x in range(n):
        for y in range(m):
            r = difflib.SequenceMatcher(None, ta[x], tb[y], autojunk=False).ratio()
            if r >= 0.5:
                S[x, y] = r
    dp = np.zeros((n + 1, m + 1))
    bk = np.zeros((n + 1, m + 1), np.int8)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best, k = dp[i - 1, j], 1
            if dp[i, j - 1] > best:
                best, k = dp[i, j - 1], 2
            if S[i - 1, j - 1] > 0 and dp[i - 1, j - 1] + S[i - 1, j - 1] > best:
                best, k = dp[i - 1, j - 1] + S[i - 1, j - 1], 0
            dp[i, j], bk[i, j] = best, k
    seq = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and bk[i, j] == 0:
            seq.append(("pair", i - 1, j - 1)); i -= 1; j -= 1
        elif i > 0 and (j == 0 or bk[i, j] == 1):
            seq.append(("a", i - 1, None)); i -= 1
        else:
            seq.append(("b", None, j - 1)); j -= 1
    seq.reverse()
    blocks, pa_, pb_ = [], [], []
    for kind, x, y in seq:
        if kind == "pair":
            if pa_ or pb_:
                blocks.append((pa_, pb_)); pa_, pb_ = [], []
            blocks.append(([A[x]], [B[y]]))
        elif kind == "a":
            pa_.append(A[x])
        else:
            pb_.append(B[y])
    if pa_ or pb_:
        blocks.append((pa_, pb_))
    return blocks


def _tail_boxes(words: List[Word], lines: List[Line], toks: List[Tok], M: Optional[np.ndarray]) -> List[BBox]:
    """From the first changed token to the end of its line, per line (A frame). Content to the right
    of an insertion / deletion / replacement shifts, and that shift belongs to the same finding."""
    out: List[BBox] = []
    seen = set()
    for t in toks:
        ln = words[t.word].line
        if ln in seen:
            continue
        seen.add(ln)
        lb = lines[ln].bbox
        if M is not None:
            lb = map_box(M, lb)
        x0 = min(tt.bbox[0] for tt in toks if words[tt.word].line == ln)
        out.append((x0, lb[1], lb[2], lb[3]))
    return out


def _gap_box(toks: List[Tok], pos: int) -> Optional[BBox]:
    """A thin box where an insertion happens (right after the previous token, else before the next)."""
    if pos > 0:
        bx = toks[pos - 1].bbox
        return (bx[2], bx[1], bx[2] + 0.4 * (bx[3] - bx[1]), bx[3])
    if toks:
        bx = toks[0].bbox
        return (bx[0] - 2, bx[1], bx[0], bx[3])
    return None


def compare_text(page_a: PageData, page_b: PageData, words_a: List[Word], lines_a: List[Line],
                 words_b: List[Word], lines_b: List[Line], al: Optional[Alignment], cfg: Config,
                 pa_no: int, pb_no: int, ocr_mode: bool,
                 ctx: Optional[VisualContext] = None) -> Tuple[List[Finding], dict]:
    """Compare text of one matched page pair. ``ocr_mode`` means OCR text on both sides. ``ctx`` (pixels of the
    aligned pair) is used to confirm that OCR-reported line moves are real; without it they cannot be confirmed."""
    M = al.M if al is not None else None
    inv = al.inverse() if al is not None else None
    findings: List[Finding] = []
    info = {"lines_a": len(lines_a), "lines_b": len(lines_b), "suppressed_ocr_disagreement": 0,
            "suppressed_layout_unconfirmed": 0}
    touched_a: set = set()        # word indices (A) / (B) that take part in a reported difference
    touched_b: set = set()
    matched_lines = 0
    agree_words = 0               # words of lines on which two OCR readings agree (counted as identical)
    compared_a: set = set()       # words (A / B) of lines that were compared token by token
    keys_a = [_line_key(l) for l in lines_a]
    keys_b = [_line_key(l) for l in lines_b]
    ops = difflib.SequenceMatcher(None, keys_a, keys_b, autojunk=False).get_opcodes()
    fuzzy_pairs: List[Tuple[int, int]] = []

    def add_whitespace_finding(i: int, j: int) -> None:
        text_a = lines_a[i].text_exact or lines_a[i].text
        text_b = lines_b[j].text_exact or lines_b[j].text
        ws = whitespace_change(text_a, text_b)
        if not ws:
            return
        tags, edit_details = classify_atomic_edit(text_a, text_b, "Modified")
        ba = lines_a[i].bbox
        bb_af = _mapped_line_bbox(lines_b[j], M)
        region = bbox_union([ba, bb_af])
        bb_own = map_box(inv, bb_af) if inv is not None else bb_af
        if ocr_mode:
            confidences = [words_a[k].conf for k in lines_a[i].words] + [words_b[k].conf for k in lines_b[j].words]
            confidence = (min(confidences) if confidences else 80.0) / 100.0 * 0.9
        else:
            confidence = 0.98
        description = f"spacing changed from '{text_a}' to '{text_b}'"
        findings.append(Finding(
            category="Modified", dtype="Text", page_a=pa_no, page_b=pb_no,
            a_content=text_a, b_content=text_b, description=description,
            bbox_a=ba, bbox_b=bb_own, confidence=round(confidence, 3),
            engines=["text-ocr" if ocr_mode else "text-native"],
            checklist_categories=tags,
            detection_method="OCR whitespace comparison" if ocr_mode else "exact native-text whitespace comparison",
            character_change=edit_details.get("character_change", {}),
            whitespace_change=edit_details.get("whitespace_change", {}),
            details={"a_changed": text_a, "b_changed": text_b, "a_tokens": [text_a], "b_tokens": [text_b],
                     "ocr_only": ocr_mode, "checklist_categories": tags,
                     "region_a_frame": region, "explain_region": region, **edit_details},
            region=region,
        ))

    # Paragraph/block reordering is identified before ordinary token diffing. Matched lines that
    # appear at a different reading-order position are moved text, not newly added/deleted text.
    reorder_pairs: List[Tuple[int, int]] = []
    if not ocr_mode:
        matched_b_by_key: Dict[str, List[int]] = {}
        matched_a_counts: Dict[str, int] = {}
        for j, key in enumerate(keys_b):
            matched_b_by_key.setdefault(key, []).append(j)
        for i, key in enumerate(keys_a):
            matched_a_counts[key] = matched_a_counts.get(key, 0) + 1
        unique_pairs: List[Tuple[int, int]] = []
        used_b = set()
        for i, key in enumerate(keys_a):
            if not key:
                continue
            candidates = matched_b_by_key.get(key, [])
            if len(candidates) == 1 and matched_a_counts.get(key, 0) == 1 and candidates[0] not in used_b:
                j = candidates[0]
                if i != j:
                    unique_pairs.append((i, j))
                used_b.add(j)
        inverted = set()
        for p, (i1, j1) in enumerate(unique_pairs):
            for i2, j2 in unique_pairs[p + 1:]:
                if (i1 < i2 and j1 > j2) or (i1 > i2 and j1 < j2):
                    inverted.update(((i1, j1), (i2, j2)))
        reorder_pairs = sorted(inverted)
    reordered_a = {i for i, _ in reorder_pairs}
    reordered_b = {j for _, j in reorder_pairs}
    if reorder_pairs:
        ia = reorder_pairs[0][0]
        ib = reorder_pairs[0][1]
        originals = [keys_a[i] for i, _ in reorder_pairs]
        modified = [keys_b[j] for _, j in reorder_pairs]
        ba = bbox_union([lines_a[i].bbox for i, _ in reorder_pairs])
        bb_af = bbox_union([map_box(M, lines_b[j].bbox) if M is not None else lines_b[j].bbox
                            for _, j in reorder_pairs])
        bb_own = map_box(inv, bb_af) if inv is not None and bb_af is not None else bb_af
        moved = Finding(
            category="Moved", dtype="Paragraph", page_a=pa_no, page_b=pb_no,
            a_content="\n".join(originals), b_content="\n".join(modified),
            description=f"paragraph/block text reordered (A positions {[i + 1 for i, _ in reorder_pairs]}, "
                        f"B positions {[j + 1 for _, j in reorder_pairs]})",
            bbox_a=ba, bbox_b=bb_own, confidence=0.97,
            engines=["text-native" if not ocr_mode else "text-ocr"],
            checklist_categories=["Paragraph order changes"],
            detection_method="normalized paragraph fingerprint and reading-order sequence alignment",
            details={
                "original_positions": [i + 1 for i, _ in reorder_pairs],
                "modified_positions": [j + 1 for _, j in reorder_pairs],
                "checklist_categories": ["Paragraph order changes"],
                "region_a_frame": bbox_union([ba, bb_af]),
            },
            region=bbox_union([ba, bb_af]),
        )
        findings.append(moved)

    for op, i1, i2, j1, j2 in ops:
        if op == "equal":
            for i, j in zip(range(i1, i2), range(j1, j2)):
                if i not in reordered_a:
                    matched_lines += 1
                    add_whitespace_finding(i, j)
            continue
        for a_ids, b_ids in _fuzzy_line_blocks(lines_a, lines_b, i1, i2, j1, j2):
            if any(i in reordered_a for i in a_ids) or any(j in reordered_b for j in b_ids):
                continue
            if len(a_ids) == 1 and len(b_ids) == 1:
                fuzzy_pairs.append((a_ids[0], b_ids[0]))
            # OCR disagreement between engines is not a document difference
            if ocr_mode and len(a_ids) == 1 and len(b_ids) == 1:
                ra = set([lines_a[a_ids[0]].text] + lines_a[a_ids[0]].alt_texts)
                rb = set([lines_b[b_ids[0]].text] + lines_b[b_ids[0]].alt_texts)
                if ra & rb:
                    info["suppressed_ocr_disagreement"] += 1
                    matched_lines += 1      # both engines read the same text on one side: the line is unchanged
                    agree_words += len(lines_a[a_ids[0]].words)
                    continue
            ta = _tokens(words_a, lines_a, a_ids, None)
            tb = _tokens(words_b, lines_b, b_ids, M)
            for i_ in a_ids:
                compared_a.update(("a", w_) for w_ in lines_a[i_].words)
            for j_ in b_ids:
                compared_a.update(("b", w_) for w_ in lines_b[j_].words)
            for ai, bi, a_pos, b_pos in _token_changes(ta, tb):
                at, bt = [ta[k] for k in ai], [tb[k] for k in bi]
                touched_a.update(t.word for t in at)
                touched_b.update(t.word for t in bt)
                a_txt, b_txt = [t.text for t in at], [t.text for t in bt]
                dtype, cat, desc = _describe_block(a_txt, b_txt)
                if dtype == "Text" and cat in ("Added", "Removed"):
                    shown = _span_text(tb, bi, words_b) if cat == "Added" else _span_text(ta, ai, words_a)
                    desc = f"text {'added' if cat == 'Added' else 'removed'}: '{shown}'"
                ba = bbox_union([t.bbox for t in at]) if at else _gap_box(ta, a_pos)
                bb_af = bbox_union([t.bbox for t in bt]) if bt else _gap_box(tb, b_pos)
                if ba is None and bb_af is None:
                    continue
                region = bbox_union([ba, bb_af])
                bbox_b_own = map_box(inv, bb_af) if (bb_af is not None and inv is not None) else bb_af
                if ocr_mode:
                    cs = [words_a[t.word].conf for t in at] + [words_b[t.word].conf for t in bt]
                    base, ocr_only = (min(cs) if cs else 80.0) / 100.0 * 0.9, True
                else:
                    base, ocr_only = 0.98, False
                first = ai[0] if ai else a_pos
                firstb = bi[0] if bi else b_pos
                a_changed = _span_text(ta, ai, words_a)
                b_changed = _span_text(tb, bi, words_b)
                if dtype == "Text" and cat in ("Added", "Removed", "Modified"):
                    number_dtype, number_cat, number_desc = _describe_block(
                        [a_changed] if a_changed else [],
                        [b_changed] if b_changed else [],
                    )
                    if number_dtype == "Number":
                        dtype, cat, desc = number_dtype, "Modified", number_desc
                tags, edit_details = classify_atomic_edit(
                    a_changed,
                    b_changed,
                    cat,
                    sentence_or_block=(cat in ("Added", "Removed") and len((a_changed or b_changed).split()) >= 4),
                    line_break_changed=(len(a_ids) != len(b_ids)),
                    paragraph_break_changed=(bool(a_ids and b_ids) and
                                             lines_a[a_ids[0]].paragraph_id != lines_b[b_ids[0]].paragraph_id),
                )
                f = Finding(category=cat, dtype=dtype, page_a=pa_no, page_b=pb_no,
                            a_content=_context(ta, first, first + len(ai), words_a),
                            b_content=_context(tb, firstb, firstb + len(bi), words_b),
                            description=desc, bbox_a=ba, bbox_b=bbox_b_own, confidence=round(base, 3),
                            engines=["text-ocr" if ocr_mode else "text-native"],
                            checklist_categories=tags,
                            detection_method="native text/word/character sequence alignment" if not ocr_mode
                                             else "OCR word/character sequence alignment",
                            character_change=edit_details.get("character_change", {}),
                            whitespace_change=edit_details.get("whitespace_change", {}),
                            details={"a_tokens": a_txt, "b_tokens": b_txt, "a_changed": a_changed,
                                     "b_changed": b_changed, "ocr_only": ocr_only,
                                     "checklist_categories": tags, "region_a_frame": region, **edit_details},
                            region=region)
                ref_a = at or ([ta[a_pos - 1]] if a_pos > 0 and a_pos - 1 < len(ta) else ta[a_pos:a_pos + 1])
                ref_b = bt or ([tb[b_pos - 1]] if b_pos > 0 and b_pos - 1 < len(tb) else tb[b_pos:b_pos + 1])
                tails = _tail_boxes(words_a, lines_a, ref_a, None) + _tail_boxes(words_b, lines_b, ref_b, M)
                f.details["explain_region"] = bbox_union([region] + tails)
                findings.append(f)

    fmt: List[Finding] = []
    if cfg.check_layout or cfg.check_fonts:
        fmt = _format_differences(page_a, page_b, words_a, lines_a, words_b, lines_b, ops, al, cfg,
                                  pa_no, pb_no, ocr_mode, fuzzy_pairs, ctx, info)
    findings += fmt

    # Merge OCR/native line-side number substitutions that were emitted as separate Added/Removed
    # findings (for example "92 mg/dL 70 - 100" and "112 mg/dL 70 - 100").
    paired_number_ids = set()
    merged_numbers = []
    removed = [f for f in findings if f.category == "Removed" and f.dtype == "Text"]
    added = [f for f in findings if f.category == "Added" and f.dtype == "Text"]
    number_pattern = r"\d+(?:[.,]\d+)*"
    for r in removed:
        a_expr = r.details.get("a_changed") or r.a_content
        if not a_expr:
            continue
        a_nums = re.findall(number_pattern, a_expr)
        if not a_nums:
            continue
        for a in added:
            b_expr = a.details.get("b_changed") or a.b_content
            b_nums = re.findall(number_pattern, b_expr)
            if not b_nums:
                continue
            if re.sub(number_pattern, "<N>", a_expr) != re.sub(number_pattern, "<N>", b_expr):
                continue
            differing = [(x, y) for x, y in zip(a_nums, b_nums) if x != y]
            if len(differing) != 1:
                continue
            old, new = differing[0]
            paired_number_ids.update((id(r), id(a)))
            tags, edit_details = classify_atomic_edit(a_expr, b_expr, "Modified")
            merged_numbers.append(Finding(
                category="Modified", dtype="Number", page_a=pa_no, page_b=pb_no,
                a_content=a_expr, b_content=b_expr,
                description=f"number changed from {old} to {new}",
                bbox_a=r.bbox_a or a.bbox_a, bbox_b=a.bbox_b or r.bbox_b,
                confidence=round(min(r.confidence, a.confidence), 3),
                engines=sorted(set(r.engines + a.engines)),
                checklist_categories=tags,
                detection_method="exact expression number substitution",
                character_change=edit_details.get("character_change", {}),
                whitespace_change=edit_details.get("whitespace_change", {}),
                details={"a_changed": a_expr, "b_changed": b_expr,
                         "checklist_categories": tags, "region_a_frame":
                         bbox_union([r.region, a.region]), **edit_details},
                region=bbox_union([r.region, a.region]),
            ))
            break
    if paired_number_ids:
        findings = [f for f in findings if id(f) not in paired_number_ids]
        findings += merged_numbers

    # ---- "verified unchanged" summary: what was compared and found identical ----------------------
    n_a, n_b = len(words_a), len(words_b)
    changed_a = touched_a
    # words of compared (non-equal) lines that no difference touches are identical too
    rest_a = sum(1 for k, w in compared_a if k == "a" and w not in changed_a)
    equal_words_a = sum(len(lines_a[i].words) for op, i1, i2, j1, j2 in ops if op == "equal" for i in range(i1, i2))
    matched_words = equal_words_a + rest_a + agree_words
    matched_words = min(matched_words, n_a, n_b)
    total = max(n_a, n_b)
    info.update({"words_a": n_a, "words_b": n_b, "matched_lines": matched_lines, "matched_words": matched_words,
                 "identical_pct": round(100.0 * matched_words / total, 1) if total else 100.0})
    return findings, info


# ------------------------------------------------------------------------------------ format


def _fmt_sig(w: Word) -> Tuple[str, float, bool, bool, int]:
    base = w.font.split("+")[-1].lower()
    for suf in ("-bold", "-italic", "-oblique", "bold", "italic", "mt", "ps"):
        base = base.replace(suf, "")
    return (base.strip("-, "), round(w.size, 1), w.bold, w.italic, w.color)


def _fmt_text(w: Word) -> str:
    style = ("bold " if w.bold else "") + ("italic " if w.italic else "")
    return f"{w.font or 'font'} {w.size:.1f} pt {style}#{w.color:06x}".replace("  ", " ")


def _format_differences(page_a, page_b, words_a, lines_a, words_b, lines_b, ops, al, cfg, pa_no, pb_no, ocr_mode,
                        fuzzy_pairs=(), ctx=None, info=None):
    M = al.M if al is not None else None
    out: List[Finding] = []
    pairs = _equal_line_pairs(lines_a, lines_b, ops)
    tol_mm = cfg.layout_tol_mm_scan if ocr_mode else cfg.layout_tol_mm_native
    tol = cfg.mm(tol_mm)

    # -- font / size / style / colour (exact, native text only)
    if cfg.check_fonts and not ocr_mode:
        for ia, ib in pairs:
            wa = [words_a[k] for k in lines_a[ia].words]
            wb = [words_b[k] for k in lines_b[ib].words]
            if len(wa) != len(wb):
                continue
            run: List[int] = []
            for k, (x, y) in enumerate(zip(wa, wb)):
                sx, sy = _fmt_sig(x), _fmt_sig(y)
                diff = (sx[0] != sy[0] or abs(sx[1] - sy[1]) > cfg.font_size_tol_pt or sx[2] != sy[2]
                        or sx[3] != sy[3] or sx[4] != sy[4])
                if diff:
                    run.append(k)
                if (not diff or k == len(wa) - 1) and run:
                    xs, ys = wa[run[0]], wb[run[0]]
                    sa_, sb_ = _fmt_sig(xs), _fmt_sig(ys)
                    what = []
                    if sa_[0] != sb_[0]:
                        what.append("font family")
                    if abs(sa_[1] - sb_[1]) > cfg.font_size_tol_pt:
                        what.append("font size")
                    if sa_[2] != sb_[2] or sa_[3] != sb_[3]:
                        what.append("bold/italic style")
                    if sa_[4] != sb_[4]:
                        what.append("text colour")
                    ba = bbox_union([wa[k].bbox for k in run])
                    bb = bbox_union([map_box(M, wb[k].bbox) if M is not None else wb[k].bbox for k in run])
                    txt_a = " ".join(wa[k].text for k in run)
                    f = Finding(category="Modified", dtype="Font", page_a=pa_no, page_b=pb_no,
                                a_content=f"'{txt_a}' in {_fmt_text(xs)}", b_content=f"'{txt_a}' in {_fmt_text(ys)}",
                                description=f"{' and '.join(what)} changed for '{txt_a}'",
                                bbox_a=ba, bbox_b=map_box(al.inverse(), bb) if al is not None else bb,
                                confidence=0.97, engines=["format-native"],
                                details={"a_font": _fmt_text(xs), "b_font": _fmt_text(ys), "region_a_frame": bbox_union([ba, bb])},
                                region=bbox_union([ba, bb]))
                    out.append(f)
                    run = []
            # (a trailing run is flushed in the loop above)

    # -- geometry: group consecutive lines (identical *and* edited) that moved by the same offset
    all_pairs = sorted([(ia, ib, True) for ia, ib in pairs] + [(ia, ib, False) for ia, ib in fuzzy_pairs])
    if cfg.check_layout and all_pairs:
        shifts = []
        for ia, ib, eq in all_pairs:
            la = lines_a[ia].bbox
            lb = _mapped_line_bbox(lines_b[ib], M)
            shifts.append((ia, ib, lb[0] - la[0], lb[1] - la[1], lb[2] - la[2], lb[3] - la[3], eq))
        groups: List[List[tuple]] = []
        for s_ in shifts:
            moved = max(abs(s_[2]), abs(s_[3])) > tol or (s_[6] and max(abs(s_[4]), abs(s_[5])) > tol)
            if not moved:
                groups.append([])
                continue
            if groups and groups[-1] and abs(groups[-1][-1][2] - s_[2]) <= tol and abs(groups[-1][-1][3] - s_[3]) <= tol:
                groups[-1].append(s_)
            else:
                groups.append([s_])
        mm = cfg.dpi / 25.4
        for g in groups:
            if not g:
                continue
            eqs = [s_ for s_ in g if s_[6]]
            dx0 = float(np.median([s_[2] for s_ in g])) / mm
            dy0 = float(np.median([s_[3] for s_ in g])) / mm
            dx1 = float(np.median([s_[4] for s_ in eqs])) / mm if eqs else dx0
            dy1 = float(np.median([s_[5] for s_ in eqs])) / mm if eqs else dy0
            ba = bbox_union([lines_a[s_[0]].bbox for s_ in g])
            bb_af = bbox_union([_mapped_line_bbox(lines_b[s_[1]], M) for s_ in g])
            width_changed = abs(dx1 - dx0) > tol_mm
            if abs(dy0) > tol_mm and abs(dx0) <= tol_mm:
                what = f"moved {'down' if dy0 > 0 else 'up'} by {abs(dy0):.1f} mm"
                if len(g) == 1 and eqs and abs(dy1 - dy0) > tol_mm:
                    what = f"height changed (top {dy0:+.1f} mm, bottom {dy1:+.1f} mm)"
            elif abs(dx0) > tol_mm and abs(dy0) <= tol_mm:
                what = (f"moved {'right' if dx0 > 0 else 'left'} by {abs(dx0):.1f} mm" if not width_changed else
                        f"alignment / width changed (left edge {dx0:+.1f} mm, right edge {dx1:+.1f} mm)")
            else:
                what = f"moved by ({dx0:+.1f}, {dy0:+.1f}) mm"
            txt = " / ".join(lines_a[s_[0]].text for s_ in g[:2]) + (" ..." if len(g) > 2 else "")
            txt_b = " / ".join(lines_b[s_[1]].text for s_ in g[:2]) + (" ..." if len(g) > 2 else "")
            sev = max(abs(dx0), abs(dy0))
            extra = {}
            if ocr_mode:
                # OCR boxes jitter: a line counts as moved only if the pixels confirm that its content really sits
                # at a different place (local image match), by more than the tolerance. Otherwise: suppress.
                if ctx is None:
                    if info is not None:
                        info["suppressed_layout_unconfirmed"] += 1
                    continue
                bx = tuple(int(v) for v in ba)
                mdx, mdy, ncc = local_move(ctx, bx, cfg, search_mm=min(40.0, max(cfg.local_shift_mm, 2.0 * sev + 2.0)))
                real = float(np.hypot(mdx, mdy))
                if ncc < LOCAL_MATCH_NCC or real <= tol_mm:
                    if info is not None:
                        info["suppressed_layout_unconfirmed"] += 1
                    continue
                extra = {"local_ncc": round(ncc, 3), "local_shift_mm": [round(mdx, 2), round(mdy, 2)], "ocr_derived": True}
            conf = 0.9 if not ocr_mode else min(0.8, 0.45 + 0.1 * sev)
            if ocr_mode and sev < 2 * tol_mm:
                conf = min(conf, 0.6)
            f = Finding(category="Moved", dtype="Layout", page_a=pa_no, page_b=pb_no,
                        a_content=txt[:200], b_content=txt_b[:200],
                        description=f"{len(g)} line(s) {what}",
                        bbox_a=ba, bbox_b=map_box(al.inverse(), bb_af) if al is not None else bb_af,
                        confidence=round(conf, 3), engines=["layout-text"],
                        details={"lines": len(g), "dx_mm": round(dx0, 2), "dy_mm": round(dy0, 2),
                                 "region_a_frame": bbox_union([ba, bb_af]), **extra},
                        region=bbox_union([ba, bb_af]))
            out.append(f)
    return out
