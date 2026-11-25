"""Builds the benchmark suite: document pairs with *known* injected differences (ground truth).

Every pair is generated procedurally. Ground-truth boxes are computed from the exact text
positions / image rectangles used when drawing, so nothing is annotated by hand or by NexaScan.

Usage:  python tests/make_suite.py [out_dir]
Writes  out_dir/<case>/A.<ext>, B.<ext> and out_dir/<case>/truth.json
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import samples as S  # noqa: E402

PT2PX = 300.0 / 72.0
A4 = (595.0, 842.0)
LETTER = (612.0, 792.0)

BASE_LINES = [
    "This agreement is made on 12-11-2026 between Alpha Traders Pvt Ltd and Beta Logistics.",
    "Payment amount is Rs. 10,000. Approved.",
    "The supplier shall deliver the goods within thirty days of the order date.",
    "Invoice No. INV-2026/0045 dated 05-11-2026 for a total of 1,25,000.50 is attached.",
    "All disputes are subject to the jurisdiction of the courts at Bengaluru.",
    "Contact: accounts@alpha-traders.example; phone +91 98765 43210.",
    "Terms: payment within 15 days (net), late fee applies, no refunds after delivery.",
    "The buyer confirms receipt of the goods in good condition.",
    "Authorised signatory for Alpha Traders Pvt Ltd:",
]
LINE_Y0, LINE_DY = 130.0, 26.0
TABLE = {"x": 72.0, "y": 440.0, "cw": [150.0, 110.0, 110.0, 80.0], "rh": 24.0,
         "rows": [["Item", "Quantity", "Rate", "Total"], ["Packaging", "500", "12.00", "6,000"],
                  ["Labels", "1,200", "2.50", "3,000"], ["Pallets", "40", "85.00", "3,400"]]}


# ------------------------------------------------------------------------------ page model


def base_page(lines: Optional[List[str]] = None, header="NexaScan Demo Corp - Confidential", footer="Page 1 of 3",
              title="Master Services Agreement", size=A4, table=True) -> dict:
    items: List[dict] = []
    items.append(dict(kind="text", x=72, y=44, s=header, size=9, font="helv", id="header"))
    items.append(dict(kind="text", x=72, y=96, s=title, size=16, font="hebo", id="title"))
    for i, t in enumerate(lines if lines is not None else BASE_LINES):
        items.append(dict(kind="text", x=72, y=LINE_Y0 + i * LINE_DY, s=t, size=11, font="helv", id=f"L{i}"))
    items.append(dict(kind="text", x=72, y=size[1] - 40, s=footer, size=9, font="helv", id="footer"))
    page = {"size": size, "items": items}
    if table:
        add_table(page, TABLE)
    return page


def add_table(page: dict, t: dict) -> None:
    x, y, rh = t["x"], t["y"], t["rh"]
    cw = t["cw"]
    nrows = len(t["rows"])
    W = sum(cw)
    for r in range(nrows + 1):
        page["items"].append(dict(kind="line", x0=x, y0=y + r * rh, x1=x + W, y1=y + r * rh, id=f"tr{r}"))
    xs = [x]
    for w in cw:
        xs.append(xs[-1] + w)
    for xx in xs:
        page["items"].append(dict(kind="line", x0=xx, y0=y, x1=xx, y1=y + nrows * rh, id=f"tc{xx:.0f}"))
    for r, row in enumerate(t["rows"]):
        for c, cell in enumerate(row):
            page["items"].append(dict(kind="text", x=xs[c] + 6, y=y + r * rh + 16, s=cell, size=10,
                                      font="hebo" if r == 0 else "helv", id=f"cell{r}{c}"))


def find(page: dict, iid: str) -> dict:
    for it in page["items"]:
        if it.get("id") == iid:
            return it
    raise KeyError(iid)


def tl(s: str, font: str, size: float) -> float:
    return pymupdf.get_text_length(s, fontname=font, fontsize=size)


def sub_box(it: dict, sub: str, occurrence: int = 0) -> List[float]:
    """Pixel box (300 DPI) of a substring of a text item."""
    s, size, font = it["s"], it["size"], it["font"]
    k = -1
    for _ in range(occurrence + 1):
        k = s.index(sub, k + 1)
    x0 = it["x"] + tl(s[:k], font, size)
    x1 = x0 + tl(sub, font, size)
    return [round(v * PT2PX, 1) for v in (x0 - 0.5, it["y"] - size * 0.95, x1 + 0.5, it["y"] + size * 0.28)]


def item_box(it: dict) -> List[float]:
    return sub_box(it, it["s"])


def rect_px(x, y, w, h) -> List[float]:
    return [round(v * PT2PX, 1) for v in (x, y, x + w, y + h)]


def png(rgba: np.ndarray) -> bytes:
    ok, enc = cv2.imencode(".png", rgba[..., [2, 1, 0, 3]])
    return enc.tobytes()


def render_pdf(path: Path, pages: List[dict]) -> None:
    doc = pymupdf.open()
    for pg in pages:
        page = doc.new_page(width=pg["size"][0], height=pg["size"][1])
        for it in pg["items"]:
            k = it["kind"]
            if k == "text":
                col = it.get("color", (0, 0, 0))
                page.insert_text((it["x"], it["y"]), it["s"], fontsize=it["size"], fontname=it["font"], color=col)
            elif k == "line":
                page.draw_line((it["x0"], it["y0"]), (it["x1"], it["y1"]), color=it.get("color", (0, 0, 0)), width=it.get("w", 0.8))
            elif k == "rect":
                page.draw_rect(pymupdf.Rect(it["x"], it["y"], it["x"] + it["w"], it["y"] + it["h"]),
                               color=it.get("stroke"), fill=it.get("fill"), width=it.get("w_line", 0.8))
            elif k == "image":
                page.insert_image(pymupdf.Rect(it["x"], it["y"], it["x"] + it["w"], it["y"] + it["h"]), stream=it["data"])
            elif k == "dot":
                page.draw_circle((it["x"], it["y"]), it["r"], color=(0, 0, 0), fill=(0, 0, 0))
        if pg.get("rotation"):
            page.set_rotation(pg["rotation"])
    doc.save(str(path), deflate=True)
    doc.close()


def T(page_a=1, page_b=1, bbox_a=None, bbox_b=None, kinds=(), category="Modified", optional=False, desc=""):
    return {"page_a": page_a, "page_b": page_b, "bbox_a": bbox_a, "bbox_b": bbox_b, "kinds": list(kinds),
            "category": category, "optional": optional, "desc": desc}


TEXT_K = ["Text", "Number", "Punctuation"]
STAMP_IMG = {
    "blue": S.stamp_png("APPROVED", (30, 60, 200), 330, 0.0, seed=1),
    "red": S.stamp_png("PAID", (200, 30, 40), 300, 0.0, seed=2),
    "red_rot": S.stamp_png("PAID", (200, 30, 40), 300, 30.0, seed=2),
    "green": S.stamp_png("VERIFIED", (20, 130, 60), 300, 0.0, seed=5),
    "blue_rej": S.stamp_png("REJECTED", (30, 60, 200), 330, 0.0, seed=6),
}
SIG1 = S.signature_png(seed=3, w=520, h=170)
SIG2 = S.signature_png(seed=9, w=520, h=170)


def stamp_item(kind, x, y, size=72):
    return dict(kind="image", x=x, y=y, w=size, h=size, data=png(STAMP_IMG[kind]), id=f"stamp_{kind}")


def sig_item(which, x, y, w=125, h=41):
    return dict(kind="image", x=x, y=y, w=w, h=h, data=png(SIG1 if which == 1 else SIG2), id=f"sig{which}")


# ------------------------------------------------------------------------------ cases (digital)


def case_identical():
    p = base_page()
    return [p], [copy.deepcopy(p)], []


def case_text_changes():
    a = base_page()
    b = copy.deepcopy(a)
    truth = []
    for lid, old, new in [("L1", "Rs. 10,000", "Rs. 15,000"), ("L2", "thirty", "forty"), ("L4", "Bengaluru", "Mumbai"),
                          ("L0", "Alpha Traders", "Gamma Traders"), ("L7", "receipt", "delivery")]:
        ia, ib = find(a, lid), find(b, lid)
        truth.append(T(bbox_a=sub_box(ia, old), bbox_b=sub_box(ib, new) if False else None, kinds=TEXT_K, desc=f"{old}->{new}"))
        ib["s"] = ib["s"].replace(old, new)
    return [a], [b], truth


def case_single_char():
    a = base_page()
    b = copy.deepcopy(a)
    truth = []
    for lid, old, new in [("L4", "Bengaluru", "Bengaluro"), ("L3", "0045", "0046"), ("L2", "deliver", "delivr"),
                          ("L7", "condition", "condltion"), ("L5", "98765", "98756"), ("L0", "Beta", "Bets")]:
        ia, ib = find(a, lid), find(b, lid)
        truth.append(T(bbox_a=sub_box(ia, old), kinds=TEXT_K, desc=f"{old}->{new}"))
        ib["s"] = ib["s"].replace(old, new)
    return [a], [b], truth


def case_punctuation():
    a = base_page()
    b = copy.deepcopy(a)
    truth = []
    edits = [("L1", "Approved.", "Approved,"),       # . -> ,
             ("L6", "(net),", "net,"),               # brackets removed
             ("L5", "example;", "example:"),         # ; -> :
             ("L7", "condition.", "condition!"),     # . -> !
             ("L3", "attached.", "attached"),        # full stop removed
             ("L2", "order date.", "order date?"),   # . -> ?
             ("L0", "Logistics.", "Logistics.."),    # extra full stop
             ("L5", "+91", "-91")]                   # symbol change
    for lid, old, new in edits:
        ia, ib = find(a, lid), find(b, lid)
        key = old if old != "Approved." else "."
        # truth box = the changed characters only (precise): use last char(s) of old
        if old == "Approved.":
            k = ia["s"].index("Approved.") + len("Approved")
            x0 = ia["x"] + tl(ia["s"][:k], ia["font"], ia["size"])
            box = [round(v * PT2PX, 1) for v in (x0 - 1, ia["y"] - 11, x0 + 4, ia["y"] + 3)]
        else:
            box = sub_box(ia, old)
        truth.append(T(bbox_a=box, kinds=["Punctuation", "Text"], desc=f"{old}->{new}"))
        ib["s"] = ib["s"].replace(old, new)
    return [a], [b], truth


def _stamp_pages(kind):
    a = base_page()
    b = copy.deepcopy(a)
    truth = []
    if kind == "missing":
        a["items"].append(stamp_item("blue", 400, 640))
        truth.append(T(bbox_a=rect_px(400, 640, 72, 72), kinds=["Stamp"], category="Removed", desc="stamp missing"))
    elif kind == "additional":
        b["items"].append(stamp_item("blue", 400, 640))
        truth.append(T(bbox_a=rect_px(400, 640, 72, 72), bbox_b=rect_px(400, 640, 72, 72), kinds=["Stamp"], category="Added", desc="stamp added"))
    elif kind == "changed":
        a["items"].append(stamp_item("blue", 400, 640))
        b["items"].append(stamp_item("blue_rej", 400, 640))
        truth.append(T(bbox_a=rect_px(400, 640, 72, 72), kinds=["Stamp"], desc="stamp text changed"))
    elif kind == "position":
        a["items"].append(stamp_item("green", 400, 640))
        b["items"].append(stamp_item("green", 300, 700))
        truth.append(T(bbox_a=rect_px(400, 640, 72, 72), bbox_b=rect_px(300, 700, 72, 72), kinds=["Stamp"], category="Moved", desc="stamp moved"))
    elif kind == "rotated":
        a["items"].append(stamp_item("red", 400, 640))
        b["items"].append(stamp_item("red_rot", 400, 640))
        truth.append(T(bbox_a=rect_px(400, 640, 72, 72), kinds=["Stamp"], desc="stamp rotated"))
    elif kind == "partial":
        a["items"].append(stamp_item("blue", 400, 640))
        st = stamp_item("blue", 400, 640)
        # B shows only the left 55 % of the stamp (cropped / partially visible)
        arr = STAMP_IMG["blue"].copy()
        arr[:, int(arr.shape[1] * 0.55):, 3] = 0
        st["data"] = png(arr)
        b["items"].append(st)
        truth.append(T(bbox_a=rect_px(400, 640, 72, 72), kinds=["Stamp"], desc="stamp partially visible"))
    return [a], [b], truth


def _sig_pages(kind):
    a = base_page()
    b = copy.deepcopy(a)
    truth = []
    box = rect_px(150, 355, 125, 41)
    if kind == "missing":
        a["items"].append(sig_item(1, 150, 355))
        truth.append(T(bbox_a=box, kinds=["Signature", "Handwriting"], category="Removed", desc="signature missing"))
    elif kind == "additional":
        b["items"].append(sig_item(1, 150, 355))
        truth.append(T(bbox_a=box, bbox_b=box, kinds=["Signature", "Handwriting"], category="Added", desc="signature added"))
    elif kind == "changed":
        a["items"].append(sig_item(1, 150, 355))
        b["items"].append(sig_item(2, 150, 355))
        truth.append(T(bbox_a=box, bbox_b=box, kinds=["Signature", "Handwriting"], desc="signature appearance changed"))
    elif kind == "moved":
        a["items"].append(sig_item(1, 150, 355))
        b["items"].append(sig_item(1, 260, 372))
        truth.append(T(bbox_a=box, bbox_b=rect_px(260, 372, 125, 41), kinds=["Signature", "Handwriting"], category="Moved", desc="signature moved"))
    elif kind == "handwritten_mark":
        # a freehand scribble (not a signature) added next to a line of text
        b["items"].append(sig_item(2, 400, 290, 100, 33))
        truth.append(T(bbox_a=rect_px(400, 290, 100, 33), bbox_b=rect_px(400, 290, 100, 33), kinds=["Signature", "Handwriting"], category="Added", desc="handwritten mark added"))
    return [a], [b], truth


def case_overlap():
    a = base_page()
    b = copy.deepcopy(a)
    truth = []
    # stamp over a text line
    L6 = find(b, "L6")
    b["items"].append(stamp_item("red", 300, LINE_Y0 + 6 * LINE_DY - 40))
    truth.append(T(bbox_a=rect_px(300, LINE_Y0 + 6 * LINE_DY - 40, 72, 72), kinds=["Stamp", "Overlap"], category="Overlapped", desc="stamp over text"))
    # signature over text
    b["items"].append(sig_item(2, 120, LINE_Y0 + 3 * LINE_DY - 25, 130, 43))
    truth.append(T(bbox_a=rect_px(120, LINE_Y0 + 3 * LINE_DY - 25, 130, 43), kinds=["Signature", "Handwriting", "Overlap"], category="Overlapped", desc="signature over text"))
    # white box hiding text (text still in text layer)
    L7 = find(b, "L7")
    box7 = sub_box(L7, "buyer confirms receipt")
    b["items"].append(dict(kind="rect", x=box7[0] / PT2PX - 1, y=box7[1] / PT2PX - 1, w=(box7[2] - box7[0]) / PT2PX + 2,
                           h=(box7[3] - box7[1]) / PT2PX + 2, fill=(1, 1, 1), stroke=None, id="whitebox"))
    truth.append(T(bbox_a=box7, kinds=["Hidden text", "Overlap", "Visual", "Text"], category="Overlapped", desc="text hidden under white box"))
    # text written over existing text
    L1 = find(b, "L1")
    b["items"].append(dict(kind="text", x=L1["x"] + 40, y=L1["y"] + 3, s="OVERWRITTEN", size=11, font="helv", id="over"))
    truth.append(T(bbox_a=rect_px(L1["x"] + 40, L1["y"] - 8, 80, 14), kinds=["Overlap", "Text"], category="Overlapped", desc="text written over text"))
    return [a], [b], truth


def case_layout():
    a = base_page()
    b = copy.deepcopy(a)
    truth = []
    # font size
    ia, ib = find(a, "L2"), find(b, "L2")
    ib["size"] = 13
    truth.append(T(bbox_a=item_box(ia), kinds=["Font", "Layout", "Visual"], desc="font size change"))
    # font family
    ia, ib = find(a, "L4"), find(b, "L4")
    ib["font"] = "tiro"
    truth.append(T(bbox_a=item_box(ia), kinds=["Font", "Layout", "Visual"], desc="font family change"))
    # alignment: line shifted right
    ia, ib = find(a, "L7"), find(b, "L7")
    ib["x"] += 28
    truth.append(T(bbox_a=item_box(ia), kinds=["Layout", "Visual", "Font"], category="Moved", desc="alignment change"))
    # table cell edit + extra row cell
    ia, ib = find(a, "cell13"), find(b, "cell13")
    ib["s"] = "6,500"
    truth.append(T(bbox_a=item_box(ia), kinds=TEXT_K + ["Number"], desc="table cell changed"))
    # header / footer text
    ia, ib = find(a, "header"), find(b, "header")
    ib["s"] = "NexaScan Demo Corp - Internal"
    truth.append(T(bbox_a=sub_box(ia, "Confidential"), kinds=TEXT_K, desc="header changed"))
    ia, ib = find(a, "footer"), find(b, "footer")
    ib["s"] = "Page 1 of 4"
    truth.append(T(bbox_a=sub_box(ia, "3"), kinds=TEXT_K, desc="footer changed"))
    return [a], [b], truth


def case_layout_spacing():
    """Paragraph spacing + margin: all lines below a point move; one grouped 'moved' difference expected."""
    a = base_page()
    b = copy.deepcopy(a)
    truth = []
    for i in range(5, 9):
        find(b, f"L{i}")["y"] += 10
    truth.append(T(bbox_a=[round(72 * PT2PX), round((LINE_Y0 + 5 * LINE_DY - 12) * PT2PX), round(520 * PT2PX), round((LINE_Y0 + 8 * LINE_DY + 4) * PT2PX)],
                   kinds=["Layout", "Visual"], category="Moved", desc="paragraph spacing changed"))
    return [a], [b], truth


def case_visual_small():
    a = base_page()
    b = copy.deepcopy(a)
    truth = []
    # tiny dot (0.5 mm) in a blank area
    b["items"].append(dict(kind="dot", x=450, y=300, r=0.9, id="dot"))
    truth.append(T(bbox_a=rect_px(448, 298, 4, 4), kinds=["Visual", "Handwriting"], category="Added", desc="tiny dot"))
    # extra line / rule
    b["items"].append(dict(kind="line", x0=72, y0=410, x1=300, y1=410, w=1.2, id="rule"))
    truth.append(T(bbox_a=rect_px(72, 408, 228, 4), kinds=["Line", "Visual"], category="Added", desc="additional line"))
    # highlight behind a word
    L5 = find(b, "L5")
    hb = sub_box(L5, "accounts@alpha-traders.example")
    b["items"].insert(0, dict(kind="rect", x=hb[0] / PT2PX, y=hb[1] / PT2PX, w=(hb[2] - hb[0]) / PT2PX, h=(hb[3] - hb[1]) / PT2PX,
                              fill=(1, 0.95, 0.3), stroke=None, id="hl"))
    truth.append(T(bbox_a=hb, kinds=["Colour", "Visual"], desc="highlight added"))
    # image (logo) changed
    logo_a = S.stamp_png("LOGO", (10, 90, 160), 120, 0.0, seed=11)
    logo_b = S.stamp_png("LOGX", (10, 90, 160), 120, 0.0, seed=12)
    a["items"].append(dict(kind="image", x=470, y=36, w=60, h=60, data=png(logo_a), id="logo"))
    b["items"].append(dict(kind="image", x=470, y=36, w=60, h=60, data=png(logo_b), id="logo"))
    truth.append(T(bbox_a=rect_px(470, 36, 60, 60), kinds=["Image", "Stamp", "Handwriting", "Visual"], desc="logo changed"))
    return [a], [b], truth


def case_multipage():
    p1, p2, p3 = base_page(), base_page(lines=["Schedule of Deliverables", "Item 1: Packaging, quantity 500 units.",
                                              "Item 2: Labels, quantity 1,200 units.", "Item 3: Pallets, quantity 40 units."],
                                  title="Schedule", footer="Page 2 of 3", table=False), base_page(
        lines=["Annexure A", "Delivery location is Warehouse 7, Peenya Industrial Area, Bengaluru 560058.",
               "Working hours are 9 am to 6 pm, Monday to Saturday."], title="Annexure", footer="Page 3 of 3", table=False)
    a_pages = [p1, p2, p3]
    b1, b3 = copy.deepcopy(p1), copy.deepcopy(p3)
    find(b3, "L1")["s"] = find(b3, "L1")["s"].replace("560058", "560085")
    p4 = base_page(lines=["Appendix B", "A completely new page that was inserted into the document."], title="Appendix",
                   footer="Page 3 of 3", table=False)
    b_pages = [b1, b3, p4]      # page 2 removed, page 3 edited, new page added
    truth = [T(page_a=2, page_b=None, kinds=["Page"], category="Removed", desc="page 2 removed"),
             T(page_a=3, page_b=2, bbox_a=sub_box(find(p3, "L1"), "560058"), kinds=TEXT_K, desc="edit on page 3"),
             T(page_a=None, page_b=3, kinds=["Page"], category="Added", desc="page added")]
    return a_pages, b_pages, truth


def case_page_reorder():
    bodies = [["Definitions and interpretation of the agreement.", "The term Effective Date means the date of signing."],
              ["Payment obligations of the buyer are set out here.", "Invoices must be settled within fifteen days."],
              ["Confidentiality duties continue after termination.", "Neither party may disclose trade secrets to anyone."]]
    pgs = [base_page(lines=bodies[i], title=f"Part {i + 1}", footer=f"Page {i + 1} of 3", table=False) for i in range(3)]
    b = [copy.deepcopy(pgs[0]), copy.deepcopy(pgs[2]), copy.deepcopy(pgs[1])]
    truth = [T(page_a=2, page_b=3, kinds=["Page"], category="Moved", desc="page moved"),
             T(page_a=3, page_b=2, kinds=["Page"], category="Moved", optional=True, desc="page moved (same swap, seen from the other side)")]
    return pgs, b, truth


def case_page_size():
    a = base_page()
    b = base_page(size=LETTER)
    for it in b["items"]:
        pass
    find(b, "L4")["s"] = find(b, "L4")["s"].replace("Bengaluru", "Chennai")
    truth = [T(kinds=["Layout"], desc="page size differs"),
             T(bbox_a=sub_box(find(a, "L4"), "Bengaluru"), kinds=TEXT_K, desc="word changed")]
    # footer sits at a different y on Letter => a layout move is a legitimate consequence
    truth.append(T(bbox_a=item_box(find(a, "footer")), kinds=["Layout", "Visual", "Text"], category="Moved", optional=True, desc="footer repositioned"))
    return [a], [b], truth


def case_native_rotated_page():
    a = base_page()
    b = copy.deepcopy(a)
    b["rotation"] = 90
    find(b, "L1")["s"] = find(b, "L1")["s"].replace("Approved.", "Rejected.")
    truth = [T(bbox_a=sub_box(find(a, "L1"), "Approved."), kinds=TEXT_K, desc="word changed on rotated page")]
    # rotated page => size/geometry differs; accept layout finding as legitimate
    truth.append(T(kinds=["Layout"], optional=True, desc="page rotation/size"))
    return [a], [b], truth


DIGITAL_CASES = {
    "d01_identical": case_identical,
    "d02_text_changes": case_text_changes,
    "d03_single_char": case_single_char,
    "d04_punctuation": case_punctuation,
    "d05_stamp_missing": lambda: _stamp_pages("missing"),
    "d06_stamp_additional": lambda: _stamp_pages("additional"),
    "d07_stamp_changed": lambda: _stamp_pages("changed"),
    "d08_stamp_moved": lambda: _stamp_pages("position"),
    "d09_stamp_rotated": lambda: _stamp_pages("rotated"),
    "d10_stamp_partial": lambda: _stamp_pages("partial"),
    "d11_sig_missing": lambda: _sig_pages("missing"),
    "d12_sig_additional": lambda: _sig_pages("additional"),
    "d13_sig_changed": lambda: _sig_pages("changed"),
    "d14_sig_moved": lambda: _sig_pages("moved"),
    "d15_handwritten_mark": lambda: _sig_pages("handwritten_mark"),
    "d16_overlap": case_overlap,
    "d17_layout": case_layout,
    "d18_layout_spacing": case_layout_spacing,
    "d19_visual_small": case_visual_small,
    "d20_multipage": case_multipage,
    "d21_page_reorder": case_page_reorder,
    "d22_page_size": case_page_size,
}


# ------------------------------------------------------------------------------ scan cases


def to_scan(pdf_path: Path, out_path: Path, kind: str, **deg) -> None:
    imgs = S.pdf_to_images(pdf_path, 300)
    outs = [S.degrade_scan(im, **deg) for im in imgs]
    if kind == "pdf":
        S.images_to_pdf(out_path, outs, 300)
    elif kind == "png":
        cv2.imwrite(str(out_path), outs[0])
    elif kind == "jpg":
        cv2.imwrite(str(out_path), outs[0], [cv2.IMWRITE_JPEG_QUALITY, 80])


def rotate_images_inplace(pdf_path: Path, rot) -> None:
    imgs = S.pdf_to_images(pdf_path, 300)
    S.images_to_pdf(pdf_path, [cv2.rotate(im, rot) for im in imgs], 300)


def scan_case_defs() -> Dict[str, dict]:
    """name -> {digital: <digital case name>, a: {kind, deg}, b: {kind, deg, rot}}"""
    mild = dict(angle=0.4, noise=4, blur=0.7, jpeg=70, shift=(5, -3), seed=1)
    mild2 = dict(angle=-0.7, noise=5, blur=0.8, jpeg=60, shift=(-9, 6), scale=0.995, seed=2)
    poor = dict(angle=-1.6, noise=9, blur=1.3, jpeg=40, shadow=0.25, brightness=-8, shift=(14, -10), seed=3)
    return {
        "s01_scan_identical": {"digital": "d01_identical", "a": {"kind": "pdf", "deg": mild}, "b": {"kind": "pdf", "deg": poor}},
        "s02_scan_text_changes": {"digital": "d02_text_changes", "a": {"kind": "pdf", "deg": mild}, "b": {"kind": "pdf", "deg": mild2}},
        "s03_scan_punct_single": {"digital": "d04_punctuation", "a": {"kind": "pdf", "deg": mild}, "b": {"kind": "pdf", "deg": mild2}},
        "s04_scan_stamps": {"digital": "d06_stamp_additional", "a": {"kind": "pdf", "deg": mild}, "b": {"kind": "pdf", "deg": mild2}},
        "s05_scan_signature": {"digital": "d13_sig_changed", "a": {"kind": "pdf", "deg": mild}, "b": {"kind": "pdf", "deg": mild2}},
        "s06_scan_overlap": {"digital": "d16_overlap", "a": {"kind": "pdf", "deg": mild}, "b": {"kind": "pdf", "deg": mild2}},
        "s07_scan_rotated90": {"digital": "d02_text_changes", "a": {"kind": "pdf", "deg": mild}, "b": {"kind": "pdf", "deg": mild2, "rot": cv2.ROTATE_90_CLOCKWISE}},
        "s08_scan_rotated180": {"digital": "d03_single_char", "a": {"kind": "pdf", "deg": mild}, "b": {"kind": "pdf", "deg": mild2, "rot": cv2.ROTATE_180}},
        "s09_scan_low_quality": {"digital": "d02_text_changes", "a": {"kind": "pdf", "deg": mild}, "b": {"kind": "pdf", "deg": poor}},
        "s10_jpg_vs_png": {"digital": "d04_punctuation", "a": {"kind": "jpg", "deg": mild}, "b": {"kind": "png", "deg": mild2}},
        "s11_scan_vs_digital": {"digital": "d02_text_changes", "a": {"kind": "digital"}, "b": {"kind": "pdf", "deg": mild2}},
        "s12_scan_multipage": {"digital": "d20_multipage", "a": {"kind": "pdf", "deg": mild}, "b": {"kind": "pdf", "deg": mild2}},
        "s13_scan_visual_small": {"digital": "d19_visual_small", "a": {"kind": "pdf", "deg": mild}, "b": {"kind": "pdf", "deg": mild2}},
    }


# ------------------------------------------------------------------------------ driver


def build_suite(out: Path) -> List[str]:
    out.mkdir(parents=True, exist_ok=True)
    names: List[str] = []
    digital_truth: Dict[str, list] = {}
    for name, fn in DIGITAL_CASES.items():
        a_pages, b_pages, truth = fn()
        d = out / name
        d.mkdir(exist_ok=True)
        render_pdf(d / "A.pdf", a_pages)
        render_pdf(d / "B.pdf", b_pages)
        (d / "truth.json").write_text(json.dumps({"truth": truth, "kind": "digital"}, indent=1))
        digital_truth[name] = truth
        names.append(name)
    for name, spec in scan_case_defs().items():
        d = out / name
        d.mkdir(exist_ok=True)
        src = out / spec["digital"]
        for side in ("a", "b"):
            sd = spec[side]
            ext = {"pdf": "pdf", "png": "png", "jpg": "jpg", "digital": "pdf"}[sd["kind"]]
            target = d / f"{side.upper()}.{ext}"
            if sd["kind"] == "digital":
                target.write_bytes((src / f"{side.upper()}.pdf").read_bytes())
                continue
            to_scan(src / f"{side.upper()}.pdf", target, sd["kind"], **sd["deg"])
            if sd.get("rot") is not None:
                rotate_images_inplace(target, sd["rot"])
        truth = copy.deepcopy(digital_truth[spec["digital"]])
        (d / "truth.json").write_text(json.dumps({"truth": truth, "kind": "scan"}, indent=1))
        names.append(name)
    (out / "cases.json").write_text(json.dumps(names))
    return names


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data" / "suite"
    n = build_suite(target)
    print(f"built {len(n)} cases in {target}")
