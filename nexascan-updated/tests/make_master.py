"""Document pairs for regression Tests 01-04 (digital PDFs with known, hand-listed differences).

NexaScan_Master_Test_Pack.zip was not available when these were written, so the pairs are rebuilt from the
expected results of the test plan (see tests/test_regression.py for the exact expectations). Every difference
is injected on purpose and listed in the EXPECTED_* constants next to the generator that creates it.

    python tests/make_master.py [out_dir]      # writes test01_A.pdf, test01_B.pdf, ... test04_B.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from samples import A4, stamp_png  # noqa: E402

BLUE = (0.12, 0.31, 0.85)
ORANGE = (0.88, 0.45, 0.10)
Run = Tuple[float, float, str, float, bool, str]       # x, y, text, size, bold, align ("l" | "r")


def _draw(page, runs: List[Run]) -> None:
    for x, y, text, size, bold, align in runs:
        fn = "hebo" if bold else "helv"
        if align == "r":
            x = x - pymupdf.get_text_length(text, fontname=fn, fontsize=size)
        page.insert_text((x, y), text, fontsize=size, fontname=fn)


def _write(path: Path, pages: List[dict]) -> None:
    doc = pymupdf.open()
    for spec in pages:
        page = doc.new_page(width=A4[0], height=A4[1])
        if spec.get("header"):
            colour, title = spec["header"]
            page.draw_rect(pymupdf.Rect(0, 0, A4[0], 64), color=None, fill=colour)
            page.insert_text((72, 40), title, fontsize=20, fontname="hebo", color=(1, 1, 1))
        _draw(page, spec.get("runs", []))
        if spec.get("stamp") is not None:
            png, rect = spec["stamp"]
            page.insert_image(pymupdf.Rect(*rect), stream=_png_bytes(png))
    doc.save(str(path))
    doc.close()


def _png_bytes(rgba: np.ndarray) -> bytes:
    import cv2
    ok, enc = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    return enc.tobytes()


# ------------------------------------------------------------------------------------------------ Test 01
# Purchase confirmation. Seven content changes, all injected in place (nothing below moves).
EXPECTED_01 = [
    "number   Rs. 10,000 -> Rs. 12,500",
    "date     12-11-2026 -> 15-11-2026 (one number)",
    "word     thirty -> sixty",
    "added    'twice' on a RIGHT-ALIGNED line (its left edge moves: must not become a separate layout row)",
    "removed  sentence 'The buyer confirms receipt ...'",
    "added    sentence 'Delivery is subject to inspection.'",
    "name     Beta Logistics -> Beta Freight",
]


def make_test01(out: Path) -> Tuple[Path, Path]:
    def runs(b: bool) -> List[Run]:
        r: List[Run] = [
            (72, 90, "Purchase Confirmation", 16, True, "l"),
            (72, 130, "This agreement is made on " + ("15-11-2026" if b else "12-11-2026")
             + " between Alpha Traders Pvt Ltd and " + ("Beta Freight." if b else "Beta Logistics."), 11, False, "l"),
            (72, 155, "Payment amount is " + ("Rs. 12,500" if b else "Rs. 10,000") + ". Approved.", 11, False, "l"),
            (72, 180, "The supplier shall deliver the goods within " + ("sixty" if b else "thirty")
             + " days of the order date.", 11, False, "l"),
            (72, 205, "All disputes are subject to the jurisdiction of the courts at Bengaluru.", 11, False, "l"),
            (72, 230, "Contact: accounts@alpha-traders.example; phone +91 98765 43210.", 11, False, "l"),
            (72, 255, "Terms: payment within 15 days (net), late fee applies, no refunds after delivery.", 11, False, "l"),
            (523, 300, "Reviewed twice" if b else "Reviewed", 11, False, "r"),
        ]
        if not b:
            r.append((72, 280, "The buyer confirms receipt of the goods in good condition.", 11, False, "l"))
        else:
            r.append((72, 330, "Delivery is subject to inspection.", 11, False, "l"))
        return r
    a, b = out / "test01_A.pdf", out / "test01_B.pdf"
    _write(a, [{"runs": runs(False)}])
    _write(b, [{"runs": runs(True)}])
    return a, b


# ------------------------------------------------------------------------------------------------ Test 02
# Laboratory report. Three result values change; the reference ranges ("70 - 100" ...) are identical.
EXPECTED_02 = ["Glucose 92 -> 118", "Cholesterol 181 -> 214", "Creatinine 0.9 -> 1.4"]
RANGES = ["70 - 100", "125 - 200", "0.6 - 1.2", "12.0 - 16.0", "4.0 - 11.0"]


def make_test02(out: Path) -> Tuple[Path, Path]:
    rows = [("Glucose (fasting)", "92", "118", "mg/dL", RANGES[0]),
            ("Cholesterol (total)", "181", "214", "mg/dL", RANGES[1]),
            ("Creatinine", "0.9", "1.4", "mg/dL", RANGES[2]),
            ("Haemoglobin", "13.8", "13.8", "g/dL", RANGES[3]),
            ("White cell count", "6.4", "6.4", "10^9/L", RANGES[4])]

    def runs(b: bool) -> List[Run]:
        r: List[Run] = [(72, 90, "Laboratory Report - Patient 0042", 16, True, "l"),
                        (72, 125, "Test", 11, True, "l"), (230, 125, "Result", 11, True, "l"),
                        (300, 125, "Unit", 11, True, "l"), (380, 125, "Reference range", 11, True, "l")]
        y = 150
        for name, va, vb, unit, rng in rows:
            r += [(72, y, name, 11, False, "l"), (230, y, vb if b else va, 11, False, "l"),
                  (300, y, unit, 11, False, "l"), (380, y, rng, 11, False, "l")]
            y += 24
        r.append((72, y + 30, "Samples were collected in the morning after a twelve hour fast.", 11, False, "l"))
        return r
    a, b = out / "test02_A.pdf", out / "test02_B.pdf"
    _write(a, [{"runs": runs(False)}])
    _write(b, [{"runs": runs(True)}])
    return a, b


# ------------------------------------------------------------------------------------------------ Test 03
# Approval letter: one stamp added in an empty area and the header band re-coloured. Nothing else changes.
def make_test03(out: Path) -> Tuple[Path, Path]:
    body: List[Run] = [
        (72, 110, "Subject: Approval of supplier onboarding request SR-2026-118", 12, True, "l"),
        (72, 145, "We confirm that the onboarding documents of Beta Logistics were reviewed.", 11, False, "l"),
        (72, 170, "The supplier meets the requirements listed in the vendor policy.", 11, False, "l"),
        (72, 195, "Please keep this letter with your records for the next audit.", 11, False, "l"),
        (72, 400, "Authorised by: Procurement Office", 11, False, "l"),
    ]
    a, b = out / "test03_A.pdf", out / "test03_B.pdf"
    _write(a, [{"header": (BLUE, "Vendor Approval"), "runs": body}])
    _write(b, [{"header": (ORANGE, "Vendor Approval"), "runs": body,
                "stamp": (stamp_png("APPROVED", (30, 60, 200), 170, 0.0, 1), (330, 330, 480, 480))}])
    return a, b


# ------------------------------------------------------------------------------------------------ Test 04
# Service level agreement, 2 pages -> 3 pages.
EXPECTED_04 = [
    "page count 2 -> 3; page 3 'APPENDIX A' reported once as an Added page",
    "three escalation e-mails appended to the three escalation lines",
    "appendix note added on page 2",
    "approval role Service Manager -> Operations Manager",
    "charge 'Rs. 1,500' added",
    "header accent colour changed",
]
UNCHANGED_04 = ["SLA-2026-0417", "Orbit Networks Pvt Ltd", "Meridian Healthcare Ltd", "01-01-2027", "24 months",
                "support@orbitnet.example", "24x7 (Mon-Sun)"]


def make_test04(out: Path) -> Tuple[Path, Path]:
    def fields(b: bool) -> List[Run]:
        data = [("Agreement ID", "SLA-2026-0417"), ("Provider", "Orbit Networks Pvt Ltd"),
                ("Customer", "Meridian Healthcare Ltd"), ("Start date", "01-01-2027"), ("Term", "24 months"),
                ("Support email", "support@orbitnet.example"), ("Support hours", "24x7 (Mon-Sun)"),
                ("Approval role", "Operations Manager" if b else "Service Manager")]
        r: List[Run] = []
        y = 110
        for k, v in data:
            r += [(72, y, k + ":", 11, True, "l"), (200, y, v, 11, False, "l")]
            y += 24
        return r

    def page1(b: bool) -> dict:
        r = fields(b)
        r += [(72, 330, "Escalation contacts", 13, True, "l"),
              (72, 355, "Level 1 - Support desk" + (" - l1-desk@orbitnet.example" if b else ""), 11, False, "l"),
              (72, 380, "Level 2 - Duty manager" + (" - l2-duty@orbitnet.example" if b else ""), 11, False, "l"),
              (72, 405, "Level 3 - Account director" + (" - l3-account@orbitnet.example" if b else ""), 11, False, "l"),
              (72, 450, "Charges", 13, True, "l"),
              (72, 475, "Standard support is included in the monthly fee.", 11, False, "l")]
        if b:
            r.append((72, 500, "An after-hours on-site visit incurs a charge of Rs. 1,500.", 11, False, "l"))
        return {"header": (ORANGE if b else BLUE, "Service Level Agreement"), "runs": r}

    def page2(b: bool) -> dict:
        r: List[Run] = [(72, 90, "Service levels", 14, True, "l"),
                        (72, 120, "Priority 1 incidents are acknowledged within fifteen minutes.", 11, False, "l"),
                        (72, 145, "Priority 2 incidents are acknowledged within one hour.", 11, False, "l"),
                        (72, 170, "Priority 3 incidents are acknowledged within one business day.", 11, False, "l"),
                        (72, 195, "Availability is measured monthly and reported to the customer.", 11, False, "l")]
        if b:
            r.append((72, 260, "Note: service credit tables are listed in Appendix A.", 11, False, "l"))
        return {"runs": r}

    appendix = {"runs": [(72, 90, "APPENDIX A", 16, True, "l"),
                         (72, 130, "Service credit table", 13, True, "l"),
                         (72, 160, "Availability below 99.5 percent: credit of five percent of the monthly fee.", 11, False, "l"),
                         (72, 185, "Availability below 99.0 percent: credit of ten percent of the monthly fee.", 11, False, "l"),
                         (72, 210, "Availability below 98.0 percent: credit of twenty percent of the monthly fee.", 11, False, "l")]}
    a, b = out / "test04_A.pdf", out / "test04_B.pdf"
    _write(a, [page1(False), page2(False)])
    _write(b, [page1(True), page2(True), appendix])
    return a, b


MAKERS = {"01": make_test01, "02": make_test02, "03": make_test03, "04": make_test04}


def make_all(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    return {k: fn(out) for k, fn in MAKERS.items()}


if __name__ == "__main__":
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data" / "master"
    for k, (a, b) in make_all(dest).items():
        print(k, a, b)
