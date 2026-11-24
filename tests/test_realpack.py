"""Regression tests on the REAL NexaScan_Master_Test_Pack (Tests 01-04), files in tests/data/pack/."""
import sys
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from nexascan.config import Config  # noqa: E402
from nexascan.pipeline import compare_documents  # noqa: E402

PACK = ROOT / "tests" / "data" / "pack"


def _run(folder, a, b, tmp_path):
    return compare_documents(str(PACK / folder / a), a, str(PACK / folder / b), b, tmp_path, Config(use_cache=False))


def _texts(r, key="differences"):
    return [i["difference"] for i in r[key]]


def test01_invoice_no_extra_layout_row(tmp_path):
    r = _run("01_PDF_Text_Comparison", "original_invoice.pdf", "revised_invoice.pdf", tmp_path)
    assert len(r["differences"]) == 7 and not r["uncertain"]
    assert not any(i["dtype"] == "Layout" for i in r["differences"])
    t = " | ".join(_texts(r))
    for s in ("15 to 20", "15 to 30", "Wireless mouse", "12,500 to 14,000", "2,250 to 2,520", "14,750 to 16,520", "1 to 2"):
        assert s in t, s


def test02_lab_report_no_false_moves(tmp_path):
    r = _run("02_JPG_Scanned_OCR", "original_lab_report.jpg", "revised_lab_report.jpg", tmp_path)
    t = " | ".join(_texts(r))
    assert "92 to 112" in t and "01 to 02" in t and any(i["dtype"] == "Stamp" for i in r["differences"])
    assert len(r["differences"]) == 3
    assert not any(i["category"] == "Moved" for i in r["differences"] + r["uncertain"])


def test03_png_one_stamp_one_colour(tmp_path):
    r = _run("03_PNG_Visual_Layout", "original_delivery.png", "revised_delivery.png", tmp_path)
    assert sum(i["dtype"] == "Stamp" for i in r["differences"]) == 1
    assert sum(i["dtype"] == "Colour" for i in r["differences"]) == 1
    assert not any(i["dtype"] == "Block" for i in r["differences"]) and not r["uncertain"]


def test04_multipage_page_added(tmp_path):
    r = _run("04_PDF_Multipage_Page_Changes", "original_agreement.pdf", "revised_agreement.pdf", tmp_path)
    t = " | ".join(_texts(r))
    assert "new page was added" in t and "Service' to 'Operations" in t and "1,500" in t
    assert any(i["dtype"] == "Colour" for i in r["differences"])
    assert len(r["differences"]) == 4
