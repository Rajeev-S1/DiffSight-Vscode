"""Deterministic Word fixtures (python-docx + LibreOffice for .doc). No external data."""
import shutil
import subprocess
import tempfile
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt


def build(path, revised=False, formatted=False):
    """Original / revised contract. ``revised`` changes content; ``formatted`` (with revised) changes only formatting."""
    content, fmt = revised and not formatted, revised and formatted
    d = Document()
    sec = d.sections[0]
    sec.header.paragraphs[0].text = "Orbital Tech - Service Agreement" + (" v2" if revised else "")
    sec.footer.paragraphs[0].text = "Confidential - page footer" + (" (revised)" if revised else "")
    d.add_heading("Service Agreement", 1)
    d.add_paragraph("This agreement is made on %s November 2026 between Orbital Tech and Cedar Grove Academy." % ("20" if content else "15"))
    d.add_paragraph("Payment is due within %s days of the invoice date." % ("30" if content else "15"))
    d.add_paragraph("The late fee is 2%% per month, thank you." if content else "The late fee is 1% per month.")
    d.add_paragraph("Clause nine: the supplier may assign this agreement." if not content else "Clause ten: either party may terminate with notice.")
    t = d.add_table(rows=3, cols=3)
    t.style = "Table Grid"
    rows = [("Item", "Qty", "Price"), ("Laptop", "3" if content else "2", "12,500"), ("Mouse", "1", "750")]
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            t.cell(i, j).text = v
    if content:
        row = t.add_row()
        for j, v in enumerate(("Keyboard", "1", "1,200")):
            row.cells[j].text = v
    ps = [d.add_paragraph(f"Formatting sample paragraph {n}.") for n in ("one", "two", "three", "four", "five")]
    if fmt:
        ps[0].runs[0].bold = True
        ps[1].runs[0].italic = True
        ps[2].runs[0].font.size = Pt(18)
        ps[3].alignment = WD_ALIGN_PARAGRAPH.CENTER
        ps[4].runs[0].underline = True
    d.save(str(path))
    return str(path)


def to_doc(docx_path, out_dir):
    """Binary .doc made by LibreOffice from a .docx (None when LibreOffice is not available)."""
    from nexascan import wordsupport as W
    soffice = W.find_soffice()
    if not soffice:
        return None
    profile = Path(tempfile.mkdtemp(prefix="lo_"))
    try:
        out = W._convert(Path(docx_path), "doc", Path(out_dir), profile)
        return str(out)
    finally:
        shutil.rmtree(profile, ignore_errors=True)
