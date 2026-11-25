"""Word (.docx / .doc) support: validation, error handling, conversion, and end-to-end comparison through the worker."""
import io
import json
import shutil
import sys
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import synth_word as SW  # noqa: E402
from nexascan import wordsupport as W  # noqa: E402
from nexascan.config import Config  # noqa: E402
from nexascan.upload import UploadError  # noqa: E402
from nexascan.worker import run_job  # noqa: E402

PACK = ROOT / "tests" / "data" / "pack"
needs_office = pytest.mark.skipif(not W.word_available(), reason="LibreOffice is not installed")


@pytest.fixture(scope="module")
def docs(tmp_path_factory):
    d = tmp_path_factory.mktemp("word")
    out = {"a": SW.build(d / "orig.docx"), "b": SW.build(d / "rev.docx", revised=True),
           "fa": SW.build(d / "forig.docx"), "fb": SW.build(d / "frev.docx", revised=True, formatted=True)}
    if W.word_available():
        out["a_doc"], out["b_doc"] = SW.to_doc(out["a"], d), SW.to_doc(out["b"], d)
    return out


def compare(tmp_path, a, b):
    """Exactly what the API + worker do for an upload: prepare() each file, then run_job()."""
    jd = tmp_path / "job"
    jd.mkdir(exist_ok=True)
    cfg = Config()
    for slot, src in (("a", a), ("b", b)):
        with open(src, "rb") as f:
            out = W.prepare(f, Path(src).name, jd, slot, cfg)
        (jd / f"input_{slot}.bin").write_bytes(out.read())
    run_job(str(jd), Path(a).name, Path(b).name, None)
    assert json.loads((jd / "progress.json").read_text())["status"] == "done"
    return json.loads((jd / "report.json").read_text(encoding="utf-8")), jd


def blob(rep):
    return " | ".join(f"{i['difference']} {i['document_a']} {i['document_b']}" for i in rep["differences"] + rep["uncertain"])


# ---------------------------------------------------------------- detection, validation and graceful errors
def test_detect_docx_and_doc(docs, tmp_path):
    assert W.detect(Path(docs["a"]), "x.docx") == "docx"
    if "a_doc" in docs:
        assert W.detect(Path(docs["a_doc"]), "x.doc") == "doc"


def test_corrupted_docx_is_rejected(docs, tmp_path):
    data = Path(docs["a"]).read_bytes()
    p = tmp_path / "broken.docx"
    p.write_bytes(data[: len(data) // 2])
    with pytest.raises(W.WordError, match="corrupted"):
        W.detect(p, "broken.docx")


def test_zip_without_word_content_and_plain_text_named_docx(tmp_path):
    import zipfile
    z = tmp_path / "x.docx"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("hello.txt", "hi")
    with pytest.raises(W.WordError, match="not a valid Word"):
        W.detect(z, "x.docx")
    t = tmp_path / "t.docx"
    t.write_text("just text, not Word")
    with pytest.raises(W.WordError, match="not a valid Word"):
        W.detect(t, "t.docx")


def test_zero_byte_word_file_is_rejected(tmp_path):
    p = tmp_path / "e.docx"
    p.write_bytes(b"")
    with pytest.raises(W.WordError, match="empty"):
        W.detect(p, "e.docx")


@needs_office
def test_document_without_any_content_is_rejected(tmp_path):
    from docx import Document
    p = tmp_path / "blank.docx"
    Document().save(str(p))
    with open(p, "rb") as f, pytest.raises(W.WordError, match="contains no text"):
        W.prepare(f, "blank.docx", tmp_path, "a", Config())


class _FakeOle:
    """Stands in for olefile.OleFileIO so a password-protected / non-Word container can be tested deterministically."""
    def __init__(self, streams):
        self.streams = streams
    def __call__(self, path):
        return self
    def exists(self, name):
        return name in self.streams
    def openstream(self, name):
        return io.BytesIO(self.streams[name])
    def close(self):
        pass


def _ole_file(tmp_path, name):
    p = tmp_path / name
    p.write_bytes(W.OLE_MAGIC + b"\0" * 600)
    return p


def test_password_protected_docx_and_doc_are_rejected(tmp_path, monkeypatch):
    import olefile
    monkeypatch.setattr(olefile, "OleFileIO", _FakeOle({"EncryptedPackage": b"x", "EncryptionInfo": b"x"}))
    with pytest.raises(W.WordError, match="password-protected"):
        W.detect(_ole_file(tmp_path, "p.docx"), "p.docx")
    fib = bytearray(32)
    fib[10:12] = (0x0100).to_bytes(2, "little")                              # FIB.fEncrypted
    monkeypatch.setattr(olefile, "OleFileIO", _FakeOle({"WordDocument": bytes(fib)}))
    with pytest.raises(W.WordError, match="password-protected"):
        W.detect(_ole_file(tmp_path, "p.doc"), "p.doc")
    monkeypatch.setattr(olefile, "OleFileIO", _FakeOle({"WordDocument": bytes(32)}))
    assert W.detect(_ole_file(tmp_path, "ok.doc"), "ok.doc") == "doc"        # same container, flag not set: accepted
    monkeypatch.setattr(olefile, "OleFileIO", _FakeOle({"Workbook": b"x"}))
    with pytest.raises(W.WordError, match="not a valid Word"):
        W.detect(_ole_file(tmp_path, "sheet.doc"), "sheet.doc")


def test_missing_libreoffice_gives_a_clear_message(docs, tmp_path, monkeypatch):
    monkeypatch.setattr(W, "find_soffice", lambda: None)
    with open(docs["a"], "rb") as f, pytest.raises(UploadError, match="LibreOffice"):
        W.prepare(f, "orig.docx", tmp_path, "a", Config())


def test_other_formats_pass_through_untouched(tmp_path):
    for p in (PACK / "01_PDF_Text_Comparison" / "original_invoice.pdf", PACK / "02_JPG_Scanned_OCR" / "original_lab_report.jpg",
              PACK / "03_PNG_Visual_Layout" / "original_delivery.png"):
        with open(p, "rb") as f:
            assert W.prepare(f, p.name, tmp_path, "a", Config()) is f           # the very same stream: existing flow
    assert not list(tmp_path.glob("word_*"))


def test_unsupported_file_message_lists_word(tmp_path):
    with pytest.raises(UploadError, match=r"Allowed: PDF, PNG, JPG/JPEG, Word \(\.docx, \.doc\)"):
        W.prepare(io.BytesIO(b"plain text file"), "notes.txt", tmp_path, "a", Config())


# ---------------------------------------------------------------- end to end: .docx
@needs_office
def test_docx_text_numbers_punctuation_tables_headers_footers(docs, tmp_path):
    rep, jd = compare(tmp_path, docs["a"], docs["b"])
    t = blob(rep)
    for needle in ("15 to 20", "15 to 30", "2%", "thank you",              # numbers, punctuation / added text
                   "2 to 3", "Keyboard", "1,200",                           # table cell change and an added table row
                   "v2", "(revised)",                                       # header and footer
                   "assign", "terminate"):                                  # deleted and added clauses
        assert needle in t, needle
    assert (jd / "report.html").is_file() and (jd / "word_a.docx").is_file()
    assert rep["document_a"]["kind"] == "pdf" and rep["diagnostics"]["word"]["format_findings"] == 0


@needs_office
def test_docx_formatting_font_size_bold_italic_alignment_underline(docs, tmp_path):
    rep, jd = compare(tmp_path, docs["fa"], docs["fb"])
    items = rep["differences"]
    fonts = [i for i in items if i["dtype"] == "Font"]
    t = " | ".join(i["difference"] for i in fonts)
    assert "paragraph one" in t and "bold/italic" in t and "paragraph two" in t     # bold, italic (existing detection)
    assert "Font size changed for 'Formatting sample paragraph three.'" in t        # size
    assert "Paragraph alignment changed from left to centered: 'Formatting sample paragraph four.'" in t
    assert "Underline added: 'Formatting sample paragraph five.'" in t
    assert rep["diagnostics"]["word"]["format_findings"] == 2 and rep["diagnostics"]["word"]["pipeline_findings_replaced"] == 2
    assert not any("moved by (+" in i["difference"] or i["dtype"] == "Line" for i in items)    # the indirect findings were replaced
    for i in fonts:
        if "word-xml" in i["engines"]:
            assert i["confidence"] == 0.95 and i["location"]["a"]["bbox_mm"] and (jd / i["evidence"]["a"]).is_file()
    assert rep["summary"]["total_differences"] == len(items) and [i["number"] for i in items] == list(range(1, len(items) + 1))
    html = (jd / "report.html").read_text(encoding="utf-8")
    assert "Paragraph alignment changed from left to centered" in html


@needs_office
def test_identical_word_documents_have_no_differences(docs, tmp_path):
    rep, _ = compare(tmp_path, docs["a"], docs["a"])
    assert rep["differences"] == [] and rep["uncertain"] == []


# ---------------------------------------------------------------- end to end: .doc and mixed
@needs_office
def test_doc_files_are_compared_like_docx(docs, tmp_path):
    rep, _ = compare(tmp_path, docs["a_doc"], docs["b_doc"])
    t = blob(rep)
    for needle in ("15 to 20", "15 to 30", "thank you", "Keyboard", "v2", "(revised)"):
        assert needle in t, needle


@needs_office
def test_doc_formatting_alignment_and_underline(docs, tmp_path):
    a, b = SW.to_doc(docs["fa"], tmp_path), SW.to_doc(docs["fb"], tmp_path)         # LibreOffice writes jc="start" etc.
    rep, _ = compare(tmp_path / "job1", a, b) if (tmp_path / "job1").mkdir() is None else (None, None)
    t = " | ".join(i["difference"] for i in rep["differences"] if i["dtype"] == "Font")
    assert "Paragraph alignment changed from left to centered" in t and "Underline added" in t and "Font size changed" in t
    assert rep["diagnostics"]["word"]["format_findings"] == 2


@needs_office
def test_alignment_variants_are_read_from_the_xml():
    from docx import Document
    import tempfile
    from docx.enum.text import WD_ALIGN_PARAGRAPH as A
    p = Path(tempfile.mkdtemp()) / "al.docx"
    d = Document()
    for text, al in (("left one", A.LEFT), ("center one", A.CENTER), ("right one", A.RIGHT), ("justify one", A.JUSTIFY)):
        d.add_paragraph(text).alignment = al
    d.save(str(p))
    got = {r["text"]: r["align"] for r in W.paragraph_records(p)}
    assert got == {"left one": "left", "center one": "centered", "right one": "right", "justify one": "justified"}


@needs_office
def test_doc_against_docx_and_word_against_pdf(docs, tmp_path):
    rep, _ = compare(tmp_path / "x", docs["a_doc"], docs["b"]) if (tmp_path / "x").mkdir() is None else (None, None)
    assert "15 to 20" in blob(rep)
    (tmp_path / "y").mkdir()
    jd = tmp_path / "y" / "job"
    jd.mkdir()
    with open(docs["a"], "rb") as f:
        (jd / "input_a.bin").write_bytes(W.prepare(f, "orig.docx", jd, "a", Config()).read())
    shutil.copy(jd / "input_a.bin", jd / "input_b.bin")                   # Word-derived PDF against a plain PDF: no Word step
    (jd / "word_a.docx").unlink()
    run_job(str(jd), "orig.docx", "orig.pdf", None)
    assert json.loads((jd / "progress.json").read_text())["status"] == "done"
    assert json.loads((jd / "report.json").read_text())["differences"] == []


# ---------------------------------------------------------------- frontend
def test_frontend_accepts_and_shows_word_formats():
    app = (ROOT / "frontend" / "src" / "App.jsx").read_text(encoding="utf-8")
    assert ".docx" in app and ".doc" in app and "Supported formats" in app and "Word" in app
    assert 'type="checkbox"' not in app
