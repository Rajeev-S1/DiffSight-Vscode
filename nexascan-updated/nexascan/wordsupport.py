"""Microsoft Word support (.docx and .doc) - an isolated adapter in front of the existing pipeline.

How it fits: a Word upload is validated by its CONTENT (not its extension), normalised to .docx and rendered to PDF with
LibreOffice (headless). The PDF then goes through the unchanged PDF pipeline, so text, numbers, punctuation, tables,
headers, footers, page content, fonts, sizes, bold/italic and layout are compared by the existing algorithms.
After the comparison, ``augment_report`` adds what only the Word file itself knows (paragraph alignment, underline),
using the same finding format as every other finding. Nothing here runs unless a Word file was uploaded.

Problems are reported with a clear message (HTTP 400, UploadError): corrupted, password-protected, empty, unsupported,
or LibreOffice not installed / too slow.
"""
from __future__ import annotations

import difflib
import io
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import BinaryIO, List, Optional, Tuple

from .applog import get_logger
from .upload import UploadError, safe_display_name, sniff_kind, stream_to_disk

log = get_logger()

OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
ZIP_MAGIC = b"PK\x03\x04"
KNOWN_OTHER = (b"%PDF-", b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff")      # PDF / PNG / JPG: the existing flow handles these
WORD_EXTS = (".doc", ".docx")
CONVERT_TIMEOUT_S = int(os.environ.get("NEXASCAN_WORD_TIMEOUT", "180"))
FRIENDLY = "Word (.docx, .doc)"


class WordError(UploadError):
    """A Word file that cannot be compared; the message is shown to the user."""


# ------------------------------------------------------------------ LibreOffice
def find_soffice() -> Optional[str]:
    env = os.environ.get("NEXASCAN_SOFFICE")
    if env and Path(env).is_file():
        return env
    for name in ("soffice", "libreoffice"):
        p = shutil.which(name)
        if p:
            return p
    for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"), os.environ.get("LOCALAPPDATA")):
        if base:
            c = Path(base) / "LibreOffice" / "program" / "soffice.exe"
            if c.is_file():
                return str(c)
    mac = Path("/Applications/LibreOffice.app/Contents/MacOS/soffice")
    return str(mac) if mac.is_file() else None


def word_available() -> bool:
    return find_soffice() is not None


def _convert(src: Path, fmt: str, outdir: Path, profile: Path) -> Path:
    soffice = find_soffice()
    if not soffice:
        raise WordError("Word documents need LibreOffice on the server, and it is not installed. "
                        "Install LibreOffice (winget install -e --id TheDocumentFoundation.LibreOffice) or upload a PDF.")
    cmd = [soffice, "--headless", "--norestore", "--nolockcheck", "--nodefault", "--nologo",
           f"-env:UserInstallation={profile.resolve().as_uri()}", "--convert-to", fmt, "--outdir", str(outdir), str(src)]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=CONVERT_TIMEOUT_S, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        raise WordError(f"The Word document took longer than {CONVERT_TIMEOUT_S} s to convert and was stopped. Try a smaller file.")
    except OSError as e:
        raise WordError(f"LibreOffice could not be started ({type(e).__name__}).")
    out = outdir / (src.stem + "." + fmt.split(":")[0])
    if not out.is_file() or out.stat().st_size == 0:
        log.warning("word conversion to %s failed (exit %s): %s", fmt, r.returncode, (r.stderr or b"")[-300:].decode("utf-8", "replace"))
        raise WordError("The Word document could not be converted - the file may be corrupted or use unsupported features.")
    return out


# ------------------------------------------------------------------ detection and validation
def detect(path: Path, filename: str) -> Optional[str]:
    """'docx' / 'doc' for a usable Word file, None when this is not a Word upload (the existing flow decides),
    WordError for a Word file that is corrupted, password-protected or not really Word."""
    ext = Path(filename or "").suffix.lower()
    is_word_name = ext in WORD_EXTS
    with open(path, "rb") as f:
        head = f.read(16)
    if not head:
        if is_word_name:
            raise WordError("The Word document is empty (0 bytes).")
        return None
    if any(head.startswith(m) for m in KNOWN_OTHER):
        return None
    if head.startswith(ZIP_MAGIC):
        try:
            with zipfile.ZipFile(path) as z:
                names = set(z.namelist())
                if z.testzip() is not None:
                    raise zipfile.BadZipFile("bad member")
        except zipfile.BadZipFile:
            if is_word_name:
                raise WordError("The Word document is corrupted and could not be opened.")
            return None
        if "word/document.xml" in names:
            return "docx"
        if is_word_name:
            raise WordError("This file is not a valid Word document (word/document.xml is missing).")
        return None
    if head.startswith(OLE_MAGIC):
        try:
            import olefile
            ole = olefile.OleFileIO(str(path))
        except Exception:  # noqa: BLE001
            if is_word_name:
                raise WordError("The Word document is corrupted and could not be opened.")
            return None
        try:
            if ole.exists("EncryptedPackage") or ole.exists("EncryptionInfo"):
                if is_word_name:
                    raise WordError("The Word document is password-protected. Remove the password and upload it again.")
                return None
            if ole.exists("WordDocument"):
                fib = ole.openstream("WordDocument").read(12)
                if len(fib) >= 12 and (int.from_bytes(fib[10:12], "little") & 0x0100):      # FIB.fEncrypted
                    raise WordError("The Word document is password-protected. Remove the password and upload it again.")
                return "doc"
        finally:
            ole.close()
        if is_word_name:
            raise WordError("This file is not a valid Word document.")
        return None
    if is_word_name:
        raise WordError("This file is not a valid Word document. (RTF or HTML saved with a .doc extension is not supported: "
                        "save it as .docx.)")
    return None


def _content_flags(docx: Path) -> Tuple[bool, bool]:
    """(has_text_or_table_or_image, readable). Raises WordError when python-docx cannot open the file."""
    try:
        import docx as pydocx
        d = pydocx.Document(str(docx))
    except Exception:  # noqa: BLE001
        raise WordError("The Word document is corrupted and could not be read.")
    if any(p.text.strip() for p in d.paragraphs):
        return True, True
    for t in d.tables:
        if any(c.text.strip() for r in t.rows for c in r.cells):
            return True, True
    for s in d.sections:
        for part in (s.header, s.footer):
            if any(p.text.strip() for p in part.paragraphs):
                return True, True
    try:
        if any("/media/" in str(part.partname) for part in d.part.package.iter_parts()):
            return True, True
    except Exception:  # noqa: BLE001
        pass
    return False, True


def prepare(fileobj: BinaryIO, filename: str, job_dir: Path, slot: str, cfg) -> BinaryIO:
    """Return ``fileobj`` untouched for anything that is not a Word upload (PDF, PNG, JPG ...); for a Word file return a
    file-like object holding its PDF rendering, and keep ``word_<slot>.docx`` in the job folder for ``augment_report``."""
    ext = Path(filename or "").suffix.lower()
    head = fileobj.read(1024)
    fileobj.seek(0)
    if sniff_kind(head) is not None:                            # PDF / PNG / JPG: exactly the existing flow
        return fileobj
    if not (ext in WORD_EXTS or head.startswith(ZIP_MAGIC) or head.startswith(OLE_MAGIC)):
        if not head:
            return fileobj                                      # empty file: the existing validation reports it
        raise UploadError(f"'{safe_display_name(filename)}' is not a supported file. Allowed: PDF, PNG, JPG/JPEG, {FRIENDLY}.")
    job_dir = Path(job_dir)
    src = job_dir / f"word_{slot}_upload.tmp"
    stream_to_disk(fileobj, src, cfg.max_file_mb * 1024 * 1024)
    try:
        kind = detect(src, filename)
        if kind is None:                                        # a zip / OLE file that is not Word
            src.unlink(missing_ok=True)
            raise UploadError(f"'{safe_display_name(filename)}' is not a supported file. Allowed: PDF, PNG, JPG/JPEG, {FRIENDLY}.")
        profile = job_dir / f"_lo_{slot}"
        try:
            source = job_dir / f"word_{slot}_source.{kind}"
            src.replace(source)
            docx = job_dir / f"word_{slot}.docx"
            if kind == "docx":
                shutil.copyfile(source, docx)
            else:                                               # .doc -> .docx (one normalised form for everything below)
                converted = _convert(source, "docx", job_dir, profile)
                converted.replace(docx)
            has_content, _ = _content_flags(docx)
            if not has_content:
                raise WordError("The Word document is empty: it contains no text, tables or images.")
            pdf = _convert(docx, "pdf", job_dir, profile)
            data = pdf.read_bytes()
            pdf.unlink(missing_ok=True)
        finally:
            shutil.rmtree(profile, ignore_errors=True)
        log.info("word upload %s (%s) converted to PDF", slot.upper(), kind)
        return io.BytesIO(data)
    except BaseException:
        src.unlink(missing_ok=True)
        raise


# ====================================================================== after the comparison: alignment and underline
# The PDF pipeline already reports text, numbers, punctuation, tables, headers, footers, fonts, sizes, bold and italic.
# It can only describe alignment and underline indirectly ("1 line moved by x mm", "ruled line added"). The Word file
# itself knows exactly, so these two are named explicitly and the indirect findings they explain are replaced.
MM = 25.4 / 72.0


def _norm(t: str) -> str:
    return " ".join((t or "").replace("\xa0", " ").split())


def paragraph_records(docx_path: Path) -> List[dict]:
    """Body paragraphs (tables included, in reading order) then header / footer paragraphs, each with its
    effective alignment and underline."""
    import docx as pydocx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    from docx.oxml.ns import qn

    # read w:jc straight from the XML: python-docx cannot parse the 'start' / 'end' values that LibreOffice writes
    names = {"left": "left", "start": "left", "center": "centered", "right": "right", "end": "right",
             "both": "justified", "justify": "justified", "distribute": "distributed", "thaiDistribute": "distributed"}
    d = pydocx.Document(str(docx_path))
    seen, out = set(), []

    def jc_of(el):
        ppr = getattr(el, "pPr", None)
        j = ppr.find(qn("w:jc")) if ppr is not None else None
        return j.get(qn("w:val")) if j is not None else None

    def align(p) -> str:
        a = jc_of(p._p)
        st = p.style
        while a is None and st is not None:
            a, st = jc_of(st.element), st.base_style
        return names.get(a, "left") if a is not None else "left"

    def underlined(p) -> bool:
        runs = [r for r in p.runs if r.text.strip()]
        if not runs:
            return False
        def one(r):
            u = r.underline
            if u is None:
                st = p.style
                while u is None and st is not None:
                    u, st = st.font.underline, st.base_style
            return bool(u)
        return sum(1 for r in runs if one(r)) * 2 >= len(runs)

    def add(p, where):
        if p._p in seen:                                 # merged table cells repeat the same paragraph
            return
        seen.add(p._p)                                   # (the element itself, not id(): lxml proxies are recycled)
        t = _norm(p.text)
        if t:
            out.append({"text": t, "align": align(p), "underline": underlined(p), "where": where})

    def walk(container, where):
        for child in container.element.body.iterchildren() if hasattr(container.element, "body") else container._element.iterchildren():
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == "p":
                add(Paragraph(child, container), where)
            elif tag == "tbl":
                tb = Table(child, container)
                for row in tb.rows:
                    for cell in row.cells:
                        for p in cell.paragraphs:
                            add(p, where)

    walk(d, "body")
    for sec in d.sections:
        for part, w in ((sec.header, "header"), (sec.footer, "footer")):
            for p in part.paragraphs:
                add(p, w)
    return out


def format_changes(docx_a: Path, docx_b: Path) -> List[dict]:
    """Paragraphs present in both files (identical or lightly edited text) whose alignment or underline changed."""
    ra, rb = paragraph_records(docx_a), paragraph_records(docx_b)
    sm = difflib.SequenceMatcher(None, [r["text"] for r in ra], [r["text"] for r in rb], autojunk=False)
    pairs = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal" or (tag == "replace" and (i2 - i1) == (j2 - j1)):
            for k in range(i2 - i1):
                a, b = ra[i1 + k], rb[j1 + k]
                if tag == "equal" or difflib.SequenceMatcher(None, a["text"], b["text"]).ratio() >= 0.75:
                    pairs.append((a, b))
    out = []
    for a, b in pairs:
        if a["align"] != b["align"]:
            out.append({"kind": "alignment", "old": a["align"], "new": b["align"], "text_a": a["text"], "text_b": b["text"], "where": a["where"]})
        if a["underline"] != b["underline"]:
            out.append({"kind": "underline", "old": "underlined" if a["underline"] else "not underlined",
                        "new": "underlined" if b["underline"] else "not underlined", "text_a": a["text"], "text_b": b["text"], "where": a["where"]})
    return out


def _lines(pdf_path: Path) -> List[dict]:
    import pymupdf
    out = []
    with pymupdf.open(str(pdf_path)) as doc:
        for pno, page in enumerate(doc, 1):
            for blk in page.get_text("dict")["blocks"]:
                for ln in blk.get("lines", []):
                    t = _norm("".join(sp["text"] for sp in ln["spans"]))
                    if t:
                        out.append({"page": pno, "bbox": tuple(ln["bbox"]), "text": t, "w": page.rect.width, "h": page.rect.height})
    return out


def _find(lines: List[dict], text: str, occurrence: int) -> Optional[dict]:
    hits = []
    for s0 in range(len(lines)):
        acc = ""
        for e in range(s0, min(len(lines), s0 + 60)):
            if lines[e]["page"] != lines[s0]["page"]:
                break
            acc = (acc + " " + lines[e]["text"]).strip()
            if acc == text:
                hits.append((s0, e))
                break
            if not text.startswith(acc):
                break
    if not hits:
        return None
    s0, e = hits[min(occurrence, len(hits) - 1)]
    box = lines[s0]["bbox"]
    for ln in lines[s0:e + 1]:
        box = (min(box[0], ln["bbox"][0]), min(box[1], ln["bbox"][1]), max(box[2], ln["bbox"][2]), max(box[3], ln["bbox"][3]))
    return {"page": lines[s0]["page"], "bbox": box, "w": lines[s0]["w"], "h": lines[s0]["h"]}


def _overlap(a, b, grow: float = 0.0) -> float:
    """Share of the smaller box that lies inside the other (mm boxes); ``grow`` widens ``b``."""
    b = (b[0] - grow, b[1] - grow, b[2] + grow, b[3] + grow)
    ix, iy = max(0.0, min(a[2], b[2]) - max(a[0], b[0])), max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return (ix * iy) / small if small > 0 else 0.0


def _snip(t: str, n: int = 70) -> str:
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + "..."


def augment_report(job_dir, report: dict) -> dict:
    """Add explicit alignment / underline findings to the finished report of a Word comparison (no-op otherwise).
    Never raises: a problem here leaves the report exactly as the pipeline produced it."""
    jd = Path(job_dir)
    try:
        da, db = jd / "word_a.docx", jd / "word_b.docx"
        if not (da.is_file() and db.is_file()):
            return report
        import pymupdf
        from .report import _box, refresh_summary, write_reports

        changes = format_changes(da, db)
        diag = report.setdefault("diagnostics", {})
        diag["word"] = {"documents": "Word + Word", "format_checks": ["alignment", "underline"], "format_findings": 0}
        if not changes:
            write_reports(report, jd)
            return report
        dpi = float((diag.get("config") or {}).get("dpi") or 300)
        la, lb = _lines(jd / "input_a.bin"), _lines(jd / "input_b.bin")
        occ_a, occ_b, items, absorbed_total = {}, {}, [], 0
        (jd / "evidence").mkdir(parents=True, exist_ok=True)
        for ch in changes:
            fa = _find(la, ch["text_a"], occ_a.get(ch["text_a"], 0))
            fb = _find(lb, ch["text_b"], occ_b.get(ch["text_b"], 0))
            occ_a[ch["text_a"]] = occ_a.get(ch["text_a"], 0) + 1
            occ_b[ch["text_b"]] = occ_b.get(ch["text_b"], 0) + 1
            if fa is None or fb is None:
                continue
            k = len(items) + 1
            ev = {}
            for side, f, src in (("a", fa, jd / "input_a.bin"), ("b", fb, jd / "input_b.bin")):
                with pymupdf.open(str(src)) as doc:
                    r = pymupdf.Rect(f["bbox"]) + (-14, -10, 14, 10)
                    pix = doc[f["page"] - 1].get_pixmap(clip=r & doc[f["page"] - 1].rect, dpi=150)
                    pix.save(str(jd / "evidence" / f"wd_{k}_{side}.png"))
                ev[side] = f"evidence/wd_{k}_{side}.png"
            ra_mm, rb_mm = tuple(v * MM for v in fa["bbox"]), tuple(v * MM for v in fb["bbox"])
            absorbed = 0
            for key in ("differences", "uncertain"):
                keep = []
                for it in report.get(key) or []:
                    la_ = ((it.get("location") or {}).get("a") or {}).get("bbox_mm")
                    lb_ = ((it.get("location") or {}).get("b") or {}).get("bbox_mm")
                    hit = False
                    if ch["kind"] == "alignment" and it.get("dtype") == "Layout" and it.get("category") == "Moved" \
                            and it.get("page_a") == fa["page"] and la_ and _overlap(la_, ra_mm, 1.0) >= 0.5:
                        hit = True
                    if ch["kind"] == "underline" and it.get("dtype") == "Line" and it.get("category") in ("Added", "Removed"):
                        box, ref = (lb_, rb_mm) if it.get("category") == "Added" else (la_, ra_mm)
                        if box and _overlap(box, ref, 1.5) >= 0.5:
                            hit = True
                    absorbed += 1 if hit else 0
                    if not hit:
                        keep.append(it)
                report[key] = keep
            absorbed_total += absorbed
            what = ("Paragraph alignment changed from %s to %s" % (ch["old"], ch["new"])) if ch["kind"] == "alignment" else \
                   ("Underline %s" % ("added" if ch["new"] == "underlined" else "removed"))
            where = "" if ch["where"] == "body" else f" ({ch['where']})"
            dims_a, dims_b = (fa["w"] * dpi / 72, fa["h"] * dpi / 72), (fb["w"] * dpi / 72, fb["h"] * dpi / 72)
            items.append({
                "number": 0, "type": "Font / Style Difference", "dtype": "Font", "category": "Modified", "status": "confirmed",
                "page": fa["page"], "page_a": fa["page"], "page_b": fb["page"],
                "document_a": f"{ch['kind']}: {ch['old']}", "document_b": f"{ch['kind']}: {ch['new']}",
                "difference": f"{what}{where}: '{_snip(ch['text_a'])}'",
                "location": {"dpi": dpi, "a": _box(tuple(v * dpi / 72 for v in fa["bbox"]), dpi, dims_a),
                             "b": _box(tuple(v * dpi / 72 for v in fb["bbox"]), dpi, dims_b)},
                "confidence": 0.95, "engines": ["word-xml"], "evidence": ev,
                "details": {"word_property": ch["kind"], "old": ch["old"], "new": ch["new"], "text": _snip(ch["text_a"], 120),
                            "replaced_pipeline_findings": absorbed}})
        if items:
            report["differences"] = list(report.get("differences") or []) + items
            diag["word"]["format_findings"] = len(items)
            diag["word"]["pipeline_findings_replaced"] = absorbed_total
            refresh_summary(report)
        write_reports(report, jd)
        log.info("word format check: %d alignment/underline finding(s) added, %d indirect finding(s) replaced", len(items), absorbed_total)
    except Exception as e:  # noqa: BLE001 - the pipeline result stays exactly as it was
        log.warning("word format check skipped: %s: %s", type(e).__name__, e)
    return report
