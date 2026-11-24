"""Service 1 – Document upload: streaming save, strict validation, hashing."""
from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path
from typing import BinaryIO

import pymupdf
from PIL import Image

from .config import Config
from .models import DocInfo

PDF_MAGIC = b"%PDF-"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPG_MAGIC = b"\xff\xd8\xff"

ALLOWED = ("PDF", "PNG", "JPEG / JPG")


class UploadError(ValueError):
    """Raised with a user-presentable message when an upload is rejected."""


def sniff_kind(head: bytes) -> str | None:
    """Detect the real format from magic bytes (the extension is never trusted)."""
    if PDF_MAGIC in head[:1024]:
        return "pdf"
    if head.startswith(PNG_MAGIC):
        return "png"
    if head.startswith(JPG_MAGIC):
        return "jpeg"
    return None


def safe_display_name(name: str) -> str:
    name = Path(name or "document").name
    name = re.sub(r"[^\w.\- ()]", "_", name)[:120]
    return name or "document"


def stream_to_disk(fileobj: BinaryIO, dest: Path, max_bytes: int) -> int:
    """Copy an upload to ``dest`` in chunks, aborting if it exceeds ``max_bytes``."""
    total = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as out:
        while True:
            chunk = fileobj.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                out.close()
                dest.unlink(missing_ok=True)
                raise UploadError(f"File is larger than the {max_bytes // (1024 * 1024)} MB limit.")
            out.write(chunk)
    return total


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_file(path: str | Path, display_name: str, cfg: Config) -> DocInfo:
    """Validate a file already on disk and return a :class:`DocInfo`."""
    path = Path(path)
    if not path.is_file():
        raise UploadError("Uploaded file not found.")
    size = path.stat().st_size
    if size == 0:
        raise UploadError(f"'{display_name}' is empty.")
    if size > cfg.max_file_mb * 1024 * 1024:
        raise UploadError(f"'{display_name}' exceeds the {cfg.max_file_mb} MB limit.")
    with open(path, "rb") as f:
        kind = sniff_kind(f.read(2048))
    if kind is None:
        raise UploadError(f"'{display_name}' is not a supported file. Allowed: PDF, PNG, JPG/JPEG.")

    if kind == "pdf":
        try:
            doc = pymupdf.open(str(path))
        except Exception as e:  # noqa: BLE001
            raise UploadError(f"'{display_name}' is not a valid PDF or is corrupted.") from e
        try:
            if doc.needs_pass or doc.is_encrypted:
                raise UploadError(f"'{display_name}' is password protected; remove the password first.")
            n = doc.page_count
            if n < 1:
                raise UploadError(f"'{display_name}' has no pages.")
            if n > cfg.max_pages:
                raise UploadError(f"'{display_name}' has {n} pages; the limit is {cfg.max_pages}.")
            scale = cfg.dpi / 72.0
            for i in range(min(n, 50)):
                r = doc[i].rect
                if r.width < 10 or r.height < 10 or r.width > 14400 or r.height > 14400:
                    raise UploadError(f"'{display_name}' page {i + 1} has an invalid size.")
                if r.width * r.height * scale * scale > cfg.max_pixels:
                    raise UploadError(f"'{display_name}' page {i + 1} is too large to render safely.")
            pages = n
        finally:
            doc.close()
    else:
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                w, h = im.size
                fmt = (im.format or "").upper()
        except Exception as e:  # noqa: BLE001
            raise UploadError(f"'{display_name}' is not a valid image or is corrupted.") from e
        if fmt not in ("PNG", "JPEG"):
            raise UploadError(f"'{display_name}' has unsupported image format {fmt}.")
        if w * h > cfg.max_pixels:
            raise UploadError(f"'{display_name}' is too large ({w}x{h}).")
        pages = 1

    return DocInfo(path=str(path), kind="pdf" if kind == "pdf" else "image", sha256=_sha256(path),
                   pages=pages, display_name=safe_display_name(display_name), size_bytes=size)


def save_and_validate(fileobj: BinaryIO, original_name: str, job_dir: Path, slot: str, cfg: Config) -> DocInfo:
    """Stream an upload into ``job_dir`` under a server-chosen name and validate it."""
    dest = job_dir / f"input_{slot}.bin"
    stream_to_disk(fileobj, dest, cfg.max_file_mb * 1024 * 1024)
    try:
        return validate_file(dest, original_name, cfg)
    except UploadError:
        dest.unlink(missing_ok=True)
        raise
