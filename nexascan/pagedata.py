"""Per-page analysis (render → orientation/deskew → text → objects) with a disk cache.

Only metadata is cached (words, lines, objects, orientation); pixel arrays are cheap to
re-render deterministically, which keeps the cache small and memory use low."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import ingest, ocr, preprocess
from .config import PIPELINE_VERSION, Config
from .models import DocInfo, PageData
from .objects import detect_objects
from .pagematch import make_thumb


def _cfg_key(cfg: Config) -> str:
    raw = json.dumps([PIPELINE_VERSION, cfg.dpi, cfg.native_min_chars, cfg.scan_cover_ratio, cfg.small_page_image_cover,
                      cfg.scan_target_long_px, cfg.scan_max_long_px])
    return hashlib.sha1(raw.encode()).hexdigest()[:10]


def _cache_path(info: DocInfo, index: int, cfg: Config) -> Path:
    d = Path(cfg.cache_dir) / info.sha256[:2]
    return d / f"{info.sha256}_{index}_{_cfg_key(cfg)}.json"


def _load_cache(info: DocInfo, index: int, cfg: Config) -> Optional[PageData]:
    if not cfg.use_cache:
        return None
    p = _cache_path(info, index, cfg)
    try:
        with open(p, "r", encoding="utf-8") as f:
            return PageData.from_json(json.load(f))
    except Exception:  # noqa: BLE001 - missing or corrupt cache entries are simply recomputed
        return None


def _save_cache(info: DocInfo, index: int, cfg: Config, page: PageData) -> None:
    if not cfg.use_cache:
        return
    p = _cache_path(info, index, cfg)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(page.to_json(), f)
        os.replace(tmp, p)
    except Exception:  # noqa: BLE001
        pass


def purge_doc_cache(sha256: str, cache_dir: str) -> int:
    """Delete every cached page entry (any page / config key) of one document. Returns the number removed.
    The cache holds the document's extracted text, so it must not outlive the job that created it."""
    if not sha256 or not all(c in "0123456789abcdef" for c in sha256.lower()):
        return 0
    d = Path(cache_dir) / sha256[:2]
    n = 0
    try:
        for p in d.glob(f"{sha256}_*.json"):
            try:
                p.unlink()
                n += 1
            except OSError:
                pass
        d.rmdir()          # only succeeds when the shard directory is now empty
    except OSError:
        pass
    return n


def purge_old_cache(cache_dir: str, max_age_s: float) -> int:
    """Delete cache files (and leftover temp files) not modified for ``max_age_s`` seconds."""
    import time
    root = Path(cache_dir)
    cutoff = time.time() - max_age_s
    n = 0
    if not root.is_dir():
        return 0
    for shard in root.iterdir():
        try:
            if not shard.is_dir():
                continue
            for p in shard.iterdir():
                try:
                    if p.is_file() and p.stat().st_mtime < cutoff:
                        p.unlink()
                        n += 1
                except OSError:
                    pass
            shard.rmdir()
        except OSError:
            pass
    return n


def get_prepared(info: DocInfo, page: PageData, cfg: Config) -> preprocess.Prepared:
    """Re-create the pixel layers of an analysed page."""
    bgr = ingest.load_processed(info, page.index, cfg, page.rotation_applied, page.skew_deg)
    return preprocess.prepare(bgr, scan=(page.source == "scan"))


def build_page(info: DocInfo, index: int, cfg: Config) -> PageData:
    cached = _load_cache(info, index, cfg)
    if cached is not None:
        return cached

    native = ingest.extract_native(info, index, cfg) if info.kind == "pdf" else None
    scan = True if native is None else ingest.is_scan_like(native, cfg)
    bgr, scale = ingest.render_raw(info, index, cfg)
    rotation, skew = 0, 0.0
    notes = []
    if scan:
        gray0 = preprocess.to_gray(bgr)
        rotation, conf = preprocess.estimate_orientation(gray0)
        if rotation:
            bgr = ingest.apply_orientation(bgr, rotation)
            notes.append(f"orientation corrected by {rotation} degrees (OSD confidence {conf:.1f})")
        sk, sconf = preprocess.estimate_skew(preprocess.to_gray(bgr))
        if abs(sk) >= 0.15 and sconf >= 0.03:
            skew = round(sk, 3)
            bgr = ingest.apply_skew(bgr, skew)
            notes.append(f"deskewed by {skew:+.2f} degrees")
    prep = preprocess.prepare(bgr, scan=scan)
    h, w = bgr.shape[:2]

    if not scan:
        words, lines, glyph_h = native["words"], native["lines"], native["glyph_h"]
        images, drawings, page_mm = native["images"], native["drawings"], native["page_mm"]
        source = "native"
    else:
        words, lines, glyph_h = ocr.ocr_image(prep.norm, cfg)
        images = native["images"] if native else []
        drawings = []
        page_mm = native["page_mm"] if native else (w * 25.4 / cfg.dpi, h * 25.4 / cfg.dpi)
        source = "scan"
        if native and native["n_chars"] >= cfg.native_min_chars:
            notes.append("page is a full-page image with a text layer; OCR is used instead of the text layer")

    objs = detect_objects(prep, words, glyph_h, [] if scan else images, drawings, cfg, scan)
    page = PageData(doc_sha=info.sha256, index=index, width=w, height=h, dpi=cfg.dpi, source=source,
                    rotation_applied=rotation, skew_deg=skew, scale=scale, noise_sigma=prep.noise,
                    glyph_h=float(glyph_h), ink_frac=float(prep.ink.mean()), words=words, lines=lines,
                    objects=objs, native_images=images, native_drawings=drawings, page_mm=tuple(page_mm),
                    thumb=make_thumb(prep.ink), notes=notes)
    _save_cache(info, index, cfg, page)
    return page


def ensure_ocr(info: DocInfo, page: PageData, cfg: Config, prep: Optional[preprocess.Prepared] = None) -> None:
    """OCR a native page as well (needed when it is compared with a scan)."""
    if page.ocr_words is not None:
        return
    prep = prep or get_prepared(info, page, cfg)
    words, lines, _ = ocr.ocr_image(prep.norm, cfg)
    page.ocr_words, page.ocr_lines = words, lines
    _save_cache(info, page.index, cfg, page)
