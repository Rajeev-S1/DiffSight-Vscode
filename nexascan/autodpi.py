"""Automatic, quality-aware resolution choice. The user never picks a DPI.

What is measured (per sampled page, then the worst page of each document counts):
  * source type      digital PDF / scanned PDF / mixed PDF / image            (ingest.is_scan_like, as in the pipeline)
  * sharpness        edge sharpness score (preprocess.estimate_sharpness)      low  = blurry
  * noise            paper noise, grey levels (preprocess.estimate_noise)      high = noisy
  * contrast         ink-to-paper contrast (preprocess.estimate_contrast)      low  = faded
  * tilt             degrees (preprocess.estimate_skew)  - reported only; tilt is corrected later and is NOT blur
  * source DPI       images: file metadata (JFIF / PNG pHYS / EXIF), checked for plausibility; unknown stays unknown.
                     PDF scan pages: pixel size of each embedded image / its placed size on the page.
                     Digital PDFs (vector text, no page-sized image): "not applicable".

Selection (all thresholds live in config.py):
  digital PDF pair            300 DPI, or 200 DPI for a large pair (> LARGE_PAIR_PAGES pages)
  clean scan                  300 DPI
  blurry / noisy / faded scan 400 DPI, but never more than source_render_headroom x the source resolution
  known source < source_low_dpi  render at min_render_dpi (200); report that rendering higher cannot restore detail
  unknown source              quality only, safe default
  images (JPG/PNG)            processed at their own pixel size (ingest._image_scale): DPI is not applicable
DPI is a rendering resolution. It never creates detail that the source does not contain.
The pipeline uses one DPI per job; for a mixed-resolution PDF the worst scanned page decides.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

from . import ingest, preprocess, upload
from .config import Config
from .models import DocInfo

LARGE_PAIR_PAGES = 20      # pages of A + B above which a clean digital pair is compared at 200 DPI
MAX_DPI = 400
BLANK_CONTRAST = 25.0       # ink-to-paper contrast below this: the page is blank (it cannot be blurry or noisy)
DPI_STEPS = (200, 300, 400)
DEFAULT_TAGS = (72.0, 96.0)   # screen defaults that many programs write into images: not a real scan resolution


# ------------------------------------------------------------------ source resolution

def _num(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return f if math.isfinite(f) else None


def image_source_dpi(path: str) -> Tuple[Optional[float], str]:
    """Stored resolution of a JPG/PNG: (dpi or None, status). Never invents a value."""
    try:
        with Image.open(path) as im:
            raw = im.info.get("dpi")
            if raw is None:                                        # EXIF XResolution / YResolution / ResolutionUnit
                try:
                    ex = im.getexif()
                    xr, yr, unit = _num(ex.get(282)), _num(ex.get(283)), ex.get(296, 2)
                    if xr and yr:
                        k = 2.54 if unit == 3 else 1.0
                        raw = (xr * k, yr * k)
                except Exception:  # noqa: BLE001
                    raw = None
    except Exception:  # noqa: BLE001
        return None, "unknown (metadata unreadable)"
    if raw is None:
        return None, "unknown (no resolution stored in the file)"
    if not isinstance(raw, (tuple, list)):
        raw = (raw, raw)
    x, y = _num(raw[0]), _num(raw[1] if len(raw) > 1 else raw[0])
    if not x or not y or x <= 0 or y <= 0 or x > 4800 or y > 4800 or x < 20 or y < 20:
        return None, "unknown (invalid resolution metadata ignored)"
    if max(x, y) / min(x, y) > 1.25:
        return None, "unknown (inconsistent horizontal / vertical resolution ignored)"
    d = min(x, y)
    if any(abs(d - t) < 0.6 for t in DEFAULT_TAGS):
        return None, f"unknown (the {d:.0f} DPI tag is a screen default, not a scan resolution)"
    return round(d, 1), "stored in the image file"


def pdf_page_source_dpi(info: DocInfo, index: int, cfg: Config) -> Optional[float]:
    """Effective resolution of the embedded scan image(s) on one PDF page, from pixel size and placed size.
    Handles rotation/scaling (placement matrix) and several images (tiles); small images (logos) are ignored."""
    with ingest._LOCK:
        page = ingest._open(info.path)[index]
        prect = page.rect
        infos = page.get_image_info(xrefs=True)
    page_area = max(1.0, prect.width * prect.height)
    vals: List[float] = []
    for im in infos:
        w_px, h_px = _num(im.get("width")), _num(im.get("height"))
        if not w_px or not h_px:
            continue
        x0, y0, x1, y1 = im["bbox"]
        area = max(0.0, min(x1, prect.x1) - max(x0, prect.x0)) * max(0.0, min(y1, prect.y1) - max(y0, prect.y0))
        if area / page_area < cfg.source_min_image_cover:
            continue
        tf = im.get("transform")
        if tf and len(tf) >= 4:                                    # unit square -> page: exact size even when rotated
            w_pt, h_pt = math.hypot(tf[0], tf[1]), math.hypot(tf[2], tf[3])
        else:
            w_pt, h_pt = abs(x1 - x0), abs(y1 - y0)
        if w_pt <= 0 or h_pt <= 0:
            continue
        vals.append(min(w_px / (w_pt / 72.0), h_px / (h_pt / 72.0)))
    return round(min(vals), 1) if vals else None


# ------------------------------------------------------------------ quality

@dataclass
class DocQuality:
    source_type: str = "unknown"          # digital PDF | scanned PDF | mixed PDF | image
    pages_sampled: int = 0
    scan_pages: int = 0
    uninspectable: int = 0
    sharpness: Optional[float] = None     # worst (lowest) scan page
    noise: Optional[float] = None         # worst (highest)
    contrast: Optional[float] = None      # worst (lowest)
    tilt: Optional[float] = None          # worst absolute tilt, degrees
    source_dpi: Optional[float] = None    # worst (lowest) known source resolution
    source_dpi_status: str = "not applicable"
    pages: List[dict] = field(default_factory=list)

    def flags(self, cfg: Config) -> List[str]:
        out = []
        if self.sharpness is not None and self.sharpness < cfg.quality_sharp_min:
            out.append("blurry")
        if self.noise is not None and self.noise > cfg.quality_noise_max:
            out.append("noisy")
        if self.contrast is not None and self.contrast < cfg.quality_contrast_min:
            out.append("low-contrast")
        return out

    def severe(self, cfg: Config) -> bool:
        f = cfg.quality_severe_factor
        return bool((self.sharpness is not None and self.sharpness < cfg.quality_sharp_min * f)
                    or (self.noise is not None and self.noise > cfg.quality_noise_max / f)
                    or (self.contrast is not None and self.contrast < cfg.quality_contrast_min * f))


def measure_gray(gray: np.ndarray) -> dict:
    """Sharpness, noise, contrast and tilt of one page image (grey, uint8). Reuses the preprocess estimators."""
    if preprocess.estimate_contrast(gray) < BLANK_CONTRAST:        # blank / no readable ink: nothing to judge
        return {"blank": True}
    tilt, _ = preprocess.estimate_skew(gray)
    return {"sharpness": round(preprocess.estimate_sharpness(gray), 1), "noise": round(preprocess.estimate_noise(gray), 2),
            "contrast": round(preprocess.estimate_contrast(gray), 1), "tilt": round(float(tilt), 2)}


def _sample(n: int, k: int) -> List[int]:
    if n <= 0:
        return []
    if n <= k:
        return list(range(n))
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def _aggregate(q: DocQuality, rows: List[dict]) -> None:
    ok = [r for r in rows if "sharpness" in r]
    q.scan_pages = len(rows)
    q.uninspectable = len([r for r in rows if "sharpness" not in r and not r.get("blank")])
    if ok:
        q.sharpness = min(r["sharpness"] for r in ok)
        q.noise = max(r["noise"] for r in ok)
        q.contrast = min(r["contrast"] for r in ok)
        q.tilt = max((r["tilt"] for r in ok), key=abs)


def analyse_image(info: DocInfo, cfg: Config) -> DocQuality:
    q = DocQuality(source_type="image", pages_sampled=1)
    row: dict = {"page": 1}
    try:
        with Image.open(info.path) as im:
            im = ImageOps.exif_transpose(im).convert("L")
            gray = np.asarray(im).copy()
        row.update(measure_gray(gray))
    except Exception as e:  # noqa: BLE001 - an image that cannot be inspected must not stop the job
        row["error"] = f"{type(e).__name__}"
    dpi, status = image_source_dpi(info.path)
    row["source_dpi"] = dpi
    q.pages = [row]
    _aggregate(q, [row])
    q.source_dpi, q.source_dpi_status = dpi, status
    return q


def analyse_pdf(info: DocInfo, cfg: Config) -> DocQuality:
    q = DocQuality()
    idxs = _sample(info.pages, cfg.inspect_pages)
    q.pages_sampled = len(idxs)
    rows: List[dict] = []
    dpis: List[float] = []
    icfg = copy.copy(cfg)
    icfg.dpi = cfg.inspect_dpi
    for idx in idxs:
        row = {"page": idx + 1, "scan": False}
        try:
            nat = ingest.extract_native(info, idx, cfg)
            row["scan"] = bool(ingest.is_scan_like(nat, cfg))
            if row["scan"]:
                src = pdf_page_source_dpi(info, idx, cfg)
                row["source_dpi"] = src
                if src is not None:
                    dpis.append(src)
                bgr, _ = ingest.render_raw(info, idx, icfg)
                row.update(measure_gray(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)))
        except Exception as e:  # noqa: BLE001 - never crash on a page that cannot be inspected
            row["error"] = f"{type(e).__name__}"
        q.pages.append(row)
        if row["scan"]:
            rows.append(row)
    _aggregate(q, rows)
    n_scan = len(rows)
    q.source_type = "digital PDF" if n_scan == 0 else ("scanned PDF" if n_scan == len(idxs) else "mixed PDF")
    if n_scan == 0:
        q.source_dpi_status = "not applicable (digital PDF with a text layer)"
    elif dpis:
        q.source_dpi, q.source_dpi_status = min(dpis), "measured from the embedded scan images"
    else:
        q.source_dpi_status = "unknown (no embedded image resolution found)"
    return q


def analyse(info: DocInfo, cfg: Config) -> DocQuality:
    return analyse_image(info, cfg) if info.kind == "image" else analyse_pdf(info, cfg)


# ------------------------------------------------------------------ decision

@dataclass
class DpiDecision:
    dpi: int
    reason: str
    info_a: DocInfo
    info_b: DocInfo
    quality_a: DocQuality
    quality_b: DocQuality
    source_dpi_a: Optional[float] = None
    source_dpi_b: Optional[float] = None
    applicable: bool = True            # False when both files are images (processed at their own pixel size)
    retry_dpi: int = 0                 # 0 = no retry possible / useful
    severe: bool = False
    message: str = ""
    notes: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)

    def __iter__(self):                # backward compatible: dpi, reason, info_a, info_b = choose_dpi(...)
        return iter((self.dpi, self.reason, self.info_a, self.info_b))

    def to_dict(self, cfg: Config) -> dict:
        def pick(f, worst):
            vals = [v for v in (getattr(self.quality_a, f), getattr(self.quality_b, f)) if v is not None]
            return round(worst(vals), 2) if vals else None
        sharp, noise = pick("sharpness", min), pick("noise", max)
        contrast = pick("contrast", min)
        tilts = [v for v in (self.quality_a.tilt, self.quality_b.tilt) if v is not None]
        tilt = max(tilts, key=abs) if tilts else None
        types = [self.quality_a.source_type, self.quality_b.source_type]
        src = [d for d in (self.source_dpi_a, self.source_dpi_b) if d is not None]
        statuses = [self.quality_a.source_dpi_status, self.quality_b.source_dpi_status]
        return {
            "mode": "auto", "used": self.dpi, "selected_dpi": self.dpi, "dpi_applicable": self.applicable,
            "reason": self.reason, "message": self.message, "notes": list(self.notes),
            "source_type": types[0] if types[0] == types[1] else f"{types[0]} / {types[1]}",
            "source_type_a": types[0], "source_type_b": types[1],
            "sharpness": sharp, "sharpness_label": None if sharp is None else ("Low" if sharp < cfg.quality_sharp_min else "Good"),
            "noise": noise, "noise_label": None if noise is None else ("High" if noise > cfg.quality_noise_max else "Low"),
            "contrast": contrast, "contrast_label": None if contrast is None else ("Low" if contrast < cfg.quality_contrast_min else "Normal"),
            "tilt": tilt, "tilt_label": None if tilt is None else ("Tilted (corrected automatically)" if abs(tilt) > cfg.quality_tilt_report else "Straight"),
            "source_dpi": min(src) if src else None,
            "source_dpi_a": self.source_dpi_a, "source_dpi_b": self.source_dpi_b,
            "source_dpi_status": statuses[0] if statuses[0] == statuses[1] else f"A: {statuses[0]}; B: {statuses[1]}",
            "quality_flags": list(self.flags), "severe_quality": self.severe,
            "retried": False, "retry_result": "not_needed", "kept": "first",
        }


def fit_dpi(infos: List[DocInfo], dpi: int, cfg: Config) -> Tuple[int, bool]:
    """Highest DPI <= dpi at which every PDF page stays inside cfg.max_pixels. Returns (dpi, was_reduced)."""
    steps = [d for d in DPI_STEPS if d <= dpi][::-1] or [DPI_STEPS[0]]
    for d in steps:
        c = copy.copy(cfg)
        c.dpi = d
        try:
            for i in infos:
                if i.kind == "pdf":
                    upload.validate_file(i.path, i.display_name, c)
            return d, d != dpi
        except upload.UploadError:
            continue
    return steps[-1], steps[-1] != dpi


def retry_dpi(first_dpi: int, cap: int = MAX_DPI) -> int:
    """Next higher resolution for the single retry (0 = already at the maximum / at the source limit)."""
    nxt = 300 if first_dpi < 300 else MAX_DPI
    return nxt if (nxt > first_dpi and nxt <= cap) else 0


def _source_cap(source_dpi: Optional[float], cfg: Config) -> int:
    """Highest useful render DPI for a known source resolution (never below min_render_dpi)."""
    if source_dpi is None:
        return MAX_DPI
    limit = source_dpi * cfg.source_render_headroom
    ok = [d for d in DPI_STEPS if d <= limit]
    return max(cfg.min_render_dpi, ok[-1] if ok else cfg.min_render_dpi)


def _num_txt(v, nd=0):
    return f"{v:.{nd}f}"


def _quality_phrase(flags, sharp, noise, contrast, scanned: bool) -> str:
    noun = "scan" if scanned else "page"
    if not flags:
        return f"Clean {noun}"
    parts = {"blurry": f"sharpness {_num_txt(sharp)}", "noisy": f"noise {_num_txt(noise, 1)}", "low-contrast": f"contrast {_num_txt(contrast)}"}
    names = {"blurry": "Blurry", "noisy": "Noisy", "low-contrast": "Low-contrast"}
    head = " and ".join(names[f].lower() if i else names[f] for i, f in enumerate(flags))
    return f"{head} {noun} ({', '.join(parts[f] for f in flags)})"


def choose_dpi(path_a: str, name_a: str, path_b: str, name_b: str, cfg: Config) -> DpiDecision:
    ia = upload.validate_file(path_a, name_a, cfg)
    ib = upload.validate_file(path_b, name_b, cfg)
    qa, qb = analyse(ia, cfg), analyse(ib, cfg)
    notes: List[str] = []
    pdf_scan = [q for q, i in ((qa, ia), (qb, ib)) if i.kind == "pdf" and q.scan_pages > 0]
    any_scan_like = bool(pdf_scan) or ia.kind == "image" or ib.kind == "image"
    applicable = ia.kind == "pdf" or ib.kind == "pdf"
    flags_all: List[str] = []
    for q in (qa, qb):
        for f in q.flags(cfg):
            if f not in flags_all:
                flags_all.append(f)
    dpi_flags: List[str] = []
    for q in pdf_scan:
        for f in q.flags(cfg):
            if f not in dpi_flags:
                dpi_flags.append(f)
    severe = qa.severe(cfg) or qb.severe(cfg)
    src_all = [d for d in (qa.source_dpi, qb.source_dpi) if d is not None]
    src_pdf = [q.source_dpi for q in pdf_scan if q.source_dpi is not None]
    src_for_dpi = min(src_pdf) if src_pdf else None
    sharp = min([q.sharpness for q in (qa, qb) if q.sharpness is not None], default=None)
    noise = max([q.noise for q in (qa, qb) if q.noise is not None], default=None)
    contrast = min([q.contrast for q in (qa, qb) if q.contrast is not None], default=None)

    low_source = bool(src_all) and min(src_all) < cfg.source_low_dpi
    total_pages = ia.pages + ib.pages
    cap = MAX_DPI

    if not any_scan_like:                                   # clean digital pair
        if total_pages > LARGE_PAIR_PAGES:
            dpi, reason = 200, f"large digital PDF pair ({total_pages} pages, exact text layer): faster setting"
            head = f"Large clean digital PDF pair ({total_pages} pages)"
        else:
            dpi, reason = 300, "digital PDF: exact text layer, standard resolution"
            head = "Clean digital PDF"
    elif not applicable:                                    # JPG/PNG against JPG/PNG
        dpi = 300
        reason = "image upload: processed at its own pixel size (DPI is not applicable; 300 is only the nominal value for millimetre tolerances)"
        head = _quality_phrase(flags_all, sharp, noise, contrast, True)
    else:
        if not pdf_scan:                                    # digital PDF compared with an image
            dpi, reason = 300, "digital PDF compared with an image: standard resolution (the image keeps its own pixel size)"
            head = _quality_phrase(flags_all, sharp, noise, contrast, True)
        else:
            want = 400 if dpi_flags else 300
            cap = _source_cap(src_for_dpi, cfg)
            dpi = min(want, cap)
            if low_source and src_for_dpi is not None and src_for_dpi < cfg.source_low_dpi:
                dpi = cfg.min_render_dpi
            if dpi < want:
                notes.append(f"Rendering is limited to {dpi} DPI because the source scan is only about {src_for_dpi:.0f} DPI.")
            reason = (f"{', '.join(dpi_flags)} scan: higher resolution" if dpi_flags else "scan without quality problems: standard resolution")
            head = _quality_phrase(dpi_flags or [], sharp, noise, contrast, True)

    fitted, reduced = fit_dpi([ia, ib], dpi, cfg) if applicable else (dpi, False)
    if reduced:
        notes.append(f"Reduced from {dpi} to {fitted} DPI because a page would exceed the pixel limit at {dpi} DPI.")
        dpi = fitted

    msg = head
    shown_src = min(src_all) if src_all else None
    if shown_src is not None:
        msg += f", source {shown_src:.0f} DPI"
    arrow = f" → analysed at {dpi} DPI" if applicable else " → processed at the image's own pixel size (DPI not applicable)"
    msg += arrow + "."
    if low_source:
        msg += " Source resolution is low; higher rendering resolution does not restore missing detail."
        notes.append("Fine details (single dots, small marks) may be missed because the source itself is low resolution.")
    elif shown_src is None and any_scan_like:
        msg += " Source resolution is unavailable, so the quality measurements were used."
    for n in notes:
        if n not in msg:
            msg += " " + n
    if flags_all and not dpi_flags and applicable and not pdf_scan:
        notes.append("The image is soft/noisy/faded; findings that depend on fine detail are less reliable.")
    r_cap = min(cap, _source_cap(src_for_dpi, cfg)) if src_for_dpi is not None else cap
    r_dpi = 0 if (not applicable or low_source) else retry_dpi(dpi, r_cap)
    return DpiDecision(dpi=dpi, reason=reason, info_a=ia, info_b=ib, quality_a=qa, quality_b=qb,
                       source_dpi_a=qa.source_dpi, source_dpi_b=qb.source_dpi, applicable=applicable,
                       retry_dpi=r_dpi, severe=severe, message=msg.strip(), notes=notes, flags=flags_all)
