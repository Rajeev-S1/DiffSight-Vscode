"""Central configuration. Physical tolerances are expressed in millimetres and
converted to pixels with :meth:`Config.mm` so behaviour is DPI independent."""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIPELINE_VERSION = "1"  # bump to invalidate caches when algorithms change


@dataclass
class Config:
    # ---- rendering / limits -------------------------------------------------
    dpi: int = 300                      # render DPI for PDFs (accuracy first); chosen automatically by autodpi.py
    dpi_info: dict = field(default_factory=dict)   # how the DPI was chosen (shown in the report)
    max_pages: int = 300

    # ---- automatic DPI: scan-quality inspection (see autodpi.py; calibrated with tests/calibrate_quality.py) ----
    inspect_dpi: int = 200              # low-resolution render used only to measure the page quality
    inspect_pages: int = 6              # pages sampled per PDF (evenly spaced, first and last included)
    quality_sharp_min: float = 20.0     # edge-sharpness score below this = blurry (clean scans 30-48; blur sigma 3 px @300 DPI = 22, 4 px = 16)
    quality_noise_max: float = 3.0      # paper-noise std (grey levels) above this = noisy (clean 0-0.5, strong synthetic noise 4-5)
    quality_contrast_min: float = 120.0 # ink-to-paper contrast below this = low contrast (clean 160-230, faded 0.5x = 110)
    quality_tilt_report: float = 1.0    # tilt (degrees) above this is only reported; never changes the DPI
    quality_severe_factor: float = 0.5  # a value this far beyond its threshold = "severe": fine-detail OCR findings become uncertain
    source_low_dpi: float = 150.0       # known source resolution below this = low (report it; render at least min_render_dpi)
    min_render_dpi: int = 200           # never render below this, even for a low-resolution source
    source_render_headroom: float = 1.5 # render at most this many times the source resolution (400 DPI only if the source can use it)
    source_min_image_cover: float = 0.05  # an embedded image smaller than this share of the page is ignored for source DPI
    severe_poor_scan: bool = False      # set by autodpi when the scan is severely degraded
    max_file_mb: int = 100
    max_pixels: int = 120_000_000       # decompression-bomb guard for images
    scan_target_long_px: int = 3300     # raster inputs are normalised towards ~300 DPI A4
    scan_max_long_px: int = 4800

    # ---- OCR ----------------------------------------------------------------
    ocr_lang: str = "eng"
    ocr_psm: int = 3
    ocr_workers: int = 2
    second_opinion: bool = True         # RapidOCR on low-confidence lines
    second_opinion_conf: float = 70.0
    use_gpu: bool = bool(int(os.environ.get("NEXASCAN_GPU", "0")))
    native_min_chars: int = 25          # below this a PDF page is treated as image-only
    scan_cover_ratio: float = 0.80      # an image covering this much of the page => scan
    small_page_image_cover: float = 0.25  # a page with < native_min_chars chars is only image-only if an image
                                          # covers at least this much of it; otherwise it stays native (exact text)

    # ---- visual comparison --------------------------------------------------
    tol_mm_native: float = 0.10         # ink-matching tolerance, digital vs digital
    tol_mm_scan: float = 0.30           # when either side is a scan
    min_ink_mm2: float = 0.045          # smaller ink changes are noise (~ a full stop at 10 pt)
    local_shift_mm: float = 3.0         # window for local re-alignment of a region
    move_threshold_mm: float = 0.8      # shifts above this are reported as "moved"
    color_delta: float = 28.0           # Lab delta-E for tint / highlight changes
    min_color_area_mm2: float = 4.0

    # ---- text comparison ----------------------------------------------------
    layout_tol_mm_native: float = 0.5
    layout_tol_mm_scan: float = 2.0
    font_size_tol_pt: float = 0.15
    check_fonts: bool = True
    check_layout: bool = True

    # ---- classification -----------------------------------------------------
    confirmed_min: float = 0.70
    uncertain_min: float = 0.35
    page_match_min: float = 0.30

    # ---- cache --------------------------------------------------------------
    cache_dir: str = str(ROOT / "data" / "cache")
    use_cache: bool = True              # per-page analysis cache (default on; the API job runner turns it off)

    def mm(self, mm: float) -> float:
        return mm * self.dpi / 25.4

    def mm2(self, mm2: float) -> float:
        return mm2 * (self.dpi / 25.4) ** 2

    def to_dict(self) -> dict:
        return asdict(self)
