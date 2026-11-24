"""Automatic, quality-aware DPI choice: quality measures, source resolution, selection table, honesty of the report."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import synth  # noqa: E402
from nexascan import autodpi  # noqa: E402
from nexascan.config import Config  # noqa: E402

PACK = ROOT / "tests" / "data" / "pack"
CFG = Config()


@pytest.fixture(scope="module")
def clean300():
    return synth.text_page_gray(300)


def decide(a, b):
    return autodpi.choose_dpi(a, Path(a).name, b, Path(b).name, Config())


# ---------------------------------------------------------------- existing behaviour (4-value unpacking stays valid)
def test_images_get_nominal_300_and_dpi_not_applicable():
    d = PACK / "02_JPG_Scanned_OCR"
    dpi, why, _, _ = autodpi.choose_dpi(str(d / "original_lab_report.jpg"), "a.jpg", str(d / "revised_lab_report.jpg"), "b.jpg", Config())
    assert dpi == 300 and "image" in why


def test_small_digital_pdf_gets_300():
    d = PACK / "01_PDF_Text_Comparison"
    dpi, _, ia, ib = autodpi.choose_dpi(str(d / "original_invoice.pdf"), "a.pdf", str(d / "revised_invoice.pdf"), "b.pdf", Config())
    assert dpi == 300 and ia.kind == ib.kind == "pdf"


def test_retry_steps_up_once_and_stops_at_max():
    assert autodpi.retry_dpi(200) == 300 and autodpi.retry_dpi(300) == 400 and autodpi.retry_dpi(400) == 0
    assert autodpi.retry_dpi(300, cap=300) == 0                 # source-limited: no retry above what the source can give


# ---------------------------------------------------------------- Test 1-3: quality-driven selection
def test_blurred_scan_selects_400(tmp_path, clean300):
    a = synth.scan_pdf(tmp_path / "a.pdf", [synth.blur(clean300, 5.0)])
    b = synth.scan_pdf(tmp_path / "b.pdf", [synth.blur(clean300, 5.0)])
    r = decide(a, b)
    assert "blurry" in r.flags and r.dpi == 400
    assert r.quality_a.sharpness < CFG.quality_sharp_min
    assert r.message.startswith("Blurry") and "sharpness" in r.message and "→ analysed at 400 DPI" in r.message


def test_noisy_scan_selects_400(tmp_path, clean300):
    a = synth.scan_pdf(tmp_path / "a.pdf", [synth.noise(clean300, 60)])
    b = synth.scan_pdf(tmp_path / "b.pdf", [synth.noise(clean300, 60, seed=9)])
    r = decide(a, b)
    assert "noisy" in r.flags and r.dpi == 400 and r.quality_a.noise > CFG.quality_noise_max
    assert "Noisy scan (noise" in r.message


def test_low_contrast_scan_selects_400(tmp_path, clean300):
    a = synth.scan_pdf(tmp_path / "a.pdf", [synth.faded(clean300, 0.4)])
    b = synth.scan_pdf(tmp_path / "b.pdf", [synth.faded(clean300, 0.4)])
    r = decide(a, b)
    assert "low-contrast" in r.flags and r.dpi == 400
    assert r.to_dict(CFG)["contrast_label"] == "Low"


def test_clean_scanned_pdf_gets_300(tmp_path, clean300):
    a = synth.scan_pdf(tmp_path / "a.pdf", [clean300])
    b = synth.scan_pdf(tmp_path / "b.pdf", [clean300])
    r = decide(a, b)
    assert r.dpi == 300 and not r.flags and r.message.startswith("Clean scan, source 300 DPI")
    assert r.retry_dpi == 400


def test_tilt_is_reported_but_never_changes_dpi(tmp_path, clean300):
    import cv2
    h, w = clean300.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 2.5, 1.0)
    tilted = cv2.warpAffine(clean300, M, (w, h), borderValue=255)
    a = synth.scan_pdf(tmp_path / "a.pdf", [tilted])
    b = synth.scan_pdf(tmp_path / "b.pdf", [tilted])
    r = decide(a, b)
    d = r.to_dict(CFG)
    assert abs(d["tilt"]) > 1.0 and d["tilt_label"].startswith("Tilted") and "blurry" not in r.flags and r.dpi == 300


# ---------------------------------------------------------------- Test 3: low source resolution is reported, never "restored"
def test_low_dpi_image_is_reported_not_upscaled(tmp_path, clean300):
    low = synth.at_resolution(clean300, 100)
    a = synth.image_file(tmp_path / "a.jpg", low, dpi=(100, 100))
    b = synth.image_file(tmp_path / "b.jpg", low, dpi=(100, 100))
    r = decide(a, b)
    assert r.source_dpi_a == 100 and r.applicable is False and r.retry_dpi == 0
    assert "source 100 DPI" in r.message and "Source resolution is low" in r.message
    assert "does not restore missing detail" in r.message and "restores" not in r.message.replace("does not restore", "")


def test_low_dpi_scan_pdf_renders_at_minimum_200_not_400(tmp_path, clean300):
    low = synth.blur(synth.at_resolution(clean300, 100), 1.6)           # blurry AND only 100 DPI: 400 would be pointless
    a = synth.scan_pdf(tmp_path / "a.pdf", [low])
    b = synth.scan_pdf(tmp_path / "b.pdf", [low])
    r = decide(a, b)
    assert r.source_dpi_a == pytest.approx(100, abs=2)
    assert r.dpi == 200 and r.retry_dpi == 0                           # never below 200, never above what the source supports
    assert "Source resolution is low" in r.message


def test_mid_resolution_source_caps_the_render(tmp_path, clean300):
    mid = synth.blur(synth.at_resolution(clean300, 200), 3.0)           # blurry 200 DPI scan: 400 not allowed, 300 is
    a = synth.scan_pdf(tmp_path / "a.pdf", [mid])
    b = synth.scan_pdf(tmp_path / "b.pdf", [mid])
    r = decide(a, b)
    assert r.dpi == 300 and any("limited to 300" in n for n in r.notes)


# ---------------------------------------------------------------- source DPI measurement
def test_image_metadata_variants(tmp_path, clean300):
    g = synth.at_resolution(clean300, 150)
    assert autodpi.image_source_dpi(synth.image_file(tmp_path / "ok.png", g, dpi=(200, 200)))[0] == pytest.approx(200, abs=1)
    assert autodpi.image_source_dpi(synth.image_file(tmp_path / "ok.jpg", g, dpi=(300, 300)))[0] == pytest.approx(300, abs=1)
    none, why = autodpi.image_source_dpi(synth.image_file(tmp_path / "none.png", g))              # missing
    assert none is None and "no resolution" in why
    for name, dpi in (("huge", (9000, 9000)), ("mixed", (100, 300)), ("tiny", (5, 5))):          # invalid / inconsistent
        v, why = autodpi.image_source_dpi(synth.image_file(tmp_path / f"{name}.png", g, dpi=dpi))
        assert v is None and "unknown" in why
    v, why = autodpi.image_source_dpi(synth.image_file(tmp_path / "scr.png", g, dpi=(72, 72)))
    assert v is None and "screen default" in why                                                  # 72/96 tags are not a scan resolution


def test_missing_metadata_uses_quality_and_says_so(tmp_path, clean300):
    a = synth.image_file(tmp_path / "a.png", clean300)
    b = synth.image_file(tmp_path / "b.png", clean300)
    r = decide(a, b)
    assert r.source_dpi_a is None and "Source resolution is unavailable" in r.message and r.dpi == 300


def test_mixed_resolution_pdf_worst_scanned_page_decides(tmp_path, clean300):
    a = synth.scan_pdf(tmp_path / "a.pdf", [clean300, synth.at_resolution(clean300, 120)])
    b = synth.scan_pdf(tmp_path / "b.pdf", [clean300, synth.at_resolution(clean300, 120)])
    r = decide(a, b)
    assert r.quality_a.source_type == "scanned PDF" and r.source_dpi_a == pytest.approx(120, abs=2)
    pages = [p["source_dpi"] for p in r.quality_a.pages]
    assert pages[0] == pytest.approx(300, abs=2) and pages[1] == pytest.approx(120, abs=2)   # per-page values are kept
    assert r.dpi == 200                                                                      # one DPI per job: the worst page


def test_vector_text_with_small_image_is_not_a_scan(tmp_path):
    a = synth.digital_pdf(tmp_path / "a.pdf", logo=True)
    b = synth.digital_pdf(tmp_path / "b.pdf", logo=True, text="Amount payable: 1500")
    r = decide(a, b)
    assert r.quality_a.source_type == "digital PDF" and r.source_dpi_a is None
    assert "not applicable" in r.to_dict(CFG)["source_dpi_status"] and r.dpi == 300 and not r.flags


def test_mixed_pdf_digital_and_scan_pages(tmp_path, clean300):
    import fitz
    d = fitz.open()
    pg = d.new_page(width=synth.A4_W, height=synth.A4_H)
    pg.insert_text((72, 100), "Digital page", fontsize=12)
    pg = d.new_page(width=synth.A4_W, height=synth.A4_H)
    pg.insert_image(pg.rect, stream=synth._png(clean300))
    p = tmp_path / "mixed.pdf"
    d.save(str(p))
    r = decide(str(p), str(p))
    assert r.quality_a.source_type == "mixed PDF" and r.quality_a.scan_pages == 1


def test_digital_pdf_pair_300_and_large_pair_200(tmp_path):
    a, b = synth.digital_pdf(tmp_path / "a.pdf", 2), synth.digital_pdf(tmp_path / "b.pdf", 2)
    assert decide(a, b).dpi == 300
    a, b = synth.digital_pdf(tmp_path / "A.pdf", 12), synth.digital_pdf(tmp_path / "B.pdf", 12, text="Amount payable: 1500")
    r = decide(a, b)
    assert r.dpi == 200 and "Large clean digital PDF pair (24 pages)" in r.message


def test_uninspectable_page_does_not_crash(tmp_path, monkeypatch, clean300):
    a = synth.scan_pdf(tmp_path / "a.pdf", [clean300])
    monkeypatch.setattr(autodpi, "measure_gray", lambda g: (_ for _ in ()).throw(RuntimeError("boom")))
    r = decide(a, a)
    assert r.dpi == 300 and r.quality_a.uninspectable == 1 and r.quality_a.sharpness is None   # safe default, still reported


def test_page_too_large_for_400_is_reduced(tmp_path, clean300):
    a = synth.scan_pdf(tmp_path / "a.pdf", [synth.blur(clean300, 5.0)])
    cfg = Config()
    cfg.max_pixels = 10_000_000                                             # A4: 300 DPI = 8.7 MP fits, 400 DPI = 15.5 MP does not
    r = autodpi.choose_dpi(a, "a.pdf", a, "b.pdf", cfg)
    assert r.dpi == 300 and any("Reduced from 400 to 300" in n for n in r.notes)


# ---------------------------------------------------------------- calibration: clean pages never go to 400
def test_clean_pack_documents_never_get_400():
    for folder, a, b in (("01_PDF_Text_Comparison", "original_invoice.pdf", "revised_invoice.pdf"),
                         ("02_JPG_Scanned_OCR", "original_lab_report.jpg", "revised_lab_report.jpg"),
                         ("03_PNG_Visual_Layout", "original_delivery.png", "revised_delivery.png"),
                         ("04_PDF_Multipage_Page_Changes", "original_agreement.pdf", "revised_agreement.pdf")):
        r = decide(str(PACK / folder / a), str(PACK / folder / b))
        assert r.dpi == 300 and r.retry_dpi in (0, 400), folder


@pytest.mark.skipif(not (ROOT / "data" / "suite").exists(), reason="run tests/make_suite.py first")
def test_clean_suite_scans_never_get_400():
    bad = []
    for d in sorted((ROOT / "data" / "suite").glob("[sd]*")):
        a, b = next(iter(d.glob("A.*")), None), next(iter(d.glob("B.*")), None)
        if a is None or b is None or d.name == "s09_scan_low_quality":
            continue
        r = decide(str(a), str(b))
        if r.dpi == 400:
            bad.append((d.name, r.message))
    assert not bad, bad


# ---------------------------------------------------------------- frontend never asks for a DPI
def test_frontend_has_no_dpi_control():
    api_js = (ROOT / "frontend" / "src" / "api.js").read_text(encoding="utf-8")
    app = (ROOT / "frontend" / "src" / "App.jsx").read_text(encoding="utf-8")
    assert 'append("dpi"' not in api_js and 'type="checkbox"' not in app and "setAccurate" not in app


def test_blank_page_is_not_called_blurry(tmp_path, clean300):
    import numpy as np
    blank = np.full_like(clean300, 255)
    a = synth.scan_pdf(tmp_path / "a.pdf", [clean300, blank])
    r = decide(a, a)
    assert not r.flags and r.dpi == 300 and r.quality_a.uninspectable == 0
    assert r.quality_a.pages[1].get("blank") is True and r.quality_a.sharpness is not None
