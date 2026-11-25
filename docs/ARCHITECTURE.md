# DiffSight – architecture, tool selection and limitations

## Tool selection (measured, not assumed)

| Need | Chosen | Why |
|---|---|---|
| PDF text + geometry | **PyMuPDF** (native text layer, per-character boxes, fonts, embedded images, vector drawings) | Exact for digital PDFs: no OCR error, so single-character / punctuation changes are found with certainty. |
| Rasterisation | PyMuPDF at 300 DPI (configurable) | Fast (~0.1–0.3 s/page) and deterministic, so images are *re-rendered* instead of cached. |
| OCR (scans/images) | **PaddleOCR Large** PP-OCRv6 medium detection and recognition with ONNX Runtime | Scanned pages are rendered at the configured DPI, detected regions are sorted and cropped, then each crop is recognized and assembled into the existing Word/Line extraction format. |
| Image ops | OpenCV, NumPy, Pillow, scikit-image (SSIM) | Free, CPU-fast. |
| Text diff | `difflib.SequenceMatcher` (line → token → character) | Direct Original-vs-Modified comparison; `autojunk=False` for accuracy. |
| Web | FastAPI + uvicorn, process-pool worker, file-based progress | Non-blocking uploads; the event loop never runs CV code. |

No LLMs, no paid APIs, no network access at runtime.

## Pipeline

```
upload ─▶ validate ─▶ per document, per page (cached on sha256+config):
          render ─▶ orientation (scan) ─▶ deskew (scan) ─▶ background flattening
          ─▶ text: native layer | PaddleOCR Large detection/recognition ─▶ objects (stamp/handwriting/image/line/block)
page matching (DP over text/visual similarity; handles inserted, removed, moved pages)
per matched pair:
   align B→A (SIFT/ORB + RANSAC similarity, ECC refinement, NCC check)
   ├─ Text engine      line → token → char diff, direct punctuation/number/whitespace classification,
   │                    font / position / alignment / spacing for equal text
   ├─ Visual engine    tolerance-based ink XOR, colour/tint delta, SSIM per region,
   │                    local re-alignment (separates "moved" and scan jitter from real change)
   ├─ Object engine    stamp-like, handwriting-like, image, line, filled-block; A↔B matching
   │                    (incl. rotation) ⇒ added / removed / modified / moved
   └─ Overlap engine   object-over-text, overlapping words, hidden text-layer content,
                        layer-separated evidence (colour layer vs dark layer)
classification: cross-validate (OCR text diffs must be backed by an independent visual change),
                merge/de-duplicate, pair removed↔added into "moved", score, split
                confirmed / uncertain / suppressed
report: JSON + self-contained HTML + plain text, cropped A/B/overlay evidence for every item
```

## Confidence model (explicit, not learned)

* Native text difference: 0.98 (exact data, no OCR).
* OCR text difference: min(OCR confidences)·0.9, +0.08 if the visual engine independently
  sees changed ink at the same place,   ×0.8 if a second OCR reading is retained. **No visual
  support ⇒ suppressed** (counted in diagnostics, not shown as a difference).
* Visual-only region: combination of changed-ink size, SSIM drop, alignment quality and
  scan noise level.
* ≥ 0.70 → *confirmed*, 0.35–0.70 → *uncertain* (listed separately), < 0.35 → suppressed.

## Known limitations (stated up front)

1. **No 100 % guarantee.** Heavily degraded scans can hide or fake single-character changes.
2. **Stamp / signature / handwriting detection is heuristic computer vision** (colour, ring
   geometry, stroke statistics, position relative to text) – no pretrained detector is bundled
   because none can be downloaded offline. Results are reported as *stamp-like object* /
   *handwriting-like mark* with a confidence and are never identified as a specific person's
   signature.
3. Black, single-colour stamps that look like table boxes are reported with low confidence.
4. Text hidden *under opaque* objects in a scan cannot be recovered; it is reported as hidden /
   unreadable, never guessed. For digital PDFs hidden text in the text layer is detected.
5. Multi-column pages in scans are ordered by line position; heavy column interleaving can make
   an OCR diff noisy (the visual engine is unaffected and the cross-validation rule suppresses
   unsupported OCR diffs).
7. Font *size/family* changes in scans are only inferred from line height and are *uncertain*;
   in digital PDFs they are exact.
8. Whitespace-only changes (double spaces) are detected through geometry/visual evidence, not
   from text.
9. Page content that extends beyond A's page area after alignment is flagged, not compared.
