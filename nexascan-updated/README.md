# DiffSight — Every Detail. Every Difference.

Document comparison (PDF / JPG / PNG) using only free, open-source tools. No paid APIs, no LLMs.
Text (native layer + OCR) and pixels are compared by independent engines; uncertain findings are
reported separately from confirmed ones. Signatures are compared by appearance only.

## Install
    pip install -r requirements.txt --break-system-packages

## Resolution (DPI) is automatic and quality-aware
The user is never asked for a DPI. `nexascan/autodpi.py` inspects both documents first:
- **Source type** (digital PDF / scanned PDF / mixed PDF / image) and, per sampled page, **sharpness, noise, contrast and tilt** (reusing `preprocess.py`). The worst scanned page of each document counts. Tilt is only reported (it is corrected later and is not blur).
- **Source resolution:** JPG/PNG - the stored metadata, checked for plausibility (invalid, inconsistent and 72/96 screen-default values are ignored; unknown stays unknown). PDF - the pixel size of each embedded scan image divided by its placed size on the page. Digital PDFs: not applicable.
- **Choice:** clean digital PDF 300 DPI (200 for a large pair, > 20 pages in total); clean scan 300; blurry / noisy / low-contrast scan 400, never more than 1.5x the source resolution; a known source below 150 DPI is rendered at 200 DPI (never lower) and reported as low resolution - higher rendering resolution does not restore missing detail. A 400 DPI render that would exceed the pixel limit is reduced and the report says so.
- **JPG/PNG** are processed at their own pixel size (`ingest._image_scale`), so DPI is "not applicable" for them and there is no DPI retry. A mixed-resolution PDF uses one DPI per job: its worst scanned page decides (per-page values are kept in the report).
- All thresholds are in `config.py` and were chosen with `python tests/calibrate_quality.py`.

**Retry** (`nexascan/retry.py`, `worker.retry_stage`): if uncertain findings remain, the job is repeated once at a higher DPI. The retry is kept if it has fewer uncertain findings, or - on a tie - if the stable findings (same type, page, place in mm and description; never confidence alone) have higher average confidence. A confirmed finding that appears only at the higher DPI is listed separately as `seen_only_at_higher_resolution` for manual review (not counted as verified); a confirmed finding that only the original run had becomes uncertain instead of vanishing. A failed retry keeps the original result. Confidence is never raised because the DPI is higher.

The report (`diagnostics.dpi_selection`, text, HTML and the web page) shows the measured values, the DPI chosen and why, and the retry outcome. `POST /api/compare` still accepts an optional `dpi` field (150-400) that switches the automatic choice off (advanced use).

## Word documents (.docx and .doc)
DiffSight compares Microsoft Word files next to PDF, JPG and PNG. Upload two Word files (or a Word file and a PDF) exactly like any other pair.
- **How:** `nexascan/wordsupport.py` is a separate adapter. The file is checked by its content (not its extension), a `.doc` is normalised to `.docx`, and LibreOffice (headless) renders it to PDF. That PDF goes through the unchanged comparison pipeline, so text, numbers, punctuation, table cells, headers, footers, page content, fonts, sizes, bold and italic are compared by the existing algorithms and appear in the usual report.
- **Alignment and underline:** the pipeline can only describe these indirectly ("1 line moved", "ruled line added"). After the comparison, `augment_report` reads them from the Word file and adds explicit findings ("Paragraph alignment changed from left to centered", "Underline added"), replacing the indirect findings they explain. This step runs only when both inputs are Word files; any problem in it leaves the normal result unchanged.
- **Needs:** LibreOffice on the machine (`winget install -e --id TheDocumentFoundation.LibreOffice` or `sudo apt install libreoffice-writer`) plus `python-docx` and `olefile` (in `requirements.txt`). PDF, JPG and PNG work without it. `NEXASCAN_SOFFICE` sets a custom LibreOffice path, `NEXASCAN_WORD_TIMEOUT` the conversion limit in seconds (default 180).
- **Rejected with a clear message (HTTP 400):** corrupted files, password-protected files, files that are not really Word (also RTF/HTML saved as `.doc`), empty documents, and unsupported types. A missing LibreOffice also gives a clear message.
- **Limits:** the layout is LibreOffice's rendering of the document, which can differ slightly from Microsoft Word (fonts that are not installed, page breaks). Comments and tracked changes are shown as rendered. Macros (`.docm`), RTF and Word templates are not supported. Word findings carry the file kind `pdf` in the report, because that is what is compared.

## Web UI
A top bar and one page, no side panel. After a comparison the default view is a table with **one row per difference**: Type (number, title and category), Page, Document A, Document B, Difference, Location (A), Location (B) (same wording as the HTML report), Confidence (with the engines) and, in the last column, **View evidence**, which opens the evidence viewer (side by side, overlay slider, highlighted). Tabs separate confirmed, uncertain and higher-resolution-only findings; search, page and category filters narrow the table. The HTML report, single-file HTML, text and JSON downloads are the backend's unchanged files (buttons at the top of the results). The scan / resolution details and the "verified unchanged" table are collapsible blocks.

## PaddleOCR Large extraction and the progress-file fix
- **Scanned extraction** (`nexascan/ocr.py`) lazily initializes the PP-OCRv6 medium detection and recognition models with ONNX Runtime, matching the detection/crop/recognition workflow in `large.py`.
- The existing PyMuPDF digital-text path remains unchanged. Scanned PDFs and native-page OCR fallback use PaddleOCR.
- **WinError 5 fix** (`nexascan/worker.py`): on Windows a file that another process has open cannot be replaced. The web page's status poll reads `progress.json` while the worker replaces it, so `os.replace` could fail with `PermissionError`. `write_json_atomic` now writes a unique, closed temporary file, retries the replace for those transient errors (12 attempts, up to about 3.5 s), falls back to an in-place write, cleans up temporary files, serialises writes within a process and logs every problem; `read_json` retries too. An unwritable progress update no longer aborts the comparison (it is logged as an error); the final status writes still raise.
- Checks for a real Windows PC: `tests\run_windows_checks.bat` and `docs/WINDOWS_CHECKLIST.md`.

## Log files (folder `logs/`)
| File | Written by | Behaviour |
|---|---|---|
| `logs/setup.log` | `setup.bat` | every setup step, versions, pip/npm output and errors; a new run overwrites it |
| `logs/start.log` | `start.bat` | checks, port, what was launched, backend health result; a new start overwrites it |
| `logs/nexascan.log` | the running product (`nexascan/applog.py`) | start-up, every job (queued, resolution chosen, first run, re-check, done or failed with traceback), rejected uploads, clean-up, shutdown. Appended across runs; at start-up it is renamed to `nexascan.log.1` when larger than 5 MB. It never contains document content or file names (job ids are shortened). `NEXASCAN_LOG_DIR` changes the folder. |

## Run on Windows (easiest)
1. Install Python 3.11+ and Node.js LTS.
2. Double-click `setup.bat` once (creates `venv`, installs Python packages and the web UI).
3. Double-click `start.bat` any time: it opens two windows (backend on 38219, frontend on 34517) and the browser.

## Run (any system, manual)
Needs Python 3.11+ and Node.js 18+ (only to build the web UI once).

    cd frontend && npm install && npm run build && cd ..     # builds the React app into frontend/dist
    python run.py                                            # open http://localhost:38219

One port serves both the API and the web UI. Default 38219; if busy, the next free port in 38200-38229 is used.
Nothing else on the machine is stopped or changed. Set `NEXASCAN_PORT` / `NEXASCAN_HOST` to override
(host defaults to 127.0.0.1). Access log is off; `NEXASCAN_LOG_ACCESS=1` turns it on.

Frontend development (hot reload): run `python run.py`, then `cd frontend && npm run dev` (port 34517, or the next free one;
/api is proxied to 38219; set `NEXASCAN_BACKEND` if the backend uses another port). If the React app is not built, the old
single-page UI in `frontend/legacy/` is served instead.

Env: `NEXASCAN_WORKERS`, `NEXASCAN_GPU=1` (optional; CPU is the default and is what was tested),
`NEXASCAN_JOB_TTL_HOURS` (default 24; jobs and cache files older than this are removed every 30 minutes),
`NEXASCAN_API_CACHE=1` (re-enable the per-page analysis cache for API jobs; off by default because it stores document text).
Jobs run in `spawn`ed worker processes that are recycled after every job (Python 3.11+); a crashed worker marks its job `error`.

## Test
    python -m pytest tests/test_units.py                             # unit tests
    python -m pytest tests/test_regression.py                        # Tests 01-04 end to end (tests/make_master.py)
    python -m pytest tests/test_realpack.py                          # Tests 01-04 on the real Master Test Pack (tests/data/pack)
    python tests/make_suite.py && python tests/benchmark.py data/suite   # 35 hand-designed cases (used during tuning)
    python tests/holdout.py 16 1000                                  # randomized cases not used for tuning

## Measured results (synthetic documents, CPU, 2 workers)
| Set | Truth items | Recall (confirmed) | Recall (+uncertain) | False positives |
|---|---|---|---|---|
| Tuned suite (35 cases) | 106 | 96.2% | 100% | 0 |
| Held-out random (16 cases) | 59 | 98.3% | 100% | 6 (precision 0.906) |
Median 1.8 s per document pair, max ~10 s, peak ~1 GB RSS.
The tuned-suite numbers are optimistic because thresholds were adjusted on it; the held-out row is the fairer estimate.

## Known limitations
- Accuracy is not 100%. Real scans of unknown quality will differ from these synthetic ones.
- Held-out false positives were mostly OCR word truncations at line ends in degraded scans
  (e.g. "covenant" read as "coven"); one was a stray ink mark reported as moved.
- Near-invisible single-letter substitutions (u/o, i/l) in low-resolution scans are reported as "uncertain".
- Stamp / signature / handwriting detection is classical-CV heuristics, not a trained model.
- Content hidden under stamps is never guessed; it is flagged as an overlap.
- See docs/ARCHITECTURE.md for the full list.
