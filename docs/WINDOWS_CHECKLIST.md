# Windows checks (could not be run on the build machine)

Run `tests\run_windows_checks.bat` after `setup.bat`. It executes the automatic part. These scenarios need a Windows PC and
must be tried by hand once (each takes a few minutes); the result goes to `logs\setup.log`:

| # | Scenario | How | Expected in `logs\setup.log` / console |
|---|---|---|---|
| 1 | Tesseract already installed and on PATH | run `setup.bat` | `found and working: ... [PATH] ... reusing it, nothing to install` |
| 2 | Installed but missing from PATH | remove its folder from PATH (keep it in Program Files), run `setup.bat`, then `start.bat` | `[common folder]`, `start.bat` prints `Tesseract OCR: C:\Program Files\Tesseract-OCR\tesseract.exe` |
| 3 | Not installed | temporarily rename `C:\Program Files\Tesseract-OCR` (and make sure `tools\Tesseract-OCR` does not exist), run `setup.bat` | `downloading ...`, `installing into ...tools\Tesseract-OCR`, `installed and verified` |
| 4 | Download failure | disconnect the network, remove `tools\Tesseract-OCR`, run `setup.bat` | 3 download attempts, `ERROR: the Tesseract installer could not be downloaded`, setup continues with a WARNING |
| 5 | Installation failure | set `NEXASCAN_TESSERACT_URL` to a small .exe, run `setup.bat` | `download is only ... bytes` or `did not produce a working tesseract.exe` |
| 6 | Restart after install | close everything, run `start.bat` | `Tesseract OCR: ...\tools\Tesseract-OCR\tesseract.exe` and the backend starts |
| 7 | OCR really works | end of `setup.bat` | `OCR self-test passed` |

Environment variables: `NEXASCAN_TESSERACT` (explicit path), `NEXASCAN_TESSERACT_URL` (installer mirror), `NEXASCAN_TOOLS_DIR`
(managed folder), `NEXASCAN_SKIP_TESSERACT_CHECK=1` (let `start.bat` run without Tesseract: digital PDFs and Word only).
