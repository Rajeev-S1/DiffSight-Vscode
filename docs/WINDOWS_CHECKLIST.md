# Windows checks (could not be run on the build machine)

Run `tests\run_windows_checks.bat` after `setup.bat`. It executes the automatic part. These scenarios need a Windows PC and
must be tried by hand once (each takes a few minutes); the result goes to `logs\setup.log`:

| # | Scenario | How | Expected in `logs\setup.log` / console |
|---|---|---|---|
| 1 | PaddleOCR packages missing | run `setup.bat` | `paddleocr` and `onnxruntime` install successfully |
| 2 | PaddleOCR model initialization fails | inspect setup/backend logs | the API reports PaddleOCR unavailable; digital PDF text-layer extraction remains available |
| 3 | Scanned comparison completes | run `tests\run_windows_checks.bat` | `real-pack comparisons PASSED` |
