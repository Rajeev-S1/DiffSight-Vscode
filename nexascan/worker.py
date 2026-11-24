"""Background job runner (executes in a separate process so the API event loop never blocks)."""
from __future__ import annotations

import json
import copy
import os
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Optional

from .applog import get_logger, job_tag

log = get_logger()

from .config import Config


class JobDeleted(BaseException):
    """The job directory vanished (DELETE while running): stop quietly. Derives from BaseException so the
    per-page ``except Exception`` handlers of the pipeline cannot swallow it."""


# ---------------------------------------------------------------------------------------------------------------
# progress.json / job.json / ... : atomic JSON files that the API process reads while the worker process replaces them.
# On Windows a file that another process has open cannot be replaced (os.replace -> PermissionError, WinError 5 access
# denied / 32 sharing violation / 33 lock violation) because Python's open() does not share the right to delete or rename.
# Every status poll of the web page opens progress.json, so a worker update could collide with it. The fixes: a unique
# temporary file that is closed before the replace, a short bounded retry for exactly those transient errors, a last
# in-place write if the replace never succeeds, readers that retry too, and a loud log line when everything failed.
# ---------------------------------------------------------------------------------------------------------------
IS_WINDOWS = os.name == "nt"
REPLACE_ATTEMPTS = 12                      # delays 0.02 s doubling up to 0.4 s: about 3.5 s in total
WINERRORS = (5, 32, 33)                    # ACCESS_DENIED, SHARING_VIOLATION, LOCK_VIOLATION
_path_locks: dict = {}
_locks_guard = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    with _locks_guard:
        return _path_locks.setdefault(str(path), threading.Lock())


def _transient(exc: BaseException) -> bool:
    """A Windows file-locking error that is expected to clear within moments."""
    return IS_WINDOWS and isinstance(exc, OSError) and (isinstance(exc, PermissionError) or getattr(exc, "winerror", None) in WINERRORS)


def _delay(attempt: int) -> float:
    return min(0.4, 0.02 * (2 ** (attempt - 1)))


def _remove(path: Path) -> bool:
    for i in range(3):
        try:
            os.unlink(path)
            return True
        except FileNotFoundError:
            return True
        except OSError:
            time.sleep(0.02 * (i + 1))        # a virus scanner may still hold the temporary file
    return False


def _purge_stale_tmp(path: Path, older_than_s: float = 300.0) -> None:
    """Remove temporary files left behind for this target by an earlier crash (only old ones, never a live write)."""
    try:
        for f in path.parent.glob(f".{path.name}.*.tmp"):
            if time.time() - f.stat().st_mtime > older_than_s:
                _remove(f)
    except OSError:
        pass


def write_json_atomic(path: Path, data: dict) -> None:
    """Atomic JSON write that survives Windows file locking. Returns quietly if the directory no longer exists
    (job deleted meanwhile). Raises OSError - after logging every attempt - only if the file really cannot be written."""
    path = Path(path)
    if not path.parent.is_dir():
        return
    payload = json.dumps(data, ensure_ascii=False)                       # a bad value fails here, before anything is touched
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:12]}.tmp")      # unique per call, process and thread
    tag = job_tag(path.parent.name)
    attempts, failed = 0, False
    with _lock_for(path):                                                # threads of this process never replace one file at once
        try:
            with open(tmp, "x", encoding="utf-8") as f:                  # exclusive create; closed (flushed) before the replace
                f.write(payload)
            while True:
                attempts += 1
                try:
                    os.replace(tmp, path)
                    break
                except FileNotFoundError:                                # directory removed between the check and the write
                    return
                except OSError as e:
                    if not _transient(e):
                        raise
                    if attempts >= REPLACE_ATTEMPTS:
                        failed = True
                        log.warning("%s: could not replace %s after %d attempts (%s); writing it in place", tag, path.name, attempts, e)
                        try:
                            with open(path, "w", encoding="utf-8") as f:      # not atomic, but allowed while a reader has it open
                                f.write(payload)
                        except OSError as e2:
                            log.error("%s: WRITE FAILED for %s after %d attempts and an in-place write: %s / %s",
                                      tag, path.name, attempts, e, e2)
                            raise e2
                        break
                    time.sleep(_delay(attempts))
            if attempts > 1 and not failed:
                log.info("%s: %s was locked by another process; the write succeeded on attempt %d", tag, path.name, attempts)
        except FileNotFoundError:
            return
        finally:
            if not _remove(tmp):
                log.warning("%s: temporary file %s could not be removed", tag, tmp.name)
            if attempts > 1:
                _purge_stale_tmp(path)


def read_json(path: Path) -> Optional[dict]:
    """Read a JSON file; None if it does not exist or is unreadable. A momentary Windows lock (the writer is replacing
    the file) or a half-written file is retried briefly before giving up."""
    last: Optional[BaseException] = None
    for attempt in range(1, 7):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        except (PermissionError, json.JSONDecodeError) as e:
            last = e
            if attempt < 6:
                time.sleep(0.01 * attempt)
        except Exception:  # noqa: BLE001
            return None
    log.warning("%s could not be read after 6 attempts: %s", Path(path).name, last)
    return None


def retry_stage(jd: Path, report: dict, first_dpi: int, retry_to: int, compare, args: tuple, cfg2, progress) -> dict:
    """Run the single higher-DPI retry, judge it with retry.evaluate_retry and leave ONE consistent result in ``jd``.
    The first result is never lost: any failure keeps it (retry_result = "failed")."""
    from . import retry as rt
    from .report import refresh_summary, write_reports

    n_unc = report["summary"]["uncertain_findings"]
    base = {**report["diagnostics"].get("dpi_selection", {}), "retried": True, "retry_dpi": retry_to,
            "first_dpi": first_dpi, "first_uncertain": n_unc}
    rd = jd / "_retry"
    shutil.rmtree(rd, ignore_errors=True)

    def keep_first(extra: dict) -> dict:
        report["diagnostics"]["dpi_selection"] = {**base, "used": first_dpi, "kept": "first",
                                                   "selected_comparison": "Original-resolution result", **extra}
        write_reports(refresh_summary(report), jd)
        return report

    try:
        progress("retry", 0.86, f"Re-checking {n_unc} uncertain finding(s) at {retry_to} DPI")
        report2 = compare(*args, rd, cfg2, lambda st, fr, msg: progress("retry", 0.86 + 0.14 * fr, f"{msg} (re-check at {retry_to} DPI)"))
    except JobDeleted:
        raise
    except Exception as e:  # noqa: BLE001 - too large, out of memory, ...: the original result stays
        shutil.rmtree(rd, ignore_errors=True)
        return keep_first({"retry_result": "failed", "retry_error": type(e).__name__, "retry_reason": "the re-check could not be completed"})
    try:
        dec = rt.evaluate_retry(report, report2)
        if dec["keep"] == "retry":
            sel = rt.finalize(report, report2, dec, retry_to, first_dpi, first_dir=jd, retry_dir=rd, keep_dir=rd)
            sel["diagnostics"]["dpi_selection"] = {**base, "used": retry_to, "selected_dpi": retry_to, "kept": "retry", "retry_result": "improved",
                                                   "selected_comparison": "Higher-resolution result", "retry_reason": dec["reason"],
                                                   "higher_resolution_only": len(sel["seen_only_at_higher_resolution"])}
            write_reports(refresh_summary(sel), rd)
            for name in ("evidence", "report.json", "report.txt", "report.html"):   # swap the retry in as the result
                target = jd / name
                if target.is_dir():
                    shutil.rmtree(target, ignore_errors=True)
                elif target.exists():
                    target.unlink()
                if (rd / name).exists():
                    shutil.move(str(rd / name), str(target))
            return sel
        sel = rt.finalize(report, report2, dec, retry_to, first_dpi, first_dir=jd, retry_dir=rd, keep_dir=jd)
        return keep_first({"retry_result": "not_improved", "retry_reason": dec["reason"],
                           "higher_resolution_only": len(sel["seen_only_at_higher_resolution"])})
    except JobDeleted:
        raise
    except Exception as e:  # noqa: BLE001
        return keep_first({"retry_result": "failed", "retry_error": type(e).__name__, "retry_reason": "the re-check result could not be evaluated"})
    finally:
        shutil.rmtree(rd, ignore_errors=True)


def run_job(job_dir: str, name_a: str, name_b: str, overrides: Optional[dict] = None) -> None:
    """Compare input_a.bin / input_b.bin inside ``job_dir`` and write the report files next to them."""
    from .pipeline import compare_documents  # imported here: keeps the API process light

    jd = Path(job_dir)
    pf = jd / "progress.json"
    started = time.time()
    tag = job_tag(jd.name)
    log.info("%s started", tag)

    failed_updates = [0]

    def progress(stage: str, frac: float, msg: str) -> None:
        if not jd.is_dir():
            raise JobDeleted()
        try:
            write_json_atomic(pf, {"status": "running", "stage": stage, "fraction": round(float(frac), 3),
                                   "message": msg, "elapsed_s": round(time.time() - started, 1)})
        except OSError as e:          # already logged in detail; one missed progress update must not abort the comparison
            failed_updates[0] += 1
            log.error("%s progress update #%d was lost (%s: %s); the comparison continues", tag, failed_updates[0], type(e).__name__, e)

    try:
        from . import autodpi

        cfg = Config()
        overrides = dict(overrides or {})
        manual = "dpi" in overrides                     # only an explicit API override disables the automatic choice
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        pa, pb = str(jd / "input_a.bin"), str(jd / "input_b.bin")
        progress("start", 0.0, "Analysing the documents and measuring the scan quality")
        info_a = info_b = None
        retry_to = 0
        if manual:
            cfg.dpi_info = {"mode": "manual", "used": cfg.dpi, "selected_dpi": cfg.dpi, "reason": "set by the caller",
                            "message": f"Analysed at {cfg.dpi} DPI (set by the caller).", "retried": False,
                            "retry_result": "not_needed", "kept": "first"}
        else:
            dec = autodpi.choose_dpi(pa, name_a, pb, name_b, cfg)
            cfg.dpi, info_a, info_b = dec.dpi, dec.info_a, dec.info_b
            cfg.severe_poor_scan = dec.severe
            cfg.dpi_info = dec.to_dict(cfg)
            log.info("%s resolution: %s", tag, dec.message)
            retry_to = dec.retry_dpi
            if retry_to:                                           # a retry must also stay inside the pixel limit
                retry_to, _ = autodpi.fit_dpi([info_a, info_b], retry_to, cfg)
                retry_to = retry_to if retry_to > cfg.dpi else 0
        share = 0.85 if retry_to else 1.0                # first run uses 0..85 % of the bar when a retry is possible
        report = compare_documents(pa, name_a, pb, name_b, jd, cfg,
                                   lambda st, fr, msg: progress(st, fr * share, msg), info_a, info_b)

        log.info("%s first run finished: %d confirmed, %d uncertain", tag, report["summary"]["total_differences"],
                 report["summary"]["uncertain_findings"])
        # ---- one retry at a higher DPI when the first result still has uncertain findings ----
        pages = report["document_a"]["pages"] + report["document_b"]["pages"]
        if retry_to and report["summary"]["uncertain_findings"] >= 1 and pages <= 2 * autodpi.LARGE_PAIR_PAGES:
            cfg2 = copy.copy(cfg)
            cfg2.dpi = retry_to
            report = retry_stage(jd, report, cfg.dpi, retry_to, compare_documents, (pa, name_a, pb, name_b), cfg2, progress)
            ds = report["diagnostics"].get("dpi_selection", {})
            log.info("%s re-check at %s DPI: %s (%s)", tag, retry_to, ds.get("retry_result"), ds.get("retry_reason"))
        from .wordsupport import augment_report              # Word-only post-step: does nothing for PDF / JPG / PNG
        report = augment_report(jd, report)
        write_json_atomic(pf, {"status": "done", "stage": "done", "fraction": 1.0, "message": "Done",
                               "elapsed_s": round(time.time() - started, 1), "summary": report["summary"]})
        log.info("%s done in %.1fs: %d confirmed, %d uncertain, %d for review at higher resolution", tag, time.time() - started,
                 report["summary"]["total_differences"], report["summary"]["uncertain_findings"],
                 len(report.get("seen_only_at_higher_resolution") or []))
    except Exception as e:  # noqa: BLE001
        log.error("%s failed after %.1fs: %s: %s", tag, time.time() - started, type(e).__name__, e, exc_info=True)
        write_json_atomic(pf, {"status": "error", "stage": "error", "fraction": 0.0,
                               "message": f"{type(e).__name__}: {e}", "elapsed_s": round(time.time() - started, 1)})
        try:
            (jd / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        except OSError:
            pass
    except JobDeleted:
        log.info("%s was deleted while it was running", tag)
        pass                        # job was deleted while running: nothing left to report to
    finally:
        from . import ingest
        ingest.close_all()          # release the PDF handles of this job
