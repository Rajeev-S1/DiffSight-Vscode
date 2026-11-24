"""Application log: logs/nexascan.log - what the running product does (start-up, jobs, DPI choice, retries, errors).

* Appended across runs; run.py renames it to nexascan.log.1 at start-up once it is larger than MAX_BYTES.
* Works in the worker processes too (each process adds its own append handler on first use).
* Never logs document content or file names - only job ids (first 8 characters), page counts and results.
* A logging problem (read-only folder, disk full) never breaks the application.
Override the folder with NEXASCAN_LOG_DIR.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = Path(os.environ.get("NEXASCAN_LOG_DIR") or (ROOT / "logs"))
LOG_FILE = LOG_DIR / "nexascan.log"
FMT = "%(asctime)s %(levelname)-7s [pid %(process)d] %(message)s"
MAX_BYTES = 5 * 1024 * 1024


def trim_at_startup() -> None:
    """Called once by run.py (a single process): keep one previous generation when the log has grown large."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        if LOG_FILE.is_file() and LOG_FILE.stat().st_size > MAX_BYTES:
            old = LOG_FILE.with_name(LOG_FILE.name + ".1")
            if old.exists():
                old.unlink()
            LOG_FILE.rename(old)
    except OSError:
        pass


def get_logger() -> logging.Logger:
    log = logging.getLogger("nexascan")
    if not any(getattr(h, "_nexascan_file", False) for h in log.handlers):
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            h = logging.FileHandler(LOG_FILE, encoding="utf-8")
            h.setFormatter(logging.Formatter(FMT))
            h._nexascan_file = True  # type: ignore[attr-defined]
            log.addHandler(h)
        except OSError:
            log.addHandler(logging.NullHandler())
        log.setLevel(logging.INFO)
        log.propagate = False
    return log


def job_tag(job_id: str) -> str:
    return "job " + str(job_id)[:8]
