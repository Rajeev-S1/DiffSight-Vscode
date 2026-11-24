"""FastAPI application: upload service + asynchronous comparison jobs."""
from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import re
import shutil
import sys
import threading
import time
import uuid
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .applog import get_logger, job_tag
from .config import ROOT, Config
from .pagedata import purge_doc_cache, purge_old_cache
from .ocr import tesseract_available
from .report import save_embedded_html
from .upload import UploadError, save_and_validate
from . import wordsupport
from .worker import read_json, run_job, write_json_atomic

JOBS_DIR = Path(os.environ.get("NEXASCAN_JOBS_DIR", ROOT / "data" / "jobs"))
JOB_TTL_H = float(os.environ.get("NEXASCAN_JOB_TTL_HOURS", "24"))
WORKERS = max(1, int(os.environ.get("NEXASCAN_WORKERS", "1")))
FRONTEND = ROOT / "frontend"
DIST = FRONTEND / "dist"
DEV_PORT = os.environ.get("NEXASCAN_DEV_PORT", "34517")
CLEANUP_INTERVAL_S = 30 * 60
API_CACHE = os.environ.get("NEXASCAN_API_CACHE", "0") == "1"   # page-analysis cache stores document text: off by default
ID_RE = re.compile(r"^[0-9a-f]{32}$")
EVIDENCE_RE = re.compile(r"^[A-Za-z0-9_.\-]+\.png$")

app = FastAPI(title="DiffSight", version=__version__, description="Every Detail. Every Difference.")
app.add_middleware(CORSMiddleware, allow_origins=[f"http://localhost:{DEV_PORT}", f"http://127.0.0.1:{DEV_PORT}"],
                   allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"])
_pool: Optional[ProcessPoolExecutor] = None
_pool_lock = threading.Lock()
log = get_logger()
_cleanup_task: Optional["asyncio.Task"] = None


def _new_pool() -> ProcessPoolExecutor:
    """Fresh worker processes ('spawn': no inherited MuPDF / OpenCV / thread state) that are recycled after
    every job so memory is returned to the OS. max_tasks_per_child needs Python 3.11+."""
    kw = {"max_workers": WORKERS, "mp_context": mp.get_context("spawn")}
    if sys.version_info >= (3, 11):
        kw["max_tasks_per_child"] = 1
    return ProcessPoolExecutor(**kw)


def _executor() -> ProcessPoolExecutor:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = _new_pool()
        return _pool


def _reset_pool(broken: ProcessPoolExecutor) -> None:
    """Drop a broken pool (only if it is still the current one) so the next submit creates a new one."""
    global _pool
    with _pool_lock:
        if _pool is broken:
            _pool = None
    # shutdown() takes a lock the pool's management thread holds while it runs done-callbacks (which is where
    # this is usually called from), so it must not run on the calling thread
    def _shutdown() -> None:
        try:
            broken.shutdown(wait=False, cancel_futures=True)
        except Exception:  # noqa: BLE001
            pass
    threading.Thread(target=_shutdown, name="nexascan-pool-shutdown", daemon=True).start()


def _on_job_done(jd: Path, pool: ProcessPoolExecutor, fut: Future) -> None:
    """Done-callback: a worker that crashed (segfault / OOM kill) or a pool that broke never reaches the
    error handling inside run_job, so the failure is recorded here and the poller sees it."""
    if fut.cancelled():
        exc: Optional[BaseException] = RuntimeError("job was cancelled before it ran")
    else:
        exc = fut.exception()
    if exc is None:
        return
    if isinstance(exc, BrokenProcessPool):
        _reset_pool(pool)
    prog = read_json(jd / "progress.json") or {}
    if prog.get("status") in ("done", "error"):
        return
    msg = ("the worker process crashed (out of memory or an internal error); please retry"
           if isinstance(exc, BrokenProcessPool) else f"{type(exc).__name__}: {exc}")
    log.error("%s failed: %s", job_tag(jd.name), msg)
    write_json_atomic(jd / "progress.json", {"status": "error", "stage": "error", "fraction": 0.0, "message": msg})


def _submit_job(jd: Path, *args) -> Future:
    for attempt in (0, 1):
        pool = _executor()
        try:
            fut = pool.submit(run_job, str(jd), *args)
            break
        except BrokenProcessPool:       # pool died before this submit: recreate it once and retry
            _reset_pool(pool)
            if attempt:
                raise
    fut.add_done_callback(lambda f, jd=jd, pool=pool: _on_job_done(jd, pool, f))
    return fut


def _forget_job(d: Path) -> None:
    """Delete a job directory together with the page-cache entries of its two documents."""
    meta = read_json(d / "job.json") or {}
    for key in ("sha256_a", "sha256_b"):
        if meta.get(key):
            purge_doc_cache(meta[key], Config().cache_dir)
    shutil.rmtree(d, ignore_errors=True)


def _cleanup_old_jobs() -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - JOB_TTL_H * 3600
    for d in JOBS_DIR.iterdir():
        try:
            if d.is_dir() and ID_RE.match(d.name) and d.stat().st_mtime < cutoff:
                _forget_job(d)
                log.info("%s expired and was removed", job_tag(d.name))
        except OSError:
            pass
    purge_old_cache(Config().cache_dir, JOB_TTL_H * 3600)    # orphaned cache files of any age past the TTL


async def _cleanup_loop() -> None:
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_S)
        try:
            await run_in_threadpool(_cleanup_old_jobs)
        except Exception:  # noqa: BLE001 - housekeeping must never kill the loop
            pass


@app.on_event("startup")
async def _startup() -> None:
    global _cleanup_task
    log.info("DiffSight API started (version %s, Tesseract %s, jobs folder %s)", __version__,
             "available" if tesseract_available() else "NOT FOUND", JOBS_DIR)
    await run_in_threadpool(_cleanup_old_jobs)
    _cleanup_task = asyncio.create_task(_cleanup_loop())


@app.on_event("shutdown")
async def _shutdown() -> None:
    log.info("DiffSight API stopped")
    if _cleanup_task is not None:
        _cleanup_task.cancel()
    if _pool is not None:
        _pool.shutdown(wait=False, cancel_futures=True)


def _job_dir(job_id: str) -> Path:
    if not ID_RE.match(job_id):
        raise HTTPException(404, "Unknown job")
    d = JOBS_DIR / job_id
    if not d.is_dir():
        raise HTTPException(404, "Unknown job")
    return d


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "tesseract": tesseract_available(), "workers": WORKERS}


@app.post("/api/compare")
async def compare(a: UploadFile = File(..., description="Document A – original"),
                  b: UploadFile = File(..., description="Document B – modified"),
                  dpi: Optional[int] = Form(None, ge=150, le=400, description="Advanced: force a DPI. Leave empty for automatic."),
                  second_opinion: bool = Form(True)) -> JSONResponse:
    """Upload two documents (PDF / PNG / JPG). Returns a job id immediately; poll /api/jobs/{id}."""
    if not tesseract_available():
        log.error("comparison refused: Tesseract OCR is not installed")
        raise HTTPException(503, "Tesseract OCR is not installed on the server.")
    job_id = uuid.uuid4().hex
    jd = JOBS_DIR / job_id
    jd.mkdir(parents=True, exist_ok=True)
    cfg = Config(dpi=dpi or 300)      # validate the page size at the dpi that will be rendered (automatic starts at 300 or lower)
    try:
        # Word (.docx/.doc) uploads are rendered to PDF here; every other upload is passed through unchanged
        fa = await run_in_threadpool(wordsupport.prepare, a.file, a.filename or "document_a", jd, "a", cfg)
        fb = await run_in_threadpool(wordsupport.prepare, b.file, b.filename or "document_b", jd, "b", cfg)
        ia = await run_in_threadpool(save_and_validate, fa, a.filename or "document_a", jd, "a", cfg)
        ib = await run_in_threadpool(save_and_validate, fb, b.filename or "document_b", jd, "b", cfg)
    except UploadError as e:
        shutil.rmtree(jd, ignore_errors=True)
        log.warning("upload rejected (HTTP 400): %s", re.sub(r"'[^']*'", "'<file>'", str(e)))
        raise HTTPException(400, str(e))
    finally:
        await a.close()
        await b.close()
    # (save_and_validate stored the uploads as input_a.bin / input_b.bin, the names the worker expects)
    write_json_atomic(jd / "progress.json", {"status": "queued", "stage": "queued", "fraction": 0.0,
                                             "message": "Waiting for a free worker"})
    write_json_atomic(jd / "job.json", {"name_a": ia.display_name, "name_b": ib.display_name,
                                        "pages_a": ia.pages, "pages_b": ib.pages, "created": time.time(),
                                        "sha256_a": ia.sha256, "sha256_b": ib.sha256})
    log.info("%s queued: A %s %d page(s), B %s %d page(s)%s", job_tag(job_id), ia.kind, ia.pages, ib.kind, ib.pages,
             f", dpi forced to {dpi}" if dpi else ", dpi automatic")
    _submit_job(jd, ia.display_name, ib.display_name,
                {**({"dpi": dpi} if dpi else {}), "second_opinion": bool(second_opinion), "use_cache": API_CACHE})
    return JSONResponse({"job_id": job_id, "document_a": {"name": ia.display_name, "pages": ia.pages},
                         "document_b": {"name": ib.display_name, "pages": ib.pages}}, status_code=202)


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    jd = _job_dir(job_id)
    prog = read_json(jd / "progress.json") or {"status": "queued", "fraction": 0.0, "message": ""}
    job = read_json(jd / "job.json") or {}
    return {"job_id": job_id, **prog, "document_a": job.get("name_a"), "document_b": job.get("name_b")}


@app.get("/api/jobs/{job_id}/report")
def report_json(job_id: str):
    p = _job_dir(job_id) / "report.json"
    if not p.is_file():
        raise HTTPException(409, "Report is not ready yet")
    return FileResponse(p, media_type="application/json")


@app.get("/api/jobs/{job_id}/report.txt")
def report_txt(job_id: str):
    p = _job_dir(job_id) / "report.txt"
    if not p.is_file():
        raise HTTPException(409, "Report is not ready yet")
    return FileResponse(p, media_type="text/plain; charset=utf-8", filename="diffsight_report.txt")


@app.get("/api/jobs/{job_id}/report.html")
def report_html(job_id: str):
    p = _job_dir(job_id) / "report.html"
    if not p.is_file():
        raise HTTPException(409, "Report is not ready yet")
    return FileResponse(p, media_type="text/html; charset=utf-8")


@app.get("/api/jobs/{job_id}/report_embedded.html")
def report_embedded(job_id: str):
    jd = _job_dir(job_id)
    rep = read_json(jd / "report.json")
    if rep is None:
        raise HTTPException(409, "Report is not ready yet")
    p = jd / "report_embedded.html"
    if not p.is_file():
        p = save_embedded_html(rep, jd)
    return FileResponse(p, media_type="text/html; charset=utf-8", filename="diffsight_report.html")


@app.get("/api/jobs/{job_id}/evidence/{name}")
def evidence(job_id: str, name: str):
    if not EVIDENCE_RE.match(name):
        raise HTTPException(404, "Not found")
    p = _job_dir(job_id) / "evidence" / name
    if not p.is_file():
        raise HTTPException(404, "Not found")
    return FileResponse(p, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    _forget_job(_job_dir(job_id))
    log.info("%s deleted by the user", job_tag(job_id))
    return {"deleted": job_id}


if (DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")


@app.get("/")
def index():
    built = DIST / "index.html"
    page = built if built.is_file() else FRONTEND / "legacy" / "index.html"   # legacy page if the React app is not built
    return FileResponse(page, media_type="text/html; charset=utf-8", headers={"Cache-Control": "no-cache"})
