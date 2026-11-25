"""Windows PermissionError (WinError 5) fix: write_json_atomic / read_json in nexascan/worker.py.

The real Windows file-locking behaviour is tested in tests/test_windows_real.py (runs only on Windows). Here the error is
SIMULATED (os.replace raising PermissionError, IS_WINDOWS switched on) and the logic is stress-tested with threads and processes."""
import builtins
import json
import logging
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import threading
import time
import uuid
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from nexascan import worker as W  # noqa: E402

PACK = ROOT / "tests" / "data" / "pack"


def old_write_json_atomic(path, data):
    """The implementation before the fix (copied verbatim), kept to prove the simulated error breaks it."""
    path = Path(path)
    if not path.parent.is_dir():
        return
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, path)
    except FileNotFoundError:
        pass
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass


class Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records = []
    def emit(self, record):
        self.records.append(record)
    def text(self):
        return "\n".join(r.getMessage() for r in self.records)


@pytest.fixture()
def logs():
    h = Capture()
    lg = logging.getLogger("nexascan")
    lg.addHandler(h)
    yield h
    lg.removeHandler(h)


@pytest.fixture()
def windows(monkeypatch):
    monkeypatch.setattr(W, "IS_WINDOWS", True)
    monkeypatch.setattr(time, "sleep", lambda s: None)            # no real waiting in the simulated retries


def tmp_files(d):
    return sorted(p.name for p in Path(d).iterdir() if p.name.endswith(".tmp"))


def fail_replace_times(monkeypatch, n):
    real, calls = os.replace, {"n": 0}
    def fake(src, dst):
        calls["n"] += 1
        if calls["n"] <= n:
            raise PermissionError(13, "Access is denied")          # what Windows reports as WinError 5
        return real(src, dst)
    monkeypatch.setattr(os, "replace", fake)
    return calls


# ---------------------------------------------------------------- the failure, simulated, and the fix
def test_simulated_winerror5_breaks_the_old_code_and_not_the_new(tmp_path, monkeypatch, windows):
    fail_replace_times(monkeypatch, 3)
    with pytest.raises(PermissionError):
        old_write_json_atomic(tmp_path / "progress.json", {"status": "running"})
    monkeypatch.undo()
    monkeypatch.setattr(W, "IS_WINDOWS", True)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    fail_replace_times(monkeypatch, 3)
    W.write_json_atomic(tmp_path / "progress.json", {"status": "running", "fraction": 0.5})      # same 3 failures: now succeeds
    assert json.loads((tmp_path / "progress.json").read_text()) == {"status": "running", "fraction": 0.5}
    assert tmp_files(tmp_path) == []


def test_retry_then_success_logs_the_attempts(tmp_path, monkeypatch, windows, logs):
    calls = fail_replace_times(monkeypatch, 4)
    W.write_json_atomic(tmp_path / "progress.json", {"n": 1})
    assert calls["n"] == 5 and json.loads((tmp_path / "progress.json").read_text()) == {"n": 1}
    assert "succeeded on attempt 5" in logs.text() and tmp_files(tmp_path) == []


def test_all_replace_attempts_fail_falls_back_to_an_in_place_write_and_logs_it(tmp_path, monkeypatch, windows, logs):
    calls = fail_replace_times(monkeypatch, 10 ** 6)
    (tmp_path / "progress.json").write_text('{"old": true}')
    W.write_json_atomic(tmp_path / "progress.json", {"new": True})
    assert calls["n"] == W.REPLACE_ATTEMPTS
    assert json.loads((tmp_path / "progress.json").read_text()) == {"new": True}
    assert "writing it in place" in logs.text() and tmp_files(tmp_path) == []


def test_total_failure_is_logged_loudly_and_raised_never_silent(tmp_path, monkeypatch, windows, logs):
    fail_replace_times(monkeypatch, 10 ** 6)
    target = tmp_path / "progress.json"
    real_open = builtins.open
    def fake_open(file, mode="r", *a, **k):
        if Path(str(file)) == target and "w" in mode:
            raise PermissionError(13, "Access is denied")
        return real_open(file, mode, *a, **k)
    monkeypatch.setattr(W, "open", fake_open, raising=False)
    with pytest.raises(PermissionError):
        W.write_json_atomic(target, {"x": 1})
    assert any(r.levelno == logging.ERROR and "WRITE FAILED" in r.getMessage() for r in logs.records)
    assert tmp_files(tmp_path) == []                                       # the temporary file is cleaned up even on failure


def test_other_errors_and_non_windows_are_not_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "IS_WINDOWS", False)
    calls = fail_replace_times(monkeypatch, 10 ** 6)
    with pytest.raises(PermissionError):
        W.write_json_atomic(tmp_path / "progress.json", {"x": 1})
    assert calls["n"] == 1 and tmp_files(tmp_path) == []


def test_temporary_names_are_unique_and_the_file_is_closed_before_replace(tmp_path, monkeypatch):
    seen = []
    real = os.replace
    def spy(src, dst):
        seen.append(Path(src).name)
        assert Path(src).read_text(encoding="utf-8")                         # fully written (and closed) at replace time
        return real(src, dst)
    monkeypatch.setattr(os, "replace", spy)
    for i in range(5):
        W.write_json_atomic(tmp_path / "progress.json", {"i": i})
    assert len(set(seen)) == 5 and all(n.startswith(".progress.json.") and n.endswith(".tmp") for n in seen)


def test_deleted_job_directory_is_still_handled_quietly(tmp_path):
    W.write_json_atomic(tmp_path / "gone" / "progress.json", {"x": 1})        # directory never existed: returns quietly


def test_stale_temporary_files_are_purged_after_trouble_but_fresh_ones_are_kept(tmp_path, monkeypatch, windows):
    old, fresh = tmp_path / ".progress.json.1.old.tmp", tmp_path / ".progress.json.2.fresh.tmp"
    old.write_text("x")
    fresh.write_text("x")
    os.utime(old, (time.time() - 3600, time.time() - 3600))
    fail_replace_times(monkeypatch, 1)                                         # trouble: purge runs
    W.write_json_atomic(tmp_path / "progress.json", {"a": 1})
    assert not old.exists() and fresh.exists()


# ---------------------------------------------------------------- readers
def test_reader_retries_a_momentary_lock_and_a_half_written_file(tmp_path, monkeypatch):
    p = tmp_path / "progress.json"
    p.write_text('{"status": "done"}')
    real_open, state = builtins.open, {"n": 0}
    def flaky(file, mode="r", *a, **k):
        if Path(str(file)) == p and "r" in mode:
            state["n"] += 1
            if state["n"] <= 2:
                raise PermissionError(13, "Access is denied")
        return real_open(file, mode, *a, **k)
    monkeypatch.setattr(W, "open", flaky, raising=False)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    assert W.read_json(p) == {"status": "done"} and state["n"] == 3
    p.write_text('{"status": "do')                                                # torn / corrupt: bounded retries, then None
    assert W.read_json(p) is None
    assert W.read_json(tmp_path / "missing.json") is None


# ---------------------------------------------------------------- repeated updates, threads, processes
def test_repeated_updates_keep_the_json_valid_and_leave_no_temp_files(tmp_path):
    p = tmp_path / "progress.json"
    for i in range(300):
        W.write_json_atomic(p, {"status": "running", "fraction": i / 300, "message": "é" * 5})
        assert json.loads(p.read_text(encoding="utf-8"))["fraction"] == i / 300
    assert tmp_files(tmp_path) == []


def test_many_threads_writing_and_reading_one_file(tmp_path):
    p = tmp_path / "progress.json"
    W.write_json_atomic(p, {"v": 0})
    stop, errors, reads = threading.Event(), [], [0]
    def writer(k):
        try:
            for i in range(150):
                W.write_json_atomic(p, {"v": k * 1000 + i})
        except Exception as e:  # noqa: BLE001
            errors.append(e)
    def reader():
        try:
            while not stop.is_set():
                d = W.read_json(p)
                assert d is not None and "v" in d
                reads[0] += 1
        except Exception as e:  # noqa: BLE001
            errors.append(e)
    rs = [threading.Thread(target=reader) for _ in range(3)]
    ws = [threading.Thread(target=writer, args=(k,)) for k in range(6)]
    for t in rs + ws:
        t.start()
    for t in ws:
        t.join()
    stop.set()
    for t in rs:
        t.join()
    assert errors == [] and reads[0] > 0 and tmp_files(tmp_path) == []
    assert "v" in json.loads(p.read_text())


def _proc_writer(path, k, n):
    from nexascan import worker as w
    for i in range(n):
        w.write_json_atomic(Path(path), {"proc": k, "i": i})
        assert w.read_json(Path(path)) is not None
    return k


def test_several_processes_writing_one_file(tmp_path):
    p = tmp_path / "progress.json"
    W.write_json_atomic(p, {"proc": -1, "i": 0})
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("spawn")) as ex:
        done = list(ex.map(_proc_writer, [str(p)] * 4, range(4), [120] * 4))
    assert sorted(done) == [0, 1, 2, 3] and tmp_files(tmp_path) == [] and "proc" in json.loads(p.read_text())


# ---------------------------------------------------------------- whole comparisons
def make_job(root, name, a, b):
    jd = Path(root) / name
    jd.mkdir(parents=True)
    shutil.copyfile(a, jd / "input_a.bin")
    shutil.copyfile(b, jd / "input_b.bin")
    return jd


INV = PACK / "01_PDF_Text_Comparison"


def check_done(jd):
    prog = json.loads((jd / "progress.json").read_text(encoding="utf-8"))
    assert prog["status"] == "done" and prog["fraction"] == 1.0
    rep = json.loads((jd / "report.json").read_text(encoding="utf-8"))
    assert (jd / "report.html").is_file() and len(rep["differences"]) == 7
    assert tmp_files(jd) == []
    return rep


def test_multiple_pdf_comparisons_in_a_row(tmp_path):
    for k in range(3):
        jd = make_job(tmp_path, f"job{k:02d}aaaa", INV / "original_invoice.pdf", INV / "revised_invoice.pdf")
        W.run_job(str(jd), "a.pdf", "b.pdf", None)
        check_done(jd)


def test_comparison_survives_lost_progress_updates_and_says_so(tmp_path, monkeypatch, logs):
    """Every intermediate progress write fails (simulated WinError 5): the comparison still completes, the loss is logged."""
    real = W.write_json_atomic
    def flaky(path, data):
        if data.get("status") == "running":
            raise PermissionError(13, "Access is denied")
        return real(path, data)
    monkeypatch.setattr(W, "write_json_atomic", flaky)
    jd = make_job(tmp_path, "lostupd1", INV / "original_invoice.pdf", INV / "revised_invoice.pdf")
    W.run_job(str(jd), "a.pdf", "b.pdf", None)
    check_done(jd)
    assert "progress update #1 was lost" in logs.text()


def test_comparison_progress_is_written_and_valid_throughout(tmp_path, monkeypatch):
    seen = []
    real = W.write_json_atomic
    def spy(path, data):
        real(path, data)
        if Path(path).name == "progress.json":
            seen.append(json.loads(Path(path).read_text(encoding="utf-8")))     # readable and valid right after every write
    monkeypatch.setattr(W, "write_json_atomic", spy)
    jd = make_job(tmp_path, "spyjob01", INV / "original_invoice.pdf", INV / "revised_invoice.pdf")
    W.run_job(str(jd), "a.pdf", "b.pdf", None)
    fr = [s["fraction"] for s in seen]
    assert len(seen) > 5 and fr[-1] == 1.0 and seen[-1]["status"] == "done" and fr == sorted(fr)


def _concurrent_job(args):
    root, k = args
    from nexascan import worker as w
    jd = Path(root) / f"conc{k:04d}"
    jd.mkdir()
    shutil.copyfile(INV / "original_invoice.pdf", jd / "input_a.bin")
    shutil.copyfile(INV / "revised_invoice.pdf", jd / "input_b.bin")
    w.run_job(str(jd), "a.pdf", "b.pdf", None)
    return str(jd)


def test_multiple_concurrent_jobs(tmp_path):
    with ProcessPoolExecutor(max_workers=3, mp_context=mp.get_context("spawn")) as ex:
        dirs = list(ex.map(_concurrent_job, [(str(tmp_path), k) for k in range(3)]))
    for d in dirs:
        check_done(Path(d))
