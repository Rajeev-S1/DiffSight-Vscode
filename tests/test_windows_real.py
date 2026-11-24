"""REAL Windows file-locking test (skipped on other systems). Run on Windows:  python -m pytest tests/test_windows_real.py -v

Reproduces the original bug: a reader keeps progress.json open (exactly what the status poll of the web page does) while the
worker replaces it. With Python's default open() Windows refuses the replace with WinError 5 / 32."""
import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from nexascan import worker as W  # noqa: E402

windows_only = pytest.mark.skipif(os.name != "nt", reason="needs real Windows file-sharing semantics")


@windows_only
def test_plain_os_replace_really_fails_while_a_reader_holds_the_file(tmp_path):
    p = tmp_path / "progress.json"
    p.write_text('{"a": 1}')
    tmp = tmp_path / "new.tmp"
    tmp.write_text('{"a": 2}')
    with open(p, "r", encoding="utf-8"):                    # the API's status poll
        with pytest.raises(PermissionError):                # WinError 5 (or 32): the bug this project had
            os.replace(tmp, p)


@windows_only
def test_write_json_atomic_waits_for_the_reader_instead_of_failing(tmp_path):
    p = tmp_path / "progress.json"
    p.write_text('{"a": 1}')
    held = threading.Event()
    def reader():
        with open(p, "r", encoding="utf-8"):
            held.set()
            time.sleep(0.6)                                  # a slow poll / a virus scanner
    t = threading.Thread(target=reader)
    t.start()
    held.wait()
    W.write_json_atomic(p, {"a": 2})                         # must not raise
    t.join()
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": 2}
    assert [f.name for f in tmp_path.iterdir() if f.name.endswith(".tmp")] == []


@windows_only
def test_hammering_the_status_poll_while_updating_progress(tmp_path):
    p = tmp_path / "progress.json"
    W.write_json_atomic(p, {"i": -1})
    stop, errors = threading.Event(), []
    def poller():
        while not stop.is_set():
            try:
                d = W.read_json(p)
                assert d is not None
            except Exception as e:  # noqa: BLE001
                errors.append(e)
    ts = [threading.Thread(target=poller) for _ in range(4)]
    for t in ts:
        t.start()
    for i in range(400):
        W.write_json_atomic(p, {"i": i})
    stop.set()
    for t in ts:
        t.join()
    assert errors == [] and json.loads(p.read_text(encoding="utf-8")) == {"i": 399}
