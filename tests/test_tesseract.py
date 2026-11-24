"""Tesseract detection, configuration and automatic installation.

The real Windows installer cannot run here: the NSIS installer step is replaced by a stand-in that creates an installation
whose ``tesseract`` is a thin wrapper around the real engine of this machine, so detection, download handling, retries,
the install ladder, verification and REAL OCR are all genuinely executed. What is NOT executed here: the real download from
digi.bib.uni-mannheim.de, the real silent NSIS installer (/S /currentuser /D=...) and the UAC elevation."""
import http.server
import os
import shutil
import socketserver
import subprocess
import sys
import threading
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from nexascan import tesseract_install as ti  # noqa: E402
from nexascan import tesseract_locator as loc  # noqa: E402

REAL = shutil.which("tesseract")
pytestmark = pytest.mark.skipif(REAL is None or os.name == "nt", reason="needs a real Tesseract on a POSIX test machine")
PACK = ROOT / "tests" / "data" / "pack"


def system_tessdata():
    out = subprocess.run([REAL, "--list-langs"], capture_output=True, text=True)
    first = (out.stdout + out.stderr).splitlines()[0]
    return Path(first.split('"')[1])


def make_install(folder, with_osd=True, broken=False):
    """An 'installed Tesseract': <folder>/tesseract (wrapper around the real engine) and <folder>/tessdata."""
    folder.mkdir(parents=True, exist_ok=True)
    exe = folder / "tesseract"
    exe.write_text('#!/bin/sh\nexit 1\n' if broken else f'#!/bin/sh\nexec "{REAL}" "$@"\n')
    exe.chmod(0o755)
    td = folder / "tessdata"
    td.mkdir(exist_ok=True)
    src = system_tessdata()
    shutil.copy(src / "eng.traineddata", td / "eng.traineddata")
    if with_osd:
        shutil.copy(src / "osd.traineddata", td / "osd.traineddata")
    return exe


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """A machine where Tesseract is NOT on PATH and not in any common folder; tools folder inside tmp_path."""
    for k in ("NEXASCAN_TESSERACT", "NEXASCAN_TESSERACT_VERIFIED", "NEXASCAN_TESSERACT_URL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "emptybin"))
    monkeypatch.setenv("NEXASCAN_TOOLS_DIR", str(tmp_path / "tools"))
    monkeypatch.setenv("ProgramFiles", str(tmp_path / "ProgramFiles"))
    monkeypatch.setenv("ProgramFiles(x86)", str(tmp_path / "ProgramFilesX86"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "LocalAppData"))
    monkeypatch.setattr(loc, "IS_WINDOWS", True)            # enumerate Windows folders only (never /usr/bin)
    monkeypatch.setattr(ti, "IS_WINDOWS", True)
    loc._CACHE.clear()
    monkeypatch.setattr("time.sleep", lambda s: None)
    yield tmp_path
    loc._CACHE.clear()


class Server:
    """Local stand-in for the download site: counts requests, can send 404, truncate, or serve an HTML error page."""
    def __init__(self, mode="ok", size=12 * 1024 * 1024):
        self.mode, self.size, self.hits, self.fail_first = mode, size, 0, 0
        outer = self
        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass
            def do_GET(self):
                outer.hits += 1
                if outer.mode == "404" or outer.hits <= outer.fail_first:
                    self.send_response(404); self.end_headers(); return
                body = b"<html>error page</html>" if outer.mode == "html" else b"\0" * outer.size
                self.send_response(200)
                self.send_header("Content-Length", str(len(body) + (5_000_000 if outer.mode == "truncated" else 0)))
                self.end_headers()
                self.wfile.write(body)
        self.httpd = socketserver.TCPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/tesseract-ocr-w64-setup-9.9.9.20300101.exe"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
    def close(self):
        self.httpd.shutdown(); self.httpd.server_close()


@pytest.fixture()
def server():
    made = []
    def make(**kw):
        s = Server(**kw)
        made.append(s)
        return s
    yield make
    for s in made:
        s.close()


def fake_nsis(calls, succeed_on=1, create=True):
    """Stand-in for the silent NSIS installer: creates the installation on attempt ``succeed_on`` (1-based)."""
    def run(installer, target, mode_args, elevated):
        calls.append((mode_args, elevated))
        if len(calls) >= succeed_on and create:
            make_install(Path(target))
            return 0
        return 1
    return run


def run_install(env, logname="setup.log"):
    ti._log_file = env / logname
    return ti.run(env / "tools" / "Tesseract-OCR")


# ---------------------------------------------------------------- scenario 1: already installed and on PATH
def test_installed_and_on_path_is_reused_without_any_download(env, monkeypatch):
    bin_dir = env / "emptybin"
    make_install(bin_dir)
    monkeypatch.setattr(ti, "download", lambda *a, **k: pytest.fail("must not download"))
    monkeypatch.setattr(ti, "_run_installer", lambda *a, **k: pytest.fail("must not install"))
    assert run_install(env) == 0
    info = loc.find_tesseract(refresh=True)
    assert info["source"] == "PATH" and loc.healthy(info) and not (env / "tools").exists()
    assert "reusing it, nothing to install" in (env / "setup.log").read_text()


# ---------------------------------------------------------------- scenario 2: installed but missing from PATH
def test_installed_but_not_on_path_is_found_in_a_common_folder_and_configured_explicitly(env, monkeypatch):
    exe = make_install(env / "ProgramFiles" / "Tesseract-OCR")
    monkeypatch.setattr(ti, "download", lambda *a, **k: pytest.fail("must not download"))
    assert run_install(env) == 0
    info = loc.find_tesseract(refresh=True)
    assert info["source"] == "common folder" and Path(info["exe"]) == exe
    class FakePyt:                                              # the backend gives pytesseract the exact path
        class pytesseract:
            tesseract_cmd = "tesseract"
    assert loc.configure(FakePyt) == str(exe) and FakePyt.pytesseract.tesseract_cmd == str(exe)
    assert os.environ["NEXASCAN_TESSERACT"] == str(exe)
    ok, text = loc.check_ocr()                                  # real OCR through this installation
    assert ok and "2026" in text


def test_explicit_environment_variable_wins_and_tessdata_is_pointed_at_the_matching_folder(env, monkeypatch):
    exe = make_install(env / "custom" / "tess")
    monkeypatch.setenv("NEXASCAN_TESSERACT", str(exe))
    monkeypatch.setenv("TESSDATA_PREFIX", str(env / "stale"))     # a stale value from some other install
    info = loc.find_tesseract(refresh=True)
    assert info["source"] == "NEXASCAN_TESSERACT"
    loc.configure(None)
    assert os.environ["TESSDATA_PREFIX"] == str(exe.parent / "tessdata")


# ---------------------------------------------------------------- scenario 3: not installed -> download and install
def test_not_installed_downloads_installs_verifies_and_is_idempotent(env, server, monkeypatch):
    srv = server()
    monkeypatch.setenv("NEXASCAN_TESSERACT_URL", srv.url)
    calls = []
    monkeypatch.setattr(ti, "_run_installer", fake_nsis(calls))
    assert run_install(env) == 0
    target = env / "tools" / "Tesseract-OCR"
    assert (target / "tesseract").is_file() and calls == [("/S /currentuser", False)] and srv.hits == 1
    info = loc.find_tesseract(refresh=True)
    assert info["source"] == "DiffSight-managed" and loc.healthy(info)
    assert not list((env / "tools" / "downloads").glob("*.exe")) and not list((env / "tools" / "downloads").glob("*.part"))
    log = (env / "setup.log").read_text()
    assert "no usable Tesseract found" in log and "downloading" in log and "installed and verified" in log and "sha256" in log
    assert run_install(env, "setup2.log") == 0                  # second run: found -> nothing downloaded or installed again
    assert srv.hits == 1 and len(calls) == 1
    assert "reusing it, nothing to install" in (env / "setup2.log").read_text()


def test_install_ladder_current_user_then_default_then_elevated(env, server, monkeypatch):
    srv = server()
    monkeypatch.setenv("NEXASCAN_TESSERACT_URL", srv.url)
    calls = []
    monkeypatch.setattr(ti, "_run_installer", fake_nsis(calls, succeed_on=3))
    assert run_install(env) == 0
    assert calls == [("/S /currentuser", False), ("/S", False), ("/S", True)]


def test_installer_that_cannot_start_for_lack_of_elevation_is_survived(env, server, monkeypatch):
    srv = server()
    monkeypatch.setenv("NEXASCAN_TESSERACT_URL", srv.url)
    calls = []
    good = fake_nsis(calls, succeed_on=3)
    def run(installer, target, mode, elevated):
        if not elevated:
            calls.append((mode, elevated))
            raise OSError(740, "The requested operation requires elevation")        # WinError 740
        return good(installer, target, mode, elevated)
    monkeypatch.setattr(ti, "_run_installer", run)
    assert run_install(env) == 0 and calls[-1][1] is True or len(calls) >= 2
    assert "requires elevation" in (env / "setup.log").read_text()


# ---------------------------------------------------------------- scenario 4: download failure
@pytest.mark.parametrize("mode", ["404", "html", "truncated"])
def test_download_failures_return_code_2_and_leave_nothing_behind(env, server, monkeypatch, mode):
    srv = server(mode=mode)
    monkeypatch.setenv("NEXASCAN_TESSERACT_URL", srv.url)
    monkeypatch.setattr(ti, "_run_installer", lambda *a, **k: pytest.fail("no installer to run"))
    assert run_install(env) == 2
    left = list((env / "tools" / "downloads").glob("*")) if (env / "tools" / "downloads").exists() else []
    assert left == [] and srv.hits == ti.DOWNLOAD_ATTEMPTS
    log = (env / "setup.log").read_text()
    assert "ERROR: the Tesseract installer could not be downloaded" in log and "internet connection" in log


def test_server_not_reachable_at_all(env, monkeypatch):
    monkeypatch.setenv("NEXASCAN_TESSERACT_URL", "http://127.0.0.1:9/tesseract-ocr-w64-setup-9.9.9.exe")
    assert run_install(env) == 2


def test_interrupted_download_is_retried_and_then_succeeds(env, server, monkeypatch):
    srv = server()
    srv.fail_first = 2                                           # two failures, third attempt works
    monkeypatch.setenv("NEXASCAN_TESSERACT_URL", srv.url)
    monkeypatch.setattr(ti, "_run_installer", fake_nsis([]))
    assert run_install(env) == 0 and srv.hits == 3


# ---------------------------------------------------------------- scenario 5: installation failure
def test_installation_failure_returns_3_keeps_the_installer_and_the_retry_reuses_it(env, server, monkeypatch):
    srv = server()
    monkeypatch.setenv("NEXASCAN_TESSERACT_URL", srv.url)
    monkeypatch.setattr(ti, "_run_installer", fake_nsis([], create=False))
    assert run_install(env) == 3
    kept = list((env / "tools" / "downloads").glob("*.exe"))
    assert len(kept) == 1 and "did not produce a working tesseract.exe" in (env / "setup.log").read_text()
    monkeypatch.setattr(ti, "_run_installer", fake_nsis([]))                   # fixed: second run does not download again
    assert run_install(env, "setup2.log") == 0 and srv.hits == 1
    assert "re-using the installer downloaded earlier" in (env / "setup2.log").read_text()


def test_missing_language_data_is_downloaded_after_installation(env, server, monkeypatch):
    srv = server()
    monkeypatch.setenv("NEXASCAN_TESSERACT_URL", srv.url)
    def run(installer, target, mode, elevated):
        make_install(Path(target), with_osd=False)
        return 0
    monkeypatch.setattr(ti, "_run_installer", run)
    served = system_tessdata() / "osd.traineddata"
    def fake_urlopen(req, timeout=0):
        if req.full_url.endswith("osd.traineddata"):
            return open(served, "rb")
        return urllib_real(req, timeout=timeout)
    import urllib.request
    urllib_real = urllib.request.urlopen
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert run_install(env) == 0 and "osd.traineddata is missing" in (env / "setup.log").read_text()


# ---------------------------------------------------------------- existing installations are never touched
def test_existing_installation_without_osd_is_left_untouched_and_a_separate_copy_is_installed(env, server, monkeypatch):
    old = env / "ProgramFiles" / "Tesseract-OCR"
    make_install(old, with_osd=False)
    before = sorted((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in old.rglob("*") if p.is_file())
    srv = server()
    monkeypatch.setenv("NEXASCAN_TESSERACT_URL", srv.url)
    monkeypatch.setattr(ti, "_run_installer", fake_nsis([]))
    assert run_install(env) == 0
    after = sorted((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in old.rglob("*") if p.is_file())
    assert before == after                                                     # nothing changed in the existing installation
    assert loc.find_tesseract(refresh=True)["source"] == "DiffSight-managed"     # the healthy copy is preferred over the incomplete one


def test_unwritable_tools_folder_falls_back_to_localappdata(env, server, monkeypatch):
    srv = server()
    monkeypatch.setenv("NEXASCAN_TESSERACT_URL", srv.url)
    monkeypatch.setattr(ti, "_run_installer", fake_nsis([]))
    tools_target = env / "tools" / "Tesseract-OCR"
    real_mkdir = Path.mkdir
    def mkdir(self, *a, **k):
        if Path(self) == tools_target:
            raise PermissionError(13, "Access is denied")
        return real_mkdir(self, *a, **k)
    monkeypatch.setattr(Path, "mkdir", mkdir)
    assert run_install(env) == 0
    assert (env / "LocalAppData" / "NexaScan" / "tools" / "Tesseract-OCR" / "tesseract").is_file()
    assert "cannot write to" in (env / "setup.log").read_text()


# ---------------------------------------------------------------- scenario 6: restart, and scenario 7: OCR really works
def test_after_a_restart_a_fresh_process_finds_the_managed_install_without_PATH(env, server, monkeypatch):
    srv = server()
    monkeypatch.setenv("NEXASCAN_TESSERACT_URL", srv.url)
    monkeypatch.setattr(ti, "_run_installer", fake_nsis([]))
    assert run_install(env) == 0
    code = "import nexascan.tesseract_locator as l; l.IS_WINDOWS = True; raise SystemExit(l.main(['--check']))"
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PATH": str(env / "emptybin")})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "[DiffSight-managed]" in r.stdout and "OCR self-test: PASSED" in r.stdout


def test_the_ocr_check_reads_real_text_and_is_not_fooled_by_a_broken_or_fake_executable(env):
    good = make_install(env / "good")
    ok, text = loc.check_ocr(str(good))
    assert ok and "nexa" in text.lower() and "2026" in text
    broken = make_install(env / "broken", broken=True)                           # exists, but does not run
    assert not loc.probe(broken)["ok"] and not loc.check_ocr(str(broken))[0]
    fake = env / "fake" / "tesseract"                                           # runs and lists languages, reads nothing useful
    fake.parent.mkdir()
    fake.write_text('#!/bin/sh\nif [ "$1" = "--version" ]; then echo "tesseract 5.9"; exit 0; fi\n'
                    'if [ "$1" = "--list-langs" ]; then printf "List of available languages (2):\\neng\\nosd\\n"; exit 0; fi\n'
                    'echo "garbage"; exit 0\n')
    fake.chmod(0o755)
    info = loc.probe(fake)
    assert loc.healthy(info) and not loc.check_ocr(str(fake))[0]               # "exists and runs" is not enough: OCR must work


def test_backend_ocr_works_with_an_empty_PATH(env):
    """The backend must not depend on the global PATH: a comparison of two scans runs with PATH empty."""
    code = (
        "import warnings; warnings.filterwarnings('ignore')\n"
        "import nexascan.ocr as o, pytesseract, os\n"
        "print('CMD', pytesseract.pytesseract.tesseract_cmd, o.tesseract_available())\n"
        "from pathlib import Path\n"
        "from nexascan.config import Config\n"
        "from nexascan.pipeline import compare_documents\n"
        f"P = Path(r'{PACK}') / '02_JPG_Scanned_OCR'\n"
        "r = compare_documents(str(P/'original_lab_report.jpg'), 'a.jpg', str(P/'revised_lab_report.jpg'), 'b.jpg', Path(os.environ['OUT']), Config(use_cache=False))\n"
        "print('CONFIRMED', len(r['differences']))\n")
    out = env / "out"
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PATH": "/nonexistent", "OUT": str(out), "NEXASCAN_TOOLS_DIR": str(env / "tools")})
    assert r.returncode == 0, r.stderr[-800:]
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("CMD")][0]
    assert os.path.isabs(line.split()[1]) and line.endswith("True")          # an exact path, not just the word "tesseract"
    assert "CONFIRMED 3" in r.stdout                                         # the scan comparison itself is unchanged
