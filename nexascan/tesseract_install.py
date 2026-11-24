"""Automatic Tesseract OCR installation for Windows (standard library only; used by setup.bat).

    python -m nexascan.tesseract_install [--target DIR] [--log FILE]

What it does, in order (and nothing more):
  1. Looks for a usable Tesseract (NEXASCAN_TESSERACT, PATH, common folders, the NexaScan-managed copy). If a healthy one
     exists it is REUSED: nothing is downloaded or installed, and no existing installation is ever touched.
  2. Otherwise downloads the official Windows installer (Tesseract at UB Mannheim, the build the Tesseract project itself
     points Windows users to) with retries, into <tools>\\downloads as ``.part`` and renames it only when complete.
  3. Installs it silently into <NexaScan>\\tools\\Tesseract-OCR (NSIS switches /S and /D=...), trying "current user" first,
     then the default mode, then - only if both fail - one elevated attempt (a Windows UAC prompt appears).
  4. Makes sure eng.traineddata and osd.traineddata exist (downloads them from the official tessdata_fast repository if not).
  5. Verifies the result by running tesseract.exe (--version and --list-langs).
Exit codes: 0 ready (reused or installed) | 2 download failed | 3 installation failed | 4 installed but verification failed |
            5 not Windows and nothing usable found (install it with the system package manager).
Every step is logged to --log (and printed). Safe to run repeatedly.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

from . import tesseract_locator as loc

IS_WINDOWS = os.name == "nt"
DOWNLOAD_BASE = "https://digi.bib.uni-mannheim.de/tesseract/"
WIKI_URL = "https://github.com/UB-Mannheim/tesseract/wiki"
TESSDATA_BASE = "https://github.com/tesseract-ocr/tessdata_fast/raw/main/"
# Used when the wiki cannot be read. Newest first; the wiki page lists the same file names.
PINNED = ["tesseract-ocr-w64-setup-5.5.3.20260724.exe", "tesseract-ocr-w64-setup-5.5.0.20241111.exe"]
MIN_INSTALLER_BYTES = 10 * 1024 * 1024          # a real installer is ~50 MB; anything smaller is an error page / truncated
DOWNLOAD_ATTEMPTS = 3
INSTALL_TIMEOUT_S = 900
USER_AGENT = "NexaScan-setup/1.0 (+Tesseract installer fetch)"

_log_file: Optional[Path] = None


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [tesseract] {msg}"
    print(line, flush=True)
    if _log_file is not None:
        try:
            with open(_log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass


# ------------------------------------------------------------------ download
def _version_key(name: str):
    m = re.search(r"setup-(?:v)?([0-9][0-9.]*)", name)
    return tuple(int(x) for x in m.group(1).split(".") if x.isdigit()) if m else ()


def installer_urls(fetch: Callable[[str], str] = None) -> List[str]:
    """Download URLs to try: NEXASCAN_TESSERACT_URL, else the newest names found on the wiki, then the pinned names."""
    if os.environ.get("NEXASCAN_TESSERACT_URL"):
        return [os.environ["NEXASCAN_TESSERACT_URL"]]
    names: List[str] = []
    try:
        html = (fetch or _fetch_text)(WIKI_URL)
        found = sorted(set(re.findall(r"tesseract-ocr-w64-setup-[0-9][0-9A-Za-z.\-]*\.exe", html)), key=_version_key, reverse=True)
        names += found[:2]
        log(f"newest installer names on the UB Mannheim wiki: {', '.join(found[:2]) or 'none found'}")
    except Exception as e:  # noqa: BLE001
        log(f"could not read the installer list ({type(e).__name__}: {e}); using the built-in names")
    names += [n for n in PINNED if n not in names]
    return [DOWNLOAD_BASE + n for n in names]


def _fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def download(url: str, dest: Path, attempts: int = DOWNLOAD_ATTEMPTS, sleep: Callable[[float], None] = time.sleep) -> str:
    """Download ``url`` to ``dest`` (via dest.part). Retries on network errors, truncated and too-small downloads.
    Returns the SHA-256 of the finished file; raises OSError when every attempt failed. A .part file never survives."""
    dest = Path(dest)
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    last: Optional[BaseException] = None
    for n in range(1, attempts + 1):
        try:
            log(f"downloading {url} (attempt {n}/{attempts})")
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            sha, size = hashlib.sha256(), 0
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
                expected = r.headers.get("Content-Length")
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    sha.update(chunk)
                    size += len(chunk)
            if expected and size != int(expected):
                raise OSError(f"interrupted download: received {size} of {expected} bytes")
            if size < MIN_INSTALLER_BYTES:
                raise OSError(f"download is only {size} bytes - not a Tesseract installer")
            os.replace(part, dest)
            log(f"download complete: {size / 1048576:.1f} MB, sha256 {sha.hexdigest()}")
            return sha.hexdigest()
        except (urllib.error.URLError, http.client.HTTPException, OSError, TimeoutError) as e:
            last = e
            log(f"download failed: {type(e).__name__}: {e}")
        finally:
            try:
                part.unlink()
            except OSError:
                pass
        if n < attempts:
            sleep(2.0 * n)
    raise OSError(f"could not download {url}: {last}")


# ------------------------------------------------------------------ installation
def _run_installer(installer: Path, target: Path, mode_args: str, elevated: bool) -> int:
    """Run the NSIS installer silently. ``/D=`` must be the last argument and must not be quoted."""
    args = f"{mode_args} /D={target}".strip()
    if not elevated:
        return subprocess.run(f'"{installer}" {args}', timeout=INSTALL_TIMEOUT_S, creationflags=loc._NO_WINDOW).returncode
    ps = (f"$p = Start-Process -FilePath '{str(installer).replace(chr(39), chr(39) * 2)}' "
          f"-ArgumentList '{args.replace(chr(39), chr(39) * 2)}' -Verb RunAs -Wait -PassThru; exit $p.ExitCode")
    return subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps], timeout=INSTALL_TIMEOUT_S).returncode


def install(installer: Path, target: Path, run: Callable = None) -> bool:
    """Silent install into ``target``; True when tesseract runs afterwards. Tries current-user, default, then elevated."""
    run = run or _run_installer
    for label, mode, elevated in (("current user", "/S /currentuser", False), ("default mode", "/S", False),
                                  ("elevated (a Windows permission prompt appears)", "/S", True)):
        log(f"installing into {target} - {label}")
        try:
            code = run(installer, target, mode, elevated)
            log(f"installer finished with exit code {code}")
        except subprocess.TimeoutExpired:
            log(f"installer did not finish within {INSTALL_TIMEOUT_S // 60} minutes")
            continue
        except OSError as e:
            log(f"installer could not be started: {type(e).__name__}: {e}")      # e.g. WinError 740: elevation required
            continue
        if (target / loc.EXE).is_file() and loc.probe(target / loc.EXE)["ok"]:
            return True
        log(f"{loc.EXE} is not usable after this attempt")
    return False


def ensure_language_data(target: Path, fetch: Callable[[str, Path], str] = None) -> None:
    """eng and osd must exist; a missing file is downloaded from the official tessdata_fast repository."""
    td = target / "tessdata"
    td.mkdir(parents=True, exist_ok=True)
    for lang in loc.NEEDED_LANGS:
        f = td / f"{lang}.traineddata"
        if f.is_file() and f.stat().st_size > 10_000:
            continue
        log(f"{f.name} is missing - downloading it from tessdata_fast")
        tmp = f.with_name(f.name + ".part")
        try:
            req = urllib.request.Request(TESSDATA_BASE + f.name, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as out:
                shutil.copyfileobj(r, out)
            if tmp.stat().st_size < 10_000:
                raise OSError("file too small")
            os.replace(tmp, f)
        except (OSError, urllib.error.URLError, http.client.HTTPException) as e:
            log(f"could not download {f.name}: {e}")
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass


# ------------------------------------------------------------------ main flow
def _writable_target(target: Path) -> Path:
    """Use ``target``; when it cannot be created (permissions) fall back to %LOCALAPPDATA%\\NexaScan\\tools\\Tesseract-OCR."""
    try:
        target.mkdir(parents=True, exist_ok=True)
        probe = target / ".write_test"
        probe.write_text("x")
        probe.unlink()
        return target
    except OSError as e:
        log(f"cannot write to {target} ({type(e).__name__}: {e})")
        alt = loc.managed_dirs()[-1]
        if alt != target:
            log(f"using {alt} instead")
            alt.mkdir(parents=True, exist_ok=True)
            return alt
        raise


def run(target: Path, downloads: Optional[Path] = None) -> int:
    log("checking for an existing Tesseract OCR installation")
    info = loc.find_tesseract(refresh=True)
    if loc.healthy(info):
        log(f"found and working: {info['exe']} [{info['source']}] {info['version']} - reusing it, nothing to install")
        return 0
    if info:
        log(f"found {info['exe']} but its language data is incomplete (eng={info['has_eng']}, osd={info['has_osd']}); "
            "it is left untouched and a separate DiffSight copy is installed")
    else:
        log("no usable Tesseract found")
    if not IS_WINDOWS:
        log("automatic installation is only available on Windows. Install Tesseract with your package manager "
            "(for example: sudo apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-osd).")
        return 5

    try:
        target = _writable_target(Path(target))
    except OSError as e:
        log(f"ERROR: no writable folder for the managed installation: {e}")
        return 3
    downloads = Path(downloads or target.parent / "downloads")

    installer: Optional[Path] = None
    cached = sorted(downloads.glob("tesseract-ocr-w64-setup-*.exe"), key=lambda f: _version_key(f.name), reverse=True) if downloads.is_dir() else []
    if cached and cached[0].stat().st_size >= MIN_INSTALLER_BYTES:
        installer = cached[0]
        log(f"re-using the installer downloaded earlier: {installer.name}")
    else:
        errors = []
        for url in installer_urls():
            try:
                download(url, downloads / url.rsplit("/", 1)[-1])
                installer = downloads / url.rsplit("/", 1)[-1]
                break
            except OSError as e:
                errors.append(str(e))
        if installer is None:
            log("ERROR: the Tesseract installer could not be downloaded. Check the internet connection / proxy / firewall "
                "and run setup.bat again, or install Tesseract manually: https://github.com/UB-Mannheim/tesseract/wiki")
            return 2

    if not install(installer, target):
        log("ERROR: the silent installation did not produce a working tesseract.exe. Install it manually from "
            "https://github.com/UB-Mannheim/tesseract/wiki (the installer file is kept in " + str(downloads) + ")")
        return 3
    ensure_language_data(target)
    final = loc.probe(target / loc.EXE)
    if not (final["ok"] and final["has_eng"] and final["has_osd"]):
        log(f"ERROR: installed, but verification failed: {final}")
        return 4
    log(f"installed and verified: {final['exe']} {final['version']} (languages: {', '.join(final['langs'])})")
    try:                                                      # the installer is no longer needed
        installer.unlink()
    except OSError:
        pass
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    global _log_file
    ap = argparse.ArgumentParser(description="Install Tesseract OCR for DiffSight (Windows)")
    ap.add_argument("--target", default=str(loc.managed_dirs()[0]))
    ap.add_argument("--log", default="")
    a = ap.parse_args(argv)
    _log_file = Path(a.log) if a.log else None
    try:
        return run(Path(a.target))
    except Exception as e:  # noqa: BLE001 - setup must get an exit code and a log line, never a traceback only
        log(f"ERROR: unexpected failure: {type(e).__name__}: {e}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
