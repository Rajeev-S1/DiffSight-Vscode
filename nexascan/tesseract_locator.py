"""Finding, configuring and verifying Tesseract OCR (standard library only, so setup can use it before pip has run).

Order in which a usable installation is looked for (the first healthy one wins; nothing is ever modified):
  1. NEXASCAN_TESSERACT         an explicit path (file or folder) set by start.bat or the user
  2. PATH                       whatever ``tesseract`` the machine already resolves
  3. common install folders     Program Files, Program Files (x86), %LOCALAPPDATA%\\Programs ... (Windows)
  4. the NexaScan-managed copy  <NexaScan>\\tools\\Tesseract-OCR (installed by setup.bat), then
                                %LOCALAPPDATA%\\NexaScan\\tools\\Tesseract-OCR (used when the first folder is not writable)
"Healthy" = it runs (``--version``) and lists both ``eng`` and ``osd`` language data. A copy that runs but lacks one of
them is reported, and used only if nothing healthy exists.

The backend uses ``configure()`` to give pytesseract the exact executable path (it no longer depends on PATH) and to point
TESSDATA_PREFIX at the matching ``tessdata`` folder. ``python -m nexascan.tesseract_locator [--check]`` prints the result;
``--check`` also runs a real OCR on a generated image (exit 0 only when the text was read correctly).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
IS_WINDOWS = os.name == "nt"
EXE = "tesseract.exe" if IS_WINDOWS else "tesseract"
NEEDED_LANGS = ("eng", "osd")
_NO_WINDOW = 0x08000000 if IS_WINDOWS else 0
_CACHE: Dict[str, Optional[dict]] = {}


def managed_dirs() -> List[Path]:
    base = Path(os.environ.get("NEXASCAN_TOOLS_DIR") or (ROOT / "tools"))
    out = [base / "Tesseract-OCR"]
    la = os.environ.get("LOCALAPPDATA")
    if la:
        out.append(Path(la) / "NexaScan" / "tools" / "Tesseract-OCR")
    return out


def _as_exe(p: str) -> Optional[Path]:
    q = Path(p)
    if q.is_dir():
        q = q / EXE
    return q if q.is_file() else None


def candidates() -> List[Tuple[str, Path]]:
    """(source, executable) pairs in priority order; duplicates removed."""
    found: List[Tuple[str, Path]] = []
    env = os.environ.get("NEXASCAN_TESSERACT")
    if env and _as_exe(env):
        found.append(("NEXASCAN_TESSERACT", _as_exe(env)))              # type: ignore[arg-type]
    w = shutil.which("tesseract")
    if w:
        found.append(("PATH", Path(w)))
    common: List[Path] = []
    for var in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432"):
        if os.environ.get(var):
            common.append(Path(os.environ[var]) / "Tesseract-OCR")
    if os.environ.get("LOCALAPPDATA"):
        common += [Path(os.environ["LOCALAPPDATA"]) / "Programs" / "Tesseract-OCR", Path(os.environ["LOCALAPPDATA"]) / "Tesseract-OCR"]
    if IS_WINDOWS:
        common.append(Path("C:/Tesseract-OCR"))
    else:
        common += [Path("/usr/bin"), Path("/usr/local/bin"), Path("/opt/homebrew/bin")]
    for d in common:
        if (d / EXE).is_file():
            found.append(("common folder", d / EXE))
    for d in managed_dirs():
        if (d / EXE).is_file():
            found.append(("DiffSight-managed", d / EXE))
    seen, out = set(), []
    for src, p in found:
        key = os.path.normcase(os.path.abspath(str(p)))
        if key not in seen:
            seen.add(key)
            out.append((src, p))
    return out


def tessdata_of(exe: Path) -> Optional[Path]:
    d = exe.parent / "tessdata"
    return d if d.is_dir() else None


def _env_for(exe: Path) -> dict:
    env = dict(os.environ)
    td = tessdata_of(exe)
    if td:
        env["TESSDATA_PREFIX"] = str(td)                                # the data that belongs to this executable
    return env


def probe(exe: Path, timeout: float = 25.0) -> dict:
    """Run the executable: version and installed languages. Never raises."""
    info = {"exe": str(exe), "ok": False, "version": "", "langs": [], "has_eng": False, "has_osd": False,
            "tessdata": str(tessdata_of(exe) or ""), "error": ""}
    try:
        r = subprocess.run([str(exe), "--version"], capture_output=True, timeout=timeout, env=_env_for(exe), creationflags=_NO_WINDOW)
        text = (r.stdout + r.stderr).decode("utf-8", "replace")
        if r.returncode != 0:
            info["error"] = f"--version exited with code {r.returncode}"
            return info
        info["version"] = text.strip().splitlines()[0] if text.strip() else ""
        info["ok"] = True
        r2 = subprocess.run([str(exe), "--list-langs"], capture_output=True, timeout=timeout, env=_env_for(exe), creationflags=_NO_WINDOW)
        langs = [ln.strip() for ln in (r2.stdout + r2.stderr).decode("utf-8", "replace").splitlines()
                 if ln.strip() and not ln.lower().startswith("list of")]
        info["langs"] = langs
        info["has_eng"], info["has_osd"] = "eng" in langs, "osd" in langs
    except subprocess.TimeoutExpired:
        info["error"] = "timed out"
    except OSError as e:
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def healthy(info: Optional[dict]) -> bool:
    return bool(info and info["ok"] and info["has_eng"] and info["has_osd"])


def find_tesseract(refresh: bool = False) -> Optional[dict]:
    """First healthy installation (else the first one that merely runs); None if there is none. Adds ``source``."""
    if not refresh and "found" in _CACHE:
        return _CACHE["found"]
    # children of a verified parent (start.bat / the API process) trust its answer instead of probing again
    pre = os.environ.get("NEXASCAN_TESSERACT")
    if not refresh and pre and os.environ.get("NEXASCAN_TESSERACT_VERIFIED") == "1" and Path(pre).is_file():
        td = tessdata_of(Path(pre))
        _CACHE["found"] = {"exe": pre, "ok": True, "version": "", "langs": [], "has_eng": True, "has_osd": True,
                           "tessdata": str(td or ""), "error": "", "source": "verified by the parent process"}
        return _CACHE["found"]
    best = None
    for src, exe in candidates():
        info = probe(exe)
        info["source"] = src
        if healthy(info):
            _CACHE["found"] = info
            return info
        if info["ok"] and best is None:
            best = info
    _CACHE["found"] = best
    return best


def configure(pytesseract_module=None) -> Optional[str]:
    """Give pytesseract the exact executable path and the matching tessdata folder. Returns the path or None."""
    info = find_tesseract()
    if not info:
        return None
    exe = info["exe"]
    if pytesseract_module is not None:
        pytesseract_module.pytesseract.tesseract_cmd = exe
    td = info.get("tessdata")
    cur = os.environ.get("TESSDATA_PREFIX")
    if td and not (cur and (Path(cur) / "eng.traineddata").is_file()):
        os.environ["TESSDATA_PREFIX"] = td                              # a stale value from another install would break OCR
    os.environ["NEXASCAN_TESSERACT"] = exe
    if healthy(info):
        os.environ["NEXASCAN_TESSERACT_VERIFIED"] = "1"                 # inherited by the worker processes
    return exe


def check_ocr(exe: Optional[str] = None) -> Tuple[bool, str]:
    """Real OCR test: draw text, read it back with the configured Tesseract (through pytesseract if available)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return False, "Pillow is not installed yet (run after the Python packages are installed)"
    info = None if exe else find_tesseract()
    exe = exe or (info["exe"] if info else None)
    if not exe:
        return False, "Tesseract was not found"
    font = None
    for make in (lambda: ImageFont.load_default(size=56), lambda: ImageFont.truetype("arial.ttf", 56), lambda: ImageFont.truetype("DejaVuSans.ttf", 56)):
        try:
            font = make()
            break
        except Exception:  # noqa: BLE001
            continue
    img = Image.new("L", (900, 140), 255)
    ImageDraw.Draw(img).text((30, 30), "NexaScan 2026", fill=0, font=font)
    tmp = Path(tempfile.mkdtemp(prefix="nexa_ocr_"))
    try:
        png = tmp / "check.png"
        img.save(png)
        r = subprocess.run([str(exe), str(png), "stdout", "--psm", "7"], capture_output=True, timeout=60, env=_env_for(Path(exe)),
                           creationflags=_NO_WINDOW)
        text = r.stdout.decode("utf-8", "replace").strip()
        ok = r.returncode == 0 and "2026" in text and "nexa" in text.lower()
        try:                                                              # and through pytesseract, exactly as the backend calls it
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = str(exe)
            via = pytesseract.image_to_string(img, config="--psm 7").strip()
            ok = ok and "2026" in via
        except ImportError:
            pass
        except Exception as e:  # noqa: BLE001 - a Tesseract that fails must give "not ok", never an exception
            ok, text = False, text or f"{type(e).__name__}: {e}"
        return ok, text or (r.stderr.decode("utf-8", "replace").strip() or "no text recognised")
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"{type(e).__name__}: {e}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    info = find_tesseract(refresh=True)
    if not info:
        print("Tesseract: NOT FOUND (looked in NEXASCAN_TESSERACT, PATH, common install folders and the DiffSight tools folder)")
        return 1
    print(f"Tesseract: {info['exe']}  [{info['source']}]  {info['version']}")
    print(f"  languages: {', '.join(info['langs']) or 'none'}   eng={'yes' if info['has_eng'] else 'NO'}  osd={'yes' if info['has_osd'] else 'NO'}")
    if not healthy(info):
        print("  WARNING: English and/or OSD language data is missing")
        return 1
    if "--check" in argv:
        ok, text = check_ocr(info["exe"])
        print(f"  OCR self-test: {'PASSED' if ok else 'FAILED'} (read: {text!r})")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
