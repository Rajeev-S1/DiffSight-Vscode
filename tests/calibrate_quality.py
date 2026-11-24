"""Print sharpness / noise / contrast / tilt for clean vs degraded pages (used to choose the defaults in config.py).

    python tests/calibrate_quality.py [data/suite]
"""
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from nexascan import ingest, preprocess  # noqa: E402
from nexascan.config import Config  # noqa: E402
from nexascan.upload import validate_file  # noqa: E402


def inspect_gray(path, name, index=0, dpi=200):
    cfg = Config(dpi=dpi)
    info = validate_file(path, name, cfg)
    bgr, _ = ingest.render_raw(info, index, cfg)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def to_inspect(gray, long_side=2338):
    """What the inspection sees of a 300 DPI scan: the page at the inspection resolution (A4 long side ~ 200 DPI)."""
    s = long_side / float(max(gray.shape))
    return cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray


def measures(gray):
    sk, conf = preprocess.estimate_skew(gray)
    return dict(sharp=round(preprocess.estimate_sharpness(gray), 1), noise=round(preprocess.estimate_noise(gray), 2),
                contrast=round(preprocess.estimate_contrast(gray), 0), tilt=round(sk, 2))


def degrade(gray, kind, level):
    g = gray.astype(np.float32)
    if kind == "blur":
        return cv2.GaussianBlur(gray, (0, 0), level)
    if kind == "noise":
        rng = np.random.default_rng(7)
        return np.clip(g + rng.normal(0, level, g.shape), 0, 255).astype(np.uint8)
    if kind == "contrast":
        return np.clip(255 - (255 - g) * level, 0, 255).astype(np.uint8)
    return gray


if __name__ == "__main__":
    suite = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data" / "suite")
    print("== real / suite pages (inspect render 100 DPI) ==")
    for d in sorted(suite.glob("s*")):
        for side in ("A", "B"):
            f = next(iter(d.glob(f"{side}.*")), None)
            if f is None:
                continue
            print(f"{d.name:26s}{side} ", measures(inspect_gray(str(f), f.name)))
    pack = ROOT / "tests" / "data" / "pack"
    for f in sorted(pack.glob("*/*")):
        if f.suffix.lower() in (".jpg", ".png", ".pdf") and "original" in f.name:
            print(f"pack {f.name:34s}", measures(inspect_gray(str(f), f.name)))
    base = next(suite.glob("s01_scan_identical/A.*"))
    clean = inspect_gray(str(base), base.name, dpi=300)         # a clean scan rendered at 300 DPI, then degraded
    print("== synthetic degradations of s01 (degrade at 300 DPI, measure at the same scale) ==")
    print("clean            ", measures(to_inspect(clean)))
    for kind, levels in (("blur", (1.0, 2.0, 3.0, 4.0, 6.0)), ("noise", (8, 15, 25, 40, 60)), ("contrast", (0.7, 0.5, 0.4, 0.3))):
        for lv in levels:
            print(f"{kind:9s}{lv:<7}", measures(to_inspect(degrade(clean, kind, lv))))
