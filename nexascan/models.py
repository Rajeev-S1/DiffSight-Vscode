"""Plain data containers shared by all services."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

BBox = Tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels of the page frame


@dataclass
class DocInfo:
    """A validated upload."""
    path: str
    kind: str                 # "pdf" | "image"
    sha256: str
    pages: int
    display_name: str
    size_bytes: int = 0


@dataclass
class Word:
    text: str
    bbox: BBox
    conf: float = 100.0       # 0..100 (native text = 100)
    line: int = 0
    source: str = "native"    # native | ocr
    font: str = ""
    size: float = 0.0         # pt, native only
    bold: bool = False
    italic: bool = False
    color: int = 0            # 0xRRGGBB, native only
    char_boxes: Optional[List[BBox]] = None
    text_exact: str = ""
    paragraph_id: int = -1
    block_id: int = -1

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(d: dict) -> "Word":
        d = dict(d)
        d["bbox"] = tuple(d["bbox"])
        if d.get("char_boxes"):
            d["char_boxes"] = [tuple(b) for b in d["char_boxes"]]
        return Word(**d)


@dataclass
class Line:
    idx: int
    words: List[int]
    bbox: BBox
    text: str
    conf: float = 100.0
    alt_texts: List[str] = field(default_factory=list)   # readings from other OCR engines
    text_exact: str = ""
    paragraph_id: int = -1
    block_id: int = -1
    whitespace_runs: List[Dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(d: dict) -> "Line":
        d = dict(d)
        d["bbox"] = tuple(d["bbox"])
        return Line(**d)


@dataclass
class PageData:
    """Everything that is derived from one page of one document and is cacheable.
    Pixel arrays are *not* stored: they are re-rendered deterministically on demand."""
    doc_sha: str
    index: int                # 0-based
    width: int
    height: int
    dpi: int
    source: str               # "native" (digital text layer) | "scan"
    rotation_applied: int = 0  # multiples of 90, clockwise
    skew_deg: float = 0.0
    scale: float = 1.0         # raster normalisation factor applied to image inputs
    noise_sigma: float = 0.0
    glyph_h: float = 0.0       # median text height in px
    ink_frac: float = 0.0
    words: List[Word] = field(default_factory=list)
    lines: List[Line] = field(default_factory=list)
    # OCR reading of a *native* page, computed only when it is compared with a scan so that
    # both sides go through the same OCR engine (systematic OCR errors then cancel out).
    ocr_words: Optional[List[Word]] = None
    ocr_lines: Optional[List[Line]] = None
    objects: List[dict] = field(default_factory=list)
    native_images: List[dict] = field(default_factory=list)
    native_drawings: List[dict] = field(default_factory=list)
    page_mm: Tuple[float, float] = (0.0, 0.0)
    thumb: List[float] = field(default_factory=list)   # 24x34 ink-density thumbnail for page matching
    notes: List[str] = field(default_factory=list)

    def to_json(self) -> dict:
        d = asdict(self)
        d["words"] = [w.to_json() for w in self.words]
        d["lines"] = [l.to_json() for l in self.lines]
        d["ocr_words"] = None if self.ocr_words is None else [w.to_json() for w in self.ocr_words]
        d["ocr_lines"] = None if self.ocr_lines is None else [l.to_json() for l in self.ocr_lines]
        return d

    @staticmethod
    def from_json(d: dict) -> "PageData":
        d = dict(d)
        d["words"] = [Word.from_json(w) for w in d["words"]]
        d["lines"] = [Line.from_json(l) for l in d["lines"]]
        if d.get("ocr_words") is not None:
            d["ocr_words"] = [Word.from_json(w) for w in d["ocr_words"]]
            d["ocr_lines"] = [Line.from_json(l) for l in d["ocr_lines"]]
        d["page_mm"] = tuple(d.get("page_mm", (0.0, 0.0)))
        return PageData(**d)


@dataclass
class Finding:
    category: str                      # Added | Removed | Modified | Moved | Overlapped | Uncertain
    dtype: str                         # Text | Punctuation | Number | Stamp | Signature | Handwriting | Image | ...
    page_a: Optional[int]              # 1-based
    page_b: Optional[int]
    a_content: str
    b_content: str
    description: str
    bbox_a: Optional[BBox]             # A page frame, pixels at report dpi
    bbox_b: Optional[BBox]             # B page frame (own frame), pixels
    confidence: float
    engines: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    status: str = "confirmed"          # confirmed | uncertain
    evidence: Dict[str, str] = field(default_factory=dict)
    id: int = 0
    checklist_categories: List[str] = field(default_factory=list)
    detection_method: str = ""
    character_change: Dict[str, Any] = field(default_factory=dict)
    whitespace_change: Dict[str, Any] = field(default_factory=dict)
    # internal: region in A-frame used for de-duplication (not exported as-is)
    region: Optional[BBox] = None

    def to_json(self) -> dict:
        d = asdict(self)
        return d


def bbox_union(boxes) -> Optional[BBox]:
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def bbox_area(b: BBox) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def bbox_inter(a: BBox, b: BBox) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def bbox_iou(a: BBox, b: BBox) -> float:
    i = bbox_inter(a, b)
    u = bbox_area(a) + bbox_area(b) - i
    return i / u if u > 0 else 0.0


def bbox_pad(b: BBox, p: float) -> BBox:
    return (b[0] - p, b[1] - p, b[2] + p, b[3] + p)
