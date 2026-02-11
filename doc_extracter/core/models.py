"""
Core data models for the document input pipeline.
Every reader produces a DocumentResult regardless of source format.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any


class PageType(Enum):
    """How the text on this page was obtained."""
    NATIVE_TEXT   = "native_text"    # PDF text layer / docx text
    OCR           = "ocr"            # Tesseract OCR from image
    HYBRID        = "hybrid"         # Mix: native where available, OCR for image regions


class IngestionStatus(Enum):
    SUCCESS       = "success"
    PARTIAL       = "partial"        # Some pages failed
    FAILED        = "failed"


@dataclass
class PageResult:
    """Text + metadata for a single page / section."""
    page_number:  int
    text:         str
    page_type:    PageType
    char_count:   int              = field(init=False)
    word_count:   int              = field(init=False)
    confidence:   Optional[float] = None   # OCR confidence 0-100, None for native
    error:        Optional[str]   = None

    def __post_init__(self):
        self.char_count = len(self.text)
        self.word_count = len(self.text.split()) if self.text else 0


@dataclass
class DocumentResult:
    """
    Unified output from any reader.
    Downstream code (chunker, embedder, etc.) only ever sees this.
    """
    file_path:    str
    file_type:    str                       # "pdf", "docx", "doc", "txt", "image"
    total_pages:  int
    pages:        List[PageResult]
    status:       IngestionStatus
    metadata:     Dict[str, Any]            = field(default_factory=dict)
    errors:       List[str]                 = field(default_factory=list)

    # ── convenience properties ──────────────────────────────────────────

    @property
    def full_text(self) -> str:
        """All pages concatenated with page markers."""
        parts = []
        for p in self.pages:
            if p.text.strip():
                parts.append(f"[PAGE {p.page_number}]\n{p.text.strip()}")
        return "\n\n".join(parts)

    @property
    def plain_text(self) -> str:
        """All pages concatenated, no markers — ready for chunker."""
        return "\n\n".join(p.text.strip() for p in self.pages if p.text.strip())

    @property
    def ocr_page_count(self) -> int:
        return sum(1 for p in self.pages if p.page_type == PageType.OCR)

    @property
    def native_page_count(self) -> int:
        return sum(1 for p in self.pages if p.page_type == PageType.NATIVE_TEXT)

    @property
    def avg_ocr_confidence(self) -> Optional[float]:
        scores = [p.confidence for p in self.pages if p.confidence is not None]
        return round(sum(scores) / len(scores), 1) if scores else None

    def summary(self) -> str:
        lines = [
            f"File       : {self.file_path}",
            f"Type       : {self.file_type}",
            f"Status     : {self.status.value}",
            f"Pages      : {self.total_pages}",
            f"Native     : {self.native_page_count}",
            f"OCR        : {self.ocr_page_count}",
        ]
        if self.avg_ocr_confidence is not None:
            lines.append(f"OCR conf.  : {self.avg_ocr_confidence}%")
        if self.errors:
            lines.append(f"Errors     : {len(self.errors)}")
        return "\n".join(lines)
