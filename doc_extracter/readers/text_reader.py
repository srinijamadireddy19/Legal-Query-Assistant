"""
Plain Text Reader
─────────────────
Handles .txt, .md, .rst, .csv files.
Splits on double-newlines into "pages" (logical paragraphs/sections).
Each chunk of MAX_CHARS characters becomes its own PageResult.
"""

import logging
from pathlib import Path
from typing import List

from ..core.base_reader import BaseReader, registry
from ..core.models import DocumentResult, IngestionStatus, PageResult, PageType

log = logging.getLogger(__name__)

MAX_CHARS_PER_PAGE = 3000   # Split long text into logical pages


@registry.register
class TextReader(BaseReader):
    SUPPORTED_EXTENSIONS = ["txt", "md", "rst", "text"]

    def read(self, file_path: str, **kwargs) -> DocumentResult:
        path = Path(file_path)
        errors: List[str] = []

        # Try common encodings
        raw = None
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                raw = path.read_text(encoding=enc)
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if raw is None:
            return DocumentResult(
                file_path=str(path),
                file_type=path.suffix.lstrip(".").lower(),
                total_pages=0,
                pages=[],
                status=IngestionStatus.FAILED,
                errors=["Could not decode file with any supported encoding."],
            )

        # Split into logical chunks
        chunks = self._split_text(raw)
        pages  = [
            PageResult(page_number=i + 1, text=chunk, page_type=PageType.NATIVE_TEXT)
            for i, chunk in enumerate(chunks)
        ]

        return DocumentResult(
            file_path=str(path),
            file_type=path.suffix.lstrip(".").lower(),
            total_pages=len(pages),
            pages=pages,
            status=IngestionStatus.SUCCESS if pages else IngestionStatus.PARTIAL,
            metadata={"encoding_used": enc, "raw_length": len(raw)},
            errors=errors,
        )

    def _split_text(self, text: str) -> List[str]:
        """
        Split on blank lines into paragraphs, then group paragraphs into
        page-sized chunks not exceeding MAX_CHARS_PER_PAGE.
        """
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks: List[str] = []
        current = []
        current_len = 0

        for para in paragraphs:
            if current_len + len(para) > MAX_CHARS_PER_PAGE and current:
                chunks.append("\n\n".join(current))
                current     = []
                current_len = 0
            current.append(para)
            current_len += len(para)

        if current:
            chunks.append("\n\n".join(current))

        return chunks or [""]
