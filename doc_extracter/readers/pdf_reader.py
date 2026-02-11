"""
PDF Reader
──────────
Strategy per page:

  1. Try pdfplumber → if text yield ≥ MIN_CHAR_THRESHOLD  →  NATIVE_TEXT
  2. Else convert page to image (pdf2image/poppler)
     → run Tesseract OCR                               →  OCR
  3. If pdfplumber gave *some* text but OCR gave more,
     pick the longer one and mark as HYBRID            →  HYBRID

This handles:
  • Normal PDFs (text layer present)
  • Fully scanned PDFs (every page is an image)
  • Mixed PDFs (some native pages, some scanned pages)
  • Searchable PDFs where OCR was baked in (native wins)
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PyPDF2 import PdfReader

from ..core.base_reader import BaseReader, registry
from ..core.models import DocumentResult, IngestionStatus, PageResult, PageType

log = logging.getLogger(__name__)

# A page with fewer than this many chars from native extraction is treated
# as image-only and handed to OCR.
MIN_CHAR_THRESHOLD = 30
OCR_DPI = 300          # Higher = better OCR quality, slower
OCR_LANG = "eng"       # tesseract lang; extend: "eng+fra" etc.


@registry.register
class PDFReader(BaseReader):
    SUPPORTED_EXTENSIONS = ["pdf"]

    def __init__(
        self,
        min_chars: int = MIN_CHAR_THRESHOLD,
        ocr_dpi: int = OCR_DPI,
        ocr_lang: str = OCR_LANG,
    ):
        self.min_chars = min_chars
        self.ocr_dpi   = ocr_dpi
        self.ocr_lang  = ocr_lang

    # ── public ────────────────────────────────────────────────────────────

    def read(self, file_path: str, **kwargs) -> DocumentResult:
        path   = Path(file_path)
        errors: list[str] = []
        pages:  list[PageResult] = []

        # Basic metadata from pypdf
        metadata = self._extract_metadata(file_path)

        try:
            with pdfplumber.open(file_path) as pdf:
                n_pages = len(pdf.pages)
                metadata["total_pages"] = n_pages

                for i, plumber_page in enumerate(pdf.pages):
                    page_num = i + 1
                    try:
                        page_result = self._process_page(
                            plumber_page, page_num, file_path, n_pages
                        )
                        pages.append(page_result)
                    except Exception as exc:
                        msg = f"Page {page_num}: {exc}"
                        log.warning(msg)
                        errors.append(msg)
                        pages.append(PageResult(
                            page_number=page_num,
                            text="",
                            page_type=PageType.OCR,
                            error=str(exc),
                        ))

        except Exception as exc:
            errors.append(f"Failed to open PDF: {exc}")
            return DocumentResult(
                file_path=str(path),
                file_type="pdf",
                total_pages=0,
                pages=[],
                status=IngestionStatus.FAILED,
                metadata=metadata,
                errors=errors,
            )

        status = self._resolve_status(pages, errors)
        return DocumentResult(
            file_path=str(path),
            file_type="pdf",
            total_pages=len(pages),
            pages=pages,
            status=status,
            metadata=metadata,
            errors=errors,
        )

    # ── private ───────────────────────────────────────────────────────────

    def _process_page(
        self,
        plumber_page,
        page_num: int,
        file_path: str,
        n_pages: int,
    ) -> PageResult:
        """Apply the 3-step strategy for one page."""

        # Step 1 — native text extraction
        native_text = (plumber_page.extract_text() or "").strip()
        native_len  = len(native_text)

        if native_len >= self.min_chars:
            log.debug(f"Page {page_num}: native text ({native_len} chars)")
            return PageResult(
                page_number=page_num,
                text=native_text,
                page_type=PageType.NATIVE_TEXT,
            )

        # Step 2 — OCR
        log.debug(
            f"Page {page_num}: native text too short ({native_len} chars) → OCR"
        )
        ocr_text, confidence = self._ocr_single_page(file_path, page_num)

        # Step 3 — hybrid: OCR gave meaningfully more text
        if native_len > 0 and len(ocr_text) > native_len * 2:
            log.debug(f"Page {page_num}: hybrid (native={native_len}, ocr={len(ocr_text)})")
            return PageResult(
                page_number=page_num,
                text=ocr_text,
                page_type=PageType.HYBRID,
                confidence=confidence,
            )

        # Pure OCR wins (native was essentially empty)
        return PageResult(
            page_number=page_num,
            text=ocr_text,
            page_type=PageType.OCR,
            confidence=confidence,
        )

    def _ocr_single_page(self, file_path: str, page_num: int):
        """
        Rasterise exactly one PDF page and run Tesseract on it.
        Returns (text, mean_confidence).
        """
        with tempfile.TemporaryDirectory() as tmp:
            images = convert_from_path(
                file_path,
                dpi=self.ocr_dpi,
                first_page=page_num,
                last_page=page_num,
                output_folder=tmp,
                fmt="png",
            )
            if not images:
                return "", None

            img = images[0]

            # Get text + confidence in one call
            data = pytesseract.image_to_data(
                img,
                lang=self.ocr_lang,
                output_type=pytesseract.Output.DICT,
            )

            # Confidence is -1 for non-word tokens; filter those out
            conf_vals = [
                int(c) for c in data["conf"]
                if str(c).lstrip("-").isdigit() and int(c) >= 0
            ]
            mean_conf = round(sum(conf_vals) / len(conf_vals), 1) if conf_vals else None

            text = pytesseract.image_to_string(img, lang=self.ocr_lang)
            return text.strip(), mean_conf

    def _extract_metadata(self, file_path: str) -> dict:
        meta: dict = {}
        try:
            reader = PdfReader(file_path)
            info   = reader.metadata or {}
            for key in ("title", "author", "subject", "creator", "producer"):
                val = getattr(info, key, None)
                if val:
                    meta[key] = val
            meta["encrypted"] = reader.is_encrypted
        except Exception:
            pass
        return meta

    @staticmethod
    def _resolve_status(pages: list, errors: list) -> IngestionStatus:
        if not pages:
            return IngestionStatus.FAILED
        if errors and len(errors) < len(pages):
            return IngestionStatus.PARTIAL
        if errors:
            return IngestionStatus.FAILED
        return IngestionStatus.SUCCESS
