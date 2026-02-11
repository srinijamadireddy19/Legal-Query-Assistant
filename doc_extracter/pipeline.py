"""
Document Ingestion Pipeline
───────────────────────────
The single public entry point for the whole system.

Usage
─────
    from doc_pipeline import DocumentIngestionPipeline

    pipeline = DocumentIngestionPipeline()
    result   = pipeline.ingest("contract.pdf")
    print(result.plain_text)

    # Or batch
    results  = pipeline.ingest_batch(["a.pdf", "b.docx", "c.png"])
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

# Auto-register all readers by importing them
from .readers import pdf_reader, docx_reader, image_reader, text_reader  # noqa: F401
from .core.base_reader import registry
from .core.models import DocumentResult, IngestionStatus

log = logging.getLogger(__name__)


class DocumentIngestionPipeline:
    """
    Detects file format → picks the right reader → returns DocumentResult.

    Supported formats (out of the box):
        PDF  : .pdf  (native text + scanned/image pages + mixed)
        Word : .docx, .doc
        Image: .jpg, .jpeg, .png, .tiff, .tif, .bmp, .webp
        Text : .txt, .md, .rst

    Parameters
    ──────────
    ocr_dpi   : DPI for PDF→image rasterisation (default 300)
    ocr_lang  : Tesseract language code, e.g. "eng", "eng+fra" (default "eng")
    preprocess: Apply image pre-processing before OCR (default True)
    """

    def __init__(
        self,
        ocr_dpi:    int  = 300,
        ocr_lang:   str  = "eng",
        preprocess: bool = True,
    ):
        self.reader_kwargs: Dict[str, Any] = {
            "ocr_dpi":    ocr_dpi,
            "ocr_lang":   ocr_lang,
            "preprocess": preprocess,
        }

    # ── single file ───────────────────────────────────────────────────────

    def ingest(self, file_path: str, **extra_kwargs) -> DocumentResult:
        """
        Ingest one file.  Returns DocumentResult regardless of success/failure.
        Never raises.
        """
        path = Path(file_path)

        if not path.exists():
            return self._failure(str(path), f"File not found: {file_path}")
        if not path.is_file():
            return self._failure(str(path), f"Not a file: {file_path}")

        ext = path.suffix.lstrip(".").lower()

        try:
            reader_cls = registry.get(ext)
        except ValueError as exc:
            return self._failure(str(path), str(exc))

        # Build reader with only the kwargs it understands
        reader = self._build_reader(reader_cls)

        log.info(f"Ingesting '{path.name}' with {reader_cls.__name__}")
        try:
            result = reader.read(str(path), **extra_kwargs)
        except Exception as exc:
            log.exception(f"Unexpected error reading {file_path}")
            return self._failure(str(path), f"Unhandled reader error: {exc}")

        return result

    # ── batch ─────────────────────────────────────────────────────────────

    def ingest_batch(
        self,
        file_paths: List[str],
        skip_failures: bool = True,
    ) -> List[DocumentResult]:
        """
        Ingest a list of files.

        Parameters
        ──────────
        file_paths    : list of path strings
        skip_failures : if False, stop on first failure
        """
        results: List[DocumentResult] = []

        for fp in file_paths:
            result = self.ingest(fp)
            results.append(result)

            if result.status == IngestionStatus.FAILED:
                log.warning(f"Failed to ingest: {fp}")
                if not skip_failures:
                    break
            else:
                log.info(
                    f"OK  {fp}  "
                    f"[{result.total_pages}p, "
                    f"{result.native_page_count} native, "
                    f"{result.ocr_page_count} OCR]"
                )

        return results

    # ── utility ───────────────────────────────────────────────────────────

    @property
    def supported_formats(self) -> List[str]:
        """List of supported file extensions."""
        return registry.supported_extensions

    def _build_reader(self, reader_cls):
        """
        Construct reader, passing only kwargs the __init__ accepts.
        (TextReader doesn't have ocr_dpi, so we can't blindly pass all.)
        """
        import inspect
        sig    = inspect.signature(reader_cls.__init__)
        params = set(sig.parameters.keys()) - {"self"}
        kwargs = {k: v for k, v in self.reader_kwargs.items() if k in params}
        return reader_cls(**kwargs)

    @staticmethod
    def _failure(file_path: str, reason: str) -> DocumentResult:
        return DocumentResult(
            file_path=file_path,
            file_type=Path(file_path).suffix.lstrip(".").lower(),
            total_pages=0,
            pages=[],
            status=IngestionStatus.FAILED,
            errors=[reason],
        )
