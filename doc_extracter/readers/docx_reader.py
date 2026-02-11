"""
DOCX / DOC Reader
─────────────────
• .docx  → python-docx (preserves paragraph/table structure)
           Pandoc as fallback (handles corrupt/unusual .docx)
• .doc   → LibreOffice converts to .docx → same pipeline above

Each section (heading-to-heading) is returned as its own "page" so that
the PageResult abstraction stays consistent with PDF output.
If the file has no headings, the whole document is one "page".
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

from docx import Document as DocxDocument


from ..core.base_reader import BaseReader, registry
from ..core.models import DocumentResult, IngestionStatus, PageResult, PageType

log = logging.getLogger(__name__)

# Heading style names recognised by python-docx
HEADING_STYLES = {
    "heading 1", "heading 2", "heading 3", "heading 4",
    "title", "subtitle",
}


@registry.register
class DocxReader(BaseReader):
    SUPPORTED_EXTENSIONS = ["docx", "doc"]

    def read(self, file_path: str, **kwargs) -> DocumentResult:
        path   = Path(file_path)
        errors: List[str] = []

        # .doc → convert to .docx first
        actual_path = str(path)
        tmp_dir     = None
        if path.suffix.lower() == ".doc":
            actual_path, tmp_dir, err = self._convert_doc_to_docx(str(path))
            if err:
                errors.append(err)
            if actual_path is None:
                return DocumentResult(
                    file_path=str(path),
                    file_type="doc",
                    total_pages=0,
                    pages=[],
                    status=IngestionStatus.FAILED,
                    errors=errors,
                )

        try:
            pages, meta, parse_errors = self._parse_docx(actual_path)
            errors.extend(parse_errors)
        except Exception as exc:
            errors.append(f"python-docx failed: {exc}. Trying pandoc…")
            pages, meta, parse_errors = self._parse_with_pandoc(actual_path)
            errors.extend(parse_errors)

        # Clean up temp dir from .doc conversion
        if tmp_dir:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

        status = (
            IngestionStatus.FAILED  if not pages
            else IngestionStatus.PARTIAL if errors
            else IngestionStatus.SUCCESS
        )

        return DocumentResult(
            file_path=str(path),
            file_type=path.suffix.lstrip(".").lower(),
            total_pages=len(pages),
            pages=pages,
            status=status,
            metadata=meta,
            errors=errors,
        )

    # ── DOCX parsing ──────────────────────────────────────────────────────

    def _parse_docx(self, file_path: str) -> Tuple[List[PageResult], dict, List[str]]:
        """
        Split document into sections by top-level headings.
        Each section becomes a PageResult (page = logical section).
        """
        doc    = DocxDocument(file_path)
        errors: List[str] = []

        # Extract core properties as metadata
        meta = self._extract_docx_metadata(doc)

        # Walk paragraphs + tables, grouping by heading boundaries
        sections = self._split_into_sections(doc)

        pages: List[PageResult] = []
        for i, (heading, body_text) in enumerate(sections, start=1):
            full_text = (f"{heading}\n\n{body_text}".strip()
                         if heading else body_text.strip())
            pages.append(PageResult(
                page_number=i,
                text=full_text,
                page_type=PageType.NATIVE_TEXT,
            ))

        return pages, meta, errors

    def _split_into_sections(self, doc: DocxDocument) -> List[Tuple[str, str]]:
        """
        Walk all block elements (paragraphs and tables) and split into
        logical sections at heading boundaries.
        Returns list of (heading_text, body_text) tuples.
        """
        # python-docx doesn't expose tables in paragraph order by default;
        # we access the raw XML child elements to preserve ordering.
        from docx.oxml.ns import qn
        from docx.text.paragraph import Paragraph as DocxPara
        from docx.table import Table as DocxTable

        body = doc.element.body
        sections: List[Tuple[str, str]] = []

        current_heading = ""
        current_body:   List[str] = []

        def flush():
            nonlocal current_heading, current_body
            if current_heading or current_body:
                sections.append((current_heading, "\n".join(current_body)))
            current_heading = ""
            current_body    = []

        for child in body.iterchildren():
            tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag

            if tag == "p":
                para = DocxPara(child, doc)
                style_name = (para.style.name or "").lower() if para.style else ""
                text = para.text.strip()

                if style_name in HEADING_STYLES and text:
                    flush()
                    current_heading = text
                else:
                    if text:
                        current_body.append(text)

            elif tag == "tbl":
                table = DocxTable(child, doc)
                table_text = self._table_to_text(table)
                if table_text:
                    current_body.append(table_text)

        flush()

        # Edge case: document had no sections at all
        if not sections:
            full = "\n".join(
                p.text for p in doc.paragraphs if p.text.strip()
            )
            sections = [("", full)]

        return sections

    def _table_to_text(self, table) -> str:
        """Convert a docx table to a plain-text grid."""
        rows = []
        for row in table.rows:
            cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
            rows.append(" | ".join(cells))
        return "\n".join(rows) if rows else ""

    def _extract_docx_metadata(self, doc: DocxDocument) -> dict:
        meta: dict = {}
        try:
            cp = doc.core_properties
            for attr in ("author", "title", "subject", "description",
                         "created", "modified", "keywords"):
                val = getattr(cp, attr, None)
                if val:
                    meta[attr] = str(val)
        except Exception:
            pass
        return meta

    # ── Pandoc fallback ───────────────────────────────────────────────────

    def _parse_with_pandoc(self, file_path: str) -> Tuple[List[PageResult], dict, List[str]]:
        errors: List[str] = []
        try:
            result = subprocess.run(
                ["pandoc", file_path, "-t", "plain", "--wrap=none"],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode != 0:
                errors.append(f"Pandoc stderr: {result.stderr[:500]}")
            text = result.stdout.strip()
            page = PageResult(
                page_number=1,
                text=text,
                page_type=PageType.NATIVE_TEXT,
            )
            return [page], {}, errors
        except Exception as exc:
            errors.append(f"Pandoc also failed: {exc}")
            return [], {}, errors

    # ── .doc → .docx conversion ───────────────────────────────────────────

    def _convert_doc_to_docx(self, file_path: str):
        """Use LibreOffice (soffice) to convert .doc → .docx."""
        tmp_dir = tempfile.mkdtemp()
        try:
            result = subprocess.run(
                [
                    "soffice", "--headless",
                    "--convert-to", "docx",
                    "--outdir", tmp_dir,
                    file_path,
                ],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode != 0:
                return None, tmp_dir, f"LibreOffice conversion failed: {result.stderr[:300]}"

            # Find the produced file
            converted = list(Path(tmp_dir).glob("*.docx"))
            if not converted:
                return None, tmp_dir, "LibreOffice produced no .docx output"

            return str(converted[0]), tmp_dir, None

        except FileNotFoundError:
            return None, tmp_dir, (
                "soffice not found. Install LibreOffice to read .doc files. "
                "(.docx files work without it.)"
            )
        except Exception as exc:
            return None, tmp_dir, f"Conversion error: {exc}"
