# Document Ingestion Pipeline

A modular, zero-compromise pipeline for reading **any legal document** into clean text —
regardless of whether it's a native PDF, a fully-scanned PDF, a Word doc, a raw image, or plain text.

## What it handles

| Format | How it works |
|---|---|
| `.pdf` (native text) | pdfplumber text extraction — fast, lossless |
| `.pdf` (scanned / image) | pdf2image → Tesseract OCR per page |
| `.pdf` (mixed) | Per-page decision: native if ≥30 chars, else OCR |
| `.docx` | python-docx (paragraphs + tables, split by heading) |
| `.doc` | LibreOffice converts → `.docx` → same pipeline |
| `.jpg` `.png` `.tiff` `.bmp` `.webp` | Pillow pre-process → Tesseract OCR |
| `.txt` `.md` `.rst` | Encoding-aware text read, split into pages |

---

## Architecture

```
doc_pipeline/
├── __init__.py
├── pipeline.py           ← DocumentIngestionPipeline  (entry point)
├── core/
│   ├── models.py         ← DocumentResult, PageResult, PageType, IngestionStatus
│   └── base_reader.py    ← BaseReader ABC + ReaderRegistry
└── readers/
    ├── pdf_reader.py     ← Native / Scanned / Hybrid PDF
    ├── docx_reader.py    ← DOCX + DOC (via LibreOffice)
    ├── image_reader.py   ← JPG / PNG / TIFF / BMP / WEBP
    └── text_reader.py    ← TXT / MD / RST
```

### Data flow

```
file.pdf  ──►  PDFReader
file.docx ──►  DocxReader    ─►  DocumentResult
file.png  ──►  ImageReader       ├── pages: List[PageResult]
file.txt  ──►  TextReader        │     ├── page_number
                                 │     ├── text
                                 │     ├── page_type  (NATIVE / OCR / HYBRID)
                                 │     └── confidence (OCR only)
                                 ├── plain_text  (property)
                                 ├── full_text   (with [PAGE N] markers)
                                 └── summary()
```

### PDF page decision logic

```
for each PDF page:
    native_text = pdfplumber.extract_text()

    if len(native_text) >= 30 chars:
        → NATIVE_TEXT  (use as-is)
    else:
        rasterise page at 300 DPI  (pdf2image / poppler)
        ocr_text, confidence = Tesseract(image)

        if native had some text AND ocr gave >2× more chars:
            → HYBRID  (use OCR text, flag the page)
        else:
            → OCR
```

---

## Quick start

```python
from doc_pipeline import DocumentIngestionPipeline

pipeline = DocumentIngestionPipeline()

# Single file — works for any supported format
result = pipeline.ingest("contract.pdf")

print(result.summary())
# File       : contract.pdf
# Type       : pdf
# Status     : success
# Pages      : 4
# Native     : 2
# OCR        : 2
# OCR conf.  : 82.4%

# Text ready for your chunker
print(result.plain_text)

# Inspect per-page detail
for page in result.pages:
    print(page.page_number, page.page_type.value, page.word_count)
```

### Batch ingestion

```python
results = pipeline.ingest_batch([
    "agreements/contract.pdf",
    "reports/q3.docx",
    "scans/invoice_jan.png",
])

for r in results:
    print(r.file_path, r.status.value, r.ocr_page_count, "OCR pages")
```

### Custom OCR settings

```python
pipeline = DocumentIngestionPipeline(
    ocr_dpi    = 400,          # Higher DPI → better quality, slower
    ocr_lang   = "eng+fra",    # Multi-language (needs lang packs installed)
    preprocess = True,         # Grayscale + sharpen + contrast before OCR
)
```

---

## Output object reference

### `DocumentResult`

| Attribute | Type | Description |
|---|---|---|
| `file_path` | `str` | Absolute path |
| `file_type` | `str` | `"pdf"`, `"docx"`, `"png"`, … |
| `total_pages` | `int` | Number of PageResult objects |
| `pages` | `List[PageResult]` | Per-page data |
| `status` | `IngestionStatus` | `SUCCESS / PARTIAL / FAILED` |
| `metadata` | `dict` | PDF metadata, DOCX core props, image size … |
| `errors` | `List[str]` | Non-fatal warnings + fatal errors |
| `.plain_text` | property | All pages joined — feed directly to chunker |
| `.full_text` | property | Like plain_text but with `[PAGE N]` markers |
| `.ocr_page_count` | property | How many pages went through OCR |
| `.avg_ocr_confidence` | property | Mean Tesseract confidence, `None` if no OCR |
| `.summary()` | method | One-line-per-field diagnostic string |

### `PageResult`

| Attribute | Type | Description |
|---|---|---|
| `page_number` | `int` | 1-indexed |
| `text` | `str` | Extracted text |
| `page_type` | `PageType` | `NATIVE_TEXT / OCR / HYBRID` |
| `char_count` | `int` | Auto-computed |
| `word_count` | `int` | Auto-computed |
| `confidence` | `float \| None` | Tesseract mean confidence 0–100 |
| `error` | `str \| None` | Per-page error, if any |

---

## Adding a new format

1. Create `readers/my_format_reader.py`
2. Inherit `BaseReader`, set `SUPPORTED_EXTENSIONS = ["xyz"]`
3. Implement `read(file_path, **kwargs) → DocumentResult`
4. Import in `pipeline.py` next to the other reader imports

That's it — the registry auto-discovers it.

```python
from ..core.base_reader import BaseReader, registry
from ..core.models import DocumentResult, IngestionStatus, PageResult, PageType

@registry.register
class MyFormatReader(BaseReader):
    SUPPORTED_EXTENSIONS = ["xyz"]

    def read(self, file_path: str, **kwargs) -> DocumentResult:
        # ... parse the file ...
        page = PageResult(page_number=1, text="...", page_type=PageType.NATIVE_TEXT)
        return DocumentResult(
            file_path=file_path, file_type="xyz",
            total_pages=1, pages=[page],
            status=IngestionStatus.SUCCESS,
        )
```

---

## Dependencies

```
pdfplumber      # PDF text extraction
pypdf           # PDF metadata
pdf2image       # PDF → images for OCR  (needs poppler)
pytesseract     # Python wrapper for Tesseract
Pillow          # Image pre-processing
python-docx     # DOCX reading
reportlab       # (test file creation only)

# System packages (already installed):
tesseract-ocr   # OCR engine
poppler-utils   # pdftotext, pdftoppm
pandoc          # DOCX fallback
```

---

## Integration with the legal RAG chunker

```python
from doc_pipeline import DocumentIngestionPipeline
from legal_rag_chunker import HierarchicalChunkingPipeline

ingest  = DocumentIngestionPipeline()
chunker = HierarchicalChunkingPipeline()

# Any format → clean text → smart chunks
result = ingest.ingest("lease_agreement.pdf")
chunks = chunker.process(result.plain_text)

for chunk in chunks.chunks:
    print(chunk.hierarchy, "→", chunk.content[:80])
```
