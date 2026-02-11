"""
Image Reader
────────────
Handles standalone image files (JPG, PNG, TIFF, BMP, WEBP).
Each image is treated as a single-page document and run through
Tesseract OCR with optional pre-processing via Pillow.

Pre-processing pipeline (applied before OCR):
  1. Convert to grayscale
  2. Adaptive binarisation (Otsu threshold via Pillow)
  3. Mild sharpening
  4. Upscale if image is very small (< 1000px wide → scale to 2000px)
"""

import logging
from pathlib import Path
from typing import Tuple

import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

from ..core.base_reader import BaseReader, registry
from ..core.models import DocumentResult, IngestionStatus, PageResult, PageType

log = logging.getLogger(__name__)

OCR_LANG = "eng"
MIN_WIDTH_FOR_UPSCALE = 1000
TARGET_UPSCALE_WIDTH  = 2000

# Force pytesseract to use installed executable (bypass PATH issues)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


@registry.register
class ImageReader(BaseReader):
    SUPPORTED_EXTENSIONS = ["jpg", "jpeg", "png", "tiff", "tif", "bmp", "webp"]

    def __init__(self, ocr_lang: str = OCR_LANG, preprocess: bool = True):
        self.ocr_lang   = ocr_lang
        self.preprocess = preprocess

    def read(self, file_path: str, **kwargs) -> DocumentResult:
        path   = Path(file_path)
        errors: list[str] = []

        try:
            img = Image.open(file_path)
        except Exception as exc:
            return DocumentResult(
                file_path=str(path),
                file_type=path.suffix.lstrip(".").lower(),
                total_pages=0,
                pages=[],
                status=IngestionStatus.FAILED,
                errors=[f"Cannot open image: {exc}"],
            )

        meta = {
            "original_size": img.size,
            "mode": img.mode,
            "format": img.format,
        }

        if self.preprocess:
            img = self._preprocess(img)

        text, confidence = self._run_ocr(img)

        page = PageResult(
            page_number=1,
            text=text,
            page_type=PageType.OCR,
            confidence=confidence,
        )

        status = IngestionStatus.SUCCESS if text.strip() else IngestionStatus.PARTIAL

        return DocumentResult(
            file_path=str(path),
            file_type=path.suffix.lstrip(".").lower(),
            total_pages=1,
            pages=[page],
            status=status,
            metadata=meta,
            errors=errors,
        )

    # ── pre-processing ───────────────────────────────────────────────────

    def _preprocess(self, img: Image.Image) -> Image.Image:
        """Normalise image for best OCR accuracy."""

        # 1. Convert to RGB then grayscale
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        gray = img.convert("L")

        # 2. Upscale small images
        w, h = gray.size
        if w < MIN_WIDTH_FOR_UPSCALE:
            scale = TARGET_UPSCALE_WIDTH / w
            new_size = (int(w * scale), int(h * scale))
            gray = gray.resize(new_size, Image.LANCZOS)
            log.debug(f"Upscaled from {w}x{h} → {new_size[0]}x{new_size[1]}")

        # 3. Sharpen
        gray = gray.filter(ImageFilter.SHARPEN)

        # 4. Increase contrast
        enhancer = ImageEnhance.Contrast(gray)
        gray = enhancer.enhance(1.5)

        return gray

    # ── OCR ──────────────────────────────────────────────────────────────

    def _run_ocr(self, img: Image.Image) -> Tuple[str, float]:
        data = pytesseract.image_to_data(
            img,
            lang=self.ocr_lang,
            output_type=pytesseract.Output.DICT,
        )
        conf_vals = [
            int(c) for c in data["conf"]
            if str(c).lstrip("-").isdigit() and int(c) >= 0
        ]
        mean_conf = round(sum(conf_vals) / len(conf_vals), 1) if conf_vals else None

        text = pytesseract.image_to_string(img, lang=self.ocr_lang)
        return text.strip(), mean_conf
