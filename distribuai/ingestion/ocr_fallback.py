"""Use local English OCR to supplement sparse native PDF text."""

import shutil
from contextlib import closing
from pathlib import Path
from threading import Lock

import pypdfium2
import pytesseract
from pytesseract import pytesseract as tesseract_engine

from distribuai import config
from distribuai.ingestion.pdf_extractor import MIN_TEXT_CHARS, PdfPage, extract_pdf_pages

OCR_DPI = 300
OCR_TIMEOUT_SECONDS = 30
_MISSING_EXECUTABLE = (
    "Tesseract OCR executable was not found. Install Tesseract and add it to PATH, "
    "or set TESSERACT_CMD to its executable path."
)

# PDFium calls and temporary changes to pytesseract's executable are serialized
# within this module. This does not synchronize direct calls from other modules.
_OCR_LOCK = Lock()


class OcrError(Exception):
    """An OCR page could not be rendered or recognized."""


def extract_pdf_with_ocr(pdf_path: Path) -> list[PdfPage]:
    """Extract native text first, then supplement flagged pages with English OCR.

    Return all pages in order. Native extraction errors propagate unchanged.
    OCR failures raise OcrError without returning partial results.
    """
    pages = extract_pdf_pages(pdf_path)
    if not any(page.needs_ocr for page in pages):
        return pages

    command = _resolve_tesseract_cmd()
    results = pages.copy()
    try:
        with _OCR_LOCK, closing(pypdfium2.PdfDocument(pdf_path)) as pdf:
            for index, page in enumerate(pages):
                if not page.needs_ocr:
                    continue
                ocr_text = _ocr_page(pdf, page.page_number, command)
                # Length is a simple heuristic, not a measure of OCR accuracy.
                text = ocr_text if len(ocr_text) > len(page.text) else page.text
                results[index] = PdfPage(page.page_number, text, len(text) < MIN_TEXT_CHARS)
    except OcrError:
        raise
    except Exception as exc:
        raise OcrError("Could not open or read the PDF for OCR.") from exc

    return results


def _resolve_tesseract_cmd() -> str:
    """Check the configured executable or fall back to Tesseract on PATH."""
    configured = config.TESSERACT_CMD
    command = configured or "tesseract"
    try:
        executable = shutil.which(command)
    except OSError as exc:
        raise OcrError(_MISSING_EXECUTABLE) from exc
    if executable is None:
        if configured:
            raise OcrError(f"The configured TESSERACT_CMD is unavailable. {_MISSING_EXECUTABLE}")
        raise OcrError(_MISSING_EXECUTABLE)
    return executable if configured else command


def _ocr_page(pdf: pypdfium2.PdfDocument, page_number: int, command: str) -> str:
    """Render and recognize one 1-based page while the caller owns the document."""
    index = page_number - 1
    try:
        if not 0 <= index < len(pdf):
            raise OcrError(f"PDF page {page_number} is out of range for OCR.")
        with (
            closing(pdf.get_page(index)) as page,
            closing(page.render(scale=OCR_DPI / 72)) as bitmap,
            closing(bitmap.to_pil()) as image,
        ):
            # The converted image can share the bitmap's storage. Close it first.
            return _recognize_text(image, command, page_number)
    except OcrError:
        raise
    except Exception as exc:
        raise OcrError(f"Could not render or convert PDF page {page_number} for OCR.") from exc


def _recognize_text(image: object, command: str, page_number: int) -> str:
    """Run the English OCR subprocess with a deadline and restore its executable."""
    previous_command = tesseract_engine.tesseract_cmd
    if command != previous_command:
        tesseract_engine.tesseract_cmd = command
    try:
        return pytesseract.image_to_string(image, lang="eng", timeout=OCR_TIMEOUT_SECONDS).strip()
    except tesseract_engine.TesseractNotFoundError as exc:
        raise OcrError(_MISSING_EXECUTABLE) from exc
    except Exception as exc:
        if isinstance(exc, tesseract_engine.TesseractError):
            detail = exc.message.lower()
            if "eng.traineddata" in detail or "failed loading language 'eng'" in detail:
                raise OcrError(
                    f"Tesseract could not load English language data for PDF page {page_number}. "
                    "Install or repair eng.traineddata and check TESSDATA_PREFIX if set."
                ) from exc
        if isinstance(exc, RuntimeError) and str(exc) == "Tesseract process timeout":
            raise OcrError(
                f"OCR timed out after {OCR_TIMEOUT_SECONDS} seconds on PDF page {page_number}."
            ) from exc
        raise OcrError(f"Tesseract OCR failed on PDF page {page_number}.") from exc
    finally:
        if command != previous_command:
            tesseract_engine.tesseract_cmd = previous_command
