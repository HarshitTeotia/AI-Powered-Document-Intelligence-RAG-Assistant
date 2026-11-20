"""Extract native PDF text and flag pages that may need OCR."""

import stat
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

# A scanned page can still contain a small text layer such as a page number.
MIN_TEXT_CHARS = 20
X_TOLERANCE = 1.5


@dataclass(frozen=True)
class PdfPage:
    """A page's original 1-based number, extracted text, and OCR flag."""

    page_number: int
    text: str
    needs_ocr: bool


class PdfExtractionError(Exception):
    """A PDF could not be opened, read, or fully extracted."""


def extract_pdf_pages(pdf_path: Path) -> list[PdfPage]:
    """Return native text for every page, in order, with OCR flags.

    Missing paths and directories raise their corresponding filesystem errors.
    Unsupported extensions and non-regular files raise ValueError. Unreadable,
    empty, invalid, or password-protected PDFs raise PdfExtractionError.
    """
    try:
        file_info = pdf_path.stat()
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise PdfExtractionError("Could not access the PDF file.") from exc

    if stat.S_ISDIR(file_info.st_mode):
        raise IsADirectoryError("Expected a PDF file, but the path is a directory.")
    if not stat.S_ISREG(file_info.st_mode):
        raise ValueError("The PDF path must point to a regular file.")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("The file must have a .pdf extension.")
    if file_info.st_size == 0:
        raise PdfExtractionError("The PDF file is empty.")

    extracted_pages: list[PdfPage] = []
    try:
        # Own the stream so it closes even if the parser fails during opening.
        with pdf_path.open("rb") as source, pdfplumber.open(source) as pdf:
            for page in pdf.pages:
                page_number = page.page_number
                try:
                    text = (page.extract_text(x_tolerance=X_TOLERANCE) or "").strip()
                except Exception as exc:
                    raise PdfExtractionError(
                        f"Could not extract text from PDF page {page_number}."
                    ) from exc
                extracted_pages.append(
                    PdfPage(page_number, text, needs_ocr=len(text) < MIN_TEXT_CHARS)
                )
    except PdfExtractionError:
        raise
    except Exception as exc:
        raise PdfExtractionError(
            "Could not open or read the PDF. It may be unreadable, invalid, "
            "or password-protected."
        ) from exc

    if not extracted_pages:
        raise PdfExtractionError("The PDF contains no readable pages.")
    return extracted_pages
