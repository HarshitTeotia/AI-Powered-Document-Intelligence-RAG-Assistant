"""Verify selective OCR, resource cleanup, and real recognition when available."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, call, patch

import pypdfium2
import pytest
from pytesseract import pytesseract as tesseract_engine
from test_pdf_extractor import _write_pdf

from distribuai.ingestion import ocr_fallback as ocr
from distribuai.ingestion.pdf_extractor import PdfExtractionError, PdfPage, extract_pdf_pages


@pytest.fixture
def tesseract_on_path(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Simulate an available executable without inspecting the test host."""
    monkeypatch.setattr(ocr.config, "TESSERACT_CMD", None)
    which = Mock(return_value="tesseract")
    monkeypatch.setattr(ocr.shutil, "which", which)
    return which


@pytest.fixture
def pdfium_document(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Provide observable document, page, bitmap, and image cleanup calls."""
    pdf = MagicMock()
    pdf.__len__.return_value = 1
    monkeypatch.setattr(ocr.pypdfium2, "PdfDocument", Mock(return_value=pdf))
    return pdf


def test_native_text_skips_pdfium_and_tesseract(tmp_path: Path) -> None:
    source = _write_pdf(tmp_path / "native.pdf", "This page has enough native text.")
    expected = extract_pdf_pages(source)

    with (
        patch.object(ocr.pypdfium2, "PdfDocument") as open_pdf,
        patch.object(ocr.shutil, "which") as which,
        patch.object(ocr.pytesseract, "image_to_string") as recognize,
    ):
        assert ocr.extract_pdf_with_ocr(source) == expected

    open_pdf.assert_not_called()
    which.assert_not_called()
    recognize.assert_not_called()


def test_mixed_document_renders_only_flagged_pages(
    tmp_path: Path, tesseract_on_path: Mock
) -> None:
    source = _write_pdf(
        tmp_path / "mixed.pdf", "", "This page keeps its native text.", "3", "4"
    )
    responses = iter(("  Text found on page one.\nSecond line.  ", "  ", "Short OCR"))

    def recognize(image: Any, *, lang: str, timeout: int) -> str:
        width, height = image.size
        assert abs(width - 2550) <= 1 and abs(height - 3300) <= 1  # US Letter at 300 DPI
        assert lang == "eng"
        assert timeout == ocr.OCR_TIMEOUT_SECONDS
        assert tesseract_engine.tesseract_cmd == "tesseract"
        return next(responses)

    with (
        patch.object(
            pypdfium2.PdfDocument, "get_page", autospec=True,
            side_effect=pypdfium2.PdfDocument.get_page,
        ) as get_page,
        patch.object(ocr.pytesseract, "image_to_string", side_effect=recognize),
    ):
        assert ocr.extract_pdf_with_ocr(source) == [
            PdfPage(1, "Text found on page one.\nSecond line.", False),
            PdfPage(2, "This page keeps its native text.", False),
            PdfPage(3, "3", True),
            PdfPage(4, "Short OCR", True),
        ]

    assert [access.args[1] for access in get_page.call_args_list] == [0, 2, 3]
    tesseract_on_path.assert_called_once_with("tesseract")


@pytest.mark.parametrize("output", ["", "  ", "Short", "Same length"])
def test_empty_shorter_or_tied_ocr_keeps_native_text(
    tmp_path: Path, tesseract_on_path: Mock, pdfium_document: MagicMock, output: str
) -> None:
    source = _write_pdf(tmp_path / "sparse.pdf", "Native text")

    with patch.object(ocr.pytesseract, "image_to_string", return_value=output):
        assert ocr.extract_pdf_with_ocr(source) == [PdfPage(1, "Native text", True)]

    assert pdfium_document.mock_calls[-4:] == [
        call.get_page().render().to_pil().close(),
        call.get_page().render().close(),
        call.get_page().close(),
        call.close(),
    ]


@pytest.mark.parametrize("configured", [False, True])
def test_missing_executable_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tesseract_on_path: Mock, configured: bool
) -> None:
    source = _write_pdf(tmp_path / "scanned.pdf", "")
    missing = str(tmp_path / "Test OCR" / "tesseract.exe") if configured else None
    monkeypatch.setattr(ocr.config, "TESSERACT_CMD", missing)
    tesseract_on_path.return_value = None

    with (
        patch.object(ocr.pypdfium2, "PdfDocument") as open_pdf,
        pytest.raises(ocr.OcrError, match="PATH.*TESSERACT_CMD") as error,
    ):
        ocr.extract_pdf_with_ocr(source)

    assert ("configured TESSERACT_CMD" in str(error.value)) is configured
    tesseract_on_path.assert_called_once_with(missing or "tesseract")
    open_pdf.assert_not_called()


@pytest.mark.parametrize(
    ("cause", "message"),
    [
        (tesseract_engine.TesseractNotFoundError(), "PATH.*TESSERACT_CMD"),
        (
            tesseract_engine.TesseractError(1, "Error opening data file eng.traineddata"),
            "English language data.*page 1",
        ),
        (RuntimeError("Tesseract process timeout"), "timed out.*page 1"),
        (tesseract_engine.TesseractError(1, "Synthetic diagnostic"), "OCR failed.*page 1"),
    ],
)
def test_execution_failures_restore_command_and_close_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tesseract_on_path: Mock,
    pdfium_document: MagicMock, cause: Exception, message: str,
) -> None:
    source = _write_pdf(tmp_path / "scanned.pdf", "")
    configured = str(tmp_path / "Test OCR" / "tesseract.exe")
    monkeypatch.setattr(ocr.config, "TESSERACT_CMD", configured)
    tesseract_on_path.return_value = configured
    previous_command = tesseract_engine.tesseract_cmd

    def fail(image: object, *, lang: str, timeout: int) -> str:
        assert tesseract_engine.tesseract_cmd == configured
        raise cause

    with (
        patch.object(ocr.pytesseract, "image_to_string", side_effect=fail),
        pytest.raises(ocr.OcrError, match=message) as error,
    ):
        ocr.extract_pdf_with_ocr(source)

    assert error.value.__cause__ is cause
    assert str(cause) not in str(error.value)
    assert tesseract_engine.tesseract_cmd == previous_command
    assert pdfium_document.mock_calls[-4:] == [
        call.get_page().render().to_pil().close(),
        call.get_page().render().close(),
        call.get_page().close(),
        call.close(),
    ]


def test_pdfium_open_failure(tmp_path: Path, tesseract_on_path: Mock) -> None:
    source = _write_pdf(tmp_path / "scanned.pdf", "")
    cause = pypdfium2.PdfiumError("Synthetic opening failure")

    with (
        patch.object(ocr.pypdfium2, "PdfDocument", side_effect=cause),
        pytest.raises(ocr.OcrError, match="open or read the PDF") as error,
    ):
        ocr.extract_pdf_with_ocr(source)

    assert error.value.__cause__ is cause


@pytest.mark.parametrize("stage", ["render", "convert"])
def test_image_preparation_failure_closes_created_resources(
    tmp_path: Path, tesseract_on_path: Mock, pdfium_document: MagicMock, stage: str
) -> None:
    source = _write_pdf(tmp_path / "scanned.pdf", "")
    cause = RuntimeError("Synthetic preparation failure")
    page = pdfium_document.get_page.return_value
    bitmap = page.render.return_value
    (page.render if stage == "render" else bitmap.to_pil).side_effect = cause

    with (
        patch.object(ocr.pytesseract, "image_to_string") as recognize,
        pytest.raises(ocr.OcrError, match="render or convert PDF page 1") as error,
    ):
        ocr.extract_pdf_with_ocr(source)

    assert error.value.__cause__ is cause
    recognize.assert_not_called()
    page.close.assert_called_once_with()
    pdfium_document.close.assert_called_once_with()
    if stage == "convert":
        bitmap.close.assert_called_once_with()


def test_invalid_page_index_aborts_ocr(
    tmp_path: Path, tesseract_on_path: Mock, pdfium_document: MagicMock
) -> None:
    source = _write_pdf(tmp_path / "scanned.pdf", "Page one has enough text.", "")

    with pytest.raises(ocr.OcrError, match="page 2 is out of range"):
        ocr.extract_pdf_with_ocr(source)

    pdfium_document.get_page.assert_not_called()
    pdfium_document.close.assert_called_once_with()


def test_native_extraction_error_propagates(tmp_path: Path) -> None:
    source = tmp_path / "invalid.pdf"
    source.write_bytes(b"not a PDF")

    with (
        patch.object(ocr.shutil, "which") as which,
        pytest.raises(PdfExtractionError, match="invalid"),
    ):
        ocr.extract_pdf_with_ocr(source)

    which.assert_not_called()


def test_real_tesseract_smoke(tmp_path: Path) -> None:
    if ocr.shutil.which(ocr.config.TESSERACT_CMD or "tesseract") is None:
        pytest.skip("Tesseract executable is unavailable via TESSERACT_CMD or PATH")
    command = ocr._resolve_tesseract_cmd()
    source = _write_pdf(tmp_path / "smoke.pdf", "First page", "OCR SMOKE TEST", font_size=36)
    native_page = extract_pdf_pages(source)[1]
    assert native_page.page_number == 2 and native_page.needs_ocr

    with pypdfium2.PdfDocument(source) as pdf:
        recognized = ocr._ocr_page(pdf, native_page.page_number, command)

    assert "OCR SMOKE TEST" in " ".join(recognized.upper().split())
