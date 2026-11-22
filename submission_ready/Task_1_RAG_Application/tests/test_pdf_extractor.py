"""Exercise native extraction with synthetic PDFs and controlled failures."""

from pathlib import Path
from unittest.mock import ANY, Mock, PropertyMock, patch

import pdfplumber
import pytest
from pdfplumber.page import Page
from pdfplumber.table import Table

from distribuai.ingestion.pdf_extractor import (
    MIN_TEXT_CHARS,
    X_TOLERANCE,
    PdfExtractionError,
    PdfPage,
    extract_pdf_pages,
)


def _write_pdf(
    path: Path, *pages: str, font_encoding: bytes = b"", font_size: int = 12
) -> Path:
    """Write a minimal PDF with one stream of ASCII text per page."""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"",  # The page tree is filled in after assigning page object numbers.
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica " + font_encoding + b">>",
    ]
    page_refs: list[str] = []
    for text in pages:
        page_id = len(objects) + 1
        page_refs.append(f"{page_id} 0 R")
        page_object = (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            "/Resources << /Font << /F1 3 0 R >> >> "
            f"/Contents {page_id + 1} 0 R >>"
        )
        objects.append(page_object.encode("ascii"))
        commands = [f"BT /F1 {font_size} Tf 18 TL 72 720 Td"]
        for line in text.splitlines():
            literal = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            commands.append(f"({literal}) Tj T*")
        commands.append("ET")
        stream = "\n".join(commands).encode("ascii")
        stream_header = f"<< /Length {len(stream)} >>\nstream\n".encode("ascii")
        objects.append(stream_header + stream + b"\nendstream")

    objects[1] = (
        f"<< /Type /Pages /Kids [{' '.join(page_refs)}] /Count {len(page_refs)} >>"
    ).encode("ascii")
    pdf = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for object_id, content in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{object_id} 0 obj\n".encode("ascii") + content + b"\nendobj\n")

    xref_offset = len(pdf)
    object_count = len(objects) + 1
    pdf.extend(f"xref\n0 {object_count}\n0000000000 65535 f \n".encode("ascii"))
    for offset in offsets:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    trailer = (
        f"trailer\n<< /Size {object_count} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    )
    pdf.extend(trailer.encode("ascii"))
    path.write_bytes(pdf)
    return path


def test_extracts_native_text_and_preserves_line_breaks(tmp_path: Path) -> None:
    """Read an uppercase-extension PDF and return trimmed multiline text."""
    # Map the first and last glyphs to strings containing spaces so pdfplumber
    # returns the outer whitespace instead of dropping it during word extraction.
    source = _write_pdf(
        tmp_path / "policy.PDF",
        "Sample policy text\nLast line ends in Z",
        font_encoding=b"/Encoding << /Differences [83 /space_S 90 /Z_space] >> ",
    )
    with pdfplumber.open(source) as pdf:
        assert pdf.pages[0].extract_text() == " Sample policy text\nLast line ends in Z "

    assert extract_pdf_pages(source) == [
        PdfPage(1, "Sample policy text\nLast line ends in Z", False)
    ]


def test_preserves_all_pages_and_flags_sparse_text(tmp_path: Path) -> None:
    """Keep blank and sparse pages in sequence and flag them for future OCR."""
    source = _write_pdf(
        tmp_path / "multipage.pdf", "First page has enough text.", "", "3", "   "
    )

    assert extract_pdf_pages(source) == [
        PdfPage(1, "First page has enough text.", False),
        PdfPage(2, "", True),
        PdfPage(3, "3", True),
        PdfPage(4, "", True),
    ]


def test_passes_x_tolerance_to_page_extraction(tmp_path: Path) -> None:
    """Use DistribuAI's horizontal gap threshold for native PDF text."""
    source = _write_pdf(tmp_path / "tolerance.pdf", "Sample policy text")

    with patch.object(
        Page, "extract_text", autospec=True, return_value="Sample policy text"
    ) as extract:
        extract_pdf_pages(source)

    extract.assert_called_once_with(ANY, x_tolerance=X_TOLERANCE)


def test_formats_table_rows_once_in_page_order(tmp_path: Path) -> None:
    """Replace table text at its original location and join wrapped cells."""
    source = _write_pdf(
        tmp_path / "table.pdf", "Before\nProduct Pack size\nPremium\nRice 10 kg\nAtta 5 kg\nAfter"
    )
    with pdfplumber.open(source) as pdf:
        lines = pdf.pages[0].extract_text_lines()
    table = Mock(spec=Table)
    table.bbox = (0, lines[1]["top"], 612, lines[4]["bottom"])
    table.extract.return_value = [
        ["Product", "Pack\nsize"], ["Premium\nRice", "10 kg"], ["Atta", "5 kg"],
    ]

    with patch.object(Page, "find_tables", autospec=True, return_value=[table]):
        pages = extract_pdf_pages(source)

    assert pages == [PdfPage(
        1, "Before\nProduct: Premium Rice | Pack size: 10 kg\nProduct: Atta | Pack size: 5 kg\nAfter",
        False,
    )]
    table.extract.assert_called_once_with(x_tolerance=X_TOLERANCE)


@pytest.mark.parametrize("detection_fails", [False, True])
def test_plain_text_when_no_tables_or_detection_fails(
    tmp_path: Path, detection_fails: bool
) -> None:
    """Use normal extraction for pages without a usable table result."""
    source = _write_pdf(tmp_path / "plain.pdf", "Normal page text\nSecond line.")
    cause = RuntimeError("Synthetic detection failure") if detection_fails else None

    with patch.object(Page, "find_tables", autospec=True, return_value=[], side_effect=cause):
        pages = extract_pdf_pages(source)

    assert pages == [PdfPage(1, "Normal page text\nSecond line.", False)]


def test_tagged_table_reuses_headers_on_continuation_page(tmp_path: Path) -> None:
    """Use semantic cells for a borderless table with page-local identifiers."""
    source = _write_pdf(
        tmp_path / "tagged.pdf", "Before\nProduct\nSize\nBasmati\nRice\n10 kg", "Atta\n5 kg\nAfter"
    )
    original_chars = Page.chars.fget
    assert original_chars is not None

    def tagged_chars(page: Page) -> list[dict[str, object]]:
        chars = original_chars(page)
        if type(page) is not Page:
            return chars
        markers = ([None, 1, 2, 3, 3, 4], [1, 2, None])[page.page_number - 1]
        by_top = dict(zip(sorted({char["top"] for char in chars}), markers, strict=True))
        for char in chars:
            char["mcid"] = by_top[char["top"]]
        return chars

    def row(kind: str, page: int, *mcids: int) -> dict[str, object]:
        return {"type": "TR", "children": [
            {"type": kind, "children": [{"type": "P", "page_number": page, "mcids": [mcid]}]}
            for mcid in mcids
        ]}

    structure = [{"type": "Table", "children": [
        row("TH", 1, 1, 2), row("TD", 1, 3, 4), row("TD", 2, 1, 2),
    ]}]
    with (
        patch.object(pdfplumber.PDF, "structure_tree", new_callable=PropertyMock, return_value=structure),
        patch.object(Page, "chars", new=property(tagged_chars)),
    ):
        pages = extract_pdf_pages(source)

    assert pages == [
        PdfPage(1, "Before\nProduct: Basmati Rice | Size: 10 kg", False),
        PdfPage(2, "Product: Atta | Size: 5 kg\nAfter", False),
    ]


def test_flags_only_text_strictly_below_threshold(tmp_path: Path) -> None:
    """A page with exactly the minimum number of characters is sufficient."""
    assert MIN_TEXT_CHARS == 20
    below = "x" * (MIN_TEXT_CHARS - 1)
    at_minimum = "x" * MIN_TEXT_CHARS
    source = _write_pdf(tmp_path / "threshold.pdf", below, at_minimum)

    assert extract_pdf_pages(source) == [PdfPage(1, below, True), PdfPage(2, at_minimum, False)]


def test_rejects_missing_file(tmp_path: Path) -> None:
    """Report a missing input before trying to parse it."""
    with pytest.raises(FileNotFoundError):
        extract_pdf_pages(tmp_path / "missing.pdf")


def test_rejects_directory(tmp_path: Path) -> None:
    """Reject a directory even if its name has the right extension."""
    directory = tmp_path / "directory.pdf"
    directory.mkdir()

    with pytest.raises(IsADirectoryError, match="directory"):
        extract_pdf_pages(directory)


def test_rejects_wrong_extension(tmp_path: Path) -> None:
    """Reject valid PDF bytes when the file lacks a PDF extension."""
    source = _write_pdf(tmp_path / "misnamed.txt", "Sample text")

    with pytest.raises(ValueError, match=r"\.pdf extension"):
        extract_pdf_pages(source)


def test_rejects_empty_file(tmp_path: Path) -> None:
    """Give an explicit error for a zero-byte PDF."""
    source = tmp_path / "empty.pdf"
    source.touch()

    with pytest.raises(PdfExtractionError, match="empty"):
        extract_pdf_pages(source)


def test_rejects_invalid_pdf(tmp_path: Path) -> None:
    """Surface a parser error with the original cause and a safe message."""
    content = "invalid synthetic content"
    source = tmp_path / "invalid.pdf"
    source.write_text(content, encoding="ascii")

    with pytest.raises(PdfExtractionError, match="invalid") as error:
        extract_pdf_pages(source)

    assert error.value.__cause__ is not None
    assert content not in str(error.value)


def test_reports_unreadable_pdf(tmp_path: Path) -> None:
    """Wrap file opening errors without discarding their cause."""
    source = _write_pdf(tmp_path / "unreadable.pdf", "Sample text")
    cause = PermissionError("synthetic permission denial")

    with patch.object(Path, "open", side_effect=cause):
        with pytest.raises(PdfExtractionError, match="open or read") as error:
            extract_pdf_pages(source)

    assert error.value.__cause__ is cause


def test_page_failure_reports_page_and_closes_stream(tmp_path: Path) -> None:
    """A failure after an earlier page aborts extraction and closes the stream."""
    source = _write_pdf(tmp_path / "page-error.pdf", "First page text", "Second page text")
    cause = RuntimeError("synthetic document content")

    with patch.object(
        Page, "extract_text", autospec=True, side_effect=["First page text", cause]
    ) as extract:
        with pytest.raises(PdfExtractionError, match="page 2") as error:
            extract_pdf_pages(source)

    assert error.value.__cause__ is cause
    assert str(cause) not in str(error.value)
    assert extract.call_args.args[0].pdf.stream.closed
