"""Extract native PDF text and flag pages that may need OCR."""

import stat
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pdfplumber
from pdfplumber import utils
from pdfplumber.page import Page

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


@dataclass(frozen=True)
class _TableBlock:
    """A formatted table and the location of its original page text."""

    bbox: tuple[float, float, float, float]
    text: str
    mcids: frozenset[int] = frozenset()


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
            tagged_tables = _tagged_table_blocks(pdf)
            for page in pdf.pages:
                page_number = page.page_number
                try:
                    text = _extract_page_text(page, tagged_tables.get(page_number, ()))
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


def _extract_page_text(page: Page, tagged: Sequence[_TableBlock]) -> str:
    """Replace table glyphs with rows and preserve nearby text in page order."""
    try:
        tables = list(tagged)
        grid_boxes = []
        table_mcids = {mcid for block in tagged for mcid in block.mcids}
        for table in page.find_tables():
            if any(utils.get_bbox_overlap(table.bbox, block.bbox) for block in tagged):
                continue
            rows = table.extract(x_tolerance=X_TOLERANCE)
            if rows and (text := _format_rows(rows[0], rows[1:])):
                tables.append(_TableBlock(table.bbox, text))
                grid_boxes.append(table.bbox)

        if tables:
            def outside_tables(obj: dict[str, Any]) -> bool:
                """Exclude tagged glyphs and glyphs whose centers lie in grid tables."""
                if obj["object_type"] != "char":
                    return True
                x = (obj["x0"] + obj["x1"]) / 2
                y = (obj["top"] + obj["bottom"]) / 2
                return obj.get("mcid") not in table_mcids and not any(
                    left <= x < right and top <= y < bottom
                    for left, top, right, bottom in grid_boxes
                )

            lines = page.filter(outside_tables).extract_text_lines(
                return_chars=False, x_tolerance=X_TOLERANCE
            )
            pieces = [(line["top"], line["x0"], line["text"]) for line in lines]
            pieces.extend((block.bbox[1], block.bbox[0], block.text) for block in tables)
            return "\n".join(text for _, _, text in sorted(pieces)).strip()
    except Exception:
        # A malformed table must not prevent the page's plain text being read.
        pass
    return (page.extract_text(x_tolerance=X_TOLERANCE) or "").strip()


def _format_rows(
    headers: Sequence[str | None], rows: Sequence[Sequence[str | None]]
) -> str:
    """Join wrapped cells and label each value with its column header."""
    labels = [" ".join((header or "").split()) for header in headers]
    lines = []
    for row in rows:
        values = [" ".join((cell or "").split()) for cell in row]
        if any(values):
            lines.append(" | ".join(
                f"{(labels[i] if i < len(labels) else '') or f'Column {i + 1}'}: "
                f"{values[i] if i < len(values) else ''}"
                for i in range(max(len(labels), len(values)))
            ))
    return "\n".join(lines) or " | ".join(label for label in labels if label)


def _structure_nodes(nodes: list[dict[str, Any]], kind: str) -> Iterator[dict[str, Any]]:
    """Find structure elements of one kind, including within grouping tags."""
    for node in nodes:
        if node.get("type") == kind:
            yield node
        else:
            yield from _structure_nodes(node.get("children", []), kind)


def _cell_mcids(
    node: dict[str, Any], page_number: int | None = None
) -> Iterator[tuple[int, int]]:
    """Yield a cell's page-local marked content identifiers."""
    page_number = node.get("page_number", page_number)
    if page_number is not None:
        for mcid in node.get("mcids", []):
            yield page_number, mcid
    for child in node.get("children", []):
        yield from _cell_mcids(child, page_number)


def _tagged_table_blocks(pdf: pdfplumber.PDF) -> dict[int, list[_TableBlock]]:
    """Read tagged borderless tables, retaining headers across page breaks."""
    blocks: dict[int, list[_TableBlock]] = defaultdict(list)
    try:
        tables = list(_structure_nodes(pdf.structure_tree, "Table"))
    except Exception:
        return blocks

    pages = {page.page_number: page for page in pdf.pages}
    char_indexes: dict[int, dict[int, list[dict[str, Any]]]] = {}

    def cell_chars(cell: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
        """Resolve a cell's tagged characters without reindexing pages."""
        selected: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for number, mcid in dict.fromkeys(_cell_mcids(cell)):
            if number not in char_indexes:
                index = defaultdict(list)
                for char in pages[number].chars:
                    if (char_mcid := char.get("mcid")) is not None:
                        index[char_mcid].append(char)
                char_indexes[number] = index
            selected[number].extend(char_indexes[number].get(mcid, ()))
        return selected

    for table in tables:
        try:
            headers: list[str] = []
            table_chars: dict[int, list[dict[str, Any]]] = defaultdict(list)
            page_rows: dict[int, list[list[str]]] = defaultdict(list)
            for row_index, row in enumerate(_structure_nodes([table], "TR")):
                cells = [cell_chars(cell) for cell in row.get("children", [])
                         if cell.get("type") in {"TH", "TD"}]
                for cell in cells:
                    for number, chars in cell.items():
                        table_chars[number].extend(chars)
                row_pages = sorted({number for cell in cells for number in cell})
                values = {
                    number: [utils.extract_text(cell.get(number, []), x_tolerance=X_TOLERANCE)
                             for cell in cells]
                    for number in row_pages
                }
                if row_index == 0:
                    headers = [" ".join(parts) for parts in zip(*values.values())]
                else:
                    for number, row_values in values.items():
                        page_rows[number].append(row_values)

            for number, chars in table_chars.items():
                if chars and (text := _format_rows(headers, page_rows[number])):
                    blocks[number].append(_TableBlock(
                        utils.objects_to_bbox(chars), text,
                        frozenset(char["mcid"] for char in chars),
                    ))
        except Exception:
            # Grid detection or plain extraction remains available for incomplete tags.
            continue
    return blocks
