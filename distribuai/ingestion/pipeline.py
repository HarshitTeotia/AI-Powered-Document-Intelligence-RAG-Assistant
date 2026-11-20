"""Ingest PDFs into an existing local document collection."""

import stat
from dataclasses import dataclass
from pathlib import Path

from chromadb.api.models.Collection import Collection

from distribuai import config
from distribuai.ingestion.ocr_fallback import OcrError, extract_pdf_with_ocr
from distribuai.ingestion.pdf_extractor import PdfExtractionError, PdfPage, extract_pdf_pages
from distribuai.rag.chunking import DocumentChunk, chunk_pdf_pages
from distribuai.rag.embeddings import EmbeddedChunk, EmbeddingError, EmbeddingTokenLimitError, embed_chunks
from distribuai.rag.vector_store import (
    VectorStoreError,
    _delete_source_chunks,
    _require_source_filename,
    _source_chunk_ids,
    store_embedded_chunks,
)


class IngestionError(Exception):
    """A PDF could not be processed, stored, or fully replaced."""


@dataclass(frozen=True)
class IngestionResult:
    """Summary of a successfully stored document version.

    Counts describe original PDF pages and stored chunks. Chunk size and overlap
    are in characters. The OCR count covers pages initially flagged for OCR.
    Empty page numbers are sorted, unique, and 1-based. The list is a snapshot
    owned by the result, even though the frozen fields cannot be reassigned.
    """

    source_file: str
    page_count: int
    chunk_count: int
    chunk_size_used: int
    overlap_used: int
    ocr_page_count: int
    stale_chunks_removed: int
    pages_without_text: list[int]


def ingest_pdf(
    pdf_path: Path, collection: Collection, source_name: str | None = None,
) -> IngestionResult:
    """Extract, embed, and upsert a whole PDF before removing its old chunks.

    Replacement is not atomic. Serialize operations affecting the same source.
    """
    filename = _validate_input(pdf_path, collection, source_name)
    pages, ocr_count = _extract_pages(pdf_path, filename)
    pages_without_text = sorted({page.page_number for page in pages if not page.text.strip()})
    if not any(page.text.strip() for page in pages):
        raise IngestionError(f"No usable text could be extracted from document {filename}.")

    records, size, overlap = _embed_document(pages, filename)
    try:
        written = store_embedded_chunks(collection, records)
    except (VectorStoreError, TypeError, ValueError) as exc:
        raise IngestionError(
            f"Could not store chunks for {filename}. Partial writes may remain. Stale cleanup was not started."
        ) from exc
    if type(written) is not int or written != len(records):
        raise IngestionError(
            f"Storage reported an unexpected chunk count for {filename}. Stale cleanup was not started."
        )

    current_ids = {record.chunk_id for record in records}
    try:
        stored_ids = _source_chunk_ids(collection, filename)
        if not current_ids <= stored_ids:
            raise VectorStoreError("Some newly written chunk IDs are missing from the source.")
        removed = _delete_source_chunks(collection, filename, sorted(stored_ids - current_ids))
    except VectorStoreError as exc:
        raise IngestionError(
            f"New chunks were stored for {filename}, but stale cleanup is incomplete."
        ) from exc

    return IngestionResult(filename, len(pages), written, size, overlap, ocr_count, removed, pages_without_text)


def _validate_input(pdf_path: Path, collection: Collection, source_name: str | None) -> str:
    """Validate the citation identity, input file, and required storage operations."""
    if not isinstance(pdf_path, Path):
        raise IngestionError("pdf_path must be a pathlib.Path.")
    try:
        filename = _require_source_filename(pdf_path.name if source_name is None else source_name)
    except (TypeError, ValueError) as exc:
        raise IngestionError("source_name must be a valid citation filename without directories.") from exc
    try:
        mode = pdf_path.stat().st_mode
    except OSError as exc:
        raise IngestionError(f"Could not access the PDF file for {filename}.") from exc
    if not stat.S_ISREG(mode):
        raise IngestionError("pdf_path must point to a regular PDF file.")
    if pdf_path.suffix.casefold() != ".pdf":
        raise IngestionError("pdf_path must have a .pdf extension.")
    if not all(callable(getattr(collection, method, None)) for method in ("get", "upsert", "delete")):
        raise IngestionError("collection must provide get(), upsert(), and delete() operations.")
    return filename


def _extract_pages(pdf_path: Path, filename: str) -> tuple[list[PdfPage], int]:
    """Count native OCR flags in a separate pass before running OCR fallback."""
    try:
        native = extract_pdf_pages(pdf_path)
        pages = extract_pdf_with_ocr(pdf_path)
    except (PdfExtractionError, OcrError, OSError, ValueError) as exc:
        raise IngestionError(f"PDF extraction or OCR failed for {filename}.") from exc
    if [page.page_number for page in native] != [page.page_number for page in pages]:
        raise IngestionError(f"Page numbers changed between extraction passes for {filename}.")
    return pages, sum(page.needs_ocr for page in native)


def _embed_document(pages: list[PdfPage], filename: str) -> tuple[list[EmbeddedChunk], int, int]:
    """Retry the whole document only when the embedding token limit is exceeded."""
    last_error: EmbeddingTokenLimitError | None = None
    for size, overlap in config.INGESTION_CHUNK_RETRY_SETTINGS:
        try:
            chunks = chunk_pdf_pages(pages, filename, chunk_size=size, chunk_overlap=overlap)
        except (TypeError, ValueError) as exc:
            raise IngestionError(f"Could not chunk document {filename}.") from exc
        if not chunks:
            raise IngestionError(f"No usable chunks were produced for {filename}.")
        try:
            records = embed_chunks(chunks)
        except EmbeddingTokenLimitError as exc:
            last_error = exc
            continue
        except (EmbeddingError, TypeError, ValueError) as exc:
            raise IngestionError(f"Could not embed document {filename}.") from exc

        _validate_embedding_output(chunks, records, filename)
        return records, size, overlap

    if last_error is None:
        raise IngestionError("No ingestion chunk configurations are available.")
    # The existing exception has a safe message with the ID, count, and limit.
    raise IngestionError(f"Chunk configurations exhausted for {filename}. {last_error}") from last_error


def _validate_embedding_output(
    chunks: list[DocumentChunk], records: list[EmbeddedChunk], filename: str
) -> None:
    """Require one unchanged citation and original passage per embedded chunk."""
    if not isinstance(records, list) or not records or len(records) != len(chunks):
        raise IngestionError(f"Embedding returned an unexpected number of chunks for {filename}.")
    for chunk, record in zip(chunks, records, strict=True):
        if not isinstance(record, EmbeddedChunk) or (
            record.chunk_id, record.source_file, record.page_number, record.chunk_index, record.text
        ) != (chunk.chunk_id, filename, chunk.page_number, chunk.chunk_index, chunk.text):
            raise IngestionError(f"Embedding returned mismatched chunk metadata for {filename}.")
