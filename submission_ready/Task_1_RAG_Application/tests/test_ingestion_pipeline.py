"""Exercise whole-PDF orchestration with synthetic pages and isolated ChromaDB."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from distribuai import config
from distribuai.ingestion import ocr_fallback, pipeline
from distribuai.ingestion.ocr_fallback import OcrError
from distribuai.ingestion.pdf_extractor import PdfExtractionError, PdfPage
from distribuai.rag.chunking import DocumentChunk
from distribuai.rag.embeddings import EmbeddedChunk, EmbeddingError, EmbeddingTokenLimitError
from distribuai.rag.vector_store import VectorStoreError, list_documents, open_vector_collection, store_embedded_chunks

_VECTOR = (1.0,) + (0.0,) * 383


def _embeddings(chunks: list[DocumentChunk]) -> list[EmbeddedChunk]:
    """Attach synthetic vectors while preserving the chunker's real metadata."""
    return [EmbeddedChunk(c.chunk_id, c.source_file, c.page_number, c.chunk_index, c.text, _VECTOR, 8)
            for c in chunks]


@pytest.fixture
def pdf(tmp_path: Path) -> Path:
    path = tmp_path / "Shipping & Returns.PDF"
    path.write_bytes(b"synthetic input; extraction is mocked")
    return path


@pytest.fixture
def extraction(monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, Mock, list[PdfPage]]:
    native = Mock(return_value=[PdfPage(1, "short", True), PdfPage(2, "", True),
                               PdfPage(3, "The native passage remains usable.", False)])
    pages = [PdfPage(1, "The scanned passage is now readable.", False),
             PdfPage(2, " \t", True), PdfPage(3, "The native passage remains usable.", False)]
    ocr = Mock(side_effect=lambda path, native_pages: (pages, sum(page.needs_ocr for page in native_pages)))
    monkeypatch.setattr(pipeline, "extract_pdf_pages", native)
    monkeypatch.setattr(pipeline, "_supplement_pdf_pages", ocr)
    monkeypatch.setattr(pipeline, "embed_chunks", Mock(side_effect=_embeddings))
    return native, ocr, pages


@pytest.fixture
def collection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "CHROMA_STORAGE_DIR", tmp_path / "unused-default")
    isolated = open_vector_collection(tmp_path / "chroma")
    yield isolated
    assert not config.CHROMA_STORAGE_DIR.exists()


def _ids(collection, filename: str) -> set[str]:
    return set(collection.get(where={"source_file": filename}, include=[])["ids"])


def test_ingests_and_reports_original_pages_and_ocr_attempts(pdf, collection, extraction) -> None:
    native, ocr, pages = extraction
    before_pages = pages.copy()
    before_pdf = pdf.read_bytes()

    result = pipeline.ingest_pdf(pdf, collection)

    assert result == pipeline.IngestionResult(pdf.name, 3, 2, 800, 100, 2, 0, [2])
    assert list_documents(collection) == [(pdf.name, 2, [1, 3])]
    stored = collection.get(where={"source_file": pdf.name})
    assert {m["page_number"]: text for m, text in zip(stored["metadatas"], stored["documents"], strict=True)} == {
        1: pages[0].text, 3: pages[2].text,
    }
    native.assert_called_once_with(pdf)
    ocr.assert_called_once_with(pdf, native.return_value)
    assert pages == before_pages and pdf.read_bytes() == before_pdf
    with pytest.raises(FrozenInstanceError):
        setattr(result, "page_count", 10)


def test_identical_reingestion_and_renamed_sources_stay_independent(pdf, collection, extraction) -> None:
    name = "Chosen & Citation.pdf"
    original = pipeline.ingest_pdf(pdf, collection, source_name=name)
    ids = _ids(collection, name)
    assert original.source_file == name
    assert pipeline.ingest_pdf(pdf, collection, source_name=name) == original
    assert collection.count() == original.chunk_count and _ids(collection, name) == ids
    renamed = pipeline.ingest_pdf(pdf, collection, source_name=pdf.name)
    assert renamed.stale_chunks_removed == 0
    assert _ids(collection, name) == ids and not (ids & _ids(collection, pdf.name))
    assert collection.count() == original.chunk_count + renamed.chunk_count


def test_ocr_supplements_the_first_native_pass_and_counts_attempts(pdf, monkeypatch) -> None:
    pages = [PdfPage(1, "short", True), PdfPage(2, "", True), PdfPage(3, "A readable native page.", False)]
    native = Mock(return_value=pages)
    document = Mock()
    recognize = Mock(side_effect=["Enough recognized text for page one.", ""])
    monkeypatch.setattr(pipeline, "extract_pdf_pages", native)
    monkeypatch.setattr(ocr_fallback, "extract_pdf_pages", Mock(side_effect=AssertionError("Duplicate native pass")))
    monkeypatch.setattr(ocr_fallback, "_resolve_tesseract_cmd", lambda: "tesseract")
    monkeypatch.setattr(ocr_fallback.pypdfium2, "PdfDocument", Mock(return_value=document))
    monkeypatch.setattr(ocr_fallback, "_ocr_page", recognize)

    extracted, attempts = pipeline._extract_pages(pdf, pdf.name)

    assert attempts == 2
    assert extracted == [PdfPage(1, "Enough recognized text for page one.", False), pages[1], pages[2]]
    assert pages[0] == PdfPage(1, "short", True)
    native.assert_called_once_with(pdf)
    assert [call.args for call in recognize.call_args_list] == [
        (document, 1, "tesseract"), (document, 2, "tesseract"),
    ]
    document.close.assert_called_once_with()


def test_edited_content_replaces_stale_ids_after_write(pdf, collection, extraction) -> None:
    pipeline.ingest_pdf(pdf, collection)
    original_ids = _ids(collection, pdf.name)
    pipeline.ingest_pdf(pdf, collection, "Independent.pdf")
    independent = _ids(collection, "Independent.pdf")
    extraction[2][0] = PdfPage(1, "New text for the edited PDF version.", False)

    result = pipeline.ingest_pdf(pdf, collection)

    assert result.stale_chunks_removed == len(original_ids)
    assert not original_ids & _ids(collection, pdf.name)
    assert len(_ids(collection, pdf.name)) == result.chunk_count
    assert _ids(collection, "Independent.pdf") == independent
    assert collection.count() == len(independent) + result.chunk_count


@pytest.mark.parametrize("failed_attempts", [1, 3])
def test_token_retry_rechunks_whole_document_and_stores_only_the_successful_attempt(
    pdf, collection, extraction, monkeypatch: pytest.MonkeyPatch, failed_attempts: int,
) -> None:
    attempts: list[list[DocumentChunk]] = []
    assert config.INGESTION_CHUNK_RETRY_SETTINGS == ((800, 100), (700, 90), (600, 80), (500, 70), (400, 60))

    def encode(chunks: list[DocumentChunk]) -> list[EmbeddedChunk]:
        assert collection.count() == 0
        assert {chunk.page_number for chunk in chunks} == {1, 3}
        attempts.append(chunks)
        if len(attempts) <= failed_attempts:
            raise EmbeddingTokenLimitError(f"Chunk {chunks[0].chunk_id} has 275 tokens, exceeding 256.")
        return _embeddings(chunks)

    monkeypatch.setattr(pipeline, "embed_chunks", encode)
    with patch.object(pipeline, "chunk_pdf_pages", wraps=pipeline.chunk_pdf_pages) as splitter:
        result = pipeline.ingest_pdf(pdf, collection)

    settings = config.INGESTION_CHUNK_RETRY_SETTINGS[:failed_attempts + 1]
    assert [(call.kwargs["chunk_size"], call.kwargs["chunk_overlap"]) for call in splitter.call_args_list] == list(settings)
    assert (result.chunk_size_used, result.overlap_used) == settings[-1]
    assert _ids(collection, pdf.name) == {chunk.chunk_id for chunk in attempts[-1]}
    assert not {chunk.chunk_id for attempt in attempts[:-1] for chunk in attempt} & _ids(collection, pdf.name)


@pytest.mark.parametrize("stage", ["extract_pdf_pages", "_supplement_pdf_pages", "embed_chunks", "chunk_pdf_pages"])
def test_non_token_failure_preserves_old_version_without_retry(
    pdf, collection, extraction, monkeypatch: pytest.MonkeyPatch, stage: str,
) -> None:
    pipeline.ingest_pdf(pdf, collection)
    before = collection.get()
    cause = {
        "extract_pdf_pages": PdfExtractionError("synthetic PDF failure"),
        "_supplement_pdf_pages": OcrError("synthetic OCR failure"),
        "embed_chunks": EmbeddingError("synthetic model failure"),
        "chunk_pdf_pages": ValueError("synthetic chunking failure"),
    }[stage]
    operation = Mock(side_effect=cause)
    monkeypatch.setattr(pipeline, stage, operation)

    with pytest.raises(pipeline.IngestionError) as error:
        pipeline.ingest_pdf(pdf, collection)

    operation.assert_called_once()
    assert error.value.__cause__ is cause
    assert collection.get() == before


def test_retry_exhaustion_preserves_old_version_and_reports_last_failure(pdf, collection, extraction) -> None:
    pipeline.ingest_pdf(pdf, collection)
    original_ids = _ids(collection, pdf.name)
    failures = []

    def reject(chunks: list[DocumentChunk]) -> list[EmbeddedChunk]:
        cause = EmbeddingTokenLimitError(f"Chunk {chunks[0].chunk_id} has 281 tokens, exceeding the model limit of 256.")
        failures.append(cause)
        raise cause

    with patch.object(pipeline, "embed_chunks", side_effect=reject):
        with pytest.raises(pipeline.IngestionError, match="configurations exhausted") as error:
            pipeline.ingest_pdf(pdf, collection)

    assert len(failures) == len(config.INGESTION_CHUNK_RETRY_SETTINGS)
    assert error.value.__cause__ is failures[-1]
    message = str(error.value)
    assert pdf.name in message and str(failures[-1]) in message
    assert all(page.text.strip() not in message for page in extraction[2] if page.text.strip())
    assert _ids(collection, pdf.name) == original_ids


@pytest.mark.parametrize("empty_text", ["", " \t\n"])
def test_unusable_document_leaves_existing_version_untouched(pdf, collection, extraction, empty_text: str) -> None:
    pipeline.ingest_pdf(pdf, collection)
    original_ids = _ids(collection, pdf.name)
    extraction[2][:] = [PdfPage(page.page_number, empty_text, True) for page in extraction[2]]
    with patch.object(pipeline, "embed_chunks") as embed:
        with pytest.raises(pipeline.IngestionError, match="No usable text"):
            pipeline.ingest_pdf(pdf, collection)
    embed.assert_not_called()
    assert _ids(collection, pdf.name) == original_ids


@pytest.mark.parametrize("broken", ["empty", "short", "metadata"])
def test_invalid_embedding_output_never_stores(pdf, collection, extraction, broken: str) -> None:
    def encode(chunks: list[DocumentChunk]) -> list[EmbeddedChunk]:
        records = _embeddings(chunks)
        if broken == "empty":
            return []
        if broken == "short":
            return records[:-1]
        return [replace(records[0], page_number=99), *records[1:]]

    with patch.object(pipeline, "embed_chunks", side_effect=encode):
        with pytest.raises(pipeline.IngestionError, match="Embedding returned"):
            pipeline.ingest_pdf(pdf, collection)
    assert collection.count() == 0


@pytest.mark.parametrize("failure", ["failed", "partial", "count"])
def test_store_failure_never_deletes_stale_chunks(pdf, collection, extraction, failure: str) -> None:
    pipeline.ingest_pdf(pdf, collection)
    original_ids = _ids(collection, pdf.name)
    extraction[2][0] = PdfPage(1, "An edited page changes the document fingerprint.", False)
    cause = VectorStoreError("synthetic storage failure")

    def fail(target, records: list[EmbeddedChunk]) -> int:
        if failure == "count":
            return len(records) - 1
        if failure == "partial":
            store_embedded_chunks(target, records[:1])
        raise cause

    with (
        patch.object(pipeline, "store_embedded_chunks", side_effect=fail),
        patch.object(collection, "delete", wraps=collection.delete) as remove,
        pytest.raises(pipeline.IngestionError, match="Stale cleanup was not started") as error,
    ):
        pipeline.ingest_pdf(pdf, collection)

    remove.assert_not_called()
    assert original_ids <= _ids(collection, pdf.name)
    assert collection.count() == len(original_ids) + (failure == "partial")
    if failure != "count":
        assert error.value.__cause__ is cause and "Partial writes may remain" in str(error.value)


def test_cleanup_failure_reports_new_records_without_rollback(pdf, collection, extraction) -> None:
    pipeline.ingest_pdf(pdf, collection)
    original_ids = _ids(collection, pdf.name)
    extraction[2][0] = PdfPage(1, "Edited text with new stable IDs.", False)
    cause = RuntimeError("synthetic delete failure")
    with patch.object(collection, "delete", side_effect=cause):
        with pytest.raises(pipeline.IngestionError, match="New chunks were stored.*cleanup is incomplete") as error:
            pipeline.ingest_pdf(pdf, collection)

    assert original_ids < _ids(collection, pdf.name)
    assert error.value.__cause__.__cause__ is cause


def test_missing_new_ids_after_upsert_blocks_stale_deletion(pdf, collection, extraction) -> None:
    pipeline.ingest_pdf(pdf, collection)
    original_ids = _ids(collection, pdf.name)
    extraction[2][0] = PdfPage(1, "Edited text with unseen new IDs.", False)
    with (
        patch.object(pipeline, "_source_chunk_ids", return_value=original_ids),
        patch.object(collection, "delete", wraps=collection.delete) as remove,
        pytest.raises(pipeline.IngestionError, match="cleanup is incomplete") as error,
    ):
        pipeline.ingest_pdf(pdf, collection)
    remove.assert_not_called()
    assert "newly written" in str(error.value.__cause__)


@pytest.mark.parametrize("source", ["", "   ", ".", "..", "/absolute.pdf", r"C:\absolute.pdf", "dir/file.pdf", r"dir\file.pdf"])
def test_invalid_source_is_rejected_before_extraction(pdf, collection, extraction, source: str) -> None:
    with pytest.raises(pipeline.IngestionError, match="source_name"):
        pipeline.ingest_pdf(pdf, collection, source)
    extraction[0].assert_not_called()
    assert collection.count() == 0


def test_invalid_paths_and_collection_are_rejected(pdf, tmp_path: Path, collection, extraction) -> None:
    wrong_extension = tmp_path / "wrong.txt"
    wrong_extension.touch()
    for path, target, message in (
        (str(pdf), collection, "pathlib.Path"), (tmp_path, collection, "regular PDF"),
        (wrong_extension, collection, "extension"), (pdf, object(), "collection"),
    ):
        with pytest.raises(pipeline.IngestionError, match=message):
            pipeline.ingest_pdf(path, target)
    with pytest.raises(pipeline.IngestionError, match="Could not access") as error:
        pipeline.ingest_pdf(tmp_path / "missing.pdf", collection)
    assert isinstance(error.value.__cause__, FileNotFoundError)
    extraction[0].assert_not_called()
