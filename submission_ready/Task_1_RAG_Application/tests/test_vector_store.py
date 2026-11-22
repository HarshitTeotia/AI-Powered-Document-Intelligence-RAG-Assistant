"""Verify local ChromaDB persistence with synthetic embeddings."""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from distribuai import config
from distribuai.rag.embeddings import EMBEDDING_DIMENSION, EmbeddedChunk
from distribuai.rag.vector_store import (
    COLLECTION_NAME,
    VectorStoreError,
    delete_document,
    list_documents,
    open_vector_collection,
    store_embedded_chunks,
)


def _record(sequence: int = 1, filename: str = "sample.pdf") -> EmbeddedChunk:
    """Build a synthetic record with known citation fields and a unit vector."""
    vector = (1.0,) + (0.0,) * (EMBEDDING_DIMENSION - 1)
    return EmbeddedChunk(f"chunk-{sequence}", filename, sequence + 1, sequence,
                         f"Synthetic passage {sequence}.", vector, 12 + sequence)


@pytest.fixture(autouse=True)
def protect_default_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Catch unintended default initialization without touching production data."""
    sentinel = tmp_path / "must-not-create"
    monkeypatch.setattr(config, "CHROMA_STORAGE_DIR", sentinel)
    yield sentinel
    assert not sentinel.exists()


@pytest.fixture
def mock_collection() -> Mock:
    """Provide a compatible collection whose writes can be inspected."""
    return Mock(
        metadata={"embedding_model": config.EMBEDDING_MODEL_ID, "embedding_dimension": 384},
        configuration_json={"hnsw": {"space": "cosine"}, "embedding_function": None},
    )


def test_open_creates_and_reuses_local_collection(tmp_path: Path) -> None:
    directory = tmp_path / "custom" / "chroma"
    first = open_vector_collection(directory)
    second = open_vector_collection(directory)

    assert directory.is_dir()
    assert first.name == second.name == COLLECTION_NAME == "distribuai_documents"
    assert first.id == second.id
    assert first.metadata == {
        "embedding_model": config.EMBEDDING_MODEL_ID,
        "embedding_dimension": EMBEDDING_DIMENSION,
    }
    assert first.configuration["hnsw"]["space"] == "cosine"
    assert first.configuration["embedding_function"] is None
    assert first.count() == 0
    with pytest.raises(ValueError, match="embedding function"):
        first.upsert(ids=["missing-vector"], documents=["Synthetic text."])
    assert first.count() == 0


def test_open_uses_the_configured_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = tmp_path / "configured-default"
    with monkeypatch.context() as overrides:
        overrides.setattr(config, "CHROMA_STORAGE_DIR", directory)
        assert open_vector_collection().count() == 0
    assert directory.is_dir()


def test_empty_input_never_opens_or_writes(mock_collection: Mock) -> None:
    with patch("distribuai.rag.vector_store.chromadb.PersistentClient") as client:
        assert store_embedded_chunks(mock_collection, []) == 0
    client.assert_not_called()
    mock_collection.upsert.assert_not_called()


def test_records_round_trip_and_re_upload_is_idempotent(tmp_path: Path) -> None:
    directory = tmp_path / "collection"
    collection = open_vector_collection(directory)
    records = [_record(1, "Policy & Rules.pdf"), _record(2, "Delivery FAQ.pdf")]
    original = records.copy()

    with patch("distribuai.rag.embeddings.load_embedding_model") as model:
        assert store_embedded_chunks(collection, records, batch_size=1) == 2
        assert store_embedded_chunks(collection, records) == 2
    model.assert_not_called()
    assert records == original
    reopened = open_vector_collection(directory)
    assert reopened.count() == 2

    for record in records:
        result = reopened.get(ids=[record.chunk_id], include=["documents", "metadatas", "embeddings"])
        assert result["ids"] == [record.chunk_id]
        assert result["documents"] == [record.text]
        assert result["metadatas"] == [{
            "source_file": record.source_file,
            "page_number": record.page_number,
            "chunk_index": record.chunk_index,
            "token_count": record.token_count,
        }]
        vector = result["embeddings"][0]
        assert len(vector) == 384 and tuple(vector) == record.vector


def test_upsert_replaces_values_under_same_id(tmp_path: Path) -> None:
    collection = open_vector_collection(tmp_path / "collection")
    original = _record()
    updated = replace(
        original, text="Updated passage.", source_file="Version 2.pdf", page_number=4,
        chunk_index=2, token_count=20, vector=(0.0, 1.0) + (0.0,) * 382,
    )
    store_embedded_chunks(collection, [original])
    store_embedded_chunks(collection, [updated])

    result = collection.get(ids=[original.chunk_id], include=["documents", "metadatas", "embeddings"])
    assert collection.count() == 1
    assert result["documents"] == [updated.text]
    assert result["metadatas"] == [{
        "source_file": "Version 2.pdf", "page_number": 4, "chunk_index": 2, "token_count": 20,
    }]
    assert tuple(result["embeddings"][0]) == updated.vector


def test_batches_keep_input_order(mock_collection: Mock) -> None:
    records = [_record(i) for i in (5, 1, 4, 3, 2)]
    assert store_embedded_chunks(mock_collection, records, batch_size=2) == 5

    assert [call.kwargs["ids"] for call in mock_collection.upsert.call_args_list] == [
        ["chunk-5", "chunk-1"], ["chunk-4", "chunk-3"], ["chunk-2"],
    ]
    first = mock_collection.upsert.call_args_list[0].kwargs
    assert first["documents"] == [records[0].text, records[1].text]
    assert first["embeddings"][0] == list(records[0].vector)


@pytest.mark.parametrize("batch_size", [0, -1, True, False, 0.5, "2"])
def test_invalid_batch_sizes_do_not_write(mock_collection: Mock, batch_size: object) -> None:
    with pytest.raises((TypeError, ValueError), match="batch_size"):
        store_embedded_chunks(mock_collection, [_record()], batch_size=batch_size)
    mock_collection.upsert.assert_not_called()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"chunk_id": " "}, "chunk ID"),
        ({"chunk_id": "chunk-1"}, "Duplicate chunk ID"),
        ({"page_number": 0}, "page_number"),
        ({"page_number": True}, "page_number"),
        ({"chunk_index": -1}, "chunk_index"),
        ({"token_count": 0}, "token_count"),
        ({"token_count": 2**63}, "token_count"),
        ({"source_file": r"C:\private\sample.pdf"}, "source filename"),
        ({"source_file": "subfolder/sample.pdf"}, "source filename"),
        ({"source_file": "NUL.pdf"}, "source filename"),
        ({"text": " \n"}, "nonempty text"),
        ({"vector": (0.0,) * 383}, "384-dimensional"),
        ({"vector": (float("nan"),) * 384}, "finite"),
        ({"vector": (float("inf"),) * 384}, "finite"),
        ({"vector": (1e40,) * 384}, "float32"),
        ({"vector": ("text",) * 384}, "numeric"),
        ({"vector": (True,) * 384}, "numeric"),
    ],
)
def test_validates_every_record_before_first_batch(
    mock_collection: Mock, changes: dict[str, object], message: str
) -> None:
    records = [_record(1), replace(_record(2), **changes)]
    with pytest.raises((TypeError, ValueError), match=message):
        store_embedded_chunks(mock_collection, records, batch_size=1)
    mock_collection.upsert.assert_not_called()


@pytest.mark.parametrize("records", [(_record(),), ["not a record"]])
def test_rejects_invalid_record_container_or_item(mock_collection: Mock, records: object) -> None:
    with pytest.raises(TypeError, match="EmbeddedChunk"):
        store_embedded_chunks(mock_collection, records)
    mock_collection.upsert.assert_not_called()


@pytest.mark.parametrize("change", [{"embedding_model": "other"}, {"embedding_dimension": 128},
                                    {"embedding_dimension": 384.0}])
def test_reopen_rejects_incompatible_metadata(tmp_path: Path, change: dict[str, object]) -> None:
    directory = tmp_path / "collection"
    collection = open_vector_collection(directory)
    store_embedded_chunks(collection, [_record()])
    collection.modify(metadata={**collection.metadata, **change})

    for _ in range(2):
        with pytest.raises(VectorStoreError, match="Incompatible collection metadata"):
            open_vector_collection(directory)
    assert collection.count() == 1


def test_reopen_rejects_wrong_stored_dimension_with_matching_metadata(tmp_path: Path) -> None:
    directory = tmp_path / "collection"
    collection = open_vector_collection(directory)
    collection.upsert(ids=["outside-write"], embeddings=[[1.0, 0.0]])

    with pytest.raises(VectorStoreError, match="Incompatible stored vector dimension"):
        open_vector_collection(directory)
    assert collection.count() == 1


@pytest.mark.parametrize(
    ("configuration", "message"),
    [({"hnsw": {"space": "l2"}}, "cosine"),
     ({"hnsw": {"space": "cosine"}, "embedding_function": {}}, "supplied embeddings")],
)
def test_rejects_incompatible_index_or_embedding_function(
    mock_collection: Mock, configuration: dict[str, object], message: str
) -> None:
    mock_collection.configuration_json = configuration
    with pytest.raises(VectorStoreError, match=message):
        store_embedded_chunks(mock_collection, [_record()])
    mock_collection.upsert.assert_not_called()


def test_rejects_non_directory_storage_path(tmp_path: Path) -> None:
    path = tmp_path / "file"
    path.touch()
    with pytest.raises(VectorStoreError, match="writable directory") as error:
        open_vector_collection(path)
    assert isinstance(error.value.__cause__, OSError)


def test_chroma_initialization_error_is_chained(tmp_path: Path) -> None:
    cause = RuntimeError("synthetic initialization error")
    with (
        patch("distribuai.rag.vector_store.chromadb.PersistentClient", side_effect=cause) as client,
        pytest.raises(VectorStoreError, match="initialize local ChromaDB") as error,
    ):
        open_vector_collection(tmp_path / "collection")

    assert error.value.__cause__ is cause
    assert client.call_args.kwargs["settings"].anonymized_telemetry is False


def test_collection_creation_error_is_chained(tmp_path: Path) -> None:
    cause = RuntimeError("synthetic create failure")
    with (
        patch("distribuai.rag.vector_store.chromadb.PersistentClient") as client,
        pytest.raises(VectorStoreError, match="open or create") as error,
    ):
        client.return_value.get_or_create_collection.side_effect = cause
        open_vector_collection(tmp_path / "collection")

    assert error.value.__cause__ is cause
    assert client.return_value.get_or_create_collection.call_args.kwargs["embedding_function"] is None


def test_later_upsert_failure_reports_prior_progress(mock_collection: Mock) -> None:
    cause = RuntimeError("synthetic storage failure")
    mock_collection.upsert.side_effect = [None, cause]

    with pytest.raises(VectorStoreError, match="batch 2 after 1 records were written") as error:
        store_embedded_chunks(mock_collection, [_record(1), _record(2), _record(3)], batch_size=1)

    assert error.value.__cause__ is cause
    assert mock_collection.upsert.call_count == 2


def test_document_listing_and_scoped_deletion_are_paginated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use real persistence and force reads/deletes through multiple pages."""
    monkeypatch.setattr("distribuai.rag.vector_store._DOCUMENT_BATCH_SIZE", 2)
    directory = tmp_path / "collection"
    collection = open_vector_collection(directory)
    assert list_documents(collection) == []
    assert delete_document(collection, "Missing.pdf") == 0
    name = "Shipping & Returns.pdf"
    records = [
        replace(_record(1, name), page_number=4),
        replace(_record(2, name), page_number=4),
        replace(_record(3, name), page_number=1),
        _record(4, "Another.pdf"),
        _record(5, "Another.pdf"),
    ]
    store_embedded_chunks(collection, records[:1])
    assert list_documents(collection) == [(name, 1, [4])]
    store_embedded_chunks(collection, records[1:])
    assert list_documents(collection) == [("Another.pdf", 2, [5, 6]), (name, 3, [1, 4])]

    pdf = tmp_path / name
    pdf.write_bytes(b"synthetic file bytes")
    assert delete_document(collection, name) == 3
    assert delete_document(collection, name) == 0
    assert pdf.read_bytes() == b"synthetic file bytes"
    reopened = open_vector_collection(directory)
    assert reopened.count() == 2
    assert list_documents(reopened) == [("Another.pdf", 2, [5, 6])]
    assert set(reopened.get(include=[])["ids"]) == {"chunk-4", "chunk-5"}


@pytest.mark.parametrize("source", ["", " ", ".", "..", "private/file.pdf", r"C:\private\file.pdf", "NUL.pdf"])
def test_delete_rejects_invalid_source_before_database_access(mock_collection: Mock, source: str) -> None:
    with pytest.raises(ValueError, match="source"):
        delete_document(mock_collection, source)
    mock_collection.get.assert_not_called()
    mock_collection.delete.assert_not_called()


@pytest.mark.parametrize(
    ("metadata", "message"),
    [(None, "citation metadata"), ({"page_number": 1}, "source filename"),
     ({"source_file": "/file.pdf", "page_number": 1}, "source filename"),
     ({"source_file": "File.pdf"}, "page number"),
     ({"source_file": "File.pdf", "page_number": True}, "page number")],
)
def test_listing_rejects_missing_or_invalid_metadata(mock_collection: Mock, metadata: object, message: str) -> None:
    mock_collection.get.return_value = {"ids": ["chunk-1"], "metadatas": [metadata]}
    with pytest.raises(VectorStoreError, match=message):
        list_documents(mock_collection)


def test_document_read_and_delete_errors_keep_their_cause(mock_collection: Mock) -> None:
    cause = RuntimeError("synthetic storage failure")
    mock_collection.get.side_effect = cause
    with pytest.raises(VectorStoreError, match="Could not read") as read_error:
        list_documents(mock_collection)
    assert read_error.value.__cause__ is cause

    mock_collection.get.side_effect = None
    mock_collection.get.return_value = {"ids": ["chunk-1"]}
    mock_collection.delete.side_effect = cause
    with pytest.raises(VectorStoreError, match="Could not delete") as delete_error:
        delete_document(mock_collection, "Rules & Policy.pdf")
    assert delete_error.value.__cause__ is cause
    mock_collection.get.assert_called_with(where={"source_file": "Rules & Policy.pdf"}, limit=256, offset=0, include=[])
    mock_collection.delete.assert_called_once_with(ids=["chunk-1"], where={"source_file": "Rules & Policy.pdf"})
