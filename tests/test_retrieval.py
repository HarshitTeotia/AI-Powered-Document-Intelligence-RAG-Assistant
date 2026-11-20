"""Exercise retrieval with synthetic vectors and local, isolated ChromaDB."""

from dataclasses import FrozenInstanceError
from math import isfinite, sqrt
from pathlib import Path
from unittest.mock import Mock

import pytest

from distribuai import config
from distribuai.rag import retrieval
from distribuai.rag.embeddings import EmbeddedChunk, EmbeddingTokenLimitError
from distribuai.rag.vector_store import open_vector_collection, store_embedded_chunks

_SOURCE = "Rules & Policy.pdf"
_QUERY_VECTOR = (1.0,) + (0.0,) * 383


@pytest.fixture
def query_embedding(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Return a known vector without loading any embedding model."""
    stub = Mock(return_value=_QUERY_VECTOR)
    monkeypatch.setattr(retrieval, "embed_query", stub)
    return stub


@pytest.fixture
def records() -> list[EmbeddedChunk]:
    """Make different cosine similarities so ranking has no ties."""
    return [
        EmbeddedChunk(
            f"chunk-{i}", _SOURCE if i % 2 else "Other.pdf", i + 1, i,
            f"Synthetic passage {i}.", (similarity, sqrt(1 - similarity**2)) + (0.0,) * 382, 5,
        )
        for i, similarity in enumerate((1.0, 0.75, 0.4, 0.0, -0.3, -0.8), start=1)
    ]


@pytest.fixture
def mock_collection() -> Mock:
    """Provide a minimal valid single-query response."""
    return Mock(count=Mock(return_value=1), query=Mock(return_value={
        "ids": [["chunk-1"]], "documents": [["Synthetic passage."]],
        "metadatas": [[{"source_file": _SOURCE, "page_number": 2, "chunk_index": 1}]],
        "distances": [[0.25]],
    }))


def test_real_cosine_ranking_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, records: list[EmbeddedChunk], query_embedding: Mock,
) -> None:
    default = tmp_path / "must-not-open"
    monkeypatch.setattr(config, "CHROMA_STORAGE_DIR", default)
    collection = open_vector_collection(tmp_path / "vectors")
    store_embedded_chunks(collection, list(reversed(records)))

    assert [r.chunk_id for r in retrieval.retrieve(collection, "Question")] == [r.chunk_id for r in records[:5]]
    best, = retrieval.retrieve(collection, "  Question  ", top_k=1)
    assert best.chunk_id == records[0].chunk_id and best.rank == 1
    query_embedding.assert_called_with("Question")

    results = retrieval.retrieve(collection, "Question", top_k=100)
    assert [r.rank for r in results] == list(range(1, 7))
    assert [r.distance for r in results] == sorted(r.distance for r in results)
    assert results[-1].similarity < 0
    for result, record in zip(results, records, strict=True):
        assert (result.chunk_id, result.source_file, result.page_number, result.chunk_index, result.text) == (
            record.chunk_id, record.source_file, record.page_number, record.chunk_index, record.text,
        )
        assert isfinite(result.distance) and isfinite(result.similarity)
        assert result.similarity == 1.0 - result.distance
        assert result.similarity == pytest.approx(record.vector[0], abs=1e-6)
    with pytest.raises(FrozenInstanceError):
        setattr(best, "rank", 0)
    assert not default.exists()


def test_real_filename_filter_searches_past_global_best(tmp_path: Path, records: list[EmbeddedChunk], query_embedding: Mock) -> None:
    collection = open_vector_collection(tmp_path / "vectors")
    store_embedded_chunks(collection, records)
    best_other, = retrieval.retrieve(collection, "Question", top_k=1, source_file="Other.pdf")
    assert best_other.chunk_id == records[1].chunk_id and best_other.source_file == "Other.pdf"
    filtered = retrieval.retrieve(collection, "Question", top_k=10, source_file=_SOURCE)
    assert [r.chunk_id for r in filtered] == [r.chunk_id for r in records[::2]]
    assert [r.rank for r in filtered] == [1, 2, 3]
    assert retrieval.retrieve(collection, "Question", source_file="Missing.pdf") == []


def test_empty_real_collection_does_not_embed(tmp_path: Path, query_embedding: Mock) -> None:
    collection = open_vector_collection(tmp_path / "vectors")
    assert retrieval.retrieve(collection, "Question") == []
    query_embedding.assert_not_called()


def test_passes_vector_and_metadata_filter_directly(mock_collection: Mock, query_embedding: Mock) -> None:
    result, = retrieval.retrieve(mock_collection, "Question", source_file=_SOURCE)
    assert result == retrieval.RetrievedChunk("chunk-1", _SOURCE, 2, 1, "Synthetic passage.", 0.25, 0.75, 1)
    mock_collection.query.assert_called_once_with(
        query_embeddings=[list(_QUERY_VECTOR)], n_results=1,
        where={"source_file": _SOURCE}, include=["documents", "metadatas", "distances"],
    )


@pytest.mark.parametrize("question", ["", "  \t", None, 7])
def test_invalid_question_fails_before_database_access(mock_collection: Mock, question: object) -> None:
    with pytest.raises((TypeError, ValueError), match="question"):
        retrieval.retrieve(mock_collection, question)
    mock_collection.count.assert_not_called()


@pytest.mark.parametrize("top_k", [0, -2, True, False, 2.5, "2"])
def test_invalid_top_k_is_rejected(mock_collection: Mock, top_k: object) -> None:
    with pytest.raises((TypeError, ValueError), match="top_k"):
        retrieval.retrieve(mock_collection, "Question", top_k=top_k)
    mock_collection.count.assert_not_called()


@pytest.mark.parametrize("source_file", ["", " ", ".", "..", "dir/file.pdf", r"C:\private\file.pdf", r"dir\file.pdf", "/file.pdf", "x?.pdf", "NUL.pdf", 8])
def test_invalid_source_filename_is_rejected(mock_collection: Mock, source_file: object) -> None:
    with pytest.raises((TypeError, ValueError), match="source_file"):
        retrieval.retrieve(mock_collection, "Question", source_file=source_file)
    mock_collection.count.assert_not_called()


def test_requires_collection_operations() -> None:
    with pytest.raises(TypeError, match=r"count\(\) and query\(\)"):
        retrieval.retrieve(object(), "Question")


@pytest.mark.parametrize("operation", ["count", "query"])
def test_database_errors_keep_original_cause(mock_collection: Mock, query_embedding: Mock, operation: str) -> None:
    cause = RuntimeError("synthetic failure")
    getattr(mock_collection, operation).side_effect = cause
    with pytest.raises(retrieval.RetrievalError, match=operation) as error:
        retrieval.retrieve(mock_collection, "Question")
    assert error.value.__cause__ is cause


def test_oversized_query_preserves_embedding_error(mock_collection: Mock, query_embedding: Mock) -> None:
    cause = EmbeddingTokenLimitError("Question exceeds the model limit.")
    query_embedding.side_effect = cause
    with pytest.raises(EmbeddingTokenLimitError) as error:
        retrieval.retrieve(mock_collection, "Question")
    assert error.value is cause
    mock_collection.query.assert_not_called()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ids", None, "ids"), ("ids", ["chunk"], "ids"), ("ids", [[""]], "chunk ID"),
        ("documents", [[]], "lengths"), ("documents", [[None]], "document"),
        ("metadatas", None, "metadatas"), ("metadatas", [[None]], "citation metadata"),
        ("metadatas", [[{"page_number": 1, "chunk_index": 1}]], "source filename"),
        ("metadatas", [[{"source_file": "/file.pdf", "page_number": 1, "chunk_index": 1}]], "source filename"),
        ("metadatas", [[{"source_file": _SOURCE, "page_number": True, "chunk_index": 1}]], "page number"),
        ("metadatas", [[{"source_file": _SOURCE, "page_number": 1, "chunk_index": 0}]], "chunk index"),
        ("distances", None, "distances"), ("distances", [[None]], "distance"),
        ("distances", [["0.25"]], "distance"), ("distances", [[True]], "distance"),
        ("distances", [[float("nan")]], "distance"), ("distances", [[float("inf")]], "distance"),
    ],
)
def test_rejects_malformed_results(
    mock_collection: Mock, query_embedding: Mock, field: str, value: object, message: str,
) -> None:
    mock_collection.query.return_value[field] = value
    with pytest.raises(retrieval.RetrievalError, match=message):
        retrieval.retrieve(mock_collection, "Question")


def test_duplicate_ids_are_rejected(mock_collection: Mock, query_embedding: Mock) -> None:
    mock_collection.count.return_value = 2
    mock_collection.query.return_value = {
        field: [values[0] * 2] for field, values in mock_collection.query.return_value.items()
    }
    with pytest.raises(retrieval.RetrievalError, match="Duplicate chunk ID at rank 2"):
        retrieval.retrieve(mock_collection, "Question")


def test_no_similarity_clamping(mock_collection: Mock, query_embedding: Mock) -> None:
    mock_collection.query.return_value["distances"] = [[-0.000001]]
    result, = retrieval.retrieve(mock_collection, "Question")
    assert result.distance == -0.000001 and result.similarity == 1.000001
