"""Verify embedding preflight, metadata, batching, and model failures offline."""

import sys
from dataclasses import FrozenInstanceError, replace
from types import ModuleType
from unittest.mock import Mock, call, patch

import pytest

from distribuai import config
from distribuai.rag import embeddings
from distribuai.rag.chunking import DocumentChunk


def _chunk(index: int, text: str | None = None) -> DocumentChunk:
    """Give each sample distinct metadata and predictable token counts."""
    return DocumentChunk(f"chunk-{index}", f"sample-{index}.pdf", index + 1, index,
                         f"sample chunk {index}" if text is None else text)


def _tokenize(text: str, **options: bool) -> dict[str, list[int]]:
    """Return a deterministic token sequence with two special tokens."""
    return {"input_ids": [101, *range(len(text.split())), 102]}


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Use the locked API's shape without importing the ML stack."""
    model = Mock(max_seq_length=8)
    model.tokenizer.side_effect = _tokenize
    model.encode.side_effect = lambda texts, **kwargs: [
        [int(text.split()[-1]), *([0] * 383)] for text in texts
    ]
    monkeypatch.setattr(embeddings, "load_embedding_model", Mock(return_value=model))
    return model


def test_empty_input_does_not_load_model() -> None:
    with patch.object(embeddings, "load_embedding_model") as load:
        assert embeddings.embed_chunks([]) == []
    load.assert_not_called()


def test_single_chunk_preserves_metadata_and_returns_immutable_floats(fake_model: Mock) -> None:
    chunk = _chunk(1)
    before = replace(chunk)
    record, = embeddings.embed_chunks([chunk])

    assert record == embeddings.EmbeddedChunk(
        chunk.chunk_id, chunk.source_file, chunk.page_number, chunk.chunk_index,
        chunk.text, (1.0,) + (0.0,) * 383, 5,
    )
    assert len(record.vector) == 384 and all(type(value) is float for value in record.vector)
    assert chunk == before
    with pytest.raises(FrozenInstanceError):
        setattr(record, "text", "changed")


def test_batches_preserve_input_order_and_request_normalization(fake_model: Mock) -> None:
    chunks = [_chunk(i) for i in (4, 2, 5, 1, 3)]
    records = embeddings.embed_chunks(chunks, batch_size=2)

    assert [r.chunk_id for r in records] == [chunk.chunk_id for chunk in chunks]
    assert [r.vector[0] for r in records] == [4.0, 2.0, 5.0, 1.0, 3.0]
    options = dict(batch_size=2, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    assert fake_model.encode.call_args_list == [
        call([chunk.text for chunk in chunks[:2]], **options),
        call([chunk.text for chunk in chunks[2:4]], **options),
        call([chunks[-1].text], **options),
    ]


def test_exact_limit_includes_special_tokens_without_truncation(fake_model: Mock) -> None:
    fake_model.max_seq_length = 5
    chunk = _chunk(1)
    record, = embeddings.embed_chunks([chunk])

    assert record.token_count == 5
    fake_model.tokenizer.assert_called_once_with(
        chunk.text, add_special_tokens=True, truncation=False, padding=False
    )


def test_late_over_limit_chunk_prevents_any_encoding(fake_model: Mock) -> None:
    fake_model.max_seq_length = 5
    private_text = "synthetic text that exceeds five tokens"
    chunks = [_chunk(1), _chunk(2, private_text)]

    with pytest.raises(embeddings.EmbeddingTokenLimitError, match="chunk-2 has 8 tokens.*limit of 5") as error:
        embeddings.embed_chunks(chunks, batch_size=1)

    assert "Re-chunk" in str(error.value) and private_text not in str(error.value)
    assert fake_model.tokenizer.call_args.kwargs["truncation"] is False
    fake_model.encode.assert_not_called()


@pytest.mark.parametrize("size", [0, -1, True, False, 1.5, "2"])
def test_invalid_batch_size_fails_before_model_load(size: int) -> None:
    with patch.object(embeddings, "load_embedding_model") as load:
        with pytest.raises((TypeError, ValueError), match="batch_size"):
            embeddings.embed_chunks([_chunk(1)], batch_size=size)
    load.assert_not_called()


@pytest.mark.parametrize(
    ("chunks", "message"),
    [
        ((_chunk(1),), "list"),
        (["not a chunk"], "DocumentChunk"),
        ([_chunk(1, ""), _chunk(2)], "nonempty text"),
        ([_chunk(1, " \t\n ")], "nonempty text"),
        ([_chunk(1), _chunk(1)], "Duplicate chunk ID"),
        ([replace(_chunk(1), source_file=r"C:\Private\sample.pdf")], "source filename"),
        ([replace(_chunk(1), page_number=0)], "positive integers"),
    ],
)
def test_rejects_invalid_chunks_before_model_load(chunks: list[DocumentChunk], message: str) -> None:
    with patch.object(embeddings, "load_embedding_model") as load:
        with pytest.raises((TypeError, ValueError), match=message):
            embeddings.embed_chunks(chunks)
    load.assert_not_called()


@pytest.fixture
def model_constructor() -> Mock:
    """Stub the lazy import and isolate the process cache for each loader test."""
    package = ModuleType("sentence_transformers")
    constructor = Mock(return_value=Mock())
    package.SentenceTransformer = constructor
    embeddings._cached_model.cache_clear()
    with patch.dict(sys.modules, {package.__name__: package}):
        try:
            yield constructor
        finally:
            embeddings._cached_model.cache_clear()


def test_loader_reuses_model_and_uses_configured_identifier(model_constructor: Mock) -> None:
    assert embeddings.load_embedding_model() is embeddings.load_embedding_model()
    model_constructor.assert_called_once_with(config.EMBEDDING_MODEL_ID)


def test_model_load_failure_is_chained_and_a_retry_can_succeed(model_constructor: Mock) -> None:
    cause = OSError("synthetic missing weights")
    model_constructor.side_effect = [cause, model_constructor.return_value]

    with pytest.raises(embeddings.EmbeddingError, match="Could not load embedding model") as error:
        embeddings.load_embedding_model()

    assert error.value.__cause__ is cause
    assert embeddings.load_embedding_model() is model_constructor.return_value


def test_reports_tokenizer_failure_without_exposing_chunk_text(fake_model: Mock) -> None:
    cause = RuntimeError("synthetic tokenizer failure")
    fake_model.tokenizer.side_effect = cause
    chunk = _chunk(1, "synthetic document content")

    with pytest.raises(embeddings.EmbeddingError, match="Tokenizer failed for chunk chunk-1") as error:
        embeddings.embed_chunks([chunk])

    assert error.value.__cause__ is cause and chunk.text not in str(error.value)
    fake_model.encode.assert_not_called()


@pytest.mark.parametrize("invalid", [{"input_ids": [[1, 2]]}, {"input_ids": []}])
def test_rejects_incompatible_tokenizer_output(fake_model: Mock, invalid: dict) -> None:
    fake_model.tokenizer.side_effect = None
    fake_model.tokenizer.return_value = invalid
    with pytest.raises(embeddings.EmbeddingError, match="invalid token IDs"):
        embeddings.embed_chunks([_chunk(1)])
    fake_model.encode.assert_not_called()


@pytest.mark.parametrize(("setting", "value"), [("tokenizer", None), ("max_seq_length", 0)])
def test_rejects_unavailable_tokenizer_or_limit(fake_model: Mock, setting: str, value: object) -> None:
    setattr(fake_model, setting, value)
    with pytest.raises(embeddings.EmbeddingError, match="tokenizer or maximum sequence length"):
        embeddings.embed_chunks([_chunk(1)])
    fake_model.encode.assert_not_called()


def test_embedding_failure_is_chained(fake_model: Mock) -> None:
    cause = RuntimeError("synthetic out-of-memory error")
    fake_model.encode.side_effect = cause

    with pytest.raises(embeddings.EmbeddingError, match="Embedding failed") as error:
        embeddings.embed_chunks([_chunk(1)])

    assert error.value.__cause__ is cause


@pytest.mark.parametrize(
    ("output", "message"),
    [
        ([], "Expected 1 embedding vectors, got 0"),
        ([[0.0] * 384] * 2, "Expected 1 embedding vectors, got 2"),
        ([[]], "dimension 384, got 0"),
        ([[0.0] * 383], "dimension 384, got 383"),
        ([[0.0] * 383 + [float("nan")]], "finite"),
        ([[0.0] * 383 + [float("inf")]], "finite"),
        ([[0.0] * 383 + ["0"]], "numeric"),
        ([[0.0] * 383 + [True]], "numeric"),
        ([0.0] * 384, "vectors, got 384"),
        (None, "Unexpected embedding output type"),
        ({"vector": [0.0] * 384}, "Unexpected embedding output type"),
        (["0" * 384], "Unexpected embedding output type"),
    ],
)
def test_rejects_invalid_embedding_output(fake_model: Mock, output: object, message: str) -> None:
    fake_model.encode.side_effect = None
    fake_model.encode.return_value = output

    with pytest.raises(embeddings.EmbeddingError, match=message):
        embeddings.embed_chunks([_chunk(1)])
