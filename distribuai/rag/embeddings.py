"""Generate local embeddings for document chunks without changing their citations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Set
from dataclasses import dataclass
from functools import cache
from math import isfinite
from numbers import Real
from pathlib import PureWindowsPath
from threading import Lock
from typing import TYPE_CHECKING

from distribuai import config
from distribuai.rag.chunking import DocumentChunk

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

EMBEDDING_DIMENSION = 384
_MODEL_LOCK = Lock()


class EmbeddingError(Exception):
    """The embedding model, tokenizer, or output could not be used."""


class EmbeddingTokenLimitError(EmbeddingError):
    """A document chunk exceeds the model's maximum sequence length."""


@dataclass(frozen=True)
class EmbeddedChunk:
    """A chunk's original text and citation metadata with its local embedding."""

    chunk_id: str
    source_file: str
    page_number: int
    chunk_index: int
    text: str
    vector: tuple[float, ...]
    token_count: int


def load_embedding_model() -> SentenceTransformer:
    """Load the configured model on demand and reuse it within this process."""
    with _MODEL_LOCK:
        return _cached_model(config.EMBEDDING_MODEL_ID)


@cache
def _cached_model(model_id: str) -> SentenceTransformer:
    """Cache successful loads and leave model files in the normal user cache."""
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_id)
    except Exception as exc:
        raise EmbeddingError(
            f"Could not load embedding model {model_id}. Check the local model cache "
            "or internet access for the initial download."
        ) from exc


def embed_chunks(chunks: list[DocumentChunk], batch_size: int = 16) -> list[EmbeddedChunk]:
    """Validate all token counts, then embed chunks in batches in input order."""
    if not isinstance(chunks, list):
        raise TypeError("chunks must be a list of DocumentChunk objects.")
    if not chunks:
        return []
    if not isinstance(batch_size, int) or isinstance(batch_size, bool):
        raise TypeError("batch_size must be an integer.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    _validate_chunks(chunks)

    model = load_embedding_model()
    counts = _count_tokens(model, chunks)
    records: list[EmbeddedChunk] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        try:
            output = model.encode(
                [chunk.text for chunk in batch],
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as exc:
            raise EmbeddingError(f"Embedding failed for the batch starting at chunk {start + 1}.") from exc

        vectors = _validate_vectors(output, len(batch))
        for offset, (chunk, vector) in enumerate(zip(batch, vectors, strict=True)):
            records.append(EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                source_file=chunk.source_file,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                vector=vector,
                token_count=counts[start + offset],
            ))
    return records


def _validate_chunks(chunks: list[DocumentChunk]) -> None:
    """Validate chunk records and citation metadata before loading the model."""
    seen: set[str] = set()
    for chunk in chunks:
        if not isinstance(chunk, DocumentChunk):
            raise TypeError("chunks must contain only DocumentChunk objects.")
        if not isinstance(chunk.chunk_id, str) or not chunk.chunk_id.strip():
            raise ValueError("Each chunk must have a nonempty string ID.")
        if chunk.chunk_id in seen:
            raise ValueError(f"Duplicate chunk ID: {chunk.chunk_id}.")
        if not isinstance(chunk.text, str) or not chunk.text.strip():
            raise ValueError(f"Chunk {chunk.chunk_id} must have nonempty text.")
        source = chunk.source_file
        if (not isinstance(source, str) or not source.strip()
                or source in {".", ".."} or PureWindowsPath(source).name != source):
            raise ValueError("Each chunk must have a source filename without directories.")
        for value in (chunk.page_number, chunk.chunk_index):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError("Chunk page numbers and indexes must be positive integers.")
        seen.add(chunk.chunk_id)


def _count_tokens(model: SentenceTransformer, chunks: list[DocumentChunk]) -> list[int]:
    """Count full token sequences, including the model's special tokens."""
    try:
        tokenizer = model.tokenizer
        limit = model.max_seq_length
    except (AttributeError, LookupError, RuntimeError, TypeError) as exc:
        raise EmbeddingError("Could not access the model tokenizer or sequence limit.") from exc
    if not callable(tokenizer) or type(limit) is not int or limit <= 0:
        raise EmbeddingError("The model tokenizer or maximum sequence length is unavailable.")

    counts: list[int] = []
    for chunk in chunks:
        try:
            ids = tokenizer(
                chunk.text, add_special_tokens=True, truncation=False, padding=False
            )["input_ids"]
        except Exception as exc:
            raise EmbeddingError(f"Tokenizer failed for chunk {chunk.chunk_id}.") from exc
        if not isinstance(ids, (list, tuple)) or not ids or any(type(token) is not int for token in ids):
            raise EmbeddingError(f"Tokenizer returned invalid token IDs for chunk {chunk.chunk_id}.")
        count = len(ids)
        if count > limit:
            raise EmbeddingTokenLimitError(
                f"Chunk {chunk.chunk_id} has {count} tokens, exceeding the model limit of {limit}. "
                "Re-chunk the document with a smaller character limit."
            )
        counts.append(count)
    return counts


def _validate_vectors(output: object, expected_count: int) -> list[tuple[float, ...]]:
    """Check model output and copy each row into a finite, immutable tuple."""
    rows = _ordered_items(output)
    if len(rows) != expected_count:
        raise EmbeddingError(f"Expected {expected_count} embedding vectors, got {len(rows)}.")

    vectors: list[tuple[float, ...]] = []
    for row in rows:
        values = _ordered_items(row)
        if len(values) != EMBEDDING_DIMENSION:
            raise EmbeddingError(
                f"Expected embedding dimension {EMBEDDING_DIMENSION}, got {len(values)}."
            )
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
            raise EmbeddingError("Embedding vectors must contain numeric values.")
        try:
            vector = tuple(float(value) for value in values)
        except (TypeError, ValueError, OverflowError) as exc:
            raise EmbeddingError("Could not convert embedding values to finite floats.") from exc
        if not all(isfinite(value) for value in vector):
            raise EmbeddingError("Embedding vectors must contain only finite values.")
        vectors.append(vector)
    return vectors


def _ordered_items(value: object) -> tuple[object, ...]:
    """Accept ordered array-like output without relying on NumPy types."""
    if isinstance(value, (str, bytes, bytearray, Mapping, Set)) or not isinstance(value, Iterable):
        raise EmbeddingError("Unexpected embedding output type. Expected an ordered array.")
    try:
        return tuple(value)
    except TypeError as exc:
        raise EmbeddingError("Unexpected embedding output type. Expected an ordered array.") from exc
