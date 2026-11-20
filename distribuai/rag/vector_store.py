"""Persist and manage precomputed document embeddings in local ChromaDB."""

from collections.abc import Iterator, Mapping, Sequence
from numbers import Real
from pathlib import Path, PureWindowsPath
from typing import override

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
from chromadb.telemetry.product import ProductTelemetryClient, ProductTelemetryEvent

from distribuai import config
from distribuai.rag.chunking import _safe_filename
from distribuai.rag.embeddings import EMBEDDING_DIMENSION, EmbeddedChunk

COLLECTION_NAME = "distribuai_documents"
_MAX_FLOAT32 = 3.4028234663852886e38
_DOCUMENT_BATCH_SIZE = 256


class VectorStoreError(Exception):
    """A local vector store operation failed or returned invalid data."""


class _DisabledProductTelemetry(ProductTelemetryClient):
    """Discard events that ChromaDB 1.0.9 still dispatches when opted out."""

    @override
    def capture(self, event: ProductTelemetryEvent) -> None:
        """Discard the product telemetry event."""


def open_vector_collection(storage_dir: Path | None = None) -> Collection:
    """Open or create the local document collection with compatible settings."""
    directory = config.CHROMA_STORAGE_DIR if storage_dir is None else storage_dir
    if not isinstance(directory, Path):
        raise TypeError("storage_dir must be a pathlib.Path.")

    try:
        directory = directory.resolve()
        directory.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(directory),
            settings=Settings(
                chroma_api_impl="chromadb.api.rust.RustBindingsAPI",
                anonymized_telemetry=False,
                chroma_product_telemetry_impl=f"{__name__}._DisabledProductTelemetry",
                allow_reset=False,
            ),
        )
    except Exception as exc:
        raise VectorStoreError(
            "Could not initialize local ChromaDB. Check that the storage path is a writable directory."
        ) from exc

    try:
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={
                "embedding_model": config.EMBEDDING_MODEL_ID,
                "embedding_dimension": EMBEDDING_DIMENSION,
            },
            embedding_function=None,
        )
    except Exception as exc:
        raise VectorStoreError("Could not open or create the DistribuAI collection.") from exc

    _validate_collection(collection)
    try:
        stored = collection.get(limit=1, include=["embeddings"])["embeddings"]
    except Exception as exc:
        raise VectorStoreError("Could not read the collection to verify its vector dimension.") from exc
    if stored is not None and len(stored) and len(stored[0]) != EMBEDDING_DIMENSION:
        raise VectorStoreError(f"Incompatible stored vector dimension. Expected {EMBEDDING_DIMENSION}.")
    return collection


def store_embedded_chunks(
    collection: Collection, records: list[EmbeddedChunk], batch_size: int = 64
) -> int:
    """Upsert validated chunks by ID and return the number of records written.

    Each batch is a separate write. A later failure can leave earlier batches
    stored, and the same IDs can be retried safely.
    """
    if not isinstance(records, list):
        raise TypeError("records must be a list of EmbeddedChunk objects.")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool):
        raise TypeError("batch_size must be an integer.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not records:
        return 0

    _validate_records(records)
    _validate_collection(collection)
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        try:
            collection.upsert(
                ids=[record.chunk_id for record in batch],
                documents=[record.text for record in batch],
                embeddings=[[float(value) for value in record.vector] for record in batch],
                metadatas=[
                    {
                        "source_file": record.source_file,
                        "page_number": record.page_number,
                        "chunk_index": record.chunk_index,
                        "token_count": record.token_count,
                    }
                    for record in batch
                ],
            )
        except Exception as exc:
            raise VectorStoreError(
                f"ChromaDB upsert failed in batch {start // batch_size + 1} after "
                f"{start} records were written. Earlier batches may remain stored."
            ) from exc
    return len(records)


def _validate_collection(collection: Collection) -> None:
    """Require the expected model, dimension, index space, and direct vectors."""
    metadata = collection.metadata or {}
    dimension = metadata.get("embedding_dimension")
    if (
        metadata.get("embedding_model") != config.EMBEDDING_MODEL_ID
        or type(dimension) is not int
        or dimension != EMBEDDING_DIMENSION
    ):
        raise VectorStoreError(
            f"Incompatible collection metadata. Expected model {config.EMBEDDING_MODEL_ID} "
            f"and embedding dimension {EMBEDDING_DIMENSION}."
        )
    configuration = collection.configuration_json
    if (configuration.get("hnsw") or {}).get("space") != "cosine":
        raise VectorStoreError("Incompatible collection index. Expected cosine distance.")
    # In 1.0.9 an explicit None is serialized as this legacy marker.
    if configuration.get("embedding_function") not in (None, {"type": "legacy"}):
        raise VectorStoreError("Incompatible collection. Expected supplied embeddings only.")


def _validate_records(records: list[EmbeddedChunk]) -> None:
    """Validate all records before any upsert, without copying their vectors."""
    seen: set[str] = set()
    for position, record in enumerate(records, start=1):
        if not isinstance(record, EmbeddedChunk):
            raise TypeError(f"Record {position} must be an EmbeddedChunk.")
        if not isinstance(record.chunk_id, str) or not record.chunk_id.strip():
            raise ValueError(f"Record {position} must have a nonempty string chunk ID.")
        if record.chunk_id in seen:
            raise ValueError(f"Duplicate chunk ID at record {position}.")
        seen.add(record.chunk_id)

        source = record.source_file
        if (
            not isinstance(source, str)
            or not source.strip()
            or source.endswith((" ", "."))
            or PureWindowsPath(source).name != source
            or PureWindowsPath(source).is_reserved()
            or any(ord(char) < 32 or char in '<>:"|?*' for char in source)
        ):
            raise ValueError(f"Record {position} must have a valid source filename without directories.")
        for field, value in (
            ("page_number", record.page_number),
            ("chunk_index", record.chunk_index),
            ("token_count", record.token_count),
        ):
            if type(value) is not int or not 0 < value < 2**63:
                raise ValueError(f"Record {position} {field} must be a positive 64-bit integer.")
        if not isinstance(record.text, str) or not record.text.strip():
            raise ValueError(f"Record {position} must have nonempty text.")
        if not isinstance(record.vector, tuple) or len(record.vector) != EMBEDDING_DIMENSION:
            raise ValueError(f"Record {position} must have a {EMBEDDING_DIMENSION}-dimensional vector.")
        for value in record.vector:
            if not isinstance(value, Real) or isinstance(value, bool):
                raise TypeError(f"Record {position} vector values must be numeric.")
            if not -_MAX_FLOAT32 <= value <= _MAX_FLOAT32:
                raise ValueError(f"Record {position} vector values must be finite float32 values.")


def list_documents(collection: Collection) -> list[tuple[str, int, list[int]]]:
    """List each filename, stored chunk count, and sorted distinct page numbers."""
    counts: dict[str, int] = {}
    pages: dict[str, set[int]] = {}
    for _, metadata in _scan_document_records(collection, include_metadata=True):
        if not isinstance(metadata, Mapping):
            raise VectorStoreError("A stored record is missing citation metadata.")
        try:
            filename = _require_source_filename(metadata.get("source_file"))
        except (TypeError, ValueError) as exc:
            raise VectorStoreError("A stored record has an invalid source filename.") from exc
        page = metadata.get("page_number")
        if type(page) is not int or page < 1:
            raise VectorStoreError("A stored record has an invalid page number.")
        counts[filename] = counts.get(filename, 0) + 1
        pages.setdefault(filename, set()).add(page)
    return [(name, counts[name], sorted(pages[name])) for name in sorted(counts)]


def delete_document(collection: Collection, source_file: str) -> int:
    """Delete only one filename's stored chunks, leaving the PDF file intact."""
    source_file = _require_source_filename(source_file)
    ids = sorted(_source_chunk_ids(collection, source_file))
    return _delete_source_chunks(collection, source_file, ids)


def _require_source_filename(value: object) -> str:
    """Accept a valid citation filename without normalizing away directories."""
    if not isinstance(value, str):
        raise TypeError("source filename must be a string.")
    if _safe_filename(value) != value:
        raise ValueError("source filename must be a plain filename without directories.")
    return value


def _source_chunk_ids(collection: Collection, source_file: str) -> set[str]:
    """Read IDs for one source without fetching text, metadata, or vectors."""
    return {chunk_id for chunk_id, _ in _scan_document_records(collection, source_file)}


def _scan_document_records(
    collection: Collection, source_file: str | None = None, *, include_metadata: bool = False
) -> Iterator[tuple[str, object]]:
    """Read a local record snapshot in pages without loading embeddings or text."""
    where = None if source_file is None else {"source_file": source_file}
    seen: set[str] = set()
    offset = 0
    while True:
        try:
            result = collection.get(
                where=where, limit=_DOCUMENT_BATCH_SIZE, offset=offset,
                include=["metadatas"] if include_metadata else [],
            )
        except Exception as exc:
            raise VectorStoreError("Could not read stored document records.") from exc
        if not isinstance(result, Mapping) or not isinstance(ids := result.get("ids"), list):
            raise VectorStoreError("ChromaDB returned invalid document record IDs.")
        metadata = result.get("metadatas") if include_metadata else [None] * len(ids)
        if not isinstance(metadata, list) or len(ids) != len(metadata):
            raise VectorStoreError("ChromaDB returned missing or incomplete citation metadata.")
        for chunk_id, item in zip(ids, metadata, strict=True):
            if not isinstance(chunk_id, str) or not chunk_id.strip() or chunk_id in seen:
                raise VectorStoreError("ChromaDB returned a missing or duplicate document record ID.")
            seen.add(chunk_id)
            yield chunk_id, item
        if len(ids) < _DOCUMENT_BATCH_SIZE:
            return
        offset += len(ids)


def _delete_source_chunks(collection: Collection, source_file: str, ids: Sequence[str]) -> int:
    """Delete selected IDs within a single source, in bounded batches."""
    for offset in range(0, len(ids), _DOCUMENT_BATCH_SIZE):
        try:
            collection.delete(
                ids=list(ids[offset:offset + _DOCUMENT_BATCH_SIZE]),
                where={"source_file": source_file},
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Could not delete stored chunks for {source_file}. "
                f"Earlier batches may have removed {offset} chunks."
            ) from exc
    return len(ids)
