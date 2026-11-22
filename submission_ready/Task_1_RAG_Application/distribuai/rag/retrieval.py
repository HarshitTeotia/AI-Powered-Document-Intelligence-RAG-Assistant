"""Query the local cosine collection and return ranked source passages."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Protocol

from distribuai.rag.chunking import _safe_filename
from distribuai.rag.embeddings import _clean_question, embed_query


class _QueryCollection(Protocol):
    """The subset of ChromaDB's collection API used for retrieval."""

    def count(self) -> int: ...

    def query(
        self, *, query_embeddings: list[list[float]], n_results: int,
        where: dict[str, str] | None, include: list[str],
    ) -> object: ...


class RetrievalError(Exception):
    """The collection could not be queried or returned malformed results."""


@dataclass(frozen=True)
class RetrievedChunk:
    """A source passage ranked by increasing cosine distance from the question."""

    chunk_id: str
    source_file: str
    page_number: int
    chunk_index: int
    text: str
    distance: float
    similarity: float
    rank: int


def retrieve(
    collection: _QueryCollection,
    question: str,
    top_k: int = 5,
    source_file: str | None = None,
) -> list[RetrievedChunk]:
    """Retrieve up to top_k passages, optionally from one exact source filename.

    ChromaDB's order is preserved. Lower cosine distance indicates a closer
    match, and similarity is 1 minus distance. Empty results are valid.
    """
    text = _clean_question(question)
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise TypeError("top_k must be an integer.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if source_file is not None:
        if not isinstance(source_file, str):
            raise TypeError("source_file must be a filename or None.")
        try:
            if _safe_filename(source_file) != source_file:
                raise ValueError("Expected a filename without directories or surrounding whitespace.")
        except ValueError as exc:
            raise ValueError("source_file must be a valid filename without directories.") from exc
    if not callable(getattr(collection, "count", None)) or not callable(getattr(collection, "query", None)):
        raise TypeError("collection must provide count() and query() operations.")

    try:
        count = collection.count()
    except Exception as exc:
        raise RetrievalError("Could not count the records in the collection.") from exc
    if type(count) is not int or count < 0:
        raise RetrievalError("The collection returned an invalid record count.")
    if count == 0:
        return []

    vector = embed_query(text)
    limit = min(top_k, count)
    try:
        response = collection.query(
            query_embeddings=[list(vector)], n_results=limit,
            where=None if source_file is None else {"source_file": source_file},
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        raise RetrievalError("ChromaDB query failed.") from exc
    return _parse_results(response, limit)


def _parse_results(response: object, limit: int) -> list[RetrievedChunk]:
    """Read a single ChromaDB result group and validate every citation."""
    if not isinstance(response, Mapping):
        raise RetrievalError("ChromaDB returned a malformed query response.")
    ids, documents, metadatas, distances = (
        _result_group(response, field) for field in ("ids", "documents", "metadatas", "distances")
    )
    if not len(ids) == len(documents) == len(metadatas) == len(distances) <= limit:
        raise RetrievalError("ChromaDB returned inconsistent result lengths.")

    seen: set[str] = set()
    results: list[RetrievedChunk] = []
    for rank, (chunk_id, text, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances, strict=True), start=1
    ):
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise RetrievalError(f"Missing or invalid chunk ID at rank {rank}.")
        if chunk_id in seen:
            raise RetrievalError(f"Duplicate chunk ID at rank {rank}.")
        if not isinstance(text, str):
            raise RetrievalError(f"Missing or invalid document at rank {rank}.")
        filename, page, index = _citation(metadata, rank)
        numeric_distance = _cosine_distance(distance, rank)
        results.append(RetrievedChunk(
            chunk_id, filename, page, index, text, numeric_distance, 1.0 - numeric_distance, rank,
        ))
        seen.add(chunk_id)
    return results


def _result_group(response: Mapping[str, object], field: str) -> list[object]:
    """Get the one nested list expected for a single query embedding."""
    groups = response.get(field)
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(groups[0], list):
        raise RetrievalError(f"Missing or malformed {field} in the ChromaDB response.")
    return groups[0]


def _citation(metadata: object, rank: int) -> tuple[str, int, int]:
    """Require storage's filename and positive page and chunk coordinates."""
    if not isinstance(metadata, Mapping):
        raise RetrievalError(f"Missing citation metadata at rank {rank}.")
    filename = metadata.get("source_file")
    try:
        if _safe_filename(filename) != filename:
            raise ValueError("Expected a filename without directories.")
    except (TypeError, ValueError) as exc:
        raise RetrievalError(f"Invalid source filename in citation metadata at rank {rank}.") from exc
    page, index = metadata.get("page_number"), metadata.get("chunk_index")
    if type(page) is not int or page <= 0 or type(index) is not int or index <= 0:
        raise RetrievalError(f"Invalid page number or chunk index in citation metadata at rank {rank}.")
    return filename, page, index


def _cosine_distance(value: object, rank: int) -> float:
    """Convert a finite numeric distance without rescaling or clamping it."""
    if not isinstance(value, Real) or isinstance(value, bool):
        raise RetrievalError(f"Invalid cosine distance at rank {rank}.")
    try:
        distance = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RetrievalError(f"Invalid cosine distance at rank {rank}.") from exc
    if not isfinite(distance):
        raise RetrievalError(f"Non-finite cosine distance at rank {rank}.")
    return distance
