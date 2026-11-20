"""Deterministic retrieval metrics with no filesystem or model dependencies."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from statistics import mean, median
from typing import Protocol

HIT_DEPTHS = (1, 3, 5)
_DASHES = str.maketrans({"\u2013": "-", "\u2014": "-"})
_SEPARATORS = str.maketrans({character: " " for character in "():|"})
_CURRENCY = re.compile(r"(?<!\w)rs(?:\.(?=\W|\d|$)|(?![\w.]))|\u20b9")


class _Passage(Protocol):
    """The read-only view of a ranked passage needed by the scorer."""

    @property
    def source_file(self) -> str: ...

    @property
    def page_number(self) -> int: ...

    @property
    def text(self) -> str: ...

    @property
    def similarity(self) -> float | None: ...


@dataclass(frozen=True)
class RetrievalScore:
    """Ranks, evidence matches, and top similarity for one answerable question."""

    file_rank: int | None
    page_rank: int | None
    reciprocal_rank: float | None
    must_contain_fraction: float
    must_contain_all: bool
    found_terms: tuple[str, ...]
    missing_terms: tuple[str, ...]
    top1_similarity: float | None
    result_count: int


@dataclass(frozen=True)
class RetrievalSummary:
    """Raw hit counts by HIT_DEPTHS and aggregate evidence and ranking metrics."""

    question_count: int
    labelled_page_count: int
    no_result_count: int
    file_hits: tuple[int, ...]
    page_hits: tuple[int, ...]
    mrr: float | None
    all_terms_count: int
    average_term_fraction: float | None
    similarity_min: float | None
    similarity_median: float | None
    similarity_max: float | None

    @property
    def missing_page_count(self) -> int:
        """Number of questions excluded from page metrics."""
        return self.question_count - self.labelled_page_count


def normalize_text(value: str) -> str:
    """Normalize evidence formatting without changing decimal or comma separators."""
    if not isinstance(value, str):
        raise TypeError("Text to normalize must be a string.")
    text = _CURRENCY.sub(" ", value.casefold().translate(_DASHES)).translate(_SEPARATORS)
    return " ".join(re.sub(r"\s*-\s*", "-", text).split())


def parse_pages(value: str) -> set[int]:
    """Read semicolon-separated positive pages, or an absent label as an empty set."""
    if not isinstance(value, str):
        raise TypeError("Page labels must be a string.")
    if not value.strip():
        return set()
    pages = set()
    for part in value.split(";"):
        token = part.strip()
        if not token.isdecimal() or int(token) < 1:
            raise ValueError(f"Invalid page label {token!r}. Expected a positive integer.")
        pages.add(int(token))
    return pages


def parse_terms(value: str) -> list[str]:
    """Split required evidence on semicolons while retaining the original terms."""
    if not isinstance(value, str):
        raise TypeError("Required terms must be a string.")
    return [term for part in value.split(";") if (term := part.strip())]


def score_retrieval(
    results: Sequence[_Passage], expected_file: str, accepted_pages: set[int], terms: list[str]
) -> RetrievalScore:
    """Score supplied retrieval order, using only the first five chunks for evidence."""
    if not isinstance(expected_file, str) or not expected_file.strip():
        raise ValueError("An expected filename is required.")
    if not isinstance(accepted_pages, set) or any(type(p) is not int or p < 1 for p in accepted_pages):
        raise ValueError("Accepted pages must be a set of positive integers.")
    normalized_terms = [normalize_text(term) for term in terms]
    if not normalized_terms or any(not term for term in normalized_terms):
        raise ValueError("Every required evidence term must have usable text.")

    file_rank = page_rank = None
    for rank, result in enumerate(results, start=1):
        if result.source_file == expected_file:
            if file_rank is None:
                file_rank = rank
            if result.page_number in accepted_pages:
                page_rank = rank
                break

    passages = [normalize_text(result.text) for result in results[:5]]
    found: list[str] = []
    missing: list[str] = []
    for original, normalized in zip(terms, normalized_terms, strict=True):
        target = found if _found_in_passage(normalized, passages) else missing
        target.append(original)

    similarity = results[0].similarity if results else None
    if similarity is not None:
        if not isinstance(similarity, Real) or isinstance(similarity, bool) or not isfinite(similarity):
            raise ValueError("Top-1 similarity must be finite or unavailable.")
        similarity = float(similarity)
    return RetrievalScore(
        file_rank=file_rank, page_rank=page_rank,
        reciprocal_rank=(1 / page_rank if page_rank is not None else 0.0) if accepted_pages else None,
        must_contain_fraction=len(found) / len(terms), must_contain_all=not missing,
        found_terms=tuple(found), missing_terms=tuple(missing), top1_similarity=similarity,
        result_count=len(results),
    )


def _found_in_passage(term: str, passages: list[str]) -> bool:
    """Match within one passage, excluding partial numeric values at term edges."""
    numeric_start = term[0].isdigit() or re.match(r"[-\u2212]\d", term)
    left = r"(?<![\d\u2212-])(?<!\d[.,])" if numeric_start else ""
    right = r"(?!\d|[.,]\d)" if term[-1].isdigit() else ""
    expression = re.compile(left + re.escape(term) + right)
    return any(expression.search(passage) is not None for passage in passages)


def aggregate_retrieval_scores(scores: Sequence[RetrievalScore]) -> RetrievalSummary:
    """Summarize answerable questions, excluding absent labels and similarities."""
    reciprocals = [score.reciprocal_rank for score in scores if score.reciprocal_rank is not None]
    similarities = [score.top1_similarity for score in scores if score.top1_similarity is not None]
    return RetrievalSummary(
        question_count=len(scores), labelled_page_count=len(reciprocals),
        no_result_count=sum(score.result_count == 0 for score in scores),
        file_hits=tuple(sum(s.file_rank is not None and s.file_rank <= k for s in scores) for k in HIT_DEPTHS),
        page_hits=tuple(sum(s.page_rank is not None and s.page_rank <= k for s in scores) for k in HIT_DEPTHS),
        mrr=mean(reciprocals) if reciprocals else None,
        all_terms_count=sum(score.must_contain_all for score in scores),
        average_term_fraction=mean(score.must_contain_fraction for score in scores) if scores else None,
        similarity_min=min(similarities) if similarities else None,
        similarity_median=median(similarities) if similarities else None,
        similarity_max=max(similarities) if similarities else None,
    )
