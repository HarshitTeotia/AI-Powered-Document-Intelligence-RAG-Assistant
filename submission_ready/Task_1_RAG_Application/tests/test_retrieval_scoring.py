"""Check pure retrieval metrics using only synthetic passages."""

from dataclasses import dataclass

import pytest

from distribuai.evaluation.scoring import (
    aggregate_retrieval_scores, normalize_text, parse_pages, parse_terms, score_retrieval,
)


@dataclass(frozen=True)
class Passage:
    source_file: str = "Policy.pdf"
    page_number: int = 1
    text: str = "Return items within 7 calendar days."
    similarity: float | None = 0.7


@pytest.mark.parametrize(("text", "expected"), [
    ("  Stra\u00dfe\n\tCONDITIONS   apply  ", "strasse conditions apply"),
    ("1-2 business days", "1-2 business days"),
    ("1 \u2013 2 business days", "1-2 business days"),
    ("1\u20142 business days", "1-2 business days"),
    ("price (rs.): 1,500.50", "price 1,500.50"),
    ("Shipping charge (Rs.): 450.00", "shipping charge 450.00"),
    ("shipping cost: Rs. 450.00", "shipping cost 450.00"),
    ("Rs.1,500.50", "1,500.50"),
    ("\u20b9450.00", "450.00"),
    ("A|B(C):D", "a b c d"),
    ("Cars, shirts, and Warsaw.", "cars, shirts, and warsaw."),
    ("Rs. 12,345.60 | FD-RIC-101 | 2026-11-20", "12,345.60 fd-ric-101 2026-11-20"),
    ("-450.00", "-450.00"),
    (" \t ", ""),
])
def test_normalize_text(text: str, expected: str) -> None:
    assert normalize_text(text) == expected
    assert normalize_text(expected) == expected


@pytest.mark.parametrize(("term", "text"), [
    ("Rs. 1,500.50", "price (rs.): 1,500.50"),
    ("Rs 450.00", "Shipping charge (Rs.): 450.00"),
    ("\u20b9 450.00", "shipping cost: Rs. 450.00"),
    ("1-2 business days", "Ships in 1 \u2013 2 BUSINESS days"),
])
def test_evidence_normalization_matches_text_in_one_passage(term: str, text: str) -> None:
    score = score_retrieval([Passage(text=text)], "Policy.pdf", {1}, [term])
    assert score.found_terms == (term,) and not score.missing_terms
    assert score.must_contain_all and score.must_contain_fraction == 1.0


@pytest.mark.parametrize("text", ["Rs. 1,500.05", "11,500.50", "91,500.50", "1,500.501", "-1,500.50", "\u22121,500.50"])
def test_different_amounts_do_not_match(text: str) -> None:
    score = score_retrieval([Passage(text=text)], "Policy.pdf", {1}, ["Rs. 1,500.50"])
    assert not score.must_contain_all and score.must_contain_fraction == 0


@pytest.mark.parametrize(("label", "expected"), [("", set()), ("   ", set()), ("2", {2}), ("1;5", {1, 5}), ("1; 2 ; 5;1", {1, 2, 5})])
def test_parse_pages(label: str, expected: set[int]) -> None:
    assert parse_pages(label) == expected


@pytest.mark.parametrize("label", ["0", "-1", "1.5", "page 3", "1;bad;2", "1;;5", "1_000"])
def test_invalid_page_label_raises(label: str) -> None:
    with pytest.raises(ValueError, match="page label"):
        parse_pages(label)


@pytest.mark.parametrize(("terms", "expected"), [
    ("7 calendar days", ["7 calendar days"]),
    (" Rs. 450.00 ; ; Rs. 15,000.00; ", ["Rs. 450.00", "Rs. 15,000.00"]),
    (";;", []),
    ("", []),
])
def test_parse_terms_retains_order_and_punctuation(terms: str, expected: list[str]) -> None:
    assert parse_terms(terms) == expected


@pytest.mark.parametrize("function", [normalize_text, parse_pages, parse_terms])
def test_string_input_required(function) -> None:
    with pytest.raises(TypeError, match="string"):
        function(None)


@pytest.mark.parametrize("rank", [1, 3, 5, None])
def test_first_file_rank_uses_supplied_order(rank: int | None) -> None:
    results = [Passage(source_file="Other.pdf")] * 5
    if rank is not None:
        results[rank - 1] = Passage()
    assert score_retrieval(results, "Policy.pdf", set(), ["7 days"]).file_rank == rank


@pytest.mark.parametrize(("rank", "rr"), [(1, 1.0), (2, 0.5), (5, 0.2)])
def test_page_rank_requires_the_right_file_and_an_accepted_page(rank: int, rr: float) -> None:
    results = [Passage(source_file="Other.pdf", page_number=3)] * (rank - 1) + [
        Passage(page_number=3), Passage(page_number=5),
    ]
    score = score_retrieval(results, "Policy.pdf", {3, 5}, ["7 calendar days"])
    assert score.file_rank == score.page_rank == rank
    assert score.reciprocal_rank == rr


def test_missing_page_labels_and_labelled_miss_differ() -> None:
    results = [Passage(), Passage(source_file="Other.pdf", page_number=2)]
    unlabelled = score_retrieval(results, "Policy.pdf", set(), ["7 calendar days"])
    labelled = score_retrieval(results, "Policy.pdf", {2}, ["7 calendar days"])
    assert unlabelled.file_rank == labelled.file_rank == 1
    assert unlabelled.page_rank is labelled.page_rank is None
    assert unlabelled.reciprocal_rank is None and labelled.reciprocal_rank == 0.0


def test_terms_use_top_five_and_never_cross_chunk_boundaries() -> None:
    results = [Passage(text=text) for text in ("refund", "window", "8 days", "policy", "450.00", "in stock")]
    terms = ["refund window", "Rs. 450.00", "in stock", "8 days"]
    original = results.copy()
    score = score_retrieval(results, "Policy.pdf", {1}, terms)

    assert score.found_terms == ("Rs. 450.00", "8 days")
    assert score.missing_terms == ("refund window", "in stock")
    assert score.must_contain_fraction == 0.5 and not score.must_contain_all
    assert results == original and terms == ["refund window", "Rs. 450.00", "in stock", "8 days"]


def test_top_similarity_and_empty_result_behavior() -> None:
    score = score_retrieval([Passage(similarity=-0.3)], "Policy.pdf", {1}, ["7 calendar days"])
    assert score.top1_similarity == -0.3
    empty = score_retrieval([], "Policy.pdf", {1}, ["7 calendar days"])
    assert empty.top1_similarity is empty.page_rank is empty.file_rank is None
    assert empty.reciprocal_rank == empty.must_contain_fraction == 0
    assert empty.result_count == 0 and not empty.must_contain_all


@pytest.mark.parametrize("terms", [[], [" "], ["Rs."]])
def test_unusable_terms_are_rejected(terms: list[str]) -> None:
    with pytest.raises(ValueError, match="usable text"):
        score_retrieval([], "Policy.pdf", set(), terms)


def test_summary_uses_raw_counts_and_excludes_unlabelled_pages() -> None:
    wrong = Passage(source_file="Other.pdf", similarity=0.4, text="Nothing matches.")
    scores = [
        score_retrieval([Passage(similarity=0.8)], "Policy.pdf", {1}, ["7 calendar days"]),
        score_retrieval([wrong, wrong, Passage()], "Policy.pdf", {1}, ["7 calendar days", "refund"]),
        score_retrieval([wrong] * 4 + [Passage()], "Policy.pdf", set(), ["7 calendar days"]),
        score_retrieval([], "Policy.pdf", {1}, ["7 calendar days"]),
    ]
    summary = aggregate_retrieval_scores(scores)
    assert (summary.question_count, summary.labelled_page_count, summary.missing_page_count) == (4, 3, 1)
    assert summary.file_hits == (1, 2, 3) and summary.page_hits == (1, 2, 2)
    assert summary.mrr == pytest.approx((1 + 1 / 3) / 3)
    assert summary.all_terms_count == 2 and summary.average_term_fraction == 0.625
    assert (summary.similarity_min, summary.similarity_median, summary.similarity_max) == (0.4, 0.4, 0.8)
    assert summary.no_result_count == 1


def test_summary_empty_and_unavailable_values_are_not_fabricated() -> None:
    empty = aggregate_retrieval_scores([])
    assert empty.question_count == empty.labelled_page_count == empty.no_result_count == 0
    assert empty.mrr is empty.average_term_fraction is empty.similarity_min is None
    summary = aggregate_retrieval_scores([
        score_retrieval([Passage(similarity=None)], "Policy.pdf", set(), ["7 calendar days"]),
    ])
    assert summary.mrr is summary.similarity_median is summary.similarity_max is None
    assert summary.missing_page_count == 1 and summary.no_result_count == 0
