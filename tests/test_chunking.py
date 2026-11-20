"""Exercise page-aware chunking with in-memory PDF page records."""

from dataclasses import FrozenInstanceError

import pytest

from distribuai import config
from distribuai.ingestion.pdf_extractor import PdfPage
from distribuai.rag.chunking import chunk_pdf_pages, clean_text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Ordinary text.", "Ordinary text."),
        (" \tPolicy\t   applies.  ", "Policy applies."),
        ("First\r\nSecond\rThird", "First\nSecond\nThird"),
        ("Paragraph one\n \t\n\nParagraph two", "Paragraph one\n\nParagraph two"),
        ("", ""),
        (" \r\n\t \n", ""),
        (
            "  INV-302/7\tRs. 1,500.50 due on 20/11/2026.\r\nTotal: $42.75. ",
            "INV-302/7 Rs. 1,500.50 due on 20/11/2026.\nTotal: $42.75.",
        ),
    ],
)
def test_clean_text(raw: str, expected: str) -> None:
    assert clean_text(raw) == expected


def test_empty_and_blank_pages_produce_no_chunks() -> None:
    assert chunk_pdf_pages([], "policy.pdf") == []
    assert chunk_pdf_pages([PdfPage(1, " \t\r\n", True)], "policy.pdf") == []


def test_short_flagged_page_is_kept_and_blank_flagged_page_is_skipped() -> None:
    pages = [PdfPage(3, "  No refunds.  ", True), PdfPage(5, "\t\n", True)]
    before = pages.copy()

    chunk, = chunk_pdf_pages(pages, r"C:\Private\policy.pdf")

    assert (chunk.source_file, chunk.page_number, chunk.chunk_index, chunk.text) == (
        "policy.pdf", 3, 1, "No refunds.",
    )
    assert pages == before
    with pytest.raises(FrozenInstanceError):
        setattr(chunk, "text", "Edited content")


def test_chunks_preserve_page_order_and_restart_indexes() -> None:
    pages = [PdfPage(7, "red green blue", False), PdfPage(2, "one two three", False)]
    chunks = chunk_pdf_pages(pages, "policy.pdf", chunk_size=9, chunk_overlap=3)

    assert [(chunk.page_number, chunk.chunk_index, chunk.text) for chunk in chunks] == [
        (7, 1, "red green"), (7, 2, "blue"), (2, 1, "one two"), (2, 2, "two three"),
    ]


def test_prefers_paragraph_boundary_over_later_sentence_boundary() -> None:
    page = PdfPage(1, "First paragraph\n\nA later sentence. More words follow here.", False)
    chunks = chunk_pdf_pages([page], "policy.pdf", chunk_size=44, chunk_overlap=0)

    assert [chunk.text for chunk in chunks] == [
        "First paragraph", "A later sentence. More words follow here.",
    ]


@pytest.mark.parametrize(
    ("text", "size", "first_chunk"),
    [
        ("Short sentence. These words continue without a stop", 27, "Short sentence."),
        ("Rs. 1,500.50 was paid. Next sentence.", 28, "Rs. 1,500.50 was paid."),
        ("Rs. 1,500.50 was paid. Next sentence.", 19, "Rs. 1,500.50 was"),
        ("Invoice No. 42 belongs to someone", 20, "Invoice No. 42"),
        ("Dr. Shah works at Acme Inc. today and tomorrow", 17, "Dr. Shah works at"),
        ("Paid 1,500.50. Next sentence.", 21, "Paid 1,500.50. Next"),
        ("Paid 1,500. Next sentence.", 18, "Paid 1,500. Next"),
        ("Balance 2.50. Next sentence.", 19, "Balance 2.50. Next"),
        ("Acme Ltd. ships today and tomorrow", 19, "Acme Ltd. ships"),
        ("Mr. Shah ships today and tomorrow", 18, "Mr. Shah ships"),
    ],
)
def test_sentence_boundaries_without_splitting_numbers_or_abbreviations(
    text: str, size: int, first_chunk: str
) -> None:
    chunks = chunk_pdf_pages([PdfPage(1, text, False)], "policy.pdf", size, 0)

    assert chunks[0].text == first_chunk
    assert all(0 < len(chunk.text) <= size for chunk in chunks)
    assert " ".join(chunk.text for chunk in chunks) == text


def test_zero_overlap_preserves_all_words_in_a_long_sentence() -> None:
    words = [f"item-{index:03d}" for index in range(120)]
    page = PdfPage(1, " ".join(words), False)
    chunks = chunk_pdf_pages([page], "policy.pdf", 49, 0)

    assert [word for chunk in chunks for word in chunk.text.split()] == words
    assert all(0 < len(chunk.text) <= 49 for chunk in chunks)


def test_overlap_shares_complete_words() -> None:
    page = PdfPage(1, "alpha beta gamma delta epsilon", False)
    chunks = chunk_pdf_pages([page], "policy.pdf", 17, 8)

    assert [chunk.text for chunk in chunks] == ["alpha beta gamma", "gamma delta", "delta epsilon"]


def test_overlap_is_dropped_to_make_room_for_the_next_word() -> None:
    page = PdfPage(1, "short ten abcdefghij", False)
    chunks = chunk_pdf_pages([page], "policy.pdf", 10, 9)

    assert [chunk.text for chunk in chunks] == ["short ten", "abcdefghij"]


def test_overlap_can_carry_a_short_paragraph_when_new_text_fits() -> None:
    page = PdfPage(1, "Intro.\n\nalpha beta gamma delta", False)
    chunks = chunk_pdf_pages([page], "policy.pdf", 24, 6)

    assert [chunk.text for chunk in chunks] == [
        "Intro.", "Intro.\n\nalpha beta gamma", "gamma delta",
    ]


def test_high_overlap_always_advances_without_losing_words() -> None:
    words = [f"w{index:03d}" for index in range(100)]
    positions = {word: i for i, word in enumerate(words)}
    chunks = chunk_pdf_pages([PdfPage(1, " ".join(words), False)], "policy.pdf", 24, 23)

    previous_start = previous_end = -1
    for chunk in chunks:
        first, *_, last = chunk.text.split()
        start, end = positions[first], positions[last]
        assert previous_start < start <= previous_end + 1
        assert previous_end < end
        assert chunk.text == " ".join(words[start:end + 1])
        assert len(chunk.text) <= 24
        previous_start, previous_end = start, end
    assert previous_end == len(words) - 1


def test_oversized_token_is_hard_split_without_losing_characters() -> None:
    page = PdfPage(1, "start ABCDEFGHIJKLMNOPQRSTUV end", True)
    chunks = chunk_pdf_pages([page], "policy.pdf", 8, 7)

    assert [chunk.text for chunk in chunks] == ["start", "ABCDEFGH", "IJKLMNOP", "QRSTUV", "end"]


def test_defaults_and_overrides_can_be_set_independently(monkeypatch: pytest.MonkeyPatch) -> None:
    assert (config.CHUNK_SIZE, config.CHUNK_OVERLAP) == (800, 100)
    page = PdfPage(1, "alpha beta gamma delta epsilon", False)
    assert chunk_pdf_pages([page], "policy.pdf")[0].text == page.text

    long_page = PdfPage(1, " ".join(f"item-{index:03d}" for index in range(100)), False)
    configured_chunks = chunk_pdf_pages([long_page], "policy.pdf")
    assert len(configured_chunks) > 1
    assert configured_chunks == chunk_pdf_pages([long_page], "policy.pdf", 800, 100)

    monkeypatch.setattr(config, "CHUNK_SIZE", 17)
    monkeypatch.setattr(config, "CHUNK_OVERLAP", 8)
    defaults = chunk_pdf_pages([page], "policy.pdf")
    assert defaults == chunk_pdf_pages([page], "policy.pdf", 17, 8)
    assert defaults != chunk_pdf_pages([page], "policy.pdf", chunk_overlap=0)
    assert defaults != chunk_pdf_pages([page], "policy.pdf", chunk_size=32)


def test_chunk_ids_are_stable_unique_and_use_only_the_filename() -> None:
    pages = [PdfPage(1, "red green blue yellow", False), PdfPage(2, "red green", False)]
    chunks = chunk_pdf_pages(pages, r"C:\Customer\Secret\policy.pdf", 9, 0)
    reprocessed = chunk_pdf_pages(pages.copy(), "/private/other/policy.pdf", 9, 0)

    assert chunks == reprocessed
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert {chunk.source_file for chunk in chunks} == {"policy.pdf"}
    assert all("/" not in chunk.chunk_id and "\\" not in chunk.chunk_id for chunk in chunks)


def test_chunk_ids_change_with_content_page_layout_and_each_setting() -> None:
    pages = [PdfPage(1, "red green", False), PdfPage(2, "blue", False)]

    def ids(source: list[PdfPage], size: int = 24, overlap: int = 0) -> set[str]:
        return {chunk.chunk_id for chunk in chunk_pdf_pages(source, "policy.pdf", size, overlap)}

    original = ids(pages)
    assert original.isdisjoint(ids([pages[0], PdfPage(2, "pink", False)]))
    assert original.isdisjoint(ids([PdfPage(1, "red", False), PdfPage(2, "green blue", False)]))
    assert original.isdisjoint(ids(list(reversed(pages))))
    assert original.isdisjoint(ids(pages, size=25))
    assert original.isdisjoint(ids(pages, overlap=1))


@pytest.mark.parametrize(
    ("size", "overlap", "error", "message"),
    [
        (0, 0, ValueError, "chunk_size"), (-1, 0, ValueError, "chunk_size"),
        (10, -1, ValueError, "chunk_overlap"), (10, 10, ValueError, "chunk_overlap"),
        (10, 11, ValueError, "chunk_overlap"), (True, 0, TypeError, "chunk_size"),
        (10, False, TypeError, "chunk_overlap"), ("10", 0, TypeError, "chunk_size"),
        (10, 2.5, TypeError, "chunk_overlap"),
    ],
)
def test_invalid_settings(size: int, overlap: int, error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        chunk_pdf_pages([], "policy.pdf", size, overlap)


@pytest.mark.parametrize("source", ["", "  ", ".", "..", "dir/", "dir/.", "dir/..", "bad\0.pdf"])
def test_invalid_filename(source: str) -> None:
    with pytest.raises(ValueError, match="filename"):
        chunk_pdf_pages([], source)


@pytest.mark.parametrize(
    ("pages", "error", "message"),
    [
        ((), TypeError, "list"),
        (["text"], TypeError, "PdfPage"),
        ([PdfPage(0, "text", False)], ValueError, "page_number"),
        ([PdfPage(-1, "text", False)], ValueError, "page_number"),
        ([PdfPage(True, "text", False)], TypeError, "page_number"),
        ([PdfPage(1.5, "text", False)], TypeError, "page_number"),
        ([PdfPage(1, "", True), PdfPage(1, "text", False)], ValueError, "Duplicate page_number"),
        ([PdfPage(1, None, False)], TypeError, "Text for page 1"),
    ],
)
def test_invalid_pages(pages: list[PdfPage], error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        chunk_pdf_pages(pages, "policy.pdf")


def test_rejects_nonstring_text_and_source_name() -> None:
    with pytest.raises(TypeError, match="text must be a string"):
        clean_text(None)
    with pytest.raises(TypeError, match="source_name must be a string"):
        chunk_pdf_pages([], None)
