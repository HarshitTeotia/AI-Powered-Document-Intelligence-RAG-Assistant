"""Clean extracted text and split individual PDF pages into citable chunks."""

import hashlib
import json
import re
from bisect import bisect_left, bisect_right
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import PureWindowsPath

from distribuai import config
from distribuai.ingestion.pdf_extractor import PdfPage

_WORD = re.compile(r"\S+")
_SENTENCE_END = re.compile(r"[.!?]['\"’”)\]}]*$")
_NON_SENTENCE_PERIOD = re.compile(
    r"(?:\b(?:Rs|No|Dr|Mr|Inc|Ltd)|(?<!\d)\d+(?:[,.]\d+)+)\.['\"’”)\]}]*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DocumentChunk:
    """Immutable text with a stable ID and 1-based citation coordinates."""

    chunk_id: str
    source_file: str
    page_number: int
    chunk_index: int
    text: str


def clean_text(text: str) -> str:
    """Normalize spacing while keeping line breaks and one blank line per gap."""
    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = (re.sub(r"[ \t]+", " ", line).strip() for line in normalized.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def chunk_pdf_pages(
    pages: list[PdfPage],
    source_name: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[DocumentChunk]:
    """Return ordered, page-specific chunks for all nonempty cleaned pages.

    Size and overlap are counted in characters. Words longer than the size limit
    are the sole exception to splitting at word boundaries.
    """
    if not isinstance(pages, list):
        raise TypeError("pages must be a list of PdfPage objects.")
    source_file = _safe_filename(source_name)
    size = config.CHUNK_SIZE if chunk_size is None else chunk_size
    overlap = config.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
    if not isinstance(size, int) or isinstance(size, bool):
        raise TypeError("chunk_size must be an integer.")
    if not isinstance(overlap, int) or isinstance(overlap, bool):
        raise TypeError("chunk_overlap must be an integer.")
    if size <= 0:
        raise ValueError("chunk_size must be positive.")
    if not 0 <= overlap < size:
        raise ValueError("chunk_overlap must be nonnegative and smaller than chunk_size.")

    cleaned_pages: list[tuple[int, str]] = []
    seen: set[int] = set()
    for page in pages:
        if not isinstance(page, PdfPage):
            raise TypeError("pages must contain only PdfPage objects.")
        number = page.page_number
        if not isinstance(number, int) or isinstance(number, bool):
            raise TypeError("page_number must be an integer.")
        if number < 1:
            raise ValueError("page_number must be positive.")
        if number in seen:
            raise ValueError(f"Duplicate page_number {number}.")
        if not isinstance(page.text, str):
            raise TypeError(f"Text for page {number} must be a string.")
        seen.add(number)
        cleaned_pages.append((number, clean_text(page.text)))

    # JSON preserves field and page boundaries, including empty pages.
    serialized = json.dumps((source_file, size, overlap, cleaned_pages), separators=(",", ":"))
    fingerprint = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return [
        DocumentChunk(f"{fingerprint}:p{number}:c{index}", source_file, number, index, chunk)
        for number, text in cleaned_pages
        for index, chunk in enumerate(_chunk_page(text, size, overlap), start=1)
    ]


def _safe_filename(source_name: str) -> str:
    """Keep a filename from either Windows or POSIX path syntax."""
    if not isinstance(source_name, str):
        raise TypeError("source_name must be a string.")
    path = source_name.strip()
    last_component = path.replace("\\", "/").rsplit("/", 1)[-1]
    filename = PureWindowsPath(path).name.strip()
    if (
        last_component in {"", ".", ".."}
        or filename in {"", ".", ".."}
        or filename.endswith(".")
        or PureWindowsPath(filename).is_reserved()
        or any(ord(char) < 32 or char in '<>:"|?*' for char in filename)
    ):
        raise ValueError("source_name must identify a valid filename.")
    return filename


def _chunk_page(text: str, size: int, overlap: int) -> Iterator[str]:
    """Split runs of whole words, emitting oversized tokens in separate pieces."""
    words: list[re.Match[str]] = []
    for word in _WORD.finditer(text):
        start, end = word.span()
        if end - start <= size:
            words.append(word)
        else:
            yield from _chunk_words(text, words, size, overlap)
            words.clear()
            for offset in range(start, end, size):
                yield text[offset:min(offset + size, end)]
    yield from _chunk_words(text, words, size, overlap)


def _chunk_words(
    text: str, words: list[re.Match[str]], size: int, overlap: int
) -> Iterator[str]:
    """Prefer natural boundaries and fit new words after any shared suffix."""
    starts = [word.start() for word in words]
    ends = [word.end() for word in words]
    paragraphs = [i for i, end in enumerate(ends) if text.startswith("\n\n", end)]
    sentences = []
    for i, word in enumerate(words):
        token = word.group()
        if _SENTENCE_END.search(token) and not _NON_SENTENCE_PERIOD.search(token):
            sentences.append(i)

    start = unread = 0
    while unread < len(words):
        end = bisect_right(ends, starts[start] + size, lo=unread) - 1
        if end < len(words) - 1:
            for boundaries in (paragraphs, sentences):
                position = bisect_right(boundaries, end)
                if position and boundaries[position - 1] >= unread:
                    end = boundaries[position - 1]
                    break

        yield text[starts[start]:ends[end]]
        unread = end + 1
        if unread == len(words):
            break

        # Retain the largest whole-word suffix that leaves room for a new word.
        # The unread index advances even when a short chunk is fully carried over.
        earliest = max(ends[end] - overlap, ends[unread] - size)
        start = bisect_left(starts, earliest, lo=start, hi=unread)
