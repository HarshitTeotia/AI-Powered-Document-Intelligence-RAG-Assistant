"""Run retrieval evaluation against an isolated, temporary ChromaDB index."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distribuai.evaluation.scoring import (  # noqa: E402
    HIT_DEPTHS, RetrievalScore, aggregate_retrieval_scores, normalize_text,
    parse_pages, parse_terms, score_retrieval,
)

if TYPE_CHECKING:
    from distribuai.rag.retrieval import RetrievedChunk

_CATEGORIES = ("answerable", "not_in_documents", "order_lookup")
_COLUMNS = frozenset("id question expected_answer must_contain source_file section page category".split())
_TEMP_PREFIX = "distribuai-eval-"


class EvaluationError(Exception):
    """An evaluation input or pipeline operation failed."""


@dataclass(frozen=True)
class _Question:
    id: str
    text: str
    category: str
    source: str
    pages: set[int]
    terms: list[str]


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _load_questions(paths: list[Path]) -> list[_Question]:
    """Load and validate every row before opening the temporary index."""
    import pandas as pd

    from distribuai.rag.vector_store import _require_source_filename

    questions: list[_Question] = []
    seen: set[str] = set()
    for path in paths:
        try:
            frame = pd.read_csv(path, encoding="utf-8", dtype=str, keep_default_na=False)
        except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise EvaluationError(f"Could not read UTF-8 CSV {path} ({type(exc).__name__}).") from exc
        missing = _COLUMNS - set(frame.columns)
        if missing:
            raise EvaluationError(f"Missing columns in {path}: {', '.join(sorted(missing))}.")
        for row_number, row in enumerate(frame.to_dict("records"), start=2):
            question_id = row["id"].strip()
            context = f"{path}, row {row_number}"
            if not question_id:
                raise EvaluationError(f"Empty question ID in {context}.")
            if question_id in seen:
                raise EvaluationError(f"Duplicate question ID {question_id!r} in {context}.")
            category, question = row["category"].strip(), row["question"].strip()
            if category not in _CATEGORIES:
                raise EvaluationError(f"Unknown category {category!r} for Q{question_id} in {context}.")
            if not question:
                raise EvaluationError(f"Empty question for Q{question_id} in {context}.")
            source = row["source_file"].strip()
            try:
                pages, terms = parse_pages(row["page"]), parse_terms(row["must_contain"])
                if category == "answerable":
                    source = _require_source_filename(source)
                    if not terms or any(not normalize_text(term) for term in terms):
                        raise ValueError("Answerable questions need usable must_contain terms.")
            except (TypeError, ValueError) as exc:
                raise EvaluationError(f"Invalid ground truth for Q{question_id} in {context}: {exc}") from exc
            seen.add(question_id)
            questions.append(_Question(question_id, question, category, source, pages, terms))
    return questions


def _discover_pdfs(uploads: Path) -> list[Path]:
    if not uploads.is_dir():
        raise EvaluationError(f"Uploads directory does not exist: {uploads}")
    pdfs = sorted(
        (path for path in uploads.iterdir() if path.suffix.casefold() == ".pdf"
         and not path.is_symlink() and path.is_file()),
        key=lambda path: (path.name.casefold(), path.name),
    )
    if not pdfs:
        raise EvaluationError(f"No PDF files found in {uploads}.")
    return pdfs


def _hash_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _verify_integrity(label: str, digests: dict[Path, str]) -> bool:
    intact = True
    for path, digest in digests.items():
        try:
            if _hash_file(path) == digest:
                continue
            print(f"Input changed: {path}", file=sys.stderr)
        except OSError as exc:
            print(f"Could not verify {path}: {exc}", file=sys.stderr)
        intact = False
    print(f"{label} integrity: {'PASS' if intact else 'FAIL'} ({len(digests)} files checked)")
    return intact


def _run_parent(csvs: list[Path], uploads: Path, top_k: int) -> int:
    """Own the input hashes and remove only the directory created by this run."""
    for path in csvs:
        if not path.is_file():
            raise EvaluationError(f"CSV file does not exist: {path}")
    pdfs = _discover_pdfs(uploads)
    pdf_hashes = {path: _hash_file(path) for path in pdfs}
    csv_hashes = {path: _hash_file(path) for path in csvs}
    temp_root = Path(tempfile.gettempdir()).resolve()
    if temp_root.is_relative_to(ROOT):
        raise EvaluationError("The OS temporary directory must be outside the repository.")
    directory = Path(tempfile.mkdtemp(prefix=_TEMP_PREFIX, dir=temp_root))
    status = 1
    cleanup_ok = False
    try:
        manifest = json.dumps({"store": str(directory), "pdfs": list(map(str, pdfs)), "csvs": list(map(str, csvs))})
        child = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--_worker", "--top-k", str(top_k)],
            input=manifest, encoding="utf-8", check=False,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        status = child.returncode
        if status:
            print(f"Evaluation child exited with status {status}.", file=sys.stderr)
    finally:
        pdf_ok = _verify_integrity("PDF", pdf_hashes)
        csv_ok = _verify_integrity("CSV", csv_hashes)
        try:
            resolved = directory.resolve(strict=True)
            if resolved != directory or resolved.parent != temp_root or not resolved.name.startswith(_TEMP_PREFIX):
                raise OSError("Temporary folder no longer resolves to its original path.")
            shutil.rmtree(resolved)
            cleanup_ok = True
            print("Temporary ChromaDB directory cleaned up.")
        except OSError as exc:
            print(f"Temporary ChromaDB cleanup failed at {directory}: {exc}", file=sys.stderr)
    return status or int(not (pdf_ok and csv_ok and cleanup_ok))


def _top_three(results: list[RetrievedChunk]) -> None:
    print("  Top 3:")
    if not results:
        print("    N/A")
    for hit in results[:3]:
        print(f"    {hit.rank}. {hit.source_file} | page {hit.page_number} | similarity {_similarity(hit.similarity)}")


def _similarity(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.17f}"


def _partial_snippet(question: _Question, score: RetrievalScore, results: list[RetrievedChunk]) -> tuple[RetrievedChunk, str]:
    """Find a nearby partial phrase in a correct-page passage, up to 150 chars."""
    correct = [r for r in results[:5] if r.source_file == question.source and r.page_number in question.pages]
    candidates = [r for r in correct if any(normalize_text(term) in normalize_text(r.text) for term in score.found_terms)]
    candidates = candidates or correct[:1]
    first = candidates[0]
    best = (0, first, " ".join(first.text.split())[:150])
    for result in candidates:
        text = " ".join(result.text.split())
        words = list(re.finditer(r"\w+", text))
        for term in score.missing_terms:
            match = SequenceMatcher(None, re.findall(r"\w+", normalize_text(term)),
                                    [w.group().casefold() for w in words], autojunk=False).find_longest_match()
            if match.size > best[0]:
                start = max(0, words[match.b].start() - 50)
                best = match.size, result, text[start:start + 150]
    return best[1], best[2]


def _diagnostics(question: _Question, score: RetrievalScore, results: list[RetrievedChunk]) -> str | None:
    file_hit = score.file_rank is not None and score.file_rank <= 5
    page_hit = score.page_rank is not None and score.page_rank <= 5
    missed = not file_hit or (bool(question.pages) and not page_hit)
    if not missed and (not page_hit or score.must_contain_all):
        return None
    outcome = "MISS" if missed else "EVIDENCE GAP"
    print(f"\n{outcome} Q{question.id}\n  Question: {question.text}")
    print(f"  Expected file: {question.source} | Accepted pages: {sorted(question.pages) or 'N/A'}")
    print(f"  File hit@5: {file_hit} | Page hit@5: {page_hit if question.pages else 'N/A'}")
    if not missed:
        result, snippet = _partial_snippet(question, score, results)
        print(f"  Retrieved correct file/page: {result.source_file} | page {result.page_number} | rank {result.rank}")
        print(f"  Required: {question.terms!r} | Found: {score.found_terms!r} | Missing: {score.missing_terms!r}")
        print(f"  Must-contain fraction: {score.must_contain_fraction:.4f}")
        print(f"  Partial-match snippet: {snippet or 'N/A'}")
    _top_three(results)
    return outcome


def _ratio(hits: int, count: int) -> str:
    return f"{hits}/{count} ({hits / count:.1%})" if count else "N/A"


def _print_summary(scores: list[RetrievalScore], similarities: dict[str, list[tuple[_Question, float | None]]]) -> None:
    summary = aggregate_retrieval_scores(scores)
    print("\n=== Answerable retrieval summary ===")
    print(f"Answerable questions: {summary.question_count}")
    print(f"Page labels: {summary.labelled_page_count} present, {summary.missing_page_count} missing")
    for k, file_hits, page_hits in zip(HIT_DEPTHS, summary.file_hits, summary.page_hits, strict=True):
        print(f"File hit@{k}: {_ratio(file_hits, summary.question_count)}")
        print(f"Page hit@{k}: {_ratio(page_hits, summary.labelled_page_count)}")
    print("MRR: " + (f"{summary.mrr:.4f}" if summary.mrr is not None else "N/A"))
    print(f"Must-contain recall@5 (all strings): {_ratio(summary.all_terms_count, summary.question_count)}")
    fraction = summary.average_term_fraction
    print("Average must-contain fraction@5: " + (f"{fraction:.1%}" if fraction is not None else "N/A"))
    print(f"Answerable questions without results: {summary.no_result_count}")
    print("Answerable top-1 similarity min/median/max: " + "/".join(map(_similarity, (
        summary.similarity_min, summary.similarity_median, summary.similarity_max,
    ))))
    absent = [value for _, value in similarities["not_in_documents"] if value is not None]
    print(f"Not-in-documents maximum similarity: {_similarity(max(absent, default=None))}")
    for category in _CATEGORIES:
        print(f"\n=== {category} similarities - ascending ===")
        values = sorted(similarities[category], key=lambda entry: (
            entry[1] is None, entry[1] if entry[1] is not None else 0.0, entry[0].id,
        ))
        if not values:
            print("N/A")
        for question, value in values:
            print(f"Q{question.id} | {_similarity(value)} | {question.text}")


def _run_worker(top_k: int) -> int:
    from distribuai.ingestion.pipeline import IngestionError, ingest_pdf
    from distribuai.rag.embeddings import EmbeddingError
    from distribuai.rag.retrieval import RetrievalError, retrieve
    from distribuai.rag.vector_store import VectorStoreError, open_vector_collection

    manifest = json.load(sys.stdin)
    directory = Path(manifest["store"]).resolve()
    if (directory.parent != Path(tempfile.gettempdir()).resolve()
            or directory.is_relative_to(ROOT) or not directory.name.startswith(_TEMP_PREFIX)
            or not directory.is_dir() or any(directory.iterdir())):
        raise EvaluationError("Worker requires an empty temporary database directory from the parent.")
    csvs = [Path(path) for path in manifest["csvs"]]
    pdfs = [Path(path) for path in manifest["pdfs"]]
    questions = _load_questions(csvs)
    depth = max(5, top_k)
    counts = Counter(question.category for question in questions)
    print(f"CSV files: {', '.join(map(str, csvs))}\nRetrieval depth: {depth}")
    print("Discovered PDFs: " + ", ".join(path.name for path in pdfs))
    print(f"Questions: {len(questions)} | " + " | ".join(f"{category}: {counts[category]}" for category in _CATEGORIES))

    try:
        collection = open_vector_collection(directory)
    except VectorStoreError as exc:
        raise EvaluationError(f"Could not open the temporary index: {exc}") from exc
    indexed = 0
    for path in pdfs:
        try:
            result = ingest_pdf(path, collection)
        except IngestionError as exc:
            raise EvaluationError(f"Could not ingest {path.name}: {exc}") from exc
        indexed += result.chunk_count
        print(f"INDEXED: {result}")
    try:
        actual_count = collection.count()
    except Exception as exc:
        raise EvaluationError("Could not count records in the temporary index.") from exc
    if actual_count != indexed:
        raise EvaluationError(f"Ingestion reported {indexed} chunks but the index holds {actual_count}.")
    print(f"Total indexed PDFs: {len(pdfs)} | Total indexed chunks: {actual_count}")

    scores: list[RetrievalScore] = []
    similarities: dict[str, list[tuple[_Question, float | None]]] = {category: [] for category in _CATEGORIES}
    diagnostics: Counter[str] = Counter()
    for question in questions:
        try:
            results = retrieve(collection, question.text, top_k=depth)
        except (EmbeddingError, RetrievalError, VectorStoreError) as exc:
            raise EvaluationError(f"Retrieval failed for Q{question.id}: {exc}") from exc
        top = results[0] if results else None
        similarity = top.similarity if top else None
        similarities[question.category].append((question, similarity))
        if question.category == "answerable":
            score = score_retrieval(results, question.source, question.pages, question.terms)
            scores.append(score)
            if outcome := _diagnostics(question, score, results):
                diagnostics[outcome] += 1
        else:
            source, page = (top.source_file, top.page_number) if top else ("N/A", "N/A")
            print(f"Q{question.id} [{question.category}] {question.text}")
            print(f"  Top-1 similarity: {_similarity(similarity)} | Source: {source} | Page: {page}")
    if not diagnostics["MISS"]:
        print("No retrieval misses.")
    if not diagnostics["EVIDENCE GAP"]:
        print("No evidence gaps.")
    _print_summary(scores, similarities)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return its process status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, action="append", help="UTF-8 evaluation CSV, repeatable (default: eval/questions.csv)")
    parser.add_argument("--uploads", type=Path, default=Path("data/uploads"), help="PDF directory (default: data/uploads)")
    parser.add_argument("--top-k", type=_positive_int, default=5, help="Retrieval depth, with at least five (default: 5)")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    try:
        Path(__file__).read_bytes().decode("ascii")
        if args._worker:
            return _run_worker(args.top_k)
        return _run_parent(args.csv or [Path("eval/questions.csv")], args.uploads, args.top_k)
    except UnicodeDecodeError:
        print("Evaluation failed: evaluate_retrieval.py must be ASCII-only.", file=sys.stderr)
        return 1
    except (EvaluationError, OSError, ValueError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
