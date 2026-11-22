"""Replay saved generation cases offline through the application's response validator."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from distribuai.rag.generation import GeminiResponseError, _validate_response, _validate_sources
from distribuai.rag.retrieval import RetrievedChunk
from eval.evaluate_retrieval import EvaluationError, _load_questions

_ERRORS = {"configuration", "retrieval", "timeout", "service", "model_unavailable", "rate_limit", "authentication"}


def _load_records(path: Path, expected_ids: set[str]) -> dict[str, dict]:
    records = {}
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"Record {number} is not valid JSON.") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Record {number} must be an object.")
            question_id = record.get("id")
            if not isinstance(question_id, str) or question_id not in expected_ids or question_id in records:
                raise ValueError(f"Record {number} has an unknown or repeated question ID.")
            fields = set(record) - {"id"}
            if fields == {"error"} and isinstance(record["error"], str) and record["error"] in _ERRORS:
                records[question_id] = record
            elif fields == {"skip"} and record["skip"] is True:
                records[question_id] = record
            elif fields == {"retrieved", "response"} and isinstance(record["retrieved"], list):
                records[question_id] = record
            else:
                raise ValueError(f"Record {number} must supply a response and evidence, a known error, or a skip.")
    if not records:
        raise ValueError("Supply at least one evaluation record.")
    return records


def _evaluate_response(record: dict) -> dict:
    try:
        chunks = [RetrievedChunk(**item) for item in record["retrieved"]]
        sources = _validate_sources(chunks)
    except (TypeError, ValueError) as exc:
        raise ValueError("A recorded passage has invalid fields or citation metadata.") from exc
    if not sources:
        raise ValueError("Replay of a model response requires the passages originally supplied to the model.")

    payload = record["response"]
    text = payload if isinstance(payload, str) else json.dumps(payload)
    try:
        answer = _validate_response(SimpleNamespace(text=text), sources, "offline-replay")
    except GeminiResponseError as exc:
        # The production validator's messages contain no prompts or document text.
        return {"outcome": "rejected_response", "reason": str(exc)}

    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    citations = []
    for citation in answer.citations:
        original = by_id[citation.chunk_id]
        citations.append({
            "source_file": citation.source_file,
            "page_number": citation.page_number,
            "source_integrity": (citation.source_file, citation.page_number) == (original.source_file, original.page_number),
            "exact_original_quote": citation.quote in original.text,
        })
    return {
        "outcome": "valid_response", "status": answer.status, "citations": citations,
        "citation_shape_valid": bool(citations) == (answer.status == "answered"),
    }


def evaluate(csv_path: Path, records_path: Path) -> dict:
    """Compare statuses and check source integrity without grading factual claims."""
    questions = {question.id: question for question in _load_questions([csv_path])}
    records = _load_records(records_path, set(questions))
    cases = []
    for question_id, record in records.items():
        question = questions[question_id]
        expected = "answered" if question.category == "answerable" else question.category
        case = {"id": question_id, "expected_status": expected}
        if "skip" in record:
            case["outcome"] = "skipped"
        elif "error" in record:
            case.update(outcome="runtime_error", reason=record["error"])
        else:
            try:
                case.update(_evaluate_response(record))
            except ValueError as exc:
                raise ValueError(f"Q{question_id}: {exc}") from exc
            if case["outcome"] == "valid_response":
                case["status_match"] = case["status"] == expected
        cases.append(case)

    outcomes = Counter(case["outcome"] for case in cases)
    valid = [case for case in cases if case["outcome"] == "valid_response"]
    citations = [citation for case in valid for citation in case["citations"]]
    return {
        "mode": "offline replay; no Gemini requests",
        "dataset_rows": len(questions),
        "recorded_rows": len(cases),
        "unrecorded_rows": len(questions) - len(cases),
        "outcomes": {name: outcomes[name] for name in ("valid_response", "rejected_response", "runtime_error", "skipped")},
        "status_matches": sum(case["status_match"] for case in valid),
        "citation_shape_matches": sum(case["citation_shape_valid"] for case in valid),
        "returned_citations": len(citations),
        "source_integrity_matches": sum(citation["source_integrity"] for citation in citations),
        "exact_original_quotes": sum(citation["exact_original_quote"] for citation in citations),
        "factual_correctness": "manual review required",
        "cases": cases,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("eval/questions.csv"))
    parser.add_argument("--records", type=Path, required=True, help="Private or synthetic JSONL recording to replay offline")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(evaluate(args.csv, args.records), indent=2))
    except (EvaluationError, OSError, ValueError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
