import json
from pathlib import Path

import pytest

from eval import evaluate_generation

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "eval" / "questions.csv"
MOCK_RECORDS = ROOT / "eval" / "fixtures" / "generation_mock.jsonl"


def test_offline_replay_keeps_wrong_statuses_and_failures_separate(capsys):
    assert evaluate_generation.main(["--csv", str(QUESTIONS), "--records", str(MOCK_RECORDS)]) == 0
    output = capsys.readouterr().out
    report = json.loads(output)
    assert report["recorded_rows"] == len(report["cases"]) == 11
    assert report["outcomes"] == {"valid_response": 6, "rejected_response": 2, "runtime_error": 2, "skipped": 1}
    assert report["status_matches"] == 5
    assert report["citation_shape_matches"] == 6
    assert report["returned_citations"] == report["source_integrity_matches"] == report["exact_original_quotes"] == 4
    cases = {case["id"]: case for case in report["cases"]}
    assert cases["6"]["citations"][0]["page_number"] == 3
    assert cases["9"]["status_match"] is True and report["factual_correctness"] == "manual review required"
    assert cases["25"]["status_match"] is False
    assert "automatic" not in output and "Rs. 100,000" not in output


@pytest.mark.parametrize("ids", ["1\n1", "9999"])
def test_replay_rejects_duplicate_or_unknown_ids(tmp_path, capsys, ids):
    records = tmp_path / "input.jsonl"
    records.write_text("\n".join(json.dumps({"id": question, "skip": True}) for question in ids.splitlines()), encoding="utf-8")
    assert evaluate_generation.main(["--csv", str(QUESTIONS), "--records", str(records)]) == 1
    assert "unknown or repeated question ID" in capsys.readouterr().err
