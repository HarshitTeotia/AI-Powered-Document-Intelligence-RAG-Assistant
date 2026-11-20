"""Check CSV validation and temporary-process orchestration without a model."""

import csv
import json
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import Mock

import pytest

from eval import evaluate_retrieval as cli


def _csv(path: Path, *, id: str = "7", **overrides: str) -> Path:
    row = {"id": id, "question": "How much?", "expected_answer": "optional reference",
           "must_contain": "\u20b9 450.00", "source_file": "Rules & Policy.pdf",
           "section": "Fees", "page": "1;5", "category": "answerable", **overrides}
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return path


def test_loads_explicit_utf8_files_in_order_and_allows_missing_labels(tmp_path: Path) -> None:
    first = _csv(tmp_path / "a.csv", id="21", page="")
    second = _csv(tmp_path / "b.csv", id="9", category="order_lookup", source_file="", page="", must_contain="")
    questions = cli._load_questions([first, second])
    assert [question.id for question in questions] == ["21", "9"]
    assert questions[0].terms == ["\u20b9 450.00"] and questions[0].pages == set()
    assert questions[1].terms == [] and questions[1].source == ""


def test_duplicate_ids_across_csvs_are_rejected(tmp_path: Path) -> None:
    first = _csv(tmp_path / "a.csv")
    second = _csv(tmp_path / "b.csv", id=" 7 ")
    with pytest.raises(cli.EvaluationError, match="Duplicate question ID '7'"):
        cli._load_questions([first, second])


@pytest.mark.parametrize(("overrides", "message"), [
    ({"category": "unknown"}, "Unknown category"), ({"question": " "}, "Empty question"),
    ({"source_file": "dir/File.pdf"}, "Invalid ground truth"), ({"page": "0"}, "Invalid page label"),
    ({"must_contain": "Rs.; "}, "usable must_contain"),
])
def test_invalid_ground_truth_is_rejected(tmp_path: Path, overrides: dict[str, str], message: str) -> None:
    file = _csv(tmp_path / "a.csv", **overrides)
    with pytest.raises(cli.EvaluationError, match=message):
        cli._load_questions([file])


def test_invalid_csv_encoding_and_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "a.csv"
    path.write_bytes(b"id,question\n1,\xff\n")
    with pytest.raises(cli.EvaluationError, match="Could not read UTF-8 CSV"):
        cli._load_questions([path])
    path.write_text("id,question\n1,Example\n", encoding="utf-8")
    with pytest.raises(cli.EvaluationError, match="Missing columns"):
        cli._load_questions([path])


def test_cli_repeated_csv_arguments_do_not_add_default(monkeypatch: pytest.MonkeyPatch) -> None:
    parent = Mock(return_value=0)
    monkeypatch.setattr(cli, "_run_parent", parent)
    assert cli.main([]) == 0
    parent.assert_called_once_with([Path("eval/questions.csv")], Path("data/uploads"), 5)
    parent.reset_mock()
    assert cli.main(["--csv", "a.csv", "--csv", "b.csv", "--uploads", "docs", "--top-k", "2"]) == 0
    parent.assert_called_once_with([Path("a.csv"), Path("b.csv")], Path("docs"), 2)


def test_discovery_is_case_insensitive_sorted_and_nonrecursive(tmp_path: Path) -> None:
    for name in ("Z.PDF", "a.pdf", "ignored.txt"):
        (tmp_path / name).touch()
    (tmp_path / "folder.pdf").mkdir()
    (tmp_path / "folder.pdf" / "nested.pdf").touch()
    assert [path.name for path in cli._discover_pdfs(tmp_path)] == ["a.pdf", "Z.PDF"]


@pytest.mark.parametrize(("child_status", "mutate_csv", "expected"), [(0, False, 0), (0, True, 1), (7, True, 7)])
def test_parent_verifies_inputs_cleans_its_store_and_preserves_child_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    child_status: int, mutate_csv: bool, expected: int,
) -> None:
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    (uploads / "synthetic.pdf").write_bytes(b"Synthetic input")
    csv_path = _csv(tmp_path / "questions.csv")
    monkeypatch.setattr(cli.tempfile, "gettempdir", lambda: str(tmp_path))
    stores: list[Path] = []

    def worker(args: list[str], **kwargs: object) -> CompletedProcess:
        manifest = json.loads(kwargs["input"])
        store = Path(manifest["store"])
        stores.append(store)
        assert store.parent == tmp_path and store.exists()
        assert manifest["csvs"] == [str(csv_path)]
        assert manifest["pdfs"] == [str(uploads / "synthetic.pdf")]
        assert "--_worker" in args and "-B" in args
        if mutate_csv:
            csv_path.write_bytes(b"Changed by synthetic worker")
        return CompletedProcess(args, child_status)

    monkeypatch.setattr(cli.subprocess, "run", worker)
    assert cli._run_parent([csv_path], uploads, 5) == expected
    assert len(stores) == 1 and not stores[0].exists()
    out = capsys.readouterr()
    assert "PDF integrity: PASS" in out.out
    assert f"CSV integrity: {'FAIL' if mutate_csv else 'PASS'}" in out.out
    assert "Temporary ChromaDB directory cleaned up." in out.out
