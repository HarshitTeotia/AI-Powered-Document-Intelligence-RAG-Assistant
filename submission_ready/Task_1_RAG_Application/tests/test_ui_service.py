from pathlib import Path
from unittest.mock import Mock

import pytest

from distribuai.ingestion.pipeline import IngestionError, IngestionResult
from distribuai.rag.generation import GeminiConfigurationError, GeminiGenerationError, GroundedAnswer
from distribuai.rag.retrieval import RetrievedChunk, RetrievalError
from distribuai.ui import service

PDF = b"%PDF-1.7\nexample contents\n%%EOF\n"
NAME = "Delivery & Shipping Policy.PDF"


@pytest.fixture(autouse=True)
def configure_mock_chat(monkeypatch):
    monkeypatch.setenv("GEMINI_CHAT_ENABLED", "true")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")


@pytest.mark.parametrize("filename", ["policy.pdf", "Delivery Notes.pdf", NAME])
def test_validate_upload_accepts_pdf_without_changing_input(filename):
    assert service.validate_upload(filename, PDF) is None
    assert PDF == b"%PDF-1.7\nexample contents\n%%EOF\n"


@pytest.mark.parametrize("filename", [
    "", "../policy.pdf", "subdir/policy.pdf", "subdir\\policy.pdf", "/policy.pdf",
    "C:\\work\\policy.pdf", "C:policy.pdf", "NUL.pdf", "policy.pdf ", "policy.txt",
])
def test_validate_upload_rejects_path_or_invalid_name(filename):
    with pytest.raises(service.UploadError):
        service.validate_upload(filename, PDF)


@pytest.mark.parametrize("data", [b"", None, b"hello", b"%PDF-xyz\n", b"%PDF-1.7"])
def test_validate_upload_rejects_invalid_content(data):
    with pytest.raises(service.UploadError):
        service.validate_upload("policy.pdf", data)


def test_validate_upload_enforces_size_limit():
    assert service.MAX_UPLOAD_BYTES == 10 * 1024 * 1024
    maximum = PDF.ljust(service.MAX_UPLOAD_BYTES)
    service.validate_upload("policy.pdf", maximum)
    with pytest.raises(service.UploadError, match="10 MiB"):
        service.validate_upload("policy.pdf", maximum + b"x")


@pytest.mark.parametrize("backend_failure", [False, True])
def test_index_upload_uses_original_name_and_cleans_up(tmp_path, monkeypatch, backend_failure):
    monkeypatch.setattr(service.tempfile, "gettempdir", lambda: str(tmp_path))
    collection = object()
    result = IngestionResult(NAME, 3, 7, 800, 100, 1, 2, [3])
    failure = IngestionError("failed to index")

    def inspect_upload(path, actual_collection, *, source_name):
        assert path.parent.parent == tmp_path
        assert not path.is_relative_to(service.config.PROJECT_ROOT)
        assert path.name == "upload.pdf"
        assert path.read_bytes() == PDF
        assert actual_collection is collection
        assert source_name == NAME
        if backend_failure:
            raise failure
        return result

    ingest = Mock(side_effect=inspect_upload)
    monkeypatch.setattr(service, "ingest_pdf", ingest)
    if backend_failure:
        with pytest.raises(IngestionError) as caught:
            service.ingest_uploaded_pdf(NAME, PDF, collection)
        assert caught.value is failure
    else:
        assert service.ingest_uploaded_pdf(NAME, PDF, collection) is result
    ingest.assert_called_once()
    assert list(tmp_path.iterdir()) == []


def test_invalid_upload_is_rejected_before_creating_a_file(monkeypatch):
    temporary_directory = Mock()
    monkeypatch.setattr(service.tempfile, "TemporaryDirectory", temporary_directory)
    with pytest.raises(service.UploadError):
        service.ingest_uploaded_pdf("../../policy.pdf", PDF, object())
    temporary_directory.assert_not_called()


def test_project_local_temp_directory_is_rejected_before_creating_a_file(tmp_path, monkeypatch):
    monkeypatch.setattr(service.config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(service.tempfile, "gettempdir", lambda: str(tmp_path / "temp"))
    with pytest.raises(service.UploadError, match="outside the project"):
        service.ingest_uploaded_pdf(NAME, PDF, object())
    assert list(tmp_path.iterdir()) == []


def test_cleanup_warning_keeps_successful_result_and_omits_sensitive_details(tmp_path, monkeypatch):
    class FailedCleanup:
        name = str(tmp_path)

        def cleanup(self):
            raise PermissionError("private path")

    result = IngestionResult(NAME, 1, 1, 800, 100, 0, 0, [])
    warnings = []
    monkeypatch.setattr(service.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(service.tempfile, "TemporaryDirectory", lambda **kwargs: FailedCleanup())
    monkeypatch.setattr(service, "ingest_pdf", Mock(return_value=result))

    assert service.ingest_uploaded_pdf(NAME, PDF, object(), on_cleanup_failure=warnings.append) is result
    assert warnings == [service.UPLOAD_CLEANUP_WARNING]
    assert "private path" not in warnings[0]
    (tmp_path / Path("upload.pdf")).unlink()


def test_ask_uses_original_question_and_all_five_retrieval_results(monkeypatch):
    question = "  When do orders ship?  "
    chunks = [RetrievedChunk(str(i), NAME, 1, i, "Orders ship daily.", 0.99, 0.01, i) for i in range(1, 6)]
    collection = object()
    answer = GroundedAnswer("answered", "Daily.", (), "test-model")
    retrieve = Mock(return_value=chunks)
    generate = Mock(return_value=answer)
    monkeypatch.setattr(service, "retrieve", retrieve)
    monkeypatch.setattr(service, "generate_grounded_answer", generate)

    assert service.ask_documents(question, collection) is answer
    retrieve.assert_called_once_with(collection, question, top_k=5)
    generate.assert_called_once_with(question, chunks)
    assert generate.call_args.args[1] is chunks


@pytest.mark.parametrize(
    ("enabled", "api_key", "message"),
    [(None, "test-key", "turned off"), ("false", "test-key", "turned off"),
     ("yes", "test-key", "turned off"), ("true", None, "needs a Gemini API key"),
     ("true", "  ", "needs a Gemini API key")],
)
def test_chat_configuration_blocks_retrieval_and_generation(monkeypatch, enabled, api_key, message):
    for key, value in [("GEMINI_CHAT_ENABLED", enabled), ("GEMINI_API_KEY", api_key)]:
        if value is None:
            monkeypatch.delenv(key)
        else:
            monkeypatch.setenv(key, value)
    retrieve = Mock()
    generate = Mock()
    monkeypatch.setattr(service, "retrieve", retrieve)
    monkeypatch.setattr(service, "generate_grounded_answer", generate)
    with pytest.raises(GeminiConfigurationError, match=message):
        service.ask_documents("Are orders dispatched daily?", object())
    retrieve.assert_not_called()
    generate.assert_not_called()


def test_empty_collection_uses_backend_refusal_without_querying(monkeypatch):
    collection = Mock()
    collection.count.return_value = 0
    create_client = Mock()
    monkeypatch.setattr("distribuai.rag.generation.genai.Client", create_client)

    answer = service.ask_documents("When do orders ship?", collection)

    assert answer.status == "not_in_documents"
    assert answer.citations == ()
    assert answer.model_name is None
    collection.query.assert_not_called()
    create_client.assert_not_called()


@pytest.mark.parametrize("question", ["", "   \n", None, "Q" * (service.MAX_QUESTION_CHARS + 1)])
def test_invalid_question_does_not_retrieve(question, monkeypatch):
    retrieve = Mock()
    monkeypatch.setattr(service, "retrieve", retrieve)
    with pytest.raises(service.QuestionError):
        service.ask_documents(question, object())
    retrieve.assert_not_called()


@pytest.mark.parametrize("failure", [RetrievalError("query failed"), GeminiGenerationError("request failed")])
def test_backend_failure_is_propagated_without_retry(failure, monkeypatch):
    retrieve = Mock(return_value=[])
    generate = Mock()
    failing = retrieve if isinstance(failure, RetrievalError) else generate
    failing.side_effect = failure
    monkeypatch.setattr(service, "retrieve", retrieve)
    monkeypatch.setattr(service, "generate_grounded_answer", generate)

    with pytest.raises(type(failure)) as caught:
        service.ask_documents("Where are my orders?", object())
    assert caught.value is failure
    retrieve.assert_called_once()
    assert generate.call_count == (0 if failing is retrieve else 1)
