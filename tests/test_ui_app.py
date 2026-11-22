import re
from html import unescape
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import dotenv
import pytest
from streamlit import config as streamlit_config
from streamlit.testing.v1 import AppTest

from distribuai import config
from distribuai.ingestion.pipeline import IngestionError, IngestionResult
from distribuai.rag import vector_store
from distribuai.rag.generation import GeminiGenerationError, GroundedAnswer, SourceCitation
from distribuai.ui import service

APP = Path(__file__).resolve().parents[1] / "distribuai" / "ui" / "app.py"
NAME = "Delivery & Shipping FAQ.pdf"
SECOND = "Edited Policy.pdf"
KEY = "test-key-never-rendered"
QUOTE = "Order before noon.\n<script>literal source text</script>"
ANSWER = GroundedAnswer("answered", "Orders ship on weekdays.", (SourceCitation("a", NAME, 2, QUOTE),), "test")
ASK_DOCUMENTS = service.ask_documents
QUICK_PROMPTS = (
    "What is the return policy?", "What are the credit terms?",
    "What are the shipping charges?", "What products are available?",
)


@pytest.fixture
def backend(tmp_path, monkeypatch):
    prior_option = streamlit_config.get_option("client.showErrorDetails")
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.setenv("GEMINI_CHAT_ENABLED", "true")
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "demo")
    collection = object()
    documents = [(NAME, 3, [1, 2]), (SECOND, 4, [1])]
    open_collection = Mock(return_value=collection)
    listing = Mock(side_effect=lambda _: documents.copy())

    def delete(_collection, name):
        index = next(i for i, document in enumerate(documents) if document[0] == name)
        return documents.pop(index)[1]

    remove = Mock(side_effect=delete)
    ask = Mock(return_value=ANSWER)
    ingest = Mock()
    monkeypatch.setattr(vector_store, "open_vector_collection", open_collection)
    monkeypatch.setattr(vector_store, "list_documents", listing)
    monkeypatch.setattr(vector_store, "delete_document", remove)
    monkeypatch.setattr(service, "ask_documents", ask)
    monkeypatch.setattr(service, "ingest_pdf", ingest)
    yield SimpleNamespace(
        documents=documents, collection=collection, open=open_collection, list=listing,
        delete=remove, ask=ask, ingest=ingest,
    )
    streamlit_config.set_option("client.showErrorDetails", prior_option)


def _run_app():
    return AppTest.from_file(str(APP), default_timeout=10).run()


def _chat(app, question):
    return app.chat_input[0].set_value(question).run()


def _button(app, label):
    return next(button for button in app.button if button.label == label)


def _text(app):
    return [element.value for element in app.get("text")]


def _sources(app):
    # AppTest represents expanders with an icon as status blocks in Streamlit 1.45.1.
    return [block for block in [*app.expander, *app.get("status")] if block.label.startswith("Sources (")]


def _assert_no_index_metrics(app):
    visible_types = (
        "text", "caption", "markdown", "heading", "metric", "info", "success", "warning", "error",
        "button", "checkbox", "selectbox", "expander", "status", "chat_input",
    )
    for element_type in visible_types:
        for element in app.get(element_type):
            assert not re.search(r"\b(?:chunks?|vectors?|overlap)\b", str(element.proto), re.IGNORECASE)


def test_empty_home_lists_no_documents_and_never_retrieves(backend):
    backend.documents.clear()
    app = _run_app()

    assert not app.exception
    assert app.title[0].value == "Document Assistant"
    heading = "<h2>AI-Powered Document Intelligence & RAG Assistant</h2>"
    assert any(heading in unescape(element.proto.body) for element in app.sidebar.get("html"))
    assert any("No documents indexed yet" in item.value for item in app.sidebar.info)
    assert "0 documents" in [item.value for item in app.sidebar.caption]
    assert not app.sidebar.metric
    _assert_no_index_metrics(app)
    assert not app.sidebar.selectbox
    assert not any(button.label == "Add demo documents" for button in app.button)
    assert app.chat_input[0].proto.disabled
    assert all(_button(app, question).proto.disabled for question in QUICK_PROMPTS)
    backend.open.assert_called_once_with()
    backend.list.assert_called_once_with(backend.collection)
    backend.ask.assert_not_called()
    backend.ingest.assert_not_called()


def test_document_list_empty_upload_and_missing_credentials_are_safe(backend, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    app = _run_app()
    assert "2 documents" in [item.value for item in app.sidebar.caption]
    assert [NAME, SECOND] == _text(app.sidebar)
    assert any("needs a Gemini API key" in item.value for item in app.main.caption)
    assert not app.checkbox
    assert app.chat_input[0].proto.disabled
    _chat(app, "Attempted submission without a key")
    _button(app, "Add to knowledge base").click().run()
    assert any("Choose a PDF before adding a document" in item.value for item in app.sidebar.error)
    assert not app.exception
    _assert_no_index_metrics(app)
    backend.ask.assert_not_called()
    backend.ingest.assert_not_called()
    backend.open.assert_called_once_with()


@pytest.mark.parametrize("setting", [None, "false"])
def test_opt_in_is_required_even_with_a_key(backend, monkeypatch, setting):
    if setting is None:
        monkeypatch.delenv("GEMINI_CHAT_ENABLED")
    else:
        monkeypatch.setenv("GEMINI_CHAT_ENABLED", setting)
    app = _run_app()
    assert app.chat_input[0].proto.disabled
    assert all(_button(app, question).proto.disabled for question in QUICK_PROMPTS)
    assert any("AI chat is turned off" in item.value for item in app.main.caption)
    assert "GEMINI_CHAT_ENABLED" not in str(app)
    _chat(app, "Attempted submission while disabled")
    assert not app.chat_message
    backend.ask.assert_not_called()


def test_process_setting_takes_precedence_over_dotenv(backend, monkeypatch, tmp_path):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("GEMINI_CHAT_ENABLED=true\n", encoding="utf-8")
    original_load = dotenv.load_dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda path, *, override: original_load(dotenv_path, override=override))
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED")
    monkeypatch.setenv("GEMINI_CHAT_ENABLED", "false")
    assert _run_app().chat_input[0].proto.disabled

    monkeypatch.delenv("GEMINI_CHAT_ENABLED")
    assert not _run_app().chat_input[0].proto.disabled
    backend.ask.assert_not_called()


def test_enabled_chat_ignores_old_consent_and_is_safe_and_session_scoped(backend):
    app = _run_app()
    app.session_state["allow_gemini"] = False
    app.run()
    assert not app.checkbox
    assert not app.chat_input[0].proto.disabled
    assert not any(item.label == "Privacy & AI settings" for item in app.sidebar.get("status"))
    assert "2 documents available" in [item.value for item in app.main.caption]
    assert any("Relevant excerpts are sent to Google" in item.value for item in app.main.caption)
    assert not any(button.label == "Clear chat" for button in app.button)
    _chat(app, "   ")
    backend.ask.assert_not_called()
    _chat(app, "  When are deliveries?  ")
    backend.ask.assert_called_once_with("  When are deliveries?  ", backend.collection)
    assert [message.name for message in app.chat_message] == ["user", "assistant"]
    assert "When are deliveries?" in _text(app.chat_message[0])
    assert app.session_state["chat_history"][0][0] == "  When are deliveries?  "
    assert ANSWER.answer in _text(app.chat_message[1])
    sources = _sources(app.chat_message[1])[0]
    assert sources.label == "Sources (1)"
    assert NAME in _text(sources)
    assert "Page 2" in [item.value for item in sources.caption]
    assert QUOTE in _text(app.chat_message[1])
    assert KEY not in str(app)

    second = GroundedAnswer("not_in_documents", "I could not find that policy.", (), "test")
    backend.ask.return_value = second
    _chat(app, "Is there a discount?")
    assert len(app.chat_message) == 4
    assert second.answer in _text(app.chat_message[3])
    assert not _sources(app.chat_message[3])
    app.run()
    assert len(app.chat_message) == 4
    assert backend.ask.call_count == 2
    assert len(_run_app().chat_message) == 0
    app.button(key="clear_chat").click().run()
    assert not app.exception
    assert not app.chat_message
    assert len(backend.documents) == 2
    backend.delete.assert_not_called()
    assert backend.ask.call_count == 2


@pytest.mark.parametrize("question", QUICK_PROMPTS)
def test_quick_prompt_and_manual_input_share_the_retrieval_and_generation_path(backend, monkeypatch, question):
    passages = [object()]
    retrieve = Mock(return_value=passages)
    generate = Mock(return_value=ANSWER)
    monkeypatch.setattr(service, "ask_documents", ASK_DOCUMENTS)
    monkeypatch.setattr(service, "retrieve", retrieve)
    monkeypatch.setattr(service, "generate_grounded_answer", generate)

    app = _run_app()
    retrieve.assert_not_called()
    generate.assert_not_called()
    assert all(not _button(app, label).proto.disabled for label in QUICK_PROMPTS)
    _button(app, question).click().run()
    assert not app.exception
    retrieve.assert_called_once_with(backend.collection, question, top_k=5)
    generate.assert_called_once_with(question, passages)
    assert len(app.chat_message) == 2
    assert app.session_state["chat_history"] == [(question, ANSWER)]
    assert question in _text(app.chat_message[0])
    assert QUOTE in _text(_sources(app.chat_message[1])[0])
    app.run()
    assert not any(button.label in QUICK_PROMPTS for button in app.button)
    assert app.button(key="clear_chat")
    assert retrieve.call_count == generate.call_count == 1

    manual = "  When do orders ship?  "
    _chat(app, manual)
    app.run()
    assert retrieve.call_args_list == [call(backend.collection, question, top_k=5), call(backend.collection, manual, top_k=5)]
    assert generate.call_args_list == [call(question, passages), call(manual, passages)]
    assert app.session_state["chat_history"] == [(question, ANSWER), (manual, ANSWER)]


def test_order_lookup_displays_existing_answer_without_sources(backend):
    answer = GroundedAnswer("order_lookup", "An authorized order-system lookup is required.", (), "test")
    backend.ask.return_value = answer
    app = _run_app()
    _chat(app, "Has my order shipped?")
    assert answer.answer in _text(app.chat_message[1])
    assert not _sources(app.chat_message[1])


def test_gemini_failure_does_not_display_credentials_or_retry(backend):
    backend.ask.side_effect = GeminiGenerationError(f"{KEY} C:\\private\\path")
    app = _run_app()
    _chat(app, "What is the return policy?")
    assert not app.exception
    assert "Gemini could not complete" in app.chat_message[1].error[0].value
    assert KEY not in str(app)
    assert "private" not in app.chat_message[1].error[0].value
    app.run()
    backend.ask.assert_called_once()


def test_delete_requires_confirmation_and_never_changes_original_pdf(backend):
    config.UPLOAD_DIR.mkdir()
    sample = config.UPLOAD_DIR / NAME
    sample.write_bytes(b"original")
    app = _run_app()
    _chat(app, "When do orders ship?")
    app.sidebar.selectbox[0].select(NAME).run()
    _button(app, "Remove document").click().run()
    backend.delete.assert_not_called()
    assert app.chat_message
    confirmation = next(item for item in app.sidebar.checkbox if "Keep the original PDF" in item.label)
    _assert_no_index_metrics(app)
    confirmation.check()
    _button(app, "Remove document").click().run()
    assert not app.exception
    backend.delete.assert_called_once_with(backend.collection, NAME)
    assert sample.read_bytes() == b"original"
    assert backend.documents == [(SECOND, 4, [1])]
    assert "1 document" in [item.value for item in app.sidebar.caption]
    assert any("Document removed successfully" in item.value for item in app.sidebar.success)
    assert app.session_state["document_feedback"][3] == (NAME, 3)
    _assert_no_index_metrics(app)
    assert not app.chat_message
    assert not app.chat_input[0].proto.disabled


def test_demo_indexing_is_explicit_reports_partial_failure_and_clears_chat(backend):
    config.UPLOAD_DIR.mkdir()
    pdf = config.UPLOAD_DIR / NAME
    broken = config.UPLOAD_DIR / "Unreadable.pdf"
    for path in (pdf, broken):
        path.write_bytes(b"original")
    result = IngestionResult(NAME, 3, 5, 700, 90, 1, 3, [3])

    def index(path, collection):
        assert collection is backend.collection
        if path == broken:
            raise IngestionError("private details")
        backend.documents[0] = (NAME, 5, [1, 2])
        return result

    backend.ingest.side_effect = index
    app = _run_app()
    _chat(app, "What are delivery times?")
    backend.ingest.assert_not_called()
    app.button(key="index_demos").click().run()
    assert not app.exception
    assert [call.args for call in backend.ingest.call_args_list] == [(pdf, backend.collection), (broken, backend.collection)]
    assert any("Could not index" in item.value for item in app.sidebar.error)
    assert "private details" not in str(app)
    assert any("Document added to your knowledge base" in item.value for item in app.sidebar.success)
    assert app.session_state["document_feedback"][0] == [result]
    assert "2 documents" in [item.value for item in app.sidebar.caption]
    _assert_no_index_metrics(app)
    assert not app.chat_message
    assert not app.chat_input[0].proto.disabled
    app.run()
    assert backend.ingest.call_count == 2
    backend.ask.assert_called_once()


def test_index_unavailable_shows_safe_error(backend):
    backend.open.side_effect = RuntimeError("private filesystem path")
    app = _run_app()
    assert not app.exception
    assert "knowledge base is unavailable" in app.sidebar.error[0].value
    assert "private filesystem path" not in str(app)
    assert app.chat_input[0].proto.disabled
    backend.list.assert_not_called()
