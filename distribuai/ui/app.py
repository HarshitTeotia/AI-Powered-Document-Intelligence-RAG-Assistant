"""Run the local document assistant with Streamlit."""

from html import escape
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

_environment_error = False
if __name__ == "__main__":
    st.set_option("client.showErrorDetails", "none")
    # OCR reads its executable setting when the backend config is first imported.
    try:
        load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    except (OSError, UnicodeError):
        _environment_error = True

from distribuai import config
from distribuai.ingestion.ocr_fallback import OcrError
from distribuai.ingestion.pdf_extractor import PdfExtractionError
from distribuai.ingestion.pipeline import IngestionError, IngestionResult
from distribuai.rag.embeddings import EmbeddingError, EmbeddingTokenLimitError
from distribuai.rag.generation import (
    GeminiConfigurationError,
    GeminiGenerationError,
    GeminiResponseError,
    GroundedAnswer,
)
from distribuai.rag.retrieval import RetrievalError
from distribuai.rag.vector_store import (
    delete_document,
    list_documents,
    open_vector_collection,
    VectorStoreError,
)
from distribuai.ui import service

_PROJECT_NAME = "AI-Powered Document Intelligence & RAG Assistant"
_EMPTY = "No documents indexed yet. Upload a PDF to get started."
_DOCUMENT_CHANGE = "Chat was cleared because the knowledge base changed."
_STYLES = Path(__file__).with_name("styles.css")
_SUGGESTIONS = (
    ("What is the return policy?", ":material/assignment_return:"),
    ("What are the credit terms?", ":material/request_quote:"),
    ("What are the shipping charges?", ":material/local_shipping:"),
    ("What products are available?", ":material/inventory_2:"),
)
_GEMINI_FAILURES = {
    "Gemini authentication or access failed. Check your API key and project permissions.":
        "Gemini denied access. Check your API key and project permissions.",
    "Gemini rate limit or quota exceeded.":
        "Gemini's rate limit or quota was reached. Check your quota or try again later.",
    "The configured Gemini model is unavailable. Check GEMINI_MODEL.":
        "Gemini could not find the configured model. Check GEMINI_MODEL locally.",
    "The Gemini request timed out.": "Gemini timed out. Try again when the service is available.",
}


def _index_error(error: Exception) -> str:
    if isinstance(error, service.UploadError):
        return str(error)
    cause = error.__cause__
    if isinstance(cause, (PdfExtractionError, OcrError)):
        return "Could not extract the PDF. Check that it opens and that Tesseract is available for scanned pages."
    if isinstance(cause, EmbeddingTokenLimitError):
        return "Some PDF text is too long to index. Try uploading a shorter document."
    if isinstance(cause, EmbeddingError):
        return "Could not prepare the PDF. Check that the local document model is available."
    if isinstance(cause, VectorStoreError):
        return "Could not finish updating the local index. Check the indexed document list before retrying."
    return "Could not index the PDF. Check that it opens and contains usable text."


def _answer_error(error: Exception) -> str:
    if isinstance(error, service.QuestionError):
        return str(error)
    if isinstance(error, EmbeddingTokenLimitError):
        return "The question is too long for the embedding model. Try asking it in fewer words."
    if isinstance(error, EmbeddingError):
        return "Could not search the documents. Check that the local embedding model is available."
    if isinstance(error, RetrievalError):
        return "Could not search the local document index. Try again when it is available."
    if isinstance(error, GeminiConfigurationError):
        return "Gemini configuration could not be loaded. Check the key and model in the app's local configuration."
    if isinstance(error, GeminiResponseError):
        return "Gemini returned an answer or citation that could not be verified. Try rephrasing the question."
    if isinstance(error, GeminiGenerationError):
        return _GEMINI_FAILURES.get(str(error), "Gemini could not complete the request. Check service availability and try again.")
    return "Could not answer the question. Try again."


def _finish_document_action(
    *,
    results: list[IngestionResult] | None = None,
    errors: list[tuple[str, str]] | None = None,
    warnings: list[str] | None = None,
    removed: tuple[str, int] | None = None,
) -> None:
    changed = bool(results) or (removed is not None and removed[1] > 0)
    st.session_state.document_feedback = (results or [], errors or [], warnings or [], removed, changed)
    if changed:
        st.session_state.chat_history.clear()
    st.rerun()


def _show_document_feedback() -> None:
    if "document_feedback" not in st.session_state:
        return
    results, errors, warnings, removed, changed = st.session_state.document_feedback
    if results:
        st.success(
            "Document added to your knowledge base." if len(results) == 1
            else "Documents added to your knowledge base."
        )
    for filename, message in errors:
        if filename:
            st.text(filename)
        st.error(message)
    for message in warnings:
        st.warning(message)
    if removed is not None:
        filename, count = removed
        st.text(filename)
        if count:
            st.success("Document removed successfully.")
        else:
            st.info("This document is already absent from the knowledge base.")
    if changed:
        st.caption(_DOCUMENT_CHANGE)


def _upload_controls(collection) -> None:
    with st.container(key="upload_panel"), st.expander("Add document", icon=":material/add:"):
        with st.form("upload", clear_on_submit=True, enter_to_submit=False, border=False):
            uploaded = st.file_uploader("Choose a PDF", type=["pdf"], label_visibility="collapsed")
            st.caption(f"Up to {service.MAX_UPLOAD_MIB} MiB per PDF.")
            submit = st.form_submit_button("Add to knowledge base", disabled=collection is None, type="primary", use_container_width=True)
    if submit and collection is not None:
        if uploaded is None:
            _finish_document_action(errors=[("", "Choose a PDF before adding a document.")])
            return
        if uploaded.size > service.MAX_UPLOAD_BYTES:
            _finish_document_action(errors=[("", f"Upload a PDF no larger than {service.MAX_UPLOAD_MIB} MiB.")])
            return
        warnings = []
        try:
            with st.spinner("Processing PDF and indexing document…"):
                result = service.ingest_uploaded_pdf(
                    uploaded.name, uploaded.getvalue(), collection, on_cleanup_failure=warnings.append,
                )
        except Exception as error:
            _finish_document_action(errors=[("", _index_error(error))], warnings=warnings)
        else:
            _finish_document_action(results=[result], warnings=warnings)

    try:
        demo_pdfs = sorted(
            path for path in config.UPLOAD_DIR.iterdir()
            if path.suffix.casefold() == ".pdf" and not path.is_symlink() and path.is_file()
        ) if config.UPLOAD_DIR.is_dir() else []
    except OSError:
        st.warning("The local demo PDF folder is unavailable.")
        demo_pdfs = []
    with st.container(key="demo_action"):
        index_demos = bool(demo_pdfs) and st.button(
            "Add demo documents", disabled=collection is None, key="index_demos", type="tertiary",
        )
    if index_demos:
        results, errors = [], []
        with st.spinner("Indexing local demo PDFs…"):
            for pdf in demo_pdfs:
                try:
                    results.append(service.ingest_pdf(pdf, collection))
                except (IngestionError, OSError) as error:
                    errors.append((pdf.name, _index_error(error)))
                except Exception:
                    errors.append((pdf.name, "An unexpected error prevented this PDF from being indexed."))
        _finish_document_action(results=results, errors=errors)


def _document_controls(collection, documents) -> None:
    with st.container(key="knowledge_summary"):
        st.subheader("Knowledge Base", anchor=False)
        if documents is None:
            st.error("The knowledge base is unavailable. Check the storage directory and reload the app.")
            return
        count = len(documents)
        st.caption(f"{count} {'document' if count == 1 else 'documents'}")
    if not documents:
        st.info(_EMPTY)
        return
    with st.container(key="document_list", height=300 if count > 6 else None):
        for index, (name, _, _) in enumerate(documents):
            with st.container(key=f"document_row_{index}"):
                st.text(name)

    with st.container(key="document_management"), st.expander("Manage documents", icon=":material/folder_managed:"):
        selected = st.selectbox("Document to remove", [name for name, _, _ in documents], index=None)
        if selected is None:
            return
        with st.form(f"delete:{selected}", clear_on_submit=True, enter_to_submit=False, border=False):
            confirmed = st.checkbox("Remove this document from the knowledge base. Keep the original PDF.")
            submit = st.form_submit_button("Remove document", icon=":material/delete_outline:")
    if submit:
        if not confirmed:
            st.warning("Confirm that you want to remove this document.")
            return
        try:
            with st.spinner("Removing document…"):
                count = delete_document(collection, selected)
        except Exception:
            _finish_document_action(errors=[("", "Could not delete the indexed document. Check the local index and try again.")])
        else:
            _finish_document_action(removed=(selected, count))


def _render_turn(index: int, question: str, result: GroundedAnswer | str) -> None:
    with st.container(key=f"user_turn_{index}"), st.chat_message("user", avatar=":material/person:"):
        st.text(question)
    with st.container(key=f"assistant_turn_{index}"), st.chat_message("assistant", avatar=":material/auto_awesome:"):
        if isinstance(result, str):
            st.error(result)
            return
        if result.status == "not_in_documents":
            st.caption("Insufficient evidence")
        elif result.status == "order_lookup":
            st.caption("Order lookup required")
        st.text(result.answer)
        if result.status == "answered" and result.citations:
            with st.container(key=f"sources_{index}"), st.expander(f"Sources ({len(result.citations)})", icon=":material/description:"):
                for number, citation in enumerate(result.citations):
                    with st.container(key=f"citation_{index}_{number}"):
                        st.text(citation.source_file)
                        st.caption(f"Page {citation.page_number}")
                        with st.container(key=f"quote_{index}_{number}"):
                            st.text(citation.quote)


def _welcome(can_ask: bool) -> str | None:
    selected = None
    with st.container(key="welcome"):
        st.html('<div class="welcome-mark" aria-hidden="true"></div>')
        st.header("Your documents, answered.", anchor=False)
        st.caption("Ask questions, discover insights, and get answers grounded in your documents.")
        with st.container(key="suggestions"):
            for index in range(0, len(_SUGGESTIONS), 2):
                for column, (question, icon) in zip(st.columns(2), _SUGGESTIONS[index:index + 2], strict=True):
                    if column.button(question, icon=icon, use_container_width=True, disabled=not can_ask):
                        selected = question
    return selected


def _submit_question(collection, question: str) -> None:
    if not question.strip():
        st.warning("Enter a question before sending.")
        return
    try:
        with st.spinner("Searching documents and preparing a grounded answer…"):
            result = service.ask_documents(question, collection)
    except Exception as error:
        result = _answer_error(error)
    st.session_state.chat_history.append((question, result))
    st.rerun()


def _render_chat(collection, documents) -> None:
    history = st.session_state.chat_history
    issue = service.chat_configuration_issue()
    can_ask = bool(documents) and issue is None
    with st.container(key="workspace_header"):
        status, actions = st.columns([4, 1], vertical_alignment="center")
        status.title("Document Assistant", anchor=False)
        if documents is None:
            status.caption("Knowledge base unavailable")
        else:
            count = len(documents)
            status.caption(f"{count} {'document' if count == 1 else 'documents'} available")
        if history:
            actions.button("Clear chat", key="clear_chat", type="tertiary", on_click=history.clear)

    with st.container(key="chat_note"):
        if issue is not None:
            st.caption(issue)
        else:
            st.caption("Powered by Gemini · Relevant excerpts are sent to Google to answer questions.")
        if _environment_error:
            st.warning("Could not read the local environment file. Check its permissions and encoding.")

    selected = None
    if history:
        for index, (question, result) in enumerate(history):
            _render_turn(index, question, result)
    else:
        selected = _welcome(can_ask)

    typed = st.chat_input(
        "Ask anything about your documents…" if documents else "Add a document to start asking questions",
        key="question_input", max_chars=service.MAX_QUESTION_CHARS,
        disabled=not can_ask,
    )
    # Native button and chat-input trigger values apply to just this run.
    question = typed if typed is not None else selected
    if can_ask and question is not None:
        _submit_question(collection, question)


def main() -> None:
    st.set_page_config(page_title=_PROJECT_NAME, initial_sidebar_state="auto")
    st.html(_STYLES)
    st.session_state.setdefault("chat_history", [])

    collection = documents = None
    try:
        if "document_collection" not in st.session_state:
            st.session_state.document_collection = open_vector_collection()
        collection = st.session_state.document_collection
        documents = list_documents(collection)
    except Exception:
        # These operations can fail before the session has a usable local index.
        pass

    with st.sidebar:
        st.html(
            '<div class="sidebar-brand"><span class="brand-mark" aria-hidden="true"></span>'
            f'<div><h2>{escape(_PROJECT_NAME)}</h2><p>Your AI document workspace</p></div></div>'
        )
        _upload_controls(collection if documents is not None else None)
        _show_document_feedback()
        _document_controls(collection, documents)
    _render_chat(collection, documents)


if __name__ == "__main__":
    main()
