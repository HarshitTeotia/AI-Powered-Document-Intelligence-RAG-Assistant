"""Adapt Streamlit inputs to the existing ingestion and answering APIs."""

import os
import re
import tempfile
import warnings
from collections.abc import Callable
from pathlib import Path

from chromadb.api.models.Collection import Collection

from distribuai import config
from distribuai.ingestion.pipeline import IngestionResult, ingest_pdf
from distribuai.rag.generation import GeminiConfigurationError, GroundedAnswer, generate_grounded_answer
from distribuai.rag.retrieval import retrieve
from distribuai.rag.vector_store import _require_source_filename

MAX_UPLOAD_MIB = 10
MAX_UPLOAD_BYTES = MAX_UPLOAD_MIB * 1024 * 1024
MAX_QUESTION_CHARS = 2_000
_PDF_HEADER = re.compile(rb"%PDF-\d+\.\d+[\r\n]")
UPLOAD_CLEANUP_WARNING = "The temporary PDF could not be removed. Check your system's temporary files."


class UploadError(ValueError):
    """An upload validation or temporary-file error with a user-facing message."""


class QuestionError(ValueError):
    """A question is missing or exceeds the interface limit."""


def validate_upload(filename: str, data: bytes) -> None:
    """Check the upload before writing. The backend parses the actual PDF."""
    try:
        _require_source_filename(filename)
    except (TypeError, ValueError):
        raise UploadError("Use a valid PDF filename without folders or paths.") from None
    if Path(filename).suffix.casefold() != ".pdf":
        raise UploadError("Choose a file with a .pdf extension.")
    if not isinstance(data, bytes) or not data:
        raise UploadError("The uploaded PDF is empty or unreadable.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadError(f"Upload a PDF no larger than {MAX_UPLOAD_MIB} MiB.")
    if _PDF_HEADER.match(data) is None:
        raise UploadError("The uploaded file does not have a valid PDF header.")


def ingest_uploaded_pdf(
    filename: str,
    data: bytes,
    collection: Collection,
    *,
    on_cleanup_failure: Callable[[str], None] | None = None,
) -> IngestionResult:
    """Index a temporary copy while preserving the original citation filename."""
    validate_upload(filename, data)
    try:
        temp_root = Path(tempfile.gettempdir()).resolve()
        if temp_root.is_relative_to(config.PROJECT_ROOT.resolve()):
            raise UploadError("Configure a system temporary directory outside the project.")
        directory = tempfile.TemporaryDirectory(prefix="distribuai-upload-", dir=temp_root)
    except OSError:
        raise UploadError("Could not create a temporary directory for the upload.") from None

    try:
        path = Path(directory.name) / "upload.pdf"
        try:
            path.write_bytes(data)
        except OSError:
            raise UploadError("Could not write the temporary PDF.") from None
        return ingest_pdf(path, collection, source_name=filename)
    finally:
        try:
            directory.cleanup()
        except OSError:
            if on_cleanup_failure is None:
                warnings.warn(UPLOAD_CLEANUP_WARNING, RuntimeWarning, stacklevel=2)
            else:
                on_cleanup_failure(UPLOAD_CLEANUP_WARNING)


def chat_configuration_issue() -> str | None:
    """Report the local prerequisites without exposing the configured credential."""
    if not config.gemini_chat_enabled():
        return "AI chat is turned off. Enable Gemini chat in the app's local configuration."
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        return "AI chat needs a Gemini API key in the app's local configuration."
    return None


def ask_documents(question: str, collection: Collection) -> GroundedAnswer:
    """Use the retrieved passages for this question only."""
    if not isinstance(question, str) or not question.strip():
        raise QuestionError("Enter a question before sending.")
    if len(question) > MAX_QUESTION_CHARS:
        raise QuestionError(f"Keep the question within {MAX_QUESTION_CHARS:,} characters.")
    issue = chat_configuration_issue()
    if issue is not None:
        raise GeminiConfigurationError(issue)
    # Retrieval handles an empty collection without loading the embedding model.
    # Generation handles empty evidence locally without creating a Gemini client.
    chunks = retrieve(collection, question, top_k=5)
    return generate_grounded_answer(question, chunks)
