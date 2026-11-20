"""Shared configuration for Python 3.12; importing it has no side effects.

Paths are relative to the repository location, independent of the working
directory. Environment settings are read at import time; .env files are not
loaded automatically. No models are loaded and no API requests are made.
"""

import os
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
UPLOAD_DIR: Path = PROJECT_ROOT / "data" / "uploads"
CHROMA_STORAGE_DIR: Path = PROJECT_ROOT / "data" / "vector_store"

EMBEDDING_MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"
# No fallback key: an unset variable stays None until future API validation.
GEMINI_API_KEY: str | None = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# Optional executable path. A blank or unset value uses Tesseract on PATH.
TESSERACT_CMD: str | None = os.environ.get("TESSERACT_CMD", "").strip() or None

# Character counts. The embedding stage must check its actual token budget.
CHUNK_SIZE: int = 800
CHUNK_OVERLAP: int = 100

# Retry whole documents at smaller character limits after a token-limit failure.
INGESTION_CHUNK_RETRY_SETTINGS: tuple[tuple[int, int], ...] = (
    (CHUNK_SIZE, CHUNK_OVERLAP),
    (700, 90),
    (600, 80),
    (500, 70),
    (400, 60),
)
