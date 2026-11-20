# DistribuAI

## Business problem

Distribution support teams need to find reliable answers across company PDFs,
including scanned documents. Manual searches can slow responses and make it
difficult to identify the document supporting an answer.

## Project overview

DistribuAI is a planned document intelligence and customer-support assistant with
local document processing and cloud-based answer generation through the Gemini API.
The intended system will use Retrieval-Augmented Generation (RAG) to answer
questions using company documents and cite its sources.

**Implemented:** project setup, PDF text extraction with selective OCR, text chunking,
local chunk embeddings, persistent vector storage, and semantic retrieval.
**Planned:** document uploads, grounded answers, and the interface.
There is no functional chatbot or runnable Streamlit application yet.

## Intended users

- Customer-support agents at distribution businesses.
- Internal staff who need to locate company document information.

## Planned features

- Upload company PDFs through the application.
- Chunk document text and embed it locally.
- Generate answers with source citations from retrieved passages.
- Provide a Streamlit interface and evaluate answer quality.

## Locked technology stack

Python 3.12 and the following ten direct Python dependencies are approved.
Required transitive dependencies are installed by pip; no additional direct
dependencies are included.

| Dependency | Version | Planned use / release metadata |
| --- | --- | --- |
| Streamlit | 1.45.1 | [UI](https://pypi.org/project/streamlit/1.45.1/) |
| pdfplumber | 0.11.6 | [PDF text extraction](https://pypi.org/project/pdfplumber/0.11.6/) |
| pytesseract | 0.3.13 | [OCR wrapper](https://pypi.org/project/pytesseract/0.3.13/) |
| pypdfium2 | 5.15.2 | [PDF page rendering](https://pypi.org/project/pypdfium2/5.15.2/) |
| sentence-transformers | 4.1.0 | [Local embeddings](https://pypi.org/project/sentence-transformers/4.1.0/) |
| ChromaDB | 1.0.9 | [Local vector storage](https://pypi.org/project/chromadb/1.0.9/) |
| Pandas | 2.2.3 | [Evaluation and analysis](https://pypi.org/project/pandas/2.2.3/) |
| Pytest | 8.3.5 | [Testing](https://pypi.org/project/pytest/8.3.5/) |
| google-genai | 1.51.0 | [Gemini API SDK](https://pypi.org/project/google-genai/1.51.0/) |
| python-dotenv | 1.2.1 | [Future .env loading](https://pypi.org/project/python-dotenv/1.2.1/) |

These are explicit release pins, not a claim that they are the latest releases.
Release existence for the two Gemini setup additions was verified against the
linked PyPI release pages. Full dependency resolution is not verified.
Direct pins do not lock the entire transitive dependency graph.

OCR uses the Tesseract system executable. Answer generation will use Google's
Gemini API with the user's own API key.
Git/GitHub is the intended version-control workflow; setup initializes only local
Git, without a remote, commit, or push.

## Planned architecture

```text
Company PDFs -> pdfplumber text extraction
             -> pages with low text: pypdfium2 rendering -> Tesseract via pytesseract
Extracted text -> chunks with source metadata
              -> all-MiniLM-L6-v2 embeddings -> local ChromaDB
Question -> local retrieval -> Google Gemini API -> answer with citations
Streamlit -> upload and question/answer interface
Pandas -> evaluation and analysis
```

PDF extraction, OCR, and chunking are implemented. Configuration lives in
`distribuai/config.py`:
repository-relative upload and vector-store paths, the embedding model identifier,
`GEMINI_API_KEY` read from `os.environ` without a fallback key, and `GEMINI_MODEL`
(default `gemini-2.5-flash`, overridden by the environment variable of the same
name). An unset key remains `None`; validation is deferred to future API integration.
Importing configuration does not load `.env` files, create directories, load
models, or contact services.

## Project structure

The current repository folder is the project root; no nested root is required.

```text
.
|-- distribuai/
|   |-- __init__.py
|   |-- config.py
|   |-- ingestion/
|   |   `-- __init__.py
|   |-- rag/
|   |   `-- __init__.py
|   `-- ui/
|       `-- __init__.py
|-- data/
|   |-- uploads/
|   |   `-- .gitkeep
|   `-- vector_store/
|       `-- .gitkeep
|-- tests/
|   `-- test_setup.py
|-- requirements.txt
|-- .env.example
|-- .gitignore
`-- README.md
```

## Prerequisites

- Windows with PowerShell, Python 3.12 (64-bit), pip, and Git.
- Network access to install Python packages during local setup.
- Internet access and a valid personal Gemini API key for future answer generation.
- Sufficient disk space for the ML packages and, in later phases, model files.

OCR requires a separate Tesseract installation with English language data
(`eng.traineddata`). Tesseract is optional for PDFs with sufficient native text.
The embedding model `sentence-transformers/all-MiniLM-L6-v2` will need a separate
download later. Setup does not download models or install system tools. Gemini
answer generation will require internet access; the application will not be fully
offline. API quotas and rate limits depend on the selected model and project tier;
check the [current Gemini rate limits](https://ai.google.dev/gemini-api/docs/rate-limits).

## Local setup instructions

Run these commands from this repository's root in Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install -r requirements.txt
```

Confirm that `python --version` reports Python 3.12 before installing packages.
If activation is blocked by a PowerShell execution policy, use the virtual
environment interpreter directly without changing that policy:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

To check resolution for the active Python 3.12 environment without installing:

```powershell
python -m pip install --dry-run --ignore-installed -r requirements.txt
```

There is no application launch command in this phase because no UI exists yet.

### PDF extraction and OCR

`extract_pdf_with_ocr(Path)` reads native text with pdfplumber first. Only pages
with fewer than 20 characters after trimming are rendered at approximately 300 DPI.
Tesseract runs in English with a 30-second timeout per page. The result preserves
page order and 1-based page numbers, keeps the longer text, and recalculates the
OCR flag. OCR can misread numbers, tables, or poor-quality scans, and longer text
does not guarantee a more accurate result.

Set optional `TESSERACT_CMD` to the executable path, including any spaces, or
leave it unset or blank to use Tesseract from `PATH`. Configuration reads the
process environment at import time. To use a value from `.env`, the caller must
load that file into the environment before importing DistribuAI configuration,
for example by calling `load_dotenv()` from the entry point.

OCR requests through this function run one at a time within the process. This
protects the PDFium calls and the temporary executable setting in pytesseract.
Other code using either library directly would need to coordinate with this path
before running in concurrent Streamlit sessions.

### Text cleaning and chunking

`clean_text()` normalizes line endings, spaces, tabs, and excess blank lines
while preserving useful line breaks, paragraphs, punctuation, and values such as
dates, prices, and business IDs. `chunk_pdf_pages()` divides the cleaned pages into
smaller passages for future retrieval and citations. It skips empty pages and
keeps short, nonempty pages even when OCR is still flagged.

The defaults are `CHUNK_SIZE = 800` and `CHUNK_OVERLAP = 100`, both in characters.
Paragraph and sentence breaks help keep related text together. If neither fits,
chunks break at whole words. A token longer than the limit, such as a long URL,
is the exception and is hard-split into separate pieces. Overlap carries a suffix
of whole words forward to preserve context when the next chunk has room for new
text. Each chunk stays on one page and records the filename without directories,
the original 1-based page number, and its 1-based index on that page.

Stable chunk IDs use a SHA-256 fingerprint of the filename, ordered cleaned pages,
and both chunk settings, followed by page and chunk indexes. This supports future
deduplication. Uploaded-file identity and replacement belong to the storage stage.
The character limits were calibrated on the four sample PDFs. They cannot guarantee
a token count for new documents, so the embedding stage validates actual token
counts against the model's sequence limit.

### Local chunk embeddings

An embedding is a numeric representation of a passage. `embed_chunks()` converts
document chunks to 384-dimensional vectors so future retrieval can compare passages
by meaning. DistribuAI uses the compact `sentence-transformers/all-MiniLM-L6-v2`
model for local inference. Embeddings are normalized so similarity comparisons can
use vector direction independently of magnitude. Each record keeps the original
chunk ID, filename, and page number for future citations.

The first real run may download model files to the normal user cache. The model
loads on demand and is reused in the process. It can run offline once cached.
Its input limit is about 256 tokens, including special tokens. The same number of
characters can tokenize to different lengths. The module counts tokens with the
model's tokenizer before it encodes any batch and rejects oversized chunks. Use a
smaller chunk size when rechunking a document that exceeds the limit.

Embedding unit tests use a mock model. For a manual check with the cached model,
run this from the project root in PowerShell with the virtual environment active:

```powershell
$env:HF_HUB_OFFLINE = '1'
@'
from math import isfinite
from distribuai.rag.chunking import DocumentChunk
from distribuai.rag.embeddings import embed_chunks

record, = embed_chunks([DocumentChunk('smoke-1', 'sample.pdf', 1, 1, 'Orders ship on weekdays.')])
assert len(record.vector) == 384 and all(map(isfinite, record.vector))
print('PASS: 384 finite embedding values')
'@ | python -B -
```

Embedding vectors are returned in memory and can be passed to the local store.

### Local vector storage

ChromaDB stores vectors, text, and citations together on disk so they survive a
restart. `open_vector_collection()` creates or reopens `distribuai_documents` in
`data/vector_store/` by default. Pass a `Path` to use a separate directory for
testing. The collection uses cosine distance for comparisons between the
normalized embeddings. It validates the configured all-MiniLM-L6-v2 model and
384 dimensions on open and accepts the existing vectors directly, with ChromaDB's
embedding function disabled.

`store_embedded_chunks()` upserts by deterministic chunk ID. Reprocessing identical
content with the same filename and chunk settings updates existing IDs rather than
increasing the record count. Records store the original text with the source
filename, 1-based page number, 1-based chunk index, and token count. Modified
content or chunk settings can produce new IDs. Previous versions remain in the
collection until document management is added. Batches are separate writes, so
earlier batches may remain stored if a later batch fails.

The persistent directory contains document text, embeddings, and citation metadata.
Protect it using the same local access controls as the source PDFs. Its contents
are Git-ignored to keep document data and local database files out of the repository.
### Semantic retrieval

`retrieve(collection, question, top_k=5, source_file=None)` returns up to five
ranked passages by default. Set `source_file` to an exact filename to search only
that document. The question and stored passages use the same all-MiniLM-L6-v2
model and normalized vectors, so scores compare direction in the same vector
space. Questions that exceed the model's token limit are rejected before encoding.

Lower cosine distance means a closer match. Cosine similarity is `1 - distance`.
Each result includes both scores, its rank, the original text, and citation
coordinates such as the filename and page number. The closest passages can still
be irrelevant. A future similarity threshold needs evaluation on labeled examples
before it can support decisions about insufficient evidence. Retrieval returns
passages for review. Gemini answer generation is planned.

`.env.example` contains blank entries for `GEMINI_API_KEY` and `TESSERACT_CMD`.
Copy it to an ignored local file and edit that file privately:

```powershell
Copy-Item .env.example .env
```

The config module reads the process environment only. Merely creating `.env` does
not load its contents; automatic loading with the approved `python-dotenv` package
is planned for a later phase. Set `GEMINI_API_KEY` in the process environment before
importing configuration when needed, and optionally set `GEMINI_MODEL` to a model
available to your API project. Do not commit or print real keys. The setup test
requires neither a key nor network access.

## Running tests

With the Python 3.12 virtual environment active:

```powershell
python -m pytest -q
```

The suite verifies the scaffold, PDF extraction, OCR, chunking, embeddings, local
storage, and retrieval. The regular OCR tests simulate the executable. One optional smoke test uses a
synthetic PDF and real Tesseract, and skips if the executable is unavailable.

## Evaluation approach

The manual retrieval check uses the fictional sample PDFs and `eval/questions.csv`
to measure source hits and evidence coverage. Page hits and reciprocal rank
require labeled pages. Answer grounding, citation accuracy in generated answers,
and refusal on insufficient evidence belong to later evaluation.

## Security and responsible-AI considerations

Customer uploads, vector-store contents, secrets, virtual environments, and local
model caches are excluded from Git; `.gitkeep` files preserve data directories.
Git ignore rules are not access controls or encryption. Store sensitive documents
only on appropriately protected machines and define retention/deletion practices
before adding real data.

Future work must treat retrieved document text as untrusted input, address prompt
injection, support insufficient-evidence responses, and retain human review for
customer-facing answers. In the planned Gemini integration, questions and retrieved
document text will be sent to Google for answer generation. Review customer consent,
data sensitivity, and Google's applicable data-use terms before enabling real-data
use. Dependency telemetry and network behavior must also be reviewed.
These application safeguards are planned, not implemented.

## Known limitations

- PDF extraction, OCR, chunking, embedding, local storage, and retrieval are
  available as Python functions. LLM calls, answer citations, and the UI are planned.
- No models or system tools were installed or downloaded.
- Gemini requires an API key and internet access, is subject to rate limits, and
  sends retrieved document text to Google. API requests are not implemented yet.
- Python 3.12 setup testing does not verify full dependency compatibility.
- OCR accuracy, answer quality, hardware performance, and production suitability
  have not been evaluated.

### Setup validation

Validation on November 20, 2026:

- The two added pins were verified on their PyPI release pages. The eight original
  pins are unchanged; the requirements file now has ten direct dependencies.
- `python -m pytest -q` passed with **1 passed, 1 warning in 0.03s** using the available
  Python 3.12.10 interpreter and pre-existing Pytest 9.1.1. The default `python`
  command points to an inaccessible Windows alias, so the available interpreter
  was invoked by its full path. The warning reports denied access when writing
  the existing Pytest cache; the test itself passed. Testing with pinned Pytest
  8.3.5 remains outstanding.
- Full dependency resolution of the updated requirements is **unverified**. No
  package installation was performed. The earlier setup resolution attempt timed
  out at `files.pythonhosted.org`; it does not validate these updated requirements.
- The original setup checked data placeholders and generated-data ignore rules.
  `.env` remains ignored, with an explicit exception to track `.env.example`.
  The correction checks passed for absent/present environment settings, the model
  override, ten dependency pins, the blank example, and the single existing test.
  Git was initialized by the sandbox account;
  checks from the host account used a per-command `safe.directory` exception for
  this repository. No global Git setting was changed. If Git reports dubious
  ownership, review the repository ownership before trusting it on your account.

## Development roadmap

1. Project scaffold and setup validation (complete).
2. PDF extraction and selective OCR (complete).
3. Chunking, embeddings, and local vector persistence.
4. Retrieval and Gemini API answers with source citations.
5. Streamlit support workflow.
6. Evaluation and security/responsible-AI review.

Each later phase requires a separate task and review.
