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
local chunk embeddings, persistent vector storage, PDF ingestion, semantic retrieval,
retrieval evaluation, and grounded Gemini answers from supplied excerpts.
**Planned:** document uploads through Streamlit, a chat interface, and end-to-end
answer-quality evaluation.
There is no functional chatbot or runnable Streamlit application yet.

## Intended users

- Customer-support agents at distribution businesses.
- Internal staff who need to locate company document information.

## Planned features

- Upload company PDFs through the application.
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
| python-dotenv | 1.2.1 | [Gemini .env loading](https://pypi.org/project/python-dotenv/1.2.1/) |

These are explicit release pins, not a claim that they are the latest releases.
Release existence for the two Gemini setup additions was verified against the
linked PyPI release pages. Full dependency resolution is not verified.
Direct pins do not lock the entire transitive dependency graph.

OCR uses the Tesseract system executable. Answer generation uses Google's
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

`distribuai/config.py` defines repository-relative upload and vector-store paths,
the embedding model identifier, `GEMINI_API_KEY` read from `os.environ` without
a fallback key, and `GEMINI_MODEL`
(default `gemini-2.5-flash`, overridden by the environment variable of the same
name). Generation validates the key when it needs a Gemini client.
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
- Internet access and a valid personal Gemini API key for answer generation.
- Sufficient disk space for the ML packages and, in later phases, model files.

OCR requires a separate Tesseract installation with English language data
(`eng.traineddata`). Tesseract is optional for PDFs with sufficient native text.
The embedding model `sentence-transformers/all-MiniLM-L6-v2` will need a separate
download later. Setup does not download models or install system tools. Gemini
answer generation requires internet access. API quotas and rate limits depend on
the selected model and project tier.
[Current Gemini rate limits](https://ai.google.dev/gemini-api/docs/rate-limits).

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
content or chunk settings can produce new IDs. Use `ingest_pdf()` to replace
stale chunks after a successful upsert.

The persistent directory contains document text, embeddings, and citation metadata.
Protect it using the same local access controls as the source PDFs. Its contents
are Git-ignored to keep document data and local database files out of the repository.

### PDF ingestion and document management

`ingest_pdf(pdf_path, collection, source_name=None)` runs extraction with OCR
fallback, page-aware chunking, embedding, and ChromaDB storage. The citation
filename defaults to the PDF basename. An explicit filename is a separate
document identity if it differs from the basename. Filenames and original
1-based page numbers are preserved for citations.

The default chunk size and overlap are 800/100 characters. If embedding detects
an oversized chunk, ingestion re-chunks the entire document at 700/90, 600/80,
500/70, and 400/60 in that order until one configuration succeeds. This keeps
one chunk configuration per stored version. No text is silently truncated, and
even 400 characters can exceed the model's token limit for some inputs.

`IngestionResult` reports the filename, original page count, stored chunk count,
successful chunk settings, OCR page count, stale chunks removed, and empty page
numbers. A separate native extraction pass counts pages flagged for OCR because
the OCR function does not expose its attempt count. The extra pass preserves the
existing API and adds extraction overhead.

Stable IDs make identical re-ingestion logically idempotent. For edited PDFs,
ingestion confirms that the new IDs are stored before deleting stale IDs under
the same filename. Upserts and deletes are separate operations. A failed upsert
can leave partial new writes, and a failed cleanup can leave both versions.
Serialize operations for the same filename. Stronger atomic replacement would
require additional design.

`list_documents()` returns each filename, its stored chunk count, and its sorted
distinct pages. `delete_document(collection, source_file)` removes only that
filename's stored chunks and returns the number removed. The original PDF is
kept. Uploaded PDFs and local vector data remain Git-ignored. Streamlit uploads
are planned.

### Semantic retrieval

`retrieve(collection, question, top_k=5, source_file=None)` returns up to five
ranked passages by default. Set `source_file` to an exact filename to search only
that document. The question and stored passages use the same all-MiniLM-L6-v2
model and normalized vectors, so scores compare direction in the same vector
space. Questions that exceed the model's token limit are rejected before encoding.

Lower cosine distance means a closer match. Cosine similarity is `1 - distance`.
Each result includes both scores, its rank, the original text, and citation
coordinates such as the filename and page number. The closest passages can still
be irrelevant. Similarity scores alone cannot establish answerability.

### Gemini grounded answer generation

Pass a question and up to five passages from semantic retrieval to
`generate_grounded_answer(question, retrieved_chunks)` in
`distribuai.rag.generation`. The Gemini API receives the question and actual
excerpts. The system instruction asks Gemini to use only the evidence and treat
all PDF text and source labels as untrusted data. It assesses the excerpts without
a similarity cutoff.

Python validates Gemini's structured JSON and checks every supporting quote
against the cited chunk's exact text. It builds citations using the original chunk
ID, filename, and page. A response with insufficient evidence gets a fixed refusal.
A question about a specific order's live status or details gets a message saying
that an authorized order-system lookup is required. Empty retrieval returns a
refusal locally. Configuration, API, and response failures raise distinct errors.

Generation lazily loads the project `.env` with `python-dotenv`. Existing
environment variables take precedence. `GEMINI_API_KEY` supplies the credential
and `GEMINI_MODEL` overrides the model configured in `distribuai/config.py`.
Importing the module requires neither a key nor an API request.

Exact quote matching verifies that the quote appears in its source. Whether it
supports every claim in the answer still requires review. Separating instructions
from untrusted data and checking citations reduce prompt-injection risk, but a
model can still be misled. Use the fictional sample PDFs for live verification.
Sending private or sensitive document excerpts to Google requires appropriate
authorization and privacy controls. A Streamlit interface, threshold calibration,
and end-to-end answer-quality evaluation are planned separately.

For a live smoke check, activate the Python 3.12 environment from the project
root and configure the key. In a child Python process, ingest the four fictional
PDFs into a temporary ChromaDB directory created by PowerShell outside the
repository. Retrieve five passages per question and check returns and credit
terms for `answered`, a question about an automatic bulk discount for
`not_in_documents`, and a dispatch question for a specific order for `order_lookup`.
Check the quotes and page numbers for both answered questions, then remove the
temporary directory from PowerShell after Python exits. Skip the live check if a
key is unavailable.

`.env.example` contains blank entries for `GEMINI_API_KEY` and `TESSERACT_CMD`.
Copy it to an ignored local file and edit that file privately:

```powershell
Copy-Item .env.example .env
```

Set `GEMINI_API_KEY` in that file or the process environment. Optionally set
`GEMINI_MODEL` to a model available to your API project. Keep real keys out of Git
and terminal output. The setup and generation unit tests require neither a key
nor network access.

## Running tests

With the Python 3.12 virtual environment active:

```powershell
python -m pytest -q
```

The suite verifies the scaffold, PDF extraction, OCR, chunking, embeddings, local
storage, retrieval, and Gemini generation. Generation tests use mocked clients.
The regular OCR tests simulate the executable. One optional smoke test uses a
synthetic PDF and real Tesseract, and skips if the executable is unavailable.

## Evaluation approach

The manual retrieval check uses the fictional sample PDFs and `eval/questions.csv`
to measure source hits and evidence coverage. Page hits and reciprocal rank
require labeled pages. Answer grounding, citation accuracy in generated answers,
and refusal on insufficient evidence belong to later evaluation.

### Running the retrieval evaluation

From the repository root in PowerShell with the virtual environment active:

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TESSERACT_CMD = 'C:\Program Files\Tesseract-OCR\tesseract.exe'

python -B eval/evaluate_retrieval.py
python -B eval/evaluate_retrieval.py --csv eval/questions.csv --csv eval/questions_extra.csv
python -B eval/evaluate_retrieval.py --csv eval/questions.csv --csv eval/questions_extra.csv --uploads data/uploads --top-k 5
```

The default uses only `eval/questions.csv`. Repeat `--csv` to choose the files to
evaluate, with unique question IDs across them. The combined examples require
`eval/questions_extra.csv` to be present. `--uploads` selects which directory to
scan for PDFs. `--top-k` changes the retrieval depth, with a minimum of five.
Hit@1/@3/@5 and evidence coverage@5 keep their fixed cutoffs. MRR uses the first
accepted file and page within the retrieved depth. Unlabelled questions still
count for file hits and evidence coverage.

The evaluator uses the real ingestion pipeline and cached embedding model in
a temporary ChromaDB index outside the repository. It reports source hits,
evidence coverage, and per-question similarities, including unsupported and
order-lookup questions. The parent checks PDF and CSV hashes and cleans up after
the database-owning child exits. The main index stays untouched. The report
measures retrieval. End-to-end answer quality and similarity threshold calibration
remain separate tasks.

## Security and responsible-AI considerations

Customer uploads, vector-store contents, secrets, virtual environments, and local
model caches are excluded from Git; `.gitkeep` files preserve data directories.
Git ignore rules are not access controls or encryption. Store sensitive documents
only on appropriately protected machines and define retention/deletion practices
before adding real data.

Gemini generation sends questions and retrieved document text to Google. Review
customer consent, data sensitivity, and Google's applicable data-use terms before
enabling real-data use. Production safeguards still require review, including
human review of customer-facing answers and dependency telemetry and network behavior.

## Known limitations

- PDF extraction, OCR, chunking, embedding, local storage, retrieval, and grounded
  generation are available as Python functions. The caller supplies the passages.
- No models or system tools were installed or downloaded.
- Gemini requires an API key and internet access, is subject to rate limits, and
  receives retrieved document text during generation.
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
3. Chunking, embeddings, and local vector persistence (complete).
4. Retrieval and Gemini API answers with source citations (Python APIs complete).
5. Streamlit support workflow.
6. Evaluation and security/responsible-AI review.

Each later phase requires a separate task and review.
