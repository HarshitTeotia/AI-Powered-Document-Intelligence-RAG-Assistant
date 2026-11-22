## Business problem

Distribution support teams search company PDFs, including scans, to find and
verify policies for customer questions.
The assistant gives support staff a way to look up shipping, returns, products, and
payment policies with source quotations. A customer service manager can use the
[pilot measurement plan](docs/TASK_1_FINAL_REPORT.md#illustrative-business-value) to assess time savings and quality. The estimates are illustrative.

## Project overview

This local document assistant uses retrieval-augmented generation (RAG) to find
relevant passages in company PDFs and draft answers with citations. Document
processing and retrieval run locally. Answer generation uses the Gemini API
when enabled.

| Component | Status | Module |
| --- | --- | --- |
| Native PDF extraction with detected table rows on one line | Implemented | distribuai.ingestion.pdf_extractor |
| OCR fallback for sparse pages | Implemented | distribuai.ingestion.ocr_fallback |
| Page-aware chunking with deterministic IDs | Implemented | distribuai.rag.chunking |
| Local embeddings with strict token checks | Implemented | distribuai.rag.embeddings |
| Persistent vector storage with idempotent upserts | Implemented | distribuai.rag.vector_store |
| Cosine-ranked retrieval | Implemented | distribuai.rag.retrieval |
| Ingestion with smaller-chunk retries and stale-chunk replacement | Implemented | distribuai.ingestion.pipeline |
| Document listing and deletion | Implemented | distribuai.rag.vector_store |
| Gemini answers with validated source quotations and the statuses `answered`, `not_in_documents`, and `order_lookup` | Implemented | distribuai.rag.generation |
| Streamlit document management and chat | Implemented | distribuai.ui.app |
| Retrieval evaluation | Implemented | eval.evaluate_retrieval |
| Offline generation replay | Implemented | eval.evaluate_generation |
| Pipeline benchmark | Implemented | eval.benchmark_pipeline |
| Live order lookup | Not implemented | Generation returns `order_lookup` when Gemini flags a live order question. No order system is queried. |
| Separate question router, similarity threshold, user authentication | Not implemented | — |

## Intended users

Customer-support agents and other staff at distribution businesses who need to find company policies.

## Technology stack

Use Python 3.12 and the [pinned dependencies](requirements.txt). The app uses Streamlit,
pdfplumber and Tesseract, local all-MiniLM-L6-v2 embeddings, ChromaDB, and the Gemini API.

<details>
<summary>Dependency pins and implementation tools</summary>

| Dependency | Version | Use |
| --- | --- | --- |
| Streamlit | 1.45.1 | UI |
| pdfplumber | 0.11.6 | PDF text extraction |
| pytesseract | 0.3.13 | OCR wrapper |
| pypdfium2 | 5.15.2 | PDF page rendering |
| sentence-transformers | 4.1.0 | Local embeddings |
| ChromaDB | 1.0.9 | Local vector storage |
| Pandas | 2.2.3 | Evaluation and analysis |
| Pytest | 8.3.5 | Testing |
| google-genai | 1.51.0 | Gemini API SDK |
| python-dotenv | 1.2.1 | Gemini .env loading |

</details>

## Architecture

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

<details>
<summary>Project structure</summary>

Tracked files for the Task 1 submission, shown to depth two.

```text
.
|-- .env.example  # Environment settings template
|-- .gitignore  # Local artifact exclusions
|-- .streamlit/  # Streamlit settings
  |-- config.toml  # Upload limit and interface theme
|-- README.md  # Setup and reviewer instructions
|-- data/  # Local data directories
  |-- uploads/  # Placeholder for approved PDFs
  |-- vector_store/  # Placeholder for the local index
|-- distribuai/  # Application source
  |-- __init__.py  # Package marker
  |-- config.py  # Paths and runtime settings
  |-- evaluation/  # Retrieval scoring
  |-- ingestion/  # PDF extraction, OCR, and ingestion
  |-- rag/  # Chunking, embeddings, storage, retrieval, and generation
  |-- ui/  # Streamlit interface and service functions
|-- docs/  # Task 1 documentation
  |-- AI_TOOLS_DISCLOSURE.md  # AI tools disclosure
  |-- DATA_SOURCE_MANIFEST.md  # Source hashes and sharing instructions
  |-- TASK_1_DATASET_AUDIT.md  # Review of question labels
  |-- TASK_1_DEMO_GUIDE.md  # Demo walkthrough
  |-- TASK_1_FINAL_REPORT.md  # Design, evaluation, and limitations
  |-- TASK_1_VALIDATION_EVIDENCE.md  # Recorded validation evidence
|-- eval/  # Evaluation data and scripts
  |-- benchmark_pipeline.py  # Pipeline timing benchmark
  |-- evaluate_generation.py  # Offline response replay
  |-- evaluate_retrieval.py  # Retrieval benchmark
  |-- fixtures/  # Saved mock responses
  |-- questions.csv  # Labeled evaluation questions
|-- pytest.ini  # Test discovery configuration
|-- requirements.txt  # Pinned Python dependencies
|-- tests/  # Automated tests
  |-- test_chunking.py  # Chunking tests
  |-- test_embeddings.py  # Embeddings tests
  |-- test_generation.py  # Generation tests
  |-- test_generation_evaluation_cli.py  # Generation evaluation cli tests
  |-- test_ingestion_pipeline.py  # Ingestion pipeline tests
  |-- test_ocr_fallback.py  # Ocr fallback tests
  |-- test_pdf_extractor.py  # Pdf extractor tests
  |-- test_retrieval.py  # Retrieval tests
  |-- test_retrieval_evaluation_cli.py  # Retrieval evaluation cli tests
  |-- test_retrieval_scoring.py  # Retrieval scoring tests
  |-- test_setup.py  # Setup tests
  |-- test_ui_app.py  # Ui app tests
  |-- test_ui_service.py  # Ui service tests
  |-- test_vector_store.py  # Vector store tests
```

</details>

## Prerequisites

- Python 3.12 (64-bit), pip, and Git. The commands below use Windows PowerShell.
- Internet access to install dependencies and download the local embedding model
  on first use. Live Gemini chat also requires internet access and an API key.
- Tesseract with English language data for pages that need OCR.

## Local setup instructions

Run from the repository root.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` in place of
`python`. Keep credentials in the process environment or the ignored local
`.env`, using [.env.example](.env.example) as a template.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEY` | Unset | Credential for Gemini requests. |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model to request. |
| `GEMINI_CHAT_ENABLED` | Disabled | Set to `true` to enable chat. Requests with evidence send the question and retrieved document excerpts to Google. |
| `TESSERACT_CMD` | Unset, uses `tesseract` from `PATH` | Optional path to the Tesseract executable. |
| `HF_HUB_OFFLINE` | Unset | Set to `1` to restrict Hugging Face Hub to cached files. |
| `TRANSFORMERS_OFFLINE` | Unset | Set to `1` for offline Transformers use. |
| `PYTHON_DOTENV_DISABLED` | Unset | The benchmark sets it to `1` in its child processes to suppress dotenv loading. |

The Streamlit entry point loads `.env` before importing the configuration.
The generation module also loads it when resolving settings. Importing
`distribuai.config` reads the process environment and does not load `.env`.
Existing process settings take precedence. Restart Streamlit after changing
the environment.

## Running the Streamlit application

From the project root with the virtual environment active:

```powershell
python -m streamlit run distribuai/ui/app.py --server.address 127.0.0.1
```

- **Add PDFs.** Open **Add document**, choose a PDF up to 10 MiB, and select
  **Add to knowledge base**. **Add demo documents** indexes PDFs already in
  `data/uploads/`. Adding a PDF under the same filename updates its entry.
- **Manage documents.** Open **Manage documents**, select a file, and confirm
  removal. This deletes indexed passages and keeps the original PDF. ChromaDB
  stores text, vectors, and citation metadata in `data/vector_store/`.
- **Ask and verify.** With chat enabled and a key configured, enter a question
  or choose a suggested question. Open **Sources** to inspect the filename,
  page, and supporting quotation. **Clear chat** clears the current session's
  messages. Changing the knowledge base clears that session's chat.

For the supervised walkthrough, use approved copies of the sample PDFs and
follow the [demo guide](docs/TASK_1_DEMO_GUIDE.md#prepare). Use an approved or
synthetic PDF to try document removal without disturbing the benchmark sources.
The local server health check passed. Live Gemini chat was not executed in this cleanup.

## Performance notes

The embedding model loads on first use and is reused within the process.
Ingestion reads native text once and applies OCR to the flagged pages.
The [pipeline benchmark](eval/benchmark_pipeline.py) uses temporary indexes
and the four sample PDFs. The command below skips network timing and uses
simulated Gemini responses. The offline benchmark completed. Network timing
was not executed in this cleanup.

```powershell
python -B eval/benchmark_pipeline.py --skip-network-check
```

## Source documents and reviewer setup

The four local fictional PDFs in `data/uploads/` are Git-ignored. A fresh checkout
does not include them. Obtain the approved copies listed in the
[source inventory](docs/TASK_1_FINAL_REPORT.md#sources-and-reproducibility) and
place them in that directory to reproduce the reported retrieval results. For a
basic ingestion demo, add an approved or explicitly synthetic PDF via the UI.
A different source set needs its own benchmark. The evaluation CSV and offline
generation replay fixtures are included with the project files.

### PDF extraction and OCR

`extract_pdf_with_ocr()` extracts native text with pdfplumber first and renders
pages with fewer than 20 trimmed characters at 300 DPI. Tesseract runs in
English with a 30-second timeout per page. The fallback keeps the longer text
and preserves the original 1-based page number.

### Text cleaning and chunking

`clean_text()` normalizes whitespace and keeps useful line breaks.
`chunk_pdf_pages()` splits within a page, skips empty pages, and retains the
filename, page number, and chunk index. The default size and overlap are 800
and 100 characters. Chunk IDs are derived from the filename, ordered cleaned
page text, chunk settings, and page and chunk indices.

### Local chunk embeddings

`embed_chunks()` and `embed_query()` use the local
`sentence-transformers/all-MiniLM-L6-v2` model to produce normalized
384-dimensional vectors. The tokenizer checks the model's input limit before
encoding. A cached model can run offline.

### Local vector storage

`open_vector_collection()` opens or creates the persistent
`distribuai_documents` collection. ChromaDB compares vectors with cosine
distance and stores the passage text and citation metadata. Reprocessing the
same content, filename, and chunk settings upserts the same IDs.

### PDF ingestion and document management

`ingest_pdf()` runs extraction, selective OCR, chunking, embedding, and storage.
It extracts native text once and counts the pages flagged for OCR from that
pass. A token-limit failure triggers a retry of the entire document with smaller
chunks. The pipeline confirms the new IDs were stored before removing stale
chunks for the same filename. Upserts and deletes are separate operations.
`list_documents()` reports indexed filenames, chunk counts, and pages.
`delete_document()` removes a filename's indexed chunks.

### Semantic retrieval

`retrieve()` returns up to five passages by default, ranked by cosine distance.
It accepts an exact filename filter and returns similarity, rank, text, and
citation coordinates. Similarity alone does not establish answerability.

### Gemini grounded answer generation

`generate_grounded_answer()` sends the current question and up to five retrieved
passages to Gemini. It validates structured JSON and accepts quotations that
match the supplied chunk exactly or match a unique span after normalizing
whitespace. It returns the original quotation and source coordinates. Empty
retrieval returns an insufficient-evidence response locally. The
`not_in_documents` and `order_lookup` statuses use fixed messages. A specific
order's details require access to an authorized order system.

## Running tests

With the Python 3.12 environment active:

```powershell
python -m pytest -q -p no:cacheprovider
```

Cleanup test run: **499 passed**, with **104 ChromaDB deprecation warnings**.
Generation tests use mocked clients. The Tesseract smoke test skips when the executable is unavailable.

## Evaluation approach

Measured by the author on fictional documents the author reports writing: file
hit@1 was **20/20** on answerable questions and page hit@1 was **14/15** on
questions with page labels. These figures are likely optimistic. The cleanup
rerun matched both. Reproduce them with the command below and the exact approved
source PDFs. See the [Task 1 report](docs/TASK_1_FINAL_REPORT.md#evaluation-results).

### Running the retrieval evaluation

Run from the repository root with the model already cached and the sample PDFs
in `data/uploads/`. The evaluator builds a temporary index and checks that input
files remain unchanged.

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
python -B eval/evaluate_retrieval.py --csv eval/questions.csv --uploads data/uploads --top-k 5
```

### Offline generation replay

```powershell
python -B -m eval.evaluate_generation --records eval/fixtures/generation_mock.jsonl
```

The cleanup replay completed. Replay checks statuses and citations using saved mock outputs. It makes no
Gemini requests and provides no live accuracy grade. See the
[manual grading rubric](docs/TASK_1_FINAL_REPORT.md#generation-replay-and-manual-review).

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

- The local prototype uses a shared index without user authentication or access
  controls.
- Citation checks verify that the quoted text appears in the source. Review the
  source to confirm it supports each claim.
- OCR can misread tables and numbers. The retrieval benchmark uses fictional
  PDFs with usable native text and does not measure scanned-PDF accuracy.

## Future work

- Connect live order lookup to an authorized order system.
- Add a separate question router.
- Evaluate whether a calibrated similarity threshold improves refusal behavior.
- Add user authentication and document access controls.
- Grade live answer quality on a held-out question set.
