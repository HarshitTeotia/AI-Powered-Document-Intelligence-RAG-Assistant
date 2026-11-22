# Task 1 validation evidence | finalization rerun

## Verified environment

Python 3.12.10, Streamlit 1.45.1, ChromaDB 1.0.9, google-genai 1.51.0, sentence-transformers 4.1.0, and Pytest 8.3.5. The existing virtual environment passed `pip check`. The package installation itself was not repeated from scratch. The local environment check found a configured key and enabled chat opt-in. Its private model setting differs from the historical `gemini-3.5-flash-lite` model. The effective value was not disclosed or changed. Confirm it locally before requesting an authorized live test.

| Check | Observed result |
| --- | --- |
| Focused retrieval, generation and UI tests | 227 passed, no failures or skips |
| Full test suite | 498 passed, no failures or skips, 104 existing ChromaDB deprecation warnings |
| Retrieval CLI | Four PDFs, 19 pages, 48 passages, 35 questions, isolated index cleaned up |
| File hit at 1, 3, and 5 | 20/20 at each depth |
| Page hit at 1, 3, and 5 | 14/15, 15/15, 15/15. Five unlabeled answerable cases excluded. |
| Page MRR and required strings | 0.9556 on 15 labeled cases. All required strings at five for 20/20 answerable cases. |
| Offline replay | 11 synthetic records. Six valid, two rejected, two simulated operational errors, one skip. 5/6 valid statuses matched, and 4/4 returned citations matched their recorded source and original text. |

The retrieval CLI reported non-failing PDF CropBox fallback notices. Ranking, labels, source PDFs, and persistent application storage were unchanged. The spoilage deadline ranked third. Synthetic replay results describe validation behavior, and a valid source quote can accompany a fabricated claim. The [Task 1 report](TASK_1_FINAL_REPORT.md) details methodology and the manual review rubric. Browser behavior was previously verified with mocks. The reported four historical live backend answers are separate evidence. No Gemini endpoint was called during this rerun.

## Clean-source reproduction

A disposable copy of the application, test suite, and evaluators ran with the existing Python 3.12 environment, an empty index, and Gemini credentials removed from the child process. All 498 tests passed and offline replay reproduced the same status count. The retrieval evaluator identified the missing source PDFs as expected. The real Streamlit app passed its localhost health check and opened in headless Chrome with the empty-index message, disabled chat, and no application exception. Both processes stopped and the disposable copy was removed. A fresh offline dependency install was blocked by unavailable local distributions. A full fresh installation requires package access.

## Commands

Run from the application root using the documented Python environment and approved inputs. Offline flags assume the embeddings are cached. Use your local Tesseract path when running the optional OCR smoke test.

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
python -m pytest -q tests/test_retrieval_scoring.py tests/test_retrieval_evaluation_cli.py tests/test_generation.py tests/test_generation_evaluation_cli.py tests/test_ui_service.py tests/test_ui_app.py
python -m pytest -q
python -B eval/evaluate_retrieval.py --csv eval/questions.csv
python -B -m eval.evaluate_generation --records eval/fixtures/generation_mock.jsonl
```

The rerun disabled the Pytest cache provider and used an external temporary root. The source PDFs are intentionally excluded from the draft snapshot. See the [data manifest](DATA_SOURCE_MANIFEST.md) and [demo guide](TASK_1_DEMO_GUIDE.md) for exact source recovery and the pending authorized live browser check.
