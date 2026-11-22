# AI-Powered Document Intelligence & RAG Assistant | Source-data manifest

The Task 1 benchmark uses four PDF files supplied locally for the fictional Fresh Distribution Pvt Ltd scenario. Their creation history was not supplied. The files are read-only inputs under `data/uploads/`. The repository and submission snapshot include directory placeholders, benchmark questions, and synthetic replay fixtures. Public distribution of the four source PDFs is pending operator approval.

| Filename | Purpose | Format | Pages | OCR required in evaluation | SHA-256 |
| --- | --- | --- | ---: | --- | --- |
| `Delivery & Shipping FAQ.pdf` | Shipping zones, transit, fees, and delivery exceptions | PDF | 6 | No | `7bc19cd841bf2f6f0afd653ad76da75e9c1d892711a266221cb3d53d09dec4e9` |
| `Payment Terms & Invoice Policy.pdf` | Business credit, invoicing, disputes, and late fees | PDF | 5 | No | `70abfafabedfb559e4b0d4aff8e9a49bd5021c053568f5ed9fbb49c12e04ff1e` |
| `Product Information Sheet.pdf` | SKUs, pack sizes, prices, and storage | PDF | 3 | No | `1a2ed13b026a3b3eb0cbe504662386c77f6aaeb29f10b5235678eda84163bcf1` |
| `Returns & Refund Policy.pdf` | Return eligibility, reporting windows, methods, and deadlines | PDF | 5 | No | `0f8f59fea1665f767becbc4e3a9b9f40403a13e3deae22976e42dda705089fa1` |

## Fictional designation and privacy review

The assignment brief identifies these as fictional demo materials. The local scan found only `.example` contact domains in extractable text. No credential patterns, phone-number candidates, or GSTIN-shaped identifiers were detected by the targeted native-text scan. The populated metadata fields were title and producer. The author field was blank and the PDF attachment count was zero in all four files. These checks support the fictional designation. Pattern screening and metadata inspection have limited coverage, including names outside known patterns and information in page graphics or annotations. Approve the exact files for public distribution after reviewing all pages and their document properties. Treat all real company and customer data as private unless separately approved.

## Reproduction and controlled inclusion

After approval, copy the four exact files into `Task_1_RAG_Application/data/uploads/` in the local submission folder, preserving their filenames. Confirm the hashes using the command below from the packaged application root. Keep the original files untouched. Upload the approved copies in the folder alongside the code and manifest. If approval is withheld, the reviewer can still use an approved or clearly labeled synthetic PDF for a basic workflow check. The reported benchmark measurements require these exact files. See the [Task 1 report](TASK_1_FINAL_REPORT.md#evaluation-results).

```powershell
Get-ChildItem -LiteralPath data/uploads -File -Filter '*.pdf' |
    Get-FileHash -Algorithm SHA256 | Select-Object Hash, Path
python -B eval/evaluate_retrieval.py --csv eval/questions.csv
```

Follow the [README](../README.md#running-the-retrieval-evaluation) for the Python and model-cache prerequisites. The retrieval evaluator uses a disposable index and checks the source hashes again after the run. Distribution approval and permission to send excerpts to Google Gemini are separate decisions.
