# AI-Powered Document Intelligence & RAG Assistant | Task 1 demo guide

**Target duration** 2 minutes 45 seconds. Allow setup time before starting the walkthrough. This guide covers the Task 1 product demo. The assignment’s face-on-camera video is a separate deliverable.

## Prepare

1. Use a complete project snapshot with the Python 3.12 environment. Obtain the four approved fictional PDFs listed in the [report](TASK_1_FINAL_REPORT.md#sources-and-reproducibility), place them in `data/uploads/`, and keep originals read-only. The Git repository excludes them by default.
2. From the project root, activate `.venv` and run the commands below. The offline flags assume the embedding model is already cached. Follow the [README](../README.md#running-the-streamlit-application) for the first setup. Adjust the Tesseract path if using OCR on a scanned document.

   ```powershell
   .\.venv\Scripts\Activate.ps1
   $env:HF_HUB_OFFLINE = '1'
   $env:TRANSFORMERS_OFFLINE = '1'
   $env:GEMINI_CHAT_ENABLED = 'false'
   python -m streamlit run distribuai/ui/app.py --server.address 127.0.0.1
   ```

3. Use a local demo instance with only the four approved fictional sources in its knowledge base. Select **Add demo documents** if they are absent. Check the displayed list. Indexing is local. The process setting above keeps Gemini chat disabled even if `.env` enables it.
4. **For a live run with explicit authorization to send excerpts**, stop Streamlit. Configure a valid `GEMINI_API_KEY` and the authorized `GEMINI_MODEL` in the ignored `.env` or process environment. The historically verified model was `gemini-3.5-flash-lite`. The current local setting differs and should be confirmed by the operator. Keep the key out of terminal output and screenshots. Run the following in the same activated PowerShell session.

   ```powershell
   $env:GEMINI_CHAT_ENABLED = 'true'
   python -m streamlit run distribuai/ui/app.py --server.address 127.0.0.1
   ```

5. Open the localhost URL in the terminal and submit only the agreed questions, once each. Check citations against the PDF, record the observed answer and status privately, and mark service or configuration failures separately. Stop after the approved calls. Restore `GEMINI_CHAT_ENABLED=false` after the session. Without authorization, use the local document workflow and the labeled mock-browser evidence described in the report.

## Walkthrough

| Time | Action | Suggested narration |
| --- | --- | --- |
| 0:00–0:20 | Show the knowledge base and four documents | “Support agents at a distributor repeatedly look up returns, delivery charges, product details, and payment terms. The assistant gives them one place to find the relevant policy.” |
| 0:20–0:55 | Ask “What is the standard return window for an unopened product?” Open **Sources** | “The returns policy describes seven calendar days from confirmed delivery for eligible non-perishable products. I can open the citation and check the conditions on page 1 before answering a customer.” |
| 0:55–1:25 | Ask “What are the standard business credit terms?” Open **Sources** | “For approved business customers, the expected terms are Net 15 calendar days from the invoice date and a standard Rs. 50,000 credit limit. The reference is the payment policy, page 2.” |
| 1:25–1:50 | Ask “Is there an automatic discount for ordering 100 cartons of the same product?” | “The supplied policies do not establish an automatic discount. The expected response is an insufficient-evidence message.” |
| 1:50–2:15 | Ask “Has order FD-2026-1042 been dispatched?” | “The PDFs explain shipping policy. Checking a particular order requires an authorized lookup in the order system.” |
| 2:15–2:45 | Sum up what a pilot would measure | “A pilot would measure handling time including review, the share of questions agents can safely handle with assistance, and unsupported claims. The business-value estimates are illustrative. Live answering sends relevant excerpts to Gemini, so the demo uses authorized fictional material.” |

The facts above are expected outcomes for verification. If the live answer differs, describe what appeared and record it as a failure or an incomplete run. A working citation confirms the quotation and source location. The agent still checks that the source supports the answer.
