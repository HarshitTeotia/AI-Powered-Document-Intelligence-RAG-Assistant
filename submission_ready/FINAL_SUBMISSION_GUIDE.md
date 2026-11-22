# SMARRTIF AI | Final submission guide

## Local package and ownership

Use `submission_ready/` for the final Drive folder. `AI_Powered_Document_Intelligence_RAG_Assistant_SMARRTIF_Submission_Draft.zip` is an offline copy of that folder before owner additions. Read the `FINAL_SUBMISSION_CHECKLIST.md` and `FINAL_ASSIGNMENT_AUDIT.md` in the folder. The archive is a draft for review.

| Folder in `submission_ready/` | Prepared contents and next action |
| --- | --- |
| `Task_1_RAG_Application/` | Latest local project snapshot, README, tests, evaluators and synthetic fixtures. The `docs/` subfolder contains the final report, evaluation evidence, dataset audit, demo guide and source manifest. Add authorized source PDFs as described below. |
| `Task_2_Consulting_Case_Study/` | Full case study, executive summary, and print-ready PDF. Add your PowerPoint in this folder. |
| `Task_3_Video/` | Add your mandatory face-on-camera explanation video here. Follow the assignment communication for the video and bonus explanation instructions. |
| `AI_Tools_Disclosure/` | Include the prepared disclosure note after candidate review. |

Use the separate `docs/PPT_AND_VIDEO_FACTS_HANDOFF.md` as a local reference while preparing your own presentation and video. The assignment deadline is five days from receipt. Confirm the receipt date and deadline from your assignment communication.

## Test the deliverable locally

Open PowerShell in `submission_ready/Task_1_RAG_Application/`. Follow the included `README.md` to install its pinned requirements into a private Python 3.12 virtual environment. Install Tesseract with English language data for scanned PDFs. Run the tests and the offline generation replay, then start the app on localhost. Keep the virtual environment and any generated data in a private test copy of the snapshot.

```powershell
python -m pytest -q
python -B -m eval.evaluate_generation --records eval/fixtures/generation_mock.jsonl
python -m streamlit run distribuai/ui/app.py --server.address 127.0.0.1
```

Configure `GEMINI_API_KEY`, `GEMINI_MODEL`, and `GEMINI_CHAT_ENABLED` through an ignored `.env` or local process environment. The package contains `.env.example` with a disabled chat default. The reported prior model was `gemini-3.5-flash-lite`. The current local model setting differs from that historical value, so confirm which model is available and approved before a live run. Enable chat with `GEMINI_CHAT_ENABLED=true` only with approval for external excerpt processing. The included `docs/TASK_1_DEMO_GUIDE.md` gives the exact four-question workflow. Keep actual Gemini credentials private.

## Include the source PDFs after approval

Source names, hashes, purpose, and screening limits are described in `Task_1_RAG_Application/docs/DATA_SOURCE_MANIFEST.md`. The snapshot excludes PDFs while public distribution approval is pending. Exact benchmark reproduction requires matching copies. From the original repository root, run the following only after reviewing and approving the exact four documents for public-link access.

```powershell
$approvedNames = @(
    'Delivery & Shipping FAQ.pdf',
    'Payment Terms & Invoice Policy.pdf',
    'Product Information Sheet.pdf',
    'Returns & Refund Policy.pdf'
)
$destination = 'submission_ready/Task_1_RAG_Application/data/uploads'
foreach ($name in $approvedNames) {
    Copy-Item -LiteralPath (Join-Path 'data/uploads' $name) -Destination $destination
}
Get-ChildItem -LiteralPath $destination -File -Filter '*.pdf' | Get-FileHash -Algorithm SHA256
```

Match the printed hashes to the manifest. Use the updated folder for upload. The existing draft ZIP stays at its original contents and would need to be rebuilt if you decide to share a ZIP. If approval is withheld, identify the source restriction for the reviewer. A labeled synthetic replacement can demonstrate PDF ingestion, and an alternative benchmark would be needed for comparable results.

Keep `.env`, keys and tokens, private spreadsheets or inbox exports, credentials, `.venv/`, caches, persistent ChromaDB files, temporary stores, raw captures and logs out of the shared folder. Local screenshots from mock tests are optional evidence and should be clearly labeled if shared. Exclude any screenshot or video frame showing credentials, private data, or unrelated workspace content.

## Submit the approved folder

1. Add your PowerPoint to Task 2 and mandatory video to Task 3. Review the disclosure and source files. Check the folder in a clean view to make sure it contains only material approved for sharing.
2. On a computer, open Google Drive and select **New → Folder upload**. Choose the completed folder. Wait for the files and subfolders to finish uploading. [Google Drive upload help](https://support.google.com/drive/answer/2424368)
3. Select the uploaded folder and open **Share**. Under **General access**, choose **Anyone with the link** and set the role to **Viewer**. Copy the link and save the sharing settings. This is required by the assignment and lets anyone with the link open the folder without signing in. If your account restricts link sharing, resolve that with the organizer before submitting. [Google Drive sharing help](https://support.google.com/drive/answer/7166529)
4. Open the folder link in an incognito or signed-out browser. Verify the root folder opens with viewing access. Open the Task 1 README and evidence, the source manifest and approved PDFs, the Task 2 report and PowerPoint, the video, and the disclosure. Check that the video is accessible to a viewer and that no restricted files were accidentally uploaded.
5. Open the official Google Form from the assignment communication and submit the verified Drive folder link as requested. The form URL was not supplied with the local assets. Retain any confirmation. The upload, public-link check, and form submission are the candidate's actions.
