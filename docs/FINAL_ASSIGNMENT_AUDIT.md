# SMARRTIF AI | Assignment compliance audit

Paths in the tables are relative to the local `submission_ready/` folder. Status describes evidence available in the local draft. The assignment receipt date and official form link must come from the candidate’s assignment communication.

## Task 1 · Applied RAG solution

| Requirement | Status | Evidence or remaining action |
| --- | --- | --- |
| Business problem | COMPLETE | `Task_1_RAG_Application/docs/TASK_1_FINAL_REPORT.md`, executive summary and design rationale |
| Decision-maker | COMPLETE | `Task_1_RAG_Application/docs/TASK_1_FINAL_REPORT.md`, customer service manager and supervised pilot |
| Data sources | COMPLETE | `Task_1_RAG_Application/docs/DATA_SOURCE_MANIFEST.md`, four source identities, page counts and hashes |
| Privacy note | COMPLETE | `Task_1_RAG_Application/docs/DATA_SOURCE_MANIFEST.md` and `Task_1_RAG_Application/docs/TASK_1_FINAL_REPORT.md`. Screening limits and external excerpt transfer are disclosed. |
| RAG justification | COMPLETE | `Task_1_RAG_Application/docs/TASK_1_FINAL_REPORT.md`, rationale and alternatives |
| Tools and models | COMPLETE | `Task_1_RAG_Application/requirements.txt` and `Task_1_RAG_Application/docs/TASK_1_VALIDATION_EVIDENCE.md`. Historical model and local configuration discrepancy are distinguished. |
| Working application | COMPLETE | `Task_1_RAG_Application/distribuai/ui/app.py`, pinned environment, and `Task_1_RAG_Application/docs/TASK_1_VALIDATION_EVIDENCE.md` |
| Demo readiness | PARTIAL | `Task_1_RAG_Application/docs/TASK_1_DEMO_GUIDE.md` includes the short walkthrough. Review-source access and authorization for a current live browser demo remain pending. |
| Retrieval evaluation | COMPLETE | `Task_1_RAG_Application/eval/evaluate_retrieval.py` and `Task_1_RAG_Application/docs/TASK_1_VALIDATION_EVIDENCE.md` |
| Generation evaluation | PARTIAL | `Task_1_RAG_Application/eval/evaluate_generation.py` and `Task_1_RAG_Application/eval/fixtures/generation_mock.jsonl` cover offline replay. Independently graded live benchmark cases remain pending. |
| Hallucination and accuracy findings | COMPLETE | `Task_1_RAG_Application/docs/TASK_1_FINAL_REPORT.md`, replay failure and manual rubric. Live population accuracy remains unmeasured. |
| Responsible AI | COMPLETE | `Task_1_RAG_Application/docs/TASK_1_FINAL_REPORT.md`, controls, privacy, narrow data and material risks |
| Limitations | COMPLETE | `Task_1_RAG_Application/docs/TASK_1_FINAL_REPORT.md`, known limitations and trust boundaries |
| Production roadmap | COMPLETE | `Task_1_RAG_Application/docs/TASK_1_FINAL_REPORT.md`, POC to production |
| Measurable expected value | COMPLETE | `Task_1_RAG_Application/docs/TASK_1_FINAL_REPORT.md`, illustrative 15 hours and Rs. 4,750 in the base case |
| README | COMPLETE | `Task_1_RAG_Application/README.md`, setup, running instructions and evidence links |
| Source data reproducibility | BLOCKED | `Task_1_RAG_Application/docs/DATA_SOURCE_MANIFEST.md` specifies the exact files. Operator approval and inclusion are pending. |

## Task 2 · Client consulting case study

| Requirement | Status | Evidence or remaining action |
| --- | --- | --- |
| Discovery plan | COMPLETE | `Task_2_Consulting_Case_Study/TASK_2_CLIENT_CONSULTING_CASE_STUDY.md`, Section 3 |
| Stakeholder questions | COMPLETE | `Task_2_Consulting_Case_Study/TASK_2_CLIENT_CONSULTING_CASE_STUDY.md`, Section 3 stakeholder table |
| Required documents | COMPLETE | `Task_2_Consulting_Case_Study/TASK_2_CLIENT_CONSULTING_CASE_STUDY.md`, Section 3 intake and handling |
| AI readiness ratings | COMPLETE | `Task_2_Consulting_Case_Study/TASK_2_CLIENT_CONSULTING_CASE_STUDY.md`, Section 4 provisional ratings |
| Five use cases | COMPLETE | `Task_2_Consulting_Case_Study/TASK_2_CLIENT_CONSULTING_CASE_STUDY.md`, Section 5 |
| Prioritization matrix | COMPLETE | `Task_2_Consulting_Case_Study/TASK_2_CLIENT_CONSULTING_CASE_STUDY.md`, Section 5 and Appendix A |
| Starting recommendation | COMPLETE | `Task_2_Consulting_Case_Study/TASK_2_CLIENT_CONSULTING_CASE_STUDY.md`, Sections 6 and 12 |
| ROI assumptions | COMPLETE | `Task_2_Consulting_Case_Study/TASK_2_CLIENT_CONSULTING_CASE_STUDY.md`, Section 8 verified scenario calculations |
| Solution architecture | COMPLETE | `Task_2_Consulting_Case_Study/TASK_2_CLIENT_CONSULTING_CASE_STUDY.md`, Section 6 diagram and future controls |
| Build versus buy | COMPLETE | `Task_2_Consulting_Case_Study/TASK_2_CLIENT_CONSULTING_CASE_STUDY.md`, Section 7 |
| Implementation roadmap and KPIs | COMPLETE | `Task_2_Consulting_Case_Study/TASK_2_CLIENT_CONSULTING_CASE_STUDY.md`, Sections 9 and 10 |
| Risk register | COMPLETE | `Task_2_Consulting_Case_Study/TASK_2_CLIENT_CONSULTING_CASE_STUDY.md`, Appendix B |
| Responsible AI | COMPLETE | `Task_2_Consulting_Case_Study/TASK_2_CLIENT_CONSULTING_CASE_STUDY.md`, Sections 6 and 11 |
| Consulting report | COMPLETE | `Task_2_Consulting_Case_Study/TASK_2_CLIENT_CONSULTING_CASE_STUDY.md` and `Task_2_Consulting_Case_Study/TASK_2_CLIENT_CONSULTING_CASE_STUDY.pdf` |
| Executive summary | COMPLETE | `Task_2_Consulting_Case_Study/TASK_2_EXECUTIVE_SUMMARY.md` |
| PowerPoint | USER ACTION REQUIRED | Candidate prepares and adds it to `Task_2_Consulting_Case_Study/`. |

## Task 3 · Candidate video

| Requirement | Status | Evidence or remaining action |
| --- | --- | --- |
| Mandatory face-on-camera video | USER ACTION REQUIRED | Candidate records and adds it to `Task_3_Video/`. `Task_3_Video/README.txt` marks the required addition. |

## Submission

| Requirement | Status | Evidence or remaining action |
| --- | --- | --- |
| Application code | COMPLETE | `Task_1_RAG_Application/` includes the current worktree snapshot, tests, and configuration example. |
| Supporting evidence | COMPLETE | `Task_1_RAG_Application/docs/TASK_1_VALIDATION_EVIDENCE.md` and the included eval fixtures and scripts |
| Dataset and source details | COMPLETE | `Task_1_RAG_Application/eval/questions.csv` and `Task_1_RAG_Application/docs/DATA_SOURCE_MANIFEST.md`. The source PDF copies remain approval-blocked. |
| Case study | COMPLETE | `Task_2_Consulting_Case_Study/TASK_2_CLIENT_CONSULTING_CASE_STUDY.pdf` and Markdown source |
| AI tools disclosure | COMPLETE | `AI_Tools_Disclosure/AI_TOOLS_DISCLOSURE.md`, ready for candidate review |
| Local submission package | COMPLETE | `FINAL_SUBMISSION_GUIDE.md`, `FINAL_SUBMISSION_CHECKLIST.md` and the four task/disclosure folders. The draft ZIP is an offline review copy. |
| Google Drive upload | USER ACTION REQUIRED | Upload the completed and approved folder following `FINAL_SUBMISSION_GUIDE.md`. |
| Public-link verification | USER ACTION REQUIRED | Set viewer access for anyone with the link and verify from a signed-out browser. |
| Official form submission | USER ACTION REQUIRED | Use the official form link from the assignment communication. |

The prepared materials are a local draft. Public source approval, presentation and video additions, and external submission still require candidate action.
