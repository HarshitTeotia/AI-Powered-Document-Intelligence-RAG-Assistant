# Task 1 benchmark audit and proposed label corrections

The audit used [`eval/questions.csv`](../eval/questions.csv) and the four local fictional source PDFs. The CSV was kept intact. The proposals below are for a separately reviewed benchmark revision. The Task 1 report uses the original labels and denominators.

## Current coverage

| Check | Result |
| --- | --- |
| Rows and IDs | 35 rows, IDs 1–35, each unique and nonempty |
| Questions | 35 nonempty questions. No exact duplicates after case, whitespace, and final question-mark normalization |
| Near-duplicates | Q31–Q35 are misspelled or paraphrased versions of Q1, Q5, Q6, Q9, and Q13. These test wording variation and should remain in the same split as their originals in future evaluation |
| Categories | 20 `answerable`, 13 `not_in_documents`, 2 `order_lookup` |
| Expected answers and evidence terms | Every row has an expected answer and at least one term. Terms for the 15 refusal or order cases describe desired response behavior. The retrieval evaluator uses terms only for answerable rows |
| Expected documents and sections | All 20 answerable rows name a local source and section. Shipping covers 6, returns 5, payments 5, and products 4 |
| Expected pages | 15 answerable rows have positive in-range page labels. Q31–Q35 are unlabeled. The 15 other rows intentionally have no expected document or page |
| Document-side term check | For each answerable row, all of its required strings were located together on at least one page of its expected file. Q5’s complete wording was found on page 5 |

`answerable` covers a policy fact supported by the document set. `not_in_documents` covers an exact requested fact or policy that the set does not establish, even when a related passage exists. `order_lookup` covers a specific transaction that needs live data. The two order cases are too few to characterize handling of all operational questions. This set emphasizes English policy questions from one fictional business. It does not measure multilingual support, scanned-document answer quality, or performance on another catalog.

## Proposed page labels

| Row | Proposed page | Supporting source evidence |
| --- | ---: | --- |
| Q31, unopened product return | 1 | `Returns & Refund Policy.pdf` states the seven-calendar-day window from confirmed delivery for eligible non-perishable products and lists the unopened condition. |
| Q32, Zone A transit after dispatch | 5 | `Delivery & Shipping FAQ.pdf` gives an example of Zone A with an estimated one-to-two-business-day window after dispatch. Page 1 contains the transit table, and page 2 explains that timelines run from dispatch. Page 5 is a self-contained reference for this wording. |
| Q33, shipping for Rs. 14,500 in Zone C | 3 | `Delivery & Shipping FAQ.pdf` gives a standard charge of Rs. 450 and a free-shipping threshold of Rs. 15,000. The worked example uses the same Rs. 14,500 pre-tax subtotal. |
| Q34, approved business credit | 2 | `Payment Terms & Invoice Policy.pdf` lists Net 15 calendar days from the invoice date and the standard credit limit of Rs. 50,000. |
| Q35, Premium Basmati Rice | 1 | `Product Information Sheet.pdf` lists FD-RIC-101, a 10 kg bag, and Rs. 1,500.50. Check the separate GST and delivery qualifiers when reviewing a full answer. |

## Proposed review clarifications

- **Q1 and Q31.** The reference answer says eligible unopened products. Spell out the non-perishable restriction and the confirmed-delivery reference date in the review checklist. The fuller conditions remain on page 1.
- **Q5 and Q32.** Page 1 establishes the Zone A duration. It does not by itself state the starting event. A reviewer should credit the complete claim using page 5 on its own or the supporting pages together. The original benchmark accepts pages 1 and 5 for Q5. Keep that result comparable to earlier runs.
- **Q10.** Payment policy page 2 contains the worked 5 November to 20 November 2026 example. The current required term is only `Net 15`. A future evidence rubric can require the due date for this worked example. The existing string check cannot establish that the generated date is right.
- **Q22–Q24 and Q27.** These ask for an exact amount or schedule. For example, the returns policy says the customer pays return shipping on excess-stock returns and gives no exact amount in that passage. Payment policy page 4 allows GST disputes and does not establish a processing fee. Credit an answer that distinguishes the supported policy from the missing amount or schedule.
- **Q19 and Q20.** The expected answers describe an authorized lookup. The app currently returns a generic order-system message. Order IDs and terms such as `tracking` in `must_contain` are not requirements for that message. Assess whether it avoids claiming dispatch, courier, or tracking information and directs the user to authorized lookup.

The evidence-string scan checks selected terms and page candidates. A review of every claim, exception, and ambiguous negative by the document owner remains appropriate before treating the benchmark as production ground truth.
