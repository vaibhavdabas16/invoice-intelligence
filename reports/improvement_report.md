# Improvement report (phase 3)

Goal: raise the submission against the six rubric categories, keeping only changes whose benefit
was measured. Baseline = the state after the remediation phase, snapshotted in
`experiments/v3_baseline/` (reports + all 20 outputs). Earlier reports are kept unchanged:
[remediation_report.md](remediation_report.md) (phase 2) and the v1 analysis in the appendix of
[failure_analysis.md](failure_analysis.md).

Measurement rules for this phase:
- The 10 labelled documents remain **the** evaluation set and are reported separately.
- Synthetic sets are labelled "exploration" in every output, and no generalisation is claimed.
- One new held-out degradation set (`stress_mild_b`, new seed and parameters) was created **after**
  the designs were fixed and used only to compare configurations.
- Thresholds were fixed before measuring. Every rule changed after a measurement is listed in §6.

## 1. Summary of changes

| # | change | rubric area | evidence | API cost | recommendation |
|---|---|---|---|---|---|
| 1 | Invoice classification confusion matrix, per-document report with silent errors marked, dataset label (evaluation vs exploration) in every output | Eval harness | n/a (reporting) | $0 | **keep** |
| 2 | **Value grounding**: every extracted value of a digital PDF is located in the text layer (page + source line); unsupported values flagged | Extraction & validation | 0 false positives on 200 values of 12 digital invoices; catches 100% of injected invoice-number, vendor, description, date-shift and amount errors (327 injected cases, 98% caught by grounding + arithmetic together) | $0 | **keep** |
| 3 | **Tax-rate consistency check** (single printed rate: tax line = rate × subtotal) | Extraction & validation | caught the only self-consistent misread seen (held-out L07, 2 runs); 0 false alarms on 318 correctly extracted documents | $0 | **keep** |
| 4 | **No automatic acceptance** of re-read disagreements (arbitration annotates only) | Human review | of 2 automatic acceptances ever observed, 1 was wrong (held-out L07) | $0; slightly more reviews | **keep** |
| 5 | **Token-confidence mode T** (effort `none` + logprobs, flag p < 0.5) | Human review / cost | mild: silent 0.67 vs 1.00 per run; held-out: 1.00 vs 0.67 (B better); field-level recall only 23% at p < 0.5; ~25–49% cheaper; ~2× review load | cheaper | **opt-in only, not default** |
| 6 | Cost per document type (digital / scanned / non-invoice) in `cost_summary.json` | Cost | n/a | $0 | **keep** |
| 7 | UI: validation workflow checklist, field-evidence table, re-read panel | Communication / bonus | rendered by tests against real pipeline metadata | $0 | **keep** |
| 8 | `docs/architecture.md` (components, sequence, data contracts, decision log), reading guide and setup troubleshooting in README | Communication | n/a | $0 | **keep** |

Not adopted after investigation: model-reported field evidence (it could be hallucinated; replaced by
deterministic grounding), logprobs on the default model (the API rejects logprobs on reasoning
models unless effort is `none`).

## 2. Evaluation harness (25%)

- **Confusion matrix** (console + `eval_results.json`). It has a separate "pipeline failed" column,
  so failures are neither hidden as correct nor counted as misclassifications.
  Evaluation set: 9/9 invoices, 1/1 non-invoice.
- **Per-document report** (`eval_per_document.md`, generated each run). It shows wrong fields as
  predicted vs expected, the totals check, the review decision and reasons, and marks every
  **silent error** explicitly.
- **Clean vs degraded separation.** Every run states its dataset kind in the console, the JSON and
  the per-document report. Only `data/labelled` is "the evaluation set".
- Unchanged and still central: field-level normalised + exact accuracy, exact document match rate,
  item-level precision and recall, review-flag precision and recall, and documented matching rules.
  The matching rules were **not** changed in this phase.

## 3. Extraction and validation (25%)

### 3.1 Value grounding (field-level evidence)

Design fixed before measuring ([src/validation/evidence.py](../src/validation/evidence.py)):
- text fields match case- and whitespace-insensitively;
- amounts match in plain, Western, Indian and European renderings, with digit boundaries;
- dates match against a fixed list of printed formats;
- quantity and currency are not checked.

The evidence is the result of a string search, so it **cannot be hallucinated**. The required JSON
schema is unchanged; evidence lives in the metadata.

| measurement | result |
|---|---|
| real outputs, 12 digital invoices, 200 values | **197 grounded + 3 derived, 0 unsupported** (after 1 rule correction, §6) |
| injected errors, 327 cases ([error_injection.md](experiments/error_injection.md)) | grounding 89%, arithmetic 37%, **either 98%** |
| invoice number / vendor / description / date shift | **100%** caught by grounding, 0% by arithmetic |
| total replaced by printed subtotal | 0% by grounding (the value *is* printed), 100% by arithmetic |
| misses (5) | 3 day/month swaps that render identically (inherent); 2 injected changes within the 0.02 rounding tolerance |

Scans and photos have no text layer, so their values are `not_checkable`. The UI and metadata say so
explicitly instead of implying verification.

### 3.2 Tax-rate consistency

Motivated by a held-out error (§6). A misread digit (`850` → `650`) was propagated into a unit
price, subtotal and total that all reconcile, so every other check passed. The printed tax
(`Sales tax @ 8.25%` = 334.13) no longer matched 8.25% of the misread subtotal (317.62). The check
runs only when all tax lines print the same rate. Result: 0 false alarms on 318 correct documents,
and it catches both silent instances.

## 4. Human review

### 4.1 Why recall was 40% on mild degradation

The remaining silent errors on mild degradation are **text misreads that both independent reads make
identically**: L09 `Pomfret` → `Reefret` in 4/4 runs, and L02's invoice number once. Across the B runs,
**7 of 29 erroneous documents (24%) were wrong identically in both reads**
([review_signals.md §2](experiments/review_signals.md)), so no re-read-based signal can see them.

### 4.2 Token confidence (evidence-based uncertainty): tested, not adopted as default

`gpt-6-luna` returns logprobs only with reasoning effort `none`. Config T = effort `none` +
logprobs + flag when a field's least likely token has p < 0.5. The threshold was fixed before the
runs.

| set | metric | B (default) | T |
|---|---|---|---|
| clean labelled | accuracy / false flags | 100% / 0 (2 runs) | 100% / 0 (1 run) |
| mild (B 4 runs, T 3 runs) | field errors per run | 1.75 | 2.67 |
| mild | **silent errors per run** | 1.00 | **0.67** |
| mild | flag recall / precision | 43% / 100% | 67% / 100% |
| held-out (B 3 runs, T 2 runs) | field errors per run | 19.0 | 20.5 |
| held-out | **silent errors per run** | **0.67** | 1.00 |
| held-out | flag recall / precision | 91% / 91% | 83% / 83% |
| all | avg $/document | baseline | 25–49% cheaper (no reasoning tokens) |

At field level, p < 0.5 flags 23% of wrong values at 77% precision. 33 of 88 wrong values had
p ≥ 0.999: **the model is often confidently wrong**, and that is exactly the Pomfret case.
The pre-registered adoption criterion was "fewer silent errors on the held-out set", and T did not
meet it. It is available as an opt-in cheaper mode (`TOKEN_CONFIDENCE=true`, `LLM_EFFORT=none`),
fully tested.

### 4.3 Never accept a value because its arithmetic is correct

Arbitration used to clear purely numeric re-read disagreements when the first read reconciled
(later, also requiring high token confidence). Observed automatic acceptances: **2, of which 1
was wrong** (held-out L07, total p = 0.975). Arbitration now **never** clears a disagreement; it
only tells the reviewer which read reconciles and which disputed values had low confidence.

### 4.4 Review flag, before vs after (default config B, re-scored exactly with the final rules)

| set | before this phase | after |
|---|---|---|
| labelled | 0 flags, 0 errors | 0 flags, 0 errors |
| mild (4 runs) | precision 100%, recall 43%, 1.00 silent/run | unchanged (the remaining silent errors are text misreads no new rule can see) |
| held-out (3 runs) | precision 91%, recall 91%, 0.67 silent/run | unchanged for B; for T the tax-rate rule removes the L07 silent error (1.00 → 0.00), **in-sample** (§6) |
| digital PDFs, injected errors | text-field errors undetected | 100% detected by grounding |

The honest summary: this phase **did not reduce silent errors on degraded scans** for the default
configuration. It closed two specific holes (text errors on digital PDFs, and self-consistent
numeric misreads) and removed a validator behaviour that created a silent error.

## 5. Cost

| item | value |
|---|---|
| digital invoice | **$0.00045** per document (1 call) |
| scanned / photographed invoice | **$0.00089** per document (2 calls + occasional repair) |
| non-invoice | $0.00030 |
| incremental API cost of kept changes | **$0** (all deterministic) |
| spend in this phase (experiments, confirmation runs, batch) | **≈ $0.085** recorded (241 calls) plus ~12 small probe calls |
| total project spend recorded | **≈ $0.28** |

Details: [cost_analysis.md](cost_analysis.md).

## 6. Rules changed after a measurement (disclosure)

| rule | first measurement | change | reason |
|---|---|---|---|
| grounding of `tax_total` | 3 false "unsupported" on multi-tax invoices (L03, U01, U08) | ground through its printed components | by the client's convention tax_total is a *sum* of tax lines, so it is never printed as one number |
| grounding of `invoice_number` | 4 injected dropped-character errors missed (`WL-88213` inside `NWL-88213`) | whole-token match | identifiers are tokens, not substrings |
| number matching whitespace | unit test: adjacent table columns merged | match on the original line | squashing whitespace merged `320.00   3,200.00` |
| tax-rate check | motivated by held-out L07 | added | it makes the held-out set **in-sample** for this rule |
| tax-rate check, mixed rates | 2 false alarms on unlabelled U01, U08 (correct extractions) | skip mixed-rate invoices | each rate applies to an unknown part of the subtotal |
| arbitration | 1 wrong automatic acceptance on held-out L07 | never auto-accept | the user's rule, confirmed by data |

The labelled evaluation set was not used to tune any rule. It scores 80/80 before and after.

## 7. Rubric-by-rubric assessment

Self-assessment against the brief's "what good looks like". It is my estimate, not a grade.

| area (weight) | what good looks like | state before this phase | what this phase added | remaining gap |
|---|---|---|---|---|
| Eval harness (25%) | sensible matching rules, one command, per-field results | met: one command, documented rules, per-field normalised + exact, review-flag metrics | confusion matrix, per-document silent-error report, explicit evaluation vs exploration labelling | the evaluation set is too easy (0 errors), so it cannot rank configurations; matcher `L.L.P.` case |
| Extraction & validation (25%) | schema-valid output, working totals check, reasonable accuracy | met: 80/80, strict schema, totals + subtotal + row checks | deterministic field evidence (page + source), tax-rate check, both measured | silent text misreads on degraded scans remain |
| Failure analysis (20%) | errors grouped by cause, clear "fix first" | met, but spread over two reports | one consolidated current analysis: extraction vs document vs evaluation causes, Pomfret, U02, U04, L07, fixed vs open, ranked fixes | causes characterised on synthetic data because the evaluation set has no errors |
| Bad-input handling (10%) | non-invoices + one more case | met (non-invoices; corrupt / empty / encrypted / unreadable / timeout) | none needed | none known |
| Cost awareness (10%) | a real number with a clear method | met | per-document-type costs, incremental cost table, measured cost of the T mode | none known |
| Code & communication (10%) | readable code, working README, honest report | met | architecture doc with decision log, reading guide, setup troubleshooting, UI showing the actual validation workflow | the repository has many reports; the README reading guide is the entry point |
| Bonus (+10) | review flag checked against data; frontend | both present | flag measured before/after on evaluation, mild, held-out and severe sets; UI shows evidence and decisions | flag recall on degraded scans is limited by correlated errors |

Where points are most likely still lost: the evaluation set cannot show a single error, and silent
text misreads on degraded scans remain. Both need data (a harder labelled set), not more features.
