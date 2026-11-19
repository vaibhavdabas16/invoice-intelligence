# Evaluation report

Command: `python -m src.eval.run` (raw console output: [eval_console_output.txt](eval_console_output.txt);
per-field rows: [results.csv](results.csv); everything incl. predictions, GT, alignments and
API metadata: [eval_results.json](eval_results.json)).

## 1. Setup

| item | value |
|---|---|
| data | 10 labelled documents (`data/labelled/L01..L10`): 6 digital PDFs, 1 two-page digital PDF, 1 phone photo (JPG), 2 image-only scanned PDFs; 9 invoices + 1 purchase order |
| model | OpenAI `gpt-6-luna`, reasoning effort `low`, Responses API, strict JSON-schema output |
| input | hybrid: page image (long edge 1568 px) + PDF text layer when present |
| prompt | `v3` (`src/extraction/prompts.py`); its wording is identical to v1 (v2 experiment removed) |
| image size | every page sent at a 1568 px long edge (PDFs rendered, photos up- or downscaled: `UPSCALE_IMAGES=true`) |
| validation | Pydantic schema, totals/subtotal/tax-line/row checks (tolerance 0.02), 1 repair call max |
| consistency check | on for image-only pages (`VERIFY_SCANS=true`): L02, L09, L10 got a second read; arithmetic arbitration for purely numeric disagreements |
| identifier check | invoice number must appear in the PDF text layer (digital PDFs) |

| eval numeric tolerance | 0.01 |

**About this run.** The shipped prompt (v3) has the v1 wording, PDFs were already rendered at
1568 px, and the only labelled photo (L02) is larger than 1568 px, so the extraction requests match
v1. The current official run is a **fresh** sample: a cache-key change made every request new. It is
the 4th independent fresh run of the shipped prompt on this set, and all 4 scored 80/80, as did 2
v2-prompt runs and 1 token-confidence (T) run. All new validators (value grounding, tax-rate
check) ran on it: 0 unsupported values, 0 tax-rate alarms, 0 review flags.

**Clean vs degraded.** Every evaluation run prints and stores which dataset it used. Only
`data/labelled` is reported as *the evaluation set*. Every other directory is labelled
"EXPLORATION SET ... (derived/synthetic; not the evaluation set, no generalisation claims)", in the
console, in `eval_results.json` (`dataset.kind`) and in the generated per-document report.

**Ground truth is never an input to the extractor.** It is read only in `src/eval/run.py`
after extraction, validated against the schema (a warning is printed if invalid), and never
modified.

## 2. Field-level accuracy (headline)

| field | accuracy | correct | total | exact raw match |
|---|---|---|---|---|
| is_invoice | 100.0% | 10 | 10 | 10 |
| vendor | 100.0% | 10 | 10 | 7 |
| invoice_number | 100.0% | 10 | 10 | 10 |
| invoice_date | 100.0% | 10 | 10 | 10 |
| currency | 100.0% | 10 | 10 | 10 |
| line_items | 100.0% | 10 | 10 | 10 |
| tax_total | 100.0% | 10 | 10 | 10 |
| total | 100.0% | 10 | 10 | 10 |

| summary metric | value |
|---|---|
| overall field accuracy (normalised) | **100.0% (80/80)** |
| overall exact raw accuracy | 96.2% (77/80) |
| exact document match rate | 100% (10/10) |
| documents with ≥ 1 field error | 0 |
| validation failures (schema or totals) | 0 |
| non-invoices correctly classified | 1/1 (L08, purchase order) |
| invoice classification (confusion matrix) | GT invoice → predicted invoice 9, non-invoice 0, failed 0; GT non-invoice → invoice 0, non-invoice 1, failed 0 (invoice precision / recall 100% / 100%) |
| line items: GT / predicted / exact matches | 66 / 66 / 66 |
| item precision / recall | 100% / 100% |
| description / quantity / amount accuracy | 100% / 100% / 100% |

The three raw-vs-normalised differences are all `vendor` letter case: the letterheads of L05, L07
and L09 print the name in capitals (`ORBIT HARDWARE SUPPLIES`) while GT uses title case.

### How much should you trust 100%?

Not much more than "it works on clean documents". With n = 10 documents, 10/10 has a 95%
Clopper-Pearson interval of **[69%, 100%]** for per-document accuracy. The labelled set is clean,
and every configuration I tried scored 100% on it (table below), so it cannot rank alternatives.
The failure modes are characterised on degraded copies of the same documents instead; see
[failure_analysis.md](failure_analysis.md).

**Per-document view:** [eval_per_document.md](eval_per_document.md) lists, for every document, the
wrong fields (predicted vs expected), the totals check, the review decision and its reasons, and marks
any **silent error** (wrong but not flagged). Every eval run generates one; for a run with errors see
[experiments/final2_stress_mild/eval_per_document.md](experiments/final2_stress_mild/eval_per_document.md)
(runs made before this feature existed do not have the file).

## 3. Matching rules

Two verdicts per field: **exact** (raw values identical; JSON numbers compared numerically, so
`3200 == 3200.0`) and **correct** (equal after normalisation). The headline is *correct*.

| field | normalisation for "correct" | deliberately NOT normalised |
|---|---|---|
| is_invoice | none (boolean) | |
| vendor | Unicode NFKC, case-fold, `&` → `and`, punctuation → space, collapse whitespace | legal suffixes: `Acme Pvt Ltd` ≠ `Acme` (reported separately as "lenient", informational only); different names are never merged |
| invoice_number | case-fold, remove whitespace | separators and leading zeros: `KOM/26-27/00418` ≠ `KOM-26-27-418` (different keys for a payables system) |
| invoice_date | parse unambiguous textual formats (`14 March 2026`) to ISO | numeric `04/09/2026` is left as-is (ambiguous), so it can only match if already ISO |
| currency | upper-case | symbols are not mapped (`₹` ≠ `INR`): the schema requires the code |
| tax_total, total | parse numbers (`"1,200.00"` = `1200`, `"₹1,23,456.50"` = `123456.5`), \|a − b\| ≤ 0.01 | `null` ≠ `0` (a missing tax is not a zero tax) |
| line_items | rows aligned one-to-one (greedy on description similarity + amount/qty agreement, so one missed row does not shift every later row); an item matches when normalised description is equal, and quantity and amount are within 0.01 (null = null) | the field is correct only if the counts are equal **and** every row matches; row order is reported separately (`order_preserved`), not folded into correctness |

Non-invoices: when GT has `is_invoice: false`, each other field is correct only if the prediction
is `null`. If the pipeline fails on a document, every field of that document counts as wrong.

Line-item attribute accuracies use the number of **GT items** as the denominator, so unmatched
GT rows count as wrong for description, quantity and amount.

## 4. Repeatability and ablations

All on the 10 labelled documents ([experiments/summary.md](experiments/summary.md)):

| run | what changed | field accuracy | avg $/doc (billed) |
|---|---|---|---|
| baseline_v1 | pre-registered config, first run | 100% | 0.000484 |
| baseline_rep2, rep3 | same config, fresh API calls | 100%, 100% | 0.000250, 0.000223 (*) |
| image_only | no text layer | 100% | 0.000430 |
| text_only | text layer only for digital PDFs (scans still use images) | 100% | 0.000295 |
| effort_none | reasoning effort `none` | 100% | 0.000483 |
| gpt5nano | model `gpt-5-nano` | 100% | 0.000524 |
| baseline_verify / official | + consistency check on scans | 100% | 0.000320 (*) |

(*) Cheaper because OpenAI's automatic prompt caching recognised identical requests sent
earlier; see [cost_analysis.md](cost_analysis.md).

Nothing separates the configurations on accuracy. The choice of hybrid, effort `low` and
gpt-6-luna rests on cost and robustness reasoning, not on a measured accuracy difference.

## 5. Robustness (stress sets derived from the labelled documents)

| set | configuration | field accuracy per run | field errors / run | false flags / run | flag precision / recall |
|---|---|---|---|---|---|
| stress_mild | v1, no re-read (1 run) | 97.5% | 2.0 | 2.0 | 0% / 0% |
| stress_mild | v1 (3 fresh runs) | 96.2%, 96.2%, 96.2% | 3.0 | 2.0 | 45% / 56% |
| stress_mild | **final** (3 fresh runs) | 97.5%, 98.8%, 97.5% | **1.67** | **0** | **100%** / 40% |
| stress_severe | v1 / final | 27.5% / 27.5% | 58 / 58 | 0 / 0 | 100% / 100% |
| stress_rot90 | v1 / final | 96.2% / 97.5% | 3 / 2 (L04 only) | **9 / 0** | 10% / 100% (precision) |
| stress_rot270 | v1 / final | 93.8% / 96.2% | 5 / 3 (L04 only) | **9 / 0** | 10% / 100% (precision) |

Full v1 / v2 / final table: [experiments/v1_vs_v2.md](experiments/v1_vs_v2.md). Rotated-set errors
are all on L04, whose two pages were stacked and then rotated into one 1568 × 554 image when the
test set was built (a resolution artifact of the test, not a rotation effect).

All severe-stress "errors" are abstentions (`null` + `illegible_fields`), not invented values.
Details, examples and fixes are in [failure_analysis.md](failure_analysis.md).

## 6. Tuning disclosure (overfitting check)

Phase 3 additions (value grounding, tax-rate check, no automatic acceptance, opt-in token
confidence) and the rules refined after measurements are listed in
[improvement_report.md §6](improvement_report.md). None of them used the labelled set for tuning; the
labelled set scores 80/80 before and after.

- Prompt `v1` was written before any labelled result was seen and **was not changed** afterwards.
  The labelling conventions from the sample pack's README (vendor = issuer, issue date not due
  date, discounts as negative rows, tax_total = 0 when there is no tax) were encoded in the prompt.
  They are the client's published business rules, not answers learned from the labelled set.
- The matching rules were written before the first evaluation and were not changed after
  seeing results (see failure analysis §D for a case where I deliberately did not change them).
- **Post-hoc changes, all made after looking at results and all disclosed:**
  1. `VERIFY_SCANS` (self-consistency re-read for image-only pages), after the first stress test.
  2. In the remediation phase: photo upscaling, arithmetic arbitration, PDF text-layer grounding,
     and removing image-quality and rotation heuristics as review triggers. Also a prompt v2
     experiment, which was **removed again**, so the shipped prompt wording is v1.

  None of them changed a labelled-set result (100% → 100%, 0 flags). Their effects were measured
  on stress sets derived from the labelled documents and on the unlabelled U02/U05, so the stress
  numbers are **in-sample** for these design decisions. Matching rules were not changed.
- Image-quality thresholds were sanity-checked on the 20 sample documents plus synthetic
  blank/blurred images; they only reject blank or tiny pages.
