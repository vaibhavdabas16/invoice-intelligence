# Failure analysis

**Current state, grouped by cause, with what to fix first.** Every example comes from a recorded run
(paths given). Nothing is invented. "GT" = ground truth. The original v1 analysis is kept unchanged
in the [appendix](#appendix-original-v1-analysis) as the record of what was found first.

## 0. Where errors can and cannot occur

| data | status |
|---|---|
| **10 labelled documents** (the evaluation set) | **0 errors** in every run: 80/80 fields in 4 independent fresh runs of the shipped prompt (incl. the final one), 2 runs of the v2 prompt and 1 of config T |
| 10 unlabelled documents (no GT) | 2 flagged; one is a model misread (U02), one is a document inconsistency (U04) |
| synthetic degradations of the labelled documents (exploration only) | where every failure category below was observed; **no generalisation claims** |

Because the evaluation set has no errors, the analysis of failure *causes* relies on degraded copies
of the same documents (GT unchanged) and on the unlabelled documents. That is a limitation in itself
(§5).

## 1. Errors grouped by cause

The first split separates **extraction failures** (the model output differs from what is printed)
from the other categories, which are not model errors.

| # | cause | type | example (run) | GT vs prediction | caught by | status |
|---|---|---|---|---|---|---|
| E1 | **degraded glyphs read as a plausible word** | extraction | L09 mild (`final_stress_mild_r1..3`) | `Pomfret, whole, kg` vs `Reefret, whole, kg` | nothing in most runs: both reads agree (correlated), token p ≈ 1 | **unresolved** |
| E2 | **uncertain digits in identifiers** | extraction | L02 mild (`final_stress_mild_r1`) | `SGS/1847` vs `SGS/2847` | re-read disagreement (when the two reads differ) | partly: caught when reads differ |
| E3 | **small identifier segments lost at low resolution** | extraction (preprocessing-induced) | L10 mild (v1 runs) | `MAP/D/09-0312` vs `MAP/09-0312` | none | **fixed**: photos upscaled like PDF pages; wrong 6/6 → correct 9/9 ([remediation §2.1](remediation_report.md)) |
| E4 | **misread digit propagated into a self-consistent read** | extraction | L07 held-out (`T_heldout_r2`) | line 2 `850.00` vs `650.00`, subtotal 4050 vs 3850, total 4384.13 vs 4184.13 | every arithmetic check passes; was **auto-accepted** by arbitration | **fixed**: tax-rate check (8.25% of 3850 ≠ 334.13) + arbitration no longer auto-accepts |
| E5 | **single misread amount** | extraction | U02 (final batch, no GT; my reading of the image) | line 4 `98.00` vs `96.00` | totals and subtotal mismatch | caught (flagged); output left as extracted |
| E6 | **page unreadable** | extraction (abstention) | 9/10 severe-stress documents | e.g. L02: every field `null`, listed in `illegible_fields` | `insufficient_data` + illegible fields | caught (100% recall); correct behaviour |
| D1 | **inconsistency in the source document** | document, not model | U04 (unlabelled) | page prints 1440 × $0.09 = **$122.40** (should be 129.60) | row check | caught, correct: a human must decide |
| M1 | **matcher false negative** | evaluation | L04 severe | `Meridian Print & Pack LLP` vs `... L.L.P.` | n/a | known; rule deliberately not changed after seeing it |
| T1 | **test-construction artifact** | evaluation | L04 rotated sets | 2 pages stacked then rotated into one 1568 × 554 image | review flag | not a pipeline failure; excluded from conclusions |

### E1. Pomfret → Reefret (the most important unresolved failure)

- **Underlying cause.** At mild degradation (837 × 1122 px, blur, JPEG q45), the strokes of `om` and
  `f` in "Pomfret" merge (crop: [remediation §2.2](remediation_report.md)). The model fills the gap
  with a word-like string. Without upscaling it wrote `Parrot` (5/6 runs); with upscaling it writes
  `Reefret` (9/9 runs).
- **Why nothing catches it.**
  - Arithmetic cannot see text.
  - The PDF text layer does not exist for scans.
  - **Both independent reads make the same error** (correlated). Across the B runs (mild + held-out, 7 runs), 7 of 29
    erroneous documents (24%) were wrong *identically* in both reads ([review_signals.md §2](experiments/review_signals.md)).
  - Token confidence does not help: the wrong value is produced with probability ≈ 1. 33 of 88 wrong
    values in the token-confidence runs had p ≥ 0.999 ([review_signals.md §3](experiments/review_signals.md)).
- **Concrete fix.** A higher-resolution **crop of the item table**, sent as a second image for
  image-only pages. Upscaling the whole page cannot restore merged strokes, but a crop from the
  original-resolution page spends more patches per glyph. Expected cost about +1–2k input tokens
  (≈ +$0.0002) per scan. **Not implemented: needs a benchmark first.**

### E4. Self-consistent misread (held-out L07): why "it adds up" is not proof

The first read misread one digit (`850` → `650`). The unit price, printed subtotal and total were
all read consistently with that misread, so the totals, subtotal and row checks all passed, and the
total had token probability 0.975. The re-read had the correct total. The arbitration rule of
that time accepted the first read because it reconciled and was confident: **a silent error created
by the validator.** Two consequences:

1. **Automatic acceptance was removed.** Every disagreement now goes to a human, and the reviewer
   is told which read reconciles. Of the 2 automatic acceptances ever observed, 1 was wrong.
2. **A tax-rate check was added.** When every tax line prints the same rate, each must equal rate ×
   subtotal. 8.25% of the misread 3,850 is 317.62, not the printed 334.13. Measured: **0 false
   alarms on 318 correctly extracted documents**, and it catches both silent L07 errors. It is not
   applied to mixed-rate invoices (U01, U08), where each rate covers an unknown part of the subtotal.
   That restriction was decided after seeing those two documents, and is disclosed.

### E5 vs D1: extraction failure vs document inconsistency

Both U02 and U04 are flagged, for opposite reasons. **U02** is a model misread: the image shows
`98.00`, the model wrote `96.00`, and the printed subtotal and total prove the read wrong.
**U04** is a correctly extracted document that is itself inconsistent: the vendor printed an amount
that does not equal quantity × unit price. The pipeline must not "fix" either one. Both carry
review reasons that let a human tell them apart: "totals do not reconcile" vs "quantity × unit price
≠ amount on row 1" with totals that reconcile.

## 2. What was fixed (measured) and what remains

| issue | measured before | measured after | where |
|---|---|---|---|
| identifier segment lost (E3) | wrong 6/6 runs | correct 9/9 | remediation §2.1 |
| false review flags from image-quality heuristics | 2 per mild run, 9 per rotated run | 0 | remediation §4 |
| self-consistent misread accepted (E4) | 1 silent error in 2 held-out T runs | 0 (re-scored with the new rules) | improvement report §3 |
| text errors on **digital** PDFs | no detector | 100% of injected invoice-number, vendor, date and description errors caught, 0 false positives | [error_injection.md](experiments/error_injection.md) |
| Pomfret → Reefret (E1) | silent | **still silent** | unresolved |
| uncertain identifier digits on scans (E2) | sometimes silent | sometimes silent | unresolved |
| silent errors per mild run, default config | 1.33 (v1) | 1.00 | unresolved in part |

## 3. What to fix first

Ranked by expected impact on **silent errors** (wrong values that reach the ledger unflagged), per
unit of effort:

| priority | fix | addresses | expected impact | cost | status |
|---|---|---|---|---|---|
| 1 | **Harder labelled set** (real phone photos, mixed tax rates, credit notes, charges outside the table) | the evaluation set cannot rank configurations or show real silent-error rates | needed before any further tuning; the current 10 documents give 0 errors under every configuration | labelling time | proposed |
| 2 | **High-resolution item-table / header crop** for image-only pages | E1, E2 at the source | targets the only remaining silent errors observed | ≈ +$0.0002 per scan | proposed, needs benchmark |
| 3 | **Tax-rate check for mixed rates** using per-line tax codes (HSN/category) | extends the E4 detector to U01/U08-style invoices | unknown, no observed failure | none (deterministic) | proposed |
| 4 | **Decide T vs B on a fresh held-out set** | T: fewer silent errors on mild (0.67 vs 1.00 per run) but more on held-out (1.00 vs 0.67), ~25–45% cheaper, ~2× review load | could reduce cost at equal quality | ≈ $0.05 of runs | proposed |
| 5 | Matcher: treat `L.L.P.` = `LLP` (pre-registered) | M1 | evaluation accuracy only | none | proposed |

## 4. Not observed in any run

- invoice / non-invoice confusion (L08 correct in every evaluation and stress run, even severely degraded; U06 and U10 correct in every batch);
- currency errors;
- total vs subtotal vs balance-due confusion on clean documents;
- hallucinated values for unreadable fields;
- malformed JSON or schema violations (strict structured outputs).

Day/month ambiguity is **untested** by the labelled set: no labelled invoice has an ambiguous numeric
date. Grounding cannot detect day/month swaps either, because a swapped `07/02/2026` renders identically
under the other convention ([error_injection.md](experiments/error_injection.md): 80% of plausible
swaps caught, only where the format disambiguates).

## 5. Limits of this analysis

- **0 errors on the evaluation set**, so causes are characterised on synthetic degradations and
  unlabelled documents. Synthetic blur and JPEG damage are not real phone photos (glare, folds,
  perspective, handwriting). These rates do not generalise.
- **Small counts.** 1–3 runs per configuration, 10 documents per set. Differences of one error per
  run are within noise.
- **Some design decisions are in-sample:**
  - the tax-rate check was motivated by a held-out error;
  - the mixed-rate exemption by two unlabelled documents;
  - the whole-token invoice-number rule and the tax-component rule by the first error-injection
    and grounding measurements.

  All are disclosed in [improvement_report.md](improvement_report.md).

---

## Appendix: original v1 analysis

The text below is the analysis written for the first submission (v1), kept unchanged.


Everything below comes from real runs recorded under `reports/` (file references
are given for each claim). No example is invented. "GT" = ground truth.

> **Status after the remediation phase (read this first).** Sections 1–6 describe the **v1**
> system as originally analysed and are kept as the record of what was found. What changed since,
> all measured in [remediation_report.md](remediation_report.md):
>
> | failure (section) | status |
> |---|---|
> | B2 `MAP/D/09-0312` → `MAP/09-0312` | **fixed**: cause was photos not being upscaled like PDF pages; wrong 6/6 runs → correct 9/9 |
> | B1 L02 invoice-number misreads | reduced: wrong in 4/6 runs → 4/9 runs |
> | A `Pomfret` → `Parrot` | **not fixed**: now `Reefret`, in 9/9 runs, and usually unflagged (source too degraded) |
> | E1 low-contrast false flags | **fixed**: image-quality and rotation heuristics are no longer review triggers (0 true positives measured) |
> | E2 U02 consistency false flag | **arithmetic arbitration** implemented; resolves purely numeric disagreements only when the first read passes every check and the re-read fails one |
> | fix #1 below (repeated-printing cross-check) | **tried and removed**: the model copies its first reading into every occurrence (0 true positives, 1 false positive); replaced by deterministic text-layer grounding for digital PDFs |
> | §6 "repair never fired" | it fired once (final batch, U02) and did not fix the error |
>
> Final mild-stress numbers (3 runs): 1.67 field errors per run (v1: 3.00), 0 false flags (v1: 2.0),
> 1.00 silent errors per run (v1: 1.33), flag precision 100% and recall 40% (v1: 45% / 56%).

### 1. The headline result has no errors, so where do failures come from?

The final configuration extracts the 10 labelled documents with **80/80 fields
correct**, repeated across **3 independent fresh runs** of the pre-registered
configuration (`baseline_v1`, `baseline_rep2`, `baseline_rep3`) and every ablation
(`image_only`, `text_only`, `effort_none`, `gpt5nano`). See
[experiments/summary.md](experiments/summary.md).

That is good news but weak evidence. With 10 documents, 10/10 correct is compatible with a
true per-document accuracy as low as ~69% (95% Clopper-Pearson lower bound). The labelled
documents are clean: 7 digital PDFs with a text layer and 3 legible scans/photos. A perfect
score on them says nothing about *how the system fails*.

To find failure modes without inventing data, I built **stress sets** from the
labelled documents (`scripts/make_stress_set.py`). Every document is re-rendered as a
degraded image (downscaled, skewed, blurred, JPEG-compressed, plus noise for
"severe"). The content is unchanged, so the original GT files are copied byte-for-byte
and stay valid. Two levels:

| set | simulation | field accuracy (no verify) | docs with error |
|---|---|---|---|
| `stress_mild` | a poor phone scan (long edge 1100 px, 1.5° skew, blur 0.8, JPEG q45) | 97.5% (78/80) | 2/10 |
| `stress_severe` | barely legible to a human (800 px, 3°, blur 1.3, noise, JPEG q25) | 27.5% (22/80) | 9/10 |

`stress_mild` was run **4 times independently** (one plain, three with the
consistency check, each a fresh set of API calls), so stochastic and systematic
errors can be told apart. The unlabelled documents add qualitative evidence (no GT;
any judgement about them is my own reading of the image and is labelled as such).

### 2. Error summary

| # | category | where seen | frequency | caught by validation? | caught by review flag? |
|---|---|---|---|---|---|
| A | Plausible word substitution in a line-item description (OCR-like, poor scan) | stress_mild L09 | 4/4 runs | no (text field) | 2 of 3 verify runs |
| B1 | Stochastic misread of identifier digits | stress_mild L02 | 3/4 runs | no | 3 of 3 verify runs |
| B2 | Systematic dropped segment of an invoice number | stress_mild L10 | 4/4 runs | no | **0 of 3 runs** |
| C | Page illegible: the model abstains (nulls + `illegible_fields`) | stress_severe, 9 docs | every run | yes (`insufficient_data`) | 9/9 |
| D | Matcher false negative (`L.L.P.` vs `LLP`) | stress_severe L04 | 1 field | n/a | n/a |
| E | Review-flag false positives (correct document flagged) | stress_mild L05/L07; U02 | every run | n/a | the flag is the error |
| F | Inconsistency in the source document itself | U04 (unlabelled) | 1 doc | yes (row check) | yes (correct behaviour) |

Categories that were looked for and **not observed** in any run: hallucinated values
for unreadable fields; invoice/non-invoice misclassification (L08 PO, U06 bank statement and
U10 cover letter were all correct, and the severe-stress L08 was still recognised as a PO);
date errors (but note: no *labelled* invoice has a genuinely ambiguous numeric date, since
L03 `23/06/2026`, L05 `27/01/2026` and L06 `30-05-2026` all have day > 12, so day/month
disambiguation is **untested by the eval**; unlabelled U09 prints `07/02/2026`, extracted as
2026-02-07, which is consistent with its INR currency and its due date `21/02/2026`, but that
is my reading, not GT); currency errors;
total vs subtotal vs amount-due confusion; missed or extra line-item rows; malformed JSON
or schema violations (**0 of 184 real API calls**, thanks to strict structured outputs).

### 3. Failure categories in detail

### A. Plausible word substitution on degraded scans

**Example** (`experiments/stress_mild/eval_results.json`, L09 row 3):

| | description | quantity | amount |
|---|---|---|---|
| GT | `Pomfret, whole, kg` | 9.5 | 6080.0 |
| prediction | `Parrot, whole, kg` | 9.5 | 6080.0 |

In another run the same row came back as `Parrot, white, kg`.

**Cause.** At ~800 px wide, after blur and JPEG compression, the word "Pomfret" is a few
grey pixels. A language model fills the gap with a *fluent* word ("Parrot", a real fish
name in Indian markets) instead of admitting uncertainty. Quantity and amount sit in
right-aligned numeric columns and survive, which is why the arithmetic checks pass.

**Why the system missed it.** Every validation layer is numeric. Totals reconcile and
qty × unit price = amount, so nothing looks wrong. The model did not list the field as
illegible either.

**Fix.** (1) The self-consistency check (now on by default for scans) catches it when the
two reads differ: it did so in 2 of 3 runs (`Parrot` vs `Pomfret` / `Parrot fish`).
(2) Higher-resolution crops of the item table for scans, so the word is actually legible
(see fix #2 below). Expected impact: medium. Descriptions matter less to payables than
amounts and identifiers.

### B. Identifier misreads: stochastic (B1) vs systematic (B2)

**B1 example** (L02, stress_mild): GT `SGS/1847`. Predictions in three runs:
`SGS/0417`, `SGS/0347`, `SGS/2047`. A different wrong answer each time.

**B2 example** (L10, stress_mild): GT `MAP/D/09-0312`, prediction `MAP/09-0312` in
**all 4 runs**.

**Cause.** B1 is genuine uncertainty: blurred digits, and the model samples a different
guess every time. B2 is a *systematic* bias: the short "D/" segment between two
slash-separated groups is consistently merged away, likely because `MAP/09-0312` looks like a
more "typical" invoice-number pattern.

**Why the system missed it.** Again no numeric check applies. The self-consistency check
catches B1 every time (3/3: two different wrong reads disagree with each other) but
**cannot catch B2** (0/3): both reads make the same mistake. **Self-consistency only
detects uncertainty, not bias.**

**Fix.**
- The invoice number is often printed twice. In L10 the page footer repeats
  `Metro Auto Parts | MAP/D/09-0312`. Ask the model to transcribe every occurrence of the
  invoice number separately and flag any disagreement between them in code. That targets B2
  directly, with no extra call.
- A zoomed crop of the header region at higher resolution for scans.

Expected impact: **high**. The invoice number is the key used for duplicate-payment
detection, so a silently wrong one is the most expensive error in this list.

### C. Illegible pages: abstention, not hallucination

**Example** (`experiments/stress_severe/eval_results.json`). For 9 of 10 severe-stress
invoices the model returned `is_invoice: true` with most fields `null`, listing them in
`illegible_fields`, e.g. L02:

```json
{"is_invoice": true, "vendor": null, "invoice_number": null, "invoice_date": null,
 "currency": null, "line_items": null, "tax_total": null, "total": null}
illegible_fields: ["vendor", "invoice_number", "invoice_date", "currency", "line_items", ...]
```

Every non-null value it did return was correct (e.g. vendor `Northwind Logistics Ltd`
for L01, `Metro Auto Parts` and currency `INR` for L10), apart from one matcher issue (D below).

**Why this is counted as an error.** The eval compares against the GT values; a null is
not a match. That is the right accounting: the data was not extracted.

**How the system handled it.** Correctly. All 9 documents ended with financial status
`insufficient_data` and `needs_review = true` (flag precision and recall 100% on this set).
This is the behaviour the prompt asks for ("a null is acceptable, a guessed value is not"),
and the review policy turns it into a routing decision.

**Fix.** None needed for safety. To reduce the manual-review volume, the pre-API image-quality
gate could reject such pages earlier (saving the ~$0.0003 call), but its current blur
threshold is deliberately conservative (see `src/preprocessing/pages.py`).

### D. Matcher false negative

L04 (stress_severe): prediction `Meridian Print & Pack L.L.P.` vs GT `Meridian Print & Pack LLP`.
The normaliser turns punctuation into spaces, giving `l l p` vs `llp`, so this counts as wrong.
A finance user would treat these as the same company. I **did not change the matching rule**
after seeing this case, because tuning eval rules on observed errors is how scores get inflated.
If the rule were changed (delete periods instead of replacing them with spaces), it should be
done up front and re-run on all sets. Impact on the headline: none (clean-set vendors match).

### E. Review-flag false positives

1. **Low-contrast warnings** (stress_mild L05 and L07, every run): the image-quality heuristic
   flagged two documents that were extracted perfectly. Over all stress_mild runs this signal
   has **0 true positives and 8 false positives**. It adds review cost without catching errors.
2. **Consistency disagreement where the first read was right** (U02, unlabelled; my own check
   against the image, not GT). Read 1: line 1 = 6 × 64.00 = 384.00. Read 2: 4 / 256.00. Read 1
   matches the image and reconciles with the printed subtotal (831.75). Read 2 does not:
   its own financial check failed (703.75 vs subtotal 831.75; see
   `outputs/_meta/U02.meta.json`, attempt `verify_initial`).

**Fix.** (a) Drop low-contrast as a standalone review trigger, or require it together with
another signal. (b) **Arithmetic arbitration**: when the two reads disagree only on
numeric fields and exactly one of them passes the totals, subtotal and row checks, accept that
read without flagging. U02 is exactly this case. Expected impact: lower review volume with no
loss of recall on any error observed so far.

### F. Inconsistent source document (not a model error)

U04 (unlabelled, digital PDF): the document prints row 1 as *1440 instance-hours × $0.09 =
$122.40*, but 1440 × 0.09 = 129.60. The extractor copied the printed values (as instructed:
"never correct a value") and the row-level check flagged it:
`quantity 1440.0 x unit price 0.09 = 129.60, but amount is 122.40`. This is the system
working as intended: a human should decide whether the vendor under-billed or the unit price
is rounded.

### 4. What to fix first

Prioritised by expected impact on *money at risk* per unit of effort:

| priority | fix | addresses | effort | evidence |
|---|---|---|---|---|
| 1 | Cross-check repeated identifiers (invoice no. in header vs footer/running header) | B2 (systematic, invisible to every current check) | small: prompt field + code compare, no extra call | L10 4/4 runs wrong, 0/3 caught |
| 2 | High-resolution crops of header + item table for image-only pages | A, B1, B2 at the source | medium; ~+2k input tokens (~$0.0002) per scan | all stress_mild errors were small-text misreads |
| 3 | Arithmetic arbitration of consistency disagreements | E2 (false flags) | small | U02 |
| 4 | Remove low-contrast as a standalone review trigger | E1 | trivial | 0 TP / 8 FP |
| 5 | Larger, harder labelled set (real phone photos, multi-page, credit notes, discounts outside the table, round-off lines) | evaluation blind spots | labelling effort | clean set saturates at 100% under every configuration |

Fix 5 matters more than it looks. The labelled set cannot distinguish image-only from
hybrid from text-only input, or effort `low` from `none`, or `gpt-6-luna` from `gpt-5-nano`.
Every one of them scores 100%. Until there is harder labelled data, configuration choices
can be justified on cost and robustness reasoning, not on measured accuracy differences.

### 5. Human-review flag: does it concentrate the errors?

| set | config | flagged (with error / without) | not flagged (with error / without) | recall | precision |
|---|---|---|---|---|---|
| labelled (official) | final | 0 / 0 | 0 / 10 | n/a (no errors) | n/a |
| stress_mild | no verify | 0 / 2 | 2 / 6 | **0%** | 0% |
| stress_mild ×3 | verify on | 2/2, 2/2, 1/2 | 1/5, 1/5, 2/5 | 67%, 67%, 33% (mean 56%) | 50%, 50%, 33% |
| stress_severe | either | 9 / 0 | 0 / 1 | 100% | 100% |

Without the consistency check, the flag was blind to every text-field error on legible-but-degraded
scans. That measured gap is why `VERIFY_SCANS` exists and is on by default. With it, the flag
catches the uncertain errors but still misses the systematic one (B2). Fix #1 targets that.
Precision is held down mostly by the low-contrast heuristic (fix #4).
The counts are small (3 erroneous documents per run), so these percentages are indicative,
not estimates with tight error bars.

### 6. Things that did not work, or did not matter

- **The validation-driven repair call never fired.** Over 184 real API calls, strict JSON-schema
  structured outputs produced 0 parse failures and 0 schema failures. The only arithmetic failure
  was in a consistency re-read (where repair is deliberately disabled). The repair path is
  unit-tested but has **no measured benefit** on this data. It stays as a safety net for
  providers or models without schema enforcement.
- **Hybrid input (image + PDF text layer) did not measurably beat image-only or text-only.**
  All three score 100% on the labelled set. Text-only is ~40% cheaper on digital PDFs. I kept
  hybrid because the text layer's reading order is scrambled for tables (seen in L01) while the
  image carries layout, but this is reasoning, not a measured win.
- **Self-consistency cannot detect systematic errors** (B2). It is a detector of uncertainty only.
- **Image-quality heuristics as review triggers**: the low-contrast rule had 0% precision.
