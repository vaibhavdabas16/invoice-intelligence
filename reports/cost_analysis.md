# Cost analysis

Per-document data: [cost_per_document.csv](cost_per_document.csv), summary:
[cost_summary.json](cost_summary.json) (written by `python -m src.main batch`).
Eval-run costs: [eval_cost_per_document.csv](eval_cost_per_document.csv).

## 1. Method

```
cost = uncached_input_tokens  × $0.10 / 1M
     + cached_input_tokens    × $0.01 / 1M
     + output_tokens          × $0.50 / 1M      (gpt-6-luna, Standard tier)
```

- **Token counts are the ones the OpenAI API reports** in `response.usage` for every call. They
  are never estimated. Image tokens are inside `input_tokens`; reasoning tokens are inside
  `output_tokens` (both billed accordingly).
- OpenAI reports prompt-cache hits inside `input_tokens` (`input_tokens_details.cached_tokens`).
  The adapter splits them out so they are priced at the cached rate.
- Every call is counted: the initial extraction, consistency re-reads, repair calls (none happened)
  and calls whose output later failed validation. Transient retries that got no response are not
  billed by the provider and are counted separately (`transient_retries`, 0 in all runs).
- Prices: [OpenAI pricing page](https://developers.openai.com/api/docs/pricing), checked
  **2026-11-19**, recorded with the date in `src/cost/pricing.py`. Unknown models return
  `cost = null` rather than $0, so a pricing gap is visible.
- **Upper bound**: `cost_usd_no_prompt_cache` bills every input token at the full rate. This is the
  worst case for a never-seen document.

## 2. Results

### Headline: cost per document type (final batch, all 20 documents, final configuration)

Generated in `reports/cost_summary.json` (`type:*` keys). "Upper bound" bills every input token at
the full rate, i.e. no benefit from OpenAI's automatic prompt caching.

| document type | documents | API calls | avg cost / document | upper bound | avg output tokens | avg latency |
|---|---|---|---|---|---|---|
| **digital invoice** (PDF with text layer) | 12 | 12 (1 each) | **$0.00045** | $0.00059 | 343 | 4.9 s |
| **scanned or photographed invoice** | 5 | 11 (initial + re-read; 1 repair) | **$0.00089** | $0.00143 | 1,190 | 16.2 s |
| non-invoice | 3 | 3 | $0.00030 | $0.00045 | 82 | 3.4 s |
| all 20 | 20 | 26 | $0.00054 | $0.00078 | 516 | 7.5 s |

A scanned invoice costs about 2× a digital one: a second independent read, plus more reasoning
tokens on degraded images.

### Incremental cost of each improvement (improvement phase)

| change | extra API calls | per-document cost impact | kept? |
|---|---|---|---|
| evaluation: confusion matrix, per-document report, dataset labels | 0 | none | yes |
| value grounding with page + source text (digital PDFs) | 0 | none (string search, milliseconds) | yes |
| tax-rate consistency check | 0 | none | yes |
| arbitration made annotation-only (no automatic acceptance) | 0 | none in API cost; more human reviews only when two reads disagree numerically (2 cases observed so far) | yes |
| UI workflow / evidence panels | 0 | none | yes |
| token-confidence mode T (effort `none` + logprobs) | 0 | **cheaper**: −26% on mild stress, −49% on held-out (far fewer reasoning/output tokens), but more raw errors and ~2× review load | **opt-in only** (`TOKEN_CONFIDENCE=true`, `LLM_EFFORT=none`) |
| photo upscaling (remediation phase) | 0 | ~+16% image tokens on photos smaller than 1568 px | yes |
| independent re-read for scans (earlier) | +1 per scan/photo invoice | ~+$0.0004 per scan | yes |

### The realistic number: first-time documents (10 unlabelled, final config)

When first sent, these documents shared only the system-prompt prefix (1,563 tokens) with earlier
requests, so only that prefix was served from OpenAI's prompt cache. That is the steady state for a
new invoice in production.

The figures below are from the **final** batch. The final configuration sends byte-identical
requests to v1 for PDFs, so U01 and U03–U10 reuse the usage recorded when they were first
extracted. U02 (a photo, now upscaled) was a fresh request; this time it needed a repair call too.
The v1 figures are given in brackets.

| | value |
|---|---|
| documents | 10 (6 digital invoices, 1 scanned PDF, 1 phone photo, 2 non-invoices) |
| API calls | 13 (10 extractions + 2 consistency re-reads on U02, U05 + 1 repair on U02) [v1: 12] |
| total cost | **$0.005226** [v1: $0.004576] |
| **average cost per document** | **$0.000523** (upper bound without prompt caching: $0.000764) [v1: $0.000458 / $0.000663] |
| average input tokens per document (uncached + cached) | 2,445 + 2,679 = 5,124 [v1: 4,596] |
| average output tokens per document | 503 [v1: 406] |
| average API latency per document | 10.2 s [v1: 8.6 s] |

By document type (from `cost_per_document.csv`):

| type | example | calls | input (uncached + cached) | output | cost |
|---|---|---|---|---|---|
| digital invoice | U01 | 1 | 2,395 + 1,563 | 234 | $0.00037 |
| non-invoice | U10 | 1 | 2,350 + 1,563 | 81 | $0.00029 |
| scanned PDF (with re-read) | U05 | 2 | 2,194 + 5,314 | 835 | $0.00069 |
| phone photo (with re-read; v1) | U02 | 2 | 1,853 + 4,973 | 1,582 | $0.00103 |
| phone photo, upscaled, + repair (final) | U02 | 3 | 3,133 + 8,973 | 2,545 | $0.00168 |

### Labelled documents

| run | avg $/doc | note |
|---|---|---|
| baseline_v1 (first time these documents were sent) | **$0.000484** | realistic first-time cost, no consistency check |
| official eval / batch (final config) | $0.000320 billed / $0.000795 upper bound | identical requests had already been sent in earlier experiments, so OpenAI served almost all input from its prompt cache (avg 3.9 uncached vs 5,283 cached input tokens). The billed number understates first-time cost; use the upper bound or the unlabelled figure. |

### Where the tokens go (baseline_v1, average per call)

| component | ~tokens | source of the estimate |
|---|---|---|
| page image (1109 × 1568 px) | ~2,200 | (image-only − text-only) run totals spread over the 7 digital PDFs, plus the text layer; consistent with 35 × 49 patches × 1.2 ≈ 2,060 |
| system prompt + JSON schema | ~1,560 | constant cached prefix reported by the API |
| PDF text layer | ~400 | (hybrid − image-only) run totals spread over the 7 digital PDFs |
| output JSON | ~390 | 82 for a non-invoice, up to 1,082 for the 24-row L04 |

The image is the largest cost component, and image tokens scale with pixel area. Capping the long
edge at 1568 px is the main input-cost lever.

## 3. Spend for this whole project

| phase | real API calls | spend |
|---|---|---|
| original build: evaluation, ablations, stress tests, batch | 184 | ≈ $0.06 |
| remediation: experiments E1/E2, v2 and final regression runs, batch | 365 | ≈ $0.144 |
| **total** | ~550 | **≈ $0.20** |

From provider-reported usage of calls actually sent (cache hits excluded; computed by
`scripts/summarize_experiments.py` and [remediation_report.md §8](remediation_report.md)).
Ad-hoc smoke calls are not in those files. The OpenAI usage dashboard is the authoritative figure.

**Per-document cost impact of the remediation changes:**
- **digital and scanned PDFs:** none (byte-identical requests);
- **photos smaller than 1568 px:** about +16% image tokens from upscaling (U02: 1,518 → 1,764 patches);
- **arbitration and text-layer grounding:** no API calls.

The removed v2 prompt had cost +12% per document, mostly from ~150 extra output tokens per invoice.

## 4. Projection and levers

At $0.0005–0.0008 per document, **1,000 invoices/month cost about $0.50–0.80**; the LLM bill is
negligible next to the cost of manual review. Levers, in order of usefulness:

| lever | effect | trade-off |
|---|---|---|
| Batch API | −50% | asynchronous (results within hours); fine for month-end backlogs |
| text-only for digital PDFs | ~−40% on digital PDFs (measured: $0.000295 vs $0.000484/doc) | loses the layout signal; matched hybrid on this set, untested on hard layouts |
| lower image resolution | roughly proportional to pixel area | small text on scans was already the main failure mode, so this would hurt accuracy |
| drop the consistency check | ~−40–50% on scans only | loses the only signal that catches uncertain text misreads |
| `gpt-5-nano` | cheaper per token, but measured **more** expensive here ($0.000524 vs $0.000484/doc): 2× output (448 reasoning tokens/call) and a higher image multiplier; deprecated | not worth it |

The cost driver that actually matters operationally is the **review rate**. In the unlabelled batch,
3 of 20 documents were flagged in v1 and 2 of 20 in the final version (both real problems: a
document that doesn't add up, and a misread amount). Each flagged document costs minutes of human
time, which is many orders of magnitude above the $0.0005 extraction cost. That is why the
remediation removed review triggers that only produced false flags (0 false flags in the final
stress runs, against 2 per mild run and 9 per rotated run in v1).
