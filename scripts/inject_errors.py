"""Synthetic error injection: does validation catch wrong values on digital PDFs?

Real digital-PDF extractions never had an error, so detection cannot be measured on
real mistakes. Instead we take each real extraction (outputs/<id>.json + its
metadata), inject ONE realistic error, and run the validators on the mutated
result:

  * grounding  - value-level evidence search in the PDF text layer (src/validation/evidence.py)
  * arithmetic - totals / subtotal / tax-line / row checks (src/validation/financial.py)

Mutation types are fixed in MUTATIONS below; each is applied with 3 seeds per document.
This measures the validators, not the model. Results are synthetic and say nothing about
how often such errors occur in practice.

Usage: python scripts/inject_errors.py   ->  reports/experiments/error_injection.md (+ .json)
"""

from __future__ import annotations

import copy
import json
import random
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ingestion.loader import load_file  # noqa: E402
from src.preprocessing.pages import prepare_document  # noqa: E402
from src.schema import LLMExtraction  # noqa: E402
from src.validation.evidence import collect_evidence  # noqa: E402
from src.validation.financial import check_financials  # noqa: E402

SEEDS = (1, 2, 3)


def _change_digit(text: str, rng: random.Random) -> str | None:
    positions = [i for i, ch in enumerate(text) if ch.isdigit()]
    if not positions:
        return None
    i = rng.choice(positions)
    new = rng.choice([d for d in "0123456789" if d != text[i]])
    return text[:i] + new + text[i + 1 :]


def _change_letter(text: str, rng: random.Random) -> str | None:
    positions = [i for i, ch in enumerate(text) if ch.isalpha()]
    if not positions:
        return None
    i = rng.choice(positions)
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    new = rng.choice([c for c in alphabet if c != text[i].lower()])
    return text[:i] + (new.upper() if text[i].isupper() else new) + text[i + 1 :]


def _mutate_number(value: float, rng: random.Random) -> float | None:
    text = f"{value:.2f}"
    for _ in range(10):
        changed = _change_digit(text, rng)
        if changed and float(changed) != value:
            return float(changed)
    return None


def m_invoice_number_drop_char(d, rng):
    s = d["invoice_number"]
    idx = [i for i, ch in enumerate(s) if ch.isalnum()]
    if len(idx) < 2:
        return None
    i = rng.choice(idx)
    d["invoice_number"] = s[:i] + s[i + 1 :]
    return d


def m_invoice_number_digit(d, rng):
    v = _change_digit(d["invoice_number"], rng)
    if v is None:
        return None
    d["invoice_number"] = v
    return d


def m_vendor_letter(d, rng):
    v = _change_letter(d["vendor"], rng)
    if v is None:
        return None
    d["vendor"] = v
    return d


def m_date_plus_one(d, rng):
    d["invoice_date"] = (date.fromisoformat(d["invoice_date"]) + timedelta(days=1)).isoformat()
    return d


def m_date_swap_day_month(d, rng):
    x = date.fromisoformat(d["invoice_date"])
    if x.day > 12 or x.day == x.month:
        return None  # not a plausible swap
    d["invoice_date"] = date(x.year, x.day, x.month).isoformat()
    return d


def m_total_digit(d, rng):
    v = _mutate_number(d["total"], rng)
    if v is None:
        return None
    d["total"] = v
    return d


def m_tax_digit(d, rng):
    if not d["tax_total"]:
        return None
    v = _mutate_number(d["tax_total"], rng)
    if v is None:
        return None
    d["tax_total"] = v
    return d


def m_line_amount_digit(d, rng):
    i = rng.randrange(len(d["line_items"]))
    v = _mutate_number(d["line_items"][i]["amount"], rng)
    if v is None:
        return None
    d["line_items"][i]["amount"] = v
    return d


def m_description_letter(d, rng):
    i = rng.randrange(len(d["line_items"]))
    v = _change_letter(d["line_items"][i]["description"], rng)
    if v is None:
        return None
    d["line_items"][i]["description"] = v
    return d


def m_total_is_subtotal(d, rng):
    # Realistic confusion: the printed subtotal taken as the total. It IS printed, so grounding passes.
    if d.get("printed_subtotal") is None or d["printed_subtotal"] == d["total"]:
        return None
    d["total"] = d["printed_subtotal"]
    return d


MUTATIONS = {
    "invoice_number: drop a character": m_invoice_number_drop_char,
    "invoice_number: change a digit": m_invoice_number_digit,
    "vendor: change a letter": m_vendor_letter,
    "invoice_date: +1 day": m_date_plus_one,
    "invoice_date: day/month swapped": m_date_swap_day_month,
    "total: change a digit": m_total_digit,
    "tax_total: change a digit": m_tax_digit,
    "line amount: change a digit": m_line_amount_digit,
    "description: change a letter": m_description_letter,
    "total replaced by printed subtotal": m_total_is_subtotal,
}


def llm_dict(doc_id: str) -> dict | None:
    out = json.loads((ROOT / "outputs" / f"{doc_id}.json").read_text(encoding="utf-8"))
    if not out.get("is_invoice"):
        return None
    aux = json.loads((ROOT / "outputs" / "_meta" / f"{doc_id}.meta.json").read_text(encoding="utf-8"))[
        "auxiliary_extraction"
    ]
    items = [
        {**item, "unit_price": up}
        for item, up in zip(out["line_items"], aux["unit_prices"] or [None] * len(out["line_items"]), strict=True)
    ]
    return {
        **out,
        "document_type": aux["document_type"],
        "line_items": items,
        "printed_subtotal": aux["printed_subtotal"],
        "tax_lines": aux["tax_lines"],
        "adjustments_outside_items": aux["adjustments_outside_items"],
        "ambiguities": [],
        "illegible_fields": [],
    }


def detect(data: dict, page_texts: list[str | None]) -> tuple[bool, bool]:
    llm = LLMExtraction.model_validate(data)
    inv = llm.to_invoice()
    tax_lines = [t.amount for t in llm.tax_lines or []]
    grounding = any(e.status == "unsupported" for e in collect_evidence(inv, page_texts, tax_lines))
    fin = check_financials(llm)
    arithmetic = (
        (not fin.passed) or fin.subtotal_matches is False or fin.tax_lines_match is False or bool(fin.row_mismatches)
    )
    return grounding, arithmetic


def main() -> int:
    docs = []
    for pdf in sorted((ROOT / "data" / "labelled").glob("*.pdf")) + sorted(
        (ROOT / "data" / "unlabelled").glob("*.pdf")
    ):
        prepared = prepare_document(load_file(pdf))
        if not prepared.has_text_layer:
            continue
        base = llm_dict(pdf.stem)
        if base is not None:
            docs.append((pdf.stem, base, [p.text_layer for p in prepared.pages]))

    # sanity: unmutated outputs must raise no alarm, otherwise detection numbers are meaningless
    baselines = {d: detect(base, pages) for d, base, pages in docs}
    baseline_alarms = {d: {"grounding": g, "arithmetic": a} for d, (g, a) in baselines.items() if g or a}

    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for name, fn in MUTATIONS.items():
        for doc_id, base, pages in docs:
            for seed in SEEDS:
                mutated = fn(copy.deepcopy(base), random.Random(f"{name}|{doc_id}|{seed}"))
                if mutated is None:
                    continue
                g, a = detect(mutated, pages)
                g0, a0 = baselines[doc_id]
                g, a = g and not g0, a and not a0  # count only NEW alarms caused by the injected error
                s = stats[name]
                s["cases"] += 1
                s["grounding"] += g
                s["arithmetic"] += a
                s["either"] += g or a
                s["neither"] += not (g or a)

    grounding_fp = sum(1 for v in baseline_alarms.values() if v["grounding"])

    def pct(n: int, d: int) -> str:
        return f"{100 * n / d:.0f}%" if d else "n/a"

    lines = [
        "# Synthetic error injection on digital PDFs",
        "",
        "Generated by `python scripts/inject_errors.py`. One error is injected into a real extraction of each",
        f"of the {len(docs)} digital-PDF invoices (6 labelled + 6 unlabelled), {len(SEEDS)} seeds per mutation type;",
        "cases where a mutation does not apply (e.g. no tax) are skipped. This measures the validators,",
        "not the model, and says nothing about how often such errors occur in practice.",
        "",
        f"Unmutated outputs raising an alarm: {baseline_alarms or 'none'}. A document's own baseline alarm is",
        "not credited as detection: only NEW alarms caused by the injected error count. (U04's arithmetic alarm",
        "is a real inconsistency in the source document, not a false positive.)",
        f"Grounding false positives on unmutated outputs: **{grounding_fp}**",
        "",
        "| injected error | cases | caught by grounding | caught by arithmetic | caught by either | missed |",
        "|---|---|---|---|---|---|",
    ]
    total = defaultdict(int)
    for name in MUTATIONS:
        s = stats[name]
        for k in ("cases", "grounding", "arithmetic", "either", "neither"):
            total[k] += s[k]
        lines.append(
            f"| {name} | {s['cases']} | {pct(s['grounding'], s['cases'])} | {pct(s['arithmetic'], s['cases'])} | "
            f"{pct(s['either'], s['cases'])} | {s['neither']} |"
        )
    lines.append(
        f"| **all** | {total['cases']} | {pct(total['grounding'], total['cases'])} | "
        f"{pct(total['arithmetic'], total['cases'])} | {pct(total['either'], total['cases'])} | {total['neither']} |"
    )
    out = ROOT / "reports" / "experiments" / "error_injection.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out.with_suffix(".json")).write_text(
        json.dumps(
            {"baseline_alarms": baseline_alarms, "stats": stats, "documents": [d for d, _, _ in docs]}, indent=2
        ),
        encoding="utf-8",
    )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
