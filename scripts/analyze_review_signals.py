"""Analyse review signals across experiment runs (no API calls).

1. Configuration comparison: B (final, effort low) vs T (effort none + token confidence), on the
   mild stress set and on the held-out set stress_mild_b.
2. Correlated errors: for every wrong document in runs with the independent re-read, did the
   re-read disagree (error visible) or agree (error invisible to the re-read)?
3. Token-confidence threshold sensitivity, field level, for transparency. The decision threshold
   (0.5) was fixed before these runs and is not re-tuned here.

Usage: python scripts/analyze_review_signals.py -> reports/experiments/review_signals.md
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "reports" / "experiments"

GROUPS = {
    "mild / B (final, effort low)": [
        "final_stress_mild_r1",
        "final_stress_mild_r2",
        "final_stress_mild_r3",
        "final2_stress_mild",
    ],
    "mild / T (effort none + token confidence)": ["T_stress_mild_r1", "T_stress_mild_r2", "T_stress_mild_r3"],
    "held-out / B": ["B_heldout_r1", "B_heldout_r2", "final2_heldout"],
    "held-out / T": ["T_heldout_r1", "T_heldout_r2"],
    "severe / B": ["final_stress_severe"],
    "severe / T": ["T_stress_severe"],
    "labelled / B": ["v3_baseline", "../"],
    "labelled / T": ["T_labelled"],
}
THRESHOLDS = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
FIELD_MAP = {"vendor", "invoice_number", "invoice_date", "currency", "tax_total", "total"}


def load(run: str) -> dict | None:
    path = EXP / run / "eval_results.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def pct(n: float, d: float) -> str:
    return f"{100 * n / d:.0f}%" if d else "n/a"


def tax_rate_alarm(doc: dict) -> bool:
    """Re-apply src.validation.financial's tax-rate rule to a stored extraction (deterministic)."""
    sys.path.insert(0, str(ROOT))
    from src.schema import LLMExtraction
    from src.validation.financial import check_financials

    pred, aux = doc["prediction"], doc.get("auxiliary_extraction")
    if not pred or not pred.get("is_invoice") or not aux:
        return False
    items = [
        {**it, "unit_price": up}
        for it, up in zip(
            pred["line_items"] or [], aux.get("unit_prices") or [None] * len(pred["line_items"] or []), strict=False
        )
    ]
    llm = LLMExtraction.model_validate(
        {
            **pred,
            "line_items": items or None,
            "document_type": aux["document_type"],
            "printed_subtotal": aux["printed_subtotal"],
            "tax_lines": aux["tax_lines"],
            "adjustments_outside_items": aux["adjustments_outside_items"],
            "ambiguities": [],
            "illegible_fields": [],
        }
    )
    return bool(check_financials(llm).tax_rate_mismatches)


def rescored_flag(doc: dict) -> bool:
    """needs_review under the final rules: + tax-rate check, and no automatic acceptance of re-read disagreements."""
    cc = doc.get("consistency_check") or {}
    auto_accepted = bool(cc.get("disagreements")) and cc.get("resolved") is True
    return doc["needs_review"] or auto_accepted or tax_rate_alarm(doc)


def group_stats(runs: list[str], rescore: bool = False) -> dict | None:
    data = [r for r in (load(x) for x in runs) if r]
    if not data:
        return None
    errs = silent = flagged_err = flagged_ok = field_errors = 0
    cost = 0.0
    for r in data:
        for d in r["documents"]:
            wrong = bool(d["wrong_fields"])
            field_errors += len(d["wrong_fields"])
            errs += wrong
            if rescored_flag(d) if rescore else d["needs_review"]:
                flagged_err += wrong
                flagged_ok += not wrong
            else:
                silent += wrong
        cost += r["cost"]["all"]["avg_cost_per_document_usd"] or 0
    n = len(data)
    return {
        "runs": n,
        "field_errors": field_errors / n,
        "docs_with_error": errs / n,
        "silent": silent / n,
        "false_flags": flagged_ok / n,
        "recall": pct(flagged_err, errs),
        "precision": pct(flagged_err, flagged_err + flagged_ok),
        "cost": cost / n,
    }


def wrong_field_labels(doc: dict) -> set[str]:
    """Labels (as used by confidence/consistency) of the predicted fields that are wrong."""
    labels = {f for f in doc["wrong_fields"] if f in FIELD_MAP}
    if "line_items" in doc["wrong_fields"] and doc.get("line_item_alignment"):
        for pair in doc["line_item_alignment"]["pairs"]:
            if pair["pred_index"] is None:
                continue
            i = pair["pred_index"] + 1
            if not pair["description_ok"]:
                labels.add(f"line {i} description")
            if not pair["quantity_ok"]:
                labels.add(f"line {i} quantity")
            if not pair["amount_ok"]:
                labels.add(f"line {i} amount")
    return labels


def correlated_errors(run_names: list[str]) -> tuple[int, int, list[str]]:
    visible = invisible = 0
    examples = []
    for run in run_names:
        r = load(run)
        if not r:
            continue
        for d in r["documents"]:
            cc = d.get("consistency_check")
            if not d["wrong_fields"] or not cc or not cc.get("checked"):
                continue
            disagreeing = " ".join(cc.get("disagreements", []))
            labels = wrong_field_labels(d)
            hit = any(re.search(rf"\b{re.escape(lbl)}\b", disagreeing) for lbl in labels)
            if hit:
                visible += 1
            else:
                invisible += 1
                examples.append(f"{run}/{d['document_id']}: {sorted(labels)}")
    return visible, invisible, examples


def threshold_table(run_names: list[str]) -> list[str]:
    rows = []  # (min_prob, is_wrong)
    for run in run_names:
        r = load(run)
        if not r:
            continue
        for d in r["documents"]:
            conf = d.get("token_confidence")
            if not conf or d["prediction"] is None or not d["prediction"].get("is_invoice"):
                continue
            wrong = wrong_field_labels(d)
            for c in conf:
                rows.append((c["min_prob"], c["field"] in wrong))
    n_wrong = sum(w for _, w in rows)
    out = [
        f"Field-level, {len(rows)} extracted values from {len(run_names)} runs; {n_wrong} of them wrong.",
        "",
        "| threshold p < | wrong values flagged (recall) | flagged values that are wrong (precision) | values flagged |",
        "|---|---|---|---|",
    ]
    for t in THRESHOLDS:
        flagged = [(p, w) for p, w in rows if p < t]
        tp = sum(w for _, w in flagged)
        marker = " **(pre-registered)**" if t == 0.5 else ""
        out.append(f"| {t}{marker} | {tp}/{n_wrong} ({pct(tp, n_wrong)}) | {pct(tp, len(flagged))} | {len(flagged)} |")
    wrong_probs = sorted(round(p, 3) for p, w in rows if w)
    out += ["", f"Lowest-token probability of every wrong value: {wrong_probs}"]
    return out


def main() -> int:
    lines = [
        "# Review signals: configuration comparison, correlated errors, token confidence",
        "",
        "Generated by `python scripts/analyze_review_signals.py` from eval_results.json files.",
        "All sets except `labelled` are synthetic degradations of the 10 labelled documents; the held-out",
        "set `stress_mild_b` was created after every design decision and used only for this comparison.",
        "Per-run means. Counts are small: read as indicative.",
        "",
        "## 1. Configurations",
        "",
        "| set / config | runs | field errors | docs with error | silent errors | false flags | flag recall | flag precision | $/doc |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    rows = [(name, runs, False) for name, runs in GROUPS.items()]
    rows += [
        ("mild / B re-scored with final rules", GROUPS["mild / B (final, effort low)"], True),
        ("held-out / B re-scored with final rules", GROUPS["held-out / B"], True),
        ("held-out / T re-scored with final rules", GROUPS["held-out / T"], True),
        ("labelled / B re-scored with final rules", GROUPS["labelled / B"], True),
    ]
    for name, runs, rescore in rows:
        s = group_stats(runs, rescore)
        if s is None:
            lines.append(f"| {name} | missing | | | | | | | |")
            continue
        lines.append(
            f"| {name} | {s['runs']} | {s['field_errors']:.2f} | {s['docs_with_error']:.2f} | {s['silent']:.2f} | "
            f"{s['false_flags']:.2f} | {s['recall']} | {s['precision']} | {s['cost']:.6f} |"
        )

    lines += ["", "## 2. Correlated errors (both reads wrong in the same way)", ""]
    lines += [
        "| config | wrong documents with a re-read | re-read disagreed (error visible) | reads agreed (invisible to re-read) |",
        "|---|---|---|---|",
    ]
    examples_all = []
    for name, runs in (
        ("B (mild + held-out)", GROUPS["mild / B (final, effort low)"] + GROUPS["held-out / B"]),
        ("T (mild + held-out)", GROUPS["mild / T (effort none + token confidence)"] + GROUPS["held-out / T"]),
    ):
        v, inv, ex = correlated_errors(runs)
        lines.append(f"| {name} | {v + inv} | {v} | {inv} ({pct(inv, v + inv)}) |")
        examples_all += [f"- {name}: {e}" for e in ex]
    lines += ["", "Errors the re-read could not see:", *examples_all]

    lines += ["", "## 3. Token-confidence threshold sensitivity (T runs, mild + held-out)", ""]
    lines += threshold_table(GROUPS["mild / T (effort none + token confidence)"] + GROUPS["held-out / T"])
    out = EXP / "review_signals.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
