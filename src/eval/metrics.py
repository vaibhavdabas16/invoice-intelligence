"""Aggregate metrics over compared documents."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from src.eval.compare import FIELDS, DocumentComparison
from src.pipeline import PipelineResult


@dataclass
class EvalRecord:
    comparison: DocumentComparison
    result: PipelineResult
    ground_truth: dict[str, Any]


def _ratio(n: int, d: int) -> float | None:
    return round(n / d, 4) if d else None


def field_metrics(records: list[EvalRecord]) -> list[dict[str, Any]]:
    rows = []
    n = len(records)
    for f in FIELDS:
        correct = sum(1 for r in records if r.comparison.fields[f].correct)
        exact = sum(1 for r in records if r.comparison.fields[f].exact)
        row: dict[str, Any] = {
            "field": f,
            "accuracy": _ratio(correct, n),
            "correct": correct,
            "total": n,
            "exact_raw_match": exact,
        }
        if f == "vendor":
            row["lenient_suffix_insensitive"] = sum(1 for r in records if r.comparison.fields[f].lenient)
        rows.append(row)
    return rows


def line_item_metrics(records: list[EvalRecord]) -> dict[str, Any]:
    gt_items = pred_items = matched = desc = qty = amt = order_ok = docs = 0
    for r in records:
        li = r.comparison.line_items
        if li is None:
            # GT and prediction disagree on whether there are line items at all.
            gt_li = r.ground_truth.get("line_items") or []
            gt_items += len(gt_li)
            continue
        docs += 1
        gt_items += li.gt_count
        pred_items += li.pred_count
        matched += li.matched_items
        desc += li.attribute_correct("description")
        qty += li.attribute_correct("quantity")
        amt += li.attribute_correct("amount")
        order_ok += 1 if li.order_preserved else 0
    return {
        "documents_with_items_compared": docs,
        "gt_items": gt_items,
        "predicted_items": pred_items,
        "exactly_matched_items": matched,
        "item_precision": _ratio(matched, pred_items),
        "item_recall": _ratio(matched, gt_items),
        "description_accuracy": _ratio(desc, gt_items),
        "quantity_accuracy": _ratio(qty, gt_items),
        "amount_accuracy": _ratio(amt, gt_items),
        "documents_with_order_preserved": order_ok,
    }


def review_flag_metrics(records: list[EvalRecord]) -> dict[str, Any]:
    flagged_err = flagged_ok = unflagged_err = unflagged_ok = 0
    errors_in_flagged = errors_total = 0
    for r in records:
        has_error = not r.comparison.all_correct
        n_err = len(r.comparison.wrong_fields)
        errors_total += n_err
        if r.result.needs_review:
            errors_in_flagged += n_err
            if has_error:
                flagged_err += 1
            else:
                flagged_ok += 1
        else:
            if has_error:
                unflagged_err += 1
            else:
                unflagged_ok += 1
    return {
        "flagged": flagged_err + flagged_ok,
        "flagged_with_errors": flagged_err,
        "flagged_without_errors": flagged_ok,
        "unflagged_with_errors": unflagged_err,
        "unflagged_without_errors": unflagged_ok,
        # precision: P(doc has an error | flagged); recall: P(flagged | doc has an error)
        "precision": _ratio(flagged_err, flagged_err + flagged_ok),
        "recall": _ratio(flagged_err, flagged_err + unflagged_err),
        "field_errors_total": errors_total,
        "field_errors_in_flagged_docs": errors_in_flagged,
        "share_of_field_errors_caught": _ratio(errors_in_flagged, errors_total),
    }


def summary_metrics(records: list[EvalRecord]) -> dict[str, Any]:
    n = len(records)
    total_cells = n * len(FIELDS)
    correct_cells = sum(r.comparison.correct_count for r in records)
    exact_cells = sum(sum(1 for f in r.comparison.fields.values() if f.exact) for r in records)
    gt_non_invoices = [r for r in records if r.ground_truth.get("is_invoice") is False]
    validation_failures = [
        r.comparison.document_id
        for r in records
        if r.result.status == "failed" or (r.result.financial is not None and not r.result.financial.passed)
    ]
    return {
        "documents": n,
        "overall_field_accuracy": _ratio(correct_cells, total_cells),
        "overall_exact_raw_accuracy": _ratio(exact_cells, total_cells),
        "correct_fields": correct_cells,
        "total_fields": total_cells,
        "exact_document_match_rate": _ratio(sum(1 for r in records if r.comparison.all_correct), n),
        "documents_with_at_least_one_error": sum(1 for r in records if not r.comparison.all_correct),
        "pipeline_failures": sum(1 for r in records if r.result.status == "failed"),
        "validation_failures": len(validation_failures),
        "validation_failure_documents": validation_failures,
        "non_invoices_in_ground_truth": len(gt_non_invoices),
        "non_invoices_correctly_classified": sum(
            1 for r in gt_non_invoices if r.comparison.fields["is_invoice"].correct
        ),
        "invoices_misclassified_as_non_invoice": sum(
            1
            for r in records
            if r.ground_truth.get("is_invoice") is True
            and r.result.invoice is not None
            and not r.result.invoice.is_invoice
        ),
    }


def classification_confusion(records: list[EvalRecord]) -> dict[str, Any]:
    """Invoice vs non-invoice confusion matrix (rows = ground truth, columns = prediction).

    A pipeline failure is its own column: it is neither a correct nor a wrong
    classification, and hiding it in either cell would distort the matrix.
    """
    cells = {gt: {"invoice": 0, "non_invoice": 0, "failed": 0} for gt in ("invoice", "non_invoice")}
    predicted_types: dict[str, dict[str, int]] = {"invoice": {}, "non_invoice": {}}
    for r in records:
        gt = "invoice" if r.ground_truth.get("is_invoice") else "non_invoice"
        inv = r.result.invoice
        pred = "failed" if inv is None else ("invoice" if inv.is_invoice else "non_invoice")
        cells[gt][pred] += 1
        dtype = (r.result.auxiliary or {}).get("document_type", "n/a")
        predicted_types[gt][dtype] = predicted_types[gt].get(dtype, 0) + 1
    tp, fn = cells["invoice"]["invoice"], cells["invoice"]["non_invoice"]
    fp, tn = cells["non_invoice"]["invoice"], cells["non_invoice"]["non_invoice"]
    return {
        "matrix": cells,
        "predicted_document_types": predicted_types,
        "invoice_precision": _ratio(tp, tp + fp),
        "invoice_recall": _ratio(tp, tp + fn),
        "non_invoice_recall": _ratio(tn, tn + fp),
    }


def _short(value: Any, limit: int = 60) -> str:
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else repr(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def per_document_markdown(records: list[EvalRecord], dataset_label: str) -> str:
    """One row per document: what was wrong, what the validators said, whether it was flagged.

    Generated, not hand-written, so it always matches the run it came from.
    """
    lines = [
        f"# Per-document results: {dataset_label}",
        "",
        "Generated by `python -m src.eval.run`. `flagged` = needs_review. A wrong field in an unflagged",
        "document is a **silent error** (the case that matters most in production).",
        "",
        "| doc | source | status | fields ok | wrong fields (predicted vs expected) | totals check | flagged "
        "| review reasons |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        c = r.comparison
        details = []
        for name in c.wrong_fields:
            f = c.fields[name]
            if name == "line_items" and c.line_items is not None:
                pred = (r.result.invoice.line_items if r.result.invoice else None) or []
                gt_items = r.ground_truth.get("line_items") or []
                for pair in c.line_items.pairs:
                    if pair.exact_match:
                        continue
                    p = pred[pair.pred_index].description if pair.pred_index is not None else "(missing)"
                    g = gt_items[pair.gt_index]["description"] if pair.gt_index is not None else "(extra row)"
                    details.append(f"item: {_short(p, 40)} vs {_short(g, 40)}")
            else:
                details.append(f"{name}: {_short(f.predicted)} vs {_short(f.expected)}")
        res = r.result
        fin = res.financial.status.value if res.financial else (res.error or {}).get("category", "-")
        kind = (res.preprocessing or {}).get("source_kind", "-")
        silent = c.wrong_fields and not res.needs_review
        flagged = "yes" if res.needs_review else ("**NO (silent error)**" if silent else "no")
        reasons = "; ".join(x[:90] for x in (res.review.reasons if res.review else [])) or "-"
        lines.append(
            f"| {c.document_id} | {kind} | {res.status} | {c.correct_count}/{len(FIELDS)} | "
            f"{'<br>'.join(details) or '-'} | {fin} | {flagged} | {reasons} |"
        )
    return "\n".join(lines) + "\n"
