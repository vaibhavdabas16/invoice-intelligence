"""Evaluation harness.

    python -m src.eval.run                 # evaluate data/labelled with the default config
    python -m src.eval.run --no-cache      # force fresh API calls
    python -m src.eval.run --tag image_only --input-mode image   # an ablation, written to reports/experiments/

Ground truth is read-only input to the *comparison* step; it is never passed
to the extractor. Results are written to reports/ (or reports/experiments/<tag>/).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from tabulate import tabulate

from src.config import get_settings
from src.cost.report import cost_row, write_cost_files
from src.eval.compare import FIELDS, compare_document
from src.eval.metrics import (
    EvalRecord,
    classification_confusion,
    field_metrics,
    line_item_metrics,
    per_document_markdown,
    review_flag_metrics,
    summary_metrics,
)
from src.extraction.providers import create_provider
from src.ingestion.loader import discover_documents
from src.runner import run_many
from src.schema import InvoiceExtraction

log = logging.getLogger("eval")


def load_ground_truth(doc_path: Path) -> dict[str, Any] | None:
    gt_path = doc_path.with_suffix(".json")
    if not gt_path.exists():
        return None
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    try:
        InvoiceExtraction.model_validate(data)
    except ValidationError as exc:
        # Report, never "fix": modifying labels would make the eval meaningless.
        log.warning("ground truth %s does not satisfy the schema: %s", gt_path.name, exc)
    return data


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def render_report(
    records: list[EvalRecord],
    fields: list[dict[str, Any]],
    items: dict[str, Any],
    summary: dict[str, Any],
    review: dict[str, Any],
    cost: dict[str, Any],
    config: dict[str, Any],
    confusion: dict[str, Any] | None = None,
    dataset_label: str = "",
) -> str:
    out: list[str] = []
    if dataset_label:
        out.append(f"DATASET: {dataset_label}\n")
    out.append("FIELD-LEVEL ACCURACY (normalised match; see README 'Matching rules')")
    out.append(
        tabulate(
            [[f["field"], _pct(f["accuracy"]), f["correct"], f["total"], f["exact_raw_match"]] for f in fields],
            headers=["FIELD", "ACCURACY", "CORRECT", "TOTAL", "EXACT(raw)"],
            tablefmt="simple",
        )
    )
    vendor = next(f for f in fields if f["field"] == "vendor")
    out.append(
        f"\nvendor, lenient (legal suffix ignored, e.g. 'Acme Pvt Ltd' == 'Acme'): "
        f"{vendor['lenient_suffix_insensitive']}/{vendor['total']} (informational, not the headline)"
    )

    out.append("\nLINE ITEMS (item-level, aligned one-to-one)")
    out.append(
        tabulate(
            [
                ["GT items / predicted items", f"{items['gt_items']} / {items['predicted_items']}"],
                ["exact item matches (desc+qty+amount)", items["exactly_matched_items"]],
                ["item precision / recall", f"{_pct(items['item_precision'])} / {_pct(items['item_recall'])}"],
                ["description accuracy", _pct(items["description_accuracy"])],
                ["quantity accuracy", _pct(items["quantity_accuracy"])],
                ["amount accuracy", _pct(items["amount_accuracy"])],
            ],
            tablefmt="simple",
        )
    )

    out.append("\nSUMMARY")
    out.append(
        tabulate(
            [
                [
                    "overall field accuracy",
                    f"{_pct(summary['overall_field_accuracy'])} "
                    f"({summary['correct_fields']}/{summary['total_fields']})",
                ],
                ["overall exact (raw) accuracy", _pct(summary["overall_exact_raw_accuracy"])],
                ["exact document match rate", _pct(summary["exact_document_match_rate"])],
                [
                    "documents with >= 1 field error",
                    f"{summary['documents_with_at_least_one_error']}/{summary['documents']}",
                ],
                [
                    "validation failures (totals/schema)",
                    f"{summary['validation_failures']} {summary['validation_failure_documents'] or ''}",
                ],
                ["pipeline failures", summary["pipeline_failures"]],
                [
                    "non-invoices correctly classified",
                    f"{summary['non_invoices_correctly_classified']}/{summary['non_invoices_in_ground_truth']}",
                ],
            ],
            tablefmt="simple",
        )
    )

    out.append("\nPER DOCUMENT")
    out.append(
        tabulate(
            [
                [
                    r.comparison.document_id,
                    r.result.status,
                    f"{r.comparison.correct_count}/{len(FIELDS)}",
                    ", ".join(r.comparison.wrong_fields) or "-",
                    r.result.financial.status.value
                    if r.result.financial
                    else (r.result.error or {}).get("category", "-"),
                    "yes" if r.result.needs_review else "no",
                    r.result.api_calls,
                ]
                for r in records
            ],
            headers=["DOC", "STATUS", "FIELDS OK", "WRONG FIELDS", "TOTALS CHECK", "REVIEW", "CALLS"],
            tablefmt="simple",
        )
    )

    if confusion is not None:
        m = confusion["matrix"]
        out.append("\nINVOICE CLASSIFICATION (rows = ground truth, columns = prediction)")
        out.append(
            tabulate(
                [[gt, m[gt]["invoice"], m[gt]["non_invoice"], m[gt]["failed"]] for gt in ("invoice", "non_invoice")],
                headers=["", "pred invoice", "pred non-invoice", "pipeline failed"],
                tablefmt="simple",
            )
        )
        out.append(
            f"invoice precision {_pct(confusion['invoice_precision'])}, invoice recall "
            f"{_pct(confusion['invoice_recall'])}, non-invoice recall {_pct(confusion['non_invoice_recall'])}"
        )

    out.append("\nHUMAN-REVIEW FLAG vs ACTUAL ERRORS")
    out.append(
        tabulate(
            [
                ["flagged", review["flagged_with_errors"], review["flagged_without_errors"]],
                ["not flagged", review["unflagged_with_errors"], review["unflagged_without_errors"]],
            ],
            headers=["", "has error", "no error"],
            tablefmt="simple",
        )
    )
    out.append(
        f"precision {_pct(review['precision'])}, recall {_pct(review['recall'])}, "
        f"field errors inside flagged docs: {review['field_errors_in_flagged_docs']}/{review['field_errors_total']}"
    )

    c = cost["all"]
    out.append("\nCOST (provider-reported tokens x published price)")
    out.append(
        f"total ${c['total_cost_usd']:.6f} | avg/doc ${c['avg_cost_per_document_usd'] or 0:.6f} "
        f"(no-prompt-cache upper bound ${c['avg_cost_per_document_usd_no_prompt_cache'] or 0:.6f}) | "
        f"avg tokens in (uncached+cached)/out {c['avg_input_tokens']}+{c['avg_cached_input_tokens']}/"
        f"{c['avg_output_tokens']} | "
        f"API calls {c['total_api_calls']} (repairs {c['total_repair_calls']}, verify {c['total_verify_calls']}, "
        f"retries {c['total_transient_retries']})"
    )
    out.append(
        f"\nconfig: model={config['model']} effort={config['effort']} input_mode={config['input_mode']} "
        f"prompt={config['prompt_version']} repair_attempts={config['max_repair_attempts']} "
        f"verify_scans={config.get('verify_scans')} "
        f"numeric_tolerance={config['numeric_tolerance']}"
    )
    return "\n".join(out)


def write_outputs(
    out_dir: Path,
    records: list[EvalRecord],
    report_text: str,
    payload: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Long-format CSV: one row per document x field.
    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["document_id", "field", "correct", "exact_raw", "predicted", "expected", "note", "needs_review"]
        )
        for r in records:
            for name in FIELDS:
                f = r.comparison.fields[name]
                writer.writerow(
                    [
                        r.comparison.document_id,
                        name,
                        f.correct,
                        f.exact,
                        json.dumps(f.predicted, ensure_ascii=False),
                        json.dumps(f.expected, ensure_ascii=False),
                        f.note,
                        r.result.needs_review,
                    ]
                )
    (out_dir / "eval_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "eval_summary.txt").write_text(report_text + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the extractor on the labelled documents.")
    parser.add_argument(
        "--data-dir", type=Path, default=None, help="directory with documents + <stem>.json ground truth"
    )
    parser.add_argument("--no-cache", action="store_true", help="ignore the response cache and call the API")
    parser.add_argument("--tag", default=None, help="write results to reports/experiments/<tag>/ instead of reports/")
    parser.add_argument("--input-mode", choices=["hybrid", "image", "text"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--effort", default=None)
    parser.add_argument("--repair-attempts", type=int, default=None)
    parser.add_argument("--upscale-images", choices=["on", "off"], default=None, help="override UPSCALE_IMAGES")
    parser.add_argument(
        "--token-confidence", choices=["on", "off"], default=None, help="override TOKEN_CONFIDENCE (needs effort none)"
    )
    parser.add_argument(
        "--verify-scans", choices=["on", "off"], default=None, help="override VERIFY_SCANS (self-consistency check)"
    )
    parser.add_argument("--tolerance", type=float, default=0.01, help="numeric tolerance for eval matching")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    settings = get_settings().with_overrides(
        input_mode=args.input_mode,
        model=args.model,
        effort=args.effort,
        max_repair_attempts=args.repair_attempts,
        verify_scans=None if args.verify_scans is None else args.verify_scans == "on",
        upscale_images=None if args.upscale_images is None else args.upscale_images == "on",
        token_confidence=None if args.token_confidence is None else args.token_confidence == "on",
    )
    data_dir = args.data_dir or settings.data_dir / "labelled"
    if not data_dir.exists():
        print(f"error: {data_dir} not found. Place the sample pack in data/ (see README).", file=sys.stderr)
        return 2

    dataset_kind = (
        "labelled_evaluation_set"
        if data_dir.resolve() == (settings.data_dir / "labelled").resolve()
        else "synthetic_or_other_exploration_set"
    )
    dataset_label = (
        "LABELLED EVALUATION SET (the 10 documents with ground truth from the sample pack)"
        if dataset_kind == "labelled_evaluation_set"
        else f"EXPLORATION SET {data_dir.name} (derived/synthetic; not the evaluation set, no generalisation claims)"
    )
    docs = [p for p in discover_documents(data_dir) if p.with_suffix(".json").exists()]
    if not docs:
        print(f"error: no documents with ground truth found in {data_dir}", file=sys.stderr)
        return 2

    provider = create_provider(settings, use_cache=not args.no_cache)
    results = run_many(docs, provider, settings, workers=args.workers)

    records: list[EvalRecord] = []
    for path, result in zip(docs, results, strict=True):
        gt = load_ground_truth(path)
        assert gt is not None
        pred = result.invoice.to_output_dict() if result.invoice is not None else None
        records.append(EvalRecord(compare_document(path.stem, pred, gt, tolerance=args.tolerance), result, gt))

    fields = field_metrics(records)
    items = line_item_metrics(records)
    summary = summary_metrics(records)
    review = review_flag_metrics(records)
    confusion = classification_confusion(records)
    out_dir = settings.reports_dir / "experiments" / args.tag if args.tag else settings.reports_dir
    cost = write_cost_files([cost_row(r, "labelled") for r in results], out_dir, stem="eval_cost")
    config = {**results[0].settings_snapshot, "numeric_tolerance": args.tolerance}

    report_text = render_report(records, fields, items, summary, review, cost, config, confusion, dataset_label)
    print(report_text)

    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": config,
        "dataset": {"path": str(data_dir), "kind": dataset_kind, "label": dataset_label},
        "field_metrics": fields,
        "classification_confusion": confusion,
        "line_item_metrics": items,
        "summary": summary,
        "review_flag": review,
        "cost": cost,
        "documents": [
            {
                "document_id": r.comparison.document_id,
                "status": r.result.status,
                "error": r.result.error,
                "wrong_fields": r.comparison.wrong_fields,
                "fields": {
                    k: {"correct": v.correct, "exact": v.exact, "lenient": v.lenient, "note": v.note}
                    for k, v in r.comparison.fields.items()
                },
                "prediction": r.result.invoice.to_output_dict() if r.result.invoice else None,
                "ground_truth": r.ground_truth,
                "line_item_alignment": r.comparison.line_items.to_dict() if r.comparison.line_items else None,
                "financial_check": r.result.financial.to_dict() if r.result.financial else None,
                "needs_review": r.result.needs_review,
                "review_reasons": r.result.review.reasons if r.result.review else [],
                "auxiliary_extraction": r.result.auxiliary,
                "consistency_check": r.result.consistency,
                "evidence": r.result.evidence,
                "token_confidence": r.result.confidence,
                "attempts": [a.to_dict() for a in r.result.attempts],
            }
            for r in records
        ],
    }
    write_outputs(out_dir, records, report_text, payload)
    (out_dir / "eval_per_document.md").write_text(per_document_markdown(records, dataset_label), encoding="utf-8")
    print(f"\nDetailed results written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
