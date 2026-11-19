"""Per-document usage/cost rows and aggregate cost summary."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from src.cost.pricing import PRICING_AS_OF, PRICING_SOURCE, price_for
from src.pipeline import PipelineResult

COST_COLUMNS = [
    "document_id",
    "subset",
    "status",
    "source_kind",
    "pages",
    "provider",
    "model",
    "api_calls",
    "transient_retries",
    "repair_calls",
    "verify_calls",
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "cost_usd",
    "cost_usd_no_prompt_cache",
    "served_from_cache",
    "processing_s",
    "api_latency_s",
]


def cost_row(result: PipelineResult, subset: str) -> dict[str, Any]:
    pre = result.preprocessing or {}
    models = sorted({a.response.model for a in result.attempts})
    return {
        "document_id": result.document_id,
        "subset": subset,
        "status": result.status,
        "source_kind": pre.get("source_kind"),
        "pages": pre.get("page_count"),
        "provider": result.settings_snapshot.get("provider"),
        "model": ",".join(models) or result.settings_snapshot.get("model"),
        "api_calls": result.api_calls,
        "transient_retries": result.transient_retries,
        "repair_calls": sum(1 for a in result.attempts if a.kind.endswith("repair")),
        "verify_calls": sum(1 for a in result.attempts if a.kind.startswith("verify_")),
        "input_tokens": result.input_tokens,
        "cached_input_tokens": result.cached_input_tokens,
        "output_tokens": result.output_tokens,
        "cost_usd": round(result.cost_usd, 8) if result.cost_usd is not None else None,
        "cost_usd_no_prompt_cache": (
            round(result.cost_usd_no_prompt_cache, 8) if result.cost_usd_no_prompt_cache is not None else None
        ),
        "served_from_cache": all(a.response.from_cache for a in result.attempts) if result.attempts else False,
        # Wall-clock for this run (near zero on cache hits) vs. the provider latency recorded at call time.
        "processing_s": round(result.duration_s, 3),
        "api_latency_s": round(sum(a.response.latency_s for a in result.attempts), 3),
    }


def _doc_category(row: dict[str, Any]) -> str:
    """digital_invoice | scan_or_photo_invoice | non_invoice | failed: the cost-relevant document types."""
    status = row.get("status")
    if status == "failed":
        return "failed"
    if status == "not_invoice":
        return "non_invoice"
    return "digital_invoice" if row.get("source_kind") == "digital_pdf" else "scan_or_photo_invoice"


def summarize_costs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def agg(subset_rows: list[dict[str, Any]]) -> dict[str, Any]:
        priced = [r for r in subset_rows if r["cost_usd"] is not None]
        called = [r for r in subset_rows if r["api_calls"] > 0]
        return {
            "documents": len(subset_rows),
            "documents_with_api_calls": len(called),
            "total_cost_usd": round(sum(r["cost_usd"] for r in priced), 6),
            "avg_cost_per_document_usd": round(mean(r["cost_usd"] for r in priced), 6) if priced else None,
            "total_cost_usd_no_prompt_cache": round(sum(r.get("cost_usd_no_prompt_cache") or 0 for r in priced), 6),
            "avg_cost_per_document_usd_no_prompt_cache": (
                round(mean(r.get("cost_usd_no_prompt_cache") or 0 for r in priced), 6) if priced else None
            ),
            # uncached input tokens (billed at full rate); cached tokens are reported separately
            "avg_input_tokens": round(mean(r["input_tokens"] for r in called), 1) if called else None,
            "avg_cached_input_tokens": round(mean(r.get("cached_input_tokens", 0) for r in called), 1)
            if called
            else None,
            "avg_total_input_tokens": round(
                mean(r["input_tokens"] + r.get("cached_input_tokens", 0) for r in called), 1
            )
            if called
            else None,
            "avg_output_tokens": round(mean(r["output_tokens"] for r in called), 1) if called else None,
            "total_input_tokens": sum(r["input_tokens"] for r in subset_rows),
            "total_output_tokens": sum(r["output_tokens"] for r in subset_rows),
            "total_api_calls": sum(r["api_calls"] for r in subset_rows),
            "total_repair_calls": sum(r["repair_calls"] for r in subset_rows),
            "total_verify_calls": sum(r.get("verify_calls", 0) for r in subset_rows),
            "total_transient_retries": sum(r["transient_retries"] for r in subset_rows),
            "avg_api_latency_s": round(mean(r["api_latency_s"] for r in called), 2) if called else None,
        }

    subsets = sorted({r["subset"] for r in rows})
    kinds = sorted({_doc_category(r) for r in rows})
    models = sorted({m for r in rows for m in str(r["model"]).split(",") if m})
    return {
        "pricing_source": PRICING_SOURCE,
        "pricing_as_of": PRICING_AS_OF,
        "prices_usd_per_mtok": {
            m: {"input": p.input_per_mtok, "output": p.output_per_mtok}
            for m in models
            if (p := price_for(m)) is not None
        },
        "method": (
            "cost = uncached input_tokens x input price + cached input tokens x cached-input price + "
            "output_tokens x output price, using the token counts "
            "reported by the provider API on each call (thinking tokens are included in output_tokens). "
            "Repair calls and retries that returned a response are included. Cached re-runs reuse the "
            "original call's recorded usage, so the figures are what a fresh extraction costs."
        ),
        "all": agg(rows),
        **{f"subset:{s}": agg([r for r in rows if r["subset"] == s]) for s in subsets},
        **{f"type:{k}": agg([r for r in rows if _doc_category(r) == k]) for k in kinds},
    }


def write_cost_files(rows: list[dict[str, Any]], out_dir: Path, stem: str = "cost") -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{stem}_per_document.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize_costs(rows)
    (out_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
