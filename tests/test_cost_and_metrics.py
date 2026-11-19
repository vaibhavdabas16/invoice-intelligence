import copy

import pytest

from src.cost.pricing import cost_usd, price_for
from src.cost.report import cost_row, summarize_costs
from src.eval.compare import compare_document
from src.eval.metrics import EvalRecord, field_metrics, review_flag_metrics, summary_metrics
from src.extraction.providers.base import TokenUsage
from src.pipeline import process_document
from tests.fakes import ScriptedProvider, llm_invoice, llm_non_invoice


def test_cost_formula_uses_published_rates():
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost_usd(usage, "gpt-6-luna") == pytest.approx(0.10 + 0.50)
    small = TokenUsage(input_tokens=2500, output_tokens=400, cache_read_input_tokens=500)
    assert cost_usd(small, "gpt-6-luna") == pytest.approx((2500 * 0.10 + 400 * 0.50 + 500 * 0.01) / 1e6)


def test_dated_snapshot_ids_resolve_but_unknown_models_do_not():
    assert price_for("gpt-6-luna-2026-06-01") == price_for("gpt-6-luna")
    assert price_for("gpt-6-lunatic") is None
    assert cost_usd(TokenUsage(1, 1), "mystery-model") is None  # unknown price is visible, not silently $0


def test_per_document_cost_includes_repairs(digital_pdf_bytes, settings):
    provider = ScriptedProvider([llm_invoice(total=1.0), llm_invoice()], usage=TokenUsage(2000, 300))
    r = process_document(digital_pdf_bytes, "inv.pdf", provider, settings)
    row = cost_row(r, "labelled")
    assert row["api_calls"] == 2 and row["repair_calls"] == 1
    assert row["input_tokens"] == 4000 and row["output_tokens"] == 600
    assert row["cost_usd"] == pytest.approx((4000 * 0.10 + 600 * 0.50) / 1e6)


def test_cost_summary_by_subset():
    rows = [
        {
            "subset": "labelled",
            "cost_usd": 0.001,
            "api_calls": 1,
            "input_tokens": 100,
            "output_tokens": 10,
            "repair_calls": 0,
            "transient_retries": 0,
            "api_latency_s": 1.0,
            "model": "gpt-6-luna",
        },
        {
            "subset": "unlabelled",
            "cost_usd": 0.003,
            "api_calls": 2,
            "input_tokens": 300,
            "output_tokens": 30,
            "repair_calls": 1,
            "transient_retries": 1,
            "api_latency_s": 3.0,
            "model": "gpt-6-luna",
        },
        {
            "subset": "unlabelled",
            "cost_usd": 0.0,
            "api_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "repair_calls": 0,
            "transient_retries": 0,
            "api_latency_s": 0.0,
            "model": "gpt-6-luna",
        },
    ]
    s = summarize_costs(rows)
    assert s["all"]["total_cost_usd"] == pytest.approx(0.004)
    assert s["subset:unlabelled"]["avg_cost_per_document_usd"] == pytest.approx(0.0015)
    assert s["all"]["avg_input_tokens"] == 200  # averaged over documents that made API calls
    assert s["prices_usd_per_mtok"]["gpt-6-luna"] == {"input": 0.10, "output": 0.50}


def _record(result, pred_override=None, gt=None):
    gt = gt or llm_to_gt(llm_invoice())
    pred = pred_override if pred_override is not None else (result.invoice.to_output_dict() if result.invoice else None)
    return EvalRecord(compare_document("d", pred, gt), result, gt)


def llm_to_gt(llm):
    return {
        "is_invoice": llm["is_invoice"],
        "vendor": llm["vendor"],
        "invoice_number": llm["invoice_number"],
        "invoice_date": llm["invoice_date"],
        "currency": llm["currency"],
        "line_items": [{k: i[k] for k in ("description", "quantity", "amount")} for i in llm["line_items"]],
        "tax_total": llm["tax_total"],
        "total": llm["total"],
    }


def test_metrics_and_review_flag_confusion(digital_pdf_bytes, settings):
    ok = process_document(digital_pdf_bytes, "a.pdf", ScriptedProvider([llm_invoice()]), settings)
    # wrong vendor, not flagged
    wrong = process_document(
        digital_pdf_bytes, "b.pdf", ScriptedProvider([llm_invoice(vendor="Acme Traders")]), settings
    )
    # flagged (ambiguity) and wrong
    flagged = process_document(
        digital_pdf_bytes,
        "c.pdf",
        ScriptedProvider([llm_invoice(invoice_number="INV-2O41", ambiguities=["unclear"])]),
        settings,
    )
    non_inv_gt = {k: None for k in llm_to_gt(llm_invoice())} | {"is_invoice": False}
    po = process_document(digital_pdf_bytes, "d.pdf", ScriptedProvider([llm_non_invoice()]), settings)
    records = [_record(ok), _record(wrong), _record(flagged), _record(po, gt=non_inv_gt)]

    s = summary_metrics(records)
    assert s["overall_field_accuracy"] == pytest.approx(30 / 32)
    assert s["exact_document_match_rate"] == 0.5
    assert s["documents_with_at_least_one_error"] == 2
    assert s["non_invoices_correctly_classified"] == 1

    vendor = next(f for f in field_metrics(records) if f["field"] == "vendor")
    assert vendor["correct"] == 3 and vendor["total"] == 4

    rf = review_flag_metrics(records)
    # "Acme Traders" is not printed in the synthetic PDF, so value grounding flags it: both errors caught.
    assert rf["flagged"] == 2 and rf["flagged_with_errors"] == 2
    assert rf["unflagged_with_errors"] == 0
    assert rf["precision"] == 1.0 and rf["recall"] == 1.0


def test_ground_truth_not_mutated_by_comparison():
    gt = llm_to_gt(llm_invoice())
    snapshot = copy.deepcopy(gt)
    compare_document("d", {**gt, "total": 1.0}, gt)
    assert gt == snapshot


def test_no_prompt_cache_upper_bound_bills_cached_tokens_at_full_rate():
    from src.cost.pricing import cost_usd_without_prompt_cache

    usage = TokenUsage(input_tokens=1000, output_tokens=100, cache_read_input_tokens=4000)
    assert cost_usd(usage, "gpt-6-luna") == pytest.approx((1000 * 0.10 + 4000 * 0.01 + 100 * 0.50) / 1e6)
    assert cost_usd_without_prompt_cache(usage, "gpt-6-luna") == pytest.approx((5000 * 0.10 + 100 * 0.50) / 1e6)


def test_cost_summary_by_document_type():
    def row(kind, status, cost):
        return {
            "subset": "x",
            "status": status,
            "source_kind": kind,
            "cost_usd": cost,
            "api_calls": 1,
            "input_tokens": 10,
            "output_tokens": 1,
            "repair_calls": 0,
            "transient_retries": 0,
            "api_latency_s": 1.0,
            "model": "gpt-6-luna",
        }

    s = summarize_costs(
        [
            row("digital_pdf", "extracted", 0.0004),
            row("image", "extracted", 0.001),
            row("scanned_pdf", "extracted", 0.0007),
            row("digital_pdf", "not_invoice", 0.0003),
        ]
    )
    assert s["type:digital_invoice"]["avg_cost_per_document_usd"] == pytest.approx(0.0004)
    assert s["type:scan_or_photo_invoice"]["avg_cost_per_document_usd"] == pytest.approx(0.00085)
    assert s["type:non_invoice"]["documents"] == 1
