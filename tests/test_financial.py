from src.schema import LLMExtraction
from src.validation.financial import FinancialStatus, check_financials
from src.validation.review import decide_review
from tests.fakes import llm_invoice, llm_non_invoice


def _ext(**kw) -> LLMExtraction:
    return LLMExtraction.model_validate(llm_invoice(**kw))


def test_reconciling_invoice_passes():
    check = check_financials(_ext())
    assert check.status is FinancialStatus.OK and check.passed
    assert check.line_items_sum == 3200.0 and check.computed_total == 3776.0


def test_mismatch_detected_and_values_untouched():
    ext = _ext(total=3800.0)
    check = check_financials(ext)
    assert check.status is FinancialStatus.MISMATCH
    assert check.difference == 24.0
    assert ext.total == 3800.0  # never silently altered


def test_rounding_tolerance_boundary():
    assert check_financials(_ext(total=3776.02), tolerance=0.02).status is FinancialStatus.OK
    assert check_financials(_ext(total=3776.03), tolerance=0.02).status is FinancialStatus.MISMATCH
    assert check_financials(_ext(total=3776.5), tolerance=1.0).status is FinancialStatus.OK


def test_discount_row_inside_table_reconciles():
    items = [
        {"description": "Design", "quantity": 1, "unit_price": 3200.0, "amount": 3200.0},
        {"description": "Discount 10%", "quantity": None, "unit_price": None, "amount": -320.0},
    ]
    check = check_financials(_ext(line_items=items, printed_subtotal=2880.0, tax_lines=[], tax_total=0.0, total=2880.0))
    assert check.status is FinancialStatus.OK


def test_charge_outside_table_is_flagged_unresolved_not_folded_in():
    ext = _ext(adjustments_outside_items=[{"label": "Shipping", "amount": 100.0}], total=3876.0)
    check = check_financials(ext)
    assert check.status is FinancialStatus.UNRESOLVED_ADJUSTMENTS
    assert not check.passed
    assert check.adjustments_sum == 100.0


def test_round_off_outside_table_is_unresolved():
    ext = _ext(adjustments_outside_items=[{"label": "Round off", "amount": 0.4}], total=3776.4)
    assert check_financials(ext).status is FinancialStatus.UNRESOLVED_ADJUSTMENTS


def test_missing_total_is_insufficient_data():
    check = check_financials(_ext(total=None))
    assert check.status is FinancialStatus.INSUFFICIENT_DATA


def test_null_tax_treated_as_zero_with_message():
    check = check_financials(_ext(tax_total=None, tax_lines=None, total=3200.0))
    assert check.status is FinancialStatus.OK
    assert any("tax_total is null" in m for m in check.messages)


def test_row_level_qty_times_price_mismatch_localised():
    items = [{"description": "Paper", "quantity": 10, "unit_price": 320.0, "amount": 3020.0}]
    check = check_financials(_ext(line_items=items, printed_subtotal=3020.0, total=3596.0))
    assert check.status is FinancialStatus.OK  # totals reconcile with the (misread) amount ...
    assert check.row_mismatches == [0]  # ... but the row check catches the misread digit


def test_rounded_unit_price_not_flagged():
    items = [{"description": "Licence", "quantity": 3, "unit_price": 333.33, "amount": 1000.0}]
    check = check_financials(_ext(line_items=items, printed_subtotal=1000.0, tax_lines=[], tax_total=0.0, total=1000.0))
    assert check.row_mismatches == []


def test_tax_lines_must_sum_to_tax_total():
    ext = _ext(tax_lines=[{"label": "CGST", "amount": 288.0}, {"label": "SGST", "amount": 288.0}])
    assert check_financials(ext).tax_lines_match is True
    ext = _ext(tax_lines=[{"label": "CGST", "amount": 288.0}])
    assert check_financials(ext).tax_lines_match is False


def test_subtotal_mismatch_detected():
    assert check_financials(_ext(printed_subtotal=3300.0)).subtotal_matches is False


def test_non_invoice_not_applicable():
    ext = LLMExtraction.model_validate(llm_non_invoice())
    assert check_financials(ext).status is FinancialStatus.NOT_APPLICABLE


# --- review policy ------------------------------------------------------------------


def test_clean_invoice_not_flagged():
    ext = _ext()
    d = decide_review(ext, check_financials(ext), preprocessing_warnings=[], repair_attempted=False)
    assert d.needs_review is False and d.reasons == []


def test_review_reasons_are_specific():
    ext = _ext(invoice_number=None, total=4000.0, ambiguities=["date could be 03/04"], illegible_fields=["vendor"])
    d = decide_review(
        ext, check_financials(ext), preprocessing_warnings=["page 1: low_contrast"], repair_attempted=True
    )
    text = " | ".join(d.reasons)
    assert d.needs_review
    for fragment in ["invoice_number", "do not reconcile", "illegible", "ambiguity", "repair"]:
        assert fragment in text
    # image-quality heuristics are informational only (0 true positives when measured)
    assert "image quality" not in text and "low_contrast" not in text


def test_classification_conflict_flagged():
    ext = _ext(document_type="purchase_order")
    d = decide_review(ext, check_financials(ext), preprocessing_warnings=[], repair_attempted=False)
    assert any("classification conflict" in r for r in d.reasons)


# --- tax-rate consistency ------------------------------------------------------------------------


def test_tax_rate_consistent_single_rate():
    ext = _ext(tax_lines=[{"label": "GST @ 18%", "amount": 576.0}])
    assert check_financials(ext).tax_rate_mismatches == []


def test_tax_rate_cgst_sgst_same_rate():
    ext = _ext(tax_lines=[{"label": "CGST @ 9%", "amount": 288.0}, {"label": "SGST @ 9%", "amount": 288.0}])
    assert check_financials(ext).tax_rate_mismatches == []


def test_self_consistent_misread_caught_by_tax_rate():
    # held-out L07 pattern: row, subtotal and total all "add up" to a misread digit, but the printed
    # tax (8.25% of the true 4050 subtotal) no longer matches 8.25% of the misread 3850
    items = [
        {"description": "Brand identity package", "quantity": 1, "unit_price": 3200.0, "amount": 3200.0},
        {"description": "Templates", "quantity": 1, "unit_price": 650.0, "amount": 650.0},
    ]
    ext = _ext(
        line_items=items,
        printed_subtotal=3850.0,
        tax_lines=[{"label": "Sales tax @ 8.25%", "amount": 334.13}],
        tax_total=334.13,
        total=4184.13,
    )
    check = check_financials(ext)
    assert check.status is FinancialStatus.OK and check.subtotal_matches  # every other check passes
    assert check.tax_rate_mismatches and "8.25% of 3850.00 is 317.6" in check.tax_rate_mismatches[0]
    d = decide_review(ext, check, preprocessing_warnings=[], repair_attempted=False)
    assert d.needs_review and any("8.25%" in r for r in d.reasons)


def test_mixed_rates_not_checked():
    lines = [{"label": "GST @ 12% (room)", "amount": 100.0}, {"label": "GST @ 5% (food)", "amount": 476.0}]
    assert check_financials(_ext(tax_lines=lines)).tax_rate_mismatches == []


def test_no_printed_rate_not_checked():
    ext = _ext(tax_lines=[{"label": "Tax", "amount": 576.0}])
    assert check_financials(ext).tax_rate_mismatches == []


def test_tax_rate_rounding_tolerance():
    # 8.25% of 4050 = 334.125, printed 334.13
    items = [{"description": "x", "quantity": 1, "unit_price": 4050.0, "amount": 4050.0}]
    ext = _ext(
        line_items=items,
        printed_subtotal=4050.0,
        tax_lines=[{"label": "Sales tax @ 8.25%", "amount": 334.13}],
        tax_total=334.13,
        total=4384.13,
    )
    assert check_financials(ext).tax_rate_mismatches == []
