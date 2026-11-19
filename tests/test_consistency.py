"""Self-consistency re-read for scans, arithmetic arbitration, and invoice-number cross-checks."""

from src.pipeline import process_document
from src.schema import InvoiceExtraction, LLMExtraction
from src.validation.confidence import FieldConfidence
from src.validation.consistency import arbitrate, compare_reads, fully_consistent
from src.validation.financial import check_financials
from tests.fakes import ScriptedProvider, llm_invoice


def _inv(**kw) -> InvoiceExtraction:
    return LLMExtraction.model_validate(llm_invoice(**kw)).to_invoice()


def _fin(**kw):
    return check_financials(LLMExtraction.model_validate(llm_invoice(**kw)))


TWO_ROWS = [
    {"description": "Gloves", "quantity": 6, "unit_price": 64.0, "amount": 384.0},
    {"description": "Mirrors", "quantity": 4, "unit_price": 38.5, "amount": 154.0},
]
# U02-shaped misread of row 1 in the re-read: qty 4 / amount 256 -> breaks subtotal and total.
TWO_ROWS_MISREAD = [
    {"description": "Gloves", "quantity": 4, "unit_price": 64.0, "amount": 256.0},
    {"description": "Mirrors", "quantity": 4, "unit_price": 38.5, "amount": 154.0},
]
GOOD = {"line_items": TWO_ROWS, "printed_subtotal": 538.0, "tax_lines": [], "tax_total": 0.0, "total": 538.0}


# --- compare_reads -------------------------------------------------------------------------


def test_identical_reads_agree_modulo_presentation():
    assert compare_reads(_inv(), _inv(vendor="ACME SUPPLIES PVT. LTD.")) == []


def test_diffs_are_typed():
    items = [{"description": "Parrot, whole, kg", "quantity": 10, "unit_price": 320.0, "amount": 3200.0}]
    diffs = compare_reads(_inv(), _inv(invoice_number="INV-241", line_items=items, total=3777.0))
    kinds = {d.message.split(" (")[0]: d.kind for d in diffs}
    assert kinds == {"invoice_number": "text", "line 1 description": "text", "total": "numeric"}


def test_line_count_disagreement_is_structural():
    extra = llm_invoice()["line_items"] * 2
    diffs = compare_reads(_inv(), _inv(line_items=extra))
    assert [(d.kind, d.message) for d in diffs] == [("structure", "line item count (1 vs 2)")]


# --- arbitration ---------------------------------------------------------------------------


def _conf(field: str, p: float) -> FieldConfidence:
    return FieldConfidence(field, "x", p, "x")


def test_u02_case_arithmetic_alone_is_not_accepted():
    first, second = _inv(**GOOD), _inv(**{**GOOD, "line_items": TWO_ROWS_MISREAD})
    f1 = _fin(**GOOD)
    f2 = _fin(**{**GOOD, "line_items": TWO_ROWS_MISREAD})
    assert fully_consistent(f1) and not fully_consistent(f2)
    decision = arbitrate(compare_reads(first, second), f1, f2)
    assert not decision.resolved and "first (kept) read reconciles" in decision.note


def test_arithmetic_plus_high_confidence_is_still_not_accepted():
    # Measured: a self-consistent misread with token p = 0.975 was auto-accepted and wrong (held-out L07).
    first, second = _inv(**GOOD), _inv(**{**GOOD, "line_items": TWO_ROWS_MISREAD})
    diffs = compare_reads(first, second)
    assert {d.field for d in diffs} == {"line 1 quantity", "line 1 amount"}
    conf = [_conf("line 1 quantity", 0.99), _conf("line 1 amount", 0.97)]
    decision = arbitrate(diffs, _fin(**GOOD), _fin(**{**GOOD, "line_items": TWO_ROWS_MISREAD}), conf, 0.5)
    assert not decision.resolved and "needs review" in decision.note


def test_low_confidence_disputed_value_is_named_in_the_note():
    diffs = compare_reads(_inv(**GOOD), _inv(**{**GOOD, "line_items": TWO_ROWS_MISREAD}))
    conf = [_conf("line 1 quantity", 0.99), _conf("line 1 amount", 0.30)]
    decision = arbitrate(diffs, _fin(**GOOD), _fin(**{**GOOD, "line_items": TWO_ROWS_MISREAD}), conf, 0.5)
    assert not decision.resolved and "low token confidence in the first read on line 1 amount" in decision.note


def test_first_read_inconsistent_is_never_swapped_or_cleared():
    f_bad = _fin(**{**GOOD, "line_items": TWO_ROWS_MISREAD})
    f_good = _fin(**GOOD)
    diffs = compare_reads(_inv(**{**GOOD, "line_items": TWO_ROWS_MISREAD}), _inv(**GOOD))
    decision = arbitrate(diffs, f_bad, f_good)
    assert not decision.resolved and "first read kept, not swapped" in decision.note


def test_both_consistent_is_insufficient_evidence():
    # Both reads reconcile internally but disagree (subtotal and total misread together).
    alt = {**GOOD, "line_items": TWO_ROWS_MISREAD, "printed_subtotal": 410.0, "total": 410.0}
    decision = arbitrate(compare_reads(_inv(**GOOD), _inv(**alt)), _fin(**GOOD), _fin(**alt))
    assert not decision.resolved and "both reads reconcile" in decision.note


def test_text_disagreement_never_resolved_by_arithmetic():
    alt = {**GOOD, "line_items": [{**TWO_ROWS[0], "description": "Gauze"}, TWO_ROWS[1]], "total": 999.0}
    decision = arbitrate(compare_reads(_inv(**GOOD), _inv(**alt)), _fin(**GOOD), _fin(**alt))
    assert not decision.resolved and "text" in decision.note


def test_row_mismatch_counts_as_inconsistent():
    rows = [{**TWO_ROWS[0], "unit_price": 60.0}, TWO_ROWS[1]]  # 6 x 60 != 384
    assert not fully_consistent(_fin(**{**GOOD, "line_items": rows}))


# --- pipeline integration ------------------------------------------------------------------


def test_verify_runs_only_for_scans(scanned_pdf_bytes, digital_pdf_bytes, settings):
    s = settings.with_overrides(verify_scans=True)
    p = ScriptedProvider([llm_invoice()])
    r = process_document(digital_pdf_bytes, "d.pdf", p, s)
    assert r.api_calls == 1 and r.consistency is None

    p = ScriptedProvider([llm_invoice(), llm_invoice()])
    r = process_document(scanned_pdf_bytes, "s.pdf", p, s)
    assert r.api_calls == 2 and r.consistency["checked"] and r.consistency["disagreements"] == []
    assert [a.kind for a in r.attempts] == ["initial", "verify_initial"]
    assert p.requests[1].sample_id == 1
    assert not r.needs_review


def test_text_disagreement_flags_and_keeps_first_read(scanned_pdf_bytes, settings):
    s = settings.with_overrides(verify_scans=True)
    p = ScriptedProvider([llm_invoice(), llm_invoice(invoice_number="INV-2O41")])
    r = process_document(scanned_pdf_bytes, "s.pdf", p, s)
    assert r.needs_review and r.output_json()["invoice_number"] == "INV-2041"
    assert any("re-read" in reason for reason in r.review.reasons)
    assert r.consistency["resolved"] is False


def test_numeric_disagreement_flagged_end_to_end_without_confidence(scanned_pdf_bytes, settings):
    s = settings.with_overrides(verify_scans=True)
    p = ScriptedProvider([llm_invoice(**GOOD), llm_invoice(**{**GOOD, "line_items": TWO_ROWS_MISREAD})])
    r = process_document(scanned_pdf_bytes, "s.pdf", p, s)
    assert r.consistency["resolved"] is False and r.consistency["reread_financial_status"] == "mismatch"
    assert any("re-read" in reason for reason in r.review.reasons)
    assert r.output_json()["line_items"][0]["amount"] == 384.0  # first read kept, never swapped
    assert r.api_calls == 2  # the inconsistent re-read is not repaired: no extra call


def test_reread_failure_flags_instead_of_crashing(scanned_pdf_bytes, settings):
    s = settings.with_overrides(verify_scans=True)
    r = process_document(scanned_pdf_bytes, "s.pdf", ScriptedProvider([llm_invoice(), "not json"]), s)
    assert r.status == "extracted" and r.needs_review
    assert any("re-read failed" in reason for reason in r.review.reasons)


def test_verify_off_by_flag(scanned_pdf_bytes, settings):
    r = process_document(scanned_pdf_bytes, "s.pdf", ScriptedProvider([llm_invoice()]), settings)
    assert r.status == "extracted" and r.api_calls == 1 and r.consistency is None


def test_defaults():
    from src.config import Settings

    assert Settings().verify_scans is True
