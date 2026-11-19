"""Field-level evidence (value grounding in the PDF text layer)."""

import pytest

from src.pipeline import process_document
from src.schema import InvoiceExtraction, LLMExtraction
from src.validation.evidence import (
    amount_renderings,
    collect_evidence,
    date_renderings,
    unsupported_reasons,
)
from tests.fakes import ScriptedProvider, llm_invoice

PAGE = """ACME SUPPLIES PVT LTD
TAX INVOICE
Invoice No: KOM/26-27/00418        Date: 14/03/2026
1  A4 paper, 500 sheets     10   320.00   3,200.00
CGST @ 9%                                  288.00
SGST @ 9%                                  288.00
Total                                  ₹3,776.00"""


def _inv(**kw) -> InvoiceExtraction:
    base = {
        "vendor": "Acme Supplies Pvt Ltd",
        "invoice_number": "KOM/26-27/00418",
        "tax_lines": [{"label": "CGST", "amount": 288.0}, {"label": "SGST", "amount": 288.0}],
    }
    return LLMExtraction.model_validate(llm_invoice(**{**base, **kw})).to_invoice()


def _by_field(evidence):
    return {e.field: e for e in evidence}


# --- renderings ------------------------------------------------------------------------------


def test_amount_renderings_cover_common_conventions():
    r = amount_renderings(300484.8)
    for text in ("300484.80", "300,484.80", "3,00,484.80", "300.484,80", "300484.8"):
        assert text in r
    assert "450.00" in amount_renderings(-450.0)  # discounts: sign shown in many ways, match the magnitude


def test_date_renderings():
    r = date_renderings("2026-04-08")
    for text in ("08 April 2026", "8 April 2026", "April 08, 2026", "08/04/2026", "2026-04-08", "08.04.2026"):
        assert text in r


# --- grounding -----------------------------------------------------------------------------


def test_all_values_grounded_with_page_and_source_line():
    ev = _by_field(collect_evidence(_inv(), [PAGE], [288.0, 288.0]))
    assert ev["invoice_number"].status == "grounded" and ev["invoice_number"].page == 1
    assert "KOM/26-27/00418" in ev["invoice_number"].source_text
    assert ev["total"].status == "grounded" and "3,776.00" in ev["total"].source_text
    assert ev["invoice_date"].status == "grounded"
    assert ev["line 1 description"].status == "grounded"
    assert unsupported_reasons(list(ev.values())) == []


def test_tax_total_grounded_through_its_printed_components():
    # 576.00 is never printed: it is CGST + SGST
    ev = _by_field(collect_evidence(_inv(), [PAGE], [288.0, 288.0]))
    assert ev["tax_total"].status == "derived" and ev["tax_total"].source_text.count("288.00") == 2


def test_tax_total_unsupported_when_components_missing():
    ev = _by_field(collect_evidence(_inv(), [PAGE], None))
    assert ev["tax_total"].status == "unsupported"


@pytest.mark.parametrize(
    "field, wrong",
    [
        ("invoice_number", "KOM/26-27/0418"),  # dropped zero
        ("invoice_number", "OM/26-27/00418"),  # dropped leading char: must not ground as a substring
        ("invoice_number", "KOM-26-27-00418"),  # separators matter
        ("vendor", "Acme Suppliers Pvt Ltd"),
        ("invoice_date", "2026-03-15"),
        ("total", 3767.0),
    ],
)
def test_wrong_values_are_unsupported(field, wrong):
    ev = _by_field(collect_evidence(_inv(**{field: wrong}), [PAGE], [288.0, 288.0]))
    assert ev[field].status == "unsupported"
    assert field in unsupported_reasons(list(ev.values()))[0]


def test_amount_digit_boundaries():
    # 15.00 must not ground inside 115.00
    items = [{"description": "A4 paper, 500 sheets", "quantity": 1, "unit_price": 15.0, "amount": 15.0}]
    page = PAGE.replace("3,200.00", "115.00")
    ev = _by_field(collect_evidence(_inv(line_items=items), [page], [288.0, 288.0]))
    assert ev["line 1 amount"].status == "unsupported"


def test_whitespace_and_case_insensitive_text():
    page = PAGE.replace("KOM/26-27/00418", "kom / 26-27 / 00418")
    ev = _by_field(collect_evidence(_inv(), [page], [288.0, 288.0]))
    assert ev["invoice_number"].status == "grounded"


def test_multi_page_reports_correct_page():
    ev = _by_field(collect_evidence(_inv(), ["page one header only", PAGE], [288.0, 288.0]))
    assert ev["total"].page == 2


def test_scan_without_text_layer_is_not_checkable():
    ev = collect_evidence(_inv(), [None], [288.0, 288.0])
    assert ev and all(e.status == "not_checkable" for e in ev)
    assert unsupported_reasons(ev) == []


def test_non_invoice_has_no_evidence():
    assert collect_evidence(InvoiceExtraction.non_invoice(), [PAGE]) == []


def test_zero_tax_not_checked():
    ev = _by_field(
        collect_evidence(_inv(tax_total=0.0, tax_lines=[], total=3200.0), [PAGE.replace("3,776.00", "3,200.00")])
    )
    assert "tax_total" not in ev


# --- pipeline integration ------------------------------------------------------------------


def test_pipeline_stores_evidence_and_flags_unsupported(digital_pdf_bytes, settings):
    ok = process_document(digital_pdf_bytes, "d.pdf", ScriptedProvider([llm_invoice()]), settings)
    assert not ok.needs_review
    statuses = {e["field"]: e["status"] for e in ok.metadata_json()["evidence"]}
    assert statuses["invoice_number"] == "grounded" and statuses["total"] == "grounded"

    bad = process_document(
        digital_pdf_bytes, "d.pdf", ScriptedProvider([llm_invoice(invoice_number="INV-241")]), settings
    )
    assert bad.needs_review and any("not found in the PDF text layer" in r for r in bad.review.reasons)
    assert bad.output_json()["invoice_number"] == "INV-241"  # flagged, never "corrected"


def test_required_output_schema_unchanged_by_evidence(digital_pdf_bytes, settings):
    r = process_document(digital_pdf_bytes, "d.pdf", ScriptedProvider([llm_invoice()]), settings)
    assert "evidence" not in r.output_json() and len(r.output_json()) == 8
