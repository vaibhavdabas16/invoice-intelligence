import pytest
from pydantic import ValidationError

from src.extraction.parsing import OutputParseError, OutputSchemaError, parse_json_object, validate_extraction
from src.schema import InvoiceExtraction
from tests.fakes import llm_invoice, llm_non_invoice


def _valid() -> dict:
    return {
        "is_invoice": True,
        "vendor": "Acme",
        "invoice_number": "INV-1",
        "invoice_date": "2026-03-14",
        "currency": "INR",
        "line_items": [{"description": "Paper", "quantity": 2.5, "amount": 10}],
        "tax_total": 0,
        "total": 10,
    }


def test_valid_invoice_accepts_int_and_decimal_numbers():
    inv = InvoiceExtraction.model_validate(_valid())
    assert inv.line_items[0].quantity == 2.5
    assert inv.total == 10


def test_output_has_exactly_the_required_fields_in_order():
    out = InvoiceExtraction.model_validate(_valid()).to_output_dict()
    assert list(out) == [
        "is_invoice",
        "vendor",
        "invoice_number",
        "invoice_date",
        "currency",
        "line_items",
        "tax_total",
        "total",
    ]


@pytest.mark.parametrize("bad_date", ["14/03/2026", "2026-02-30", "2026-3-14", "March 14, 2026"])
def test_invalid_dates_rejected(bad_date):
    with pytest.raises(ValidationError):
        InvoiceExtraction.model_validate({**_valid(), "invoice_date": bad_date})


@pytest.mark.parametrize("bad_currency", ["inr", "Rs", "₹", "XYZ", "RUPEES"])
def test_invalid_currency_rejected(bad_currency):
    with pytest.raises(ValidationError):
        InvoiceExtraction.model_validate({**_valid(), "currency": bad_currency})


def test_string_amount_rejected_not_coerced():
    with pytest.raises(ValidationError):
        InvoiceExtraction.model_validate({**_valid(), "total": "1,200.00"})


def test_non_finite_amount_rejected():
    with pytest.raises(ValidationError):
        InvoiceExtraction.model_validate({**_valid(), "total": float("nan")})


def test_missing_field_rejected():
    data = _valid()
    del data["vendor"]
    with pytest.raises(ValidationError):
        InvoiceExtraction.model_validate(data)


def test_extra_field_rejected():
    with pytest.raises(ValidationError):
        InvoiceExtraction.model_validate({**_valid(), "needs_review": True})


def test_non_invoice_must_have_null_fields():
    with pytest.raises(ValidationError):
        InvoiceExtraction.model_validate({**_valid(), "is_invoice": False})
    ok = InvoiceExtraction.non_invoice().to_output_dict()
    assert ok["is_invoice"] is False and all(v is None for k, v in ok.items() if k != "is_invoice")


def test_empty_vendor_string_rejected():
    with pytest.raises(ValidationError):
        InvoiceExtraction.model_validate({**_valid(), "vendor": "  "})


# --- parsing of raw model text --------------------------------------------------


def test_parse_handles_code_fences_and_preamble():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Here you go: {"a": 1} thanks') == {"a": 1}


@pytest.mark.parametrize("text", ["", "not json", '{"a": 1', "[1, 2]"])
def test_parse_rejects_malformed(text):
    with pytest.raises(OutputParseError):
        parse_json_object(text)


def test_validate_extraction_projects_to_required_schema():
    llm, inv = validate_extraction(llm_invoice())
    assert inv.vendor == "Acme Supplies Pvt Ltd"
    assert "unit_price" not in inv.to_output_dict()["line_items"][0]


def test_non_invoice_projection_forces_nulls_even_if_model_filled_fields():
    data = {**llm_non_invoice(), "vendor": "Some Bank", "total": 100.0}
    _, inv = validate_extraction(data)
    assert inv.vendor is None and inv.total is None and inv.is_invoice is False


def test_schema_error_lists_all_problems():
    with pytest.raises(OutputSchemaError) as exc:
        validate_extraction({**llm_invoice(), "invoice_date": "14/03/2026", "currency": "rupees"})
    assert len(exc.value.problems) == 2


def test_unknown_document_type_rejected():
    with pytest.raises(OutputSchemaError):
        validate_extraction({**llm_invoice(), "document_type": "tax_form"})
