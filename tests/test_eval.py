"""Evaluation harness: normalisation, line-item alignment, metrics."""

import copy

import pytest

from src.eval.compare import align_line_items, compare_document
from src.eval.normalize import (
    normalize_date,
    normalize_invoice_number,
    normalize_text,
    normalize_vendor_lenient,
    numbers_equal,
    to_number,
)

GT = {
    "is_invoice": True,
    "vendor": "Meridian Print & Pack LLP",
    "invoice_number": "MPP-2026-0207",
    "invoice_date": "2026-02-17",
    "currency": "INR",
    "line_items": [
        {"description": "Corrugated box 12x9x6 in (3-ply)", "quantity": 500, "amount": 9250.0},
        {"description": "Bubble wrap roll 1m x 100m", "quantity": 30, "amount": 35400.0},
        {"description": "Freight, Pune to Nashik", "quantity": 1, "amount": 6200.0},
    ],
    "tax_total": 1000.0,
    "total": 51850.0,
}
NON_INVOICE = {k: None for k in GT} | {"is_invoice": False}


# --- normalisation ------------------------------------------------------------------


@pytest.mark.parametrize("a,b", [("1,200.00", 1200), ("₹1,23,456.50", 123456.5), (1200, 1200.004)])
def test_number_parsing_equivalences(a, b):
    assert numbers_equal(a, b, 0.01)


def test_number_tolerance_and_nulls():
    assert not numbers_equal(100.0, 100.02, 0.01)
    assert numbers_equal(None, None, 0.01)
    assert not numbers_equal(None, 0, 0.01)  # null is not zero
    assert to_number(True) is None


def test_text_normalisation():
    assert normalize_text("  Meridian  Print & Pack, LLP ") == normalize_text("MERIDIAN PRINT AND PACK LLP")
    assert normalize_text("Acme") != normalize_text("Acme Corp")


def test_vendor_lenient_strips_legal_suffix_only():
    assert normalize_vendor_lenient("Acme Supplies Pvt. Ltd.") == normalize_vendor_lenient("Acme Supplies")
    assert normalize_vendor_lenient("Acme Supplies") != normalize_vendor_lenient("Apex Supplies")


def test_invoice_number_keeps_meaningful_characters():
    assert normalize_invoice_number(" kom/26-27/00418 ") == normalize_invoice_number("KOM/26-27/00418")
    assert normalize_invoice_number("KOM/26-27/418") != normalize_invoice_number("KOM/26-27/00418")
    assert normalize_invoice_number("KOM-26-27-00418") != normalize_invoice_number("KOM/26-27/00418")


def test_date_normalisation_only_for_unambiguous_formats():
    assert normalize_date("14 March 2026") == "2026-03-14"
    assert normalize_date("March 14, 2026") == "2026-03-14"
    assert normalize_date("04/09/2026") == "04/09/2026"  # ambiguous: left as-is, so it cannot match


# --- line items ------------------------------------------------------------------------


def test_line_items_reordered_still_aligned_but_order_reported():
    pred = list(reversed(GT["line_items"]))
    li = align_line_items(pred, GT["line_items"], 0.01)
    assert li.matched_items == 3 and li.order_preserved is False


def test_missing_row_does_not_cascade():
    pred = [GT["line_items"][0], GT["line_items"][2]]
    li = align_line_items(pred, GT["line_items"], 0.01)
    assert li.matched_items == 2
    assert li.attribute_correct("amount") == 2  # the third GT item counts as wrong, others stay right


def test_attribute_level_accuracy():
    pred = copy.deepcopy(GT["line_items"])
    pred[1]["quantity"] = 3  # misread
    pred[2]["description"] = "Freight Pune-Nashik"  # paraphrased
    li = align_line_items(pred, GT["line_items"], 0.01)
    assert li.attribute_correct("quantity") == 2
    assert li.attribute_correct("description") == 2
    assert li.attribute_correct("amount") == 3
    assert li.matched_items == 1


def test_extra_predicted_row_hurts_precision():
    pred = GT["line_items"] + [{"description": "Subtotal", "quantity": None, "amount": 50850.0}]
    li = align_line_items(pred, GT["line_items"], 0.01)
    assert li.matched_items == 3 and li.pred_count == 4


# --- document comparison -------------------------------------------------------------


def test_perfect_prediction():
    c = compare_document("d", copy.deepcopy(GT), GT)
    assert c.all_correct and all(f.exact for f in c.fields.values())


def test_normalised_vs_exact_distinguished():
    pred = copy.deepcopy(GT) | {"vendor": "MERIDIAN PRINT AND PACK LLP", "total": 51850}
    c = compare_document("d", pred, GT)
    assert c.fields["vendor"].correct and not c.fields["vendor"].exact
    assert c.fields["total"].correct and c.fields["total"].exact  # 51850 == 51850.0 as JSON numbers


def test_different_vendor_is_wrong_even_leniently():
    c = compare_document("d", copy.deepcopy(GT) | {"vendor": "Meridian Packaging LLP"}, GT)
    assert not c.fields["vendor"].correct and not c.fields["vendor"].lenient


def test_line_items_field_requires_all_items():
    pred = copy.deepcopy(GT)
    pred["line_items"] = pred["line_items"][:2]
    c = compare_document("d", pred, GT)
    assert not c.fields["line_items"].correct and c.wrong_fields == ["line_items"]


def test_non_invoice_correct_when_all_null():
    c = compare_document("d", copy.deepcopy(NON_INVOICE), NON_INVOICE)
    assert c.all_correct


def test_invoice_predicted_for_non_invoice_counts_errors():
    c = compare_document("d", copy.deepcopy(GT), NON_INVOICE)
    assert not c.fields["is_invoice"].correct and len(c.wrong_fields) == 8


def test_failed_pipeline_counts_every_field_wrong_even_for_non_invoice():
    c = compare_document("d", None, NON_INVOICE)
    assert c.correct_count == 0


def test_zero_tax_is_not_null_tax():
    c = compare_document("d", copy.deepcopy(GT) | {"tax_total": None}, GT | {"tax_total": 0.0})
    assert not c.fields["tax_total"].correct
