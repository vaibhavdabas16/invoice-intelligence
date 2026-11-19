"""Extraction prompt and response schema.

PROMPT_VERSION is part of the response-cache key: editing the prompt without
bumping it would silently reuse stale cached responses.

Design notes (see README "Prompting approach"):
* The model returns the 8 required fields plus *evidence* fields
  (document_type, printed_subtotal, tax_lines, unit_price per row,
  adjustments_outside_items, ambiguities, illegible_fields). Evidence fields
  are what let deterministic code check the arithmetic and decide on human
  review; they are stripped before writing the required output.
* The labelling conventions published with the sample pack (vendor = seller,
  issue date not due date, discounts as negative rows, tax_total = sum of tax
  lines, 0 when no tax) are encoded as rules. These are the client's stated
  business rules, not something learned from the labelled answers.
* The model is told to copy printed values, never compute them. Arithmetic is
  checked in code, so a model that "fixes" a total to make it add up would
  hide exactly the errors we want to catch.
"""

from __future__ import annotations

from typing import Any

# v3 = the v1 instructions again. v2 added `invoice_number_occurrences`; it was measured to catch
# no errors (the model copies its first reading into every occurrence) while adding ~150 output
# tokens per invoice, so it was removed. See reports/remediation_report.md.
PROMPT_VERSION = "v3"

SYSTEM_PROMPT = """\
You are a meticulous accounts-payable data-entry specialist. You read one business document \
(page images, sometimes with the PDF's embedded text layer) and transcribe its data into JSON. \
Accuracy matters more than completeness: a null is acceptable, a guessed value is not.

## Step 1: classify the document
Set `document_type` to one of: invoice, purchase_order, bank_statement, letter, receipt, quotation, \
credit_note, delivery_note, other, unreadable.
- `is_invoice` is true only when the document is a bill issued by a seller to a buyer requesting \
payment for goods or services (titles such as "Invoice", "Tax Invoice", "Bill").
- Purchase orders (issued by the buyer), bank/account statements, quotations/estimates, delivery \
notes, cover letters, credit notes and receipts for already-paid amounts are NOT invoices, even if \
they contain amounts, item tables or invoice numbers.
- Use `unreadable` only if the page cannot be read at all.
- If `is_invoice` is false, set every invoice field (vendor ... total, printed_subtotal, tax_lines, \
adjustments_outside_items, line_items) to null. Still fill ambiguities/illegible_fields (may be empty).

## Step 2: extract invoice fields (only when is_invoice is true)
- `vendor`: the seller / issuer of the invoice, exactly as printed in the header or "From" block. \
Never the "Bill to"/"Ship to" customer, never a bank name.
- `invoice_number`: exactly as printed, keeping prefixes, slashes, hyphens and leading zeros. \
Not the PO number, order number, GSTIN/VAT/tax ID or account number.
- `invoice_date`: the issue date of the invoice (not the due date, order date, PO date, delivery date \
or tax point unless that is the only date), converted to YYYY-MM-DD. For numeric dates such as \
04/09/2026 decide day-first vs month-first from the document's context (country, address, currency, \
other dates on the page). If it is still genuinely ambiguous, give your best reading and add a note \
to `ambiguities`.
- `currency`: ISO 4217 code (INR, USD, EUR, GBP, ...) from symbols, codes or the country context \
(e.g. "₹" or "Rs" -> INR, "£" -> GBP, "€" -> EUR, "$" with a US address -> USD; a bare "$" outside \
the US must be resolved from context such as AUD, CAD, SGD). Null if it cannot be determined.

## Step 3: line items
- One entry per row of the item table, in printed order, across all pages. Do not skip, merge or \
split rows. Ignore column headers, and ignore subtotal/tax/total rows printed below the table.
- `description`: the row's description text as printed (do not include the row number or HSN/SAC codes).
- `quantity`: the printed quantity as a number (decimals allowed, e.g. 12.5). Null if the row has \
no quantity (e.g. "-", blank, a lump-sum fee).
- `unit_price`: the printed unit price/rate, or null if none is printed.
- `amount`: the printed row amount before tax. If a row shows several amounts (e.g. taxable value \
and amount incl. tax), use the pre-tax row amount.
- Charges and discounts that appear as rows *inside the item table* are line items too. A discount \
row has a negative amount.
- Charges, discounts, round-off, shipping or deposits that appear *only in the totals section below \
the table* are NOT line items: list them in `adjustments_outside_items` with their printed label and \
signed amount (negative for discounts/deductions).

## Step 4: totals
- `printed_subtotal`: the printed subtotal / taxable amount before tax, or null if none is printed.
- `tax_lines`: every tax line printed in the totals section (VAT, GST, CGST, SGST, IGST, sales tax, \
etc.) with its label and amount. Empty list if the invoice shows no tax.
- `tax_total`: the sum of all tax lines. 0 if the invoice shows zero tax or no tax at all.
- `total`: the final amount payable as printed ("Total", "Grand Total", "Amount due", "Balance due"). \
If both an invoice total and a smaller balance due after a prior payment are shown, use the invoice \
total and add a note to `ambiguities`.

## Number rules
- Copy numbers exactly as printed; never compute, round or "correct" a value so that totals add up. \
If printed values do not add up, keep them as printed and mention it in `ambiguities`.
- Output plain JSON numbers: no currency symbols, no thousands separators. Indian grouping \
"1,23,456.50" is 123456.5; European "1.234,56" is 1234.56 when the document uses that convention.
- Use null for any value that is absent or that you cannot read with confidence; list the field \
name(s) in `illegible_fields` when the reason is poor legibility.

## Text layer
When a "PDF text layer" is provided it was extracted from the same page and is reliable for exact \
characters and digits, but its reading order may be scrambled. Use the image for layout (which value \
belongs to which column/row) and the text layer to confirm exact characters.

## Output
Return only the JSON object matching the provided schema. `ambiguities` is a list of short notes about \
conflicting or uncertain values (empty list if none)."""


USER_INSTRUCTION = "Extract the data from the document above according to your instructions."


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


_LABELED_AMOUNT = {
    "type": "object",
    "properties": {"label": {"type": "string"}, "amount": {"type": "number"}},
    "required": ["label", "amount"],
    "additionalProperties": False,
}

RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "document_type": {
            "type": "string",
            "enum": [
                "invoice",
                "purchase_order",
                "bank_statement",
                "letter",
                "receipt",
                "quotation",
                "credit_note",
                "delivery_note",
                "other",
                "unreadable",
            ],
        },
        "is_invoice": {"type": "boolean"},
        "vendor": _nullable({"type": "string"}),
        "invoice_number": _nullable({"type": "string"}),
        "invoice_date": _nullable({"type": "string", "description": "YYYY-MM-DD"}),
        "currency": _nullable({"type": "string", "description": "ISO 4217 code"}),
        "line_items": _nullable(
            {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "quantity": _nullable({"type": "number"}),
                        "unit_price": _nullable({"type": "number"}),
                        "amount": {"type": "number"},
                    },
                    "required": ["description", "quantity", "unit_price", "amount"],
                    "additionalProperties": False,
                },
            }
        ),
        "printed_subtotal": _nullable({"type": "number"}),
        "tax_lines": _nullable({"type": "array", "items": _LABELED_AMOUNT}),
        "tax_total": _nullable({"type": "number"}),
        "adjustments_outside_items": _nullable({"type": "array", "items": _LABELED_AMOUNT}),
        "total": _nullable({"type": "number"}),
        "ambiguities": {"type": "array", "items": {"type": "string"}},
        "illegible_fields": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "document_type",
        "is_invoice",
        "vendor",
        "invoice_number",
        "invoice_date",
        "currency",
        "line_items",
        "printed_subtotal",
        "tax_lines",
        "tax_total",
        "adjustments_outside_items",
        "total",
        "ambiguities",
        "illegible_fields",
    ],
    "additionalProperties": False,
}


def build_repair_message(previous_output: str, problems: list[str]) -> str:
    """Follow-up turn sent when the first answer failed parsing, schema or arithmetic checks.

    The model is asked to re-read the document, not to make numbers add up:
    if the printed document itself does not reconcile, the honest answer is
    the printed values plus an ambiguity note.
    """
    bullet_list = "\n".join(f"- {p}" for p in problems)
    return (
        "Your previous answer failed automated checks:\n"
        f"{bullet_list}\n\n"
        "Re-read the document carefully (every line-item row, every amount, the tax lines and the "
        "final total) and return a corrected JSON object. Typical causes are a missed or duplicated "
        "row, a misread digit, a value taken from the wrong column, or a charge/discount that belongs "
        "in adjustments_outside_items. Do NOT alter values to force the arithmetic to match: if the "
        "printed document itself does not add up, keep the printed values and explain in "
        "`ambiguities`.\n\n"
        f"Previous answer:\n{previous_output}"
    )


def build_page_hints(warnings: list[str]) -> str | None:
    if not warnings:
        return None
    lines = ["Preprocessing notes about the page images:"]
    for w in warnings:
        if "rotation" in w:
            lines.append(f"- {w}: the page may be rotated by 90 degrees; read the text in its actual orientation.")
        else:
            lines.append(f"- {w}: image quality may be poor; use null rather than guessing unclear values.")
    return "\n".join(lines)
