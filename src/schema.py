"""Pydantic models.

Two layers:

* `InvoiceExtraction` is the *required* output schema from the assignment.
  It is what gets written to `outputs/*.json` and is validated strictly.
* `LLMExtraction` is what we ask the model to return: the required fields plus
  auxiliary evidence (document type, printed subtotal, tax lines, unit prices,
  adjustments outside the item table, ambiguities). The auxiliary fields power
  validation and the review flag and are *never* written into the required
  output.
"""

from __future__ import annotations

import math
import re
from datetime import date
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.validation.currencies import ISO_4217_CODES

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_CURRENCY = re.compile(r"^[A-Z]{3}$")

REQUIRED_FIELDS: tuple[str, ...] = (
    "is_invoice",
    "vendor",
    "invoice_number",
    "invoice_date",
    "currency",
    "line_items",
    "tax_total",
    "total",
)


def _check_finite(value: float | None, field_name: str) -> float | None:
    if value is not None and not math.isfinite(value):
        raise ValueError(f"{field_name} must be a finite number")
    return value


def validate_iso_date(value: str | None) -> str | None:
    if value is None:
        return None
    if not _ISO_DATE.match(value):
        raise ValueError(f"invoice_date must be YYYY-MM-DD, got {value!r}")
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invoice_date is not a real calendar date: {value!r}") from exc
    return value


def validate_currency(value: str | None) -> str | None:
    if value is None:
        return None
    if not _CURRENCY.match(value):
        raise ValueError(f"currency must be a 3-letter upper-case ISO 4217 code, got {value!r}")
    if value not in ISO_4217_CODES:
        raise ValueError(f"currency {value!r} is not an ISO 4217 code")
    return value


class LineItem(BaseModel):
    # strict=True: "1,200.00" (a string) is rejected instead of silently coerced.
    model_config = ConfigDict(extra="forbid", strict=True)

    description: str = Field(min_length=1)
    quantity: float | None
    amount: float

    @field_validator("quantity", "amount")
    @classmethod
    def _finite(cls, v: float | None, info: Any) -> float | None:
        return _check_finite(v, info.field_name)


class InvoiceExtraction(BaseModel):
    """The assignment's output schema. Field order matches the brief."""

    model_config = ConfigDict(extra="forbid", strict=True)

    is_invoice: bool
    vendor: str | None
    invoice_number: str | None
    invoice_date: str | None
    currency: str | None
    line_items: list[LineItem] | None
    tax_total: float | None
    total: float | None

    @field_validator("vendor", "invoice_number")
    @classmethod
    def _non_blank(cls, v: str | None, info: Any) -> str | None:
        if v is not None and not v.strip():
            raise ValueError(f"{info.field_name} must be null rather than an empty string")
        return v

    @field_validator("invoice_date")
    @classmethod
    def _date(cls, v: str | None) -> str | None:
        return validate_iso_date(v)

    @field_validator("currency")
    @classmethod
    def _currency(cls, v: str | None) -> str | None:
        return validate_currency(v)

    @field_validator("tax_total", "total")
    @classmethod
    def _finite(cls, v: float | None, info: Any) -> float | None:
        return _check_finite(v, info.field_name)

    @model_validator(mode="after")
    def _non_invoice_fields_are_null(self) -> InvoiceExtraction:
        if not self.is_invoice:
            populated = [f for f in REQUIRED_FIELDS[1:] if getattr(self, f) is not None]
            if populated:
                raise ValueError(f"is_invoice is false but these fields are not null: {populated}")
        return self

    @classmethod
    def non_invoice(cls) -> InvoiceExtraction:
        return cls(
            is_invoice=False,
            vendor=None,
            invoice_number=None,
            invoice_date=None,
            currency=None,
            line_items=None,
            tax_total=None,
            total=None,
        )

    def to_output_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Extended schema requested from the LLM
# ---------------------------------------------------------------------------


class DocumentType(str, Enum):
    INVOICE = "invoice"
    PURCHASE_ORDER = "purchase_order"
    BANK_STATEMENT = "bank_statement"
    LETTER = "letter"
    RECEIPT = "receipt"
    QUOTATION = "quotation"
    CREDIT_NOTE = "credit_note"
    DELIVERY_NOTE = "delivery_note"
    OTHER = "other"
    UNREADABLE = "unreadable"


class LabeledAmount(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    label: str
    amount: float


class LLMLineItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    description: str = Field(min_length=1)
    quantity: float | None
    unit_price: float | None
    amount: float


class LLMExtraction(BaseModel):
    """Raw model response. Validated strictly, then projected onto `InvoiceExtraction`."""

    model_config = ConfigDict(extra="forbid", strict=True)

    document_type: DocumentType
    is_invoice: bool
    vendor: str | None
    invoice_number: str | None
    invoice_date: str | None
    currency: str | None
    line_items: list[LLMLineItem] | None
    printed_subtotal: float | None
    tax_lines: list[LabeledAmount] | None
    tax_total: float | None
    adjustments_outside_items: list[LabeledAmount] | None
    total: float | None
    ambiguities: list[str]
    illegible_fields: list[str]

    @model_validator(mode="before")
    @classmethod
    def _coerce_document_type(cls, data: Any) -> Any:
        # strict mode rejects plain strings for Enum fields; accept the value form.
        if isinstance(data, dict) and isinstance(data.get("document_type"), str):
            try:
                data = {**data, "document_type": DocumentType(data["document_type"])}
            except ValueError:
                pass  # leave as-is so validation reports a clear error
        return data

    def to_invoice(self) -> InvoiceExtraction:
        """Project onto the required schema; raises `ValidationError` on violations.

        For non-invoices every other field is forced to null, as the brief
        requires, regardless of what the model put there.
        """
        if not self.is_invoice:
            return InvoiceExtraction.non_invoice()
        items = (
            [LineItem(description=i.description, quantity=i.quantity, amount=i.amount) for i in self.line_items]
            if self.line_items is not None
            else None
        )
        return InvoiceExtraction(
            is_invoice=True,
            vendor=self.vendor,
            invoice_number=self.invoice_number,
            invoice_date=self.invoice_date,
            currency=self.currency,
            line_items=items,
            tax_total=self.tax_total,
            total=self.total,
        )
