"""Financial consistency checks.

The required check from the brief:  sum(line_item.amount) + tax_total == total
(within a rounding tolerance). Around it we run cheaper, more specific checks
that help *localise* a failure:

* subtotal check   sum(amounts) vs printed subtotal  -> a missed/misread row
* tax-lines check  sum(tax_lines) vs tax_total       -> a mis-added tax
* row checks       quantity x unit_price vs amount   -> a misread digit in a row

These checks never modify extracted values. When the document has charges or
discounts outside the item table (shipping, round-off, deposits), the brief's
formula legitimately does not hold; we report that as `unresolved_adjustments`
instead of pretending the formula applies or silently folding them in.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum

from src.schema import LLMExtraction


class FinancialStatus(str, Enum):
    OK = "ok"  # line items + tax == total
    MISMATCH = "mismatch"  # does not reconcile and nothing printed explains it
    UNRESOLVED_ADJUSTMENTS = "unresolved_adjustments"  # reconciles only via charges outside the item table
    INSUFFICIENT_DATA = "insufficient_data"  # total or line items missing
    NOT_APPLICABLE = "not_applicable"  # not an invoice


@dataclass
class FinancialCheck:
    status: FinancialStatus
    tolerance: float
    line_items_sum: float | None = None
    tax_total: float | None = None
    computed_total: float | None = None
    printed_total: float | None = None
    difference: float | None = None
    adjustments_sum: float | None = None
    subtotal_matches: bool | None = None
    tax_lines_match: bool | None = None
    row_mismatches: list[int] = field(default_factory=list)  # 0-based indices
    tax_rate_mismatches: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status in (FinancialStatus.OK, FinancialStatus.NOT_APPLICABLE)

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["status"] = self.status.value
        d["passed"] = self.passed
        return d


def _close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol + 1e-9


def _row_tolerance(amount: float, tolerance: float) -> float:
    # Unit prices are often printed rounded (e.g. 3 x 333.33 = 999.99 vs 1000.00),
    # so per-row checks allow 0.5% relative slack in addition to the absolute tolerance.
    return max(tolerance, abs(amount) * 0.005)


_RATE = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def _tax_rate_mismatches(extraction: LLMExtraction, tolerance: float) -> list[str]:
    """Each tax line must equal its printed rate x the taxable base.

    Catches self-consistent misreads: a digit misread in a row whose unit price, subtotal and
    total were "made to add up" passes every other check, but the printed tax amount no longer
    matches the printed rate (seen in a held-out run: subtotal 3850 for 4050, tax 334.13 = 8.25%
    of 4050). Only checked when EVERY tax line prints the SAME rate (VAT 20%, or CGST 9% + SGST
    9%): with mixed rates (GST 12% on rooms, 5% on food) each rate applies to an unknown part of
    the subtotal, so there is nothing to check.
    """
    lines = extraction.tax_lines or []
    rates = [_RATE.search(t.label) for t in lines]
    if not lines or any(r is None for r in rates):
        return []
    values = {float(r.group(1)) for r in rates if r is not None}
    if len(values) != 1:
        return []
    rate = values.pop()
    items = extraction.line_items or []
    base = extraction.printed_subtotal if extraction.printed_subtotal is not None else sum(i.amount for i in items)
    if not base:
        return []
    expected = rate / 100 * base
    out = []
    for t in lines:
        if abs(t.amount - expected) > max(tolerance, 0.01 * abs(expected)):
            out.append(f"tax line {t.label!r} is {t.amount:.2f}, but {rate:g}% of {base:.2f} is {expected:.2f}")
    return out


def check_financials(extraction: LLMExtraction, *, tolerance: float = 0.02) -> FinancialCheck:
    if not extraction.is_invoice:
        return FinancialCheck(status=FinancialStatus.NOT_APPLICABLE, tolerance=tolerance)

    check = FinancialCheck(status=FinancialStatus.INSUFFICIENT_DATA, tolerance=tolerance)
    items = extraction.line_items or []
    check.printed_total = extraction.total
    check.tax_total = extraction.tax_total

    # Row-level: quantity x unit price ~= amount
    for idx, item in enumerate(items):
        if item.quantity is not None and item.unit_price is not None:
            expected = item.quantity * item.unit_price
            if not _close(expected, item.amount, _row_tolerance(item.amount, tolerance)):
                check.row_mismatches.append(idx)
                check.messages.append(
                    f"line item {idx + 1} ({item.description!r}): quantity {item.quantity} x unit price "
                    f"{item.unit_price} = {expected:.2f}, but amount is {item.amount:.2f}"
                )

    # Tax lines add up to tax_total
    if extraction.tax_lines is not None and extraction.tax_total is not None:
        tax_sum = round(sum(t.amount for t in extraction.tax_lines), 2)
        check.tax_lines_match = _close(tax_sum, extraction.tax_total, tolerance)
        if not check.tax_lines_match:
            check.messages.append(f"tax lines sum to {tax_sum:.2f} but tax_total is {extraction.tax_total:.2f}")

    check.tax_rate_mismatches = _tax_rate_mismatches(extraction, tolerance)
    check.messages.extend(check.tax_rate_mismatches)

    if not items or extraction.total is None:
        missing = [n for n, v in (("line_items", items or None), ("total", extraction.total)) if v is None]
        check.messages.append(f"cannot reconcile totals: missing {', '.join(missing)}")
        return check

    items_sum = round(sum(i.amount for i in items), 2)
    check.line_items_sum = items_sum

    if extraction.printed_subtotal is not None:
        check.subtotal_matches = _close(items_sum, extraction.printed_subtotal, tolerance)
        if not check.subtotal_matches:
            check.messages.append(
                f"line items sum to {items_sum:.2f} but the printed subtotal is {extraction.printed_subtotal:.2f}"
            )

    tax = extraction.tax_total if extraction.tax_total is not None else 0.0
    if extraction.tax_total is None:
        check.messages.append("tax_total is null; treated as 0 for the totals check")
    computed = round(items_sum + tax, 2)
    check.computed_total = computed
    check.difference = round(extraction.total - computed, 2)

    if _close(computed, extraction.total, tolerance):
        check.status = FinancialStatus.OK
        return check

    adjustments = extraction.adjustments_outside_items or []
    if adjustments:
        adj_sum = round(sum(a.amount for a in adjustments), 2)
        check.adjustments_sum = adj_sum
        if _close(computed + adj_sum, extraction.total, tolerance):
            check.status = FinancialStatus.UNRESOLVED_ADJUSTMENTS
            labels = ", ".join(a.label for a in adjustments)
            check.messages.append(
                f"line items + tax = {computed:.2f} differs from total {extraction.total:.2f}; the gap is explained "
                f"only by charges outside the item table ({labels}) which the required schema cannot represent"
            )
            return check

    check.status = FinancialStatus.MISMATCH
    check.messages.append(
        f"line items ({items_sum:.2f}) + tax ({tax:.2f}) = {computed:.2f}, but the extracted total is "
        f"{extraction.total:.2f} (difference {check.difference:+.2f}, tolerance {tolerance})"
    )
    return check
