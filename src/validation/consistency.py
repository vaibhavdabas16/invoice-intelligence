"""Self-consistency check for documents without a text layer (scans, photos).

Motivation (measured, see reports/failure_analysis.md): on degraded scans the
model sometimes produces fluent but wrong text, e.g. "Parrot" for a blurry
"Pomfret". Arithmetic checks cannot see these. A second, independent
extraction of the same page tends to disagree exactly where the first read was
uncertain, so disagreement is a cheap, model-agnostic review signal.

Arbitration (see `arbitrate`) never clears a disagreement: every disagreement
goes to a human. It only annotates which read reconciles with the printed
totals and which disputed values had low token confidence. Values are never
swapped between reads.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.eval.normalize import normalize_invoice_number, normalize_text, numbers_equal
from src.schema import InvoiceExtraction
from src.validation.confidence import FieldConfidence
from src.validation.financial import FinancialCheck


@dataclass(frozen=True)
class ReadDiff:
    kind: str  # "text" | "numeric" | "structure"
    message: str
    field: str = ""  # same labels as src/validation/confidence.py ("total", "line 4 amount", ...)


def compare_reads(a: InvoiceExtraction, b: InvoiceExtraction, *, tolerance: float = 0.01) -> list[ReadDiff]:
    """Typed disagreements between two extractions of the same document."""
    diffs: list[ReadDiff] = []
    if a.is_invoice != b.is_invoice:
        return [ReadDiff("structure", f"is_invoice ({a.is_invoice} vs {b.is_invoice})")]
    if not a.is_invoice:
        return diffs

    def text(name: str, norm: Callable[[Any], Any]) -> None:
        va, vb = getattr(a, name), getattr(b, name)
        if norm(va) != norm(vb):
            diffs.append(ReadDiff("text", f"{name} ({va!r} vs {vb!r})", name))

    text("vendor", normalize_text)
    text("invoice_number", normalize_invoice_number)
    text("invoice_date", lambda v: v)
    text("currency", lambda v: v)
    for name in ("tax_total", "total"):
        if not numbers_equal(getattr(a, name), getattr(b, name), tolerance):
            diffs.append(ReadDiff("numeric", f"{name} ({getattr(a, name)} vs {getattr(b, name)})", name))

    items_a, items_b = a.line_items or [], b.line_items or []
    if len(items_a) != len(items_b):
        diffs.append(ReadDiff("structure", f"line item count ({len(items_a)} vs {len(items_b)})"))
        return diffs
    for i, (ia, ib) in enumerate(zip(items_a, items_b, strict=True), start=1):
        if normalize_text(ia.description) != normalize_text(ib.description):
            diffs.append(
                ReadDiff(
                    "text", f"line {i} description ({ia.description!r} vs {ib.description!r})", f"line {i} description"
                )
            )
        if not numbers_equal(ia.quantity, ib.quantity, tolerance):
            diffs.append(
                ReadDiff("numeric", f"line {i} quantity ({ia.quantity} vs {ib.quantity})", f"line {i} quantity")
            )
        if not numbers_equal(ia.amount, ib.amount, tolerance):
            diffs.append(ReadDiff("numeric", f"line {i} amount ({ia.amount} vs {ib.amount})", f"line {i} amount"))
    return diffs


def fully_consistent(check: FinancialCheck) -> bool:
    """Every arithmetic check that could run passed (totals, subtotal, tax lines, rows)."""
    return (
        check.passed
        and check.subtotal_matches is not False
        and check.tax_lines_match is not False
        and not check.row_mismatches
        and not check.tax_rate_mismatches
    )


@dataclass(frozen=True)
class Arbitration:
    resolved: bool
    note: str


def arbitrate(
    diffs: list[ReadDiff],
    first: FinancialCheck,
    second: FinancialCheck,
    first_confidences: list[FieldConfidence] | None = None,
    threshold: float = 0.5,
) -> Arbitration:
    """Describe a re-read disagreement for the reviewer. Never clears it.

    Earlier versions cleared purely numeric disagreements when the first read reconciled
    (later: reconciled AND had high token confidence). Measured outcome of automatic acceptance:
    2 cases, 1 correct (U02, v2 batch) and 1 wrong (held-out L07: a misread digit propagated
    into a self-consistent unit price, subtotal and total with token p = 0.975). A value that
    adds up can still be wrong, so every disagreement now goes to a human; the note tells them
    which read reconciles and which disputed values the model was unsure about.
    """
    if not diffs:
        return Arbitration(True, "reads agree")
    if any(d.kind != "numeric" for d in diffs):
        return Arbitration(False, "reads disagree on text or structure")
    first_ok, second_ok = fully_consistent(first), fully_consistent(second)
    hint = {
        (True, False): "the first (kept) read reconciles with the printed totals; the re-read does not",
        (False, True): "only the re-read reconciles with the printed totals (first read kept, not swapped)",
        (True, True): "both reads reconcile internally",
        (False, False): "neither read reconciles",
    }[(first_ok, second_ok)]
    if first_confidences is not None:
        by_field = {c.field: c for c in first_confidences}
        unsure = [d.field for d in diffs if (c := by_field.get(d.field)) is not None and c.min_prob < threshold]
        if unsure:
            hint += f"; low token confidence in the first read on {', '.join(unsure)}"
    return Arbitration(False, f"numeric disagreement, needs review: {hint}")
