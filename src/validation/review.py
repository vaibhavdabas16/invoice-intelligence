"""Human-review decision (bonus: `needs_review`).

Rule-based on purpose: each reason is a concrete, explainable signal a finance
reviewer can act on. There is no numeric "confidence"; an uncalibrated LLM
self-reported confidence would look like a probability without being one.
Whether these rules actually concentrate the errors is measured by the eval
harness (precision / recall of the flag against real field errors).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.schema import DocumentType, LLMExtraction
from src.validation.financial import FinancialCheck, FinancialStatus

CRITICAL_FIELDS = ("vendor", "invoice_number", "invoice_date", "currency", "total")


@dataclass
class ReviewDecision:
    needs_review: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {"needs_review": self.needs_review, "reasons": self.reasons}


def decide_review(
    llm: LLMExtraction,
    financial: FinancialCheck,
    *,
    preprocessing_warnings: list[str],
    repair_attempted: bool,
    consistency_disagreements: list[str] | None = None,
    evidence_issues: list[str] | None = None,
    confidence_issues: list[str] | None = None,
) -> ReviewDecision:
    """`preprocessing_warnings` (contrast, sharpness, rotation, resolution) are deliberately NOT
    review triggers: measured over the stress and rotated sets they flagged correct documents
    (0 true positives) while the errors were caught by other signals. They stay in metadata and
    are passed to the model as hints. See reports/remediation_report.md."""
    reasons: list[str] = []

    if llm.document_type is DocumentType.UNREADABLE:
        reasons.append("model reported the document as unreadable")
    if llm.is_invoice != (llm.document_type is DocumentType.INVOICE):
        reasons.append(
            f"classification conflict: is_invoice={llm.is_invoice} but document_type={llm.document_type.value}"
        )

    if llm.is_invoice:
        missing = [f for f in CRITICAL_FIELDS if getattr(llm, f) is None]
        if not llm.line_items:
            missing.append("line_items")
        if missing:
            reasons.append(f"missing critical fields: {', '.join(missing)}")

        if financial.status is FinancialStatus.MISMATCH:
            reasons.append(f"totals do not reconcile (difference {financial.difference:+.2f})")
        elif financial.status is FinancialStatus.UNRESOLVED_ADJUSTMENTS:
            reasons.append("totals reconcile only via charges outside the item table")
        if financial.subtotal_matches is False:
            reasons.append("line items do not sum to the printed subtotal")
        if financial.tax_lines_match is False:
            reasons.append("tax lines do not sum to tax_total")
        for message in financial.tax_rate_mismatches:
            reasons.append(message)
        if financial.row_mismatches:
            rows = ", ".join(str(i + 1) for i in financial.row_mismatches)
            reasons.append(f"quantity x unit price != amount on row(s) {rows}")

    if llm.illegible_fields:
        reasons.append(f"model reported illegible fields: {', '.join(llm.illegible_fields)}")
    if llm.ambiguities:
        reasons.append(f"model reported {len(llm.ambiguities)} ambiguity note(s)")
    del preprocessing_warnings  # informational only (see docstring)
    for issue in [*(evidence_issues or []), *(confidence_issues or [])]:
        reasons.append(issue)
    if consistency_disagreements:
        shown = "; ".join(consistency_disagreements[:5])
        more = f" (+{len(consistency_disagreements) - 5} more)" if len(consistency_disagreements) > 5 else ""
        reasons.append(f"independent re-read of the scan disagrees: {shown}{more}")
    if repair_attempted:
        reasons.append("first extraction failed automated checks and needed a repair call")

    return ReviewDecision(needs_review=bool(reasons), reasons=reasons)
