"""Field-by-field comparison of a prediction against ground truth.

For every field we record two verdicts:
  * `exact`   - raw values identical (numbers within tolerance).
  * `correct` - equal after the documented normalisation. This is the headline.

Line items are aligned one-to-one (order-insensitive greedy matching on a
similarity score) so that one missed row does not make every following row
"wrong". The *field* `line_items` is correct only if both lists have the same
length and every aligned pair matches on description, quantity and amount.
Row order is reported separately, not folded into correctness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Any

from src.eval.normalize import (
    normalize_currency,
    normalize_date,
    normalize_invoice_number,
    normalize_text,
    normalize_vendor_lenient,
    numbers_equal,
)

FIELDS = ("is_invoice", "vendor", "invoice_number", "invoice_date", "currency", "line_items", "tax_total", "total")


@dataclass
class FieldResult:
    field: str
    correct: bool
    exact: bool
    predicted: Any
    expected: Any
    lenient: bool | None = None  # only for vendor
    note: str = ""


@dataclass
class ItemPair:
    gt_index: int | None
    pred_index: int | None
    description_ok: bool
    quantity_ok: bool
    amount_ok: bool
    description_similarity: float

    @property
    def exact_match(self) -> bool:
        return self.description_ok and self.quantity_ok and self.amount_ok


@dataclass
class LineItemResult:
    gt_count: int
    pred_count: int
    pairs: list[ItemPair] = field(default_factory=list)
    order_preserved: bool | None = None

    @property
    def matched_items(self) -> int:
        return sum(1 for p in self.pairs if p.gt_index is not None and p.pred_index is not None and p.exact_match)

    def attribute_correct(self, attr: str) -> int:
        """Count of GT items whose aligned prediction has the attribute right (unaligned GT items count wrong)."""
        return sum(
            1 for p in self.pairs if p.gt_index is not None and p.pred_index is not None and getattr(p, f"{attr}_ok")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "gt_count": self.gt_count,
            "pred_count": self.pred_count,
            "matched_items": self.matched_items,
            "order_preserved": self.order_preserved,
            "pairs": [asdict(p) for p in self.pairs],
        }


@dataclass
class DocumentComparison:
    document_id: str
    fields: dict[str, FieldResult]
    line_items: LineItemResult | None

    @property
    def correct_count(self) -> int:
        return sum(1 for f in self.fields.values() if f.correct)

    @property
    def all_correct(self) -> bool:
        return all(f.correct for f in self.fields.values())

    @property
    def wrong_fields(self) -> list[str]:
        return [name for name, f in self.fields.items() if not f.correct]


# ---------------------------------------------------------------------------
# Line items
# ---------------------------------------------------------------------------


def _desc_similarity(a: str, b: str) -> float:
    na, nb = normalize_text(a) or "", normalize_text(b) or ""
    return SequenceMatcher(None, na, nb).ratio()


def align_line_items(pred: list[dict[str, Any]], gt: list[dict[str, Any]], tolerance: float) -> LineItemResult:
    result = LineItemResult(gt_count=len(gt), pred_count=len(pred))
    candidates: list[tuple[float, int, int, float]] = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            sim = _desc_similarity(str(p.get("description", "")), str(g.get("description", "")))
            amount_ok = numbers_equal(p.get("amount"), g.get("amount"), tolerance)
            qty_ok = numbers_equal(p.get("quantity"), g.get("quantity"), tolerance)
            # Pair only rows that plausibly describe the same thing.
            if sim < 0.5 and not amount_ok:
                continue
            score = 2.0 * sim + 1.0 * amount_ok + 0.5 * qty_ok
            candidates.append((score, gi, pi, sim))
    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))

    used_g: set[int] = set()
    used_p: set[int] = set()
    for _score, gi, pi, sim in candidates:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        g, p = gt[gi], pred[pi]
        result.pairs.append(
            ItemPair(
                gt_index=gi,
                pred_index=pi,
                description_ok=normalize_text(str(p.get("description", "")))
                == normalize_text(str(g.get("description", ""))),
                quantity_ok=numbers_equal(p.get("quantity"), g.get("quantity"), tolerance),
                amount_ok=numbers_equal(p.get("amount"), g.get("amount"), tolerance),
                description_similarity=round(sim, 3),
            )
        )
    for gi in range(len(gt)):
        if gi not in used_g:
            result.pairs.append(ItemPair(gi, None, False, False, False, 0.0))
    for pi in range(len(pred)):
        if pi not in used_p:
            result.pairs.append(ItemPair(None, pi, False, False, False, 0.0))
    result.pairs.sort(key=lambda p: (p.gt_index is None, p.gt_index or 0, p.pred_index or 0))

    matched = [p for p in result.pairs if p.gt_index is not None and p.pred_index is not None]
    pred_order = [p.pred_index for p in sorted(matched, key=lambda p: p.gt_index or 0)]
    result.order_preserved = pred_order == sorted(pred_order)  # type: ignore[type-var]
    return result


def _line_items_field(pred: Any, gt: Any, tolerance: float) -> tuple[FieldResult, LineItemResult | None]:
    if gt is None or pred is None:
        ok = gt is None and pred is None
        return FieldResult("line_items", ok, ok, pred, gt), None
    li = align_line_items(pred, gt, tolerance)
    correct = li.gt_count == li.pred_count and li.matched_items == li.gt_count
    exact = (
        correct
        and all(p.get("description") == g.get("description") for p, g in zip(pred, gt, strict=False))
        and bool(li.order_preserved)
    )
    note = f"{li.matched_items}/{li.gt_count} items matched; predicted {li.pred_count}"
    if li.order_preserved is False:
        note += "; order differs"
    return FieldResult(
        "line_items", correct, exact, f"<{li.pred_count} items>", f"<{li.gt_count} items>", note=note
    ), li


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


def compare_document(
    document_id: str, pred: dict[str, Any] | None, gt: dict[str, Any], *, tolerance: float = 0.01
) -> DocumentComparison:
    """`pred` is None when the pipeline failed: every field is then counted wrong."""
    p = pred or {}
    failed = pred is None
    out: dict[str, FieldResult] = {}

    def fr(name: str, correct: bool, exact: bool, **kw: Any) -> None:
        out[name] = FieldResult(name, correct and not failed, exact and not failed, p.get(name), gt.get(name), **kw)

    fr("is_invoice", p.get("is_invoice") is gt["is_invoice"], p.get("is_invoice") is gt["is_invoice"])

    pv, gv = p.get("vendor"), gt.get("vendor")
    fr(
        "vendor",
        normalize_text(pv) == normalize_text(gv),
        pv == gv,
        lenient=(normalize_vendor_lenient(pv) == normalize_vendor_lenient(gv)) and not failed,
    )
    pn, gn = p.get("invoice_number"), gt.get("invoice_number")
    fr("invoice_number", normalize_invoice_number(pn) == normalize_invoice_number(gn), pn == gn)
    pd, gd = p.get("invoice_date"), gt.get("invoice_date")
    fr("invoice_date", normalize_date(pd) == normalize_date(gd), pd == gd)
    pc, gc = p.get("currency"), gt.get("currency")
    fr("currency", normalize_currency(pc) == normalize_currency(gc), pc == gc)

    li_field, li = _line_items_field(p.get("line_items"), gt.get("line_items"), tolerance)
    li_field.correct = li_field.correct and not failed
    li_field.exact = li_field.exact and not failed
    out["line_items"] = li_field

    for name in ("tax_total", "total"):
        # exact: identical JSON numbers (3200 == 3200.0); correct: within tolerance after number parsing.
        fr(name, numbers_equal(p.get(name), gt.get(name), tolerance), p.get(name) == gt.get(name))

    if failed:
        for f in out.values():
            f.note = (f.note + "; " if f.note else "") + "pipeline failed, no prediction"
    return DocumentComparison(document_id=document_id, fields=out, line_items=li)
