"""Field-level evidence: where in the document each extracted value is printed.

For digital PDFs, every extracted value is searched for in the PDF's embedded
text layer, and the page number and source line are recorded. The evidence is
produced by deterministic string search, never by the model, so it cannot be
hallucinated: if a value is not printed, the search does not find it.

Rules (fixed before measuring; see reports/improvement_report.md):

* text fields (vendor, line-item descriptions): case-insensitive,
  whitespace-insensitive substring match. Punctuation must match exactly.
* invoice_number: the same, but as a WHOLE token (no letter or digit directly
  before or after), so a number with a dropped leading character ("WL-88213")
  does not ground inside the printed "NWL-88213". (Tightened after the error-
  injection measurement; see the improvement report.)
* amounts (line-item amounts, tax_total, total): the absolute value rendered as
  plain, Western-grouped (1,234.50), Indian-grouped (1,23,456.50) and
  European (1.234,50) text, with 2, 1 or 0 decimals where that is the same
  number; digit boundaries are enforced so 15.00 cannot match inside 115.00.
  `tax_total == 0` is not checked (invoices often print no tax line at all).
  `tax_total` is defined as the SUM of the tax lines (client convention), so on
  invoices with several tax lines (CGST + SGST) it is never printed as one number.
  It is grounded directly if printed, otherwise "derived": every printed tax line
  is found and the lines sum to tax_total. (Correction made after the first
  measurement flagged 3 multi-tax invoices; see the improvement report.)
* invoice_date: the ISO date rendered in a fixed list of common printed formats.
* quantity and currency are not checked: small integers match by accident, and
  the currency code is usually inferred from a symbol rather than printed.

A value that is not found is `unsupported`; on a digital PDF that sets
needs_review. Grounding is necessary, not sufficient: a wrong value that happens
to be printed elsewhere on the page (e.g. a subtotal used as the total) still
grounds. Scans and photos have no text layer, so their values are
`not_checkable` and rely on the re-read and token-confidence checks instead.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date

from src.schema import InvoiceExtraction

_WS = re.compile(r"\s+")

DATE_FORMATS = (
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%d %B %Y",
    "%d %b %Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%d-%b-%Y",
    "%d %B, %Y",
    "%Y/%m/%d",
)


@dataclass
class FieldEvidence:
    field: str
    value: str
    status: str  # "grounded" | "derived" | "unsupported" | "not_checkable"
    page: int | None = None  # 1-based
    source_text: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _squash(text: str) -> str:
    return _WS.sub("", text).casefold()


def _indian_group(integer: int) -> str:
    s = str(integer)
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    groups: list[str] = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return ",".join(groups) + "," + tail


def amount_renderings(value: float) -> list[str]:
    v = abs(value)
    out: set[str] = set()
    decimals = [2]
    if round(v, 1) == round(v, 2):
        decimals.append(1)
    if v == int(v):
        decimals.append(0)
    for d in decimals:
        plain = f"{v:.{d}f}"
        integer, _, frac = plain.partition(".")
        frac_part = f".{frac}" if frac else ""
        out.add(plain)
        out.add(f"{int(integer):,}{frac_part}")
        out.add(_indian_group(int(integer)) + frac_part)
        out.add(f"{int(integer):,}".replace(",", ".") + (f",{frac}" if frac else ""))
        # space as thousands separator (1 234,50 / 1 234.50)
        out.add(f"{int(integer):,}".replace(",", " ") + (f",{frac}" if frac else ""))
        out.add(f"{int(integer):,}".replace(",", " ") + frac_part)
    return sorted(out, key=len, reverse=True)


def date_renderings(iso: str) -> list[str]:
    try:
        d = date.fromisoformat(iso)
    except ValueError:
        return []
    out: set[str] = set()
    for fmt in DATE_FORMATS:
        text = d.strftime(fmt)
        out.add(text)
        # also without leading zeros on day/month ("8 April 2026", "3/4/2026")
        out.add(re.sub(r"(?<!\d)0(\d)", r"\1", text))
    return sorted(out, key=len, reverse=True)


def _find_text(needle: str, pages: list[str]) -> tuple[int, str] | None:
    target = _squash(needle)
    if not target:
        return None
    for page_no, page in enumerate(pages, start=1):
        for line in page.splitlines():
            if target in _squash(line):
                return page_no, line.strip()
        if target in _squash(page):  # value wrapped across lines
            return page_no, needle
    return None


def _find_token(value: str, pages: list[str]) -> tuple[int, str] | None:
    chars = [re.escape(ch) for ch in _WS.sub("", value)]
    if not chars:
        return None
    pattern = re.compile(r"(?<![A-Za-z0-9])" + r"\s*".join(chars) + r"(?![A-Za-z0-9])", re.IGNORECASE)
    for page_no, page in enumerate(pages, start=1):
        for line in page.splitlines():
            if pattern.search(line):
                return page_no, line.strip()
        if pattern.search(page):
            return page_no, value
    return None


def _find_number(renderings: list[str], pages: list[str]) -> tuple[int, str] | None:
    for page_no, page in enumerate(pages, start=1):
        for line in page.splitlines():
            # Whitespace is kept: removing it would merge adjacent table columns
            # ("320.00   3,200.00" -> "320.003,200.00") and break the digit boundaries.
            for r in renderings:
                if re.search(rf"(?<![\d.,]){re.escape(r)}(?![\d])", line):
                    return page_no, line.strip()
    return None


def collect_evidence(
    invoice: InvoiceExtraction, page_texts: list[str | None], tax_lines: list[float] | None = None
) -> list[FieldEvidence]:
    """Evidence for every checkable field. `page_texts[i]` is page i's text layer (None for image pages).

    `tax_lines` are the individual tax amounts the model read (auxiliary output); they are only
    used to ground a tax_total that is a sum and therefore not printed itself.
    """
    if not invoice.is_invoice:
        return []
    pages = [t or "" for t in page_texts]
    checkable = any(t for t in page_texts)

    def record(field: str, value: object, found: tuple[int, str] | None) -> FieldEvidence:
        if not checkable:
            return FieldEvidence(field, str(value), "not_checkable")
        if found is None:
            return FieldEvidence(field, str(value), "unsupported")
        return FieldEvidence(field, str(value), "grounded", found[0], found[1])

    ev: list[FieldEvidence] = []
    if invoice.vendor is not None:
        ev.append(record("vendor", invoice.vendor, _find_text(invoice.vendor, pages) if checkable else None))
    if invoice.invoice_number is not None:
        number = invoice.invoice_number
        ev.append(record("invoice_number", number, _find_token(number, pages) if checkable else None))
    if invoice.invoice_date is not None:
        found = None
        if checkable:
            for r in date_renderings(invoice.invoice_date):
                found = _find_text(r, pages)
                if found:
                    break
        ev.append(record("invoice_date", invoice.invoice_date, found))
    for name in ("tax_total", "total"):
        value = getattr(invoice, name)
        if value is None or (name == "tax_total" and value == 0):
            continue
        found = _find_number(amount_renderings(value), pages) if checkable else None
        if checkable and found is None and name == "tax_total" and tax_lines:
            parts = [_find_number(amount_renderings(t), pages) for t in tax_lines]
            if all(parts) and abs(sum(tax_lines) - value) <= 0.02:
                sources = " + ".join(p[1] for p in parts if p)
                ev.append(FieldEvidence(name, str(value), "derived", parts[0][0] if parts[0] else None, sources))
                continue
        ev.append(record(name, value, found))
    for i, item in enumerate(invoice.line_items or [], start=1):
        ev.append(
            record(
                f"line {i} description", item.description, _find_text(item.description, pages) if checkable else None
            )
        )
        ev.append(
            record(
                f"line {i} amount",
                item.amount,
                _find_number(amount_renderings(item.amount), pages) if checkable else None,
            )
        )
    return ev


def unsupported_reasons(evidence: list[FieldEvidence]) -> list[str]:
    bad = [e for e in evidence if e.status == "unsupported"]
    if not bad:
        return []
    shown = "; ".join(f"{e.field} {e.value!r}" for e in bad[:4])
    more = f" (+{len(bad) - 4} more)" if len(bad) > 4 else ""
    return [f"value(s) not found in the PDF text layer: {shown}{more}"]
