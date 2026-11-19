"""Normalisation rules used by the evaluation harness.

Principle: normalise away *presentation* differences that a finance user would
not consider a different answer (case, whitespace, "&" vs "and", thousands
separators), and nothing that changes meaning. Every rule here is listed in
README.md -> "Matching rules".
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from typing import Any

# Legal-form suffixes stripped only by the *lenient* vendor rule (reported separately).
LEGAL_SUFFIXES = (
    "private limited",
    "pvt ltd",
    "pvt",
    "limited",
    "ltd",
    "llp",
    "llc",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
    "gmbh",
    "plc",
    "sa",
    "ag",
    "bv",
)


def normalize_text(value: str | None) -> str | None:
    """Unicode NFKC, case-fold, '&' -> 'and', punctuation -> space, collapse whitespace."""
    if value is None:
        return None
    s = unicodedata.normalize("NFKC", value).casefold()
    s = s.replace("&", " and ")
    s = re.sub(r"[^\w\s]", " ", s)  # punctuation is presentation, word characters are content
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_vendor_lenient(value: str | None) -> str | None:
    """normalize_text + strip trailing legal-form suffixes ('Acme Pvt Ltd' == 'Acme')."""
    s = normalize_text(value)
    if s is None:
        return None
    changed = True
    while changed:
        changed = False
        for suffix in LEGAL_SUFFIXES:
            if s.endswith(" " + suffix):
                s = s[: -len(suffix) - 1].strip()
                changed = True
    return s


def normalize_invoice_number(value: str | None) -> str | None:
    """Case-fold and remove whitespace only.

    Separators ('/', '-', '.') and leading zeros are kept: 'KOM/26-27/00418'
    and 'KOM-26-27-418' are different identifiers to a payables system.
    """
    if value is None:
        return None
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value)).casefold()


def normalize_currency(value: str | None) -> str | None:
    return value.strip().upper() if value is not None else None


_UNAMBIGUOUS_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d %B %Y",
    "%d %b %Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%d-%b-%Y",
    "%d %B, %Y",
)


def normalize_date(value: str | None) -> str | None:
    """Return ISO YYYY-MM-DD for unambiguous formats; otherwise the stripped input.

    Purely numeric day/month forms (04/09/2026) are deliberately NOT parsed:
    they are ambiguous, and guessing would turn a real error into a match.
    """
    if value is None:
        return None
    s = value.strip()
    for fmt in _UNAMBIGUOUS_DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    try:
        return date.fromisoformat(s).isoformat()
    except ValueError:
        return s


def to_number(value: Any) -> float | None:
    """Parse JSON numbers and number-like strings ('1,200.00', '₹1,200', '1 200') to float.

    Used for comparison only, so that '1,200.00' vs 1200 counts as equal.
    Booleans are not numbers.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = re.sub(r"[^\d.\-]", "", value.replace("−", "-"))
        try:
            return float(s) if s not in ("", "-", ".") else None
        except ValueError:
            return None
    return None


def numbers_equal(a: Any, b: Any, tolerance: float) -> bool:
    """Both null -> equal; one null -> not equal; else |a - b| <= tolerance."""
    if a is None and b is None:
        return True
    na, nb = to_number(a), to_number(b)
    if na is None or nb is None:
        return False
    return abs(na - nb) <= tolerance + 1e-9
