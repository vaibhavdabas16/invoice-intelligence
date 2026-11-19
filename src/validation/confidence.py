"""Token-confidence review signal (optional; TOKEN_CONFIDENCE=true, LLM_EFFORT=none).

For every required field, take the lowest probability among the output tokens
that spell its value. A low value means the model seriously considered another
character there, e.g. invoice number "SGS/24-47" where the token "24" had
p = 0.012 (the page prints "SGS/1847").

What it is NOT: a calibrated probability that the field is correct. It is a
per-token sampling probability, used only as a threshold signal for review,
and it cannot see errors the model makes confidently (p close to 1).

Spans: a small JSON scanner records the character span of every scalar value;
tokens are mapped to characters by concatenation. Formatting-only characters
are excluded by construction: string quotes, and the trailing ".00" of numbers
(920.0 vs 920.00 is not a different value, but the model is often unsure which
to write).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null")

REQUIRED_SCALARS = ("vendor", "invoice_number", "invoice_date", "currency", "tax_total", "total")
ITEM_FIELDS = ("description", "quantity", "amount")


@dataclass
class FieldConfidence:
    field: str
    value: str
    min_prob: float
    weakest_token: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["min_prob"] = round(self.min_prob, 4)
        return d


class _Scanner:
    def __init__(self, text: str) -> None:
        self.text = text
        self.i = 0
        self.spans: dict[str, tuple[int, int, str]] = {}

    def _ws(self) -> None:
        while self.i < len(self.text) and self.text[self.i] in " \t\r\n":
            self.i += 1

    def _string(self) -> tuple[str, int, int]:
        assert self.text[self.i] == '"'
        start = self.i + 1
        self.i += 1
        while self.text[self.i] != '"':
            self.i += 2 if self.text[self.i] == "\\" else 1
        end = self.i
        self.i += 1
        return json.loads(self.text[start - 1 : end + 1]), start, end

    def value(self, path: str) -> None:
        self._ws()
        ch = self.text[self.i]
        if ch == "{":
            self.i += 1
            self._ws()
            if self.text[self.i] == "}":
                self.i += 1
                return
            while True:
                self._ws()
                key, _, _ = self._string()
                self._ws()
                assert self.text[self.i] == ":"
                self.i += 1
                self.value(f"{path}.{key}" if path else key)
                self._ws()
                if self.text[self.i] == ",":
                    self.i += 1
                    continue
                assert self.text[self.i] == "}"
                self.i += 1
                return
        if ch == "[":
            self.i += 1
            self._ws()
            if self.text[self.i] == "]":
                self.i += 1
                return
            idx = 0
            while True:
                self.value(f"{path}[{idx}]")
                idx += 1
                self._ws()
                if self.text[self.i] == ",":
                    self.i += 1
                    continue
                assert self.text[self.i] == "]"
                self.i += 1
                return
        if ch == '"':
            _, start, end = self._string()
            self.spans[path] = (start, end, "string")
            return
        m = _NUMBER.match(self.text, self.i)
        assert m is not None, f"unexpected character at {self.i}"
        kind = "number" if m.group(0)[0] in "-0123456789" else "literal"
        self.spans[path] = (m.start(), m.end(), kind)
        self.i = m.end()


def value_spans(text: str) -> dict[str, tuple[int, int, str]]:
    """{json_path: (start, end, kind)} for every scalar value in a JSON document."""
    scanner = _Scanner(text)
    scanner.value("")
    return scanner.spans


def _significant_end(text: str, start: int, end: int) -> int:
    literal = text[start:end]
    if "." in literal and "e" not in literal.lower():
        stripped = literal.rstrip("0").rstrip(".")
        return start + max(len(stripped), 1)
    return end


def _label(path: str) -> str:
    m = re.fullmatch(r"line_items\[(\d+)\]\.(\w+)", path)
    return f"line {int(m.group(1)) + 1} {m.group(2)}" if m else path


def field_confidences(text: str, token_logprobs: list[list[Any]]) -> list[FieldConfidence] | None:
    """None if token data is missing or does not reproduce the text exactly (then nothing is claimed)."""
    if not token_logprobs or "".join(t[0] for t in token_logprobs) != text:
        return None
    try:
        spans = value_spans(text)
    except (AssertionError, IndexError, json.JSONDecodeError):
        return None
    offsets = []
    pos = 0
    for token, logprob in token_logprobs:
        offsets.append((pos, pos + len(token), token, math.exp(logprob)))
        pos += len(token)

    paths = [p for p in REQUIRED_SCALARS if p in spans]
    paths += sorted(
        (p for p in spans if re.fullmatch(rf"line_items\[\d+\]\.({'|'.join(ITEM_FIELDS)})", p)),
        key=lambda p: (int(re.search(r"\d+", p).group()), ITEM_FIELDS.index(p.rsplit(".", 1)[1])),  # type: ignore[union-attr]
    )
    out: list[FieldConfidence] = []
    for path in paths:
        start, end, kind = spans[path]
        if kind == "literal":  # null / true / false: no value to doubt here
            continue
        if kind == "number":
            end = _significant_end(text, start, end)
        covering = [o for o in offsets if o[0] < end and o[1] > start]
        if not covering:
            continue
        weakest = min(covering, key=lambda o: o[3])
        out.append(FieldConfidence(_label(path), text[spans[path][0] : spans[path][1]], weakest[3], weakest[2]))
    return out


def low_confidence_reasons(confidences: list[FieldConfidence] | None, threshold: float) -> list[str]:
    if not confidences:
        return []
    low = [c for c in confidences if c.min_prob < threshold]
    if not low:
        return []
    shown = "; ".join(f"{c.field} {c.value!r} (token {c.weakest_token!r} p={c.min_prob:.2f})" for c in low[:4])
    more = f" (+{len(low) - 4} more)" if len(low) > 4 else ""
    return [f"low token confidence: {shown}{more}"]
