"""Turn raw model text into a validated `LLMExtraction`.

Structured outputs make malformed JSON rare, but providers without schema
enforcement (or truncated responses) still need defensive parsing: strip code
fences, then fall back to the outermost {...} span. Anything beyond that is a
parse failure, which the extractor turns into a repair attempt.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from src.schema import InvoiceExtraction, LLMExtraction

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


class OutputParseError(ValueError):
    pass


class OutputSchemaError(ValueError):
    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise OutputParseError("model returned an empty response")
    fence = _FENCE.match(stripped)
    if fence:
        stripped = fence.group(1)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start == -1 or end <= start:
            raise OutputParseError(f"response is not JSON: {stripped[:120]!r}") from None
        try:
            value = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise OutputParseError(f"response contains malformed JSON ({exc.msg} at char {exc.pos})") from None
    if not isinstance(value, dict):
        raise OutputParseError(f"expected a JSON object, got {type(value).__name__}")
    return value


def _format_validation_error(exc: ValidationError) -> list[str]:
    problems = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "(root)"
        problems.append(f"{loc}: {err['msg']}")
    return problems


def validate_extraction(data: dict[str, Any]) -> tuple[LLMExtraction, InvoiceExtraction]:
    """Validate the extended response, then project it onto the required schema.

    Raises `OutputSchemaError` listing every problem (so a repair prompt can
    address all of them at once).
    """
    try:
        llm = LLMExtraction.model_validate(data)
    except ValidationError as exc:
        raise OutputSchemaError(_format_validation_error(exc)) from None
    try:
        invoice = llm.to_invoice()
    except ValidationError as exc:
        raise OutputSchemaError(_format_validation_error(exc)) from None
    return llm, invoice
