"""Test doubles: a scripted LLM provider so unit tests never hit the network or spend money."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from src.extraction.providers.base import LLMProvider, LLMRequest, LLMResponse, TokenUsage


class ScriptedProvider(LLMProvider):
    """Returns queued responses in order. Queue items may be str/dict (response text) or an Exception to raise."""

    name = "fake"

    def __init__(self, script: Iterable[Any], model: str = "gpt-6-luna", usage: TokenUsage | None = None):
        self.model = model
        self.script = list(script)
        self.requests: list[LLMRequest] = []
        self.usage = usage or TokenUsage(input_tokens=1000, output_tokens=200)

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if not self.script:
            raise AssertionError("ScriptedProvider ran out of scripted responses")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        text = item if isinstance(item, str) else json.dumps(item)
        return LLMResponse(
            text=text,
            usage=TokenUsage(**self.usage.to_dict()),
            provider=self.name,
            model=self.model,
            stop_reason="end_turn",
            request_id="req_test",
            latency_s=0.01,
        )


def llm_invoice(**overrides: Any) -> dict[str, Any]:
    """A valid extended-schema invoice response whose arithmetic reconciles."""
    base: dict[str, Any] = {
        "document_type": "invoice",
        "is_invoice": True,
        "vendor": "Acme Supplies Pvt Ltd",
        "invoice_number": "INV-2041",
        "invoice_date": "2026-03-14",
        "currency": "INR",
        "line_items": [
            {"description": "A4 paper, 500 sheets", "quantity": 10, "unit_price": 320.0, "amount": 3200.0},
        ],
        "printed_subtotal": 3200.0,
        "tax_lines": [{"label": "GST 18%", "amount": 576.0}],
        "tax_total": 576.0,
        "adjustments_outside_items": [],
        "total": 3776.0,
        "ambiguities": [],
        "illegible_fields": [],
    }
    base.update(overrides)
    return base


def llm_non_invoice(document_type: str = "purchase_order") -> dict[str, Any]:
    return {
        "document_type": document_type,
        "is_invoice": False,
        "vendor": None,
        "invoice_number": None,
        "invoice_date": None,
        "currency": None,
        "line_items": None,
        "printed_subtotal": None,
        "tax_lines": None,
        "tax_total": None,
        "adjustments_outside_items": None,
        "total": None,
        "ambiguities": [],
        "illegible_fields": [],
    }
