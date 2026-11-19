"""Failure handling end to end, through the REAL OpenAI adapter with a fake SDK client.

The retry/repair paths never fired in 184 real API calls, so these tests are
the only evidence that they work. Each test drives `process_document` ->
`InvoiceExtractor` -> `call_with_retries` -> `OpenAIProvider` -> fake client,
so error mapping, retry limits, backoff, repair and review are exercised
together. No network, no API spend.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx2
import openai
import pytest

from src.extraction.providers import base as base_module
from src.extraction.providers.openai_provider import OpenAIProvider
from src.pipeline import process_document
from tests.fakes import llm_invoice

_REQ = httpx2.Request("POST", "https://api.openai.com/v1/responses")


def _ok(payload: Any) -> SimpleNamespace:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(
        id="resp_x",
        model="gpt-6-luna",
        status="completed",
        error=None,
        incomplete_details=None,
        output=[SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text")])],
        output_text=text,
        usage=SimpleNamespace(
            input_tokens=3000,
            output_tokens=300,
            input_tokens_details=SimpleNamespace(cached_tokens=0),
            output_tokens_details=SimpleNamespace(reasoning_tokens=0),
        ),
    )


def _status(cls: type, code: int, headers: dict[str, str] | None = None, body: Any = None) -> Exception:
    return cls("boom", response=httpx2.Response(code, request=_REQ, headers=headers or {}), body=body)


def http_429(**kw: Any) -> Exception:
    return _status(openai.RateLimitError, 429, **kw)


def http_500() -> Exception:
    return _status(openai.InternalServerError, 500)


def timeout() -> Exception:
    return openai.APITimeoutError(request=_REQ)


class FakeOpenAIClient:
    """Pops one scripted outcome per `responses.create` call."""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.calls = 0
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **_: Any) -> Any:
        self.calls += 1
        if not self.script:
            raise AssertionError("client called more times than scripted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return _ok(item)


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []
    monkeypatch.setattr(base_module.time, "sleep", recorded.append)
    monkeypatch.setattr(base_module.random, "random", lambda: 0.5)  # jitter factor exactly 1.0
    return recorded


def run(doc: bytes, script: list[Any], settings: Any, **overrides: Any):
    client = FakeOpenAIClient(script)
    provider = OpenAIProvider("gpt-6-luna", client=client)
    result = process_document(doc, "inv.pdf", provider, settings.with_overrides(**overrides))
    return result, client


# --- transient failures ----------------------------------------------------------------------


def test_http_429_retried_with_exponential_backoff(digital_pdf_bytes, settings, sleeps):
    r, client = run(digital_pdf_bytes, [http_429(), http_429(), llm_invoice()], settings, retry_base_delay_s=1.0)
    assert r.status == "extracted" and client.calls == 3
    assert sleeps == [1.0, 2.0]  # base * 2**attempt, jitter pinned to 1.0
    assert r.transient_retries == 2 and r.api_calls == 1  # retries are not billed as extra responses


def test_http_429_retry_after_header_overrides_shorter_backoff(digital_pdf_bytes, settings, sleeps):
    r, _ = run(
        digital_pdf_bytes, [http_429(headers={"retry-after": "20"}), llm_invoice()], settings, retry_base_delay_s=1.0
    )
    assert r.status == "extracted" and sleeps == [20.0]


def test_http_429_insufficient_quota_is_not_retried(digital_pdf_bytes, settings, sleeps):
    quota = http_429(body={"error": {"code": "insufficient_quota"}})
    r, client = run(digital_pdf_bytes, [quota, llm_invoice()], settings)
    assert r.status == "failed" and r.error["category"] == "provider_error"
    assert client.calls == 1 and sleeps == []


def test_http_500_retried_then_succeeds(digital_pdf_bytes, settings, sleeps):
    r, client = run(digital_pdf_bytes, [http_500(), llm_invoice()], settings, retry_base_delay_s=0.5)
    assert r.status == "extracted" and client.calls == 2 and sleeps == [0.5]


def test_timeout_retried_then_succeeds(digital_pdf_bytes, settings, sleeps):
    r, client = run(digital_pdf_bytes, [timeout(), llm_invoice()], settings)
    assert r.status == "extracted" and client.calls == 2


def test_max_retry_exhaustion_is_bounded_and_categorised(digital_pdf_bytes, settings, sleeps):
    r, client = run(digital_pdf_bytes, [timeout()] * 10, settings, max_transient_retries=3, retry_base_delay_s=1.0)
    assert client.calls == 4  # 1 attempt + 3 retries, never more
    assert sleeps == [1.0, 2.0, 4.0]
    assert r.status == "failed" and r.error["category"] == "provider_timeout"
    assert "gave up after 4 attempts" in r.error["message"] and r.error["hint"]
    assert r.needs_review and r.api_calls == 0 and r.cost_usd == 0  # nothing billed, nothing invented
    assert "error" in r.output_json() and "is_invoice" not in r.output_json()


def test_500_exhaustion_does_not_trigger_repair_calls(digital_pdf_bytes, settings, sleeps):
    r, client = run(digital_pdf_bytes, [http_500()] * 10, settings, max_transient_retries=2)
    assert client.calls == 3 and r.error["category"] == "provider_error"


def test_auth_error_propagates_immediately(digital_pdf_bytes, settings, sleeps):
    r, client = run(digital_pdf_bytes, [_status(openai.AuthenticationError, 401), llm_invoice()], settings)
    assert client.calls == 1 and sleeps == [] and r.error["category"] == "provider_auth"


# --- bad model output --------------------------------------------------------------------------


def test_malformed_json_repaired_once(digital_pdf_bytes, settings, sleeps):
    r, client = run(digital_pdf_bytes, ['{"is_invoice": true, "vendor": "Ac', llm_invoice()], settings)
    assert r.status == "extracted" and client.calls == 2
    assert [a.kind for a in r.attempts] == ["initial", "repair"] and r.needs_review


def test_malformed_json_twice_fails_without_third_call(digital_pdf_bytes, settings, sleeps):
    r, client = run(digital_pdf_bytes, ["nope", "still nope", llm_invoice()], settings)
    assert client.calls == 2 and r.status == "failed" and r.error["category"] == "extraction_parse_error"
    assert r.cost_usd > 0  # both failed responses were billed and are reported


def test_invalid_schema_repaired(digital_pdf_bytes, settings, sleeps):
    bad = llm_invoice(currency="Rupees", invoice_date="14/03/2026")
    r, client = run(digital_pdf_bytes, [bad, llm_invoice()], settings)
    assert r.status == "extracted" and client.calls == 2
    assert r.output_json()["currency"] == "INR"


def test_missing_fields_rejected_then_failed(digital_pdf_bytes, settings, sleeps):
    missing = {k: v for k, v in llm_invoice().items() if k not in ("total", "vendor")}
    r, client = run(digital_pdf_bytes, [missing, missing], settings)
    assert client.calls == 2 and r.status == "failed" and r.error["category"] == "schema_validation_error"
    assert "total" in r.error["message"] and "vendor" in r.error["message"]


def test_financial_mismatch_repaired_then_reviewed(digital_pdf_bytes, settings, sleeps):
    r, client = run(digital_pdf_bytes, [llm_invoice(total=3876.0), llm_invoice()], settings)
    assert client.calls == 2 and r.financial.passed
    assert r.needs_review and any("repair" in x for x in r.review.reasons)


def test_persistent_financial_mismatch_is_flagged_not_fixed(digital_pdf_bytes, settings, sleeps):
    r, client = run(digital_pdf_bytes, [llm_invoice(total=3876.0)] * 5, settings)
    assert client.calls == 2  # one repair maximum
    assert r.status == "extracted" and r.output_json()["total"] == 3876.0  # printed value kept, not "fixed"
    assert r.needs_review and any("do not reconcile" in x for x in r.review.reasons)


def test_repair_call_hitting_provider_error_keeps_valid_first_answer(digital_pdf_bytes, settings, sleeps):
    r, client = run(digital_pdf_bytes, [llm_invoice(total=3876.0), timeout(), timeout(), timeout()], settings)
    assert r.status == "extracted" and r.output_json()["total"] == 3876.0 and r.needs_review
    assert client.calls == 4  # initial + repair attempt with 2 retries, then stop


def test_truncated_output_triggers_repair(digital_pdf_bytes, settings, sleeps):
    truncated = _ok("{}")
    truncated.status, truncated.incomplete_details = "incomplete", SimpleNamespace(reason="max_output_tokens")

    class Client(FakeOpenAIClient):
        def _create(self, **kw: Any) -> Any:
            self.calls += 1
            return truncated if self.calls == 1 else _ok(llm_invoice())

    provider = OpenAIProvider("gpt-6-luna", client=(client := Client([])))
    r = process_document(digital_pdf_bytes, "inv.pdf", provider, settings)
    assert r.status == "extracted" and client.calls == 2
    assert "truncated" in r.attempts[0].problems[0]


# --- no unnecessary duplicate calls -------------------------------------------------------------


def test_valid_first_answer_makes_exactly_one_call(digital_pdf_bytes, settings, sleeps):
    r, client = run(digital_pdf_bytes, [llm_invoice()], settings, verify_scans=True)
    assert client.calls == 1 and sleeps == []  # digital PDF: no re-read, no repair


def test_non_invoice_makes_one_call_even_for_scans(scanned_pdf_bytes, settings, sleeps):
    from tests.fakes import llm_non_invoice

    r, client = run(scanned_pdf_bytes, [llm_non_invoice()], settings, verify_scans=True)
    assert client.calls == 1 and r.status == "not_invoice"


def test_bad_input_makes_no_call(settings, sleeps):
    client = FakeOpenAIClient([])
    r = process_document(b"%PDF-1.4 garbage", "x.pdf", OpenAIProvider("m", client=client), settings)
    assert client.calls == 0 and r.error["category"] == "corrupt_file"


def test_cache_prevents_duplicate_calls_across_runs(digital_pdf_bytes, settings, sleeps, tmp_path):
    from src.extraction.providers.cache import CachingProvider

    client = FakeOpenAIClient([llm_invoice()])
    provider = CachingProvider(OpenAIProvider("gpt-6-luna", client=client), tmp_path / "c")
    for _ in range(3):
        r = process_document(digital_pdf_bytes, "inv.pdf", provider, settings)
        assert r.status == "extracted"
    assert client.calls == 1
