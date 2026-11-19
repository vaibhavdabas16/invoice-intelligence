"""Extractor + pipeline behaviour with a scripted provider (no network)."""

import json

import pytest

from src.errors import ErrorCategory, ProviderError
from src.extraction.providers.base import LLMResponse, TokenUsage, call_with_retries
from src.extraction.providers.cache import CachingProvider
from src.pipeline import process_document
from tests.fakes import ScriptedProvider, llm_invoice, llm_non_invoice


def _timeout() -> ProviderError:
    return ProviderError(ErrorCategory.PROVIDER_TIMEOUT, "timed out", retryable=True)


def test_happy_path_single_call(digital_pdf_bytes, settings):
    provider = ScriptedProvider([llm_invoice()])
    r = process_document(digital_pdf_bytes, "inv.pdf", provider, settings)
    assert r.status == "extracted"
    assert r.api_calls == 1 and r.needs_review is False
    assert r.output_json()["total"] == 3776.0
    # The request carried the page image and the text layer.
    parts = provider.requests[0].messages[0].parts
    assert any(type(p).__name__ == "ImagePart" for p in parts)
    assert any("text_layer" in getattr(p, "text", "") for p in parts)


def test_image_mode_omits_text_layer(digital_pdf_bytes, settings):
    provider = ScriptedProvider([llm_invoice()])
    process_document(digital_pdf_bytes, "inv.pdf", provider, settings.with_overrides(input_mode="image"))
    assert not any("text_layer" in getattr(p, "text", "") for p in provider.requests[0].messages[0].parts)


def test_non_invoice_classification(digital_pdf_bytes, settings):
    r = process_document(digital_pdf_bytes, "po.pdf", ScriptedProvider([llm_non_invoice()]), settings)
    assert r.status == "not_invoice"
    out = r.output_json()
    assert out["is_invoice"] is False and all(out[k] is None for k in out if k != "is_invoice")
    assert r.needs_review is False


def test_financial_mismatch_triggers_one_repair_that_fixes_it(digital_pdf_bytes, settings):
    provider = ScriptedProvider([llm_invoice(total=3876.0), llm_invoice()])
    r = process_document(digital_pdf_bytes, "inv.pdf", provider, settings)
    assert r.api_calls == 2
    assert r.output_json()["total"] == 3776.0
    assert r.financial.passed
    # Repair turn quoted the concrete failure and the previous answer.
    repair_text = provider.requests[1].messages[-1].parts[0].text
    assert "3876.00" in repair_text and "Do NOT alter values" in repair_text
    assert r.needs_review  # a repair is itself a review signal


def test_persistent_mismatch_keeps_first_answer_and_flags(digital_pdf_bytes, settings):
    provider = ScriptedProvider([llm_invoice(total=3876.0), llm_invoice(total=3999.0)])
    r = process_document(digital_pdf_bytes, "inv.pdf", provider, settings)
    assert r.status == "extracted"
    assert r.output_json()["total"] == 3876.0  # tie -> keep the original; nothing silently "fixed"
    assert not r.financial.passed and r.needs_review
    assert any("do not reconcile" in reason for reason in r.review.reasons)


def test_repair_limit_is_respected(digital_pdf_bytes, settings):
    provider = ScriptedProvider(["not json"] * 5)
    r = process_document(digital_pdf_bytes, "inv.pdf", provider, settings.with_overrides(max_repair_attempts=2))
    assert r.api_calls == 3  # 1 initial + 2 repairs, then stop
    assert r.status == "failed" and r.error["category"] == "extraction_parse_error"
    assert len(provider.script) == 2


def test_malformed_json_repaired(digital_pdf_bytes, settings):
    provider = ScriptedProvider(['{"is_invoice": tru', llm_invoice()])
    r = process_document(digital_pdf_bytes, "inv.pdf", provider, settings)
    assert r.status == "extracted" and r.api_calls == 2


def test_schema_violation_repaired(digital_pdf_bytes, settings):
    provider = ScriptedProvider([llm_invoice(invoice_date="14/03/2026"), llm_invoice()])
    r = process_document(digital_pdf_bytes, "inv.pdf", provider, settings)
    assert r.output_json()["invoice_date"] == "2026-03-14"
    assert "invoice_date" in provider.requests[1].messages[-1].parts[0].text


def test_schema_violation_unrepaired_fails_with_category(digital_pdf_bytes, settings):
    bad = llm_invoice(currency="rupees")
    r = process_document(digital_pdf_bytes, "inv.pdf", ScriptedProvider([bad, bad]), settings)
    assert r.status == "failed" and r.error["category"] == "schema_validation_error"
    assert r.api_calls == 2 and r.cost_usd > 0  # failed attempts still cost money and are logged
    assert "error" in r.output_json()


def test_timeout_retried_then_succeeds(digital_pdf_bytes, settings):
    provider = ScriptedProvider([_timeout(), _timeout(), llm_invoice()])
    r = process_document(digital_pdf_bytes, "inv.pdf", provider, settings)
    assert r.status == "extracted"
    assert r.transient_retries == 2 and r.api_calls == 1


def test_timeout_retries_exhausted_fails_gracefully(digital_pdf_bytes, settings):
    provider = ScriptedProvider([_timeout()] * 10)
    r = process_document(digital_pdf_bytes, "inv.pdf", provider, settings)
    assert r.status == "failed" and r.error["category"] == "provider_timeout"
    assert "gave up after 3 attempts" in r.error["message"]
    assert len(provider.script) == 7  # 1 try + 2 retries consumed, no more


def test_auth_error_not_retried(digital_pdf_bytes, settings):
    provider = ScriptedProvider([ProviderError(ErrorCategory.PROVIDER_AUTH, "bad key"), llm_invoice()])
    r = process_document(digital_pdf_bytes, "inv.pdf", provider, settings)
    assert r.error["category"] == "provider_auth" and len(provider.script) == 1


def test_bad_input_fails_without_api_call(blank_png_bytes, settings):
    provider = ScriptedProvider([])
    r = process_document(blank_png_bytes, "blank.png", provider, settings)
    assert r.status == "failed" and r.error["category"] == "unreadable_document"
    assert r.api_calls == 0 and provider.requests == []
    assert r.error["hint"]


def test_empty_file_fails_gracefully(settings):
    r = process_document(b"", "empty.pdf", ScriptedProvider([]), settings)
    assert r.error["category"] == "empty_file"


def test_unexpected_exception_is_contained(digital_pdf_bytes, settings):
    r = process_document(digital_pdf_bytes, "inv.pdf", ScriptedProvider([RuntimeError("boom")]), settings)
    assert r.status == "failed" and r.error["category"] == "internal_error"


def test_backoff_is_exponential_and_bounded():
    delays = []
    calls = {"n": 0}

    def fn() -> LLMResponse:
        calls["n"] += 1
        raise _timeout()

    with pytest.raises(ProviderError):
        call_with_retries(fn, max_retries=3, base_delay_s=1.0, sleep=delays.append)
    assert calls["n"] == 4 and len(delays) == 3
    assert delays[0] < delays[1] < delays[2]  # 1, 2, 4 (+/- 25% jitter)
    assert all(0.75 * 2**i <= d <= 1.25 * 2**i for i, d in enumerate(delays))


def test_retry_after_header_respected():
    delays = []
    err = _timeout()
    err.retry_after_s = 30.0
    script = [err]

    def fn() -> LLMResponse:
        if script:
            raise script.pop()
        return LLMResponse("{}", TokenUsage(), "fake", "m", "end_turn", None, 0.0)

    call_with_retries(fn, max_retries=2, base_delay_s=1.0, sleep=delays.append)
    assert delays == [30.0]


def test_cache_serves_repeat_requests_without_calling_provider(digital_pdf_bytes, settings, tmp_path):
    inner = ScriptedProvider([llm_invoice()])
    cached = CachingProvider(inner, tmp_path / "cache")
    r1 = process_document(digital_pdf_bytes, "inv.pdf", cached, settings)
    r2 = process_document(digital_pdf_bytes, "inv.pdf", cached, settings)
    assert len(inner.requests) == 1
    assert r2.attempts[0].response.from_cache and not r1.attempts[0].response.from_cache
    assert r2.input_tokens == r1.input_tokens  # original usage preserved for cost reporting
    assert r2.output_json() == r1.output_json()


def test_metadata_is_json_serialisable(digital_pdf_bytes, settings):
    r = process_document(
        digital_pdf_bytes, "inv.pdf", ScriptedProvider([llm_invoice(total=1.0), llm_invoice()]), settings
    )
    meta = json.loads(json.dumps(r.metadata_json()))
    assert meta["usage"]["api_calls"] == 2
    assert [a["kind"] for a in meta["attempts"]] == ["initial", "repair"]
    assert meta["config"]["prompt_version"]
