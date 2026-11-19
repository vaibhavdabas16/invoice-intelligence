"""Provider adapters: request mapping, response/usage parsing and error mapping, with a fake SDK client."""

from types import SimpleNamespace

import httpx2
import openai
import pytest

from src.errors import ErrorCategory, ProviderError
from src.extraction.prompts import RESPONSE_JSON_SCHEMA
from src.extraction.providers.base import ChatMessage, ImagePart, LLMRequest, TextPart
from src.extraction.providers.openai_provider import OpenAIProvider

REQ = LLMRequest(
    system="sys",
    messages=(
        ChatMessage("user", (TextPart("Page 1"), ImagePart(b"\x89PNG fake", "image/png"))),
        ChatMessage("assistant", (TextPart('{"prev": 1}'),)),
        ChatMessage("user", (TextPart("fix it"),)),
    ),
    json_schema=RESPONSE_JSON_SCHEMA,
    max_output_tokens=1000,
)


def _response(status="completed", text='{"ok": true}', reason=None, refusal=False):
    content = [SimpleNamespace(type="refusal", refusal="no")] if refusal else [SimpleNamespace(type="output_text")]
    return SimpleNamespace(
        id="resp_1",
        model="gpt-6-luna-2026-06-01",
        status=status,
        error=None,
        incomplete_details=SimpleNamespace(reason=reason) if reason else None,
        output=[SimpleNamespace(type="message", content=content)],
        output_text=text,
        usage=SimpleNamespace(
            input_tokens=2500,
            output_tokens=400,
            input_tokens_details=SimpleNamespace(cached_tokens=500),
            output_tokens_details=SimpleNamespace(reasoning_tokens=150),
        ),
    )


class FakeClient:
    def __init__(self, result):
        self.calls = []
        self.result = result
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_request_mapping_and_usage_parsing():
    client = FakeClient(_response())
    resp = OpenAIProvider("gpt-6-luna", effort="low", client=client).complete(REQ)
    call = client.calls[0]
    assert call["instructions"] == "sys" and call["store"] is False
    assert call["reasoning"] == {"effort": "low"}
    assert call["text"]["format"]["strict"] is True and call["text"]["format"]["type"] == "json_schema"
    user = call["input"][0]["content"]
    assert user[0] == {"type": "input_text", "text": "Page 1"}
    assert user[1]["type"] == "input_image" and user[1]["image_url"].startswith("data:image/png;base64,")
    assert user[1]["detail"] == "high"
    assert call["input"][1] == {"role": "assistant", "content": '{"prev": 1}'}
    # cached tokens are split out of input_tokens so they are priced at the cached rate
    assert resp.usage.input_tokens == 2000 and resp.usage.cache_read_input_tokens == 500
    assert resp.usage.output_tokens == 400 and resp.extra["reasoning_tokens"] == 150
    assert resp.stop_reason == "end_turn" and resp.text == '{"ok": true}'


def test_truncated_response_reported_as_max_tokens():
    resp = OpenAIProvider("m", client=FakeClient(_response("incomplete", reason="max_output_tokens"))).complete(REQ)
    assert resp.stop_reason == "max_tokens"


def test_refusal_raises():
    with pytest.raises(ProviderError) as exc:
        OpenAIProvider("m", client=FakeClient(_response(refusal=True))).complete(REQ)
    assert exc.value.category is ErrorCategory.PROVIDER_REFUSAL


_REQ = httpx2.Request("POST", "https://api.openai.com/v1/responses")


def _status_error(cls, status, code=None, headers=None):
    body = {"error": {"code": code}} if code else None
    err = cls("msg", response=httpx2.Response(status, request=_REQ, headers=headers or {}), body=body)
    return err


@pytest.mark.parametrize(
    "exc, category, retryable",
    [
        (openai.APITimeoutError(request=_REQ), ErrorCategory.PROVIDER_TIMEOUT, True),
        (openai.APIConnectionError(request=_REQ), ErrorCategory.PROVIDER_ERROR, True),
        (_status_error(openai.AuthenticationError, 401), ErrorCategory.PROVIDER_AUTH, False),
        (_status_error(openai.RateLimitError, 429), ErrorCategory.PROVIDER_ERROR, True),
        (_status_error(openai.RateLimitError, 429, code="insufficient_quota"), ErrorCategory.PROVIDER_ERROR, False),
        (_status_error(openai.InternalServerError, 503), ErrorCategory.PROVIDER_ERROR, True),
        (_status_error(openai.BadRequestError, 400), ErrorCategory.PROVIDER_ERROR, False),
    ],
)
def test_error_mapping(exc, category, retryable):
    with pytest.raises(ProviderError) as raised:
        OpenAIProvider("m", client=FakeClient(exc)).complete(REQ)
    assert raised.value.category is category and raised.value.retryable is retryable


def test_rate_limit_retry_after_parsed():
    exc = _status_error(openai.RateLimitError, 429, headers={"retry-after": "7"})
    with pytest.raises(ProviderError) as raised:
        OpenAIProvider("m", client=FakeClient(exc)).complete(REQ)
    assert raised.value.retry_after_s == 7.0


def test_response_schema_is_strict_mode_compatible():
    """OpenAI strict mode: every object lists all properties as required and forbids extras."""

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(RESPONSE_JSON_SCHEMA)
