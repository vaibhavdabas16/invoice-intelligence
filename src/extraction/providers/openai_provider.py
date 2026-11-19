"""OpenAI adapter (Responses API).

* Images are sent as base64 data URLs with `detail="high"`. Our preprocessing
  already caps the long edge at MAX_IMAGE_EDGE_PX, so the provider does not
  need to downscale further and we control the pixel (= token) budget.
* Structured outputs: `text.format = json_schema` with `strict: true`, so the
  response is guaranteed to match the schema's shape. Pydantic still checks
  semantics (ISO dates, currency codes, invoice/non-invoice consistency).
* `store=False`: the documents are confidential; we do not ask OpenAI to keep
  response state server-side.
* SDK retries are disabled (`max_retries=0`); our bounded retry loop is the
  only place retries happen, so every retry is counted.
"""

from __future__ import annotations

import base64
import time
from typing import Any

import openai

from src.errors import ErrorCategory, ProviderError
from src.extraction.providers.base import (
    ChatMessage,
    ImagePart,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    TextPart,
    TokenUsage,
)


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(
        self,
        model: str,
        *,
        effort: str | None = "low",
        timeout_s: float = 90.0,
        image_detail: str = "high",
        logprobs: bool = False,
        client: Any | None = None,
    ) -> None:
        if logprobs and effort not in (None, "none"):
            # The API rejects logprobs on reasoning models unless reasoning effort is "none".
            raise ValueError("token logprobs require LLM_EFFORT=none")
        self.model = model
        self.effort = effort
        self.image_detail = image_detail
        self.logprobs = logprobs
        self._client = client or openai.OpenAI(timeout=timeout_s, max_retries=0)

    def cache_identity(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.model,
            "effort": self.effort,
            "image_detail": self.image_detail,
            "logprobs": self.logprobs,
        }

    def _to_input_item(self, message: ChatMessage) -> dict[str, Any]:
        if message.role == "assistant":
            # Prior model output (used in repair turns) is plain text.
            text = "".join(p.text for p in message.parts if isinstance(p, TextPart))
            return {"role": "assistant", "content": text}
        content: list[dict[str, Any]] = []
        for part in message.parts:
            if isinstance(part, ImagePart):
                b64 = base64.standard_b64encode(part.data).decode("ascii")
                content.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:{part.media_type};base64,{b64}",
                        "detail": self.image_detail,
                    }
                )
            elif isinstance(part, TextPart):
                content.append({"type": "input_text", "text": part.text})
        return {"role": "user", "content": content}

    def complete(self, request: LLMRequest) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "instructions": request.system,
            "input": [self._to_input_item(m) for m in request.messages],
            "max_output_tokens": request.max_output_tokens,
            "store": False,
        }
        if self.effort:
            kwargs["reasoning"] = {"effort": self.effort}
        if self.logprobs:
            kwargs["include"] = ["message.output_text.logprobs"]
        if request.json_schema is not None:
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "invoice_extraction",
                    "schema": request.json_schema,
                    "strict": True,
                }
            }

        started = time.perf_counter()
        try:
            response = self._client.responses.create(**kwargs)
        except openai.APITimeoutError as exc:
            raise ProviderError(ErrorCategory.PROVIDER_TIMEOUT, f"request timed out: {exc}", retryable=True) from exc
        except openai.APIConnectionError as exc:
            raise ProviderError(ErrorCategory.PROVIDER_ERROR, f"connection error: {exc}", retryable=True) from exc
        except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
            raise ProviderError(ErrorCategory.PROVIDER_AUTH, f"authentication failed: {exc.message}") from exc
        except openai.RateLimitError as exc:
            # insufficient_quota is a billing problem, not a transient one: do not retry it.
            code = getattr(exc, "code", None)
            if code is None and isinstance(exc.body, dict):
                nested = exc.body.get("error")
                code = nested.get("code") if isinstance(nested, dict) else exc.body.get("code")
            err = ProviderError(
                ErrorCategory.PROVIDER_ERROR,
                f"rate limited ({code or 'rate_limit'}): {exc.message}",
                retryable=code != "insufficient_quota",
            )
            retry_after = exc.response.headers.get("retry-after") if exc.response is not None else None
            if retry_after:
                try:
                    err.retry_after_s = float(retry_after)
                except ValueError:
                    pass
            raise err from exc
        except openai.APIStatusError as exc:
            retryable = exc.status_code >= 500 or exc.status_code in (408, 409)
            raise ProviderError(
                ErrorCategory.PROVIDER_ERROR, f"API error {exc.status_code}: {exc.message}", retryable=retryable
            ) from exc
        latency = time.perf_counter() - started

        # Refusals arrive as a content part of type "refusal" instead of output_text.
        for item in response.output or []:
            for part in getattr(item, "content", None) or []:
                if getattr(part, "type", None) == "refusal":
                    raise ProviderError(
                        ErrorCategory.PROVIDER_REFUSAL, f"model refused: {getattr(part, 'refusal', '')}"
                    )

        stop_reason = "end_turn"
        if response.status == "incomplete":
            reason = getattr(response.incomplete_details, "reason", None)
            stop_reason = "max_tokens" if reason == "max_output_tokens" else f"incomplete:{reason}"
        elif response.status not in (None, "completed"):
            raise ProviderError(
                ErrorCategory.PROVIDER_ERROR, f"response status {response.status}: {response.error}", retryable=True
            )

        usage = response.usage
        cached = getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0) or 0
        reasoning = getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", 0) or 0
        # OpenAI's input_tokens *includes* cached tokens; split them so cost uses the cached rate.
        return LLMResponse(
            text=response.output_text or "",
            usage=TokenUsage(
                input_tokens=(usage.input_tokens if usage else 0) - cached,
                output_tokens=usage.output_tokens if usage else 0,
                cache_read_input_tokens=cached,
                cache_creation_input_tokens=0,
            ),
            provider=self.name,
            model=response.model,
            stop_reason=stop_reason,
            request_id=getattr(response, "_request_id", None) or response.id,
            latency_s=round(latency, 3),
            extra={"reasoning_tokens": reasoning, **_token_logprobs(response)},
        )


def _token_logprobs(response: Any) -> dict[str, Any]:
    """[[token, logprob], ...] for the output text, when the request asked for them."""
    tokens: list[list[Any]] = []
    for item in getattr(response, "output", None) or []:
        for part in getattr(item, "content", None) or []:
            if getattr(part, "type", None) == "output_text":
                for t in getattr(part, "logprobs", None) or []:
                    tokens.append([t.token, float(t.logprob)])
    return {"token_logprobs": tokens} if tokens else {}
