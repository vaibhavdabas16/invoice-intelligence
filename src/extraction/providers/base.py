"""Provider-neutral LLM interface.

The extractor only talks to `LLMProvider.complete(LLMRequest) -> LLMResponse`.
Adding another vendor means writing one adapter that maps these dataclasses to
its SDK and its exceptions to `ProviderError`; nothing else changes.
"""

from __future__ import annotations

import logging
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from src.errors import ProviderError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImagePart:
    data: bytes
    media_type: str


@dataclass(frozen=True)
class TextPart:
    text: str


ContentPart = ImagePart | TextPart


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["user", "assistant"]
    parts: tuple[ContentPart, ...]


@dataclass(frozen=True)
class LLMRequest:
    system: str
    messages: tuple[ChatMessage, ...]
    json_schema: dict[str, Any] | None
    max_output_tokens: int
    # Distinguishes deliberate independent samples of the same input (self-consistency check)
    # so the local response cache does not return the first sample again. Not sent to the API.
    sample_id: int = 0


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
        }


@dataclass
class LLMResponse:
    text: str
    usage: TokenUsage
    provider: str
    model: str
    stop_reason: str | None
    request_id: str | None
    latency_s: float
    transient_retries: int = 0
    # True when served from the local response cache: no API call, no spend in *this* run.
    from_cache: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    name: str
    model: str

    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse: ...

    def cache_identity(self) -> dict[str, Any]:
        """Everything about the provider configuration that changes the output."""
        return {"provider": self.name, "model": self.model}


def call_with_retries(
    fn: Callable[[], LLMResponse],
    *,
    max_retries: int,
    base_delay_s: float,
    sleep: Callable[[float], None] | None = None,
) -> LLMResponse:
    """Run `fn`, retrying only `ProviderError(retryable=True)` with exponential backoff + jitter.

    Bounded: at most `max_retries` extra attempts. Non-retryable errors (auth,
    bad request, refusal) propagate immediately.
    """
    attempt = 0
    while True:
        try:
            response = fn()
            response.transient_retries = attempt
            return response
        except ProviderError as exc:
            if not exc.retryable or attempt >= max_retries:
                if exc.retryable:
                    exc.message = f"{exc.message} (gave up after {attempt + 1} attempts)"
                raise
            delay = base_delay_s * (2**attempt) * (0.75 + random.random() * 0.5)
            retry_after = getattr(exc, "retry_after_s", None)
            if isinstance(retry_after, (int, float)) and retry_after > delay:
                delay = float(retry_after)
            log.warning(
                "transient provider error (%s); retry %d/%d in %.1fs", exc.message, attempt + 1, max_retries, delay
            )
            (sleep or time.sleep)(delay)  # resolved at call time so tests can patch time.sleep
            attempt += 1
