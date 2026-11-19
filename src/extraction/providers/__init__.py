"""LLM provider adapters and factory.

To add a provider: implement `LLMProvider.complete()` in a new module (map
`LLMRequest` to the vendor SDK and its exceptions to `ProviderError`) and add a
branch here. Nothing else in the pipeline changes.
"""

from __future__ import annotations

from src.config import Settings
from src.extraction.providers.base import LLMProvider
from src.extraction.providers.cache import CachingProvider


def create_provider(settings: Settings, *, use_cache: bool | None = None) -> LLMProvider:
    """Build the configured provider, wrapped in the response cache unless disabled."""
    provider: LLMProvider
    # Provider SDKs are imported lazily so only the configured one needs to be installed.
    if settings.provider == "openai":
        from src.extraction.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(
            settings.model,
            effort=settings.effort or None,
            timeout_s=settings.request_timeout_s,
            image_detail=settings.image_detail,
            logprobs=settings.token_confidence,
        )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER {settings.provider!r}; supported: openai")
    cache = settings.use_cache if use_cache is None else use_cache
    return CachingProvider(provider, settings.cache_dir) if cache else provider
