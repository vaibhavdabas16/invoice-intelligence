"""On-disk response cache.

Why: LLM output is not bit-for-bit deterministic and every call costs money.
Caching by a hash of (provider config, prompt, schema, exact input bytes)
makes re-running the eval free and reproducible. Any change to the prompt,
model, effort or document produces a new key, so stale answers are never
reused. Pass `--no-cache` to force fresh calls.

Cached entries keep the *original* token usage so cost reports reflect what
the extraction actually cost when it was made.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.extraction.providers.base import ImagePart, LLMProvider, LLMRequest, LLMResponse, TextPart, TokenUsage


def request_fingerprint(provider: LLMProvider, request: LLMRequest, extra: dict[str, Any] | None = None) -> str:
    def part_repr(p: ImagePart | TextPart) -> dict[str, str]:
        if isinstance(p, ImagePart):
            return {"image": hashlib.sha256(p.data).hexdigest(), "media_type": p.media_type}
        return {"text": p.text}

    payload = {
        "provider": provider.cache_identity(),
        "system": request.system,
        "schema": request.json_schema,
        "max_output_tokens": request.max_output_tokens,
        "messages": [{"role": m.role, "parts": [part_repr(p) for p in m.parts]} for m in request.messages],
        "extra": extra or {},
        "sample_id": request.sample_id,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class CachingProvider(LLMProvider):
    def __init__(self, inner: LLMProvider, cache_dir: Path) -> None:
        self.inner = inner
        self.name = inner.name
        self.model = inner.model
        self.cache_dir = cache_dir

    def cache_identity(self) -> dict[str, Any]:
        return self.inner.cache_identity()

    def _path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def complete(self, request: LLMRequest) -> LLMResponse:
        key = request_fingerprint(self.inner, request)
        path = self._path(key)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return LLMResponse(
                    text=data["text"],
                    usage=TokenUsage(**data["usage"]),
                    provider=data["provider"],
                    model=data["model"],
                    stop_reason=data.get("stop_reason"),
                    request_id=data.get("request_id"),
                    latency_s=data.get("latency_s", 0.0),
                    transient_retries=0,
                    from_cache=True,
                    extra=data.get("extra", {}),
                )
            except (json.JSONDecodeError, KeyError, TypeError):
                path.unlink(missing_ok=True)  # corrupt cache entry: fall through to a real call

        response = self.inner.complete(request)
        # Only cache complete answers; truncated output should be retried fresh next time.
        if response.stop_reason in (None, "end_turn", "stop_sequence"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "text": response.text,
                        "usage": response.usage.to_dict(),
                        "provider": response.provider,
                        "model": response.model,
                        "stop_reason": response.stop_reason,
                        "request_id": response.request_id,
                        "latency_s": response.latency_s,
                        "extra": response.extra,
                    },
                    indent=1,
                ),
                encoding="utf-8",
            )
        return response
