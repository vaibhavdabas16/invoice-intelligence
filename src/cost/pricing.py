"""Model pricing (USD per million tokens) and cost computation.

Sources (verified live on 2026-11-19, Standard tier, short context):
* https://developers.openai.com/api/docs/pricing
Prices change; update this table and PRICING_AS_OF together.

What is billed:
* Image inputs are converted to input tokens by the provider; the API's
  reported `input_tokens` already include them, so we never estimate image
  tokens ourselves.
* Reasoning/thinking tokens are included in the reported `output_tokens` and
  billed at the output rate.
* OpenAI reports cached prompt tokens inside `input_tokens`; the adapter moves
  them to `cache_read_input_tokens` so they are priced at the cached rate.
  (Our prompts are mostly unique images, so caching is expected to be ~0.)
* The Batch API (~50% cheaper) is not used: we want
  per-document latency and simple resumability.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.extraction.providers.base import TokenUsage

PRICING_AS_OF = "2026-11-19"
PRICING_SOURCE = "https://developers.openai.com/api/docs/pricing"


@dataclass(frozen=True)
class ModelPrice:
    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float
    cache_write_per_mtok: float


PRICES: dict[str, ModelPrice] = {
    # input, output, cached input, cache write (OpenAI has no cache-write surcharge -> input price)
    "gpt-6-luna": ModelPrice(0.10, 0.50, 0.01, 0.10),
    "gpt-5.6-luna": ModelPrice(0.20, 1.20, 0.02, 0.20),
    "gpt-5.4-nano": ModelPrice(0.20, 1.25, 0.02, 0.20),
    "gpt-5.4-mini": ModelPrice(0.75, 4.50, 0.075, 0.75),
    "gpt-5-nano": ModelPrice(0.05, 0.40, 0.005, 0.05),
    "gpt-5-mini": ModelPrice(0.25, 2.00, 0.025, 0.25),
    "gpt-4.1-mini": ModelPrice(0.40, 1.60, 0.10, 0.40),
    "gpt-4o-mini": ModelPrice(0.15, 0.60, 0.075, 0.15),
}


def price_for(model: str) -> ModelPrice | None:
    if model in PRICES:
        return PRICES[model]
    # Responses may echo a dated snapshot id (e.g. "gpt-6-luna-2026-06-01"); match the longest known prefix.
    for known in sorted(PRICES, key=len, reverse=True):
        if model.startswith(known + "-") or model == known:
            return PRICES[known]
    return None


def cost_usd_without_prompt_cache(usage: TokenUsage, model: str) -> float | None:
    """Upper bound: every input token billed at the full input rate.

    The provider's automatic prompt caching makes repeat requests (re-runs of the
    same documents) much cheaper than a first-time extraction. This figure is what
    a never-seen-before document costs in the worst case.
    """
    full = TokenUsage(
        input_tokens=usage.input_tokens + usage.cache_read_input_tokens + usage.cache_creation_input_tokens,
        output_tokens=usage.output_tokens,
    )
    return cost_usd(full, model)


def cost_usd(usage: TokenUsage, model: str) -> float | None:
    """Cost of one call. None (not 0) when the model's price is unknown, so gaps are visible."""
    price = price_for(model)
    if price is None:
        return None
    return (
        usage.input_tokens * price.input_per_mtok
        + usage.output_tokens * price.output_per_mtok
        + usage.cache_read_input_tokens * price.cache_read_per_mtok
        + usage.cache_creation_input_tokens * price.cache_write_per_mtok
    ) / 1_000_000
