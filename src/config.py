"""Runtime configuration.

All settings come from environment variables (optionally loaded from a local
`.env`). Nothing secret is hard-coded; the OpenAI SDK reads `OPENAI_API_KEY` from
the environment itself.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value and value.strip() else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value and value.strip() else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # --- LLM provider -------------------------------------------------------
    provider: str = field(default_factory=lambda: _env_str("LLM_PROVIDER", "openai"))
    model: str = field(default_factory=lambda: _env_str("LLM_MODEL", "gpt-6-luna"))
    # Reasoning effort for models that support it (none | low | medium | high).
    effort: str = field(default_factory=lambda: _env_str("LLM_EFFORT", "low"))
    max_output_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_OUTPUT_TOKENS", 16000))
    request_timeout_s: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT_S", 90.0))
    # Retries for *transient* provider failures (timeouts, 429, 5xx, network).
    max_transient_retries: int = field(default_factory=lambda: _env_int("LLM_MAX_RETRIES", 3))
    retry_base_delay_s: float = field(default_factory=lambda: _env_float("LLM_RETRY_BASE_DELAY_S", 2.0))
    # Validation-aware repair calls when parsing / schema / arithmetic fails.
    max_repair_attempts: int = field(default_factory=lambda: _env_int("LLM_MAX_REPAIR_ATTEMPTS", 1))

    # --- Input handling -----------------------------------------------------
    # hybrid = page images + PDF text layer (when present); image = images only;
    # text = text layer only (falls back to images when there is no text layer).
    input_mode: str = field(default_factory=lambda: _env_str("INPUT_MODE", "hybrid"))
    # OpenAI image detail level (low | high | auto | original); "high" with our own downscaling.
    image_detail: str = field(default_factory=lambda: _env_str("IMAGE_DETAIL", "high"))
    # Upscale small photos/images to max_image_edge_px (PDF pages are always rendered at that size).
    upscale_images: bool = field(default_factory=lambda: _env_bool("UPSCALE_IMAGES", True))
    max_image_edge_px: int = field(default_factory=lambda: _env_int("MAX_IMAGE_EDGE_PX", 1568))
    render_dpi: int = field(default_factory=lambda: _env_int("RENDER_DPI", 150))
    max_pages: int = field(default_factory=lambda: _env_int("MAX_PAGES", 10))
    max_file_mb: float = field(default_factory=lambda: _env_float("MAX_FILE_MB", 20.0))

    # --- Validation ---------------------------------------------------------
    # Token-confidence review signal: request token logprobs (needs LLM_EFFORT=none) and flag any
    # required field whose least likely token has probability below the threshold. The threshold
    # was fixed at 0.5 before measuring; it is a token probability, not a calibrated confidence.
    token_confidence: bool = field(default_factory=lambda: _env_bool("TOKEN_CONFIDENCE", False))
    confidence_threshold: float = field(default_factory=lambda: _env_float("CONFIDENCE_THRESHOLD", 0.5))
    # Self-consistency check: a second independent extraction for documents with no text
    # layer (scans/photos); any disagreement on key fields sets needs_review.
    verify_scans: bool = field(default_factory=lambda: _env_bool("VERIFY_SCANS", True))
    amount_tolerance: float = field(default_factory=lambda: _env_float("AMOUNT_TOLERANCE", 0.02))

    # --- Caching ------------------------------------------------------------
    use_cache: bool = field(default_factory=lambda: _env_bool("LLM_USE_CACHE", True))
    cache_dir: Path = field(
        default_factory=lambda: Path(_env_str("LLM_CACHE_DIR", str(PROJECT_ROOT / ".cache" / "llm")))
    )

    # --- Paths --------------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: Path(_env_str("DATA_DIR", str(PROJECT_ROOT / "data"))))
    outputs_dir: Path = field(default_factory=lambda: Path(_env_str("OUTPUTS_DIR", str(PROJECT_ROOT / "outputs"))))
    reports_dir: Path = field(default_factory=lambda: Path(_env_str("REPORTS_DIR", str(PROJECT_ROOT / "reports"))))

    def with_overrides(self, **kwargs: Any) -> Settings:
        """Return a copy with selected fields replaced (used by CLI flags and tests)."""
        from dataclasses import replace

        return replace(self, **{k: v for k, v in kwargs.items() if v is not None})


def get_settings() -> Settings:
    return Settings()
