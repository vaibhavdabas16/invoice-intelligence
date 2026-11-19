"""Run the pipeline over many documents with a small thread pool.

Order of results always matches the input order, so reports are stable.
Each document is isolated: a failure becomes a failed `PipelineResult`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.config import Settings
from src.extraction.providers.base import LLMProvider
from src.pipeline import PipelineResult, process_path

log = logging.getLogger(__name__)


def run_many(
    paths: Sequence[Path],
    provider: LLMProvider,
    settings: Settings,
    *,
    workers: int = 4,
    on_done: Callable[[PipelineResult], None] | None = None,
) -> list[PipelineResult]:
    def work(path: Path) -> PipelineResult:
        result = process_path(path, provider, settings)
        log.info(
            "%-12s %-12s calls=%d in=%d out=%d review=%s%s",
            result.source_name,
            result.status,
            result.api_calls,
            result.input_tokens,
            result.output_tokens,
            result.needs_review,
            f" error={result.error['category']}" if result.error else "",
        )
        if on_done:
            on_done(result)
        return result

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(work, paths))
