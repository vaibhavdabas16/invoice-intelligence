"""LLM extraction with validation-driven repair.

Flow per document:
  1. One extraction call (classification + extraction in the same call: a
     separate classification call would double latency for no measured gain,
     since the model has to read the page either way).
  2. Deterministic checks: JSON parse -> Pydantic schema -> financial checks.
  3. If parsing/schema failed or the totals do not reconcile, at most
     `max_repair_attempts` follow-up calls that show the model the specific
     failures and ask it to re-read the document.
  4. Keep the best attempt (furthest through the checks; ties keep the
     earlier answer, since a repair that still fails has not shown it is better).

Transient provider failures are retried separately with backoff (see
`call_with_retries`); they never count as repair attempts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.config import Settings
from src.errors import ErrorCategory, PipelineError
from src.extraction.parsing import OutputParseError, OutputSchemaError, parse_json_object, validate_extraction
from src.extraction.prompts import (
    PROMPT_VERSION,
    RESPONSE_JSON_SCHEMA,
    SYSTEM_PROMPT,
    USER_INSTRUCTION,
    build_page_hints,
    build_repair_message,
)
from src.extraction.providers.base import (
    ChatMessage,
    ContentPart,
    ImagePart,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    TextPart,
    call_with_retries,
)
from src.preprocessing.pages import PreparedDocument
from src.schema import InvoiceExtraction, LLMExtraction
from src.validation.financial import FinancialCheck, FinancialStatus, check_financials

log = logging.getLogger(__name__)

# Stages an attempt can reach; higher is better.
STAGE_PARSE_FAILED = 0
STAGE_SCHEMA_FAILED = 1
STAGE_FINANCIAL_FAILED = 2
STAGE_PASSED = 3


@dataclass
class AttemptRecord:
    kind: str  # "initial" | "repair"
    stage: int
    problems: list[str]
    response: LLMResponse
    raw_text: str

    def to_dict(self) -> dict[str, Any]:
        r = self.response
        return {
            "kind": self.kind,
            "stage": ["parse_failed", "schema_failed", "financial_failed", "passed"][self.stage],
            "problems": self.problems,
            "provider": r.provider,
            "model": r.model,
            "usage": r.usage.to_dict(),
            "reasoning_tokens": r.extra.get("reasoning_tokens"),
            "stop_reason": r.stop_reason,
            "request_id": r.request_id,
            "latency_s": r.latency_s,
            "transient_retries": r.transient_retries,
            "from_cache": r.from_cache,
        }


@dataclass
class _Evaluated:
    stage: int
    problems: list[str]
    llm: LLMExtraction | None = None
    invoice: InvoiceExtraction | None = None
    financial: FinancialCheck | None = None
    response: LLMResponse | None = None


@dataclass
class ExtractionResult:
    llm: LLMExtraction
    invoice: InvoiceExtraction
    financial: FinancialCheck
    attempts: list[AttemptRecord] = field(default_factory=list)
    response: LLMResponse | None = None  # the response the kept answer came from

    @property
    def repair_attempted(self) -> bool:
        return any(a.kind.endswith("repair") for a in self.attempts)


class ExtractionFailed(PipelineError):
    """Model output never passed parse/schema checks. Carries attempts so cost is still logged."""

    def __init__(self, category: ErrorCategory, message: str, attempts: list[AttemptRecord]) -> None:
        super().__init__(category, message)
        self.attempts = attempts


class ProviderFailed(PipelineError):
    def __init__(self, error: PipelineError, attempts: list[AttemptRecord]) -> None:
        super().__init__(error.category, error.message)
        self.attempts = attempts


class InvoiceExtractor:
    def __init__(self, provider: LLMProvider, settings: Settings, *, sleep: Any = None) -> None:
        self.provider = provider
        self.settings = settings
        self._sleep = sleep

    # -- request construction -------------------------------------------------

    def build_document_parts(self, doc: PreparedDocument) -> list[ContentPart]:
        mode = self.settings.input_mode
        use_text_only = mode == "text" and all(p.text_layer for p in doc.pages)
        parts: list[ContentPart] = []
        n = len(doc.pages)
        for page in doc.pages:
            label = f"Page {page.index + 1} of {n}"
            if not use_text_only:
                parts.append(TextPart(f"{label} (image):"))
                parts.append(ImagePart(page.image_bytes, page.media_type))
            if page.text_layer and mode in ("hybrid", "text"):
                parts.append(TextPart(f"{label} - PDF text layer:\n<text_layer>\n{page.text_layer}\n</text_layer>"))
        hints = build_page_hints(doc.warnings)
        if hints:
            parts.append(TextPart(hints))
        parts.append(TextPart(USER_INSTRUCTION))
        return parts

    def _request(self, messages: list[ChatMessage], sample_id: int = 0) -> LLMRequest:
        return LLMRequest(
            system=SYSTEM_PROMPT,
            messages=tuple(messages),
            json_schema=RESPONSE_JSON_SCHEMA,
            max_output_tokens=self.settings.max_output_tokens,
            sample_id=sample_id,
        )

    def _call(self, request: LLMRequest) -> LLMResponse:
        kwargs: dict[str, Any] = {}
        if self._sleep is not None:
            kwargs["sleep"] = self._sleep
        return call_with_retries(
            lambda: self.provider.complete(request),
            max_retries=self.settings.max_transient_retries,
            base_delay_s=self.settings.retry_base_delay_s,
            **kwargs,
        )

    # -- checking -------------------------------------------------------------

    def _evaluate(self, response: LLMResponse) -> _Evaluated:
        if response.stop_reason == "max_tokens":
            return _Evaluated(STAGE_PARSE_FAILED, ["response was truncated at the output token limit"])
        try:
            data = parse_json_object(response.text)
        except OutputParseError as exc:
            return _Evaluated(STAGE_PARSE_FAILED, [str(exc)])
        try:
            llm, invoice = validate_extraction(data)
        except OutputSchemaError as exc:
            return _Evaluated(STAGE_SCHEMA_FAILED, exc.problems)
        financial = check_financials(llm, tolerance=self.settings.amount_tolerance)
        if financial.status is FinancialStatus.MISMATCH:
            return _Evaluated(STAGE_FINANCIAL_FAILED, list(financial.messages), llm, invoice, financial)
        return _Evaluated(STAGE_PASSED, [], llm, invoice, financial)

    # -- main entry -------------------------------------------------------------

    def extract(self, doc: PreparedDocument, *, sample_id: int = 0, kind_prefix: str = "") -> ExtractionResult:
        first_user = ChatMessage("user", tuple(self.build_document_parts(doc)))
        messages = [first_user]
        attempts: list[AttemptRecord] = []
        evaluated: list[_Evaluated] = []

        for attempt_no in range(1 + max(0, self.settings.max_repair_attempts)):
            try:
                response = self._call(self._request(messages, sample_id))
            except PipelineError as exc:
                if evaluated and evaluated[-1].stage >= STAGE_FINANCIAL_FAILED:
                    # A repair call failed at the provider, but we already hold a schema-valid answer.
                    log.warning("repair call failed (%s); keeping earlier answer", exc.message)
                    break
                raise ProviderFailed(exc, attempts) from exc
            result = self._evaluate(response)
            result.response = response
            attempts.append(
                AttemptRecord(
                    kind=kind_prefix + ("initial" if attempt_no == 0 else "repair"),
                    stage=result.stage,
                    problems=result.problems,
                    response=response,
                    raw_text=response.text,
                )
            )
            evaluated.append(result)
            if result.stage == STAGE_PASSED:
                break
            # Prepare a repair turn for the next iteration (if any remain).
            messages = [
                first_user,
                ChatMessage("assistant", (TextPart(response.text.strip() or "(empty response)"),)),
                ChatMessage("user", (TextPart(build_repair_message(response.text.strip(), result.problems)),)),
            ]

        best = max(evaluated, key=lambda e: e.stage)  # max() returns the first maximal element: ties keep earlier
        if best.llm is None or best.invoice is None or best.financial is None:
            category = (
                ErrorCategory.EXTRACTION_PARSE_ERROR
                if best.stage == STAGE_PARSE_FAILED
                else ErrorCategory.SCHEMA_VALIDATION_ERROR
            )
            raise ExtractionFailed(category, "; ".join(best.problems), attempts)
        return ExtractionResult(
            llm=best.llm, invoice=best.invoice, financial=best.financial, attempts=attempts, response=best.response
        )


def prompt_metadata() -> dict[str, str]:
    return {"prompt_version": PROMPT_VERSION}
