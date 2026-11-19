"""End-to-end processing of one document.

`process_document` never raises for expected failures: every outcome
(success, non-invoice, bad input, provider outage, unparseable model output)
becomes a `PipelineResult` with a categorised error, so one bad file cannot
crash a batch. Unexpected exceptions (bugs) are also captured, labelled
`internal_error`, and logged with a traceback.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import Settings
from src.cost.pricing import cost_usd, cost_usd_without_prompt_cache
from src.errors import PipelineError
from src.extraction.extractor import (
    AttemptRecord,
    ExtractionFailed,
    ExtractionResult,
    InvoiceExtractor,
    ProviderFailed,
    prompt_metadata,
)
from src.extraction.providers.base import LLMProvider
from src.ingestion.loader import RawDocument, load_bytes
from src.preprocessing.pages import PreparedDocument, prepare_document
from src.schema import InvoiceExtraction
from src.validation.confidence import FieldConfidence, field_confidences, low_confidence_reasons
from src.validation.consistency import arbitrate, compare_reads
from src.validation.evidence import collect_evidence, unsupported_reasons
from src.validation.financial import FinancialCheck
from src.validation.review import ReviewDecision, decide_review

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    document_id: str
    source_name: str
    status: str  # "extracted" | "not_invoice" | "failed"
    invoice: InvoiceExtraction | None = None
    error: dict[str, Any] | None = None
    financial: FinancialCheck | None = None
    review: ReviewDecision | None = None
    auxiliary: dict[str, Any] | None = None
    preprocessing: dict[str, Any] | None = None
    attempts: list[AttemptRecord] = field(default_factory=list)
    consistency: dict[str, Any] | None = None
    evidence: list[dict[str, Any]] | None = None
    confidence: list[dict[str, Any]] | None = None
    duration_s: float = 0.0
    settings_snapshot: dict[str, Any] = field(default_factory=dict)

    # -- cost / usage roll-ups ------------------------------------------------

    @property
    def input_tokens(self) -> int:
        return sum(a.response.usage.input_tokens for a in self.attempts)

    @property
    def cached_input_tokens(self) -> int:
        return sum(a.response.usage.cache_read_input_tokens for a in self.attempts)

    @property
    def output_tokens(self) -> int:
        return sum(a.response.usage.output_tokens for a in self.attempts)

    @property
    def api_calls(self) -> int:
        return len(self.attempts)

    @property
    def transient_retries(self) -> int:
        return sum(a.response.transient_retries for a in self.attempts)

    @property
    def cost_usd(self) -> float | None:
        costs = [cost_usd(a.response.usage, a.response.model) for a in self.attempts]
        if any(c is None for c in costs):
            return None
        return float(sum(c for c in costs if c is not None))

    @property
    def cost_usd_no_prompt_cache(self) -> float | None:
        costs = [cost_usd_without_prompt_cache(a.response.usage, a.response.model) for a in self.attempts]
        if any(c is None for c in costs):
            return None
        return float(sum(c for c in costs if c is not None))

    @property
    def needs_review(self) -> bool:
        # Failures always need a human; otherwise defer to the review policy.
        if self.status == "failed":
            return True
        return bool(self.review and self.review.needs_review)

    # -- serialisation ----------------------------------------------------------

    def output_json(self) -> dict[str, Any]:
        """The file written to outputs/<doc>.json.

        On success this is exactly the assignment schema. A failed document has no
        trustworthy extraction, so instead of inventing one (e.g. is_invoice=false,
        which would be a false claim) we write an explicit error object.
        """
        if self.invoice is not None:
            return self.invoice.to_output_dict()
        return {"error": self.error, "needs_review": True}

    def metadata_json(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source_file": self.source_name,
            "status": self.status,
            "needs_review": self.needs_review,
            "review_reasons": self.review.reasons if self.review else (["processing failed"] if self.error else []),
            "error": self.error,
            "financial_check": self.financial.to_dict() if self.financial else None,
            "auxiliary_extraction": self.auxiliary,
            "consistency_check": self.consistency,
            "evidence": self.evidence,
            "token_confidence": self.confidence,
            "preprocessing": self.preprocessing,
            "usage": {
                "api_calls": self.api_calls,
                "transient_retries": self.transient_retries,
                "input_tokens": self.input_tokens,
                "cached_input_tokens": self.cached_input_tokens,
                "output_tokens": self.output_tokens,
                "cost_usd": self.cost_usd,
                "cost_usd_no_prompt_cache": self.cost_usd_no_prompt_cache,
                "served_from_cache": all(a.response.from_cache for a in self.attempts) if self.attempts else False,
            },
            "attempts": [a.to_dict() for a in self.attempts],
            "duration_s": round(self.duration_s, 3),
            "config": self.settings_snapshot,
        }


def _settings_snapshot(settings: Settings) -> dict[str, Any]:
    return {
        "provider": settings.provider,
        "model": settings.model,
        "effort": settings.effort,
        "input_mode": settings.input_mode,
        "max_image_edge_px": settings.max_image_edge_px,
        "amount_tolerance": settings.amount_tolerance,
        "max_repair_attempts": settings.max_repair_attempts,
        "verify_scans": settings.verify_scans,
        "upscale_images": settings.upscale_images,
        "token_confidence": settings.token_confidence,
        "confidence_threshold": settings.confidence_threshold,
        **prompt_metadata(),
    }


def _token_confidences(
    extraction: ExtractionResult, settings: Settings, result: PipelineResult
) -> list[FieldConfidence] | None:
    """Per-field lowest token probability of the kept answer (TOKEN_CONFIDENCE), else None."""
    if not settings.token_confidence or extraction.response is None:
        return None
    tokens = extraction.response.extra.get("token_logprobs")
    confidences = field_confidences(extraction.response.text, tokens) if tokens else None
    result.confidence = [c.to_dict() for c in confidences] if confidences is not None else None
    return confidences


def _consistency_check(
    doc: PreparedDocument,
    first: ExtractionResult,
    provider: LLMProvider,
    settings: Settings,
    result: PipelineResult,
    first_confidences: list[FieldConfidence] | None = None,
) -> list[str] | None:
    """Second independent read for scanned/photographed invoices (VERIFY_SCANS).

    Returns unresolved disagreements (empty = agree, or resolved by arithmetic),
    or None when the check does not apply. The first read's values are always
    the ones kept; arbitration only decides whether a human must look.
    """
    if not settings.verify_scans or doc.has_text_layer or not first.invoice.is_invoice:
        return None
    try:
        second = InvoiceExtractor(provider, settings.with_overrides(max_repair_attempts=0)).extract(
            doc, sample_id=1, kind_prefix="verify_"
        )
    except (ExtractionFailed, ProviderFailed) as exc:
        result.attempts.extend(exc.attempts)
        result.consistency = {"checked": False, "error": exc.category.value}
        return [f"consistency re-read failed ({exc.category.value})"]
    result.attempts.extend(second.attempts)
    diffs = compare_reads(first.invoice, second.invoice, tolerance=settings.amount_tolerance)
    decision = arbitrate(diffs, first.financial, second.financial, first_confidences, settings.confidence_threshold)
    result.consistency = {
        "checked": True,
        "disagreements": [d.message for d in diffs],
        "disagreement_kinds": sorted({d.kind for d in diffs}),
        "resolved": decision.resolved,
        "resolution": decision.note,
        "reread_financial_status": second.financial.status.value,
    }
    return [] if decision.resolved else [d.message for d in diffs]


def process_document(raw_bytes: bytes, source_name: str, provider: LLMProvider, settings: Settings) -> PipelineResult:
    started = time.perf_counter()
    result = PipelineResult(
        document_id=Path(source_name).stem,
        source_name=Path(source_name).name,
        status="failed",
        settings_snapshot=_settings_snapshot(settings),
    )
    try:
        raw: RawDocument = load_bytes(raw_bytes, source_name, max_file_mb=settings.max_file_mb)
        doc: PreparedDocument = prepare_document(
            raw,
            max_edge=settings.max_image_edge_px,
            max_pages=settings.max_pages,
            upscale_images=settings.upscale_images,
        )
        result.preprocessing = doc.summary()

        extraction = InvoiceExtractor(provider, settings).extract(doc)
        result.attempts = extraction.attempts
        result.invoice = extraction.invoice
        result.financial = extraction.financial
        llm = extraction.llm
        result.auxiliary = {
            "document_type": llm.document_type.value,
            "printed_subtotal": llm.printed_subtotal,
            "tax_lines": [t.model_dump() for t in llm.tax_lines] if llm.tax_lines is not None else None,
            "adjustments_outside_items": (
                [a.model_dump() for a in llm.adjustments_outside_items]
                if llm.adjustments_outside_items is not None
                else None
            ),
            "unit_prices": [i.unit_price for i in llm.line_items] if llm.line_items is not None else None,
            "ambiguities": llm.ambiguities,
            "illegible_fields": llm.illegible_fields,
        }
        confidences = _token_confidences(extraction, settings, result)
        disagreements = _consistency_check(doc, extraction, provider, settings, result, confidences)
        evidence = collect_evidence(
            extraction.invoice,
            [p.text_layer for p in doc.pages],
            [t.amount for t in llm.tax_lines] if llm.tax_lines else None,
        )
        result.evidence = [e.to_dict() for e in evidence]
        result.review = decide_review(
            llm,
            extraction.financial,
            preprocessing_warnings=doc.warnings,
            repair_attempted=extraction.repair_attempted,
            consistency_disagreements=disagreements,
            evidence_issues=unsupported_reasons(evidence),
            confidence_issues=low_confidence_reasons(confidences, settings.confidence_threshold),
        )
        result.status = "extracted" if extraction.invoice.is_invoice else "not_invoice"
    except (ExtractionFailed, ProviderFailed) as exc:
        result.attempts = exc.attempts
        result.error = exc.to_dict()
    except PipelineError as exc:
        result.error = exc.to_dict()
    except Exception as exc:
        log.exception("unexpected error while processing %s", source_name)
        result.error = {
            "category": "internal_error",
            "message": f"{type(exc).__name__}: {exc}",
            "hint": "This is a bug.",
        }
    result.duration_s = time.perf_counter() - started
    return result


def process_path(path: str | Path, provider: LLMProvider, settings: Settings) -> PipelineResult:
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        return PipelineResult(
            document_id=path.stem,
            source_name=path.name,
            status="failed",
            error={"category": "io_error", "message": str(exc), "hint": "Check the path and permissions."},
            settings_snapshot=_settings_snapshot(settings),
        )
    return process_document(data, path.name, provider, settings)
