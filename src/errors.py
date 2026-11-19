"""Error taxonomy.

Every failure the pipeline can produce maps to exactly one `ErrorCategory`, so
batch reports can say *why* a document failed (bad input vs. provider outage
vs. model output problems) rather than just "error".
"""

from __future__ import annotations

from enum import Enum


class ErrorCategory(str, Enum):
    # Bad input, detected before any API call (no cost incurred).
    EMPTY_FILE = "empty_file"
    UNSUPPORTED_FORMAT = "unsupported_format"
    CORRUPT_FILE = "corrupt_file"
    ENCRYPTED_FILE = "encrypted_file"
    FILE_TOO_LARGE = "file_too_large"
    UNREADABLE_DOCUMENT = "unreadable_document"
    # Provider / transport failures (after retries were exhausted).
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_ERROR = "provider_error"
    PROVIDER_AUTH = "provider_auth"
    PROVIDER_REFUSAL = "provider_refusal"
    # Model output problems that survived the repair attempt.
    EXTRACTION_PARSE_ERROR = "extraction_parse_error"
    SCHEMA_VALIDATION_ERROR = "schema_validation_error"


_HINTS: dict[ErrorCategory, str] = {
    ErrorCategory.EMPTY_FILE: "The file has zero bytes. Re-export or re-upload the document.",
    ErrorCategory.UNSUPPORTED_FORMAT: "Supported formats are PDF, JPEG, PNG, WEBP, TIFF, BMP and GIF.",
    ErrorCategory.CORRUPT_FILE: "The file could not be parsed. Re-download it or ask the sender for a fresh copy.",
    ErrorCategory.ENCRYPTED_FILE: "The PDF is password-protected. Remove the password and resubmit.",
    ErrorCategory.FILE_TOO_LARGE: "The file exceeds the configured size or page limit (MAX_FILE_MB / MAX_PAGES).",
    ErrorCategory.UNREADABLE_DOCUMENT: "The page is blank or too degraded to read. Rescan at higher quality.",
    ErrorCategory.PROVIDER_TIMEOUT: "The LLM provider timed out after retries. Re-run; the batch is resumable.",
    ErrorCategory.PROVIDER_ERROR: "The LLM provider returned an error after retries. Re-run later.",
    ErrorCategory.PROVIDER_AUTH: "Check OPENAI_API_KEY in .env (or the configured provider credentials).",
    ErrorCategory.PROVIDER_REFUSAL: "The provider declined the request. Route to manual processing.",
    ErrorCategory.EXTRACTION_PARSE_ERROR: "Model output was not valid JSON even after a repair call. Route to review.",
    ErrorCategory.SCHEMA_VALIDATION_ERROR: "Model output broke the schema even after a repair call. Route to review.",
}


class PipelineError(Exception):
    """An expected, categorised failure. Never used for programming errors."""

    def __init__(self, category: ErrorCategory, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.category = category
        self.message = message
        self.retryable = retryable

    @property
    def hint(self) -> str:
        return _HINTS.get(self.category, "")

    def to_dict(self) -> dict[str, object]:
        return {"category": self.category.value, "message": self.message, "hint": self.hint}


class ProviderError(PipelineError):
    """Raised by LLM providers; `retryable` marks transient failures."""

    # Seconds the provider asked us to wait (e.g. a Retry-After header), if any.
    retry_after_s: float | None = None
