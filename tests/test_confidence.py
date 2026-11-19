"""Token-confidence signal: span mapping, formatting tokens, thresholds, pipeline wiring."""

import json
import math

import pytest

from src.extraction.providers.base import ChatMessage, LLMRequest, LLMResponse, TextPart
from src.extraction.providers.cache import CachingProvider
from src.extraction.providers.openai_provider import OpenAIProvider
from src.pipeline import process_document
from src.validation.confidence import field_confidences, low_confidence_reasons, value_spans
from tests.fakes import ScriptedProvider, llm_invoice


def _uniform_tokens(text: str, size: int = 3) -> list[list]:
    return [[text[i : i + size], math.log(0.999)] for i in range(0, len(text), size)]


def test_value_spans_paths_and_kinds():
    text = '{"vendor":"Acme","line_items":[{"description":"A","quantity":null,"amount":12.50}],"total":3.0}'
    spans = value_spans(text)
    assert text[spans["vendor"][0] : spans["vendor"][1]] == "Acme"  # quotes excluded
    assert spans["line_items[0].quantity"][2] == "literal"
    assert text[spans["line_items[0].amount"][0] : spans["line_items[0].amount"][1]] == "12.50"


def test_low_probability_digit_in_invoice_number_is_reported():
    text = json.dumps({"invoice_number": "SGS/24-47", "total": 917.0}, separators=(",", ":"))
    toks = [
        ['{"invoice_number":"', 0.0],
        ["SGS/", 0.0],
        ["24", math.log(0.012)],
        ["-47", 0.0],
        ['","total":', 0.0],
        ["917", 0.0],
        [".0", 0.0],
        ["}", 0.0],
    ]
    assert "".join(t[0] for t in toks) == text
    conf = {c.field: c for c in field_confidences(text, toks)}
    assert conf["invoice_number"].weakest_token == "24"
    assert conf["invoice_number"].min_prob == pytest.approx(0.012)
    reasons = low_confidence_reasons(list(conf.values()), 0.5)
    assert reasons and "invoice_number 'SGS/24-47'" in reasons[0] and "p=0.01" in reasons[0]


def test_trailing_decimal_formatting_tokens_ignored():
    # the model is unsure between 920.0 and 920.00: that is not doubt about the value
    text = '{"total":920.00}'
    toks = [['{"total":', 0.0], ["920", 0.0], [".", math.log(0.6)], ["00", math.log(0.5)], ["}", 0.0]]
    assert field_confidences(text, toks)[0].min_prob > 0.99


def test_significant_decimal_digit_counts():
    text = '{"total":920.50}'
    toks = [['{"total":', 0.0], ["920", 0.0], [".", 0.0], ["5", math.log(0.3)], ["0}", 0.0]]
    assert field_confidences(text, toks)[0].min_prob == pytest.approx(0.3)


def test_line_item_labels_and_null_skipped():
    data = {"line_items": [{"description": "Pomfret, whole, kg", "quantity": None, "amount": 6080.0}]}
    text = json.dumps(data, separators=(",", ":"))
    conf = field_confidences(text, _uniform_tokens(text))
    assert [c.field for c in conf] == ["line 1 description", "line 1 amount"]


def test_mismatched_tokens_claim_nothing():
    assert field_confidences('{"total":1}', [["{", 0.0]]) is None
    assert field_confidences('{"total":1}', []) is None


def test_threshold_boundary():
    text = '{"vendor":"Acme"}'
    conf = field_confidences(text, [['{"vendor":"', 0.0], ["Acme", math.log(0.5)], ['"}', 0.0]])
    assert low_confidence_reasons(conf, 0.5) == []  # p == threshold is not "low"
    assert low_confidence_reasons(conf, 0.51)


def test_provider_requires_effort_none_for_logprobs():
    with pytest.raises(ValueError):
        OpenAIProvider("gpt-6-luna", effort="low", logprobs=True, client=object())
    OpenAIProvider("gpt-6-luna", effort="none", logprobs=True, client=object())


class LogprobProvider(ScriptedProvider):
    """ScriptedProvider that also returns token logprobs (one token per character)."""

    def __init__(self, script, low_paths=()):
        super().__init__(script)
        self.low = set(low_paths)

    def complete(self, request):
        resp: LLMResponse = super().complete(request)
        spans = value_spans(resp.text)
        low_ranges = [spans[p][:2] for p in self.low if p in spans]
        resp.extra["token_logprobs"] = [
            [ch, math.log(0.2 if any(a <= i < b for a, b in low_ranges) else 0.999)] for i, ch in enumerate(resp.text)
        ]
        return resp


def test_pipeline_flags_low_confidence_and_stores_metadata(digital_pdf_bytes, settings):
    s = settings.with_overrides(token_confidence=True, effort="none")
    r = process_document(digital_pdf_bytes, "d.pdf", LogprobProvider([llm_invoice()], ["invoice_number"]), s)
    assert r.needs_review and any("low token confidence" in x for x in r.review.reasons)
    conf = {c["field"]: c for c in r.metadata_json()["token_confidence"]}
    assert conf["invoice_number"]["min_prob"] == pytest.approx(0.2) and conf["total"]["min_prob"] > 0.99
    assert r.output_json()["invoice_number"] == "INV-2041"  # flagged, not changed


def test_pipeline_confident_answer_not_flagged(digital_pdf_bytes, settings):
    s = settings.with_overrides(token_confidence=True, effort="none")
    r = process_document(digital_pdf_bytes, "d.pdf", LogprobProvider([llm_invoice()]), s)
    assert not r.needs_review


def test_confidence_off_records_nothing(digital_pdf_bytes, settings):
    r = process_document(digital_pdf_bytes, "d.pdf", LogprobProvider([llm_invoice()], ["invoice_number"]), settings)
    assert r.confidence is None and not r.needs_review


def test_cache_preserves_token_logprobs(tmp_path):
    class P(ScriptedProvider):
        def complete(self, request):
            r = super().complete(request)
            r.extra["token_logprobs"] = [["{}", -0.1]]
            return r

    cache = CachingProvider(P(["{}"]), tmp_path)
    req = LLMRequest("s", (ChatMessage("user", (TextPart("x"),)),), None, 10)
    cache.complete(req)
    again = cache.complete(req)
    assert again.from_cache and again.extra["token_logprobs"] == [["{}", -0.1]]
