"""A fresh clone without an API key must fail with an actionable message, not a traceback."""

import pytest
from fastapi.testclient import TestClient

from src import main as cli
from src.api import app as api_module
from src.config import Settings
from src.errors import ErrorCategory, ProviderError
from src.eval import run as eval_run
from src.extraction.providers import create_provider


@pytest.fixture
def no_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_factory_raises_categorised_error(no_key):
    with pytest.raises(ProviderError) as exc:
        create_provider(Settings(use_cache=False))
    assert exc.value.category is ErrorCategory.PROVIDER_AUTH
    assert ".env" in exc.value.message


def test_cli_extract_exits_2_with_message(no_key, tmp_path, digital_pdf_bytes, capsys):
    pdf = tmp_path / "inv.pdf"
    pdf.write_bytes(digital_pdf_bytes)
    assert cli.main(["extract", str(pdf), "--no-cache"]) == 2
    err = capsys.readouterr().err
    assert "OPENAI_API_KEY is not set" in err and "Traceback" not in err


def test_eval_exits_2_with_message(no_key, tmp_path, digital_pdf_bytes, capsys):
    (tmp_path / "L01.pdf").write_bytes(digital_pdf_bytes)
    (tmp_path / "L01.json").write_text("{}")
    assert eval_run.main(["--data-dir", str(tmp_path), "--no-cache"]) == 2
    assert "OPENAI_API_KEY is not set" in capsys.readouterr().err


def test_api_returns_503(no_key, digital_pdf_bytes):
    api_module._provider.cache_clear()
    api_module.app.state.provider_override = None
    with TestClient(api_module.app) as client:
        r = client.post("/extract", files={"file": ("inv.pdf", digital_pdf_bytes, "application/pdf")})
    api_module._provider.cache_clear()
    assert r.status_code == 503 and "OPENAI_API_KEY" in r.json()["detail"]
