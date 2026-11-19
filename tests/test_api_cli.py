"""API and CLI smoke tests with a fake provider (no network)."""

import json

import pytest
from fastapi.testclient import TestClient

from src import main as cli
from src.api import app as api_module
from tests.fakes import ScriptedProvider, llm_invoice, llm_non_invoice


@pytest.fixture
def client():
    api_module.app.state.provider_override = None
    with TestClient(api_module.app) as c:
        yield c
    api_module.app.state.provider_override = None


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["provider"]


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Invoice Intelligence" in r.text


def test_extract_invoice(client, digital_pdf_bytes):
    api_module.app.state.provider_override = ScriptedProvider([llm_invoice()])
    r = client.post("/extract", files={"file": ("inv.pdf", digital_pdf_bytes, "application/pdf")})
    assert r.status_code == 200
    body = r.json()
    assert body["result"]["invoice_number"] == "INV-2041"
    assert body["metadata"]["financial_check"]["status"] == "ok"
    assert body["metadata"]["needs_review"] is False


def test_extract_non_invoice(client, digital_pdf_bytes):
    api_module.app.state.provider_override = ScriptedProvider([llm_non_invoice("bank_statement")])
    body = client.post("/extract", files={"file": ("stmt.pdf", digital_pdf_bytes, "application/pdf")}).json()
    assert body["result"]["is_invoice"] is False
    assert body["metadata"]["auxiliary_extraction"]["document_type"] == "bank_statement"


def test_extract_corrupt_file_is_a_graceful_200(client):
    api_module.app.state.provider_override = ScriptedProvider([])
    body = client.post("/extract", files={"file": ("bad.pdf", b"<html>nope</html>", "application/pdf")}).json()
    assert body["metadata"]["status"] == "failed"
    assert body["metadata"]["error"]["category"] == "corrupt_file"
    assert body["result"]["needs_review"] is True


def test_cli_extract_prints_required_schema(tmp_path, digital_pdf_bytes, capsys, monkeypatch):
    pdf = tmp_path / "inv.pdf"
    pdf.write_bytes(digital_pdf_bytes)
    monkeypatch.setattr(cli, "create_provider", lambda settings, use_cache=None: ScriptedProvider([llm_invoice()]))
    assert cli.main(["extract", str(pdf)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert list(out) == [
        "is_invoice",
        "vendor",
        "invoice_number",
        "invoice_date",
        "currency",
        "line_items",
        "tax_total",
        "total",
    ]


def test_cli_extract_failure_exit_code(tmp_path, capsys, monkeypatch):
    bad = tmp_path / "empty.pdf"
    bad.write_bytes(b"")
    monkeypatch.setattr(cli, "create_provider", lambda settings, use_cache=None: ScriptedProvider([]))
    assert cli.main(["extract", str(bad)]) == 1
    assert "empty_file" in capsys.readouterr().err


def test_cli_batch_is_resumable(tmp_path, digital_pdf_bytes, monkeypatch):
    data = tmp_path / "data" / "labelled"
    data.mkdir(parents=True)
    (data / "A.pdf").write_bytes(digital_pdf_bytes)
    (data / "B.pdf").write_bytes(b"")  # bad input must not stop the batch
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))
    provider = ScriptedProvider([llm_invoice()])
    monkeypatch.setattr(cli, "create_provider", lambda settings, use_cache=None: provider)
    out = tmp_path / "outputs"

    rc = cli.main(["batch", "--subsets", "labelled", "--out", str(out), "--workers", "1"])
    assert rc == 1  # one failed document is reported via exit code, after everything else finished
    assert json.loads((out / "A.json").read_text())["total"] == 3776.0
    assert "error" in json.loads((out / "B.json").read_text())
    assert (tmp_path / "reports" / "cost_per_document.csv").exists()
    manifest = json.loads((out / "_manifest.json").read_text())
    assert {m["document_id"]: m["status"] for m in manifest} == {"A": "extracted", "B": "failed"}

    # Second run: A is reused (no API call), only the failed B is retried.
    calls_before = len(provider.requests)
    cli.main(["batch", "--subsets", "labelled", "--out", str(out), "--workers", "1"])
    assert len(provider.requests) == calls_before
