"""Execute the web UI's pure render functions in Node against pipeline metadata (skipped without Node)."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from src.pipeline import process_document
from tests.fakes import ScriptedProvider, llm_invoice

HTML = Path(__file__).resolve().parent.parent / "src" / "api" / "static" / "index.html"

NODE = shutil.which("node")

HARNESS = r"""
const fs = require("fs");
const html = fs.readFileSync(process.argv[1], "utf8");
const js = html.split("<script>")[1].split("</script>")[0];
const stub = { addEventListener() {}, textContent: "", innerHTML: "", hidden: false, classList: { add() {}, remove() {} } };
global.document = { getElementById: () => stub };
global.fetch = () => new Promise(() => {});
global.URL = { createObjectURL: () => "", revokeObjectURL() {} };
eval(js + "; global.renderWorkflow = renderWorkflow; global.renderEvidence = renderEvidence; global.renderReread = renderReread;");
const data = JSON.parse(fs.readFileSync(0, "utf8"));
const m = data.metadata, r = data.result;
const out = renderWorkflow(r, m) + (r.is_invoice ? renderEvidence(m) : "") + (m.consistency_check ? renderReread(m.consistency_check) : "");
process.stdout.write(out);
"""


def _render(payload: dict) -> str:
    proc = subprocess.run(
        [NODE or "node", "-e", HARNESS, str(HTML)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed")


def test_ui_renders_clean_digital_invoice(digital_pdf_bytes, settings):
    r = process_document(digital_pdf_bytes, "d.pdf", ScriptedProvider([llm_invoice()]), settings)
    html = _render({"result": r.output_json(), "metadata": r.metadata_json()})
    assert "Validation workflow" in html and "Decision: no review needed" in html
    assert "Evidence in PDF text layer" in html and "grounded" in html


def test_ui_renders_review_case_with_reread(scanned_pdf_bytes, settings):
    s = settings.with_overrides(verify_scans=True)
    p = ScriptedProvider([llm_invoice(), llm_invoice(invoice_number="INV-2O41")])
    r = process_document(scanned_pdf_bytes, "s.pdf", p, s)
    html = _render({"result": r.output_json(), "metadata": r.metadata_json()})
    assert "Decision: needs human review" in html
    assert "Independent re-read" in html and "INV-2O41" in html
    assert "not checkable" in html  # scans have no text layer to ground against
