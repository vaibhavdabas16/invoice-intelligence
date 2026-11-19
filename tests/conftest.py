"""Shared fixtures. All documents are synthesised on the fly: tests never need the
confidential sample pack and never call a real LLM API."""

from __future__ import annotations

import io
from pathlib import Path

import pymupdf
import pytest
from PIL import Image, ImageDraw

from src.config import Settings

INVOICE_LINES = [
    "ACME SUPPLIES PVT LTD",
    "TAX INVOICE",
    "Invoice No: INV-2041        Date: 14/03/2026",
    "Bill to: Example Buyer Ltd",
    "1  A4 paper, 500 sheets     10   320.00   3,200.00",
    "Subtotal                               3,200.00",
    "GST 18%                                  576.00",
    "Total                                  3,776.00",
]


def _draw_invoice_image(size: tuple[int, int] = (1240, 1754)) -> Image.Image:
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    y = 80
    for line in INVOICE_LINES * 3:  # repeat so there is enough "text" structure
        draw.text((80, y), line, fill="black")
        y += 40
    return img


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        use_cache=False,
        cache_dir=tmp_path / "cache",
        retry_base_delay_s=0.0,
        max_transient_retries=2,
        max_repair_attempts=1,
        verify_scans=False,  # tests that need the consistency check enable it explicitly
        outputs_dir=tmp_path / "outputs",
        reports_dir=tmp_path / "reports",
        data_dir=tmp_path / "data",
    )


@pytest.fixture
def digital_pdf_bytes() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    y = 72
    for line in INVOICE_LINES:
        page.insert_text((72, y), line, fontsize=11)
        y += 20
    return doc.tobytes()


@pytest.fixture
def scanned_pdf_bytes() -> bytes:
    """A PDF whose only content is a raster image (no text layer), like a scanner produces."""
    buf = io.BytesIO()
    _draw_invoice_image().save(buf, format="PNG")
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_image(page.rect, stream=buf.getvalue())
    return doc.tobytes()


@pytest.fixture
def photo_jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    _draw_invoice_image((1500, 2000)).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


@pytest.fixture
def blank_png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (1200, 1600), (250, 250, 250)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def encrypted_pdf_bytes(digital_pdf_bytes: bytes) -> bytes:
    doc = pymupdf.open(stream=digital_pdf_bytes, filetype="pdf")
    return doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner")


@pytest.fixture
def truncated_pdf_bytes(digital_pdf_bytes: bytes) -> bytes:
    # Keep the header, destroy everything else: a typical interrupted download.
    return b"%PDF-1.4\n" + b"\x00garbage" * 50
