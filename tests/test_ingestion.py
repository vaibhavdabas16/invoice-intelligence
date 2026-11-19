import io

import pytest
from PIL import Image

from src.errors import ErrorCategory, PipelineError
from src.ingestion.loader import FileKind, load_bytes, sniff_type
from src.preprocessing.pages import SourceKind, prepare_document


def _prep(data: bytes, name: str, **kw):
    return prepare_document(load_bytes(data, name), **kw)


def test_digital_pdf_has_text_layer_and_image(digital_pdf_bytes):
    doc = _prep(digital_pdf_bytes, "inv.pdf")
    assert doc.source_kind is SourceKind.DIGITAL_PDF
    assert doc.pages[0].text_layer and "INV-2041" in doc.pages[0].text_layer
    assert doc.pages[0].media_type == "image/png"
    assert max(doc.pages[0].width, doc.pages[0].height) == 1568


def test_scanned_pdf_detected_and_rendered(scanned_pdf_bytes):
    doc = _prep(scanned_pdf_bytes, "scan.pdf")
    assert doc.source_kind is SourceKind.SCANNED_PDF
    assert doc.pages[0].text_layer is None
    assert doc.pages[0].quality is not None
    assert doc.pages[0].media_type == "image/jpeg"


def test_image_ingested_and_downscaled(photo_jpeg_bytes):
    doc = _prep(photo_jpeg_bytes, "photo.jpg", max_edge=1000)
    assert doc.source_kind is SourceKind.IMAGE
    assert max(doc.pages[0].width, doc.pages[0].height) == 1000


def test_magic_bytes_win_over_extension(photo_jpeg_bytes):
    raw = load_bytes(photo_jpeg_bytes, "misnamed.pdf")
    assert raw.kind is FileKind.IMAGE


def test_png_and_webp_sniffed():
    buf = io.BytesIO()
    Image.new("RGB", (10, 10)).save(buf, format="WEBP")
    assert sniff_type(buf.getvalue()) == ("image/webp", FileKind.IMAGE)


def test_empty_file():
    with pytest.raises(PipelineError) as exc:
        load_bytes(b"", "empty.pdf")
    assert exc.value.category is ErrorCategory.EMPTY_FILE


def test_unsupported_format():
    with pytest.raises(PipelineError) as exc:
        load_bytes(b"PK\x03\x04 a zip/docx file", "invoice.docx")
    assert exc.value.category is ErrorCategory.UNSUPPORTED_FORMAT


def test_html_disguised_as_pdf_is_corrupt():
    with pytest.raises(PipelineError) as exc:
        load_bytes(b"<!DOCTYPE html><html>Access denied</html>", "invoice.pdf")
    assert exc.value.category is ErrorCategory.CORRUPT_FILE


def test_truncated_pdf_is_corrupt(truncated_pdf_bytes):
    with pytest.raises(PipelineError) as exc:
        _prep(truncated_pdf_bytes, "broken.pdf")
    assert exc.value.category is ErrorCategory.CORRUPT_FILE


def test_corrupt_image_is_corrupt(photo_jpeg_bytes):
    with pytest.raises(PipelineError) as exc:
        _prep(photo_jpeg_bytes[:200], "cut.jpg")
    assert exc.value.category is ErrorCategory.CORRUPT_FILE


def test_encrypted_pdf(encrypted_pdf_bytes):
    with pytest.raises(PipelineError) as exc:
        _prep(encrypted_pdf_bytes, "locked.pdf")
    assert exc.value.category is ErrorCategory.ENCRYPTED_FILE


def test_blank_scan_is_unreadable_before_any_api_call(blank_png_bytes):
    with pytest.raises(PipelineError) as exc:
        _prep(blank_png_bytes, "blank.png")
    assert exc.value.category is ErrorCategory.UNREADABLE_DOCUMENT


def test_tiny_image_is_unreadable():
    buf = io.BytesIO()
    Image.effect_noise((80, 100), 60).convert("RGB").save(buf, format="PNG")
    with pytest.raises(PipelineError) as exc:
        _prep(buf.getvalue(), "thumb.png")
    assert exc.value.category is ErrorCategory.UNREADABLE_DOCUMENT


def test_file_size_limit():
    with pytest.raises(PipelineError) as exc:
        load_bytes(b"%PDF-1.4" + b"0" * 2_000_000, "big.pdf", max_file_mb=1)
    assert exc.value.category is ErrorCategory.FILE_TOO_LARGE


def test_page_limit(digital_pdf_bytes):
    with pytest.raises(PipelineError) as exc:
        _prep(digital_pdf_bytes, "inv.pdf", max_pages=0)
    assert exc.value.category is ErrorCategory.FILE_TOO_LARGE


def test_rotated_scan_gets_warning(photo_jpeg_bytes):
    img = Image.open(io.BytesIO(photo_jpeg_bytes)).rotate(90, expand=True)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    doc = _prep(buf.getvalue(), "rotated.jpg")
    assert any("rotation" in w for w in doc.warnings)


def test_upright_scan_has_no_rotation_warning(photo_jpeg_bytes):
    doc = _prep(photo_jpeg_bytes, "upright.jpg")
    assert not any("rotation" in w for w in doc.warnings)
