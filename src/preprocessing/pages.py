"""Turn a raw document into model-ready pages.

For every page we produce:
  * a downscaled raster image (long edge <= MAX_IMAGE_EDGE_PX; vision models
    bill by pixel area, so this is the main input-cost lever), and
  * the PDF text layer, when one exists (digital PDFs). The text layer carries
    exact characters/digits but loses table structure; the image carries the
    layout. Sending both ("hybrid") lets the model use each for what it is
    good at.

Image-quality heuristics run here too. They are deliberately simple and are
used for two things only: refusing obviously blank/unusable pages *before*
paying for an API call, and adding a human-review reason. They are not
presented as calibrated quality scores.
"""

from __future__ import annotations

import io
import math
from dataclasses import asdict, dataclass, field
from enum import Enum

import pymupdf
from PIL import Image, ImageFilter, ImageOps, ImageStat, UnidentifiedImageError

from src.errors import ErrorCategory, PipelineError
from src.ingestion.loader import FileKind, RawDocument

# A page needs at least this many non-whitespace characters to count as having a text layer.
MIN_TEXT_LAYER_CHARS = 20

# Quality thresholds (on a grayscale copy with long edge 1000 px).
# BLANK: luminance std-dev below this means an essentially uniform page.
BLANK_CONTRAST_STD = 6.0
# LOW_CONTRAST: readable but faint/washed-out.
LOW_CONTRAST_STD = 20.0
# BLUR: std-dev of a Laplacian response; sharp text produces strong edges.
SEVERE_BLUR_LAPLACIAN_STD = 1.5
LOW_SHARPNESS_LAPLACIAN_STD = 8.0
MIN_SHORT_EDGE_PX = 300

Image.MAX_IMAGE_PIXELS = 80_000_000  # guard against decompression bombs


class SourceKind(str, Enum):
    DIGITAL_PDF = "digital_pdf"
    SCANNED_PDF = "scanned_pdf"
    MIXED_PDF = "mixed_pdf"
    IMAGE = "image"


@dataclass
class PageQuality:
    width: int
    height: int
    contrast_std: float
    laplacian_std: float
    # Ratio of text-line periodicity along rows vs columns; < 1 suggests the page is rotated 90 degrees.
    orientation_ratio: float
    warnings: list[str] = field(default_factory=list)


@dataclass
class PreparedPage:
    index: int
    image_bytes: bytes
    media_type: str
    width: int
    height: int
    text_layer: str | None
    quality: PageQuality | None  # only computed for pages without a text layer


@dataclass
class PreparedDocument:
    raw: RawDocument
    source_kind: SourceKind
    pages: list[PreparedPage]
    warnings: list[str]

    @property
    def has_text_layer(self) -> bool:
        return any(p.text_layer for p in self.pages)

    def summary(self) -> dict[str, object]:
        return {
            "source_kind": self.source_kind.value,
            "page_count": len(self.pages),
            "has_text_layer": self.has_text_layer,
            "image_sizes": [(p.width, p.height) for p in self.pages],
            "quality": [asdict(p.quality) if p.quality else None for p in self.pages],
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _fit_long_edge(img: Image.Image, target: int) -> Image.Image:
    """Resize (up or down) so the long edge equals `target`."""
    w, h = img.size
    scale = target / max(w, h)
    if abs(scale - 1) < 1e-3:
        return img
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)


def _downscale(img: Image.Image, max_edge: int) -> Image.Image:
    w, h = img.size
    scale = max_edge / max(w, h)
    if scale >= 1:
        return img
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)


def _encode(img: Image.Image, *, lossless: bool) -> tuple[bytes, str]:
    buf = io.BytesIO()
    if lossless:
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue(), "image/png"
    img.convert("RGB").save(buf, format="JPEG", quality=88, optimize=True)
    return buf.getvalue(), "image/jpeg"


def _profile_variance(img: Image.Image, axis: str) -> float:
    """Variance of mean darkness along rows ('rows') or columns ('cols').

    Upright text produces strongly alternating row means (text line / gap), so
    row variance dominates. A 90-degree rotation swaps that.
    """
    w, h = img.size
    strip = img.resize((1, h), Image.Resampling.BOX) if axis == "rows" else img.resize((w, 1), Image.Resampling.BOX)
    flat = getattr(strip, "get_flattened_data", None)  # Pillow >= 12.1; getdata() is deprecated
    values = list(flat() if flat else strip.getdata())
    n = len(values)
    if n < 2:
        return 0.0
    # Remove slow trends (page shading, vignetting) with a first difference.
    diffs = [values[i + 1] - values[i] for i in range(n - 1)]
    mean = sum(diffs) / len(diffs)
    return sum((d - mean) ** 2 for d in diffs) / len(diffs)


def assess_quality(img: Image.Image) -> PageQuality:
    gray = ImageOps.grayscale(img)
    gray = _downscale(gray, 1000)
    contrast = ImageStat.Stat(gray).stddev[0]
    lap = gray.filter(ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1, offset=128))
    lap_std = ImageStat.Stat(lap).stddev[0]

    # Orientation on an autocontrasted copy so faint scans still show line structure.
    norm = ImageOps.autocontrast(gray, cutoff=1)
    rows = _profile_variance(norm, "rows")
    cols = _profile_variance(norm, "cols")
    ratio = rows / cols if cols > 0 else math.inf

    warnings: list[str] = []
    w, h = img.size
    if min(w, h) < MIN_SHORT_EDGE_PX:
        warnings.append(f"low_resolution ({w}x{h})")
    if contrast < LOW_CONTRAST_STD:
        warnings.append(f"low_contrast (std={contrast:.1f})")
    if lap_std < LOW_SHARPNESS_LAPLACIAN_STD:
        warnings.append(f"low_sharpness (laplacian_std={lap_std:.1f})")
    if ratio < 0.6:
        warnings.append(f"possible_90deg_rotation (row/col ratio={ratio:.2f})")
    return PageQuality(
        width=w,
        height=h,
        contrast_std=round(contrast, 2),
        laplacian_std=round(lap_std, 2),
        orientation_ratio=round(ratio, 3) if math.isfinite(ratio) else 999.0,
        warnings=warnings,
    )


def _is_unreadable(q: PageQuality) -> str | None:
    if q.contrast_std < BLANK_CONTRAST_STD:
        return f"page is blank or near-uniform (contrast std={q.contrast_std})"
    if min(q.width, q.height) < MIN_SHORT_EDGE_PX // 2:
        return f"resolution too low to read ({q.width}x{q.height})"
    if q.laplacian_std < SEVERE_BLUR_LAPLACIAN_STD:
        return f"page is too blurred to read (laplacian std={q.laplacian_std})"
    return None


# ---------------------------------------------------------------------------
# PDF / image preparation
# ---------------------------------------------------------------------------


def _prepare_pdf(raw: RawDocument, *, max_edge: int, max_pages: int) -> PreparedDocument:
    try:
        pdf = pymupdf.open(stream=raw.data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several exception types for broken files
        raise PipelineError(ErrorCategory.CORRUPT_FILE, f"{raw.source_name}: cannot open PDF ({exc})") from exc

    with pdf:
        if pdf.needs_pass:
            raise PipelineError(ErrorCategory.ENCRYPTED_FILE, f"{raw.source_name}: PDF is password-protected")
        if pdf.page_count == 0:
            raise PipelineError(ErrorCategory.CORRUPT_FILE, f"{raw.source_name}: PDF has no pages")
        if pdf.page_count > max_pages:
            raise PipelineError(
                ErrorCategory.FILE_TOO_LARGE,
                f"{raw.source_name}: {pdf.page_count} pages exceeds MAX_PAGES={max_pages}",
            )

        pages: list[PreparedPage] = []
        warnings: list[str] = []
        for i in range(pdf.page_count):
            page = pdf[i]
            try:
                text = page.get_text("text").strip()
                rect = page.rect
                zoom = max_edge / max(rect.width, rect.height)
                pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
            except Exception as exc:
                raise PipelineError(
                    ErrorCategory.CORRUPT_FILE, f"{raw.source_name}: page {i + 1} cannot be rendered ({exc})"
                ) from exc
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            has_text = len("".join(text.split())) >= MIN_TEXT_LAYER_CHARS
            quality = None
            if not has_text:
                quality = assess_quality(img)
                reason = _is_unreadable(quality)
                if reason:
                    raise PipelineError(ErrorCategory.UNREADABLE_DOCUMENT, f"{raw.source_name}: page {i + 1}: {reason}")
                warnings.extend(f"page {i + 1}: {w}" for w in quality.warnings)
            data, media = _encode(img, lossless=has_text)
            pages.append(
                PreparedPage(
                    index=i,
                    image_bytes=data,
                    media_type=media,
                    width=img.width,
                    height=img.height,
                    text_layer=text if has_text else None,
                    quality=quality,
                )
            )

    n_text = sum(1 for p in pages if p.text_layer)
    kind = (
        SourceKind.DIGITAL_PDF
        if n_text == len(pages)
        else SourceKind.SCANNED_PDF
        if n_text == 0
        else SourceKind.MIXED_PDF
    )
    return PreparedDocument(raw=raw, source_kind=kind, pages=pages, warnings=warnings)


def _prepare_image(raw: RawDocument, *, max_edge: int, max_pages: int, upscale: bool) -> PreparedDocument:
    try:
        img = Image.open(io.BytesIO(raw.data))
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise PipelineError(ErrorCategory.CORRUPT_FILE, f"{raw.source_name}: cannot decode image ({exc})") from exc

    frames: list[Image.Image] = []
    n_frames = getattr(img, "n_frames", 1)
    if n_frames > max_pages:
        raise PipelineError(
            ErrorCategory.FILE_TOO_LARGE, f"{raw.source_name}: {n_frames} frames exceeds MAX_PAGES={max_pages}"
        )
    for i in range(n_frames):
        img.seek(i)
        frame = ImageOps.exif_transpose(img.copy()) or img.copy()  # honour phone-camera rotation tags
        frames.append(frame.convert("RGB"))

    pages: list[PreparedPage] = []
    warnings: list[str] = []
    for i, frame in enumerate(frames):
        quality = assess_quality(frame)
        reason = _is_unreadable(quality)
        if reason:
            raise PipelineError(ErrorCategory.UNREADABLE_DOCUMENT, f"{raw.source_name}: page {i + 1}: {reason}")
        warnings.extend(f"page {i + 1}: {w}" for w in quality.warnings)
        # Scanned PDFs are always rendered at max_edge (which upsamples their embedded image);
        # with `upscale`, photos get the same treatment so small text gets as many pixels.
        scaled = _fit_long_edge(frame, max_edge) if upscale else _downscale(frame, max_edge)
        data, media = _encode(scaled, lossless=False)
        pages.append(
            PreparedPage(
                index=i,
                image_bytes=data,
                media_type=media,
                width=scaled.width,
                height=scaled.height,
                text_layer=None,
                quality=quality,
            )
        )
    return PreparedDocument(raw=raw, source_kind=SourceKind.IMAGE, pages=pages, warnings=warnings)


def prepare_document(
    raw: RawDocument, *, max_edge: int = 1568, max_pages: int = 10, upscale_images: bool = False
) -> PreparedDocument:
    if raw.kind is FileKind.PDF:
        return _prepare_pdf(raw, max_edge=max_edge, max_pages=max_pages)
    return _prepare_image(raw, max_edge=max_edge, max_pages=max_pages, upscale=upscale_images)
