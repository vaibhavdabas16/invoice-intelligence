"""Document ingestion: read bytes, detect the real file type, reject bad files early.

Type detection uses magic bytes, not file extensions, so a `.pdf` that is
really HTML (a common failure when a download link expires) is caught here
instead of producing a confusing parser error later.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from src.errors import ErrorCategory, PipelineError


class FileKind(str, Enum):
    PDF = "pdf"
    IMAGE = "image"


# (signature, mime type, kind)
_SIGNATURES: list[tuple[bytes, str, FileKind]] = [
    (b"%PDF-", "application/pdf", FileKind.PDF),
    (b"\xff\xd8\xff", "image/jpeg", FileKind.IMAGE),
    (b"\x89PNG\r\n\x1a\n", "image/png", FileKind.IMAGE),
    (b"GIF87a", "image/gif", FileKind.IMAGE),
    (b"GIF89a", "image/gif", FileKind.IMAGE),
    (b"II*\x00", "image/tiff", FileKind.IMAGE),
    (b"MM\x00*", "image/tiff", FileKind.IMAGE),
    (b"BM", "image/bmp", FileKind.IMAGE),
]


@dataclass(frozen=True)
class RawDocument:
    document_id: str
    source_name: str
    data: bytes
    kind: FileKind
    mime_type: str
    sha256: str

    @property
    def size_bytes(self) -> int:
        return len(self.data)


def sniff_type(data: bytes) -> tuple[str, FileKind] | None:
    head = data[:16]
    # PDFs may legally have junk before the header; the spec allows it within the first 1 KB.
    if b"%PDF-" in data[:1024]:
        return "application/pdf", FileKind.PDF
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp", FileKind.IMAGE
    for signature, mime, kind in _SIGNATURES:
        if head.startswith(signature):
            return mime, kind
    return None


def load_bytes(data: bytes, source_name: str, *, max_file_mb: float = 20.0) -> RawDocument:
    if len(data) == 0:
        raise PipelineError(ErrorCategory.EMPTY_FILE, f"{source_name}: file is empty (0 bytes)")
    if len(data) > max_file_mb * 1024 * 1024:
        raise PipelineError(
            ErrorCategory.FILE_TOO_LARGE,
            f"{source_name}: {len(data) / 1e6:.1f} MB exceeds limit of {max_file_mb} MB",
        )
    sniffed = sniff_type(data)
    if sniffed is None:
        suffix = Path(source_name).suffix.lower()
        if suffix == ".pdf":
            # Claims to be a PDF but has no PDF header: corrupt/truncated, not merely unsupported.
            raise PipelineError(ErrorCategory.CORRUPT_FILE, f"{source_name}: no PDF header found; file is corrupt")
        raise PipelineError(
            ErrorCategory.UNSUPPORTED_FORMAT,
            f"{source_name}: unrecognised file type (extension {suffix or 'none'})",
        )
    mime, kind = sniffed
    return RawDocument(
        document_id=Path(source_name).stem,
        source_name=Path(source_name).name,
        data=data,
        kind=kind,
        mime_type=mime,
        sha256=hashlib.sha256(data).hexdigest(),
    )


def load_file(path: str | Path, *, max_file_mb: float = 20.0) -> RawDocument:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        raise IsADirectoryError(path)
    return load_bytes(path.read_bytes(), path.name, max_file_mb=max_file_mb)


SUPPORTED_SUFFIXES = {".pdf", ".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".gif"}


def discover_documents(directory: str | Path) -> list[Path]:
    """All candidate documents in a directory (ground-truth JSON is ignored)."""
    directory = Path(directory)
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
