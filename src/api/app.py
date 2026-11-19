"""HTTP API + minimal web UI.

    uvicorn src.api.app:app            (or: python -m src.main serve)

    POST /extract      multipart file upload -> {"result": <required schema>, "metadata": {...}}
    GET  /health       liveness + configured provider/model
    GET  /             single-page UI (upload, preview, fields, validation, review flag)

Errors in the *document* (corrupt, unreadable, non-invoice, provider outage)
are part of a normal 200 response with `metadata.status` / `metadata.error`,
because they are expected outcomes the UI must display. HTTP errors are used
only for problems with the *request* itself (no file, file too large).
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from src.config import Settings, get_settings
from src.extraction.providers import create_provider
from src.extraction.providers.base import LLMProvider
from src.pipeline import process_document

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Invoice Intelligence", version="1.0.0")


@lru_cache(maxsize=1)
def _settings() -> Settings:
    return get_settings()


@lru_cache(maxsize=1)
def _provider() -> LLMProvider:
    return create_provider(_settings())


def get_provider() -> LLMProvider:
    """Indirection so tests can substitute a fake provider."""
    return _provider()


@app.get("/health")
def health() -> dict[str, Any]:
    s = _settings()
    return {"status": "ok", "provider": s.provider, "model": s.model, "input_mode": s.input_mode}


@app.post("/extract")
async def extract(file: UploadFile = File(...)) -> dict[str, Any]:
    settings = _settings()
    data = await file.read(int(settings.max_file_mb * 1024 * 1024) + 1)
    if len(data) > settings.max_file_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"file exceeds {settings.max_file_mb} MB")
    name = file.filename or "upload"
    # The pipeline is synchronous (SDK call + image processing); keep the event loop free.
    result = await asyncio.to_thread(process_document, data, name, app_provider(), settings)
    return {"result": result.output_json(), "metadata": result.metadata_json()}


def app_provider() -> LLMProvider:
    override = getattr(app.state, "provider_override", None)  # set by tests
    return override if override is not None else get_provider()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
