"""Command-line interface.

    python -m src.main extract invoice.pdf > out.json        # required-schema JSON on stdout
    python -m src.main extract invoice.pdf --full             # + validation, review, usage metadata
    python -m src.main batch                                  # all documents in data/labelled + data/unlabelled
    python -m src.main serve                                  # FastAPI app + web UI on http://127.0.0.1:8000

Exit codes for `extract`: 0 success (incl. non-invoice), 1 processing failed, 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from src.config import Settings, get_settings
from src.cost.report import write_cost_files
from src.errors import ProviderError
from src.extraction.providers import create_provider
from src.extraction.providers.base import LLMProvider
from src.ingestion.loader import discover_documents
from src.pipeline import PipelineResult, process_path
from src.runner import run_many

log = logging.getLogger("cli")


def _settings_from_args(args: argparse.Namespace) -> Settings:
    return get_settings().with_overrides(
        model=getattr(args, "model", None),
        effort=getattr(args, "effort", None),
        input_mode=getattr(args, "input_mode", None),
    )


def _provider_or_exit(settings: Settings, args: argparse.Namespace) -> LLMProvider | None:
    try:
        return create_provider(settings, use_cache=not args.no_cache)
    except ProviderError as exc:
        print(f"error [{exc.category.value}]: {exc.message}", file=sys.stderr)
        return None


def cmd_extract(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    path = Path(args.file)
    if not path.is_file():
        print(f"error: {path} is not a file", file=sys.stderr)
        return 2
    provider = _provider_or_exit(settings, args)
    if provider is None:
        return 2
    result = process_path(path, provider, settings)
    payload: dict[str, Any] = (
        {"result": result.output_json(), "metadata": result.metadata_json()} if args.full else result.output_json()
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if result.status == "failed":
        err = result.error or {}
        print(
            f"extraction failed [{err.get('category')}]: {err.get('message')}\nhint: {err.get('hint')}", file=sys.stderr
        )
        return 1
    if result.needs_review:
        print("needs_review: " + "; ".join(result.review.reasons if result.review else []), file=sys.stderr)
    return 0


def _round(value: float | None, digits: int = 8) -> float | None:
    return round(value, digits) if value is not None else None


def _cost_row_from_meta(meta: dict[str, Any], subset: str) -> dict[str, Any]:
    usage = meta.get("usage", {})
    attempts = meta.get("attempts", [])
    pre = meta.get("preprocessing") or {}
    return {
        "document_id": meta["document_id"],
        "subset": subset,
        "status": meta["status"],
        "source_kind": pre.get("source_kind"),
        "pages": pre.get("page_count"),
        "provider": meta.get("config", {}).get("provider"),
        "model": ",".join(sorted({a["model"] for a in attempts})) or meta.get("config", {}).get("model"),
        "api_calls": usage.get("api_calls", 0),
        "transient_retries": usage.get("transient_retries", 0),
        "repair_calls": sum(1 for a in attempts if a["kind"].endswith("repair")),
        "verify_calls": sum(1 for a in attempts if a["kind"].startswith("verify_")),
        "input_tokens": usage.get("input_tokens", 0),
        "cached_input_tokens": usage.get("cached_input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "cost_usd": _round(usage.get("cost_usd")),
        "cost_usd_no_prompt_cache": _round(usage.get("cost_usd_no_prompt_cache")),
        "served_from_cache": usage.get("served_from_cache"),
        "processing_s": meta.get("duration_s"),
        "api_latency_s": round(sum(a.get("latency_s", 0) for a in attempts), 3),
    }


def _is_reusable(meta_path: Path, settings: Settings) -> bool:
    """A previous result can be reused if it succeeded under the same extraction config."""
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    cfg = meta.get("config", {})
    from src.extraction.prompts import PROMPT_VERSION

    return (
        meta.get("status") != "failed"
        and cfg.get("model") == settings.model
        and cfg.get("input_mode") == settings.input_mode
        and cfg.get("effort") == settings.effort
        and cfg.get("verify_scans") == settings.verify_scans
        and cfg.get("upscale_images") == settings.upscale_images
        and cfg.get("prompt_version") == PROMPT_VERSION
    )


def cmd_batch(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    out_dir: Path = args.out or settings.outputs_dir
    meta_dir = out_dir / "_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    subsets: list[tuple[str, Path]] = []
    for sub in args.subsets:
        d = settings.data_dir / sub
        if not d.exists():
            print(f"warning: {d} does not exist, skipping", file=sys.stderr)
            continue
        subsets.extend((sub, p) for p in discover_documents(d))
    if not subsets:
        print("error: no documents found (see README: place the sample pack in data/)", file=sys.stderr)
        return 2

    todo = [(s, p) for s, p in subsets if args.force or not _is_reusable(meta_dir / f"{p.stem}.meta.json", settings)]
    print(f"{len(subsets)} documents, {len(subsets) - len(todo)} already done, {len(todo)} to process", file=sys.stderr)

    provider = _provider_or_exit(settings, args)
    if provider is None:
        return 2

    def save(result: PipelineResult) -> None:
        # Written per document as soon as it finishes: an interrupted batch resumes where it stopped.
        (out_dir / f"{result.document_id}.json").write_text(
            json.dumps(result.output_json(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (meta_dir / f"{result.document_id}.meta.json").write_text(
            json.dumps(result.metadata_json(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    run_many([p for _, p in todo], provider, settings, workers=args.workers, on_done=save)

    # Aggregate over *all* documents (processed now or earlier) from their metadata files.
    rows, manifest = [], []
    for subset, path in subsets:
        meta = json.loads((meta_dir / f"{path.stem}.meta.json").read_text(encoding="utf-8"))
        rows.append(_cost_row_from_meta(meta, subset))
        manifest.append(
            {
                "document_id": meta["document_id"],
                "source_file": f"{subset}/{meta['source_file']}",
                "subset": subset,
                "status": meta["status"],
                "document_type": (meta.get("auxiliary_extraction") or {}).get("document_type"),
                "needs_review": meta["needs_review"],
                "review_reasons": meta["review_reasons"],
                "financial_status": (meta.get("financial_check") or {}).get("status"),
                "error": meta["error"],
                "output_file": f"{meta['document_id']}.json",
            }
        )
    (out_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = write_cost_files(rows, settings.reports_dir, stem="cost")

    from tabulate import tabulate

    print(
        tabulate(
            [
                [
                    m["document_id"],
                    m["status"],
                    m["document_type"],
                    m["financial_status"],
                    "yes" if m["needs_review"] else "no",
                    (m["error"] or {}).get("category", ""),
                ]
                for m in manifest
            ],
            headers=["DOC", "STATUS", "DOC TYPE", "TOTALS CHECK", "REVIEW", "ERROR"],
        )
    )
    a = summary["all"]
    print(
        f"\ntotal cost ${a['total_cost_usd']:.6f} | avg/doc ${a['avg_cost_per_document_usd'] or 0:.6f} "
        f"(no-prompt-cache upper bound ${a['avg_cost_per_document_usd_no_prompt_cache'] or 0:.6f}) | "
        f"avg tokens in (uncached+cached)/out {a['avg_input_tokens']}+{a['avg_cached_input_tokens']}/"
        f"{a['avg_output_tokens']} | calls {a['total_api_calls']}"
    )
    print(f"outputs: {out_dir}  |  cost report: {settings.reports_dir / 'cost_per_document.csv'}")
    return 1 if any(m["status"] == "failed" for m in manifest) else 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("src.api.app:app", host=args.host, port=args.port, reload=False)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.main", description="Invoice extraction pipeline")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--no-cache", action="store_true", help="bypass the LLM response cache")
        p.add_argument("--model", default=None)
        p.add_argument("--effort", default=None)
        p.add_argument("--input-mode", choices=["hybrid", "image", "text"], default=None)

    p = sub.add_parser("extract", help="extract one document and print JSON")
    p.add_argument("file")
    p.add_argument("--full", action="store_true", help="include validation/review/usage metadata")
    common(p)
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("batch", help="process all documents into outputs/ (resumable)")
    p.add_argument("--subsets", nargs="+", default=["labelled", "unlabelled"], help="sub-folders of data/")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--force", action="store_true", help="reprocess documents that already have outputs")
    p.add_argument("--workers", type=int, default=4)
    common(p)
    p.set_defaults(func=cmd_batch)

    p = sub.add_parser("serve", help="run the API + web UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
