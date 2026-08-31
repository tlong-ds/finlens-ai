"""Command-line orchestration for FinLens indexing services."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.config import Settings
from src.data.indexing.collection import ensure_collection
from src.data.indexing.manifest import (
    LOGGER,
    BaselineError,
    Config,
    build_manifest,
    manifest_stats,
    resolve_csv_path,
)
from src.data.indexing.reconciliation import (
    doctor,
    load_tables,
    parse_query_buckets,
    reconcile_points,
    retrieve_tables,
    upsert_points,
    verify_ingestion,
)
from src.data.indexing.service import run_indexing_pipeline, self_test


def _effective_argv(argv: Sequence[str] | None) -> list[str]:
    values = list(sys.argv[1:] if argv is None else argv)
    return values or ["index"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "ViFinQA metadata-only Granite/Qdrant baseline. "
            "Không truyền command sẽ tự chạy full indexing."
        )
    )
    parser.add_argument("--metadata", type=Path, help="Override tables_metadata.json")
    parser.add_argument("--manifest", type=Path, help="Override manifest JSONL")
    parser.add_argument("--rejects", type=Path, help="Override rejects JSONL")
    parser.add_argument("--state", type=Path, help="Override SQLite checkpoint")
    parser.add_argument("--stock-codes", type=Path, help="Override code_stock.csv")
    parser.add_argument("--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)

    index = commands.add_parser(
        "index", help="Build/reuse manifest and index all metadata into Qdrant"
    )
    index.add_argument("--force", action="store_true", help="Re-embed all points")
    index.add_argument(
        "--rebuild-manifest",
        action="store_true",
        help="Rebuild even when source is unchanged",
    )
    index.add_argument("--no-check-files", action="store_true")
    index.add_argument("--skip-source-hash", action="store_true")
    index.add_argument("--skip-verify", action="store_true")

    build = commands.add_parser(
        "build-manifest", help="Stream metadata and build JSONL"
    )
    build.add_argument("--ticker")
    build.add_argument("--year", type=int)
    build.add_argument("--limit", type=int)
    build.add_argument("--no-check-files", action="store_true")
    build.add_argument("--skip-source-hash", action="store_true")

    commands.add_parser("stats", help="Show manifest statistics")

    doctor_parser = commands.add_parser(
        "doctor", help="Check local inputs, model and Qdrant"
    )
    doctor_parser.add_argument("--skip-model", action="store_true")
    doctor_parser.add_argument("--skip-qdrant", action="store_true")

    commands.add_parser(
        "init-collection", help="Create/validate collection and indexes"
    )

    upsert = commands.add_parser("upsert", help="Embed manifest and upload points")
    upsert.add_argument("--ticker")
    upsert.add_argument("--year", type=int)
    upsert.add_argument("--limit", type=int)
    upsert.add_argument(
        "--resume", action="store_true", help="Default: skip unchanged success"
    )
    upsert.add_argument("--force", action="store_true", help="Re-embed selected points")

    verify = commands.add_parser("verify", help="Compare manifest with Qdrant")
    verify.add_argument("--ticker")
    verify.add_argument("--year", type=int)
    verify.add_argument("--sample-size", type=int, default=100)
    verify.add_argument("--skip-count", action="store_true")

    retrieve = commands.add_parser("retrieve", help="Route and retrieve table metadata")
    retrieve.add_argument("question")
    retrieve.add_argument("--ticker", action="append", dest="tickers")
    retrieve.add_argument("--year", action="append", type=int, dest="years")
    retrieve.add_argument(
        "--report-type",
        action="append",
        choices=("consolidated", "separate", "aggregated", "other"),
        dest="report_types",
    )
    retrieve.add_argument("--top-k-per-bucket", type=int)
    retrieve.add_argument("--max-candidates", type=int, default=50)
    retrieve.add_argument(
        "--load", action="store_true", help="Load retrieved CSVs with pandas"
    )

    route = commands.add_parser(
        "route", help="Inspect query buckets without model/Qdrant"
    )
    route.add_argument("question")
    route.add_argument("--ticker", action="append", dest="tickers")
    route.add_argument("--year", action="append", type=int, dest="years")
    route.add_argument(
        "--report-type",
        action="append",
        choices=("consolidated", "separate", "aggregated", "other"),
        dest="report_types",
    )

    resolve = commands.add_parser(
        "resolve", help="Resolve a retrieved table_id to local CSV"
    )
    resolve.add_argument("table_id")

    reconcile = commands.add_parser(
        "reconcile", help="Diff manifest IDs and Qdrant IDs"
    )
    reconcile.add_argument("--dry-run", action="store_true")
    reconcile.add_argument("--prune", action="store_true")
    reconcile.add_argument("--confirm", action="store_true")

    commands.add_parser("self-test", help="Run dependency-free tests in this file")
    return parser


def _config_from_args(args: argparse.Namespace, settings: Settings) -> Config:
    config = Config.from_settings(settings)
    changes: dict[str, Any] = {}
    for argument, field_name in (
        ("metadata", "metadata_path"),
        ("manifest", "manifest_path"),
        ("rejects", "rejects_path"),
        ("state", "state_path"),
        ("stock_codes", "stock_codes_path"),
    ):
        value = getattr(args, argument, None)
        if value is not None:
            changes[field_name] = Path(value).resolve()
    return replace(config, **changes) if changes else config


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def main(argv: Sequence[str] | None = None) -> int:
    # PowerShell may expose a legacy code page; keep Vietnamese CLI output valid.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = _build_parser()
    args = parser.parse_args(_effective_argv(argv))
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    settings = Settings.from_env(validate=False)
    config = _config_from_args(args, settings)
    try:
        if args.command == "index":
            result = run_indexing_pipeline(
                config,
                force=args.force,
                rebuild_manifest=args.rebuild_manifest,
                check_files=not args.no_check_files,
                source_hash=not args.skip_source_hash,
                verify=not args.skip_verify,
            )
        elif args.command == "build-manifest":
            result = build_manifest(
                config,
                ticker=args.ticker,
                year=args.year,
                limit=args.limit,
                check_files=not args.no_check_files,
                source_hash=not args.skip_source_hash,
            )
        elif args.command == "stats":
            result = manifest_stats(config.manifest_path)
        elif args.command == "doctor":
            result = doctor(
                config,
                check_model=not args.skip_model,
                check_qdrant=not args.skip_qdrant,
            )
        elif args.command == "init-collection":
            result = ensure_collection(config)
        elif args.command == "upsert":
            if args.resume and args.force:
                raise BaselineError("Không dùng đồng thời --resume và --force")
            result = upsert_points(
                config,
                ticker=args.ticker,
                year=args.year,
                limit=args.limit,
                force=args.force,
            )
        elif args.command == "verify":
            result = verify_ingestion(
                config,
                ticker=args.ticker,
                year=args.year,
                sample_size=args.sample_size,
                skip_count=args.skip_count,
            )
        elif args.command == "retrieve":
            result = retrieve_tables(
                args.question,
                config,
                ticker_overrides=args.tickers,
                year_overrides=args.years,
                report_type_overrides=args.report_types,
                top_k_per_bucket=args.top_k_per_bucket,
                max_candidates=args.max_candidates,
            )
            if args.load:
                frames = load_tables(result, config.project_root)
                result = {
                    "results": result,
                    "loaded_shapes": {
                        table_id: list(frame.shape)
                        for table_id, frame in frames.items()
                    },
                }
        elif args.command == "route":
            buckets, semantic_query = parse_query_buckets(
                args.question,
                config.stock_codes_path,
                ticker_overrides=args.tickers,
                year_overrides=args.years,
                report_type_overrides=args.report_types,
            )
            result = {"semantic_query": semantic_query, "buckets": buckets}
        elif args.command == "resolve":
            result = {
                "table_id": args.table_id,
                "csv_path": str(resolve_csv_path(args.table_id, config.project_root)),
            }
        elif args.command == "reconcile":
            if args.dry_run and args.prune:
                raise BaselineError("Chọn --dry-run hoặc --prune, không chọn cả hai")
            result = reconcile_points(config, prune=args.prune, confirm=args.confirm)
        elif args.command == "self-test":
            result = self_test()
        else:
            parser.error(f"Command không được hỗ trợ: {args.command}")
            return 2
        _print_json(result)
        return 0
    except (BaselineError, FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2
    except Exception as exc:  # pragma: no cover - external runtime failures
        if args.verbose:
            LOGGER.exception("Indexing thất bại")
        else:
            LOGGER.error("Indexing thất bại (%s): %s", exc.__class__.__name__, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
