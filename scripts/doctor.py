"""Offline FinLens configuration and dataset-path diagnostics."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

from src.config import Settings

PACKAGES = ("e2b", "httpx", "ijson", "langgraph", "openai", "pandas", "qdrant-client")


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _build_parser().parse_args(argv)
    settings = Settings.from_env(validate=False)
    root = settings.project_root
    dataset_paths = {
        "questions": root / "ViFinQA/questions/questions.jsonl",
        "stock_codes": root / settings.stock_codes_path,
        "financial_statements": root / "ViFinQA/financial_statements",
        "tables_metadata": root / settings.tables_metadata_path,
        "data": root / "data",
    }
    report = {
        "python": platform.python_version(),
        "project_root": ".",
        "settings": settings.public_dict(),
        "packages": {name: _version(name) for name in PACKAGES},
        "dataset_paths": {
            name: {
                "path": path.relative_to(root).as_posix()
                if path.is_relative_to(root)
                else Path(path.name).as_posix(),
                "exists": path.exists(),
            }
            for name, path in dataset_paths.items()
        },
        "network_checks_performed": False,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
