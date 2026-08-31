"""Prepare ViFinQA OCR tables and metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import Settings
from src.data.preparation.service import prepare


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "financial_statements_dir",
        nargs="?",
        type=Path,
        default=Path("ViFinQA/financial_statements"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    settings = Settings.from_env(validate=False)
    source = args.financial_statements_dir
    if not source.is_absolute():
        source = settings.project_root / source
    prepare(str(source), settings=settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
