"""Submission schema conversion and deterministic ZIP packaging."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

from src.experiments.storage import write_json


def write_submission_json(records: dict[int, dict[str, Any]], path: Path) -> None:
    write_json(path, [records[key] for key in sorted(records)])


def build_zip(run_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(run_dir / "submission.json", "submission.json")
        data_dir = run_dir / "data"
        if data_dir.is_dir():
            for csv_path in sorted(data_dir.glob("*.csv")):
                archive.write(csv_path, f"data/{csv_path.name}")
