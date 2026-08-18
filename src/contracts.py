"""Shared Qdrant payload and retrieval contracts for FinLens tables."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypedDict

MIN_YEAR = 2015
MAX_YEAR = 2025

PAYLOAD_FIELDS = (
    "table_id",
    "doc_id",
    "ticker",
    "company_name",
    "year",
    "report_type",
    "table_type",
)
FILTER_FIELDS = ("ticker", "company_name", "year", "report_type", "table_type")
REPORT_TYPES = frozenset({"consolidated", "separate", "aggregated", "other"})
TABLE_TYPES = frozenset(
    {"balance_sheet", "income_statement", "cash_flow", "note_table"}
)
TABLE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class QdrantTablePayload(TypedDict):
    table_id: str
    doc_id: str
    ticker: str
    company_name: str
    year: int
    report_type: str
    table_type: str


def validate_table_id(value: Any) -> str:
    """Validate the canonical table identifier used by Qdrant and local CSVs."""
    table_id = str(value).strip() if value is not None else ""
    if (
        not table_id
        or table_id in {".", ".."}
        or not TABLE_ID_PATTERN.fullmatch(table_id)
    ):
        raise ValueError(f"table_id không hợp lệ: {table_id!r}")
    return table_id


def resolve_csv_path(table_id: str, project_root: Path) -> Path:
    """Resolve ``data/{table_id}.csv`` without trusting payload file paths."""
    safe_id = validate_table_id(table_id)
    data_root = (Path(project_root).resolve() / "data").resolve()
    candidate = (data_root / f"{safe_id}.csv").resolve()
    try:
        candidate.relative_to(data_root)
    except ValueError as exc:
        raise ValueError(f"CSV path vượt ngoài data/: {candidate}") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"Không tìm thấy CSV cho {safe_id}: {candidate}")
    return candidate


def validate_qdrant_payload(value: Mapping[str, Any]) -> QdrantTablePayload:
    """Return a normalized payload only when it exactly matches schema v2."""
    if set(value) != set(PAYLOAD_FIELDS):
        raise ValueError(
            "Payload Qdrant phải có đúng 7 trường: " + ", ".join(PAYLOAD_FIELDS)
        )

    string_fields = (
        "table_id",
        "doc_id",
        "ticker",
        "company_name",
        "report_type",
        "table_type",
    )
    normalized: dict[str, Any] = {}
    for field in string_fields:
        raw = value[field]
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"Payload field {field} phải là chuỗi không rỗng")
        normalized[field] = " ".join(raw.split())

    normalized["table_id"] = validate_table_id(normalized["table_id"])
    normalized["ticker"] = normalized["ticker"].upper()

    year = value["year"]
    if isinstance(year, bool) or not isinstance(year, int):
        raise ValueError("Payload field year phải là số nguyên")
    if not MIN_YEAR <= year <= MAX_YEAR:
        raise ValueError(f"Payload field year phải nằm trong {MIN_YEAR}–{MAX_YEAR}")
    normalized["year"] = year

    if normalized["report_type"] not in REPORT_TYPES:
        raise ValueError(f"report_type không hợp lệ: {normalized['report_type']}")
    if normalized["table_type"] not in TABLE_TYPES:
        raise ValueError(f"table_type không hợp lệ: {normalized['table_type']}")
    return normalized  # type: ignore[return-value]
