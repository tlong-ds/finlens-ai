"""Question-driven planning support grounded in reranker-selected tables."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

EVIDENCE_PLAN_SCHEMA_VERSION = 3
PLANNING_CELL_MAX_CHARS = 160

_DATAFRAME_ALIAS_PATTERN = re.compile(r"^df_[1-9][0-9]*$")
_LABEL_COLUMNS = ("item_label_norm", "row_label_raw")
_CODE_COLUMN = "item_code"
_TITLE_COLUMN = "note_title"


def _compact_value(value: Any) -> str:
    return " ".join(str(value).split())[:PLANNING_CELL_MAX_CHARS]


def _optional_compact(value: Any) -> str | None:
    compact = _compact_value(value)
    return compact or None


def _catalog_row(dataframe: pd.DataFrame, position: int) -> dict[str, Any]:
    """Build one stable row identity from the loaded DataFrame."""
    row = dataframe.iloc[position]
    label_column = next(
        (column for column in _LABEL_COLUMNS if column in dataframe.columns),
        dataframe.columns[0] if len(dataframe.columns) else None,
    )
    item: dict[str, Any] = {"row_position": position}
    if label_column is not None:
        item["label"] = _compact_value(row[label_column])
    if _CODE_COLUMN in dataframe.columns:
        code = _optional_compact(row[_CODE_COLUMN])
        if code is not None:
            item["code"] = code
    if _TITLE_COLUMN in dataframe.columns:
        title = _optional_compact(row[_TITLE_COLUMN])
        if title is not None:
            item["title"] = title
    return item


def _rerank_detail_positions(
    rerank_context: Mapping[str, Any],
    row_count: int,
) -> list[int]:
    """Translate reranker CSV line numbers into DataFrame positions."""
    raw_rows = rerank_context.get("detailed_rows")
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        return []
    positions: list[int] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, Mapping):
            continue
        raw_position = raw_row.get("row_position")
        if isinstance(raw_position, int) and not isinstance(raw_position, bool):
            position = raw_position
        else:
            csv_line = raw_row.get("row")
            if not isinstance(csv_line, int) or isinstance(csv_line, bool):
                continue
            position = csv_line - 2
        if 0 <= position < row_count and position not in positions:
            positions.append(position)
    return positions


def _row_values(dataframe: pd.DataFrame, position: int) -> dict[str, str]:
    row = dataframe.iloc[position]
    return {str(column): _compact_value(row[column]) for column in dataframe.columns}


def build_planning_inventory(
    dataframes: Mapping[str, pd.DataFrame],
    alias_metadata: Mapping[str, Mapping[str, Any]],
    rerank_contexts: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Expose every row identity and reuse reranker-selected detailed rows.

    The row catalog has no scan cutoff. Full cell values remain bounded to the
    detailed rows already selected by the reranker. Any catalog row named by the
    planner is hydrated from the loaded DataFrame after planning.
    """
    contexts = rerank_contexts or {}
    inventory: list[dict[str, Any]] = []
    for alias, dataframe in dataframes.items():
        metadata = alias_metadata.get(alias)
        if metadata is None:
            raise ValueError(f"Missing metadata for alias {alias}")
        raw_context = contexts.get(alias) or {}
        detail_positions = _rerank_detail_positions(raw_context, len(dataframe))
        if not detail_positions and len(dataframe) <= 8:
            detail_positions = list(range(len(dataframe)))
        inventory.append(
            {
                "alias": alias,
                "metadata": dict(metadata),
                "row_count": len(dataframe),
                "columns": [
                    {"name": str(column), "dtype": str(dataframe[column].dtype)}
                    for column in dataframe.columns
                ],
                "table_titles": list(raw_context.get("table_titles") or []),
                "match_summary": dict(raw_context.get("match_summary") or {}),
                "row_catalog": [
                    _catalog_row(dataframe, position)
                    for position in range(len(dataframe))
                ],
                "detailed_rows": [
                    {
                        "row_position": position,
                        "values": _row_values(dataframe, position),
                    }
                    for position in detail_positions
                ],
            }
        )
    return inventory


def _planned_row_positions(
    generation_plan: Mapping[str, Any],
    allowed_aliases: set[str],
) -> dict[str, list[int]]:
    positions: dict[str, list[int]] = {alias: [] for alias in allowed_aliases}
    raw_evidence = generation_plan.get("evidence")
    if not isinstance(raw_evidence, Sequence) or isinstance(raw_evidence, (str, bytes)):
        return positions
    for raw_item in raw_evidence:
        if not isinstance(raw_item, Mapping):
            continue
        alias = raw_item.get("alias")
        if not isinstance(alias, str) or alias not in positions:
            continue
        raw_rows = raw_item.get("rows")
        if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
            continue
        for raw_row in raw_rows:
            raw_position = (
                raw_row.get("row_position") if isinstance(raw_row, Mapping) else raw_row
            )
            if (
                isinstance(raw_position, int)
                and not isinstance(raw_position, bool)
                and raw_position not in positions[alias]
            ):
                positions[alias].append(raw_position)
    return positions


def aliases_declared_in_plan(
    generation_plan: Mapping[str, Any],
    dataframes: Mapping[str, pd.DataFrame] | None = None,
) -> set[str]:
    """Extract table aliases declared in generation_plan with at least one valid row and purpose."""
    if not isinstance(generation_plan, Mapping):
        return set()
    raw_evidence = generation_plan.get("evidence")
    if not isinstance(raw_evidence, Sequence) or isinstance(raw_evidence, (str, bytes)):
        return set()

    declared_aliases: set[str] = set()
    for raw_item in raw_evidence:
        if not isinstance(raw_item, Mapping):
            continue
        alias = raw_item.get("alias")
        if not isinstance(alias, str) or not alias:
            continue
        raw_rows = raw_item.get("rows")
        if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
            continue
        df = dataframes.get(alias) if dataframes is not None else None
        max_rows = len(df) if df is not None else None
        has_valid_row = False
        for raw_row in raw_rows:
            if not isinstance(raw_row, Mapping):
                continue
            position = raw_row.get("row_position")
            purpose = raw_row.get("purpose")
            if (
                isinstance(position, int)
                and not isinstance(position, bool)
                and (max_rows is None or 0 <= position < max_rows)
                and isinstance(purpose, str)
                and purpose.strip()
            ):
                has_valid_row = True
                break
        if has_valid_row:
            declared_aliases.add(alias)

    return declared_aliases


def hydrate_planned_rows(
    generation_plan: Mapping[str, Any],
    dataframes: Mapping[str, pd.DataFrame],
    inventory: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Hydrate planner-selected positions plus reranker-detailed fallback rows."""
    positions = _planned_row_positions(generation_plan, set(dataframes))
    for table in inventory:
        alias = table.get("alias")
        if not isinstance(alias, str) or alias not in positions:
            continue
        raw_details = table.get("detailed_rows")
        if not isinstance(raw_details, Sequence) or isinstance(
            raw_details, (str, bytes)
        ):
            continue
        for raw_detail in raw_details:
            if not isinstance(raw_detail, Mapping):
                continue
            position = raw_detail.get("row_position")
            if (
                isinstance(position, int)
                and not isinstance(position, bool)
                and position not in positions[alias]
            ):
                positions[alias].append(position)

    hydrated: dict[str, list[dict[str, Any]]] = {}
    for alias, dataframe in dataframes.items():
        hydrated[alias] = [
            {
                "row_position": position,
                "values": _row_values(dataframe, position),
            }
            for position in positions[alias]
            if 0 <= position < len(dataframe)
        ]
    return hydrated
