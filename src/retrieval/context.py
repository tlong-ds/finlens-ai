"""Question-aware CSV context construction for table ranking."""

from __future__ import annotations

import csv
import json
import logging
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.contracts import resolve_csv_path, validate_qdrant_payload
from src.retrieval.dense import (
    _QUESTION_STOP_WORDS,
    FPT_RERANK_DOCUMENT_MAX_CHARS,
    RERANK_CONTEXT_DETAIL_MAX_ROWS,
    RERANK_CONTEXT_MAX_CELL_CHARS,
    RERANK_CONTEXT_MAX_LABEL_CHARS,
    RERANK_CONTEXT_MAX_TITLE_CHARS,
    RERANK_CONTEXT_SEED_ROWS,
    RERANK_CONTEXT_SMALL_TABLE_ROWS,
    RERANK_CONTEXT_TARGET_CHARS,
    Candidate,
    RerankerError,
)

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _compact_cell(value: Any) -> str:
    return " ".join(str(value).split())[:RERANK_CONTEXT_MAX_CELL_CHARS]


def _question_tokens(question: str) -> set[str]:
    return {
        token
        for token in re.findall(r"\w+", question.casefold(), flags=re.UNICODE)
        if len(token) > 1 and token not in _QUESTION_STOP_WORDS
    }


def _row_tokens(value: str) -> set[str]:
    return set(re.findall(r"\w+", value.casefold(), flags=re.UNICODE))


def _select_detailed_row_indexes(
    rows: Sequence[Mapping[str, Any]],
) -> list[int]:
    """Select lexical seed rows and their immediate neighbours."""
    if len(rows) <= RERANK_CONTEXT_SMALL_TABLE_ROWS:
        return list(range(len(rows)))

    ranked = sorted(
        range(len(rows)),
        key=lambda index: (-int(rows[index]["match_score"]), index),
    )
    seeds = [index for index in ranked if int(rows[index]["match_score"]) > 0][
        :RERANK_CONTEXT_SEED_ROWS
    ]
    if len(seeds) < RERANK_CONTEXT_SEED_ROWS:
        seeds.extend(index for index in range(len(rows)) if index not in seeds)
        seeds = seeds[:RERANK_CONTEXT_SEED_ROWS]

    expanded = {
        neighbour
        for index in seeds
        for neighbour in (index - 1, index, index + 1)
        if 0 <= neighbour < len(rows)
    }
    return sorted(expanded)[:RERANK_CONTEXT_DETAIL_MAX_ROWS]


def build_csv_rerank_context(question: str, table_id: str) -> dict[str, Any]:
    """Build layered table context without hiding unmatched row labels."""
    try:
        csv_file = resolve_csv_path(table_id, PROJECT_ROOT)
        with csv_file.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            raw_columns = next(reader)
            columns = [_compact_cell(column) for column in raw_columns]
            if not columns or not any(columns):
                raise ValueError("CSV không có header hữu ích")

            query_tokens = _question_tokens(question)
            label_index = (
                columns.index("item_label_norm")
                if "item_label_norm" in columns
                else columns.index("row_label_raw")
                if "row_label_raw" in columns
                else 0
            )
            code_index = columns.index("item_code") if "item_code" in columns else None
            title_index = (
                columns.index("note_title") if "note_title" in columns else None
            )
            rows: list[dict[str, Any]] = []
            titles: list[str] = []
            seen_titles: set[str] = set()
            for row_number, raw_row in enumerate(reader, start=2):
                cells = [_compact_cell(value) for value in raw_row]
                if not any(cells):
                    continue
                label = cells[label_index] if label_index < len(cells) else ""
                code = (
                    cells[code_index]
                    if code_index is not None and code_index < len(cells)
                    else ""
                )
                title = (
                    " ".join(str(raw_row[title_index]).split())[
                        :RERANK_CONTEXT_MAX_TITLE_CHARS
                    ]
                    if title_index is not None and title_index < len(cells)
                    else ""
                )
                if title and title not in seen_titles and len(titles) < 3:
                    seen_titles.add(title)
                    titles.append(title)
                match_text = " ".join(part for part in (label, title) if part)
                rows.append(
                    {
                        "row": row_number,
                        "cells": cells,
                        "label": label[:RERANK_CONTEXT_MAX_LABEL_CHARS],
                        "code": code,
                        "match_score": len(query_tokens & _row_tokens(match_text)),
                    }
                )
    except (OSError, UnicodeError, csv.Error, StopIteration, ValueError) as exc:
        raise RerankerError(
            f"Không dựng được rerank context từ CSV: {table_id}"
        ) from exc

    context: dict[str, Any] = {
        "columns": columns,
        "row_count": len(rows),
        "table_titles": titles,
        "row_catalog": [
            {
                "row": row["row"],
                **({"code": row["code"]} if row["code"] else {}),
                "label": row["label"],
            }
            for row in rows
            if row["label"]
        ],
        "detailed_rows": [],
    }
    essential_size = len(json.dumps(context, ensure_ascii=False))
    if essential_size > RERANK_CONTEXT_TARGET_CHARS:
        logger.warning(
            "Rerank essential context exceeds target: table_id=%s chars=%d target=%d",
            table_id,
            essential_size,
            RERANK_CONTEXT_TARGET_CHARS,
        )

    for index in _select_detailed_row_indexes(rows):
        row = rows[index]
        compact_row = {
            "row": row["row"],
            "cells": row["cells"],
        }
        candidate_context = {
            **context,
            "detailed_rows": [*context["detailed_rows"], compact_row],
        }
        serialized = json.dumps(candidate_context, ensure_ascii=False)
        if len(serialized) > RERANK_CONTEXT_TARGET_CHARS:
            break
        context = candidate_context
    return context


def _validate_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> list[Candidate]:
    validated: list[Candidate] = []
    seen: set[str] = set()
    for raw_candidate in candidates:
        try:
            candidate = dict(raw_candidate)
            table_id = str(candidate["table_id"])
            payload = validate_qdrant_payload(candidate["metadata"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RerankerError("Candidate retrieval không đúng contract") from exc
        if table_id != payload["table_id"] or table_id in seen:
            raise RerankerError(
                f"Candidate table_id không hợp lệ hoặc bị trùng: {table_id}"
            )
        seen.add(table_id)
        candidate["metadata"] = payload
        validated.append(candidate)
    return validated


def _attach_context_to_validated(
    question: str,
    candidates: Sequence[Mapping[str, Any]],
) -> list[Candidate]:
    enriched: list[Candidate] = []
    for raw_candidate in candidates:
        candidate = dict(raw_candidate)
        if not isinstance(candidate.get("rerank_context"), Mapping):
            candidate["rerank_context"] = build_csv_rerank_context(
                question,
                str(candidate["table_id"]),
            )
        enriched.append(candidate)
    return enriched


def attach_rerank_context(
    question: str,
    candidates: Sequence[Mapping[str, Any]],
) -> list[Candidate]:
    """Validate Qdrant candidates and attach layered context from local CSVs."""
    return _attach_context_to_validated(question, _validate_candidates(candidates))


def _fpt_candidate_document(
    question: str,
    candidate: Mapping[str, Any],
) -> str:
    """Pack valid, query-prioritized JSON within the FPT document limit."""
    metadata = candidate["metadata"]
    raw_context = candidate["rerank_context"]
    if not isinstance(raw_context, Mapping):
        raise RerankerError("FPT candidate rerank_context must be an object")

    from src.retrieval.selection import _build_match_summary

    match_summary = _build_match_summary(question, raw_context)
    context: dict[str, Any] = {
        "match_summary": match_summary,
        "columns": list(raw_context.get("columns") or []),
        "table_titles": list(raw_context.get("table_titles") or []),
        "row_count": raw_context.get("row_count", 0),
        "detailed_rows": [],
        "row_catalog": [],
    }
    payload = {
        "table_type": metadata.get("table_type"),
        "index_text": metadata.get("index_text", ""),
        "context": context,
    }

    def serialize() -> str:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    serialized = serialize()
    if len(serialized) > FPT_RERANK_DOCUMENT_MAX_CHARS:
        raise RerankerError(
            "FPT required table metadata exceeds the 6000-character document limit"
        )

    prioritized_row_numbers = [
        row.get("row")
        for name in ("exact_phrase_rows", "strong_overlap_rows")
        for row in match_summary.get(name, [])
        if isinstance(row, Mapping)
    ]
    priority_by_row = {
        row_number: position
        for position, row_number in enumerate(prioritized_row_numbers)
    }

    raw_detailed = raw_context.get("detailed_rows")
    detailed_rows = (
        [dict(row) for row in raw_detailed if isinstance(row, Mapping)]
        if isinstance(raw_detailed, Sequence)
        and not isinstance(raw_detailed, (str, bytes))
        else []
    )
    detailed_rows = [
        row
        for _, row in sorted(
            enumerate(detailed_rows),
            key=lambda item: (
                priority_by_row.get(item[1].get("row"), len(priority_by_row)),
                item[0],
            ),
        )
    ]

    raw_catalog = raw_context.get("row_catalog")
    row_catalog = (
        [dict(row) for row in raw_catalog if isinstance(row, Mapping)]
        if isinstance(raw_catalog, Sequence)
        and not isinstance(raw_catalog, (str, bytes))
        else []
    )
    row_catalog = [
        row
        for _, row in sorted(
            enumerate(row_catalog),
            key=lambda item: (
                priority_by_row.get(item[1].get("row"), len(priority_by_row)),
                item[0],
            ),
        )
    ]

    for field, rows in (("detailed_rows", detailed_rows), ("row_catalog", row_catalog)):
        packed_rows = context[field]
        for row in rows:
            packed_rows.append(row)
            candidate_serialized = serialize()
            if len(candidate_serialized) > FPT_RERANK_DOCUMENT_MAX_CHARS:
                packed_rows.pop()
                break
            serialized = candidate_serialized
    return serialized
