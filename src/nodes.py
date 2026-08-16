"""LangGraph node adapters for Level 1 table retrieval."""

from __future__ import annotations

import logging
import re
from typing import Any, Mapping

from src.llm import generate_structured
from src.retrieval import rerank, retrieve

logger = logging.getLogger(__name__)

_TABLE_TYPES = {"balance_sheet", "income_statement", "cash_flow", "note_table"}
_REPORT_TYPES = {"consolidated", "separate", "standalone"}
_TICKER_RE = re.compile(r"[A-Z][A-Z0-9.-]{1,9}")

_PARSE_SYSTEM_PROMPT = """You extract conservative metadata filters for financial-table retrieval.
Return only one JSON object. Omit any field that is not explicit or highly certain.
Allowed fields are ticker, year, report_type, and table_type; every value is an array.
Allowed table_type values: balance_sheet, income_statement, cash_flow, note_table.
Allowed report_type values: consolidated, separate, standalone.
Do not infer a single table_type when the question may require multiple statements."""


def _unique(values: list[Any]) -> list[Any]:
    return list(dict.fromkeys(values))


def validate_filters(value: Mapping[str, Any]) -> dict[str, list[str | int]]:
    """Drop unknown, malformed, or unsupported filter values locally."""
    validated: dict[str, list[str | int]] = {}

    tickers = value.get("ticker")
    if isinstance(tickers, list):
        clean_tickers = [
            item.strip().upper()
            for item in tickers
            if isinstance(item, str) and _TICKER_RE.fullmatch(item.strip().upper())
        ]
        if clean_tickers:
            validated["ticker"] = _unique(clean_tickers)

    years = value.get("year")
    if isinstance(years, list):
        clean_years = [
            item
            for item in years
            if isinstance(item, int)
            and not isinstance(item, bool)
            and 1900 <= item <= 2100
        ]
        if clean_years:
            validated["year"] = _unique(clean_years)

    for field, allowed in (
        ("report_type", _REPORT_TYPES),
        ("table_type", _TABLE_TYPES),
    ):
        values = value.get(field)
        if isinstance(values, list):
            clean_values = [
                item.strip().lower()
                for item in values
                if isinstance(item, str) and item.strip().lower() in allowed
            ]
            if clean_values:
                validated[field] = _unique(clean_values)

    return validated


def parse_query_node(state: Mapping[str, Any]) -> dict[str, Any]:
    """Parse conservative metadata filters from the original question."""
    question = str(state["question"])
    raw_filters = generate_structured(
        f"Question: {question}", system_prompt=_PARSE_SYSTEM_PROMPT
    )
    filters = validate_filters(raw_filters)
    logger.info("Question: %s", question)
    logger.info("Parsed filters: %s", filters)
    return {"filters": filters}


def retrieve_tables_node(state: Mapping[str, Any]) -> dict[str, Any]:
    """Retrieve Top-N tables using the question and parsed filters."""
    candidates = retrieve(
        question=str(state["question"]),
        filters=state.get("filters", {}),
    )
    return {"candidates": candidates}


def rerank_tables_node(state: Mapping[str, Any]) -> dict[str, Any]:
    """Rerank candidates and return the final Top-K tables."""
    retrieved_tables = rerank(
        question=str(state["question"]),
        candidates=state.get("candidates", []),
    )
    logger.info(
        "Final table IDs: %s",
        [item.get("table_id") for item in retrieved_tables],
    )
    return {"retrieved_tables": retrieved_tables}
