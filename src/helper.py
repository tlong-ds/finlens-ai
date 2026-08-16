"""Reusable validation and routing helpers for graph nodes."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Mapping
from numbers import Number
from pathlib import Path
from typing import Any, Literal

from langgraph.types import Command

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_QUESTIONS_PATH = _PROJECT_ROOT / "ViFinQA" / "questions" / "questions.jsonl"

_TABLE_TYPES = {"balance_sheet", "income_statement", "cash_flow", "note_table"}
_REPORT_TYPES = {"consolidated", "separate", "standalone"}
_TICKER_RE = re.compile(r"[A-Z][A-Z0-9.-]{1,9}")


def _unique(values: list[Any]) -> list[Any]:
    return list(dict.fromkeys(values))


def normalize_question(value: str) -> str:
    """Normalize a question for canonical matching."""
    normalized = unicodedata.normalize("NFC", value)
    return " ".join(normalized.split()).casefold()


def find_question(query_text: str) -> dict[str, Any]:
    """Find exactly one canonical ViFinQA question record."""
    normalized_query = normalize_question(query_text)
    matches: list[dict[str, Any]] = []

    with _QUESTIONS_PATH.open(encoding="utf-8") as questions_file:
        for line in questions_file:
            if not line.strip():
                continue
            record = json.loads(line)
            if normalize_question(str(record["question"])) == normalized_query:
                matches.append(record)

    if len(matches) != 1:
        raise ValueError(
            "question must match exactly one question in "
            "ViFinQA/questions/questions.jsonl"
        )
    return matches[0]


def generator_feedback(response: Mapping[str, Any]) -> str | None:
    """Return actionable feedback for an invalid generator response."""
    required_keys = {"pandas_query", "evidence_variables"}
    if set(response) != required_keys:
        return "Generator response must contain exactly pandas_query and evidence_variables."
    if not isinstance(response["pandas_query"], str) or not response[
        "pandas_query"
    ].strip():
        return "pandas_query must be a non-empty string."
    evidence_variables = response["evidence_variables"]
    if not isinstance(evidence_variables, list) or not evidence_variables:
        return "evidence_variables must be a non-empty list of DataFrame aliases."
    if not all(isinstance(alias, str) for alias in evidence_variables):
        return "Every evidence variable must be a string alias."
    return None


def validator_feedback(response: Mapping[str, Any]) -> str | None:
    """Return actionable feedback for a rejected validator response."""
    required_keys = {"valid", "feedback"}
    if set(response) != required_keys:
        return "Validator response must contain exactly valid and feedback."
    if not isinstance(response["valid"], bool):
        return "Validator valid must be a boolean."
    if not isinstance(response["feedback"], str):
        return "Validator feedback must be a string."
    if not response["valid"]:
        return response["feedback"].strip() or "The validator rejected the code."
    return None


def numeric_result(value: Any) -> tuple[float | None, str | None]:
    """Convert a finite numeric sandbox result to float or return feedback."""
    if isinstance(value, bool) or not isinstance(value, Number):
        return None, f"Sandbox result must be numeric, not {type(value).__name__}."
    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None, f"Sandbox result cannot be converted to float: {type(value).__name__}."
    if not math.isfinite(numeric_value):
        return None, "Sandbox result must be finite, not NaN or infinity."
    return numeric_value, None


def concise_error(error: BaseException) -> str:
    """Return a bounded single-line exception message."""
    message = " ".join(str(error).split())
    return message[:500] or type(error).__name__


def ordered_unique(values: list[str]) -> list[str]:
    """Deduplicate strings without changing their order."""
    return list(dict.fromkeys(values))


def retry_or_exhausted(
    state: Mapping[str, Any],
    update: dict[str, Any],
    *,
    attempt: int | None = None,
) -> Command[Literal["generate_code", "execute_code"]]:
    """Route feedback to regeneration or execution's terminal failure path."""
    current_attempt = int(state.get("attempt", 0)) if attempt is None else attempt
    destination = (
        "generate_code"
        if current_attempt < int(state.get("max_attempts", 5))
        else "execute_code"
    )
    return Command(update=update, goto=destination)


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
