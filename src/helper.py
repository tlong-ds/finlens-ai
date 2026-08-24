"""Reusable validation and routing helpers for graph nodes."""

from __future__ import annotations

import json
import math
import unicodedata
from collections.abc import Mapping
from numbers import Number
from pathlib import Path
from typing import Any, Literal

from langgraph.types import Command
from src.routing import validate_llm_filters

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_QUESTIONS_PATH = _PROJECT_ROOT / "ViFinQA" / "questions" / "questions.jsonl"

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
        return "Response generator phải có đúng pandas_query và evidence_variables."
    if not isinstance(response["pandas_query"], str) or not response[
        "pandas_query"
    ].strip():
        return "pandas_query phải là chuỗi không rỗng."
    evidence_variables = response["evidence_variables"]
    if not isinstance(evidence_variables, list):
        return "evidence_variables phải là mảng alias DataFrame."
    if not all(isinstance(alias, str) for alias in evidence_variables):
        return "Mọi evidence variable phải là alias dạng chuỗi."
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
    """Strict compatibility wrapper around the shared LLM filter contract."""
    return validate_llm_filters(value)
