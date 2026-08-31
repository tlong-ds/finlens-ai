"""Question matching, parsing, and initial retrieval nodes."""

from __future__ import annotations

import json
import logging
import unicodedata
from collections.abc import Mapping
from typing import Any

from src.config import Settings
from src.retrieval.dense import (
    RETRIEVAL_TOP_K,
    NoMatchingCandidatesError,
    retrieve,
)
from src.retrieval.routing import QueryRoutingError, parse_query_with_diagnostics

logger = logging.getLogger(__name__)


def _normalize_question(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split()).casefold()


def _find_question(query_text: str, settings: Settings) -> dict[str, Any]:
    questions_path = settings.project_root / "ViFinQA/questions/questions.jsonl"
    normalized_query = _normalize_question(query_text)
    matches: list[dict[str, Any]] = []
    with questions_path.open(encoding="utf-8") as questions_file:
        for line in questions_file:
            if line.strip():
                record = json.loads(line)
                if _normalize_question(str(record["question"])) == normalized_query:
                    matches.append(record)
    if len(matches) != 1:
        raise ValueError(
            "question must match exactly one question in "
            "ViFinQA/questions/questions.jsonl"
        )
    return matches[0]


def match_question_node(
    state: Mapping[str, Any], *, settings: Settings
) -> dict[str, Any]:
    """Resolve the input to exactly one canonical ViFinQA question."""
    question = state.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must not be empty")

    max_attempts = state.get("max_attempts", 2)
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
        raise TypeError("max_attempts must be an integer")
    if not 1 <= max_attempts <= 5:
        raise ValueError("max_attempts must be between 1 and 5")

    question_record = _find_question(question, settings)
    return {
        "question": str(question_record["question"]),
        "question_record": question_record,
        "max_attempts": max_attempts,
        "attempt": 0,
        "feedback": "",
    }


def parse_query_node(state: Mapping[str, Any], *, settings: Settings) -> dict[str, Any]:
    """Parse and reconcile strict metadata filters from the canonical question."""
    question = str(state.get("question") or "")
    question_record = state.get("question_record") or {}
    parsed = parse_query_with_diagnostics(
        question,
        settings=settings,
        question_id=question_record.get("id", "unknown"),
    )
    return {
        "filters": parsed["filters"],
        "semantic_query": parsed["semantic_query"],
    }


def retrieve_tables_node(
    state: Mapping[str, Any], *, settings: Settings
) -> dict[str, Any]:
    """Retrieve balanced Top-N candidates from one metadata bucket per ticker."""
    query_text = str(state.get("semantic_query") or "")
    filters = dict(state.get("filters", {}) or {})
    raw_tickers = filters.get("ticker", [])
    if (
        not isinstance(raw_tickers, list)
        or not raw_tickers
        or not all(isinstance(ticker, str) and ticker for ticker in raw_tickers)
    ):
        raise QueryRoutingError("Filter ticker phải là một mảng chuỗi không rỗng")
    tickers = list(dict.fromkeys(raw_tickers))
    question_record = state.get("question_record") or {}
    question_id = question_record.get("id", "unknown")
    quota = (RETRIEVAL_TOP_K + len(tickers) - 1) // len(tickers)
    bucket_results: list[list[dict[str, Any]]] = []
    relaxed_report_type = False

    for ticker in tickers:
        bucket_filters = {**filters, "ticker": [ticker]}
        try:
            bucket = retrieve(
                query_text=query_text,
                filters=bucket_filters,
                top_n=quota,
                settings=settings,
            )
        except NoMatchingCandidatesError:
            if not bucket_filters.get("report_type"):
                raise
            fallback_bucket_filters = dict(bucket_filters)
            fallback_bucket_filters.pop("report_type", None)
            relaxed_report_type = True
            logger.info(
                "question_id=%s no candidates for ticker=%s with report_type; "
                "retrying that bucket without report_type",
                question_id,
                ticker,
            )
            bucket = retrieve(
                query_text=query_text,
                filters=fallback_bucket_filters,
                top_n=quota,
                settings=settings,
            )
        bucket_results.append(bucket)

    candidates: list[dict[str, Any]] = []
    seen_table_ids: set[str] = set()
    max_bucket_size = max((len(bucket) for bucket in bucket_results), default=0)
    for bucket_rank in range(max_bucket_size):
        for bucket in bucket_results:
            if bucket_rank >= len(bucket):
                continue
            candidate = dict(bucket[bucket_rank])
            table_id = str(candidate.get("table_id") or "")
            if not table_id or table_id in seen_table_ids:
                continue
            seen_table_ids.add(table_id)
            candidate.pop("retrieval_rank", None)
            candidate["dense_rank"] = len(candidates) + 1
            candidates.append(candidate)
            if len(candidates) == RETRIEVAL_TOP_K:
                break
        if len(candidates) == RETRIEVAL_TOP_K:
            break

    result: dict[str, Any] = {"candidates": candidates}
    if relaxed_report_type:
        effective_filters = dict(filters)
        effective_filters.pop("report_type", None)
        result["filters"] = effective_filters
    return result
