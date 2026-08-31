"""Values shared by retrieval and answer-execution graph nodes."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict

import pandas as pd


class RetrievalState(TypedDict):
    question: str
    max_attempts: NotRequired[int]
    question_record: NotRequired[dict[str, Any]]
    filters: NotRequired[dict[str, list[str | int]]]
    semantic_query: NotRequired[str]
    candidates: NotRequired[list[dict[str, Any]]]
    reranked_tables: NotRequired[list[dict[str, Any]]]
    retrieved_tables: NotRequired[list[dict[str, Any]]]
    selector_diagnostics: NotRequired[dict[str, Any]]
    dataframes: NotRequired[dict[str, pd.DataFrame]]
    evidence_sources: NotRequired[dict[str, dict[str, str]]]
    alias_metadata: NotRequired[dict[str, dict[str, Any]]]
    rerank_contexts: NotRequired[dict[str, dict[str, Any]]]
    planning_inventory: NotRequired[list[dict[str, Any]]]
    generation_plan: NotRequired[dict[str, Any]]
    planned_context: NotRequired[dict[str, Any]]
    attempt: NotRequired[int]
    feedback: NotRequired[str]
    pandas_query: NotRequired[str]
    evidence_variables: NotRequired[list[str]]
    answer_record: NotRequired[dict[str, Any]]


__all__ = ["RetrievalState"]
