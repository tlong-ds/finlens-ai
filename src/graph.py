"""Compiled LangGraph workflow for retrieval and answer execution."""

from __future__ import annotations

import inspect
from typing import Any, NotRequired, TypedDict

import pandas as pd
from langgraph.graph import START, StateGraph
from langgraph.types import RetryPolicy

from src.llm import LLMTransientError
from src.nodes import (
    execute_code_node,
    generate_code_node,
    load_tables_node,
    match_question_node,
    materialize_buckets_node,
    parse_query_node,
    plan_generation_context_node,
    rerank_bucket_tables_node,
    retrieve_bucket_tables_node,
    rewrite_bucket_queries_node,
    select_bucket_tables_node,
    validate_table_coverage_node,
)
from src.retrieval import TransientRetrievalError


class RetrievalState(TypedDict):
    """Values shared by retrieval and answer-execution graph nodes."""

    question: str
    max_attempts: NotRequired[int]
    question_record: NotRequired[dict[str, Any]]
    filters: NotRequired[dict[str, list[str | int]]]
    semantic_query: NotRequired[str]
    bucket_specs: NotRequired[list[dict[str, Any]]]
    bucket_states: NotRequired[dict[str, dict[str, Any]]]
    active_bucket_keys: NotRequired[list[str]]
    retrieval_repair_round: NotRequired[int]
    coverage_validation: NotRequired[dict[str, Any]]
    validation_history: NotRequired[list[dict[str, Any]]]
    bucket_pipeline_metrics: NotRequired[dict[str, int]]
    candidates: NotRequired[list[dict[str, Any]]]
    reranked_tables: NotRequired[list[dict[str, Any]]]
    retrieved_tables: NotRequired[list[dict[str, Any]]]
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


_LLM_RETRY_POLICY = RetryPolicy(max_attempts=3, retry_on=LLMTransientError)
_QDRANT_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    retry_on=TransientRetrievalError,
)


builder = StateGraph(RetrievalState)


def _add_node(name: str, action: Any, retry_policy: RetryPolicy | None = None) -> None:
    """Support both LangGraph's ``retry`` and ``retry_policy`` keyword names."""
    kwargs: dict[str, Any] = {}
    if retry_policy is not None:
        parameters = inspect.signature(builder.add_node).parameters
        keyword = "retry_policy" if "retry_policy" in parameters else "retry"
        kwargs[keyword] = retry_policy
    builder.add_node(name, action, **kwargs)


_add_node("match_question", match_question_node)
_add_node("parse_query", parse_query_node, _LLM_RETRY_POLICY)
_add_node("materialize_buckets", materialize_buckets_node)
_add_node("rewrite_bucket_queries", rewrite_bucket_queries_node, _LLM_RETRY_POLICY)
_add_node("retrieve_bucket_tables", retrieve_bucket_tables_node, _QDRANT_RETRY_POLICY)
_add_node("rerank_bucket_tables", rerank_bucket_tables_node)
_add_node("select_bucket_tables", select_bucket_tables_node, _LLM_RETRY_POLICY)
_add_node("validate_table_coverage", validate_table_coverage_node, _LLM_RETRY_POLICY)
_add_node("load_tables", load_tables_node)
_add_node(
    "plan_generation_context",
    plan_generation_context_node,
    _LLM_RETRY_POLICY,
)
_add_node("generate_code", generate_code_node, _LLM_RETRY_POLICY)
_add_node("execute_code", execute_code_node)

builder.add_edge(START, "match_question")
builder.add_edge("match_question", "parse_query")
builder.add_edge("parse_query", "materialize_buckets")
builder.add_edge("materialize_buckets", "rewrite_bucket_queries")
builder.add_edge("rewrite_bucket_queries", "retrieve_bucket_tables")
builder.add_edge("retrieve_bucket_tables", "rerank_bucket_tables")
builder.add_edge("rerank_bucket_tables", "select_bucket_tables")
builder.add_edge("select_bucket_tables", "validate_table_coverage")
builder.add_edge("load_tables", "plan_generation_context")

graph = builder.compile()
