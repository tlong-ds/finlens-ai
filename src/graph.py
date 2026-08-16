"""Compiled LangGraph workflow for retrieval and answer execution."""

from __future__ import annotations

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
    parse_query_node,
    rerank_tables_node,
    retrieve_tables_node,
    validate_code_node,
)
from src.retrieval import TransientRetrievalError


class RetrievalState(TypedDict):
    """Values shared by retrieval and answer-execution graph nodes."""

    question: str
    max_attempts: NotRequired[int]
    question_record: NotRequired[dict[str, Any]]
    filters: NotRequired[dict[str, list[str | int]]]
    candidates: NotRequired[list[dict[str, Any]]]
    retrieved_tables: NotRequired[list[dict[str, Any]]]
    dataframes: NotRequired[dict[str, pd.DataFrame]]
    evidence_sources: NotRequired[dict[str, dict[str, str]]]
    dataframe_description: NotRequired[str]
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
builder.add_node("match_question", match_question_node)
builder.add_node("parse_query", parse_query_node, retry_policy=_LLM_RETRY_POLICY)
builder.add_node(
    "retrieve_tables",
    retrieve_tables_node,
    retry_policy=_QDRANT_RETRY_POLICY,
)
builder.add_node("rerank_tables", rerank_tables_node)
builder.add_node("load_tables", load_tables_node)
builder.add_node("generate_code", generate_code_node, retry_policy=_LLM_RETRY_POLICY)
builder.add_node("validate_code", validate_code_node, retry_policy=_LLM_RETRY_POLICY)
builder.add_node("execute_code", execute_code_node)

builder.add_edge(START, "match_question")
builder.add_edge("match_question", "parse_query")
builder.add_edge("parse_query", "retrieve_tables")
builder.add_edge("retrieve_tables", "rerank_tables")
builder.add_edge("rerank_tables", "load_tables")
builder.add_edge("load_tables", "generate_code")

graph = builder.compile()
