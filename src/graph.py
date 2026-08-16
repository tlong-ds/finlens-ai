"""Level 1 LangGraph workflow definition."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from src.nodes import parse_query_node, rerank_tables_node, retrieve_tables_node


class RetrievalState(TypedDict):
    """Only values shared between graph nodes."""

    question: str
    filters: NotRequired[dict[str, list[str | int]]]
    candidates: NotRequired[list[dict[str, Any]]]
    retrieved_tables: NotRequired[list[dict[str, Any]]]


builder = StateGraph(RetrievalState)
builder.add_node("parse_query", parse_query_node)
builder.add_node("retrieve_tables", retrieve_tables_node)
builder.add_node("rerank_tables", rerank_tables_node)

builder.add_edge(START, "parse_query")
builder.add_edge("parse_query", "retrieve_tables")
builder.add_edge("retrieve_tables", "rerank_tables")
builder.add_edge("rerank_tables", END)

graph = builder.compile()
