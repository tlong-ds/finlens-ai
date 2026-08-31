"""Graph construction with explicit dependency injection."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

from langgraph.graph import START, StateGraph
from langgraph.types import RetryPolicy

from src.config import Settings
from src.pipeline.nodes.answer import execute_code_node, generate_code_node
from src.pipeline.nodes.question import (
    match_question_node,
    parse_query_node,
    retrieve_tables_node,
)
from src.pipeline.nodes.tables import (
    load_tables_node,
    plan_generation_context_node,
    rerank_tables_node,
    select_tables_node,
)
from src.pipeline.state import RetrievalState
from src.providers.llm import LLMTransientError
from src.retrieval.dense import TransientRetrievalError

Node = Callable[[Any], Any]


@dataclass(frozen=True, slots=True)
class PipelineDependencies:
    """Node actions supplied to ``build_graph`` for isolated tests or adapters."""

    match_question: Node
    parse_query: Node
    retrieve_tables: Node
    rerank_tables: Node
    select_tables: Node
    load_tables: Node
    plan_generation_context: Node
    generate_code: Node
    execute_code: Node

    @classmethod
    def from_settings(cls, settings: Settings) -> "PipelineDependencies":
        """Bind the validated settings once at the application boundary."""

        def bind(node: Node) -> Node:
            return partial(node, settings=settings)

        return cls(
            match_question=bind(match_question_node),
            parse_query=bind(parse_query_node),
            retrieve_tables=bind(retrieve_tables_node),
            rerank_tables=bind(rerank_tables_node),
            select_tables=bind(select_tables_node),
            load_tables=bind(load_tables_node),
            plan_generation_context=bind(plan_generation_context_node),
            generate_code=bind(generate_code_node),
            execute_code=bind(execute_code_node),
        )


def build_graph(
    settings: Settings,
    dependencies: PipelineDependencies,
) -> Any:
    """Compile the authoritative pipeline using explicit runtime dependencies.

    ``settings`` is validated at the caller boundary. Provider adapters consume it
    when constructing ``dependencies``; graph construction itself performs no I/O.
    """
    if not isinstance(settings, Settings):
        raise TypeError("settings must be a Settings instance")
    if not isinstance(dependencies, PipelineDependencies):
        raise TypeError("dependencies must be a PipelineDependencies instance")

    builder = StateGraph(RetrievalState)
    llm_retry = RetryPolicy(max_attempts=3, retry_on=LLMTransientError)
    qdrant_retry = RetryPolicy(max_attempts=3, retry_on=TransientRetrievalError)

    def add(name: str, action: Node, retry: RetryPolicy | None = None) -> None:
        kwargs: dict[str, Any] = {}
        if retry is not None:
            keyword = (
                "retry_policy"
                if "retry_policy" in inspect.signature(builder.add_node).parameters
                else "retry"
            )
            kwargs[keyword] = retry
        builder.add_node(name, action, **kwargs)

    add("match_question", dependencies.match_question)
    add("parse_query", dependencies.parse_query, llm_retry)
    add("retrieve_tables", dependencies.retrieve_tables, qdrant_retry)
    add("rerank_tables", dependencies.rerank_tables)
    add("select_tables", dependencies.select_tables, llm_retry)
    add("load_tables", dependencies.load_tables)
    add("plan_generation_context", dependencies.plan_generation_context, llm_retry)
    add("generate_code", dependencies.generate_code, llm_retry)
    add("execute_code", dependencies.execute_code)
    builder.add_edge(START, "match_question")
    builder.add_edge("match_question", "parse_query")
    builder.add_edge("parse_query", "retrieve_tables")
    builder.add_edge("retrieve_tables", "rerank_tables")
    builder.add_edge("rerank_tables", "select_tables")
    builder.add_edge("select_tables", "load_tables")
    builder.add_edge("load_tables", "plan_generation_context")
    return builder.compile()


graph = build_graph(
    (default_settings := Settings.from_env(validate=False)),
    PipelineDependencies.from_settings(default_settings),
)
