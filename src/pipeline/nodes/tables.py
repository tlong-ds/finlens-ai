"""Reranking, selection, loading, and planning nodes."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Literal

import pandas as pd
from langgraph.types import Command

from src.config import Settings
from src.contracts import resolve_csv_path, validate_qdrant_payload
from src.generation.planning import (
    aliases_declared_in_plan,
    build_planning_inventory,
    hydrate_planned_rows,
)
from src.generation.prompts import (
    PLANNER_SYSTEM_PROMPT,
    build_planner_prompt,
)
from src.providers.llm import LLMResponseError, generate_structured
from src.retrieval.reranking import rerank_with_fpt
from src.retrieval.selection import select_tables_with_diagnostics

logger = logging.getLogger(__name__)


def rerank_tables_node(
    state: Mapping[str, Any], *, settings: Settings
) -> dict[str, Any]:
    """Use FPT BGE-M3 to rerank up to 80 candidates into a top-20 list."""
    reranked_tables = rerank_with_fpt(
        question=str(state.get("question") or ""),
        candidates=state.get("candidates", []),
        settings=settings,
    )
    logger.info(
        "FPT top table IDs: %s",
        [item.get("table_id") for item in reranked_tables],
    )
    return {"reranked_tables": reranked_tables}


def select_tables_node(
    state: Mapping[str, Any], *, settings: Settings
) -> dict[str, Any]:
    """Select exact tables and retain bounded coverage diagnostics."""
    retrieved_tables, selector_diagnostics = select_tables_with_diagnostics(
        question=str(state.get("question") or ""),
        candidates=state.get("reranked_tables", []),
        settings=settings,
    )
    logger.info(
        "Planner table IDs: %s",
        [item.get("table_id") for item in retrieved_tables],
    )
    return {
        "retrieved_tables": retrieved_tables,
        "selector_diagnostics": selector_diagnostics,
    }


def load_tables_node(state: Mapping[str, Any], *, settings: Settings) -> dict[str, Any]:
    """Load every reranked CSV table for planning and sandbox execution."""
    dataframes: dict[str, pd.DataFrame] = {}
    evidence_sources: dict[str, dict[str, Any]] = {}
    alias_metadata: dict[str, dict[str, Any]] = {}
    rerank_contexts: dict[str, dict[str, Any]] = {}

    for index, candidate in enumerate(state.get("retrieved_tables", []), start=1):
        alias = f"df_{index}"
        metadata_value = candidate["metadata"]
        if not isinstance(metadata_value, Mapping):
            raise TypeError("retrieved table metadata must be an object")

        metadata = validate_qdrant_payload(metadata_value)
        table_id = metadata["table_id"]
        doc_id = metadata["doc_id"]
        start_line = metadata["start_line"]
        csv_file = resolve_csv_path(table_id, settings.project_root)
        csv_path = csv_file.relative_to(settings.project_root).as_posix()
        dataframe = pd.read_csv(csv_file)

        dataframes[alias] = dataframe
        raw_rerank_context = candidate.get("rerank_context")
        rerank_contexts[alias] = (
            dict(raw_rerank_context) if isinstance(raw_rerank_context, Mapping) else {}
        )
        evidence_sources[alias] = {
            "csv_path": csv_path,
            "doc_id": doc_id,
            "relevant_table": f"{doc_id}|{start_line}",
        }
        # Keep provenance outside the DataFrame. CSV round-tripping deliberately
        # discards DataFrame.attrs, so attaching metadata to pandas objects would
        # be unreliable in the sandbox as well.
        alias_metadata[alias] = {
            "ticker": metadata["ticker"],
            "company_name": metadata["company_name"],
            "year": metadata["year"],
            "report_type": metadata["report_type"],
            "table_type": metadata["table_type"],
        }
    if not dataframes:
        raise RuntimeError("Retrieval returned no tables")
    return {
        "dataframes": dataframes,
        "evidence_sources": evidence_sources,
        "alias_metadata": alias_metadata,
        "rerank_contexts": rerank_contexts,
    }


def plan_generation_context_node(
    state: Mapping[str, Any],
    *,
    settings: Settings,
) -> Command[Literal["generate_code"]]:
    """Plan how to answer the user question from selector-chosen tables."""
    dataframes = state.get("dataframes") or {}
    alias_metadata = state.get("alias_metadata") or {}
    inventory = build_planning_inventory(
        dataframes,
        alias_metadata,
        state.get("rerank_contexts") or {},
    )
    question = str(state.get("question") or "")
    selected_aliases = set(dataframes)
    generation_plan: dict[str, Any] | None = None
    response_error: Exception | None = None
    feedback = ""
    for response_attempt in range(1, 4):
        try:
            candidate_plan = generate_structured(
                build_planner_prompt(question, inventory, feedback),
                settings=settings,
                system_prompt=PLANNER_SYSTEM_PROMPT,
                native=False,
            )
            planned_aliases = aliases_declared_in_plan(candidate_plan, dataframes)
            missing = selected_aliases - planned_aliases
            if missing:
                feedback = (
                    f"Kế hoạch phải khai báo bằng chứng cho tất cả các bảng đã chọn "
                    f"({', '.join(sorted(selected_aliases))}). Các bảng còn thiếu hoặc "
                    f"chưa có row hợp lệ kèm purpose: {', '.join(sorted(missing))}."
                )
                continue
            generation_plan = candidate_plan
            break
        except LLMResponseError as error:
            response_error = error
            feedback = (
                "Phản hồi trước không phải JSON object hợp lệ. Chỉ trả về JSON đúng "
                "schema được yêu cầu."
            )
    if generation_plan is None:
        if response_error is not None:
            raise response_error
        raise ValueError(
            f"Planner không thể tạo kế hoạch hợp lệ cho tất cả các bảng đã chọn: {feedback}"
        )
    selected_rows = hydrate_planned_rows(generation_plan, dataframes, inventory)
    return Command(
        update={
            "planning_inventory": inventory,
            "generation_plan": generation_plan,
            "planned_context": {
                "inventory": inventory,
                "alias_metadata": dict(alias_metadata),
                "selected_rows": selected_rows,
            },
            "feedback": "",
            "pandas_query": "",
            "evidence_variables": [],
        },
        goto="generate_code",
    )
