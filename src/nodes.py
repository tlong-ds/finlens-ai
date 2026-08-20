"""LangGraph nodes for retrieval and validated pandas answer execution."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from langgraph.graph import END
from langgraph.types import Command

from src.sandbox import run_code
from src.contracts import resolve_csv_path, validate_qdrant_payload
from src.helper import (
    concise_error,
    find_question,
    generator_feedback,
    numeric_result,
    ordered_unique,
    retry_or_exhausted,
    validator_feedback,
)
from src.llm import LLMResponseError, generate_structured
from src.prompt import (
    GENERATOR_SYSTEM_PROMPT,
    PARSE_SYSTEM_PROMPT,
    TICKER_RESOLUTION_SYSTEM_PROMPT,
    VALIDATOR_SYSTEM_PROMPT,
    build_generator_prompt,
    build_parse_prompt,
    build_ticker_resolution_prompt,
    build_validator_prompt,
)
from src.retrieval import rerank, retrieve
from src.routing import (
    QueryRoutingError,
    load_company_catalog,
    parse_years,
    reconcile_query_filters,
    resolve_tickers_conservatively,
    validate_llm_ticker_resolution,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SANDBOX_TIMEOUT_SECONDS = 5.0
_STRUCTURED_RESPONSE_ATTEMPTS = 2


def match_question_node(state: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the input to exactly one canonical ViFinQA question."""
    question = state.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must not be empty")

    max_attempts = state.get("max_attempts", 5)
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
        raise TypeError("max_attempts must be an integer")
    if not 1 <= max_attempts <= 5:
        raise ValueError("max_attempts must be between 1 and 5")

    question_record = find_question(question)
    return {
        "question": str(question_record["question"]),
        "question_record": question_record,
        "max_attempts": max_attempts,
        "attempt": 0,
        "feedback": "",
    }


def _resolve_tickers_with_llm(
    question: str,
    deterministic_candidates: list[dict[str, Any]],
    reason: str,
) -> list[str]:
    """Resolve an unresolved question against the bounded stock catalog."""
    canonical, _ = load_company_catalog()
    company_catalog = [
        {"ticker": ticker, "company_name": company}
        for ticker, company in sorted(canonical.items())
    ]
    feedback = ""
    last_error = ""
    for _ in range(_STRUCTURED_RESPONSE_ATTEMPTS):
        try:
            response = generate_structured(
                build_ticker_resolution_prompt(
                    question,
                    company_catalog,
                    feedback,
                    candidates=deterministic_candidates,
                    reason=reason,
                ),
                system_prompt=TICKER_RESOLUTION_SYSTEM_PROMPT,
            )
            tickers = validate_llm_ticker_resolution(
                response,
                question,
                canonical,
                allowed_tickers=[
                    str(candidate["ticker"])
                    for candidate in deterministic_candidates
                ] or None,
            )
            if not tickers:
                raise QueryRoutingError(
                    "LLM không xác định được doanh nghiệp chính trong câu hỏi"
                )
            return tickers
        except (LLMResponseError, QueryRoutingError) as error:
            last_error = concise_error(error)
            feedback = "Response trước không hợp lệ: " + last_error
    raise QueryRoutingError(
        "Không resolve được ticker bằng deterministic resolver hoặc LLM fallback: "
        + last_error
    )


def parse_query_node(state: Mapping[str, Any]) -> dict[str, Any]:
    """Parse and reconcile strict metadata filters from the canonical question."""
    question = str(state.get("question") or "")
    resolution, _ = resolve_tickers_conservatively(question)
    ticker_overrides: list[str]
    if resolution.status == "resolved":
        ticker_overrides = list(resolution.tickers)
    else:
        ticker_overrides = _resolve_tickers_with_llm(
            question,
            [
                {
                    "ticker": candidate.ticker,
                    "entity_text": candidate.entity_text,
                    "match_type": candidate.match_type,
                    "matched_variant": candidate.matched_variant,
                }
                for candidate in resolution.candidates
            ],
            resolution.reason,
        )
    if not parse_years(question):
        raise QueryRoutingError(
            "Không resolve được năm trong phạm vi 2015–2025; hệ thống không search global"
        )

    feedback = ""
    last_error = ""
    for _ in range(_STRUCTURED_RESPONSE_ATTEMPTS):
        try:
            raw_filters = generate_structured(
                build_parse_prompt(question, feedback),
                system_prompt=PARSE_SYSTEM_PROMPT,
            )
            filters, semantic_query = reconcile_query_filters(
                question,
                raw_filters,
                ticker_overrides=ticker_overrides,
            )
            break
        except (LLMResponseError, QueryRoutingError) as error:
            last_error = concise_error(error)
            feedback = "Response trước không hợp lệ: " + last_error
    else:
        raise LLMResponseError(
            "Không parse được metadata filter hợp lệ: " + last_error
        )
    logger.info("Question: %s", question)
    logger.info("Parsed filters: %s", filters)
    logger.info("Semantic query: %s", semantic_query)
    return {"filters": filters, "semantic_query": semantic_query}


def retrieve_tables_node(state: Mapping[str, Any]) -> dict[str, Any]:
    """Retrieve Top-N tables using the question and parsed filters."""
    candidates = retrieve(
        query_text=str(state.get("semantic_query") or ""),
        filters=state.get("filters", {}),
    )
    return {"candidates": candidates}


def rerank_tables_node(state: Mapping[str, Any]) -> dict[str, Any]:
    """Rerank candidates and return up to the configured maximum tables."""
    retrieved_tables = rerank(
        question=str(state.get("question") or ""),
        candidates=state.get("candidates", []),
    )
    logger.info(
        "Final table IDs: %s",
        [item.get("table_id") for item in retrieved_tables],
    )
    return {"retrieved_tables": retrieved_tables}


def load_tables_node(state: Mapping[str, Any]) -> dict[str, Any]:
    """Load retrieved CSV tables and describe them for pandas generation."""
    dataframes: dict[str, pd.DataFrame] = {}
    evidence_sources: dict[str, dict[str, str]] = {}
    descriptions: list[str] = []

    for index, candidate in enumerate(state.get("retrieved_tables", []), start=1):
        alias = f"df_{index}"
        metadata_value = candidate["metadata"]
        if not isinstance(metadata_value, Mapping):
            raise TypeError("retrieved table metadata must be an object")

        metadata = validate_qdrant_payload(metadata_value)
        table_id = metadata["table_id"]
        doc_id = metadata["doc_id"]
        start_line = metadata["start_line"]
        csv_file = resolve_csv_path(table_id, _PROJECT_ROOT)
        csv_path = csv_file.relative_to(_PROJECT_ROOT.resolve()).as_posix()
        dataframe = pd.read_csv(csv_file)

        dataframes[alias] = dataframe
        evidence_sources[alias] = {
            "csv_path": csv_path,
            "doc_id": doc_id,
            "relevant_table": f"{doc_id}|{start_line}",
        }
        schema = {
            "alias": alias,
            "metadata": metadata,
            "columns": [
                {"name": str(column), "dtype": str(dataframe[column].dtype)}
                for column in dataframe.columns
            ],
            "sample_rows": json.loads(
                dataframe.head(8).to_json(orient="records", force_ascii=False)
            ),
        }
        descriptions.append(json.dumps(schema, ensure_ascii=False, default=str))

    if not dataframes:
        raise RuntimeError("Retrieval returned no tables")
    return {
        "dataframes": dataframes,
        "evidence_sources": evidence_sources,
        "dataframe_description": "\n".join(descriptions),
    }


def generate_code_node(
    state: Mapping[str, Any],
) -> Command[Literal["validate_code", "generate_code", "execute_code"]]:
    """Generate one pandas candidate and increment the attempt counter."""
    attempt = int(state.get("attempt", 0)) + 1
    update: dict[str, Any] = {
        "attempt": attempt,
        "pandas_query": "",
        "evidence_variables": [],
    }
    generator_prompt = build_generator_prompt(
        question=str(state.get("question") or ""),
        dataframe_description=str(state.get("dataframe_description") or ""),
        feedback=str(state.get("feedback") or ""),
    )
    try:
        generated = generate_structured(
            generator_prompt,
            system_prompt=GENERATOR_SYSTEM_PROMPT,
        )
    except LLMResponseError as error:
        update["feedback"] = f"Generator call failed: {concise_error(error)}"
        return retry_or_exhausted(state, update, attempt=attempt)

    feedback = generator_feedback(generated)
    if feedback:
        update["feedback"] = feedback
        return retry_or_exhausted(state, update, attempt=attempt)

    evidence_variables = generated["evidence_variables"]
    evidence_sources = state.get("evidence_sources") or {}
    unknown_aliases = [
        alias for alias in evidence_variables if alias not in evidence_sources
    ]
    if unknown_aliases:
        update["feedback"] = "Unknown evidence variables: " + ", ".join(
            unknown_aliases
        )
        return retry_or_exhausted(state, update, attempt=attempt)

    update.update(
        {
            "feedback": "",
            "pandas_query": generated["pandas_query"],
            "evidence_variables": evidence_variables,
        }
    )
    return Command(update=update, goto="validate_code")


def validate_code_node(
    state: Mapping[str, Any],
) -> Command[Literal["execute_code", "generate_code"]]:
    """Ask the LLM to validate the generated pandas without executing it."""
    validator_prompt = build_validator_prompt(
        question=str(state.get("question") or ""),
        available_aliases=list(state.get("dataframes") or {}),
        dataframe_description=str(state.get("dataframe_description") or ""),
        pandas_query=str(state.get("pandas_query") or ""),
        evidence_variables=list(state.get("evidence_variables") or []),
    )
    try:
        validation = generate_structured(
            validator_prompt,
            system_prompt=VALIDATOR_SYSTEM_PROMPT,
        )
    except LLMResponseError as error:
        return retry_or_exhausted(
            state,
            {
                "feedback": f"Validator call failed: {concise_error(error)}",
            },
        )

    feedback = validator_feedback(validation)
    if feedback:
        return retry_or_exhausted(state, {"feedback": feedback})
    return Command(update={"feedback": ""}, goto="execute_code")


def execute_code_node(
    state: Mapping[str, Any],
) -> Command[Literal["generate_code", "__end__"]]:
    """Execute approved code and build the final answer for numeric success."""
    feedback = state.get("feedback") or ""
    attempt = int(state.get("attempt", 0))
    max_attempts = int(state.get("max_attempts", 5))
    if feedback and attempt >= max_attempts:
        raise RuntimeError(
            "Unable to produce a valid numeric result after "
            f"{attempt} attempts: {feedback}"
        )

    try:
        result = run_code(
            str(state.get("pandas_query") or ""),
            state.get("dataframes") or {},
            timeout_sec=_SANDBOX_TIMEOUT_SECONDS,
        )
    except (RuntimeError, TimeoutError, ValueError) as error:
        execution_feedback = f"Sandbox execution failed: {concise_error(error)}"
        if attempt >= max_attempts:
            raise RuntimeError(
                "Unable to produce a valid numeric result after "
                f"{attempt} attempts: {execution_feedback}"
            ) from error
        return retry_or_exhausted(
            state,
            {"feedback": execution_feedback},
        )

    answer, feedback = numeric_result(result)
    if feedback:
        if attempt >= max_attempts:
            raise RuntimeError(
                "Unable to produce a valid numeric result after "
                f"{attempt} attempts: {feedback}"
            )
        return retry_or_exhausted(state, {"feedback": feedback})

    aliases = ordered_unique(list(state.get("evidence_variables") or []))
    sources = state.get("evidence_sources") or {}
    question_record = state.get("question_record") or {}
    answer_record = {
        "id": question_record["id"],
        "question": state.get("question"),
        "answer": answer,
        "evidence": {alias: sources[alias]["csv_path"] for alias in aliases},
        "relevant_docs": ordered_unique(
            [sources[alias]["doc_id"] for alias in aliases]
        ),
        "relevant_tables": ordered_unique(
            [sources[alias]["relevant_table"] for alias in aliases]
        ),
        "pandas_query": state.get("pandas_query"),
    }
    return Command(
        update={"answer_record": answer_record, "feedback": ""},
        goto=END,
    )
