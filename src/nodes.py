"""LangGraph nodes for retrieval and pandas answer execution."""

from __future__ import annotations

import ast
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
)
from src.llm import LLMResponseError, generate_structured
from src.parser import parse_query_with_diagnostics
from src.prompt import GENERATOR_SYSTEM_PROMPT, build_generator_prompt
from src.retrieval import (
    RETRIEVAL_TOP_K,
    NoMatchingCandidatesError,
    rerank,
    retrieve,
)
from src.routing import QueryRoutingError

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SANDBOX_TIMEOUT_SECONDS = 5.0
_UNSUPPORTED_DATAFRAME_ATTRIBUTES = {"metadata", "attrs"}


def _unsupported_dataframe_attribute_feedback(
    code: str,
    aliases: set[str],
) -> str | None:
    """Reject generated access to provenance as if it were a DataFrame attribute.

    Table provenance belongs to the graph state and is supplied to the generator as
    ``alias_metadata``.  A pandas DataFrame has no stable ``metadata`` contract, and
    ``attrs`` is intentionally not populated when CSVs are loaded.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    dataframe_names = set(aliases)
    # Generated code sometimes assigns an alias to a shorter local name before
    # indexing it. Follow simple name-to-name assignments so that
    # ``table = df_1; table.metadata`` cannot bypass the contract.
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Name)
                and node.value.id in dataframe_names
            ):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id not in dataframe_names:
                    dataframe_names.add(target.id)
                    changed = True

    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if (
            isinstance(node.value, ast.Name)
            and node.value.id in dataframe_names
            and node.attr in _UNSUPPORTED_DATAFRAME_ATTRIBUTES
        ):
            return (
                f"Không dùng {node.value.id}.{node.attr}: DataFrame không mang "
                "metadata/provenance. Hãy lấy năm, ticker và loại báo cáo từ "
                "alias_metadata được cung cấp trong prompt."
            )
    return None


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


def parse_query_node(state: Mapping[str, Any]) -> dict[str, Any]:
    """Parse and reconcile strict metadata filters from the canonical question."""
    question = str(state.get("question") or "")
    question_record = state.get("question_record") or {}
    parsed = parse_query_with_diagnostics(
        question,
        question_id=question_record.get("id", "unknown"),
    )
    return {
        "filters": parsed["filters"],
        "semantic_query": parsed["semantic_query"],
    }


def retrieve_tables_node(state: Mapping[str, Any]) -> dict[str, Any]:
    """Retrieve balanced Top-N candidates from one metadata bucket per ticker."""
    query_text = str(state.get("semantic_query") or "")
    filters = dict(state.get("filters", {}) or {})
    raw_tickers = filters.get("ticker", [])
    if not isinstance(raw_tickers, list) or not raw_tickers or not all(
        isinstance(ticker, str) and ticker for ticker in raw_tickers
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
    alias_metadata: dict[str, dict[str, Any]] = {}
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
        "alias_metadata": alias_metadata,
        "dataframe_description": "\n".join(descriptions),
    }


def generate_code_node(
    state: Mapping[str, Any],
) -> Command[Literal["generate_code", "execute_code"]]:
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
        alias_metadata=state.get("alias_metadata") or {},
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

    attribute_feedback = _unsupported_dataframe_attribute_feedback(
        generated["pandas_query"],
        set(state.get("dataframes") or {}),
    )
    if attribute_feedback:
        update["feedback"] = attribute_feedback
        return retry_or_exhausted(state, update, attempt=attempt)

    update.update(
        {
            "feedback": "",
            "pandas_query": generated["pandas_query"],
            "evidence_variables": evidence_variables,
        }
    )
    return Command(update=update, goto="execute_code")


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
            alias_metadata=state.get("alias_metadata"),
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
