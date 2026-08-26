"""LangGraph nodes for retrieval and pandas answer execution."""

from __future__ import annotations

import ast
import logging
import os
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from itertools import product
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
from src.llm import LLMResponseError, LLMTransientError, generate_structured
from src.parser import parse_query_with_diagnostics
from src.retrieval import (
    BUCKET_FINALIST_POOL_MAX,
    BUCKET_RETRIEVAL_TOP_N,
    BUCKET_RERANK_DEFAULT_TOP_N,
    RETRIEVAL_TOP_K,
    NoMatchingCandidatesError,
    rerank_with_fpt,
    retrieve,
    select_bucket_tables_with_diagnostics,
    select_tables,
)
from src.planning import (
    build_planning_inventory,
    generated_evidence_variables,
    generated_context_coverage_feedback,
    generated_semantic_feedback,
    generation_plan_feedback,
    normalize_generated_code,
    normalize_generated_selectors,
    normalize_generated_semantics,
    parse_malformed_generator_json,
    generated_rounding_feedback,
    hydrate_planned_rows,
)
from src.prompt import (
    BUCKET_QUERY_REWRITE_SYSTEM_PROMPT,
    COVERAGE_VALIDATOR_SYSTEM_PROMPT,
    GENERATOR_RESPONSE_SCHEMA,
    GENERATOR_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    build_bucket_query_rewrite_prompt,
    build_coverage_validator_prompt,
    build_generator_prompt,
    build_planner_prompt,
)
from src.routing import QueryRoutingError

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SANDBOX_TIMEOUT_SECONDS = 5.0
_UNSUPPORTED_DATAFRAME_ATTRIBUTES = {"metadata", "attrs"}
_BUCKET_WORKERS = 2
_MAX_RETRIEVAL_REPAIR_ROUNDS = 2
_BUCKET_REPAIR_RERANK_DEFAULT_TOP_N = 20


def _bucket_worker_count(active_count: int) -> int:
    raw_workers = os.getenv("FINLENS_BUCKET_WORKERS", str(_BUCKET_WORKERS))
    try:
        workers = int(raw_workers)
    except ValueError as exc:
        raise ValueError("FINLENS_BUCKET_WORKERS must be an integer") from exc
    if workers < 1 or workers > 8:
        raise ValueError("FINLENS_BUCKET_WORKERS must be between 1 and 8")
    return min(workers, active_count)


class TableContextUnsolvableError(RuntimeError):
    """Raised when bounded bucket repair cannot produce sufficient evidence."""

    def __init__(self, message: str, diagnostics: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


def _numeric_result_error(attempt: int, feedback: str) -> str:
    """Format a terminal numeric-result error with correct singular grammar."""
    noun = "attempt" if attempt == 1 else "attempts"
    return (
        f"Unable to produce a valid numeric result after {attempt} {noun}: {feedback}"
    )


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

    max_attempts = state.get("max_attempts", 1)
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


def _bucket_specs_by_key(state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw_specs = state.get("bucket_specs") or []
    if not isinstance(raw_specs, list):
        raise TypeError("bucket_specs must be a list")
    specs: dict[str, dict[str, Any]] = {}
    for raw_spec in raw_specs:
        if not isinstance(raw_spec, Mapping):
            raise TypeError("bucket spec must be an object")
        spec = dict(raw_spec)
        key = spec.get("bucket_key")
        if not isinstance(key, str) or not key or key in specs:
            raise ValueError("bucket_key must be unique and non-empty")
        specs[key] = spec
    if not specs:
        raise ValueError("bucket_specs must not be empty")
    return specs


def _bucket_runtime_map(state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw_states = state.get("bucket_states") or {}
    if not isinstance(raw_states, Mapping):
        raise TypeError("bucket_states must be an object")
    return {
        str(key): dict(value)
        for key, value in raw_states.items()
        if isinstance(key, str) and isinstance(value, Mapping)
    }


def _active_bucket_keys(
    state: Mapping[str, Any],
    specs: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    raw_keys = state.get("active_bucket_keys") or []
    if not isinstance(raw_keys, list):
        raise TypeError("active_bucket_keys must be a list")
    keys = list(dict.fromkeys(str(key) for key in raw_keys))
    if not keys or any(key not in specs for key in keys):
        raise ValueError("active_bucket_keys contains no valid bucket")
    return keys


def _pipeline_metrics(state: Mapping[str, Any]) -> dict[str, int]:
    raw_metrics = state.get("bucket_pipeline_metrics") or {}
    if not isinstance(raw_metrics, Mapping):
        raw_metrics = {}
    names = (
        "rewrite_llm_calls",
        "qdrant_calls",
        "reranker_calls",
        "selector_llm_calls",
        "validator_llm_calls",
    )
    return {name: int(raw_metrics.get(name, 0)) for name in names}


def materialize_buckets_node(state: Mapping[str, Any]) -> dict[str, Any]:
    """Expand flat parser filters into stable ticker/year/report buckets."""
    filters = state.get("filters") or {}
    if not isinstance(filters, Mapping):
        raise QueryRoutingError("filters phải là một object")
    tickers = filters.get("ticker")
    years = filters.get("year")
    report_types = filters.get("report_type")
    if (
        not isinstance(tickers, list)
        or not tickers
        or not all(isinstance(ticker, str) and ticker for ticker in tickers)
    ):
        raise QueryRoutingError("Filter ticker phải là một mảng chuỗi không rỗng")
    if (
        not isinstance(years, list)
        or not years
        or any(isinstance(year, bool) or not isinstance(year, int) for year in years)
    ):
        raise QueryRoutingError("Filter year phải là một mảng số nguyên không rỗng")
    if (
        not isinstance(report_types, list)
        or len(report_types) != 1
        or not isinstance(report_types[0], str)
    ):
        raise QueryRoutingError("Filter report_type phải chứa đúng một giá trị")

    semantic_query = str(state.get("semantic_query") or "").strip()
    if not semantic_query:
        raise QueryRoutingError("semantic_query không được rỗng")
    specs: list[dict[str, Any]] = []
    runtimes: dict[str, dict[str, Any]] = {}
    combinations = product(
        list(dict.fromkeys(tickers)),
        list(dict.fromkeys(years)),
        list(dict.fromkeys(report_types)),
    )
    for index, (ticker, year, report_type) in enumerate(combinations, start=1):
        bucket_key = f"b{index:02d}"
        spec = {
            "bucket_key": bucket_key,
            "ticker": ticker,
            "year": year,
            "report_type": report_type,
            "filters": {
                "ticker": [ticker],
                "year": [year],
                "report_type": [report_type],
            },
        }
        specs.append(spec)
        runtimes[bucket_key] = {
            "bucket_key": bucket_key,
            "query": semantic_query,
            "query_round": 0,
            "filter_relaxed": False,
            "effective_filters": dict(spec["filters"]),
            "latest_candidates": [],
            "finalists": [],
            "selected_tables": [],
            "selector_diagnostics": {},
            "status": "pending",
            "feedback": "",
        }
    if not specs:
        raise QueryRoutingError("Không materialize được metadata bucket")
    return {
        "bucket_specs": specs,
        "bucket_states": runtimes,
        "active_bucket_keys": [spec["bucket_key"] for spec in specs],
        "retrieval_repair_round": 0,
        "validation_history": [],
        "bucket_pipeline_metrics": _pipeline_metrics({}),
    }


def _rewrite_one_bucket(
    question: str,
    semantic_query: str,
    spec: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> str:
    prompt = build_bucket_query_rewrite_prompt(
        question,
        semantic_query,
        spec,
        feedback=str(runtime.get("feedback") or ""),
        previous_query=str(runtime.get("query") or ""),
    )
    response = generate_structured(
        prompt,
        system_prompt=BUCKET_QUERY_REWRITE_SYSTEM_PROMPT,
        native=False,
    )
    if set(response) != {"search_query"}:
        raise LLMResponseError("Bucket rewrite phải trả đúng key search_query")
    query = response.get("search_query")
    if not isinstance(query, str) or not query.strip():
        raise LLMResponseError("Bucket rewrite trả search_query rỗng")
    return " ".join(query.split())[:1_000]


def rewrite_bucket_queries_node(state: Mapping[str, Any]) -> dict[str, Any]:
    """Rewrite active bucket queries independently, bypassing initial single bucket."""
    specs = _bucket_specs_by_key(state)
    runtimes = _bucket_runtime_map(state)
    active_keys = _active_bucket_keys(state, specs)
    repair_round = int(state.get("retrieval_repair_round", 0))
    semantic_query = str(state.get("semantic_query") or "").strip()
    question = str(state.get("question") or "")
    bypass = len(specs) == 1 and repair_round == 0
    rewritten: dict[str, str] = {}
    if bypass:
        rewritten[active_keys[0]] = semantic_query
    else:
        with ThreadPoolExecutor(
            max_workers=_bucket_worker_count(len(active_keys)),
            thread_name_prefix="bucket-rewrite",
        ) as executor:
            futures = {
                key: executor.submit(
                    _rewrite_one_bucket,
                    question,
                    semantic_query,
                    specs[key],
                    runtimes[key],
                )
                for key in active_keys
            }
            rewritten = {key: future.result() for key, future in futures.items()}
    for key in active_keys:
        runtime = runtimes[key]
        runtime.update(
            {
                "query": rewritten[key],
                "query_round": repair_round,
                "status": "query_ready",
            }
        )
    metrics = _pipeline_metrics(state)
    metrics["rewrite_llm_calls"] += 0 if bypass else len(active_keys)
    return {
        "bucket_states": runtimes,
        "bucket_pipeline_metrics": metrics,
    }


def _retrieve_one_bucket(
    query: str,
    spec: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, list[str | int]], bool, int]:
    raw_filters = spec.get("filters")
    if not isinstance(raw_filters, Mapping):
        raise QueryRoutingError("Bucket filters phải là một object")
    exact_filters = {str(key): list(values) for key, values in raw_filters.items()}
    calls = 1
    try:
        candidates = retrieve(
            query_text=query,
            filters=exact_filters,
            top_n=BUCKET_RETRIEVAL_TOP_N,
        )
        effective_filters = exact_filters
        relaxed = False
    except NoMatchingCandidatesError:
        if not exact_filters.get("report_type"):
            raise
        effective_filters = dict(exact_filters)
        effective_filters.pop("report_type", None)
        calls += 1
        candidates = retrieve(
            query_text=query,
            filters=effective_filters,
            top_n=BUCKET_RETRIEVAL_TOP_N,
        )
        relaxed = True
    validated: list[dict[str, Any]] = []
    for raw_candidate in candidates:
        candidate = dict(raw_candidate)
        metadata = candidate.get("metadata")
        if not isinstance(metadata, Mapping):
            raise TypeError("Bucket retrieval candidate metadata must be an object")
        if metadata.get("ticker") != spec.get("ticker") or metadata.get("year") != spec.get("year"):
            raise RuntimeError("Qdrant trả candidate ngoài ticker/year bucket")
        candidate["bucket_key"] = spec["bucket_key"]
        candidate["bucket_query"] = query
        validated.append(candidate)
    return validated, effective_filters, relaxed, calls


def retrieve_bucket_tables_node(state: Mapping[str, Any]) -> dict[str, Any]:
    """Run one exact Qdrant query for each active metadata bucket."""
    specs = _bucket_specs_by_key(state)
    runtimes = _bucket_runtime_map(state)
    active_keys = _active_bucket_keys(state, specs)
    with ThreadPoolExecutor(
        max_workers=_bucket_worker_count(len(active_keys)),
        thread_name_prefix="bucket-retrieve",
    ) as executor:
        futures = {
            key: executor.submit(
                _retrieve_one_bucket,
                str(runtimes[key].get("query") or ""),
                specs[key],
            )
            for key in active_keys
        }
        results = {key: future.result() for key, future in futures.items()}
    qdrant_calls = 0
    for key in active_keys:
        candidates, effective_filters, relaxed, calls = results[key]
        qdrant_calls += calls
        runtimes[key].update(
            {
                "latest_candidates": candidates,
                "effective_filters": effective_filters,
                "filter_relaxed": relaxed,
                "status": "retrieved",
            }
        )
    metrics = _pipeline_metrics(state)
    metrics["qdrant_calls"] += qdrant_calls
    return {
        "bucket_states": runtimes,
        "bucket_pipeline_metrics": metrics,
    }


def _bucket_rerank_depth() -> int:
    raw_depth = os.getenv(
        "FINLENS_BUCKET_RERANK_TOP_N",
        str(BUCKET_RERANK_DEFAULT_TOP_N),
    )
    try:
        depth = int(raw_depth)
    except ValueError as exc:
        raise ValueError("FINLENS_BUCKET_RERANK_TOP_N must be an integer") from exc
    if depth not in {3, 5, 8, 10, 15, 20}:
        raise ValueError(
            "FINLENS_BUCKET_RERANK_TOP_N must be one of 3, 5, 8, 10, 15, 20"
        )
    return depth


def _bucket_repair_rerank_depth() -> int:
    raw_depth = os.getenv(
        "FINLENS_BUCKET_REPAIR_RERANK_TOP_N",
        str(_BUCKET_REPAIR_RERANK_DEFAULT_TOP_N),
    )
    try:
        depth = int(raw_depth)
    except ValueError as exc:
        raise ValueError(
            "FINLENS_BUCKET_REPAIR_RERANK_TOP_N must be an integer"
        ) from exc
    if depth not in {10, 15, 20}:
        raise ValueError(
            "FINLENS_BUCKET_REPAIR_RERANK_TOP_N must be one of 10, 15, 20"
        )
    return depth


def _max_retrieval_repair_rounds() -> int:
    raw_rounds = os.getenv(
        "FINLENS_MAX_RETRIEVAL_REPAIRS",
        str(_MAX_RETRIEVAL_REPAIR_ROUNDS),
    )
    try:
        rounds = int(raw_rounds)
    except ValueError as exc:
        raise ValueError(
            "FINLENS_MAX_RETRIEVAL_REPAIRS must be an integer"
        ) from exc
    if rounds < 0 or rounds > _MAX_RETRIEVAL_REPAIR_ROUNDS:
        raise ValueError(
            "FINLENS_MAX_RETRIEVAL_REPAIRS must be between 0 and 2"
        )
    return rounds


def _coverage_validator_disabled() -> bool:
    return os.getenv("FINLENS_DISABLE_COVERAGE_VALIDATOR", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _semantic_validator_fail_closed() -> bool:
    return os.getenv(
        "FINLENS_FAIL_CLOSED_SEMANTIC_VALIDATOR", ""
    ).strip().lower() in {"1", "true", "yes"}


def _merge_bucket_finalists(
    newest: list[dict[str, Any]],
    previous: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_candidate in [*newest, *previous]:
        candidate = dict(raw_candidate)
        table_id = str(candidate.get("table_id") or "")
        if not table_id or table_id in seen:
            continue
        seen.add(table_id)
        merged.append(candidate)
        if len(merged) == BUCKET_FINALIST_POOL_MAX:
            break
    return merged


def rerank_bucket_tables_node(state: Mapping[str, Any]) -> dict[str, Any]:
    """Rerank each active bucket independently and retain a bounded union."""
    specs = _bucket_specs_by_key(state)
    runtimes = _bucket_runtime_map(state)
    active_keys = _active_bucket_keys(state, specs)
    repair_round = int(state.get("retrieval_repair_round", 0))
    depth = (
        _bucket_repair_rerank_depth()
        if repair_round > 0
        else _bucket_rerank_depth()
    )
    with ThreadPoolExecutor(
        max_workers=_bucket_worker_count(len(active_keys)),
        thread_name_prefix="bucket-rerank",
    ) as executor:
        futures = {
            key: executor.submit(
                rerank_with_fpt,
                str(runtimes[key].get("query") or ""),
                list(runtimes[key].get("latest_candidates") or []),
                top_n=depth,
            )
            for key in active_keys
        }
        reranked = {key: future.result() for key, future in futures.items()}
    for key in active_keys:
        newest = []
        for raw_candidate in reranked[key]:
            candidate = dict(raw_candidate)
            candidate["bucket_key"] = key
            candidate["bucket_query"] = runtimes[key].get("query")
            candidate["bucket_rerank_round"] = repair_round
            newest.append(candidate)
        previous = list(runtimes[key].get("finalists") or [])
        if not previous and runtimes[key].get("selected_tables"):
            previous = list(runtimes[key]["selected_tables"])
        runtimes[key].update(
            {
                "finalists": _merge_bucket_finalists(newest, previous),
                "status": "reranked",
                "rerank_depth": depth,
            }
        )
    metrics = _pipeline_metrics(state)
    metrics["reranker_calls"] += len(active_keys)
    return {
        "bucket_states": runtimes,
        "bucket_pipeline_metrics": metrics,
    }


def select_bucket_tables_node(state: Mapping[str, Any]) -> dict[str, Any]:
    """Run exactly one selector LLM call for every active bucket."""
    specs = _bucket_specs_by_key(state)
    runtimes = _bucket_runtime_map(state)
    active_keys = _active_bucket_keys(state, specs)
    question = str(state.get("question") or "")
    with ThreadPoolExecutor(
        max_workers=_bucket_worker_count(len(active_keys)),
        thread_name_prefix="bucket-selector",
    ) as executor:
        futures = {
            key: executor.submit(
                select_bucket_tables_with_diagnostics,
                question,
                specs[key],
                list(runtimes[key].get("finalists") or []),
            )
            for key in active_keys
        }
        selections = {key: future.result() for key, future in futures.items()}
    for key in active_keys:
        selected, diagnostics = selections[key]
        compact_diagnostics = {
            name: diagnostics.get(name)
            for name in (
                "bucket_key",
                "concepts",
                "covered_concepts_by_key",
                "uncovered_concept_keys",
                "invalid_values",
                "selection_field",
                "fallback_used",
                "llm_selected_keys",
                "selected_keys",
                "policy_added_keys",
                "selection_sources",
            )
        }
        runtimes[key].update(
            {
                "selected_tables": selected,
                "selector_diagnostics": compact_diagnostics,
                "status": "selected",
            }
        )
    metrics = _pipeline_metrics(state)
    metrics["selector_llm_calls"] += len(active_keys)
    return {
        "bucket_states": runtimes,
        "bucket_pipeline_metrics": metrics,
    }


def _deterministic_coverage_failures(
    specs: Mapping[str, Mapping[str, Any]],
    runtimes: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    for key, spec in specs.items():
        runtime = runtimes.get(key, {})
        selected = runtime.get("selected_tables") or []
        if not isinstance(selected, list) or not selected:
            failures.append(
                {
                    "bucket_key": key,
                    "concept": "evidence table",
                    "role": "direct",
                    "reason": "bucket không có selected table",
                    "suggested_query": str(runtime.get("query") or ""),
                }
            )
            continue
        for candidate in selected:
            metadata = candidate.get("metadata") if isinstance(candidate, Mapping) else None
            if not isinstance(metadata, Mapping):
                failures.append(
                    {
                        "bucket_key": key,
                        "concept": "valid table metadata",
                        "role": "direct",
                        "reason": "selected table thiếu metadata",
                        "suggested_query": str(runtime.get("query") or ""),
                    }
                )
                break
            if metadata.get("ticker") != spec.get("ticker") or metadata.get("year") != spec.get("year"):
                failures.append(
                    {
                        "bucket_key": key,
                        "concept": "đúng ticker và năm",
                        "role": "direct",
                        "reason": "selected table nằm ngoài bucket scope",
                        "suggested_query": str(runtime.get("query") or ""),
                    }
                )
                break
    return failures


def _validator_bucket_payload(
    spec: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    selected_tables: list[dict[str, Any]] = []
    for raw_candidate in runtime.get("selected_tables") or []:
        candidate = dict(raw_candidate)
        metadata = candidate.get("metadata") or {}
        selected_tables.append(
            {
                "table_id": candidate.get("table_id"),
                "table_type": metadata.get("table_type"),
                "actual_report_type": metadata.get("report_type"),
                "covered_concept_keys": candidate.get("covered_concept_keys", []),
                "context": candidate.get("rerank_context", {}),
            }
        )
    diagnostics = runtime.get("selector_diagnostics") or {}
    return {
        "bucket_key": spec["bucket_key"],
        "ticker": spec["ticker"],
        "year": spec["year"],
        "report_type": spec["report_type"],
        "filter_relaxed": bool(runtime.get("filter_relaxed")),
        "search_query": runtime.get("query"),
        "selector_concepts": (
            diagnostics.get("concepts", []) if isinstance(diagnostics, Mapping) else []
        ),
        "selected_tables": selected_tables,
    }


def _validate_coverage_response(
    response: Mapping[str, Any],
    bucket_payloads: list[Mapping[str, Any]],
) -> dict[str, Any]:
    answerable = response.get("answerable")
    if not isinstance(answerable, bool):
        raise LLMResponseError("Validator answerable phải là boolean")
    bucket_keys = [str(bucket["bucket_key"]) for bucket in bucket_payloads]
    available = set(bucket_keys)
    raw_statuses = response.get("bucket_statuses")
    if not isinstance(raw_statuses, list):
        raw_statuses = []
    statuses: list[dict[str, Any]] = []
    seen_statuses: set[str] = set()
    for raw_status in raw_statuses:
        if not isinstance(raw_status, Mapping):
            continue
        key = raw_status.get("bucket_key")
        sufficient = raw_status.get("sufficient")
        reason = raw_status.get("reason")
        if (
            not isinstance(key, str)
            or key not in available
            or key in seen_statuses
            or not isinstance(sufficient, bool)
            or not isinstance(reason, str)
        ):
            continue
        seen_statuses.add(key)
        statuses.append(
            {
                "bucket_key": key,
                "sufficient": sufficient,
                "reason": reason.strip(),
                "required_operands": [
                    operand.strip()
                    for operand in raw_status.get("required_operands", [])
                    if isinstance(operand, str) and operand.strip()
                ]
                if isinstance(raw_status.get("required_operands"), list)
                else [],
            }
        )
    raw_missing = response.get("missing_requirements")
    if not isinstance(raw_missing, list):
        raw_missing = []
    missing: list[dict[str, str]] = []
    for raw_item in raw_missing:
        if not isinstance(raw_item, Mapping):
            continue
        key = raw_item.get("bucket_key")
        if not isinstance(key, str) or key not in available:
            continue
        missing.append(
            {
                "bucket_key": key,
                "concept": str(raw_item.get("concept") or "").strip(),
                "role": str(raw_item.get("role") or "direct").strip(),
                "reason": str(raw_item.get("reason") or "").strip(),
                "suggested_query": str(raw_item.get("suggested_query") or "").strip(),
            }
        )
    raw_targets = response.get("target_bucket_keys")
    if not isinstance(raw_targets, list):
        raw_targets = []
    targets = list(
        dict.fromkeys(
            key for key in raw_targets if isinstance(key, str) and key in available
        )
    )
    insufficient = [
        status["bucket_key"] for status in statuses if not status["sufficient"]
    ]
    for key in [*insufficient, *(item["bucket_key"] for item in missing)]:
        if key not in targets:
            targets.append(key)
    feedback = response.get("feedback")
    if not isinstance(feedback, str):
        feedback = ""
    if answerable and (targets or missing or insufficient):
        answerable = False
    for key in bucket_keys:
        if key in seen_statuses:
            continue
        statuses.append(
            {
                "bucket_key": key,
                "sufficient": key not in targets,
                "reason": "salvaged from validator aggregate decision",
                "required_operands": [],
            }
        )
    status_by_key = {status["bucket_key"]: status for status in statuses}
    statuses = [status_by_key[key] for key in bucket_keys]

    raw_proofs = response.get("coverage_proofs")
    if not isinstance(raw_proofs, list):
        raw_proofs = []
    proofs: list[dict[str, Any]] = []
    for raw_proof in raw_proofs:
        if not isinstance(raw_proof, Mapping):
            continue
        key = raw_proof.get("bucket_key")
        operand = raw_proof.get("operand")
        table_id = raw_proof.get("table_id")
        row = raw_proof.get("row")
        raw_columns = raw_proof.get("columns")
        derivation = raw_proof.get("derivation")
        if (
            not isinstance(key, str)
            or key not in available
            or not isinstance(operand, str)
            or not operand.strip()
            or not isinstance(table_id, str)
            or not table_id.strip()
            or isinstance(row, bool)
            or not isinstance(row, int)
            or not isinstance(raw_columns, list)
            or not raw_columns
            or not all(isinstance(column, str) and column for column in raw_columns)
            or not isinstance(derivation, str)
            or not derivation.strip()
        ):
            continue
        proofs.append(
            {
                "bucket_key": key,
                "operand": operand.strip(),
                "table_id": table_id.strip(),
                "row": row,
                "columns": list(dict.fromkeys(raw_columns)),
                "derivation": derivation.strip(),
            }
        )

    proof_errors: list[dict[str, str]] = []
    if answerable:
        inventory: dict[str, dict[str, tuple[set[int], set[str]]]] = {}
        for bucket in bucket_payloads:
            key = str(bucket["bucket_key"])
            tables: dict[str, tuple[set[int], set[str]]] = {}
            raw_tables = bucket.get("selected_tables")
            if not isinstance(raw_tables, list):
                raw_tables = []
            for table in raw_tables:
                if not isinstance(table, Mapping):
                    continue
                table_id = table.get("table_id")
                context = table.get("context")
                if not isinstance(table_id, str) or not isinstance(context, Mapping):
                    continue
                rows: set[int] = set()
                for field in ("row_catalog", "detailed_rows"):
                    raw_rows = context.get(field)
                    if not isinstance(raw_rows, list):
                        continue
                    for raw_row in raw_rows:
                        if not isinstance(raw_row, Mapping):
                            continue
                        row = raw_row.get("row")
                        if isinstance(row, int) and not isinstance(row, bool):
                            rows.add(row)
                raw_columns = context.get("columns")
                columns = {
                    column
                    for column in raw_columns
                    if isinstance(column, str) and column
                } if isinstance(raw_columns, list) else set()
                tables[table_id] = (rows, columns)
            inventory[key] = tables

        explicit_status_keys = {
            str(raw_status.get("bucket_key"))
            for raw_status in raw_statuses
            if isinstance(raw_status, Mapping)
            and isinstance(raw_status.get("bucket_key"), str)
        }
        for status in statuses:
            key = status["bucket_key"]
            required = status["required_operands"]
            if key not in explicit_status_keys or not status["sufficient"]:
                proof_errors.append(
                    {
                        "bucket_key": key,
                        "concept": "coverage proof",
                        "reason": "answerable=true nhưng thiếu bucket_status sufficient=true",
                    }
                )
                continue
            if not required:
                proof_errors.append(
                    {
                        "bucket_key": key,
                        "concept": "required operands",
                        "reason": "answerable=true nhưng required_operands rỗng",
                    }
                )
                continue
            bucket_proofs = [proof for proof in proofs if proof["bucket_key"] == key]
            for operand in required:
                matching = [
                    proof
                    for proof in bucket_proofs
                    if proof["operand"].casefold() == operand.casefold()
                ]
                valid = False
                for proof in matching:
                    table = inventory.get(key, {}).get(proof["table_id"])
                    if table is None:
                        continue
                    rows, columns = table
                    if proof["row"] in rows and set(proof["columns"]).issubset(columns):
                        valid = True
                        break
                if not valid:
                    proof_errors.append(
                        {
                            "bucket_key": key,
                            "concept": operand,
                            "reason": (
                                "không có coverage proof trỏ tới exact table/row/columns "
                                "trong selected evidence"
                            ),
                        }
                    )

    if proof_errors:
        answerable = False
        for error in proof_errors:
            key = error["bucket_key"]
            if key not in targets:
                targets.append(key)
            missing.append(
                {
                    "bucket_key": key,
                    "concept": error["concept"],
                    "role": "direct",
                    "reason": error["reason"],
                    "suggested_query": "",
                }
            )
        for status in statuses:
            if status["bucket_key"] in targets:
                status["sufficient"] = False
                status["reason"] = "coverage proof không đối chiếu được với inventory"
        feedback = (
            "Validator phải bổ sung evidence có exact table_id, row và value columns "
            "cho các operand chưa chứng minh được."
        )
    if not answerable and not targets:
        raise LLMResponseError("Validator báo thiếu nhưng không chỉ ra target bucket")
    return {
        "answerable": answerable,
        "bucket_statuses": statuses,
        "coverage_proofs": proofs,
        "proof_errors": proof_errors,
        "missing_requirements": missing,
        "target_bucket_keys": targets,
        "feedback": feedback.strip(),
        "source": "llm_proof_invalid" if proof_errors else "llm",
    }


def _flatten_selected_tables(
    specs: Mapping[str, Mapping[str, Any]],
    runtimes: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key in specs:
        for raw_candidate in runtimes[key].get("selected_tables") or []:
            candidate = dict(raw_candidate)
            table_id = str(candidate.get("table_id") or "")
            if not table_id or table_id in seen:
                continue
            seen.add(table_id)
            tables.append(candidate)
    return tables


def validate_table_coverage_node(
    state: Mapping[str, Any],
) -> Command[Literal["rewrite_bucket_queries", "load_tables"]]:
    """Audit global evidence and route insufficient buckets through bounded repair."""
    specs = _bucket_specs_by_key(state)
    runtimes = _bucket_runtime_map(state)
    if _coverage_validator_disabled():
        decision = {
            "answerable": True,
            "bucket_statuses": [
                {
                    "bucket_key": key,
                    "sufficient": True,
                    "reason": "coverage validator disabled for benchmark",
                }
                for key in specs
            ],
            "missing_requirements": [],
            "target_bucket_keys": [],
            "feedback": "",
            "source": "disabled",
        }
        for runtime in runtimes.values():
            runtime.update(
                {
                    "status": "locked",
                    "feedback": "",
                    "latest_candidates": [],
                    "finalists": [],
                }
            )
        history = list(state.get("validation_history") or [])
        history.append(
            {
                "repair_round": int(state.get("retrieval_repair_round", 0)),
                **decision,
            }
        )
        return Command(
            update={
                "bucket_states": runtimes,
                "active_bucket_keys": [],
                "retrieved_tables": _flatten_selected_tables(specs, runtimes),
                "coverage_validation": decision,
                "validation_history": history,
                "bucket_pipeline_metrics": _pipeline_metrics(state),
            },
            goto="load_tables",
        )
    deterministic_failures = _deterministic_coverage_failures(specs, runtimes)
    metrics = _pipeline_metrics(state)
    if deterministic_failures:
        target_keys = list(
            dict.fromkeys(item["bucket_key"] for item in deterministic_failures)
        )
        decision = {
            "answerable": False,
            "bucket_statuses": [
                {
                    "bucket_key": key,
                    "sufficient": key not in target_keys,
                    "reason": (
                        "deterministic checks passed"
                        if key not in target_keys
                        else "deterministic coverage failure"
                    ),
                }
                for key in specs
            ],
            "missing_requirements": deterministic_failures,
            "target_bucket_keys": target_keys,
            "feedback": "Evidence chưa vượt qua deterministic coverage checks.",
            "source": "deterministic",
        }
    else:
        payload = [
            _validator_bucket_payload(specs[key], runtimes[key]) for key in specs
        ]
        prompt = build_coverage_validator_prompt(
            str(state.get("question") or ""),
            payload,
        )
        decision: dict[str, Any] | None = None
        last_response_error: LLMResponseError | None = None
        validator_prompt = prompt
        for validator_attempt in range(2):
            metrics["validator_llm_calls"] += 1
            try:
                response = generate_structured(
                    validator_prompt,
                    system_prompt=COVERAGE_VALIDATOR_SYSTEM_PROMPT,
                    native=False,
                )
                candidate_decision = _validate_coverage_response(response, payload)
            except LLMResponseError as exc:
                last_response_error = exc
                logger.warning(
                    "Coverage validator response is unusable (attempt %d/2): %s",
                    validator_attempt + 1,
                    exc,
                )
                if validator_attempt == 0:
                    validator_prompt = (
                        prompt
                        + "\nLần trả lời trước không phải JSON đúng contract. "
                        "Hãy đọc lại inventory và chỉ trả một JSON object hợp lệ."
                    )
                continue
            logger.debug(
                "Coverage validator prompt=%s response=%s",
                validator_prompt,
                response,
            )
            decision = candidate_decision
            if decision.get("source") != "llm_proof_invalid":
                break
            if validator_attempt == 0:
                proof_feedback = "; ".join(
                    f"{item['bucket_key']}: {item['concept']}"
                    for item in decision.get("proof_errors", [])
                )
                validator_prompt = (
                    prompt
                    + "\nCoverage proof lần trước không đối chiếu được với inventory: "
                    + proof_feedback
                    + ". Hãy tạo lại exact table_id, row và columns từ input; "
                    "nếu không thể thì trả answerable=false."
                )
        if decision is None:
            assert last_response_error is not None
            raise last_response_error
        if decision.get("source") == "llm_proof_invalid":
            decision = {
                **decision,
                "answerable": True,
                "verified_answerable": False,
                "advisory": True,
                "source": "llm_unproven_advisory",
                "reason": (
                    "The semantic validator claimed sufficient evidence but its exact "
                    "proof references were not verifiable. Planner must independently "
                    "audit the selected inventory; no retrieval repair is justified by "
                    "a proof-format failure alone."
                ),
            }

    repair_round = int(state.get("retrieval_repair_round", 0))
    history = list(state.get("validation_history") or [])
    history.append({"repair_round": repair_round, **decision})
    if decision["answerable"]:
        for runtime in runtimes.values():
            runtime.update(
                {
                    "status": "locked",
                    "feedback": "",
                    "latest_candidates": [],
                    "finalists": [],
                }
            )
        return Command(
            update={
                "bucket_states": runtimes,
                "active_bucket_keys": [],
                "retrieved_tables": _flatten_selected_tables(specs, runtimes),
                "coverage_validation": decision,
                "validation_history": history,
                "bucket_pipeline_metrics": metrics,
            },
            goto="load_tables",
        )

    max_repair_rounds = _max_retrieval_repair_rounds()
    if repair_round >= max_repair_rounds:
        if decision.get("source") == "llm" and not _semantic_validator_fail_closed():
            advisory_decision = {
                **decision,
                "answerable": False,
                "advisory": True,
                "source": "llm_exhausted_advisory",
                "reason": (
                    "Semantic validator remained uncertain after bounded repair; "
                    "planner receives the accumulated evidence for a final grounded audit."
                ),
            }
            history[-1] = {
                "repair_round": repair_round,
                **advisory_decision,
            }
            for runtime in runtimes.values():
                runtime.update(
                    {
                        "status": "locked",
                        "feedback": "",
                        "latest_candidates": [],
                        "finalists": [],
                    }
                )
            return Command(
                update={
                    "bucket_states": runtimes,
                    "active_bucket_keys": [],
                    "retrieved_tables": _flatten_selected_tables(specs, runtimes),
                    "coverage_validation": advisory_decision,
                    "validation_history": history,
                    "bucket_pipeline_metrics": metrics,
                },
                goto="load_tables",
            )
        diagnostics = {
            "repair_round": repair_round,
            "max_repair_rounds": max_repair_rounds,
            "decision": decision,
            "history": history,
            "pipeline_metrics": metrics,
        }
        raise TableContextUnsolvableError(
            "Table evidence vẫn thiếu sau "
            f"{max_repair_rounds} retrieval repair rounds: "
            + ", ".join(decision["target_bucket_keys"]),
            diagnostics,
        )

    target_keys = list(decision["target_bucket_keys"])
    missing_by_bucket: dict[str, list[Mapping[str, Any]]] = {
        key: [] for key in target_keys
    }
    for item in decision["missing_requirements"]:
        missing_by_bucket.setdefault(item["bucket_key"], []).append(item)
    for key, runtime in runtimes.items():
        if key not in target_keys:
            runtime.update(
                {
                    "status": "locked",
                    "feedback": "",
                    "latest_candidates": [],
                    "finalists": [],
                }
            )
            continue
        feedback_parts = [decision.get("feedback") or ""]
        for item in missing_by_bucket.get(key, []):
            feedback_parts.append(
                " | ".join(
                    part
                    for part in (
                        str(item.get("concept") or ""),
                        str(item.get("role") or ""),
                        str(item.get("reason") or ""),
                        str(item.get("suggested_query") or ""),
                    )
                    if part
                )
            )
        if not runtime.get("finalists"):
            runtime["finalists"] = list(runtime.get("selected_tables") or [])
        runtime.update(
            {
                "status": "needs_repair",
                "feedback": "\n".join(part for part in feedback_parts if part),
                "latest_candidates": [],
            }
        )
    return Command(
        update={
            "bucket_states": runtimes,
            "active_bucket_keys": target_keys,
            "retrieval_repair_round": repair_round + 1,
            "coverage_validation": decision,
            "validation_history": history,
            "bucket_pipeline_metrics": metrics,
        },
        goto="rewrite_bucket_queries",
    )


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
    """Use FPT BGE-M3 to rerank up to 80 candidates into a top-20 list."""
    reranked_tables = rerank_with_fpt(
        question=str(state.get("question") or ""),
        candidates=state.get("candidates", []),
    )
    logger.info(
        "FPT top table IDs: %s",
        [item.get("table_id") for item in reranked_tables],
    )
    return {"reranked_tables": reranked_tables}


def select_tables_node(state: Mapping[str, Any]) -> dict[str, Any]:
    """Prune FPT top-20 tables for the planner with recall-first LLM selection."""
    retrieved_tables = select_tables(
        question=str(state.get("question") or ""),
        candidates=state.get("reranked_tables", []),
    )
    logger.info(
        "Planner table IDs: %s",
        [item.get("table_id") for item in retrieved_tables],
    )
    return {"retrieved_tables": retrieved_tables}


def load_tables_node(state: Mapping[str, Any]) -> dict[str, Any]:
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
        csv_file = resolve_csv_path(table_id, _PROJECT_ROOT)
        csv_path = csv_file.relative_to(_PROJECT_ROOT.resolve()).as_posix()
        dataframe = pd.read_csv(csv_file)

        dataframes[alias] = dataframe
        raw_rerank_context = candidate.get("rerank_context")
        rerank_contexts[alias] = (
            dict(raw_rerank_context)
            if isinstance(raw_rerank_context, Mapping)
            else {}
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
            "table_id": table_id,
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
) -> Command[Literal["generate_code"]]:
    """Plan how to answer the user question from reranker-selected table context."""
    dataframes = state.get("dataframes") or {}
    alias_metadata = state.get("alias_metadata") or {}
    inventory = build_planning_inventory(
        dataframes,
        alias_metadata,
        state.get("rerank_contexts") or {},
    )
    question = str(state.get("question") or "")
    coverage_validation = state.get("coverage_validation")
    proof_evidence: list[dict[str, Any]] = []
    if isinstance(coverage_validation, Mapping):
        raw_proofs = coverage_validation.get("coverage_proofs")
        table_aliases = {
            str(metadata.get("table_id")): alias
            for alias, metadata in alias_metadata.items()
            if isinstance(metadata, Mapping) and metadata.get("table_id")
        }
        grouped_proofs: dict[str, list[dict[str, Any]]] = {}
        seen_proofs: set[tuple[str, int, tuple[str, ...]]] = set()
        if isinstance(raw_proofs, list):
            for proof in raw_proofs:
                if not isinstance(proof, Mapping):
                    continue
                alias = table_aliases.get(str(proof.get("table_id") or ""))
                csv_row = proof.get("row")
                raw_columns = proof.get("columns")
                if (
                    alias not in dataframes
                    or isinstance(csv_row, bool)
                    or not isinstance(csv_row, int)
                    or not isinstance(raw_columns, list)
                    or not raw_columns
                ):
                    continue
                position = csv_row - 2
                dataframe = dataframes[alias]
                columns = [
                    column
                    for column in raw_columns
                    if isinstance(column, str) and column in dataframe.columns
                ]
                signature = (alias, position, tuple(columns))
                if (
                    position < 0
                    or position >= len(dataframe)
                    or len(columns) != len(raw_columns)
                    or signature in seen_proofs
                ):
                    continue
                seen_proofs.add(signature)
                operand = str(proof.get("operand") or "operand").strip()
                derivation = str(proof.get("derivation") or "direct").strip()
                grouped_proofs.setdefault(alias, []).append(
                    {
                        "row_position": position,
                        "columns": columns,
                        "purpose": f"{operand} ({derivation})",
                    }
                )
        proof_evidence = [
            {"alias": alias, "rows": rows}
            for alias, rows in grouped_proofs.items()
        ]
    coverage_advisory = ""
    if (
        isinstance(coverage_validation, Mapping)
        and coverage_validation.get("source") in {
            "llm_exhausted_advisory",
            "llm_unproven_advisory",
        }
    ):
        missing = coverage_validation.get("missing_requirements")
        missing_summary = "; ".join(
            str(item.get("concept") or item.get("suggested_query") or "").strip()
            for item in missing
            if isinstance(item, Mapping)
        ) if isinstance(missing, list) else ""
        coverage_advisory = (
            "Coverage validator did not provide a fully machine-verifiable coverage "
            "decision. Independently verify every primitive operand in "
            "inventory; derived ratios and totals may be computed from complete "
            "components instead of requiring a same-named row."
            + (f" Reported uncertainty: {missing_summary}." if missing_summary else "")
        )
    generation_plan: dict[str, Any] | None = None
    response_error: LLMResponseError | None = None
    plan_feedback = ""
    for response_attempt in range(1, 4):
        feedback_parts = [coverage_advisory]
        if response_attempt > 1:
            feedback_parts.append(plan_feedback or (
                "Phản hồi trước không phải JSON object hợp lệ. Chỉ trả về JSON đúng "
                "schema được yêu cầu."
            ))
        feedback = "\n".join(part for part in feedback_parts if part)
        planner_prompt = build_planner_prompt(question, inventory, feedback)
        try:
            candidate_plan = generate_structured(
                planner_prompt,
                system_prompt=PLANNER_SYSTEM_PROMPT,
                native=len(planner_prompt) <= 24_000,
            )
        except LLMResponseError as error:
            response_error = error
            plan_feedback = concise_error(error)
            continue
        plan_feedback = generation_plan_feedback(candidate_plan, inventory) or ""
        if plan_feedback:
            generation_plan = None
            continue
        candidate_plan = {
            key: candidate_plan[key]
            for key in ("evidence", "calculation", "unit_conversion", "audit")
        }
        planned_aliases = [
            str(item["alias"])
            for item in candidate_plan["evidence"]
            if isinstance(item, Mapping) and isinstance(item.get("alias"), str)
        ]
        plan_feedback = generated_context_coverage_feedback(
            planned_aliases,
            question,
            alias_metadata,
        ) or ""
        if plan_feedback:
            generation_plan = None
            continue
        generation_plan = candidate_plan
        break
    if generation_plan is None:
        fallback_reason = plan_feedback or (
            concise_error(response_error)
            if response_error is not None
            else "invalid response"
        )
        logger.warning(
            "Planner could not produce a grounded contract after 3 attempts; "
            "using inventory advisory fallback: %s",
            fallback_reason,
        )
        generation_plan = {
            "evidence": proof_evidence,
            "calculation": (
                "Planner contract unavailable. Generator must independently map every "
                "primitive operand to exact aliases, rows and columns in inventory. "
                "Validated coverage-proof rows are grounded seeds, not permission to "
                "skip other required operands."
            ),
            "unit_conversion": "Infer only from exact selected table units and question.",
            "audit": (
                "Do not assume the missing planner output is evidence; inspect all explicit "
                f"ticker/year buckets. Planner error: {fallback_reason}"
            ),
        }
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
    allowed_aliases = set(state.get("evidence_sources") or [])
    generator_prompt = build_generator_prompt(
        question=str(state.get("question") or ""),
        generation_plan=state.get("generation_plan") or {},
        planned_context=state.get("planned_context") or {},
        feedback=str(state.get("feedback") or ""),
    )
    native_json = attempt == 1 and len(generator_prompt) <= 24_000
    structured_options = {
        "system_prompt": GENERATOR_SYSTEM_PROMPT,
        "json_schema": GENERATOR_RESPONSE_SCHEMA,
        "sdk_max_retries": 0,
        "invalid_json_parser": parse_malformed_generator_json,
    }
    try:
        try:
            generated = generate_structured(
                generator_prompt,
                native=native_json,
                **structured_options,
            )
        except LLMTransientError:
            if not native_json:
                raise
            generated = generate_structured(
                generator_prompt,
                native=False,
                **structured_options,
            )
    except LLMResponseError as error:
        update["feedback"] = f"Generator call failed: {concise_error(error)}"
        return retry_or_exhausted(state, update, attempt=attempt)

    feedback = generator_feedback(generated)
    if feedback:
        update["feedback"] = feedback
        return retry_or_exhausted(state, update, attempt=attempt)

    normalized_code, code_feedback = normalize_generated_code(
        generated["pandas_query"]
    )
    if code_feedback:
        update["feedback"] = code_feedback
        return retry_or_exhausted(state, update, attempt=attempt)

    normalized_code, selector_feedback = normalize_generated_selectors(
        normalized_code,
        state.get("dataframes") or {},
    )
    if selector_feedback:
        update["feedback"] = selector_feedback
        return retry_or_exhausted(state, update, attempt=attempt)

    normalized_code, semantic_normalization_feedback = normalize_generated_semantics(
        normalized_code,
        str(state.get("question") or ""),
        state.get("dataframes") or {},
    )
    if semantic_normalization_feedback:
        update["feedback"] = semantic_normalization_feedback
        return retry_or_exhausted(state, update, attempt=attempt)

    evidence_variables, alias_feedback = generated_evidence_variables(
        normalized_code, allowed_aliases
    )
    if alias_feedback:
        update["feedback"] = alias_feedback
        return retry_or_exhausted(state, update, attempt=attempt)

    coverage_feedback = generated_context_coverage_feedback(
        evidence_variables,
        str(state.get("question") or ""),
        state.get("alias_metadata") or {},
    )
    if coverage_feedback:
        update["feedback"] = coverage_feedback
        return retry_or_exhausted(state, update, attempt=attempt)

    rounding_feedback = generated_rounding_feedback(
        normalized_code, str(state.get("question") or "")
    )
    if rounding_feedback:
        update["feedback"] = rounding_feedback
        return retry_or_exhausted(state, update, attempt=attempt)

    semantic_feedback = generated_semantic_feedback(
        normalized_code,
        str(state.get("question") or ""),
        state.get("dataframes") or {},
    )
    if semantic_feedback:
        update["feedback"] = semantic_feedback
        return retry_or_exhausted(state, update, attempt=attempt)

    attribute_feedback = _unsupported_dataframe_attribute_feedback(
        normalized_code,
        set(state.get("dataframes") or {}),
    )
    if attribute_feedback:
        update["feedback"] = attribute_feedback
        return retry_or_exhausted(state, update, attempt=attempt)

    update.update(
        {
            "feedback": "",
            "pandas_query": normalized_code,
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
    max_attempts = int(state.get("max_attempts", 1))
    if feedback and attempt >= max_attempts:
        raise RuntimeError(_numeric_result_error(attempt, feedback))

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
                _numeric_result_error(attempt, execution_feedback)
            ) from error
        return retry_or_exhausted(
            state,
            {"feedback": execution_feedback},
        )

    answer, feedback = numeric_result(result)
    if feedback:
        if attempt >= max_attempts:
            raise RuntimeError(_numeric_result_error(attempt, feedback))
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
