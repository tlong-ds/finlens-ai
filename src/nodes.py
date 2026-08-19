"""LangGraph nodes for retrieval and validated pandas answer execution."""

from __future__ import annotations

import ast
import json
import logging
import re
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from src.sandbox import run_code
from src.contracts import resolve_csv_path, validate_qdrant_payload
from src.helper import (
    concise_error,
    find_question,
    generator_feedback,
    numeric_result,
    ordered_unique,
    validator_feedback,
)
from src.llm import LLMResponseError, generate_structured
from src.prompt import (
    GENERATOR_SYSTEM_PROMPT,
    PARSE_SYSTEM_PROMPT,
    VALIDATOR_SYSTEM_PROMPT,
    build_generator_prompt,
    build_parse_prompt,
    build_validator_prompt,
)
from src.run_log import RunAuditLog
from src.retrieval import rerank, retrieve
from src.routing import (
    QueryRoutingError,
    parse_years,
    reconcile_query_filters,
    resolve_tickers,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SANDBOX_TIMEOUT_SECONDS = 5.0
_STRUCTURED_RESPONSE_ATTEMPTS = 2
_DATAFRAME_ALIAS_PATTERN = re.compile(r"df_\d+")
_LABEL_COLUMN_PATTERN = re.compile(r"(?:^|_)(?:row|item)?_?label(?:_|$)", re.I)
_ITEM_CODE_COLUMN_PATTERN = re.compile(r"(?:^|_)item_code(?:_|$)", re.I)
_SEMANTIC_LOOKUP_ROWS_PER_TABLE = 12
_DYNAMIC_NAMESPACE_CALLS = {
    "compile",
    "eval",
    "exec",
    "globals",
    "locals",
    "__import__",
}


def _normalize_lookup_text(value: Any) -> str:
    """Normalize labels for matching while preserving originals for pandas code."""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", text, flags=re.UNICODE).split())


def _lookup_score(normalized_label: str, question_tokens: set[str]) -> tuple[int, int]:
    label_tokens = set(normalized_label.split())
    overlap = len(label_tokens & question_tokens)
    return overlap, -len(label_tokens - question_tokens)


def _json_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _build_semantic_lookup(
    dataframe: pd.DataFrame,
    question: str,
) -> dict[str, Any]:
    """Return relevant exact labels and item codes, ranked by question overlap."""
    label_columns = [
        column
        for column in dataframe.columns
        if _LABEL_COLUMN_PATTERN.search(str(column))
    ]
    code_columns = [
        column
        for column in dataframe.columns
        if _ITEM_CODE_COLUMN_PATTERN.search(str(column))
    ]
    if not label_columns and not code_columns:
        return {
            "label_columns": [],
            "item_code_columns": [],
            "matching_rows": [],
        }

    question_tokens = set(_normalize_lookup_text(question).split())
    ranked_rows: list[tuple[tuple[int, int], int, dict[str, Any]]] = []
    for position, (_, row) in enumerate(dataframe.iterrows()):
        labels: dict[str, Any] = {}
        normalized_labels: dict[str, str] = {}
        for column in label_columns:
            original = _json_scalar(row[column])
            if original is None or not str(original).strip():
                continue
            labels[str(column)] = original
            normalized_labels[str(column)] = _normalize_lookup_text(original)

        item_codes: dict[str, Any] = {}
        for column in code_columns:
            item_code = _json_scalar(row[column])
            if item_code is not None:
                item_codes[str(column)] = item_code
        if not labels and not item_codes:
            continue

        score = max(
            (_lookup_score(value, question_tokens) for value in normalized_labels.values()),
            default=(0, 0),
        )
        ranked_rows.append(
            (
                score,
                position,
                {
                    "labels": labels,
                    "normalized_labels": normalized_labels,
                    "item_codes": item_codes,
                },
            )
        )

    ranked_rows.sort(key=lambda item: (-item[0][0], -item[0][1], item[1]))
    return {
        "label_columns": [str(column) for column in label_columns],
        "item_code_columns": [str(column) for column in code_columns],
        "matching_rows": [
            record
            for _, _, record in ranked_rows[:_SEMANTIC_LOOKUP_ROWS_PER_TABLE]
        ],
    }


class _ModuleResultAssignmentVisitor(ast.NodeVisitor):
    """Find result assignments outside nested function and class scopes."""

    def __init__(self) -> None:
        self.found = False

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == "result" and isinstance(node.ctx, ast.Store):
            self.found = True

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return


def _is_zero_index(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
        and node.value == 0
    )


def _iloc_mask_name(node: ast.Subscript) -> str | None:
    """Return the named boolean mask used by ``...loc[mask, ...].iloc[0]``."""
    if not (
        isinstance(node.value, ast.Attribute)
        and node.value.attr == "iloc"
        and _is_zero_index(node.slice)
    ):
        return None

    selection = node.value.value
    if not isinstance(selection, ast.Subscript):
        return None
    if isinstance(selection.value, ast.Attribute) and selection.value.attr == "loc":
        mask = selection.slice
        if isinstance(mask, ast.Tuple) and mask.elts:
            mask = mask.elts[0]
    else:
        mask = selection.slice
    return mask.id if isinstance(mask, ast.Name) else None


def _guarded_mask_lines(tree: ast.AST) -> dict[str, list[int]]:
    """Find explicit ``if not mask.any(): raise ValueError(...)`` guards."""
    guarded: dict[str, list[int]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.UnaryOp):
            continue
        if not isinstance(node.test.op, ast.Not):
            continue
        call = node.test.operand
        if not (
            isinstance(call, ast.Call)
            and not call.args
            and not call.keywords
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "any"
            and isinstance(call.func.value, ast.Name)
        ):
            continue

        has_specific_error = False
        for statement in node.body:
            if not isinstance(statement, ast.Raise) or not isinstance(
                statement.exc, ast.Call
            ):
                continue
            exception_call = statement.exc
            if not (
                isinstance(exception_call.func, ast.Name)
                and exception_call.func.id == "ValueError"
                and exception_call.args
                and isinstance(exception_call.args[0], ast.Constant)
                and isinstance(exception_call.args[0].value, str)
                and exception_call.args[0].value.strip()
            ):
                continue
            has_specific_error = True
            break
        if has_specific_error:
            guarded.setdefault(call.func.value.id, []).append(node.lineno)
    return guarded


def _unguarded_iloc_feedback(tree: ast.AST) -> str | None:
    guarded_lines = _guarded_mask_lines(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript) or not (
            isinstance(node.value, ast.Attribute)
            and node.value.attr == "iloc"
            and _is_zero_index(node.slice)
        ):
            continue
        mask_name = _iloc_mask_name(node)
        if mask_name is None:
            return (
                "Mọi .iloc[0] phải lấy từ df.loc[mask, ...] với mask là biến riêng "
                "đã được kiểm tra bằng if not mask.any(): raise ValueError(...)."
            )
        if not any(
            guard_line < node.lineno for guard_line in guarded_lines.get(mask_name, [])
        ):
            return (
                f"Mask {mask_name} phải được kiểm tra trước .iloc[0] bằng "
                "if not mask.any(): raise ValueError('<lỗi cụ thể>')."
            )
    return None


def _attach_attempt_history(error: BaseException, audit_log: RunAuditLog) -> None:
    """Expose generated candidates to callers without coupling them to log.json."""
    try:
        error.attempt_history = audit_log.attempts_snapshot()  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        logger.warning("Unable to attach attempt history to %s", type(error).__name__)


def _code_contract_feedback(
    code: str,
    evidence_variables: list[str],
    available_aliases: set[str],
) -> str | None:
    """Validate syntax and deterministic generated-code invariants."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as error:
        location = f" dòng {error.lineno}" if error.lineno else ""
        return f"pandas_query có cú pháp Python không hợp lệ{location}: {error.msg}."

    if len(evidence_variables) != len(set(evidence_variables)):
        return "evidence_variables không được chứa alias trùng lặp."

    declared_aliases = set(evidence_variables)
    unknown_declared = sorted(declared_aliases - available_aliases)
    if unknown_declared:
        return "Unknown evidence variables: " + ", ".join(unknown_declared)

    used_aliases: set[str] = set()
    overwritten_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load) and _DATAFRAME_ALIAS_PATTERN.fullmatch(
                node.id
            ):
                used_aliases.add(node.id)
            if isinstance(node.ctx, ast.Store) and (
                node.id == "pd" or _DATAFRAME_ALIAS_PATTERN.fullmatch(node.id)
            ):
                overwritten_names.add(node.id)
        elif isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Name)
                and node.func.id in _DYNAMIC_NAMESPACE_CALLS
            ):
                return (
                    "pandas_query không được truy cập namespace động bằng "
                    f"{node.func.id}()."
                )
        elif isinstance(node, ast.Import):
            if len(node.names) != 1 or not (
                node.names[0].name == "pandas" and node.names[0].asname == "pd"
            ):
                return "pandas_query chỉ được phép import pandas as pd."
        elif isinstance(node, ast.ImportFrom):
            return "pandas_query không được phép dùng import-from."

    if overwritten_names:
        return "pandas_query không được gán đè alias đầu vào: " + ", ".join(
            sorted(overwritten_names)
        )

    unknown_used = sorted(used_aliases - available_aliases)
    if unknown_used:
        return "pandas_query sử dụng DataFrame không tồn tại: " + ", ".join(
            unknown_used
        )

    missing_evidence = sorted(used_aliases - declared_aliases)
    if missing_evidence:
        return "Thiếu evidence_variables cho DataFrame: " + ", ".join(
            missing_evidence
        )

    unused_evidence = sorted(declared_aliases - used_aliases)
    if unused_evidence:
        return "evidence_variables khai báo nhưng không được dùng: " + ", ".join(
            unused_evidence
        )

    iloc_feedback = _unguarded_iloc_feedback(tree)
    if iloc_feedback:
        return iloc_feedback

    result_visitor = _ModuleResultAssignmentVisitor()
    result_visitor.visit(tree)
    if not result_visitor.found:
        return "pandas_query phải gán kết quả cuối cùng vào biến result."
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
    tickers, _, _ = resolve_tickers(question)
    if not tickers:
        raise QueryRoutingError(
            "Không resolve được ticker; hệ thống không search global"
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
            filters, semantic_query = reconcile_query_filters(question, raw_filters)
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
    """Rerank candidates and return the final Top-K tables."""
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
    entity_year_aliases: dict[str, dict[str, list[dict[str, str]]]] = {}
    semantic_lookup: dict[str, dict[str, Any]] = {}
    descriptions: list[str] = []
    question = str(state.get("question") or "")

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
        ticker = metadata["ticker"]
        year = str(metadata["year"])
        entity_year_aliases.setdefault(ticker, {}).setdefault(year, []).append(
            {
                "alias": alias,
                "company_name": metadata["company_name"],
                "report_type": metadata["report_type"],
                "table_type": metadata["table_type"],
                "table_id": table_id,
            }
        )
        semantic_lookup[alias] = _build_semantic_lookup(dataframe, question)
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
        "entity_year_aliases": entity_year_aliases,
        "semantic_lookup": semantic_lookup,
        "dataframe_description": "\n".join(descriptions),
    }


def generate_code_node(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Generate, check, execute, and semantically validate pandas candidates."""
    max_attempts = int(state.get("max_attempts", 5))
    question = str(state.get("question") or "")
    dataframe_description = str(state.get("dataframe_description") or "")
    dataframes = state.get("dataframes") or {}
    evidence_sources = state.get("evidence_sources") or {}
    entity_year_aliases = state.get("entity_year_aliases") or {}
    semantic_lookup = state.get("semantic_lookup") or {}
    available_aliases = set(dataframes)
    feedback = str(state.get("feedback") or "")
    last_error: BaseException | None = None
    question_record = state.get("question_record") or {}
    audit_log = RunAuditLog(
        {
            "question_id": question_record.get("id"),
            "question": question,
            "max_attempts": max_attempts,
            "filters": state.get("filters"),
            "semantic_query": state.get("semantic_query"),
            "retrieved_table_ids": [
                item.get("table_id")
                for item in state.get("retrieved_tables", [])
                if isinstance(item, Mapping)
            ],
            "dataframe_description": dataframe_description,
            "entity_year_aliases": entity_year_aliases,
            "semantic_lookup": semantic_lookup,
            "evidence_sources": evidence_sources,
        }
    )

    for attempt in range(1, max_attempts + 1):
        last_error = None
        attempt_log_index = audit_log.start_attempt(attempt, feedback)
        generator_prompt = build_generator_prompt(
            question=question,
            dataframe_description=dataframe_description,
            feedback=feedback,
            entity_year_aliases=entity_year_aliases,
            semantic_lookup=semantic_lookup,
        )
        try:
            generated = generate_structured(
                generator_prompt,
                system_prompt=GENERATOR_SYSTEM_PROMPT,
            )
        except LLMResponseError as error:
            last_error = error
            feedback = f"Generator call failed: {concise_error(error)}"
            audit_log.update_attempt(
                attempt_log_index,
                generation={
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error) or type(error).__name__,
                },
            )
            audit_log.finish_attempt(attempt_log_index, "retry", feedback)
            continue
        except BaseException as error:
            message = f"Generator call failed: {concise_error(error)}"
            audit_log.update_attempt(
                attempt_log_index,
                generation={
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error) or type(error).__name__,
                },
            )
            audit_log.finish_attempt(attempt_log_index, "aborted", message)
            audit_log.finish("error", final_error=message)
            _attach_attempt_history(error, audit_log)
            raise

        audit_log.update_attempt(
            attempt_log_index,
            generation={"status": "ok", "response": generated},
        )

        feedback = generator_feedback(generated) or ""
        if feedback:
            audit_log.update_attempt(
                attempt_log_index,
                contract_validation={"valid": False, "feedback": feedback},
            )
            audit_log.finish_attempt(attempt_log_index, "retry", feedback)
            continue

        pandas_query = generated["pandas_query"]
        evidence_variables = generated["evidence_variables"]
        feedback = _code_contract_feedback(
            pandas_query,
            evidence_variables,
            available_aliases,
        ) or ""
        audit_log.update_attempt(
            attempt_log_index,
            contract_validation={"valid": not feedback, "feedback": feedback or None},
        )
        if feedback:
            audit_log.finish_attempt(attempt_log_index, "retry", feedback)
            continue

        try:
            result = run_code(
                pandas_query,
                dataframes,
                timeout_sec=_SANDBOX_TIMEOUT_SECONDS,
            )
        except (RuntimeError, TimeoutError, ValueError) as error:
            last_error = error
            feedback = f"Sandbox execution failed: {concise_error(error)}"
            audit_log.update_attempt(
                attempt_log_index,
                code_execution={
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error) or type(error).__name__,
                },
            )
            audit_log.finish_attempt(attempt_log_index, "retry", feedback)
            continue
        except BaseException as error:
            message = f"Sandbox execution failed: {concise_error(error)}"
            audit_log.update_attempt(
                attempt_log_index,
                code_execution={
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error) or type(error).__name__,
                },
            )
            audit_log.finish_attempt(attempt_log_index, "aborted", message)
            audit_log.finish("error", final_error=message)
            _attach_attempt_history(error, audit_log)
            raise

        answer, feedback_value = numeric_result(result)
        feedback = feedback_value or ""
        audit_log.update_attempt(
            attempt_log_index,
            code_execution={
                "status": "ok" if not feedback else "invalid",
                "result": answer,
                "feedback": feedback or None,
            },
        )
        if feedback:
            audit_log.finish_attempt(attempt_log_index, "retry", feedback)
            continue

        validator_prompt = build_validator_prompt(
            question=question,
            available_aliases=list(dataframes),
            dataframe_description=dataframe_description,
            pandas_query=pandas_query,
            evidence_variables=evidence_variables,
            execution_result=answer,
            entity_year_aliases=entity_year_aliases,
            semantic_lookup=semantic_lookup,
        )
        try:
            validation = generate_structured(
                validator_prompt,
                system_prompt=VALIDATOR_SYSTEM_PROMPT,
            )
        except LLMResponseError as error:
            last_error = error
            feedback = f"Validator call failed: {concise_error(error)}"
            audit_log.update_attempt(
                attempt_log_index,
                semantic_validation={
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error) or type(error).__name__,
                },
            )
            audit_log.finish_attempt(attempt_log_index, "retry", feedback)
            continue
        except BaseException as error:
            message = f"Validator call failed: {concise_error(error)}"
            audit_log.update_attempt(
                attempt_log_index,
                semantic_validation={
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error) or type(error).__name__,
                },
            )
            audit_log.finish_attempt(attempt_log_index, "aborted", message)
            audit_log.finish("error", final_error=message)
            _attach_attempt_history(error, audit_log)
            raise

        feedback = validator_feedback(validation) or ""
        audit_log.update_attempt(
            attempt_log_index,
            semantic_validation={
                "status": "ok",
                "response": validation,
                "accepted": not feedback,
                "feedback": feedback or None,
            },
        )
        if feedback:
            audit_log.finish_attempt(attempt_log_index, "retry", feedback)
            continue

        aliases = ordered_unique(evidence_variables)
        answer_record = {
            "id": question_record["id"],
            "question": question,
            "answer": answer,
            "evidence": {
                alias: evidence_sources[alias]["csv_path"] for alias in aliases
            },
            "relevant_docs": ordered_unique(
                [evidence_sources[alias]["doc_id"] for alias in aliases]
            ),
            "relevant_tables": ordered_unique(
                [evidence_sources[alias]["relevant_table"] for alias in aliases]
            ),
            "pandas_query": pandas_query,
        }
        audit_log.finish_attempt(attempt_log_index, "accepted")
        audit_log.finish("success", answer_record=answer_record)
        return {
            "attempt": attempt,
            "feedback": "",
            "pandas_query": pandas_query,
            "evidence_variables": evidence_variables,
            "answer_record": answer_record,
        }

    exhausted = RuntimeError(
        "Unable to produce a valid numeric result after "
        f"{max_attempts} attempts: {feedback or 'Unknown generation failure'}"
    )
    audit_log.finish("failed", final_error=str(exhausted))
    _attach_attempt_history(exhausted, audit_log)
    if last_error is not None:
        raise exhausted from last_error
    raise exhausted
