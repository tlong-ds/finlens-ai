"""Code generation, retry, sandbox execution, and answer assembly nodes."""

from __future__ import annotations

import ast
import math
from collections.abc import Mapping
from numbers import Number
from typing import Any, Literal

from langgraph.graph import END
from langgraph.types import Command

from src.config import Settings
from src.generation.normalization import (
    generated_context_coverage_feedback,
    generated_evidence_variables,
    generated_rounding_feedback,
    generated_semantic_feedback,
    normalize_generated_code,
    normalize_generated_selectors,
    normalize_generated_semantics,
    parse_malformed_generator_json,
)
from src.generation.prompts import (
    GENERATOR_RESPONSE_SCHEMA,
    GENERATOR_SYSTEM_PROMPT,
    build_generator_prompt,
)
from src.providers.e2b import run_code
from src.providers.llm import (
    LLMResponseError,
    LLMTransientError,
    generate_structured,
)

_SANDBOX_TIMEOUT_SECONDS = 30.0
_UNSUPPORTED_DATAFRAME_ATTRIBUTES = {"metadata", "attrs"}


def _generator_feedback(response: Mapping[str, Any]) -> str | None:
    if set(response) != {"pandas_query", "evidence_variables"}:
        return "Response generator phải có đúng pandas_query và evidence_variables."
    if (
        not isinstance(response["pandas_query"], str)
        or not response["pandas_query"].strip()
    ):
        return "pandas_query phải là chuỗi không rỗng."
    evidence_variables = response["evidence_variables"]
    if not isinstance(evidence_variables, list) or not all(
        isinstance(alias, str) for alias in evidence_variables
    ):
        return "evidence_variables phải là mảng alias DataFrame dạng chuỗi."
    return None


def _numeric_result(value: Any) -> tuple[float | None, str | None]:
    if isinstance(value, bool) or not isinstance(value, Number):
        return None, f"Sandbox result must be numeric, not {type(value).__name__}."
    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None, "Sandbox result cannot be converted to float."
    if not math.isfinite(numeric_value):
        return None, "Sandbox result must be finite, not NaN or infinity."
    return numeric_value, None


def _concise_error(error: BaseException) -> str:
    message = " ".join(str(error).split())
    return message[:500] or type(error).__name__


def _ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _retry_or_exhausted(
    state: Mapping[str, Any],
    update: dict[str, Any],
    *,
    attempt: int | None = None,
) -> Command[Literal["generate_code", "execute_code"]]:
    current_attempt = int(state.get("attempt", 0)) if attempt is None else attempt
    destination = (
        "generate_code"
        if current_attempt < int(state.get("max_attempts", 1))
        else "execute_code"
    )
    return Command(update=update, goto=destination)


# Local aliases keep the moved node body byte-for-byte stable.
retry_or_exhausted = _retry_or_exhausted
concise_error = _concise_error
numeric_result = _numeric_result
ordered_unique = _ordered_unique


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


def generate_code_node(
    state: Mapping[str, Any],
    *,
    settings: Settings,
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
                settings=settings,
                native=native_json,
                **structured_options,
            )
        except LLMTransientError:
            if not native_json:
                raise
            generated = generate_structured(
                generator_prompt,
                settings=settings,
                native=False,
                **structured_options,
            )
    except LLMResponseError as error:
        update["feedback"] = f"Generator call failed: {_concise_error(error)}"
        return _retry_or_exhausted(state, update, attempt=attempt)

    feedback = _generator_feedback(generated)
    if feedback:
        update["feedback"] = feedback
        return _retry_or_exhausted(state, update, attempt=attempt)

    normalized_code, code_feedback = normalize_generated_code(generated["pandas_query"])
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

    selected_aliases = list(state.get("evidence_sources") or {})
    code_aliases = evidence_variables
    if code_aliases != selected_aliases:
        missing_aliases = [a for a in selected_aliases if a not in set(code_aliases)]
        if attempt < 2 and missing_aliases:
            feedback = (
                f"Generated code must use every selector-selected alias. "
                f"Expected {selected_aliases}, got {code_aliases}."
            )
            update["feedback"] = feedback
            return retry_or_exhausted(state, update, attempt=attempt)
        elif missing_aliases:
            normalized_code = (
                f"{normalized_code}\n_finlens_unused = [{', '.join(missing_aliases)}]"
            )
            evidence_variables = selected_aliases

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
    *,
    settings: Settings,
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
            settings,
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

    sources = state.get("evidence_sources") or {}
    aliases = list(sources)  # selector order
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
