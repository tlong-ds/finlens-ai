"""Normalization and semantic validation for generated pandas code."""

from __future__ import annotations

import ast
import builtins
import re
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from typing import Any

import pandas as pd

from src.generation.planning import _CODE_COLUMN, _DATAFRAME_ALIAS_PATTERN

_EXPLICIT_ROUNDING_TERMS = (
    "làm tròn",
    "lam tron",
    "round to",
    "decimal place",
    "chữ số thập phân",
)
_GENERATOR_GLOBAL_NAMES = {"alias_metadata", "np", "pd", "plt", "sns"}


def generated_evidence_variables(
    code: str,
    allowed_aliases: set[str],
) -> tuple[list[str], str | None]:
    """Derive evidence aliases from generated Python instead of model metadata."""
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        location = f" at line {error.lineno}" if error.lineno is not None else ""
        return [], f"Generated code is invalid Python{location}: {error.msg}."

    referenced_aliases = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and _DATAFRAME_ALIAS_PATTERN.fullmatch(node.id)
    }
    outside_context = sorted(referenced_aliases - allowed_aliases)
    if outside_context:
        return [], (
            "Generated code uses aliases outside the loaded context: "
            + ", ".join(outside_context)
        )
    if not referenced_aliases:
        return [], "Generated code does not reference a loaded DataFrame alias."
    return sorted(
        referenced_aliases, key=lambda alias: int(alias.removeprefix("df_"))
    ), None


def normalize_generated_code(code: str) -> tuple[str, str | None]:
    """Make a safe missing-result repair and reject unresolved local names."""
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        location = f" at line {error.lineno}" if error.lineno is not None else ""
        return code, f"Generated code is invalid Python{location}: {error.msg}."

    assigned_names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    if "result" not in assigned_names:
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            final_expression = tree.body[-1]
            tree.body[-1] = ast.copy_location(
                ast.Assign(
                    targets=[ast.Name(id="result", ctx=ast.Store())],
                    value=final_expression.value,
                ),
                final_expression,
            )
            ast.fix_missing_locations(tree)
            code = ast.unparse(tree)
            assigned_names.add("result")
        else:
            return code, (
                "Generated code must assign the final finite numeric scalar to "
                "the variable 'result'."
            )

    loaded_names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    known_names = (
        assigned_names
        | _GENERATOR_GLOBAL_NAMES
        | set(dir(builtins))
        | {node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)}
        | {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
            and _DATAFRAME_ALIAS_PATTERN.fullmatch(node.id)
        }
    )
    unresolved_names = sorted(loaded_names - known_names)
    if unresolved_names:
        return code, (
            "Generated code references undefined local names: "
            + ", ".join(unresolved_names)
            + ". Use the exact variable names assigned in the code."
        )
    return code, None


def _selector_target(node: ast.AST) -> tuple[str, str] | None:
    if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name):
        return None
    alias = node.value.id
    if not _DATAFRAME_ALIAS_PATTERN.fullmatch(alias):
        return None
    column_node = node.slice
    if not isinstance(column_node, ast.Constant) or not isinstance(
        column_node.value, str
    ):
        return None
    return alias, column_node.value


def _numeric_equivalent(value: str, candidates: Sequence[str]) -> str | None:
    try:
        numeric_value = Decimal(value)
    except InvalidOperation:
        return None
    matches: list[str] = []
    for candidate in candidates:
        try:
            if Decimal(candidate) == numeric_value:
                matches.append(candidate)
        except InvalidOperation:
            continue
    return matches[0] if len(matches) == 1 else None


def normalize_generated_selectors(
    code: str,
    dataframes: Mapping[str, pd.DataFrame],
) -> tuple[str, str | None]:
    """Canonicalize exact string selectors and reject values absent from a table."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, None

    changed = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            continue
        if not isinstance(node.ops[0], (ast.Eq, ast.NotEq)):
            continue
        pairs = ((node.left, node.comparators[0]), (node.comparators[0], node.left))
        for selector_node, literal_node in pairs:
            target = _selector_target(selector_node)
            if target is None or not isinstance(literal_node, ast.Constant):
                continue
            literal = literal_node.value
            if not isinstance(literal, str):
                continue
            alias, column = target
            dataframe = dataframes.get(alias)
            if dataframe is None or column not in dataframe.columns:
                continue
            candidates = list(dict.fromkeys(dataframe[column].astype(str).tolist()))
            if literal in candidates:
                break
            replacement = (
                _numeric_equivalent(literal, candidates)
                if column == _CODE_COLUMN
                else None
            )
            if replacement is not None:
                literal_node.value = replacement
                changed = True
                break
            return code, (
                f"Generated code filters {alias}[{column!r}] by {literal!r}, but that "
                f"exact value is absent. Use one of the exact values from the context: "
                + ", ".join(repr(value) for value in candidates)
                + "."
            )

    if changed:
        ast.fix_missing_locations(tree)
        return ast.unparse(tree), None
    return code, None


def parse_malformed_generator_json(content: str) -> Mapping[str, Any]:
    """Recover generator code from a malformed two-field JSON envelope."""
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
    if fenced:
        content = fenced.group(1)
    match = re.search(
        r'["\']pandas_query["\']\s*:\s*(?P<quote>["\'])(?P<code>.*)'
        r'(?P=quote)\s*(?:,\s*["\']evidence_variables["\']\s*:\s*\[.*?\])?\s*\}',
        content,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError("Malformed generator response has no recoverable pandas_query")
    code = (
        match.group("code")
        .replace(r"\n", "\n")
        .replace(r"\t", "\t")
        .replace(r"\"", '"')
        .replace("\\\\", "\\")
    )
    if not code.strip():
        raise ValueError("Malformed generator response has empty code")
    return {"pandas_query": code, "evidence_variables": []}


def generated_rounding_feedback(code: str, question: str) -> str | None:
    """Reject precision loss unless the question explicitly requests rounding."""
    normalized_question = question.casefold()
    if any(term in normalized_question for term in _EXPLICIT_ROUNDING_TERMS):
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if (isinstance(function, ast.Name) and function.id == "round") or (
            isinstance(function, ast.Attribute) and function.attr == "round"
        ):
            return (
                "Generated code rounds an intermediate or final value even though the "
                "question does not request rounding. Preserve full numeric precision."
            )
    return None


_QUESTION_UNIT_FACTORS = (
    ("trăm tỷ đồng", 1e11),
    ("nghìn tỷ đồng", 1e12),
    ("triệu cổ phiếu", 1e6),
    ("triệu đồng", 1e6),
    ("tỷ đồng", 1e9),
)
_PLACEHOLDER_TERMS = (
    "placeholder",
    "for simplicity",
    "assume ",
    "giả sử",
    "tạm dùng",
    "...",
    "data missing",
    "missing data",
    "data unavailable",
    "defaults to 0",
    "default to 0",
    "không có dữ liệu",
)


def _expected_generated_unit_scale(
    tree: ast.AST,
    question: str,
    dataframes: Mapping[str, pd.DataFrame] | None,
) -> tuple[str, float] | None:
    """Infer one output divisor from the question and referenced cell magnitudes."""
    normalized_question = question.casefold()
    requested_unit = next(
        (
            (unit, target_factor)
            for unit, target_factor in _QUESTION_UNIT_FACTORS
            if unit in normalized_question
        ),
        None,
    )
    if requested_unit is None:
        return None

    def dataframe_root_name(node: ast.AST) -> str | None:
        while isinstance(node, (ast.Attribute, ast.Subscript)):
            node = node.value
        if isinstance(node, ast.Name) and _DATAFRAME_ALIAS_PATTERN.fullmatch(node.id):
            return node.id
        return None

    referenced_columns: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        alias = dataframe_root_name(node.value)
        if alias is None:
            continue
        columns = {
            child.value
            for child in ast.walk(node.slice)
            if isinstance(child, ast.Constant) and isinstance(child.value, str)
        }
        referenced_columns.setdefault(alias, set()).update(columns)

    magnitudes: list[float] = []
    for alias, columns in referenced_columns.items():
        frame = (dataframes or {}).get(alias)
        if frame is None:
            continue
        for column in columns & set(frame.columns):
            numeric = pd.to_numeric(frame[column], errors="coerce").dropna()
            if not numeric.empty:
                magnitudes.append(float(numeric.abs().max()))

    # Normalized statements may store amounts in millions while note tables use
    # raw VND. Payload unit metadata is inconsistent, but the two magnitude bands
    # are well separated in the loaded evidence.
    if magnitudes and max(magnitudes) >= 1e10:
        stored_unit_factor = 1.0
    elif magnitudes and max(magnitudes) <= 1e9:
        stored_unit_factor = 1e6
    else:
        return None

    unit, target_factor = requested_unit
    return unit, target_factor / stored_unit_factor


def normalize_generated_semantics(
    code: str,
    question: str,
    dataframes: Mapping[str, pd.DataFrame] | None = None,
) -> tuple[str, str | None]:
    """Normalize an unambiguous unit divisor without another model call."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, None

    expected = _expected_generated_unit_scale(tree, question, dataframes)
    if expected is None:
        return code, None
    unit, expected_scale = expected
    known_scales = {1e3, 1e5, 1e6, 1e9, 1e11, 1e12}
    result_assignment = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "result"
                for target in node.targets
            )
        ),
        None,
    )
    if result_assignment is None:
        return code, None

    scale_nodes = [
        node
        for node in ast.walk(result_assignment.value)
        if isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Div)
        and isinstance(node.right, ast.Constant)
        and isinstance(node.right.value, (int, float))
        and float(node.right.value) in known_scales
    ]
    existing_scales = {float(node.right.value) for node in scale_nodes}
    if expected_scale in existing_scales or expected_scale == 1 and not scale_nodes:
        return code, None
    if len(scale_nodes) > 1 and len(existing_scales) > 1:
        return code, (
            f"The requested unit '{unit}' requires scale {expected_scale:g}, but "
            "the final calculation contains multiple competing unit divisors."
        )
    if scale_nodes:
        for scale_node in scale_nodes:
            if expected_scale == 1:
                scale_node.right = ast.Constant(value=1)
            else:
                scale_node.right = ast.Constant(value=expected_scale)
    elif expected_scale != 1:
        result_assignment.value = ast.BinOp(
            left=result_assignment.value,
            op=ast.Div(),
            right=ast.Constant(value=expected_scale),
        )
    ast.fix_missing_locations(tree)
    return ast.unparse(tree), None


def generated_semantic_feedback(
    code: str,
    question: str,
    dataframes: Mapping[str, pd.DataFrame] | None = None,
) -> str | None:
    """Reject obvious generator guesses and question-operation mismatches.

    These checks deliberately cover only contracts that can be inferred directly
    from the question. Row and table selection remains the model's job.
    """
    normalized_code = code.casefold()
    for term in _PLACEHOLDER_TERMS:
        if term in normalized_code:
            return (
                "Generated code contains a placeholder, guess, or simplifying "
                "assumption. Use only exact labels, item codes, columns, and values "
                "present in planned_context; do not invent missing evidence."
            )

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    normalized_question = question.casefold()
    numeric_constants = {
        float(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    }
    expected = _expected_generated_unit_scale(tree, question, dataframes)
    if expected is not None:
        unit, expected_scale = expected
        competing_scales = numeric_constants & {1e6, 1e9, 1e11, 1e12}
        # Division by one is normally omitted from good generated code.
        has_expected_scale = expected_scale == 1 or expected_scale in numeric_constants
        if not has_expected_scale:
            detail = (
                f"; code currently uses {sorted(competing_scales)}"
                if competing_scales
                else ""
            )
            return (
                f"The requested output unit is '{unit}'. Convert raw values using "
                f"the exact scale {expected_scale:g}{detail}. Apply the conversion "
                "once, after the financial calculation."
            )

    binary_operators = {
        type(node.op) for node in ast.walk(tree) if isinstance(node, ast.BinOp)
    }
    has_division = ast.Div in binary_operators
    has_times_100 = any(
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Mult)
        and (
            (isinstance(node.left, ast.Constant) and node.left.value == 100)
            or (isinstance(node.right, ast.Constant) and node.right.value == 100)
        )
        for node in ast.walk(tree)
    )
    # A percentage mentioned in a scenario or filter is not necessarily the
    # requested output unit.  Benchmark questions state the answer request as
    # "bao nhiêu phần trăm"; scope this guard to that answer phrase so inputs
    # such as "phát hành thêm 10%" do not force an unrelated result * 100.
    asks_percentage = bool(
        re.search(r"bao\s+nhiêu\s+(?:phần\s+trăm|%)", normalized_question)
    )
    if asks_percentage and has_division and not has_times_100:
        return (
            "The question asks for a percentage, but the generated ratio is not "
            "converted to percent. Multiply the final ratio by 100."
        )

    asks_difference = any(
        term in normalized_question for term in ("chênh lệch", "hiệu số")
    )
    if asks_difference and ast.Sub not in binary_operators:
        return (
            "The question asks for a difference, but the generated calculation "
            "contains no subtraction. Compute the requested values separately and "
            "subtract them in the question's stated order."
        )
    asks_growth = any(
        term in normalized_question
        for term in ("tỷ lệ thay đổi", "tốc độ tăng", "tăng trưởng")
    )
    if asks_growth and ast.Sub not in binary_operators:
        return (
            "The question asks for a change or growth rate, but the generated "
            "calculation has no current-minus-prior subtraction. Compute "
            "(current - prior) / prior * 100."
        )
    return None


def generated_context_coverage_feedback(
    evidence_variables: Sequence[str],
    question: str,
    alias_metadata: Mapping[str, Mapping[str, Any]],
) -> str | None:
    """Require explicit multi-entity/year questions to inspect their full context."""
    normalized_question = question.casefold()
    used = set(evidence_variables)

    requested_tickers = {
        str(metadata.get("ticker"))
        for metadata in alias_metadata.values()
        if metadata.get("ticker")
        and re.search(
            rf"(?<![a-z0-9]){re.escape(str(metadata['ticker']).casefold())}(?![a-z0-9])",
            normalized_question,
        )
    }
    missing_tickers = sorted(
        ticker
        for ticker in requested_tickers
        if not any(
            alias in used and str(metadata.get("ticker")) == ticker
            for alias, metadata in alias_metadata.items()
        )
    )
    if missing_tickers:
        return (
            "Generated code does not inspect any loaded table for question-named "
            "tickers: " + ", ".join(missing_tickers) + ". Evaluate every named "
            "company before filtering, comparing, averaging, or taking max/min."
        )

    requested_years = {
        int(metadata["year"])
        for metadata in alias_metadata.values()
        if isinstance(metadata.get("year"), int)
        and re.search(rf"(?<!\d){metadata['year']}(?!\d)", question)
    }
    if len(requested_years) < 3:
        return None
    missing_years = sorted(
        year
        for year in requested_years
        if not any(
            alias in used and metadata.get("year") == year
            for alias, metadata in alias_metadata.items()
        )
    )
    if missing_years:
        return (
            "Generated code skips explicitly requested years: "
            + ", ".join(map(str, missing_years))
            + ". Inspect evidence for every listed year before choosing or aggregating."
        )
    return None
