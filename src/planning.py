"""Question-driven planning support grounded in reranker-selected tables."""

from __future__ import annotations

import ast
import builtins
import re
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from typing import Any

import pandas as pd

EVIDENCE_PLAN_SCHEMA_VERSION = 3
PLANNING_CELL_MAX_CHARS = 160

_DATAFRAME_ALIAS_PATTERN = re.compile(r"^df_[1-9][0-9]*$")
_LABEL_COLUMNS = ("item_label_norm", "row_label_raw")
_CODE_COLUMN = "item_code"
_TITLE_COLUMN = "note_title"
_EXPLICIT_ROUNDING_TERMS = (
    "làm tròn",
    "lam tron",
    "round to",
    "decimal place",
    "chữ số thập phân",
)
_GENERATOR_GLOBAL_NAMES = {"alias_metadata", "np", "pd", "plt", "sns"}


def _compact_value(value: Any) -> str:
    return " ".join(str(value).split())[:PLANNING_CELL_MAX_CHARS]


def _optional_compact(value: Any) -> str | None:
    compact = _compact_value(value)
    return compact or None


def _catalog_row(dataframe: pd.DataFrame, position: int) -> dict[str, Any]:
    """Build one stable row identity from the loaded DataFrame."""
    row = dataframe.iloc[position]
    label_column = next(
        (column for column in _LABEL_COLUMNS if column in dataframe.columns),
        dataframe.columns[0] if len(dataframe.columns) else None,
    )
    item: dict[str, Any] = {"row_position": position}
    if label_column is not None:
        item["label"] = _compact_value(row[label_column])
    if _CODE_COLUMN in dataframe.columns:
        code = _optional_compact(row[_CODE_COLUMN])
        if code is not None:
            item["code"] = code
    if _TITLE_COLUMN in dataframe.columns:
        title = _optional_compact(row[_TITLE_COLUMN])
        if title is not None:
            item["title"] = title
    return item


def _rerank_detail_positions(
    rerank_context: Mapping[str, Any],
    row_count: int,
) -> list[int]:
    """Translate reranker CSV line numbers into DataFrame positions."""
    raw_rows = rerank_context.get("detailed_rows")
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        return []
    positions: list[int] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, Mapping):
            continue
        raw_position = raw_row.get("row_position")
        if isinstance(raw_position, int) and not isinstance(raw_position, bool):
            position = raw_position
        else:
            csv_line = raw_row.get("row")
            if not isinstance(csv_line, int) or isinstance(csv_line, bool):
                continue
            position = csv_line - 2
        if 0 <= position < row_count and position not in positions:
            positions.append(position)
    return positions


def _row_values(dataframe: pd.DataFrame, position: int) -> dict[str, str]:
    row = dataframe.iloc[position]
    return {
        str(column): _compact_value(row[column])
        for column in dataframe.columns
    }


def build_planning_inventory(
    dataframes: Mapping[str, pd.DataFrame],
    alias_metadata: Mapping[str, Mapping[str, Any]],
    rerank_contexts: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Expose every row identity and reuse reranker-selected detailed rows.

    The row catalog has no scan cutoff. Full cell values remain bounded to the
    detailed rows already selected by the reranker. Any catalog row named by the
    planner is hydrated from the loaded DataFrame after planning.
    """
    contexts = rerank_contexts or {}
    inventory: list[dict[str, Any]] = []
    for alias, dataframe in dataframes.items():
        metadata = alias_metadata.get(alias)
        if metadata is None:
            raise ValueError(f"Missing metadata for alias {alias}")
        raw_context = contexts.get(alias) or {}
        detail_positions = _rerank_detail_positions(raw_context, len(dataframe))
        if not detail_positions and len(dataframe) <= 8:
            detail_positions = list(range(len(dataframe)))
        inventory.append(
            {
                "alias": alias,
                "metadata": dict(metadata),
                "row_count": len(dataframe),
                "columns": [
                    {"name": str(column), "dtype": str(dataframe[column].dtype)}
                    for column in dataframe.columns
                ],
                "table_titles": list(raw_context.get("table_titles") or []),
                "match_summary": dict(raw_context.get("match_summary") or {}),
                "row_catalog": [
                    _catalog_row(dataframe, position)
                    for position in range(len(dataframe))
                ],
                "detailed_rows": [
                    {
                        "row_position": position,
                        "values": _row_values(dataframe, position),
                    }
                    for position in detail_positions
                ],
            }
        )
    return inventory


def generation_plan_feedback(
    generation_plan: Mapping[str, Any],
    inventory: Sequence[Mapping[str, Any]],
) -> str | None:
    """Validate planner references before generator retries depend on the plan."""
    required_keys = {"evidence", "calculation", "unit_conversion", "audit"}
    if not required_keys.issubset(generation_plan):
        return (
            "Planner response must contain evidence, calculation, "
            "unit_conversion, and audit."
        )
    for field in ("calculation", "unit_conversion", "audit"):
        value = generation_plan.get(field)
        if not isinstance(value, str) or not value.strip():
            return f"Planner field {field} must be a non-empty string."

    table_inventory: dict[str, tuple[set[int], set[str]]] = {}
    for raw_table in inventory:
        if not isinstance(raw_table, Mapping):
            continue
        alias = raw_table.get("alias")
        if not isinstance(alias, str):
            continue
        raw_rows = raw_table.get("row_catalog")
        rows = {
            row["row_position"]
            for row in raw_rows
            if isinstance(row, Mapping)
            and isinstance(row.get("row_position"), int)
            and not isinstance(row.get("row_position"), bool)
        } if isinstance(raw_rows, Sequence) and not isinstance(raw_rows, (str, bytes)) else set()
        raw_columns = raw_table.get("columns")
        columns = {
            column["name"]
            for column in raw_columns
            if isinstance(column, Mapping)
            and isinstance(column.get("name"), str)
        } if isinstance(raw_columns, Sequence) and not isinstance(raw_columns, (str, bytes)) else set()
        table_inventory[alias] = (rows, columns)

    raw_evidence = generation_plan.get("evidence")
    if not isinstance(raw_evidence, Sequence) or isinstance(
        raw_evidence, (str, bytes)
    ) or not raw_evidence:
        return "Planner evidence must be a non-empty array."
    for item in raw_evidence:
        if not isinstance(item, Mapping):
            return "Every planner evidence item must be an object."
        alias = item.get("alias")
        if not isinstance(alias, str) or alias not in table_inventory:
            return f"Planner evidence references unknown alias: {alias!r}."
        raw_rows = item.get("rows")
        if not isinstance(raw_rows, Sequence) or isinstance(
            raw_rows, (str, bytes)
        ) or not raw_rows:
            return f"Planner evidence for {alias} must select at least one row."
        available_rows, available_columns = table_inventory[alias]
        for row in raw_rows:
            if not isinstance(row, Mapping):
                return f"Planner row selection for {alias} must be an object."
            position = row.get("row_position")
            raw_selected_columns = row.get("columns")
            purpose = row.get("purpose")
            if (
                isinstance(position, bool)
                or not isinstance(position, int)
                or position not in available_rows
            ):
                return f"Planner row_position {position!r} does not exist in {alias}."
            if (
                not isinstance(raw_selected_columns, Sequence)
                or isinstance(raw_selected_columns, (str, bytes))
                or not raw_selected_columns
                or not all(
                    isinstance(column, str) and column in available_columns
                    for column in raw_selected_columns
                )
            ):
                return f"Planner columns for {alias} row {position} are not in inventory."
            if not isinstance(purpose, str) or not purpose.strip():
                return f"Planner purpose for {alias} row {position} must be non-empty."
    return None


def _planned_row_positions(
    generation_plan: Mapping[str, Any],
    allowed_aliases: set[str],
) -> dict[str, list[int]]:
    positions: dict[str, list[int]] = {alias: [] for alias in allowed_aliases}
    raw_evidence = generation_plan.get("evidence")
    if not isinstance(raw_evidence, Sequence) or isinstance(
        raw_evidence, (str, bytes)
    ):
        return positions
    for raw_item in raw_evidence:
        if not isinstance(raw_item, Mapping):
            continue
        alias = raw_item.get("alias")
        if not isinstance(alias, str) or alias not in positions:
            continue
        raw_rows = raw_item.get("rows")
        if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
            continue
        for raw_row in raw_rows:
            raw_position = (
                raw_row.get("row_position")
                if isinstance(raw_row, Mapping)
                else raw_row
            )
            if (
                isinstance(raw_position, int)
                and not isinstance(raw_position, bool)
                and raw_position not in positions[alias]
            ):
                positions[alias].append(raw_position)
    return positions


def hydrate_planned_rows(
    generation_plan: Mapping[str, Any],
    dataframes: Mapping[str, pd.DataFrame],
    inventory: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Hydrate planner-selected positions plus reranker-detailed fallback rows."""
    positions = _planned_row_positions(generation_plan, set(dataframes))
    for table in inventory:
        alias = table.get("alias")
        if not isinstance(alias, str) or alias not in positions:
            continue
        raw_details = table.get("detailed_rows")
        if not isinstance(raw_details, Sequence) or isinstance(
            raw_details, (str, bytes)
        ):
            continue
        for raw_detail in raw_details:
            if not isinstance(raw_detail, Mapping):
                continue
            position = raw_detail.get("row_position")
            if (
                isinstance(position, int)
                and not isinstance(position, bool)
                and position not in positions[alias]
            ):
                positions[alias].append(position)

    hydrated: dict[str, list[dict[str, Any]]] = {}
    for alias, dataframe in dataframes.items():
        hydrated[alias] = [
            {
                "row_position": position,
                "values": _row_values(dataframe, position),
            }
            for position in positions[alias]
            if 0 <= position < len(dataframe)
        ]
    return hydrated


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
    return sorted(referenced_aliases, key=lambda alias: int(alias.removeprefix("df_"))), None


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
        | {
            node.arg
            for node in ast.walk(tree)
            if isinstance(node, ast.arg)
        }
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
        .replace(r'\"', '"')
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
            and any(isinstance(target, ast.Name) and target.id == "result" for target in node.targets)
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
        type(node.op)
        for node in ast.walk(tree)
        if isinstance(node, ast.BinOp)
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
        term in normalized_question
        for term in ("chênh lệch", "hiệu số")
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
