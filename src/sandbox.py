"""Execute generated pandas code inside an isolated E2B microVM."""

from __future__ import annotations

import json
import keyword
import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

import pandas as pd
from e2b_code_interpreter import Sandbox, TimeoutException

_RESULT_PATH = "/tmp/__finlens_result.json"
_MAX_RESULT_BYTES = 1_024
_SANDBOX_TTL_BUFFER_SECONDS = 10
_RESERVED_ALIASES = {"pd", "result"}


def _validate_inputs(
    code: str,
    dataframes: Mapping[str, pd.DataFrame],
    timeout_sec: float,
) -> None:
    if not isinstance(code, str) or not code.strip():
        raise ValueError("code must be a non-empty string")
    if (
        isinstance(timeout_sec, bool)
        or not isinstance(timeout_sec, Real)
        or not math.isfinite(float(timeout_sec))
        or timeout_sec <= 0
    ):
        raise ValueError("timeout_sec must be a positive finite number")
    if not dataframes:
        raise ValueError("dataframes must not be empty")

    for alias, dataframe in dataframes.items():
        if (
            not isinstance(alias, str)
            or not alias.isidentifier()
            or keyword.iskeyword(alias)
            or alias in _RESERVED_ALIASES
            or alias.startswith("_finlens_")
        ):
            raise ValueError(f"Invalid DataFrame alias: {alias!r}")
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                f"DataFrame alias {alias!r} must map to a pandas DataFrame"
            )
        if not all(isinstance(column, str) for column in dataframe.columns):
            raise ValueError(f"DataFrame alias {alias!r} must use string column names")
        if not dataframe.columns.is_unique:
            raise ValueError(f"DataFrame alias {alias!r} must use unique column names")


def _dataframe_payload(dataframe: pd.DataFrame) -> str:
    return dataframe.to_json(
        orient="table",
        date_format="iso",
        force_ascii=False,
        index=False,
    )


def _dataframe_dtypes(dataframe: pd.DataFrame) -> str:
    return json.dumps(
        {column: str(dtype) for column, dtype in dataframe.dtypes.items()},
        ensure_ascii=False,
    )


def _result_serializer_code() -> str:
    return f"""
import json as _finlens_json
import math as _finlens_math
import numbers as _finlens_numbers

if "result" not in globals():
    _finlens_payload = {{"status": "missing"}}
elif isinstance(result, bool) or not isinstance(result, _finlens_numbers.Number):
    _finlens_payload = {{
        "status": "invalid",
        "message": "Sandbox result must be a numeric scalar, not " + type(result).__name__,
    }}
else:
    try:
        _finlens_value = float(result)
    except (TypeError, ValueError, OverflowError):
        _finlens_payload = {{
            "status": "invalid",
            "message": "Sandbox result cannot be converted to float",
        }}
    else:
        if _finlens_math.isfinite(_finlens_value):
            _finlens_payload = {{"status": "ok", "value": _finlens_value}}
        else:
            _finlens_payload = {{
                "status": "invalid",
                "message": "Sandbox result must be finite",
            }}

with open({_RESULT_PATH!r}, "w", encoding="utf-8") as _finlens_file:
    _finlens_json.dump(_finlens_payload, _finlens_file, allow_nan=False)
"""


def _raise_execution_error(execution: Any) -> None:
    if execution.error:
        raise RuntimeError(execution.error.traceback or execution.error.value)


def _read_numeric_result(sandbox: Sandbox) -> float:
    result_info = sandbox.files.get_info(_RESULT_PATH)
    if result_info.size > _MAX_RESULT_BYTES:
        raise RuntimeError(
            f"Sandbox result exceeds the {_MAX_RESULT_BYTES}-byte limit"
        )

    result_text = sandbox.files.read(_RESULT_PATH, format="text")
    if len(result_text.encode("utf-8")) > _MAX_RESULT_BYTES:
        raise RuntimeError(
            f"Sandbox result exceeds the {_MAX_RESULT_BYTES}-byte limit"
        )

    try:
        payload = json.loads(result_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Sandbox returned invalid result JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("status"), str):
        raise RuntimeError("Sandbox returned an invalid result payload")

    status = payload["status"]
    if status == "missing" and set(payload) == {"status"}:
        raise ValueError("Code did not assign a 'result' variable.")
    if status == "invalid" and set(payload) == {"status", "message"}:
        message = payload.get("message")
        if not isinstance(message, str) or not message:
            raise RuntimeError("Sandbox returned invalid result feedback")
        raise ValueError(message)
    if status != "ok" or set(payload) != {"status", "value"}:
        raise RuntimeError("Sandbox returned an invalid result payload")

    value = payload["value"]
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RuntimeError("Sandbox returned a non-numeric result value")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise RuntimeError("Sandbox returned a non-finite result value")
    return numeric_value


def run_code(
    code: str,
    dataframes: dict[str, pd.DataFrame],
    timeout_sec: float = 5.0,
) -> float:
    """Execute generated code in a fresh network-disabled E2B microVM.

    The supplied DataFrames are available under their dictionary aliases. Generated code
    must assign a finite numeric scalar to ``result``. Only a small validated JSON payload
    crosses from the untrusted VM back into the host process.

    Raises:
        TimeoutError: If generated-code execution exceeds ``timeout_sec``.
        TypeError: If a DataFrame value has the wrong type.
        ValueError: If inputs or the generated result are invalid.
        RuntimeError: If execution or the result protocol fails.
    """
    _validate_inputs(code, dataframes, timeout_sec)
    sandbox_timeout = math.ceil(float(timeout_sec)) + _SANDBOX_TTL_BUFFER_SECONDS

    try:
        with Sandbox.create(
            timeout=sandbox_timeout,
            secure=True,
            allow_internet_access=False,
        ) as sandbox:
            load_lines: list[str] = []
            for index, (alias, dataframe) in enumerate(dataframes.items(), start=1):
                dataframe_path = f"/tmp/__finlens_dataframe_{index}.json"
                sandbox.files.write(dataframe_path, _dataframe_payload(dataframe))
                dtype_payload = _dataframe_dtypes(dataframe)
                load_lines.append(
                    f"{alias} = pd.read_json({dataframe_path!r}, orient='table').astype("
                    f"_finlens_json.loads({dtype_payload!r}))"
                )

            execution = sandbox.run_code(
                "import json as _finlens_json\nimport pandas as pd\n"
                + "\n".join(load_lines)
                + "\n\n"
                + code,
                timeout=float(timeout_sec),
            )
            _raise_execution_error(execution)

            serialization = sandbox.run_code(
                _result_serializer_code(),
                timeout=min(float(timeout_sec), 5.0),
            )
            _raise_execution_error(serialization)
            return _read_numeric_result(sandbox)
    except TimeoutException as exc:
        raise TimeoutError(f"Execution exceeded {timeout_sec}s timeout") from exc


if __name__ == "__main__":
    example_dataframes = {
        "df_1": pd.DataFrame({"value": [1.5, 2.5]}),
    }
    example_code = "result = df_1['value'].sum()"
    print(run_code(example_code, example_dataframes))
