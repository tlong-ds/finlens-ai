"""Safe code execution: runs code in an isolated E2B Firecracker microVM, with timeout and error handling."""

import pickle
from typing import Any

import pandas as pd
from e2b_code_interpreter import Sandbox, TimeoutException

_RESULT_PATH = "/tmp/__result__.pkl"
_MISSING_RESULT_SENTINEL = "__FINLENS_RESULT_UNDEFINED__"


def run_code(
    code: str,
    dataframes: dict[str, pd.DataFrame],
    timeout_sec: float = 5.0
) -> Any:
    """
    Execute pandas code in a restricted namespace.

    Args:
        code: Python code snippet to execute. Must assign result to 'result' variable.
        dataframes: Dict of {table_id: DataFrame} available to the code.
        timeout_sec: Execution timeout in seconds.

    Returns:
        The value assigned to 'result' variable in the executed code.

    Raises:
        TimeoutError: If execution exceeds timeout.
        ValueError: If code does not assign a 'result' variable.
        RuntimeError: If code execution fails.
    """
    try:
        with Sandbox.create(timeout=max(1, round(timeout_sec))) as sbx:
            for table_id, df in dataframes.items():
                sbx.files.write(f"{table_id}.csv", df.to_csv(index=False))

            load_lines = "\n".join(
                f"{table_id} = pd.read_csv('{table_id}.csv')" for table_id in dataframes
            )

            wrapped_code = f"""
import pandas as pd
import pickle

{load_lines}

{code}

with open('{_RESULT_PATH}', 'wb') as _f:
    pickle.dump(result if 'result' in dir() else '{_MISSING_RESULT_SENTINEL}', _f)
"""

            execution = sbx.run_code(wrapped_code, timeout=timeout_sec)

            if execution.error:
                raise RuntimeError(execution.error.traceback or execution.error.value)

            result = pickle.loads(sbx.files.read(_RESULT_PATH, format="bytes"))
    except TimeoutException as exc:
        raise TimeoutError(f"Execution exceeded {timeout_sec}s timeout") from exc

    if isinstance(result, str) and result == _MISSING_RESULT_SENTINEL:
        raise ValueError("Code did not assign a 'result' variable.")

    return result

if __name__ == "__main__":

    dataframes = {
        "table_001": pd.DataFrame({"1": ["0"], "0": ["1"]})
    }

    code = """
result = table_001.columns.tolist()
"""

    print(run_code(code, dataframes))
