"""Concurrent-safe JSON audit log for generated-code execution."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LOG_FILE = _PROJECT_ROOT / "log.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunAuditLog:
    """Persist one graph run in the shared audit-log file."""

    def __init__(self, input_payload: Mapping[str, Any]) -> None:
        self.run_id = uuid.uuid4().hex
        self.path = Path(
            os.getenv("FINLENS_RUN_LOG_FILE", str(_DEFAULT_LOG_FILE))
        ).resolve()
        self.lock_path = self.path.with_name(f".{self.path.name}.lock")
        self.data: dict[str, Any] = {
            "run_id": self.run_id,
            "status": "running",
            "started_at": _utc_now(),
            "finished_at": None,
            "input": dict(input_payload),
            "attempts": [],
            "answer_record": None,
            "final_error": None,
        }
        self._persist()

    def start_attempt(self, attempt: int, feedback_in: str) -> int:
        self.data["attempts"].append(
            {
                "attempt": attempt,
                "started_at": _utc_now(),
                "finished_at": None,
                "feedback_in": feedback_in or None,
                "generation": None,
                "contract_validation": None,
                "code_execution": None,
                "semantic_validation": None,
                "outcome": "running",
                "feedback_out": None,
            }
        )
        self._persist()
        return len(self.data["attempts"]) - 1

    def update_attempt(self, index: int, **values: Any) -> None:
        self.data["attempts"][index].update(values)
        self._persist()

    def finish_attempt(self, index: int, outcome: str, feedback: str = "") -> None:
        self.data["attempts"][index].update(
            {
                "finished_at": _utc_now(),
                "outcome": outcome,
                "feedback_out": feedback or None,
            }
        )
        self._persist()

    def finish(
        self,
        status: str,
        *,
        answer_record: Mapping[str, Any] | None = None,
        final_error: str | None = None,
    ) -> None:
        self.data.update(
            {
                "status": status,
                "finished_at": _utc_now(),
                "answer_record": dict(answer_record) if answer_record else None,
                "final_error": final_error,
            }
        )
        self._persist()

    def attempts_snapshot(self) -> list[dict[str, Any]]:
        """Return a detached, JSON-safe copy for failure reporting."""
        return json.loads(json.dumps(self.data["attempts"], default=str))

    def _persist(self) -> None:
        """Upsert this run while preserving concurrent writers' snapshots."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.lock_path.open("a", encoding="utf-8") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    runs = self._read_runs()
                    for index, run in enumerate(runs):
                        if run.get("run_id") == self.run_id:
                            runs[index] = self.data
                            break
                    else:
                        runs.append(self.data)
                    self._write_runs(runs)
                finally:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            logger.warning("Unable to persist run audit log %s: %s", self.path, error)

    def _read_runs(self) -> list[dict[str, Any]]:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return []

        with self.path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or not isinstance(payload.get("runs"), list):
            raise ValueError("audit log must be a JSON object containing a runs array")
        runs = payload["runs"]
        if not all(isinstance(run, dict) for run in runs):
            raise ValueError("every audit-log run must be a JSON object")
        return runs

    def _write_runs(self, runs: list[dict[str, Any]]) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            suffix=".tmp",
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            with temporary_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {"runs": runs},
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            os.replace(temporary_path, self.path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
