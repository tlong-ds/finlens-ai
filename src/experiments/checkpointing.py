"""Run IDs and resumable status transitions shared by experiment runners."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid4().hex[:8]}"


def validate_run_id(value: str) -> str:
    run_id = value.strip()
    if (
        not run_id
        or run_id in {".", ".."}
        or len(run_id) > 128
        or not all(character.isalnum() or character in "._-" for character in run_id)
    ):
        raise ValueError("run-id may contain only letters, numbers, '.', '_' or '-'")
    return run_id
