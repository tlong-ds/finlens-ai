"""Run the FinLens graph on a labelled validation set and compute metrics.

Examples:

    python run_valset.py --run-id experiment ids --ids 1,5,7
    python run_valset.py --run-id experiment full
    python run_valset.py --run-id experiment --resume full

Each run is isolated under ``val_submission/runs/<run-id>/``.  Besides the
submission package, a run contains aggregate/per-question metrics and a JSON
trace for every question with the output emitted by each LangGraph node.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from generate_submission import (
    build_zip,
    load_existing_submission,
    to_submission_item,
    write_submission_json,
)

from src.experiments.provenance import write_run_manifest
from src.pipeline.graph import default_settings as SETTINGS
from src.pipeline.graph import graph
from src.providers.fpt import effective_fpt_reranker_config

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN_PATH = PROJECT_ROOT / "golden_100.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "val_submission"
RUNS_DIRECTORY_NAME = "runs"
STATUS_SCHEMA_VERSION = 1
TRACE_SCHEMA_VERSION = 2
DEFAULT_ANSWER_RTOL = 1e-6
DEFAULT_ANSWER_ATOL = 1e-6

METRIC_NAMES = (
    "EXECUTION ACCURACY",
    "TABLES F2-MACRO",
    "DOCS F2-MACRO",
    "TABLES PRECISION",
    "TABLES RECALL",
    "TABLES MRR5",
    "DOCS PRECISION",
    "DOCS RECALL",
    "DOCS MRR5",
    "ANSWER ACCURACY",
)

_write_lock = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid4().hex[:8]}"


def _validate_run_id(value: str) -> str:
    run_id = value.strip()
    if (
        not run_id
        or run_id in {".", ".."}
        or len(run_id) > 128
        or not all(character.isalnum() or character in "._-" for character in run_id)
    ):
        raise ValueError("run-id must contain only letters, numbers, '.', '_' or '-'")
    return run_id


def _atomic_replace(target: Path, write_fn: Any, *, suffix: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=".tmp-", suffix=suffix)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        write_fn(tmp_path)
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _write_json(value: Any, target: Path) -> None:
    def _write(tmp_path: Path) -> None:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)

    _atomic_replace(target, _write, suffix=".json")


def _write_jsonl(values: Sequence[Mapping[str, Any]], target: Path) -> None:
    def _write(tmp_path: Path) -> None:
        with tmp_path.open("w", encoding="utf-8") as handle:
            for value in values:
                handle.write(json.dumps(value, ensure_ascii=False) + "\n")

    _atomic_replace(target, _write, suffix=".jsonl")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _retry_error_ids(path: Path, message: str | None) -> set[int] | None:
    """Return failed validation ids matching an exact error message."""
    if not message:
        return None
    if not path.exists():
        return set()
    matched: set[int] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("error") == message:
                matched.add(int(record["id"]))
    return matched


def _failure_messages(path: Path) -> dict[int, str]:
    messages: dict[int, str] = {}
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                    messages[int(record["id"])] = str(
                        record.get("error", "unknown error")
                    )
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
    return messages


def load_golden(path: Path) -> list[dict[str, Any]]:
    """Load and validate the labelled validation records."""
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list) or not raw:
        raise ValueError("Golden file must be a non-empty JSON array")

    records: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    required = {"id", "question", "answer", "relevant_docs", "relevant_tables"}
    for index, value in enumerate(raw):
        if not isinstance(value, Mapping) or not required.issubset(value):
            raise ValueError(f"Golden record {index} is missing required fields")
        question_id = value["id"]
        if isinstance(question_id, bool) or not isinstance(question_id, int):
            raise ValueError(f"Golden record {index} has a non-integer id")
        if question_id in seen_ids:
            raise ValueError(f"Duplicate golden question id: {question_id}")
        seen_ids.add(question_id)

        question = value["question"]
        answer = value["answer"]
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Golden id={question_id} has an empty question")
        if (
            isinstance(answer, bool)
            or not isinstance(answer, (int, float))
            or not math.isfinite(float(answer))
        ):
            raise ValueError(f"Golden id={question_id} has a non-finite answer")

        normalized: dict[str, Any] = {
            "id": question_id,
            "question": question,
            "answer": float(answer),
        }
        for field in ("relevant_docs", "relevant_tables"):
            items = value[field]
            if not isinstance(items, list) or not all(
                isinstance(item, str) and item.strip() for item in items
            ):
                raise ValueError(
                    f"Golden id={question_id} field {field} must be a string array"
                )
            normalized[field] = _ordered_unique(items)
        records.append(normalized)
    return records


def _ordered_unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def _parse_ids(raw: str) -> set[int]:
    try:
        values = {int(part.strip()) for part in raw.split(",") if part.strip()}
    except ValueError as exc:
        raise ValueError("--ids must be a comma-separated list of integers") from exc
    if not values:
        raise ValueError("--ids must contain at least one question id")
    return values


def _select_golden(
    args: argparse.Namespace, golden: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if args.command == "ids":
        requested = _parse_ids(args.ids)
        known = {int(record["id"]) for record in golden}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError("Unknown validation ids: " + ", ".join(map(str, unknown)))
        selected = [record for record in golden if int(record["id"]) in requested]
    else:
        selected = list(golden)
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        raise ValueError("No validation questions selected")
    return selected


def _score_ranked(
    predicted: Sequence[str], relevant: Sequence[str]
) -> dict[str, float]:
    predicted_unique = _ordered_unique(predicted)
    relevant_set = set(relevant)
    correct = len(set(predicted_unique) & relevant_set)
    precision = correct / len(predicted_unique) if predicted_unique else 0.0
    recall = correct / len(relevant_set) if relevant_set else 1.0
    denominator = 4.0 * precision + recall
    f2 = (5.0 * precision * recall / denominator) if denominator else 0.0
    reciprocal_rank = 0.0
    for rank, item in enumerate(predicted_unique[:5], start=1):
        if item in relevant_set:
            reciprocal_rank = 1.0 / rank
            break
    return {
        "precision": precision,
        "recall": recall,
        "f2": f2,
        "mrr5": reciprocal_rank,
    }


def _answers_match(
    predicted: Any,
    expected: float,
    *,
    rtol: float,
    atol: float,
) -> bool:
    if (
        isinstance(predicted, bool)
        or not isinstance(predicted, (int, float))
        or not math.isfinite(float(predicted))
    ):
        return False
    return math.isclose(float(predicted), expected, rel_tol=rtol, abs_tol=atol)


def score_question(
    golden: Mapping[str, Any],
    prediction: Mapping[str, Any] | None,
    *,
    execution_ok: bool,
    answer_rtol: float,
    answer_atol: float,
) -> dict[str, Any]:
    """Score one submission-shaped prediction against one golden record."""
    predicted_docs = list(prediction.get("relevant_docs", [])) if prediction else []
    predicted_tables = list(prediction.get("relevant_tables", [])) if prediction else []
    docs = _score_ranked(predicted_docs, list(golden["relevant_docs"]))
    tables = _score_ranked(predicted_tables, list(golden["relevant_tables"]))
    answer_correct = _answers_match(
        prediction.get("answer") if prediction else None,
        float(golden["answer"]),
        rtol=answer_rtol,
        atol=answer_atol,
    )
    return {
        "id": int(golden["id"]),
        "tables": tables,
        "docs": docs,
        "answer_correct": answer_correct,
        "execution_correct": bool(execution_ok and answer_correct),
        "predicted_answer": prediction.get("answer") if prediction else None,
        "expected_answer": float(golden["answer"]),
    }


def calculate_metrics(
    golden_records: Sequence[Mapping[str, Any]],
    predictions: Mapping[int, Mapping[str, Any]],
    execution_ok: Mapping[int, bool],
    *,
    answer_rtol: float = DEFAULT_ANSWER_RTOL,
    answer_atol: float = DEFAULT_ANSWER_ATOL,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Return official aggregate metrics and per-question metric details."""
    details = [
        score_question(
            golden,
            predictions.get(int(golden["id"])),
            execution_ok=execution_ok.get(int(golden["id"]), False),
            answer_rtol=answer_rtol,
            answer_atol=answer_atol,
        )
        for golden in golden_records
    ]
    count = len(details)
    if not count:
        return {name: 0.0 for name in METRIC_NAMES}, []

    def mean(path: tuple[str, ...]) -> float:
        values: list[float] = []
        for detail in details:
            value: Any = detail
            for key in path:
                value = value[key]
            values.append(float(value))
        return sum(values) / count

    metrics = {
        "EXECUTION ACCURACY": mean(("execution_correct",)),
        "TABLES F2-MACRO": mean(("tables", "f2")),
        "DOCS F2-MACRO": mean(("docs", "f2")),
        "TABLES PRECISION": mean(("tables", "precision")),
        "TABLES RECALL": mean(("tables", "recall")),
        "TABLES MRR5": mean(("tables", "mrr5")),
        "DOCS PRECISION": mean(("docs", "precision")),
        "DOCS RECALL": mean(("docs", "recall")),
        "DOCS MRR5": mean(("docs", "mrr5")),
        "ANSWER ACCURACY": mean(("answer_correct",)),
    }
    return metrics, details


def _json_safe(value: Any, *, depth: int = 0) -> Any:
    """Convert graph updates to bounded, readable JSON artifacts."""
    if depth > 12:
        return "<maximum serialization depth reached>"
    if isinstance(value, pd.DataFrame):
        return {
            "type": "DataFrame",
            "rows": len(value),
            "columns": [str(column) for column in value.columns],
            "dtypes": {
                str(column): str(dtype) for column, dtype in value.dtypes.items()
            },
            "sample_rows": json.loads(
                value.head(8).to_json(orient="records", force_ascii=False)
            ),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item, depth=depth + 1) for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item, depth=depth + 1) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


def _rankings_from_candidates(candidates: Any) -> dict[str, list[str]]:
    docs: list[str] = []
    tables: list[str] = []
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        return {"docs": docs, "tables": tables}
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        metadata = candidate.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        doc_id = metadata.get("doc_id")
        start_line = metadata.get("start_line")
        if isinstance(doc_id, str) and doc_id:
            docs.append(doc_id)
            if isinstance(start_line, int) and not isinstance(start_line, bool):
                tables.append(f"{doc_id}|{start_line}")
    return {"docs": _ordered_unique(docs), "tables": _ordered_unique(tables)}


def trace_graph_question(question: str, max_attempts: int) -> dict[str, Any]:
    """Run one graph stream and retain serializable output from every node."""
    started_at = _utc_now()
    start_clock = time.monotonic()
    events: list[dict[str, Any]] = []
    answer_record: dict[str, Any] | None = None
    stages = {
        "retriever": {"docs": [], "tables": []},
        "reranker": {"docs": [], "tables": []},
        "selector": {"docs": [], "tables": []},
    }
    selector_diagnostics: dict[str, Any] | None = None
    error: dict[str, str] | None = None

    try:
        stream = graph.stream(
            {"question": question, "max_attempts": max_attempts},
            stream_mode="updates",
        )
        for chunk in stream:
            if not isinstance(chunk, Mapping):
                events.append(
                    {
                        "sequence": len(events) + 1,
                        "node": "unknown",
                        "elapsed_seconds": time.monotonic() - start_clock,
                        "output": _json_safe(chunk),
                    }
                )
                continue
            for node_name, output in chunk.items():
                events.append(
                    {
                        "sequence": len(events) + 1,
                        "node": str(node_name),
                        "elapsed_seconds": time.monotonic() - start_clock,
                        "output": _json_safe(output),
                    }
                )
                if not isinstance(output, Mapping):
                    continue
                if node_name == "retrieve_tables":
                    stages["retriever"] = _rankings_from_candidates(
                        output.get("candidates")
                    )
                elif node_name == "rerank_tables":
                    stages["reranker"] = _rankings_from_candidates(
                        output.get("reranked_tables")
                    )
                elif node_name == "select_tables":
                    stages["selector"] = _rankings_from_candidates(
                        output.get("retrieved_tables")
                    )
                    raw_selector_diagnostics = output.get("selector_diagnostics")
                    if isinstance(raw_selector_diagnostics, Mapping):
                        selector_diagnostics = dict(raw_selector_diagnostics)
                candidate_answer = output.get("answer_record")
                if isinstance(candidate_answer, Mapping):
                    answer_record = dict(candidate_answer)
    except Exception as exc:
        logger.exception("Validation question failed: %s", question)
        raw_selector_diagnostics = getattr(exc, "diagnostics", None)
        if isinstance(raw_selector_diagnostics, Mapping):
            selector_diagnostics = dict(raw_selector_diagnostics)
        error = {"type": type(exc).__name__, "message": str(exc)}

    if answer_record is None and error is None:
        error = {
            "type": "RuntimeError",
            "message": "Graph stream completed without an answer_record",
        }
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "duration_seconds": time.monotonic() - start_clock,
        "events": events,
        "stage_rankings": stages,
        "selector_diagnostics": selector_diagnostics,
        "answer_record": answer_record,
        "error": error,
    }


def _prepare_run_directory(
    output_root: Path,
    *,
    requested_run_id: str | None,
    resume: bool,
) -> tuple[str, Path]:
    if resume and not requested_run_id:
        raise ValueError("--resume requires an explicit --run-id")
    run_id = _validate_run_id(requested_run_id or _new_run_id())
    output_root = output_root.resolve()
    run_dir = output_root / RUNS_DIRECTORY_NAME / run_id
    if run_dir.exists() and not resume:
        raise FileExistsError(
            f"Run already exists: {run_dir}. Use --resume with the same --run-id."
        )
    if resume and not run_dir.is_dir():
        raise FileNotFoundError(f"Cannot resume missing run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts" / "questions").mkdir(parents=True, exist_ok=True)
    _write_json(
        {
            "run_id": run_id,
            "run_dir": run_dir.relative_to(output_root).as_posix(),
            "updated_at": _utc_now(),
        },
        output_root / "latest_run.json",
    )
    return run_id, run_dir


def _status_counts(questions: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    statuses = [str(item.get("status") or "pending") for item in questions.values()]
    return {
        "selected": len(statuses),
        "pending": statuses.count("pending"),
        "running": statuses.count("running"),
        "succeeded": statuses.count("succeeded"),
        "failed": statuses.count("failed"),
    }


def _write_status(status: dict[str, Any], path: Path) -> None:
    status["updated_at"] = _utc_now()
    status["counts"] = _status_counts(status["questions"])
    _write_json(status, path)


def _run_config(
    args: argparse.Namespace,
    *,
    golden_path: Path,
    golden_sha256: str,
    selected_ids: list[int],
) -> dict[str, Any]:
    return {
        "golden_path": str(golden_path.resolve()),
        "golden_sha256": golden_sha256,
        "selected_question_ids": selected_ids,
        "max_attempts": args.max_attempts,
        "answer_rtol": args.answer_rtol,
        "answer_atol": args.answer_atol,
        "llm_model": SETTINGS.llm_model,
        "llm_temperature": SETTINGS.llm_temperature,
        "qdrant_collection": SETTINGS.qdrant_collection,
        "embedding_model": SETTINGS.embedding_model,
        "embedding_revision": SETTINGS.embedding_revision,
        "fpt_reranker": effective_fpt_reranker_config(SETTINGS),
        "concurrency": args.concurrency,
        "force": bool(args.force),
    }


def _initialize_status(
    *,
    run_id: str,
    run_dir: Path,
    command: str,
    selected: Sequence[Mapping[str, Any]],
    records: Mapping[int, Mapping[str, Any]],
    config: Mapping[str, Any],
    resume: bool,
    force: bool,
) -> dict[str, Any]:
    status_path = run_dir / "status.json"
    immutable_fields = {
        "golden_path",
        "golden_sha256",
        "selected_question_ids",
        "max_attempts",
        "answer_rtol",
        "answer_atol",
        "llm_model",
        "llm_temperature",
        "qdrant_collection",
        "embedding_model",
        "embedding_revision",
        "fpt_reranker",
    }
    if resume:
        if not status_path.is_file():
            raise FileNotFoundError(
                f"Missing status file for resumed run: {status_path}"
            )
        with status_path.open(encoding="utf-8") as handle:
            status = json.load(handle)
        if status.get("run_id") != run_id or status.get("command") != command:
            raise ValueError("Resumed run does not match run-id or command")
        old_config = status.get("config", {})
        changed = sorted(
            field
            for field in immutable_fields
            if old_config.get(field) != config.get(field)
        )
        if changed:
            raise ValueError(
                "Resumed run changed immutable config: " + ", ".join(changed)
            )
    else:
        status = {
            "schema_version": STATUS_SCHEMA_VERSION,
            "run_id": run_id,
            "command": command,
            "state": "running",
            "created_at": _utc_now(),
            "finished_at": None,
            "config": dict(config),
            "questions": {},
            "invocations": [],
        }

    status["state"] = "running"
    status["finished_at"] = None
    status.setdefault("invocations", []).append(
        {
            "number": len(status.get("invocations", [])) + 1,
            "started_at": _utc_now(),
            "concurrency": config["concurrency"],
            "force": force,
        }
    )
    tracked = status.setdefault("questions", {})
    for golden in selected:
        question_id = int(golden["id"])
        key = str(question_id)
        previous = tracked.get(key, {})
        already_done = question_id in records and not force
        tracked[key] = {
            "id": question_id,
            "question": str(golden["question"]),
            "status": "succeeded" if already_done else "pending",
            "run_attempts": int(previous.get("run_attempts", 0)),
            "execution_ok": bool(previous.get("execution_ok", already_done)),
            "answer": records[question_id].get("answer") if already_done else None,
            "error": None,
            "started_at": previous.get("started_at") if already_done else None,
            "finished_at": previous.get("finished_at") if already_done else None,
            "artifact": f"artifacts/questions/{question_id}.json",
        }
    _write_status(status, status_path)
    return status


def _mark_question(
    status: dict[str, Any],
    status_path: Path,
    question_id: int,
    state: str,
    *,
    answer: float | None = None,
    error: str | None = None,
) -> None:
    item = status["questions"][str(question_id)]
    item["status"] = state
    if state == "running":
        item["run_attempts"] = int(item.get("run_attempts", 0)) + 1
        item["started_at"] = _utc_now()
        item["finished_at"] = None
        item["execution_ok"] = False
    elif state in {"succeeded", "failed"}:
        item["finished_at"] = _utc_now()
        item["execution_ok"] = state == "succeeded"
    item["answer"] = answer
    item["error"] = error
    _write_status(status, status_path)


def _append_failure(
    path: Path,
    *,
    golden: Mapping[str, Any],
    error: Mapping[str, Any],
    run_attempt: int,
) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "id": int(golden["id"]),
                    "question": golden["question"],
                    "error": error.get("message", "unknown error"),
                    "error_type": error.get("type", "Error"),
                    "run_attempt": run_attempt,
                    "failed_at": _utc_now(),
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def _score_stage(
    golden: Mapping[str, Any], stage: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    return {
        "tables": _score_ranked(
            list(stage.get("tables", [])), list(golden["relevant_tables"])
        ),
        "docs": _score_ranked(
            list(stage.get("docs", [])), list(golden["relevant_docs"])
        ),
    }


def _write_question_artifact(
    run_dir: Path,
    *,
    golden: Mapping[str, Any],
    trace: Mapping[str, Any],
    prediction: Mapping[str, Any] | None,
    run_attempt: int,
    answer_rtol: float,
    answer_atol: float,
) -> None:
    execution_ok = trace.get("error") is None and prediction is not None
    artifact = dict(trace)
    artifact.update(
        {
            "id": int(golden["id"]),
            "question": golden["question"],
            "run_attempt": run_attempt,
            "golden": dict(golden),
            "prediction": dict(prediction) if prediction else None,
            "metrics": score_question(
                golden,
                prediction,
                execution_ok=execution_ok,
                answer_rtol=answer_rtol,
                answer_atol=answer_atol,
            ),
            "stage_metrics": {
                name: _score_stage(golden, stage)
                for name, stage in dict(trace.get("stage_rankings", {})).items()
            },
        }
    )
    question_id = int(golden["id"])
    latest_path = run_dir / "artifacts" / "questions" / f"{question_id}.json"
    attempt_path = (
        run_dir
        / "artifacts"
        / "questions"
        / str(question_id)
        / f"attempt-{run_attempt}.json"
    )
    _write_json(artifact, attempt_path)
    _write_json(artifact, latest_path)


def _aggregate_stage_metrics(
    run_dir: Path,
    selected: Sequence[Mapping[str, Any]],
    stage_name: str,
) -> dict[str, float]:
    scored: list[dict[str, Any]] = []
    for golden in selected:
        artifact_path = (
            run_dir / "artifacts" / "questions" / f"{int(golden['id'])}.json"
        )
        if not artifact_path.is_file():
            continue
        with artifact_path.open(encoding="utf-8") as handle:
            artifact = json.load(handle)
        stage = artifact.get("stage_rankings", {}).get(stage_name, {})
        scored.append(_score_stage(golden, stage))
    if not scored:
        return {
            "TABLES F2-MACRO": 0.0,
            "DOCS F2-MACRO": 0.0,
            "TABLES PRECISION": 0.0,
            "TABLES RECALL": 0.0,
            "TABLES MRR5": 0.0,
            "DOCS PRECISION": 0.0,
            "DOCS RECALL": 0.0,
            "DOCS MRR5": 0.0,
        }

    def mean(scope: str, metric: str) -> float:
        return sum(float(item[scope][metric]) for item in scored) / len(scored)

    return {
        "TABLES F2-MACRO": mean("tables", "f2"),
        "DOCS F2-MACRO": mean("docs", "f2"),
        "TABLES PRECISION": mean("tables", "precision"),
        "TABLES RECALL": mean("tables", "recall"),
        "TABLES MRR5": mean("tables", "mrr5"),
        "DOCS PRECISION": mean("docs", "precision"),
        "DOCS RECALL": mean("docs", "recall"),
        "DOCS MRR5": mean("docs", "mrr5"),
    }


def _write_metrics(
    run_id: str,
    run_dir: Path,
    selected: Sequence[Mapping[str, Any]],
    records: Mapping[int, Mapping[str, Any]],
    status: Mapping[str, Any],
    *,
    answer_rtol: float,
    answer_atol: float,
) -> dict[str, Any]:
    completed_ids = {
        int(key)
        for key, item in status["questions"].items()
        if item.get("status") in {"succeeded", "failed"}
    }
    evaluated = [record for record in selected if int(record["id"]) in completed_ids]
    execution_ok = {
        int(key): bool(item.get("execution_ok"))
        for key, item in status["questions"].items()
        if int(key) in completed_ids
    }
    metrics, details = calculate_metrics(
        evaluated,
        records,
        execution_ok,
        answer_rtol=answer_rtol,
        answer_atol=answer_atol,
    )
    payload = {
        "run_id": run_id,
        "updated_at": _utc_now(),
        "selected_queries": len(selected),
        "evaluated_queries": len(evaluated),
        "answer_tolerance": {"rtol": answer_rtol, "atol": answer_atol},
        "metrics": metrics,
        "diagnostic_stage_metrics": {
            "retriever": _aggregate_stage_metrics(run_dir, evaluated, "retriever"),
            "reranker": _aggregate_stage_metrics(run_dir, evaluated, "reranker"),
            "selector": _aggregate_stage_metrics(run_dir, evaluated, "selector"),
        },
    }
    _write_json(payload, run_dir / "metrics.json")
    _write_jsonl(details, run_dir / "metrics_per_question.jsonl")
    return payload


def _finish_status(status: dict[str, Any], path: Path) -> None:
    counts = _status_counts(status["questions"])
    status["state"] = "completed_with_failures" if counts["failed"] else "completed"
    status["finished_at"] = _utc_now()
    _write_status(status, path)


def _add_file_logger(run_dir: Path, verbose: bool) -> logging.Handler:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler = logging.FileHandler(run_dir / "artifacts" / "run.log", encoding="utf-8")
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    root_logger.addHandler(handler)
    return handler


def _run(args: argparse.Namespace) -> int:
    golden_path = args.golden.resolve()
    golden = load_golden(golden_path)
    selected = _select_golden(args, golden)
    selected_ids = [int(record["id"]) for record in selected]

    run_id, run_dir = _prepare_run_directory(
        args.output_dir,
        requested_run_id=args.run_id,
        resume=args.resume,
    )
    if not args.resume:
        write_run_manifest(
            run_dir,
            settings=SETTINGS,
            arguments=vars(args),
            question_files=[golden_path],
            tolerances={
                "answer_rtol": args.answer_rtol,
                "answer_atol": args.answer_atol,
            },
            retries={
                "max_attempts": args.max_attempts,
                "concurrency": args.concurrency,
            },
        )
    file_handler = _add_file_logger(run_dir, args.verbose)
    try:
        submission_path = run_dir / "submission.json"
        records = load_existing_submission(submission_path)
        if args.force:
            for question_id in selected_ids:
                records.pop(question_id, None)
            write_submission_json(records, submission_path)

        config = _run_config(
            args,
            golden_path=golden_path,
            golden_sha256=_sha256(golden_path),
            selected_ids=selected_ids,
        )
        status = _initialize_status(
            run_id=run_id,
            run_dir=run_dir,
            command=args.command,
            selected=selected,
            records=records,
            config=config,
            resume=args.resume,
            force=args.force,
        )
        status_path = run_dir / "status.json"
        failures_path = run_dir / "failures.jsonl"
        retry_ids = _retry_error_ids(failures_path, args.retry_error)
        if retry_ids is not None:
            failure_messages = _failure_messages(failures_path)
            for record in selected:
                qid = int(record["id"])
                if qid not in records and qid not in retry_ids:
                    status["questions"][str(qid)]["status"] = "failed"
                    status["questions"][str(qid)]["error"] = failure_messages.get(qid)
            _write_status(status, status_path)

        pending = [
            record
            for record in selected
            if int(record["id"]) not in records
            and (retry_ids is None or int(record["id"]) in retry_ids)
        ]
        print(
            f"{len(selected)} selected, {len(selected) - len(pending)} already done, "
            f"{len(pending)} to run (concurrency={args.concurrency}, run={run_id})"
        )

        def worker(golden_record: Mapping[str, Any]) -> dict[str, Any]:
            question_id = int(golden_record["id"])
            with _write_lock:
                _mark_question(status, status_path, question_id, "running")
            return trace_graph_question(
                str(golden_record["question"]), args.max_attempts
            )

        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {executor.submit(worker, record): record for record in pending}
            for future in as_completed(futures):
                golden_record = futures[future]
                question_id = int(golden_record["id"])
                try:
                    trace = future.result()
                except (
                    Exception
                ) as exc:  # defensive: worker normally captures graph errors
                    logger.exception("Unhandled worker failure id=%s", question_id)
                    trace = {
                        "schema_version": TRACE_SCHEMA_VERSION,
                        "events": [],
                        "stage_rankings": {},
                        "answer_record": None,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    }

                raw_answer = trace.get("answer_record")
                error = trace.get("error")
                prediction: dict[str, Any] | None = None
                if isinstance(raw_answer, Mapping) and error is None:
                    if int(raw_answer.get("id", -1)) != question_id:
                        error = {
                            "type": "ValueError",
                            "message": (
                                f"Graph returned id={raw_answer.get('id')} for "
                                f"validation id={question_id}"
                            ),
                        }
                        trace = dict(trace)
                        trace["error"] = error
                    else:
                        try:
                            prediction = to_submission_item(dict(raw_answer), run_dir)
                        except Exception as exc:
                            logger.exception(
                                "Could not package validation prediction id=%s",
                                question_id,
                            )
                            error = {
                                "type": type(exc).__name__,
                                "message": f"Could not package prediction: {exc}",
                            }
                            trace = dict(trace)
                            trace["error"] = error

                with _write_lock:
                    _write_question_artifact(
                        run_dir,
                        golden=golden_record,
                        trace=trace,
                        prediction=prediction,
                        run_attempt=status["questions"][str(question_id)][
                            "run_attempts"
                        ],
                        answer_rtol=args.answer_rtol,
                        answer_atol=args.answer_atol,
                    )
                    if prediction is not None:
                        records[question_id] = prediction
                        write_submission_json(records, submission_path)
                        _mark_question(
                            status,
                            status_path,
                            question_id,
                            "succeeded",
                            answer=float(prediction["answer"]),
                        )
                        print(f"OK   id={question_id}")
                    else:
                        error_value = (
                            error
                            if isinstance(error, Mapping)
                            else {
                                "type": "RuntimeError",
                                "message": "Question failed without an error payload",
                            }
                        )
                        _append_failure(
                            failures_path,
                            golden=golden_record,
                            error=error_value,
                            run_attempt=status["questions"][str(question_id)][
                                "run_attempts"
                            ],
                        )
                        _mark_question(
                            status,
                            status_path,
                            question_id,
                            "failed",
                            error=str(error_value.get("message")),
                        )
                        print(
                            f"FAIL id={question_id}: {error_value.get('message')}",
                            file=sys.stderr,
                        )
                    _write_metrics(
                        run_id,
                        run_dir,
                        selected,
                        records,
                        status,
                        answer_rtol=args.answer_rtol,
                        answer_atol=args.answer_atol,
                    )

        with _write_lock:
            write_submission_json(records, submission_path)
            build_zip(run_dir, run_dir / "submission.zip")
            _finish_status(status, status_path)
            metrics_payload = _write_metrics(
                run_id,
                run_dir,
                selected,
                records,
                status,
                answer_rtol=args.answer_rtol,
                answer_atol=args.answer_atol,
            )

        counts = status["counts"]
        print(
            f"Done. succeeded={counts['succeeded']} failed={counts['failed']} "
            f"run={run_id} dir={run_dir}"
        )
        for name in METRIC_NAMES:
            print(f"{name}: {metrics_payload['metrics'][name]:.6f}")
        return 1 if counts["failed"] else 0
    finally:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the labelled ViFinQA validation set, create a submission package, "
            "and compute retrieval/answer/execution metrics."
        )
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=DEFAULT_GOLDEN_PATH,
        help="labelled validation JSON (default: ./golden_100.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="validation output root (default: ./val_submission)",
    )
    parser.add_argument(
        "--run-id", help="run directory name; autogenerated when omitted"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an existing --run-id and retry only unfinished/failed questions",
    )
    parser.add_argument("--max-attempts", type=int, choices=range(1, 6), default=2)
    parser.add_argument("--answer-rtol", type=float, default=DEFAULT_ANSWER_RTOL)
    parser.add_argument("--answer-atol", type=float, default=DEFAULT_ANSWER_ATOL)
    parser.add_argument("--verbose", action="store_true")

    commands = parser.add_subparsers(dest="command", required=True)
    ids = commands.add_parser("ids", help="run specific validation question ids")
    ids.add_argument("--ids", required=True, help="comma-separated validation ids")
    ids.add_argument("--concurrency", type=int, default=5)
    ids.add_argument("--limit", type=int, help="run only the first N selected ids")
    ids.add_argument(
        "--force", action="store_true", help="rerun completed ids on resume"
    )
    ids.add_argument(
        "--retry-error",
        help="on --resume, retry only failed ids whose exact error message matches this value",
    )

    full = commands.add_parser(
        "full", help="run every record in the golden validation file"
    )
    full.add_argument("--concurrency", type=int, default=5)
    full.add_argument(
        "--limit", type=int, help="run only the first N validation records"
    )
    full.add_argument(
        "--force", action="store_true", help="rerun completed ids on resume"
    )
    full.add_argument(
        "--retry-error",
        help="on --resume, retry only failed ids whose exact error message matches this value",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if (
        not math.isfinite(args.answer_rtol)
        or not math.isfinite(args.answer_atol)
        or args.answer_rtol < 0
        or args.answer_atol < 0
    ):
        parser.error("answer tolerances must be finite and non-negative")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        return _run(args)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"Cannot initialize validation run: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
