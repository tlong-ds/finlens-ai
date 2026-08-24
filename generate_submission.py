"""Generate ViFinQA answers in isolated, resumable experiment runs.

Two subcommands:

    python generate_submission.py [run options] single (--question-id ID | --query "...")
    python generate_submission.py [run options] full [--concurrency N] [--limit N] [--ids 1,2,3] [--force]

Each invocation creates ``<output-dir>/runs/<run-id>/`` with ``status.json``,
``submission.json``, ``failures.jsonl``, evidence data, and ``submission.zip``.
An existing run is only reused when both ``--run-id`` and ``--resume`` are set.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv

from src.graph import graph

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
QUESTIONS_PATH = PROJECT_ROOT / "ViFinQA" / "questions" / "questions.jsonl"
RUNS_DIRECTORY_NAME = "runs"
STATUS_SCHEMA_VERSION = 1
IMMUTABLE_RUN_CONFIG_FIELDS = (
    "max_attempts",
    "llm_model",
    "llm_temperature",
    "qdrant_collection",
    "embedding_model",
    "embedding_revision",
)

_write_lock = threading.Lock()

LOG_PATH = PROJECT_ROOT / "log.json"
_LOG_LOCK_PATH = PROJECT_ROOT / ".log.json.lock"


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp suitable for run metadata."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_log_entry(entry: dict[str, Any]) -> None:
    """Append one entry to the global log.json, safe across threads and processes.

    Uses a sibling ``.log.json.lock`` file as an advisory lock so that concurrent
    runs writing failures at the same time do not corrupt the JSON array.
    The lock is held only during the read-modify-write cycle and released
    immediately after the atomic replace, so contention is negligible.
    """
    import fcntl

    with open(_LOG_LOCK_PATH, "a", encoding="utf-8") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        try:
            if LOG_PATH.exists() and LOG_PATH.stat().st_size > 0:
                with open(LOG_PATH, encoding="utf-8") as fh:
                    entries: list[dict[str, Any]] = json.load(fh)
            else:
                entries = []
            entries.append(entry)
            _write_json(entries, LOG_PATH)
        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)


def _new_run_id() -> str:
    """Create a sortable run identifier with a collision-resistant suffix."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid4().hex[:8]}"


def _validate_run_id(value: str) -> str:
    """Validate a run ID before using it as a directory name."""
    run_id = value.strip()
    if (
        not run_id
        or run_id in {".", ".."}
        or len(run_id) > 128
        or not all(character.isalnum() or character in "._-" for character in run_id)
    ):
        raise ValueError(
            "run-id must contain only letters, numbers, '.', '_' or '-'"
        )
    return run_id


def load_questions() -> list[dict[str, Any]]:
    """Read the canonical ViFinQA question set into [{"id": int, "question": str}, ...]."""
    questions: list[dict[str, Any]] = []
    with open(QUESTIONS_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            questions.append({"id": int(record["id"]), "question": str(record["question"])})
    return questions


def load_existing_submission(json_path: Path) -> dict[int, dict[str, Any]]:
    """Load a prior submission.json (if any), keyed by question id."""
    if not json_path.exists():
        return {}
    with open(json_path, encoding="utf-8") as fh:
        items = json.load(fh)
    return {int(item["id"]): item for item in items}


def run_question(question_text: str, max_attempts: int) -> dict[str, Any]:
    """Invoke the retrieval graph for one canonical question and return its answer_record."""
    result = graph.invoke({"question": question_text, "max_attempts": max_attempts})
    return result["answer_record"]


def to_submission_item(answer_record: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Copy evidence CSVs into output_dir/data and shape the record for submission.json."""
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    evidence = []
    for alias, csv_path in answer_record["evidence"].items():
        destination = data_dir / Path(csv_path).name
        if not destination.exists():
            shutil.copy2(PROJECT_ROOT / csv_path, destination)
        evidence.append({"variable": alias, "csv_path": f"data/{destination.name}"})

    return {
        "id": answer_record["id"],
        "question": answer_record["question"],
        "answer": answer_record["answer"],
        "relevant_docs": answer_record["relevant_docs"],
        "relevant_tables": answer_record["relevant_tables"],
        "evidence": evidence,
        "pandas_query": answer_record["pandas_query"],
    }


def _atomic_replace(target: Path, write_fn: Any, *, suffix: str) -> None:
    """Write via write_fn(tmp_path) into a sibling temp file, then atomically replace target."""
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
    """Atomically write one JSON value."""
    target.parent.mkdir(parents=True, exist_ok=True)

    def _write(tmp_path: Path) -> None:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(value, fh, ensure_ascii=False, indent=2)

    _atomic_replace(target, _write, suffix=".json")


def _prepare_run_directory(
    output_root: Path,
    *,
    requested_run_id: str | None,
    resume: bool,
) -> tuple[str, Path]:
    """Create a fresh run directory or explicitly reopen an existing one."""
    if resume and not requested_run_id:
        raise ValueError("--resume requires an explicit --run-id")

    run_id = _validate_run_id(requested_run_id or _new_run_id())
    runs_dir = output_root.resolve() / RUNS_DIRECTORY_NAME
    run_dir = runs_dir / run_id
    if run_dir.exists() and not resume:
        raise FileExistsError(
            f"Run already exists: {run_dir}. Use --resume with the same --run-id."
        )
    if resume and not run_dir.is_dir():
        raise FileNotFoundError(f"Cannot resume missing run: {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        {
            "run_id": run_id,
            "run_dir": run_dir.relative_to(output_root.resolve()).as_posix(),
            "status_path": (
                run_dir / "status.json"
            ).relative_to(output_root.resolve()).as_posix(),
            "updated_at": _utc_now(),
        },
        output_root.resolve() / "latest_run.json",
    )
    return run_id, run_dir


def _status_counts(questions: dict[str, dict[str, Any]]) -> dict[str, int]:
    """Summarize per-question run states."""
    statuses = [str(item.get("status") or "pending") for item in questions.values()]
    return {
        "selected": len(statuses),
        "pending": statuses.count("pending"),
        "running": statuses.count("running"),
        "succeeded": statuses.count("succeeded"),
        "failed": statuses.count("failed"),
    }


def _write_status(status: dict[str, Any], status_path: Path) -> None:
    """Refresh derived status fields and atomically persist them."""
    status["updated_at"] = _utc_now()
    status["counts"] = _status_counts(status["questions"])
    _write_json(status, status_path)


def _initialize_status(
    *,
    run_id: str,
    run_dir: Path,
    command: str,
    questions: list[dict[str, Any]],
    records: dict[int, dict[str, Any]],
    config: dict[str, Any],
    resume: bool,
    force: bool,
) -> dict[str, Any]:
    """Create or resume run status without mixing incompatible experiments."""
    status_path = run_dir / "status.json"
    if resume:
        if not status_path.is_file():
            raise FileNotFoundError(f"Missing status file for resumed run: {status_path}")
        with open(status_path, encoding="utf-8") as fh:
            status = json.load(fh)
        if status.get("run_id") != run_id or status.get("command") != command:
            raise ValueError("Resumed run does not match run-id or command")
        previous_ids = {int(value) for value in status.get("selected_question_ids", [])}
        selected_ids = {int(question["id"]) for question in questions}
        if previous_ids != selected_ids:
            raise ValueError("Resumed run must use the same selected question IDs")
        original_config = status.get("config", {})
        changed_fields = [
            field
            for field in IMMUTABLE_RUN_CONFIG_FIELDS
            if original_config.get(field) != config.get(field)
        ]
        if changed_fields:
            raise ValueError(
                "Resumed run changed immutable experiment config: "
                + ", ".join(changed_fields)
            )
    else:
        status = {
            "schema_version": STATUS_SCHEMA_VERSION,
            "run_id": run_id,
            "command": command,
            "state": "running",
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "finished_at": None,
            "config": config,
            "selected_question_ids": [int(question["id"]) for question in questions],
            "questions": {},
            "invocations": [],
        }

    status["state"] = "running"
    status["finished_at"] = None
    status.setdefault("invocations", []).append(
        {
            "number": len(status.get("invocations", [])) + 1,
            "started_at": _utc_now(),
            "config": config,
        }
    )
    tracked = status.setdefault("questions", {})
    for question in questions:
        question_id = int(question["id"])
        key = str(question_id)
        previous = tracked.get(key, {})
        has_result = question_id in records and not force
        tracked[key] = {
            "id": question_id,
            "question": str(question["question"]),
            "status": "succeeded" if has_result else "pending",
            "run_attempts": int(previous.get("run_attempts", 0)),
            "error": None,
            "answer": records[question_id].get("answer") if has_result else None,
            "started_at": previous.get("started_at") if has_result else None,
            "finished_at": previous.get("finished_at") if has_result else None,
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
    """Update one question and persist a coherent run snapshot."""
    item = status["questions"][str(question_id)]
    item["status"] = state
    if state == "running":
        item["run_attempts"] = int(item.get("run_attempts", 0)) + 1
        item["started_at"] = _utc_now()
        item["finished_at"] = None
    elif state in {"succeeded", "failed"}:
        item["finished_at"] = _utc_now()
    item["answer"] = answer
    item["error"] = error
    _write_status(status, status_path)


def _finish_status(status: dict[str, Any], status_path: Path) -> None:
    """Mark a run complete while retaining failed-question details for resume."""
    counts = _status_counts(status["questions"])
    status["state"] = "completed_with_failures" if counts["failed"] else "completed"
    status["finished_at"] = _utc_now()
    _write_status(status, status_path)


def write_submission_json(records: dict[int, dict[str, Any]], json_path: Path) -> None:
    items = [records[key] for key in sorted(records)]

    def _write(tmp_path: Path) -> None:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(items, fh, ensure_ascii=False, indent=2)

    _atomic_replace(json_path, _write, suffix=".json")


def build_zip(output_dir: Path, zip_path: Path) -> None:
    def _write(tmp_path: Path) -> None:
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(output_dir / "submission.json", arcname="submission.json")
            data_dir = output_dir / "data"
            if data_dir.exists():
                for csv_file in sorted(data_dir.glob("*.csv")):
                    zf.write(csv_file, arcname=f"data/{csv_file.name}")

    _atomic_replace(zip_path, _write, suffix=".zip")


def save_and_package(records: dict[int, dict[str, Any]], output_dir: Path) -> None:
    """Persist submission.json and rebuild submission.zip, serialized across threads."""
    with _write_lock:
        write_submission_json(records, output_dir / "submission.json")
        build_zip(output_dir, output_dir / "submission.zip")


def _run_config(args: argparse.Namespace, *, selected_ids: list[int]) -> dict[str, Any]:
    """Capture the non-secret settings needed to interpret an experiment run."""
    return {
        "max_attempts": args.max_attempts,
        "selected_question_ids": selected_ids,
        "llm_model": os.getenv("LLM_MODEL"),
        "llm_temperature": os.getenv("LLM_TEMPERATURE", "0"),
        "qdrant_collection": os.getenv("QDRANT_COLLECTION"),
        "embedding_model": os.getenv("EMBEDDING_MODEL"),
        "embedding_revision": os.getenv("EMBEDDING_REVISION"),
        "concurrency": getattr(args, "concurrency", 1),
        "force": bool(getattr(args, "force", False)),
    }


def _append_failure(
    failures_path: Path,
    *,
    question: dict[str, Any],
    error: BaseException,
    attempt: int,
) -> None:
    """Append one timestamped failure attempt to the run-local audit log."""
    with open(failures_path, "a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "id": int(question["id"]),
                    "question": str(question["question"]),
                    "error": str(error),
                    "run_attempt": attempt,
                    "failed_at": _utc_now(),
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def _run_single(args: argparse.Namespace) -> int:
    questions = load_questions()
    if args.question_id is not None:
        by_id = {q["id"]: q for q in questions}
        if args.question_id not in by_id:
            print(f"Unknown question id: {args.question_id}", file=sys.stderr)
            return 1
        question = by_id[args.question_id]
    else:
        matches = [item for item in questions if item["question"] == args.query]
        if len(matches) != 1:
            print("Query must exactly match one canonical question", file=sys.stderr)
            return 1
        question = matches[0]

    try:
        run_id, run_dir = _prepare_run_directory(
            args.output_dir,
            requested_run_id=args.run_id,
            resume=args.resume,
        )
        records = load_existing_submission(run_dir / "submission.json")
        if args.force:
            records.pop(int(question["id"]), None)
        status = _initialize_status(
            run_id=run_id,
            run_dir=run_dir,
            command="single",
            questions=[question],
            records=records,
            config=_run_config(args, selected_ids=[int(question["id"])]),
            resume=args.resume,
            force=args.force,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"Cannot initialize run: {exc}", file=sys.stderr)
        return 2

    status_path = run_dir / "status.json"
    failures_path = run_dir / "failures.jsonl"
    question_id = int(question["id"])
    if question_id in records and not args.force:
        _finish_status(status, status_path)
        print(f"SKIP id={question_id} already completed in run={run_id}")
        return 0

    try:
        _mark_question(status, status_path, question_id, "running")
        answer_record = run_question(str(question["question"]), args.max_attempts)
    except Exception as exc:
        logger.exception("Failed to answer question: %s", question["question"])
        run_attempt = status["questions"][str(question_id)]["run_attempts"]
        _append_failure(
            failures_path,
            question=question,
            error=exc,
            attempt=run_attempt,
        )
        append_log_entry(
            {
                "id": int(question["id"]),
                "question": str(question["question"]),
                "error": str(exc),
                "run_attempt": run_attempt,
                "failed_at": _utc_now(),
            }
        )
        _mark_question(
            status,
            status_path,
            question_id,
            "failed",
            error=str(exc),
        )
        save_and_package(records, run_dir)
        _finish_status(status, status_path)
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    records[question_id] = to_submission_item(answer_record, run_dir)
    save_and_package(records, run_dir)
    _mark_question(
        status,
        status_path,
        question_id,
        "succeeded",
        answer=float(answer_record["answer"]),
    )
    _finish_status(status, status_path)
    print(
        f"OK id={answer_record['id']} answer={answer_record['answer']} "
        f"run={run_id} dir={run_dir}"
    )
    return 0


def _parse_ids(raw: str | None) -> set[int] | None:
    if not raw:
        return None
    return {int(part.strip()) for part in raw.split(",") if part.strip()}


def _run_full(args: argparse.Namespace) -> int:
    questions = load_questions()
    only_ids = _parse_ids(args.ids)
    if only_ids is not None:
        questions = [q for q in questions if q["id"] in only_ids]
    if args.limit is not None:
        questions = questions[: args.limit]

    try:
        run_id, run_dir = _prepare_run_directory(
            args.output_dir,
            requested_run_id=args.run_id,
            resume=args.resume,
        )
        json_path = run_dir / "submission.json"
        records = load_existing_submission(json_path)
        if args.force:
            for question in questions:
                records.pop(int(question["id"]), None)
            write_submission_json(records, json_path)
        status = _initialize_status(
            run_id=run_id,
            run_dir=run_dir,
            command="full",
            questions=questions,
            records=records,
            config=_run_config(
                args,
                selected_ids=[int(question["id"]) for question in questions],
            ),
            resume=args.resume,
            force=args.force,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"Cannot initialize run: {exc}", file=sys.stderr)
        return 2

    done_ids = set() if args.force else set(records)
    pending = [q for q in questions if q["id"] not in done_ids]
    skipped = len(questions) - len(pending)
    print(
        f"{len(questions)} selected, {skipped} already done, {len(pending)} to run "
        f"(concurrency={args.concurrency}, run={run_id})"
    )

    succeeded = 0
    failed_ids: list[int] = []
    failures_path = run_dir / "failures.jsonl"
    status_path = run_dir / "status.json"

    def _run_tracked(question: dict[str, Any]) -> dict[str, Any]:
        with _write_lock:
            _mark_question(
                status,
                status_path,
                int(question["id"]),
                "running",
            )
        return run_question(str(question["question"]), args.max_attempts)

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(_run_tracked, question): question for question in pending
        }
        for future in as_completed(futures):
            question = futures[future]
            try:
                answer_record = future.result()
            except Exception as exc:
                failed_ids.append(question["id"])
                logger.exception("Question id=%s failed", question["id"])
                print(f"FAIL id={question['id']}: {exc}", file=sys.stderr)
                with _write_lock:
                    run_attempt = status["questions"][str(question["id"])]["run_attempts"]
                    _append_failure(
                        failures_path,
                        question=question,
                        error=exc,
                        attempt=run_attempt,
                    )
                    _mark_question(
                        status,
                        status_path,
                        int(question["id"]),
                        "failed",
                        error=str(exc),
                    )
                append_log_entry(
                    {
                        "id": int(question["id"]),
                        "question": str(question["question"]),
                        "error": str(exc),
                        "run_attempt": run_attempt,
                        "failed_at": _utc_now(),
                    }
                )
                continue

            with _write_lock:
                records[int(answer_record["id"])] = to_submission_item(answer_record, run_dir)
                write_submission_json(records, json_path)
                _mark_question(
                    status,
                    status_path,
                    int(question["id"]),
                    "succeeded",
                    answer=float(answer_record["answer"]),
                )
            succeeded += 1
            print(f"OK   id={question['id']}")

    with _write_lock:
        write_submission_json(records, json_path)
        build_zip(run_dir, run_dir / "submission.zip")
        _finish_status(status, status_path)

    print(
        f"Done. already_done={skipped} succeeded={succeeded} failed={len(failed_ids)} "
        f"total_in_submission={len(records)} run={run_id} dir={run_dir}"
    )
    if failed_ids:
        print(f"Failed ids: {sorted(failed_ids)}", file=sys.stderr)
    return 1 if failed_ids else 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate ViFinQA answers and package them into submission.zip."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "submission",
        help=(
            "Experiment output root; each run is stored under runs/<run-id> "
            "(default: ./submission)"
        ),
    )
    parser.add_argument(
        "--run-id",
        help="Run directory name; autogenerated for a new run when omitted",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing --run-id instead of creating a fresh run",
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)

    single = commands.add_parser(
        "single", help="Answer one question in a new or explicitly resumed run"
    )
    target = single.add_mutually_exclusive_group(required=True)
    target.add_argument("--question-id", type=int, help="id from ViFinQA/questions/questions.jsonl")
    target.add_argument("--query", help="exact canonical question text")
    single.add_argument(
        "--force",
        action="store_true",
        help="rerun the question even if the resumed run already has an answer",
    )

    full = commands.add_parser("full", help="Answer all ViFinQA questions concurrently")
    full.add_argument("--concurrency", type=int, default=5)
    full.add_argument("--limit", type=int, help="only run the first N selected questions")
    full.add_argument("--ids", help="comma-separated question ids to restrict to")
    full.add_argument(
        "--force", action="store_true", help="rerun ids even if already present in submission.json"
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.command == "single":
        return _run_single(args)
    if args.command == "full":
        return _run_full(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
