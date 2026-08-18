"""Generate ViFinQA answers via the retrieval graph and package a submission.zip.

Two subcommands:

    python generate_submission.py single (--question-id ID | --query "...")
    python generate_submission.py full [--concurrency N] [--limit N] [--ids 1,2,3] [--force]

Both write into the same --output-dir (default: ./submission), producing
submission.json, data/<csv files>, and submission.zip. Re-running either
subcommand upserts into the existing package rather than starting over.
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
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.graph import graph

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
QUESTIONS_PATH = PROJECT_ROOT / "ViFinQA" / "questions" / "questions.jsonl"

_write_lock = threading.Lock()


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


def _run_single(args: argparse.Namespace) -> int:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.question_id is not None:
        by_id = {q["id"]: q["question"] for q in load_questions()}
        if args.question_id not in by_id:
            print(f"Unknown question id: {args.question_id}", file=sys.stderr)
            return 1
        question_text = by_id[args.question_id]
    else:
        question_text = args.query

    try:
        answer_record = run_question(question_text, args.max_attempts)
    except Exception as exc:
        logger.exception("Failed to answer question: %s", question_text)
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    records = load_existing_submission(output_dir / "submission.json")
    records[int(answer_record["id"])] = to_submission_item(answer_record, output_dir)
    save_and_package(records, output_dir)
    print(f"OK id={answer_record['id']} answer={answer_record['answer']}")
    return 0


def _parse_ids(raw: str | None) -> set[int] | None:
    if not raw:
        return None
    return {int(part.strip()) for part in raw.split(",") if part.strip()}


def _run_full(args: argparse.Namespace) -> int:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    questions = load_questions()
    only_ids = _parse_ids(args.ids)
    if only_ids is not None:
        questions = [q for q in questions if q["id"] in only_ids]
    if args.limit is not None:
        questions = questions[: args.limit]

    json_path = output_dir / "submission.json"
    records = load_existing_submission(json_path)

    done_ids = set() if args.force else set(records)
    pending = [q for q in questions if q["id"] not in done_ids]
    skipped = len(questions) - len(pending)
    print(
        f"{len(questions)} selected, {skipped} already done, {len(pending)} to run "
        f"(concurrency={args.concurrency})"
    )

    succeeded = 0
    failed_ids: list[int] = []
    failures_path = output_dir / "failures.jsonl"

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(run_question, q["question"], args.max_attempts): q for q in pending
        }
        for future in as_completed(futures):
            question = futures[future]
            try:
                answer_record = future.result()
            except Exception as exc:
                failed_ids.append(question["id"])
                logger.exception("Question id=%s failed", question["id"])
                print(f"FAIL id={question['id']}: {exc}", file=sys.stderr)
                with _write_lock, open(failures_path, "a", encoding="utf-8") as fh:
                    fh.write(
                        json.dumps(
                            {"id": question["id"], "question": question["question"], "error": str(exc)},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                continue

            with _write_lock:
                records[int(answer_record["id"])] = to_submission_item(answer_record, output_dir)
                write_submission_json(records, json_path)
                build_zip(output_dir, output_dir / "submission.zip")
            succeeded += 1
            print(f"OK   id={question['id']}")

    print(
        f"Done. already_done={skipped} succeeded={succeeded} failed={len(failed_ids)} "
        f"total_in_submission={len(records)}"
    )
    if failed_ids:
        print(f"Failed ids: {sorted(failed_ids)}", file=sys.stderr)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate ViFinQA answers and package them into submission.zip."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "submission",
        help="Directory holding submission.json, data/, and submission.zip (default: ./submission)",
    )
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)

    single = commands.add_parser(
        "single", help="Answer one question and upsert it into the submission package"
    )
    target = single.add_mutually_exclusive_group(required=True)
    target.add_argument("--question-id", type=int, help="id from ViFinQA/questions/questions.jsonl")
    target.add_argument("--query", help="exact canonical question text")

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
