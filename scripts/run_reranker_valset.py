"""Replay frozen FPT top-20 candidates and evaluate only the LLM selector.

Example:

    python run_reranker_valset.py \
        --run-id reranker-04 \
        --source-run val_submission/runs/reranker-03 \
        --concurrency 4

The source run supplies the exact ``rerank_tables`` output for every question.
This runner never calls the parser, Qdrant, generator, or sandbox.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.experiments.provenance import write_run_manifest
from src.pipeline.graph import default_settings as SETTINGS
from src.providers.llm import LLMTransientError
from src.retrieval.selection import select_tables_with_diagnostics

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN_PATH = PROJECT_ROOT / "golden_100.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "val_submission" / "reranker_runs"
SCHEMA_VERSION = 2
PROVIDER_ATTEMPTS = 3
STAGE_NAMES = (
    "reranker",
    "selector_input",
    "scout_union",
    "finalists",
    "final_llm",
    "selector",
)


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


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=".tmp-", suffix=".json"
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=".tmp-", suffix=".jsonl"
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for value in values:
                handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_golden(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or not value:
        raise ValueError("Golden file must be a non-empty JSON array")
    records: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise ValueError(f"Golden record {index} must be an object")
        question_id = raw.get("id")
        question = raw.get("question")
        relevant_docs = raw.get("relevant_docs")
        relevant_tables = raw.get("relevant_tables")
        if (
            isinstance(question_id, bool)
            or not isinstance(question_id, int)
            or question_id in seen_ids
        ):
            raise ValueError(f"Golden record {index} has an invalid or duplicate id")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Golden id={question_id} has an empty question")
        if not isinstance(relevant_docs, list) or not all(
            isinstance(item, str) and item for item in relevant_docs
        ):
            raise ValueError(f"Golden id={question_id} has invalid relevant_docs")
        if not isinstance(relevant_tables, list) or not all(
            isinstance(item, str) and item for item in relevant_tables
        ):
            raise ValueError(f"Golden id={question_id} has invalid relevant_tables")
        seen_ids.add(question_id)
        records.append(
            {
                "id": question_id,
                "question": question,
                "relevant_docs": list(dict.fromkeys(relevant_docs)),
                "relevant_tables": list(dict.fromkeys(relevant_tables)),
            }
        )
    return records


def _parse_ids(value: str) -> set[int]:
    try:
        parsed = {int(part.strip()) for part in value.split(",") if part.strip()}
    except ValueError as exc:
        raise ValueError("--ids must be a comma-separated list of integers") from exc
    if not parsed:
        raise ValueError("--ids must contain at least one id")
    return parsed


def _select_records(
    records: Sequence[Mapping[str, Any]], ids: str | None, limit: int | None
) -> list[dict[str, Any]]:
    selected = [dict(record) for record in records]
    if ids:
        requested = _parse_ids(ids)
        known = {int(record["id"]) for record in records}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError("Unknown validation ids: " + ", ".join(map(str, unknown)))
        selected = [record for record in selected if int(record["id"]) in requested]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("No validation questions selected")
    return selected


def _score_ranked(
    predicted: Sequence[str], relevant: Sequence[str]
) -> dict[str, float]:
    predicted_unique = list(dict.fromkeys(predicted))
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
    return {"precision": precision, "recall": recall, "f2": f2, "mrr5": reciprocal_rank}


def _rankings_from_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    docs: list[str] = []
    tables: list[str] = []
    for candidate in candidates:
        metadata = candidate.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        doc_id = metadata.get("doc_id")
        start_line = metadata.get("start_line")
        if isinstance(doc_id, str) and doc_id:
            docs.append(doc_id)
            if isinstance(start_line, int) and not isinstance(start_line, bool):
                tables.append(f"{doc_id}|{start_line}")
    return {
        "docs": list(dict.fromkeys(docs)),
        "tables": list(dict.fromkeys(tables)),
    }


def _rankings_from_keys(
    diagnostics: Mapping[str, Any], keys: Sequence[str]
) -> dict[str, list[str]]:
    catalog = diagnostics.get("candidate_catalog", {})
    docs: list[str] = []
    tables: list[str] = []
    if not isinstance(catalog, Mapping):
        return {"docs": [], "tables": []}
    for key in keys:
        item = catalog.get(key)
        if not isinstance(item, Mapping):
            continue
        doc_id = item.get("doc_id")
        table_ref = item.get("table_ref")
        if isinstance(doc_id, str):
            docs.append(doc_id)
        if isinstance(table_ref, str):
            tables.append(table_ref)
    return {
        "docs": list(dict.fromkeys(docs)),
        "tables": list(dict.fromkeys(tables)),
    }


def _score_stage(
    golden: Mapping[str, Any], rankings: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    return {
        "tables": _score_ranked(
            list(rankings.get("tables", [])), list(golden["relevant_tables"])
        ),
        "docs": _score_ranked(
            list(rankings.get("docs", [])), list(golden["relevant_docs"])
        ),
    }


def _load_source_candidates(
    source_run: Path, record: Mapping[str, Any]
) -> tuple[Path, list[dict[str, Any]]]:
    question_id = int(record["id"])
    artifact_path = source_run / "artifacts" / "questions" / f"{question_id}.json"
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"Missing source artifact for id={question_id}: {artifact_path}"
        )
    with artifact_path.open(encoding="utf-8") as handle:
        artifact = json.load(handle)
    if (
        not isinstance(artifact, Mapping)
        or artifact.get("question") != record["question"]
    ):
        raise ValueError(f"Source artifact question mismatch for id={question_id}")
    events = artifact.get("events")
    if not isinstance(events, list):
        raise ValueError(f"Source artifact has no events for id={question_id}")
    raw_candidates: Any = None
    for event in events:
        if isinstance(event, Mapping) and event.get("node") == "rerank_tables":
            output = event.get("output")
            if isinstance(output, Mapping):
                raw_candidates = output.get("reranked_tables")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError(
            f"Source artifact has no FPT reranker candidates for id={question_id}"
        )
    if not all(isinstance(candidate, Mapping) for candidate in raw_candidates):
        raise ValueError(
            f"Source artifact has malformed candidates for id={question_id}"
        )
    return artifact_path, [dict(candidate) for candidate in raw_candidates]


def _loss_attribution(
    golden: Mapping[str, Any], stage_rankings: Mapping[str, Mapping[str, Sequence[str]]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for table_ref in golden["relevant_tables"]:
        presence = {
            stage: table_ref in stage_rankings.get(stage, {}).get("tables", [])
            for stage in STAGE_NAMES
        }
        if presence["selector"]:
            terminal = "selected"
        else:
            terminal = next(
                (stage for stage in STAGE_NAMES if not presence[stage]),
                "selector",
            )
        output.append(
            {"table_ref": table_ref, "stage_presence": presence, "terminal": terminal}
        )
    return output


def _run_question(
    record: Mapping[str, Any], source_run: Path, settings=SETTINGS
) -> dict[str, Any]:
    started_at = _utc_now()
    started_clock = time.monotonic()
    source_artifact: Path | None = None
    candidates: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    ranked: list[dict[str, Any]] = []
    error: dict[str, str] | None = None
    provider_attempts = 0
    try:
        source_artifact, candidates = _load_source_candidates(source_run, record)
        for attempt in range(1, PROVIDER_ATTEMPTS + 1):
            provider_attempts = attempt
            try:
                ranked, diagnostics = select_tables_with_diagnostics(
                    str(record["question"]), candidates, settings=settings
                )
                break
            except LLMTransientError:
                if attempt == PROVIDER_ATTEMPTS:
                    raise
                time.sleep(0.5 * (2 ** (attempt - 1)))
    except Exception as exc:
        logger.exception("Selector-only question failed id=%s", record["id"])
        error = {"type": type(exc).__name__, "message": str(exc)}

    stage_rankings = {
        "reranker": _rankings_from_candidates(candidates),
        "selector_input": _rankings_from_keys(
            diagnostics, diagnostics.get("selector_input_keys", [])
        ),
        "scout_union": _rankings_from_keys(
            diagnostics, diagnostics.get("scout_nominated_keys", [])
        ),
        "finalists": _rankings_from_keys(
            diagnostics, diagnostics.get("finalist_keys", [])
        ),
        "final_llm": _rankings_from_keys(
            diagnostics, diagnostics.get("final_llm_keys", [])
        ),
        "selector": _rankings_from_candidates(ranked),
    }
    stage_metrics = {
        stage: _score_stage(record, rankings)
        for stage, rankings in stage_rankings.items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "id": int(record["id"]),
        "question": record["question"],
        "started_at": started_at,
        "finished_at": _utc_now(),
        "duration_seconds": time.monotonic() - started_clock,
        "provider_attempts": provider_attempts,
        "source_artifact": str(source_artifact) if source_artifact else None,
        "golden": dict(record),
        "fpt_reranked_candidates": candidates,
        "selector_diagnostics": diagnostics,
        "selected_candidates": ranked,
        "stage_rankings": stage_rankings,
        "stage_metrics": stage_metrics,
        "loss_attribution": _loss_attribution(record, stage_rankings),
        "error": error,
    }


def _mean_stage_metrics(
    artifacts: Sequence[Mapping[str, Any]], stage: str
) -> dict[str, float]:
    scored = [
        artifact.get("stage_metrics", {}).get(stage, {}) for artifact in artifacts
    ]

    def mean(scope: str, metric: str) -> float:
        if not scored:
            return 0.0
        return sum(
            float(item.get(scope, {}).get(metric, 0.0)) for item in scored
        ) / len(scored)

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


def _question_slice(golden: Mapping[str, Any]) -> str:
    tables = list(golden["relevant_tables"])
    if len(tables) == 1:
        return "single_gold_table"
    counts = Counter(table_ref.rsplit("|", 1)[0] for table_ref in tables)
    if any(count > 1 for count in counts.values()):
        return "multi_table_per_document"
    return "one_table_per_document"


def _aggregate_diagnostics(artifacts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    loss_counts: Counter[str] = Counter()
    absent_by_stage: Counter[str] = Counter()
    required_gold = 0
    required_predicted = 0
    required_correct = 0
    completion_sources: Counter[str] = Counter()
    uncovered_concepts = 0
    coverage_lock_questions = 0
    coverage_locked_buckets = 0
    policy_added_buckets = 0
    lexical_rescue_questions = 0
    lexical_rescue_candidates = 0
    unresolved_required_buckets = 0
    for artifact in artifacts:
        for item in artifact.get("loss_attribution", []):
            loss_counts[str(item.get("terminal"))] += 1
            presence = item.get("stage_presence", {})
            if isinstance(presence, Mapping):
                absent_by_stage.update(
                    stage for stage in STAGE_NAMES if not presence.get(stage, False)
                )
        diagnostics = artifact.get("selector_diagnostics", {})
        catalog = diagnostics.get("candidate_catalog", {})
        bucket_docs: dict[str, str] = {}
        if isinstance(catalog, Mapping):
            for value in catalog.values():
                if isinstance(value, Mapping):
                    bucket_key = value.get("bucket_key")
                    doc_id = value.get("doc_id")
                    if isinstance(bucket_key, str) and isinstance(doc_id, str):
                        bucket_docs[bucket_key] = doc_id
        golden_docs = set(artifact.get("golden", {}).get("relevant_docs", []))
        predicted_keys = diagnostics.get("required_bucket_keys", [])
        predicted_docs = {
            bucket_docs[key]
            for key in predicted_keys
            if isinstance(key, str) and key in bucket_docs
        }
        available_gold_docs = golden_docs & set(bucket_docs.values())
        required_gold += len(available_gold_docs)
        required_predicted += len(predicted_docs)
        required_correct += len(available_gold_docs & predicted_docs)
        completion = diagnostics.get("coverage_completion", {})
        if isinstance(completion, Mapping):
            completion_sources.update(map(str, completion.values()))
        uncovered_concepts += len(diagnostics.get("uncovered_concept_keys", []))
        locked = diagnostics.get("coverage_locked_bucket_keys", [])
        lexical = diagnostics.get("lexical_finalist_keys", [])
        coverage_lock_questions += bool(locked)
        coverage_locked_buckets += len(locked)
        policy_added_buckets += len(
            diagnostics.get("policy_added_required_bucket_keys", [])
        )
        lexical_rescue_questions += bool(lexical)
        lexical_rescue_candidates += len(lexical)
        unresolved_required_buckets += len(
            diagnostics.get("unresolved_required_bucket_keys", [])
        )
    return {
        "gold_table_terminal_counts": dict(sorted(loss_counts.items())),
        "gold_table_absent_by_stage": dict(sorted(absent_by_stage.items())),
        "required_bucket_gold_recall": (
            required_correct / required_gold if required_gold else 1.0
        ),
        "required_bucket_precision": (
            required_correct / required_predicted if required_predicted else 0.0
        ),
        "coverage_completion_sources": dict(sorted(completion_sources.items())),
        "uncovered_concept_count": uncovered_concepts,
        "coverage_lock_questions": coverage_lock_questions,
        "coverage_locked_bucket_count": coverage_locked_buckets,
        "policy_added_required_bucket_count": policy_added_buckets,
        "lexical_rescue_questions": lexical_rescue_questions,
        "lexical_rescue_candidate_count": lexical_rescue_candidates,
        "unresolved_required_bucket_count": unresolved_required_buckets,
    }


def _write_summary(
    run_dir: Path,
    run_id: str,
    source_run: Path,
    selected_count: int,
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ordered = sorted(artifacts, key=lambda item: int(item["id"]))
    stage_metrics = {
        stage: _mean_stage_metrics(ordered, stage) for stage in STAGE_NAMES
    }
    slice_metrics: dict[str, Any] = {}
    for slice_name in (
        "single_gold_table",
        "one_table_per_document",
        "multi_table_per_document",
    ):
        subset = [
            artifact
            for artifact in ordered
            if _question_slice(artifact["golden"]) == slice_name
        ]
        slice_metrics[slice_name] = {
            "queries": len(subset),
            "selector": _mean_stage_metrics(subset, "selector"),
        }
    payload = {
        "run_id": run_id,
        "updated_at": _utc_now(),
        "source_run": str(source_run),
        "selected_queries": selected_count,
        "evaluated_queries": len(ordered),
        "failed_queries": sum(
            artifact.get("error") is not None for artifact in ordered
        ),
        "stage_metrics": stage_metrics,
        "slice_metrics": slice_metrics,
        "diagnostics": _aggregate_diagnostics(ordered),
    }
    _atomic_json(run_dir / "metrics.json", payload)
    _atomic_jsonl(
        run_dir / "metrics_per_question.jsonl",
        [
            {
                "id": artifact["id"],
                "stage_metrics": artifact["stage_metrics"],
                "loss_attribution": artifact["loss_attribution"],
                "error": artifact["error"],
            }
            for artifact in ordered
        ],
    )
    return payload


def _validate_source_run(source_run: Path, golden_sha256: str) -> dict[str, Any]:
    status_path = source_run / "status.json"
    if not status_path.is_file():
        raise FileNotFoundError(f"Missing source status: {status_path}")
    with status_path.open(encoding="utf-8") as handle:
        status = json.load(handle)
    if not isinstance(status, Mapping):
        raise ValueError("Source status must be an object")
    source_hash = status.get("config", {}).get("golden_sha256")
    if source_hash != golden_sha256:
        raise ValueError("Source run and --golden use different golden files")
    return dict(status)


def _run(args: argparse.Namespace) -> int:
    golden_path = args.golden.resolve()
    golden_sha256 = _sha256(golden_path)
    records = load_golden(golden_path)
    selected = _select_records(records, args.ids, args.limit)
    source_run = args.source_run.resolve()
    _validate_source_run(source_run, golden_sha256)
    run_id = _validate_run_id(args.run_id or _new_run_id())
    run_dir = args.output_dir.resolve() / run_id
    status_path = run_dir / "status.json"
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"Reranker run already exists: {run_dir}")
    if args.resume:
        if not status_path.is_file():
            raise FileNotFoundError(f"Missing selector status: {status_path}")
        with status_path.open(encoding="utf-8") as handle:
            status = json.load(handle)
        config = status.get("config", {})
        if (
            config.get("golden_sha256") != golden_sha256
            or config.get("source_run") != str(source_run)
            or config.get("selected_question_ids")
            != [int(record["id"]) for record in selected]
        ):
            raise ValueError("Resumed selector run changed immutable config")
    else:
        run_dir.mkdir(parents=True)
        write_run_manifest(
            run_dir,
            settings=SETTINGS,
            arguments=vars(args),
            question_files=[golden_path],
            retries={
                "provider_attempts": PROVIDER_ATTEMPTS,
                "concurrency": args.concurrency,
            },
        )
        status = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "state": "running",
            "created_at": _utc_now(),
            "finished_at": None,
            "config": {
                "golden_path": str(golden_path),
                "golden_sha256": golden_sha256,
                "source_run": str(source_run),
                "selected_question_ids": [int(record["id"]) for record in selected],
                "llm_model": SETTINGS.llm_model,
                "llm_temperature": SETTINGS.llm_temperature,
                "concurrency": args.concurrency,
            },
            "questions": {},
        }

    log_path = run_dir / "artifacts" / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logging.getLogger().addHandler(file_handler)
    artifacts_by_id: dict[int, dict[str, Any]] = {}
    try:
        pending: list[dict[str, Any]] = []
        for record in selected:
            question_id = int(record["id"])
            artifact_path = run_dir / "artifacts" / "questions" / f"{question_id}.json"
            if args.resume and artifact_path.is_file():
                with artifact_path.open(encoding="utf-8") as handle:
                    artifact = json.load(handle)
                if artifact.get("error") is None:
                    artifacts_by_id[question_id] = artifact
                    status["questions"][str(question_id)] = {
                        "status": "succeeded",
                        "artifact": f"artifacts/questions/{question_id}.json",
                    }
                    continue
            pending.append(record)
            status["questions"][str(question_id)] = {"status": "pending"}
        _atomic_json(status_path, status)
        print(
            f"{len(selected)} selected, {len(artifacts_by_id)} already done, "
            f"{len(pending)} to run (concurrency={args.concurrency}, run={run_id})"
        )
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {
                executor.submit(_run_question, record, source_run): record
                for record in pending
            }
            for future in as_completed(futures):
                artifact = future.result()
                question_id = int(artifact["id"])
                artifacts_by_id[question_id] = artifact
                artifact_path = (
                    run_dir / "artifacts" / "questions" / f"{question_id}.json"
                )
                _atomic_json(artifact_path, artifact)
                succeeded = artifact["error"] is None
                status["questions"][str(question_id)] = {
                    "status": "succeeded" if succeeded else "failed",
                    "error": artifact["error"],
                    "artifact": f"artifacts/questions/{question_id}.json",
                }
                status["updated_at"] = _utc_now()
                _atomic_json(status_path, status)
                _write_summary(
                    run_dir,
                    run_id,
                    source_run,
                    len(selected),
                    list(artifacts_by_id.values()),
                )
                print(("OK  " if succeeded else "FAIL") + f" id={question_id}")

        artifacts = [artifacts_by_id[int(record["id"])] for record in selected]
        metrics = _write_summary(run_dir, run_id, source_run, len(selected), artifacts)
        failures = sum(artifact["error"] is not None for artifact in artifacts)
        status["state"] = "completed_with_failures" if failures else "completed"
        status["finished_at"] = _utc_now()
        status["counts"] = {
            "selected": len(selected),
            "succeeded": len(selected) - failures,
            "failed": failures,
        }
        _atomic_json(status_path, status)
        print(
            json.dumps(
                metrics["stage_metrics"]["selector"], ensure_ascii=False, indent=2
            )
        )
        return 1 if failures else 0
    finally:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay frozen FPT top-20 candidates and evaluate only the selector"
    )
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--ids", help="optional comma-separated validation ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        return _run(args)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"Cannot initialize selector run: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
