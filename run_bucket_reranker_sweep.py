"""Benchmark legacy global top-20 against bucket-aware top-3/5/8/10.

The bucket pipeline retrieves and reranks every bucket once at depth 10.  The four
per-bucket variants are deterministic prefixes of that frozen result, so the sweep
uses one Qdrant/rewrite/FPT pass per question rather than duplicate passes.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.llm import LLMTransientError
from src.nodes import (
    materialize_buckets_node,
    rerank_bucket_tables_node,
    retrieve_bucket_tables_node,
    rewrite_bucket_queries_node,
)
from src.retrieval import TransientRetrievalError


ROOT = Path(__file__).resolve().parent
DEFAULT_GOLDEN = ROOT / "golden_100.json"
DEFAULT_PARSER_SOURCE = (
    ROOT / "val_submission" / "parser_runs" / "parser-report-v1" / "artifacts" / "questions"
)
DEFAULT_BASELINE = (
    ROOT
    / "val_submission"
    / "fpt_reranker_runs"
    / "fpt-bge-m3-full100-top20-v1"
    / "per_question.json"
)
DEFAULT_OUTPUT = ROOT / "val_submission" / "bucket_reranker_runs"
DEPTHS = (3, 5, 8, 10)
BASELINE_NAME = "baseline_global_top20"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _ordered_unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _score(predicted: Sequence[str], relevant: Sequence[str]) -> dict[str, float]:
    predicted_unique = _ordered_unique(list(predicted))
    relevant_set = set(relevant)
    correct = len(set(predicted_unique) & relevant_set)
    precision = correct / len(predicted_unique) if predicted_unique else 0.0
    recall = correct / len(relevant_set) if relevant_set else 1.0
    denominator = 4.0 * precision + recall
    f2 = 5.0 * precision * recall / denominator if denominator else 0.0
    mrr5 = 0.0
    for rank, item in enumerate(predicted_unique[:5], start=1):
        if item in relevant_set:
            mrr5 = 1.0 / rank
            break
    return {"precision": precision, "recall": recall, "f2": f2, "mrr5": mrr5}


def _candidate_refs(candidates: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[str]]:
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
    return _ordered_unique(docs), _ordered_unique(tables)


def _baseline_by_id(path: Path) -> dict[int, dict[str, Any]]:
    values = _load_json(path)
    if not isinstance(values, list):
        raise ValueError("Baseline per_question artifact must be a list")
    return {int(item["id"]): dict(item) for item in values if isinstance(item, Mapping)}


def _baseline_candidates(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = item.get("fpt_top60")
    if not isinstance(raw, list):
        raw = item.get("fpt_top50")
    if not isinstance(raw, list):
        raise ValueError(f"Baseline id={item.get('id')} has no FPT candidate list")
    return [dict(candidate) for candidate in raw[:20] if isinstance(candidate, Mapping)]


def _load_parser_state(path: Path, golden: Mapping[str, Any]) -> dict[str, Any]:
    artifact = _load_json(path / f"{int(golden['id'])}.json")
    if not isinstance(artifact, Mapping) or artifact.get("error") is not None:
        raise ValueError(f"Parser source is invalid for id={golden['id']}")
    filters = artifact.get("filters")
    semantic_query = artifact.get("semantic_query")
    if not isinstance(filters, Mapping) or not isinstance(semantic_query, str):
        raise ValueError(f"Parser source lacks filters/query for id={golden['id']}")
    base = materialize_buckets_node(
        {"filters": dict(filters), "semantic_query": semantic_query}
    )
    return {
        "question": str(golden["question"]),
        "semantic_query": semantic_query,
        **base,
    }


def _run_question(
    golden: Mapping[str, Any],
    parser_source: Path,
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.monotonic()
    state = _load_parser_state(parser_source, golden)
    for attempt in range(1, 4):
        try:
            state.update(rewrite_bucket_queries_node(state))
            break
        except LLMTransientError:
            if attempt == 3:
                raise
            time.sleep(0.5 * (2 ** (attempt - 1)))
    for attempt in range(1, 4):
        try:
            state.update(retrieve_bucket_tables_node(state))
            break
        except TransientRetrievalError:
            if attempt == 3:
                raise
            time.sleep(0.5 * (2 ** (attempt - 1)))
    state.update(rerank_bucket_tables_node(state))
    bucket_states = state["bucket_states"]
    if not isinstance(bucket_states, Mapping):
        raise TypeError("bucket_states must be an object")

    variants: dict[str, Any] = {}
    baseline_docs, baseline_tables = _candidate_refs(_baseline_candidates(baseline))
    variants[BASELINE_NAME] = {
        "tables": _score(baseline_tables, list(golden["relevant_tables"])),
        "docs": _score(baseline_docs, list(golden["relevant_docs"])),
        "candidate_count": len(baseline_tables),
    }
    for depth in DEPTHS:
        candidates: list[Mapping[str, Any]] = []
        per_bucket_counts: list[int] = []
        for runtime in bucket_states.values():
            if not isinstance(runtime, Mapping):
                continue
            finalists = runtime.get("finalists")
            bucket_candidates = (
                [item for item in finalists if isinstance(item, Mapping)][:depth]
                if isinstance(finalists, list)
                else []
            )
            candidates.extend(bucket_candidates)
            per_bucket_counts.append(len(bucket_candidates))
        docs, tables = _candidate_refs(candidates)
        variants[f"per_bucket_top{depth}"] = {
            "tables": _score(tables, list(golden["relevant_tables"])),
            "docs": _score(docs, list(golden["relevant_docs"])),
            "candidate_count": len(tables),
            "mean_candidates_per_bucket": (
                sum(per_bucket_counts) / len(per_bucket_counts)
                if per_bucket_counts
                else 0.0
            ),
        }
    return {
        "id": int(golden["id"]),
        "question": golden["question"],
        "bucket_count": len(bucket_states),
        "queries": {
            key: runtime.get("query")
            for key, runtime in bucket_states.items()
            if isinstance(runtime, Mapping)
        },
        "pipeline_metrics": state.get("bucket_pipeline_metrics", {}),
        "duration_seconds": time.monotonic() - started,
        "variants": variants,
        "error": None,
    }


def _aggregate(artifacts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    successful = [item for item in artifacts if item.get("error") is None]
    names = [BASELINE_NAME, *(f"per_bucket_top{depth}" for depth in DEPTHS)]
    variants: dict[str, Any] = {}
    for name in names:
        values = [item["variants"][name] for item in successful]
        count = len(values)
        if not count:
            continue
        variants[name] = {
            "tables": {
                metric: sum(float(value["tables"][metric]) for value in values) / count
                for metric in ("precision", "recall", "f2", "mrr5")
            },
            "docs": {
                metric: sum(float(value["docs"][metric]) for value in values) / count
                for metric in ("precision", "recall", "f2", "mrr5")
            },
            "complete_table_coverage": sum(
                float(value["tables"]["recall"]) == 1.0 for value in values
            )
            / count,
            "mean_candidate_count": sum(float(value["candidate_count"]) for value in values)
            / count,
        }
        if name != BASELINE_NAME:
            variants[name]["mean_candidates_per_bucket"] = sum(
                float(value["mean_candidates_per_bucket"]) for value in values
            ) / count

    if not successful:
        return {
            "evaluated_questions": len(artifacts),
            "successful_questions": 0,
            "variants": {},
            "selected_variant": None,
            "promotion_gate": {
                "complete_table_coverage_non_regression": False,
                "document_recall_non_regression": False,
                "all_questions_successful": False,
                "passed": False,
            },
            "mean_duration_seconds": 0.0,
        }

    per_bucket = [f"per_bucket_top{depth}" for depth in DEPTHS]
    best_coverage = max(variants[name]["complete_table_coverage"] for name in per_bucket)
    eligible = [
        name
        for name in per_bucket
        if round(
            best_coverage - variants[name]["complete_table_coverage"], 12
        )
        <= 0.01
    ]
    winner = min(
        eligible,
        key=lambda name: (
            int(name.removeprefix("per_bucket_top")),
            -float(variants[name]["tables"]["f2"]),
        ),
    )
    baseline = variants[BASELINE_NAME]
    selected = variants[winner]
    gate = {
        "complete_table_coverage_non_regression": (
            selected["complete_table_coverage"] >= baseline["complete_table_coverage"]
        ),
        "document_recall_non_regression": (
            selected["docs"]["recall"] >= baseline["docs"]["recall"]
        ),
        "all_questions_successful": len(successful) == len(artifacts),
    }
    gate["passed"] = all(gate.values())
    return {
        "evaluated_questions": len(artifacts),
        "successful_questions": len(successful),
        "variants": variants,
        "selected_variant": winner,
        "promotion_gate": gate,
        "mean_duration_seconds": (
            sum(float(item["duration_seconds"]) for item in successful) / len(successful)
            if successful
            else 0.0
        ),
    }


def _parse_ids(value: str) -> set[int]:
    parsed = {int(part.strip()) for part in value.split(",") if part.strip()}
    if not parsed:
        raise ValueError("--ids must contain at least one id")
    return parsed


def _run_and_persist_question(
    golden_item: Mapping[str, Any],
    parser_source: Path,
    baseline_item: Mapping[str, Any],
    artifact_path: Path,
) -> dict[str, Any]:
    question_id = int(golden_item["id"])
    try:
        artifact = _run_question(golden_item, parser_source, baseline_item)
    except Exception as exc:  # keep the sweep resumable question by question
        artifact = {
            "id": question_id,
            "question": golden_item["question"],
            "duration_seconds": 0.0,
            "variants": {},
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    _atomic_json(artifact_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--parser-source", type=Path, default=DEFAULT_PARSER_SOURCE)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    if not args.run_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in args.run_id):
        raise ValueError("--run-id contains unsupported characters")
    golden = _load_json(args.golden)
    if not isinstance(golden, list):
        raise ValueError("Golden file must contain a list")
    selected = [dict(item) for item in golden if isinstance(item, Mapping)]
    if args.ids:
        ids = _parse_ids(args.ids)
        selected = [item for item in selected if int(item["id"]) in ids]
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        raise ValueError("No benchmark questions selected")
    if args.workers < 1 or args.workers > 8:
        raise ValueError("--workers must be between 1 and 8")

    baseline = _baseline_by_id(args.baseline)
    run_dir = args.output_root / args.run_id
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"Run exists: {run_dir}; pass --resume")
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = run_dir / "questions"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    os.environ["FINLENS_BUCKET_RERANK_TOP_N"] = "10"

    status = {
        "run_id": args.run_id,
        "updated_at": _utc_now(),
        "selected_ids": [int(item["id"]) for item in selected],
        "configurations": [BASELINE_NAME, *(f"per_bucket_top{depth}" for depth in DEPTHS)],
    }
    _atomic_json(run_dir / "status.json", status)
    artifacts_by_id: dict[int, dict[str, Any]] = {}
    pending: list[tuple[dict[str, Any], Path]] = []
    for golden_item in selected:
        question_id = int(golden_item["id"])
        artifact_path = artifacts_dir / f"{question_id}.json"
        if args.resume and artifact_path.is_file():
            artifact = _load_json(artifact_path)
            if isinstance(artifact, Mapping) and artifact.get("error") is None:
                artifacts_by_id[question_id] = dict(artifact)
                continue
        pending.append((golden_item, artifact_path))

    with ThreadPoolExecutor(
        max_workers=min(args.workers, len(pending)) if pending else 1,
        thread_name_prefix="bucket-benchmark",
    ) as executor:
        futures = {
            executor.submit(
                _run_and_persist_question,
                golden_item,
                args.parser_source,
                baseline[int(golden_item["id"])],
                artifact_path,
            ): int(golden_item["id"])
            for golden_item, artifact_path in pending
        }
        for future in as_completed(futures):
            question_id = futures[future]
            artifact = future.result()
            artifacts_by_id[question_id] = artifact
            print(
                json.dumps(
                    {"id": question_id, "error": artifact.get("error")},
                    ensure_ascii=False,
                ),
                flush=True,
            )

    artifacts = [artifacts_by_id[int(item["id"])] for item in selected]

    metrics = _aggregate(artifacts)
    metrics.update({"run_id": args.run_id, "updated_at": _utc_now()})
    _atomic_json(run_dir / "metrics.json", metrics)
    return 0 if metrics["promotion_gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
