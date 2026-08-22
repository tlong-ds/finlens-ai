"""Evaluate FinLens query parsing and table retrieval on the validation set.

This runner stops after dense or hybrid retrieval.  It never calls reranking,
code generation, or the sandbox.  It writes isolated artifacts under
``val_submission/parser_runs/<run-id>/``.  Retrieval can use either the parser's
semantic query or the original question, and parser artifacts can be replayed so
that retrieval experiments use identical filters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
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

from dotenv import load_dotenv

from src.llm import LLMTransientError
from src.parser import parse_query_with_diagnostics
from src.retrieval import (
    RETRIEVAL_MODE_DEFAULT,
    RETRIEVAL_TOP_K,
    NoMatchingCandidatesError,
    TransientRetrievalError,
    retrieve,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_GOLDEN_PATH = PROJECT_ROOT / "golden_100.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "val_submission" / "parser_runs"
SCHEMA_VERSION = 1
_PROVIDER_ATTEMPTS = 3
_RETRIEVAL_ATTEMPTS = 3
DEFAULT_TOP_KS = (5, 10, 20, RETRIEVAL_TOP_K)
_DOC_ID_PATTERN = re.compile(
    r"^(?P<ticker>.+)_financial_statements_(?P<year>\d{4})"
    r"(?:_(?P<report_type>consolidated|separate|aggregated))?$"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]


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
    seen: set[int] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"Golden record {index} must be an object")
        question_id = item.get("id")
        question = item.get("question")
        relevant_docs = item.get("relevant_docs")
        relevant_tables = item.get("relevant_tables")
        if (
            isinstance(question_id, bool)
            or not isinstance(question_id, int)
            or question_id in seen
        ):
            raise ValueError(f"Golden record {index} has an invalid or duplicate id")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Golden id={question_id} has an empty question")
        if not isinstance(relevant_docs, list) or not all(
            isinstance(doc_id, str) and doc_id for doc_id in relevant_docs
        ):
            raise ValueError(f"Golden id={question_id} has invalid relevant_docs")
        if not isinstance(relevant_tables, list) or not all(
            isinstance(table_ref, str) and table_ref for table_ref in relevant_tables
        ):
            raise ValueError(f"Golden id={question_id} has invalid relevant_tables")
        seen.add(question_id)
        records.append(
            {
                "id": question_id,
                "question": question,
                "relevant_docs": list(dict.fromkeys(relevant_docs)),
                "relevant_tables": list(dict.fromkeys(relevant_tables)),
            }
        )
    return records


def derive_golden_filters(record: Mapping[str, Any]) -> dict[str, list[str | int]]:
    """Derive parser labels from canonical golden document identifiers."""
    tickers: list[str] = []
    years: list[int] = []
    report_types: list[str] = []
    for doc_id in record["relevant_docs"]:
        match = _DOC_ID_PATTERN.fullmatch(str(doc_id))
        if not match:
            raise ValueError(f"Unsupported golden doc id: {doc_id}")
        tickers.append(match.group("ticker"))
        years.append(int(match.group("year")))
        report_types.append(match.group("report_type") or "other")
    return {
        "ticker": list(dict.fromkeys(tickers)),
        "year": list(dict.fromkeys(years)),
        "report_type": list(dict.fromkeys(report_types)),
    }


def _dimension_score(expected: Sequence[Any], predicted: Sequence[Any]) -> dict[str, Any]:
    expected_set = set(expected)
    predicted_set = set(predicted)
    return {
        "exact": expected_set == predicted_set,
        "coverage": expected_set <= predicted_set,
        "missing": sorted(expected_set - predicted_set),
        "extra": sorted(predicted_set - expected_set),
    }


def score_parser_output(
    expected: Mapping[str, Sequence[str | int]],
    filters: Mapping[str, Sequence[str | int]] | None,
) -> dict[str, Any]:
    predicted = filters or {}
    return {
        field: _dimension_score(expected[field], predicted.get(field, []))
        for field in ("ticker", "year", "report_type")
    }


def _ordered_unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def _score_ranked(
    predicted: Sequence[str], relevant: Sequence[str]
) -> dict[str, float]:
    """Score one ranking with the same formulas used by ``run_valset.py``."""
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


def _rankings_from_candidates(
    candidates: Sequence[Mapping[str, Any]], top_k: int
) -> dict[str, list[str]]:
    docs: list[str] = []
    tables: list[str] = []
    for candidate in candidates[:top_k]:
        metadata = candidate.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        doc_id = metadata.get("doc_id")
        start_line = metadata.get("start_line")
        if not isinstance(doc_id, str) or not doc_id:
            continue
        docs.append(doc_id)
        if isinstance(start_line, int) and not isinstance(start_line, bool):
            tables.append(f"{doc_id}|{start_line}")
    return {
        "docs": _ordered_unique(docs),
        "tables": _ordered_unique(tables),
    }


def score_retrieval_output(
    golden: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    top_ks: Sequence[int],
) -> dict[str, dict[str, Any]]:
    """Score every requested prefix of one dense-retrieval ranking."""
    scored: dict[str, dict[str, Any]] = {}
    for top_k in top_ks:
        rankings = _rankings_from_candidates(candidates, top_k)
        scored[str(top_k)] = {
            "candidate_count": min(top_k, len(candidates)),
            "rankings": rankings,
            "tables": _score_ranked(
                rankings["tables"], list(golden["relevant_tables"])
            ),
            "docs": _score_ranked(
                rankings["docs"], list(golden["relevant_docs"])
            ),
        }
    return scored


def _changed_repair_fields(diagnostics: Mapping[str, Any]) -> list[str]:
    attempts = diagnostics.get("attempts", [])
    if not isinstance(attempts, list) or len(attempts) < 2:
        return []
    first = attempts[0].get("raw_filters")
    final = attempts[-1].get("raw_filters")
    if not isinstance(first, Mapping) or not isinstance(final, Mapping):
        return []
    return sorted(
        field
        for field in set(first) | set(final)
        if first.get(field) != final.get(field)
    )


def aggregate_parser_metrics(artifacts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(artifacts)
    succeeded = [
        artifact
        for artifact in artifacts
        if (
            artifact.get("parser_error") is None
            if "parser_error" in artifact
            else artifact.get("error") is None
        )
    ]
    semantic_calls = [
        int(artifact.get("diagnostics", {}).get("semantic_attempts", 0))
        for artifact in artifacts
    ]
    provider_attempts = [int(artifact.get("provider_attempts", 0)) for artifact in artifacts]
    aggregate: dict[str, Any] = {
        "evaluated_queries": count,
        "successful_queries": len(succeeded),
        "parser_success_rate": len(succeeded) / count if count else 0.0,
        "mean_semantic_calls_per_question": (
            sum(semantic_calls) / count if count else 0.0
        ),
        "semantic_retry_questions": sum(calls > 1 for calls in semantic_calls),
        "mean_provider_attempts_per_question": (
            sum(provider_attempts) / count if count else 0.0
        ),
        "provider_retry_questions": sum(attempts > 1 for attempts in provider_attempts),
    }
    for field in ("ticker", "year", "report_type"):
        scores = [artifact["metrics"][field] for artifact in artifacts]
        aggregate[f"{field}_exact_set_accuracy"] = (
            sum(bool(score["exact"]) for score in scores) / count if count else 0.0
        )
        aggregate[f"{field}_gold_coverage"] = (
            sum(bool(score["coverage"]) for score in scores) / count if count else 0.0
        )
        aggregate[f"{field}_overselection_questions"] = sum(
            bool(score["extra"]) for score in scores
        )

    report_confusion: Counter[str] = Counter()
    report_class_total: Counter[str] = Counter()
    report_class_correct: Counter[str] = Counter()
    repair_mutations: Counter[str] = Counter()
    for artifact in artifacts:
        expected_reports = artifact["expected_filters"]["report_type"]
        filters = artifact.get("filters") or {}
        predicted_reports = filters.get("report_type", [])
        expected_label = "+".join(sorted(map(str, expected_reports)))
        predicted_label = (
            "+".join(sorted(map(str, predicted_reports))) or "<error-or-missing>"
        )
        report_confusion[f"{expected_label}->{predicted_label}"] += 1
        if len(expected_reports) == 1:
            report_class_total[str(expected_reports[0])] += 1
            if set(expected_reports) == set(predicted_reports):
                report_class_correct[str(expected_reports[0])] += 1
        repair_mutations.update(artifact.get("repair_changed_fields", []))

    aggregate["report_confusion"] = dict(sorted(report_confusion.items()))
    aggregate["report_accuracy_by_class"] = {
        label: {
            "correct": report_class_correct[label],
            "total": total,
            "accuracy": report_class_correct[label] / total if total else 0.0,
        }
        for label, total in sorted(report_class_total.items())
    }
    aggregate["repair_changed_fields"] = dict(sorted(repair_mutations.items()))
    return aggregate


def aggregate_retrieval_health(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, float | int]:
    """Summarize whether dense retrieval ran successfully and needed retries."""
    count = len(artifacts)
    attempts = [int(artifact.get("retrieval_attempts", 0)) for artifact in artifacts]
    succeeded = sum(
        isinstance(artifact.get("retrieval"), Mapping)
        and artifact.get("parser_error") is None
        and artifact.get("retrieval_error") is None
        for artifact in artifacts
    )
    return {
        "retrieval_successful_queries": succeeded,
        "retrieval_success_rate": succeeded / count if count else 0.0,
        "mean_retrieval_attempts_per_question": (
            sum(attempts) / count if count else 0.0
        ),
        "retrieval_retry_questions": sum(attempt > 1 for attempt in attempts),
    }


def aggregate_retrieval_metrics(
    artifacts: Sequence[Mapping[str, Any]], top_ks: Sequence[int]
) -> dict[str, dict[str, float]]:
    """Macro-average retrieval rankings, counting failed retrievals as empty."""
    aggregate: dict[str, dict[str, float]] = {}
    for top_k in top_ks:
        per_question: list[Mapping[str, Any]] = []
        for artifact in artifacts:
            retrieval = artifact.get("retrieval")
            scored = (
                retrieval.get("metrics_by_top_k", {}).get(str(top_k))
                if isinstance(retrieval, Mapping)
                else None
            )
            if isinstance(scored, Mapping):
                per_question.append(scored)
                continue
            golden = artifact.get("golden_retrieval") or {
                "relevant_docs": [],
                "relevant_tables": [],
            }
            per_question.append(
                score_retrieval_output(golden, [], [top_k])[str(top_k)]
            )

        count = len(per_question)

        def mean(scope: str, metric: str) -> float:
            if not count:
                return 0.0
            return sum(
                float(item[scope][metric]) for item in per_question
            ) / count

        aggregate[str(top_k)] = {
            "TABLES F2-MACRO": mean("tables", "f2"),
            "DOCS F2-MACRO": mean("docs", "f2"),
            "TABLES PRECISION": mean("tables", "precision"),
            "TABLES RECALL": mean("tables", "recall"),
            "TABLES MRR5": mean("tables", "mrr5"),
            "DOCS PRECISION": mean("docs", "precision"),
            "DOCS RECALL": mean("docs", "recall"),
            "DOCS MRR5": mean("docs", "mrr5"),
        }
    return aggregate


def _retrieve_balanced(
    query_text: str,
    filters: Mapping[str, Sequence[str | int]],
    *,
    top_n: int,
    retrieval_mode: str = "dense",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Mirror the graph's per-ticker balanced retrieval."""
    raw_tickers = filters.get("ticker", [])
    if not isinstance(raw_tickers, list) or not raw_tickers or not all(
        isinstance(ticker, str) and ticker for ticker in raw_tickers
    ):
        raise ValueError("Filter ticker must be a non-empty string array")
    tickers = list(dict.fromkeys(raw_tickers))
    quota = (top_n + len(tickers) - 1) // len(tickers)
    bucket_results: list[list[dict[str, Any]]] = []
    bucket_diagnostics: list[dict[str, Any]] = []

    for ticker in tickers:
        bucket_filters = {**filters, "ticker": [ticker]}
        effective_filters = dict(bucket_filters)
        relaxed_report_type = False
        try:
            bucket = retrieve(
                query_text=query_text,
                filters=bucket_filters,
                top_n=quota,
                mode=retrieval_mode,
            )
        except NoMatchingCandidatesError:
            if not bucket_filters.get("report_type"):
                raise
            effective_filters.pop("report_type", None)
            relaxed_report_type = True
            bucket = retrieve(
                query_text=query_text,
                filters=effective_filters,
                top_n=quota,
                mode=retrieval_mode,
            )
        bucket_results.append(bucket)
        bucket_diagnostics.append(
            {
                "ticker": ticker,
                "requested_top_n": quota,
                "candidate_count": len(bucket),
                "retrieval_mode": retrieval_mode,
                "relaxed_report_type": relaxed_report_type,
                "effective_filters": effective_filters,
            }
        )

    candidates: list[dict[str, Any]] = []
    seen_table_ids: set[str] = set()
    max_bucket_size = max((len(bucket) for bucket in bucket_results), default=0)
    for bucket_rank in range(max_bucket_size):
        for bucket in bucket_results:
            if bucket_rank >= len(bucket):
                continue
            candidate = dict(bucket[bucket_rank])
            table_id = str(candidate.get("table_id") or "")
            if not table_id or table_id in seen_table_ids:
                continue
            seen_table_ids.add(table_id)
            candidate["retrieval_rank"] = len(candidates) + 1
            candidates.append(candidate)
            if len(candidates) == top_n:
                break
        if len(candidates) == top_n:
            break
    return candidates, bucket_diagnostics


def _compact_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "rank": index,
            "table_id": candidate.get("table_id"),
            "retrieval_score": candidate.get("retrieval_score"),
            "retrieval_rank": candidate.get("retrieval_rank", index),
            "dense_score": candidate.get("dense_score"),
            "dense_rank": candidate.get("dense_rank"),
            "bm25_score": candidate.get("bm25_score"),
            "bm25_rank": candidate.get("bm25_rank"),
            "rrf_score": candidate.get("rrf_score"),
            "metadata": dict(candidate.get("metadata") or {}),
        }
        for index, candidate in enumerate(candidates, start=1)
    ]


def _parse_with_transient_retry(
    record: Mapping[str, Any],
    top_ks: Sequence[int],
    *,
    retrieval_query_source: str = "semantic",
    retrieval_mode: str = "dense",
    parser_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    question_id = int(record["id"])
    started_at = _utc_now()
    started_clock = time.monotonic()
    provider_attempts = 0
    diagnostics: dict[str, Any] = {}
    filters: dict[str, list[str | int]] | None = None
    semantic_query: str | None = None
    parser_error: dict[str, str] | None = None
    if retrieval_query_source not in {"semantic", "question"}:
        raise ValueError("retrieval_query_source must be semantic or question")
    if parser_source is not None:
        source_filters = parser_source.get("filters")
        source_semantic_query = parser_source.get("semantic_query")
        source_diagnostics = parser_source.get("diagnostics", {})
        source_error = parser_source.get("parser_error")
        if (
            int(parser_source.get("id", -1)) != question_id
            or parser_source.get("question") != record["question"]
            or source_error is not None
            or not isinstance(source_filters, Mapping)
            or not isinstance(source_semantic_query, str)
            or not source_semantic_query.strip()
            or not isinstance(source_diagnostics, Mapping)
        ):
            raise ValueError(f"Parser source artifact is invalid for id={question_id}")
        filters = {
            str(field): list(values)
            for field, values in source_filters.items()
            if isinstance(values, list)
        }
        semantic_query = source_semantic_query
        diagnostics = dict(source_diagnostics)
        provider_attempts = int(parser_source.get("provider_attempts", 0))
    else:
        for provider_attempt in range(1, _PROVIDER_ATTEMPTS + 1):
            provider_attempts = provider_attempt
            try:
                result = parse_query_with_diagnostics(
                    str(record["question"]), question_id=question_id
                )
                filters = dict(result["filters"])
                semantic_query = str(result["semantic_query"])
                diagnostics = dict(result["diagnostics"])
                break
            except LLMTransientError as exc:
                if provider_attempt == _PROVIDER_ATTEMPTS:
                    parser_error = {
                        "stage": "parser",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                    break
                time.sleep(0.5 * (2 ** (provider_attempt - 1)))
            except Exception as exc:
                possible_diagnostics = getattr(exc, "diagnostics", {})
                if isinstance(possible_diagnostics, Mapping):
                    diagnostics = dict(possible_diagnostics)
                parser_error = {
                    "stage": "parser",
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                break

    expected = derive_golden_filters(record)
    golden_retrieval = {
        "relevant_docs": list(record["relevant_docs"]),
        "relevant_tables": list(record["relevant_tables"]),
    }
    candidates: list[dict[str, Any]] = []
    bucket_diagnostics: list[dict[str, Any]] = []
    retrieval_attempts = 0
    retrieval_error: dict[str, str] | None = None
    retrieval_query = (
        str(record["question"])
        if retrieval_query_source == "question"
        else semantic_query
    )
    if parser_error is None and filters is not None and semantic_query is not None:
        for retrieval_attempt in range(1, _RETRIEVAL_ATTEMPTS + 1):
            retrieval_attempts = retrieval_attempt
            try:
                candidates, bucket_diagnostics = _retrieve_balanced(
                    str(retrieval_query),
                    filters,
                    top_n=max(top_ks),
                    retrieval_mode=retrieval_mode,
                )
                break
            except TransientRetrievalError as exc:
                if retrieval_attempt == _RETRIEVAL_ATTEMPTS:
                    retrieval_error = {
                        "stage": "retrieval",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                    break
                time.sleep(0.5 * (2 ** (retrieval_attempt - 1)))
            except Exception as exc:
                retrieval_error = {
                    "stage": "retrieval",
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                break

    retrieval_metrics = score_retrieval_output(
        golden_retrieval,
        candidates,
        top_ks,
    )
    error = parser_error or retrieval_error
    return {
        "schema_version": SCHEMA_VERSION,
        "id": question_id,
        "question": record["question"],
        "started_at": started_at,
        "finished_at": _utc_now(),
        "duration_seconds": time.monotonic() - started_clock,
        "provider_attempts": provider_attempts,
        "retrieval_attempts": retrieval_attempts,
        "expected_filters": expected,
        "golden_retrieval": golden_retrieval,
        "filters": filters,
        "semantic_query": semantic_query,
        "diagnostics": diagnostics,
        "repair_changed_fields": _changed_repair_fields(diagnostics),
        "metrics": score_parser_output(expected, filters),
        "retrieval": {
            "mode": retrieval_mode,
            "query_source": retrieval_query_source,
            "query_text": retrieval_query,
            "requested_top_ks": list(top_ks),
            "max_top_k": max(top_ks),
            "bucket_diagnostics": bucket_diagnostics,
            "candidates": _compact_candidates(candidates),
            "metrics_by_top_k": retrieval_metrics,
        },
        "parser_error": parser_error,
        "retrieval_error": retrieval_error,
        "error": error,
    }


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
    return selected


def _write_summary(
    run_dir: Path,
    run_id: str,
    artifacts: Sequence[Mapping[str, Any]],
    top_ks: Sequence[int],
) -> dict[str, Any]:
    metrics = aggregate_parser_metrics(artifacts)
    metrics.update(aggregate_retrieval_health(artifacts))
    metrics["retrieval_by_top_k"] = aggregate_retrieval_metrics(artifacts, top_ks)
    payload = {"run_id": run_id, "updated_at": _utc_now(), "metrics": metrics}
    _atomic_json(
        run_dir / "metrics.json",
        payload,
    )
    details = [
        {
            "id": artifact["id"],
            "metrics": artifact["metrics"],
            "error": artifact["error"],
            "semantic_attempts": artifact.get("diagnostics", {}).get(
                "semantic_attempts", 0
            ),
            "repair_changed_fields": artifact.get("repair_changed_fields", []),
            "retrieval_attempts": artifact.get("retrieval_attempts", 0),
            "retrieval_by_top_k": artifact.get("retrieval", {}).get(
                "metrics_by_top_k", {}
            ),
        }
        for artifact in sorted(artifacts, key=lambda item: int(item["id"]))
    ]
    _atomic_jsonl(run_dir / "metrics_per_question.jsonl", details)
    return payload


def _run(args: argparse.Namespace) -> int:
    golden_path = args.golden.resolve()
    records = load_golden(golden_path)
    selected = _select_records(records, args.ids, args.limit)
    if not selected:
        raise ValueError("No validation questions selected")
    run_id = _validate_run_id(args.run_id or _new_run_id())
    run_dir = args.output_dir.resolve() / run_id
    status_path = run_dir / "status.json"
    parser_source_dir = (
        args.parser_source_run.resolve() if args.parser_source_run else None
    )
    parser_sources: dict[int, dict[str, Any]] = {}
    if parser_source_dir is not None:
        for record in selected:
            question_id = int(record["id"])
            source_path = (
                parser_source_dir / "artifacts" / "questions" / f"{question_id}.json"
            )
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"Missing parser source artifact for id={question_id}: {source_path}"
                )
            with source_path.open(encoding="utf-8") as handle:
                source = json.load(handle)
            if not isinstance(source, Mapping):
                raise ValueError(f"Invalid parser source artifact: {source_path}")
            parser_sources[question_id] = dict(source)
    config = {
        "golden_path": str(golden_path),
        "golden_sha256": _sha256(golden_path),
        "selected_question_ids": [int(record["id"]) for record in selected],
        "llm_model": os.getenv("LLM_MODEL"),
        "llm_temperature": os.getenv("LLM_TEMPERATURE", "0"),
        "qdrant_collection": os.getenv("QDRANT_COLLECTION"),
        "embedding_model": os.getenv("EMBEDDING_MODEL"),
        "embedding_revision": os.getenv("EMBEDDING_REVISION"),
        "retrieval_top_ks": list(args.top_ks),
        "retrieval_mode": args.retrieval_mode,
        "retrieval_query_source": args.retrieval_query,
        "parser_source_run": str(parser_source_dir) if parser_source_dir else None,
        "concurrency": args.concurrency,
    }
    if run_dir.exists() and not args.resume:
        raise FileExistsError(
            f"Parser run already exists: {run_dir}. Use --resume to continue it."
        )
    if args.resume:
        if not status_path.is_file():
            raise FileNotFoundError(f"Missing parser status file: {status_path}")
        with status_path.open(encoding="utf-8") as handle:
            status = json.load(handle)
        for field in (
            "golden_sha256",
            "selected_question_ids",
            "llm_model",
            "llm_temperature",
            "qdrant_collection",
            "embedding_model",
            "embedding_revision",
            "retrieval_top_ks",
            "retrieval_mode",
            "retrieval_query_source",
            "parser_source_run",
        ):
            if status.get("config", {}).get(field) != config[field]:
                raise ValueError(f"Resumed parser run changed immutable config: {field}")
    else:
        run_dir.mkdir(parents=True)
        status = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "state": "running",
            "created_at": _utc_now(),
            "finished_at": None,
            "config": config,
            "questions": {},
        }

    log_path = run_dir / "artifacts" / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
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
                executor.submit(
                    _parse_with_transient_retry,
                    record,
                    args.top_ks,
                    retrieval_query_source=args.retrieval_query,
                    retrieval_mode=args.retrieval_mode,
                    parser_source=parser_sources.get(int(record["id"])),
                ): record
                for record in pending
            }
            for future in as_completed(futures):
                artifact = future.result()
                question_id = int(artifact["id"])
                artifacts_by_id[question_id] = artifact
                _atomic_json(
                    run_dir / "artifacts" / "questions" / f"{question_id}.json",
                    artifact,
                )
                succeeded = artifact["error"] is None
                status["questions"][str(question_id)] = {
                    "status": "succeeded" if succeeded else "failed",
                    "error": artifact["error"],
                    "artifact": f"artifacts/questions/{question_id}.json",
                }
                _atomic_json(status_path, status)
                _write_summary(
                    run_dir,
                    run_id,
                    list(artifacts_by_id.values()),
                    args.top_ks,
                )
                print(("OK  " if succeeded else "FAIL") + f" id={question_id}")

        artifacts = [artifacts_by_id[int(record["id"])] for record in selected]
        metrics_payload = _write_summary(run_dir, run_id, artifacts, args.top_ks)
        metrics = metrics_payload["metrics"]
        failures = sum(artifact.get("error") is not None for artifact in artifacts)
        status["state"] = "completed_with_failures" if failures else "completed"
        status["finished_at"] = _utc_now()
        status["counts"] = {
            "selected": len(selected),
            "succeeded": len(selected) - failures,
            "failed": failures,
        }
        _atomic_json(status_path, status)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return 1 if failures else 0
    finally:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()


def _parse_top_ks(value: str) -> tuple[int, ...]:
    try:
        parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError("--top-k must be a comma-separated list of integers") from exc
    if not parsed or any(top_k < 1 for top_k in parsed):
        raise ValueError("--top-k values must be positive integers")
    return tuple(sorted(set(parsed)))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate FinLens query parsing and dense/hybrid retrieval"
    )
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--ids", help="optional comma-separated validation ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--top-k",
        default=",".join(map(str, DEFAULT_TOP_KS)),
        help="comma-separated retrieval cutoffs (default: 5,10,20,50)",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=("dense", "hybrid"),
        default=RETRIEVAL_MODE_DEFAULT,
        help="retrieval strategy (default: hybrid)",
    )
    parser.add_argument(
        "--retrieval-query",
        choices=("semantic", "question"),
        default="semantic",
        help="text embedded for dense retrieval (default: semantic)",
    )
    parser.add_argument(
        "--parser-source-run",
        type=Path,
        help="reuse filters/semantic queries from an existing parser run directory",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    try:
        args.top_ks = _parse_top_ks(args.top_k)
    except ValueError as exc:
        parser.error(str(exc))
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        return _run(args)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"Cannot initialize parser run: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
