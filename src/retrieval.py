"""Dense retrieval and hierarchical LLM reranking for financial tables."""

from __future__ import annotations

import json
import logging
import math
import csv
import re
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import httpx
from qdrant_client import models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from src.contracts import (
    FILTER_FIELDS,
    resolve_csv_path,
    validate_qdrant_payload,
)
from src.embeddings import (
    DENSE_VECTOR_NAME,
    DenseEmbeddingModel,
    EmbeddingError,
)
from src.llm import LLMResponseError, generate_structured
from src.prompt import RERANK_SYSTEM_PROMPT, build_rerank_prompt
from src.qdrant import QdrantConnectionError, get_collection_name, get_qdrant_client

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RETRIEVAL_TOP_K = 50
RERANK_TOP_K = 10
RERANK_BATCH_SIZE = 10
RERANK_BATCH_SHORTLIST = 5
RERANK_RESPONSE_ATTEMPTS = 2
RERANK_CONTEXT_SCAN_ROWS = 500
RERANK_CONTEXT_MAX_ROWS = 12
RERANK_CONTEXT_MAX_COLUMNS = 24
RERANK_CONTEXT_MAX_CELL_CHARS = 160
RERANK_CONTEXT_MAX_CHARS = 6_000

_QUESTION_STOP_WORDS = frozenset(
    {
        "bao",
        "bằng",
        "của",
        "cuối",
        "đến",
        "đồng",
        "là",
        "năm",
        "ngày",
        "nhiêu",
        "số",
        "tháng",
        "theo",
        "trong",
        "triệu",
        "tỷ",
        "vào",
    }
)

Candidate = dict[str, Any]


class RetrievalError(RuntimeError):
    """Raised when table retrieval fails."""


class TransientRetrievalError(RetrievalError):
    """Raised for temporary Qdrant failures that are safe to retry unchanged."""


class RerankerError(RuntimeError):
    """Raised when table reranking fails."""


def build_qdrant_filter(
    filters: Mapping[str, Sequence[str | int]] | None,
) -> models.Filter | None:
    """Translate validated metadata constraints into a strict Qdrant filter."""
    raw_filters = filters or {}
    unknown = set(raw_filters) - set(FILTER_FIELDS)
    if unknown:
        raise ValueError("Unsupported Qdrant filters: " + ", ".join(sorted(unknown)))

    conditions: list[models.FieldCondition] = []
    for field in FILTER_FIELDS:
        raw_values = raw_filters.get(field, [])
        if isinstance(raw_values, (str, bytes)):
            raise TypeError(f"Filter {field} must be a sequence, not a string")
        values = list(raw_values)
        if values:
            if field == "year":
                if any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in values
                ):
                    raise TypeError("Filter year must contain only integers")
                match = models.MatchAny(any=cast(list[int], values))
            else:
                if any(not isinstance(value, str) for value in values):
                    raise TypeError(f"Filter {field} must contain only strings")
                match = models.MatchAny(any=cast(list[str], values))
            conditions.append(models.FieldCondition(key=field, match=match))
    return models.Filter(must=cast(Any, conditions)) if conditions else None


@lru_cache(maxsize=1)
def _get_embedding_model() -> DenseEmbeddingModel:
    return DenseEmbeddingModel.from_env()


def embed_query(query_text: str) -> list[float]:
    """Embed a non-empty semantic query with the shared Granite contract."""
    if not query_text.strip():
        raise RetrievalError("Semantic query must not be empty")
    try:
        return _get_embedding_model().encode_queries([query_text.strip()])[0]
    except (EmbeddingError, TypeError, ValueError) as exc:
        raise RetrievalError("Query embedding failed") from exc


def _point_to_candidate(point: Any, dense_rank: int) -> Candidate:
    try:
        payload = validate_qdrant_payload(dict(point.payload or {}))
        retrieval_score = float(point.score)
    except (TypeError, ValueError) as exc:
        raise RetrievalError(
            f"Qdrant point {point.id} có dữ liệu không hợp lệ"
        ) from exc
    if not math.isfinite(retrieval_score):
        raise RetrievalError(
            f"Qdrant point {point.id} có retrieval score không hữu hạn"
        )
    return {
        "table_id": payload["table_id"],
        "metadata": payload,
        "retrieval_score": retrieval_score,
        "dense_rank": dense_rank,
    }


def retrieve(
    query_text: str,
    filters: Mapping[str, Sequence[str | int]] | None = None,
    *,
    top_n: int = RETRIEVAL_TOP_K,
) -> list[Candidate]:
    """Embed a semantic query and return Top-N validated Qdrant candidates."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    try:
        response = get_qdrant_client().query_points(
            collection_name=get_collection_name(),
            query=embed_query(query_text),
            using=DENSE_VECTOR_NAME,
            query_filter=build_qdrant_filter(filters),
            limit=top_n,
            with_payload=True,
            with_vectors=False,
        )
        candidates = [
            _point_to_candidate(point, dense_rank=index)
            for index, point in enumerate(response.points, start=1)
        ]
    except (RetrievalError, QdrantConnectionError):
        raise
    except (httpx.TransportError, ResponseHandlingException) as exc:
        raise TransientRetrievalError("Temporary Qdrant retrieval failure") from exc
    except UnexpectedResponse as exc:
        status_code = exc.status_code
        if status_code in {408, 409, 425, 429} or (
            status_code is not None and status_code >= 500
        ):
            raise TransientRetrievalError("Temporary Qdrant retrieval failure") from exc
        raise RetrievalError("Qdrant table retrieval failed") from exc
    except Exception as exc:
        raise RetrievalError("Qdrant table retrieval failed") from exc

    if not candidates:
        raise RetrievalError("Không tìm thấy bảng nào khớp metadata filter")
    logger.info("Retrieved %d table candidates", len(candidates))
    logger.debug(
        "Retrieval scores: %s",
        [(item["table_id"], item["retrieval_score"]) for item in candidates],
    )
    return candidates


def _compact_cell(value: Any) -> str:
    return " ".join(str(value).split())[:RERANK_CONTEXT_MAX_CELL_CHARS]


def _question_tokens(question: str) -> set[str]:
    return {
        token
        for token in re.findall(r"\w+", question.casefold(), flags=re.UNICODE)
        if len(token) > 1 and token not in _QUESTION_STOP_WORDS
    }


def build_csv_rerank_context(question: str, table_id: str) -> str:
    """Build bounded, question-aware rerank context directly from a table CSV."""
    try:
        csv_file = resolve_csv_path(table_id, PROJECT_ROOT)
        with csv_file.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            raw_columns = next(reader)
            columns = [
                _compact_cell(column)
                for column in raw_columns[:RERANK_CONTEXT_MAX_COLUMNS]
            ]
            if not columns or not any(columns):
                raise ValueError("CSV không có header hữu ích")

            query_tokens = _question_tokens(question)
            scored_rows: list[tuple[int, int, list[str]]] = []
            scanned = 0
            for row_number, raw_row in enumerate(reader, start=2):
                if scanned >= RERANK_CONTEXT_SCAN_ROWS:
                    break
                scanned += 1
                cells = [
                    _compact_cell(value)
                    for value in raw_row[:RERANK_CONTEXT_MAX_COLUMNS]
                ]
                if not any(cells):
                    continue
                row_tokens = set(
                    re.findall(r"\w+", " ".join(cells).casefold(), flags=re.UNICODE)
                )
                score = len(query_tokens & row_tokens)
                scored_rows.append((score, row_number, cells))
    except (OSError, UnicodeError, csv.Error, StopIteration, ValueError) as exc:
        raise RerankerError(f"Không dựng được rerank context từ CSV: {table_id}") from exc

    ranked_rows = sorted(scored_rows, key=lambda item: (-item[0], item[1]))[
        :RERANK_CONTEXT_MAX_ROWS
    ]
    context: dict[str, Any] = {
        "columns": columns,
        "rows_scanned": scanned,
        "relevant_rows": [],
    }
    for _, row_number, cells in ranked_rows:
        compact_row = {
            "row": row_number,
            "cells": cells,
        }
        candidate_context = {**context, "relevant_rows": [*context["relevant_rows"], compact_row]}
        serialized = json.dumps(candidate_context, ensure_ascii=False)
        if len(serialized) > RERANK_CONTEXT_MAX_CHARS:
            break
        context = candidate_context
    return json.dumps(context, ensure_ascii=False)


def attach_rerank_context(
    question: str,
    candidates: Sequence[Mapping[str, Any]],
) -> list[Candidate]:
    """Validate Qdrant candidates and attach bounded context from local CSVs."""
    enriched: list[Candidate] = []
    seen: set[str] = set()
    for raw_candidate in candidates:
        try:
            candidate = dict(raw_candidate)
            table_id = str(candidate["table_id"])
            payload = validate_qdrant_payload(candidate["metadata"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RerankerError("Candidate retrieval không đúng contract") from exc
        if table_id != payload["table_id"] or table_id in seen:
            raise RerankerError(
                f"Candidate table_id không hợp lệ hoặc bị trùng: {table_id}"
            )
        seen.add(table_id)
        candidate["metadata"] = payload
        candidate["rerank_context"] = build_csv_rerank_context(question, table_id)
        enriched.append(candidate)
    return enriched


def _validate_rerank_response(
    response: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    top_k: int,
) -> list[Candidate]:
    if set(response) != {"ranked_candidates"}:
        raise ValueError("JSON phải có duy nhất key ranked_candidates")
    ranked = response["ranked_candidates"]
    if not isinstance(ranked, list) or len(ranked) != top_k:
        raise ValueError(f"ranked_candidates phải có đúng {top_k} phần tử")

    by_id = {str(item["table_id"]): item for item in candidates}
    selected: list[Candidate] = []
    seen: set[str] = set()
    for position, raw in enumerate(ranked, start=1):
        if not isinstance(raw, Mapping) or set(raw) != {"table_id", "score", "reason"}:
            raise ValueError("Mỗi kết quả phải có table_id, score và reason")
        table_id = raw["table_id"]
        score = raw["score"]
        reason = raw["reason"]
        if not isinstance(table_id, str) or table_id not in by_id:
            raise ValueError(f"LLM trả table_id không thuộc ứng viên: {table_id!r}")
        if table_id in seen:
            raise ValueError(f"LLM trả table_id trùng: {table_id}")
        if (
            isinstance(score, bool)
            or not isinstance(score, int)
            or not 0 <= score <= 100
        ):
            raise ValueError(f"score của {table_id} phải là số nguyên 0–100")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"reason của {table_id} phải là chuỗi không rỗng")
        seen.add(table_id)
        item = dict(by_id[table_id])
        item.update(
            {
                "rerank_score": score / 100.0,
                "rerank_reason": " ".join(reason.split())[:500],
                "rerank_rank": position,
            }
        )
        selected.append(item)
    return selected


def _rank_listwise(
    question: str,
    candidates: Sequence[Mapping[str, Any]],
    top_k: int,
) -> list[Candidate]:
    feedback = ""
    last_error = ""
    for _ in range(RERANK_RESPONSE_ATTEMPTS):
        try:
            response = generate_structured(
                build_rerank_prompt(question, candidates, top_k, feedback),
                system_prompt=RERANK_SYSTEM_PROMPT,
            )
            return _validate_rerank_response(response, candidates, top_k)
        except LLMResponseError as exc:
            last_error = str(exc)
        except ValueError as exc:
            last_error = str(exc)
        feedback = "Response trước không hợp lệ: " + last_error
    raise RerankerError("LLM reranker trả dữ liệu không hợp lệ: " + last_error)


def rerank(
    question: str,
    candidates: Sequence[Mapping[str, Any]],
    *,
    top_k: int = RERANK_TOP_K,
) -> list[Candidate]:
    """Hierarchically rerank dense candidates with the configured LLM."""
    if not question.strip():
        raise ValueError("question must not be empty")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not candidates:
        raise RerankerError("Không có candidate để rerank")

    enriched = attach_rerank_context(question, candidates)
    final_size = min(top_k, len(enriched))
    if len(enriched) <= top_k:
        result = _rank_listwise(question, enriched, final_size)
    else:
        batch_count = math.ceil(len(enriched) / RERANK_BATCH_SIZE)
        batches: list[list[Candidate]] = [[] for _ in range(batch_count)]
        for index, candidate in enumerate(enriched):
            batches[index % batch_count].append(candidate)

        finalists: list[Candidate] = []
        for batch in batches:
            shortlist_size = min(RERANK_BATCH_SHORTLIST, len(batch))
            finalists.extend(_rank_listwise(question, batch, shortlist_size))
        result = _rank_listwise(question, finalists, min(final_size, len(finalists)))

    logger.info("LLM reranked candidates to %d tables", len(result))
    logger.debug(
        "Rerank scores: %s",
        [(item["table_id"], item["rerank_score"]) for item in result],
    )
    return result
