"""Dense retrieval and hierarchical LLM reranking for financial tables."""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import httpx
from qdrant_client import models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from src.contracts import FILTER_FIELDS, validate_qdrant_payload
from src.embeddings import (
    DENSE_VECTOR_NAME,
    EMBEDDING_VECTOR_SIZE,
    DenseEmbeddingModel,
    EmbeddingError,
)
from src.llm import LLMResponseError, generate_structured
from src.prompt import RERANK_SYSTEM_PROMPT, build_rerank_prompt
from src.qdrant import QdrantConnectionError, get_collection_name, get_qdrant_client

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT / "intermediate" / "qdrant_manifest_granite_97m_r2_v1.jsonl"
)
RETRIEVAL_TOP_K = 50
RERANK_TOP_K = 10
RERANK_BATCH_SIZE = 10
RERANK_BATCH_SHORTLIST = 5
RERANK_RESPONSE_ATTEMPTS = 2

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


def get_manifest_path() -> Path:
    return Path(os.getenv("QDRANT_MANIFEST_PATH", str(DEFAULT_MANIFEST_PATH))).resolve()


@lru_cache(maxsize=4)
def _load_manifest_cached(path_text: str, mtime_ns: int) -> dict[str, dict[str, Any]]:
    del mtime_ns  # The value only invalidates the cache when the file changes.
    path = Path(path_text)
    entries: dict[str, dict[str, Any]] = {}
    header_seen = False
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RerankerError(
                    f"Manifest JSONL lỗi tại dòng {line_number}: {path}"
                ) from exc
            if not isinstance(item, dict):
                raise RerankerError(
                    f"Manifest record không phải object tại dòng {line_number}"
                )
            if item.get("record_type") == "header":
                if header_seen or line_number != 1:
                    raise RerankerError("Manifest phải có đúng một header ở dòng đầu")
                header_seen = True
                if item.get("vector_name") != DENSE_VECTOR_NAME:
                    raise RerankerError("Manifest dùng sai named vector")
                if item.get("vector_size") != EMBEDDING_VECTOR_SIZE:
                    raise RerankerError("Manifest không dùng vector Granite 384 chiều")
                if item.get("payload_schema_version") != 2:
                    raise RerankerError("Manifest không dùng payload schema version 2")
                continue
            if item.get("record_type") != "point":
                raise RerankerError(
                    f"Manifest record_type không hợp lệ tại dòng {line_number}"
                )
            try:
                payload = validate_qdrant_payload(item["payload"])
                table_id = payload["table_id"]
                index_text = item["index_text"]
            except (KeyError, TypeError, ValueError) as exc:
                raise RerankerError(
                    f"Manifest point không hợp lệ tại dòng {line_number}"
                ) from exc
            if not isinstance(index_text, str) or not index_text.strip():
                raise RerankerError(f"Manifest thiếu index_text tại dòng {line_number}")
            if table_id in entries:
                raise RerankerError(f"Manifest trùng table_id: {table_id}")
            entries[table_id] = {"payload": payload, "index_text": index_text.strip()}
    if not header_seen:
        raise RerankerError(f"Manifest thiếu header: {path}")
    return entries


def load_manifest_index(path: Path | None = None) -> dict[str, dict[str, Any]]:
    manifest_path = (path or get_manifest_path()).resolve()
    try:
        stat = manifest_path.stat()
    except OSError as exc:
        raise RerankerError(
            f"Không đọc được manifest reranker: {manifest_path}"
        ) from exc
    return _load_manifest_cached(str(manifest_path), stat.st_mtime_ns)


def enrich_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    manifest_path: Path | None = None,
) -> list[Candidate]:
    """Attach the exact embedded index_text and reject stale manifest data."""
    manifest = load_manifest_index(manifest_path)
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
        manifest_item = manifest.get(table_id)
        if manifest_item is None:
            raise RerankerError(f"Manifest không có table_id từ Qdrant: {table_id}")
        if manifest_item["payload"] != payload:
            raise RerankerError(
                f"Payload manifest đã lệch Qdrant cho table_id: {table_id}"
            )
        candidate["metadata"] = payload
        candidate["index_text"] = manifest_item["index_text"]
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
    manifest_path: Path | None = None,
) -> list[Candidate]:
    """Hierarchically rerank dense candidates with the configured LLM."""
    if not question.strip():
        raise ValueError("question must not be empty")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not candidates:
        raise RerankerError("Không có candidate để rerank")

    enriched = enrich_candidates(candidates, manifest_path=manifest_path)
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
