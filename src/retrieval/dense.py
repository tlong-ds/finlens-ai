"""Hybrid retrieval and coverage-aware LLM reranking for financial tables."""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Mapping, Sequence
from typing import Any, cast

import httpx
from qdrant_client import models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from src.config import Settings
from src.contracts import (
    FILTER_FIELDS,
    validate_qdrant_payload,
)
from src.providers.embeddings import (
    DENSE_VECTOR_NAME,
    DenseEmbeddingModel,
    EmbeddingError,
)
from src.providers.qdrant import (
    QdrantConnectionError,
    get_collection_name,
    get_qdrant_client,
)
from src.retrieval.bm25 import BM25IndexError, search_bm25

logger = logging.getLogger(__name__)

RETRIEVAL_TOP_K = 80
RETRIEVAL_MODE_DEFAULT = "hybrid"
RRF_K = 60
FPT_RERANK_TOP_N = 20
FPT_RERANK_DOCUMENT_MAX_CHARS = 6_000
RERANK_SCOUT_COUNT = 2
RERANK_SCOUT_OUTPUT_MAX = 8
RERANK_OUTPUT_MAX = 18
RERANK_FINALIST_LEXICAL_RESCUE_MAX = 8
RERANK_FINALIST_LEXICAL_RESCUE_PER_BUCKET = 2
RERANK_CONCEPT_ROLES = frozenset(
    {
        "direct",
        "numerator",
        "denominator",
        "beginning_balance",
        "ending_balance",
        "comparison_operand",
    }
)
RERANK_CONTEXT_SMALL_TABLE_ROWS = 8
RERANK_CONTEXT_SEED_ROWS = 3
RERANK_CONTEXT_DETAIL_MAX_ROWS = 9
RERANK_CONTEXT_MAX_CELL_CHARS = 160
RERANK_CONTEXT_MAX_LABEL_CHARS = 160
RERANK_CONTEXT_MAX_TITLE_CHARS = 240
RERANK_CONTEXT_TARGET_CHARS = 4_000

_QUESTION_STOP_WORDS = frozenset(
    {
        "bao",
        "bằng",
        "các",
        "có",
        "công",
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
        "ty",
        "tỷ",
        "vào",
        "và",
    }
)

Candidate = dict[str, Any]

_embedding_models: dict[Settings, DenseEmbeddingModel] = {}
_embedding_model_lock = threading.Lock()


class RetrievalError(RuntimeError):
    """Raised when table retrieval fails."""


class NoMatchingCandidatesError(RetrievalError):
    """Raised when a valid Qdrant query returns no matching candidates."""


class TransientRetrievalError(RetrievalError):
    """Raised for temporary Qdrant failures that are safe to retry unchanged."""


class RerankerError(RuntimeError):
    """Raised when table reranking fails."""


class SelectorResponseError(ValueError):
    """Raised when a final selector response violates the coverage contract."""

    def __init__(self, errors: Sequence[str], coverage_status: Mapping[str, Any]):
        self.errors = list(errors)
        self.coverage_status = dict(coverage_status)
        super().__init__("; ".join(self.errors))


class SelectorSelectionError(RerankerError):
    """Raised after the bounded correction fails, retaining diagnostics."""

    def __init__(self, message: str, diagnostics: Mapping[str, Any]):
        self.diagnostics = dict(diagnostics)
        super().__init__(message)


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


def _get_embedding_model(settings: Settings) -> DenseEmbeddingModel:
    """Return one fully loaded encoder, even under concurrent first access."""
    model = _embedding_models.get(settings)
    if model is None:
        with _embedding_model_lock:
            model = _embedding_models.get(settings)
            if model is None:
                model = DenseEmbeddingModel.from_settings(settings)
                model.load()
                _embedding_models[settings] = model
    return model


def embed_query(query_text: str, *, settings: Settings) -> list[float]:
    """Embed a non-empty semantic query with the shared Granite contract."""
    if not query_text.strip():
        raise RetrievalError("Semantic query must not be empty")
    try:
        return _get_embedding_model(settings).encode_queries([query_text.strip()])[0]
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
        "dense_score": retrieval_score,
        "dense_rank": dense_rank,
    }


def _retrieve_dense(
    query_text: str,
    filters: Mapping[str, Sequence[str | int]] | None = None,
    *,
    top_n: int,
    settings: Settings,
) -> list[Candidate]:
    """Return validated dense candidates without enforcing a non-empty result."""
    try:
        response = get_qdrant_client(settings).query_points(
            collection_name=get_collection_name(settings),
            query=embed_query(query_text, settings=settings),
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

    logger.info("Retrieved %d dense table candidates", len(candidates))
    logger.debug(
        "Retrieval scores: %s",
        [(item["table_id"], item["retrieval_score"]) for item in candidates],
    )
    return candidates


def reciprocal_rank_fusion(
    dense_candidates: Sequence[Mapping[str, Any]],
    bm25_candidates: Sequence[Mapping[str, Any]],
    *,
    top_n: int,
    rrf_k: int = RRF_K,
) -> list[Candidate]:
    """Fuse dense and lexical rankings without calibrating their raw scores."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    if rrf_k < 1:
        raise ValueError("rrf_k must be at least 1")

    by_table_id: dict[str, Candidate] = {}
    for source, candidates in (
        ("dense", dense_candidates),
        ("bm25", bm25_candidates),
    ):
        rank_field = f"{source}_rank"
        score_field = f"{source}_score"
        for fallback_rank, raw_candidate in enumerate(candidates, start=1):
            candidate = dict(raw_candidate)
            table_id = str(candidate.get("table_id") or "")
            metadata = candidate.get("metadata")
            if not table_id or not isinstance(metadata, Mapping):
                raise RetrievalError(f"Invalid {source} retrieval candidate")
            rank = candidate.get(rank_field, fallback_rank)
            if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
                raise RetrievalError(f"Invalid {source} rank for {table_id}")
            existing = by_table_id.get(table_id)
            if existing is None:
                existing = {
                    "table_id": table_id,
                    "metadata": dict(metadata),
                    "rrf_score": 0.0,
                }
                by_table_id[table_id] = existing
            elif dict(existing["metadata"]) != dict(metadata):
                raise RetrievalError(
                    f"Dense/BM25 metadata mismatch for table {table_id}"
                )
            existing[rank_field] = rank
            if score_field in candidate:
                existing[score_field] = candidate[score_field]
            elif source == "dense" and "retrieval_score" in candidate:
                existing[score_field] = candidate["retrieval_score"]
            existing["rrf_score"] = float(existing["rrf_score"]) + 1.0 / (rrf_k + rank)

    ordered = sorted(
        by_table_id.values(),
        key=lambda candidate: (
            -float(candidate["rrf_score"]),
            min(
                int(candidate.get("dense_rank", 10**9)),
                int(candidate.get("bm25_rank", 10**9)),
            ),
            str(candidate["table_id"]),
        ),
    )[:top_n]
    for rank, candidate in enumerate(ordered, start=1):
        candidate["retrieval_rank"] = rank
        candidate["retrieval_score"] = candidate["rrf_score"]
        candidate["retrieval_mode"] = "hybrid"
    return ordered


def retrieve(
    query_text: str,
    filters: Mapping[str, Sequence[str | int]] | None = None,
    *,
    top_n: int = RETRIEVAL_TOP_K,
    mode: str | None = None,
    settings: Settings,
) -> list[Candidate]:
    """Return dense or BM25+dense RRF candidates under identical filters."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    retrieval_mode = (mode or settings.retrieval_mode).strip().lower()
    if retrieval_mode not in {"dense", "hybrid"}:
        raise ValueError("retrieval mode must be dense or hybrid")

    dense_candidates = _retrieve_dense(
        query_text, filters, top_n=top_n, settings=settings
    )
    if retrieval_mode == "dense":
        if not dense_candidates:
            raise NoMatchingCandidatesError(
                "Không tìm thấy bảng nào khớp metadata filter"
            )
        result: list[Candidate] = []
        for rank, raw_candidate in enumerate(dense_candidates, start=1):
            candidate = dict(raw_candidate)
            candidate["retrieval_rank"] = rank
            candidate["retrieval_mode"] = "dense"
            result.append(candidate)
        return result

    try:
        bm25_candidates = search_bm25(
            query_text,
            filters,
            top_n=top_n,
            settings=settings,
        )
    except BM25IndexError as exc:
        raise RetrievalError("BM25 table retrieval failed") from exc
    candidates = reciprocal_rank_fusion(
        dense_candidates,
        bm25_candidates,
        top_n=top_n,
    )
    if not candidates:
        raise NoMatchingCandidatesError("Không tìm thấy bảng nào khớp metadata filter")
    logger.info(
        "Hybrid retrieval fused dense=%d bm25=%d output=%d",
        len(dense_candidates),
        len(bm25_candidates),
        len(candidates),
    )
    return candidates
