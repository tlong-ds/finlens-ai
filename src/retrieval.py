"""Dense retrieval and minimal reranking for financial tables."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Mapping, Sequence

import httpx
from qdrant_client import models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from src.embeddings import DENSE_VECTOR_NAME, DenseEmbeddingModel, EmbeddingError
from src.qdrant import QdrantConnectionError, get_collection_name, get_qdrant_client

logger = logging.getLogger(__name__)

RETRIEVAL_TOP_K = 50
RERANK_TOP_K = 10
FILTER_FIELDS = ("ticker", "company_name", "year", "report_type", "table_type")

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
    """Translate validated metadata constraints into a Qdrant filter."""
    conditions: list[models.FieldCondition] = []
    for field in FILTER_FIELDS:
        values = list((filters or {}).get(field, []))
        if values:
            conditions.append(
                models.FieldCondition(
                    key=field,
                    match=models.MatchAny(any=values),
                )
            )
    return models.Filter(must=conditions) if conditions else None


@lru_cache(maxsize=1)
def _get_embedding_model() -> DenseEmbeddingModel:
    return DenseEmbeddingModel.from_env()


def embed_query(question: str) -> list[float]:
    """Embed a non-empty question with the shared Granite query contract."""
    if not question.strip():
        raise RetrievalError("Question must not be empty")
    try:
        return _get_embedding_model().encode_queries([question.strip()])[0]
    except (EmbeddingError, TypeError, ValueError) as exc:
        raise RetrievalError("Query embedding failed") from exc


def _point_to_candidate(point: Any) -> Candidate:
    payload = dict(point.payload or {})
    point_id = str(point.id)
    return {
        "table_id": str(payload.get("table_id", point_id)),
        "text": str(payload.get("text") or payload.get("retrieval_context") or ""),
        "metadata": payload,
        "retrieval_score": float(point.score),
    }


def retrieve(
    question: str,
    filters: Mapping[str, Sequence[str | int]] | None = None,
    *,
    top_n: int = RETRIEVAL_TOP_K,
) -> list[Candidate]:
    """Embed a question and return Top-N Qdrant table candidates."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    try:
        response = get_qdrant_client().query_points(
            collection_name=get_collection_name(),
            query=embed_query(question),
            using=DENSE_VECTOR_NAME,
            query_filter=build_qdrant_filter(filters),
            limit=top_n,
            with_payload=True,
        )
        candidates = [_point_to_candidate(point) for point in response.points]
    except (RetrievalError, QdrantConnectionError):
        raise
    except (httpx.TransportError, ResponseHandlingException) as exc:
        raise TransientRetrievalError("Temporary Qdrant retrieval failure") from exc
    except UnexpectedResponse as exc:
        if exc.status_code in {408, 409, 425, 429} or exc.status_code >= 500:
            raise TransientRetrievalError(
                "Temporary Qdrant retrieval failure"
            ) from exc
        raise RetrievalError("Qdrant table retrieval failed") from exc
    except Exception as exc:
        raise RetrievalError("Qdrant table retrieval failed") from exc

    logger.info("Retrieved %d table candidates", len(candidates))
    logger.debug(
        "Retrieval scores: %s",
        [(item["table_id"], item["retrieval_score"]) for item in candidates],
    )
    return candidates


def rerank(
    question: str,
    candidates: Sequence[Mapping[str, Any]],
    *,
    top_k: int = RERANK_TOP_K,
) -> list[Candidate]:
    """Return Top-K tables using a retrieval-score fallback.

    Replace this scoring block once a reranker provider/model is selected. The
    function contract and graph do not need to change.
    """
    del question  # Reserved for the future (question, table) reranker.
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    try:
        scored = []
        for candidate in candidates:
            item = dict(candidate)
            item["rerank_score"] = float(item.get("retrieval_score", 0.0))
            scored.append(item)
        scored.sort(key=lambda item: item["rerank_score"], reverse=True)
        result = scored[:top_k]
    except (TypeError, ValueError) as exc:
        raise RerankerError("Candidate data is invalid for reranking") from exc

    logger.info("Reranked candidates to %d tables", len(result))
    logger.debug(
        "Rerank scores: %s",
        [(item.get("table_id"), item["rerank_score"]) for item in result],
    )
    return result
