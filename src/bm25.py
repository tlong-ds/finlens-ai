"""Runtime BM25 ranking over ``index_text`` payloads stored in Qdrant."""

from __future__ import annotations

import logging
import math
import os
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, cast

import httpx
from qdrant_client import models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from src.contracts import FILTER_FIELDS, validate_qdrant_payload
from src.qdrant import get_collection_name, get_qdrant_client


logger = logging.getLogger(__name__)

BM25_K1_DEFAULT = 1.2
BM25_B_DEFAULT = 0.75
QDRANT_SCROLL_BATCH_DEFAULT = 512
QDRANT_BM25_MAX_DOCUMENTS_DEFAULT = 20_000

_STOP_WORDS = frozenset(
    {
        "bao",
        "bang",
        "cac",
        "cho",
        "co",
        "cong",
        "cua",
        "cuoi",
        "den",
        "dong",
        "giua",
        "la",
        "nam",
        "ngay",
        "nhieu",
        "so",
        "theo",
        "trong",
        "trieu",
        "ty",
        "vao",
        "va",
    }
)


class BM25IndexError(RuntimeError):
    """Raised when Qdrant payloads cannot be collected or ranked safely."""


class TransientBM25IndexError(BM25IndexError):
    """Raised when Qdrant payload scrolling fails transiently."""


def _positive_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise BM25IndexError(f"{name} must be numeric") from exc
    if not math.isfinite(value) or value <= 0:
        raise BM25IndexError(f"{name} must be positive and finite")
    return value


def _bounded_float_env(name: str, default: float, low: float, high: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise BM25IndexError(f"{name} must be numeric") from exc
    if not math.isfinite(value) or not low <= value <= high:
        raise BM25IndexError(f"{name} must be between {low} and {high}")
    return value


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise BM25IndexError(f"{name} must be an integer") from exc
    if value < 1:
        raise BM25IndexError(f"{name} must be positive")
    return value


def _fold_token(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value.casefold())
    return "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn"
    ).replace("đ", "d")


def _tokens(text: str) -> list[str]:
    tokens = [
        _fold_token(token)
        for token in re.findall(r"\w+", text, flags=re.UNICODE)
        if len(token) > 1
    ]
    filtered = [token for token in tokens if token not in _STOP_WORDS]
    return filtered or tokens


def _query_tokens(query_text: str) -> list[str]:
    tokens = list(dict.fromkeys(_tokens(query_text)))
    if not tokens:
        raise BM25IndexError("BM25 query must contain at least one token")
    return tokens


def _build_qdrant_filter(
    filters: Mapping[str, Sequence[str | int]] | None,
) -> models.Filter | None:
    raw_filters = filters or {}
    unknown = set(raw_filters) - set(FILTER_FIELDS)
    if unknown:
        raise ValueError(
            "Unsupported BM25 filters: " + ", ".join(sorted(unknown))
        )

    conditions: list[Any] = []
    for field in FILTER_FIELDS:
        raw_values = raw_filters.get(field, [])
        if isinstance(raw_values, (str, bytes)):
            raise TypeError(f"Filter {field} must be a sequence, not a string")
        values = list(raw_values)
        if not values:
            continue
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
    return models.Filter(must=conditions) if conditions else None


def _scroll_payloads(
    filters: Mapping[str, Sequence[str | int]] | None,
    *,
    client: Any,
    collection_name: str,
) -> list[dict[str, Any]]:
    batch_size = _positive_int_env(
        "QDRANT_BM25_SCROLL_BATCH", QDRANT_SCROLL_BATCH_DEFAULT
    )
    maximum = _positive_int_env(
        "QDRANT_BM25_MAX_DOCUMENTS", QDRANT_BM25_MAX_DOCUMENTS_DEFAULT
    )
    query_filter = _build_qdrant_filter(filters)
    payloads: list[dict[str, Any]] = []
    offset: Any = None

    try:
        while True:
            points, offset = client.scroll(
                collection_name=collection_name,
                scroll_filter=query_filter,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                try:
                    payload = validate_qdrant_payload(dict(point.payload or {}))
                except (TypeError, ValueError) as exc:
                    raise BM25IndexError(
                        f"Qdrant BM25 point {point.id} has invalid payload"
                    ) from exc
                payloads.append(payload)
                if len(payloads) > maximum:
                    raise BM25IndexError(
                        "Qdrant BM25 candidate scope exceeds "
                        f"QDRANT_BM25_MAX_DOCUMENTS={maximum}; add stricter filters"
                    )
            if offset is None:
                break
    except BM25IndexError:
        raise
    except (httpx.TransportError, ResponseHandlingException) as exc:
        raise TransientBM25IndexError(
            "Temporary Qdrant BM25 payload failure"
        ) from exc
    except UnexpectedResponse as exc:
        status_code = exc.status_code
        if status_code in {408, 409, 425, 429} or (
            status_code is not None and status_code >= 500
        ):
            raise TransientBM25IndexError(
                "Temporary Qdrant BM25 payload failure"
            ) from exc
        raise BM25IndexError("Cannot read BM25 payloads from Qdrant") from exc
    except Exception as exc:
        raise BM25IndexError("Cannot read BM25 payloads from Qdrant") from exc

    return payloads


def _score_payloads(
    query_text: str,
    payloads: Sequence[Mapping[str, Any]],
    *,
    top_n: int,
) -> list[dict[str, Any]]:
    query_tokens = _query_tokens(query_text)
    query_counts = Counter(query_tokens)
    document_terms: list[Counter[str]] = []
    document_lengths: list[int] = []
    document_frequency: Counter[str] = Counter()

    for payload in payloads:
        tokens = _tokens(str(payload["index_text"]))
        counts = Counter(tokens)
        document_terms.append(counts)
        document_lengths.append(len(tokens))
        document_frequency.update(token for token in query_counts if counts[token])

    document_count = len(payloads)
    if document_count == 0:
        return []
    average_length = sum(document_lengths) / document_count
    if average_length <= 0:
        return []

    k1 = _positive_float_env("BM25_K1", BM25_K1_DEFAULT)
    b = _bounded_float_env("BM25_B", BM25_B_DEFAULT, 0.0, 1.0)
    scored: list[tuple[float, str, dict[str, Any]]] = []

    for payload, counts, document_length in zip(
        payloads, document_terms, document_lengths, strict=True
    ):
        score = 0.0
        length_normalization = k1 * (
            1.0 - b + b * document_length / average_length
        )
        for token, query_frequency in query_counts.items():
            term_frequency = counts[token]
            if term_frequency == 0:
                continue
            frequency = document_frequency[token]
            inverse_document_frequency = math.log(
                1.0 + (document_count - frequency + 0.5) / (frequency + 0.5)
            )
            score += (
                inverse_document_frequency
                * term_frequency
                * (k1 + 1.0)
                / (term_frequency + length_normalization)
                * query_frequency
            )
        if score > 0 and math.isfinite(score):
            normalized_payload = dict(payload)
            scored.append(
                (score, str(normalized_payload["table_id"]), normalized_payload)
            )

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [
        {
            "table_id": table_id,
            "metadata": payload,
            "bm25_score": score,
            "bm25_rank": rank,
        }
        for rank, (score, table_id, payload) in enumerate(scored[:top_n], start=1)
    ]


def search_bm25(
    query_text: str,
    filters: Mapping[str, Sequence[str | int]] | None = None,
    *,
    top_n: int,
    client: Any | None = None,
    collection_name: str | None = None,
) -> list[dict[str, Any]]:
    """Rank filtered Qdrant ``index_text`` payloads with BM25 at query time."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    resolved_client = client or get_qdrant_client()
    resolved_collection = collection_name or get_collection_name()
    payloads = _scroll_payloads(
        filters,
        client=resolved_client,
        collection_name=resolved_collection,
    )
    candidates = _score_payloads(query_text, payloads, top_n=top_n)
    logger.info(
        "Retrieved %d BM25 candidates from %d Qdrant payloads",
        len(candidates),
        len(payloads),
    )
    return candidates
