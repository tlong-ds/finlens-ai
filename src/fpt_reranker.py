"""Strict FPT BGE reranker client used by the online graph."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import httpx


DEFAULT_ENDPOINT = "https://mkp-api.fptcloud.com/v1/rerank"
DEFAULT_MODEL = "bge-reranker-v2-m3"
DEFAULT_TIMEOUT_SECONDS = 90.0
DEFAULT_MAX_ATTEMPTS = 3
_TRANSIENT_STATUS_CODES = frozenset({408, 409, 425, 429})


class FptRerankerError(RuntimeError):
    """Raised when FPT returns a permanent or malformed reranker response."""


class TransientFptRerankerError(FptRerankerError):
    """Raised after retryable FPT failures exhaust the configured attempts."""


@dataclass(frozen=True)
class FptRerankerConfig:
    """Non-secret settings for one reproducible FPT reranker call."""

    endpoint: str = DEFAULT_ENDPOINT
    model: str = DEFAULT_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_attempts: int = DEFAULT_MAX_ATTEMPTS

    @classmethod
    def from_env(cls) -> "FptRerankerConfig":
        endpoint = os.getenv("FPT_RERANK_URL", DEFAULT_ENDPOINT).strip()
        model = os.getenv("FPT_RERANK_MODEL", DEFAULT_MODEL).strip()
        try:
            timeout_seconds = float(
                os.getenv("FPT_RERANK_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS))
            )
        except ValueError as exc:
            raise FptRerankerError("FPT_RERANK_TIMEOUT must be numeric") from exc
        config = cls(
            endpoint=endpoint,
            model=model,
            timeout_seconds=timeout_seconds,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.endpoint or not self.model:
            raise FptRerankerError("FPT reranker endpoint and model are required")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise FptRerankerError("FPT reranker timeout must be positive and finite")
        if self.max_attempts < 1:
            raise FptRerankerError("FPT reranker max_attempts must be positive")

    def public_dict(self) -> dict[str, Any]:
        """Return settings safe to persist in experiment artifacts."""
        return asdict(self)


def effective_fpt_reranker_config() -> dict[str, Any]:
    """Expose the effective non-secret FPT configuration for run manifests."""
    return FptRerankerConfig.from_env().public_dict()


def _api_key() -> str:
    value = os.getenv("FPT_API_KEY", "").strip()
    if not value:
        raise FptRerankerError("FPT_API_KEY is not configured")
    return value


def _is_transient_status(status_code: int) -> bool:
    return status_code in _TRANSIENT_STATUS_CODES or status_code >= 500


def _validate_results(
    body: Any,
    *,
    document_count: int,
    expected_count: int,
) -> list[tuple[int, float]]:
    if not isinstance(body, Mapping):
        raise FptRerankerError("FPT reranker response must be a JSON object")
    raw_results = body.get("results")
    if not isinstance(raw_results, list):
        raise FptRerankerError("FPT reranker response has no results list")
    if len(raw_results) != expected_count:
        raise FptRerankerError(
            "FPT reranker returned a partial result: "
            f"expected={expected_count} actual={len(raw_results)}"
        )

    results: list[tuple[int, float]] = []
    seen: set[int] = set()
    for raw_item in raw_results:
        if not isinstance(raw_item, Mapping):
            raise FptRerankerError("FPT reranker result must be an object")
        index = raw_item.get("index")
        score = raw_item.get("relevance_score")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= document_count
            or index in seen
        ):
            raise FptRerankerError("FPT reranker returned an invalid document index")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise FptRerankerError("FPT reranker returned an invalid relevance score")
        seen.add(index)
        results.append((index, float(score)))
    return results


def rerank_documents(
    query: str,
    documents: Sequence[str],
    *,
    top_n: int,
    config: FptRerankerConfig | None = None,
) -> list[tuple[int, float]]:
    """Return validated ``(document_index, score)`` pairs in FPT rank order."""
    if not query.strip():
        raise ValueError("FPT reranker query must not be empty")
    if not documents or any(not isinstance(item, str) or not item for item in documents):
        raise ValueError("FPT reranker documents must be non-empty strings")
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise ValueError("FPT reranker top_n must be a positive integer")

    effective_config = config or FptRerankerConfig.from_env()
    effective_config.validate()
    expected_count = min(top_n, len(documents))
    payload = {
        "model": effective_config.model,
        "query": query,
        "documents": list(documents),
        "top_n": expected_count,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_api_key()}",
    }

    last_transient: BaseException | None = None
    for attempt in range(1, effective_config.max_attempts + 1):
        try:
            response = httpx.post(
                effective_config.endpoint,
                headers=headers,
                json=payload,
                timeout=effective_config.timeout_seconds,
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_transient = exc
        else:
            if _is_transient_status(response.status_code):
                last_transient = FptRerankerError(
                    f"FPT reranker transient HTTP {response.status_code}"
                )
            elif response.is_error:
                raise FptRerankerError(
                    f"FPT reranker permanent HTTP {response.status_code}"
                )
            else:
                try:
                    body = response.json()
                except ValueError as exc:
                    raise FptRerankerError(
                        "FPT reranker response is not valid JSON"
                    ) from exc
                return _validate_results(
                    body,
                    document_count=len(documents),
                    expected_count=expected_count,
                )

        if attempt < effective_config.max_attempts:
            time.sleep(float(attempt))

    raise TransientFptRerankerError(
        "FPT reranker failed after "
        f"{effective_config.max_attempts} transient attempts: {last_transient}"
    ) from last_transient
