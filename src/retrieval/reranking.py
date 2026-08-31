"""Strict external reranking over bounded table context."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any

from src.config import Settings
from src.providers.fpt import rerank_documents
from src.retrieval.context import (
    _attach_context_to_validated,
    _fpt_candidate_document,
    _validate_candidates,
)
from src.retrieval.dense import (
    FPT_RERANK_TOP_N,
    RETRIEVAL_TOP_K,
    Candidate,
    RerankerError,
)

logger = logging.getLogger(__name__)


def _select_balanced_candidates(
    scored_candidates: list[tuple[Candidate, float]],
    maximum: int,
) -> list[Candidate]:
    """Select up to `maximum` candidates ensuring multi-bucket and statement coverage."""
    if len(scored_candidates) <= maximum:
        result: list[Candidate] = []
        for rank, (cand, score) in enumerate(scored_candidates, start=1):
            c = dict(cand)
            c.update(
                {
                    "rerank_rank": rank,
                    "rerank_score": score,
                    "rerank_source": "fpt_bge_m3",
                }
            )
            result.append(c)
        return result

    # Group by bucket: (ticker, year, report_type)
    bucket_map: dict[tuple[Any, ...], list[tuple[Candidate, float]]] = {}
    for cand, score in scored_candidates:
        meta = cand.get("metadata", {})
        bucket = (meta.get("ticker"), meta.get("year"), meta.get("report_type"))
        bucket_map.setdefault(bucket, []).append((cand, score))

    selected_ids: set[str] = set()
    selected_scored: list[tuple[Candidate, float]] = []

    # 1. If multiple buckets exist, ensure top candidates and core statement types per bucket are included
    if len(bucket_map) > 1:
        for _bucket, items in bucket_map.items():
            top_cand, top_score = items[0]
            cand_id = str(top_cand.get("metadata", {}).get("table_id") or "")
            if cand_id and cand_id not in selected_ids:
                selected_ids.add(cand_id)
                selected_scored.append((top_cand, top_score))

            seen_types: set[str] = set()
            for cand, score in items:
                raw_ttype = str(cand.get("metadata", {}).get("table_type") or "")
                idx_text = str(
                    cand.get("metadata", {}).get("index_text") or ""
                ).casefold()
                if raw_ttype == "balance_sheet":
                    if "tài sản" in idx_text:
                        ttype = "balance_sheet_assets"
                    elif "nợ" in idx_text or "nguồn vốn" in idx_text:
                        ttype = "balance_sheet_liabilities"
                    else:
                        ttype = "balance_sheet"
                else:
                    ttype = raw_ttype

                if (
                    ttype
                    in {
                        "balance_sheet_assets",
                        "balance_sheet_liabilities",
                        "balance_sheet",
                        "income_statement",
                        "cash_flow",
                    }
                    and ttype not in seen_types
                ):
                    seen_types.add(ttype)
                    cand_id = str(cand.get("metadata", {}).get("table_id") or "")
                    if (
                        cand_id
                        and cand_id not in selected_ids
                        and len(selected_scored) < maximum
                    ):
                        selected_ids.add(cand_id)
                        selected_scored.append((cand, score))

    # 2. Fill remaining slots from global BGE ranked order
    for cand, score in scored_candidates:
        if len(selected_scored) >= maximum:
            break
        cand_id = str(cand.get("metadata", {}).get("table_id") or "")
        if cand_id and cand_id not in selected_ids:
            selected_ids.add(cand_id)
            selected_scored.append((cand, score))

    # Sort final selected candidates by BGE score descending
    selected_scored.sort(key=lambda item: -item[1])

    # Assign final rank 1..N
    result: list[Candidate] = []
    for rank, (cand, score) in enumerate(selected_scored[:maximum], start=1):
        c = dict(cand)
        c.update(
            {
                "rerank_rank": rank,
                "rerank_score": score,
                "rerank_source": "fpt_bge_m3",
            }
        )
        result.append(c)
    return result


def rerank_with_fpt(
    question: str,
    candidates: Sequence[Mapping[str, Any]],
    *,
    settings: Settings,
) -> list[Candidate]:
    """Rerank up to 80 retrieval candidates with FPT BGE and keep top 20 with balanced bucket coverage."""
    if not question.strip():
        raise ValueError("question must not be empty")
    if not candidates:
        raise RerankerError("Không có candidate để FPT rerank")
    if len(candidates) > RETRIEVAL_TOP_K:
        raise RerankerError(
            f"FPT reranker received {len(candidates)} candidates; maximum is "
            f"{RETRIEVAL_TOP_K}"
        )

    enriched = _attach_context_to_validated(
        question,
        _validate_candidates(candidates),
    )
    ranked_pairs = rerank_documents(
        question,
        [_fpt_candidate_document(question, candidate) for candidate in enriched],
        top_n=len(enriched),
        settings=settings,
    )
    scored_candidates = [(enriched[index], score) for index, score in ranked_pairs]
    result = _select_balanced_candidates(scored_candidates, FPT_RERANK_TOP_N)
    logger.info(
        "FPT rerank completed: input=%d output=%d",
        len(candidates),
        len(result),
    )
    return result


def _salvage_rerank_response(
    response: Mapping[str, Any],
    by_key: Mapping[str, Mapping[str, Any]],
    maximum: int,
) -> list[str]:
    if maximum < 1:
        raise ValueError("Reranker maximum must be positive")
    ranked = response.get("ranked_candidate_keys")
    if not isinstance(ranked, list):
        raise ValueError("ranked_candidate_keys phải là một mảng")

    selected_keys: list[str] = []
    seen: set[str] = set()
    dropped = 0
    for raw_key in ranked:
        if not isinstance(raw_key, str) or raw_key not in by_key or raw_key in seen:
            dropped += 1
            continue
        seen.add(raw_key)
        selected_keys.append(raw_key)
        if len(selected_keys) == maximum:
            break

    if not selected_keys:
        raise ValueError("LLM không trả candidate_key hợp lệ")

    if dropped or len(ranked) > maximum:
        logger.warning(
            "Salvaged reranker response: dropped=%d kept=%d returned=%d maximum=%d",
            dropped,
            len(selected_keys),
            len(ranked),
            maximum,
        )
    return selected_keys


def _fallback_sort_key(candidate: Mapping[str, Any]) -> tuple[float, float, str]:
    """Order selector candidates by FPT rank, then the original retrieval rank."""
    retrieval_rank = candidate.get(
        "rerank_rank",
        candidate.get("retrieval_rank", candidate.get("dense_rank")),
    )
    rank_value = (
        float(retrieval_rank)
        if isinstance(retrieval_rank, (int, float))
        and not isinstance(retrieval_rank, bool)
        else math.inf
    )
    retrieval_score = candidate.get("rerank_score", candidate.get("retrieval_score"))
    score_value = (
        float(retrieval_score)
        if isinstance(retrieval_score, (int, float))
        and not isinstance(retrieval_score, bool)
        and math.isfinite(float(retrieval_score))
        else -math.inf
    )
    return rank_value, -score_value, str(candidate.get("table_id") or "")
