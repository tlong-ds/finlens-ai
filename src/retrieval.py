"""Hybrid retrieval and coverage-aware LLM reranking for financial tables."""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
import threading
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import httpx
from qdrant_client import models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from src.bm25 import BM25IndexError, search_bm25
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
from src.prompt import (
    RERANK_SCOUT_SYSTEM_PROMPT,
    RERANK_SYSTEM_PROMPT,
    build_rerank_prompt,
    build_rerank_scout_prompt,
)
from src.qdrant import QdrantConnectionError, get_collection_name, get_qdrant_client

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RETRIEVAL_TOP_K = 50
RETRIEVAL_MODE_DEFAULT = "hybrid"
RRF_K = 60
RERANK_SHORTLIST_MAX = 30
RERANK_SHORTLIST_RESCUE_MAX = 4
RERANK_SCOUT_COUNT = 2
RERANK_SCOUT_OUTPUT_MAX = 8
RERANK_OUTPUT_MIN = 8
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

_embedding_model: DenseEmbeddingModel | None = None
_embedding_model_lock = threading.Lock()


class RetrievalError(RuntimeError):
    """Raised when table retrieval fails."""


class NoMatchingCandidatesError(RetrievalError):
    """Raised when a valid Qdrant query returns no matching candidates."""


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


def _get_embedding_model() -> DenseEmbeddingModel:
    """Return one fully loaded encoder, even under concurrent first access."""
    global _embedding_model
    if _embedding_model is None:
        with _embedding_model_lock:
            if _embedding_model is None:
                model = DenseEmbeddingModel.from_env()
                model.load()
                _embedding_model = model
    return _embedding_model


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
        "dense_score": retrieval_score,
        "dense_rank": dense_rank,
    }


def _retrieve_dense(
    query_text: str,
    filters: Mapping[str, Sequence[str | int]] | None = None,
    *,
    top_n: int,
) -> list[Candidate]:
    """Return validated dense candidates without enforcing a non-empty result."""
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
            if (
                isinstance(rank, bool)
                or not isinstance(rank, int)
                or rank < 1
            ):
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
            existing["rrf_score"] = float(existing["rrf_score"]) + 1.0 / (
                rrf_k + rank
            )

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
) -> list[Candidate]:
    """Return dense or BM25+dense RRF candidates under identical filters."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    retrieval_mode = (
        mode or os.getenv("RETRIEVAL_MODE", RETRIEVAL_MODE_DEFAULT)
    ).strip().lower()
    if retrieval_mode not in {"dense", "hybrid"}:
        raise ValueError("retrieval mode must be dense or hybrid")

    dense_candidates = _retrieve_dense(query_text, filters, top_n=top_n)
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
        )
    except BM25IndexError as exc:
        raise RetrievalError("BM25 table retrieval failed") from exc
    candidates = reciprocal_rank_fusion(
        dense_candidates,
        bm25_candidates,
        top_n=top_n,
    )
    if not candidates:
        raise NoMatchingCandidatesError(
            "Không tìm thấy bảng nào khớp metadata filter"
        )
    logger.info(
        "Hybrid retrieval fused dense=%d bm25=%d output=%d",
        len(dense_candidates),
        len(bm25_candidates),
        len(candidates),
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


def _row_tokens(value: str) -> set[str]:
    return set(re.findall(r"\w+", value.casefold(), flags=re.UNICODE))


def _select_detailed_row_indexes(
    rows: Sequence[Mapping[str, Any]],
) -> list[int]:
    """Select lexical seed rows and their immediate neighbours."""
    if len(rows) <= RERANK_CONTEXT_SMALL_TABLE_ROWS:
        return list(range(len(rows)))

    ranked = sorted(
        range(len(rows)),
        key=lambda index: (-int(rows[index]["match_score"]), index),
    )
    seeds = [
        index
        for index in ranked
        if int(rows[index]["match_score"]) > 0
    ][:RERANK_CONTEXT_SEED_ROWS]
    if len(seeds) < RERANK_CONTEXT_SEED_ROWS:
        seeds.extend(
            index
            for index in range(len(rows))
            if index not in seeds
        )
        seeds = seeds[:RERANK_CONTEXT_SEED_ROWS]

    expanded = {
        neighbour
        for index in seeds
        for neighbour in (index - 1, index, index + 1)
        if 0 <= neighbour < len(rows)
    }
    return sorted(expanded)[:RERANK_CONTEXT_DETAIL_MAX_ROWS]


def build_csv_rerank_context(question: str, table_id: str) -> dict[str, Any]:
    """Build layered table context without hiding unmatched row labels."""
    try:
        csv_file = resolve_csv_path(table_id, PROJECT_ROOT)
        with csv_file.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            raw_columns = next(reader)
            columns = [_compact_cell(column) for column in raw_columns]
            if not columns or not any(columns):
                raise ValueError("CSV không có header hữu ích")

            query_tokens = _question_tokens(question)
            label_index = (
                columns.index("item_label_norm")
                if "item_label_norm" in columns
                else columns.index("row_label_raw")
                if "row_label_raw" in columns
                else 0
            )
            code_index = columns.index("item_code") if "item_code" in columns else None
            title_index = columns.index("note_title") if "note_title" in columns else None
            rows: list[dict[str, Any]] = []
            titles: list[str] = []
            seen_titles: set[str] = set()
            for row_number, raw_row in enumerate(reader, start=2):
                cells = [_compact_cell(value) for value in raw_row]
                if not any(cells):
                    continue
                label = cells[label_index] if label_index < len(cells) else ""
                code = (
                    cells[code_index]
                    if code_index is not None and code_index < len(cells)
                    else ""
                )
                title = (
                    " ".join(str(raw_row[title_index]).split())[
                        :RERANK_CONTEXT_MAX_TITLE_CHARS
                    ]
                    if title_index is not None and title_index < len(cells)
                    else ""
                )
                if title and title not in seen_titles and len(titles) < 3:
                    seen_titles.add(title)
                    titles.append(title)
                match_text = " ".join(part for part in (label, title) if part)
                rows.append(
                    {
                        "row": row_number,
                        "cells": cells,
                        "label": label[:RERANK_CONTEXT_MAX_LABEL_CHARS],
                        "code": code,
                        "match_score": len(query_tokens & _row_tokens(match_text)),
                    }
                )
    except (OSError, UnicodeError, csv.Error, StopIteration, ValueError) as exc:
        raise RerankerError(f"Không dựng được rerank context từ CSV: {table_id}") from exc

    context: dict[str, Any] = {
        "columns": columns,
        "row_count": len(rows),
        "table_titles": titles,
        "row_catalog": [
            {
                "row": row["row"],
                **({"code": row["code"]} if row["code"] else {}),
                "label": row["label"],
            }
            for row in rows
            if row["label"]
        ],
        "detailed_rows": [],
    }
    essential_size = len(json.dumps(context, ensure_ascii=False))
    if essential_size > RERANK_CONTEXT_TARGET_CHARS:
        logger.warning(
            "Rerank essential context exceeds target: table_id=%s chars=%d target=%d",
            table_id,
            essential_size,
            RERANK_CONTEXT_TARGET_CHARS,
        )

    for index in _select_detailed_row_indexes(rows):
        row = rows[index]
        compact_row = {
            "row": row["row"],
            "cells": row["cells"],
        }
        candidate_context = {
            **context,
            "detailed_rows": [*context["detailed_rows"], compact_row],
        }
        serialized = json.dumps(candidate_context, ensure_ascii=False)
        if len(serialized) > RERANK_CONTEXT_TARGET_CHARS:
            break
        context = candidate_context
    return context


def _validate_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> list[Candidate]:
    validated: list[Candidate] = []
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
        validated.append(candidate)
    return validated


def _attach_context_to_validated(
    question: str,
    candidates: Sequence[Mapping[str, Any]],
) -> list[Candidate]:
    enriched: list[Candidate] = []
    for raw_candidate in candidates:
        candidate = dict(raw_candidate)
        candidate["rerank_context"] = build_csv_rerank_context(
            question,
            str(candidate["table_id"]),
        )
        enriched.append(candidate)
    return enriched


def attach_rerank_context(
    question: str,
    candidates: Sequence[Mapping[str, Any]],
) -> list[Candidate]:
    """Validate Qdrant candidates and attach layered context from local CSVs."""
    return _attach_context_to_validated(question, _validate_candidates(candidates))


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
    """Order candidates by effective retrieval rank, then retrieval score."""
    retrieval_rank = candidate.get("retrieval_rank", candidate.get("dense_rank"))
    rank_value = (
        float(retrieval_rank)
        if isinstance(retrieval_rank, (int, float))
        and not isinstance(retrieval_rank, bool)
        else math.inf
    )
    retrieval_score = candidate.get("retrieval_score")
    score_value = (
        float(retrieval_score)
        if isinstance(retrieval_score, (int, float))
        and not isinstance(retrieval_score, bool)
        and math.isfinite(float(retrieval_score))
        else -math.inf
    )
    return rank_value, -score_value, str(candidate.get("table_id") or "")


def _normalized_words(value: str) -> tuple[str, list[str]]:
    words = re.findall(r"\w+", value.casefold(), flags=re.UNICODE)
    return " ".join(words), words


def _lexical_rescue_score(
    question: str,
    candidate: Mapping[str, Any],
) -> tuple[int, int, float, float] | None:
    """Score strong row/title matches without making a hard table decision."""
    question_normalized, _ = _normalized_words(question)
    query_tokens = _question_tokens(question)
    if not query_tokens:
        return None

    context = candidate.get("rerank_context")
    if not isinstance(context, Mapping):
        return None
    raw_catalog = context.get("row_catalog")
    raw_titles = context.get("table_titles")
    texts: list[str] = []
    if isinstance(raw_catalog, Sequence) and not isinstance(raw_catalog, (str, bytes)):
        for row in raw_catalog:
            if isinstance(row, Mapping) and isinstance(row.get("label"), str):
                texts.append(str(row["label"]))
    if isinstance(raw_titles, Sequence) and not isinstance(raw_titles, (str, bytes)):
        texts.extend(str(title) for title in raw_titles if isinstance(title, str))

    best: tuple[int, int, float, float] | None = None
    for text in texts:
        text_normalized, raw_words = _normalized_words(text)
        text_tokens = _question_tokens(text)
        if not text_tokens:
            continue
        overlap = len(query_tokens & text_tokens)
        single_acronym = bool(
            len(raw_words) == 1
            and re.fullmatch(r"[A-Z][A-Z0-9/.-]{2,9}", text.strip())
        )
        exact_phrase = int(
            bool(text_normalized)
            and text_normalized in question_normalized
            and (len(raw_words) >= 2 or single_acronym)
        )
        if not exact_phrase and overlap < 2:
            continue
        label_coverage = overlap / len(text_tokens)
        question_coverage = overlap / len(query_tokens)
        score = (exact_phrase, overlap, label_coverage, question_coverage)
        if best is None or score > best:
            best = score
    return best


def _build_match_summary(
    question: str,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Put high-signal row/title matches before the lossless table context."""
    question_normalized, _ = _normalized_words(question)
    query_tokens = _question_tokens(question)
    scored_rows: list[tuple[int, int, float, dict[str, Any]]] = []
    raw_catalog = context.get("row_catalog")
    if isinstance(raw_catalog, Sequence) and not isinstance(raw_catalog, (str, bytes)):
        for raw_row in raw_catalog:
            if not isinstance(raw_row, Mapping) or not isinstance(
                raw_row.get("label"), str
            ):
                continue
            label = str(raw_row["label"])
            label_normalized, label_words = _normalized_words(label)
            label_tokens = _question_tokens(label)
            overlap = len(query_tokens & label_tokens)
            single_acronym = bool(
                len(label_words) == 1
                and re.fullmatch(r"[A-Z][A-Z0-9/.-]{2,9}", label.strip())
            )
            exact = int(
                bool(label_normalized)
                and label_normalized in question_normalized
                and (len(label_words) >= 2 or single_acronym)
            )
            if not exact and overlap < 2:
                continue
            row = {
                "row": raw_row.get("row"),
                **({"code": raw_row["code"]} if raw_row.get("code") else {}),
                "label": label,
                "overlap_tokens": overlap,
            }
            coverage = overlap / len(label_tokens) if label_tokens else 0.0
            scored_rows.append((exact, overlap, coverage, row))

    scored_rows.sort(key=lambda item: (-item[0], -item[1], -item[2], str(item[3]["row"])))
    exact_rows = [item[3] for item in scored_rows if item[0]][:3]
    exact_row_numbers = {row["row"] for row in exact_rows}
    strong_rows = [
        item[3]
        for item in scored_rows
        if item[3]["row"] not in exact_row_numbers
    ][:5]

    exact_phrase_titles: list[str] = []
    matching_titles: list[str] = []
    raw_titles = context.get("table_titles")
    if isinstance(raw_titles, Sequence) and not isinstance(raw_titles, (str, bytes)):
        exact_phrase_titles = [
            str(title)
            for title in raw_titles
            if isinstance(title, str)
            and (title_normalized := _normalized_words(title)[0])
            and title_normalized in question_normalized
            and len(_normalized_words(title)[1]) >= 2
        ][:3]
        exact_title_set = set(exact_phrase_titles)
        matching_titles = sorted(
            (
                str(title)
                for title in raw_titles
                if isinstance(title, str)
                and str(title) not in exact_title_set
                and query_tokens & _question_tokens(title)
            ),
            key=lambda title: (
                -len(query_tokens & _question_tokens(title)),
                title,
            ),
        )[:3]
    return {
        "exact_phrase_rows": exact_rows,
        "exact_phrase_titles": exact_phrase_titles,
        "strong_overlap_rows": strong_rows,
        "table_titles": matching_titles,
    }


def _prioritized_rerank_context(
    question: str,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Reorder context for attention while preserving the complete row catalog."""
    return {
        "match_summary": _build_match_summary(question, context),
        "table_titles": context.get("table_titles", []),
        "columns": context.get("columns", []),
        "row_count": context.get("row_count", 0),
        "row_catalog": context.get("row_catalog", []),
        "detailed_rows": context.get("detailed_rows", []),
    }


def _diversified_shortlist(
    question: str,
    candidates: Sequence[Mapping[str, Any]],
    maximum: int = RERANK_SHORTLIST_MAX,
    rescue_maximum: int = RERANK_SHORTLIST_RESCUE_MAX,
) -> list[Candidate]:
    """Preserve buckets/diversity and rescue strong row matches within one cap."""
    if maximum < 1 or rescue_maximum < 0:
        raise ValueError("Invalid rerank shortlist size")
    buckets: dict[str, list[Mapping[str, Any]]] = {}
    for candidate in sorted(candidates, key=_fallback_sort_key):
        doc_id = str(candidate["metadata"]["doc_id"])
        buckets.setdefault(doc_id, []).append(candidate)

    ordered_doc_ids = sorted(
        buckets,
        key=lambda doc_id: _fallback_sort_key(buckets[doc_id][0]),
    )
    if len(ordered_doc_ids) > maximum:
        logger.warning(
            "Rerank candidate pool has more document buckets than shortlist: "
            "buckets=%d maximum=%d",
            len(ordered_doc_ids),
            maximum,
        )
        ordered_doc_ids = ordered_doc_ids[:maximum]

    selected: list[Candidate] = []
    selected_ids: set[str] = set()

    def add(candidate: Mapping[str, Any]) -> None:
        item = dict(candidate)
        selected.append(item)
        selected_ids.add(str(item["table_id"]))

    for doc_id in ordered_doc_ids:
        add(buckets[doc_id][0])

    eligible_doc_ids = set(ordered_doc_ids)
    rescue_candidates: list[
        tuple[tuple[int, int, float, float], Mapping[str, Any]]
    ] = []
    for candidate in candidates:
        if str(candidate["metadata"]["doc_id"]) not in eligible_doc_ids:
            continue
        score = _lexical_rescue_score(question, candidate)
        if score is not None:
            rescue_candidates.append((score, candidate))
    rescue_added = 0
    for _, candidate in sorted(
        rescue_candidates,
        key=lambda item: (
            -float(item[0][0]),
            -float(item[0][1]),
            -item[0][2],
            -item[0][3],
            *_fallback_sort_key(item[1]),
        ),
    ):
        if len(selected) == maximum or rescue_added == rescue_maximum:
            break
        if str(candidate["table_id"]) in selected_ids:
            continue
        add(candidate)
        rescue_added += 1

    while len(selected) < maximum:
        changed = False
        for doc_id in ordered_doc_ids:
            selected_types = {
                str(item["metadata"]["table_type"])
                for item in selected
                if item["metadata"]["doc_id"] == doc_id
            }
            candidate = next(
                (
                    item
                    for item in buckets[doc_id]
                    if str(item["table_id"]) not in selected_ids
                    and str(item["metadata"]["table_type"]) not in selected_types
                ),
                None,
            )
            if candidate is not None:
                add(candidate)
                changed = True
                if len(selected) == maximum:
                    break
        if not changed:
            break

    while len(selected) < maximum:
        changed = False
        for doc_id in ordered_doc_ids:
            candidate = next(
                (
                    item
                    for item in buckets[doc_id]
                    if str(item["table_id"]) not in selected_ids
                ),
                None,
            )
            if candidate is not None:
                add(candidate)
                changed = True
                if len(selected) == maximum:
                    break
        if not changed:
            break
    logger.debug(
        "Built rerank shortlist: input=%d buckets=%d rescue_added=%d output=%d",
        len(candidates),
        len(ordered_doc_ids),
        rescue_added,
        len(selected),
    )
    return selected


def _dynamic_output_cap(
    required_bucket_count: int,
    coverage_cell_count: int,
    finalist_count: int,
) -> int:
    """Allow one slot per declared coverage cell without exceeding the hard cap."""
    if (
        required_bucket_count < 1
        or coverage_cell_count < 0
        or finalist_count < required_bucket_count
    ):
        raise ValueError("Invalid rerank bucket or shortlist size")
    return min(
        finalist_count,
        RERANK_OUTPUT_MAX,
        max(RERANK_OUTPUT_MIN, required_bucket_count, coverage_cell_count),
    )


def _coverage_locked_buckets(
    question: str,
    available_buckets: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[str]]:
    """Lock structurally required comparison buckets without resolving tables."""
    normalized = " ".join(question.casefold().split())
    tickers = {str(bucket.get("ticker") or "") for bucket in available_buckets}
    years = {bucket.get("year") for bucket in available_buckets}
    reasons: list[str] = []
    multi_ticker_terms = (
        "trong nhóm",
        "xét nhóm",
        "nhóm doanh nghiệp",
        "gồm",
        "giữa",
        "so sánh",
        "so với",
        "trung bình",
        "bình quân",
        "hiệu số",
        "chênh lệch",
        "tổng chi phí",
        "tổng giá trị",
    )
    multi_year_terms = (
        "trong giai đoạn",
        "trong các năm",
        "năm có",
        "năm nào",
        "cao nhất",
        "thấp nhất",
        "lớn nhất",
        "nhỏ nhất",
        "trung vị",
        "cả ba năm",
        "cả hai năm",
    )
    if len(tickers) > 1 and any(term in normalized for term in multi_ticker_terms):
        reasons.append("multi_ticker_aggregation_or_comparison")
    if len(years) > 1 and any(term in normalized for term in multi_year_terms):
        reasons.append("multi_year_selection_or_filter")
    if not reasons:
        return [], []
    return [str(bucket["bucket_key"]) for bucket in available_buckets], reasons


def _exact_lexical_finalist_keys(
    question: str,
    by_key: Mapping[str, Mapping[str, Any]],
    bucket_by_candidate_key: Mapping[str, str],
) -> list[str]:
    """Bypass scout pruning for a bounded set of exact row/title matches."""
    scored: list[tuple[tuple[int, int, float, float], str]] = []
    for key, candidate in by_key.items():
        score = _lexical_rescue_score(question, candidate)
        if score is not None and score[0] == 1:
            scored.append((score, key))
    ordered = sorted(
        scored,
        key=lambda item: (
            -item[0][0],
            -item[0][1],
            -item[0][2],
            -item[0][3],
            *_fallback_sort_key(by_key[item[1]]),
        ),
    )
    selected: list[str] = []
    per_bucket: Counter[str] = Counter()
    for _, key in ordered:
        bucket_key = bucket_by_candidate_key[key]
        if per_bucket[bucket_key] >= RERANK_FINALIST_LEXICAL_RESCUE_PER_BUCKET:
            continue
        selected.append(key)
        per_bucket[bucket_key] += 1
        if len(selected) == RERANK_FINALIST_LEXICAL_RESCUE_MAX:
            break
    return selected


def _build_prompt_contract(
    question: str,
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Candidate],
    dict[str, str],
]:
    doc_ids: list[str] = []
    for candidate in candidates:
        doc_id = str(candidate["metadata"]["doc_id"])
        if doc_id not in doc_ids:
            doc_ids.append(doc_id)
    bucket_key_by_doc = {
        doc_id: f"b{index:02d}"
        for index, doc_id in enumerate(doc_ids, start=1)
    }

    available_buckets: list[dict[str, Any]] = []
    for doc_id in doc_ids:
        metadata = next(
            candidate["metadata"]
            for candidate in candidates
            if candidate["metadata"]["doc_id"] == doc_id
        )
        available_buckets.append(
            {
                "bucket_key": bucket_key_by_doc[doc_id],
                "ticker": metadata["ticker"],
                "company_name": metadata["company_name"],
                "year": metadata["year"],
                "report_type": metadata["report_type"],
            }
        )

    by_key: dict[str, Candidate] = {}
    prompt_candidates: list[dict[str, Any]] = []
    bucket_by_candidate_key: dict[str, str] = {}
    for index, raw_candidate in enumerate(candidates, start=1):
        key = f"c{index:02d}"
        candidate = dict(raw_candidate)
        doc_id = str(candidate["metadata"]["doc_id"])
        bucket_key = bucket_key_by_doc[doc_id]
        by_key[key] = candidate
        bucket_by_candidate_key[key] = bucket_key
        prompt_candidates.append(
            {
                "candidate_key": key,
                "bucket_key": bucket_key,
                "table_type": candidate["metadata"]["table_type"],
                "dense_rank": candidate.get(
                    "retrieval_rank", candidate.get("dense_rank")
                ),
                "context": _prioritized_rerank_context(
                    question,
                    candidate["rerank_context"],
                ),
            }
        )
    return available_buckets, prompt_candidates, by_key, bucket_by_candidate_key


def _balanced_scout_chunks(
    candidates: Sequence[Mapping[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Split candidates into two stable chunks while spreading every bucket."""
    chunks: list[list[dict[str, Any]]] = [[], []]
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for candidate in candidates:
        grouped.setdefault(str(candidate["bucket_key"]), []).append(candidate)

    for bucket_candidates in grouped.values():
        first_chunk = 0 if len(chunks[0]) <= len(chunks[1]) else 1
        for index, candidate in enumerate(bucket_candidates):
            chunks[(first_chunk + index) % RERANK_SCOUT_COUNT].append(dict(candidate))
    return chunks


def _ensure_nomination_bucket_coverage(
    nominated_keys: Sequence[str],
    by_key: Mapping[str, Mapping[str, Any]],
    bucket_by_candidate_key: Mapping[str, str],
    required_bucket_keys: Sequence[str],
) -> list[str]:
    """Ensure the final arbiter can inspect at least one candidate per bucket."""
    selected = list(dict.fromkeys(nominated_keys))
    selected_buckets = {bucket_by_candidate_key[key] for key in selected}
    for bucket_key in required_bucket_keys:
        if bucket_key in selected_buckets:
            continue
        anchor = next(
            key
            for key in by_key
            if bucket_by_candidate_key[key] == bucket_key
        )
        selected.append(anchor)
        selected_buckets.add(bucket_key)
    return selected


def _salvage_final_response(
    response: Mapping[str, Any],
    by_key: Mapping[str, Mapping[str, Any]],
    bucket_by_candidate_key: Mapping[str, str],
    available_bucket_keys: Sequence[str],
) -> dict[str, Any]:
    """Salvage the structured final decision without inventing required buckets."""
    raw_required = response.get("required_bucket_keys")
    if not isinstance(raw_required, list):
        raise ValueError("required_bucket_keys phải là một mảng")
    available = set(available_bucket_keys)
    required_bucket_keys: list[str] = []
    dropped_required = 0
    for raw_key in raw_required:
        if (
            not isinstance(raw_key, str)
            or raw_key not in available
            or raw_key in required_bucket_keys
        ):
            dropped_required += 1
            continue
        required_bucket_keys.append(raw_key)
    if not required_bucket_keys:
        raise ValueError("Final LLM không trả required bucket hợp lệ")

    raw_requirements = response.get("bucket_requirements")
    if not isinstance(raw_requirements, list):
        raw_requirements = []
    requirements: list[dict[str, Any]] = []
    seen_requirement_buckets: set[str] = set()
    concept_bucket_by_key: dict[str, str] = {}
    invalid_requirements = 0
    for raw_requirement in raw_requirements:
        if not isinstance(raw_requirement, Mapping):
            invalid_requirements += 1
            continue
        bucket_key = raw_requirement.get("bucket_key")
        if (
            not isinstance(bucket_key, str)
            or bucket_key not in required_bucket_keys
            or bucket_key in seen_requirement_buckets
        ):
            invalid_requirements += 1
            continue
        raw_concepts = raw_requirement.get("concepts")
        if not isinstance(raw_concepts, list):
            raw_concepts = []
        concepts: list[dict[str, str]] = []
        for raw_concept in raw_concepts:
            if not isinstance(raw_concept, Mapping):
                invalid_requirements += 1
                continue
            concept_key = raw_concept.get("concept_key")
            description = raw_concept.get("description")
            role = raw_concept.get("role")
            if (
                not isinstance(concept_key, str)
                or not concept_key.strip()
                or concept_key in concept_bucket_by_key
                or not isinstance(description, str)
                or not description.strip()
                or role not in RERANK_CONCEPT_ROLES
            ):
                invalid_requirements += 1
                continue
            concept_bucket_by_key[concept_key] = bucket_key
            concepts.append(
                {
                    "concept_key": concept_key,
                    "description": description.strip(),
                    "role": str(role),
                }
            )
        seen_requirement_buckets.add(bucket_key)
        requirements.append({"bucket_key": bucket_key, "concepts": concepts})
    for bucket_key in required_bucket_keys:
        if bucket_key not in seen_requirement_buckets:
            requirements.append({"bucket_key": bucket_key, "concepts": []})

    raw_selections = response.get("ranked_selections")
    if not isinstance(raw_selections, list):
        raw_selections = []
    selected_keys: list[str] = []
    covered_concepts_by_key: dict[str, list[str]] = {}
    invalid_selections = 0
    for raw_selection in raw_selections:
        if not isinstance(raw_selection, Mapping):
            invalid_selections += 1
            continue
        candidate_key = raw_selection.get("candidate_key")
        if (
            not isinstance(candidate_key, str)
            or candidate_key not in by_key
            or candidate_key in selected_keys
        ):
            invalid_selections += 1
            continue
        bucket_key = bucket_by_candidate_key[candidate_key]
        if bucket_key not in required_bucket_keys:
            invalid_selections += 1
            continue
        raw_covered = raw_selection.get("covered_concept_keys")
        if not isinstance(raw_covered, list):
            raw_covered = []
        covered: list[str] = []
        for concept_key in raw_covered:
            if (
                isinstance(concept_key, str)
                and concept_bucket_by_key.get(concept_key) == bucket_key
                and concept_key not in covered
            ):
                covered.append(concept_key)
            else:
                invalid_selections += 1
        selected_keys.append(candidate_key)
        covered_concepts_by_key[candidate_key] = covered

    covered_concepts = {
        concept_key
        for values in covered_concepts_by_key.values()
        for concept_key in values
    }
    if concept_bucket_by_key and not selected_keys:
        raise ValueError(
            "Final LLM khai báo concepts nhưng không trả ranked_selection hợp lệ"
        )
    return {
        "required_bucket_keys": required_bucket_keys,
        "bucket_requirements": requirements,
        "selected_keys": selected_keys,
        "covered_concepts_by_key": covered_concepts_by_key,
        "uncovered_concept_keys": sorted(
            set(concept_bucket_by_key) - covered_concepts
        ),
        "coverage_cell_count": len(concept_bucket_by_key),
        "dropped_required_bucket_values": dropped_required,
        "invalid_requirement_values": invalid_requirements,
        "invalid_selection_values": invalid_selections,
    }


def _complete_required_bucket_coverage(
    llm_keys: Sequence[str],
    nominated_keys: Sequence[str],
    lexical_rescue_keys: Sequence[str],
    finalist_keys: Sequence[str],
    bucket_by_candidate_key: Mapping[str, str],
    required_bucket_keys: Sequence[str],
    coverage_locked_bucket_keys: Sequence[str],
) -> tuple[list[str], dict[str, str], list[str]]:
    """Complete required buckets without treating an ordinary anchor as evidence."""
    selected = list(dict.fromkeys(llm_keys))
    completion_sources: dict[str, str] = {}
    unresolved: list[str] = []
    locked = set(coverage_locked_bucket_keys)
    for bucket_key in required_bucket_keys:
        if any(bucket_by_candidate_key[key] == bucket_key for key in selected):
            continue
        candidate_key = next(
            (
                key
                for key in nominated_keys
                if bucket_by_candidate_key[key] == bucket_key
            ),
            None,
        )
        source = "coverage_completion_scout"
        if candidate_key is None:
            candidate_key = next(
                (
                    key
                    for key in lexical_rescue_keys
                    if bucket_by_candidate_key[key] == bucket_key
                ),
                None,
            )
            source = "coverage_completion_lexical"
        if candidate_key is None and bucket_key in locked:
            candidate_key = next(
                key
                for key in finalist_keys
                if bucket_by_candidate_key[key] == bucket_key
            )
            source = "locked_bucket_presence"
        if candidate_key is None:
            unresolved.append(bucket_key)
            continue
        selected.append(candidate_key)
        completion_sources[candidate_key] = source
    return selected, completion_sources, unresolved


def _trim_coverage_aware(
    selected_keys: Sequence[str],
    bucket_by_candidate_key: Mapping[str, str],
    required_bucket_keys: Sequence[str],
    covered_concepts_by_key: Mapping[str, Sequence[str]],
    output_cap: int,
) -> list[str]:
    """Trim redundant tail candidates while protecting bucket/concept claims."""
    ordered = list(dict.fromkeys(selected_keys))
    if len(ordered) <= output_cap:
        return ordered
    required = set(required_bucket_keys)
    protected: list[str] = []
    covered_buckets: set[str] = set()
    covered_concepts: set[str] = set()
    for key in ordered:
        bucket_key = bucket_by_candidate_key[key]
        new_concepts = set(covered_concepts_by_key.get(key, ())) - covered_concepts
        if bucket_key in required and (
            bucket_key not in covered_buckets or new_concepts
        ):
            protected.append(key)
            covered_buckets.add(bucket_key)
            covered_concepts.update(new_concepts)
    kept = protected[:output_cap]
    for key in ordered:
        if len(kept) == output_cap:
            break
        if key not in kept:
            kept.append(key)
    return kept


def _materialize_ranking(
    selected_keys: Sequence[str],
    by_key: Mapping[str, Mapping[str, Any]],
    source_by_key: Mapping[str, str],
) -> list[Candidate]:
    """Map opaque keys back to candidates and attach compatible rank metadata."""
    result: list[Candidate] = []
    count = len(selected_keys)
    for position, key in enumerate(selected_keys, start=1):
        item = dict(by_key[key])
        source = source_by_key.get(key, "llm")
        item.update(
            {
                "rerank_score": (count - position + 1) / count,
                "rerank_reason": source,
                "rerank_rank": position,
                "rerank_source": source,
            }
        )
        result.append(item)
    return result


def rerank_with_diagnostics(
    question: str,
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[list[Candidate], dict[str, Any]]:
    """Rerank candidates and expose loss-attribution diagnostics."""
    if not question.strip():
        raise ValueError("question must not be empty")
    if not candidates:
        raise RerankerError("Không có candidate để rerank")

    validated = _validate_candidates(candidates)
    enriched_candidates = _attach_context_to_validated(question, validated)
    enriched = _diversified_shortlist(question, enriched_candidates)
    available_buckets, prompt_candidates, by_key, bucket_by_key = (
        _build_prompt_contract(question, enriched)
    )
    available_bucket_keys = [
        str(bucket["bucket_key"])
        for bucket in available_buckets
    ]
    coverage_locked_bucket_keys, coverage_lock_reasons = _coverage_locked_buckets(
        question, available_buckets
    )
    lexical_rescue_keys = _exact_lexical_finalist_keys(
        question, by_key, bucket_by_key
    )
    scout_chunks = _balanced_scout_chunks(prompt_candidates)
    nominated_keys: list[str] = []
    nomination_priorities: dict[str, tuple[Any, ...]] = {}
    scout_prompt_chars: list[int] = []
    scout_valid_counts: list[int] = []
    scout_diagnostics: list[dict[str, Any]] = []
    scout_prompt_payloads: list[dict[str, Any]] = []
    for scout_index, chunk in enumerate(scout_chunks, start=1):
        if not chunk:
            scout_prompt_chars.append(0)
            scout_valid_counts.append(0)
            scout_diagnostics.append(
                {"scout_index": scout_index, "input_keys": [], "response": None, "nominated_keys": [], "error": None}
            )
            continue
        scout_maximum = min(RERANK_SCOUT_OUTPUT_MAX, len(chunk))
        scout_prompt = build_rerank_scout_prompt(
            question,
            chunk,
            scout_maximum,
        )
        scout_prompt_payloads.append(json.loads(scout_prompt))
        scout_prompt_chars.append(len(scout_prompt))
        chunk_by_key = {
            str(candidate["candidate_key"]): by_key[str(candidate["candidate_key"])]
            for candidate in chunk
        }
        scout_keys: list[str] = []
        scout_response: Mapping[str, Any] | None = None
        scout_error: str | None = None
        try:
            scout_response = generate_structured(
                scout_prompt,
                system_prompt=RERANK_SCOUT_SYSTEM_PROMPT,
            )
            scout_keys = _salvage_rerank_response(
                scout_response,
                chunk_by_key,
                scout_maximum,
            )
        except (LLMResponseError, ValueError) as exc:
            scout_error = str(exc)
            logger.warning(
                "Rerank scout %d output is unusable: %s",
                scout_index,
                exc,
            )
        for position, key in enumerate(scout_keys, start=1):
            nomination_priorities.setdefault(
                key,
                (position, *_fallback_sort_key(by_key[key])),
            )
        nominated_keys.extend(scout_keys)
        scout_valid_counts.append(len(scout_keys))
        scout_diagnostics.append(
            {
                "scout_index": scout_index,
                "input_keys": [str(item["candidate_key"]) for item in chunk],
                "response": dict(scout_response) if scout_response is not None else None,
                "nominated_keys": scout_keys,
                "error": scout_error,
            }
        )

    nominated_keys = sorted(
        set(nominated_keys),
        key=lambda key: nomination_priorities[key],
    )

    finalist_seed_keys = list(dict.fromkeys([*nominated_keys, *lexical_rescue_keys]))
    finalist_keys = _ensure_nomination_bucket_coverage(
        finalist_seed_keys,
        by_key,
        bucket_by_key,
        available_bucket_keys,
    )
    prompt_candidate_by_key = {
        str(candidate["candidate_key"]): candidate
        for candidate in prompt_candidates
    }
    final_prompt_candidates = [
        prompt_candidate_by_key[key]
        for key in finalist_keys
    ]
    final_by_key = {key: by_key[key] for key in finalist_keys}
    final_bucket_by_key = {
        key: bucket_by_key[key]
        for key in finalist_keys
    }
    hard_maximum = min(RERANK_OUTPUT_MAX, len(final_prompt_candidates))
    final_prompt = build_rerank_prompt(
        question,
        available_buckets,
        final_prompt_candidates,
        hard_maximum,
        coverage_locked_bucket_keys,
    )
    final_response: Mapping[str, Any] | None = None
    final_decision: dict[str, Any] | None = None
    final_error: str | None = None
    try:
        final_response = generate_structured(
            final_prompt,
            system_prompt=RERANK_SYSTEM_PROMPT,
        )
        final_decision = _salvage_final_response(
            final_response,
            final_by_key,
            final_bucket_by_key,
            available_bucket_keys,
        )
    except (LLMResponseError, ValueError) as exc:
        final_error = str(exc)
        logger.warning(
            "Final reranker output is unusable; using scout nominations: %s",
            exc,
        )

    completion_sources: dict[str, str] = {}
    unresolved_required_bucket_keys: list[str] = []
    policy_added_required_bucket_keys: list[str] = []
    if final_decision is not None:
        final_keys = list(final_decision["selected_keys"])
        required_bucket_keys = list(final_decision["required_bucket_keys"])
        policy_added_required_bucket_keys = [
            key
            for key in coverage_locked_bucket_keys
            if key not in required_bucket_keys
        ]
        required_bucket_keys.extend(policy_added_required_bucket_keys)
        bucket_requirements = list(final_decision["bucket_requirements"])
        bucket_requirements.extend(
            {"bucket_key": key, "concepts": []}
            for key in policy_added_required_bucket_keys
        )
        output_cap = _dynamic_output_cap(
            len(required_bucket_keys),
            int(final_decision["coverage_cell_count"]),
            len(finalist_keys),
        )
        (
            selected_keys,
            completion_sources,
            unresolved_required_bucket_keys,
        ) = _complete_required_bucket_coverage(
            final_keys,
            nominated_keys,
            lexical_rescue_keys,
            finalist_keys,
            final_bucket_by_key,
            required_bucket_keys,
            coverage_locked_bucket_keys,
        )
        selected_keys = _trim_coverage_aware(
            selected_keys,
            final_bucket_by_key,
            required_bucket_keys,
            final_decision["covered_concepts_by_key"],
            output_cap,
        )
        source_by_key = {key: "llm" for key in final_keys}
        source_by_key.update(completion_sources)
    else:
        if not nominated_keys and not coverage_locked_bucket_keys:
            raise RerankerError(
                "Final reranker và cả hai scout đều không trả candidate hợp lệ"
            )
        required_bucket_keys = list(coverage_locked_bucket_keys)
        bucket_requirements = []
        final_keys = []
        output_cap = min(RERANK_OUTPUT_MAX, len(finalist_keys))
        selected_keys = nominated_keys[:output_cap]
        source_by_key = {key: "scout_fallback" for key in selected_keys}
        if required_bucket_keys:
            (
                selected_keys,
                completion_sources,
                unresolved_required_bucket_keys,
            ) = _complete_required_bucket_coverage(
                selected_keys,
                nominated_keys,
                lexical_rescue_keys,
                finalist_keys,
                final_bucket_by_key,
                required_bucket_keys,
                coverage_locked_bucket_keys,
            )
            selected_keys = _trim_coverage_aware(
                selected_keys,
                final_bucket_by_key,
                required_bucket_keys,
                {},
                output_cap,
            )
            source_by_key.update(completion_sources)

    result = _materialize_ranking(selected_keys, by_key, source_by_key)
    candidate_catalog = {
        key: {
            "bucket_key": bucket_by_key[key],
            "table_id": candidate["table_id"],
            "table_ref": (
                f"{candidate['metadata']['doc_id']}|"
                f"{candidate['metadata']['start_line']}"
            ),
            "doc_id": candidate["metadata"]["doc_id"],
            "table_type": candidate["metadata"]["table_type"],
            "retrieval_rank": candidate.get(
                "retrieval_rank", candidate.get("dense_rank")
            ),
        }
        for key, candidate in by_key.items()
    }
    diagnostics = {
        "input_candidate_count": len(candidates),
        "available_buckets": available_buckets,
        "candidate_catalog": candidate_catalog,
        "shortlist_keys": list(by_key),
        "scout_prompts": scout_prompt_payloads,
        "scouts": scout_diagnostics,
        "scout_nominated_keys": nominated_keys,
        "lexical_finalist_keys": lexical_rescue_keys,
        "finalist_keys": finalist_keys,
        "final_prompt": json.loads(final_prompt),
        "final_response": dict(final_response) if final_response is not None else None,
        "final_error": final_error,
        "coverage_locked_bucket_keys": coverage_locked_bucket_keys,
        "coverage_lock_reasons": coverage_lock_reasons,
        "policy_added_required_bucket_keys": policy_added_required_bucket_keys,
        "required_bucket_keys": required_bucket_keys,
        "bucket_requirements": bucket_requirements,
        "uncovered_concept_keys": (
            final_decision["uncovered_concept_keys"] if final_decision else []
        ),
        "final_llm_keys": final_keys,
        "coverage_completion": completion_sources,
        "unresolved_required_bucket_keys": unresolved_required_bucket_keys,
        "selected_keys": selected_keys,
        "output_cap": output_cap,
    }
    logger.info(
        "Rerank completed: input=%d buckets=%d shortlist=%d scout_chunks=%s "
        "scout_prompt_chars=%s scout_valid=%s finalists=%d final_prompt_chars=%d "
        "lexical_rescue=%d coverage_locked=%s required_buckets=%s output_cap=%d "
        "final_valid=%d coverage_added=%d unresolved=%s output=%d",
        len(candidates),
        len(available_buckets),
        len(enriched),
        [len(chunk) for chunk in scout_chunks],
        scout_prompt_chars,
        scout_valid_counts,
        len(finalist_keys),
        len(final_prompt),
        len(lexical_rescue_keys),
        coverage_locked_bucket_keys,
        required_bucket_keys,
        output_cap,
        len(final_keys),
        len(completion_sources),
        unresolved_required_bucket_keys,
        len(result),
    )
    logger.debug(
        "Rerank scores: %s",
        [(item["table_id"], item["rerank_score"]) for item in result],
    )
    return result, diagnostics


def rerank(
    question: str,
    candidates: Sequence[Mapping[str, Any]],
) -> list[Candidate]:
    """Nominate with two bounded scouts, then choose with one final arbiter."""
    result, _ = rerank_with_diagnostics(question, candidates)
    return result
