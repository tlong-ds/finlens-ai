"""Dense retrieval and coverage-aware LLM reranking for financial tables."""

from __future__ import annotations

import csv
import json
import logging
import math
import re
import threading
from collections.abc import Mapping, Sequence
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
RERANK_SHORTLIST_MAX = 30
RERANK_SHORTLIST_RESCUE_MAX = 4
RERANK_SCOUT_COUNT = 2
RERANK_SCOUT_OUTPUT_MAX = 8
RERANK_OUTPUT_MIN = 8
RERANK_OUTPUT_MAX = 18
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
        raise NoMatchingCandidatesError(
            "Không tìm thấy bảng nào khớp metadata filter"
        )
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
    """Order candidates deterministically by dense rank, then retrieval score."""
    dense_rank = candidate.get("dense_rank")
    rank_value = (
        float(dense_rank)
        if isinstance(dense_rank, (int, float)) and not isinstance(dense_rank, bool)
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

    matching_titles: list[str] = []
    raw_titles = context.get("table_titles")
    if isinstance(raw_titles, Sequence) and not isinstance(raw_titles, (str, bytes)):
        matching_titles = sorted(
            (
                str(title)
                for title in raw_titles
                if isinstance(title, str)
                and query_tokens & _question_tokens(title)
            ),
            key=lambda title: (
                -len(query_tokens & _question_tokens(title)),
                title,
            ),
        )[:3]
    return {
        "exact_phrase_rows": exact_rows,
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


def _dynamic_output_cap(bucket_count: int, shortlist_size: int) -> int:
    if bucket_count < 1 or shortlist_size < bucket_count:
        raise ValueError("Invalid rerank bucket or shortlist size")
    normal_cap = min(
        RERANK_OUTPUT_MAX,
        max(RERANK_OUTPUT_MIN, 2 * bucket_count + 2),
    )
    return min(shortlist_size, max(bucket_count, normal_cap))


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

    required_buckets: list[dict[str, Any]] = []
    for doc_id in doc_ids:
        metadata = next(
            candidate["metadata"]
            for candidate in candidates
            if candidate["metadata"]["doc_id"] == doc_id
        )
        required_buckets.append(
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
                "dense_rank": candidate.get("dense_rank"),
                "context": _prioritized_rerank_context(
                    question,
                    candidate["rerank_context"],
                ),
            }
        )
    return required_buckets, prompt_candidates, by_key, bucket_by_candidate_key


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


def _complete_bucket_coverage(
    llm_keys: Sequence[str],
    by_key: Mapping[str, Mapping[str, Any]],
    bucket_by_candidate_key: Mapping[str, str],
    required_bucket_keys: Sequence[str],
    output_cap: int,
) -> list[str]:
    selected = list(llm_keys)
    protected: set[str] = set()
    for bucket_key in required_bucket_keys:
        representative = next(
            (
                key
                for key in selected
                if bucket_by_candidate_key[key] == bucket_key
            ),
            None,
        )
        if representative is None:
            representative = next(
                key
                for key in by_key
                if bucket_by_candidate_key[key] == bucket_key
            )
            selected.append(representative)
        protected.add(representative)

    while len(selected) > output_cap:
        removable_index = next(
            (
                index
                for index in range(len(selected) - 1, -1, -1)
                if selected[index] not in protected
            ),
            None,
        )
        if removable_index is None:
            break
        selected.pop(removable_index)
    return selected


def _materialize_ranking(
    selected_keys: Sequence[str],
    by_key: Mapping[str, Mapping[str, Any]],
    llm_keys: Sequence[str],
) -> list[Candidate]:
    """Map opaque keys back to candidates and attach compatible rank metadata."""
    result: list[Candidate] = []
    count = len(selected_keys)
    llm_key_set = set(llm_keys)
    for position, key in enumerate(selected_keys, start=1):
        item = dict(by_key[key])
        source = "llm" if key in llm_key_set else "coverage_completion"
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


def rerank(
    question: str,
    candidates: Sequence[Mapping[str, Any]],
) -> list[Candidate]:
    """Nominate with two bounded scouts, then choose with one final arbiter."""
    if not question.strip():
        raise ValueError("question must not be empty")
    if not candidates:
        raise RerankerError("Không có candidate để rerank")

    validated = _validate_candidates(candidates)
    enriched_candidates = _attach_context_to_validated(question, validated)
    enriched = _diversified_shortlist(question, enriched_candidates)
    required_buckets, prompt_candidates, by_key, bucket_by_key = (
        _build_prompt_contract(question, enriched)
    )
    required_bucket_keys = [
        str(bucket["bucket_key"])
        for bucket in required_buckets
    ]
    scout_chunks = _balanced_scout_chunks(prompt_candidates)
    nominated_keys: list[str] = []
    scout_prompt_chars: list[int] = []
    scout_valid_counts: list[int] = []
    for scout_index, chunk in enumerate(scout_chunks, start=1):
        if not chunk:
            scout_prompt_chars.append(0)
            scout_valid_counts.append(0)
            continue
        scout_maximum = min(RERANK_SCOUT_OUTPUT_MAX, len(chunk))
        scout_prompt = build_rerank_scout_prompt(
            question,
            chunk,
            scout_maximum,
        )
        scout_prompt_chars.append(len(scout_prompt))
        chunk_by_key = {
            str(candidate["candidate_key"]): by_key[str(candidate["candidate_key"])]
            for candidate in chunk
        }
        scout_keys: list[str] = []
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
            logger.warning(
                "Rerank scout %d output is unusable: %s",
                scout_index,
                exc,
            )
        nominated_keys.extend(scout_keys)
        scout_valid_counts.append(len(scout_keys))

    finalist_keys = _ensure_nomination_bucket_coverage(
        nominated_keys,
        by_key,
        bucket_by_key,
        required_bucket_keys,
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
    output_cap = _dynamic_output_cap(
        len(required_buckets),
        len(final_prompt_candidates),
    )
    final_prompt = build_rerank_prompt(
        question,
        required_buckets,
        final_prompt_candidates,
        output_cap,
    )
    final_keys: list[str] = []
    try:
        final_response = generate_structured(
            final_prompt,
            system_prompt=RERANK_SYSTEM_PROMPT,
        )
        final_keys = _salvage_rerank_response(
            final_response,
            final_by_key,
            output_cap,
        )
    except (LLMResponseError, ValueError) as exc:
        logger.warning(
            "Final reranker output is unusable; using finalist bucket anchors: %s",
            exc,
        )

    if final_keys:
        selected_keys = final_keys
    else:
        selected_keys = _complete_bucket_coverage(
            [],
            final_by_key,
            final_bucket_by_key,
            required_bucket_keys,
            output_cap,
        )
    result = _materialize_ranking(selected_keys, by_key, final_keys)
    coverage_added = sum(
        item["rerank_source"] == "coverage_completion"
        for item in result
    )
    logger.info(
        "Rerank completed: input=%d buckets=%d shortlist=%d scout_chunks=%s "
        "scout_prompt_chars=%s scout_valid=%s finalists=%d final_prompt_chars=%d "
        "output_cap=%d final_valid=%d coverage_added=%d output=%d",
        len(candidates),
        len(required_buckets),
        len(enriched),
        [len(chunk) for chunk in scout_chunks],
        scout_prompt_chars,
        scout_valid_counts,
        len(finalist_keys),
        len(final_prompt),
        output_cap,
        len(final_keys),
        coverage_added,
        len(result),
    )
    logger.debug(
        "Rerank scores: %s",
        [(item["table_id"], item["rerank_score"]) for item in result],
    )
    return result
