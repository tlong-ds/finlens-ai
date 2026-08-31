"""Coverage-aware final table selection."""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.config import Settings
from src.generation.prompts import (
    SELECTOR_SCOUT_SYSTEM_PROMPT,
    SELECTOR_SYSTEM_PROMPT,
    build_selector_prompt,
    build_selector_scout_prompt,
)
from src.providers.llm import LLMResponseError, generate_structured
from src.retrieval.context import (
    _attach_context_to_validated,
    _question_tokens,
    _validate_candidates,
)
from src.retrieval.dense import (
    FPT_RERANK_TOP_N,
    RERANK_CONCEPT_ROLES,
    RERANK_FINALIST_LEXICAL_RESCUE_MAX,
    RERANK_FINALIST_LEXICAL_RESCUE_PER_BUCKET,
    RERANK_OUTPUT_MAX,
    RERANK_SCOUT_COUNT,
    RERANK_SCOUT_OUTPUT_MAX,
    Candidate,
    RerankerError,
    SelectorResponseError,
    SelectorSelectionError,
)
from src.retrieval.reranking import _fallback_sort_key, _salvage_rerank_response

logger = logging.getLogger(__name__)


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
            len(raw_words) == 1 and re.fullmatch(r"[A-Z][A-Z0-9/.-]{2,9}", text.strip())
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

    scored_rows.sort(
        key=lambda item: (-item[0], -item[1], -item[2], str(item[3]["row"]))
    )
    exact_rows = [item[3] for item in scored_rows if item[0]][:3]
    exact_row_numbers = {row["row"] for row in exact_rows}
    strong_rows = [
        item[3] for item in scored_rows if item[3]["row"] not in exact_row_numbers
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


def _dynamic_output_cap(
    required_bucket_count: int,
    coverage_cell_count: int,
    finalist_count: int,
) -> int:
    """Calculate minimal exact cap from required buckets and operand coverage."""
    if (
        required_bucket_count < 1
        or coverage_cell_count < 0
        or finalist_count < required_bucket_count
    ):
        raise ValueError("Invalid rerank bucket or shortlist size")
    return min(
        finalist_count,
        RERANK_OUTPUT_MAX,
        max(1, required_bucket_count, coverage_cell_count),
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
        doc_id: f"b{index:02d}" for index, doc_id in enumerate(doc_ids, start=1)
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
                "bge_rank": candidate.get("rerank_rank"),
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
            key for key in by_key if bucket_by_candidate_key[key] == bucket_key
        )
        selected.append(anchor)
        selected_buckets.add(bucket_key)
    return selected


def _validate_final_response(
    response: Mapping[str, Any],
    by_key: Mapping[str, Mapping[str, Any]],
    bucket_by_candidate_key: Mapping[str, str],
    available_bucket_keys: Sequence[str],
    coverage_locked_bucket_keys: Sequence[str],
    maximum: int,
) -> dict[str, Any]:
    """Reject any final decision that does not prove exact concept coverage."""
    errors: list[str] = []
    expected_fields = {
        "required_bucket_keys",
        "bucket_requirements",
        "ranked_selections",
    }
    unexpected_fields = sorted(set(response) - expected_fields)
    missing_fields = sorted(expected_fields - set(response))
    if unexpected_fields:
        errors.append("key không được phép: " + ", ".join(unexpected_fields))
    if missing_fields:
        errors.append("thiếu key: " + ", ".join(missing_fields))

    raw_required = response.get("required_bucket_keys")
    if not isinstance(raw_required, list):
        errors.append("required_bucket_keys phải là một mảng")
        raw_required = []
    available = set(available_bucket_keys)
    required_bucket_keys: list[str] = []
    for raw_key in raw_required:
        if not isinstance(raw_key, str) or raw_key not in available:
            errors.append(f"required bucket không hợp lệ: {raw_key!r}")
            continue
        if raw_key in required_bucket_keys:
            errors.append(f"required bucket bị lặp: {raw_key}")
            continue
        required_bucket_keys.append(raw_key)
    if not required_bucket_keys:
        errors.append("không có required bucket hợp lệ")
    missing_locked_buckets = sorted(
        set(coverage_locked_bucket_keys) - set(required_bucket_keys)
    )
    if missing_locked_buckets:
        errors.append(
            "thiếu coverage_locked bucket: " + ", ".join(missing_locked_buckets)
        )

    raw_requirements = response.get("bucket_requirements")
    if not isinstance(raw_requirements, list):
        errors.append("bucket_requirements phải là một mảng")
        raw_requirements = []
    requirements: list[dict[str, Any]] = []
    seen_requirement_buckets: set[str] = set()
    concept_bucket_by_key: dict[str, str] = {}
    for raw_requirement in raw_requirements:
        if not isinstance(raw_requirement, Mapping):
            errors.append("bucket_requirement phải là object")
            continue
        bucket_key = raw_requirement.get("bucket_key")
        if not isinstance(bucket_key, str) or bucket_key not in required_bucket_keys:
            errors.append(f"requirement ngoài required bucket: {bucket_key!r}")
            continue
        if bucket_key in seen_requirement_buckets:
            errors.append(f"bucket_requirement bị lặp: {bucket_key}")
            continue
        raw_concepts = raw_requirement.get("concepts")
        if not isinstance(raw_concepts, list):
            errors.append(f"concepts của {bucket_key} phải là một mảng")
            raw_concepts = []
        if not raw_concepts:
            errors.append(f"required bucket không khai báo concept: {bucket_key}")
        concepts: list[dict[str, str]] = []
        for raw_concept in raw_concepts:
            if not isinstance(raw_concept, Mapping):
                errors.append(f"concept của {bucket_key} phải là object")
                continue
            concept_key = raw_concept.get("concept_key")
            description = raw_concept.get("description")
            role = raw_concept.get("role")
            if not isinstance(concept_key, str) or not concept_key.strip():
                errors.append(f"concept_key rỗng/không hợp lệ trong {bucket_key}")
                continue
            if concept_key in concept_bucket_by_key:
                errors.append(f"concept_key bị lặp: {concept_key}")
                continue
            if not isinstance(description, str) or not description.strip():
                errors.append(f"concept thiếu description: {concept_key}")
                continue
            if role not in RERANK_CONCEPT_ROLES:
                errors.append(f"concept role không hợp lệ: {concept_key}")
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
    missing_requirement_buckets = sorted(
        set(required_bucket_keys) - seen_requirement_buckets
    )
    if missing_requirement_buckets:
        errors.append(
            "required bucket thiếu requirement: "
            + ", ".join(missing_requirement_buckets)
        )

    raw_selections = response.get("ranked_selections")
    if not isinstance(raw_selections, list):
        errors.append("ranked_selections phải là một mảng")
        raw_selections = []
    selected_keys: list[str] = []
    covered_concepts_by_key: dict[str, list[str]] = {}
    empty_coverage_candidate_keys: list[str] = []
    for raw_selection in raw_selections:
        if not isinstance(raw_selection, Mapping):
            errors.append("ranked_selection phải là object")
            continue
        candidate_key = raw_selection.get("candidate_key")
        if not isinstance(candidate_key, str) or candidate_key not in by_key:
            errors.append(f"candidate_key không xác định: {candidate_key!r}")
            continue
        if candidate_key in selected_keys:
            errors.append(f"candidate_key bị lặp: {candidate_key}")
            continue
        bucket_key = bucket_by_candidate_key[candidate_key]
        if bucket_key not in required_bucket_keys:
            errors.append(
                f"candidate {candidate_key} nằm ngoài required bucket {bucket_key}"
            )
            continue
        raw_covered = raw_selection.get("covered_concept_keys")
        if not isinstance(raw_covered, list):
            errors.append(f"covered_concept_keys của {candidate_key} phải là mảng")
            raw_covered = []
        if not raw_covered:
            empty_coverage_candidate_keys.append(candidate_key)
            errors.append(f"candidate không cover concept nào: {candidate_key}")
        covered: list[str] = []
        for concept_key in raw_covered:
            if not isinstance(concept_key, str):
                errors.append(
                    f"candidate {candidate_key} cover concept không xác định: {concept_key!r}"
                )
                continue
            if concept_key not in concept_bucket_by_key:
                if concept_key.startswith(f"{bucket_key}_") or concept_key.startswith(
                    bucket_key
                ):
                    concept_bucket_by_key[concept_key] = bucket_key
                else:
                    suffix = (
                        concept_key.split("_", 1)[-1]
                        if "_" in concept_key
                        else concept_key
                    )
                    target_key = f"{bucket_key}_{suffix}"
                    concept_bucket_by_key[target_key] = bucket_key
                    concept_key = target_key
            elif concept_bucket_by_key[concept_key] != bucket_key:
                suffix = (
                    concept_key.split("_", 1)[-1] if "_" in concept_key else concept_key
                )
                target_key = f"{bucket_key}_{suffix}"
                concept_bucket_by_key[target_key] = bucket_key
                concept_key = target_key
            if concept_key in covered:
                errors.append(
                    f"candidate {candidate_key} lặp concept coverage: {concept_key}"
                )
                continue
            covered.append(concept_key)
        selected_keys.append(candidate_key)
        covered_concepts_by_key[candidate_key] = covered

    if not selected_keys:
        errors.append("không có ranked_selection hợp lệ")
    if len(selected_keys) > maximum:
        errors.append(f"số selection {len(selected_keys)} vượt giới_hạn_cứng {maximum}")

    covered_concepts = {
        concept_key
        for values in covered_concepts_by_key.values()
        for concept_key in values
    }
    uncovered_concept_keys = sorted(set(concept_bucket_by_key) - covered_concepts)
    if uncovered_concept_keys:
        errors.append("concept chưa được cover: " + ", ".join(uncovered_concept_keys))
    selected_bucket_keys = list(
        dict.fromkeys(bucket_by_candidate_key[key] for key in selected_keys)
    )
    unrepresented_required_bucket_keys = sorted(
        set(required_bucket_keys) - set(selected_bucket_keys)
    )
    if unrepresented_required_bucket_keys:
        errors.append(
            "required bucket chưa có selection: "
            + ", ".join(unrepresented_required_bucket_keys)
        )

    coverage_status = {
        "valid": not errors,
        "errors": errors,
        "required_bucket_keys": required_bucket_keys,
        "coverage_locked_bucket_keys": list(coverage_locked_bucket_keys),
        "missing_locked_bucket_keys": missing_locked_buckets,
        "declared_concept_keys": list(concept_bucket_by_key),
        "covered_concept_keys": sorted(covered_concepts),
        "uncovered_concept_keys": uncovered_concept_keys,
        "selected_candidate_keys": selected_keys,
        "empty_coverage_candidate_keys": empty_coverage_candidate_keys,
        "selected_bucket_keys": selected_bucket_keys,
        "unrepresented_required_bucket_keys": unrepresented_required_bucket_keys,
    }
    if errors:
        raise SelectorResponseError(errors, coverage_status)

    return {
        "required_bucket_keys": required_bucket_keys,
        "bucket_requirements": requirements,
        "selected_keys": selected_keys,
        "covered_concepts_by_key": covered_concepts_by_key,
        "coverage_cell_count": len(concept_bucket_by_key),
        "coverage_status": coverage_status,
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
    """Map opaque keys back to candidates and attach selector metadata."""
    result: list[Candidate] = []
    count = len(selected_keys)
    for position, key in enumerate(selected_keys, start=1):
        item = dict(by_key[key])
        source = source_by_key.get(key, "llm")
        item.update(
            {
                "selection_score": (count - position + 1) / count,
                "selection_reason": source,
                "selection_rank": position,
                "selection_source": source,
            }
        )
        result.append(item)
    return result


def _run_selector_scout(
    scout_index: int,
    scout_prompt: str,
    chunk_by_key: Mapping[str, Mapping[str, Any]],
    scout_maximum: int,
    settings: Settings,
) -> tuple[Mapping[str, Any] | None, list[str], str | None]:
    """Run one independent selector scout and salvage its nominations."""
    scout_response: Mapping[str, Any] | None = None
    try:
        scout_response = generate_structured(
            scout_prompt,
            settings=settings,
            system_prompt=SELECTOR_SCOUT_SYSTEM_PROMPT,
            native=False,
        )
        scout_keys = _salvage_rerank_response(
            scout_response,
            chunk_by_key,
            scout_maximum,
        )
        return scout_response, scout_keys, None
    except (LLMResponseError, ValueError) as exc:
        logger.warning(
            "Selector scout %d output is unusable: %s",
            scout_index,
            exc,
        )
        return scout_response, [], str(exc)


def _bounded_selector_response(
    response: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Keep only bounded expected selector fields in validation artifacts."""
    if response is None:
        return None

    def bounded(value: Any, depth: int = 0) -> Any:
        if depth >= 5:
            return "<depth-limited>"
        if isinstance(value, str):
            return value[:500]
        if isinstance(value, Mapping):
            return {
                str(key)[:100]: bounded(item, depth + 1)
                for key, item in list(value.items())[:25]
            }
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [bounded(item, depth + 1) for item in list(value)[:25]]
        if value is None or isinstance(value, (bool, int, float)):
            return value
        return str(value)[:500]

    return {
        key: bounded(response.get(key))
        for key in (
            "required_bucket_keys",
            "bucket_requirements",
            "ranked_selections",
        )
        if key in response
    }


def _bounded_scout_response(
    response: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if response is None:
        return None
    raw_keys = response.get("ranked_candidate_keys")
    return {
        "ranked_candidate_keys": (
            [str(key)[:100] for key in raw_keys[:FPT_RERANK_TOP_N]]
            if isinstance(raw_keys, list)
            else None
        )
    }


def select_tables_with_diagnostics(
    question: str,
    candidates: Sequence[Mapping[str, Any]],
    *,
    settings: Settings,
) -> tuple[list[Candidate], dict[str, Any]]:
    """Select minimal exact tables from FPT top-20 candidates with diagnostics."""
    if not question.strip():
        raise ValueError("question must not be empty")
    if not candidates:
        raise RerankerError("Không có candidate để rerank")
    if len(candidates) > FPT_RERANK_TOP_N:
        raise RerankerError(
            f"Selector received {len(candidates)} candidates; maximum FPT input is "
            f"{FPT_RERANK_TOP_N}"
        )

    validated = _validate_candidates(candidates)
    enriched = _attach_context_to_validated(question, validated)
    available_buckets, prompt_candidates, by_key, bucket_by_key = (
        _build_prompt_contract(question, enriched)
    )
    available_bucket_keys = [str(bucket["bucket_key"]) for bucket in available_buckets]
    coverage_locked_bucket_keys, coverage_lock_reasons = _coverage_locked_buckets(
        question, available_buckets
    )
    lexical_rescue_keys = _exact_lexical_finalist_keys(question, by_key, bucket_by_key)
    scout_chunks = _balanced_scout_chunks(prompt_candidates)
    nominated_keys: list[str] = []
    nomination_priorities: dict[str, tuple[Any, ...]] = {}
    scout_prompt_chars: list[int] = []
    scout_valid_counts: list[int] = []
    scout_diagnostics: list[dict[str, Any]] = []
    scout_jobs: list[tuple[int, str, dict[str, Mapping[str, Any]], int]] = []
    for scout_index, chunk in enumerate(scout_chunks, start=1):
        if not chunk:
            scout_prompt_chars.append(0)
            continue
        scout_maximum = min(RERANK_SCOUT_OUTPUT_MAX, len(chunk))
        scout_prompt = build_selector_scout_prompt(
            question,
            chunk,
            scout_maximum,
        )
        scout_prompt_chars.append(len(scout_prompt))
        chunk_by_key = {
            str(candidate["candidate_key"]): by_key[str(candidate["candidate_key"])]
            for candidate in chunk
        }
        scout_jobs.append((scout_index, scout_prompt, chunk_by_key, scout_maximum))

    scout_results: dict[
        int, tuple[Mapping[str, Any] | None, list[str], str | None]
    ] = {}
    if scout_jobs:
        with ThreadPoolExecutor(
            max_workers=len(scout_jobs),
            thread_name_prefix="rerank-scout",
        ) as executor:
            futures = {
                scout_index: executor.submit(
                    _run_selector_scout,
                    scout_index,
                    scout_prompt,
                    chunk_by_key,
                    scout_maximum,
                    settings,
                )
                for (
                    scout_index,
                    scout_prompt,
                    chunk_by_key,
                    scout_maximum,
                ) in scout_jobs
            }
            scout_results = {
                scout_index: future.result() for scout_index, future in futures.items()
            }

    for scout_index, chunk in enumerate(scout_chunks, start=1):
        if not chunk:
            scout_valid_counts.append(0)
            scout_diagnostics.append(
                {
                    "scout_index": scout_index,
                    "input_keys": [],
                    "response": None,
                    "nominated_keys": [],
                    "error": None,
                }
            )
            continue
        scout_response, scout_keys, scout_error = scout_results[scout_index]
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
                "response": _bounded_scout_response(scout_response),
                "nominated_keys": scout_keys,
                "error": scout_error,
            }
        )

    nominated_keys = sorted(
        set(nominated_keys),
        key=lambda key: nomination_priorities[key],
    )

    finalist_keys = _ensure_nomination_bucket_coverage(
        list(dict.fromkeys([*nominated_keys, *lexical_rescue_keys])),
        by_key,
        bucket_by_key,
        available_bucket_keys,
    )
    prompt_candidate_by_key = {
        str(candidate["candidate_key"]): candidate for candidate in prompt_candidates
    }
    final_prompt_candidates = [prompt_candidate_by_key[key] for key in finalist_keys]
    final_by_key = {key: by_key[key] for key in finalist_keys}
    final_bucket_by_key = {key: bucket_by_key[key] for key in finalist_keys}
    hard_maximum = min(RERANK_OUTPUT_MAX, len(final_prompt_candidates))
    final_response: Mapping[str, Any] | None = None
    final_decision: dict[str, Any] | None = None
    final_attempts: list[dict[str, Any]] = []
    final_prompt_chars: list[int] = []
    feedback = ""
    previous_response: Mapping[str, Any] | None = None
    for final_attempt in range(1, 3):
        final_response = None
        final_prompt = build_selector_prompt(
            question,
            available_buckets,
            final_prompt_candidates,
            hard_maximum,
            coverage_locked_bucket_keys,
            feedback,
            previous_response,
        )
        final_prompt_chars.append(len(final_prompt))
        coverage_status: dict[str, Any]
        attempt_error: str | None = None
        try:
            final_response = generate_structured(
                final_prompt,
                settings=settings,
                system_prompt=SELECTOR_SYSTEM_PROMPT,
                native=False,
            )
            final_decision = _validate_final_response(
                final_response,
                final_by_key,
                final_bucket_by_key,
                available_bucket_keys,
                coverage_locked_bucket_keys,
                hard_maximum,
            )
            coverage_status = dict(final_decision["coverage_status"])
        except SelectorResponseError as exc:
            final_decision = None
            coverage_status = dict(exc.coverage_status)
            attempt_error = str(exc)
        except LLMResponseError as exc:
            final_decision = None
            attempt_error = str(exc)
            coverage_status = {
                "valid": False,
                "errors": [attempt_error],
                "required_bucket_keys": [],
                "coverage_locked_bucket_keys": coverage_locked_bucket_keys,
                "declared_concept_keys": [],
                "covered_concept_keys": [],
                "uncovered_concept_keys": [],
                "selected_candidate_keys": [],
                "empty_coverage_candidate_keys": [],
                "selected_bucket_keys": [],
                "unrepresented_required_bucket_keys": coverage_locked_bucket_keys,
            }
        final_attempts.append(
            {
                "attempt": final_attempt,
                "response": _bounded_selector_response(final_response),
                "coverage_status": coverage_status,
                "error": attempt_error,
            }
        )
        if final_decision is not None:
            break
        if final_attempt == 1:
            previous_response = final_response
            feedback = (
                "Phản hồi trước bị từ chối. Sửa đúng các lỗi coverage sau, đặc biệt "
                "mọi concept chưa cover, candidate coverage rỗng và bucket còn thiếu: "
                + (attempt_error or "response không đúng schema")
            )[:4_000]

    correction_attempt = {
        "attempted": len(final_attempts) == 2,
        "feedback": feedback or None,
        "initial_response": (
            final_attempts[0]["response"] if len(final_attempts) == 2 else None
        ),
        "succeeded": final_decision is not None,
    }
    if final_decision is None:
        failure_diagnostics = {
            "finalist_keys": finalist_keys,
            "final_response": _bounded_selector_response(final_response),
            "selected_keys": [],
            "coverage_status": final_attempts[-1]["coverage_status"],
            "correction_attempt": correction_attempt,
            "selection_source": {},
        }
        raise SelectorSelectionError(
            "Final selector vẫn vi phạm coverage contract sau một lần sửa: "
            + str(final_attempts[-1]["error"] or "response không hợp lệ"),
            failure_diagnostics,
        )

    selected_keys = list(final_decision["selected_keys"])
    required_bucket_keys = list(final_decision["required_bucket_keys"])
    bucket_requirements = list(final_decision["bucket_requirements"])
    output_cap = hard_maximum
    selection_source = "llm_correction" if len(final_attempts) == 2 else "llm"
    source_by_key = {key: selection_source for key in selected_keys}

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
            "bge_rank": candidate.get("rerank_rank"),
            "bge_score": candidate.get("rerank_score"),
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
        "selector_input_keys": list(by_key),
        "scouts": scout_diagnostics,
        "scout_nominated_keys": nominated_keys,
        "lexical_finalist_keys": lexical_rescue_keys,
        "finalist_keys": finalist_keys,
        "final_response": _bounded_selector_response(final_response),
        "coverage_locked_bucket_keys": coverage_locked_bucket_keys,
        "coverage_lock_reasons": coverage_lock_reasons,
        "required_bucket_keys": required_bucket_keys,
        "bucket_requirements": bucket_requirements,
        "coverage_status": final_decision["coverage_status"],
        "correction_attempt": correction_attempt,
        "selected_keys": selected_keys,
        "selection_source": source_by_key,
        "output_cap": output_cap,
    }
    logger.info(
        "Table selection completed: input=%d buckets=%d scout_chunks=%s "
        "scout_prompt_chars=%s scout_valid=%s finalists=%d final_prompt_chars=%d "
        "lexical_rescue=%d coverage_locked=%s required_buckets=%s output_cap=%d "
        "final_valid=%d correction_attempted=%s output=%d",
        len(candidates),
        len(available_buckets),
        [len(chunk) for chunk in scout_chunks],
        scout_prompt_chars,
        scout_valid_counts,
        len(finalist_keys),
        final_prompt_chars[-1],
        len(lexical_rescue_keys),
        coverage_locked_bucket_keys,
        required_bucket_keys,
        output_cap,
        len(selected_keys),
        correction_attempt["attempted"],
        len(result),
    )
    logger.debug(
        "Selector scores: %s",
        [(item["table_id"], item["selection_score"]) for item in result],
    )
    return result, diagnostics


def select_tables(
    question: str,
    candidates: Sequence[Mapping[str, Any]],
    *,
    settings: Settings,
) -> list[Candidate]:
    """Select planner evidence with two scouts and one final LLM call."""
    result, _ = select_tables_with_diagnostics(question, candidates, settings=settings)
    return result
