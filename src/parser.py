"""LLM query parsing shared by the graph and parser-only evaluation."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from src.helper import concise_error
from src.llm import LLMResponseError, generate_structured
from src.prompt import PARSE_SYSTEM_PROMPT, build_parse_prompt
from src.routing import (
    QueryRoutingError,
    build_ticker_shortlist,
    reconcile_query_filters,
    serialize_ticker_candidates,
)

logger = logging.getLogger(__name__)

_PARSE_RESPONSE_ATTEMPTS = 2


class ParserAttemptsExhausted(QueryRoutingError):
    """A bounded parse failure carrying diagnostics for offline evaluation."""

    def __init__(self, message: str, diagnostics: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


def parse_query_with_diagnostics(
    question: str,
    *,
    question_id: int | str = "unknown",
) -> dict[str, Any]:
    """Parse one question and retain diagnostics without changing graph state."""
    ticker_candidates = build_ticker_shortlist(question)
    candidate_context = serialize_ticker_candidates(ticker_candidates)
    logger.info(
        "question_id=%s ticker shortlist: %s",
        question_id,
        [
            (item["candidate_key"], item["ticker"], item["match_type"])
            for item in candidate_context
        ],
    )
    feedback = ""
    previous_response: Mapping[str, Any] | None = None
    last_error = ""
    attempts: list[dict[str, Any]] = []
    for parse_attempt in range(1, _PARSE_RESPONSE_ATTEMPTS + 1):
        current_response: Mapping[str, Any] | None = None
        attempt_diagnostic: dict[str, Any] = {
            "attempt": parse_attempt,
            "raw_filters": None,
            "validation_error": None,
        }
        try:
            current_response = generate_structured(
                build_parse_prompt(
                    question,
                    candidate_context,
                    feedback=feedback,
                    previous_response=previous_response,
                ),
                system_prompt=PARSE_SYSTEM_PROMPT,
            )
            logger.info(
                "question_id=%s parser_attempt=%d raw_filters=%s",
                question_id,
                parse_attempt,
                current_response,
            )
            attempt_diagnostic["raw_filters"] = dict(current_response)
            filters, semantic_query = reconcile_query_filters(
                question,
                current_response,
                ticker_candidates=ticker_candidates,
            )
            attempts.append(attempt_diagnostic)
            break
        except (LLMResponseError, QueryRoutingError) as error:
            last_error = concise_error(error)
            attempt_diagnostic["validation_error"] = last_error
            if current_response is not None:
                attempt_diagnostic["raw_filters"] = dict(current_response)
            attempts.append(attempt_diagnostic)
            feedback = last_error
            if current_response is not None:
                previous_response = current_response
            logger.info(
                "question_id=%s parser_attempt=%d validation_error=%s",
                question_id,
                parse_attempt,
                last_error,
            )
    else:
        diagnostics = {
            "ticker_candidates": candidate_context,
            "attempts": attempts,
            "semantic_attempts": len(attempts),
        }
        raise ParserAttemptsExhausted(
            "Không parse được metadata filter hợp lệ sau "
            f"{_PARSE_RESPONSE_ATTEMPTS} lần: {last_error}",
            diagnostics,
        )

    logger.info("question_id=%s question=%s", question_id, question)
    logger.info("question_id=%s parsed_filters=%s", question_id, filters)
    logger.info("question_id=%s semantic_query=%s", question_id, semantic_query)
    return {
        "filters": filters,
        "semantic_query": semantic_query,
        "diagnostics": {
            "ticker_candidates": candidate_context,
            "attempts": attempts,
            "semantic_attempts": len(attempts),
        },
    }
