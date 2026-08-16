"""Phase 2 entry point for Level 1 table retrieval."""

from __future__ import annotations

import logging
from typing import Any

from src.graph import graph

logger = logging.getLogger(__name__)


class RetrievalPipelineError(RuntimeError):
    """Raised at the application boundary when the retrieval graph fails."""


def query(question: str) -> list[dict[str, Any]]:
    """Run the Level 1 graph and return its final retrieved tables."""
    if not question.strip():
        raise ValueError("question must not be empty")

    try:
        result = graph.invoke({"question": question.strip()})
    except Exception as exc:
        logger.exception("Table retrieval pipeline failed")
        raise RetrievalPipelineError("Table retrieval pipeline failed") from exc
    return result.get("retrieved_tables", [])
