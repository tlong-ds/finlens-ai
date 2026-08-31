"""Pure metric helpers for retrieval and answer experiments."""

from __future__ import annotations

from collections.abc import Sequence


def ranked_scores(
    predicted: Sequence[str], relevant: Sequence[str]
) -> dict[str, float]:
    predicted_unique = list(dict.fromkeys(predicted))
    relevant_set = set(relevant)
    hits = sum(item in relevant_set for item in predicted_unique)
    precision = hits / len(predicted_unique) if predicted_unique else 0.0
    recall = hits / len(relevant_set) if relevant_set else 0.0
    f2 = (
        5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0
    )
    first = next(
        (
            rank
            for rank, item in enumerate(predicted_unique[:5], 1)
            if item in relevant_set
        ),
        None,
    )
    return {
        "precision": precision,
        "recall": recall,
        "f2": f2,
        "mrr5": 1 / first if first else 0.0,
    }
