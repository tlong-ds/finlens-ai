from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.retrieval import RerankerError, attach_rerank_context, rerank


def payload(index: int) -> dict[str, object]:
    return {
        "table_id": f"VJC_financial_statements_2018_separate_table_{index}",
        "doc_id": "VJC_financial_statements_2018_separate",
        "ticker": "VJC",
        "company_name": "CTCP Hàng không Vietjet",
        "year": 2018,
        "report_type": "separate",
        "table_type": "note_table",
        "start_line": index,
    }


def candidates(count: int) -> list[dict[str, object]]:
    return [
        {
            "table_id": payload(index)["table_id"],
            "metadata": payload(index),
            "retrieval_score": 1 - index / 100,
            "dense_rank": index,
        }
        for index in range(1, count + 1)
    ]


def write_csvs(root: Path, count: int) -> None:
    data_dir = root / "data"
    data_dir.mkdir()
    for index in range(1, count + 1):
        table_id = payload(index)["table_id"]
        (data_dir / f"{table_id}.csv").write_text(
            "Chỉ tiêu,2018\nLãi tiền gửi,123\nChi phí lãi vay,45\n",
            encoding="utf-8",
        )


def valid_llm_response(prompt: str, **_: object) -> dict[str, object]:
    request = json.loads(prompt)
    requested = request["số_lượng_phải_chọn"]
    selected = request["ứng_viên"][:requested]
    return {
        "ranked_candidates": [
            {
                "table_id": item["table_id"],
                "score": 100 - index,
                "reason": "Có chỉ tiêu phù hợp",
            }
            for index, item in enumerate(selected)
        ]
    }


class RerankerTests(unittest.TestCase):
    def test_hierarchical_rerank_uses_batches_and_final(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_csvs(root, 50)
            with (
                patch("src.retrieval.PROJECT_ROOT", root),
                patch(
                    "src.retrieval.generate_structured",
                    side_effect=valid_llm_response,
                ) as call,
            ):
                result = rerank(
                    "Lãi tiền gửi là bao nhiêu?", candidates(50)
                )
        self.assertEqual(len(result), 10)
        self.assertEqual(call.call_count, 6)  # five batches and one final
        self.assertEqual([item["rerank_rank"] for item in result], list(range(1, 11)))
        self.assertTrue(all("rerank_context" in item for item in result))

    def test_rerank_retries_invalid_schema_then_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_csvs(root, 2)
            with (
                patch("src.retrieval.PROJECT_ROOT", root),
                patch(
                    "src.retrieval.generate_structured",
                    return_value={"ranked_candidates": []},
                ) as call,
            ):
                with self.assertRaisesRegex(RerankerError, "không hợp lệ"):
                    rerank("Câu hỏi", candidates(2))
        self.assertEqual(call.call_count, 2)

    def test_attaches_question_aware_context_without_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_csvs(root, 1)
            with patch("src.retrieval.PROJECT_ROOT", root):
                enriched = attach_rerank_context(
                    "Lãi tiền gửi là bao nhiêu?", candidates(1)
                )
        context = json.loads(enriched[0]["rerank_context"])
        self.assertEqual(context["columns"], ["Chỉ tiêu", "2018"])
        self.assertEqual(context["relevant_rows"][0]["cells"][0], "Lãi tiền gửi")

    def test_rejects_duplicate_qdrant_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_csvs(root, 1)
            duplicate = [*candidates(1), *candidates(1)]
            with (
                patch("src.retrieval.PROJECT_ROOT", root),
                self.assertRaisesRegex(RerankerError, "bị trùng"),
            ):
                attach_rerank_context("Câu hỏi", duplicate)


if __name__ == "__main__":
    unittest.main()
