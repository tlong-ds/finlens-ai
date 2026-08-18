from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.retrieval import RerankerError, enrich_candidates, rerank


def payload(index: int) -> dict[str, object]:
    return {
        "table_id": f"VJC_financial_statements_2018_separate_table_{index}",
        "doc_id": "VJC_financial_statements_2018_separate",
        "ticker": "VJC",
        "company_name": "CTCP Hàng không Vietjet",
        "year": 2018,
        "report_type": "separate",
        "table_type": "note_table",
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


def write_manifest(path: Path, count: int) -> None:
    records = [
        {
            "record_type": "header",
            "vector_name": "dense",
            "vector_size": 384,
            "payload_schema_version": 2,
        }
    ]
    records.extend(
        {
            "record_type": "point",
            "table_id": payload(index)["table_id"],
            "payload": payload(index),
            "index_text": f"Loại bảng: thuyết minh\nChỉ tiêu: mục tài chính {index}",
        }
        for index in range(1, count + 1)
    )
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
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
            manifest = Path(temp_dir) / "manifest.jsonl"
            write_manifest(manifest, 50)
            with patch(
                "src.retrieval.generate_structured", side_effect=valid_llm_response
            ) as call:
                result = rerank(
                    "Lãi tiền gửi là bao nhiêu?", candidates(50), manifest_path=manifest
                )
        self.assertEqual(len(result), 10)
        self.assertEqual(call.call_count, 6)  # five batches and one final
        self.assertEqual([item["rerank_rank"] for item in result], list(range(1, 11)))
        self.assertTrue(all("index_text" in item for item in result))

    def test_rerank_retries_invalid_schema_then_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "manifest.jsonl"
            write_manifest(manifest, 2)
            with patch(
                "src.retrieval.generate_structured",
                return_value={"ranked_candidates": []},
            ) as call:
                with self.assertRaisesRegex(RerankerError, "không hợp lệ"):
                    rerank("Câu hỏi", candidates(2), manifest_path=manifest)
        self.assertEqual(call.call_count, 2)

    def test_rejects_stale_manifest_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "manifest.jsonl"
            write_manifest(manifest, 1)
            stale = candidates(1)
            stale[0]["metadata"] = {**stale[0]["metadata"], "year": 2019}
            with self.assertRaisesRegex(RerankerError, "lệch Qdrant"):
                enrich_candidates(stale, manifest_path=manifest)


if __name__ == "__main__":
    unittest.main()
