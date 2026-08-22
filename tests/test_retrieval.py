from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.prompt import RERANK_SYSTEM_PROMPT
from src.retrieval import (
    RerankerError,
    _dynamic_output_cap,
    attach_rerank_context,
    rerank,
)


TABLE_TYPES = ("balance_sheet", "income_statement", "cash_flow", "note_table")


def payload(
    index: int,
    *,
    doc_index: int = 1,
    table_type: str = "note_table",
) -> dict[str, object]:
    ticker = f"C{doc_index:02d}"
    doc_id = f"{ticker}_financial_statements_2018_separate"
    return {
        "table_id": f"{doc_id}_table_{index}",
        "doc_id": doc_id,
        "ticker": ticker,
        "company_name": f"Công ty {ticker}",
        "year": 2018,
        "report_type": "separate",
        "table_type": table_type,
        "start_line": index,
    }


def candidate(
    index: int,
    *,
    doc_index: int = 1,
    table_type: str = "note_table",
) -> dict[str, object]:
    metadata = payload(index, doc_index=doc_index, table_type=table_type)
    return {
        "table_id": metadata["table_id"],
        "metadata": metadata,
        "retrieval_score": 1 - index / 1000,
        "dense_rank": index,
    }


def candidates(count: int, *, documents: int = 1) -> list[dict[str, object]]:
    return [
        candidate(
            index,
            doc_index=(index - 1) % documents + 1,
            table_type=TABLE_TYPES[(index - 1) % len(TABLE_TYPES)],
        )
        for index in range(1, count + 1)
    ]


def write_csvs(root: Path, items: list[dict[str, object]]) -> None:
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)
    for item in items:
        table_id = str(item["table_id"])
        (data_dir / f"{table_id}.csv").write_text(
            "row_label_raw,2018,note_title\n"
            "Lãi tiền gửi,123,Chi tiết doanh thu tài chính\n"
            "Chi phí lãi vay,45,Chi tiết doanh thu tài chính\n",
            encoding="utf-8",
        )


def valid_llm_response(prompt: str, **_: object) -> dict[str, object]:
    request = json.loads(prompt)
    maximum = request["số_lượng_tối_đa"]
    selected = request.get("candidates")
    if selected is None:
        selected = [
            candidate
            for bucket in request["candidate_buckets"]
            for candidate in bucket["candidates"]
        ]
    selected = selected[:maximum]
    return {
        "ranked_candidate_keys": [item["candidate_key"] for item in selected]
    }


class RerankerTests(unittest.TestCase):
    def test_rerank_prompt_is_coverage_first_and_audits_exact_tables(self) -> None:
        self.assertIn("Ưu tiên coverage trước precision", RERANK_SYSTEM_PROMPT)
        self.assertIn("toán_hạng x required_bucket", RERANK_SYSTEM_PROMPT)
        self.assertIn("Phân biệt stock và flow", RERANK_SYSTEM_PROMPT)
        self.assertIn("giữ cả hai", RERANK_SYSTEM_PROMPT)
        self.assertNotIn("Chọn tập nhỏ nhất", RERANK_SYSTEM_PROMPT)

    def test_fifty_candidates_use_two_scouts_and_one_final_call(self) -> None:
        items = candidates(50)
        requests: list[dict[str, object]] = []

        def capture(prompt: str, **kwargs: object) -> dict[str, object]:
            requests.append(json.loads(prompt))
            return valid_llm_response(prompt, **kwargs)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_csvs(root, items)
            with (
                patch("src.retrieval.PROJECT_ROOT", root),
                patch("src.retrieval.generate_structured", side_effect=capture) as call,
            ):
                result = rerank("Lãi tiền gửi là bao nhiêu?", items)

        self.assertEqual(call.call_count, 3)
        self.assertEqual(
            [len(request["candidates"]) for request in requests[:2]],
            [15, 15],
        )
        self.assertIn("candidate_buckets", requests[2])
        self.assertLessEqual(
            sum(
                len(bucket["candidates"])
                for bucket in requests[2]["candidate_buckets"]
            ),
            16,
        )
        self.assertEqual(len(result), 8)
        self.assertEqual([item["rerank_rank"] for item in result], list(range(1, 9)))

    def test_shortlist_rescues_strong_row_match_beyond_dense_top_thirty(self) -> None:
        items = [candidate(index) for index in range(1, 41)]
        requests: list[dict[str, object]] = []

        def capture(prompt: str, **kwargs: object) -> dict[str, object]:
            requests.append(json.loads(prompt))
            return valid_llm_response(prompt, **kwargs)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_csvs(root, items)
            rescued = root / "data" / f"{items[34]['table_id']}.csv"
            rescued.write_text(
                "row_label_raw,2018,note_title\n"
                "Hàng tồn kho cuối năm,123,Chi tiết hàng tồn kho\n",
                encoding="utf-8",
            )
            with (
                patch("src.retrieval.PROJECT_ROOT", root),
                patch("src.retrieval.generate_structured", side_effect=capture) as call,
            ):
                rerank("Hàng tồn kho cuối năm là bao nhiêu?", items)

        dense_ranks = [
            item["dense_rank"]
            for request in requests[:2]
            for item in request["candidates"]
        ]
        self.assertEqual(call.call_count, 3)
        self.assertEqual(len(dense_ranks), 30)
        self.assertIn(35, dense_ranks)
        self.assertNotIn(30, dense_ranks)
        rescued_card = next(
            item
            for request in requests[:2]
            for item in request["candidates"]
            if item["dense_rank"] == 35
        )
        self.assertEqual(
            rescued_card["context"]["match_summary"]["exact_phrase_rows"][0][
                "label"
            ],
            "Hàng tồn kho cuối năm",
        )

    def test_prompt_uses_opaque_keys_and_structured_layered_context(self) -> None:
        items = candidates(2)
        requests: list[dict[str, object]] = []

        def capture(prompt: str, **kwargs: object) -> dict[str, object]:
            requests.append(json.loads(prompt))
            return valid_llm_response(prompt, **kwargs)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_csvs(root, items)
            with (
                patch("src.retrieval.PROJECT_ROOT", root),
                patch("src.retrieval.generate_structured", side_effect=capture),
            ):
                rerank("Câu hỏi", items)

        final_request = requests[2]
        prompt_candidates = [
            candidate
            for bucket in final_request["candidate_buckets"]
            for candidate in bucket["candidates"]
        ]
        self.assertEqual(
            {item["candidate_key"] for item in prompt_candidates},
            {"c01", "c02"},
        )
        serialized = json.dumps(requests, ensure_ascii=False)
        self.assertNotIn(str(items[0]["table_id"]), serialized)
        self.assertNotIn(str(items[0]["metadata"]["doc_id"]), serialized)
        context = prompt_candidates[0]["context"]
        self.assertIsInstance(context, dict)
        self.assertIn("match_summary", context)
        self.assertEqual(context["columns"], ["row_label_raw", "2018", "note_title"])
        self.assertEqual(
            [row["label"] for row in context["row_catalog"]],
            ["Lãi tiền gửi", "Chi phí lãi vay"],
        )

    def test_partial_response_is_salvaged_without_filling_covered_bucket(self) -> None:
        items = candidates(3)
        response = {
            "ranked_candidate_keys": ["c02", "not-a-key", "c02"],
            "extra_key": "ignored",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_csvs(root, items)
            with (
                patch("src.retrieval.PROJECT_ROOT", root),
                patch("src.retrieval.generate_structured", return_value=response) as call,
            ):
                result = rerank("Câu hỏi", items)

        self.assertEqual(call.call_count, 3)
        self.assertEqual([item["table_id"] for item in result], [items[1]["table_id"]])
        self.assertEqual([item["rerank_source"] for item in result], ["llm"])

    def test_valid_final_response_is_not_blindly_completed(self) -> None:
        items = candidates(6, documents=3)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_csvs(root, items)
            with (
                patch("src.retrieval.PROJECT_ROOT", root),
                patch(
                    "src.retrieval.generate_structured",
                    return_value={"ranked_candidate_keys": ["c01"]},
                ),
            ):
                result = rerank("So sánh ba công ty", items)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["metadata"]["doc_id"], items[0]["metadata"]["doc_id"])
        self.assertEqual(result[0]["rerank_source"], "llm")

    def test_unusable_three_call_pipeline_returns_bucket_anchors(self) -> None:
        items = candidates(6, documents=3)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_csvs(root, items)
            with (
                patch("src.retrieval.PROJECT_ROOT", root),
                patch(
                    "src.retrieval.generate_structured",
                    return_value={"ranked_candidate_keys": []},
                ) as call,
            ):
                result = rerank("Câu hỏi", items)

        self.assertEqual(call.call_count, 3)
        self.assertEqual(len(result), 3)
        self.assertTrue(
            all(item["rerank_source"] == "coverage_completion" for item in result)
        )

    def test_valid_final_output_can_omit_a_bucket_without_local_completion(self) -> None:
        items = candidates(36, documents=9)

        def omit_last_bucket(prompt: str, **_: object) -> dict[str, object]:
            request = json.loads(prompt)
            if "candidate_buckets" not in request:
                return valid_llm_response(prompt)
            keys = [
                candidate["candidate_key"]
                for bucket in request["candidate_buckets"]
                if bucket["bucket_key"] != "b09"
                for candidate in bucket["candidates"]
            ]
            return {
                "ranked_candidate_keys": keys[: request["số_lượng_tối_đa"]]
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_csvs(root, items)
            with (
                patch("src.retrieval.PROJECT_ROOT", root),
                patch("src.retrieval.generate_structured", side_effect=omit_last_bucket),
            ):
                result = rerank("So sánh chín công ty", items)

        self.assertLessEqual(len(result), 18)
        self.assertNotIn(
            "C09_financial_statements_2018_separate",
            {item["metadata"]["doc_id"] for item in result},
        )
        self.assertTrue(all(item["rerank_source"] == "llm" for item in result))

    def test_context_keeps_all_columns_and_unmatched_financial_labels(self) -> None:
        item = candidate(1, table_type="balance_sheet")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            data_dir.mkdir()
            (data_dir / f"{item['table_id']}.csv").write_text(
                "item_code,item_label_raw,item_label_norm,note_ref,period_current,period_prior,unit,entity_type\n"
                "300,C. NỢ PHẢI TRẢ,NỢ PHẢI TRẢ,,100,90,VND,normal\n"
                "400,D. VỐN CHỦ SỞ HỮU,VỐN CHỦ SỞ HỮU,,50,45,VND,normal\n",
                encoding="utf-8",
            )
            with patch("src.retrieval.PROJECT_ROOT", root):
                enriched = attach_rerank_context("Tỷ số D/E là bao nhiêu?", [item])

        context = enriched[0]["rerank_context"]
        self.assertEqual(len(context["columns"]), 8)
        self.assertEqual(
            [row["label"] for row in context["row_catalog"]],
            ["NỢ PHẢI TRẢ", "VỐN CHỦ SỞ HỮU"],
        )
        self.assertEqual([row["code"] for row in context["row_catalog"]], ["300", "400"])
        self.assertEqual(len(context["detailed_rows"]), 2)

    def test_large_context_preserves_catalog_and_adds_seed_neighbours(self) -> None:
        item = candidate(1, table_type="cash_flow")
        rows = [
            f"{index},Dòng {index},Dòng {index},,{index},{index - 1},VND,normal"
            for index in range(1, 13)
        ]
        rows[6] = "7,Lưu chuyển tiền thuần,Lưu chuyển tiền thuần,,7,6,VND,normal"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            data_dir.mkdir()
            (data_dir / f"{item['table_id']}.csv").write_text(
                "item_code,item_label_raw,item_label_norm,note_ref,period_current,period_prior,unit,entity_type\n"
                + "\n".join(rows)
                + "\n",
                encoding="utf-8",
            )
            with patch("src.retrieval.PROJECT_ROOT", root):
                enriched = attach_rerank_context("CFO hay lưu chuyển tiền thuần", [item])

        context = enriched[0]["rerank_context"]
        self.assertEqual(len(context["row_catalog"]), 12)
        detailed_numbers = {row["row"] for row in context["detailed_rows"]}
        self.assertTrue({7, 8, 9}.issubset(detailed_numbers))
        self.assertLessEqual(len(context["detailed_rows"]), 9)

    def test_shortlist_is_deterministic_and_diversifies_table_types(self) -> None:
        items = [
            candidate(index, table_type=TABLE_TYPES[(index - 1) % 4])
            for index in range(1, 41)
        ]
        requests: list[dict[str, object]] = []

        def capture(prompt: str, **kwargs: object) -> dict[str, object]:
            request = json.loads(prompt)
            requests.append(request)
            return valid_llm_response(prompt, **kwargs)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_csvs(root, items)
            with (
                patch("src.retrieval.PROJECT_ROOT", root),
                patch("src.retrieval.generate_structured", side_effect=capture),
            ):
                rerank("Câu hỏi", items)
                rerank("Câu hỏi", items)

        first_chunks = [
            [item["dense_rank"] for item in request["candidates"]]
            for request in requests[:2]
        ]
        second_chunks = [
            [item["dense_rank"] for item in request["candidates"]]
            for request in requests[3:5]
        ]
        self.assertEqual(first_chunks, second_chunks)
        self.assertTrue({1, 2, 3, 4}.issubset({rank for chunk in first_chunks for rank in chunk}))

    def test_dynamic_output_cap(self) -> None:
        self.assertEqual(_dynamic_output_cap(1, 30), 8)
        self.assertEqual(_dynamic_output_cap(3, 30), 8)
        self.assertEqual(_dynamic_output_cap(4, 30), 10)
        self.assertEqual(_dynamic_output_cap(7, 30), 16)
        self.assertEqual(_dynamic_output_cap(9, 30), 18)
        self.assertEqual(_dynamic_output_cap(20, 30), 20)

    def test_rejects_duplicate_qdrant_candidate(self) -> None:
        items = candidates(1)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_csvs(root, items)
            duplicate = [*items, *items]
            with (
                patch("src.retrieval.PROJECT_ROOT", root),
                self.assertRaisesRegex(RerankerError, "bị trùng"),
            ):
                attach_rerank_context("Câu hỏi", duplicate)


if __name__ == "__main__":
    unittest.main()
