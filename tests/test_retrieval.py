from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.prompt import RERANK_SYSTEM_PROMPT
from src.retrieval import (
    RerankerError,
    _build_match_summary,
    _dynamic_output_cap,
    attach_rerank_context,
    reciprocal_rank_fusion,
    retrieve,
    rerank,
    rerank_with_diagnostics,
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
    selected = request.get("candidates")
    if selected is not None:
        maximum = request["số_lượng_tối_đa"]
        return {
            "ranked_candidate_keys": [
                item["candidate_key"] for item in selected[:maximum]
            ]
        }

    buckets = request["candidate_buckets"]
    requirements = []
    concept_by_bucket = {}
    for bucket in buckets:
        bucket_key = bucket["bucket_key"]
        concept_key = f"{bucket_key}_k01"
        concept_by_bucket[bucket_key] = concept_key
        requirements.append(
            {
                "bucket_key": bucket_key,
                "concepts": [
                    {
                        "concept_key": concept_key,
                        "description": "Chỉ tiêu cần đọc",
                        "role": "direct",
                    }
                ],
            }
        )
    flattened = [candidate for bucket in buckets for candidate in bucket["candidates"]]
    selected = flattened[: min(8, request["giới_hạn_cứng"])]
    return {
        "required_bucket_keys": [bucket["bucket_key"] for bucket in buckets],
        "bucket_requirements": requirements,
        "ranked_selections": [
            {
                "candidate_key": item["candidate_key"],
                "covered_concept_keys": [concept_by_bucket[item["bucket_key"]]],
            }
            for item in selected
        ],
    }


class RerankerTests(unittest.TestCase):
    def test_match_summary_exposes_exact_table_title_separately(self) -> None:
        summary = _build_match_summary(
            "Chi phí khác của SAM năm 2023 là bao nhiêu?",
            {
                "row_catalog": [{"row": 1, "label": "Chi phí khác"}],
                "table_titles": ["CHI PHÍ KHÁC", "Chi phí quản lý doanh nghiệp"],
            },
        )
        self.assertEqual(summary["exact_phrase_titles"], ["CHI PHÍ KHÁC"])
        self.assertEqual(summary["exact_phrase_rows"][0]["label"], "Chi phí khác")

    def test_rerank_prompt_is_coverage_first_and_audits_exact_tables(self) -> None:
        self.assertIn("Ưu tiên coverage trước precision", RERANK_SYSTEM_PROMPT)
        self.assertIn("concept x required bucket", RERANK_SYSTEM_PROMPT)
        self.assertIn("Phân biệt stock và flow", RERANK_SYSTEM_PROMPT)
        self.assertIn("giữ cả hai", RERANK_SYSTEM_PROMPT)
        self.assertIn("required_bucket_keys", RERANK_SYSTEM_PROMPT)
        self.assertIn("BẮT BUỘC giữ note table", RERANK_SYSTEM_PROMPT)
        self.assertIn("nhận ký cược, ký quỹ", RERANK_SYSTEM_PROMPT)
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
        self.assertIn("available_buckets", requests[2])
        self.assertNotIn("required_buckets", requests[2])
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
            "required_bucket_keys": ["b01"],
            "bucket_requirements": [
                {
                    "bucket_key": "b01",
                    "concepts": [
                        {
                            "concept_key": "b01_k01",
                            "description": "Lãi tiền gửi",
                            "role": "direct",
                        }
                    ],
                }
            ],
            "ranked_selections": [
                {"candidate_key": "c02", "covered_concept_keys": ["b01_k01"]},
                {"candidate_key": "not-a-key", "covered_concept_keys": []},
                {"candidate_key": "c02", "covered_concept_keys": ["b01_k01"]},
            ],
            "extra_key": "ignored",
        }

        def scouts_then_partial(prompt: str, **kwargs: object) -> dict[str, object]:
            if "candidates" in json.loads(prompt):
                return valid_llm_response(prompt, **kwargs)
            return response

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_csvs(root, items)
            with (
                patch("src.retrieval.PROJECT_ROOT", root),
                patch(
                    "src.retrieval.generate_structured",
                    side_effect=scouts_then_partial,
                ) as call,
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
                    side_effect=lambda prompt, **kwargs: (
                        valid_llm_response(prompt, **kwargs)
                        if "candidates" in json.loads(prompt)
                        else {
                            "required_bucket_keys": ["b01"],
                            "bucket_requirements": [
                                {
                                    "bucket_key": "b01",
                                    "concepts": [
                                        {
                                            "concept_key": "b01_k01",
                                            "description": "Chỉ tiêu",
                                            "role": "direct",
                                        }
                                    ],
                                }
                            ],
                            "ranked_selections": [
                                {
                                    "candidate_key": "c01",
                                    "covered_concept_keys": ["b01_k01"],
                                }
                            ],
                        }
                    ),
                ),
            ):
                result = rerank("Câu hỏi chỉ cần công ty đầu tiên", items)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["metadata"]["doc_id"], items[0]["metadata"]["doc_id"])
        self.assertEqual(result[0]["rerank_source"], "llm")

    def test_required_bucket_is_completed_from_best_scout_nomination(self) -> None:
        items = candidates(6, documents=2)

        def select_only_first_bucket(prompt: str, **kwargs: object) -> dict[str, object]:
            request = json.loads(prompt)
            if "candidates" in request:
                return valid_llm_response(prompt, **kwargs)
            return {
                "required_bucket_keys": ["b01", "b02"],
                "bucket_requirements": [
                    {
                        "bucket_key": bucket_key,
                        "concepts": [
                            {
                                "concept_key": f"{bucket_key}_k01",
                                "description": "Chỉ tiêu so sánh",
                                "role": "comparison_operand",
                            }
                        ],
                    }
                    for bucket_key in ("b01", "b02")
                ],
                "ranked_selections": [
                    {
                        "candidate_key": "c01",
                        "covered_concept_keys": ["b01_k01"],
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_csvs(root, items)
            with (
                patch("src.retrieval.PROJECT_ROOT", root),
                patch(
                    "src.retrieval.generate_structured",
                    side_effect=select_only_first_bucket,
                ),
            ):
                result, diagnostics = rerank_with_diagnostics("So sánh hai công ty", items)

        self.assertEqual(
            {item["metadata"]["doc_id"] for item in result},
            {items[0]["metadata"]["doc_id"], items[1]["metadata"]["doc_id"]},
        )
        self.assertEqual(
            list(diagnostics["coverage_completion"].values()),
            ["coverage_completion_scout"],
        )

    def test_required_bucket_uses_anchor_only_when_scout_has_no_nominee(self) -> None:
        items = candidates(6, documents=2)

        def omit_second_bucket(prompt: str, **_: object) -> dict[str, object]:
            request = json.loads(prompt)
            if "candidates" in request:
                keys = [
                    item["candidate_key"]
                    for item in request["candidates"]
                    if item["bucket_key"] == "b01"
                ]
                return {"ranked_candidate_keys": keys or []}
            return {
                "required_bucket_keys": ["b01", "b02"],
                "bucket_requirements": [],
                "ranked_selections": [
                    {"candidate_key": "c01", "covered_concept_keys": []}
                ],
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_csvs(root, items)
            with (
                patch("src.retrieval.PROJECT_ROOT", root),
                patch("src.retrieval.generate_structured", side_effect=omit_second_bucket),
            ):
                _, diagnostics = rerank_with_diagnostics("So sánh hai công ty", items)

        self.assertEqual(
            list(diagnostics["coverage_completion"].values()),
            ["locked_bucket_presence"],
        )

    def test_one_bucket_can_select_multiple_tables_for_distinct_concepts(self) -> None:
        items = candidates(3)

        def select_three_concepts(prompt: str, **kwargs: object) -> dict[str, object]:
            request = json.loads(prompt)
            if "candidates" in request:
                return valid_llm_response(prompt, **kwargs)
            concepts = [
                {
                    "concept_key": f"b01_k{index:02d}",
                    "description": description,
                    "role": role,
                }
                for index, (description, role) in enumerate(
                    (
                        ("Tử số", "numerator"),
                        ("Mẫu số", "denominator"),
                        ("Giá trị so sánh", "comparison_operand"),
                    ),
                    start=1,
                )
            ]
            return {
                "required_bucket_keys": ["b01"],
                "bucket_requirements": [
                    {"bucket_key": "b01", "concepts": concepts}
                ],
                "ranked_selections": [
                    {
                        "candidate_key": f"c{index:02d}",
                        "covered_concept_keys": [f"b01_k{index:02d}"],
                    }
                    for index in range(1, 4)
                ],
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_csvs(root, items)
            with (
                patch("src.retrieval.PROJECT_ROOT", root),
                patch("src.retrieval.generate_structured", side_effect=select_three_concepts),
            ):
                result, diagnostics = rerank_with_diagnostics("Tính một tỷ lệ", items)

        self.assertEqual(len(result), 3)
        self.assertEqual(diagnostics["uncovered_concept_keys"], [])

    def test_unusable_final_uses_scout_nominations_without_dense_completion(self) -> None:
        items = candidates(6, documents=3)

        def scouts_then_invalid_final(prompt: str, **kwargs: object) -> dict[str, object]:
            if "candidates" in json.loads(prompt):
                return valid_llm_response(prompt, **kwargs)
            return {}

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_csvs(root, items)
            with (
                patch("src.retrieval.PROJECT_ROOT", root),
                patch(
                    "src.retrieval.generate_structured",
                    side_effect=scouts_then_invalid_final,
                ) as call,
            ):
                result = rerank("Câu hỏi", items)

        self.assertEqual(call.call_count, 3)
        self.assertEqual(len(result), 6)
        self.assertTrue(all(item["rerank_source"] == "scout_fallback" for item in result))

    def test_concepts_without_selections_use_scout_fallback_not_anchor(self) -> None:
        items = candidates(6, documents=2)

        def empty_concept_mapping(prompt: str, **kwargs: object) -> dict[str, object]:
            request = json.loads(prompt)
            if "candidates" in request:
                return valid_llm_response(prompt, **kwargs)
            return {
                "required_bucket_keys": ["b01", "b02"],
                "bucket_requirements": [
                    {
                        "bucket_key": bucket_key,
                        "concepts": [
                            {
                                "concept_key": f"{bucket_key}_k01",
                                "description": "Chỉ tiêu",
                                "role": "direct",
                            }
                        ],
                    }
                    for bucket_key in ("b01", "b02")
                ],
                "ranked_selections": [],
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_csvs(root, items)
            with (
                patch("src.retrieval.PROJECT_ROOT", root),
                patch(
                    "src.retrieval.generate_structured",
                    side_effect=empty_concept_mapping,
                ),
            ):
                result, diagnostics = rerank_with_diagnostics("Câu hỏi", items)

        self.assertIn("không trả ranked_selection", diagnostics["final_error"])
        self.assertTrue(all(item["rerank_source"] == "scout_fallback" for item in result))
        self.assertEqual(diagnostics["coverage_completion"], {})

    def test_exact_lexical_candidate_bypasses_scout_pruning(self) -> None:
        items = [candidate(index) for index in range(1, 41)]

        def omit_exact_in_scout(prompt: str, **_: object) -> dict[str, object]:
            request = json.loads(prompt)
            if "candidates" in request:
                non_exact = [
                    item["candidate_key"]
                    for item in request["candidates"]
                    if not item["context"]["match_summary"]["exact_phrase_rows"]
                ]
                return {"ranked_candidate_keys": non_exact[:1]}
            exact = [
                item
                for bucket in request["candidate_buckets"]
                for item in bucket["candidates"]
                if item["context"]["match_summary"]["exact_phrase_rows"]
            ]
            self.assertEqual(len(exact), 1)
            key = exact[0]["candidate_key"]
            return {
                "required_bucket_keys": ["b01"],
                "bucket_requirements": [
                    {
                        "bucket_key": "b01",
                        "concepts": [
                            {
                                "concept_key": "b01_k01",
                                "description": "Hàng tồn kho cuối năm",
                                "role": "ending_balance",
                            }
                        ],
                    }
                ],
                "ranked_selections": [
                    {"candidate_key": key, "covered_concept_keys": ["b01_k01"]}
                ],
            }

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
                patch(
                    "src.retrieval.generate_structured",
                    side_effect=omit_exact_in_scout,
                ),
            ):
                result, diagnostics = rerank_with_diagnostics(
                    "Hàng tồn kho cuối năm là bao nhiêu?", items
                )

        self.assertEqual(result[0]["table_id"], items[34]["table_id"])
        self.assertTrue(diagnostics["lexical_finalist_keys"])
        self.assertNotIn(
            diagnostics["lexical_finalist_keys"][0],
            diagnostics["scout_nominated_keys"],
        )

    def test_comparison_policy_restores_bucket_omitted_by_final(self) -> None:
        items = candidates(36, documents=9)

        def omit_last_bucket(prompt: str, **_: object) -> dict[str, object]:
            request = json.loads(prompt)
            if "candidate_buckets" not in request:
                return valid_llm_response(prompt)
            selected = [
                candidate
                for bucket in request["candidate_buckets"]
                if bucket["bucket_key"] != "b09"
                for candidate in bucket["candidates"]
            ]
            return {
                "required_bucket_keys": [f"b{index:02d}" for index in range(1, 9)],
                "bucket_requirements": [],
                "ranked_selections": [
                    {"candidate_key": item["candidate_key"], "covered_concept_keys": []}
                    for item in selected[: request["giới_hạn_cứng"]]
                ],
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_csvs(root, items)
            with (
                patch("src.retrieval.PROJECT_ROOT", root),
                patch("src.retrieval.generate_structured", side_effect=omit_last_bucket),
            ):
                result, diagnostics = rerank_with_diagnostics(
                    "So sánh chín công ty", items
                )

        self.assertLessEqual(len(result), 18)
        self.assertIn(
            "C09_financial_statements_2018_separate",
            {item["metadata"]["doc_id"] for item in result},
        )
        self.assertEqual(diagnostics["policy_added_required_bucket_keys"], ["b09"])
        self.assertTrue(
            all(
                item["rerank_source"] in {"llm", "coverage_completion_scout"}
                or item["rerank_source"] == "locked_bucket_presence"
                for item in result
            )
        )

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
        self.assertEqual(_dynamic_output_cap(1, 3, 30), 8)
        self.assertEqual(_dynamic_output_cap(3, 9, 30), 9)
        self.assertEqual(_dynamic_output_cap(7, 14, 30), 14)
        self.assertEqual(_dynamic_output_cap(9, 18, 30), 18)
        self.assertEqual(_dynamic_output_cap(9, 27, 30), 18)

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


class HybridRetrievalTests(unittest.TestCase):
    def test_rrf_rewards_candidates_present_in_both_rankings(self) -> None:
        first = candidate(1)
        second = candidate(2)
        third = candidate(3)
        dense = [
            {**first, "dense_rank": 1, "dense_score": 0.9},
            {**second, "dense_rank": 2, "dense_score": 0.8},
        ]
        bm25 = [
            {
                "table_id": second["table_id"],
                "metadata": second["metadata"],
                "bm25_rank": 1,
                "bm25_score": 12.0,
            },
            {
                "table_id": third["table_id"],
                "metadata": third["metadata"],
                "bm25_rank": 2,
                "bm25_score": 8.0,
            },
        ]

        fused = reciprocal_rank_fusion(dense, bm25, top_n=3, rrf_k=60)

        self.assertEqual(
            [item["table_id"] for item in fused],
            [second["table_id"], first["table_id"], third["table_id"]],
        )
        self.assertEqual([item["retrieval_rank"] for item in fused], [1, 2, 3])
        self.assertAlmostEqual(fused[0]["rrf_score"], 1 / 61 + 1 / 62)

    def test_dense_mode_does_not_call_bm25(self) -> None:
        dense = [candidate(1)]
        with (
            patch("src.retrieval._retrieve_dense", return_value=dense),
            patch("src.retrieval.search_bm25") as bm25_mock,
        ):
            result = retrieve("query", {"ticker": ["C01"]}, top_n=1, mode="dense")

        bm25_mock.assert_not_called()
        self.assertEqual(result[0]["retrieval_mode"], "dense")

    def test_hybrid_mode_calls_both_retrievers(self) -> None:
        dense_item = candidate(1)
        lexical_item = candidate(2)
        lexical = {
            "table_id": lexical_item["table_id"],
            "metadata": lexical_item["metadata"],
            "bm25_rank": 1,
            "bm25_score": 3.0,
        }
        with (
            patch("src.retrieval._retrieve_dense", return_value=[dense_item]) as dense_mock,
            patch("src.retrieval.search_bm25", return_value=[lexical]) as bm25_mock,
        ):
            result = retrieve(
                "query", {"ticker": ["C01"]}, top_n=2, mode="hybrid"
            )

        dense_mock.assert_called_once()
        bm25_mock.assert_called_once()
        self.assertEqual(len(result), 2)
        self.assertTrue(all(item["retrieval_mode"] == "hybrid" for item in result))


if __name__ == "__main__":
    unittest.main()
