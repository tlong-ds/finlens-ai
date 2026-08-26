from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.graph import graph
from src.llm import LLMResponseError
from src.nodes import (
    TableContextUnsolvableError,
    _bucket_repair_rerank_depth,
    _bucket_rerank_depth,
    load_tables_node,
    materialize_buckets_node,
    parse_query_node,
    plan_generation_context_node,
    rerank_bucket_tables_node,
    rerank_tables_node,
    retrieve_bucket_tables_node,
    retrieve_tables_node,
    rewrite_bucket_queries_node,
    select_bucket_tables_node,
    select_tables_node,
    validate_table_coverage_node,
)
from src.retrieval import NoMatchingCandidatesError, RetrievalError
from src.routing import QueryRoutingError


QUESTION = (
    "Lãi tiền gửi năm 2018 của công ty mẹ CTCP Hàng không Vietjet (VJC) "
    "là bao nhiêu triệu đồng?"
)


def candidate() -> dict[str, object]:
    metadata = {
        "table_id": "VJC_financial_statements_2018_separate_table_1",
        "doc_id": "VJC_financial_statements_2018_separate",
        "ticker": "VJC",
        "company_name": "CTCP Hàng không Vietjet",
        "year": 2018,
        "report_type": "separate",
        "table_type": "note_table",
        "start_line": 42,
        "index_text": "Lãi tiền gửi VJC 2018",
    }
    return {
        "table_id": metadata["table_id"],
        "metadata": metadata,
        "retrieval_score": 0.9,
        "dense_rank": 1,
        "rerank_score": 0.98,
        "rerank_context": {
            "columns": ["item_label_norm", "period_current"],
            "row_count": 1,
            "table_titles": ["LÃI TIỀN GỬI"],
            "row_catalog": [{"row": 2, "label": "Lãi tiền gửi"}],
            "detailed_rows": [
                {"row": 2, "cells": ["Lãi tiền gửi", "123"]}
            ],
        },
    }


class GraphTests(unittest.TestCase):
    @staticmethod
    def _scoped_candidate(ticker: str, year: int, index: int = 1) -> dict[str, object]:
        item = candidate()
        table_id = f"{ticker}_financial_statements_{year}_separate_table_{index}"
        item["table_id"] = table_id
        item["metadata"] = {
            **dict(item["metadata"]),
            "table_id": table_id,
            "doc_id": f"{ticker}_financial_statements_{year}_separate",
            "ticker": ticker,
            "year": year,
            "start_line": 40 + index,
        }
        return item

    def test_parse_uses_one_llm_request_and_materializes_candidate_key(self) -> None:
        with patch(
            "src.parser.generate_structured",
            return_value={
                "ticker": ["c01"],
                "year": [2018],
                "report_type": ["separate"],
            },
        ) as generate_mock:
            result = parse_query_node({"question": QUESTION})

        generate_mock.assert_called_once()
        self.assertEqual(result["filters"]["ticker"], ["VJC"])
        self.assertEqual(result["filters"]["year"], [2018])
        self.assertEqual(result["filters"]["report_type"], ["separate"])
        prompt = generate_mock.call_args.args[0]
        self.assertIn('"candidate_key": "c01"', prompt)

    def test_parse_repairs_response_that_adds_table_type(self) -> None:
        responses = [
            {
                "ticker": ["c01"],
                "year": [2018],
                "report_type": ["separate"],
                "table_type": "note_table",
            },
            {
                "ticker": ["c01"],
                "year": [2018],
                "report_type": ["separate"],
            },
        ]
        with patch(
            "src.parser.generate_structured",
            side_effect=responses,
        ) as generate_mock:
            result = parse_query_node(
                {"question": QUESTION, "question_record": {"id": 1}}
            )

        self.assertEqual(generate_mock.call_count, 2)
        repair_prompt = generate_mock.call_args_list[1].args[0]
        self.assertIn("table_type", repair_prompt)
        self.assertEqual(result["filters"]["report_type"], ["separate"])
        self.assertNotIn("table_type", result["filters"])

    def test_parse_repairs_invalid_schema_once(self) -> None:
        responses = [
            {"ticker": ["c01"], "year": 2018, "report_type": ["separate"]},
            {
                "ticker": ["c01"],
                "year": [2018],
                "report_type": ["separate"],
            },
        ]
        with patch(
            "src.parser.generate_structured", side_effect=responses
        ) as generate_mock:
            result = parse_query_node(
                {"question": QUESTION, "question_record": {"id": 1}}
            )

        self.assertEqual(generate_mock.call_count, 2)
        repair_prompt = generate_mock.call_args_list[1].args[0]
        self.assertIn('"response_trước"', repair_prompt)
        self.assertIn("year phải là một mảng số nguyên", repair_prompt)
        self.assertEqual(result["filters"]["year"], [2018])

    def test_parse_fails_after_two_invalid_responses(self) -> None:
        invalid = {
            "ticker": ["c01"],
            "year": 2018,
            "report_type": ["separate"],
        }
        with patch(
            "src.parser.generate_structured", return_value=invalid
        ) as generate_mock:
            with self.assertRaisesRegex(
                QueryRoutingError, "sau 2 lần"
            ):
                parse_query_node(
                    {"question": QUESTION, "question_record": {"id": 1}}
                )
        self.assertEqual(generate_mock.call_count, 2)

    def test_materializes_cartesian_buckets_in_parser_order(self) -> None:
        result = materialize_buckets_node(
            {
                "semantic_query": "Doanh thu thuần",
                "filters": {
                    "ticker": ["VJC", "ACB"],
                    "year": [2023, 2024],
                    "report_type": ["consolidated"],
                },
            }
        )
        self.assertEqual(
            [
                (item["bucket_key"], item["ticker"], item["year"])
                for item in result["bucket_specs"]
            ],
            [
                ("b01", "VJC", 2023),
                ("b02", "VJC", 2024),
                ("b03", "ACB", 2023),
                ("b04", "ACB", 2024),
            ],
        )
        self.assertEqual(result["active_bucket_keys"], ["b01", "b02", "b03", "b04"])

    def test_initial_single_bucket_bypasses_query_rewrite_llm(self) -> None:
        state = {
            "question": QUESTION,
            "semantic_query": "Lãi tiền gửi",
            **materialize_buckets_node(
                {
                    "semantic_query": "Lãi tiền gửi",
                    "filters": {
                        "ticker": ["VJC"],
                        "year": [2018],
                        "report_type": ["separate"],
                    },
                }
            ),
        }
        with patch("src.nodes.generate_structured") as generate_mock:
            result = rewrite_bucket_queries_node(state)
        generate_mock.assert_not_called()
        self.assertEqual(result["bucket_states"]["b01"]["query"], "Lãi tiền gửi")
        self.assertEqual(result["bucket_pipeline_metrics"]["rewrite_llm_calls"], 0)

    def test_multi_bucket_rewrites_one_query_per_bucket(self) -> None:
        base = materialize_buckets_node(
            {
                "semantic_query": "Doanh thu thuần",
                "filters": {
                    "ticker": ["VJC", "ACB"],
                    "year": [2024],
                    "report_type": ["consolidated"],
                },
            }
        )

        def rewrite(prompt: str, **_: object) -> dict[str, str]:
            payload = __import__("json").loads(prompt)
            ticker = payload["bucket"]["ticker"]
            return {"search_query": f"Doanh thu thuần {ticker}"}

        with patch("src.nodes.generate_structured", side_effect=rewrite) as generate_mock:
            result = rewrite_bucket_queries_node(
                {"question": "So sánh VJC và ACB", "semantic_query": "Doanh thu thuần", **base}
            )
        self.assertEqual(generate_mock.call_count, 2)
        self.assertEqual(result["bucket_states"]["b01"]["query"], "Doanh thu thuần VJC")
        self.assertEqual(result["bucket_states"]["b02"]["query"], "Doanh thu thuần ACB")

    def test_bucket_retrieval_uses_exact_filters_and_isolates_results(self) -> None:
        base = materialize_buckets_node(
            {
                "semantic_query": "Doanh thu",
                "filters": {
                    "ticker": ["VJC", "ACB"],
                    "year": [2024],
                    "report_type": ["separate"],
                },
            }
        )
        calls: list[dict[str, object]] = []

        def retrieve_one(*, query_text: str, filters: dict[str, object], top_n: int) -> list[dict[str, object]]:
            calls.append({"query_text": query_text, "filters": filters, "top_n": top_n})
            ticker = str(filters["ticker"][0])
            return [self._scoped_candidate(ticker, 2024)]

        with patch("src.nodes.retrieve", side_effect=retrieve_one):
            result = retrieve_bucket_tables_node(base)
        self.assertEqual({call["top_n"] for call in calls}, {40})
        self.assertEqual(
            {tuple(call["filters"]["ticker"]) for call in calls},
            {("VJC",), ("ACB",)},
        )
        self.assertEqual(result["bucket_states"]["b01"]["latest_candidates"][0]["bucket_key"], "b01")
        self.assertEqual(result["bucket_states"]["b02"]["latest_candidates"][0]["bucket_key"], "b02")

    def test_bucket_reranker_uses_configured_depth_and_accumulates_finalists(self) -> None:
        base = materialize_buckets_node(
            {
                "semantic_query": "Doanh thu",
                "filters": {
                    "ticker": ["VJC"],
                    "year": [2024],
                    "report_type": ["separate"],
                },
            }
        )
        old = self._scoped_candidate("VJC", 2024, 1)
        new = self._scoped_candidate("VJC", 2024, 2)
        base["bucket_states"]["b01"].update(
            {"latest_candidates": [new], "finalists": [old], "query": "Doanh thu VJC"}
        )
        with (
            patch.dict("os.environ", {"FINLENS_BUCKET_RERANK_TOP_N": "5"}),
            patch("src.nodes.rerank_with_fpt", return_value=[new]) as rerank_mock,
        ):
            result = rerank_bucket_tables_node(base)
        self.assertEqual(rerank_mock.call_args.kwargs["top_n"], 5)
        self.assertEqual(
            [item["table_id"] for item in result["bucket_states"]["b01"]["finalists"]],
            [new["table_id"], old["table_id"]],
        )

    def test_bucket_reranker_uses_top_ten_initially_and_top_twenty_for_repair(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(_bucket_rerank_depth(), 10)
            self.assertEqual(_bucket_repair_rerank_depth(), 20)
        with patch.dict("os.environ", {"FINLENS_BUCKET_RERANK_TOP_N": "15"}):
            self.assertEqual(_bucket_rerank_depth(), 15)

        base = materialize_buckets_node(
            {
                "semantic_query": "Doanh thu",
                "filters": {
                    "ticker": ["VJC"],
                    "year": [2024],
                    "report_type": ["separate"],
                },
            }
        )
        item = self._scoped_candidate("VJC", 2024)
        base["retrieval_repair_round"] = 1
        base["bucket_states"]["b01"].update(
            {"latest_candidates": [item], "query": "Doanh thu repair"}
        )
        with patch("src.nodes.rerank_with_fpt", return_value=[item]) as rerank:
            rerank_bucket_tables_node(base)
        self.assertEqual(rerank.call_args.kwargs["top_n"], 20)

    def test_selector_runs_exactly_one_call_for_each_active_bucket(self) -> None:
        base = materialize_buckets_node(
            {
                "semantic_query": "Doanh thu",
                "filters": {
                    "ticker": ["VJC", "ACB"],
                    "year": [2024],
                    "report_type": ["separate"],
                },
            }
        )
        for key, ticker in (("b01", "VJC"), ("b02", "ACB")):
            base["bucket_states"][key]["finalists"] = [
                self._scoped_candidate(ticker, 2024)
            ]

        def choose(_: str, bucket: dict[str, object], candidates: list[dict[str, object]]):
            return [candidates[0]], {"bucket_key": bucket["bucket_key"], "concepts": [], "uncovered_concept_keys": []}

        with patch(
            "src.nodes.select_bucket_tables_with_diagnostics", side_effect=choose
        ) as selector_mock:
            result = select_bucket_tables_node({"question": "So sánh", **base})
        self.assertEqual(selector_mock.call_count, 2)
        self.assertEqual(result["bucket_pipeline_metrics"]["selector_llm_calls"], 2)

    def test_validator_locks_sufficient_bucket_and_repairs_only_missing_bucket(self) -> None:
        base = materialize_buckets_node(
            {
                "semantic_query": "Doanh thu",
                "filters": {
                    "ticker": ["VJC", "ACB"],
                    "year": [2024],
                    "report_type": ["separate"],
                },
            }
        )
        for key, ticker in (("b01", "VJC"), ("b02", "ACB")):
            base["bucket_states"][key].update(
                {
                    "selected_tables": [self._scoped_candidate(ticker, 2024)],
                    "selector_diagnostics": {"concepts": [], "uncovered_concept_keys": []},
                }
            )
        response = {
            "answerable": False,
            "bucket_statuses": [
                {"bucket_key": "b01", "sufficient": True, "reason": "đủ"},
                {"bucket_key": "b02", "sufficient": False, "reason": "thiếu doanh thu"},
            ],
            "missing_requirements": [
                {
                    "bucket_key": "b02",
                    "concept": "Doanh thu thuần",
                    "role": "comparison_operand",
                    "reason": "không có row",
                    "suggested_query": "doanh thu thuần ACB",
                }
            ],
            "target_bucket_keys": ["b02"],
            "feedback": "Bổ sung doanh thu ACB",
        }
        with patch("src.nodes.generate_structured", return_value=response):
            command = validate_table_coverage_node({"question": "So sánh", **base})
        self.assertEqual(command.goto, "rewrite_bucket_queries")
        self.assertEqual(command.update["active_bucket_keys"], ["b02"])
        self.assertEqual(command.update["bucket_states"]["b01"]["status"], "locked")
        self.assertEqual(command.update["bucket_states"]["b02"]["status"], "needs_repair")
        self.assertEqual(command.update["retrieval_repair_round"], 1)

    def test_semantic_validator_becomes_advisory_after_two_repair_rounds(self) -> None:
        base = materialize_buckets_node(
            {
                "semantic_query": "Doanh thu",
                "filters": {
                    "ticker": ["VJC"],
                    "year": [2024],
                    "report_type": ["separate"],
                },
            }
        )
        base["retrieval_repair_round"] = 2
        base["bucket_states"]["b01"].update(
            {
                "selected_tables": [self._scoped_candidate("VJC", 2024)],
                "selector_diagnostics": {"concepts": [], "uncovered_concept_keys": []},
            }
        )
        response = {
            "answerable": False,
            "bucket_statuses": [
                {"bucket_key": "b01", "sufficient": False, "reason": "thiếu"}
            ],
            "missing_requirements": [],
            "target_bucket_keys": ["b01"],
            "feedback": "Thiếu bảng",
        }
        with patch("src.nodes.generate_structured", return_value=response):
            command = validate_table_coverage_node({"question": "Câu hỏi", **base})
        self.assertEqual(command.goto, "load_tables")
        self.assertEqual(
            command.update["coverage_validation"]["source"],
            "llm_exhausted_advisory",
        )
        self.assertTrue(command.update["coverage_validation"]["advisory"])

    def test_semantic_validator_strict_mode_still_fails_closed(self) -> None:
        base = materialize_buckets_node(
            {
                "semantic_query": "Doanh thu",
                "filters": {
                    "ticker": ["VJC"],
                    "year": [2024],
                    "report_type": ["separate"],
                },
            }
        )
        base["retrieval_repair_round"] = 2
        base["bucket_states"]["b01"]["selected_tables"] = [
            self._scoped_candidate("VJC", 2024)
        ]
        response = {
            "answerable": False,
            "bucket_statuses": [],
            "missing_requirements": [],
            "target_bucket_keys": ["b01"],
            "feedback": "Thiếu bảng",
        }
        with (
            patch.dict(
                "os.environ",
                {"FINLENS_FAIL_CLOSED_SEMANTIC_VALIDATOR": "1"},
            ),
            patch("src.nodes.generate_structured", return_value=response),
        ):
            with self.assertRaisesRegex(TableContextUnsolvableError, "sau 2"):
                validate_table_coverage_node({"question": "Câu hỏi", **base})

    def test_deterministic_missing_bucket_still_fails_closed(self) -> None:
        base = materialize_buckets_node(
            {
                "semantic_query": "Doanh thu",
                "filters": {
                    "ticker": ["VJC"],
                    "year": [2024],
                    "report_type": ["separate"],
                },
            }
        )
        base["retrieval_repair_round"] = 2
        with self.assertRaisesRegex(TableContextUnsolvableError, "sau 2"):
            validate_table_coverage_node({"question": "Câu hỏi", **base})

    def test_validator_can_disable_repairs_for_ab_benchmark(self) -> None:
        base = materialize_buckets_node(
            {
                "semantic_query": "Doanh thu",
                "filters": {
                    "ticker": ["VJC"],
                    "year": [2024],
                    "report_type": ["separate"],
                },
            }
        )
        base["bucket_states"]["b01"].update(
            {
                "selected_tables": [self._scoped_candidate("VJC", 2024)],
                "selector_diagnostics": {"concepts": [], "uncovered_concept_keys": []},
            }
        )
        response = {
            "answerable": False,
            "bucket_statuses": [],
            "missing_requirements": [],
            "target_bucket_keys": ["b01"],
        }
        with (
            patch.dict(
                "os.environ",
                {
                    "FINLENS_MAX_RETRIEVAL_REPAIRS": "0",
                    "FINLENS_FAIL_CLOSED_SEMANTIC_VALIDATOR": "1",
                },
            ),
            patch("src.nodes.generate_structured", return_value=response),
        ):
            with self.assertRaisesRegex(TableContextUnsolvableError, "sau 0"):
                validate_table_coverage_node({"question": "Câu hỏi", **base})

    def test_validator_off_ab_mode_routes_directly_to_load(self) -> None:
        base = materialize_buckets_node(
            {
                "semantic_query": "Doanh thu",
                "filters": {
                    "ticker": ["VJC"],
                    "year": [2024],
                    "report_type": ["separate"],
                },
            }
        )
        base["bucket_states"]["b01"].update(
            {
                "selected_tables": [self._scoped_candidate("VJC", 2024)],
                "selector_diagnostics": {"concepts": [], "uncovered_concept_keys": []},
            }
        )
        with (
            patch.dict("os.environ", {"FINLENS_DISABLE_COVERAGE_VALIDATOR": "1"}),
            patch("src.nodes.generate_structured") as generate_mock,
        ):
            command = validate_table_coverage_node({"question": "Câu hỏi", **base})
        generate_mock.assert_not_called()
        self.assertEqual(command.goto, "load_tables")
        self.assertEqual(command.update["coverage_validation"]["source"], "disabled")
        self.assertEqual(command.update["retrieved_tables"][0]["metadata"]["ticker"], "VJC")

    def test_planner_receives_semantic_validator_advisory_without_hard_coding_concepts(self) -> None:
        prompts: list[str] = []

        def plan(prompt: str, **_: object) -> dict[str, object]:
            prompts.append(prompt)
            return {
                "evidence": [
                    {
                        "alias": "df_1",
                        "rows": [
                            {
                                "row_position": 0,
                                "columns": ["period_current"],
                                "purpose": "Audit primitive operand",
                            }
                        ],
                    }
                ],
                "calculation": "Audit primitive operands",
                "unit_conversion": "None",
                "audit": "Complete",
            }

        state = {
            "question": "Tính một tỷ lệ tài chính",
            "dataframes": {"df_1": object()},
            "alias_metadata": {"df_1": {}},
            "rerank_contexts": {"df_1": {}},
            "coverage_validation": {
                "source": "llm_exhausted_advisory",
                "missing_requirements": [
                    {"concept": "mẫu số chưa chắc chắn"}
                ],
            },
        }
        inventory = [
            {
                "alias": "df_1",
                "columns": [{"name": "period_current"}],
                "row_catalog": [{"row_position": 0}],
            }
        ]
        with (
            patch("src.nodes.build_planning_inventory", return_value=inventory),
            patch("src.nodes.hydrate_planned_rows", return_value={}),
            patch("src.nodes.generate_structured", side_effect=plan),
        ):
            command = plan_generation_context_node(state)

        self.assertEqual(command.goto, "generate_code")
        request = prompts[0]
        self.assertIn("primitive operand", request)
        self.assertIn("mẫu số chưa chắc chắn", request)

    def test_planner_retries_invalid_contract_and_incomplete_year_scope(self) -> None:
        inventory = [
            {
                "alias": f"df_{index}",
                "columns": [{"name": "period_current"}],
                "row_catalog": [{"row_position": 0}],
            }
            for index in range(1, 4)
        ]

        def valid_plan(aliases: list[str]) -> dict[str, object]:
            return {
                "evidence": [
                    {
                        "alias": alias,
                        "rows": [
                            {
                                "row_position": 0,
                                "columns": ["period_current"],
                                "purpose": "Vốn chủ sở hữu và tổng tài sản",
                            }
                        ],
                    }
                    for alias in aliases
                ],
                "calculation": "Tính từng tỷ trọng rồi lấy trung bình",
                "unit_conversion": "Nhân 100",
                "audit": "Đủ ba năm",
            }

        responses = [
            {"analysis": "not a planner contract"},
            valid_plan(["df_1"]),
            valid_plan(["df_1", "df_2", "df_3"]),
        ]
        prompts: list[str] = []

        def generate(prompt: str, **_: object) -> dict[str, object]:
            prompts.append(prompt)
            return responses[len(prompts) - 1]

        state = {
            "question": "Tỷ trọng trung bình các năm 2015, 2018 và 2024 là bao nhiêu?",
            "dataframes": {f"df_{index}": object() for index in range(1, 4)},
            "alias_metadata": {
                "df_1": {"year": 2015},
                "df_2": {"year": 2018},
                "df_3": {"year": 2024},
            },
            "rerank_contexts": {},
        }
        with (
            patch("src.nodes.build_planning_inventory", return_value=inventory),
            patch("src.nodes.hydrate_planned_rows", return_value={}),
            patch("src.nodes.generate_structured", side_effect=generate),
        ):
            command = plan_generation_context_node(state)

        self.assertEqual(command.goto, "generate_code")
        self.assertEqual(len(prompts), 3)
        self.assertIn("contain evidence", prompts[1])
        self.assertIn("skips explicitly requested years", prompts[2])

    def test_planner_contract_failure_falls_back_to_inventory_advisory(self) -> None:
        inventory = [
            {
                "alias": "df_1",
                "columns": [{"name": "period_current"}],
                "row_catalog": [{"row_position": 0}],
                "detailed_rows": [],
            }
        ]
        state = {
            "question": "Doanh thu là bao nhiêu?",
            "dataframes": {"df_1": pd.DataFrame({"period_current": [123]})},
            "alias_metadata": {
                "df_1": {"year": 2024, "table_id": "table_1"}
            },
            "rerank_contexts": {},
            "coverage_validation": {
                "source": "llm_unproven_advisory",
                "coverage_proofs": [
                    {
                        "table_id": "table_1",
                        "row": 2,
                        "columns": ["period_current"],
                        "operand": "Doanh thu",
                        "derivation": "direct",
                    }
                ],
                "missing_requirements": [],
            },
        }
        with (
            patch("src.nodes.build_planning_inventory", return_value=inventory),
            patch("src.nodes.hydrate_planned_rows", return_value={}),
            patch(
                "src.nodes.generate_structured",
                return_value={"analysis": "not a planner contract"},
            ) as planner,
        ):
            command = plan_generation_context_node(state)

        self.assertEqual(planner.call_count, 3)
        self.assertEqual(command.goto, "generate_code")
        plan = command.update["generation_plan"]
        self.assertEqual(plan["evidence"][0]["alias"], "df_1")
        self.assertEqual(plan["evidence"][0]["rows"][0]["row_position"], 0)
        self.assertIn("Planner contract unavailable", plan["calculation"])

    def test_validator_salvages_partial_bucket_statuses_without_losing_target(self) -> None:
        base = materialize_buckets_node(
            {
                "semantic_query": "So sánh doanh thu",
                "filters": {
                    "ticker": ["VJC", "ACB"],
                    "year": [2024],
                    "report_type": ["separate"],
                },
            }
        )
        for key, ticker in (("b01", "VJC"), ("b02", "ACB")):
            base["bucket_states"][key].update(
                {
                    "selected_tables": [self._scoped_candidate(ticker, 2024)],
                    "selector_diagnostics": {
                        "concepts": [],
                        "uncovered_concept_keys": [],
                    },
                }
            )
        response = {
            "answerable": False,
            "bucket_statuses": [
                {"bucket_key": "b02", "sufficient": False}
            ],
            "missing_requirements": [],
            "target_bucket_keys": ["b02"],
        }
        with patch("src.nodes.generate_structured", return_value=response):
            command = validate_table_coverage_node({"question": "Câu hỏi", **base})
        self.assertEqual(command.update["active_bucket_keys"], ["b02"])
        statuses = command.update["coverage_validation"]["bucket_statuses"]
        self.assertEqual([item["bucket_key"] for item in statuses], ["b01", "b02"])
        self.assertTrue(statuses[0]["sufficient"])
        self.assertFalse(statuses[1]["sufficient"])

    def test_uncovered_selector_concept_is_left_to_global_semantic_audit(self) -> None:
        base = materialize_buckets_node(
            {
                "semantic_query": "Biên lợi nhuận gộp",
                "filters": {
                    "ticker": ["NKG"],
                    "year": [2024],
                    "report_type": ["consolidated"],
                },
            }
        )
        base["bucket_states"]["b01"].update(
            {
                "selected_tables": [self._scoped_candidate("NKG", 2024)],
                "selector_diagnostics": {
                    "concepts": [],
                    "uncovered_concept_keys": ["k02"],
                },
            }
        )
        response = {
            "answerable": True,
            "bucket_statuses": [
                {
                    "bucket_key": "b01",
                    "sufficient": True,
                    "reason": "đủ",
                    "required_operands": ["Biên lợi nhuận gộp"],
                }
            ],
            "coverage_proofs": [
                {
                    "bucket_key": "b01",
                    "operand": "Biên lợi nhuận gộp",
                    "table_id": "NKG_financial_statements_2024_separate_table_1",
                    "row": 2,
                    "columns": ["period_current"],
                    "derivation": "direct",
                }
            ],
            "missing_requirements": [],
            "target_bucket_keys": [],
            "feedback": "",
        }
        with patch("src.nodes.generate_structured", return_value=response) as audit:
            command = validate_table_coverage_node({"question": "Câu hỏi", **base})
        audit.assert_called_once()
        self.assertEqual(command.goto, "load_tables")

    def test_validator_routes_unverifiable_positive_proof_as_planner_advisory(self) -> None:
        base = materialize_buckets_node(
            {
                "semantic_query": "Biên lợi nhuận gộp",
                "filters": {
                    "ticker": ["NKG"],
                    "year": [2024],
                    "report_type": ["consolidated"],
                },
            }
        )
        table = self._scoped_candidate("NKG", 2024)
        base["bucket_states"]["b01"]["selected_tables"] = [table]
        response = {
            "answerable": True,
            "bucket_statuses": [
                {
                    "bucket_key": "b01",
                    "sufficient": True,
                    "reason": "đủ",
                    "required_operands": ["Lợi nhuận gộp"],
                }
            ],
            "coverage_proofs": [
                {
                    "bucket_key": "b01",
                    "operand": "Lợi nhuận gộp",
                    "table_id": table["table_id"],
                    "row": 999,
                    "columns": ["period_current"],
                    "derivation": "direct",
                }
            ],
            "missing_requirements": [],
            "target_bucket_keys": [],
            "feedback": "",
        }
        with patch("src.nodes.generate_structured", return_value=response):
            command = validate_table_coverage_node({"question": "Câu hỏi", **base})

        self.assertEqual(command.goto, "load_tables")
        decision = command.update["coverage_validation"]
        self.assertEqual(decision["source"], "llm_unproven_advisory")
        self.assertTrue(decision["advisory"])
        self.assertFalse(decision["verified_answerable"])
        self.assertEqual(decision["target_bucket_keys"], ["b01"])
        self.assertEqual(decision["proof_errors"][0]["concept"], "Lợi nhuận gộp")

    def test_validator_retries_malformed_json_once_on_same_inventory(self) -> None:
        base = materialize_buckets_node(
            {
                "semantic_query": "Doanh thu",
                "filters": {
                    "ticker": ["VJC"],
                    "year": [2024],
                    "report_type": ["separate"],
                },
            }
        )
        table = self._scoped_candidate("VJC", 2024)
        base["bucket_states"]["b01"]["selected_tables"] = [table]
        valid_response = {
            "answerable": True,
            "bucket_statuses": [
                {
                    "bucket_key": "b01",
                    "sufficient": True,
                    "reason": "đủ",
                    "required_operands": ["Doanh thu"],
                }
            ],
            "coverage_proofs": [
                {
                    "bucket_key": "b01",
                    "operand": "Doanh thu",
                    "table_id": table["table_id"],
                    "row": 2,
                    "columns": ["period_current"],
                    "derivation": "direct",
                }
            ],
            "missing_requirements": [],
            "target_bucket_keys": [],
            "feedback": "",
        }
        with patch(
            "src.nodes.generate_structured",
            side_effect=[LLMResponseError("invalid JSON"), valid_response],
        ) as audit:
            command = validate_table_coverage_node({"question": "Câu hỏi", **base})

        self.assertEqual(command.goto, "load_tables")
        self.assertEqual(audit.call_count, 2)
        self.assertEqual(
            command.update["bucket_pipeline_metrics"]["validator_llm_calls"], 2
        )

    def test_unverifiable_positive_proof_does_not_consume_repair_budget(self) -> None:
        base = materialize_buckets_node(
            {
                "semantic_query": "Doanh thu",
                "filters": {
                    "ticker": ["VJC"],
                    "year": [2024],
                    "report_type": ["separate"],
                },
            }
        )
        table = self._scoped_candidate("VJC", 2024)
        base["retrieval_repair_round"] = 2
        base["bucket_states"]["b01"]["selected_tables"] = [table]
        response = {
            "answerable": True,
            "bucket_statuses": [
                {
                    "bucket_key": "b01",
                    "sufficient": True,
                    "reason": "đủ",
                    "required_operands": ["Doanh thu"],
                }
            ],
            "coverage_proofs": [],
            "missing_requirements": [],
            "target_bucket_keys": [],
            "feedback": "",
        }
        with patch("src.nodes.generate_structured", return_value=response):
            command = validate_table_coverage_node({"question": "Câu hỏi", **base})
        self.assertEqual(command.goto, "load_tables")
        self.assertEqual(
            command.update["coverage_validation"]["source"],
            "llm_unproven_advisory",
        )

    def test_retrieve_falls_back_to_all_report_types_on_no_match(self) -> None:
        calls: list[dict[str, object]] = []

        def retrieve_side_effect(
            *, query_text: str, filters: dict[str, object], top_n: int
        ) -> list[dict[str, object]]:
            calls.append(
                {"query_text": query_text, "filters": filters, "top_n": top_n}
            )
            if len(calls) == 1:
                raise NoMatchingCandidatesError("no match")
            return [candidate()]

        initial_filters = {
            "ticker": ["VJC"],
            "year": [2018],
            "report_type": ["consolidated"],
            "table_type": ["note_table"],
        }
        with patch("src.nodes.retrieve", side_effect=retrieve_side_effect):
            result = retrieve_tables_node(
                {
                    "semantic_query": "Lãi tiền gửi",
                    "filters": initial_filters,
                }
            )

        self.assertEqual(result["candidates"], [candidate()])
        self.assertEqual(calls[0]["filters"], initial_filters)
        self.assertEqual(
            calls[1]["filters"],
            {
                "ticker": ["VJC"],
                "year": [2018],
                "table_type": ["note_table"],
            },
        )
        self.assertEqual(result["filters"], calls[1]["filters"])

    def test_retrieve_materializes_balanced_bucket_per_ticker(self) -> None:
        calls: list[dict[str, object]] = []

        def ticker_candidate(ticker: str, index: int) -> dict[str, object]:
            item = candidate()
            table_id = f"{ticker}_financial_statements_2018_separate_table_{index}"
            item["table_id"] = table_id
            item["metadata"] = {
                **dict(item["metadata"]),
                "table_id": table_id,
                "doc_id": f"{ticker}_financial_statements_2018_separate",
                "ticker": ticker,
            }
            return item

        def retrieve_side_effect(
            *, query_text: str, filters: dict[str, object], top_n: int
        ) -> list[dict[str, object]]:
            calls.append(
                {"query_text": query_text, "filters": filters, "top_n": top_n}
            )
            ticker = str(filters["ticker"][0])
            return [ticker_candidate(ticker, 1), ticker_candidate(ticker, 2)]

        with patch("src.nodes.retrieve", side_effect=retrieve_side_effect):
            result = retrieve_tables_node(
                {
                    "semantic_query": "Doanh thu thuần",
                    "filters": {
                        "ticker": ["VJC", "ACB"],
                        "year": [2018],
                        "report_type": ["consolidated"],
                    },
                }
            )

        self.assertEqual(
            [call["filters"]["ticker"] for call in calls], [["VJC"], ["ACB"]]
        )
        self.assertEqual([call["top_n"] for call in calls], [40, 40])
        self.assertEqual(
            [item["metadata"]["ticker"] for item in result["candidates"]],
            ["VJC", "ACB", "VJC", "ACB"],
        )
        self.assertEqual(
            [item["dense_rank"] for item in result["candidates"]], [1, 2, 3, 4]
        )

    def test_retrieve_does_not_fallback_without_report_type_filter(self) -> None:
        error = NoMatchingCandidatesError("no match")
        with patch("src.nodes.retrieve", side_effect=error) as retrieve_mock:
            with self.assertRaises(NoMatchingCandidatesError):
                retrieve_tables_node(
                    {
                        "semantic_query": "Lãi tiền gửi",
                        "filters": {"ticker": ["VJC"], "year": [2018]},
                    }
                )
        retrieve_mock.assert_called_once()

    def test_retrieve_does_not_fallback_on_other_retrieval_errors(self) -> None:
        error = RetrievalError("Qdrant query failed")
        with patch("src.nodes.retrieve", side_effect=error) as retrieve_mock:
            with self.assertRaises(RetrievalError):
                retrieve_tables_node(
                    {
                        "semantic_query": "Lãi tiền gửi",
                        "filters": {
                            "ticker": ["VJC"],
                            "year": [2018],
                            "report_type": ["consolidated"],
                        },
                    }
                )
        retrieve_mock.assert_called_once()

    def test_retrieve_propagates_no_match_when_fallback_is_empty(self) -> None:
        error = NoMatchingCandidatesError("no match")
        with patch("src.nodes.retrieve", side_effect=[error, error]) as retrieve_mock:
            with self.assertRaises(NoMatchingCandidatesError):
                retrieve_tables_node(
                    {
                        "semantic_query": "Lãi tiền gửi",
                        "filters": {
                            "ticker": ["VJC"],
                            "year": [2018],
                            "report_type": ["consolidated"],
                        },
                    }
                )
        self.assertEqual(retrieve_mock.call_count, 2)

    def test_retrieve_does_not_retry_when_first_query_has_candidates(self) -> None:
        with patch("src.nodes.retrieve", return_value=[candidate()]) as retrieve_mock:
            result = retrieve_tables_node(
                {
                    "semantic_query": "Lãi tiền gửi",
                    "filters": {
                        "ticker": ["VJC"],
                        "year": [2018],
                        "report_type": ["consolidated"],
                    },
                }
            )
        self.assertEqual(result["candidates"], [candidate()])
        retrieve_mock.assert_called_once()

    def test_load_tables_derives_csv_path_from_table_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            data_dir.mkdir()
            table_id = candidate()["table_id"]
            (data_dir / f"{table_id}.csv").write_text(
                "Chỉ tiêu,2018\nLãi tiền gửi,123\n", encoding="utf-8"
            )
            with patch("src.nodes._PROJECT_ROOT", root):
                result = load_tables_node({"retrieved_tables": [candidate()]})
        self.assertEqual(
            result["evidence_sources"]["df_1"]["csv_path"], f"data/{table_id}.csv"
        )
        self.assertEqual(
            result["evidence_sources"]["df_1"]["relevant_table"],
            "VJC_financial_statements_2018_separate|42",
        )
        self.assertEqual(result["alias_metadata"]["df_1"]["table_id"], table_id)

    def test_fpt_reranker_and_selector_are_separate_nodes(self) -> None:
        item = candidate()
        with patch("src.nodes.rerank_with_fpt", return_value=[item]) as fpt_mock:
            reranked = rerank_tables_node(
                {"question": QUESTION, "candidates": [item]}
            )
        with patch("src.nodes.select_tables", return_value=[item]) as selector_mock:
            selected = select_tables_node(
                {"question": QUESTION, "reranked_tables": reranked["reranked_tables"]}
            )
        self.assertEqual(reranked, {"reranked_tables": [item]})
        self.assertEqual(selected, {"retrieved_tables": [item]})
        fpt_mock.assert_called_once()
        selector_mock.assert_called_once()

    def test_full_graph_returns_numeric_answer_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            data_dir.mkdir()
            table_id = candidate()["table_id"]
            (data_dir / f"{table_id}.csv").write_text(
                "Chỉ tiêu,2018\nLãi tiền gửi,123\n", encoding="utf-8"
            )

            def structured_response(
                _: str, *, system_prompt: str | None = None, **__: object
            ) -> dict[str, object]:
                if system_prompt and "bộ định tuyến" in system_prompt:
                    return {
                        "ticker": ["c01"],
                        "year": [2018],
                        "report_type": ["separate"],
                    }
                if system_prompt and "validator độc lập" in system_prompt:
                    return {
                        "answerable": True,
                        "bucket_statuses": [
                            {
                                "bucket_key": "b01",
                                "sufficient": True,
                                "reason": "đủ lãi tiền gửi",
                                "required_operands": ["Lãi tiền gửi"],
                            }
                        ],
                        "coverage_proofs": [
                            {
                                "bucket_key": "b01",
                                "operand": "Lãi tiền gửi",
                                "table_id": table_id,
                                "row": 2,
                                "columns": ["period_current"],
                                "derivation": "direct",
                            }
                        ],
                        "missing_requirements": [],
                        "target_bucket_keys": [],
                        "feedback": "",
                    }
                if system_prompt and "lập kế hoạch bằng chứng" in system_prompt:
                    return {
                        "evidence": [
                            {
                                "alias": "df_1",
                                "rows": [
                                    {
                                        "row_position": 0,
                                        "columns": ["2018"],
                                        "purpose": "Lãi tiền gửi",
                                    }
                                ],
                            }
                        ],
                        "calculation": "Đọc lãi tiền gửi",
                        "unit_conversion": "Không đổi",
                        "audit": "Đủ",
                    }
                return {
                    "pandas_query": "result = float(df_1.loc[0, '2018'])",
                    "evidence_variables": ["df_1"],
                }

            def bucket_selector_response(
                _: str, **__: object
            ) -> dict[str, object]:
                return {
                    "concepts": [
                        {
                            "concept_key": "k01",
                            "description": "Lãi tiền gửi",
                            "role": "direct",
                        }
                    ],
                    "ranked_selections": [
                        {
                            "candidate_key": "c01",
                            "covered_concept_keys": ["k01"],
                        }
                    ],
                }

            with (
                patch("src.nodes._PROJECT_ROOT", root),
                patch("src.parser.generate_structured", side_effect=structured_response),
                patch("src.nodes.generate_structured", side_effect=structured_response),
                patch(
                    "src.retrieval.generate_structured",
                    side_effect=bucket_selector_response,
                ),
                patch("src.nodes.retrieve", return_value=[candidate()]),
                patch("src.nodes.rerank_with_fpt", return_value=[candidate()]),
                patch("src.nodes.run_code", return_value=123.0),
            ):
                result = graph.invoke({"question": QUESTION, "max_attempts": 2})

        answer = result["answer_record"]
        self.assertEqual(answer["id"], 1)
        self.assertEqual(answer["answer"], 123.0)
        self.assertEqual(
            answer["relevant_docs"], ["VJC_financial_statements_2018_separate"]
        )
        self.assertEqual(
            answer["relevant_tables"],
            ["VJC_financial_statements_2018_separate|42"],
        )


if __name__ == "__main__":
    unittest.main()
