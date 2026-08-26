from __future__ import annotations

import json
import unittest

from src.prompt import (
    COVERAGE_VALIDATOR_SYSTEM_PROMPT,
    GENERATOR_SYSTEM_PROMPT,
    PARSE_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    build_coverage_validator_prompt,
    build_parse_prompt,
    build_planner_prompt,
)
from src.retrieval import build_qdrant_filter
from src.routing import (
    QueryRoutingError,
    build_ticker_shortlist,
    reconcile_query_filters,
    serialize_ticker_candidates,
    validate_llm_filters,
)


class RoutingTests(unittest.TestCase):
    @staticmethod
    def _candidate_key(
        question: str, ticker: str
    ) -> tuple[str, tuple[object, ...]]:
        candidates = build_ticker_shortlist(question)
        context = serialize_ticker_candidates(candidates)
        key = next(item["candidate_key"] for item in context if item["ticker"] == ticker)
        return key, candidates

    def test_shortlist_scans_every_exact_ticker_without_early_return(self) -> None:
        question = "Trong bốn mã cổ phiếu AAA, DCM, DPM và GVR năm 2016"
        candidates = build_ticker_shortlist(question)
        tickers = {candidate.ticker for candidate in candidates}
        self.assertTrue({"AAA", "DCM", "DPM", "GVR"}.issubset(tickers))

    def test_shortlist_keeps_collision_candidates_for_llm_resolution(self) -> None:
        candidates = build_ticker_shortlist("CTCP Chứng khoán FPT năm 2023")
        by_ticker = {candidate.ticker: candidate for candidate in candidates}
        self.assertEqual(by_ticker["FPT"].match_type, "exact_ticker")
        self.assertEqual(by_ticker["FTS"].match_type, "exact_alias")
        self.assertEqual(by_ticker["FOX"].match_type, "collision")

    def test_collision_ignores_legal_entity_tokens(self) -> None:
        candidates = build_ticker_shortlist("CTCP Tasco năm 2023")
        self.assertEqual([candidate.ticker for candidate in candidates], ["HUT"])

    def test_shortlist_keeps_main_and_related_company_for_llm(self) -> None:
        question = (
            "Vay dài hạn với Công ty Cổ phần Hoàng Anh Gia Lai của công ty mẹ "
            "CTCP Nông nghiệp Quốc tế Hoàng Anh Gia Lai (HNG) cuối năm 2017"
        )
        tickers = {candidate.ticker for candidate in build_ticker_shortlist(question)}
        self.assertTrue({"HNG", "HAG"}.issubset(tickers))

    def test_fuzzy_company_name_handles_typo_but_never_fuzzes_ticker(self) -> None:
        candidates = build_ticker_shortlist(
            "Phân bón và Hóa chấc Dầu khí năm 2023"
        )
        by_ticker = {candidate.ticker: candidate for candidate in candidates}
        self.assertEqual(by_ticker["DPM"].match_type, "fuzzy_company_name")
        self.assertIn("Phân bón", by_ticker["DPM"].matched_text)

    def test_shortlist_is_deterministic_and_bounded(self) -> None:
        question = "Các công ty chứng khoán năm 2023"
        first = build_ticker_shortlist(question, max_candidates=3)
        second = build_ticker_shortlist(question, max_candidates=3)
        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 3)

    def test_prompt_serializes_compact_candidate_context_without_scores(self) -> None:
        candidates = build_ticker_shortlist("CTCP Chứng khoán FPT năm 2023")
        context = serialize_ticker_candidates(candidates)
        payload = json.loads(
            build_parse_prompt("CTCP Chứng khoán FPT năm 2023", context)
        )
        self.assertEqual(payload["ticker_candidates"], context)
        self.assertEqual(
            set(context[0]),
            {"candidate_key", "ticker", "company_name", "matched_text", "match_type"},
        )
        self.assertNotIn("score", payload["ticker_candidates"][0])

    def test_report_prompt_uses_dataset_decision_tree_and_minimal_pairs(self) -> None:
        self.assertIn("quy ước nhãn của bộ dữ liệu", PARSE_SYSTEM_PROMPT)
        self.assertIn('"số liệu công ty mẹ"', PARSE_SYSTEM_PROMPT)
        self.assertIn('trả ["separate"] và tuyệt đối không trả consolidated', PARSE_SYSTEM_PROMPT)
        self.assertIn('"của Ngân hàng A" -> ["consolidated"]', PARSE_SYSTEM_PROMPT)
        self.assertNotIn("table_type", PARSE_SYSTEM_PROMPT)

    def test_coverage_validator_audits_operands_not_derived_metric_rows(self) -> None:
        payload = json.loads(build_coverage_validator_prompt("Quick ratio", []))
        self.assertEqual(
            payload["derivation_contract"]["quick_ratio"],
            ["tài sản ngắn hạn", "hàng tồn kho", "nợ ngắn hạn"],
        )
        self.assertIn("KHÔNG đòi một row trực tiếp", COVERAGE_VALIDATOR_SYSTEM_PROMPT)
        self.assertIn("bảng ma trận", COVERAGE_VALIDATOR_SYSTEM_PROMPT)
        self.assertIn("tổng nợ vay", COVERAGE_VALIDATOR_SYSTEM_PROMPT)
        self.assertIn("table_total", payload["derivation_contract"])
        self.assertTrue(payload["proof_contract"]["required_for_answerable_true"])
        self.assertIn("coverage_proofs", COVERAGE_VALIDATOR_SYSTEM_PROMPT)
        self.assertIn("EBIT = lợi nhuận trước thuế", PLANNER_SYSTEM_PROMPT)
        self.assertIn("Không thay EBIT", GENERATOR_SYSTEM_PROMPT)

    def test_planner_prompt_compacts_semantic_catalog_but_keeps_matrix_cells(self) -> None:
        common = {
            "columns": [{"name": "period_current"}],
            "row_count": 1,
            "detailed_rows": [
                {"row_position": 0, "values": {"period_current": "123"}}
            ],
        }
        payload = json.loads(
            build_planner_prompt(
                "Câu hỏi",
                [
                    {
                        **common,
                        "alias": "df_1",
                        "row_catalog": [
                            {"row_position": 0, "label": "Tổng tài sản"}
                        ],
                    },
                    {
                        **common,
                        "alias": "df_2",
                        "row_catalog": [{"row_position": 0, "label": "1"}],
                    },
                ],
            )
        )

        semantic, matrix = payload["inventory"]
        self.assertNotIn("detailed_rows", semantic)
        self.assertIn("detailed_rows", matrix)

    def test_candidate_key_validation_rejects_unknown_and_duplicates(self) -> None:
        question = "Quỹ khen thưởng của HT1 cuối năm 2019 là bao nhiêu?"
        candidates = build_ticker_shortlist(question)
        with self.assertRaisesRegex(QueryRoutingError, "không có trong shortlist"):
            validate_llm_filters({"ticker": ["c99"]}, candidates)
        with self.assertRaisesRegex(QueryRoutingError, "trùng"):
            validate_llm_filters({"ticker": ["c01", "c01"]}, candidates)

    def test_reconciles_canonical_identity_and_semantic_query(self) -> None:
        question = (
            "Lãi tiền gửi năm 2018 của công ty mẹ CTCP Hàng không Vietjet "
            "(VJC) là bao nhiêu triệu đồng?"
        )
        key, candidates = self._candidate_key(question, "VJC")
        filters, semantic_query = reconcile_query_filters(
            question,
            {
                "ticker": [key],
                "year": [2018],
                "report_type": ["separate"],
            },
            ticker_candidates=candidates,
        )
        self.assertEqual(filters["ticker"], ["VJC"])
        self.assertNotIn("company_name", filters)
        self.assertEqual(filters["year"], [2018])
        self.assertEqual(filters["report_type"], ["separate"])
        self.assertNotIn("VJC", semantic_query)
        self.assertNotIn("2018", semantic_query)
        self.assertIn("Lãi tiền gửi", semantic_query)

    def test_honors_llm_report_type_without_table_filter(self) -> None:
        question = (
            "Lãi tiền gửi năm 2018 của công ty mẹ CTCP Hàng không Vietjet "
            "(VJC) là bao nhiêu triệu đồng?"
        )
        key, candidates = self._candidate_key(question, "VJC")
        filters, _ = reconcile_query_filters(
            question,
            {
                "ticker": [key],
                "year": [2018],
                "report_type": ["consolidated"],
            },
            ticker_candidates=candidates,
        )
        self.assertEqual(filters["report_type"], ["consolidated"])
        self.assertNotIn("table_type", filters)

    def test_rejects_table_type_from_llm_contract(self) -> None:
        question = "Dòng tiền của VJC năm 2024 là bao nhiêu?"
        key, candidates = self._candidate_key(question, "VJC")
        with self.assertRaisesRegex(QueryRoutingError, "table_type"):
            reconcile_query_filters(
                question,
                {
                    "ticker": [key],
                    "year": [2024],
                    "report_type": ["consolidated"],
                    "table_type": ["cash_flow"],
                },
                ticker_candidates=candidates,
            )

    def test_does_not_infer_table_type_when_llm_omits_it(self) -> None:
        question = "Dòng tiền của VJC năm 2024 là bao nhiêu?"
        key, candidates = self._candidate_key(question, "VJC")
        filters, _ = reconcile_query_filters(
            question,
            {
                "ticker": [key],
                "year": [2024],
                "report_type": ["consolidated"],
            },
            ticker_candidates=candidates,
        )
        self.assertNotIn("table_type", filters)

    def test_rejects_llm_response_without_year(self) -> None:
        question = "Lợi nhuận của CTCP Hàng không Vietjet (VJC) là bao nhiêu?"
        key, candidates = self._candidate_key(question, "VJC")
        with self.assertRaisesRegex(QueryRoutingError, "year.*không rỗng"):
            reconcile_query_filters(
                question,
                {"ticker": [key], "report_type": ["consolidated"]},
                ticker_candidates=candidates,
            )

    def test_rejects_invalid_year_shapes_without_local_coercion(self) -> None:
        question = "Lợi nhuận của VJC năm 2024 là bao nhiêu?"
        key, candidates = self._candidate_key(question, "VJC")
        for invalid_year in (2024, "2024", None, [], [True], [2026]):
            with self.subTest(year=invalid_year):
                with self.assertRaises(QueryRoutingError):
                    reconcile_query_filters(
                        question,
                        {
                            "ticker": [key],
                            "year": invalid_year,
                            "report_type": ["consolidated"],
                        },
                        ticker_candidates=candidates,
                    )

    def test_llm_years_are_deduped_without_expansion_or_sorting(self) -> None:
        question = "Tăng trưởng của GAS từ năm 2019 đến năm 2021 là bao nhiêu?"
        key, candidates = self._candidate_key(question, "GAS")
        filters, _ = reconcile_query_filters(
            question,
            {
                "ticker": [key],
                "year": [2021, 2019, 2021],
                "report_type": ["consolidated"],
            },
            ticker_candidates=candidates,
        )
        self.assertEqual(filters["year"], [2021, 2019])

    def test_routes_multiple_companies_and_years(self) -> None:
        question = (
            "Giá trị trung bình từ năm 2019 đến 2021 của Tổng Công ty Khí Việt Nam "
            "- CTCP (GAS) và Tổng Công ty Điện lực Dầu khí Việt Nam - CTCP (POW) "
            "là bao nhiêu?"
        )
        candidates = build_ticker_shortlist(question)
        context = serialize_ticker_candidates(candidates)
        keys = [
            item["candidate_key"]
            for item in context
            if item["ticker"] in {"GAS", "POW"}
        ]
        filters, _ = reconcile_query_filters(
            question,
            {
                "ticker": keys,
                "year": [2019, 2020, 2021],
                "report_type": ["consolidated"],
            },
            ticker_candidates=candidates,
        )
        self.assertEqual(filters["ticker"], ["GAS", "POW"])
        self.assertEqual(filters["year"], [2019, 2020, 2021])
        self.assertNotIn("company_name", filters)

    def test_report_type_contract_matches_indexing(self) -> None:
        question = "Doanh thu của VJC năm 2024 là bao nhiêu?"
        key, candidates = self._candidate_key(question, "VJC")
        self.assertEqual(
            validate_llm_filters(
                {
                    "ticker": [key],
                    "year": [2024],
                    "report_type": ["other"],
                },
                candidates,
            ),
            {"ticker": ["VJC"], "year": [2024], "report_type": ["other"]},
        )
        for invalid_report_type in ([], ["separate", "consolidated"], None):
            with self.subTest(report_type=invalid_report_type):
                with self.assertRaisesRegex(QueryRoutingError, "đúng một"):
                    validate_llm_filters(
                        {
                            "ticker": [key],
                            "year": [2024],
                            "report_type": invalid_report_type,
                        },
                        candidates,
                    )
        with self.assertRaisesRegex(QueryRoutingError, "report_type không hợp lệ"):
            validate_llm_filters(
                {
                    "ticker": [key],
                    "year": [2024],
                    "report_type": ["standalone"],
                },
                candidates,
            )

    def test_rejects_company_name_llm_filter(self) -> None:
        with self.assertRaisesRegex(QueryRoutingError, "company_name"):
            validate_llm_filters({"company_name": ["CTCP Hàng không Vietjet"]})

    def test_builds_match_any_for_all_payload_filter_fields(self) -> None:
        qdrant_filter = build_qdrant_filter(
            {
                "ticker": ["VJC", "ACB"],
                "year": [2018, 2022],
                "report_type": ["separate"],
                "table_type": ["note_table"],
            }
        )
        self.assertIsNotNone(qdrant_filter)
        self.assertEqual(
            [condition.key for condition in qdrant_filter.must],
            [
                "ticker",
                "year",
                "report_type",
                "table_type",
            ],
        )
        self.assertEqual(qdrant_filter.must[0].match.any, ["VJC", "ACB"])

    def test_rejects_company_name_qdrant_filter(self) -> None:
        with self.assertRaisesRegex(ValueError, "company_name"):
            build_qdrant_filter({"company_name": ["CTCP Hàng không Vietjet"]})

    def test_rejects_unknown_qdrant_filter(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported Qdrant filters"):
            build_qdrant_filter({"csv_path": ["data/example.csv"]})


if __name__ == "__main__":
    unittest.main()
