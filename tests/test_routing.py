from __future__ import annotations

import unittest

from src.retrieval import build_qdrant_filter
from src.routing import QueryRoutingError, reconcile_query_filters, validate_llm_filters


class RoutingTests(unittest.TestCase):
    def test_reconciles_canonical_identity_and_semantic_query(self) -> None:
        question = (
            "Lãi tiền gửi năm 2018 của công ty mẹ CTCP Hàng không Vietjet "
            "(VJC) là bao nhiêu triệu đồng?"
        )
        filters, semantic_query = reconcile_query_filters(
            question,
            {
                "ticker": ["VJC"],
                "company_name": ["CTCP Hàng không Vietjet"],
                "year": [2018],
                "report_type": ["separate"],
            },
        )
        self.assertEqual(filters["ticker"], ["VJC"])
        self.assertEqual(filters["company_name"], ["CTCP Hàng không Vietjet"])
        self.assertEqual(filters["year"], [2018])
        self.assertEqual(filters["report_type"], ["separate"])
        self.assertNotIn("VJC", semantic_query)
        self.assertNotIn("2018", semantic_query)
        self.assertIn("Lãi tiền gửi", semantic_query)

    def test_drops_non_explicit_llm_report_and_table_guesses(self) -> None:
        question = (
            "Lãi tiền gửi năm 2018 của công ty mẹ CTCP Hàng không Vietjet "
            "(VJC) là bao nhiêu triệu đồng?"
        )
        filters, _ = reconcile_query_filters(
            question,
            {
                "ticker": ["VJC"],
                "year": [2018],
                "report_type": ["consolidated"],
                "table_type": ["income_statement"],
            },
        )
        self.assertEqual(filters["report_type"], ["separate"])
        self.assertNotIn("table_type", filters)

    def test_rejects_question_without_year(self) -> None:
        with self.assertRaisesRegex(QueryRoutingError, "Không resolve được năm"):
            reconcile_query_filters(
                "Lợi nhuận của CTCP Hàng không Vietjet (VJC) là bao nhiêu?",
                {"ticker": ["VJC"]},
            )

    def test_routes_multiple_companies_and_years(self) -> None:
        question = (
            "Giá trị trung bình từ năm 2019 đến 2021 của Tổng Công ty Khí Việt Nam "
            "- CTCP (GAS) và Tổng Công ty Điện lực Dầu khí Việt Nam - CTCP (POW) "
            "là bao nhiêu?"
        )
        filters, _ = reconcile_query_filters(
            question,
            {"ticker": ["GAS", "POW"], "year": [2019, 2020, 2021]},
        )
        self.assertEqual(filters["ticker"], ["GAS", "POW"])
        self.assertEqual(filters["year"], [2019, 2020, 2021])
        self.assertEqual(len(filters["company_name"]), 2)

    def test_report_type_contract_matches_indexing(self) -> None:
        self.assertEqual(
            validate_llm_filters({"report_type": ["aggregated", "other"]}),
            {"report_type": ["aggregated", "other"]},
        )
        with self.assertRaisesRegex(QueryRoutingError, "report_type không hợp lệ"):
            validate_llm_filters({"report_type": ["standalone"]})

    def test_builds_match_any_for_all_payload_filter_fields(self) -> None:
        qdrant_filter = build_qdrant_filter(
            {
                "ticker": ["VJC", "ACB"],
                "company_name": ["CTCP Hàng không Vietjet"],
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
                "company_name",
                "year",
                "report_type",
                "table_type",
            ],
        )
        self.assertEqual(qdrant_filter.must[0].match.any, ["VJC", "ACB"])

    def test_rejects_unknown_qdrant_filter(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported Qdrant filters"):
            build_qdrant_filter({"csv_path": ["data/example.csv"]})


if __name__ == "__main__":
    unittest.main()
