from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.graph import graph
from src.nodes import load_tables_node, parse_query_node, retrieve_tables_node
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
    }
    return {
        "table_id": metadata["table_id"],
        "metadata": metadata,
        "retrieval_score": 0.9,
        "dense_rank": 1,
        "rerank_score": 0.98,
    }


class GraphTests(unittest.TestCase):
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
        self.assertEqual([call["top_n"] for call in calls], [25, 25])
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
                _: str, *, system_prompt: str | None = None
            ) -> dict[str, object]:
                if system_prompt and "bộ định tuyến" in system_prompt:
                    return {
                        "ticker": ["c01"],
                        "year": [2018],
                        "report_type": ["separate"],
                    }
                return {
                    "pandas_query": "result = float(df_1.loc[0, '2018'])",
                    "evidence_variables": ["df_1"],
                }

            with (
                patch("src.nodes._PROJECT_ROOT", root),
                patch("src.parser.generate_structured", side_effect=structured_response),
                patch("src.nodes.generate_structured", side_effect=structured_response),
                patch("src.nodes.retrieve", return_value=[candidate()]),
                patch("src.nodes.rerank", return_value=[candidate()]),
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
