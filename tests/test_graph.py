from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.graph import graph
from src.nodes import load_tables_node, retrieve_tables_node
from src.retrieval import NoMatchingCandidatesError, RetrievalError


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
        "rerank_score": 0.98,
    }


class GraphTests(unittest.TestCase):
    def test_retrieve_falls_back_to_all_report_types_on_no_match(self) -> None:
        calls: list[dict[str, object]] = []

        def retrieve_side_effect(*, query_text: str, filters: dict[str, object]) -> list[dict[str, object]]:
            calls.append({"query_text": query_text, "filters": filters})
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
                        "ticker": ["VJC"],
                        "year": [2018],
                        "report_type": ["separate"],
                    }
                return {
                    "pandas_query": "result = float(df_1.loc[0, '2018'])",
                    "evidence_variables": ["df_1"],
                }

            with (
                patch("src.nodes._PROJECT_ROOT", root),
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
