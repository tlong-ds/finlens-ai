from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_parser_valset


def candidate(doc_id: str, start_line: int, rank: int) -> dict[str, object]:
    return {
        "table_id": f"{doc_id}_table_{rank}",
        "retrieval_score": 1.0 / rank,
        "dense_rank": rank,
        "metadata": {
            "doc_id": doc_id,
            "start_line": start_line,
        },
    }


def parser_artifact(
    question_id: int,
    *,
    expected_report: str = "separate",
    predicted_report: str = "separate",
) -> dict[str, object]:
    expected = {
        "ticker": ["VJC"],
        "year": [2018],
        "report_type": [expected_report],
    }
    filters = {
        "ticker": ["VJC"],
        "year": [2018],
        "report_type": [predicted_report],
    }
    diagnostics = {
        "semantic_attempts": 1,
        "attempts": [
            {
                "attempt": 1,
                "raw_filters": dict(filters),
                "validation_error": None,
            }
        ],
        "ticker_candidates": [],
    }
    return {
        "schema_version": 1,
        "id": question_id,
        "question": "Question",
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:01+00:00",
        "duration_seconds": 1.0,
        "provider_attempts": 1,
        "expected_filters": expected,
        "golden_retrieval": {
            "relevant_docs": ["VJC_financial_statements_2018_separate"],
            "relevant_tables": [
                "VJC_financial_statements_2018_separate|10"
            ],
        },
        "filters": filters,
        "semantic_query": "query",
        "diagnostics": diagnostics,
        "repair_changed_fields": [],
        "metrics": run_parser_valset.score_parser_output(expected, filters),
        "retrieval_attempts": 1,
        "retrieval": {
            "requested_top_ks": [1],
            "max_top_k": 1,
            "bucket_diagnostics": [],
            "candidates": [],
            "metrics_by_top_k": run_parser_valset.score_retrieval_output(
                {
                    "relevant_docs": [
                        "VJC_financial_statements_2018_separate"
                    ],
                    "relevant_tables": [
                        "VJC_financial_statements_2018_separate|10"
                    ],
                },
                [],
                [1],
            ),
        },
        "parser_error": None,
        "retrieval_error": None,
        "error": None,
    }


class ParserValsetTests(unittest.TestCase):
    def test_derives_ticker_year_and_other_report_from_golden_docs(self) -> None:
        filters = run_parser_valset.derive_golden_filters(
            {
                "relevant_docs": [
                    "FTS_financial_statements_2022",
                    "FTS_financial_statements_2023",
                ]
            }
        )
        self.assertEqual(filters["ticker"], ["FTS"])
        self.assertEqual(filters["year"], [2022, 2023])
        self.assertEqual(filters["report_type"], ["other"])

    def test_scores_exact_coverage_and_overselection_independently(self) -> None:
        score = run_parser_valset.score_parser_output(
            {
                "ticker": ["VJC"],
                "year": [2023, 2024],
                "report_type": ["consolidated"],
            },
            {
                "ticker": ["VJC"],
                "year": [2022, 2023, 2024],
                "report_type": ["consolidated"],
            },
        )
        self.assertTrue(score["ticker"]["exact"])
        self.assertTrue(score["year"]["coverage"])
        self.assertFalse(score["year"]["exact"])
        self.assertEqual(score["year"]["extra"], [2022])

    def test_aggregates_report_classes(self) -> None:
        first = parser_artifact(1)
        second = parser_artifact(
            2, expected_report="consolidated", predicted_report="separate"
        )
        metrics = run_parser_valset.aggregate_parser_metrics([first, second])
        self.assertEqual(metrics["parser_success_rate"], 1.0)
        self.assertEqual(metrics["report_type_exact_set_accuracy"], 0.5)
        self.assertEqual(
            metrics["report_accuracy_by_class"]["separate"]["accuracy"], 1.0
        )
        self.assertEqual(
            metrics["report_accuracy_by_class"]["consolidated"]["accuracy"], 0.0
        )
        self.assertNotIn("table_type_emission_rate", metrics)

    def test_scores_retrieval_prefixes_with_run_valset_formulas(self) -> None:
        golden = {
            "relevant_docs": ["doc-1", "doc-2"],
            "relevant_tables": ["doc-1|10", "doc-2|20"],
        }
        candidates = [
            candidate("wrong", 1, 1),
            candidate("doc-1", 10, 2),
            candidate("doc-2", 20, 3),
        ]
        scored = run_parser_valset.score_retrieval_output(
            golden, candidates, [1, 2, 3]
        )

        self.assertEqual(scored["1"]["tables"]["recall"], 0.0)
        self.assertAlmostEqual(scored["2"]["tables"]["precision"], 0.5)
        self.assertAlmostEqual(scored["2"]["tables"]["recall"], 0.5)
        self.assertAlmostEqual(scored["2"]["tables"]["mrr5"], 0.5)
        self.assertAlmostEqual(scored["3"]["tables"]["f2"], 10 / 11)

    def test_balanced_retrieval_interleaves_ticker_buckets(self) -> None:
        def side_effect(
            *,
            query_text: str,
            filters: dict[str, object],
            top_n: int,
            mode: str,
        ) -> list[dict[str, object]]:
            self.assertEqual(mode, "hybrid")
            ticker = str(filters["ticker"][0])
            return [
                candidate(f"{ticker}-doc", 10, 1),
                candidate(f"{ticker}-doc", 20, 2),
            ][:top_n]

        with patch("run_parser_valset.retrieve", side_effect=side_effect) as mocked:
            candidates, buckets = run_parser_valset._retrieve_balanced(
                "query",
                {
                    "ticker": ["AAA", "BBB"],
                    "year": [2024],
                    "report_type": ["consolidated"],
                },
                top_n=4,
                retrieval_mode="hybrid",
            )

        self.assertEqual(mocked.call_count, 2)
        self.assertEqual([bucket["requested_top_n"] for bucket in buckets], [2, 2])
        self.assertEqual(
            [item["metadata"]["doc_id"] for item in candidates],
            ["AAA-doc", "BBB-doc", "AAA-doc", "BBB-doc"],
        )
        self.assertEqual(
            [item["retrieval_rank"] for item in candidates], [1, 2, 3, 4]
        )

    def test_parser_artifact_includes_retrieval_metrics(self) -> None:
        record = {
            "id": 1,
            "question": "Question",
            "relevant_docs": ["VJC_financial_statements_2018_separate"],
            "relevant_tables": ["VJC_financial_statements_2018_separate|10"],
        }
        parsed = {
            "filters": {
                "ticker": ["VJC"],
                "year": [2018],
                "report_type": ["separate"],
            },
            "semantic_query": "query",
            "diagnostics": {"semantic_attempts": 1, "attempts": []},
        }
        retrieved = [candidate("VJC_financial_statements_2018_separate", 10, 1)]
        with (
            patch("run_parser_valset.parse_query_with_diagnostics", return_value=parsed),
            patch(
                "run_parser_valset._retrieve_balanced",
                return_value=(retrieved, [{"ticker": "VJC"}]),
            ) as retrieve_mock,
        ):
            artifact = run_parser_valset._parse_with_transient_retry(record, [1, 5])

        self.assertIsNone(artifact["error"])
        self.assertEqual(artifact["retrieval_attempts"], 1)
        self.assertEqual(
            artifact["retrieval"]["metrics_by_top_k"]["1"]["tables"]["recall"],
            1.0,
        )
        retrieve_mock.assert_called_once_with(
            "query", parsed["filters"], top_n=5, retrieval_mode="dense"
        )

    def test_original_question_can_replace_semantic_query_for_retrieval(self) -> None:
        record = {
            "id": 1,
            "question": "Original financial question",
            "relevant_docs": ["VJC_financial_statements_2018_separate"],
            "relevant_tables": ["VJC_financial_statements_2018_separate|10"],
        }
        source = parser_artifact(1)
        source["question"] = record["question"]
        source["semantic_query"] = "short semantic query"
        retrieved = [candidate("VJC_financial_statements_2018_separate", 10, 1)]
        with (
            patch("run_parser_valset.parse_query_with_diagnostics") as parse_mock,
            patch(
                "run_parser_valset._retrieve_balanced",
                return_value=(retrieved, [{"ticker": "VJC"}]),
            ) as retrieve_mock,
        ):
            artifact = run_parser_valset._parse_with_transient_retry(
                record,
                [1],
                retrieval_query_source="question",
                parser_source=source,
            )

        parse_mock.assert_not_called()
        retrieve_mock.assert_called_once_with(
            record["question"],
            source["filters"],
            top_n=1,
            retrieval_mode="dense",
        )
        self.assertEqual(artifact["retrieval"]["query_source"], "question")
        self.assertEqual(artifact["retrieval"]["query_text"], record["question"])

    def test_aggregate_retrieval_counts_failed_question_as_empty(self) -> None:
        first = parser_artifact(1)
        golden = first["golden_retrieval"]
        retrieved = [candidate("VJC_financial_statements_2018_separate", 10, 1)]
        first["retrieval"]["metrics_by_top_k"] = (
            run_parser_valset.score_retrieval_output(golden, retrieved, [1])
        )
        second = parser_artifact(2)
        second["error"] = {"stage": "retrieval", "message": "failed"}
        second["retrieval_error"] = second["error"]

        metrics = run_parser_valset.aggregate_retrieval_metrics([first, second], [1])
        health = run_parser_valset.aggregate_retrieval_health([first, second])
        parser_metrics = run_parser_valset.aggregate_parser_metrics([first, second])

        self.assertEqual(metrics["1"]["TABLES RECALL"], 0.5)
        self.assertEqual(metrics["1"]["DOCS RECALL"], 0.5)
        self.assertEqual(health["retrieval_success_rate"], 0.5)
        self.assertEqual(parser_metrics["parser_success_rate"], 1.0)

    def test_parses_sorted_unique_top_k_values(self) -> None:
        self.assertEqual(run_parser_valset._parse_top_ks("20,5,10,5"), (5, 10, 20))
        with self.assertRaisesRegex(ValueError, "positive"):
            run_parser_valset._parse_top_ks("0,5")

    def test_runner_writes_parser_artifacts_and_resume_skips_success(self) -> None:
        golden = [
            {
                "id": 1,
                "question": "Question",
                "answer": 1.0,
                "relevant_docs": ["VJC_financial_statements_2018_separate"],
                "relevant_tables": [],
            }
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            golden_path = root / "golden.json"
            golden_path.write_text(json.dumps(golden), encoding="utf-8")
            output_root = root / "parser-runs"
            artifact = parser_artifact(1)
            with patch(
                "run_parser_valset._parse_with_transient_retry",
                return_value=artifact,
            ) as parse_mock:
                exit_code = run_parser_valset.main(
                    [
                        "--golden",
                        str(golden_path),
                        "--output-dir",
                        str(output_root),
                        "--run-id",
                        "test",
                    ]
                )
            self.assertEqual(exit_code, 0)
            parse_mock.assert_called_once()
            self.assertTrue((output_root / "test" / "metrics.json").is_file())
            metrics = json.loads(
                (output_root / "test" / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertIn("retrieval_by_top_k", metrics["metrics"])
            self.assertTrue(
                (output_root / "test" / "artifacts" / "questions" / "1.json").is_file()
            )

            with patch("run_parser_valset._parse_with_transient_retry") as resume_mock:
                resumed = run_parser_valset.main(
                    [
                        "--golden",
                        str(golden_path),
                        "--output-dir",
                        str(output_root),
                        "--run-id",
                        "test",
                        "--resume",
                    ]
                )
            self.assertEqual(resumed, 0)
            resume_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
