from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_parser_valset


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
        "filters": filters,
        "semantic_query": "query",
        "diagnostics": diagnostics,
        "repair_changed_fields": [],
        "metrics": run_parser_valset.score_parser_output(expected, filters),
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
