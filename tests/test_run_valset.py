from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import run_valset


GOLDEN = [
    {
        "id": 1,
        "question": "Câu hỏi một?",
        "answer": 10.0,
        "relevant_docs": ["doc-1", "doc-2"],
        "relevant_tables": ["doc-1|10", "doc-2|20"],
    },
    {
        "id": 2,
        "question": "Câu hỏi hai?",
        "answer": 20.0,
        "relevant_docs": ["doc-3"],
        "relevant_tables": ["doc-3|30"],
    },
]


def traced_success(question_id: int) -> dict[str, object]:
    golden = GOLDEN[question_id - 1]
    return {
        "schema_version": 2,
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:01+00:00",
        "duration_seconds": 1.0,
        "events": [{"sequence": 1, "node": "execute_code", "output": {}}],
        "stage_rankings": {
            "retriever": {
                "docs": golden["relevant_docs"],
                "tables": golden["relevant_tables"],
            },
            "reranker": {
                "docs": golden["relevant_docs"],
                "tables": golden["relevant_tables"],
            },
            "selector": {
                "docs": golden["relevant_docs"],
                "tables": golden["relevant_tables"],
            },
        },
        "answer_record": {
            "id": question_id,
            "question": golden["question"],
            "answer": golden["answer"],
            "evidence": {},
            "relevant_docs": golden["relevant_docs"],
            "relevant_tables": golden["relevant_tables"],
            "pandas_query": f"result = {golden['answer']}",
        },
        "error": None,
    }


class MetricTests(unittest.TestCase):
    def test_macro_metrics_and_mrr5(self) -> None:
        predictions = {
            1: {
                "answer": 10.0,
                "relevant_docs": ["doc-2"],
                "relevant_tables": ["doc-2|20", "wrong|1", "doc-1|10"],
            }
        }
        metrics, details = run_valset.calculate_metrics(
            GOLDEN,
            predictions,
            {1: True, 2: False},
        )

        self.assertEqual(len(details), 2)
        self.assertAlmostEqual(metrics["TABLES PRECISION"], (2 / 3) / 2)
        self.assertAlmostEqual(metrics["TABLES RECALL"], 1 / 2)
        self.assertAlmostEqual(metrics["TABLES F2-MACRO"], (10 / 11) / 2)
        self.assertAlmostEqual(metrics["TABLES MRR5"], 1 / 2)
        self.assertAlmostEqual(metrics["DOCS PRECISION"], 1 / 2)
        self.assertAlmostEqual(metrics["DOCS RECALL"], 0.5 / 2)
        self.assertAlmostEqual(metrics["DOCS F2-MACRO"], (5 / 9) / 2)
        self.assertAlmostEqual(metrics["ANSWER ACCURACY"], 0.5)
        self.assertAlmostEqual(metrics["EXECUTION ACCURACY"], 0.5)

    def test_answer_tolerance_is_configurable(self) -> None:
        prediction = {
            1: {
                "answer": 10.01,
                "relevant_docs": [],
                "relevant_tables": [],
            }
        }
        strict, _ = run_valset.calculate_metrics(
            GOLDEN[:1], prediction, {1: True}, answer_rtol=0.0, answer_atol=0.001
        )
        loose, _ = run_valset.calculate_metrics(
            GOLDEN[:1], prediction, {1: True}, answer_rtol=0.0, answer_atol=0.02
        )
        self.assertEqual(strict["ANSWER ACCURACY"], 0.0)
        self.assertEqual(loose["ANSWER ACCURACY"], 1.0)


class TraceTests(unittest.TestCase):
    def test_dataframe_trace_is_json_serializable_and_bounded(self) -> None:
        frame = pd.DataFrame({"label": ["a", "b"], "value": [1.0, 2.0]})
        safe = run_valset._json_safe({"df_1": frame})
        json.dumps(safe)
        self.assertEqual(safe["df_1"]["rows"], 2)
        self.assertEqual(safe["df_1"]["columns"], ["label", "value"])

    def test_graph_stream_captures_module_outputs_and_rankings(self) -> None:
        candidate = {
            "metadata": {
                "doc_id": "doc-1",
                "start_line": 10,
            }
        }

        class FakeGraph:
            def stream(self, inputs: object, stream_mode: str):
                self.inputs = inputs
                self.stream_mode = stream_mode
                yield {"parse_query": {"filters": {"year": [2020]}}}
                yield {"retrieve_tables": {"candidates": [candidate]}}
                yield {"rerank_tables": {"reranked_tables": [candidate]}}
                yield {"select_tables": {"retrieved_tables": [candidate]}}
                yield {
                    "execute_code": {
                        "answer_record": {
                            "id": 1,
                            "question": "Câu hỏi một?",
                            "answer": 10.0,
                        }
                    }
                }

        fake = FakeGraph()
        with patch("run_valset.graph", fake):
            trace = run_valset.trace_graph_question("Câu hỏi một?", 5)

        self.assertIsNone(trace["error"])
        self.assertEqual(
            [event["node"] for event in trace["events"]],
            [
                "parse_query",
                "retrieve_tables",
                "rerank_tables",
                "select_tables",
                "execute_code",
            ],
        )
        self.assertEqual(trace["stage_rankings"]["retriever"]["tables"], ["doc-1|10"])
        self.assertEqual(trace["stage_rankings"]["reranker"]["docs"], ["doc-1"])
        self.assertEqual(trace["stage_rankings"]["selector"]["docs"], ["doc-1"])

    def test_real_compiled_graph_update_stream_yields_final_answer(self) -> None:
        question = (
            "Lãi tiền gửi năm 2018 của công ty mẹ CTCP Hàng không Vietjet (VJC) "
            "là bao nhiêu triệu đồng?"
        )
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
        candidate = {
            "table_id": metadata["table_id"],
            "metadata": metadata,
            "retrieval_score": 0.9,
            "dense_rank": 1,
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

        def structured_response(
            _: str, *, system_prompt: str | None = None, **__: object
        ):
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
                            "reason": "đủ",
                            "required_operands": ["Lãi tiền gửi"],
                        }
                    ],
                    "coverage_proofs": [
                        {
                            "bucket_key": "b01",
                            "operand": "Lãi tiền gửi",
                            "table_id": metadata["table_id"],
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
                    "calculation": "Đọc giá trị",
                    "unit_conversion": "Không đổi",
                    "audit": "Đủ",
                }
            return {
                "pandas_query": "result = float(df_1.loc[0, '2018'])",
                "evidence_variables": ["df_1"],
            }

        def bucket_selector_response(_: str, **__: object):
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

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "data").mkdir()
            (root / "data" / f"{metadata['table_id']}.csv").write_text(
                "Chỉ tiêu,2018\nLãi tiền gửi,123\n", encoding="utf-8"
            )
            with (
                patch("src.nodes._PROJECT_ROOT", root),
                patch("src.parser.generate_structured", side_effect=structured_response),
                patch("src.nodes.generate_structured", side_effect=structured_response),
                patch(
                    "src.retrieval.generate_structured",
                    side_effect=bucket_selector_response,
                ),
                patch("src.nodes.retrieve", return_value=[candidate]),
                patch("src.nodes.rerank_with_fpt", return_value=[candidate]),
                patch("src.nodes.run_code", return_value=123.0),
            ):
                trace = run_valset.trace_graph_question(question, 2)

        self.assertIsNone(trace["error"])
        self.assertEqual(trace["answer_record"]["answer"], 123.0)
        self.assertEqual(
            [event["node"] for event in trace["events"]],
            [
                "match_question",
                "parse_query",
                "materialize_buckets",
                "rewrite_bucket_queries",
                "retrieve_bucket_tables",
                "rerank_bucket_tables",
                "select_bucket_tables",
                "validate_table_coverage",
                "load_tables",
                "plan_generation_context",
                "generate_code",
                "execute_code",
            ],
        )


class ValidationRunTests(unittest.TestCase):
    def _write_golden(self, directory: Path) -> Path:
        path = directory / "golden.json"
        path.write_text(json.dumps(GOLDEN, ensure_ascii=False), encoding="utf-8")
        return path

    def test_ids_run_writes_submission_metrics_zip_and_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            golden_path = self._write_golden(root)
            with patch(
                "run_valset.trace_graph_question", return_value=traced_success(1)
            ):
                result = run_valset.main(
                    [
                        "--golden",
                        str(golden_path),
                        "--output-dir",
                        str(root / "output"),
                        "--run-id",
                        "test-run",
                        "ids",
                        "--ids",
                        "1",
                        "--concurrency",
                        "1",
                    ]
                )

            self.assertEqual(result, 0)
            run_dir = root / "output" / "runs" / "test-run"
            metrics = json.loads((run_dir / "metrics.json").read_text())
            self.assertEqual(metrics["metrics"]["ANSWER ACCURACY"], 1.0)
            self.assertEqual(metrics["metrics"]["TABLES F2-MACRO"], 1.0)
            artifact = json.loads(
                (run_dir / "artifacts" / "questions" / "1.json").read_text()
            )
            self.assertEqual(artifact["id"], 1)
            self.assertIn("stage_metrics", artifact)
            with zipfile.ZipFile(run_dir / "submission.zip") as archive:
                self.assertEqual(archive.namelist(), ["submission.json"])

    def test_resume_retries_only_failed_question(self) -> None:
        failed = {
            "schema_version": 1,
            "events": [],
            "stage_rankings": {},
            "answer_record": None,
            "error": {"type": "RuntimeError", "message": "synthetic failure"},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            golden_path = self._write_golden(root)
            base = [
                "--golden",
                str(golden_path),
                "--output-dir",
                str(root / "output"),
                "--run-id",
                "resume-run",
            ]
            with patch(
                "run_valset.trace_graph_question",
                side_effect=[traced_success(1), failed],
            ):
                first = run_valset.main(
                    [*base, "full", "--concurrency", "1"]
                )

            with patch(
                "run_valset.trace_graph_question", return_value=traced_success(2)
            ) as rerun:
                resumed = run_valset.main(
                    [*base, "--resume", "full", "--concurrency", "1"]
                )

            self.assertEqual((first, resumed), (1, 0))
            rerun.assert_called_once_with(GOLDEN[1]["question"], 5)
            run_dir = root / "output" / "runs" / "resume-run"
            records = json.loads((run_dir / "submission.json").read_text())
            self.assertEqual([record["id"] for record in records], [1, 2])
            status = json.loads((run_dir / "status.json").read_text())
            self.assertEqual(status["state"], "completed")

    def test_resume_rejects_changed_bucket_rerank_depth(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            golden_path = self._write_golden(root)
            base = [
                "--golden",
                str(golden_path),
                "--output-dir",
                str(root / "output"),
                "--run-id",
                "immutable-retrieval-run",
            ]
            with (
                patch.dict(
                    "os.environ", {"FINLENS_BUCKET_RERANK_TOP_N": "10"}
                ),
                patch(
                    "run_valset.trace_graph_question",
                    side_effect=[traced_success(1), traced_success(2)],
                ),
            ):
                self.assertEqual(
                    run_valset.main([*base, "full", "--concurrency", "1"]),
                    0,
                )

            with patch.dict(
                "os.environ", {"FINLENS_BUCKET_RERANK_TOP_N": "20"}
            ):
                self.assertEqual(
                    run_valset.main(
                        [*base, "--resume", "full", "--concurrency", "1"]
                    ),
                    2,
                )


if __name__ == "__main__":
    unittest.main()
