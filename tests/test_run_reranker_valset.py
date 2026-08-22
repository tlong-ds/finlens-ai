from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_reranker_valset


GOLDEN = [
    {
        "id": 1,
        "question": "Lãi tiền gửi là bao nhiêu?",
        "answer": 10.0,
        "relevant_docs": ["AAA_financial_statements_2018_separate"],
        "relevant_tables": ["AAA_financial_statements_2018_separate|10"],
    }
]


def source_candidate() -> dict[str, object]:
    metadata = {
        "table_id": "AAA_financial_statements_2018_separate_table_1",
        "doc_id": "AAA_financial_statements_2018_separate",
        "ticker": "AAA",
        "company_name": "Công ty AAA",
        "year": 2018,
        "report_type": "separate",
        "table_type": "note_table",
        "start_line": 10,
    }
    return {
        "table_id": metadata["table_id"],
        "metadata": metadata,
        "retrieval_score": 0.9,
        "dense_rank": 1,
    }


def successful_rerank() -> tuple[list[dict[str, object]], dict[str, object]]:
    candidate = source_candidate()
    ranked = [{**candidate, "rerank_rank": 1, "rerank_source": "llm"}]
    catalog = {
        "c01": {
            "bucket_key": "b01",
            "table_id": candidate["table_id"],
            "table_ref": "AAA_financial_statements_2018_separate|10",
            "doc_id": "AAA_financial_statements_2018_separate",
            "table_type": "note_table",
            "retrieval_rank": 1,
        }
    }
    diagnostics = {
        "candidate_catalog": catalog,
        "shortlist_keys": ["c01"],
        "scout_nominated_keys": ["c01"],
        "finalist_keys": ["c01"],
        "final_llm_keys": ["c01"],
        "selected_keys": ["c01"],
        "required_bucket_keys": ["b01"],
        "coverage_completion": {},
        "uncovered_concept_keys": [],
    }
    return ranked, diagnostics


class RerankerRunnerTests(unittest.TestCase):
    def _prepare(self, root: Path) -> tuple[Path, Path]:
        golden_path = root / "golden.json"
        golden_path.write_text(json.dumps(GOLDEN, ensure_ascii=False), encoding="utf-8")
        source_run = root / "source"
        artifact_dir = source_run / "artifacts" / "questions"
        artifact_dir.mkdir(parents=True)
        (source_run / "status.json").write_text(
            json.dumps(
                {
                    "config": {
                        "golden_sha256": run_reranker_valset._sha256(golden_path)
                    }
                }
            ),
            encoding="utf-8",
        )
        (artifact_dir / "1.json").write_text(
            json.dumps(
                {
                    "id": 1,
                    "question": GOLDEN[0]["question"],
                    "events": [
                        {
                            "node": "retrieve_tables",
                            "output": {"candidates": [source_candidate()]},
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return golden_path, source_run

    def test_runner_replays_source_and_writes_stage_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            golden_path, source_run = self._prepare(root)
            output_root = root / "output"
            with patch(
                "run_reranker_valset.rerank_with_diagnostics",
                return_value=successful_rerank(),
            ) as rerank_call:
                result = run_reranker_valset.main(
                    [
                        "--golden",
                        str(golden_path),
                        "--output-dir",
                        str(output_root),
                        "--source-run",
                        str(source_run),
                        "--run-id",
                        "test-run",
                        "--concurrency",
                        "1",
                    ]
                )

            self.assertEqual(result, 0)
            rerank_call.assert_called_once()
            run_dir = output_root / "test-run"
            metrics = json.loads((run_dir / "metrics.json").read_text())
            self.assertEqual(
                metrics["stage_metrics"]["reranker"]["TABLES F2-MACRO"], 1.0
            )
            self.assertEqual(
                metrics["diagnostics"]["required_bucket_gold_recall"], 1.0
            )
            self.assertEqual(metrics["diagnostics"]["coverage_lock_questions"], 0)
            self.assertEqual(
                metrics["diagnostics"]["unresolved_required_bucket_count"], 0
            )
            artifact = json.loads(
                (run_dir / "artifacts" / "questions" / "1.json").read_text()
            )
            self.assertEqual(artifact["loss_attribution"][0]["terminal"], "selected")
            self.assertNotIn("answer", metrics)

    def test_resume_skips_successful_question(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            golden_path, source_run = self._prepare(root)
            output_root = root / "output"
            base = [
                "--golden",
                str(golden_path),
                "--output-dir",
                str(output_root),
                "--source-run",
                str(source_run),
                "--run-id",
                "resume-run",
                "--concurrency",
                "1",
            ]
            with patch(
                "run_reranker_valset.rerank_with_diagnostics",
                return_value=successful_rerank(),
            ):
                self.assertEqual(run_reranker_valset.main(base), 0)
            with patch(
                "run_reranker_valset.rerank_with_diagnostics"
            ) as rerank_call:
                self.assertEqual(run_reranker_valset.main([*base, "--resume"]), 0)
            rerank_call.assert_not_called()

    def test_loss_attribution_identifies_shortlist_drop(self) -> None:
        stages = {
            stage: {"docs": [], "tables": []}
            for stage in run_reranker_valset.STAGE_NAMES
        }
        stages["retriever"] = {
            "docs": GOLDEN[0]["relevant_docs"],
            "tables": GOLDEN[0]["relevant_tables"],
        }
        loss = run_reranker_valset._loss_attribution(GOLDEN[0], stages)
        self.assertEqual(loss[0]["terminal"], "shortlist")


if __name__ == "__main__":
    unittest.main()
