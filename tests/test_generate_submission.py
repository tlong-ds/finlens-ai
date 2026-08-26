from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import generate_submission


QUESTIONS = [
    {"id": 1, "question": "Câu hỏi một?"},
    {"id": 2, "question": "Câu hỏi hai?"},
]


def answer_record(question_id: int) -> dict[str, object]:
    return {
        "id": question_id,
        "question": QUESTIONS[question_id - 1]["question"],
        "answer": float(question_id),
        "evidence": {},
        "relevant_docs": [],
        "relevant_tables": [],
        "pandas_query": f"result = {question_id}.0",
    }


class SubmissionRunTests(unittest.TestCase):
    def test_single_creates_isolated_run_and_refuses_implicit_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            with (
                patch("generate_submission.load_questions", return_value=QUESTIONS),
                patch(
                    "generate_submission.run_question",
                    return_value=answer_record(1),
                ) as run,
            ):
                result = generate_submission.main(
                    [
                        "--output-dir",
                        str(output_root),
                        "--run-id",
                        "experiment-1",
                        "single",
                        "--question-id",
                        "1",
                    ]
                )
                collision = generate_submission.main(
                    [
                        "--output-dir",
                        str(output_root),
                        "--run-id",
                        "experiment-1",
                        "single",
                        "--question-id",
                        "1",
                    ]
                )
                resumed = generate_submission.main(
                    [
                        "--output-dir",
                        str(output_root),
                        "--run-id",
                        "experiment-1",
                        "--resume",
                        "single",
                        "--question-id",
                        "1",
                    ]
                )

            self.assertEqual((result, collision, resumed), (0, 2, 0))
            self.assertEqual(run.call_count, 1)
            run_dir = output_root / "runs" / "experiment-1"
            status = json.loads((run_dir / "status.json").read_text())
            self.assertEqual(status["state"], "completed")
            self.assertEqual(status["counts"]["succeeded"], 1)
            self.assertTrue((run_dir / "submission.json").is_file())
            with zipfile.ZipFile(run_dir / "submission.zip") as archive:
                self.assertIn("submission.json", archive.namelist())

            latest = json.loads((output_root / "latest_run.json").read_text())
            self.assertEqual(latest["run_id"], "experiment-1")

    def test_resume_rejects_changed_model_config(self) -> None:
        arguments = [
            "--run-id",
            "immutable-config",
            "single",
            "--question-id",
            "1",
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            arguments[0:0] = ["--output-dir", temp_dir]
            with (
                patch("generate_submission.load_questions", return_value=QUESTIONS),
                patch(
                    "generate_submission.run_question",
                    return_value=answer_record(1),
                ),
                patch.dict("os.environ", {"LLM_MODEL": "model-a"}),
            ):
                first_result = generate_submission.main(arguments)

            resumed_arguments = [*arguments[:4], "--resume", *arguments[4:]]
            with (
                patch("generate_submission.load_questions", return_value=QUESTIONS),
                patch.dict("os.environ", {"LLM_MODEL": "model-b"}),
            ):
                resumed_result = generate_submission.main(resumed_arguments)

        self.assertEqual((first_result, resumed_result), (0, 2))

    def test_full_resume_retries_only_failed_questions(self) -> None:
        def first_attempt(question: str, _: int) -> dict[str, object]:
            if question == QUESTIONS[1]["question"]:
                raise RuntimeError("synthetic failure")
            return answer_record(1)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            with (
                patch("generate_submission.load_questions", return_value=QUESTIONS),
                patch(
                    "generate_submission.run_question",
                    side_effect=first_attempt,
                ),
            ):
                first_result = generate_submission.main(
                    [
                        "--output-dir",
                        str(output_root),
                        "--run-id",
                        "experiment-2",
                        "full",
                        "--concurrency",
                        "1",
                    ]
                )

            run_dir = output_root / "runs" / "experiment-2"
            first_status = json.loads((run_dir / "status.json").read_text())
            self.assertEqual(first_status["state"], "completed_with_failures")
            self.assertEqual(first_status["counts"]["succeeded"], 1)
            self.assertEqual(first_status["counts"]["failed"], 1)

            with (
                patch("generate_submission.load_questions", return_value=QUESTIONS),
                patch(
                    "generate_submission.run_question",
                    return_value=answer_record(2),
                ) as resumed_run,
            ):
                resumed_result = generate_submission.main(
                    [
                        "--output-dir",
                        str(output_root),
                        "--run-id",
                        "experiment-2",
                        "--resume",
                        "full",
                        "--concurrency",
                        "1",
                    ]
                )

            self.assertEqual((first_result, resumed_result), (1, 0))
            resumed_run.assert_called_once_with(QUESTIONS[1]["question"], 3)
            final_status = json.loads((run_dir / "status.json").read_text())
            self.assertEqual(final_status["state"], "completed")
            self.assertEqual(final_status["counts"]["succeeded"], 2)
            self.assertEqual(final_status["counts"]["failed"], 0)
            records = json.loads((run_dir / "submission.json").read_text())
            self.assertEqual([record["id"] for record in records], [1, 2])
            self.assertEqual(len((run_dir / "failures.jsonl").read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
