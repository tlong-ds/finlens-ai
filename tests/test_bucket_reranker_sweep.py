from __future__ import annotations

import unittest

import run_bucket_reranker_sweep as sweep


def variant(recall: float, f2: float, doc_recall: float = 1.0) -> dict[str, object]:
    return {
        "tables": {"precision": f2, "recall": recall, "f2": f2, "mrr5": f2},
        "docs": {
            "precision": doc_recall,
            "recall": doc_recall,
            "f2": doc_recall,
            "mrr5": doc_recall,
        },
        "candidate_count": 5,
        "mean_candidates_per_bucket": 5.0,
    }


class BucketRerankerSweepTests(unittest.TestCase):
    def test_selects_smallest_depth_within_one_point_of_best_coverage(self) -> None:
        artifacts = []
        for index in range(100):
            variants = {
                sweep.BASELINE_NAME: variant(1.0 if index < 80 else 0.0, 0.3),
                "per_bucket_top3": variant(1.0 if index < 85 else 0.0, 0.4),
                "per_bucket_top5": variant(1.0 if index < 90 else 0.0, 0.5),
                "per_bucket_top8": variant(1.0 if index < 94 else 0.0, 0.55),
                "per_bucket_top10": variant(1.0 if index < 95 else 0.0, 0.56),
            }
            artifacts.append(
                {
                    "id": index + 1,
                    "variants": variants,
                    "duration_seconds": 1.0,
                    "error": None,
                }
            )
        result = sweep._aggregate(artifacts)
        self.assertEqual(result["selected_variant"], "per_bucket_top8")
        self.assertTrue(result["promotion_gate"]["passed"])

    def test_gate_fails_when_any_question_errors(self) -> None:
        good_variants = {
            sweep.BASELINE_NAME: variant(1.0, 0.3),
            **{
                f"per_bucket_top{depth}": variant(1.0, 0.4)
                for depth in sweep.DEPTHS
            },
        }
        result = sweep._aggregate(
            [
                {
                    "id": 1,
                    "variants": good_variants,
                    "duration_seconds": 1.0,
                    "error": None,
                },
                {"id": 2, "variants": {}, "duration_seconds": 0.0, "error": {}},
            ]
        )
        self.assertFalse(result["promotion_gate"]["all_questions_successful"])
        self.assertFalse(result["promotion_gate"]["passed"])


if __name__ == "__main__":
    unittest.main()
