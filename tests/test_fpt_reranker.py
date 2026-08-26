from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import httpx

from src.fpt_reranker import (
    FptRerankerConfig,
    FptRerankerError,
    TransientFptRerankerError,
    rerank_documents,
)


def response(status_code: int, body: object) -> Mock:
    item = Mock()
    item.status_code = status_code
    item.is_error = status_code >= 400
    item.json.return_value = body
    return item


class FptRerankerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = FptRerankerConfig(
            endpoint="https://fpt.invalid/v1/rerank",
            model="bge-reranker-v2-m3",
            timeout_seconds=5.0,
            max_attempts=3,
        )

    def test_top_n_payload_and_rank_order_are_preserved(self) -> None:
        documents = [f"table-{index}" for index in range(25)]
        body = {
            "results": [
                {"index": index, "relevance_score": 1.0 - rank / 100}
                for rank, index in enumerate(range(24, 4, -1), start=1)
            ]
        }
        with (
            patch.dict("os.environ", {"FPT_API_KEY": "secret"}),
            patch("src.fpt_reranker.httpx.post", return_value=response(200, body)) as post,
        ):
            result = rerank_documents(
                "question", documents, top_n=20, config=self.config
            )

        self.assertEqual([index for index, _ in result], list(range(24, 4, -1)))
        request = post.call_args.kwargs
        self.assertEqual(request["json"]["top_n"], 20)
        self.assertEqual(request["json"]["model"], "bge-reranker-v2-m3")
        self.assertEqual(request["json"]["documents"], documents)
        self.assertEqual(request["headers"]["Authorization"], "Bearer secret")

    def test_transient_failures_retry_three_times_then_fail_without_fallback(self) -> None:
        with (
            patch.dict("os.environ", {"FPT_API_KEY": "secret"}),
            patch(
                "src.fpt_reranker.httpx.post",
                side_effect=httpx.ConnectError("offline"),
            ) as post,
            patch("src.fpt_reranker.time.sleep") as sleep,
        ):
            with self.assertRaises(TransientFptRerankerError):
                rerank_documents(
                    "question", ["table"], top_n=1, config=self.config
                )
        self.assertEqual(post.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_permanent_http_error_does_not_retry(self) -> None:
        with (
            patch.dict("os.environ", {"FPT_API_KEY": "secret"}),
            patch(
                "src.fpt_reranker.httpx.post",
                return_value=response(400, {"error": "bad request"}),
            ) as post,
        ):
            with self.assertRaises(FptRerankerError):
                rerank_documents(
                    "question", ["table"], top_n=1, config=self.config
                )
        post.assert_called_once()

    def test_partial_duplicate_and_nonfinite_results_are_rejected(self) -> None:
        invalid_bodies = (
            {"results": [{"index": 0, "relevance_score": 1.0}]},
            {
                "results": [
                    {"index": 0, "relevance_score": 1.0},
                    {"index": 0, "relevance_score": 0.9},
                ]
            },
            {
                "results": [
                    {"index": 0, "relevance_score": 1.0},
                    {"index": 1, "relevance_score": float("nan")},
                ]
            },
        )
        for body in invalid_bodies:
            with self.subTest(body=body):
                with (
                    patch.dict("os.environ", {"FPT_API_KEY": "secret"}),
                    patch(
                        "src.fpt_reranker.httpx.post",
                        return_value=response(200, body),
                    ),
                ):
                    with self.assertRaises(FptRerankerError):
                        rerank_documents(
                            "question",
                            ["table-a", "table-b"],
                            top_n=2,
                            config=self.config,
                        )


if __name__ == "__main__":
    unittest.main()
