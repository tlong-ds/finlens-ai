from __future__ import annotations

import unittest

from src.llm import _normalize_base_url


class LLMConfigurationTests(unittest.TestCase):
    def test_keeps_api_root(self) -> None:
        self.assertEqual(
            _normalize_base_url("https://example.test/v1"), "https://example.test/v1"
        )

    def test_accepts_full_chat_completions_endpoint(self) -> None:
        self.assertEqual(
            _normalize_base_url("https://example.test/v1/chat/completions"),
            "https://example.test/v1",
        )


if __name__ == "__main__":
    unittest.main()
