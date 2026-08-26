from __future__ import annotations

import unittest
from unittest.mock import patch

import httpx

from src.bm25 import BM25IndexError, TransientBM25IndexError, search_bm25


def payload(
    table_id: str,
    index_text: str,
    *,
    ticker: str = "AAA",
    year: int = 2024,
    report_type: str = "consolidated",
    start_line: int = 10,
) -> dict[str, object]:
    doc_id = f"{ticker}_financial_statements_{year}_{report_type}"
    return {
        "table_id": table_id,
        "doc_id": doc_id,
        "ticker": ticker,
        "company_name": f"Công ty {ticker}",
        "year": year,
        "report_type": report_type,
        "table_type": "note_table",
        "start_line": start_line,
        "index_text": index_text,
    }


class _Point:
    def __init__(self, point_id: str, point_payload: dict[str, object]) -> None:
        self.id = point_id
        self.payload = point_payload


class _ScrollClient:
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.points = [
            _Point(f"point-{index}", point_payload)
            for index, point_payload in enumerate(payloads, start=1)
        ]
        self.calls: list[dict[str, object]] = []

    def scroll(self, **kwargs: object) -> tuple[list[_Point], int | None]:
        self.calls.append(dict(kwargs))
        selected = self.points
        query_filter = kwargs.get("scroll_filter")
        for condition in getattr(query_filter, "must", []) or []:
            values = set(condition.match.any)
            selected = [
                point
                for point in selected
                if point.payload.get(condition.key) in values
            ]
        offset = int(kwargs.get("offset") or 0)
        limit = int(kwargs.get("limit") or len(selected))
        page = selected[offset : offset + limit]
        next_offset = offset + limit if offset + limit < len(selected) else None
        return page, next_offset


class BM25Tests(unittest.TestCase):
    def test_search_reads_qdrant_payload_and_applies_metadata_filters(self) -> None:
        client = _ScrollClient(
            [
                payload(
                    "AAA_financial_statements_2024_consolidated_table_1",
                    "Chỉ tiêu: Doanh thu thuần bán hàng và cung cấp dịch vụ",
                ),
                payload(
                    "AAA_financial_statements_2024_consolidated_table_2",
                    "Chỉ tiêu: Tiền và các khoản tương đương tiền",
                    start_line=20,
                ),
                payload(
                    "BBB_financial_statements_2024_consolidated_table_1",
                    "Chỉ tiêu: Doanh thu thuần",
                    ticker="BBB",
                ),
            ]
        )

        with patch.dict("os.environ", {"QDRANT_BM25_SCROLL_BATCH": "1"}):
            result = search_bm25(
                "doanh thu thuần",
                {
                    "ticker": ["AAA"],
                    "year": [2024],
                    "report_type": ["consolidated"],
                },
                top_n=5,
                client=client,
                collection_name="tables",
            )

        self.assertEqual(
            [item["table_id"] for item in result],
            ["AAA_financial_statements_2024_consolidated_table_1"],
        )
        self.assertEqual(result[0]["bm25_rank"], 1)
        self.assertGreater(result[0]["bm25_score"], 0)
        self.assertGreater(len(client.calls), 1)
        self.assertTrue(all(call["collection_name"] == "tables" for call in client.calls))
        self.assertTrue(all(call["with_vectors"] is False for call in client.calls))

    def test_unicode_tokenizer_matches_vietnamese_without_diacritics(self) -> None:
        client = _ScrollClient(
            [
                payload(
                    "AAA_financial_statements_2024_consolidated_table_1",
                    "Chỉ tiêu: Tiền và các khoản tương đương tiền",
                )
            ]
        )

        result = search_bm25(
            "tien tuong duong",
            {"ticker": ["AAA"]},
            top_n=1,
            client=client,
            collection_name="tables",
        )

        self.assertEqual(len(result), 1)

    def test_invalid_qdrant_payload_is_rejected(self) -> None:
        invalid = payload(
            "AAA_financial_statements_2024_consolidated_table_1",
            "Doanh thu thuần",
        )
        invalid.pop("index_text")

        with self.assertRaisesRegex(BM25IndexError, "invalid payload"):
            search_bm25(
                "doanh thu",
                {"ticker": ["AAA"]},
                top_n=1,
                client=_ScrollClient([invalid]),
                collection_name="tables",
            )

    def test_transport_failure_is_classified_as_transient(self) -> None:
        request = httpx.Request("POST", "https://qdrant.example/scroll")
        client = _ScrollClient([])
        with patch.object(
            client,
            "scroll",
            side_effect=httpx.ConnectError("temporary", request=request),
        ):
            with self.assertRaises(TransientBM25IndexError):
                search_bm25(
                    "doanh thu",
                    {"ticker": ["AAA"]},
                    top_n=1,
                    client=client,
                    collection_name="tables",
                )

    def test_zero_length_normalization_is_supported(self) -> None:
        client = _ScrollClient(
            [
                payload(
                    "AAA_financial_statements_2024_consolidated_table_1",
                    "Doanh thu thuần",
                )
            ]
        )

        with patch.dict("os.environ", {"BM25_B": "0"}):
            result = search_bm25(
                "doanh thu",
                {"ticker": ["AAA"]},
                top_n=1,
                client=client,
                collection_name="tables",
            )

        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
