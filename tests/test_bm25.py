from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.bm25 import search_bm25


def manifest_point(
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
        "record_type": "point",
        "table_id": table_id,
        "index_text": index_text,
        "payload": {
            "table_id": table_id,
            "doc_id": doc_id,
            "ticker": ticker,
            "company_name": f"Công ty {ticker}",
            "year": year,
            "report_type": report_type,
            "table_type": "note_table",
            "start_line": start_line,
        },
    }


class BM25Tests(unittest.TestCase):
    def test_search_uses_index_text_and_metadata_filters(self) -> None:
        records = [
            {"record_type": "header"},
            manifest_point(
                "AAA_financial_statements_2024_consolidated_table_1",
                "Chỉ tiêu: Doanh thu thuần bán hàng và cung cấp dịch vụ",
            ),
            manifest_point(
                "AAA_financial_statements_2024_consolidated_table_2",
                "Chỉ tiêu: Tiền và các khoản tương đương tiền",
                start_line=20,
            ),
            manifest_point(
                "BBB_financial_statements_2024_consolidated_table_1",
                "Chỉ tiêu: Doanh thu thuần",
                ticker="BBB",
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                encoding="utf-8",
            )
            index = root / "bm25.sqlite3"

            result = search_bm25(
                "doanh thu thuần",
                {
                    "ticker": ["AAA"],
                    "year": [2024],
                    "report_type": ["consolidated"],
                },
                top_n=5,
                manifest_path=manifest,
                index_path=index,
            )

        self.assertEqual(
            [item["table_id"] for item in result],
            ["AAA_financial_statements_2024_consolidated_table_1"],
        )
        self.assertEqual(result[0]["bm25_rank"], 1)
        self.assertGreater(result[0]["bm25_score"], 0)

    def test_unicode_tokenizer_matches_vietnamese_without_diacritics(self) -> None:
        records = [
            {"record_type": "header"},
            manifest_point(
                "AAA_financial_statements_2024_consolidated_table_1",
                "Chỉ tiêu: Tiền và các khoản tương đương tiền",
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                encoding="utf-8",
            )
            result = search_bm25(
                "tien tuong duong",
                {"ticker": ["AAA"]},
                top_n=1,
                manifest_path=manifest,
                index_path=root / "bm25.sqlite3",
            )

        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
