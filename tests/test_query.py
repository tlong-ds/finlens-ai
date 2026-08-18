from __future__ import annotations

import unittest
from unittest.mock import patch

from query import query


class DeprecatedQueryWrapperTests(unittest.TestCase):
    def test_query_keeps_returning_retrieved_tables(self) -> None:
        tables = [{"table_id": "example_table_1"}]
        with patch("query.graph.invoke", return_value={"retrieved_tables": tables}):
            self.assertEqual(query("Câu hỏi benchmark"), tables)


if __name__ == "__main__":
    unittest.main()
