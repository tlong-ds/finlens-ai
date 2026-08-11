"""Phase 1: Parse OCR files, extract tables, generate metadata."""

from pathlib import Path
from typing import Any


def prepare(financial_statements_dir: str) -> None:
    """
    Run Phase 1: parse OCR .txt files, extract tables, save CSVs, generate metadata.

    Args:
        financial_statements_dir: Path to directory containing OCR .txt files
                                  (e.g., financial_statements/TICKER/YEAR/*.txt)
    """
    pass


def parse_ocr_files(dir: str) -> list[tuple[str, dict[str, Any], str, list[str]]]:
    """
    Parse OCR files and extract tables.

    Args:
        dir: Path to financial_statements directory

    Returns:
        List of (table_id, table_data_dict, table_type, semantic_fields)
    """
    pass


def save_tables_as_csv(tables: list[tuple[str, dict[str, Any], str, list[str]]]) -> None:
    """
    Save extracted tables to CSV files in data/ directory.

    Args:
        tables: List of (table_id, table_data_dict, table_type, semantic_fields)
    """
    pass


def generate_metadata(
    tables: list[tuple[str, dict[str, Any], str, list[str]]],
    docs: list[dict[str, Any]]
) -> None:
    """
    Generate docs_metadata.json and tables_metadata.json.

    Args:
        tables: List of (table_id, table_data_dict, table_type, semantic_fields)
        docs: List of document metadata dicts
    """
    pass
