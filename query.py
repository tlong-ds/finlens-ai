"""Phase 2: Retrieve tables, describe schema, generate code, execute, format result."""

from typing import Any
import pandas as pd


def query(question: str, table_ids: list[str] | None = None) -> str:
    """
    Answer a question using text-to-pandas pipeline.

    Args:
        question: Natural language question about financial data
        table_ids: Optional list of specific table IDs to use. If None, retrieve tables via semantic search.

    Returns:
        String answer to the question
    """
    pass


def retrieve_tables(question: str, table_ids: list[str] | None = None) -> list[str]:
    """
    Retrieve relevant table IDs for a question.

    Args:
        question: Natural language question
        table_ids: If provided, use these directly. Otherwise perform semantic search.

    Returns:
        List of relevant table_ids
    """
    pass


def load_tables(table_ids: list[str]) -> dict[str, pd.DataFrame]:
    """
    Load tables from CSV files into DataFrames.

    Args:
        table_ids: List of table IDs to load

    Returns:
        Dict mapping table_id to DataFrame
    """
    pass


def describe_schema(dataframes: dict[str, pd.DataFrame]) -> str:
    """
    Describe schema of DataFrames: columns, dtypes, sample rows.

    Args:
        dataframes: Dict of {table_id: DataFrame}

    Returns:
        String description of schema
    """
    pass


def generate_code(question: str, schema_description: str) -> str:
    """
    Call LLM to generate pandas code for the question.

    Args:
        question: Natural language question
        schema_description: Description of available tables and columns

    Returns:
        Pandas code snippet as string
    """
    pass


def format_result(value: Any) -> str:
    """
    Format result value (scalar, Series, DataFrame) to readable string.

    Args:
        value: Result from executed pandas code

    Returns:
        Formatted string representation
    """
    pass
