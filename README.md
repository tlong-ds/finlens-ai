# text2pandas

Generate pandas code to answer natural language questions about financial statements.

## Architecture

### Phase 1: Preparation (one-time, offline)
- Parse OCR `.txt` files from `financial_statements/`
- Extract `<table>` HTML blocks
- Normalize and save as CSV files to `data/`
- Generate metadata files (`metadata/docs_metadata.json`, `metadata/tables_metadata.json`)

### Phase 2: Query (runtime, per question)
- Retrieve relevant tables (semantic search or manual selection)
- Load tables from `data/*.csv`
- Describe schema to LLM
- Generate pandas code
- Execute code in restricted sandbox
- Format and return answer

## Usage

### Setup
```bash
pip install -r requirements.txt
```

### Preparation
```python
from text2pandas import prepare

prepare("financial_statements")
```

### Querying
```python
from text2pandas import query

# Automatic table retrieval via semantic search
answer = query("What is the total revenue in 2025?")
print(answer)

# Manual table selection
answer = query(
    "Compare profit margins",
    table_ids=["table_001", "table_002"]
)
print(answer)
```

## Module Structure

```
text2pandas/
├── __init__.py          # Public API
├── _prepare.py          # Phase 1: OCR parsing, table extraction, metadata generation
├── _query.py            # Phase 2: Table retrieval, schema description, LLM, execution
└── _sandbox.py          # Safe code execution environment
```

## Data Structure

```
metadata/
├── docs_metadata.json   # Document catalog (doc_id, doc_path, ticker, year)
└── tables_metadata.json # Table catalog (table_id, doc_id, table_type, semantic_fields, csv_path)

data/
├── table_001.csv        # Extracted tables
├── table_002.csv
└── ...
```

## API Reference

### `prepare(financial_statements_dir: str) -> None`
Parse OCR files and generate metadata/data.

### `query(question: str, table_ids: list[str] | None = None) -> str`
Answer a natural language question about financial data.

**Parameters:**
- `question`: The question to answer
- `table_ids`: Optional list of specific tables to use. If None, uses semantic search.

**Returns:** String answer
