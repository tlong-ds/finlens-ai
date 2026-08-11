# finlens-agent

Generate pandas code to answer natural language questions about financial statements.

## Prerequisite: Download dataset:

```git clone https://huggingface.co/datasets/AIGuruTinix/ViFinQA```

## Architecture 

```
metadata/
├── docs_metadata.json   # Document catalog (doc_id, doc_path, ticker, year)
└── tables_metadata.json # Table catalog (table_id, doc_id, table_type, semantic_fields, csv_path)

data/
├── table_001.csv        # Extracted tables
├── table_002.csv
└── ...

prepare.py -- Phase 1: text parsing, table extraction, metadata generation
query.py -- Phase 2: table retrieval, schema description, LLM, execution
sandbox.py -- safe code execution environment
```

## Usage

### Setup
```bash
pip install -r requirements.txt
```

