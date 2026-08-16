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
src/graph.py -- supported compiled workflow: retrieval, generation, validation, execution
src/nodes.py -- graph node implementations
src/helper.py -- graph validation and routing helpers
src/prompt.py -- graph prompt templates and builders
src/sandbox.py -- isolated code execution environment
query.py -- deprecated retrieval-only wrapper
```

## Usage

### Setup
```bash
pip install -r requirements.txt
```

`src/sandbox.py` executes generated pandas code inside an isolated
[E2B](https://e2b.dev) Firecracker microVM with outbound internet disabled. DataFrames
are transferred with dtype metadata, and only a bounded, validated numeric JSON result
returns to the host. Create an E2B account, obtain an API key, and export it:
```bash
export E2B_API_KEY=e2b_***
```

Configure the embedding provider, Qdrant collection, and OpenAI-compatible LLM endpoint,
then invoke the compiled graph with a canonical ViFinQA question:

```python
from src.graph import graph

result = graph.invoke({"question": query_text, "max_attempts": 5})
answer = result["answer_record"]
```

`max_attempts` defaults to `5` and accepts values from `1` through `5`. The final
`answer_record` contains the numeric answer, evidence paths, relevant documents and
tables, and the accepted pandas query. Transient LLM and Qdrant failures are retried
up to three times with backoff without consuming a semantic generation attempt.
`query.py` remains only as a deprecated retrieval-only compatibility wrapper; new
callers should use the compiled graph.
