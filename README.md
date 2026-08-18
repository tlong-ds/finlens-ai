# finlens-agent

Trả lời câu hỏi tài chính ViFinQA bằng dense retrieval, LLM reranking và mã pandas được kiểm định.

## Prerequisite: Download dataset:

```git clone https://huggingface.co/datasets/AIGuruTinix/ViFinQA```

## Architecture

```
metadata/
├── docs_metadata.json   # Document catalog (doc_id, doc_path, ticker, year)
└── tables_metadata.json # Table catalog (table_id, doc_id, ticker, company_name, table_type, semantic_fields, csv_path)

data/
├── table_001.csv        # Extracted tables
├── table_002.csv
└── ...

prepare.py -- Phase 1: text parsing, table extraction, metadata generation
src/graph.py -- supported compiled workflow: retrieval, generation, validation, execution
src/nodes.py -- graph node implementations
src/contracts.py -- shared Qdrant payload schema and safe CSV resolution
src/routing.py -- strict Vietnamese metadata routing and semantic-query cleanup
src/helper.py -- graph validation and routing helpers
src/prompt.py -- Vietnamese graph prompt templates and builders
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

Chạy `prepare.py` và `data_indexing.py` trước để collection Qdrant, manifest JSONL và
các file `data/{table_id}.csv` được sinh từ cùng một phiên dữ liệu. Collection phải dùng
named vector `dense` 384 chiều. Qdrant payload có đúng bảy trường: `table_id`, `doc_id`,
`ticker`, `company_name`, `year`, `report_type`, `table_type`. Manifest cung cấp
`index_text` cho LLM reranker; graph không chờ `index_text` hoặc `csv_path` trong payload.

Configure the embedding provider, Qdrant collection, `QDRANT_MANIFEST_PATH`, and
OpenAI-compatible LLM endpoint, then invoke the compiled graph with an exact canonical
ViFinQA question:

```dotenv
QDRANT_COLLECTION=finlens_tables_metadata_granite_97m_multilingual_r2_v1
QDRANT_ALIAS=finlens_tables_current
EMBEDDING_MODEL=ibm-granite/granite-embedding-97m-multilingual-r2
EMBEDDING_REVISION=835ad14087e140460703cf0fae09f97d469d65c2
EMBEDDING_DEVICE=auto
EMBEDDING_MAX_LENGTH=512
```

The default Granite multilingual encoder uses 384-dimensional normalized dense
vectors. Index metadata and queries are encoded without model-specific prefixes.
The current V1 collection uses Vietnamese-only metadata for `index_text`: the
Vietnamese table type label, Vietnamese semantic summary/keywords, and
`canonical_name_vi`. English canonical names are excluded. After changing this
contract, run `python data_indexing.py index --rebuild-manifest --force` during
a maintenance window, then reconcile stale points and verify the collection
before serving queries.

Qdrant payloads include both `ticker` and the canonical `company_name` from
`ViFinQA/code_stock.csv`; both fields are keyword-indexed and can be used as exact
metadata filters.

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

Routing yêu cầu resolve được ít nhất một ticker và một năm từ 2015–2025; graph không
fallback sang global search. Dense retrieval lấy Top-50, LLM dùng cùng cấu hình `.env`
xếp hạng theo batch và chọn Top-10. Nếu manifest lệch payload Qdrant hoặc LLM reranker
vẫn trả sai schema sau hai lần sửa, graph dừng với lỗi rõ ràng thay vì âm thầm dùng dense score.
