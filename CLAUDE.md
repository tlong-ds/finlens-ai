# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

FinLens generates pandas code to answer natural-language questions about Vietnamese financial
statements (the ViFinQA dataset). It is a two-phase pipeline:

1. **Offline data prep / indexing** (`prepare.py`, `data_indexing.py`) — parse OCR'd financial
   statements into tables, build metadata catalogs, embed table descriptions, and upsert them
   into Qdrant.
2. **Online retrieval + answer generation** (`src/graph.py` and friends) — a compiled LangGraph
   workflow that takes a canonical ViFinQA question, retrieves the relevant tables, generates
   pandas code with an LLM, validates it with a second LLM pass, and executes it in an isolated
   E2B sandbox to produce a single numeric answer.

## Setup

```bash
pip install -r requirements.txt
```

Required environment (see `.env.example`, loaded via `python-dotenv` / `.envrc` + `direnv`):
- `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`, `QDRANT_TIMEOUT` — Qdrant Cloud connection.
- `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `LLM_TIMEOUT`, `LLM_TEMPERATURE` — any
  OpenAI-compatible chat completions endpoint.
- `E2B_API_KEY` — required by `src/sandbox.py` to create the isolated microVM used for code
  execution.

The ViFinQA dataset is a separate git repo/submodule, not vendored: `git clone
https://huggingface.co/datasets/AIGuruTinix/ViFinQA`. It's gitignored here
(`ViFinQA/`), as are the generated `data/` and `metadata/` directories.

## Common commands

There is no configured test runner, linter, or formatter in this repo (no pytest/ruff/black
config). `data_indexing.py` has its own dependency-free self-check:

```bash
python data_indexing.py self-test     # in-file checks, no Qdrant/model calls
python data_indexing.py doctor        # verify local inputs, embedding model, Qdrant reachability
```

`data_indexing.py` is a single-file CLI (argparse subcommands) covering the whole offline
pipeline:

```bash
python data_indexing.py build-manifest [--ticker T] [--year Y] [--limit N]   # stream metadata -> JSONL manifest
python data_indexing.py stats                                                # manifest statistics
python data_indexing.py init-collection                                      # create/validate Qdrant collection
python data_indexing.py upsert [--resume | --force] [--ticker T] [--year Y]  # embed manifest, upload to Qdrant
python data_indexing.py verify [--sample-size N]                             # compare manifest vs Qdrant
python data_indexing.py index [--force] [--rebuild-manifest]                 # full pipeline: build-manifest+upsert+verify
python data_indexing.py retrieve "<question>" [--ticker T]... [--load]       # ad hoc retrieval, bypassing the graph
python data_indexing.py route "<question>"                                   # inspect query buckets, no model/Qdrant calls
python data_indexing.py resolve <table_id>                                   # table_id -> local CSV path
python data_indexing.py reconcile [--dry-run | --prune]                      # diff manifest IDs vs Qdrant IDs
```

`prepare.py` runs Phase 1 (OCR text parsing, table extraction, metadata generation) over
`ViFinQA/financial_statements` and writes into `data/` and `metadata/`.

Invoking the retrieval+answer graph directly:

```python
from src.graph import graph

result = graph.invoke({"question": query_text, "max_attempts": 1})
answer = result["answer_record"]
```

`question` must match exactly one canonical question in `ViFinQA/questions/questions.jsonl`
(see `src/helper.py:find_question`) — the graph is built to answer the fixed ViFinQA benchmark,
not arbitrary free text. `max_attempts` must be between 1 and 5.

## Architecture

### Offline pipeline (`prepare.py`, `data_indexing.py`)

- `prepare.py` parses OCR'd financial-statement text/HTML into structured tables (balance
  sheet, income statement, cash flow, note tables), classifying report type (consolidated /
  separate / standalone) and entity type (TCTD/CTCK/BH — bank/securities/insurance, which use
  different accounting chart-of-account layouts) via regex heuristics tuned for OCR artifacts in
  Vietnamese text. Output: `metadata/docs_metadata.json`, `metadata/tables_metadata.json`, and
  per-table CSVs under `data/`.
- `data_indexing.py` is a large, mostly self-contained CLI that turns those tables into a
  searchable Qdrant collection: it builds a manifest (JSONL) from the metadata catalog with a
  SQLite checkpoint (`.cache/qdrant_sync_v1.sqlite3`) for idempotent/resumable runs, embeds
  table text with the configured embedding model and upserts dense vectors + payload
  (`table_id`, `doc_id`, `ticker`, `company_name`, `year`, `report_type`, `table_type`,
  `start_line`, `index_text`) into Qdrant, aliased for atomic collection swaps. It also exposes
  `retrieve`/`route`/`resolve` for testing retrieval without going through the LangGraph
  pipeline.

### Online graph (`src/`)

`src/graph.py` compiles a `langgraph.graph.StateGraph` over a single `RetrievalState`
TypedDict, threaded through these nodes (`src/nodes.py`):

```
match_question -> parse_query -> retrieve_tables -> rerank_tables -> select_tables
                                                                      |
                                                                      v
                     END <- execute_code <-> generate_code <- plan_generation_context <- load_tables
```

- `match_question_node` resolves free text to exactly one canonical question record from
  `ViFinQA/questions/questions.jsonl` and initializes `attempt`/`feedback`/`max_attempts`.
- `parse_query_node` asks the LLM for conservative metadata filters (ticker/year/report_type/
  table_type); `src/helper.py:validate_filters` locally drops anything malformed or not in the
  allowed vocab before it ever reaches Qdrant.
- `retrieve_tables_node` calls `src/retrieval.py`'s `retrieve()` (Qdrant dense search, optionally
  fused with BM25) and returns a balanced top 80. Retrieval embeds the semantic query
  with the shared Granite encoder and validates the exact nine-field Qdrant payload. In hybrid
  mode, BM25 scrolls only payloads matching the metadata filters and scores their `index_text`
  at query time; it does not read a local manifest or SQLite lexical index. The table-ranking
  stage resolves each candidate CSV from `table_id` and builds bounded question-aware context
  from its columns and relevant rows. `rerank_tables_node` sends that same bounded context
  contract used by the benchmark to FPT `bge-reranker-v2-m3` and keeps top 20. The call is strict,
  retries transient failures three times, and has no quality-degrading fallback. `select_tables_node`
  then uses two LLM scouts plus a final high-recall selector to prune for the planner with a hard
  cap of 18. There is no heuristic 30-table shortlist. The offline manifest is not read at runtime.
- `load_tables_node` reads the retrieved tables' CSVs into DataFrames (aliased `df_1`, `df_2`,
  ...) and builds a compact JSON schema description (columns, dtypes, sample rows) for the
  generator prompt.
- `generate_code_node` / `execute_code_node` form a bounded retry loop using
  `langgraph.types.Command` for explicit routing: the generator LLM proposes pandas code +
  `evidence_variables` (which DataFrame aliases were actually used), then the sandbox executes
  it. Any failure (bad LLM JSON, unknown alias, sandbox error, non-numeric result) sets
  `feedback` and routes back to `generate_code` via
  `src/helper.py:retry_or_exhausted` — until `attempt >= max_attempts`, at which point the loop
  routes to `execute_code`/raises instead of looping forever. Fresh code is generated on every
  retry (feedback is fed back into the prompt); nothing is cached across attempts.
- On success, `execute_code_node` builds `answer_record`: the numeric answer, the pandas query,
  and evidence (source CSV paths, doc IDs, `doc_id|start_line` table refs) derived from
  `evidence_variables`, so every accepted answer carries traceable provenance.

LLM/Qdrant transient failures are handled at the graph level: `parse_query`, `select_tables`, and
`generate_code` are wrapped with a `RetryPolicy` retrying
`LLMTransientError` (`src/llm.py`), and `retrieve_tables` retries `TransientRetrievalError`
(`src/retrieval.py`) — both up to 3 attempts with backoff, and neither consumes a semantic
`attempt` from `max_attempts`. The FPT reranker performs its own three bounded transient retries
so that exhausted questions fail and can be resumed by the experiment runners.

### Sandboxed execution (`src/sandbox.py`)

Generated pandas code never runs in-process. `src/sandbox.py:run_code` spins up a fresh,
network-disabled, `secure=True` E2B Firecracker microVM per call: DataFrames are serialized
with explicit dtype metadata and reloaded inside the VM, the candidate code runs, and a
*separate* serialization step writes a strictly-shaped, size-capped
(`_MAX_RESULT_BYTES`) JSON payload (`{"status": "ok", "value": <finite float>}` or an
`"invalid"`/`"missing"` status) that is the only thing read back into the host process. Treat
this boundary as a security boundary: don't widen what crosses back (e.g. don't return
DataFrames/objects), and don't relax `secure=True` / `allow_internet_access=False`.

There is a second, unrelated `sandbox.py` at the repo root — it's an older pickle-based
prototype, not imported by anything (`src/nodes.py` imports `src.sandbox`, not root
`sandbox`). Don't confuse the two; prefer `src/sandbox.py` for any sandbox changes.

### Deprecated

`query.py` is a deprecated retrieval-only wrapper around `graph.invoke` (returns just
`retrieved_tables`, no answer generation). New callers should invoke `src.graph.graph` directly.

## Official submission format

Submit one ZIP archive through **My Submissions** at
`http://leaderboard.aiguru.com.vn/`. The archive must contain exactly one result JSON file and a
`data/` directory at the archive root; do not wrap them in another parent directory:

```text
submission.zip
├── submission.json
└── data/
    ├── <table_1>.csv
    ├── <table_2>.csv
    └── ...
```

`submission.json` must be a JSON array. Every included prediction must have exactly this shape
(the placeholders below describe types and are not literal JSON):

```json
[
  {
    "id": 1,
    "question": "<financial question>",
    "answer": 63075000000.0,
    "relevant_docs": ["AAA_financial_statements_2015_consolidated"],
    "relevant_tables": ["AAA_financial_statements_2015_consolidated|350"],
    "evidence": [
      {
        "variable": "df_1",
        "csv_path": "data/AAA_financial_statements_2015_consolidated_table_1.csv"
      }
    ],
    "pandas_query": "result = df_1.loc[df_1['year'] == 2023, 'net_revenue'].iloc[0]"
  }
]
```

Field contract:

- `id` is the integer question ID, and `question` is the canonical question text.
- `answer` is a finite floating-point result.
- `relevant_docs` contains the relevant report IDs. A report ID is the final filename component
  with its `.txt` extension removed.
- `relevant_tables` contains directly relevant tables in `<report_id>|<start_line>` form, where
  `start_line` is the table's starting line in the organizer-provided OCR report.
- `evidence` lists every CSV DataFrame needed to run `pandas_query`. Each `variable` must be a
  unique valid Python identifier within that question and must match the DataFrame name used by
  the query. Each `csv_path` must be a relative path beginning with `data/`, and the referenced
  file must exist in the ZIP.
- `pandas_query` is a pandas statement or program that can be rerun against only the declared
  evidence DataFrames and reproduces `answer` as a numeric scalar. Do not declare unused evidence,
  reference undeclared DataFrames, or submit constant placeholder answers unrelated to evidence.

Before calling a run ready, validate the JSON schema and types, unique IDs and evidence variable
names, canonical question text, finite answers, `relevant_docs`/`relevant_tables` provenance,
presence of every referenced CSV, replayability of every `pandas_query`, and the ZIP's root-level
layout. For repository readiness checks, fewer than 1,000 predictions is not by itself a blocker;
validate all records that are present unless the user explicitly requests a completeness check.
Missing or malformed records and missing evidence files are not valid predictions.
