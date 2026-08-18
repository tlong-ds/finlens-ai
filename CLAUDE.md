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

result = graph.invoke({"question": query_text, "max_attempts": 5})
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
  table text with `BAAI/bge-m3` (via FlagEmbedding), and upserts dense vectors + payload
  (`table_id`, `doc_id`, `ticker`, `year`, `report_type`, `table_type`) into Qdrant, aliased for
  atomic collection swaps. It also exposes `retrieve`/`route`/`resolve` for testing retrieval
  without going through the LangGraph pipeline.

### Online graph (`src/`)

`src/graph.py` compiles a `langgraph.graph.StateGraph` over a single `RetrievalState`
TypedDict, threaded through these nodes (`src/nodes.py`):

```
match_question -> parse_query -> retrieve_tables -> rerank_tables -> load_tables -> generate_code
                                                                                        |  ^
                                                                                        v  |
                                                                                  validate_code
                                                                                        |
                                                                                        v
                                                                                  execute_code -> END
```

- `match_question_node` resolves free text to exactly one canonical question record from
  `ViFinQA/questions/questions.jsonl` and initializes `attempt`/`feedback`/`max_attempts`.
- `parse_query_node` asks the LLM for conservative metadata filters (ticker/year/report_type/
  table_type); `src/helper.py:validate_filters` locally drops anything malformed or not in the
  allowed vocab before it ever reaches Qdrant.
- `retrieve_tables_node` / `rerank_tables_node` call `src/retrieval.py`'s `retrieve()` (Qdrant
  dense search) and `rerank()`. **Both are currently stubs**: `embed_query()` raises
  `RetrievalError` unconditionally (no embedding provider wired in yet), and `rerank()` is just
  a retrieval-score sort placeholder — replace the scoring block once a real
  (question, table) reranker is chosen; the function contract and graph don't need to change.
- `load_tables_node` reads the retrieved tables' CSVs into DataFrames (aliased `df_1`, `df_2`,
  ...) and builds a compact JSON schema description (columns, dtypes, sample rows) for the
  generator prompt.
- `generate_code_node` / `validate_code_node` / `execute_code_node` form a bounded retry loop
  using `langgraph.types.Command` for explicit routing: the generator LLM proposes pandas code
  + `evidence_variables` (which DataFrame aliases were actually used), a second LLM pass
  validates it without executing it, and only validated code reaches the sandbox. Any failure
  at any stage (bad LLM JSON, unknown alias, validator rejection, sandbox error, non-numeric
  result) sets `feedback` and routes back to `generate_code` via
  `src/helper.py:retry_or_exhausted` — until `attempt >= max_attempts`, at which point the loop
  routes to `execute_code`/raises instead of looping forever. Fresh code is generated on every
  retry (feedback is fed back into the prompt); nothing is cached across attempts.
- On success, `execute_code_node` builds `answer_record`: the numeric answer, the pandas query,
  and evidence (source CSV paths, doc IDs, `doc_id|table_N` table refs) derived from
  `evidence_variables`, so every accepted answer carries traceable provenance.

Transient failures are handled at the graph level, not inside nodes: `parse_query`,
`generate_code`, and `validate_code` are wrapped with a `RetryPolicy` retrying
`LLMTransientError` (`src/llm.py`), and `retrieve_tables` retries `TransientRetrievalError`
(`src/retrieval.py`) — both up to 3 attempts with backoff, and neither consumes a semantic
`attempt` from `max_attempts`.

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
