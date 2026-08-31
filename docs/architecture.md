# Architecture

## Online pipeline

`src.pipeline.build_graph(settings, dependencies)` compiles the authoritative
LangGraph workflow. `PipelineDependencies.from_settings(settings)` binds settings
without monkey-patching globals.

```text
question
  -> match_question
  -> parse_query
  -> retrieve_tables
  -> rerank_tables
  -> select_tables
  -> load_tables
  -> plan_generation_context
  -> generate_code
  -> execute_code
  -> answer_record
```

The first generation attempt uses an OpenAI-compatible structured response. A bad
response, invalid selector, unsupported operation, sandbox failure, or non-numeric
result returns concise feedback to `generate_code` until `max_attempts` is reached.
The default is two attempts. Transient LLM and Qdrant failures use provider retry
policies and do not consume a semantic generation attempt.

The node responsibilities are intentionally separate:

- `src.pipeline.nodes.question` resolves a canonical ViFinQA question, parses
  strict metadata filters, and starts retrieval.
- `src.pipeline.nodes.tables` reranks, selects exact tables, assigns aliases,
  loads CSVs, and prepares the planner inventory.
- `src.pipeline.nodes.answer` generates normalized pandas code, executes it in
  E2B, and assembles the answer and evidence contract.
- `src.retrieval.selection` owns the selected tables and aliases. The generation
  planner consumes those choices and cannot reselect evidence.

## Retrieval

`src.retrieval` separates dense search, BM25, bounded CSV context, FPT reranking,
LLM selection, and metadata routing. Dense and lexical candidates are fused with
the existing reciprocal-rank algorithm. Candidate limits, prompts, strict coverage
enforcement, and output contracts remain unchanged.

Qdrant points use a named `dense` vector and the exact payload schema defined in
`src.contracts.QDRANT_PAYLOAD_FIELDS`:

```text
table_id, doc_id, ticker, company_name, year, report_type,
table_type, start_line, index_text
```

At runtime, CSV paths are derived from validated table IDs. Question-aware rerank
context is bounded and built directly from the candidate CSV files; the indexing
manifest is never consulted by the online graph.

## Offline data services

`src.data.preparation` parses OCR sources, classifies statements, normalizes table
content and taxonomy, and writes table/document metadata. `scripts/prepare.py` owns
CLI parsing and orchestration.

`src.data.indexing` builds the portable manifest, manages the collection and
aliases, records checkpoints, reconciles point IDs, and provides the indexing
service. `scripts/data_indexing.py` owns its subcommands.

## Providers and execution

Provider constructors receive one `Settings` instance. Only `src.config` reads
`.env` and process environment values. Public diagnostics redact credential values.

Generated code is serialized with its selected DataFrames to a fresh secure E2B
microVM with outbound internet disabled. Only a bounded JSON numeric result crosses
back into the host process.

## Experiments

`src.experiments` contains atomic writes, checkpoint/status transitions, metrics,
provenance hashing, submission validation, and ZIP packaging shared by the runner
programs. Each new run writes `manifest.json` before processing questions.
