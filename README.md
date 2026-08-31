# FinLens AI

FinLens answers Vietnamese financial-statement questions from the ViFinQA
benchmark. It retrieves the relevant tables, selects the minimum evidence set,
plans the required rows and calculations, generates pandas code, and executes the
code in a network-disabled E2B sandbox.

## Setup

The project uses Python 3.12.11 and `pyproject.toml` as its only dependency
declaration. `uv.lock` is committed for reproducible installs.

```bash
uv sync --frozen --group dev
```

Copy the ViFinQA dataset into `ViFinQA/`. Configure the OpenAI-compatible LLM,
Qdrant, embedding, FPT reranker, and E2B providers in `.env`. All configuration is
loaded once through `src.config.Settings`; provider modules do not read the
environment themselves.

Run the offline preparation and indexing phases before invoking the graph:

```bash
python scripts/prepare.py --help
python scripts/data_indexing.py self-test
python scripts/data_indexing.py index --help
```

`self-test` performs eleven local checks and makes no model or Qdrant call.

## Pipeline API

The canonical API lives in `src.pipeline`:

```python
from src.config import Settings
from src.pipeline import PipelineDependencies, build_graph

settings = Settings.from_env()
pipeline = build_graph(settings, PipelineDependencies.from_settings(settings))
result = pipeline.invoke({"question": query_text, "max_attempts": 2})
answer = result["answer_record"]
```

`from src.pipeline import graph` provides a default compiled graph for interactive
use. `max_attempts` defaults to two and accepts values from one through five.

The graph order is:

```text
match_question -> parse_query -> retrieve_tables -> rerank_tables
    -> select_tables -> load_tables -> plan_generation_context
    -> generate_code -> execute_code
```

Generation and execution form a bounded retry loop. Selection owns table choice
and DataFrame aliases. Planning may only map the selected tables to rows, columns,
units, and calculations; it cannot replace or drop aliases.

## Retrieval and payload contract

Dense retrieval and runtime BM25 are fused with reciprocal-rank fusion. The FPT
reranker and LLM selector retain their existing candidate limits, strict coverage
checks, and failure behavior. Runtime retrieval never reads the offline manifest.

Every Qdrant point has exactly these nine payload fields:

```text
table_id, doc_id, ticker, company_name, year, report_type,
table_type, start_line, index_text
```

The collection uses the named dense vector `dense`. CSV paths are resolved from
validated `table_id` values beneath the project `data/` directory.

## Executable programs

All programs live under `scripts/`:

```bash
python scripts/doctor.py
python scripts/prepare.py --help
python scripts/data_indexing.py --help
python scripts/generate_submission.py --help
python scripts/run_valset.py --help
python scripts/run_parser_valset.py --help
python scripts/run_reranker_valset.py --help
```

Submission and validation outputs are experiment runs under `submission/runs/`
or `val_submission/`. Every newly created run includes a credential-safe,
path-portable `manifest.json` with source state, lock/question hashes, arguments,
tolerances, retry and concurrency settings, provider identities, payload schema,
and prompt/pipeline hashes.

Generated runs, logs, pointers, locks, caches, ZIP archives, `.DS_Store`, and
graphify output are ignored. Existing local historical artifacts are not deleted.

## Repository layout

```text
src/
  pipeline/        graph, state, and lifecycle nodes
  retrieval/       dense, BM25, context, reranking, selection, routing
  generation/      planning, normalization, and prompts
  providers/       explicitly configured external adapters
  data/            preparation and indexing services
  experiments/     run storage, checkpoints, metrics, provenance, packaging
scripts/            executable CLI programs
```

See [Architecture](docs/architecture.md) and
[Reproducibility](docs/reproducibility.md) for the detailed contracts.
