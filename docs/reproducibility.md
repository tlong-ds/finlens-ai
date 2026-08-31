# Reproducibility

## Environment

The supported interpreter is Python 3.12.11. `pyproject.toml` is the sole dependency
source and `uv.lock` pins the resolved environment.

```bash
uv sync --frozen --group dev
uv lock --check
```

If dependency downloads are unavailable, do not regenerate or discard the lock;
use an already-synced environment for the offline checks and rerun the two commands
when network access is restored.

## Offline acceptance checks

These checks must not contact an LLM, Qdrant, FPT, an embedding registry, or E2B:

```bash
ruff check src scripts
ruff format --check src scripts
python -m compileall src scripts
python scripts/data_indexing.py self-test
python scripts/doctor.py
```

Also smoke-import every public package and the canonical graph API:

```bash
python - <<'PY'
import src.config
import src.contracts
import src.data.indexing
import src.data.preparation
import src.experiments
import src.generation
import src.pipeline
import src.providers
import src.retrieval
from src.pipeline import build_graph, graph
PY
```

Run `--help` for every file in `scripts/`. Search for imports of deleted flat
modules and root runners, and confirm generated paths are absent from `git ls-files`.
There is intentionally no pytest or unittest step.

## Run manifests

Each new submission or validation run writes a path-portable `manifest.json` with:

- Git commit, dirty state, and diff hash;
- `uv.lock`, question-source, prompt, and pipeline hashes;
- CLI arguments, tolerances, retry and concurrency settings;
- provider identities without credentials;
- the exact Qdrant payload schema;
- timestamps and other portable run identifiers.

Absolute local paths and API keys are excluded. Status files and question-level
checkpoints use atomic replacement so interrupted runs can resume without treating
partial writes as terminal results.

## Generated state

Experiment outputs stay on disk but out of version control. The ignore rules cover
run directories, latest-run pointers, logs, JSON locks, caches, ZIPs, `.DS_Store`,
and graphify output. Golden inputs and documentation assets remain tracked.
