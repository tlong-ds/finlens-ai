# FinLens project constraints

- Python is pinned to 3.12.11. Dependencies belong only in `pyproject.toml`; keep
  `uv.lock` and use the Ruff-only `dev` group.
- `src` is the canonical package. Do not add flat-module compatibility shims or a
  second package facade. Executable programs belong in `scripts/`.
- Load `src.config.Settings` once at an application boundary and pass it explicitly
  to providers. Never add provider-level `load_dotenv()` or `os.getenv()` calls.
- Preserve the graph lifecycle, two-attempt default, nine-field Qdrant payload,
  retrieval/reranking/selection limits, planner boundary, sandbox boundary, and
  submission schemas unless a task explicitly changes those contracts.
- Selection owns tables and aliases. Planning only maps selected evidence to rows,
  columns, units, and calculations.
- Generated pandas code runs only in the secure, network-disabled E2B adapter.
- Run artifacts are credential-safe and path-portable. Generated runs, logs,
  pointers, locks, caches, ZIPs, `.DS_Store`, and graphify output stay untracked.
- There is intentionally no `tests/` suite. Use the offline acceptance checks in
  [Reproducibility](docs/reproducibility.md); do not recreate pytest or unittest
  coverage without an explicit request.

Project documentation:

- [README](README.md)
- [Architecture](docs/architecture.md)
- [Reproducibility](docs/reproducibility.md)
