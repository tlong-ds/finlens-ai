"""Credential-safe, path-portable experiment manifests."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.config import Settings
from src.contracts import PAYLOAD_FIELDS, PAYLOAD_SCHEMA_VERSION
from src.experiments.storage import sha256_file, write_json


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=root, check=False, capture_output=True, text=True
    )
    return result.stdout.strip()


def _hash_sources(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        if path.is_file():
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def write_run_manifest(
    run_dir: Path,
    *,
    settings: Settings,
    arguments: Mapping[str, Any],
    question_files: Sequence[Path] = (),
    tolerances: Mapping[str, Any] | None = None,
    retries: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the complete reproducibility envelope before any provider call."""
    root = settings.project_root
    lock_path = root / "uv.lock"
    source_paths = [
        *sorted((root / "src/pipeline").rglob("*.py")),
        *sorted((root / "src/retrieval").rglob("*.py")),
        *sorted((root / "src/generation").rglob("*.py")),
    ]
    manifest = {
        "schema_version": 1,
        "git": {
            "commit": _git(root, "rev-parse", "HEAD") or None,
            "branch": _git(root, "branch", "--show-current") or None,
            "dirty": bool(_git(root, "status", "--porcelain")),
            "diff_sha256": hashlib.sha256(
                _git(root, "diff", "--binary", "HEAD").encode()
            ).hexdigest(),
        },
        "hashes": {
            "uv_lock": sha256_file(lock_path) if lock_path.is_file() else None,
            "questions": {
                path.name: sha256_file(path)
                for path in question_files
                if path.is_file()
            },
            "pipeline_and_prompts": _hash_sources(source_paths),
        },
        "arguments": json.loads(json.dumps(dict(arguments), default=str)),
        "tolerances": dict(tolerances or {}),
        "retries_and_concurrency": dict(retries or {}),
        "providers": {
            "llm": {"base_url": settings.llm_base_url, "model": settings.llm_model},
            "qdrant": {
                "collection": settings.qdrant_collection,
                "alias": settings.qdrant_alias,
            },
            "embedding": {
                "model": settings.embedding_model,
                "revision": settings.embedding_revision,
            },
            "reranker": {
                "endpoint": settings.fpt_rerank_url,
                "model": settings.fpt_rerank_model,
            },
            "sandbox": "e2b",
        },
        "payload": {
            "schema_version": PAYLOAD_SCHEMA_VERSION,
            "fields": list(PAYLOAD_FIELDS),
        },
        "settings": settings.public_dict(),
    }
    write_json(run_dir / "manifest.json", manifest)
    return manifest
