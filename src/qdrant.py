"""Qdrant infrastructure for the Level 1 retrieval pipeline."""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from qdrant_client import QdrantClient


class QdrantConnectionError(RuntimeError):
    """Raised when Qdrant configuration or client creation fails."""


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise QdrantConnectionError(f"Missing required environment variable: {name}")
    return value


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """Create one lazily initialized client configured for Qdrant Cloud."""
    load_dotenv()
    try:
        return QdrantClient(
            url=_required_env("QDRANT_URL"),
            api_key=_required_env("QDRANT_API_KEY"),
            timeout=float(os.getenv("QDRANT_TIMEOUT", "30")),
        )
    except QdrantConnectionError:
        raise
    except (TypeError, ValueError) as exc:
        raise QdrantConnectionError("Invalid Qdrant configuration") from exc


def get_collection_name() -> str:
    """Return the read alias, falling back to the physical collection."""
    load_dotenv()
    alias = os.getenv("QDRANT_ALIAS", "").strip()
    if alias:
        return alias
    return _required_env("QDRANT_COLLECTION")
