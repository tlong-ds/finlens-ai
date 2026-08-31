"""Qdrant infrastructure for the Level 1 retrieval pipeline."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from qdrant_client import QdrantClient

if TYPE_CHECKING:
    from src.config import Settings


class QdrantConnectionError(RuntimeError):
    """Raised when Qdrant configuration or client creation fails."""


def _required_setting(name: str, value: str) -> str:
    value = value.strip()
    if not value:
        raise QdrantConnectionError(f"Missing required setting: {name}")
    return value


@lru_cache(maxsize=8)
def get_qdrant_client(settings: Settings) -> QdrantClient:
    """Create one lazily initialized client configured for Qdrant Cloud."""
    try:
        return QdrantClient(
            url=_required_setting("QDRANT_URL", settings.qdrant_url),
            api_key=_required_setting("QDRANT_API_KEY", settings.qdrant_api_key),
            timeout=settings.qdrant_timeout,
        )
    except QdrantConnectionError:
        raise
    except (TypeError, ValueError) as exc:
        raise QdrantConnectionError("Invalid Qdrant configuration") from exc


def get_collection_name(settings: Settings) -> str:
    """Return the read alias, falling back to the physical collection."""
    alias = settings.qdrant_alias.strip()
    if alias:
        return alias
    return _required_setting("QDRANT_COLLECTION", settings.qdrant_collection)
