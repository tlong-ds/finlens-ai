"""Validated, credential-safe FinLens runtime configuration."""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values

from src.providers.embeddings import (
    EMBEDDING_MAX_LENGTH,
    EMBEDDING_MODEL_DEFAULT,
    EMBEDDING_REVISION_DEFAULT,
)
from src.providers.fpt import DEFAULT_ENDPOINT, DEFAULT_MODEL, DEFAULT_TIMEOUT_SECONDS


class SettingsError(ValueError):
    """Raised when runtime configuration is incomplete or invalid."""


_SECRET_FIELDS = {"llm_api_key", "qdrant_api_key", "fpt_api_key", "e2b_api_key"}


def _as_int(name: str, value: str, *, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise SettingsError(f"{name} must be an integer") from exc
    if parsed < minimum:
        raise SettingsError(f"{name} must be at least {minimum}")
    return parsed


def _as_float(
    name: str, value: str, *, minimum: float = 0.0, strict: bool = True
) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise SettingsError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed) or (parsed <= minimum if strict else parsed < minimum):
        qualifier = "greater than" if strict else "at least"
        raise SettingsError(f"{name} must be finite and {qualifier} {minimum}")
    return parsed


@dataclass(frozen=True, slots=True)
class Settings:
    """Single settings object created at an application boundary and then injected."""

    project_root: Path
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "finlens_tables"
    qdrant_alias: str = "finlens_tables_current"
    qdrant_timeout: float = 30.0
    qdrant_manifest_path: Path = Path("intermediate/qdrant_manifest.jsonl")
    retrieval_mode: str = "hybrid"
    qdrant_bm25_scroll_batch: int = 512
    qdrant_bm25_max_documents: int = 20_000
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    embedding_model: str = EMBEDDING_MODEL_DEFAULT
    embedding_revision: str = EMBEDDING_REVISION_DEFAULT
    embedding_model_path: str = ""
    embedding_device: str = "auto"
    embedding_max_length: int = EMBEDDING_MAX_LENGTH
    embedding_batch_size: int = 32
    llm_base_url: str = "http://localhost:8000/v1"
    llm_api_key: str = ""
    llm_model: str = ""
    llm_timeout: float = 60.0
    llm_temperature: float = 0.0
    fpt_api_key: str = ""
    fpt_rerank_url: str = DEFAULT_ENDPOINT
    fpt_rerank_model: str = DEFAULT_MODEL
    fpt_rerank_timeout: float = DEFAULT_TIMEOUT_SECONDS
    e2b_api_key: str = ""
    tables_metadata_path: Path = Path("metadata/tables_metadata.json")
    stock_codes_path: Path = Path("ViFinQA/code_stock.csv")
    qdrant_rejects_path: Path = Path("intermediate/qdrant_rejects.jsonl")
    qdrant_state_path: Path = Path(".cache/qdrant_sync.sqlite3")
    upsert_batch_size: int = 256
    qdrant_upload_parallel: int = 1
    qdrant_upload_retries: int = 3
    qdrant_prefer_grpc: bool = True

    @classmethod
    def from_env(
        cls,
        *,
        project_root: Path | None = None,
        environ: Mapping[str, str] | None = None,
        env_file: Path | None = None,
        validate: bool = True,
    ) -> "Settings":
        root = (project_root or Path.cwd()).resolve()
        file_values = dotenv_values(env_file or root / ".env")
        merged = {
            key: str(value) for key, value in file_values.items() if value is not None
        }
        merged.update(dict(os.environ if environ is None else environ))

        def get(name: str, default: str = "") -> str:
            return str(merged.get(name, default)).strip()

        configured_root = Path(get("FINLENS_PROJECT_ROOT", str(root))).resolve()
        settings = cls(
            project_root=configured_root,
            qdrant_url=get("QDRANT_URL", "http://localhost:6333"),
            qdrant_api_key=get("QDRANT_API_KEY"),
            qdrant_collection=get("QDRANT_COLLECTION", "finlens_tables"),
            qdrant_alias=get("QDRANT_ALIAS", "finlens_tables_current"),
            qdrant_timeout=_as_float("QDRANT_TIMEOUT", get("QDRANT_TIMEOUT", "30")),
            qdrant_manifest_path=Path(
                get("QDRANT_MANIFEST_PATH", "intermediate/qdrant_manifest.jsonl")
            ),
            retrieval_mode=get("RETRIEVAL_MODE", "hybrid").lower(),
            qdrant_bm25_scroll_batch=_as_int(
                "QDRANT_BM25_SCROLL_BATCH", get("QDRANT_BM25_SCROLL_BATCH", "512")
            ),
            qdrant_bm25_max_documents=_as_int(
                "QDRANT_BM25_MAX_DOCUMENTS", get("QDRANT_BM25_MAX_DOCUMENTS", "20000")
            ),
            bm25_k1=_as_float("BM25_K1", get("BM25_K1", "1.2")),
            bm25_b=_as_float(
                "BM25_B", get("BM25_B", "0.75"), minimum=0.0, strict=False
            ),
            embedding_model=get("EMBEDDING_MODEL", EMBEDDING_MODEL_DEFAULT),
            embedding_revision=get("EMBEDDING_REVISION", EMBEDDING_REVISION_DEFAULT),
            embedding_model_path=get("EMBEDDING_MODEL_PATH"),
            embedding_device=get("EMBEDDING_DEVICE", "auto"),
            embedding_max_length=_as_int(
                "EMBEDDING_MAX_LENGTH",
                get("EMBEDDING_MAX_LENGTH", str(EMBEDDING_MAX_LENGTH)),
            ),
            embedding_batch_size=_as_int(
                "EMBED_BATCH_SIZE", get("EMBED_BATCH_SIZE", "32")
            ),
            llm_base_url=get("LLM_BASE_URL", "http://localhost:8000/v1"),
            llm_api_key=get("LLM_API_KEY"),
            llm_model=get("LLM_MODEL"),
            llm_timeout=_as_float("LLM_TIMEOUT", get("LLM_TIMEOUT", "60")),
            llm_temperature=_as_float(
                "LLM_TEMPERATURE",
                get("LLM_TEMPERATURE", "0"),
                minimum=0.0,
                strict=False,
            ),
            fpt_api_key=get("FPT_API_KEY"),
            fpt_rerank_url=get("FPT_RERANK_URL", DEFAULT_ENDPOINT),
            fpt_rerank_model=get("FPT_RERANK_MODEL", DEFAULT_MODEL),
            fpt_rerank_timeout=_as_float(
                "FPT_RERANK_TIMEOUT",
                get("FPT_RERANK_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS)),
            ),
            e2b_api_key=get("E2B_API_KEY"),
            tables_metadata_path=Path(
                get("TABLES_METADATA_PATH", "metadata/tables_metadata.json")
            ),
            stock_codes_path=Path(get("STOCK_CODES_PATH", "ViFinQA/code_stock.csv")),
            qdrant_rejects_path=Path(
                get("QDRANT_REJECTS_PATH", "intermediate/qdrant_rejects.jsonl")
            ),
            qdrant_state_path=Path(
                get("QDRANT_STATE_PATH", ".cache/qdrant_sync.sqlite3")
            ),
            upsert_batch_size=_as_int(
                "UPSERT_BATCH_SIZE", get("UPSERT_BATCH_SIZE", "256")
            ),
            qdrant_upload_parallel=_as_int(
                "QDRANT_UPLOAD_PARALLEL", get("QDRANT_UPLOAD_PARALLEL", "1")
            ),
            qdrant_upload_retries=_as_int(
                "QDRANT_UPLOAD_RETRIES", get("QDRANT_UPLOAD_RETRIES", "3")
            ),
            qdrant_prefer_grpc=get("QDRANT_PREFER_GRPC", "true").casefold()
            in {"1", "true", "yes", "on"},
        )
        if validate:
            settings.validate()
        return settings

    def validate(self) -> None:
        missing = []
        for label, value in (
            ("QDRANT_URL", self.qdrant_url),
            (
                "QDRANT_COLLECTION or QDRANT_ALIAS",
                self.qdrant_alias or self.qdrant_collection,
            ),
            ("LLM_BASE_URL", self.llm_base_url),
            ("LLM_MODEL", self.llm_model),
            ("FPT_RERANK_URL", self.fpt_rerank_url),
            ("FPT_RERANK_MODEL", self.fpt_rerank_model),
        ):
            if not value:
                missing.append(label)
        if self.retrieval_mode not in {"dense", "hybrid"}:
            raise SettingsError("RETRIEVAL_MODE must be 'dense' or 'hybrid'")
        if not 0.0 <= self.bm25_b <= 1.0:
            raise SettingsError("BM25_B must be between 0 and 1")
        if missing:
            raise SettingsError("Missing required settings: " + ", ".join(missing))

    def public_dict(self) -> dict[str, object]:
        values = asdict(self)
        for name in _SECRET_FIELDS:
            values[name] = "<configured>" if values[name] else "<missing>"
        values["project_root"] = "."
        manifest = self.qdrant_manifest_path
        if manifest.is_absolute():
            try:
                manifest = manifest.relative_to(self.project_root)
            except ValueError:
                manifest = Path(manifest.name)
        values["qdrant_manifest_path"] = manifest.as_posix()
        for name in (
            "tables_metadata_path",
            "stock_codes_path",
            "qdrant_rejects_path",
            "qdrant_state_path",
        ):
            path = Path(values[name])
            if path.is_absolute():
                try:
                    path = path.relative_to(self.project_root)
                except ValueError:
                    path = Path(path.name)
            values[name] = path.as_posix()
        return values


def settings_field_names() -> tuple[str, ...]:
    return tuple(field.name for field in fields(Settings))


__all__ = ["Settings", "SettingsError", "settings_field_names"]
