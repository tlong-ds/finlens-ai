"""Local BM25 retrieval over the same manifest used to build Qdrant."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
import tempfile
import threading
from collections.abc import Mapping, Sequence
from contextlib import closing
from pathlib import Path
from typing import Any

from src.contracts import FILTER_FIELDS, validate_qdrant_payload


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT / "intermediate" / "qdrant_manifest_granite_97m_r2_v1.jsonl"
)
DEFAULT_INDEX_PATH = PROJECT_ROOT / ".cache" / "bm25_manifest_v1.sqlite3"
BM25_SCHEMA_VERSION = 1

_INDEX_LOCK = threading.Lock()
_READY_INDEXES: set[tuple[str, int, int, str]] = set()
_STOP_WORDS = frozenset(
    {
        "bao",
        "bằng",
        "các",
        "cho",
        "có",
        "công",
        "của",
        "cuối",
        "đến",
        "đồng",
        "giữa",
        "là",
        "năm",
        "ngày",
        "nhiêu",
        "số",
        "so",
        "theo",
        "trong",
        "triệu",
        "tỷ",
        "vào",
        "và",
    }
)


class BM25IndexError(RuntimeError):
    """Raised when the local lexical index cannot be built or queried."""


def get_manifest_path() -> Path:
    raw = os.getenv("QDRANT_MANIFEST_PATH")
    path = Path(raw) if raw else DEFAULT_MANIFEST_PATH
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def get_index_path() -> Path:
    raw = os.getenv("BM25_INDEX_PATH")
    path = Path(raw) if raw else DEFAULT_INDEX_PATH
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _index_identity(manifest_path: Path, index_path: Path) -> tuple[str, int, int, str]:
    stat = manifest_path.stat()
    return str(manifest_path), stat.st_size, stat.st_mtime_ns, str(index_path)


def _is_current(index_path: Path, identity: tuple[str, int, int, str]) -> bool:
    if not index_path.is_file():
        return False
    try:
        with closing(sqlite3.connect(index_path)) as connection:
            row = connection.execute(
                "SELECT schema_version, manifest_path, manifest_size, manifest_mtime_ns "
                "FROM bm25_metadata LIMIT 1"
            ).fetchone()
    except sqlite3.Error:
        return False
    return row == (BM25_SCHEMA_VERSION, identity[0], identity[1], identity[2])


def _build_index(
    manifest_path: Path,
    index_path: Path,
    identity: tuple[str, int, int, str],
) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=index_path.parent,
        prefix=".bm25-",
        suffix=".sqlite3",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    point_count = 0
    try:
        with closing(sqlite3.connect(temporary_path)) as connection:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA temp_store=MEMORY")
            connection.execute("PRAGMA cache_size=-65536")
            connection.execute(
                "CREATE TABLE bm25_metadata ("
                "schema_version INTEGER NOT NULL, "
                "manifest_path TEXT NOT NULL, "
                "manifest_size INTEGER NOT NULL, "
                "manifest_mtime_ns INTEGER NOT NULL, "
                "point_count INTEGER NOT NULL)"
            )
            connection.execute(
                "CREATE VIRTUAL TABLE documents USING fts5("
                "table_id UNINDEXED, doc_id UNINDEXED, ticker UNINDEXED, "
                "company_name UNINDEXED, year UNINDEXED, "
                "report_type UNINDEXED, table_type UNINDEXED, "
                "start_line UNINDEXED, index_text, "
                "tokenize='unicode61 remove_diacritics 2')"
            )
            batch: list[tuple[Any, ...]] = []
            with manifest_path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if record.get("record_type") == "header":
                        continue
                    if record.get("record_type") != "point":
                        raise ValueError(
                            f"Unknown manifest record type at line {line_number}"
                        )
                    payload = validate_qdrant_payload(record["payload"])
                    if record.get("table_id") != payload["table_id"]:
                        raise ValueError(
                            f"Manifest table_id mismatch at line {line_number}"
                        )
                    index_text = record.get("index_text")
                    if not isinstance(index_text, str) or not index_text.strip():
                        raise ValueError(
                            f"Empty manifest index_text at line {line_number}"
                        )
                    batch.append(
                        (
                            payload["table_id"],
                            payload["doc_id"],
                            payload["ticker"],
                            payload["company_name"],
                            payload["year"],
                            payload["report_type"],
                            payload["table_type"],
                            payload["start_line"],
                            index_text,
                        )
                    )
                    if len(batch) >= 1_000:
                        connection.executemany(
                            "INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?)",
                            batch,
                        )
                        point_count += len(batch)
                        batch.clear()
                if batch:
                    connection.executemany(
                        "INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?)",
                        batch,
                    )
                    point_count += len(batch)
            if point_count < 1:
                raise ValueError("Manifest contains no point records")
            connection.execute(
                "INSERT INTO bm25_metadata VALUES (?,?,?,?,?)",
                (
                    BM25_SCHEMA_VERSION,
                    identity[0],
                    identity[1],
                    identity[2],
                    point_count,
                ),
            )
            connection.execute("INSERT INTO documents(documents) VALUES ('optimize')")
            connection.commit()
        os.replace(temporary_path, index_path)
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        KeyError,
        ValueError,
        sqlite3.Error,
    ) as exc:
        raise BM25IndexError(f"Cannot build BM25 index from {manifest_path}") from exc
    finally:
        temporary_path.unlink(missing_ok=True)
    logger.info("Built BM25 index with %d documents at %s", point_count, index_path)


def ensure_index(
    manifest_path: Path | None = None,
    index_path: Path | None = None,
) -> Path:
    """Build the SQLite FTS5 index once and return its validated path."""
    manifest = (manifest_path or get_manifest_path()).resolve()
    index = (index_path or get_index_path()).resolve()
    if not manifest.is_file():
        raise BM25IndexError(f"BM25 manifest does not exist: {manifest}")
    try:
        identity = _index_identity(manifest, index)
    except OSError as exc:
        raise BM25IndexError(f"Cannot stat BM25 manifest: {manifest}") from exc
    if identity in _READY_INDEXES and _is_current(index, identity):
        return index
    with _INDEX_LOCK:
        if not _is_current(index, identity):
            _build_index(manifest, index, identity)
        _READY_INDEXES.add(identity)
    return index


def _query_tokens(query_text: str) -> list[str]:
    raw = list(
        dict.fromkeys(
            token
            for token in re.findall(r"\w+", query_text.casefold(), flags=re.UNICODE)
            if len(token) > 1
        )
    )
    filtered = [token for token in raw if token not in _STOP_WORDS]
    return filtered or raw


def _match_expression(query_text: str) -> str:
    tokens = _query_tokens(query_text)
    if not tokens:
        raise BM25IndexError("BM25 query must contain at least one token")
    return " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)


def _filter_clauses(
    filters: Mapping[str, Sequence[str | int]] | None,
) -> tuple[list[str], list[str | int]]:
    raw_filters = filters or {}
    unknown = set(raw_filters) - set(FILTER_FIELDS)
    if unknown:
        raise ValueError("Unsupported BM25 filters: " + ", ".join(sorted(unknown)))
    clauses: list[str] = []
    parameters: list[str | int] = []
    for field in FILTER_FIELDS:
        raw_values = raw_filters.get(field, [])
        if isinstance(raw_values, (str, bytes)):
            raise TypeError(f"Filter {field} must be a sequence, not a string")
        values = list(raw_values)
        if not values:
            continue
        if field == "year":
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in values
            ):
                raise TypeError("Filter year must contain only integers")
        elif any(not isinstance(value, str) for value in values):
            raise TypeError(f"Filter {field} must contain only strings")
        clauses.append(f"{field} IN ({','.join('?' for _ in values)})")
        parameters.extend(values)
    return clauses, parameters


def search_bm25(
    query_text: str,
    filters: Mapping[str, Sequence[str | int]] | None = None,
    *,
    top_n: int,
    manifest_path: Path | None = None,
    index_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return metadata-filtered BM25 candidates ordered by lexical rank."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    expression = _match_expression(query_text)
    clauses, filter_parameters = _filter_clauses(filters)
    sql = (
        "SELECT table_id, doc_id, ticker, company_name, year, report_type, "
        "table_type, start_line, bm25(documents) AS score "
        "FROM documents WHERE documents MATCH ?"
    )
    if clauses:
        sql += " AND " + " AND ".join(clauses)
    sql += " ORDER BY score, table_id LIMIT ?"
    parameters: list[str | int] = [expression, *filter_parameters, top_n]
    database = ensure_index(manifest_path, index_path)
    try:
        with closing(sqlite3.connect(database)) as connection:
            connection.execute("PRAGMA query_only=ON")
            connection.execute("PRAGMA busy_timeout=30000")
            rows = connection.execute(sql, parameters).fetchall()
    except sqlite3.Error as exc:
        raise BM25IndexError("BM25 search failed") from exc

    candidates: list[dict[str, Any]] = []
    for rank, row in enumerate(rows, start=1):
        raw_score = float(row[8])
        if not math.isfinite(raw_score):
            raise BM25IndexError("BM25 returned a non-finite score")
        payload = validate_qdrant_payload(
            {
                "table_id": row[0],
                "doc_id": row[1],
                "ticker": row[2],
                "company_name": row[3],
                "year": int(row[4]),
                "report_type": row[5],
                "table_type": row[6],
                "start_line": int(row[7]),
            }
        )
        candidates.append(
            {
                "table_id": payload["table_id"],
                "metadata": payload,
                "bm25_score": -raw_score,
                "bm25_rank": rank,
            }
        )
    logger.info("Retrieved %d BM25 table candidates", len(candidates))
    return candidates
