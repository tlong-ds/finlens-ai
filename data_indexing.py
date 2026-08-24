from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
import unicodedata
import uuid
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from src.embeddings import (
    EMBEDDING_MAX_LENGTH,
    EMBEDDING_MODEL_DEFAULT,
    EMBEDDING_REVISION_DEFAULT,
    EMBEDDING_VECTOR_SIZE,
    DENSE_VECTOR_NAME,
    DenseEmbeddingModel,
    EmbeddingError,
)
from src.contracts import (
    MAX_YEAR,
    MIN_YEAR,
    PAYLOAD_FIELDS,
    PAYLOAD_SCHEMA_VERSION,
    REPORT_TYPES,
    TABLE_ID_PATTERN,
    resolve_csv_path as resolve_contract_csv_path,
)


LOGGER = logging.getLogger("finlens.data_indexing")

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_METADATA_PATH = PROJECT_ROOT / "metadata" / "tables_metadata.json"
DEFAULT_STOCK_CODES_PATH = PROJECT_ROOT / "ViFinQA" / "code_stock.csv"
DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT / "intermediate" / "qdrant_manifest_granite_97m_r2_v1.jsonl"
)
DEFAULT_REJECTS_PATH = (
    PROJECT_ROOT / "intermediate" / "qdrant_rejects_granite_97m_r2_v1.jsonl"
)
DEFAULT_STATE_PATH = PROJECT_ROOT / ".cache" / "qdrant_sync_granite_97m_r2_v1.sqlite3"

DATASET_ID = "AIGuruTinix/ViFinQA"
DATASET_REVISION = "0450088ab22ec946f04f097586967ca405955b3b"
COLLECTION_DEFAULT = "finlens_tables_metadata_granite_97m_multilingual_r2_v1"
ALIAS_DEFAULT = "finlens_tables_current"
VECTOR_SIZE = EMBEDDING_VECTOR_SIZE
INDEX_TEXT_VERSION = 1
FINLENS_NAMESPACE_UUID = uuid.uuid5(
    uuid.NAMESPACE_URL, "https://finlens.ai/qdrant/table"
)
YEAR_PATTERN = re.compile(r"(?<!\d)(20\d{2})(?!\d)")
YEAR_RANGE_PATTERN = re.compile(
    r"(?<!\d)(20\d{2})\s*(?:-|–|—|đến|den|tới|toi)\s*(20\d{2})(?!\d)",
    re.IGNORECASE,
)

TABLE_TYPE_VI = {
    "balance_sheet": "Bảng cân đối kế toán",
    "income_statement": "Báo cáo kết quả kinh doanh",
    "cash_flow": "Báo cáo lưu chuyển tiền tệ",
    "note_table": "Bảng thuyết minh báo cáo tài chính",
}
REPORT_TYPE_TERMS = (
    "consolidated",
    "separate",
    "aggregated",
    "hợp nhất",
    "riêng",
    "tổng hợp",
    "công ty mẹ",
    "báo cáo tài chính",
)
GENERIC_SUMMARY_TERMS = {
    "",
    "note_unknown",
    "thuyết minh note_unknown",
    "bảng thuyết minh note_unknown",
    "thuyết minh",
    "bảng thuyết minh",
    "bảng thuyết minh báo cáo tài chính",
}
# Common Vietnamese words used to keep untagged summaries/keywords in the
# embedding text.  ``canonical_name_vi`` is trusted by field provenance, while
# free-form ``semantic_summary`` and ``keywords`` need this lightweight,
# deterministic guard to avoid embedding English-only text.
VIETNAMESE_SIGNAL_WORDS = frozenset(
    {
        "báo", "cáo", "bảng", "doanh", "thu", "lợi", "nhuận", "chi",
        "phí", "tài", "sản", "nợ", "vốn", "chủ", "sở", "hữu", "tiền",
        "khoản", "phải", "trả", "người", "bán", "mua", "hàng", "tồn",
        "kho", "ngắn", "dài", "hạn", "năm", "quý", "kỳ", "tháng",
        "hoạt", "động", "kinh", "doanh", "thuế", "lãi", "vay", "cổ",
        "phiếu", "đầu", "tư", "dòng", "lưu", "chuyển", "thuyết", "minh",
        "giá", "trị", "khấu", "hao", "doanh", "nghiệp", "công", "ty",
    }
)


class BaselineError(RuntimeError):
    """Expected, user-actionable baseline failure."""


class RoutingError(BaselineError):
    """The query cannot be safely routed to ticker/year buckets."""


@dataclass(frozen=True)
class Config:
    project_root: Path = PROJECT_ROOT
    metadata_path: Path = DEFAULT_METADATA_PATH
    stock_codes_path: Path = DEFAULT_STOCK_CODES_PATH
    manifest_path: Path = DEFAULT_MANIFEST_PATH
    rejects_path: Path = DEFAULT_REJECTS_PATH
    state_path: Path = DEFAULT_STATE_PATH
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    collection_name: str = COLLECTION_DEFAULT
    alias_name: str = ALIAS_DEFAULT
    prefer_grpc: bool = True
    qdrant_timeout: float = 60.0
    embedding_model: str = EMBEDDING_MODEL_DEFAULT
    embedding_revision: str = EMBEDDING_REVISION_DEFAULT
    embedding_model_path: str | None = None
    embedding_device: str = "auto"
    embedding_batch_size: int = 32
    upsert_batch_size: int = 256
    upload_parallel: int = 1
    upload_retries: int = 3
    max_length: int = EMBEDDING_MAX_LENGTH

    @classmethod
    def from_env(cls) -> "Config":
        root = Path(os.getenv("FINLENS_PROJECT_ROOT", str(PROJECT_ROOT))).resolve()
        return cls(
            project_root=root,
            metadata_path=_env_path("TABLES_METADATA_PATH", root / "metadata" / "tables_metadata.json"),
            stock_codes_path=_env_path("STOCK_CODES_PATH", root / "ViFinQA" / "code_stock.csv"),
            manifest_path=_env_path(
                "QDRANT_MANIFEST_PATH",
                root / "intermediate" / "qdrant_manifest_granite_97m_r2_v1.jsonl",
            ),
            rejects_path=_env_path(
                "QDRANT_REJECTS_PATH",
                root / "intermediate" / "qdrant_rejects_granite_97m_r2_v1.jsonl",
            ),
            state_path=_env_path(
                "QDRANT_STATE_PATH",
                root / ".cache" / "qdrant_sync_granite_97m_r2_v1.sqlite3",
            ),
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
            collection_name=os.getenv("QDRANT_COLLECTION", COLLECTION_DEFAULT),
            alias_name=os.getenv("QDRANT_ALIAS", ALIAS_DEFAULT),
            prefer_grpc=_env_bool("QDRANT_PREFER_GRPC", True),
            qdrant_timeout=float(os.getenv("QDRANT_TIMEOUT", "60")),
            embedding_model=os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL_DEFAULT),
            embedding_revision=os.getenv("EMBEDDING_REVISION", EMBEDDING_REVISION_DEFAULT),
            embedding_model_path=os.getenv("EMBEDDING_MODEL_PATH") or None,
            embedding_device=os.getenv("EMBEDDING_DEVICE", "auto"),
            embedding_batch_size=int(os.getenv("EMBED_BATCH_SIZE", "32")),
            upsert_batch_size=int(os.getenv("UPSERT_BATCH_SIZE", "256")),
            upload_parallel=int(os.getenv("QDRANT_UPLOAD_PARALLEL", "1")),
            upload_retries=int(os.getenv("QDRANT_UPLOAD_RETRIES", "3")),
            max_length=int(
                os.getenv("EMBEDDING_MAX_LENGTH", str(EMBEDDING_MAX_LENGTH))
            ),
        )


def _env_path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).resolve()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def fold_text(value: Any) -> str:
    text = normalize_text(value).lower().replace("đ", "d")
    return "".join(
        char for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )


def validate_table_id(table_id: str) -> str:
    table_id = normalize_text(table_id)
    if not table_id or not TABLE_ID_PATTERN.fullmatch(table_id):
        raise BaselineError(f"table_id không hợp lệ: {table_id!r}")
    if table_id in {".", ".."}:
        raise BaselineError(f"table_id không an toàn: {table_id!r}")
    return table_id


def resolve_csv_path(table_id: str, project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve the canonical CSV path and block path traversal."""
    try:
        return resolve_contract_csv_path(validate_table_id(table_id), project_root)
    except ValueError as exc:
        raise BaselineError(str(exc)) from exc


def _iter_json_array_stdlib(path: Path, chunk_size: int = 1 << 20) -> Iterator[dict[str, Any]]:
    """Incrementally decode a top-level JSON array without ``json.load``."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        position = 0
        eof = False

        def read_more() -> None:
            nonlocal buffer, position, eof
            if position:
                buffer = buffer[position:]
                position = 0
            chunk = handle.read(chunk_size)
            if chunk:
                buffer += chunk
            else:
                eof = True

        read_more()
        while not eof and not buffer.strip():
            read_more()
        position = len(buffer) - len(buffer.lstrip())
        if position >= len(buffer) or buffer[position] != "[":
            raise BaselineError(f"JSON source phải là top-level array: {path}")
        position += 1

        while True:
            while True:
                while position < len(buffer) and (buffer[position].isspace() or buffer[position] == ","):
                    position += 1
                if position < len(buffer) or eof:
                    break
                read_more()

            if position < len(buffer) and buffer[position] == "]":
                return
            if eof and position >= len(buffer):
                raise BaselineError(f"JSON array kết thúc không hợp lệ: {path}")

            try:
                value, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError as exc:
                if eof:
                    raise BaselineError(f"JSON không hợp lệ tại offset {exc.pos}: {path}") from exc
                read_more()
                continue
            if not isinstance(value, dict):
                raise BaselineError("Mỗi phần tử tables_metadata.json phải là object")
            yield value
            position = end
            if position > chunk_size:
                read_more()


def iter_tables_metadata(path: Path) -> Iterator[dict[str, Any]]:
    """Stream metadata with ijson, with a streaming stdlib fallback."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        import ijson  # type: ignore
    except ImportError:
        LOGGER.warning("ijson chưa được cài; dùng streaming JSON decoder chuẩn (chậm hơn)")
        yield from _iter_json_array_stdlib(path)
        return

    with path.open("rb") as handle:
        for item in ijson.items(handle, "item"):
            if not isinstance(item, dict):
                raise BaselineError("Mỗi phần tử tables_metadata.json phải là object")
            yield item


def _dedupe_terms(values: Iterable[Any], limit: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        term = normalize_text(value).strip(" -–—:;,.|/\\")
        folded = fold_text(term)
        if (
            len(term) < 2
            or len(term) > 160
            or folded in seen
            or folded in {"note_unknown", "unknown"}
            or not any(char.isalpha() for char in term)
            or term.count("\\") >= 2
        ):
            continue
        seen.add(folded)
        result.append(term)
        if len(result) >= limit:
            break
    return result


def _is_vietnamese_text(value: Any) -> bool:
    """Return whether free-form metadata has a Vietnamese-language signal."""
    text = normalize_text(value)
    if not text:
        return False
    # Vietnamese-specific letters provide a strong signal, including for short
    # terms such as ``lãi`` or ``quỹ``.
    if re.search(r"[ăâđêôơưĂÂĐÊÔƠƯ]", text):
        return True
    # Accept common unaccented Vietnamese metadata while rejecting English-only
    # labels such as ``Net revenue`` or ``Operating profit``.
    words = set(re.findall(r"[a-zA-ZÀ-ỹĐđ]+", text.lower()))
    return bool(words & VIETNAMESE_SIGNAL_WORDS)


def _vietnamese_terms(values: Iterable[Any], limit: int) -> list[str]:
    return _dedupe_terms(
        (value for value in values if _is_vietnamese_text(value)),
        limit,
    )


def _strip_identity(text: Any, record: Mapping[str, Any]) -> str:
    cleaned = normalize_text(text)
    identities = [
        record.get("table_id"),
        record.get("doc_id"),
        record.get("ticker"),
        record.get("company_name"),
        record.get("csv_path"),
        record.get("year"),
    ]
    for identity in identities:
        token = normalize_text(identity)
        if token:
            cleaned = re.sub(re.escape(token), " ", cleaned, flags=re.IGNORECASE)
    for term in REPORT_TYPE_TERMS:
        cleaned = re.sub(rf"(?<!\w){re.escape(term)}(?!\w)", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b20\d{2}\b", " ", cleaned)
    cleaned = re.sub(r"\bnote[_\s-]*unknown\b", " ", cleaned, flags=re.IGNORECASE)
    indicator_prefix = re.search(r"\bgồm\s+các\s+chỉ\s+tiêu\s*:\s*", cleaned, flags=re.I)
    if indicator_prefix:
        cleaned = cleaned[indicator_prefix.end():]
    cleaned = re.sub(
        r"^(?:báo cáo|bảng|nội dung|của|năm)(?:\s+(?:báo cáo|bảng|của|năm))*\s*",
        "",
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -–—:;,.|/")
    if fold_text(cleaned) in {fold_text(value) for value in GENERIC_SUMMARY_TERMS}:
        return ""
    return cleaned


def build_index_text(record: Mapping[str, Any]) -> tuple[str, str]:
    """Build the Vietnamese-only V1 text from semantic metadata."""
    table_type = normalize_text(record.get("table_type"))
    type_label = TABLE_TYPE_VI.get(table_type, "")
    retrieval = record.get("retrieval_context") or {}
    if not isinstance(retrieval, Mapping):
        retrieval = {}

    raw_summary = retrieval.get("semantic_summary")
    summary_quality = "ok"
    if "note_unknown" in fold_text(raw_summary):
        summary_quality = "note_unknown"
    summary = _strip_identity(raw_summary, record)
    if not _is_vietnamese_text(summary):
        summary = ""

    semantic_fields = record.get("semantic_fields") or []
    canonical_values: list[Any] = []
    if isinstance(semantic_fields, list):
        for field in semantic_fields:
            if not isinstance(field, Mapping):
                continue
            canonical_values.append(field.get("canonical_name_vi"))
    canonical = _vietnamese_terms(
        (_strip_identity(value, record) for value in canonical_values), 32
    )

    raw_keywords = retrieval.get("keywords") or []
    keyword_values = raw_keywords if isinstance(raw_keywords, list) else []
    keywords = _vietnamese_terms(
        (_strip_identity(value, record) for value in keyword_values), 32
    )
    canonical_folded = {fold_text(item) for item in canonical}
    keywords = [item for item in keywords if fold_text(item) not in canonical_folded][:32]

    sections: list[str] = []
    if type_label:
        sections.append(f"Loại bảng: {type_label}")
    if summary:
        sections.append(f"Nội dung: {summary}")
    if canonical:
        sections.append("Chỉ tiêu: " + "; ".join(canonical))
    if keywords:
        sections.append("Từ khóa: " + "; ".join(keywords))
    return "\n".join(sections).strip(), summary_quality


def has_useful_note_metadata(record: Mapping[str, Any]) -> bool:
    """Whether a note can be distinguished without embedding CSV content."""
    if normalize_text(record.get("table_type")) != "note_table":
        return True
    retrieval = record.get("retrieval_context") or {}
    if not isinstance(retrieval, Mapping):
        retrieval = {}
    summary = _strip_identity(retrieval.get("semantic_summary"), record)
    if _is_vietnamese_text(summary):
        return True
    raw_keywords = retrieval.get("keywords") or []
    if isinstance(raw_keywords, list) and _vietnamese_terms(
        (_strip_identity(value, record) for value in raw_keywords), 1
    ):
        return True
    semantic_fields = record.get("semantic_fields") or []
    if isinstance(semantic_fields, list):
        for field in semantic_fields:
            if not isinstance(field, Mapping):
                continue
            if _vietnamese_terms(
                (_strip_identity(field.get("canonical_name_vi"), record),), 1
            ):
                return True
    return False


def build_payload(record: Mapping[str, Any], index_text: str) -> dict[str, Any]:
    start_line = record.get("start_line")
    if isinstance(start_line, bool) or not isinstance(start_line, int) or start_line < 1:
        raise BaselineError("missing_or_invalid_start_line")
    if not isinstance(index_text, str) or not index_text.strip():
        raise BaselineError("missing_or_empty_index_text")
    payload = {
        "table_id": validate_table_id(normalize_text(record.get("table_id"))),
        "doc_id": normalize_text(record.get("doc_id")),
        "ticker": normalize_text(record.get("ticker")).upper(),
        "company_name": normalize_text(record.get("company_name")),
        "year": int(record.get("year")),
        "report_type": normalize_text(record.get("report_type")),
        "table_type": normalize_text(record.get("table_type")),
        "start_line": start_line,
        "index_text": index_text.strip(),
    }
    if tuple(payload) != PAYLOAD_FIELDS:
        raise AssertionError("Payload allowlist bị thay đổi")
    return payload


def make_point_id(table_id: str) -> str:
    return str(uuid.uuid5(FINLENS_NAMESPACE_UUID, f"table:{validate_table_id(table_id)}"))


def compute_content_hash(
    table_id: str,
    index_text: str,
    model: str = EMBEDDING_MODEL_DEFAULT,
    revision: str = EMBEDDING_REVISION_DEFAULT,
    payload: Mapping[str, Any] | None = None,
) -> str:
    payload_json = json.dumps(
        dict(payload or {}),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    content = "\n".join(
        (
            validate_table_id(table_id),
            index_text,
            str(INDEX_TEXT_VERSION),
            str(PAYLOAD_SCHEMA_VERSION),
            payload_json,
            model,
            revision,
        )
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _record_rejection(record: Mapping[str, Any], reason: str) -> dict[str, Any]:
    return {
        "table_id": normalize_text(record.get("table_id")) or None,
        "doc_id": normalize_text(record.get("doc_id")) or None,
        "reason": reason,
    }


def _candidate_to_manifest(
    record: Mapping[str, Any], config: Config, check_files: bool
) -> dict[str, Any]:
    required = (
        "table_id",
        "doc_id",
        "ticker",
        "company_name",
        "year",
        "table_type",
    )
    missing = [field for field in required if not normalize_text(record.get(field))]
    if missing:
        raise BaselineError("missing_required:" + ",".join(missing))
    if normalize_text(record.get("table_type")) == "table_of_contents":
        raise BaselineError("table_of_contents")
    if not has_useful_note_metadata(record):
        raise BaselineError("uninformative_note_metadata")
    if not normalize_text(record.get("csv_path")):
        raise BaselineError("empty_csv_path")

    index_text, metadata_quality = build_index_text(record)
    if not index_text:
        raise BaselineError("empty_index_text")

    payload = build_payload(record, index_text)
    if not (MIN_YEAR <= payload["year"] <= MAX_YEAR):
        raise BaselineError("year_out_of_range")
    if check_files:
        try:
            resolve_csv_path(payload["table_id"], config.project_root)
        except FileNotFoundError as exc:
            raise BaselineError("derived_csv_missing") from exc

    point_id = make_point_id(payload["table_id"])
    return {
        "record_type": "point",
        "point_id": point_id,
        "table_id": payload["table_id"],
        "index_text": index_text,
        "payload": payload,
        "content_hash": compute_content_hash(
            payload["table_id"],
            index_text,
            config.embedding_model,
            config.embedding_revision,
            payload,
        ),
        "index_text_version": INDEX_TEXT_VERSION,
        "metadata_quality": metadata_quality,
    }


def build_manifest(
    config: Config,
    *,
    ticker: str | None = None,
    year: int | None = None,
    limit: int | None = None,
    check_files: bool = True,
    source_hash: bool = True,
) -> dict[str, Any]:
    """Create an atomic JSONL manifest without opening any CSV content."""
    metadata_path = config.metadata_path
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config.rejects_path.parent.mkdir(parents=True, exist_ok=True)
    source_stat = metadata_path.stat()
    if source_hash:
        LOGGER.info("Đang tính SHA-256 cho %s", metadata_path)
    source_sha = sha256_file(metadata_path) if source_hash else None
    header = {
        "record_type": "header",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "source_path": str(metadata_path),
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "source_sha256": source_sha,
        "embedding_model": config.embedding_model,
        "embedding_revision": config.embedding_revision,
        "collection_name": config.collection_name,
        "vector_name": DENSE_VECTOR_NAME,
        "vector_size": VECTOR_SIZE,
        "index_text_version": INDEX_TEXT_VERSION,
        "payload_schema_version": PAYLOAD_SCHEMA_VERSION,
        "git_commit": _git_commit(config.project_root),
    }

    ticker_filter = ticker.upper() if ticker else None
    accepted = 0
    rejected = 0
    scanned = 0
    duplicate_ids: set[str] = set()
    rejected_reasons: Counter[str] = Counter()
    accepted_types: Counter[str] = Counter()
    qualities: Counter[str] = Counter()

    manifest_tmp = _temporary_sibling(config.manifest_path)
    rejects_tmp = _temporary_sibling(config.rejects_path)
    try:
        with manifest_tmp.open("w", encoding="utf-8", newline="\n") as manifest_handle, rejects_tmp.open(
            "w", encoding="utf-8", newline="\n"
        ) as rejects_handle:
            _write_jsonl(manifest_handle, header)
            for record in iter_tables_metadata(metadata_path):
                scanned += 1
                if scanned % 10_000 == 0:
                    LOGGER.info(
                        "Build manifest: scanned=%d accepted=%d rejected=%d",
                        scanned,
                        accepted,
                        rejected,
                    )
                if ticker_filter and normalize_text(record.get("ticker")).upper() != ticker_filter:
                    continue
                try:
                    record_year = int(record.get("year"))
                except (TypeError, ValueError):
                    record_year = None
                if year is not None and record_year != year:
                    continue
                try:
                    item = _candidate_to_manifest(record, config, check_files)
                    table_id = item["table_id"]
                    if table_id in duplicate_ids:
                        raise BaselineError("duplicate_table_id")
                    duplicate_ids.add(table_id)
                except (BaselineError, TypeError, ValueError) as exc:
                    reason = str(exc) or exc.__class__.__name__
                    _write_jsonl(rejects_handle, _record_rejection(record, reason))
                    rejected += 1
                    rejected_reasons[reason] += 1
                    continue

                _write_jsonl(manifest_handle, item)
                accepted += 1
                accepted_types[item["payload"]["table_type"]] += 1
                qualities[item["metadata_quality"]] += 1
                if limit is not None and accepted >= limit:
                    break

        os.replace(manifest_tmp, config.manifest_path)
        os.replace(rejects_tmp, config.rejects_path)
    except BaseException:
        manifest_tmp.unlink(missing_ok=True)
        rejects_tmp.unlink(missing_ok=True)
        raise

    stats = {
        "scanned": scanned,
        "accepted": accepted,
        "rejected": rejected,
        "accepted_by_table_type": dict(sorted(accepted_types.items())),
        "metadata_quality": dict(sorted(qualities.items())),
        "rejected_by_reason": dict(rejected_reasons.most_common()),
        "manifest": str(config.manifest_path),
        "rejects": str(config.rejects_path),
    }
    LOGGER.info("Manifest hoàn tất: %s", json.dumps(stats, ensure_ascii=False))
    return stats


def _temporary_sibling(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(handle)
    return Path(name)


def _write_jsonl(handle: Any, value: Mapping[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def iter_manifest(path: Path, include_header: bool = False) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BaselineError(f"Manifest JSONL lỗi tại dòng {line_number}: {path}") from exc
            if include_header or item.get("record_type") == "point":
                yield item


def read_manifest_header(path: Path) -> dict[str, Any]:
    first = next(iter_manifest(path, include_header=True), None)
    if not first or first.get("record_type") != "header":
        raise BaselineError(f"Manifest thiếu header: {path}")
    return first


def manifest_stats(path: Path) -> dict[str, Any]:
    header = read_manifest_header(path)
    count = 0
    types: Counter[str] = Counter()
    qualities: Counter[str] = Counter()
    tickers: set[str] = set()
    years: set[int] = set()
    for item in iter_manifest(path):
        count += 1
        payload = item["payload"]
        types[payload["table_type"]] += 1
        qualities[item.get("metadata_quality", "unknown")] += 1
        tickers.add(payload["ticker"])
        years.add(payload["year"])
    return {
        "header": header,
        "points": count,
        "table_types": dict(types.most_common()),
        "metadata_quality": dict(qualities.most_common()),
        "tickers": len(tickers),
        "year_min": min(years) if years else None,
        "year_max": max(years) if years else None,
    }


class EmbeddingModel(DenseEmbeddingModel):
    """Indexing adapter for the shared pinned Granite encoder."""

    def __init__(self, config: Config):
        self.config = config
        super().__init__(
            model_id=config.embedding_model,
            revision=config.embedding_revision,
            model_path=config.embedding_model_path,
            device=config.embedding_device,
            batch_size=config.embedding_batch_size,
            max_length=config.max_length,
        )

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode index documents with the Granite text contract."""
        try:
            return self.encode_passages(texts)
        except EmbeddingError as exc:
            raise BaselineError(str(exc)) from exc

    def encode_query(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode retrieval queries with the Granite text contract."""
        try:
            return self.encode_queries(texts)
        except EmbeddingError as exc:
            raise BaselineError(str(exc)) from exc


def _qdrant_imports() -> tuple[Any, Any]:
    try:
        from qdrant_client import QdrantClient  # type: ignore
        from qdrant_client import models  # type: ignore
    except ImportError as exc:
        raise BaselineError(
            "Thiếu qdrant-client. Chạy: pip install -r requirements.txt"
        ) from exc
    return QdrantClient, models


def create_qdrant_client(config: Config) -> Any:
    QdrantClient, _ = _qdrant_imports()
    kwargs: dict[str, Any] = {
        "url": config.qdrant_url,
        "prefer_grpc": config.prefer_grpc,
        "timeout": config.qdrant_timeout,
    }
    if config.qdrant_api_key:
        kwargs["api_key"] = config.qdrant_api_key
    return QdrantClient(**kwargs)


def _enum_value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw).lower()


def ensure_collection(config: Config, client: Any | None = None) -> dict[str, Any]:
    """Create or fail-fast validate the physical collection and indexes."""
    client = client or create_qdrant_client(config)
    _, models = _qdrant_imports()
    exists = client.collection_exists(config.collection_name)
    if not exists:
        client.create_collection(
            collection_name=config.collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                    on_disk=True,
                )
            },
            on_disk_payload=True,
        )
        LOGGER.info("Đã tạo collection %s", config.collection_name)
    else:
        info = client.get_collection(config.collection_name)
        vectors = info.config.params.vectors
        if not isinstance(vectors, Mapping) or DENSE_VECTOR_NAME not in vectors:
            raise BaselineError(
                f"Collection tồn tại nhưng thiếu named vector {DENSE_VECTOR_NAME!r}"
            )
        dense = vectors[DENSE_VECTOR_NAME]
        if int(dense.size) != VECTOR_SIZE or _enum_value(dense.distance) != "cosine":
            raise BaselineError(
                "Collection schema không khớp: "
                f"size={dense.size}, distance={dense.distance}; "
                f"kỳ vọng size={VECTOR_SIZE}, cosine"
            )

    index_schemas = {
        "ticker": models.PayloadSchemaType.KEYWORD,
        "company_name": models.PayloadSchemaType.KEYWORD,
        "year": models.PayloadSchemaType.INTEGER,
        "report_type": models.PayloadSchemaType.KEYWORD,
        "table_type": models.PayloadSchemaType.KEYWORD,
        "doc_id": models.PayloadSchemaType.KEYWORD,
    }
    for field_name, field_schema in index_schemas.items():
        try:
            client.create_payload_index(
                collection_name=config.collection_name,
                field_name=field_name,
                field_schema=field_schema,
                wait=True,
            )
        except Exception as exc:
            # Existing indexes are reported differently by Qdrant versions.
            message = str(exc).lower()
            if "already exists" not in message and "already indexed" not in message:
                raise

    return {
        "collection": config.collection_name,
        "alias": config.alias_name,
        "vector_name": DENSE_VECTOR_NAME,
        "vector_size": VECTOR_SIZE,
        "distance": "Cosine",
        "created": not exists,
    }


def activate_alias(
    client: Any,
    collection: str,
    alias: str,
    models: Any | None = None,
) -> dict[str, Any]:
    """Atomically point the read alias at a fully verified collection."""
    if models is None:
        _, models = _qdrant_imports()
    aliases = client.get_aliases().aliases
    matching = [item for item in aliases if item.alias_name == alias]
    if matching:
        previous = matching[0].collection_name
        if previous == collection:
            return {
                "alias": alias,
                "previous_collection": previous,
                "collection": collection,
                "changed": False,
            }
        operations = [
            models.DeleteAliasOperation(
                delete_alias=models.DeleteAlias(alias_name=alias)
            )
        ]
    else:
        previous = None
        operations = []
    operations.append(
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(
                collection_name=collection,
                alias_name=alias,
            )
        )
    )
    client.update_collection_aliases(
        change_aliases_operations=operations
    )
    LOGGER.info("Alias %s đã chuyển từ %s sang %s", alias, previous, collection)
    return {
        "alias": alias,
        "previous_collection": previous,
        "collection": collection,
        "changed": True,
    }


def init_state_db(path: Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS sync_state (
            point_id TEXT PRIMARY KEY,
            table_id TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            collection_name TEXT NOT NULL,
            status TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_sync_collection_status "
        "ON sync_state(collection_name, status)"
    )
    connection.commit()
    return connection


def _successful_hashes(connection: sqlite3.Connection, collection: str) -> dict[str, str]:
    rows = connection.execute(
        "SELECT point_id, content_hash FROM sync_state "
        "WHERE collection_name = ? AND status = 'success'",
        (collection,),
    )
    return {point_id: content_hash for point_id, content_hash in rows}


def _mark_success(
    connection: sqlite3.Connection,
    collection: str,
    items: Sequence[Mapping[str, Any]],
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    connection.executemany(
        """
        INSERT INTO sync_state
            (point_id, table_id, content_hash, collection_name, status, updated_at)
        VALUES (?, ?, ?, ?, 'success', ?)
        ON CONFLICT(point_id) DO UPDATE SET
            table_id = excluded.table_id,
            content_hash = excluded.content_hash,
            collection_name = excluded.collection_name,
            status = excluded.status,
            updated_at = excluded.updated_at
        """,
        [
            (
                item["point_id"],
                item["table_id"],
                item["content_hash"],
                collection,
                now,
            )
            for item in items
        ],
    )
    connection.commit()


def _matches_scope(
    item: Mapping[str, Any], ticker: str | None, year: int | None
) -> bool:
    payload = item["payload"]
    if ticker and payload["ticker"] != ticker.upper():
        return False
    if year is not None and payload["year"] != year:
        return False
    return True


def _batches(values: Iterable[Any], size: int) -> Iterator[list[Any]]:
    if size <= 0:
        raise ValueError("batch size phải > 0")
    batch: list[Any] = []
    for value in values:
        batch.append(value)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _validate_manifest_header(header: Mapping[str, Any], config: Config) -> None:
    expected = {
        "embedding_model": config.embedding_model,
        "embedding_revision": config.embedding_revision,
        "collection_name": config.collection_name,
        "vector_name": DENSE_VECTOR_NAME,
        "vector_size": VECTOR_SIZE,
        "index_text_version": INDEX_TEXT_VERSION,
        "payload_schema_version": PAYLOAD_SCHEMA_VERSION,
    }
    mismatches = {
        key: (header.get(key), value)
        for key, value in expected.items()
        if header.get(key) != value
    }
    if mismatches:
        raise BaselineError(f"Manifest header không khớp config: {mismatches}")


def upsert_points(
    config: Config,
    *,
    ticker: str | None = None,
    year: int | None = None,
    limit: int | None = None,
    force: bool = False,
    client: Any | None = None,
    encoder: EmbeddingModel | None = None,
) -> dict[str, Any]:
    """Embed manifest text and idempotently upload named dense vectors."""
    header = read_manifest_header(config.manifest_path)
    _validate_manifest_header(header, config)
    client = client or create_qdrant_client(config)
    ensure_collection(config, client)
    encoder = encoder or EmbeddingModel(config)
    _, models = _qdrant_imports()
    connection = init_state_db(config.state_path)
    successful = {} if force else _successful_hashes(connection, config.collection_name)

    selected = 0
    skipped = 0
    embedded = 0
    upserted = 0
    started = time.monotonic()

    def pending_items() -> Iterator[dict[str, Any]]:
        nonlocal selected, skipped
        for item in iter_manifest(config.manifest_path):
            if not _matches_scope(item, ticker, year):
                continue
            if limit is not None and selected >= limit:
                break
            selected += 1
            if successful.get(item["point_id"]) == item["content_hash"]:
                skipped += 1
                continue
            yield item

    try:
        for batch_number, batch in enumerate(
            _batches(pending_items(), config.upsert_batch_size), 1
        ):
            vectors = encoder.encode([item["index_text"] for item in batch])
            embedded += len(batch)
            points = [
                models.PointStruct(
                    id=item["point_id"],
                    vector={DENSE_VECTOR_NAME: vector},
                    payload=item["payload"],
                )
                for item, vector in zip(batch, vectors)
            ]
            client.upload_points(
                collection_name=config.collection_name,
                points=points,
                batch_size=config.upsert_batch_size,
                parallel=config.upload_parallel,
                max_retries=config.upload_retries,
                wait=True,
            )
            _mark_success(connection, config.collection_name, batch)
            upserted += len(batch)
            elapsed = max(time.monotonic() - started, 1e-9)
            LOGGER.info(
                "batch=%d selected=%d skipped=%d upserted=%d rate=%.2f point/s",
                batch_number,
                selected,
                skipped,
                upserted,
                upserted / elapsed,
            )
    finally:
        connection.close()

    return {
        "selected": selected,
        "skipped": skipped,
        "embedded": embedded,
        "upserted": upserted,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def _qdrant_filter(models: Any, bucket: Mapping[str, Any]) -> Any:
    conditions = [
        models.FieldCondition(key="ticker", match=models.MatchValue(value=bucket["ticker"])),
        models.FieldCondition(key="year", match=models.MatchValue(value=int(bucket["year"]))),
        models.FieldCondition(
            key="report_type", match=models.MatchValue(value=bucket["report_type"])
        ),
    ]
    if bucket.get("table_type"):
        conditions.append(
            models.FieldCondition(
                key="table_type", match=models.MatchValue(value=bucket["table_type"])
            )
        )
    return models.Filter(must=conditions)


def verify_ingestion(
    config: Config,
    *,
    ticker: str | None = None,
    year: int | None = None,
    sample_size: int = 100,
    skip_count: bool = False,
    client: Any | None = None,
) -> dict[str, Any]:
    client = client or create_qdrant_client(config)
    _, models = _qdrant_imports()
    expected_count = 0
    sample: list[dict[str, Any]] = []
    for item in iter_manifest(config.manifest_path):
        if not _matches_scope(item, ticker, year):
            continue
        expected_count += 1
        if len(sample) < max(sample_size, 0):
            sample.append(item)
    count_value: int | None = None
    if not skip_count:
        conditions = []
        if ticker:
            conditions.append(
                models.FieldCondition(
                    key="ticker", match=models.MatchValue(value=ticker.upper())
                )
            )
        if year is not None:
            conditions.append(
                models.FieldCondition(key="year", match=models.MatchValue(value=year))
            )
        query_filter = models.Filter(must=conditions) if conditions else None
        count_value = int(
            client.count(
                collection_name=config.collection_name,
                count_filter=query_filter,
                exact=True,
            ).count
        )
        if count_value != expected_count:
            raise BaselineError(
                f"Count mismatch: Qdrant={count_value}, manifest={expected_count}"
            )

    by_id = {item["point_id"]: item for item in sample}
    points = client.retrieve(
        collection_name=config.collection_name,
        ids=list(by_id),
        with_payload=True,
        with_vectors=False,
    ) if by_id else []
    if len(points) != len(sample):
        raise BaselineError(
            f"Sample missing: Qdrant={len(points)}, manifest={len(sample)}"
        )
    for point in points:
        point_id = str(point.id)
        expected = by_id.get(point_id)
        if expected is None:
            raise BaselineError(f"Qdrant trả point ngoài sample: {point_id}")
        payload = dict(point.payload or {})
        if set(payload) != set(PAYLOAD_FIELDS):
            raise BaselineError(
                f"Payload {point_id} không đúng allowlist: {sorted(payload)}"
            )
        if payload != expected["payload"]:
            raise BaselineError(f"Payload mismatch tại point {point_id}")
    return {
        "expected_count": expected_count,
        "qdrant_count": count_value,
        "sample_verified": len(points),
        "payload_fields": list(PAYLOAD_FIELDS),
    }


def load_company_aliases(path: Path) -> tuple[set[str], dict[str, list[str]]]:
    """Return known tickers and normalized company aliases by ticker."""
    known: set[str] = set()
    aliases: dict[str, list[str]] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ticker = normalize_text(row.get("Mã CK") or row.get("ticker")).upper()
            company = normalize_text(row.get("Tên công ty") or row.get("company"))
            if not ticker or not company:
                continue
            known.add(ticker)
            variants = {company, _strip_company_prefix(company)}
            aliases[ticker] = sorted(
                {value for value in variants if len(fold_text(value)) >= 5},
                key=len,
                reverse=True,
            )
    if not known:
        raise BaselineError(f"Không đọc được ticker từ {path}")
    return known, aliases


def _strip_company_prefix(name: str) -> str:
    folded_prefixes = (
        "ngân hàng thương mại cổ phần ",
        "ngân hàng tmcp ",
        "ngân hàng ",
        "tổng công ty cổ phần ",
        "tổng công ty ",
        "tập đoàn ",
        "công ty cổ phần ",
        "công ty tnhh ",
        "công ty ",
        "ctcp ",
    )
    result = normalize_text(name)
    changed = True
    while changed:
        changed = False
        folded = fold_text(result)
        for prefix in folded_prefixes:
            folded_prefix = fold_text(prefix)
            if folded.startswith(folded_prefix):
                words_to_remove = len(prefix.split())
                result = " ".join(result.split()[words_to_remove:])
                changed = True
                break
    result = re.sub(r"\s*[-–—]\s*(?:CTCP|Công ty cổ phần)\s*$", "", result, flags=re.I)
    return normalize_text(result)


def parse_years(question: str) -> list[int]:
    years: set[int] = set()
    for match in YEAR_RANGE_PATTERN.finditer(question):
        start, end = int(match.group(1)), int(match.group(2))
        low, high = sorted((start, end))
        if high - low > 20:
            continue
        years.update(range(max(low, MIN_YEAR), min(high, MAX_YEAR) + 1))
    for value in YEAR_PATTERN.findall(question):
        year = int(value)
        if MIN_YEAR <= year <= MAX_YEAR:
            years.add(year)
    return sorted(years)


def resolve_tickers(
    question: str, stock_codes_path: Path
) -> tuple[list[str], dict[str, list[str]]]:
    known, aliases = load_company_aliases(stock_codes_path)
    direct = {
        token.upper()
        for token in re.findall(r"(?<![A-Za-z0-9])[A-Za-z]{3}(?![A-Za-z0-9])", question)
        if token.upper() in known
    }
    folded_question = f" {fold_text(question)} "
    matched_aliases: dict[str, list[str]] = {}
    for ticker, names in aliases.items():
        for name in names:
            folded_name = fold_text(name)
            if len(folded_name) >= 5 and f" {folded_name} " in folded_question:
                direct.add(ticker)
                matched_aliases.setdefault(ticker, []).append(name)
                break
    strongly_explicit = {
        token.upper()
        for token in re.findall(
            r"(?:\(\s*|\bmã\s+)([A-Za-z]{3})(?:\s*\)|\b)", question, flags=re.I
        )
        if token.upper() in known
    }
    # A legal company name can end in a token that is also another ticker,
    # e.g. "CTCP Chứng khoán FPT" maps to FTS, not ticker FPT.
    for company_ticker, names in matched_aliases.items():
        alias_tokens = {
            token.upper()
            for name in names
            for token in re.findall(r"(?<![A-Za-z0-9])[A-Za-z]{3}(?![A-Za-z0-9])", name)
        }
        direct = {
            ticker
            for ticker in direct
            if ticker == company_ticker
            or ticker in strongly_explicit
            or ticker not in alias_tokens
        }
    direct.update(matched_aliases)
    return sorted(direct), matched_aliases


def parse_report_types(question: str) -> list[str]:
    folded = fold_text(question)
    selected: list[str] = []
    if "cong ty me" in folded or re.search(r"\brieng\b", folded):
        selected.append("separate")
    if "hop nhat" in folded:
        selected.append("consolidated")
    if "tong hop" in folded:
        selected.append("aggregated")
    return selected or ["consolidated", "separate", "aggregated", "other"]


def parse_table_type(question: str) -> str | None:
    folded = fold_text(question)
    if "bang can doi" in folded:
        return "balance_sheet"
    if "ket qua kinh doanh" in folded or "bao cao ket qua" in folded:
        return "income_statement"
    if "luu chuyen tien te" in folded or "dong tien" in folded:
        return "cash_flow"
    if "thuyet minh" in folded:
        return "note_table"
    return None


def build_semantic_query(
    question: str,
    tickers: Sequence[str],
    matched_aliases: Mapping[str, Sequence[str]],
) -> str:
    semantic = normalize_text(question)
    for ticker in tickers:
        semantic = re.sub(
            rf"(?<![A-Za-z0-9]){re.escape(ticker)}(?![A-Za-z0-9])",
            " ",
            semantic,
            flags=re.IGNORECASE,
        )
        for alias in matched_aliases.get(ticker, []):
            semantic = re.sub(re.escape(alias), " ", semantic, flags=re.IGNORECASE)
    semantic = YEAR_RANGE_PATTERN.sub(" ", semantic)
    semantic = re.sub(r"\bnăm\s*(?=20\d{2}\b)", " ", semantic, flags=re.I)
    semantic = YEAR_PATTERN.sub(" ", semantic)
    report_phrases = (
        "theo báo cáo tài chính hợp nhất",
        "theo báo cáo tài chính riêng",
        "theo báo cáo hợp nhất",
        "theo báo cáo riêng",
        "báo cáo tài chính hợp nhất",
        "báo cáo tài chính riêng",
        "báo cáo hợp nhất",
        "báo cáo riêng",
        "công ty mẹ",
        "hợp nhất",
        "riêng",
        "consolidated",
        "separate",
        "aggregated",
        "other",
    )
    for phrase in report_phrases:
        semantic = re.sub(re.escape(phrase), " ", semantic, flags=re.IGNORECASE)
    if matched_aliases:
        semantic = re.sub(
            r"\b(?:CTCP|công ty(?:\s+cổ phần)?|tập đoàn|tổng công ty|"
            r"ngân hàng(?:\s+TMCP)?)\b",
            " ",
            semantic,
            flags=re.I,
        )
    semantic = re.sub(r"\b(?:là\s+)?bao\s+nhiêu\b", " ", semantic, flags=re.I)
    semantic = re.sub(r"\b(?:của|năm)\b", " ", semantic, flags=re.I)
    semantic = re.sub(r"\(\s*\)", " ", semantic)
    semantic = re.sub(r"\s+", " ", semantic).strip(" ,.;:-–—?")
    previous = None
    while semantic != previous:
        previous = semantic
        semantic = re.sub(
            r"\s+\b(?:của|tại|trong|năm|theo|từ|đến|giai đoạn)\b\s*$",
            "",
            semantic,
            flags=re.I,
        ).strip()
    if not semantic:
        raise RoutingError("Câu hỏi không còn nội dung tài chính sau khi bỏ identity token")
    return semantic


def parse_query_buckets(
    question: str,
    stock_codes_path: Path = DEFAULT_STOCK_CODES_PATH,
    *,
    ticker_overrides: Sequence[str] | None = None,
    year_overrides: Sequence[int] | None = None,
    report_type_overrides: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    resolved_tickers, matched_aliases = resolve_tickers(question, stock_codes_path)
    tickers = sorted({value.upper() for value in (ticker_overrides or resolved_tickers)})
    years = sorted(set(year_overrides or parse_years(question)))
    report_types = list(dict.fromkeys(report_type_overrides or parse_report_types(question)))
    if not tickers:
        raise RoutingError("Không resolve được ticker; baseline không search global")
    if not years:
        raise RoutingError("Không resolve được năm 2015–2025; baseline không search global")
    invalid_years = [value for value in years if not MIN_YEAR <= int(value) <= MAX_YEAR]
    if invalid_years:
        raise RoutingError(f"Năm ngoài phạm vi dataset: {invalid_years}")
    allowed_report_types = REPORT_TYPES
    if any(value not in allowed_report_types for value in report_types):
        raise RoutingError(f"report_type không hợp lệ: {report_types}")
    table_type = parse_table_type(question)
    buckets = [
        {
            "ticker": ticker,
            "year": int(year),
            "report_type": report_type,
            "table_type": table_type,
        }
        for ticker in tickers
        for year in years
        for report_type in report_types
    ]
    semantic_query = build_semantic_query(question, tickers, matched_aliases)
    return buckets, semantic_query


def retrieve_tables(
    question: str,
    config: Config | None = None,
    *,
    ticker_overrides: Sequence[str] | None = None,
    year_overrides: Sequence[int] | None = None,
    report_type_overrides: Sequence[str] | None = None,
    top_k_per_bucket: int | None = None,
    max_candidates: int = 50,
    client: Any | None = None,
    encoder: EmbeddingModel | None = None,
) -> list[dict[str, Any]]:
    config = config or Config.from_env()
    buckets, semantic_query = parse_query_buckets(
        question,
        config.stock_codes_path,
        ticker_overrides=ticker_overrides,
        year_overrides=year_overrides,
        report_type_overrides=report_type_overrides,
    )
    per_bucket = top_k_per_bucket or (10 if len(buckets) == 1 else 5)
    if per_bucket <= 0 or max_candidates <= 0:
        raise ValueError("top-k và max-candidates phải > 0")
    encoder = encoder or EmbeddingModel(config)
    query_vector = encoder.encode_query([semantic_query])[0]
    client = client or create_qdrant_client(config)
    _, models = _qdrant_imports()
    collection = config.alias_name or config.collection_name

    merged: dict[str, dict[str, Any]] = {}
    for bucket in buckets:
        response = client.query_points(
            collection_name=collection,
            query=query_vector,
            using=DENSE_VECTOR_NAME,
            query_filter=_qdrant_filter(models, bucket),
            limit=per_bucket,
            with_payload=True,
            with_vectors=False,
        )
        for hit in response.points:
            payload = dict(hit.payload or {})
            if set(payload) != set(PAYLOAD_FIELDS):
                raise BaselineError(
                    f"Point {hit.id} có payload ngoài allowlist: {sorted(payload)}"
                )
            table_id = payload["table_id"]
            candidate = {
                "table_id": table_id,
                "doc_id": payload["doc_id"],
                "score": float(hit.score),
                "bucket": dict(bucket),
            }
            previous = merged.get(table_id)
            if previous is None or candidate["score"] > previous["score"]:
                merged[table_id] = candidate
    return sorted(merged.values(), key=lambda value: value["score"], reverse=True)[
        :max_candidates
    ]


def load_tables(
    results: Iterable[str | Mapping[str, Any]],
    project_root: Path = PROJECT_ROOT,
    **read_csv_kwargs: Any,
) -> dict[str, Any]:
    """Load retrieved CSVs; this is the only baseline function that reads them."""
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise BaselineError("Thiếu pandas. Chạy: pip install -r requirements.txt") from exc
    loaded: dict[str, Any] = {}
    for result in results:
        table_id = result if isinstance(result, str) else result.get("table_id")
        safe_id = validate_table_id(normalize_text(table_id))
        if safe_id not in loaded:
            loaded[safe_id] = pd.read_csv(
                resolve_csv_path(safe_id, project_root), **read_csv_kwargs
            )
    return loaded


def doctor(
    config: Config,
    *,
    check_model: bool = True,
    check_qdrant: bool = True,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "metadata_exists": config.metadata_path.is_file(),
        "stock_codes_exists": config.stock_codes_path.is_file(),
    }
    if not all(report.values()):
        raise BaselineError(f"Thiếu input local: {report}")
    if check_model:
        vectors = EmbeddingModel(config).encode(
            ["Loại bảng: Báo cáo kết quả kinh doanh\nChỉ tiêu: Doanh thu thuần"]
        )
        report["model"] = {
            "id": config.embedding_model,
            "revision": config.embedding_revision,
            "shape": [len(vectors), len(vectors[0])],
            "finite": all(math.isfinite(value) for value in vectors[0]),
        }
    if check_qdrant:
        client = create_qdrant_client(config)
        collections = client.get_collections().collections
        report["qdrant"] = {
            "connected": True,
            "url": config.qdrant_url,
            "collection_exists": any(
                item.name == config.collection_name for item in collections
            ),
        }
    return report


def reconcile_points(
    config: Config,
    *,
    prune: bool = False,
    confirm: bool = False,
    client: Any | None = None,
) -> dict[str, Any]:
    """Compare IDs and optionally delete Qdrant points absent from the manifest."""
    if prune and not confirm:
        raise BaselineError("Prune yêu cầu đồng thời --prune --confirm")
    expected = {item["point_id"] for item in iter_manifest(config.manifest_path)}
    client = client or create_qdrant_client(config)
    _, models = _qdrant_imports()
    actual: set[str] = set()
    offset: Any = None
    while True:
        records, offset = client.scroll(
            collection_name=config.collection_name,
            limit=1000,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        actual.update(str(point.id) for point in records)
        if offset is None:
            break
    extra = sorted(actual - expected)
    missing = sorted(expected - actual)
    deleted = 0
    if prune and extra:
        for batch in _batches(extra, config.upsert_batch_size):
            client.delete(
                collection_name=config.collection_name,
                points_selector=models.PointIdsList(points=batch),
                wait=True,
            )
            deleted += len(batch)
    return {
        "manifest_points": len(expected),
        "qdrant_points": len(actual),
        "missing_in_qdrant": len(missing),
        "extra_in_qdrant": len(extra),
        "deleted": deleted,
        "dry_run": not prune,
        "extra_sample": extra[:20],
        "missing_sample": missing[:20],
    }


def self_test() -> dict[str, Any]:
    """Dependency-free unit/smoke tests kept in this single source file."""
    import io
    from unittest.mock import patch

    checks: list[str] = []
    sample = {
        "table_id": "AAA_financial_statements_2023_consolidated_table_9",
        "doc_id": "AAA_financial_statements_2023_consolidated",
        "ticker": "AAA",
        "company_name": "Công ty cổ phần An Phát",
        "year": 2023,
        "report_type": "consolidated",
        "table_type": "income_statement",
        "start_line": 321,
        "csv_path": "data/AAA_financial_statements_2023_consolidated_table_9.csv",
        "semantic_fields": [
            {"canonical_name_vi": "Doanh thu thuần", "canonical_name_en": "Net revenue"},
            {"canonical_name_vi": "Lợi nhuận sau thuế"},
        ],
        "retrieval_context": {
            "semantic_summary": "Bảng AAA năm 2023 hợp nhất về doanh thu và lợi nhuận",
            "keywords": ["Doanh thu thuần", "AAA", "2023", "Lợi nhuận sau thuế"],
        },
    }
    index_text, quality = build_index_text(sample)
    folded_index = fold_text(index_text)
    assert "doanh thu" in folded_index
    assert "net revenue" not in folded_index
    english_only = dict(sample)
    english_only["semantic_fields"] = [
        {"canonical_name_en": "Operating profit"},
    ]
    english_only["retrieval_context"] = {
        "semantic_summary": "Operating profit and net revenue",
        "keywords": ["Operating profit", "Net revenue"],
    }
    english_index, _ = build_index_text(english_only)
    assert "Operating profit" not in english_index
    assert "Net revenue" not in english_index
    assert "aaa" not in folded_index
    assert "2023" not in index_text
    assert "hop nhat" not in folded_index
    assert quality == "ok"
    checks.append("metadata_only_index_text")

    payload = build_payload(sample, index_text)
    assert tuple(payload) == PAYLOAD_FIELDS
    assert set(payload) == set(PAYLOAD_FIELDS)
    assert "csv_path" not in payload
    assert payload["start_line"] == 321
    invalid_start_line = dict(sample)
    invalid_start_line["start_line"] = 0
    try:
        build_payload(invalid_start_line, index_text)
    except BaselineError as exc:
        assert str(exc) == "missing_or_invalid_start_line"
    else:
        raise AssertionError("start_line không hợp lệ vẫn được chấp nhận")
    checks.append("nine_field_payload_and_start_line_validation")

    point_id = make_point_id(sample["table_id"])
    assert point_id == make_point_id(sample["table_id"])
    assert point_id != make_point_id(sample["table_id"] + "_other")
    original_hash = compute_content_hash(sample["table_id"], index_text)
    assert original_hash != compute_content_hash(sample["table_id"], index_text + " changed")
    assert compute_content_hash(
        sample["table_id"], index_text, payload=payload
    ) != compute_content_hash(
        sample["table_id"],
        index_text,
        payload={**payload, "company_name": "Tên công ty đã đổi"},
    )
    checks.append("deterministic_id_and_hash")

    with tempfile.TemporaryDirectory(prefix="finlens-index-test-") as temp_name:
        root = Path(temp_name)
        data_dir = root / "data"
        metadata_dir = root / "metadata"
        intermediate_dir = root / "intermediate"
        data_dir.mkdir()
        metadata_dir.mkdir()
        intermediate_dir.mkdir()

        csv_file = data_dir / f"{sample['table_id']}.csv"
        csv_file.write_text("must,not,be,read\n", encoding="utf-8")
        assert resolve_csv_path(sample["table_id"], root) == csv_file.resolve()
        try:
            resolve_csv_path("../escape", root)
        except BaselineError:
            pass
        else:
            raise AssertionError("Path traversal không bị chặn")
        checks.append("safe_csv_resolver")

        unknown = dict(sample)
        unknown["table_id"] = "AAA_financial_statements_2023_consolidated_table_10"
        unknown["csv_path"] = f"data/{unknown['table_id']}.csv"
        unknown["table_type"] = "note_table"
        unknown["semantic_fields"] = []
        unknown["retrieval_context"] = {
            "semantic_summary": "Thuyết minh note_unknown AAA 2023 hợp nhất",
            "keywords": ["Chi phí lãi vay"],
        }
        (data_dir / f"{unknown['table_id']}.csv").write_text(
            "also,must,not,be,read\n", encoding="utf-8"
        )
        uninformative = dict(unknown)
        uninformative["table_id"] = (
            "AAA_financial_statements_2023_consolidated_table_11"
        )
        uninformative["csv_path"] = f"data/{uninformative['table_id']}.csv"
        uninformative["retrieval_context"] = {
            "semantic_summary": "Thuyết minh note_unknown AAA 2023 hợp nhất",
            "keywords": ["note_unknown"],
        }
        (data_dir / f"{uninformative['table_id']}.csv").write_text(
            "boilerplate,must,not,be,indexed\n", encoding="utf-8"
        )
        metadata_path = metadata_dir / "tables_metadata.json"
        metadata_path.write_text(
            json.dumps([sample, unknown, uninformative], ensure_ascii=False),
            encoding="utf-8",
        )
        config = Config(
            project_root=root,
            metadata_path=metadata_path,
            stock_codes_path=root / "code_stock.csv",
            manifest_path=intermediate_dir / "manifest.jsonl",
            rejects_path=intermediate_dir / "rejects.jsonl",
            state_path=root / ".cache" / "state.sqlite3",
        )

        original_io_open = io.open

        def guarded_open(file: Any, *args: Any, **kwargs: Any) -> Any:
            if isinstance(file, (str, os.PathLike)) and Path(file).suffix.lower() == ".csv":
                raise AssertionError(f"Indexing đã cố mở CSV: {file}")
            return original_io_open(file, *args, **kwargs)

        with patch("io.open", new=guarded_open):
            stats = build_manifest(config, check_files=True, source_hash=False)
        assert stats["accepted"] == 2
        assert stats["rejected_by_reason"] == {"uninformative_note_metadata": 1}
        manifest_items = list(iter_manifest(config.manifest_path))
        assert manifest_items[1]["metadata_quality"] == "note_unknown"
        assert "Nội dung:" not in manifest_items[1]["index_text"]
        assert all(set(item["payload"]) == set(PAYLOAD_FIELDS) for item in manifest_items)
        checks.append("selective_note_indexing_without_opening_csv")
        assert manifest_is_current(config)
        assert not manifest_is_current(
            replace(config, embedding_revision="different-revision")
        )
        assert _effective_argv([]) == ["index"]
        checks.append("default_command_and_manifest_reuse")

        class FakeAliasModel:
            def __init__(self, **values: Any) -> None:
                self.__dict__.update(values)

        class FakeAliasModels:
            DeleteAliasOperation = FakeAliasModel
            DeleteAlias = FakeAliasModel
            CreateAliasOperation = FakeAliasModel
            CreateAlias = FakeAliasModel

        class FakeAliasClient:
            def __init__(self) -> None:
                self.operations: list[Any] = []

            def get_aliases(self) -> Any:
                alias_item = FakeAliasModel(
                    alias_name="finlens_tables_current",
                    collection_name="finlens_tables_metadata_granite_97m_multilingual_r2_v0",
                )
                return FakeAliasModel(aliases=[alias_item])

            def update_collection_aliases(
                self, *, change_aliases_operations: list[Any]
            ) -> None:
                self.operations = change_aliases_operations

        fake_alias_client = FakeAliasClient()
        alias_result = activate_alias(
            fake_alias_client,
            "finlens_tables_metadata_granite_97m_multilingual_r2_v1",
            "finlens_tables_current",
            FakeAliasModels,
        )
        assert alias_result["changed"] is True
        assert alias_result["previous_collection"].endswith("_v0")
        assert len(fake_alias_client.operations) == 2
        checks.append("verified_collection_alias_switch")

        stock_path = root / "code_stock.csv"
        stock_path.write_text(
            "Mã CK,Tên công ty\n"
            "VJC,CTCP Hàng không VietJet\n"
            "AAA,Công ty cổ phần An Phát\n"
            "FPT,CTCP FPT\n"
            "FTS,CTCP Chứng khoán FPT\n",
            encoding="utf-8-sig",
        )
        buckets, semantic_query = parse_query_buckets(
            "Doanh thu của CTCP Hàng không VietJet năm 2022 theo báo cáo công ty mẹ?",
            stock_path,
        )
        assert buckets == [
            {
                "ticker": "VJC",
                "year": 2022,
                "report_type": "separate",
                "table_type": None,
            }
        ]
        folded_query = fold_text(semantic_query)
        assert "doanh thu" in folded_query
        assert "vietjet" not in folded_query
        assert "2022" not in semantic_query
        checks.append("strict_query_router")

        assert parse_years("giai đoạn 2019 đến 2021 và năm 2023") == [2019, 2020, 2021, 2023]
        ambiguous, _ = resolve_tickers(
            "Lợi nhuận của CTCP Chứng khoán FPT năm 2023", stock_path
        )
        assert ambiguous == ["FTS"]
        checks.append("year_range_expansion")

        class FakePointStruct:
            def __init__(self, **kwargs: Any):
                self.id = kwargs["id"]
                self.vector = kwargs["vector"]
                self.payload = kwargs["payload"]

        class FakeModels:
            PointStruct = FakePointStruct

        class FakeEncoder:
            def encode(self, texts: Sequence[str]) -> list[list[float]]:
                return [[1.0] + [0.0] * (VECTOR_SIZE - 1) for _ in texts]

        class FakeClient:
            def __init__(self) -> None:
                self.uploads: list[list[Any]] = []

            def upload_points(self, **kwargs: Any) -> None:
                self.uploads.append(kwargs["points"])

        fake_client = FakeClient()
        module = sys.modules[__name__]
        with patch.object(module, "_qdrant_imports", return_value=(None, FakeModels)), patch.object(
            module, "ensure_collection", return_value={}
        ):
            first_run = upsert_points(
                config,
                limit=1,
                force=True,
                client=fake_client,
                encoder=FakeEncoder(),  # type: ignore[arg-type]
            )
            second_run = upsert_points(
                config,
                limit=1,
                client=fake_client,
                encoder=FakeEncoder(),  # type: ignore[arg-type]
            )
        assert first_run["upserted"] == 1
        assert second_run["upserted"] == 0 and second_run["skipped"] == 1
        uploaded = fake_client.uploads[0][0]
        assert set(uploaded.vector) == {DENSE_VECTOR_NAME}
        assert set(uploaded.payload) == set(PAYLOAD_FIELDS)
        checks.append("named_vector_upsert_and_resume")

        state = init_state_db(config.state_path)
        hashes = _successful_hashes(state, config.collection_name)
        state.close()
        assert hashes[manifest_items[0]["point_id"]] == manifest_items[0]["content_hash"]
        checks.append("sqlite_resume_checkpoint")

    return {"status": "ok", "checks": checks, "count": len(checks)}


def _load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE pairs without overriding process environment."""
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                os.environ.setdefault(key, value)


def manifest_is_current(config: Config) -> bool:
    """Return whether the existing manifest matches the current source/config."""
    if not config.manifest_path.is_file() or not config.metadata_path.is_file():
        return False
    try:
        header = read_manifest_header(config.manifest_path)
        _validate_manifest_header(header, config)
        source_stat = config.metadata_path.stat()
        return (
            Path(header["source_path"]).resolve() == config.metadata_path.resolve()
            and header.get("source_size") == source_stat.st_size
            and header.get("source_mtime_ns") == source_stat.st_mtime_ns
            and header.get("dataset_id") == DATASET_ID
            and header.get("dataset_revision") == DATASET_REVISION
        )
    except (BaselineError, KeyError, OSError, TypeError, ValueError):
        return False


def _check_indexing_dependencies(config: Config) -> None:
    """Fail before the expensive metadata scan when runtime packages are absent."""
    _qdrant_imports()
    try:
        import sentence_transformers  # noqa: F401  # type: ignore
    except ImportError as exc:
        raise BaselineError(
            "Thiếu sentence-transformers. Chạy: pip install -r requirements.txt"
        ) from exc


def run_indexing_pipeline(
    config: Config,
    *,
    force: bool = False,
    rebuild_manifest: bool = False,
    check_files: bool = True,
    source_hash: bool = True,
    verify: bool = True,
) -> dict[str, Any]:
    """Run the complete metadata-only indexing workflow for ``python data_indexing.py``."""
    LOGGER.info("Bắt đầu metadata-only indexing lên Qdrant")
    _check_indexing_dependencies(config)

    client = create_qdrant_client(config)
    collection_result = ensure_collection(config, client)

    reuse_manifest = not rebuild_manifest and manifest_is_current(config)
    if reuse_manifest:
        LOGGER.info("Manifest hiện tại còn hợp lệ; bỏ qua bước build lại")
        manifest_result: dict[str, Any] = {
            "reused": True,
            "manifest": str(config.manifest_path),
        }
    else:
        manifest_result = build_manifest(
            config,
            check_files=check_files,
            source_hash=source_hash,
        )
        manifest_result["reused"] = False

    # A newly created collection cannot reuse checkpoint rows from an older
    # collection with the same name.
    effective_force = force or bool(collection_result.get("created"))
    encoder = EmbeddingModel(config)
    upsert_result = upsert_points(
        config,
        force=effective_force,
        client=client,
        encoder=encoder,
    )

    verification_result: dict[str, Any] | None = None
    alias_result: dict[str, Any] | None = None
    if verify:
        verification_result = verify_ingestion(config, client=client)
        alias_result = activate_alias(
            client,
            config.collection_name,
            config.alias_name,
        )
    else:
        LOGGER.warning(
            "Bỏ qua verify nên alias %s chưa được chuyển sang %s",
            config.alias_name,
            config.collection_name,
        )
    LOGGER.info("Indexing hoàn tất: upserted=%d", upsert_result["upserted"])
    return {
        "status": "ok",
        "manifest": manifest_result,
        "collection": collection_result,
        "upsert": upsert_result,
        "verify": verification_result,
        "alias": alias_result,
    }


def _effective_argv(argv: Sequence[str] | None) -> list[str]:
    values = list(sys.argv[1:] if argv is None else argv)
    return values or ["index"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "ViFinQA metadata-only Granite/Qdrant baseline. "
            "Không truyền command sẽ tự chạy full indexing."
        )
    )
    parser.add_argument("--metadata", type=Path, help="Override tables_metadata.json")
    parser.add_argument("--manifest", type=Path, help="Override manifest JSONL")
    parser.add_argument("--rejects", type=Path, help="Override rejects JSONL")
    parser.add_argument("--state", type=Path, help="Override SQLite checkpoint")
    parser.add_argument("--stock-codes", type=Path, help="Override code_stock.csv")
    parser.add_argument("--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)

    index = commands.add_parser(
        "index", help="Build/reuse manifest and index all metadata into Qdrant"
    )
    index.add_argument("--force", action="store_true", help="Re-embed all points")
    index.add_argument(
        "--rebuild-manifest", action="store_true", help="Rebuild even when source is unchanged"
    )
    index.add_argument("--no-check-files", action="store_true")
    index.add_argument("--skip-source-hash", action="store_true")
    index.add_argument("--skip-verify", action="store_true")

    build = commands.add_parser("build-manifest", help="Stream metadata and build JSONL")
    build.add_argument("--ticker")
    build.add_argument("--year", type=int)
    build.add_argument("--limit", type=int)
    build.add_argument("--no-check-files", action="store_true")
    build.add_argument("--skip-source-hash", action="store_true")

    commands.add_parser("stats", help="Show manifest statistics")

    doctor_parser = commands.add_parser("doctor", help="Check local inputs, model and Qdrant")
    doctor_parser.add_argument("--skip-model", action="store_true")
    doctor_parser.add_argument("--skip-qdrant", action="store_true")

    commands.add_parser("init-collection", help="Create/validate collection and indexes")

    upsert = commands.add_parser("upsert", help="Embed manifest and upload points")
    upsert.add_argument("--ticker")
    upsert.add_argument("--year", type=int)
    upsert.add_argument("--limit", type=int)
    upsert.add_argument("--resume", action="store_true", help="Default: skip unchanged success")
    upsert.add_argument("--force", action="store_true", help="Re-embed selected points")

    verify = commands.add_parser("verify", help="Compare manifest with Qdrant")
    verify.add_argument("--ticker")
    verify.add_argument("--year", type=int)
    verify.add_argument("--sample-size", type=int, default=100)
    verify.add_argument("--skip-count", action="store_true")

    retrieve = commands.add_parser("retrieve", help="Route and retrieve table metadata")
    retrieve.add_argument("question")
    retrieve.add_argument("--ticker", action="append", dest="tickers")
    retrieve.add_argument("--year", action="append", type=int, dest="years")
    retrieve.add_argument(
        "--report-type",
        action="append",
        choices=("consolidated", "separate", "aggregated", "other"),
        dest="report_types",
    )
    retrieve.add_argument("--top-k-per-bucket", type=int)
    retrieve.add_argument("--max-candidates", type=int, default=50)
    retrieve.add_argument("--load", action="store_true", help="Load retrieved CSVs with pandas")

    route = commands.add_parser("route", help="Inspect query buckets without model/Qdrant")
    route.add_argument("question")
    route.add_argument("--ticker", action="append", dest="tickers")
    route.add_argument("--year", action="append", type=int, dest="years")
    route.add_argument(
        "--report-type",
        action="append",
        choices=("consolidated", "separate", "aggregated", "other"),
        dest="report_types",
    )

    resolve = commands.add_parser("resolve", help="Resolve a retrieved table_id to local CSV")
    resolve.add_argument("table_id")

    reconcile = commands.add_parser("reconcile", help="Diff manifest IDs and Qdrant IDs")
    reconcile.add_argument("--dry-run", action="store_true")
    reconcile.add_argument("--prune", action="store_true")
    reconcile.add_argument("--confirm", action="store_true")

    commands.add_parser("self-test", help="Run dependency-free tests in this file")
    return parser


def _config_from_args(args: argparse.Namespace) -> Config:
    config = Config.from_env()
    changes: dict[str, Any] = {}
    for argument, field_name in (
        ("metadata", "metadata_path"),
        ("manifest", "manifest_path"),
        ("rejects", "rejects_path"),
        ("state", "state_path"),
        ("stock_codes", "stock_codes_path"),
    ):
        value = getattr(args, argument, None)
        if value is not None:
            changes[field_name] = Path(value).resolve()
    return replace(config, **changes) if changes else config


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def main(argv: Sequence[str] | None = None) -> int:
    # PowerShell may expose a legacy code page; keep Vietnamese CLI output valid.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="backslashreplace")
    _load_dotenv(PROJECT_ROOT / ".env")
    parser = _build_parser()
    args = parser.parse_args(_effective_argv(argv))
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = _config_from_args(args)
    try:
        if args.command == "index":
            result = run_indexing_pipeline(
                config,
                force=args.force,
                rebuild_manifest=args.rebuild_manifest,
                check_files=not args.no_check_files,
                source_hash=not args.skip_source_hash,
                verify=not args.skip_verify,
            )
        elif args.command == "build-manifest":
            result = build_manifest(
                config,
                ticker=args.ticker,
                year=args.year,
                limit=args.limit,
                check_files=not args.no_check_files,
                source_hash=not args.skip_source_hash,
            )
        elif args.command == "stats":
            result = manifest_stats(config.manifest_path)
        elif args.command == "doctor":
            result = doctor(
                config,
                check_model=not args.skip_model,
                check_qdrant=not args.skip_qdrant,
            )
        elif args.command == "init-collection":
            result = ensure_collection(config)
        elif args.command == "upsert":
            if args.resume and args.force:
                raise BaselineError("Không dùng đồng thời --resume và --force")
            result = upsert_points(
                config,
                ticker=args.ticker,
                year=args.year,
                limit=args.limit,
                force=args.force,
            )
        elif args.command == "verify":
            result = verify_ingestion(
                config,
                ticker=args.ticker,
                year=args.year,
                sample_size=args.sample_size,
                skip_count=args.skip_count,
            )
        elif args.command == "retrieve":
            result = retrieve_tables(
                args.question,
                config,
                ticker_overrides=args.tickers,
                year_overrides=args.years,
                report_type_overrides=args.report_types,
                top_k_per_bucket=args.top_k_per_bucket,
                max_candidates=args.max_candidates,
            )
            if args.load:
                frames = load_tables(result, config.project_root)
                result = {
                    "results": result,
                    "loaded_shapes": {
                        table_id: list(frame.shape) for table_id, frame in frames.items()
                    },
                }
        elif args.command == "route":
            buckets, semantic_query = parse_query_buckets(
                args.question,
                config.stock_codes_path,
                ticker_overrides=args.tickers,
                year_overrides=args.years,
                report_type_overrides=args.report_types,
            )
            result = {"semantic_query": semantic_query, "buckets": buckets}
        elif args.command == "resolve":
            result = {
                "table_id": args.table_id,
                "csv_path": str(resolve_csv_path(args.table_id, config.project_root)),
            }
        elif args.command == "reconcile":
            if args.dry_run and args.prune:
                raise BaselineError("Chọn --dry-run hoặc --prune, không chọn cả hai")
            result = reconcile_points(config, prune=args.prune, confirm=args.confirm)
        elif args.command == "self-test":
            result = self_test()
        else:
            parser.error(f"Command không được hỗ trợ: {args.command}")
            return 2
        _print_json(result)
        return 0
    except (BaselineError, FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2
    except Exception as exc:  # pragma: no cover - external runtime failures
        if args.verbose:
            LOGGER.exception("Indexing thất bại")
        else:
            LOGGER.error("Indexing thất bại (%s): %s", exc.__class__.__name__, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
