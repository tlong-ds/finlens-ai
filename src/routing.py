"""Strict Vietnamese query routing aligned with the Qdrant payload schema."""

from __future__ import annotations

import csv
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.contracts import FILTER_FIELDS, MAX_YEAR, MIN_YEAR, REPORT_TYPES, TABLE_TYPES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STOCK_CODES_PATH = PROJECT_ROOT / "ViFinQA" / "code_stock.csv"
YEAR_PATTERN = re.compile(r"(?<!\d)(20\d{2})(?!\d)")
YEAR_RANGE_PATTERN = re.compile(
    r"(?<!\d)(20\d{2})\s*(?:-|–|—|đến|den|tới|toi)\s*(20\d{2})(?!\d)",
    re.IGNORECASE,
)


class QueryRoutingError(ValueError):
    """Raised when a question cannot be mapped to safe metadata buckets."""


def _as_filter_values(values: Iterable[str | int]) -> list[str | int]:
    return list(values)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def fold_text(value: Any) -> str:
    text = normalize_text(value).lower().replace("đ", "d")
    return "".join(
        char
        for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )


def _strip_company_prefix(name: str) -> str:
    prefixes = (
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
        for prefix in prefixes:
            if folded.startswith(fold_text(prefix)):
                result = " ".join(result.split()[len(prefix.split()) :])
                changed = True
                break
    result = re.sub(r"\s*[-–—]\s*(?:CTCP|Công ty cổ phần)\s*$", "", result, flags=re.I)
    return normalize_text(result)


@lru_cache(maxsize=4)
def load_company_catalog(
    path_text: str = str(DEFAULT_STOCK_CODES_PATH),
) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    """Load canonical company names and safe aliases keyed by ticker."""
    canonical: dict[str, str] = {}
    aliases: dict[str, tuple[str, ...]] = {}
    path = Path(path_text)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            ticker = normalize_text(row.get("Mã CK") or row.get("ticker")).upper()
            company = normalize_text(row.get("Tên công ty") or row.get("company"))
            if not ticker or not company:
                continue
            if ticker in canonical and canonical[ticker] != company:
                raise QueryRoutingError(f"Tên công ty bị xung đột cho mã {ticker}")
            canonical[ticker] = company
            variants = {company, _strip_company_prefix(company)}
            aliases[ticker] = tuple(
                sorted(
                    (value for value in variants if len(fold_text(value)) >= 5),
                    key=len,
                    reverse=True,
                )
            )
    if not canonical:
        raise QueryRoutingError(f"Không đọc được danh mục mã cổ phiếu: {path}")
    return canonical, aliases


def resolve_tickers(
    question: str, stock_codes_path: Path = DEFAULT_STOCK_CODES_PATH
) -> tuple[list[str], dict[str, list[str]], dict[str, str]]:
    canonical, alias_catalog = load_company_catalog(
        str(Path(stock_codes_path).resolve())
    )
    known = set(canonical)
    direct = {
        token.upper()
        for token in re.findall(r"(?<![A-Za-z0-9])[A-Za-z]{3}(?![A-Za-z0-9])", question)
        if token.upper() in known
    }
    folded_question = f" {fold_text(question)} "
    matched_aliases: dict[str, list[str]] = {}
    for ticker, alias_names in alias_catalog.items():
        for name in alias_names:
            if f" {fold_text(name)} " in folded_question:
                direct.add(ticker)
                matched_aliases.setdefault(ticker, []).append(name)
                break

    strongly_explicit = {
        token.upper()
        for token in re.findall(
            r"(?:\(\s*|\bmã\s+)([A-Za-z]{3})(?:\s*\)|\b)",
            question,
            flags=re.I,
        )
        if token.upper() in known
    }
    for company_ticker, matched_names in matched_aliases.items():
        alias_tokens = {
            token.upper()
            for name in matched_names
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
    return sorted(direct), matched_aliases, canonical


def parse_years(question: str) -> list[int]:
    years: set[int] = set()
    for match in YEAR_RANGE_PATTERN.finditer(question):
        start, end = int(match.group(1)), int(match.group(2))
        low, high = sorted((start, end))
        if high - low <= 20:
            years.update(range(max(low, MIN_YEAR), min(high, MAX_YEAR) + 1))
    years.update(
        year
        for year in map(int, YEAR_PATTERN.findall(question))
        if MIN_YEAR <= year <= MAX_YEAR
    )
    return sorted(years)


def parse_report_types(question: str) -> list[str]:
    folded = fold_text(question)
    selected: list[str] = []
    if "cong ty me" in folded or re.search(r"\brieng\b", folded):
        selected.append("separate")
    if "hop nhat" in folded:
        selected.append("consolidated")
    if "tong hop" in folded:
        selected.append("aggregated")
    return selected


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


def validate_llm_filters(value: Mapping[str, Any]) -> dict[str, list[str | int]]:
    """Strictly validate the LLM JSON before deterministic reconciliation."""
    unknown = set(value) - set(FILTER_FIELDS)
    if unknown:
        raise QueryRoutingError(
            "Trường filter không được hỗ trợ: " + ", ".join(sorted(unknown))
        )

    validated: dict[str, list[str | int]] = {}
    for field in ("ticker", "company_name"):
        raw = value.get(field, [])
        if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
            raise QueryRoutingError(f"{field} phải là một mảng chuỗi")
        cleaned = list(
            dict.fromkeys(normalize_text(item) for item in raw if normalize_text(item))
        )
        if cleaned:
            if field == "ticker":
                cleaned = [item.upper() for item in cleaned]
            validated[field] = _as_filter_values(cleaned)

    raw_years = value.get("year", [])
    if not isinstance(raw_years, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in raw_years
    ):
        raise QueryRoutingError("year phải là một mảng số nguyên")
    if any(not MIN_YEAR <= item <= MAX_YEAR for item in raw_years):
        raise QueryRoutingError(f"year phải nằm trong {MIN_YEAR}–{MAX_YEAR}")
    if raw_years:
        validated["year"] = _as_filter_values(dict.fromkeys(raw_years))

    for field, allowed in (("report_type", REPORT_TYPES), ("table_type", TABLE_TYPES)):
        raw = value.get(field, [])
        if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
            raise QueryRoutingError(f"{field} phải là một mảng chuỗi")
        cleaned = list(dict.fromkeys(normalize_text(item).lower() for item in raw))
        invalid = [item for item in cleaned if item not in allowed]
        if invalid:
            raise QueryRoutingError(f"{field} không hợp lệ: {invalid}")
        if cleaned:
            validated[field] = _as_filter_values(cleaned)
    return validated


def build_semantic_query(
    question: str,
    tickers: Sequence[str],
    company_aliases: Mapping[str, Sequence[str]],
) -> str:
    semantic = normalize_text(question)
    for ticker in tickers:
        semantic = re.sub(
            rf"(?<![A-Za-z0-9]){re.escape(ticker)}(?![A-Za-z0-9])",
            " ",
            semantic,
            flags=re.I,
        )
        for alias in company_aliases.get(ticker, []):
            semantic = re.sub(re.escape(alias), " ", semantic, flags=re.I)
    semantic = YEAR_RANGE_PATTERN.sub(" ", semantic)
    semantic = re.sub(r"\bnăm\s*(?=20\d{2}\b)", " ", semantic, flags=re.I)
    semantic = YEAR_PATTERN.sub(" ", semantic)
    for phrase in (
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
        "tổng hợp",
        "consolidated",
        "separate",
        "aggregated",
    ):
        semantic = re.sub(re.escape(phrase), " ", semantic, flags=re.I)
    if company_aliases:
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
        raise QueryRoutingError("Câu hỏi không còn nội dung tài chính để tìm kiếm")
    return semantic


def reconcile_query_filters(
    question: str,
    llm_value: Mapping[str, Any],
    stock_codes_path: Path = DEFAULT_STOCK_CODES_PATH,
) -> tuple[dict[str, list[str | int]], str]:
    """Combine strict LLM semantics with deterministic identity routing."""
    parsed = validate_llm_filters(llm_value)
    tickers, matched_aliases, canonical = resolve_tickers(question, stock_codes_path)
    years = parse_years(question)
    if not tickers:
        raise QueryRoutingError(
            "Không resolve được ticker; hệ thống không search global"
        )
    if not years:
        raise QueryRoutingError(
            f"Không resolve được năm trong phạm vi {MIN_YEAR}–{MAX_YEAR}; hệ thống không search global"
        )

    llm_tickers = set(parsed.get("ticker", []))
    if llm_tickers and llm_tickers != set(tickers):
        raise QueryRoutingError(
            f"Ticker do LLM parse {sorted(llm_tickers)} không khớp câu hỏi {tickers}"
        )
    llm_years = set(parsed.get("year", []))
    if llm_years and llm_years != set(years):
        raise QueryRoutingError(
            f"Năm do LLM parse {sorted(llm_years)} không khớp câu hỏi {years}"
        )

    explicit_reports = parse_report_types(question)
    # Identity filters must be conservative. Explicit Vietnamese phrases win; when
    # absent, search both normal statement variants instead of trusting an LLM guess.
    report_types = explicit_reports or ["consolidated", "separate"]

    explicit_table = parse_table_type(question)
    # A metric can live in a note even when its wording resembles a main statement.
    # Only apply table_type when the question names the statement explicitly.
    table_types = [explicit_table] if explicit_table else []

    filters: dict[str, list[str | int]] = {
        "ticker": _as_filter_values(tickers),
        "company_name": _as_filter_values(canonical[ticker] for ticker in tickers),
        "year": _as_filter_values(years),
        "report_type": _as_filter_values(report_types),
    }
    if table_types:
        filters["table_type"] = _as_filter_values(table_types)
    return filters, build_semantic_query(question, tickers, matched_aliases)
