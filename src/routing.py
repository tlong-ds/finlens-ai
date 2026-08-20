"""Strict Vietnamese query routing aligned with the Qdrant payload schema."""

from __future__ import annotations

import csv
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.contracts import FILTER_FIELDS, MAX_YEAR, MIN_YEAR, REPORT_TYPES, TABLE_TYPES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STOCK_CODES_PATH = PROJECT_ROOT / "ViFinQA" / "code_stock.csv"
YEAR_PATTERN = re.compile(r"(?<!\d)(20\d{2})(?!\d)")
YEAR_RANGE_PATTERN = re.compile(
    r"(?<!\d)(20\d{2})(?!\d)"
    r"\s*(?:(?:-|–|—)|(?:đến|den|tới|toi))\s*"
    r"(?:(?:đầu|cuối)\s+)?(?:năm\s+)?"
    r"(20\d{2})(?!\d)",
    re.IGNORECASE,
)
YEAR_RELATIVE_PATTERN = re.compile(
    r"\b(?:trước|sau|tiếp theo|liền trước|liền sau|so với|từ|đến)\b",
    re.IGNORECASE,
)


class QueryRoutingError(ValueError):
    """Raised when a question cannot be mapped to safe metadata buckets."""


@dataclass(frozen=True)
class TickerCandidate:
    ticker: str
    entity_text: str
    match_type: str
    score: int
    matched_variant: str | None = None


@dataclass(frozen=True)
class TickerResolution:
    status: str
    tickers: tuple[str, ...]
    candidates: tuple[TickerCandidate, ...]
    matched_aliases: dict[str, tuple[str, ...]]
    reason: str = ""


@dataclass(frozen=True)
class YearCandidate:
    expression: str
    years: tuple[int, ...]
    match_type: str


@dataclass(frozen=True)
class YearResolution:
    status: str
    years: tuple[int, ...]
    candidates: tuple[YearCandidate, ...]
    reason: str = ""


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


def contains_folded_phrase(text: str, phrase: str) -> bool:
    """Match a normalized phrase across punctuation but not inside a token."""
    folded_text = fold_text(text)
    folded_phrase = fold_text(phrase)
    if not folded_phrase:
        return False
    return re.search(
        rf"(?<![a-z0-9]){re.escape(folded_phrase)}(?![a-z0-9])",
        folded_text,
    ) is not None


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


@lru_cache(maxsize=4)
def build_ticker_collision_index(
    path_text: str = str(DEFAULT_STOCK_CODES_PATH),
) -> dict[str, tuple[str, ...]]:
    """Map a bare token to all catalog companies whose names contain it."""
    canonical, _ = load_company_catalog(path_text)
    collisions: dict[str, set[str]] = {}
    for ticker, company in canonical.items():
        tokens = {
            token.upper()
            for token in re.findall(r"(?<![A-Za-z0-9])[A-Za-z0-9]+(?![A-Za-z0-9])", company)
            if len(token) >= 2
        }
        for token in tokens:
            collisions.setdefault(token, set()).add(ticker)
    return {token: tuple(sorted(tickers)) for token, tickers in collisions.items()}


def _matched_company_variants(
    question: str,
    alias_catalog: Mapping[str, Sequence[str]],
) -> list[tuple[str, str]]:
    matches: list[tuple[str, str]] = []
    for ticker, aliases in alias_catalog.items():
        for alias in aliases:
            if contains_folded_phrase(question, alias):
                matches.append((ticker, alias))
                break

    # A shorter catalog name nested in a longer name is not an independent
    # company match, e.g. HAG inside HNG's full legal name.
    return [
        (ticker, alias)
        for ticker, alias in matches
        if not any(
            other_ticker != ticker
            and len(fold_text(other_alias)) > len(fold_text(alias))
            and f" {fold_text(alias)} " in f" {fold_text(other_alias)} "
            for other_ticker, other_alias in matches
        )
    ]


def resolve_tickers_conservatively(
    question: str,
    stock_codes_path: Path = DEFAULT_STOCK_CODES_PATH,
) -> tuple[TickerResolution, dict[str, str]]:
    """Resolve only high-confidence identities; defer ambiguity to the LLM."""
    path_text = str(Path(stock_codes_path).resolve())
    canonical, alias_catalog = load_company_catalog(path_text)
    known = set(canonical)
    collision_index = build_ticker_collision_index(path_text)

    explicit = {
        token.upper()
        for token in re.findall(
            r"(?:\(\s*|\bmã(?:\s+cổ\s+phiếu)?\s+)([A-Za-z0-9]+)(?:\s*\)|\b)",
            question,
            flags=re.I,
        )
        if token.upper() in known
    }
    if explicit:
        candidates = tuple(
            TickerCandidate(ticker, ticker, "explicit_ticker", 100)
            for ticker in sorted(explicit)
        )
        return (
            TickerResolution("resolved", tuple(sorted(explicit)), candidates, {}),
            canonical,
        )

    variant_matches = _matched_company_variants(question, alias_catalog)
    matched_by_ticker: dict[str, tuple[str, ...]] = {}
    alias_tickers: set[str] = set()
    candidates: list[TickerCandidate] = []
    for ticker, alias in variant_matches:
        alias_tickers.add(ticker)
        matched_by_ticker.setdefault(ticker, tuple())
        matched_by_ticker[ticker] = (*matched_by_ticker[ticker], alias)
        candidates.append(
            TickerCandidate(
                ticker,
                alias,
                "company_name",
                95 if alias == canonical[ticker] else 90,
                alias,
            )
        )

    # A one-token shortened name such as "Masan" is not safe when that token
    # occurs in several catalog company names. Defer it to the LLM with all
    # catalog owners as a bounded shortlist.
    for ticker, alias in variant_matches:
        alias_tokens = re.findall(
            r"(?<![A-Za-z0-9])[A-Za-z0-9]+(?![A-Za-z0-9])", alias
        )
        if len(alias_tokens) != 1:
            continue
        token = alias_tokens[0].upper()
        owners = collision_index.get(token, (ticker,))
        if len(owners) <= 1:
            continue
        known_candidate_tickers = {candidate.ticker for candidate in candidates}
        candidates.extend(
            TickerCandidate(owner, token, "ambiguous_company_variant", 40)
            for owner in owners
            if owner not in known_candidate_tickers
        )
        return (
            TickerResolution(
                "needs_llm",
                (),
                tuple(candidates),
                matched_by_ticker,
                f"Tên rút gọn {alias!r} có thể thuộc: {', '.join(owners)}",
            ),
            canonical,
        )

    direct_tokens = {
        token.upper()
        for token in re.findall(
            r"(?<![A-Za-z0-9])[A-Za-z0-9]+(?![A-Za-z0-9])", question
        )
        if token.upper() in known
    }
    alias_tokens = {
        part.upper()
        for aliases in matched_by_ticker.values()
        for alias in aliases
        for part in re.findall(
            r"(?<![A-Za-z0-9])[A-Za-z0-9]+(?![A-Za-z0-9])", alias
        )
    }
    direct_tokens -= alias_tokens

    for ticker in sorted(direct_tokens):
        owners = collision_index.get(ticker, (ticker,))
        candidates.append(TickerCandidate(ticker, ticker, "bare_ticker", 70))
        if len(owners) > 1:
            known_candidate_tickers = {candidate.ticker for candidate in candidates}
            candidates.extend(
                TickerCandidate(owner, ticker, "collision_candidate", 40)
                for owner in owners
                if owner not in known_candidate_tickers
            )
            return (
                TickerResolution(
                    "needs_llm",
                    (),
                    tuple(candidates),
                    matched_by_ticker,
                    f"Bare ticker {ticker} có thể thuộc: {', '.join(owners)}",
                ),
                canonical,
            )

    resolved_tickers = set(alias_tickers) | direct_tokens
    if resolved_tickers:
        return (
            TickerResolution(
                "resolved",
                tuple(sorted(resolved_tickers)),
                tuple(candidates),
                matched_by_ticker,
            ),
            canonical,
        )

    return (
        TickerResolution(
            "needs_llm", (), tuple(), {}, "Không có match deterministic đủ chắc chắn"
        ),
        canonical,
    )


def resolve_tickers(
    question: str, stock_codes_path: Path = DEFAULT_STOCK_CODES_PATH
) -> tuple[list[str], dict[str, list[str]], dict[str, str]]:
    resolution, canonical = resolve_tickers_conservatively(
        question, stock_codes_path
    )
    if resolution.status != "resolved":
        return [], {}, canonical
    return (
        list(resolution.tickers),
        {ticker: list(aliases) for ticker, aliases in resolution.matched_aliases.items()},
        canonical,
    )


def extract_year_candidates(question: str) -> tuple[YearCandidate, ...]:
    """Extract year evidence without deciding which years the query needs."""
    candidates: list[YearCandidate] = []
    covered_spans: list[tuple[int, int]] = []
    for match in YEAR_RANGE_PATTERN.finditer(question):
        start, end = int(match.group(1)), int(match.group(2))
        low, high = sorted((start, end))
        if high - low > 20:
            years = tuple(
                year
                for year in (low, high)
                if MIN_YEAR <= year <= MAX_YEAR
            )
        else:
            years = tuple(
                range(max(low, MIN_YEAR), min(high, MAX_YEAR) + 1)
            )
        candidates.append(YearCandidate(match.group(0), years, "range"))
        covered_spans.append(match.span())

    for match in YEAR_PATTERN.finditer(question):
        if any(start <= match.start() < end for start, end in covered_spans):
            continue
        year = int(match.group(1))
        if MIN_YEAR <= year <= MAX_YEAR:
            candidates.append(YearCandidate(match.group(0), (year,), "explicit"))

    return tuple(candidates)


def _candidate_years(candidates: Sequence[YearCandidate]) -> tuple[int, ...]:
    return tuple(
        sorted({year for candidate in candidates for year in candidate.years})
    )


def resolve_years_conservatively(question: str) -> YearResolution:
    """Resolve only unambiguous absolute years; defer temporal semantics to LLM."""
    candidates = list(extract_year_candidates(question))
    if not candidates:
        return YearResolution(
            "needs_llm", (), tuple(candidates), "Không có năm tuyệt đối hợp lệ trong câu hỏi"
        )

    if any(candidate.match_type == "range" for candidate in candidates):
        return YearResolution(
            "needs_llm",
            (),
            tuple(candidates),
            "Khoảng năm cần phân tích ngữ nghĩa bởi year resolver riêng",
        )

    if YEAR_RELATIVE_PATTERN.search(question):
        anchored_years = _candidate_years(candidates)
        relative_years = tuple(
            sorted(
                {
                    year + delta
                    for year in anchored_years
                    for delta in (-1, 0, 1)
                    if MIN_YEAR <= year + delta <= MAX_YEAR
                }
            )
        )
        if relative_years:
            candidates.append(
                YearCandidate(
                    "relative-year neighborhood",
                    relative_years,
                    "relative_candidate",
                )
            )
        return YearResolution(
            "needs_llm",
            (),
            tuple(candidates),
            "Câu hỏi có quan hệ thời gian cần phân tích ngữ nghĩa",
        )

    years = _candidate_years(candidates)
    if years:
        return YearResolution("resolved", years, tuple(candidates))
    return YearResolution(
        "needs_llm", (), candidates, "Không có năm nằm trong phạm vi dữ liệu"
    )


def parse_years(question: str) -> list[int]:
    """Compatibility wrapper returning only high-confidence deterministic years."""
    resolution = resolve_years_conservatively(question)
    return list(resolution.years) if resolution.status == "resolved" else []


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
    for field in ("ticker",):
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


def validate_llm_ticker_resolution(
    value: Mapping[str, Any],
    question: str,
    canonical: Mapping[str, str],
    allowed_tickers: Sequence[str] | None = None,
) -> list[str]:
    """Validate a catalog-constrained LLM ticker resolution response."""
    if set(value) != {"matches", "unresolved_entities"}:
        raise QueryRoutingError(
            "Ticker resolver phải có đúng matches và unresolved_entities"
        )

    matches = value["matches"]
    unresolved = value["unresolved_entities"]
    if not isinstance(matches, list) or not isinstance(unresolved, list):
        raise QueryRoutingError(
            "matches và unresolved_entities phải là các mảng"
        )
    if not all(isinstance(item, str) and normalize_text(item) for item in unresolved):
        raise QueryRoutingError("unresolved_entities phải là mảng chuỗi không rỗng")

    resolved: list[str] = []
    seen_entities: set[str] = set()
    allowed = set(allowed_tickers or canonical)
    for item in matches:
        if not isinstance(item, Mapping):
            raise QueryRoutingError("Mỗi ticker match phải là một object")
        if set(item) != {"entity_text", "ticker"}:
            raise QueryRoutingError("Ticker match phải có entity_text và ticker")
        if not isinstance(item["entity_text"], str) or not isinstance(
            item["ticker"], str
        ):
            raise QueryRoutingError("entity_text và ticker phải là chuỗi")
        entity_text = normalize_text(item["entity_text"])
        ticker = normalize_text(item["ticker"]).upper()
        folded_entity = fold_text(entity_text)
        if not entity_text or not folded_entity:
            raise QueryRoutingError("entity_text không được rỗng")
        if not contains_folded_phrase(question, entity_text):
            raise QueryRoutingError(
                f"entity_text không xuất hiện trong câu hỏi: {entity_text!r}"
            )
        if ticker not in canonical:
            raise QueryRoutingError(f"Ticker không có trong catalog: {ticker}")
        if ticker not in allowed:
            raise QueryRoutingError(f"Ticker không nằm trong candidate shortlist: {ticker}")
        if folded_entity in seen_entities or ticker in resolved:
            raise QueryRoutingError("Ticker resolver trả match trùng")
        seen_entities.add(folded_entity)
        resolved.append(ticker)

    return sorted(resolved)


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
    ticker_overrides: Sequence[str] | None = None,
    year_overrides: Sequence[int] | None = None,
) -> tuple[dict[str, list[str | int]], str]:
    """Combine deterministic matches with one initial LLM metadata response."""
    parsed = validate_llm_filters(llm_value)
    if ticker_overrides is None:
        ticker_resolution, canonical = resolve_tickers_conservatively(
            question, stock_codes_path
        )
        if ticker_resolution.status == "resolved":
            tickers = list(ticker_resolution.tickers)
            matched_aliases = {
                ticker: list(aliases)
                for ticker, aliases in ticker_resolution.matched_aliases.items()
            }
        else:
            _, alias_catalog = load_company_catalog(
                str(Path(stock_codes_path).resolve())
            )
            tickers = list(parsed.get("ticker", []))
            unknown = [ticker for ticker in tickers if ticker not in canonical]
            if unknown:
                raise QueryRoutingError(
                    "Ticker do LLM parse không có trong catalog: " + ", ".join(unknown)
                )
            matched_aliases = {
                ticker: [
                    alias
                    for alias in alias_catalog.get(ticker, ())
                    if contains_folded_phrase(question, alias)
                ]
                for ticker in tickers
            }
            matched_aliases = {
                ticker: aliases
                for ticker, aliases in matched_aliases.items()
                if aliases
            }
    else:
        canonical, alias_catalog = load_company_catalog(
            str(Path(stock_codes_path).resolve())
        )
        tickers = sorted(
            {
                normalize_text(ticker).upper()
                for ticker in ticker_overrides
                if normalize_text(ticker)
            }
        )
        unknown = [ticker for ticker in tickers if ticker not in canonical]
        if unknown:
            raise QueryRoutingError(
                "Ticker override không có trong catalog: " + ", ".join(unknown)
            )
        matched_aliases = {
            ticker: [
                alias
                for alias in alias_catalog.get(ticker, ())
                if contains_folded_phrase(question, alias)
            ]
            for ticker in tickers
        }
        matched_aliases = {
            ticker: aliases for ticker, aliases in matched_aliases.items() if aliases
        }
    if year_overrides is None:
        year_resolution = resolve_years_conservatively(question)
        years = (
            list(year_resolution.years)
            if year_resolution.status == "resolved"
            else list(parsed.get("year", []))
        )
    else:
        years = list(dict.fromkeys(year_overrides))
        if any(
            isinstance(year, bool)
            or not isinstance(year, int)
            or not MIN_YEAR <= year <= MAX_YEAR
            for year in years
        ):
            raise QueryRoutingError(
                f"Year override phải là các số nguyên trong phạm vi {MIN_YEAR}–{MAX_YEAR}"
            )
    if not tickers:
        raise QueryRoutingError(
            "Không resolve được ticker; hệ thống không search global"
        )
    if not years:
        raise QueryRoutingError(
            f"Không resolve được năm trong phạm vi {MIN_YEAR}–{MAX_YEAR}; hệ thống không search global"
        )

    # High-confidence deterministic matches win. For ambiguous ticker/year
    # expressions, the same initial metadata LLM response supplies the value;
    # no additional resolver call is made.

    explicit_reports = parse_report_types(question)
    # Identity filters must be conservative. Explicit Vietnamese phrases win; when
    # absent, search every dataset report variant instead of trusting an LLM guess.
    report_types = explicit_reports or [
        "consolidated",
        "separate",
        "aggregated",
        "other",
    ]

    explicit_table = parse_table_type(question)
    # A metric can live in a note even when its wording resembles a main statement.
    # Only apply table_type when the question names the statement explicitly.
    table_types = [explicit_table] if explicit_table else []

    filters: dict[str, list[str | int]] = {
        "ticker": _as_filter_values(tickers),
        "year": _as_filter_values(years),
        "report_type": _as_filter_values(report_types),
    }
    if table_types:
        filters["table_type"] = _as_filter_values(table_types)
    return filters, build_semantic_query(question, tickers, matched_aliases)
