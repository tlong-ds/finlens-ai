"""Strict Vietnamese query routing aligned with the Qdrant payload schema."""

from __future__ import annotations

import csv
import logging
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.config import Settings
from src.contracts import MAX_YEAR, MIN_YEAR, REPORT_TYPES
from src.generation.prompts import PARSE_SYSTEM_PROMPT, build_parse_prompt
from src.providers.llm import LLMResponseError, generate_structured

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STOCK_CODES_PATH = PROJECT_ROOT / "ViFinQA" / "code_stock.csv"
YEAR_PATTERN = re.compile(r"(?<!\d)(20\d{2})(?!\d)")
YEAR_RANGE_PATTERN = re.compile(
    r"(?<!\d)(20\d{2})(?!\d)"
    r"\s*(?:(?:-|–|—)|(?:đến|den|tới|toi))\s*"
    r"(?:(?:đầu|cuối)\s+)?(?:năm\s+)?"
    r"(20\d{2})(?!\d)",
    re.IGNORECASE,
)
TICKER_SHORTLIST_MAX = 15
TICKER_FUZZY_THRESHOLD = 0.80

_MATCH_TYPE_PRIORITY = {
    "explicit_ticker": 0,
    "exact_ticker": 1,
    "exact_alias": 2,
    "normalized_alias": 3,
    "collision": 4,
    "fuzzy_company_name": 5,
}
_COLLISION_STOP_WORDS = frozenset(
    {
        "bao",
        "co",
        "cong",
        "ctcp",
        "dau",
        "doan",
        "hang",
        "ngan",
        "phan",
        "phat",
        "tap",
        "thuong",
        "tong",
        "trien",
        "ty",
        "viet",
        "nam",
        "tmcp",
        "tnhh",
    }
)


class QueryRoutingError(ValueError):
    """Raised when a question cannot be mapped to safe metadata buckets."""


@dataclass(frozen=True)
class TickerCandidate:
    ticker: str
    company_name: str
    matched_text: str
    match_type: str
    score: float


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


def normalize_match_text(value: Any) -> str:
    """Fold accents and punctuation into a stable token sequence."""
    return " ".join(re.findall(r"[a-z0-9]+", fold_text(value)))


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
    """Map informative company-name tokens to all catalog owners."""
    canonical, _ = load_company_catalog(path_text)
    collisions: dict[str, set[str]] = {}
    for ticker, company in canonical.items():
        tokens = {
            token
            for token in normalize_match_text(company).split()
            if len(token) >= 3 and token not in _COLLISION_STOP_WORDS
        }
        for token in tokens:
            collisions.setdefault(token, set()).add(ticker)
    return {token: tuple(sorted(tickers)) for token, tickers in collisions.items()}


def _question_windows(question: str, max_size: int) -> dict[int, list[tuple[str, str]]]:
    tokens = list(re.finditer(r"[^\W_]+", question, flags=re.UNICODE))
    windows: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for size in range(1, min(len(tokens), max_size) + 1):
        for start in range(0, len(tokens) - size + 1):
            first = tokens[start]
            last = tokens[start + size - 1]
            original = question[first.start() : last.end()]
            windows[size].append((normalize_match_text(original), original))
    return dict(windows)


def _exact_alias_match(
    alias: str,
    windows: Mapping[int, Sequence[tuple[str, str]]],
) -> str:
    normalized_alias = normalize_match_text(alias)
    size = len(normalized_alias.split())
    return next(
        (
            original
            for normalized, original in windows.get(size, ())
            if normalized == normalized_alias
        ),
        "",
    )


def _best_alias_match(
    alias: str,
    windows: Mapping[int, Sequence[tuple[str, str]]],
) -> tuple[float, str]:
    normalized_alias = normalize_match_text(alias)
    alias_tokens = normalized_alias.split()
    alias_set = set(alias_tokens)
    best_score = 0.0
    best_text = ""
    for size in range(max(1, len(alias_tokens) - 2), len(alias_tokens) + 3):
        for normalized_window, original_window in windows.get(size, ()):
            window_set = set(normalized_window.split())
            denominator = len(alias_set) + len(window_set)
            token_f1 = (
                2 * len(alias_set & window_set) / denominator if denominator else 0.0
            )
            sequence_score = SequenceMatcher(
                None, normalized_alias, normalized_window
            ).ratio()
            score = 0.7 * sequence_score + 0.3 * token_f1
            if score > best_score or (
                score == best_score and len(original_window) < len(best_text)
            ):
                best_score = score
                best_text = original_window
    return best_score, best_text


def _candidate_sort_key(candidate: TickerCandidate) -> tuple[int, float, int, str]:
    return (
        _MATCH_TYPE_PRIORITY[candidate.match_type],
        -candidate.score,
        -len(normalize_match_text(candidate.matched_text)),
        candidate.ticker,
    )


def _prefer_candidate(
    current: TickerCandidate | None, candidate: TickerCandidate
) -> TickerCandidate:
    if current is None or _candidate_sort_key(candidate) < _candidate_sort_key(current):
        return candidate
    return current


def _diverse_fuzzy_candidates(
    candidates: Sequence[TickerCandidate], limit: int
) -> list[TickerCandidate]:
    groups: dict[str, list[TickerCandidate]] = defaultdict(list)
    for candidate in sorted(candidates, key=_candidate_sort_key):
        groups[normalize_match_text(candidate.matched_text)].append(candidate)
    ordered_groups = sorted(
        groups.values(), key=lambda group: _candidate_sort_key(group[0])
    )
    selected: list[TickerCandidate] = []
    index = 0
    while len(selected) < limit:
        added = False
        for group in ordered_groups:
            if index < len(group):
                selected.append(group[index])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        index += 1
    return selected


def build_ticker_shortlist(
    question: str,
    stock_codes_path: Path = DEFAULT_STOCK_CODES_PATH,
    *,
    max_candidates: int = TICKER_SHORTLIST_MAX,
) -> tuple[TickerCandidate, ...]:
    """Build a deterministic, catalog-bound ticker shortlist for one LLM call."""
    if max_candidates < 1:
        raise ValueError("max_candidates must be at least 1")
    path_text = str(Path(stock_codes_path).resolve())
    canonical, alias_catalog = load_company_catalog(path_text)
    known = set(canonical)
    candidates_by_ticker: dict[str, TickerCandidate] = {}

    explicit = {
        token.upper()
        for token in re.findall(
            r"(?:\(\s*|\bmã(?:\s+cổ\s+phiếu)?\s+)([A-Za-z0-9]+)(?:\s*\)|\b)",
            question,
            flags=re.I,
        )
        if token.upper() in known
    }
    for ticker in explicit:
        candidates_by_ticker[ticker] = TickerCandidate(
            ticker, canonical[ticker], ticker, "explicit_ticker", 1.0
        )

    for match in re.finditer(
        r"(?<![A-Za-z0-9])([A-Za-z0-9]{2,8})(?![A-Za-z0-9])", question
    ):
        raw_token = match.group(1)
        ticker = raw_token.upper()
        if ticker not in known or not raw_token.isupper():
            continue
        match_type = "explicit_ticker" if ticker in explicit else "exact_ticker"
        candidate = TickerCandidate(
            ticker, canonical[ticker], raw_token, match_type, 1.0
        )
        candidates_by_ticker[ticker] = _prefer_candidate(
            candidates_by_ticker.get(ticker), candidate
        )

    max_alias_tokens = max(
        len(normalize_match_text(alias).split())
        for aliases in alias_catalog.values()
        for alias in aliases
    )
    windows = _question_windows(question, max_alias_tokens + 2)
    for ticker, aliases in alias_catalog.items():
        for alias in aliases:
            matched_text = _exact_alias_match(alias, windows)
            if matched_text:
                match_type = (
                    "exact_alias" if alias == canonical[ticker] else "normalized_alias"
                )
                score = 1.0
            else:
                if alias == canonical[ticker] and len(aliases) > 1:
                    continue
                score, matched_text = _best_alias_match(alias, windows)
                if not matched_text or score < TICKER_FUZZY_THRESHOLD:
                    continue
                match_type = "fuzzy_company_name"
            candidate = TickerCandidate(
                ticker, canonical[ticker], matched_text, match_type, score
            )
            candidates_by_ticker[ticker] = _prefer_candidate(
                candidates_by_ticker.get(ticker), candidate
            )

    collision_index = build_ticker_collision_index(path_text)
    seeds = list(candidates_by_ticker.values())
    for seed in seeds:
        seed_tokens = normalize_match_text(seed.matched_text).split()
        if seed.match_type in {"explicit_ticker", "exact_ticker"}:
            collision_tokens = [seed.ticker.lower()]
        elif len(seed_tokens) <= 2:
            collision_tokens = [
                token for token in seed_tokens if token not in _COLLISION_STOP_WORDS
            ]
        else:
            collision_tokens = []
        owners = {
            owner
            for token in collision_tokens
            for owner in collision_index.get(token, ())
        }
        for owner in owners:
            if owner == seed.ticker:
                continue
            collision = TickerCandidate(
                owner,
                canonical[owner],
                seed.matched_text,
                "collision",
                seed.score,
            )
            candidates_by_ticker[owner] = _prefer_candidate(
                candidates_by_ticker.get(owner), collision
            )

    non_fuzzy = sorted(
        (
            candidate
            for candidate in candidates_by_ticker.values()
            if candidate.match_type != "fuzzy_company_name"
        ),
        key=_candidate_sort_key,
    )
    exact_count = sum(
        candidate.match_type not in {"collision", "fuzzy_company_name"}
        for candidate in non_fuzzy
    )
    if exact_count > max_candidates:
        raise QueryRoutingError(
            f"Câu hỏi có hơn {max_candidates} ticker match chính xác; không thể cắt an toàn"
        )
    selected = non_fuzzy[:max_candidates]
    fuzzy_slots = max_candidates - len(selected)
    if fuzzy_slots:
        fuzzy = [
            candidate
            for candidate in candidates_by_ticker.values()
            if candidate.match_type == "fuzzy_company_name"
        ]
        selected.extend(_diverse_fuzzy_candidates(fuzzy, fuzzy_slots))
    if not selected:
        raise QueryRoutingError(
            "Không tạo được ticker candidate; hệ thống không search global"
        )
    return tuple(selected)


def serialize_ticker_candidates(
    candidates: Sequence[TickerCandidate],
) -> list[dict[str, str]]:
    """Serialize only the evidence the parser LLM needs for disambiguation."""
    return [
        {
            "candidate_key": f"c{index:02d}",
            "ticker": candidate.ticker,
            "company_name": candidate.company_name,
            "matched_text": candidate.matched_text,
            "match_type": candidate.match_type,
        }
        for index, candidate in enumerate(candidates, start=1)
    ]


def validate_llm_filters(
    value: Mapping[str, Any],
    ticker_candidates: Sequence[TickerCandidate] = (),
) -> dict[str, list[str | int]]:
    """Validate the parse JSON and materialize candidate keys into tickers."""
    unknown = set(value) - {"ticker", "year", "report_type"}
    if unknown:
        raise QueryRoutingError(
            "Trường filter không được hỗ trợ: " + ", ".join(sorted(unknown))
        )

    validated: dict[str, list[str | int]] = {}
    raw_ticker_keys = value.get("ticker", [])
    if not isinstance(raw_ticker_keys, list) or not all(
        isinstance(item, str) and normalize_text(item) for item in raw_ticker_keys
    ):
        raise QueryRoutingError("ticker phải là một mảng candidate_key không rỗng")
    ticker_keys = [normalize_text(item).lower() for item in raw_ticker_keys]
    if len(ticker_keys) != len(set(ticker_keys)):
        raise QueryRoutingError("ticker chứa candidate_key trùng")
    candidate_by_key = {
        f"c{index:02d}": candidate
        for index, candidate in enumerate(ticker_candidates, start=1)
    }
    unknown_keys = [key for key in ticker_keys if key not in candidate_by_key]
    if unknown_keys:
        raise QueryRoutingError(
            "Ticker candidate_key không có trong shortlist: " + ", ".join(unknown_keys)
        )
    if not ticker_keys:
        raise QueryRoutingError("ticker phải là một mảng candidate_key không rỗng")
    selected_keys = set(ticker_keys)
    validated["ticker"] = [
        candidate.ticker
        for key, candidate in candidate_by_key.items()
        if key in selected_keys
    ]

    raw_years = value.get("year", [])
    if not isinstance(raw_years, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in raw_years
    ):
        raise QueryRoutingError("year phải là một mảng số nguyên")
    if any(not MIN_YEAR <= item <= MAX_YEAR for item in raw_years):
        raise QueryRoutingError(f"year phải nằm trong {MIN_YEAR}–{MAX_YEAR}")
    if not raw_years:
        raise QueryRoutingError("year phải là một mảng số nguyên không rỗng")
    validated["year"] = _as_filter_values(dict.fromkeys(raw_years))

    raw_report_types = value.get("report_type", [])
    if (
        not isinstance(raw_report_types, list)
        or len(raw_report_types) != 1
        or not isinstance(raw_report_types[0], str)
    ):
        raise QueryRoutingError("report_type phải chứa đúng một loại báo cáo")
    report_type = normalize_text(raw_report_types[0]).lower()
    if report_type not in REPORT_TYPES:
        raise QueryRoutingError(f"report_type không hợp lệ: {report_type!r}")
    validated["report_type"] = [report_type]

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
    ticker_candidates: Sequence[TickerCandidate] | None = None,
) -> tuple[dict[str, list[str | int]], str]:
    """Map the LLM candidate selection into the existing flat filter contract."""
    candidates = tuple(
        ticker_candidates
        if ticker_candidates is not None
        else build_ticker_shortlist(question, stock_codes_path)
    )
    parsed = validate_llm_filters(llm_value, candidates)
    tickers = [str(ticker) for ticker in parsed["ticker"]]
    years = [int(year) for year in parsed["year"]]
    report_types = [str(report_type) for report_type in parsed["report_type"]]
    selected = {candidate.ticker: candidate for candidate in candidates}
    matched_aliases = {
        ticker: [selected[ticker].matched_text]
        for ticker in tickers
        if ticker in selected and selected[ticker].matched_text != ticker
    }
    filters: dict[str, list[str | int]] = {
        "ticker": _as_filter_values(tickers),
        "year": _as_filter_values(years),
        "report_type": _as_filter_values(report_types),
    }
    return filters, build_semantic_query(question, tickers, matched_aliases)


_PARSE_RESPONSE_ATTEMPTS = 2
logger = logging.getLogger(__name__)


class ParserAttemptsExhausted(QueryRoutingError):
    """A bounded parse failure carrying diagnostics for offline evaluation."""

    def __init__(self, message: str, diagnostics: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


def _concise_error(error: BaseException) -> str:
    message = " ".join(str(error).split())
    return message[:500] or error.__class__.__name__


def parse_query_with_diagnostics(
    question: str,
    *,
    settings: Settings,
    question_id: int | str = "unknown",
) -> dict[str, Any]:
    """Parse one question and retain bounded diagnostics."""
    ticker_candidates = build_ticker_shortlist(
        question, settings.project_root / settings.stock_codes_path
    )
    candidate_context = serialize_ticker_candidates(ticker_candidates)
    feedback = ""
    previous_response: Mapping[str, Any] | None = None
    last_error = ""
    attempts: list[dict[str, Any]] = []
    for parse_attempt in range(1, _PARSE_RESPONSE_ATTEMPTS + 1):
        current_response: Mapping[str, Any] | None = None
        attempt_diagnostic: dict[str, Any] = {
            "attempt": parse_attempt,
            "raw_filters": None,
            "validation_error": None,
        }
        try:
            current_response = generate_structured(
                build_parse_prompt(
                    question,
                    candidate_context,
                    feedback=feedback,
                    previous_response=previous_response,
                ),
                settings=settings,
                system_prompt=PARSE_SYSTEM_PROMPT,
                native=False,
            )
            attempt_diagnostic["raw_filters"] = dict(current_response)
            filters, semantic_query = reconcile_query_filters(
                question,
                current_response,
                stock_codes_path=settings.project_root / settings.stock_codes_path,
                ticker_candidates=ticker_candidates,
            )
            attempts.append(attempt_diagnostic)
            break
        except (LLMResponseError, QueryRoutingError) as error:
            last_error = _concise_error(error)
            attempt_diagnostic["validation_error"] = last_error
            if current_response is not None:
                attempt_diagnostic["raw_filters"] = dict(current_response)
                previous_response = current_response
            attempts.append(attempt_diagnostic)
            feedback = last_error
    else:
        diagnostics = {
            "ticker_candidates": candidate_context,
            "attempts": attempts,
            "semantic_attempts": len(attempts),
        }
        raise ParserAttemptsExhausted(
            "Không parse được metadata filter hợp lệ sau "
            f"{_PARSE_RESPONSE_ATTEMPTS} lần: {last_error}",
            diagnostics,
        )
    logger.info("question_id=%s parsed_filters=%s", question_id, filters)
    return {
        "filters": filters,
        "semantic_query": semantic_query,
        "diagnostics": {
            "ticker_candidates": candidate_context,
            "attempts": attempts,
            "semantic_attempts": len(attempts),
        },
    }
