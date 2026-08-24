"""Phase 1: Parse OCR files, extract tables, generate metadata.

Consolidates the full data-processing pipeline (GĐ 0–6) from
implementation_plan.md into a single file.  Run with:

    python prepare.py

Uses ``ViFinQA/financial_statements`` as the default input directory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import re
import unicodedata
from bisect import bisect_right
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd


# ═════════════════════════════════════════════════════════════════════
# Utilities — Regex patterns
# ═════════════════════════════════════════════════════════════════════

PAGE_MARKER_RE = re.compile(r'={3,}\s*PAGE\s+(\d+)\s*={3,}')

TABLE_TAG_RE = re.compile(r'<table[^>]*>.*?</table>', re.DOTALL | re.IGNORECASE)
HTML_CELL_RE = re.compile(
    r'<t[dh]\b(?P<attrs>[^>]*)>(?P<content>.*?)</t[dh]>',
    re.DOTALL | re.IGNORECASE,
)
HTML_TAG_RE = re.compile(r'<[^>]+>')
HTML_ROWSPAN_RE = re.compile(r'\browspan\s*=\s*["\']?(\d+)', re.IGNORECASE)
# This intentionally requires a thousands/decimal separator: row codes such
# as ``09`` and years such as ``2025`` are not financial amount evidence.
FINANCIAL_AMOUNT_TOKEN_RE = re.compile(r'\(?-?\d{1,3}(?:[.,]\d{3})+\)?')
MERGED_LABEL_START_RE = re.compile(
    r'(?:^|\s)(?:tăng|giảm|thu|chi|lãi|lỗ|khấu\s+hao|hoàn\s+nhập|'
    r'tổng\s+lợi\s+nhuận|điều\s+chỉnh)\b',
    re.IGNORECASE,
)

# Mã mẫu biểu — handles OCR variations ("Mâu"→"Mẫu", "MẪU SỐ B", extra spaces, – vs -)
# Groups: (1) number e.g. "02", (2) optional letter e.g. "b", (3) entity suffix e.g. "TCTD"
# Also matches OCR-degraded forms like "M?U S? B 02/TCTD-HN" where diacritics are lost
MAU_BIEU_RE = re.compile(
    r'[Mm][ẫâấ\w]u\s*'       # "Mẫu" / "Mâu" / "M?u" (OCR)
    r'(?:[Ss][ốô\w]?\s*)?'    # optional "Số" / "S?"
    r'[Bb]\s*(\d{1,2})\s*'    # "B 02" → group(1) = "02"
    r'([a-zA-Z]?)\s*'         # optional letter like "b" → group(2)
    r'[-–/\\]\s*'             # separator
    r'(DN|TCTD|CTCK|BH|HN)'  # entity suffix → group(3)
    r'(?:\s*[-–/]\s*HN)?',   # optional "-HN" suffix (hợp nhất)
    re.IGNORECASE,
)

# Entity type detection patterns (for scanning file content)
ENTITY_DETECT_PATTERNS = {
    'TCTD': re.compile(
        r'[Bb]\s*\d{1,2}\s*[a-z]?\s*[-–/]\s*TCTD'
        r'|(?:chế\s*độ|hệ\s*thống)\s*kế\s*toán[^\n]{0,120}'
        r'tổ\s*chức\s*tín\s*dụng',
        re.IGNORECASE,
    ),
    'CTCK': re.compile(
        r'[Bb]\s*\d{1,2}\s*[a-z]?\s*[-–/]\s*CTCK'
        r'|[Mm][ẫâấ]u\s*[Bb]\s*\d{1,2}\s*[a-z]?\s*[-–/]\s*CTCK'
        r'|chế\s*độ\s*kế\s*toán[^\n]{0,120}công\s*ty\s*chứng\s*khoán'
        r'|thông\s*tư\s*(?:số\s*)?210\s*/\s*2014\s*/\s*TT\s*-?\s*BTC',
        re.IGNORECASE,
    ),
    'BH': re.compile(
        r'dự\s*phòng\s*nghiệp\s*vụ\s*bảo\s*hiểm'
        r'|bảo\s*hiểm\s*nhân\s*thọ'
        r'|phí\s*bảo\s*hiểm\s*gốc',
        re.IGNORECASE,
    ),
}

# Used only to make an otherwise equal fallback score deterministic.  BH is
# deliberately preferred because an insurance group can legitimately mention
# both banks and securities companies throughout its notes.
ENTITY_FALLBACK_PRIORITY = {'TCTD': 1, 'CTCK': 2, 'BH': 3}
ENTITY_FALLBACK_MIN_COUNTS = {'TCTD': 3, 'CTCK': 3, 'BH': 100}

REPORT_TYPE_VI = {
    'consolidated': 'hợp nhất',
    'separate': 'riêng',
    'aggregated': 'tổng hợp',
    'other': 'không xác định',
}

# Heading patterns for table type classification (with OCR-error tolerance)
HEADING_BALANCE_SHEET_RE = re.compile(
    r'b[ảa]ng\s*c[âấ]n\s*đ[ốổ]i\s*k[ếể]\s*to[áa]n'
    r'|b[áa]o\s*c[áa]o\s*t[ìi]nh\s*h[ìi]nh\s*t[àa]i\s*ch[íi]nh',
    re.IGNORECASE,
)
HEADING_INCOME_STMT_RE = re.compile(
    r'b[áa]o\s*c[áa]o\s*k[ếể]t\s*qu[ảa]\s*ho[ạa]t\s*đ[ộô]ng\s*kinh\s*doanh',
    re.IGNORECASE,
)
HEADING_CASH_FLOW_RE = re.compile(
    r'b[áa]o\s*c[áa]o\s*l[ưu][uù]?\s*chuy[ểê][nrn]\s*ti[ềe]n\s*t[ệe]',
    re.IGNORECASE,
)
HEADING_NOTES_RE = re.compile(
    r'thuy[ếê]t\s*minh\s*b[áa]o\s*c[áa]o\s*t[àa]i\s*ch[íi]nh'
    r'|b[ảa]n\s*thuy[ếê]t\s*minh',
    re.IGNORECASE,
)

# Structural fingerprint patterns (found inside table cell text)
STRUCTURAL_BALANCE_SHEET_RE = re.compile(
    r'T[ỔỒ]NG\s*(C[ỘỒ]NG\s*)?T[ÀA]I\s*S[ẢA]N'
    r'|T[ỔỒ]NG\s*(C[ỘỒ]NG\s*)?NGU[ỒỒ]N\s*V[ỐỒ]N',
    re.IGNORECASE,
)
STRUCTURAL_INCOME_STMT_RE = re.compile(
    r'L[ỢỢ]I\s*NHU[ẬẬ]N\s*SAU\s*THU[ẾỀ]'
    r'|L[ỢỢ]I\s*NHU[ẬẬ]N\s*THU[ẦẦ]N',
    re.IGNORECASE,
)
STRUCTURAL_CASH_FLOW_RE = re.compile(
    r'L[ƯƯ]U\s*CHUY[ỂỂ]N\s*TI[ỀỀ]N\s*THU[ẦẦ]N'
    r'|TI[ỀỀ]N\s*V[ÀÀ]\s*T[ƯƯ][ƠƠ]NG\s*\u0110[ƯƯ][ƠƠ]NG\s*TI[ỀỀ]N\s*CU[ỐỒ]I',
    re.IGNORECASE,
)

TOC_HEADING_RE = re.compile(
    r'\bmục\s*lục\b|\btable\s+of\s+contents\b',
    re.IGNORECASE,
)
TOC_LIST_HEADING_RE = re.compile(r'\bdanh\s*mục\b', re.IGNORECASE)
TOC_PAGE_RE = re.compile(r'^\s*\d+\s*(?:[-–]\s*\d+)?\s*$')


# ═════════════════════════════════════════════════════════════════════
# Utilities — Mã mẫu biểu → table_type mapping
# ═════════════════════════════════════════════════════════════════════

MAU_BIEU_TABLE_TYPE: dict[tuple[str, str], str] = {
    # DN / HN (Doanh nghiệp, hợp nhất)
    ('01', 'DN'): 'balance_sheet',
    ('01', 'HN'): 'balance_sheet',
    ('02', 'DN'): 'income_statement',
    ('02', 'HN'): 'income_statement',
    ('03', 'DN'): 'cash_flow',
    ('03', 'HN'): 'cash_flow',
    ('09', 'DN'): 'note_table',
    ('09', 'HN'): 'note_table',
    # TCTD (Ngân hàng)
    ('02', 'TCTD'): 'balance_sheet',
    ('03', 'TCTD'): 'income_statement',
    ('04', 'TCTD'): 'cash_flow',
    ('05', 'TCTD'): 'note_table',
    # CTCK (Chứng khoán)
    ('01', 'CTCK'): 'balance_sheet',
    ('02', 'CTCK'): 'income_statement',
    ('03', 'CTCK'): 'cash_flow',
    ('05', 'CTCK'): 'note_table',
}

# Vietnamese table type display names (for retrieval_context templates)
TABLE_TYPE_VI: dict[str, str] = {
    'balance_sheet': 'Bảng cân đối kế toán',
    'income_statement': 'Báo cáo kết quả hoạt động kinh doanh',
    'cash_flow': 'Báo cáo lưu chuyển tiền tệ',
    'note_table': 'Thuyết minh báo cáo tài chính',
}


# ═════════════════════════════════════════════════════════════════════
# Utilities — Label normalization
# ═════════════════════════════════════════════════════════════════════

# Remove formula in parentheses like (20=10-11), (100 = 110 + 120 + ...)
FORMULA_IN_PARENS_RE = re.compile(r'\(\s*\d+\s*=[\s\d+\-×*/]+\)')
# Remove leading numbering like "1.", "A.", "I.", "II.", "a)"
# A decimal/thousands-formatted amount can start with ``1.`` or ``2.``.
# Require whitespace after a numeric list marker so ``2.528.849`` is never
# shortened to ``528.849`` when a malformed table maps it into a label field.
LEADING_NUMBER_RE = re.compile(r'^\s*(?:[IVXivx]+\.|[A-Za-z]\.|[A-Za-z]\)|\d+\.\s+)')
# Collapse whitespace
MULTI_SPACE_RE = re.compile(r'\s+')


def normalize_item_label(raw: str) -> str:
    """Clean item_label_raw → item_label_norm."""
    s = raw.strip()
    s = FORMULA_IN_PARENS_RE.sub('', s)
    s = LEADING_NUMBER_RE.sub('', s)
    s = MULTI_SPACE_RE.sub(' ', s)
    return s.strip()


def _fold_text(value: Any) -> str:
    """Lower-case text and strip Vietnamese diacritics for rule matching."""
    text = str(value or '')
    text = ''.join(
        char for char in unicodedata.normalize('NFD', text)
        if unicodedata.category(char) != 'Mn'
    )
    return text.replace('đ', 'd').replace('Đ', 'D').lower()


_GENERIC_NOTE_LABELS = {
    'stt',
    'so thu tu',
    'ma so',
    'chi tieu',
    'noi dung',
    'dien giai',
    'danh muc',
    'ten',
    'tong cong',
    'cong',
}
_GENERIC_NOTE_MARKERS = (
    'bao cao tai chinh',
    'thuyet minh bao cao tai chinh',
    'cac thuyet minh nay la bo phan',
    'mau so b09',
    'mau so b 09',
    'cho nam tai chinh',
    'ket thuc ngay',
)
_ADDRESS_MARKERS = ('dia chi', 'lo ', 'cum cong nghiep', 'thi tran', 'huyen ', 'tinh ')


def _clean_note_title(value: Any) -> str:
    """Keep the semantic note title while dropping form/legal boilerplate."""
    title = MULTI_SPACE_RE.sub(' ', str(value or '')).strip(' -–—:;,')
    if not title:
        return ''
    title = re.split(
        r'\b(?:m[ẫâa]u\s+s[ốo]|ban\s+h[àa]nh|theo\s+th[oô]ng\s+t[ưừu])\b',
        title,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(' -–—:;,')
    folded = _fold_text(title).replace('_', ' ').strip()
    if folded in {
        'unknown',
        'note unknown',
        'bao cao tai chinh',
        'thuyet minh bao cao tai chinh',
    }:
        return ''
    return title[:240]


def _clean_note_retrieval_term(value: Any) -> str:
    """Return a useful table label, excluding page headers and OCR boilerplate."""
    term = MULTI_SPACE_RE.sub(' ', str(value or '')).strip(' -–—:;,|/\\')
    if len(term) < 3 or len(term) > 180:
        return ''
    folded = _fold_text(term)
    if folded in _GENERIC_NOTE_LABELS:
        return ''
    if any(marker in folded for marker in _GENERIC_NOTE_MARKERS):
        return ''
    if sum(marker in folded for marker in _ADDRESS_MARKERS) >= 2:
        return ''
    if not any(char.isalpha() for char in term):
        return ''
    if len(term.split()) > 28:
        return ''
    return term


def _note_retrieval_terms(table: dict, limit: int = 32) -> list[str]:
    """Derive metadata-only terms from a note title and raw row labels."""
    candidates: list[str] = []
    title = _clean_note_title(table.get('note_title'))
    if title:
        candidates.append(title)

    for row in table.get('rows') or []:
        if not row:
            continue
        # The first cell is often an STT/code.  Select the first useful text
        # cell so a table such as ["1", "Công ty mẹ"] remains retrievable.
        for cell in row:
            term = _clean_note_retrieval_term(cell)
            if term:
                candidates.append(term)
                break

    terms: list[str] = []
    seen: set[str] = set()
    for term in candidates:
        key = _fold_text(term)
        if not key or key in seen:
            continue
        seen.add(key)
        terms.append(term)
        if len(terms) >= limit:
            break
    return terms


# ═════════════════════════════════════════════════════════════════════
# Utilities — Vietnamese number parser
# ═════════════════════════════════════════════════════════════════════

_NUMBER_STRIP_RE = re.compile(r'[^\d.,\-()]')


def parse_vietnamese_number(s: str) -> Optional[float]:
    """
    Parse a Vietnamese-formatted number string to float.

    Handles:
    - Thousand separators with dots: "120.355.231" → 120355231.0
    - Negative in parentheses: "(1.234)" → -1234.0
    - Regular negative: "-1.234" → -1234.0
    - Empty / dash / non-numeric: → None
    """
    if not s or not isinstance(s, str):
        return None

    s = s.strip()
    if not s or s in ('-', '–', '—', '*', 'x', 'X', 'N/A', ''):
        return None
    # Dates appear frequently in multi-level headers.  Treating 31/12/2025
    # as 31122025 creates fabricated financial values downstream.
    if _is_date_cell(s):
        return None

    negative = False
    if s.startswith('(') and s.endswith(')'):
        negative = True
        s = s[1:-1].strip()
    elif s.startswith('-'):
        negative = True
        s = s[1:].strip()

    # Remove any non-numeric chars except .,
    s = _NUMBER_STRIP_RE.sub('', s)
    if not s:
        return None

    # Vietnamese convention: dots = thousands, comma = decimal
    if ',' in s:
        s = s.replace('.', '').replace(',', '.')
    else:
        s = s.replace('.', '')

    try:
        value = float(s)
        return -value if negative else value
    except ValueError:
        return None


# ═════════════════════════════════════════════════════════════════════
# Utilities — HTML table parser (stdlib, no external deps)
# ═════════════════════════════════════════════════════════════════════

class _TableHTMLParser(HTMLParser):
    """Parse a single <table> HTML string into a 2D list of strings."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] = []
        self._current_cell: list[str] = []
        self._colspans: list[int] = []
        self._rowspans: list[int] = []
        self._pending_rowspans: dict[int, tuple[int, str]] = {}
        self._in_td = False
        self._in_table = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        tag_lower = tag.lower()
        if tag_lower == 'table':
            self._in_table = True
        elif tag_lower == 'tr' and self._in_table:
            self._current_row = []
            self._colspans = []
            self._rowspans = []
        elif tag_lower in ('td', 'th') and self._in_table:
            self._in_td = True
            self._current_cell = []
            attrs_dict = dict(attrs)
            self._colspans.append(int(attrs_dict.get('colspan', '1') or '1'))
            self._rowspans.append(int(attrs_dict.get('rowspan', '1') or '1'))

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower in ('td', 'th') and self._in_td:
            self._in_td = False
            cell_text = ''.join(self._current_cell).strip()
            self._current_row.append(cell_text)
        elif tag_lower == 'tr' and self._in_table and (
            self._current_row or self._pending_rowspans
        ):
            expanded: list[str] = []
            source_cell = 0
            column = 0
            while source_cell < len(self._current_row) or column in self._pending_rowspans:
                pending = self._pending_rowspans.get(column)
                if pending:
                    remaining, value = pending
                    expanded.append(value)
                    if remaining <= 1:
                        del self._pending_rowspans[column]
                    else:
                        self._pending_rowspans[column] = (remaining - 1, value)
                    column += 1
                    continue

                cell = self._current_row[source_cell]
                colspan = self._colspans[source_cell] if source_cell < len(self._colspans) else 1
                rowspan = self._rowspans[source_cell] if source_cell < len(self._rowspans) else 1
                expanded.append(cell)
                for offset in range(1, colspan):
                    expanded.append('')
                if rowspan > 1:
                    self._pending_rowspans[column] = (rowspan - 1, cell)
                    for offset in range(1, colspan):
                        self._pending_rowspans[column + offset] = (rowspan - 1, '')
                source_cell += 1
                column += colspan
            self.rows.append(expanded)
        elif tag_lower == 'table':
            self._in_table = False

    def handle_data(self, data: str) -> None:
        if self._in_td:
            self._current_cell.append(data)


def parse_html_table(html: str) -> list[list[str]]:
    """
    Parse an HTML table string into a 2D list of cell values.

    Expands ``colspan`` and ``rowspan`` into a rectangular cell grid.  The
    top-left value of a rowspan is repeated in subsequent rows so a header
    remains usable without borrowing cells from another source table.
    """
    parser = _TableHTMLParser()
    try:
        parser.feed(html)
    except Exception:
        return []
    return parser.rows


# ═════════════════════════════════════════════════════════════════════
# Utilities — Text helpers
# ═════════════════════════════════════════════════════════════════════

def extract_tables_from_text(text: str) -> list[tuple[str, int, int]]:
    """
    Extract all <table>...</table> segments from text.

    Returns:
        List of (table_html, start_pos, end_pos).
    """
    return [(m.group(), m.start(), m.end()) for m in TABLE_TAG_RE.finditer(text)]


def get_preceding_text(full_text: str, table_start: int, max_chars: int = 300) -> str:
    """Get up to *max_chars* characters of text before *table_start*."""
    start = max(0, table_start - max_chars)
    return full_text[start:table_start].strip()


def detect_folder_type(doc_name: str) -> str:
    """Detect consolidated / separate / aggregated / other from doc folder name."""
    low = doc_name.lower()
    if 'consolidated' in low:
        return 'consolidated'
    if 'separate' in low:
        return 'separate'
    if 'aggregated' in low:
        return 'aggregated'
    return 'other'


# ═════════════════════════════════════════════════════════════════════
# Inventory & entity_type classification
# ═════════════════════════════════════════════════════════════════════

def _detect_entity_type_from_content(content: str) -> str:
    """
    Detect entity_type from file content using mã mẫu biểu as primary signal.

    Scans the FULL file content for mã mẫu biểu patterns (B01-DN, B02/TCTD, etc.)
    and returns the most frequent entity suffix found.
    """
    suffix_counts: Counter[str] = Counter()
    for m in MAU_BIEU_RE.finditer(content):
        suffix = m.group(3).upper()
        # Normalize HN → DN (Mẫu B01-HN is the hợp nhất variant of DN)
        if suffix == 'HN':
            suffix = 'DN'
        suffix_counts[suffix] += 1

    pattern_counts = Counter({
        entity_type: len(pattern.findall(content))
        for entity_type, pattern in ENTITY_DETECT_PATTERNS.items()
    })
    return _resolve_entity_type(suffix_counts, pattern_counts)


def _resolve_entity_type(
    suffix_counts: Counter[str],
    pattern_counts: Counter[str],
) -> str:
    """Resolve an entity type from exact form codes, then strong text clues."""
    non_dn = {key: value for key, value in suffix_counts.items()
              if key != 'DN' and value > 0}
    if non_dn:
        return max(
            non_dn,
            key=lambda key: (
                non_dn[key],
                pattern_counts.get(key, 0),
                ENTITY_FALLBACK_PRIORITY.get(key, 0),
            ),
        )

    # A detected DN/HN form is stronger evidence than incidental industry
    # terminology in the notes (for example a DN owning a bank subsidiary).
    if suffix_counts.get('DN', 0) > 0:
        return 'DN'

    evidence = {
        key: value for key, value in pattern_counts.items()
        if value >= ENTITY_FALLBACK_MIN_COUNTS.get(key, 1)
    }
    if evidence:
        return max(
            evidence,
            key=lambda key: (evidence[key], ENTITY_FALLBACK_PRIORITY.get(key, 0)),
        )
    return 'DN'


def scan_inventory(fs_dir: str) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Scan the financial statements directory tree and build inventory.

    Returns:
        (inventory_df, entity_type_map) where *entity_type_map* maps
        each ticker to its detected entity_type (DN/TCTD/CTCK/BH).
    """
    fs_path = Path(fs_dir)
    inventory_data: list[dict] = []

    # Accumulate ALL mã mẫu biểu suffix counts across ALL files per ticker
    ticker_suffix_counts: dict[str, Counter] = {}
    ticker_pattern_counts: dict[str, Counter] = {}

    for ticker_dir in sorted(fs_path.iterdir()):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name

        for year_dir in sorted(ticker_dir.iterdir()):
            if not year_dir.is_dir():
                continue
            try:
                year = int(year_dir.name)
            except ValueError:
                continue

            for doc_dir in sorted(year_dir.iterdir()):
                if not doc_dir.is_dir():
                    continue

                doc_id = doc_dir.name
                file_path = doc_dir / f"{doc_id}_extracted.txt"
                if not file_path.exists():
                    continue

                folder_type = detect_folder_type(doc_id)
                rel_path = str(file_path.relative_to(fs_path)).replace('\\', '/')

                try:
                    size = file_path.stat().st_size
                    content = file_path.read_text(encoding='utf-8')
                    n_pages = len(PAGE_MARKER_RE.findall(content))

                    # Accumulate mã mẫu biểu suffix counts for this ticker
                    if ticker not in ticker_suffix_counts:
                        ticker_suffix_counts[ticker] = Counter()
                        ticker_pattern_counts[ticker] = Counter()
                    for m in MAU_BIEU_RE.finditer(content):
                        suffix = m.group(3).upper()
                        if suffix == 'HN':
                            suffix = 'DN'
                        ticker_suffix_counts[ticker][suffix] += 1
                    for entity_type, pattern in ENTITY_DETECT_PATTERNS.items():
                        ticker_pattern_counts[ticker][entity_type] += len(
                            pattern.findall(content)
                        )

                    inventory_data.append({
                        'ticker': ticker,
                        'year': year,
                        'folder_type': folder_type,
                        'file_path': rel_path,
                        'doc_id': doc_id,
                        'size': size,
                        'n_pages': n_pages,
                    })
                except Exception as exc:
                    logging.warning("Error reading %s: %s", file_path, exc)

    # Resolve entity_type per ticker from accumulated suffix counts
    entity_type_map: dict[str, str] = {}
    for ticker, counts in ticker_suffix_counts.items():
        entity_type_map[ticker] = _resolve_entity_type(
            counts,
            ticker_pattern_counts.get(ticker, Counter()),
        )

    return pd.DataFrame(inventory_data), entity_type_map


# ═════════════════════════════════════════════════════════════════════
# Parse .txt → raw tables
# ═════════════════════════════════════════════════════════════════════

def parse_all_files(inventory: pd.DataFrame, entity_type_map: dict[str, str], fs_dir: str) -> list[dict]:
    """
    Parse tables from all files in the inventory.
    """
    fs_path = Path(fs_dir)
    results = []

    count = 0

    for _, row in inventory.iterrows():
        file_rel_path = row['file_path']
        file_path = fs_path / file_rel_path
        doc_id = row['doc_id']
        ticker = row['ticker']
        year = row['year']
        folder_type = row['folder_type']
        entity_type = entity_type_map.get(ticker, 'DN')
        consolidated = (folder_type == 'consolidated')

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logging.warning(f"Failed to read {file_path}: {e}")
            continue

        pages = []
        last_end = 0
        seen_page_marker = False
        current_page_start = 0

        # 1-based source line lookup for audit metadata.  Bisecting precomputed
        # line starts is substantially cheaper than recounting newlines for
        # every table in a report.
        line_starts = [0]
        line_starts.extend(match.end() for match in re.finditer(r'\n', content))

        for m in PAGE_MARKER_RE.finditer(content):
            page_text = content[last_end:m.start()]
            if seen_page_marker:
                pages.append((page_text, current_page_start))

            seen_page_marker = True
            last_end = m.end()
            current_page_start = last_end

        if seen_page_marker:
            pages.append((content[last_end:], current_page_start))

        table_counter = 1

        for page_number, (page_text, page_start) in enumerate(pages, start=1):
            tables = extract_tables_from_text(page_text)
            page_has_toc_heading = bool(TOC_HEADING_RE.search(page_text))

            for table_html, start_pos, end_pos in tables:
                table_id = f"{doc_id}_table_{table_counter}"
                table_counter += 1

                rows = parse_html_table(table_html)
                if not rows:
                    logging.warning(f"Empty rows parsed from table {table_id} in {file_rel_path}")
                    continue

                preceding_text = get_preceding_text(page_text, start_pos, 300)
                start_line = bisect_right(line_starts, page_start + start_pos)
                parse_proofs = []
                if re.search(r'\b(?:rowspan|colspan)\s*=', table_html, re.IGNORECASE):
                    parse_proofs.append({
                        'stage': 'parse_html',
                        'rule': 'expand_table_spans',
                        'action': 'expand_rowspan_colspan',
                        'table_id': table_id,
                        'doc_id': doc_id,
                        'ticker': ticker,
                        'year': year,
                        'source_line': start_line,
                        'source_table_ids': [table_id],
                        'reason': 'source HTML declares rowspan or colspan',
                        'benefit': 'preserve source cell alignment without merging tables',
                    })

                record = {
                    'doc_id': doc_id,
                    'table_id': table_id,
                    'start_line': start_line,
                    'preceding_text': preceding_text,
                    'table_html': table_html,
                    'rows': rows,
                    'ticker': ticker,
                    'year': year,
                    'entity_type': entity_type,
                    'consolidated': consolidated,
                    'doc_path': file_rel_path,
                    'folder_type': folder_type,
                    # Internal audit context only.  It is intentionally not
                    # included in the public metadata/Qdrant payload.
                    'page_number': page_number,
                    'page_has_toc_heading': page_has_toc_heading,
                    'page_table_count': len(tables),
                    'source_table_index': table_counter - 1,
                    'repair_proofs': parse_proofs,
                }
                results.append(record)

        count += 1
        if count % 100 == 0:
            print(f"Processed {count} files...")

    return results


# ═════════════════════════════════════════════════════════════════════
# Classify table_type
# ═════════════════════════════════════════════════════════════════════

NOTE_HEADING_RE = re.compile(
    r'(?m)^[ \t]*'
    r'(?P<number>(?:[IVXLCDM]+|\d+|[A-Z])(?:\.\d+)*)'
    r'\s*\.\s*'
    r'(?P<title>[^\r\n<]{2,180})\s*$',
    re.IGNORECASE | re.UNICODE,
)


def _extract_note_heading(text: str) -> tuple[str, str]:
    """Return the deepest numbered heading and its original title."""
    candidates: list[tuple[int, int, str, str]] = []
    for match in NOTE_HEADING_RE.finditer(text or ''):
        number = match.group('number').strip().rstrip('.')
        title = MULTI_SPACE_RE.sub(' ', match.group('title')).strip(' .:-')
        if not re.search(r'[A-Za-zÀ-ỹ]', title):
            continue
        depth = number.count('.') + 1
        candidates.append((depth, match.start(), number, title))

    if not candidates:
        return '', ''

    _, _, number, title = max(candidates, key=lambda item: (item[0], item[1]))
    return number, title


def _extract_note_subtype(text: str) -> str:
    """Extract a stable note subtype, preferring the most specific heading."""
    number, title = _extract_note_heading(text)
    if not number:
        return 'note_unknown'
    number_key = number.replace('.', '_')
    title_key = re.sub(r'[^0-9A-Za-zÀ-ỹ]+', '_', title, flags=re.UNICODE)
    title_key = title_key.strip('_').lower()
    return f"note_{number_key}_{title_key}" if title_key else f"note_{number_key}"


def _set_note_heading_fields(tbl: dict, text: str) -> None:
    number, title = _extract_note_heading(text)
    tbl['note_number'] = number
    tbl['note_title'] = title
    tbl['note_subtype'] = _extract_note_subtype(text)


def _is_table_of_contents(
    preceding_text: str,
    rows: list[list[str]],
    page_has_toc_heading: bool = False,
    page_table_count: int = 0,
) -> bool:
    """Detect a document TOC without treating a generic “Nội dung” column as one."""
    first_rows = rows[:3]
    first_rows_text = ' '.join(str(cell) for row in first_rows for cell in row)
    nearby_text = f"{preceding_text[-300:]} {first_rows_text}"
    if TOC_HEADING_RE.search(nearby_text):
        return True

    # Some OCR engines emit each TOC row as its own <table>.  The individual
    # row cannot satisfy the usual “two page references” rule, but the page
    # itself is still unambiguously a table of contents.
    if page_has_toc_heading:
        if page_table_count >= 3 and rows and all(len(row) <= 2 for row in rows):
            row_text = ' '.join(str(cell) for row in rows for cell in row)
            if TOC_PAGE_RE.search(row_text) or re.search(r'\d+\s*[-–]\s*\d+', row_text):
                return True

    folded_cells = [_fold_text(cell).strip() for row in first_rows for cell in row]
    has_content_column = any(cell in ('noi dung', 'ten bao cao') for cell in folded_cells)
    has_page_column = any(cell in ('trang', 'so trang') for cell in folded_cells)
    page_refs = sum(
        bool(TOC_PAGE_RE.match(str(cell)))
        for row in rows[:12]
        for cell in row
    )
    return (
        page_refs >= 2
        and (
            (has_content_column and has_page_column)
            or TOC_LIST_HEADING_RE.search(preceding_text[-300:]) is not None
        )
    )


def classify_tables(raw_tables: list[dict]) -> list[dict]:
    """Classify tables using a 3-tier pipeline."""
    for tbl in raw_tables:
        preceding_text = tbl.get('preceding_text', '')
        rows = tbl.get('rows', [])
        tbl['note_subtype'] = None  # default for non-note tables
        tbl['note_number'] = ''
        tbl['note_title'] = ''

        if tbl.get('skip_export_reason') == 'header_only':
            tbl['table_type'] = 'header_only'
            tbl['classification_method'] = 'header_recovery'
            continue
        if tbl.get('skip_export_reason') == 'administrative_boilerplate':
            tbl['table_type'] = 'boilerplate'
            tbl['classification_method'] = 'boilerplate_detection'
            continue

        first_2_rows_text = ""
        for r in rows[:2]:
            first_2_rows_text += " ".join(str(cell) for cell in r) + " "

        search_text_t1_t2 = preceding_text + " " + first_2_rows_text

        # A TOC often names all three financial statements.  Detect it before
        # the financial heading tiers so those names cannot win classification.
        if _is_table_of_contents(
            preceding_text,
            rows,
            bool(tbl.get('page_has_toc_heading')),
            int(tbl.get('page_table_count') or 0),
        ):
            tbl['table_type'] = 'table_of_contents'
            tbl['classification_method'] = 'toc_detection'
            _append_proof(
                tbl,
                stage='classification',
                rule='table_of_contents',
                action='skip_toc',
                reason='page/table evidence identifies a table-of-contents fragment',
                benefit='prevent navigation rows from becoming financial CSVs',
            )
            continue

        # Tier 1: Regex mã mẫu biểu
        m = MAU_BIEU_RE.search(search_text_t1_t2)
        if m:
            number = m.group(1).zfill(2)
            suffix = m.group(3).upper()
            if (number, suffix) in MAU_BIEU_TABLE_TYPE:
                tbl['table_type'] = MAU_BIEU_TABLE_TYPE[(number, suffix)]
                tbl['classification_method'] = 'regex_mau_bieu'
                if tbl['table_type'] == 'note_table':
                    _set_note_heading_fields(tbl, search_text_t1_t2)
                continue

        # Tier 2: Heading text fallback
        if HEADING_BALANCE_SHEET_RE.search(search_text_t1_t2):
            tbl['table_type'] = 'balance_sheet'
            tbl['classification_method'] = 'heading_text'
            continue
        if HEADING_INCOME_STMT_RE.search(search_text_t1_t2):
            tbl['table_type'] = 'income_statement'
            tbl['classification_method'] = 'heading_text'
            continue
        if HEADING_CASH_FLOW_RE.search(search_text_t1_t2):
            tbl['table_type'] = 'cash_flow'
            tbl['classification_method'] = 'heading_text'
            continue
        if HEADING_NOTES_RE.search(search_text_t1_t2):
            tbl['table_type'] = 'note_table'
            tbl['classification_method'] = 'heading_text'
            _set_note_heading_fields(tbl, search_text_t1_t2)
            continue

        # Tier 3: Structural fingerprint
        all_text = " ".join(" ".join(str(cell) for cell in r) for r in rows)
        if STRUCTURAL_BALANCE_SHEET_RE.search(all_text):
            tbl['table_type'] = 'balance_sheet'
            tbl['classification_method'] = 'structural_fingerprint'
            continue
        if STRUCTURAL_INCOME_STMT_RE.search(all_text):
            tbl['table_type'] = 'income_statement'
            tbl['classification_method'] = 'structural_fingerprint'
            continue
        if STRUCTURAL_CASH_FLOW_RE.search(all_text):
            tbl['table_type'] = 'cash_flow'
            tbl['classification_method'] = 'structural_fingerprint'
            continue

        # Default
        tbl['table_type'] = 'note_table'
        tbl['classification_method'] = 'default'
        _set_note_heading_fields(tbl, preceding_text)

    return raw_tables


# ═════════════════════════════════════════════════════════════════════
# Normalize & export CSV
# ═════════════════════════════════════════════════════════════════════

_CODE_HEADER_RE = re.compile(r'\b(?:ma\s*so|ma\s*chi\s*tieu|stt|so\s*thu\s*tu)\b')
_LABEL_HEADER_RE = re.compile(r'\b(?:chi\s*tieu|noi\s*dung|khoan\s*muc)\b')
_NOTE_HEADER_RE = re.compile(r'\bthuyet\s*minh\b')
_PERIOD_HEADER_RE = re.compile(
    r'\b(?:nam\s*(?:nay|truoc|hien\s*hanh)|ky\s*(?:nay|truoc)|'
    r'so\s*(?:cuoi|dau)\s*nam|ngay\s*\d{1,2}|'
    r'thang\s*\d{1,2}|(?:19|20)\d{2}|vnd|trieu|nghin)\b'
    r'|\d{1,2}\s*[./-]\s*\d{1,2}\s*[./-]\s*(?:19|20)?\d{2}'
    r'|(?:19|20)\d{2}\s*(?:vnd|dong|trieu|nghin)'
)
_ITEM_CODE_RE = re.compile(
    r'^(?:[A-Z]|[IVXLCDM]+|\d{1,4}(?:\.\d{1,3})*[A-Z]?)\.?$',
    re.IGNORECASE,
)


def _header_cell_role(cell: Any) -> str:
    folded = MULTI_SPACE_RE.sub(' ', _fold_text(cell)).strip()
    if not folded:
        return ''
    if _CODE_HEADER_RE.search(folded):
        return 'code'
    if _NOTE_HEADER_RE.search(folded):
        return 'note'
    if _LABEL_HEADER_RE.search(folded):
        return 'label'
    if _PERIOD_HEADER_RE.search(folded) or folded in {
        'dong', 'don vi dong', 'don vi: dong',
    }:
        return 'value'
    return ''


def _leading_header_row_count(rows: list[list[str]]) -> int:
    """Count consecutive header rows while tolerating rowspan-flattened units."""
    count = 0
    for row in rows[:3]:
        roles = [_header_cell_role(cell) for cell in row]
        nonempty_roles = [
            role for cell, role in zip(row, roles) if str(cell).strip()
        ]
        has_numeric_data = any(
            parse_vietnamese_number(str(cell)) is not None and not role
            for cell, role in zip(row, roles)
            if str(cell).strip()
        )
        if count and has_numeric_data and not all(
            role == 'value' for role in nonempty_roles
        ):
            break
        if any(roles):
            count += 1
        else:
            break
    return count


_DATE_CELL_RE = re.compile(
    r'^\s*\d{1,2}\s*[./-]\s*\d{1,2}\s*[./-]\s*(?:19|20)\d{2}\s*$'
)
_DATE_WITH_UNIT_RE = re.compile(
    r'^\s*\d{1,2}\s*[./-]\s*\d{1,2}\s*[./-]\s*(?:19|20)\d{2}'
    r'\s*(?:(?:triệu|trieu|nghìn|nghin)\s*)?(?:vnd|đồng|dong)?\s*$',
    re.IGNORECASE,
)
_UNIT_RE = re.compile(r'\b(?:vnd|đồng|dong|triệu|trieu|nghìn|nghin)\b', re.IGNORECASE)
# A percent unit must be explicit in the OCR value or its own value-column
# header.  Wording such as ``lãi suất`` can name a monetary maturity bucket,
# so it is not unit evidence by itself.
PERCENT_LITERAL_RE = re.compile(r'%')
_ADMINISTRATIVE_BOILERPLATE_RE = re.compile(
    r'\b(?:mẫu\s*số|mau\s*so|ban\s*hành|ban\s*hanh|thông\s*tư|thong\s*tu|'
    r'bộ\s*tài\s*chính|bo\s*tai\s*chinh)\b',
    re.IGNORECASE,
)


def _table_width(rows: list[list[str]]) -> int:
    return max((len(row) for row in rows), default=0)


def _is_date_cell(value: Any) -> bool:
    text = str(value)
    return bool(_DATE_CELL_RE.fullmatch(text) or _DATE_WITH_UNIT_RE.fullmatch(text))


def _has_financial_number(value: Any) -> bool:
    """Return True only for a value-shaped financial amount, never a date."""
    text = str(value).strip()
    if not text or _is_date_cell(text):
        return False
    return bool(re.search(r'(?<!\d)\d{1,3}(?:[.,]\d{3})+(?!\d)', text))


def _is_header_only_table(rows: list[list[str]]) -> bool:
    """Conservatively identify a detached header table.

    This deliberately rejects short tables containing percentages or any
    financial amount: a small data table must never be consumed as a header.
    """
    if not 1 <= len(rows) <= 3:
        return False
    cells = [str(cell).strip() for row in rows for cell in row if str(cell).strip()]
    if not cells or any(_has_financial_number(cell) for cell in cells):
        return False
    if any(re.search(r'\d+(?:[,.]\d+)?\s*%', cell) for cell in cells):
        return False
    folded = ' '.join(_fold_text(cell) for cell in cells)
    role_count = sum(bool(_header_cell_role(cell)) for cell in cells)
    return role_count >= 2 or bool(
        re.search(r'\b(?:thuyet minh|so cuoi|so dau|nam nay|nam truoc|don vi)\b', folded)
        and _UNIT_RE.search(folded)
    )


def _is_administrative_boilerplate(rows: list[list[str]]) -> bool:
    """Detect a form/legal fragment which must not become a data header."""
    if not 1 <= len(rows) <= 3 or _table_width(rows) != 1:
        return False
    text = ' '.join(str(cell).strip() for row in rows for cell in row if str(cell).strip())
    return bool(_ADMINISTRATIVE_BOILERPLATE_RE.search(text))


def _has_recoverable_value_header(rows: list[list[str]]) -> bool:
    """A detached header is useful only when it names at least two values."""
    if _table_width(rows) < 2:
        return False
    value_columns = {
        column
        for row in rows
        for column, cell in enumerate(row)
        if _header_cell_role(cell) == 'value'
    }
    return len(value_columns) >= 2


def _is_data_table(rows: list[list[str]]) -> bool:
    if len(rows) < 2:
        return False
    cells = [str(cell).strip() for row in rows[1:] for cell in row if str(cell).strip()]
    return len(cells) >= 2 and (
        sum(_has_financial_number(cell) for cell in cells) >= 2
        or len(rows) >= 4
    )


def _has_two_aligned_amount_rows(rows: list[list[str]], width: int) -> bool:
    """Require two body rows that really carry the same value layout."""
    matching_rows = []
    for row in rows:
        if len(row) != width:
            continue
        amount_columns = tuple(
            index for index, cell in enumerate(row) if _has_financial_number(cell)
        )
        if len(amount_columns) >= 2:
            matching_rows.append(amount_columns)
    return any(
        matching_rows.count(columns) >= 2 for columns in set(matching_rows)
    )


def _append_proof(table: dict, *, stage: str, rule: str, action: str,
                  source_table_ids: list[str] | None = None,
                  reason: str, benefit: str,
                  evidence: dict[str, Any] | None = None) -> None:
    """Attach an in-memory, source-traceable proof for a transformation."""
    proof = {
        'stage': stage,
        'rule': rule,
        'action': action,
        'table_id': table.get('table_id'),
        'doc_id': table.get('doc_id'),
        'ticker': table.get('ticker'),
        'year': table.get('year'),
        'source_line': table.get('start_line'),
        'source_table_ids': source_table_ids or [table.get('table_id')],
        'source_table_sha256': hashlib.sha256(
            str(table.get('table_html') or '').encode('utf-8')
        ).hexdigest(),
        'source_row_widths': [len(row) for row in table.get('rows') or []],
        'reason': reason,
        'benefit': benefit,
    }
    if evidence:
        proof['evidence'] = evidence
    table.setdefault('repair_proofs', []).append(proof)


def recover_headers_without_merging(raw_tables: list[dict]) -> list[dict]:
    """Attach a proven adjacent header to a body table without merging tables.

    A header may cross exactly one page boundary, but it must be the immediate
    previous source table in the same document and have the same expanded
    width.  The original header table is retained for audit and merely marked
    non-exportable; no rows are moved between table IDs.
    """
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for table in raw_tables:
        by_doc[str(table.get('doc_id') or '')].append(table)

    for tables in by_doc.values():
        tables.sort(key=lambda table: int(table.get('start_line') or 0))
        for table in tables:
            if _is_administrative_boilerplate(table.get('rows') or []):
                table['skip_export_reason'] = 'administrative_boilerplate'
                _append_proof(
                    table,
                    stage='header_recovery',
                    rule='administrative_boilerplate',
                    action='skip_boilerplate',
                    reason='one-column form/legal fragment is not a data header',
                    benefit='prevent false header inheritance without exporting boilerplate',
                )
        for header, body in zip(tables, tables[1:]):
            header_rows = header.get('rows') or []
            body_rows = body.get('rows') or []
            page_gap = int(body.get('page_number') or 0) - int(header.get('page_number') or 0)
            if not (
                0 <= page_gap <= 1
                and _is_header_only_table(header_rows)
                and not header.get('skip_export_reason')
                and _has_recoverable_value_header(header_rows)
                and _is_data_table(body_rows)
                and _table_width(header_rows) == _table_width(body_rows)
                and _has_two_aligned_amount_rows(body_rows, _table_width(header_rows))
            ):
                continue

            body['effective_header_rows'] = [list(row) for row in header_rows]
            body['header_source_table_id'] = header.get('table_id')
            header['skip_export_reason'] = 'header_only'
            _append_proof(
                body,
                stage='header_recovery',
                rule='adjacent_header_only',
                action='inherit_header_context',
                source_table_ids=[header.get('table_id'), body.get('table_id')],
                reason='adjacent header-only table has matching expanded width',
                benefit='restore column meaning without merging source tables',
            )
            _append_proof(
                header,
                stage='header_recovery',
                rule='adjacent_header_only',
                action='skip_header_only',
                source_table_ids=[header.get('table_id'), body.get('table_id')],
                reason='table is used only as proven header context for the next body table',
                benefit='prevent an empty/header-only CSV from entering the catalog',
            )
    return raw_tables


def _column_samples(
    rows: list[list[str]],
    column: int,
    start_row: int,
    limit: int = 40,
) -> list[str]:
    values = []
    for row in rows[start_row:start_row + limit]:
        if column < len(row):
            value = str(row[column]).strip()
            if value:
                values.append(value)
    return values


def _label_column_score(values: list[str]) -> tuple[float, float, int]:
    if not values:
        return (0.0, 0.0, 0)
    descriptive = [
        value for value in values
        if re.search(r'[A-Za-zÀ-ỹ]', value)
        and not _ITEM_CODE_RE.fullmatch(value)
    ]
    avg_length = sum(len(value) for value in descriptive) / max(len(descriptive), 1)
    return (len(descriptive) / len(values), avg_length, len(descriptive))


def _code_column_score(values: list[str]) -> tuple[float, int, float]:
    if not values:
        return (0.0, 0, float('-inf'))
    matches = [value for value in values if _ITEM_CODE_RE.fullmatch(value)]
    avg_length = sum(len(value) for value in matches) / max(len(matches), 1)
    return (len(matches) / len(values), len(matches), -avg_length)


def _numeric_column_score(values: list[str]) -> tuple[float, int]:
    if not values:
        return (0.0, 0)
    parsed = sum(parse_vietnamese_number(value) is not None for value in values)
    return (parsed / len(values), parsed)


def _detect_statement_columns(
    rows: list[list[str]],
) -> tuple[int, int, int, list[int], int]:
    """Infer label/code/note/value columns from both headers and cell shapes."""
    if not rows:
        return -1, -1, -1, [], 0

    header_row_count = _leading_header_row_count(rows)
    header_candidates = rows[:max(header_row_count, 1)]
    # The widest row is normally the real header.  Unit-only rows produced by
    # unhandled rowspans are shorter and must not shift value columns left.
    primary_header = max(
        header_candidates,
        key=lambda row: (len(row), sum(bool(_header_cell_role(cell)) for cell in row)),
    )

    label_col = -1
    code_col = -1
    note_col = -1
    val_cols: list[int] = []
    for column, cell in enumerate(primary_header):
        role = _header_cell_role(cell)
        if role == 'label' and label_col == -1:
            label_col = column
        elif role == 'code' and code_col == -1:
            code_col = column
        elif role == 'note' and note_col == -1:
            note_col = column
        elif role == 'value':
            val_cols.append(column)

    width = max(len(row) for row in rows[:max(header_row_count + 20, 20)])
    excluded = {column for column in (code_col, note_col) if column >= 0}
    excluded.update(val_cols)

    if label_col == -1:
        label_candidates = [column for column in range(width) if column not in excluded]
        if label_candidates:
            label_col = max(
                label_candidates,
                key=lambda column: _label_column_score(
                    _column_samples(rows, column, header_row_count)
                ),
            )

    if not val_cols:
        numeric_candidates = []
        for column in range(width):
            if column in (label_col, note_col, code_col):
                continue
            score = _numeric_column_score(
                _column_samples(rows, column, header_row_count)
            )
            if score[0] >= 0.35 and score[1] >= 2:
                numeric_candidates.append(column)
        # Financial statements in ViFinQA put current/prior periods on the
        # right.  Taking the two right-most numeric columns avoids mistaking a
        # leading STT column for a value column when a header was lost by OCR.
        val_cols = numeric_candidates[-2:]

    if code_col == -1:
        code_candidates = [
            column for column in range(width)
            if column not in {label_col, note_col, *val_cols}
        ]
        scored_codes = [
            (
                _code_column_score(_column_samples(rows, column, header_row_count)),
                -column,
                column,
            )
            for column in code_candidates
        ]
        if scored_codes:
            best_score, _, best_column = max(scored_codes)
            if best_score[0] >= 0.55 and best_score[1] >= 2:
                code_col = best_column

    if label_col == -1:
        used = {column for column in (code_col, note_col, *val_cols) if column >= 0}
        label_col = next((column for column in range(width) if column not in used), 0)

    val_cols = [
        column for column in val_cols
        if column not in (label_col, code_col, note_col)
    ][:2]
    return label_col, code_col, note_col, val_cols, header_row_count


def _effective_header_rows(table: dict) -> list[list[str]]:
    return [list(row) for row in table.get('effective_header_rows') or []]


def _detect_unit(table: dict, rows: list[list[str]]) -> str:
    header_text = ' '.join(
        str(cell) for row in (_effective_header_rows(table) or rows[:3]) for cell in row
    ).lower()
    if re.search(r'\b(?:triệu|trieu)\b', header_text):
        return 'million_VND'
    if re.search(r'\b(?:nghìn|nghin)\b', header_text):
        return 'thousand_VND'
    return 'VND'


def _has_inline_section_headers(rows: list[list[str]]) -> bool:
    """A numbered heading inside a table means one schema cannot be trusted."""
    for row in rows[1:]:
        nonempty = [str(cell).strip() for cell in row if str(cell).strip()]
        if len(nonempty) == 1 and re.match(r'^\d+(?:\.\d+)+\.?\s+', nonempty[0]):
            return True
    return False


def _has_more_than_two_value_columns(rows: list[list[str]]) -> bool:
    header_rows = rows[:max(_leading_header_row_count(rows), 1)]
    widest = max(header_rows, key=len, default=[])
    return sum(_header_cell_role(cell) == 'value' for cell in widest) > 2


def _note_requires_raw_fallback(rows: list[list[str]]) -> bool:
    """Keep text cells in mixed note tables instead of silently nulling them."""
    if len(rows) < 2:
        return False
    has_number = False
    has_text_value = False
    for row in rows[1:]:
        for cell in row[1:]:
            value = str(cell).strip()
            if not value or value in {'-', '–', '—'}:
                continue
            has_number = has_number or parse_vietnamese_number(value) is not None
            has_text_value = has_text_value or (
                parse_vietnamese_number(value) is None and not _is_date_cell(value)
            )
    return has_number and has_text_value


def _unsafe_merged_cell_evidence(table: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return OCR evidence when a rowspan joins multiple unrecoverable rows.

    This is deliberately high-precision.  A rowspan alone is a valid layout
    construct and has already been expanded by ``parse_html_table``.  We only
    reject a statement table when the source also proves that multiple labels
    and multiple monetary values have been concatenated, leaving no OCR-backed
    one-to-one mapping between them.
    """
    table_html = str(table.get('table_html') or '')
    if not table_html:
        return []

    merged_labels: list[dict[str, Any]] = []
    multi_amount_cells: list[dict[str, Any]] = []
    for match in HTML_CELL_RE.finditer(table_html):
        text = MULTI_SPACE_RE.sub(' ', HTML_TAG_RE.sub('', match.group('content'))).strip()
        if not text:
            continue
        rowspan_match = HTML_ROWSPAN_RE.search(match.group('attrs'))
        rowspan = int(rowspan_match.group(1)) if rowspan_match else 1
        label_starts = MERGED_LABEL_START_RE.findall(text)
        if rowspan > 1 and len(label_starts) >= 2 and re.search(r'[A-Za-zÀ-ỹ]', text):
            merged_labels.append({
                'rowspan': rowspan,
                'cell_text': text,
                'label_signal_count': len(label_starts),
            })

        amounts = FINANCIAL_AMOUNT_TOKEN_RE.findall(text)
        if len(amounts) >= 2:
            multi_amount_cells.append({
                'cell_text': text,
                'amount_tokens': amounts,
            })

    if not merged_labels or not multi_amount_cells:
        return []
    return [{
        'mapping_conflict': 'rowspan_concatenates_multiple_labels_and_amounts',
        'merged_label_cells': merged_labels[:5],
        'multi_amount_cells': multi_amount_cells[:10],
    }]


def _raw_fallback_dataframe(rows: list[list[str]], note_number: str, note_title: str) -> pd.DataFrame:
    width = _table_width(rows)
    raw_data = []
    for row_index, row in enumerate(rows):
        raw_cells = [str(cell).strip() for cell in row]
        if not any(raw_cells):
            continue
        record: dict[str, Any] = {
            'row_index': row_index,
            'row_label_raw': raw_cells[0] if raw_cells else '',
            'note_number': note_number,
            'note_title': note_title,
        }
        for column_index in range(1, width):
            record[f'value_{column_index}_raw'] = (
                raw_cells[column_index] if column_index < len(raw_cells) else ''
            )
        raw_data.append(record)
    return pd.DataFrame(raw_data)


def _percent_unit_evidence_for_row(
    row: list[str],
    value_columns: list[int],
    header_rows: list[list[str]],
) -> list[dict[str, Any]]:
    """Return only explicit OCR percent evidence for this row's value columns."""
    evidence: list[dict[str, Any]] = []
    for column in value_columns:
        if 0 <= column < len(row) and PERCENT_LITERAL_RE.search(str(row[column])):
            evidence.append({
                'source': 'value_cell', 'column': column, 'text': str(row[column]),
            })
    if evidence:
        return evidence
    for header_row_index, header in enumerate(header_rows):
        for column in value_columns:
            if column < len(header) and PERCENT_LITERAL_RE.search(str(header[column])):
                evidence.append({
                    'source': 'value_header', 'header_row_index': header_row_index,
                    'column': column, 'text': str(header[column]),
                })
    return evidence


def _percent_unit_for_row(
    row: list[str], value_columns: list[int], header_rows: list[list[str]],
) -> bool:
    """Compatibility wrapper for explicit-percent unit detection."""
    return bool(_percent_unit_evidence_for_row(row, value_columns, header_rows))


def _mapping_failure_evidence(
    rows: list[list[str]],
    label_col: int,
    code_col: int,
    note_col: int,
    value_columns: list[int],
    data_start: int,
) -> list[dict[str, Any]]:
    """Prove that a proposed statement mapping puts an amount in its label."""
    failures: list[dict[str, Any]] = []
    if code_col == label_col and code_col >= 0:
        # A canonical statement row needs independent code and label fields.
        # Choosing either side of an OCR-shifted layout would silently discard
        # the other, so preserve the table verbatim instead.
        return [{
            'mapping_conflict': 'code_and_label_use_same_column',
            'source_rows': [
                [str(cell) for cell in row]
                for row in rows[data_start:data_start + 5]
            ],
            'proposed_label_column': label_col,
            'proposed_code_column': code_col,
            'proposed_note_column': note_col,
            'proposed_value_columns': value_columns,
        }]
    for row_index, row in enumerate(rows[data_start:], start=data_start):
        label = str(row[label_col]).strip() if 0 <= label_col < len(row) else ''
        note = str(row[note_col]).strip() if 0 <= note_col < len(row) else ''
        if not _has_financial_number(label) or not re.search(r'[A-Za-zÀ-ỹ]', note):
            continue
        failures.append({
            'row_index': row_index,
            'source_row': [str(cell) for cell in row],
            'proposed_label_column': label_col,
            'proposed_note_column': note_col,
            'proposed_value_columns': value_columns,
        })
    return failures


def _composed_header_names(header_rows: list[list[str]], width: int) -> list[str]:
    """Compose one source-faithful name per column from a multi-row header."""
    names = []
    for column in range(width):
        parts = []
        for row in header_rows:
            if column >= len(row):
                continue
            part = MULTI_SPACE_RE.sub(' ', str(row[column]).strip())
            if part and part not in parts:
                parts.append(part)
        names.append(' '.join(parts) if parts else f'value_{column}')
    return names


def normalize_and_export(classified_tables: list[dict], output_dir: str) -> list[dict]:
    """Normalize classified tables and export to CSV."""
    os.makedirs(output_dir, exist_ok=True)

    for tbl in classified_tables:
        table_type = tbl.get('table_type')
        rows = tbl.get('rows', [])
        tbl.pop('csv_path', None)
        if table_type in ('table_of_contents', 'header_only') or tbl.get('skip_export_reason'):
            continue
        if not rows:
            continue

        table_id = tbl.get('table_id', 'unknown_table')
        entity_type = tbl.get('entity_type', 'DN')

        if table_type in ('balance_sheet', 'income_statement', 'cash_flow'):
            unsafe_merged_cells = _unsafe_merged_cell_evidence(tbl)
            if unsafe_merged_cells:
                _append_proof(
                    tbl,
                    stage='normalization',
                    rule='unsafe_merged_cell_mapping',
                    action='raw_fallback',
                    reason=(
                        'rowspan cell concatenates multiple financial labels and '
                        'other cells concatenate multiple amounts without an OCR-backed '
                        'one-to-one mapping'
                    ),
                    benefit=(
                        'preserve OCR cell positions without inventing financial rows or '
                        'assigning an amount to an unproven label'
                    ),
                    evidence={'affected_cells': unsafe_merged_cells},
                )
                df = _raw_fallback_dataframe(
                    rows,
                    str(tbl.get('note_number') or ''),
                    str(tbl.get('note_title') or ''),
                )
                if df.empty:
                    tbl['skip_export_reason'] = 'normalization_produced_no_rows'
                    continue
                csv_path = os.path.join(output_dir, f"{table_id}.csv")
                df.to_csv(csv_path, index=False)
                tbl['csv_path'] = f"data/{table_id}.csv"
                tbl['raw_fallback'] = True
                tbl['unsafe_merged_cell'] = True
                continue

            detection_rows = _effective_header_rows(tbl) + rows
            if _has_inline_section_headers(rows) or _has_more_than_two_value_columns(detection_rows):
                _append_proof(
                    tbl,
                    stage='normalization',
                    rule='non_binary_statement_schema',
                    action='raw_fallback',
                    reason='source table has inline sections or more than two value columns',
                    benefit='preserve every OCR cell instead of shifting or dropping columns',
                )
                df = _raw_fallback_dataframe(
                    rows,
                    str(tbl.get('note_number') or ''),
                    str(tbl.get('note_title') or ''),
                )
                if df.empty:
                    tbl['skip_export_reason'] = 'normalization_produced_no_rows'
                    continue
                csv_path = os.path.join(output_dir, f"{table_id}.csv")
                df.to_csv(csv_path, index=False)
                tbl['csv_path'] = f"data/{table_id}.csv"
                tbl['raw_fallback'] = True
                continue

            label_col, code_col, note_col, val_cols, data_start = (
                _detect_statement_columns(detection_rows)
            )

            # Header context belongs to another source table.  It informs
            # column detection only; body rows keep their own table identity.
            if _effective_header_rows(tbl):
                data_start = 0

            # ``detection_rows`` also contains every body row and must never
            # be used as unit evidence.  A percent elsewhere in a statement
            # cannot change the unit of this row.
            source_header_rows = _effective_header_rows(tbl) or rows[:data_start]
            base_unit = _detect_unit(tbl, source_header_rows)

            mapping_failures = _mapping_failure_evidence(
                rows, label_col, code_col, note_col, val_cols, data_start
            )
            if mapping_failures:
                _append_proof(
                    tbl,
                    stage='normalization',
                    rule='column_mapping_unreliable',
                    action='raw_fallback',
                    reason='proposed statement label column contains an OCR amount',
                    benefit='preserve source cell positions instead of exporting shifted values',
                    evidence={'affected_rows': mapping_failures[:20]},
                )
                df = _raw_fallback_dataframe(
                    rows,
                    str(tbl.get('note_number') or ''),
                    str(tbl.get('note_title') or ''),
                )
                if df.empty:
                    tbl['skip_export_reason'] = 'normalization_produced_no_rows'
                    continue
                csv_path = os.path.join(output_dir, f"{table_id}.csv")
                df.to_csv(csv_path, index=False)
                tbl['csv_path'] = f"data/{table_id}.csv"
                tbl['raw_fallback'] = True
                continue

            data = []
            percent_unit_proofs: list[dict[str, Any]] = []
            for r in rows[data_start:]:
                item_code = (
                    str(r[code_col]).strip()
                    if code_col >= 0 and code_col < len(r)
                    else ''
                )
                item_label_raw = str(r[label_col]).strip() if label_col < len(r) else ""

                if not item_label_raw and not item_code:
                    continue

                item_label_norm = normalize_item_label(item_label_raw)
                note_ref = (
                    str(r[note_col]).strip()
                    if note_col >= 0 and note_col < len(r)
                    else ''
                )

                period_current = parse_vietnamese_number(str(r[val_cols[0]])) if len(val_cols) > 0 and val_cols[0] < len(r) else None
                period_prior = parse_vietnamese_number(str(r[val_cols[1]])) if len(val_cols) > 1 and val_cols[1] < len(r) else None
                percent_evidence = _percent_unit_evidence_for_row(
                    r, val_cols, source_header_rows
                )
                row_unit = 'percent' if percent_evidence else base_unit
                if percent_evidence:
                    percent_unit_proofs.append({
                        'row_index': len(data) + data_start,
                        'item_label_raw': item_label_raw,
                        'unit': row_unit,
                        'ocr_evidence': percent_evidence,
                    })

                data.append({
                    'item_code': item_code,
                    'item_label_raw': item_label_raw,
                    'item_label_norm': item_label_norm,
                    'note_ref': note_ref,
                    'period_current': period_current,
                    'period_prior': period_prior,
                    'unit': row_unit,
                    'entity_type': entity_type
                })

            if not data:
                tbl['skip_export_reason'] = 'normalization_produced_no_rows'
                _append_proof(
                    tbl,
                    stage='normalization',
                    rule='empty_normalized_table',
                    action='skip_export',
                    reason='no source row produced a usable statement record',
                    benefit='prevent an empty CSV from entering the catalog',
                )
                continue
            if percent_unit_proofs:
                _append_proof(
                    tbl,
                    stage='normalization',
                    rule='unit_from_ocr_evidence',
                    action='assign_row_unit',
                    reason=(
                        'only rows with an explicit percent sign in an OCR value '
                        'or matching value header use percent'
                    ),
                    benefit=(
                        'prevent a percentage row or an interest-rate wording from '
                        'changing monetary rows in the same table'
                    ),
                    evidence={
                        'base_table_unit': base_unit,
                        'percent_rows': percent_unit_proofs[:50],
                    },
                )
            df = pd.DataFrame(data)
            csv_path = os.path.join(output_dir, f"{table_id}.csv")
            df.to_csv(csv_path, index=False)
            tbl['csv_path'] = f"data/{table_id}.csv"

        elif table_type == 'note_table':
            header_rows = _effective_header_rows(tbl)
            header = header_rows[-1] if header_rows else (rows[0] if rows else [])
            composed_headers = (
                _composed_header_names(header_rows, _table_width(header_rows))
                if header_rows else []
            )
            note_subtype = tbl.get('note_subtype', '')
            note_number = tbl.get('note_number', '')
            note_title = tbl.get('note_title', '')

            m = re.search(r'note_([0-9A-Z_]+)_(.*)', note_subtype or '')
            if m and not note_number:
                note_number = m.group(1).replace('_', '.')
            if m and not note_title:
                note_title = m.group(2).replace('_', ' ').title()

            # When a note contains both text and numeric values, the old
            # numeric-only export silently erased the text.  Use the existing
            # generic raw schema for the single source table instead.
            # A body may repeat a compact version of an inherited header.  It
            # supplies no data and must never become a row such as 31122023.
            body_header_count = _leading_header_row_count(rows) if header_rows else 0
            note_rows = rows[body_header_count:] if header_rows else rows[1:]
            if _has_inline_section_headers(rows) or _note_requires_raw_fallback(rows):
                _append_proof(
                    tbl,
                    stage='normalization',
                    rule='mixed_note_cells',
                    action='raw_fallback',
                    reason='numeric note export would discard non-numeric source cells',
                    benefit='preserve every OCR cell without splitting or merging tables',
                )
                df = _raw_fallback_dataframe(rows, note_number, note_title)
                if df.empty:
                    tbl['skip_export_reason'] = 'normalization_produced_no_rows'
                    continue
                csv_path = os.path.join(output_dir, f"{table_id}.csv")
                df.to_csv(csv_path, index=False)
                tbl['csv_path'] = f"data/{table_id}.csv"
                tbl['raw_fallback'] = True
                continue

            data = []
            for r in note_rows:
                if not r: continue
                row_label_raw = str(r[0]).strip()
                row_data = {
                    'row_label_raw': row_label_raw,
                }

                has_numeric = False
                for i, cell in enumerate(r[1:], start=1):
                    val = parse_vietnamese_number(str(cell))
                    if val is not None:
                        has_numeric = True
                    header_name = (
                        composed_headers[i]
                        if i < len(composed_headers) and composed_headers[i]
                        else (
                            str(header[i]).strip()
                            if i < len(header) and str(header[i]).strip()
                            else f"value_{i}"
                        )
                    )
                    row_data[header_name] = val

                if has_numeric:
                    row_data['note_number'] = note_number
                    row_data['note_title'] = note_title
                    data.append(row_data)

            if data:
                df = pd.DataFrame(data)
            else:
                # Preserve useful text-only notes (related parties, product
                # lists, accounting-policy details, etc.) instead of dropping
                # them merely because no cell parses as a number.  Unknown
                # page headers/boilerplate have no useful retrieval terms and
                # remain intentionally unexported.
                retrieval_terms = _note_retrieval_terms(tbl)
                if not retrieval_terms:
                    continue
                df = _raw_fallback_dataframe(rows, note_number, note_title)
                if df.empty:
                    continue

            csv_path = os.path.join(output_dir, f"{table_id}.csv")
            df.to_csv(csv_path, index=False)
            tbl['csv_path'] = f"data/{table_id}.csv"

    return classified_tables


# ═════════════════════════════════════════════════════════════════════
# Build taxonomy
# ═════════════════════════════════════════════════════════════════════

def _to_canonical_id(name_vi: str) -> str:
    """Convert Vietnamese label to snake_case English-ish ID. Simple transliteration."""
    # Remove diacritics
    s = ''.join(c for c in unicodedata.normalize('NFD', name_vi)
                if unicodedata.category(c) != 'Mn')
    s = s.replace('đ', 'd').replace('Đ', 'D')
    s = s.lower()
    # Replace spaces and non-alphanumeric with underscores
    s = re.sub(r'[^a-z0-9]+', '_', s)
    s = s.strip('_')
    # Truncate to 50 chars
    return s[:50].rstrip('_')


def _is_semantic_label(value: str) -> bool:
    """Reject OCR amounts/codes which cannot be a searchable label alias."""
    text = str(value or '').strip()
    return bool(text and re.search(r'[A-Za-zÀ-ỹ]', text) and not _has_financial_number(text))


def build_taxonomy(tables: list[dict], output_dir: str) -> dict[str, list[dict]]:
    project_root = Path(__file__).resolve().parent

    # entity_type → table_type → item_code → normalized-label frequencies
    label_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    # Some TCTD statements use a hierarchical STT (I, 1, a, ...), not a
    # globally unique regulatory code.  Such an STT can identify several rows
    # in the same table and must not be allowed to merge unrelated concepts.
    ambiguous_keys: set[tuple[str, str, str]] = set()
    exclusions: list[dict[str, Any]] = []

    for t in tables:
        if t.get('table_type') in ('balance_sheet', 'income_statement', 'cash_flow'):
            entity_type = t.get('entity_type')
            table_type = t.get('table_type')
            if t.get('raw_fallback'):
                if t.get('unsafe_merged_cell'):
                    exclusions.append({
                        'table_id': t.get('table_id', ''),
                        'doc_id': t.get('doc_id', ''),
                        'ticker': t.get('ticker', ''),
                        'year': t.get('year', 0),
                        'source_line': t.get('start_line', 0),
                        'statement': table_type,
                        'reason': 'unsafe_merged_cell_raw_fallback',
                    })
                # Raw fallback has no proven item-label schema.  Its text must
                # not become a taxonomy alias or dense-retrieval keyword.
                continue
            csv_path = project_root / t.get('csv_path', '')

            if not csv_path.exists():
                continue

            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    labels_in_table: dict[str, set[str]] = defaultdict(set)
                    for row in reader:
                        item_code = (row.get('item_code') or '').strip()
                        item_label_norm = (row.get('item_label_norm') or '').strip()
                        if item_code and item_label_norm and _is_semantic_label(item_label_norm):
                            label_counts[entity_type][table_type][item_code][item_label_norm] += 1
                            labels_in_table[item_code].add(item_label_norm)
                        elif item_code and item_label_norm:
                            exclusions.append({
                                'table_id': t.get('table_id', ''),
                                'doc_id': t.get('doc_id', ''),
                                'ticker': t.get('ticker', ''),
                                'year': t.get('year', 0),
                                'source_line': t.get('start_line', 0),
                                'statement': table_type,
                                'item_code': item_code,
                                'rejected_alias': item_label_norm,
                                'reason': 'numeric_or_code_like_label',
                            })
                    for item_code, labels in labels_in_table.items():
                        if len(labels) > 1:
                            ambiguous_keys.add((entity_type, table_type, item_code))
            except (OSError, csv.Error) as exc:
                logging.warning("Failed to build taxonomy from %s: %s", csv_path, exc)

    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    _save_jsonl(exclusions, out_dir_path / 'taxonomy_exclusions.jsonl')

    result = {}
    universal_concepts: dict[str, set[str]] = defaultdict(set)

    for entity_type in ('DN', 'TCTD', 'CTCK', 'BH'):
        taxonomy_list = []
        if entity_type in label_counts:
            for statement in ('balance_sheet', 'income_statement', 'cash_flow'):
                statement_counts = label_counts[entity_type].get(statement, {})
                for item_code, counts in statement_counts.items():
                    key = (entity_type, statement, item_code)
                    # For a genuinely reusable item code, retain OCR/wording
                    # variations as aliases.  For a repeated STT, emit one
                    # concept per label so unrelated rows remain separated.
                    count_groups = (
                        [Counter({label: count}) for label, count in counts.items()]
                        if (
                            key in ambiguous_keys
                            or (
                                entity_type in ('TCTD', 'CTCK')
                                and len(counts) > 1
                            )
                        )
                        else [counts]
                    )
                    for concept_counts in count_groups:
                        most_common_label = concept_counts.most_common(1)[0][0]
                        aliases = [label for label, _ in concept_counts.most_common()]
                        entry = {
                            'item_code': item_code,
                            'canonical_id': _to_canonical_id(most_common_label),
                            'name_vi': most_common_label,
                            'name_en': '',
                            'aliases': aliases,
                            'statement': statement,
                            'entity_type': entity_type,
                            'unit': 'VND',
                            'source': ''
                        }
                        taxonomy_list.append(entry)

                    universal_concepts[f'{statement}:{item_code}'].add(entity_type)

        # Save taxonomy
        with open(out_dir_path / f'taxonomy_{entity_type}.json', 'w', encoding='utf-8') as f:
            json.dump(taxonomy_list, f, ensure_ascii=False, indent=2)

        result[entity_type] = taxonomy_list

    # Save universal concepts
    with open(out_dir_path / 'universal_concepts.json', 'w', encoding='utf-8') as f:
        json.dump(
            {key: sorted(entity_types) for key, entity_types in universal_concepts.items()},
            f,
            ensure_ascii=False,
            indent=2,
        )

    return result


# ═════════════════════════════════════════════════════════════════════
# Semantic fields + retrieval context
# ═════════════════════════════════════════════════════════════════════

def assign_semantic_fields(tables: list[dict], taxonomy: dict[str, list[dict]]) -> list[dict]:
    project_root = Path(__file__).resolve().parent

    # build quick lookup
    tax_lookup = {}
    alias_lookup = {}
    for et, entries in taxonomy.items():
        tax_lookup[et] = defaultdict(list)
        alias_lookup[et] = defaultdict(list)
        for entry in entries:
            statement = entry.get('statement', '')
            item_code = str(entry.get('item_code', ''))
            tax_lookup[et][(statement, item_code)].append(entry)
            for alias in {entry.get('name_vi', ''), *entry.get('aliases', [])}:
                alias_key = _fold_text(normalize_item_label(str(alias)))
                if alias_key:
                    alias_lookup[et][(statement, alias_key)].append(entry)

    for t in tables:
        table_type = t.get('table_type')
        if table_type in ('balance_sheet', 'income_statement', 'cash_flow'):
            semantic_fields = []
            if t.get('raw_fallback'):
                t['semantic_fields'] = semantic_fields
                continue
            seen_fields = set()
            entity_type = t.get('entity_type')
            csv_path = project_root / t.get('csv_path', '')

            if csv_path.exists():
                try:
                    with open(csv_path, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            item_code = (row.get('item_code') or '').strip()
                            item_label = (row.get('item_label_norm') or '').strip()
                            label_key = _fold_text(normalize_item_label(item_label))
                            candidates = tax_lookup.get(entity_type, {}).get(
                                (table_type, item_code), []
                            ) if item_code else []

                            tax_entry = None
                            if len(candidates) == 1:
                                tax_entry = candidates[0]
                            elif candidates and label_key:
                                exact = [
                                    entry for entry in candidates
                                    if label_key in {
                                        _fold_text(normalize_item_label(str(alias)))
                                        for alias in {
                                            entry.get('name_vi', ''),
                                            *entry.get('aliases', []),
                                        }
                                    }
                                ]
                                if len(exact) == 1:
                                    tax_entry = exact[0]

                            # Tables with no code (common in older TCTD files)
                            # may still reuse an unambiguous label from the same
                            # entity type and statement.
                            if tax_entry is None and label_key:
                                by_alias = alias_lookup.get(entity_type, {}).get(
                                    (table_type, label_key), []
                                )
                                unique_aliases = {
                                    (entry['canonical_id'], entry['item_code']): entry
                                    for entry in by_alias
                                }
                                if len(unique_aliases) == 1:
                                    tax_entry = next(iter(unique_aliases.values()))

                            if tax_entry is not None:
                                field_key = (
                                    tax_entry['canonical_id'],
                                    tax_entry['item_code'],
                                    item_label,
                                )
                                if field_key in seen_fields:
                                    continue
                                seen_fields.add(field_key)
                                semantic_fields.append({
                                    'field_id': tax_entry['canonical_id'],
                                    'canonical_name_vi': tax_entry['name_vi'],
                                    'aliases': tax_entry['aliases'],
                                    'item_code': item_code,
                                    'unit': (row.get('unit') or 'VND').strip(),
                                })
                except (OSError, csv.Error) as exc:
                    logging.warning("Failed to assign semantics from %s: %s", csv_path, exc)
            t['semantic_fields'] = semantic_fields
        else:
            t['semantic_fields'] = []

    return tables


def build_retrieval_context(tables: list[dict]) -> list[dict]:
    for t in tables:
        table_type = t.get('table_type')
        if table_type == 'table_of_contents':
            t['retrieval_context'] = {}
            continue
        ticker = t.get('ticker', '')
        year = t.get('year', '')
        report_type = t.get('folder_type', 'other')
        if report_type not in REPORT_TYPE_VI:
            report_type = 'other'
        folder_type_vi = REPORT_TYPE_VI[report_type]

        keywords = []
        semantic_summary = ""

        if table_type in ('balance_sheet', 'income_statement', 'cash_flow'):
            # Aggregate keywords from semantic_fields
            semantic_fields = t.get('semantic_fields', [])
            all_kw = set()
            top_names = []
            for sf in semantic_fields:
                if sf.get('canonical_name_vi'):
                    all_kw.add(sf['canonical_name_vi'])
                    if len(top_names) < 5:
                        top_names.append(sf['canonical_name_vi'])
                if sf.get('canonical_name_en'):
                    all_kw.add(sf['canonical_name_en'])
                for alias in sf.get('aliases', []):
                    all_kw.add(alias)

            keywords = sorted(all_kw)
            table_type_vi = TABLE_TYPE_VI.get(table_type, table_type)
            top_names_str = ', '.join(top_names)

            semantic_summary = f"Báo cáo {table_type_vi} {folder_type_vi} của {ticker} năm {year}, gồm các chỉ tiêu: {top_names_str}"

        elif table_type == 'note_table':
            note_subtype = t.get('note_subtype', '')
            note_title = _clean_note_title(t.get('note_title', ''))
            keywords = _note_retrieval_terms(t)

            if note_title:
                note_number = str(t.get('note_number') or '').strip()
                subject = f"{note_number} - {note_title}" if note_number else note_title
                semantic_summary = f"Thuyết minh {subject} của {ticker} năm {year}"
            elif keywords:
                semantic_summary = (
                    f"Thuyết minh về {', '.join(keywords[:5])} "
                    f"của {ticker} năm {year}"
                )
            else:
                # Keep an explicit low-quality marker so indexing can reject
                # notes that cannot be distinguished using metadata alone.
                subtype_str = note_subtype or 'note_unknown'
                semantic_summary = f"Thuyết minh {subtype_str} của {ticker} năm {year}"

        embedding_text = f"{ticker} | {table_type} | {year} | {report_type} | {semantic_summary}"

        t['retrieval_context'] = {
            'keywords': keywords,
            'semantic_summary': semantic_summary,
            'embedding_text': embedding_text
        }

    return tables


# ═════════════════════════════════════════════════════════════════════
# Generate metadata
# ═════════════════════════════════════════════════════════════════════

def load_company_names(path: str | Path) -> dict[str, str]:
    """Load the canonical company name for each ticker from ``code_stock.csv``."""
    company_names: dict[str, str] = {}
    with open(path, 'r', encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            ticker = str(row.get('Mã CK') or row.get('ticker') or '').strip().upper()
            company_name = str(
                row.get('Tên công ty') or row.get('company_name') or ''
            ).strip()
            if not ticker or not company_name:
                continue
            previous = company_names.get(ticker)
            if previous and previous != company_name:
                raise ValueError(f'Conflicting company names for ticker {ticker}')
            company_names[ticker] = company_name
    return company_names


def generate_all_metadata(
    tables: list[dict],
    output_dir: str,
    company_names: Mapping[str, str] | None = None,
    inventory: pd.DataFrame | list[dict] | None = None,
    entity_type_map: Mapping[str, str] | None = None,
) -> None:
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    names_by_ticker = {
        str(ticker).strip().upper(): str(company_name).strip()
        for ticker, company_name in (company_names or {}).items()
    }

    docs_map = {}
    tables_metadata = []

    for t in tables:
        # The final catalog is intentionally one-to-one with exported CSVs.
        # TOC tables and tables which produced no CSV remain available in the
        # intermediate audit files, but must not enter the searchable catalog.
        if t.get('table_type') == 'table_of_contents' or not t.get('csv_path'):
            continue
        doc_id = t.get('doc_id')
        if not doc_id:
            continue

        if doc_id not in docs_map:
            doc_report_type = t.get('folder_type', 'other')
            if doc_report_type not in REPORT_TYPE_VI:
                doc_report_type = 'other'
            docs_map[doc_id] = {
                'doc_id': doc_id,
                'doc_path': t.get('doc_path', ''),
                'ticker': t.get('ticker', ''),
                'year': t.get('year', 0),
                'entity_type': t.get('entity_type', ''),
                'report_type': doc_report_type,
                'consolidated': doc_report_type == 'consolidated',
                'table_count': 0,
            }
        docs_map[doc_id]['table_count'] += 1

        table_id = t.get('table_id', '')
        ticker = str(t.get('ticker', '')).strip().upper()
        company_name = str(t.get('company_name', '')).strip()
        if not company_name:
            company_name = names_by_ticker.get(ticker, '')
        report_type = t.get('folder_type', '')
        if report_type not in ('consolidated', 'separate', 'aggregated', 'other'):
            report_type = 'consolidated' if t.get('consolidated', False) else 'separate'

        tm = {
            'table_id': table_id,
            'doc_id': doc_id,
            'start_line': t.get('start_line', 0),
            'ticker': ticker,
            'company_name': company_name,
            'year': t.get('year', 0),
            'report_type': report_type,
            'table_type': t.get('table_type', ''),
            'csv_path': t.get('csv_path', ''),
            'semantic_fields': t.get('semantic_fields', []),
            'retrieval_context': t.get('retrieval_context', {})
        }
        tables_metadata.append(tm)

    if inventory is not None:
        inventory_records = (
            inventory.to_dict('records')
            if hasattr(inventory, 'to_dict')
            else list(inventory)
        )
        docs_metadata = []
        for row in inventory_records:
            doc_id = str(row.get('doc_id') or '').strip()
            if not doc_id:
                continue
            report_type = row.get('folder_type') or 'other'
            if report_type not in REPORT_TYPE_VI:
                report_type = 'other'
            ticker = str(row.get('ticker') or '').strip().upper()
            docs_metadata.append({
                'doc_id': doc_id,
                'doc_path': str(row.get('file_path') or '').replace('\\', '/'),
                'ticker': ticker,
                'year': int(row.get('year') or 0),
                'entity_type': (entity_type_map or {}).get(ticker, 'DN'),
                'report_type': report_type,
                # Kept for compatibility with existing consumers.  It is not
                # sufficient to distinguish separate/aggregated/other.
                'consolidated': report_type == 'consolidated',
                'table_count': sum(
                    1 for table in tables
                    if table.get('doc_id') == doc_id and table.get('csv_path')
                ),
            })
    else:
        docs_metadata = list(docs_map.values())

    with open(out_dir_path / 'docs_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(docs_metadata, f, ensure_ascii=False, indent=2)

    with open(out_dir_path / 'tables_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(tables_metadata, f, ensure_ascii=False, indent=2)


# ═════════════════════════════════════════════════════════════════════
# Orchestrator
# ═════════════════════════════════════════════════════════════════════

def _ensure_dirs(root: Path) -> None:
    for d in ("data", "metadata", "intermediate", "taxonomy"):
        (root / d).mkdir(parents=True, exist_ok=True)


def _clear_generated_csvs(output_dir: Path) -> int:
    """Remove only generated CSV artifacts before a full rebuild.

    ``normalize_and_export`` already skips TOC tables, but old CSVs remain in
    ``data/`` when the directory is reused.  Clearing only ``*.csv`` makes the
    rebuild deterministic while preserving ``.gitkeep`` and unrelated files.
    """
    removed = 0
    for path in output_dir.glob('*.csv'):
        if path.is_file():
            path.unlink()
            removed += 1
    return removed


def _save_skipped_tables(tables: list[dict], path: Path) -> None:
    """Persist non-exported tables with a reason for auditability."""
    records = []
    for table in tables:
        if table.get('table_type') == 'table_of_contents':
            reason = 'table_of_contents'
        elif table.get('skip_export_reason'):
            reason = str(table['skip_export_reason'])
        elif not table.get('csv_path'):
            reason = 'normalization_produced_no_csv'
        else:
            continue
        records.append({
            'table_id': table.get('table_id'),
            'doc_id': table.get('doc_id'),
            'ticker': table.get('ticker'),
            'year': table.get('year'),
            'report_type': table.get('folder_type', 'other'),
            'table_type': table.get('table_type'),
            'reason': reason,
        })
    _save_jsonl(records, path)


def _save_transform_proofs(tables: list[dict], path: Path) -> None:
    """Write one machine-readable PoC for every intentional transformation."""
    proofs: list[dict[str, Any]] = []
    for table in tables:
        for proof in table.get('repair_proofs') or []:
            record = dict(proof)
            record['result_csv_path'] = table.get('csv_path', '')
            csv_path = str(table.get('csv_path') or '')
            if csv_path:
                csv_file = path.parent.parent / csv_path
                if csv_file.is_file():
                    with csv_file.open(encoding='utf-8-sig', newline='') as handle:
                        record['after_columns'] = next(csv.reader(handle), [])
                    record['result_csv_sha256'] = hashlib.sha256(
                        csv_file.read_bytes()
                    ).hexdigest()
            proofs.append(record)
    _save_jsonl(proofs, path)


def validate_generated_outputs(
    tables: list[dict],
    data_dir: str | Path,
    metadata_dir: str | Path,
) -> None:
    """Validate the final one-to-one CSV/table metadata contract."""
    usable = [table for table in tables if table.get('csv_path')]
    table_ids = [str(table.get('table_id') or '') for table in usable]
    if any(table.get('table_type') == 'table_of_contents' for table in usable):
        raise ValueError('table_of_contents không được xuất vào catalog cuối')
    if any(table.get('table_type') == 'header_only' for table in usable):
        raise ValueError('header_only không được xuất vào catalog cuối')
    if any(not table_id for table_id in table_ids):
        raise ValueError('table metadata có table_id rỗng')
    if len(table_ids) != len(set(table_ids)):
        raise ValueError('table_id bị trùng trong catalog cuối')

    data_root = Path(data_dir)
    expected_csvs = {f'{table_id}.csv' for table_id in table_ids}
    actual_csvs = {path.name for path in data_root.glob('*.csv')}
    if expected_csvs != actual_csvs:
        missing = sorted(expected_csvs - actual_csvs)[:5]
        orphan = sorted(actual_csvs - expected_csvs)[:5]
        raise ValueError(
            f'CSV/catalog mismatch: missing={missing}, orphan={orphan}'
        )
    for table in usable:
        csv_path = str(table.get('csv_path') or '')
        if csv_path != f"data/{table['table_id']}.csv":
            raise ValueError(f"csv_path không canonical cho {table['table_id']}")
        if table.get('folder_type', 'other') not in {
            'consolidated', 'separate', 'aggregated', 'other',
        }:
            raise ValueError(f"report_type không hợp lệ cho {table['table_id']}")
        csv_file = data_root / f"{table['table_id']}.csv"
        if csv_file.stat().st_size <= 2:
            raise ValueError(f"CSV rỗng không được export: {table['table_id']}")

    metadata_path = Path(metadata_dir) / 'tables_metadata.json'
    with metadata_path.open(encoding='utf-8') as handle:
        metadata = json.load(handle)
    metadata_ids = [str(row.get('table_id') or '') for row in metadata]
    if metadata_ids != table_ids:
        raise ValueError('tables_metadata.json không khớp với các table đã export')


def _save_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def prepare(financial_statements_dir: str) -> None:
    """
    Run the full data-processing pipeline (Stage 0 → 6).

    Args:
        financial_statements_dir: Path to directory containing the OCR
            ``.txt`` files organised as
            ``{TICKER}/{YEAR}/{DOC_DIR}/{DOC_DIR}_extracted.txt``.
    """
    fs_dir = Path(financial_statements_dir)
    root = Path(__file__).resolve().parent
    _ensure_dirs(root)

    # ── Inventory & entity_type ───────────────────────────
    print("═══ Stage 0 - Inventory & entity_type ═══")
    inventory, entity_map = scan_inventory(str(fs_dir))
    inventory.to_csv(root / "intermediate" / "inventory.csv", index=False)
    (root / "intermediate" / "ticker_entity_type_map.json").write_text(
        json.dumps(entity_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  {len(inventory)} documents · {len(entity_map)} tickers")

    # ── Stage 1: Parse .txt → raw tables ───────────────────────────
    print("═══ Stage 1: Parse .txt → raw tables ═══")
    raw_tables = parse_all_files(inventory, entity_map, str(fs_dir))
    _save_jsonl(raw_tables, root / "intermediate" / "raw_tables.jsonl")
    print(f"  {len(raw_tables)} tables extracted")

    # Header context is attached before classification/normalization, but
    # source tables remain separate and retain their original IDs.
    recover_headers_without_merging(raw_tables)

    # ── Stage 2: Classify table_type ───────────────────────────────
    print("═══ Stage 2: Classify table_type ═══")
    classified = classify_tables(raw_tables)
    _save_jsonl(classified, root / "intermediate" / "classified_tables.jsonl")
    counts = Counter(t["table_type"] for t in classified)
    print(f"  {len(classified)} tables classified: {dict(counts)}")

    # ── Stage 3: Normalize & export CSV ────────────────────────────
    print("═══ Stage 3: Normalize & export CSV ═══")
    removed_stale = _clear_generated_csvs(root / "data")
    if removed_stale:
        print(f"  Removed {removed_stale} stale CSV artifacts")
    exported = normalize_and_export(classified, str(root / "data"))
    _save_skipped_tables(exported, root / "intermediate" / "skipped_tables.jsonl")
    _save_transform_proofs(exported, root / "intermediate" / "transform_proofs.jsonl")
    exported_tables = [table for table in exported if table.get('csv_path')]
    exported_count = sum(bool(table.get('csv_path')) for table in exported)
    print(f"  {exported_count} CSV files written")

    # ── Stage 4: Build taxonomy ────────────────────────────────────
    print("═══ Stage 4: Build taxonomy ═══")
    taxonomy = build_taxonomy(exported_tables, str(root / "taxonomy"))
    total_entries = sum(len(v) for v in taxonomy.values())
    print(f"  {len(taxonomy)} entity-types · {total_entries} entries")

    # ── Stage 5: Semantic fields + retrieval_context (no LLM) ──────
    print("═══ Stage 5: Semantic fields + retrieval context ═══")
    with_sem = assign_semantic_fields(exported_tables, taxonomy)
    with_ctx = build_retrieval_context(with_sem)
    print(f"  Assigned semantic fields & retrieval context to {len(with_ctx)} tables")

    # ── Stage 6: Metadata ──────────────────────────────────────────
    print("═══ Stage 6: Generate metadata ═══")
    company_names = load_company_names(root / "ViFinQA" / "code_stock.csv")
    generate_all_metadata(
        with_ctx,
        str(root / "metadata"),
        company_names,
        inventory=inventory,
        entity_type_map=entity_map,
    )
    validate_generated_outputs(with_ctx, root / "data", root / "metadata")
    print("  docs_metadata.json · tables_metadata.json")

    print("\n✓ Pipeline complete")
    print(f"  CSVs       → {root / 'data'}")
    print(f"  Metadata   → {root / 'metadata'}")
    print(f"  Taxonomy   → {root / 'taxonomy'}")
    print(f"  Intermediate → {root / 'intermediate'}")


# ═════════════════════════════════════════════════════════════════════
# Legacy API kept for backward-compat with existing imports
# ═════════════════════════════════════════════════════════════════════

def parse_ocr_files(
    dir: str,
) -> list[tuple[str, dict[str, Any], str, list[str]]]:
    """Thin wrapper kept for backward-compatibility."""
    raise NotImplementedError(
        "Use prepare() for the full pipeline. "
        "Direct parse_ocr_files() is no longer supported."
    )


def save_tables_as_csv(
    tables: list[tuple[str, dict[str, Any], str, list[str]]],
) -> None:
    raise NotImplementedError("Use prepare() instead.")


def generate_metadata(
    tables: list[tuple[str, dict[str, Any], str, list[str]]],
    docs: list[dict[str, Any]],
) -> None:
    raise NotImplementedError("Use prepare() instead.")


# ═════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    DEFAULT_FS_DIR = "ViFinQA/financial_statements"
    prepare(DEFAULT_FS_DIR)
