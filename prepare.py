"""Phase 1: Parse OCR files, extract tables, generate metadata.

Consolidates the full data-processing pipeline (GĐ 0–6) from
implementation_plan.md into a single file.  Run with:

    python prepare.py

Uses ``ViFinQA/financial_statements`` as the default input directory.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import unicodedata
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Optional

import pandas as pd


# ═════════════════════════════════════════════════════════════════════
# Utilities — Regex patterns
# ═════════════════════════════════════════════════════════════════════

PAGE_MARKER_RE = re.compile(r'={3,}\s*PAGE\s+(\d+)\s*={3,}')

TABLE_TAG_RE = re.compile(r'<table[^>]*>.*?</table>', re.DOTALL | re.IGNORECASE)

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
        r'|tổ\s*chức\s*tín\s*dụng',
        re.IGNORECASE,
    ),
    'CTCK': re.compile(
        r'[Bb]\s*\d{1,2}\s*[a-z]?\s*[-–/]\s*CTCK'
        r'|[Mm][ẫâấ]u\s*[Bb]\s*\d{1,2}\s*[a-z]?\s*[-–/]\s*CTCK'
        r'|Công\s*ty\s*(?:Cổ\s*phần\s*)?Chứng\s*khoán',
        re.IGNORECASE,
    ),
    'BH': re.compile(
        r'dự\s*phòng\s*nghiệp\s*vụ\s*bảo\s*hiểm'
        r'|bảo\s*hiểm\s*nhân\s*thọ'
        r'|phí\s*bảo\s*hiểm\s*gốc',
        re.IGNORECASE,
    ),
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
LEADING_NUMBER_RE = re.compile(r'^\s*(?:[IVXivx]+\.|[A-Za-z]\.|[A-Za-z]\)|\d+\.)\s*')
# Collapse whitespace
MULTI_SPACE_RE = re.compile(r'\s+')


def normalize_item_label(raw: str) -> str:
    """Clean item_label_raw → item_label_norm."""
    s = raw.strip()
    s = FORMULA_IN_PARENS_RE.sub('', s)
    s = LEADING_NUMBER_RE.sub('', s)
    s = MULTI_SPACE_RE.sub(' ', s)
    return s.strip()


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
        self._in_td = False
        self._in_table = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        tag_lower = tag.lower()
        if tag_lower == 'table':
            self._in_table = True
        elif tag_lower == 'tr' and self._in_table:
            self._current_row = []
            self._colspans = []
        elif tag_lower in ('td', 'th') and self._in_table:
            self._in_td = True
            self._current_cell = []
            attrs_dict = dict(attrs)
            self._colspans.append(int(attrs_dict.get('colspan', '1') or '1'))

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower in ('td', 'th') and self._in_td:
            self._in_td = False
            cell_text = ''.join(self._current_cell).strip()
            self._current_row.append(cell_text)
        elif tag_lower == 'tr' and self._in_table and self._current_row:
            expanded: list[str] = []
            for i, cell in enumerate(self._current_row):
                colspan = self._colspans[i] if i < len(self._colspans) else 1
                expanded.append(cell)
                for _ in range(colspan - 1):
                    expanded.append('')
            self.rows.append(expanded)
        elif tag_lower == 'table':
            self._in_table = False

    def handle_data(self, data: str) -> None:
        if self._in_td:
            self._current_cell.append(data)


def parse_html_table(html: str) -> list[list[str]]:
    """
    Parse an HTML table string into a 2D list of cell values.

    Handles colspan by expanding cells. Rowspan is not handled
    (rare in Vietnamese financial statement tables).
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

    if suffix_counts:
        return suffix_counts.most_common(1)[0][0]

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
                    for m in MAU_BIEU_RE.finditer(content):
                        suffix = m.group(3).upper()
                        if suffix == 'HN':
                            suffix = 'DN'
                        ticker_suffix_counts[ticker][suffix] += 1

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
        if not counts:
            entity_type_map[ticker] = 'DN'
            continue

        # Remove DN from counts to see if any non-DN type exists
        non_dn = {k: v for k, v in counts.items() if k != 'DN'}
        if non_dn:
            # If ANY non-DN mã mẫu biểu exists, that's the entity type
            # (entity_type is a fixed property of the company)
            winner = max(non_dn, key=non_dn.get)
            entity_type_map[ticker] = winner
        else:
            entity_type_map[ticker] = 'DN'

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
        current_page = None

        for m in PAGE_MARKER_RE.finditer(content):
            page_text = content[last_end:m.start()]
            if current_page is not None:
                pages.append((current_page, page_text))

            current_page = int(m.group(1))
            last_end = m.end()

        if current_page is not None:
            pages.append((current_page, content[last_end:]))

        table_counter = 1

        for page_number, page_text in pages:
            tables = extract_tables_from_text(page_text)

            for table_html, start_pos, end_pos in tables:
                table_id = f"{doc_id}_table_{table_counter}"
                table_counter += 1

                rows = parse_html_table(table_html)
                if not rows:
                    logging.warning(f"Empty rows parsed from table {table_id} in {file_rel_path}")
                    continue

                preceding_text = get_preceding_text(page_text, start_pos, 300)

                record = {
                    'doc_id': doc_id,
                    'table_id': table_id,
                    'page_number': page_number,
                    'preceding_text': preceding_text,
                    'table_html': table_html,
                    'rows': rows,
                    'ticker': ticker,
                    'year': year,
                    'entity_type': entity_type,
                    'consolidated': consolidated,
                    'doc_path': file_rel_path,
                    'folder_type': folder_type,
                }
                results.append(record)

        count += 1
        if count % 100 == 0:
            print(f"Processed {count} files...")

    return results


# ═════════════════════════════════════════════════════════════════════
# Classify table_type
# ═════════════════════════════════════════════════════════════════════

def _extract_note_subtype(text: str) -> str:
    """Extract note subtype from text based on numbered headings."""
    match = re.search(r'\b([0-9A-Z]+(?:\.\d+)?)\s*\.\s+([A-ZÀ-Ỹa-zà-ỹ\s]+)', text)
    if match:
        number = match.group(1).replace('.', '_')
        title = match.group(2).strip()
        title_snake = re.sub(r'\s+', '_', title).lower()
        return f"note_{number}_{title_snake}"
    return "note_unknown"


def classify_tables(raw_tables: list[dict]) -> list[dict]:
    """Classify tables using a 3-tier pipeline."""
    for tbl in raw_tables:
        preceding_text = tbl.get('preceding_text', '')
        rows = tbl.get('rows', [])
        tbl['note_subtype'] = None  # default for non-note tables

        first_2_rows_text = ""
        for r in rows[:2]:
            first_2_rows_text += " ".join(str(cell) for cell in r) + " "

        search_text_t1_t2 = preceding_text + " " + first_2_rows_text

        # Tier 1: Regex mã mẫu biểu
        m = MAU_BIEU_RE.search(search_text_t1_t2)
        if m:
            number = m.group(1).zfill(2)
            suffix = m.group(3).upper()
            if (number, suffix) in MAU_BIEU_TABLE_TYPE:
                tbl['table_type'] = MAU_BIEU_TABLE_TYPE[(number, suffix)]
                tbl['classification_method'] = 'regex_mau_bieu'
                if tbl['table_type'] == 'note_table':
                    tbl['note_subtype'] = _extract_note_subtype(search_text_t1_t2)
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
            tbl['note_subtype'] = _extract_note_subtype(search_text_t1_t2)
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
        tbl['note_subtype'] = _extract_note_subtype(preceding_text)

    return raw_tables


# ═════════════════════════════════════════════════════════════════════
# Normalize & export CSV
# ═════════════════════════════════════════════════════════════════════

def normalize_and_export(classified_tables: list[dict], output_dir: str) -> list[dict]:
    """Normalize classified tables and export to CSV."""
    os.makedirs(output_dir, exist_ok=True)

    for tbl in classified_tables:
        table_type = tbl.get('table_type')
        rows = tbl.get('rows', [])
        if not rows:
            continue

        table_id = tbl.get('table_id', 'unknown_table')
        entity_type = tbl.get('entity_type', 'DN')

        if table_type in ('balance_sheet', 'income_statement', 'cash_flow'):
            label_col = 0
            code_col = -1
            note_col = -1
            val_cols = []

            for row_idx in [0, 1]:
                if row_idx >= len(rows):
                    break
                header = rows[row_idx]
                for c, cell in enumerate(header):
                    cell_lower = str(cell).lower()
                    if 'mã số' in cell_lower or 'mã' in cell_lower:
                        if code_col == -1: code_col = c
                    elif 'thuyết minh' in cell_lower:
                        if note_col == -1: note_col = c
                    elif re.search(r'\b(năm|kỳ|vnd|đ|20\d\d)\b', cell_lower) or '/' in cell_lower:
                        if c not in val_cols and c != label_col and c != code_col and c != note_col:
                            val_cols.append(c)
                if code_col != -1 and val_cols:
                    break

            if code_col == -1: code_col = 1
            if note_col == -1: note_col = 2
            if not val_cols: val_cols = [3, 4] if len(rows[0]) > 4 else [3]

            data = []
            for r in rows[2:]:
                if len(r) <= code_col: continue
                item_code = str(r[code_col]).strip()
                item_label_raw = str(r[label_col]).strip() if label_col < len(r) else ""

                if not item_label_raw and not item_code:
                    continue

                item_label_norm = normalize_item_label(item_label_raw)
                note_ref = str(r[note_col]).strip() if note_col < len(r) else ""

                period_current = parse_vietnamese_number(str(r[val_cols[0]])) if len(val_cols) > 0 and val_cols[0] < len(r) else None
                period_prior = parse_vietnamese_number(str(r[val_cols[1]])) if len(val_cols) > 1 and val_cols[1] < len(r) else None

                data.append({
                    'item_code': item_code,
                    'item_label_raw': item_label_raw,
                    'item_label_norm': item_label_norm,
                    'note_ref': note_ref,
                    'period_current': period_current,
                    'period_prior': period_prior,
                    'unit': 'VND',
                    'entity_type': entity_type
                })

            df = pd.DataFrame(data)
            csv_path = os.path.join(output_dir, f"{table_id}.csv")
            df.to_csv(csv_path, index=False)
            tbl['csv_path'] = f"data/{table_id}.csv"

        elif table_type == 'note_table':
            header = rows[0] if rows else []
            note_subtype = tbl.get('note_subtype', '')
            note_number = ''
            note_title = ''

            m = re.search(r'note_([0-9A-Z_]+)_(.*)', note_subtype)
            if m:
                note_number = m.group(1).replace('_', '.')
                note_title = m.group(2).replace('_', ' ').title()

            data = []
            for r in rows[1:]:
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
                    header_name = str(header[i]).strip() if i < len(header) and str(header[i]).strip() else f"value_{i}"
                    row_data[header_name] = val

                if has_numeric:
                    row_data['note_number'] = note_number
                    row_data['note_title'] = note_title
                    data.append(row_data)

            if data:
                df = pd.DataFrame(data)
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


def build_taxonomy(tables: list[dict], output_dir: str) -> dict[str, list[dict]]:
    project_root = Path(__file__).resolve().parent

    # Store variations: entity_type → item_code → list of item_label_norm
    label_counts = defaultdict(lambda: defaultdict(Counter))
    # Store statement: entity_type → item_code → table_type
    statement_map = defaultdict(dict)

    for t in tables:
        if t.get('table_type') in ('balance_sheet', 'income_statement', 'cash_flow'):
            entity_type = t.get('entity_type')
            table_type = t.get('table_type')
            csv_path = project_root / t.get('csv_path', '')

            if not csv_path.exists():
                continue

            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        item_code = row.get('item_code')
                        item_label_norm = row.get('item_label_norm')
                        if item_code and item_label_norm:
                            label_counts[entity_type][item_code][item_label_norm] += 1
                            statement_map[entity_type][item_code] = table_type
            except Exception:
                pass

    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    result = {}
    universal_concepts = defaultdict(list)

    for entity_type in ('DN', 'TCTD', 'CTCK', 'BH'):
        taxonomy_list = []
        if entity_type in label_counts:
            for item_code, counts in label_counts[entity_type].items():
                most_common_label = counts.most_common(1)[0][0]
                canonical_id = _to_canonical_id(most_common_label)
                aliases = list(counts.keys())
                statement = statement_map[entity_type].get(item_code, '')

                entry = {
                    'item_code': item_code,
                    'canonical_id': canonical_id,
                    'name_vi': most_common_label,
                    'name_en': '',
                    'aliases': aliases,
                    'statement': statement,
                    'entity_type': entity_type,
                    'unit': 'VND',
                    'source': ''
                }
                taxonomy_list.append(entry)

                # add to universal concepts based on item code just as structure
                universal_concepts[item_code].append(entity_type)

        # Save taxonomy
        with open(out_dir_path / f'taxonomy_{entity_type}.json', 'w', encoding='utf-8') as f:
            json.dump(taxonomy_list, f, ensure_ascii=False, indent=2)

        result[entity_type] = taxonomy_list

    # Save universal concepts
    with open(out_dir_path / 'universal_concepts.json', 'w', encoding='utf-8') as f:
        json.dump(universal_concepts, f, ensure_ascii=False, indent=2)

    return result


# ═════════════════════════════════════════════════════════════════════
# Semantic fields + retrieval context
# ═════════════════════════════════════════════════════════════════════

def assign_semantic_fields(tables: list[dict], taxonomy: dict[str, list[dict]]) -> list[dict]:
    project_root = Path(__file__).resolve().parent

    # build quick lookup
    tax_lookup = {}
    for et, entries in taxonomy.items():
        tax_lookup[et] = {e['item_code']: e for e in entries}

    for t in tables:
        table_type = t.get('table_type')
        if table_type in ('balance_sheet', 'income_statement', 'cash_flow'):
            semantic_fields = []
            entity_type = t.get('entity_type')
            csv_path = project_root / t.get('csv_path', '')

            if csv_path.exists():
                try:
                    with open(csv_path, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            item_code = row.get('item_code')
                            if item_code and entity_type in tax_lookup and item_code in tax_lookup[entity_type]:
                                tax_entry = tax_lookup[entity_type][item_code]
                                semantic_fields.append({
                                    'field_id': tax_entry['canonical_id'],
                                    'canonical_name_vi': tax_entry['name_vi'],
                                    'aliases': tax_entry['aliases'],
                                    'item_code': item_code,
                                    'unit': 'VND',
                                })
                except Exception:
                    pass
            t['semantic_fields'] = semantic_fields
        elif table_type == 'note_table':
            t['semantic_fields'] = []

    return tables


def build_retrieval_context(tables: list[dict]) -> list[dict]:
    for t in tables:
        table_type = t.get('table_type')
        ticker = t.get('ticker', '')
        year = t.get('year', '')
        is_consolidated = t.get('consolidated', False)
        folder_type_vi = 'hợp nhất' if is_consolidated else 'riêng'

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

            keywords = list(all_kw)
            table_type_vi = TABLE_TYPE_VI.get(table_type, table_type)
            top_names_str = ', '.join(top_names)

            semantic_summary = f"Báo cáo {table_type_vi} {folder_type_vi} của {ticker} năm {year}, gồm các chỉ tiêu: {top_names_str}"

        elif table_type == 'note_table':
            note_subtype = t.get('note_subtype', '')
            note_title = t.get('note_title', '')

            # note_subtype + note_title words for keywords
            kw_text = f"{note_subtype} {note_title}"
            keywords = list(set([w for w in kw_text.split() if w]))

            subtype_str = note_subtype if note_subtype else 'báo cáo tài chính'
            semantic_summary = f"Thuyết minh {subtype_str} của {ticker} năm {year}"

        folder_type_str = 'consolidated' if is_consolidated else 'separate'

        embedding_text = f"{ticker} | {table_type} | {year} | {folder_type_str} | {semantic_summary}"

        t['retrieval_context'] = {
            'keywords': keywords,
            'semantic_summary': semantic_summary,
            'embedding_text': embedding_text
        }

    return tables


# ═════════════════════════════════════════════════════════════════════
# Generate metadata
# ═════════════════════════════════════════════════════════════════════

def generate_all_metadata(tables: list[dict], output_dir: str) -> None:
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    docs_map = {}
    tables_metadata = []
    page_index = defaultdict(lambda: defaultdict(list))

    for t in tables:
        doc_id = t.get('doc_id')
        if not doc_id:
            continue

        if doc_id not in docs_map:
            docs_map[doc_id] = {
                'doc_id': doc_id,
                'doc_path': t.get('doc_path', ''),
                'ticker': t.get('ticker', ''),
                'year': t.get('year', 0),
                'entity_type': t.get('entity_type', ''),
                'consolidated': t.get('consolidated', False)
            }

        table_id = t.get('table_id', '')

        tm = {
            'table_id': table_id,
            'doc_id': doc_id,
            'page_number': t.get('page_number', 0),
            'table_type': t.get('table_type', ''),
            'csv_path': t.get('csv_path', ''),
            'semantic_fields': t.get('semantic_fields', []),
            'retrieval_context': t.get('retrieval_context', {})
        }
        tables_metadata.append(tm)

        page_number = t.get('page_number', 0)
        page_index[doc_id][str(page_number)].append(table_id)

    docs_metadata = list(docs_map.values())

    with open(out_dir_path / 'docs_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(docs_metadata, f, ensure_ascii=False, indent=2)

    with open(out_dir_path / 'tables_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(tables_metadata, f, ensure_ascii=False, indent=2)

    # convert defaultdict to dict for page_index
    page_index_dict = {doc_id: dict(pages) for doc_id, pages in page_index.items()}

    with open(out_dir_path / 'page_index.json', 'w', encoding='utf-8') as f:
        json.dump(page_index_dict, f, ensure_ascii=False, indent=2)


# ═════════════════════════════════════════════════════════════════════
# Orchestrator
# ═════════════════════════════════════════════════════════════════════

def _ensure_dirs(root: Path) -> None:
    for d in ("data", "metadata", "intermediate", "taxonomy"):
        (root / d).mkdir(parents=True, exist_ok=True)


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

    # ── Stage 2: Classify table_type ───────────────────────────────
    print("═══ Stage 2: Classify table_type ═══")
    classified = classify_tables(raw_tables)
    _save_jsonl(classified, root / "intermediate" / "classified_tables.jsonl")
    counts = Counter(t["table_type"] for t in classified)
    print(f"  {len(classified)} tables classified: {dict(counts)}")

    # ── Stage 3: Normalize & export CSV ────────────────────────────
    print("═══ Stage 3: Normalize & export CSV ═══")
    exported = normalize_and_export(classified, str(root / "data"))
    print(f"  {len(exported)} CSV files written")

    # ── Stage 4: Build taxonomy ────────────────────────────────────
    print("═══ Stage 4: Build taxonomy ═══")
    taxonomy = build_taxonomy(exported, str(root / "taxonomy"))
    total_entries = sum(len(v) for v in taxonomy.values())
    print(f"  {len(taxonomy)} entity-types · {total_entries} entries")

    # ── Stage 5: Semantic fields + retrieval_context (no LLM) ──────
    print("═══ Stage 5: Semantic fields + retrieval context ═══")
    with_sem = assign_semantic_fields(exported, taxonomy)
    with_ctx = build_retrieval_context(with_sem)
    print(f"  Assigned semantic fields & retrieval context to {len(with_ctx)} tables")

    # ── Stage 6: Metadata ──────────────────────────────────────────
    print("═══ Stage 6: Generate metadata ═══")
    generate_all_metadata(with_ctx, str(root / "metadata"))
    print("  docs_metadata.json · tables_metadata.json · page_index.json")

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
