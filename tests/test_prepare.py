import csv
import json
import sys
import tempfile
import types
import unittest
from collections import Counter
from pathlib import Path


try:
    import pandas  # noqa: F401
except ModuleNotFoundError:
    # Most tests exercise the stdlib-only pipeline logic.  Keep those tests
    # runnable in a bare checkout where requirements.txt is not installed.
    sys.modules['pandas'] = types.ModuleType('pandas')

import prepare


ROOT = Path(__file__).resolve().parents[1]


def _write_statement_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=['item_code', 'item_label_raw', 'item_label_norm'],
        )
        writer.writeheader()
        writer.writerows(rows)


class StatementColumnDetectionTests(unittest.TestCase):
    def test_tctd_without_stt_has_no_fabricated_code(self) -> None:
        rows = [
            ['', 'Thuyết minh', 'Năm nay Triệu đồng', 'Năm trước Triệu đồng'],
            ['Thu nhập lãi và các khoản tương tự', '26', '6.684.626', '6.525.900'],
            ['Chi phí lãi và các chi phí tương tự', '27', '(4.000.000)', '(3.500.000)'],
        ]
        self.assertEqual(
            prepare._detect_statement_columns(rows),
            (0, -1, 1, [2, 3], 1),
        )

    def test_tctd_stt_and_label_are_not_swapped(self) -> None:
        rows = [
            ['STT', 'CHỈ TIÊU', 'Thuyết minh', '31/12/2024', '31/12/2023'],
            ['A', 'TÀI SẢN', '', '', ''],
            ['1', 'Tiền mặt', '5', '315.917', '503.043'],
        ]
        self.assertEqual(
            prepare._detect_statement_columns(rows),
            (1, 0, 2, [3, 4], 1),
        )

    def test_bh_ma_so_precedes_label(self) -> None:
        rows = [
            ['Mã số', 'TÀI SẢN', 'Thuyết minh', 'Năm 2024', 'Năm 2023'],
            ['100', 'A. TÀI SẢN NGẮN HẠN', '', '1.000.000', '900.000'],
        ]
        self.assertEqual(
            prepare._detect_statement_columns(rows),
            (1, 0, 2, [3, 4], 1),
        )

    @unittest.skipUnless(hasattr(prepare.pd, 'DataFrame'), 'pandas is not installed')
    def test_normalize_preserves_first_data_row_and_blank_missing_code(self) -> None:
        table = {
            'table_id': 'tctd_no_code',
            'table_type': 'income_statement',
            'entity_type': 'TCTD',
            'rows': [
                ['', 'Thuyết minh', 'Năm nay Triệu đồng', 'Năm trước Triệu đồng'],
                ['Thu nhập lãi và các khoản tương tự', '26', '6.684.626', '6.525.900'],
                ['Chi phí lãi và các chi phí tương tự', '27', '(4.000.000)', '(3.500.000)'],
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            prepare.normalize_and_export([table], tmp)
            with (Path(tmp) / 'tctd_no_code.csv').open(
                encoding='utf-8', newline=''
            ) as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['item_code'], '')
        self.assertEqual(
            rows[0]['item_label_raw'],
            'Thu nhập lãi và các khoản tương tự',
        )
        self.assertEqual(rows[0]['note_ref'], '26')
        self.assertEqual(float(rows[0]['period_current']), 6_684_626)


class NoteExportTests(unittest.TestCase):
    @unittest.skipUnless(hasattr(prepare.pd, 'DataFrame'), 'pandas is not installed')
    def test_identified_text_only_note_is_exported_as_raw_csv(self) -> None:
        table = {
            'table_id': 'related_parties_text',
            'table_type': 'note_table',
            'note_number': '7.2',
            'note_title': 'Nghiệp vụ và số dư với các bên liên quan',
            'note_subtype': 'note_7_2_nghiep_vu_ben_lien_quan',
            'ticker': 'AAA',
            'year': 2023,
            'consolidated': False,
            'rows': [
                ['Công ty Cổ phần Alpha'],
                ['Công ty TNHH Beta'],
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            prepare.normalize_and_export([table], tmp)
            csv_path = Path(tmp) / 'related_parties_text.csv'
            with csv_path.open(encoding='utf-8', newline='') as handle:
                exported = list(csv.DictReader(handle))

        self.assertEqual(table['csv_path'], 'data/related_parties_text.csv')
        self.assertEqual(len(exported), 2)
        self.assertEqual(exported[0]['row_label_raw'], 'Công ty Cổ phần Alpha')
        self.assertEqual(exported[0]['note_number'], '7.2')

        prepare.build_retrieval_context([table])
        context = table['retrieval_context']
        self.assertIn('Nghiệp vụ và số dư với các bên liên quan', context['keywords'])
        self.assertIn('Công ty Cổ phần Alpha', context['keywords'])
        self.assertIn('Nghiệp vụ và số dư với các bên liên quan', context['semantic_summary'])

    @unittest.skipUnless(hasattr(prepare.pd, 'DataFrame'), 'pandas is not installed')
    def test_meaningful_unknown_text_note_is_exported(self) -> None:
        table = {
            'table_id': 'unknown_but_useful',
            'table_type': 'note_table',
            'note_number': '',
            'note_title': '',
            'note_subtype': 'note_unknown',
            'ticker': 'AAA',
            'year': 2023,
            'consolidated': True,
            'rows': [['Các bên liên quan'], ['Công ty Cổ phần Alpha']],
        }
        with tempfile.TemporaryDirectory() as tmp:
            prepare.normalize_and_export([table], tmp)
            self.assertTrue((Path(tmp) / 'unknown_but_useful.csv').is_file())

        prepare.build_retrieval_context([table])
        self.assertIn('Các bên liên quan', table['retrieval_context']['keywords'])
        self.assertNotIn('note_unknown', table['retrieval_context']['semantic_summary'])

    @unittest.skipUnless(hasattr(prepare.pd, 'DataFrame'), 'pandas is not installed')
    def test_text_note_uses_label_after_stt_as_retrieval_term(self) -> None:
        table = {
            'table_id': 'unknown_with_stt',
            'table_type': 'note_table',
            'note_number': '',
            'note_title': '',
            'note_subtype': 'note_unknown',
            'ticker': 'AAA',
            'year': 2023,
            'consolidated': True,
            'rows': [['STT', 'Bên liên quan'], ['1', 'Công ty mẹ']],
        }
        with tempfile.TemporaryDirectory() as tmp:
            prepare.normalize_and_export([table], tmp)
            self.assertTrue((Path(tmp) / 'unknown_with_stt.csv').is_file())

        prepare.build_retrieval_context([table])
        self.assertIn('Bên liên quan', table['retrieval_context']['keywords'])
        self.assertIn('Công ty mẹ', table['retrieval_context']['keywords'])
        self.assertNotIn('STT', table['retrieval_context']['keywords'])

    @unittest.skipUnless(hasattr(prepare.pd, 'DataFrame'), 'pandas is not installed')
    def test_unknown_page_boilerplate_is_not_exported(self) -> None:
        table = {
            'table_id': 'page_header_only',
            'table_type': 'note_table',
            'note_number': '',
            'note_title': '',
            'note_subtype': 'note_unknown',
            'ticker': 'AAA',
            'year': 2023,
            'consolidated': True,
            'rows': [
                ['Lô CN11, cụm công nghiệp An Đồng, thị trấn Nam Sách, huyện Nam Sách, tỉnh Hải Dương'],
                ['THUYẾT MINH BÁO CÁO TÀI CHÍNH', 'MẪU SỐ B09 - DN/HN'],
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            prepare.normalize_and_export([table], tmp)
            self.assertFalse((Path(tmp) / 'page_header_only.csv').exists())

        self.assertNotIn('csv_path', table)
        prepare.build_retrieval_context([table])
        self.assertEqual(table['retrieval_context']['keywords'], [])
        self.assertIn('note_unknown', table['retrieval_context']['semantic_summary'])


class ClassificationTests(unittest.TestCase):
    def test_fragmented_toc_page_is_detected(self) -> None:
        table = {
            'preceding_text': '',
            'page_has_toc_heading': True,
            'page_table_count': 5,
            'rows': [['Báo cáo lưu chuyển tiền tệ riêng năm 2020', '14 - 16']],
        }
        prepare.classify_tables([table])
        self.assertEqual(table['table_type'], 'table_of_contents')

    def test_toc_is_detected_before_financial_headings(self) -> None:
        table = {
            'preceding_text': 'MỤC LỤC',
            'rows': [
                ['Nội dung', 'Trang'],
                ['Bảng cân đối kế toán', '6 - 9'],
                ['Báo cáo kết quả hoạt động kinh doanh', '10 - 12'],
            ],
        }
        prepare.classify_tables([table])
        self.assertEqual(table['table_type'], 'table_of_contents')
        self.assertEqual(table['classification_method'], 'toc_detection')
        prepare.assign_semantic_fields([table], {})
        prepare.build_retrieval_context([table])
        self.assertEqual(table['semantic_fields'], [])
        self.assertEqual(table['retrieval_context'], {})

    def test_note_heading_prefers_deepest_heading_and_preserves_title(self) -> None:
        text = (
            'V. THÔNG TIN BỔ SUNG\n'
            'V.1. Chi phí trả trước ngắn hạn\n'
        )
        self.assertEqual(
            prepare._extract_note_heading(text),
            ('V.1', 'Chi phí trả trước ngắn hạn'),
        )
        self.assertEqual(
            prepare._extract_note_subtype(text),
            'note_V_1_chi_phí_trả_trước_ngắn_hạn',
        )

    def test_exact_form_code_overrides_fallback_text(self) -> None:
        resolved = prepare._resolve_entity_type(
            Counter({'TCTD': 1}),
            Counter({'BH': 20, 'TCTD': 1}),
        )
        self.assertEqual(resolved, 'TCTD')
        self.assertEqual(
            prepare._resolve_entity_type(Counter(), Counter({'BH': 120, 'TCTD': 2})),
            'BH',
        )
        self.assertEqual(
            prepare._resolve_entity_type(
                Counter({'DN': 1}),
                Counter({'TCTD': 500, 'BH': 500}),
            ),
            'DN',
        )


class SourceFaithfulRepairTests(unittest.TestCase):
    def test_html_parser_expands_rowspan(self) -> None:
        rows = prepare.parse_html_table(
            '<table><tr><td rowspan="2">Chỉ tiêu</td><td>Năm 2025</td></tr>'
            '<tr><td>VND</td></tr></table>'
        )
        self.assertEqual(rows, [['Chỉ tiêu', 'Năm 2025'], ['Chỉ tiêu', 'VND']])

    def test_date_is_never_a_financial_number(self) -> None:
        self.assertIsNone(prepare.parse_vietnamese_number('31/12/2025'))
        self.assertIsNone(prepare.parse_vietnamese_number('31.12.2025'))
        self.assertIsNone(prepare.parse_vietnamese_number('31.12.2025Triệu VND'))
        self.assertEqual(prepare.parse_vietnamese_number('31.122.025'), 31_122_025.0)

    def test_header_recovery_uses_adjacent_table_without_merging(self) -> None:
        header = {
            'table_id': 'header', 'doc_id': 'AAA_doc', 'ticker': 'AAA', 'year': 2024,
            'start_line': 10, 'page_number': 8,
            'rows': [['', '31/12/2024', '31/12/2023'], ['', 'Triệu VND', 'Triệu VND']],
            'repair_proofs': [],
        }
        body = {
            'table_id': 'body', 'doc_id': 'AAA_doc', 'ticker': 'AAA', 'year': 2024,
            'start_line': 12, 'page_number': 9,
            'rows': [['Tiền mặt', '123.456', '100.000'], ['Cộng', '123.456', '100.000']],
            'repair_proofs': [],
        }
        prepare.recover_headers_without_merging([header, body])
        self.assertEqual(header['skip_export_reason'], 'header_only')
        self.assertEqual(body['header_source_table_id'], 'header')
        self.assertEqual(body['effective_header_rows'], header['rows'])
        self.assertEqual(body['rows'][0][0], 'Tiền mặt')

    def test_header_recovery_rejects_more_than_one_page_boundary(self) -> None:
        header = {
            'table_id': 'header', 'doc_id': 'AAA_doc', 'start_line': 10, 'page_number': 8,
            'rows': [['', '31/12/2024', '31/12/2023'], ['', 'VND', 'VND']],
        }
        body = {
            'table_id': 'body', 'doc_id': 'AAA_doc', 'start_line': 12, 'page_number': 10,
            'rows': [['Tiền mặt', '123.456', '100.000'], ['Cộng', '123.456', '100.000']],
        }
        prepare.recover_headers_without_merging([header, body])
        self.assertNotIn('effective_header_rows', body)
        self.assertNotIn('skip_export_reason', header)

    def test_administrative_form_is_not_a_header(self) -> None:
        header = {
            'table_id': 'form', 'doc_id': 'AAA_doc', 'start_line': 10, 'page_number': 8,
            'rows': [['Mẫu số B 09 - DN/HN'], ['Ban hành kèm theo Thông tư']],
            'repair_proofs': [],
        }
        body = {
            'table_id': 'body', 'doc_id': 'AAA_doc', 'start_line': 12, 'page_number': 8,
            'rows': [['Hợp đồng A'], ['Hợp đồng B'], ['Hợp đồng C'], ['Hợp đồng D']],
            'repair_proofs': [],
        }
        prepare.recover_headers_without_merging([header, body])
        self.assertEqual(header['skip_export_reason'], 'administrative_boilerplate')
        self.assertNotIn('effective_header_rows', body)

    def test_gas_style_mapping_is_proven_unreliable(self) -> None:
        rows = [
            ['Mã số', 'Thuyết minh', '2020VND', '2019VND'],
            ['50', 'Lưu chuyển tiền thuần', '208.330.794.876', '(2.169.668.946.861)'],
            ['60', 'Tiền và tương đương tiền đầu năm', '3', '2.528.849.109.924'],
        ]
        label, code, note, values, start = prepare._detect_statement_columns(rows)
        evidence = prepare._mapping_failure_evidence(
            rows, label, code, note, values, start
        )
        self.assertEqual(evidence[0]['mapping_conflict'], 'code_and_label_use_same_column')

    def test_merged_labels_and_amounts_are_proven_unsafe(self) -> None:
        table_html = (
            '<table><tr><td>Mã số</td><td>Chỉ tiêu</td><td>Năm nay</td></tr>'
            '<tr><td>09</td><td rowspan="2">Tăng các khoản phải thu '
            'Giảm hàng tồn kho</td><td>(14.246.868.862) 43.488.477.309</td></tr>'
            '<tr><td>10</td><td>(40.120.058.951) 6.572.055.093</td></tr></table>'
        )
        evidence = prepare._unsafe_merged_cell_evidence({'table_html': table_html})
        self.assertEqual(
            evidence[0]['mapping_conflict'],
            'rowspan_concatenates_multiple_labels_and_amounts',
        )
        self.assertEqual(
            evidence[0]['merged_label_cells'][0]['cell_text'],
            'Tăng các khoản phải thu Giảm hàng tồn kho',
        )

    def test_ordinary_span_is_not_marked_unsafe(self) -> None:
        table_html = (
            '<table><tr><td rowspan="2">Chỉ tiêu</td><td>Năm 2025</td></tr>'
            '<tr><td>VND</td></tr></table>'
        )
        self.assertEqual(prepare._unsafe_merged_cell_evidence({'table_html': table_html}), [])

    @unittest.skipUnless(hasattr(prepare.pd, 'DataFrame'), 'pandas is not installed')
    def test_note_mixed_text_uses_raw_fallback(self) -> None:
        table = {
            'table_id': 'mixed_note', 'table_type': 'note_table',
            'note_number': '5.1', 'note_title': 'Bên liên quan',
            'rows': [
                ['Bên liên quan', 'Năm nay', 'Năm trước'],
                ['Công ty Alpha', 'Mua hàng', '1.234.000'],
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            prepare.normalize_and_export([table], tmp)
            with (Path(tmp) / 'mixed_note.csv').open(encoding='utf-8', newline='') as handle:
                exported = list(csv.DictReader(handle))
        self.assertTrue(table['raw_fallback'])
        self.assertEqual(exported[1]['value_1_raw'], 'Mua hàng')
        self.assertEqual(exported[1]['value_2_raw'], '1.234.000')

    @unittest.skipUnless(hasattr(prepare.pd, 'DataFrame'), 'pandas is not installed')
    def test_statement_unit_comes_from_header(self) -> None:
        table = {
            'table_id': 'million_unit', 'table_type': 'balance_sheet', 'entity_type': 'TCTD',
            'rows': [
                ['', '31/12/2024 Triệu VND', '31/12/2023 Triệu VND'],
                ['Tiền mặt', '328.595', '232.410'],
                ['Cộng', '328.595', '232.410'],
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            prepare.normalize_and_export([table], tmp)
            with (Path(tmp) / 'million_unit.csv').open(encoding='utf-8', newline='') as handle:
                exported = list(csv.DictReader(handle))
        self.assertEqual(exported[0]['unit'], 'million_VND')

    @unittest.skipUnless(hasattr(prepare.pd, 'DataFrame'), 'pandas is not installed')
    def test_statement_percent_uses_existing_unit_column(self) -> None:
        table = {
            'table_id': 'percent_row', 'table_type': 'income_statement', 'entity_type': 'DN',
            'rows': [
                ['Chỉ tiêu', 'Năm 2024 VND', 'Năm 2023 VND'],
                ['Tỷ lệ trích quỹ', '15%', '12%'],
                ['Lợi nhuận', '1.000.000', '900.000'],
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            prepare.normalize_and_export([table], tmp)
            with (Path(tmp) / 'percent_row.csv').open(encoding='utf-8', newline='') as handle:
                exported = list(csv.DictReader(handle))
        self.assertEqual(exported[0]['period_current'], '15.0')
        self.assertEqual(exported[0]['unit'], 'percent')
        self.assertEqual(exported[1]['unit'], 'VND')

    @unittest.skipUnless(hasattr(prepare.pd, 'DataFrame'), 'pandas is not installed')
    def test_mixed_percent_and_money_rows_do_not_leak_unit(self) -> None:
        table = {
            'table_id': 'aaa_2015_table_37_style',
            'table_type': 'income_statement', 'entity_type': 'DN',
            'repair_proofs': [],
            'rows': [
                ['', 'Năm 2015 VND', 'Năm 2014 VND'],
                ['Lợi nhuận phân bổ', '40.015.870.985', '46.963.561.091'],
                ['Tỷ lệ trích quỹ', '15%', '15%'],
                ['Lợi nhuận để tính lãi', '31.802.089.144', '41.523.039.919'],
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            prepare.normalize_and_export([table], tmp)
            with (Path(tmp) / 'aaa_2015_table_37_style.csv').open(
                encoding='utf-8', newline=''
            ) as handle:
                exported = list(csv.DictReader(handle))
        self.assertEqual([row['unit'] for row in exported], ['VND', 'percent', 'VND'])
        proof = next(
            proof for proof in table['repair_proofs']
            if proof['rule'] == 'unit_from_ocr_evidence'
        )
        self.assertEqual(proof['evidence']['percent_rows'][0]['item_label_raw'], 'Tỷ lệ trích quỹ')

    @unittest.skipUnless(hasattr(prepare.pd, 'DataFrame'), 'pandas is not installed')
    def test_interest_rate_wording_does_not_override_explicit_money_unit(self) -> None:
        table = {
            'table_id': 'acb_2023_table_104_style',
            'table_type': 'balance_sheet', 'entity_type': 'TCTD',
            'rows': [
                ['Tại ngày', 'Không chịu lãi suất', 'Từ 1 tháng đến 3 tháng', 'Đơn vị: Triệu VND'],
                ['Tiền gửi tại Ngân hàng Nhà nước', '13.657.531', '-', '13.657.531'],
                ['Tổng tài sản', '613.317.275', '-', '613.317.275'],
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            prepare.normalize_and_export([table], tmp)
            with (Path(tmp) / 'acb_2023_table_104_style.csv').open(
                encoding='utf-8', newline=''
            ) as handle:
                exported = list(csv.DictReader(handle))
        self.assertTrue(exported)
        self.assertTrue(all(row['unit'] == 'million_VND' for row in exported))

    def test_percent_header_is_explicit_evidence_without_percent_value(self) -> None:
        evidence = prepare._percent_unit_evidence_for_row(
            ['Tỷ lệ an toàn', '8.0', '9.0'],
            [1, 2],
            [['Chỉ tiêu', 'Lãi suất (%/năm)', 'Lãi suất (%/năm)']],
        )
        self.assertEqual(evidence[0]['source'], 'value_header')

    @unittest.skipUnless(hasattr(prepare.pd, 'DataFrame'), 'pandas is not installed')
    def test_unreliable_statement_mapping_uses_raw_fallback(self) -> None:
        table = {
            'table_id': 'gas_style', 'table_type': 'cash_flow', 'entity_type': 'DN',
            'repair_proofs': [],
            'rows': [
                ['Mã số', 'Thuyết minh', '2020VND', '2019VND'],
                ['50', 'Lưu chuyển tiền thuần', '208.330.794.876', '(2.169.668.946.861)'],
                ['60', 'Tiền và tương đương tiền đầu năm', '3', '2.528.849.109.924'],
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            prepare.normalize_and_export([table], tmp)
            with (Path(tmp) / 'gas_style.csv').open(encoding='utf-8', newline='') as handle:
                exported = list(csv.DictReader(handle))
        self.assertTrue(table['raw_fallback'])
        self.assertEqual(exported[2]['row_label_raw'], '60')
        self.assertEqual(exported[2]['value_1_raw'], 'Tiền và tương đương tiền đầu năm')
        self.assertEqual(exported[2]['value_3_raw'], '2.528.849.109.924')

    @unittest.skipUnless(hasattr(prepare.pd, 'DataFrame'), 'pandas is not installed')
    def test_unsafe_merged_statement_uses_raw_fallback_without_splitting(self) -> None:
        table_html = (
            '<table><tr><td>Mã số</td><td>Chỉ tiêu</td><td>Năm nay</td><td>Năm trước</td></tr>'
            '<tr><td>09</td><td rowspan="2">Tăng các khoản phải thu '
            'Giảm hàng tồn kho</td><td>(14.246.868.862) 43.488.477.309</td>'
            '<td>(40.120.058.951) 6.572.055.093</td></tr>'
            '<tr><td>10</td><td></td><td></td></tr></table>'
        )
        table = {
            'table_id': 'vif_unsafe', 'table_type': 'cash_flow', 'entity_type': 'DN',
            'table_html': table_html, 'repair_proofs': [],
            'rows': prepare.parse_html_table(table_html),
        }
        with tempfile.TemporaryDirectory() as tmp:
            prepare.normalize_and_export([table], tmp)
            with (Path(tmp) / 'vif_unsafe.csv').open(encoding='utf-8', newline='') as handle:
                exported = list(csv.DictReader(handle))
        self.assertTrue(table['raw_fallback'])
        self.assertTrue(table['unsafe_merged_cell'])
        self.assertEqual(exported[1]['value_2_raw'], '(14.246.868.862) 43.488.477.309')
        proof = table['repair_proofs'][-1]
        self.assertEqual(proof['rule'], 'unsafe_merged_cell_mapping')
        self.assertEqual(proof['action'], 'raw_fallback')


class TaxonomyTests(unittest.TestCase):
    def test_taxonomy_and_semantics_are_scoped_by_statement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            balance_csv = tmp_path / 'balance.csv'
            income_csv = tmp_path / 'income.csv'
            _write_statement_csv(balance_csv, [{
                'item_code': '10',
                'item_label_raw': 'Tiền và tương đương tiền',
                'item_label_norm': 'Tiền và tương đương tiền',
            }])
            _write_statement_csv(income_csv, [{
                'item_code': '10',
                'item_label_raw': 'Doanh thu bán hàng',
                'item_label_norm': 'Doanh thu bán hàng',
            }])
            tables = [
                {
                    'entity_type': 'DN',
                    'table_type': 'balance_sheet',
                    'csv_path': str(balance_csv),
                },
                {
                    'entity_type': 'DN',
                    'table_type': 'income_statement',
                    'csv_path': str(income_csv),
                },
            ]
            taxonomy = prepare.build_taxonomy(tables, str(tmp_path / 'taxonomy'))
            code_10 = [entry for entry in taxonomy['DN'] if entry['item_code'] == '10']
            self.assertEqual(len(code_10), 2)
            self.assertEqual(
                {entry['statement'] for entry in code_10},
                {'balance_sheet', 'income_statement'},
            )
            for entry in code_10:
                self.assertEqual(entry['aliases'], [entry['name_vi']])

            prepare.assign_semantic_fields(tables, taxonomy)
            self.assertEqual(
                tables[0]['semantic_fields'][0]['canonical_name_vi'],
                'Tiền và tương đương tiền',
            )
            self.assertEqual(
                tables[1]['semantic_fields'][0]['canonical_name_vi'],
                'Doanh thu bán hàng',
            )

            universal = json.loads(
                (tmp_path / 'taxonomy' / 'universal_concepts.json').read_text(
                    encoding='utf-8'
                )
            )
            self.assertIn('balance_sheet:10', universal)
            self.assertIn('income_statement:10', universal)

    def test_repeated_tctd_stt_is_split_and_matched_by_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            statement_csv = tmp_path / 'tctd.csv'
            _write_statement_csv(statement_csv, [
                {
                    'item_code': '1',
                    'item_label_raw': 'Tiền mặt',
                    'item_label_norm': 'Tiền mặt',
                },
                {
                    'item_code': '1',
                    'item_label_raw': 'Tiền gửi khách hàng',
                    'item_label_norm': 'Tiền gửi khách hàng',
                },
            ])
            tables = [{
                'entity_type': 'TCTD',
                'table_type': 'balance_sheet',
                'csv_path': str(statement_csv),
            }]
            taxonomy = prepare.build_taxonomy(tables, str(tmp_path / 'taxonomy'))
            entries = [
                entry for entry in taxonomy['TCTD']
                if entry['statement'] == 'balance_sheet' and entry['item_code'] == '1'
            ]
            self.assertEqual(len(entries), 2)

            prepare.assign_semantic_fields(tables, taxonomy)
            self.assertEqual(
                {field['canonical_name_vi'] for field in tables[0]['semantic_fields']},
                {'Tiền mặt', 'Tiền gửi khách hàng'},
            )

    def test_taxonomy_rejects_amount_like_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            statement_csv = tmp_path / 'cashflow.csv'
            _write_statement_csv(statement_csv, [
                {
                    'item_code': '60',
                    'item_label_raw': 'Tiền và tương đương tiền đầu năm',
                    'item_label_norm': 'Tiền và tương đương tiền đầu năm',
                },
                {
                    'item_code': '60',
                    'item_label_raw': '2.528.849.109.924',
                    'item_label_norm': '528.849.109.924',
                },
            ])
            table = {
                'table_id': 'gas_style', 'doc_id': 'GAS_doc', 'ticker': 'GAS',
                'year': 2020, 'start_line': 283, 'entity_type': 'DN',
                'table_type': 'cash_flow', 'csv_path': str(statement_csv),
            }
            taxonomy_dir = tmp_path / 'taxonomy'
            taxonomy = prepare.build_taxonomy([table], str(taxonomy_dir))
            entry = next(
                item for item in taxonomy['DN']
                if item['statement'] == 'cash_flow' and item['item_code'] == '60'
            )
            self.assertEqual(entry['aliases'], ['Tiền và tương đương tiền đầu năm'])
            exclusions = (taxonomy_dir / 'taxonomy_exclusions.jsonl').read_text(
                encoding='utf-8'
            )
            self.assertIn('528.849.109.924', exclusions)

    def test_taxonomy_excludes_unsafe_raw_fallback_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            taxonomy_dir = Path(tmp) / 'taxonomy'
            taxonomy = prepare.build_taxonomy([{
                'table_id': 'VIF_2025_table_5', 'doc_id': 'VIF_doc',
                'ticker': 'VIF', 'year': 2025, 'start_line': 406,
                'entity_type': 'DN', 'table_type': 'cash_flow',
                'raw_fallback': True, 'unsafe_merged_cell': True,
            }], str(taxonomy_dir))
            exclusions = (taxonomy_dir / 'taxonomy_exclusions.jsonl').read_text(
                encoding='utf-8'
            )
        self.assertEqual(taxonomy['DN'], [])
        self.assertIn('unsafe_merged_cell_raw_fallback', exclusions)
        self.assertIn('VIF_2025_table_5', exclusions)


class MetadataTests(unittest.TestCase):
    def test_tables_metadata_contains_filter_and_audit_fields(self) -> None:
        table = {
            'table_id': 'AAA_2024_table_1',
            'doc_id': 'AAA_financial_statements_2024_separate',
            'doc_path': 'AAA/2024/doc/doc_extracted.txt',
            'start_line': 321,
            'ticker': 'AAA',
            'year': 2024,
            'folder_type': 'separate',
            'entity_type': 'DN',
            'consolidated': False,
            'table_type': 'balance_sheet',
            'csv_path': 'data/AAA_2024_table_1.csv',
            'semantic_fields': [],
            'retrieval_context': {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            prepare.generate_all_metadata(
                [table],
                str(output_dir),
                {'AAA': 'Công ty cổ phần An Phát'},
            )
            tables_metadata = json.loads(
                (output_dir / 'tables_metadata.json').read_text(encoding='utf-8')
            )
            self.assertFalse((output_dir / 'page_index.json').exists())

        metadata = tables_metadata[0]
        self.assertNotIn('page_number', metadata)
        self.assertEqual(metadata['start_line'], 321)
        self.assertEqual(metadata['ticker'], 'AAA')
        self.assertEqual(metadata['company_name'], 'Công ty cổ phần An Phát')
        self.assertEqual(metadata['year'], 2024)
        self.assertEqual(metadata['report_type'], 'separate')

    def test_metadata_excludes_toc_and_tables_without_csv(self) -> None:
        tables = [
            {
                'table_id': 'AAA_toc',
                'doc_id': 'AAA_doc',
                'table_type': 'table_of_contents',
                'folder_type': 'consolidated',
                'csv_path': '',
            },
            {
                'table_id': 'AAA_unexported',
                'doc_id': 'AAA_doc',
                'table_type': 'note_table',
                'folder_type': 'consolidated',
                'csv_path': '',
            },
            {
                'table_id': 'AAA_statement',
                'doc_id': 'AAA_doc',
                'table_type': 'balance_sheet',
                'folder_type': 'consolidated',
                'csv_path': 'data/AAA_statement.csv',
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            prepare.generate_all_metadata(tables, tmp)
            metadata = json.loads(
                (Path(tmp) / 'tables_metadata.json').read_text(encoding='utf-8')
            )
        self.assertEqual([row['table_id'] for row in metadata], ['AAA_statement'])

    def test_inventory_drives_document_metadata_for_documents_without_tables(self) -> None:
        tables = [{
            'table_id': 'AAA_statement',
            'doc_id': 'AAA_doc',
            'table_type': 'balance_sheet',
            'folder_type': 'consolidated',
            'csv_path': 'data/AAA_statement.csv',
        }]
        inventory = [
            {
                'doc_id': 'AAA_doc',
                'file_path': 'AAA/2024/AAA_doc/AAA_doc_extracted.txt',
                'ticker': 'AAA',
                'year': 2024,
                'folder_type': 'consolidated',
            },
            {
                'doc_id': 'AAA_empty_doc',
                'file_path': 'AAA/2024/AAA_empty_doc/AAA_empty_doc_extracted.txt',
                'ticker': 'AAA',
                'year': 2024,
                'folder_type': 'other',
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            prepare.generate_all_metadata(
                tables,
                tmp,
                inventory=inventory,
                entity_type_map={'AAA': 'DN'},
            )
            docs = json.loads(
                (Path(tmp) / 'docs_metadata.json').read_text(encoding='utf-8')
            )
        self.assertEqual(len(docs), 2)
        empty_doc = next(row for row in docs if row['doc_id'] == 'AAA_empty_doc')
        self.assertEqual(empty_doc['report_type'], 'other')
        self.assertEqual(empty_doc['table_count'], 0)

    def test_retrieval_context_preserves_non_binary_report_type(self) -> None:
        table = {
            'table_type': 'note_table',
            'folder_type': 'aggregated',
            'ticker': 'AAA',
            'year': 2024,
            'note_title': 'Doanh thu',
            'note_number': '1',
            'semantic_fields': [],
        }
        prepare.build_retrieval_context([table])
        self.assertIn('| aggregated |', table['retrieval_context']['embedding_text'])


class ViFinQARegressionTests(unittest.TestCase):
    def test_real_bvh_toc_and_statement_layout(self) -> None:
        files = list(
            (ROOT / 'ViFinQA' / 'financial_statements' / 'BVH' / '2024').rglob(
                '*_extracted.txt'
            )
        )
        if not files:
            self.skipTest('ViFinQA BVH sample is not available')

        content = files[0].read_text(encoding='utf-8')
        html_tables = prepare.extract_tables_from_text(content)
        toc_html, toc_start, _ = html_tables[0]
        toc = {
            'preceding_text': prepare.get_preceding_text(content, toc_start),
            'rows': prepare.parse_html_table(toc_html),
        }
        prepare.classify_tables([toc])
        self.assertEqual(toc['table_type'], 'table_of_contents')

        statement_rows = next(
            rows
            for html, _, _ in html_tables
            for rows in [prepare.parse_html_table(html)]
            if rows and 'Mã số' in rows[0]
        )
        self.assertEqual(
            prepare._detect_statement_columns(statement_rows),
            (1, 0, 2, [3, 4], 1),
        )

    @unittest.skipUnless(hasattr(prepare.pd, 'DataFrame'), 'pandas is not installed')
    def test_real_tctd_csv_keeps_stt_and_label_in_their_columns(self) -> None:
        files = list(
            (ROOT / 'ViFinQA' / 'financial_statements' / 'BID' / '2015').rglob(
                '*_extracted.txt'
            )
        )
        if not files:
            self.skipTest('ViFinQA BID sample is not available')

        content = files[0].read_text(encoding='utf-8')
        statement_rows = next(
            rows
            for html, _, _ in prepare.extract_tables_from_text(content)
            for rows in [prepare.parse_html_table(html)]
            if rows and rows[0][:3] == ['STT', 'CHỈ TIÊU', 'Thuyết minh']
        )
        table = {
            'table_id': 'bid_real_statement',
            'table_type': 'balance_sheet',
            'entity_type': 'TCTD',
            'rows': statement_rows,
        }
        with tempfile.TemporaryDirectory() as tmp:
            prepare.normalize_and_export([table], tmp)
            with (Path(tmp) / 'bid_real_statement.csv').open(
                encoding='utf-8', newline=''
            ) as handle:
                exported = list(csv.DictReader(handle))

        first_coded_row = next(row for row in exported if row['item_code'])
        self.assertEqual(first_coded_row['item_code'], statement_rows[1][0])
        self.assertEqual(first_coded_row['item_label_raw'], statement_rows[1][1])
        self.assertNotEqual(
            first_coded_row['item_code'],
            first_coded_row['item_label_raw'],
        )

    @unittest.skipUnless(hasattr(prepare.pd, 'DataFrame'), 'pandas is not installed')
    def test_real_source_start_line_points_to_table_tag(self) -> None:
        fs_root = ROOT / 'ViFinQA' / 'financial_statements'
        source_path = next(
            (fs_root / 'BVH' / '2024').rglob('*_extracted.txt'),
            None,
        )
        if source_path is None:
            self.skipTest('ViFinQA BVH sample is not available')

        doc_id = source_path.parent.name
        inventory = prepare.pd.DataFrame([{
            'file_path': source_path.relative_to(fs_root).as_posix(),
            'doc_id': doc_id,
            'ticker': 'BVH',
            'year': 2024,
            'folder_type': 'consolidated',
        }])
        parsed = prepare.parse_all_files(
            inventory,
            {'BVH': 'BH'},
            str(fs_root),
        )
        first_table = parsed[0]
        self.assertNotIn('page_number', first_table)
        source_lines = source_path.read_text(encoding='utf-8').splitlines()
        self.assertTrue(
            source_lines[first_table['start_line'] - 1].lstrip().startswith('<table')
        )


if __name__ == '__main__':
    unittest.main()
