from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from shingadip.data_processing import load_operations_source, read_operations_file


class _UploadedFileStub:
    def __init__(self, name: str, payload: bytes = b"test") -> None:
        self.name = name
        self._payload = payload

    def getvalue(self) -> bytes:
        return self._payload


class DataProcessingTests(unittest.TestCase):
    def test_read_operations_file_supports_xls_uploads(self) -> None:
        uploaded = _UploadedFileStub("operations_2025.xls")
        source_df = pd.DataFrame(
            [
                ["Дата операции", "Номер документа", "Контрагент", "Сумма", "Описание операции"],
                ["2026-04-01", "INV-9001", "TOO Demo", "150000", "Поставка"],
            ]
        )

        with patch("shingadip.data_processing.pd.read_excel", return_value=source_df) as read_excel_mock:
            result = read_operations_file(uploaded)

        self.assertEqual(result.loc[0, "counterparty"], "TOO Demo")
        self.assertEqual(result.loc[0, "amount"], 150000.0)
        self.assertEqual(read_excel_mock.call_args.kwargs.get("engine"), "xlrd")

    def test_load_operations_source_detects_one_c_export_and_combines_columns(self) -> None:
        uploaded = _UploadedFileStub("journal_1c.xlsx")
        source_df = pd.DataFrame(
            [
                ["Отчет по операциям 1С", None, None, None, None, None, None],
                ["Период", "Регистратор", "Содержание", "Сумма", "Счет Дт", "Счет Кт", "Контрагент"],
                ["2026-03-01", "Поступление товаров INV-1001", "Закупка цемента", "245000", "1310", "3310", "TOO Alpha"],
                ["2026-03-02", "Поступление товаров INV-1002", "Транспортные услуги", "98000", "7210", "3310", "IP Logistic"],
            ]
        )

        with patch("shingadip.data_processing.pd.read_excel", return_value=source_df):
            result = load_operations_source(uploaded, import_mode="auto")

        self.assertEqual(result.detected_mode, "one_c_export")
        self.assertEqual(result.resolved_mode, "one_c_export")
        self.assertEqual(result.dataframe.loc[0, "document_number"], "Поступление товаров INV-1001")
        self.assertEqual(result.dataframe.loc[0, "account"], "1310 / 3310")
        self.assertEqual(result.dataframe.loc[0, "counterparty"], "TOO Alpha")

    def test_load_operations_source_detects_osv_and_marks_summary(self) -> None:
        uploaded = _UploadedFileStub("ОСВ 2025г.xls")
        source_df = pd.DataFrame(
            [
                ['TOO "Жихан-Бетон"', None, None, None, None, None, None],
                ["Оборотно-сальдовая ведомость за 2025 г.", None, None, None, None, None, None],
                [None, None, None, None, None, None, None],
                ["Выводимые данные: БУ", None, None, None, None, None, None],
                [None, None, None, None, None, None, None],
                ["Счет, Наименование", "Сальдо на начало периода", None, "Обороты за период", None, "Сальдо на конец периода", None],
                [None, "Дебет", "Кредит", "Дебет", "Кредит", "Дебет", "Кредит"],
                ["1000, Денежные средства", "410342.34", None, "373566806.94", "372972403.93", "1004745.35", None],
                ["1200, Краткосрочная дебиторская задолженность", "26403663.49", None, "414853325.8", "418885901.3", "22371087.99", None],
            ]
        )

        with patch("shingadip.data_processing.pd.read_excel", return_value=source_df):
            result = load_operations_source(uploaded, import_mode="auto")

        self.assertEqual(result.detected_mode, "osv_summary")
        self.assertTrue(result.source_is_summary)
        self.assertEqual(result.period_label, "2025 г")
        self.assertEqual(result.dataframe.loc[0, "document_type"], "ОСВ")
        self.assertEqual(result.dataframe.loc[0, "account"], "1000")
        self.assertEqual(result.dataframe.loc[0, "counterparty"], "Денежные средства")

    def test_load_operations_source_applies_manual_mapping(self) -> None:
        uploaded = _UploadedFileStub("custom.csv")
        source_df = pd.DataFrame(
            [
                ["Дата платежа", "Док №", "Партнер", "Итого", "Комментарий"],
                ["2026-04-10", "PAY-100", "TOO Manual", "88000", "Оплата услуг"],
            ]
        )

        with patch("shingadip.data_processing._read_csv_from_bytes", return_value=source_df):
            result = load_operations_source(
                uploaded,
                import_mode="operations_table",
                column_mapping={
                    "operation_date": "Дата платежа",
                    "document_number": "Док №",
                    "counterparty": "Партнер",
                    "amount": "Итого",
                    "description": "Комментарий",
                },
            )

        row = result.dataframe.iloc[0]
        self.assertEqual(row["document_number"], "PAY-100")
        self.assertEqual(row["counterparty"], "TOO Manual")
        self.assertEqual(row["amount"], 88000.0)


if __name__ == "__main__":
    unittest.main()
