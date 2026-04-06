from __future__ import annotations

import unittest

import pandas as pd

from shingadip.analysis import analyze_operations
from shingadip.data_processing import standardize_operations
from shingadip.documents import DocumentExtraction


class AnalysisTests(unittest.TestCase):
    def test_standardize_operations_maps_russian_columns(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "Дата операции": "2026-03-01",
                    "Номер документа": "INV-1",
                    "Контрагент": "TOO Test",
                    "Сумма": "100000",
                    "Описание операции": "Закупка",
                }
            ]
        )
        normalized = standardize_operations(frame)
        self.assertEqual(normalized.loc[0, "document_number"], "INV-1")
        self.assertEqual(normalized.loc[0, "counterparty"], "TOO Test")
        self.assertEqual(normalized.loc[0, "amount"], 100000.0)
        self.assertEqual(normalized.loc[0, "missing_required_fields"], [])

    def test_analysis_flags_duplicate_and_missing_fields(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "operation_date": pd.Timestamp("2026-03-01"),
                    "document_number": "INV-1",
                    "counterparty": "TOO Test",
                    "amount": 100000.0,
                    "description": "Обычная покупка",
                    "vat": 12000.0,
                    "document_type": "Invoice",
                    "currency": "KZT",
                    "account": "3310",
                    "responsible_employee": "A",
                    "operation_id": "OP-0001",
                    "source_row": 2,
                    "missing_required_fields": [],
                    "operation_date_display": "2026-03-01",
                    "amount_display": "100 000.00",
                },
                {
                    "operation_date": pd.Timestamp("2026-03-02"),
                    "document_number": "INV-1",
                    "counterparty": "TOO New",
                    "amount": 3000000.0,
                    "description": None,
                    "vat": 0.0,
                    "document_type": "Invoice",
                    "currency": "KZT",
                    "account": "3310",
                    "responsible_employee": "B",
                    "operation_id": "OP-0002",
                    "source_row": 3,
                    "missing_required_fields": ["description"],
                    "operation_date_display": "2026-03-02",
                    "amount_display": "3 000 000.00",
                },
            ]
        )
        result = analyze_operations(frame, [])
        flagged = result.loc[result["operation_id"] == "OP-0002"].iloc[0]
        self.assertIn(flagged["status"], {"WARNING", "RISK"})
        self.assertIn("Номер документа встречается", flagged["reason_details"])
        self.assertIn("Отсутствуют обязательные поля", flagged["reason_details"])

    def test_analysis_detects_document_mismatch(self) -> None:
        frame = standardize_operations(
            pd.DataFrame(
                [
                    {
                        "Дата операции": "2026-03-04",
                        "Номер документа": "INV-1004",
                        "Контрагент": "TOO Vector Parts",
                        "Сумма": "450000",
                        "Описание операции": "Запчасти",
                    }
                ]
            )
        )
        document = DocumentExtraction(
            file_name="INV-1004.pdf",
            stored_path="sample",
            extraction_method="pdf_text",
            extracted_text="",
            document_number="INV-1004",
            document_date=pd.Timestamp("2026-03-04"),
            counterparty="TOO Vector Parts",
            amount=470000.0,
            currency="KZT",
            description="Spare parts",
        )
        result = analyze_operations(frame, [document])
        row = result.iloc[0]
        self.assertEqual(row["document_check_status"], "MISMATCH")
        self.assertIn("Сумма операции не совпадает", row["reason_details"])


if __name__ == "__main__":
    unittest.main()
