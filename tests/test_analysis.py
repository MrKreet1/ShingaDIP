from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from shingadip.analysis import analyze_operations
from shingadip.data_processing import standardize_operations
from shingadip.documents import DocumentExtraction, _normalize_llm_document_payload


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

    def test_llm_document_payload_normalization(self) -> None:
        payload = {
            "document_number": "INV-2201",
            "document_date": "2026-03-12",
            "seller": "TOO Vision Vendor",
            "amount": "145 000 KZT",
            "description": "Consulting services",
            "extracted_text": "Invoice INV-2201 dated 2026-03-12",
        }
        fields, extracted_text = _normalize_llm_document_payload(payload, "")
        self.assertEqual(fields["document_number"], "INV-2201")
        self.assertEqual(fields["counterparty"], "TOO Vision Vendor")
        self.assertEqual(fields["amount"], 145000.0)
        self.assertEqual(fields["currency"], "KZT")
        self.assertEqual(str(fields["document_date"].date()), "2026-03-12")
        self.assertIn("INV-2201", extracted_text)

    def test_no_documents_do_not_create_missing_document_warning(self) -> None:
        frame = standardize_operations(
            pd.DataFrame(
                [
                    {
                        "Дата операции": "2026-04-01",
                        "Номер документа": "INV-7001",
                        "Контрагент": "TOO Stable Vendor",
                        "Сумма": "120000",
                        "Описание операции": "Обычная поставка",
                    }
                ]
            )
        )
        result = analyze_operations(frame, [])
        row = result.iloc[0]
        self.assertEqual(row["document_check_status"], "NOT_PROVIDED")
        self.assertNotIn("no_primary_document", row["reason_codes"])
        self.assertEqual(row["status"], "OK")

    def test_rare_counterparty_is_not_flagged_without_other_signals(self) -> None:
        rows = []
        for index in range(20):
            rows.append(
                {
                    "operation_date": pd.Timestamp("2026-04-01"),
                    "document_number": f"INV-{7000 + index}",
                    "counterparty": "TOO Base Vendor" if index < 19 else "IP Rare Vendor",
                    "amount": 100000.0,
                    "description": "Обычная поставка",
                    "vat": 12000.0,
                    "document_type": "Invoice",
                    "currency": "KZT",
                    "account": "3310",
                    "responsible_employee": "A",
                    "operation_id": f"OP-{index + 1:04d}",
                    "source_row": index + 2,
                    "missing_required_fields": [],
                    "operation_date_display": "2026-04-01",
                    "amount_display": "100 000.00",
                    "ml_anomaly_flag": False,
                    "ml_anomaly_strength": 0.0,
                }
            )
        frame = pd.DataFrame(rows)

        with patch("shingadip.analysis.detect_ml_anomalies", return_value=frame):
            result = analyze_operations(frame, [])

        row = result.loc[result["counterparty"] == "IP Rare Vendor"].iloc[0]
        self.assertNotIn("atypical_counterparty", row["reason_codes"])
        self.assertEqual(row["status"], "OK")


if __name__ == "__main__":
    unittest.main()
