from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from PIL import Image

from shingadip.ai import AISettings, generate_row_commentary, resolve_lightonocr_model_identifier
from shingadip.analysis import analyze_operations
from shingadip.data_processing import standardize_operations
from shingadip.documents import DocumentExtraction, _normalize_llm_document_payload, extract_document


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

    def test_generate_row_commentary_adds_machine_and_ai_layers(self) -> None:
        results_df = pd.DataFrame(
            [
                {
                    "operation_id": "OP-9001",
                    "status": "RISK",
                    "risk_score": 84,
                    "amount": 2750000.0,
                    "amount_display": "2 750 000.00",
                    "counterparty": "IP Rare Vendor 1",
                    "document_number": "INV-9001",
                    "document_check_status": "MISSING",
                    "reason_codes": "missing_required_fields | amount_outlier | atypical_counterparty | no_primary_document | ml_anomaly",
                    "reason_details": "Отсутствует описание операции, сумма выше типового диапазона, документ не найден.",
                    "description": None,
                    "missing_required_fields": ["description"],
                    "matched_document": None,
                }
            ]
        )

        enriched = generate_row_commentary(results_df, AISettings(use_lm_studio=False))
        row = enriched.iloc[0]

        self.assertEqual(row["priority"], "HIGH")
        self.assertEqual(row["confidence"], "HIGH")
        self.assertTrue(str(row["risk_category"]).strip())
        self.assertTrue(str(row["primary_risk_driver"]).strip())
        self.assertTrue(str(row["machine_payload_json"]).strip().startswith("{"))
        self.assertTrue(str(row["short_ai_comment"]).strip())
        self.assertTrue(str(row["full_ai_comment"]).strip())
        self.assertTrue(str(row["recommended_action"]).strip())
        self.assertEqual(row["ai_comment"], row["short_ai_comment"])

    def test_extract_document_lightonocr_image_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "INV-3301.png"
            Image.new("RGB", (640, 420), "white").save(image_path)
            settings = AISettings(
                use_document_model=True,
                document_analysis_mode="lightonocr",
                lightonocr_endpoint="http://127.0.0.1:8080/v1/chat/completions",
            )

            with patch(
                "shingadip.documents.request_document_model_completion",
                return_value=(
                    "Invoice INV-3301\n"
                    "Date: 2026-04-05\n"
                    "Seller: TOO LightOn Trade\n"
                    "Amount: 125 000 KZT\n"
                    "Description: Office supplies"
                ),
            ):
                extracted = extract_document(image_path, settings)

        self.assertEqual(extracted.extraction_method, "lightonocr")
        self.assertEqual(extracted.document_number, "INV-3301")
        self.assertEqual(extracted.counterparty, "TOO LightOn Trade")
        self.assertEqual(extracted.amount, 125000.0)
        self.assertEqual(extracted.currency, "KZT")

    def test_extract_document_lightonocr_fallback_to_pdf_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "INV-4401.pdf"
            pdf_path.write_bytes(b"%PDF-1.4 test")
            settings = AISettings(
                use_document_model=True,
                document_analysis_mode="lightonocr",
            )

            with patch(
                "shingadip.documents._extract_text_from_pdf",
                return_value=(
                    "Invoice INV-4401\n"
                    "Date: 2026-04-07\n"
                    "Seller: TOO PDF Supplier\n"
                    "Amount: 98 000 KZT\n"
                    "Description: Stationery"
                ),
            ), patch("shingadip.documents._render_pdf_pages", return_value=[]):
                extracted = extract_document(pdf_path, settings)

        self.assertEqual(extracted.extraction_method, "pdf_text")
        self.assertEqual(extracted.document_number, "INV-4401")
        self.assertEqual(extracted.amount, 98000.0)
        self.assertIn("LightOnOCR", " ".join(extracted.warnings))

    def test_lightonocr_output_is_used_in_operation_matching(self) -> None:
        operations = standardize_operations(
            pd.DataFrame(
                [
                    {
                        "Дата операции": "2026-04-05",
                        "Номер документа": "INV-5501",
                        "Контрагент": "TOO Match Vendor",
                        "Сумма": "450000",
                        "Описание операции": "Поставка комплектующих",
                    }
                ]
            )
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "INV-5501.png"
            Image.new("RGB", (720, 480), "white").save(image_path)
            settings = AISettings(
                use_document_model=True,
                document_analysis_mode="lightonocr",
            )

            with patch(
                "shingadip.documents.request_document_model_completion",
                return_value=(
                    "Invoice INV-5501\n"
                    "Date: 2026-04-05\n"
                    "Seller: TOO Match Vendor\n"
                    "Amount: 470 000 KZT\n"
                    "Description: Поставка комплектующих"
                ),
            ):
                extracted = extract_document(image_path, settings)

        result = analyze_operations(operations, [extracted])
        row = result.iloc[0]
        self.assertEqual(row["document_check_status"], "MISMATCH")
        self.assertIn("Сумма операции не совпадает", row["reason_details"])

    def test_resolve_lightonocr_model_identifier_supports_lm_studio_alias(self) -> None:
        resolved = resolve_lightonocr_model_identifier(
            ["qwen2.5-7b-instruct", "lightonocr-2-1b", "qwen2.5-vl-7b-instruct"]
        )
        self.assertEqual(resolved, "lightonocr-2-1b")


if __name__ == "__main__":
    unittest.main()
