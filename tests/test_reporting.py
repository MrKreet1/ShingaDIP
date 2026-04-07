from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from shingadip.reporting import (
    build_audit_conclusion,
    build_counterparty_summary,
    build_document_reconciliation,
    build_reason_summary,
    build_risk_register,
    build_summary,
    save_report_bundle,
)


class ReportingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.results_df = pd.DataFrame(
            [
                {
                    "operation_id": "OP-0001",
                    "operation_date": pd.Timestamp("2026-04-01"),
                    "operation_date_display": "2026-04-01",
                    "document_number": "INV-1001",
                    "counterparty": "TOO Alpha",
                    "amount": 100000.0,
                    "amount_display": "100 000.00",
                    "status": "OK",
                    "risk_score": 10,
                    "reason_codes": "",
                    "reason_details": "Существенных отклонений не обнаружено.",
                    "recommended_action": "Действие не требуется.",
                    "document_check_status": "OK",
                    "matched_document": "INV-1001.pdf",
                    "matched_document_amount": 100000.0,
                    "matched_document_date": "2026-04-01",
                    "matched_document_counterparty": "TOO Alpha",
                    "priority": "LOW",
                    "confidence": "HIGH",
                    "risk_category": "Низкий остаточный риск",
                    "primary_risk_driver": "существенные отклонения не выявлены",
                    "short_ai_comment": "Операция не содержит существенных признаков риска.",
                    "full_ai_comment": "Операция оценена как низкорисковая и не требует дополнительной эскалации.",
                    "ai_comment": "OK",
                },
                {
                    "operation_id": "OP-0002",
                    "operation_date": pd.Timestamp("2026-04-02"),
                    "operation_date_display": "2026-04-02",
                    "document_number": "INV-1002",
                    "counterparty": "TOO Beta",
                    "amount": 240000.0,
                    "amount_display": "240 000.00",
                    "status": "WARNING",
                    "risk_score": 35,
                    "reason_codes": "no_primary_document | ml_anomaly",
                    "reason_details": "Не найден подходящий первичный документ для сверки.",
                    "recommended_action": "Запросить документ.",
                    "document_check_status": "MISSING",
                    "matched_document": None,
                    "matched_document_amount": None,
                    "matched_document_date": None,
                    "matched_document_counterparty": None,
                    "priority": "MEDIUM",
                    "confidence": "MEDIUM",
                    "risk_category": "Документарный риск",
                    "primary_risk_driver": "отсутствие подтверждающего документа",
                    "short_ai_comment": "Операция требует проверки из-за отсутствия документа.",
                    "full_ai_comment": "По операции не найден подтверждающий документ, поэтому требуется выборочная ручная проверка.",
                    "ai_comment": "WARNING",
                },
                {
                    "operation_id": "OP-0003",
                    "operation_date": pd.Timestamp("2026-04-03"),
                    "operation_date_display": "2026-04-03",
                    "document_number": "INV-1003",
                    "counterparty": "TOO Beta",
                    "amount": 500000.0,
                    "amount_display": "500 000.00",
                    "status": "RISK",
                    "risk_score": 75,
                    "reason_codes": "document_amount_mismatch | ml_anomaly",
                    "reason_details": "Сумма операции не совпадает с суммой в первичном документе.",
                    "recommended_action": "Проверить сумму.",
                    "document_check_status": "MISMATCH",
                    "matched_document": "INV-1003.pdf",
                    "matched_document_amount": 460000.0,
                    "matched_document_date": "2026-04-04",
                    "matched_document_counterparty": "TOO Gamma",
                    "priority": "HIGH",
                    "confidence": "HIGH",
                    "risk_category": "Документарный и аномальный риск",
                    "primary_risk_driver": "расхождение суммы с документом",
                    "short_ai_comment": "Операция имеет высокий риск из-за расхождения с документом.",
                    "full_ai_comment": "Операция классифицирована как рискованная, поскольку сумма и реквизиты не совпадают с документом.",
                    "ai_comment": "RISK",
                },
            ]
        )

    def test_build_risk_register_includes_only_warning_and_risk(self) -> None:
        register = build_risk_register(self.results_df)
        self.assertEqual(register["ID операции"].tolist(), ["OP-0003", "OP-0002"])
        self.assertEqual(
            register.columns.tolist(),
            [
                "ID операции",
                "Дата",
                "Номер документа",
                "Контрагент",
                "Сумма",
                "Статус",
                "Риск-балл",
                "Приоритет проверки",
                "Уверенность",
                "Категория риска",
                "Ведущий фактор риска",
                "Причина отклонения",
                "Рекомендуемое действие",
                "Статус сверки документа",
                "Краткий AI-комментарий",
            ],
        )

    def test_build_reason_summary_calculates_metrics(self) -> None:
        summary = build_reason_summary(self.results_df)
        ml_row = summary.loc[summary["Причина"] == "ML-анализ выявил аномалию"].iloc[0]
        self.assertEqual(int(ml_row["Количество случаев"]), 2)
        self.assertEqual(float(ml_row["Доля от общего числа операций, %"]), 66.67)
        self.assertEqual(float(ml_row["Средний риск по причине"]), 55.0)

    def test_build_counterparty_summary_aggregates_by_counterparty(self) -> None:
        summary = build_counterparty_summary(self.results_df)
        beta_row = summary.loc[summary["Контрагент"] == "TOO Beta"].iloc[0]
        self.assertEqual(int(beta_row["Количество операций"]), 2)
        self.assertEqual(float(beta_row["Общая сумма операций"]), 740000.0)
        self.assertEqual(int(beta_row["Количество WARNING"]), 1)
        self.assertEqual(int(beta_row["Количество RISK"]), 1)
        self.assertEqual(int(beta_row["Количество операций без документа"]), 1)
        self.assertEqual(int(beta_row["Количество операций с расхождением документа"]), 1)

    def test_build_document_reconciliation_maps_match_flags(self) -> None:
        reconciliation = build_document_reconciliation(self.results_df)
        op1 = reconciliation.loc[reconciliation["ID операции"] == "OP-0001"].iloc[0]
        op2 = reconciliation.loc[reconciliation["ID операции"] == "OP-0002"].iloc[0]
        op3 = reconciliation.loc[reconciliation["ID операции"] == "OP-0003"].iloc[0]

        self.assertEqual(op1["Статус сверки"], "MATCH")
        self.assertEqual(op1["Совпадение суммы"], "Да")
        self.assertEqual(op1["Совпадение даты"], "Да")
        self.assertEqual(op1["Совпадение контрагента"], "Да")

        self.assertEqual(op2["Статус сверки"], "MISSING")
        self.assertEqual(op2["Совпадение суммы"], "Н/Д")

        self.assertEqual(op3["Статус сверки"], "MISMATCH")
        self.assertEqual(op3["Совпадение суммы"], "Нет")
        self.assertEqual(op3["Совпадение даты"], "Нет")
        self.assertEqual(op3["Совпадение контрагента"], "Нет")

    def test_build_audit_conclusion_contains_problematic_counterparties(self) -> None:
        summary = build_summary(self.results_df, [])
        summary["dataset_comment"] = "Требуется дополнительная проверка."
        report_tables = {
            "risk_register": build_risk_register(self.results_df),
            "reason_summary": build_reason_summary(self.results_df),
            "counterparty_summary": build_counterparty_summary(self.results_df),
            "document_reconciliation": build_document_reconciliation(self.results_df),
        }
        conclusion = build_audit_conclusion(summary, self.results_df, report_tables)
        self.assertIn("TOO Beta", conclusion["problematic_counterparties"])
        self.assertIn("WARNING", conclusion["text"])
        self.assertEqual(conclusion["document_coverage_quality"], "умеренное")
        self.assertEqual(len(conclusion["top_risk_operations"]), 2)

    def test_save_report_bundle_writes_extended_csv_files(self) -> None:
        summary = build_summary(self.results_df, [])
        summary["dataset_comment"] = "Тестовое заключение."
        conclusion = build_audit_conclusion(summary, self.results_df)

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = save_report_bundle(self.results_df, summary, Path(temp_dir), audit_conclusion=conclusion)

            self.assertTrue(Path(paths["csv"]).exists())
            self.assertTrue(Path(paths["json"]).exists())
            self.assertTrue(Path(paths["risk_register"]).exists())
            self.assertTrue(Path(paths["reason_summary"]).exists())
            self.assertTrue(Path(paths["counterparty_summary"]).exists())
            self.assertTrue(Path(paths["document_reconciliation"]).exists())


if __name__ == "__main__":
    unittest.main()
