from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from shingadip.documents import DocumentExtraction


def build_summary(results_df: pd.DataFrame, documents: list[DocumentExtraction]) -> dict[str, object]:
    total_operations = len(results_df)
    ok_count = int((results_df["status"] == "OK").sum())
    warning_count = int((results_df["status"] == "WARNING").sum())
    risk_count = int((results_df["status"] == "RISK").sum())
    average_risk_score = round(float(results_df["risk_score"].mean() if total_operations else 0.0), 2)
    matched_count = int(results_df["matched_document"].notna().sum())
    top_reasons = summarize_reasons(results_df)
    if top_reasons.empty:
        top_reason_text = "существенные отклонения не выявлены"
    else:
        top_reason_text = ", ".join(top_reasons["Причина"].head(3).tolist())

    return {
        "total_operations": total_operations,
        "ok_count": ok_count,
        "warning_count": warning_count,
        "risk_count": risk_count,
        "average_risk_score": average_risk_score,
        "documents_uploaded": len(documents),
        "matched_operations": matched_count,
        "document_coverage": f"{matched_count}/{total_operations}",
        "top_reason_text": top_reason_text,
    }


def summarize_reasons(results_df: pd.DataFrame) -> pd.DataFrame:
    exploded = []
    for codes in results_df["reason_codes"].fillna(""):
        for code in [item.strip() for item in str(codes).split("|") if item.strip()]:
            exploded.append(code)
    if not exploded:
        return pd.DataFrame(columns=["Причина", "Количество"])

    counts = pd.Series(exploded).value_counts()
    label_map = {
        "missing_required_fields": "Пустые обязательные поля",
        "duplicate_document_number": "Дубликаты номера документа",
        "amount_outlier": "Нетипично большая сумма",
        "suspicious_description": "Подозрительное описание",
        "atypical_counterparty": "Нетипичный контрагент",
        "no_primary_document": "Отсутствует подтверждающий документ",
        "document_amount_mismatch": "Несоответствие суммы документу",
        "document_date_mismatch": "Несоответствие даты документу",
        "document_counterparty_mismatch": "Несоответствие контрагента документу",
        "document_incomplete": "Неполные реквизиты документа",
        "ml_anomaly": "ML-анализ выявил аномалию",
    }
    return (
        counts.rename_axis("code")
        .reset_index(name="Количество")
        .assign(Причина=lambda frame: frame["code"].map(label_map).fillna(frame["code"]))
        [["Причина", "Количество"]]
    )


def to_display_frame(results_df: pd.DataFrame) -> pd.DataFrame:
    display = results_df.copy()
    display["operation_date"] = display["operation_date_display"]
    display["amount"] = display["amount_display"]
    return display[
        [
            "operation_id",
            "operation_date",
            "document_number",
            "counterparty",
            "amount",
            "status",
            "risk_score",
            "document_check_status",
            "reason_details",
            "ai_comment",
        ]
    ].rename(
        columns={
            "operation_id": "ID",
            "operation_date": "Дата",
            "document_number": "Документ",
            "counterparty": "Контрагент",
            "amount": "Сумма",
            "status": "Статус",
            "risk_score": "Риск-балл",
            "document_check_status": "Сверка с документом",
            "reason_details": "Причины",
            "ai_comment": "AI-комментарий",
        }
    )


def export_results_csv(results_df: pd.DataFrame) -> bytes:
    return results_df.to_csv(index=False).encode("utf-8-sig")


def save_report_bundle(results_df: pd.DataFrame, summary: dict[str, object], target_dir: Path) -> dict[str, str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    csv_path = target_dir / "audit_results.csv"
    json_path = target_dir / "audit_summary.json"
    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"csv": str(csv_path), "json": str(json_path)}
