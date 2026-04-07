from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from shingadip.analysis import text_similarity
from shingadip.documents import DocumentExtraction


REASON_LABELS = {
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


def reason_code_to_label(code: str) -> str:
    return REASON_LABELS.get(code, code)


def build_summary(results_df: pd.DataFrame, documents: list[DocumentExtraction]) -> dict[str, object]:
    total_operations = len(results_df)
    ok_count = int((results_df["status"] == "OK").sum())
    warning_count = int((results_df["status"] == "WARNING").sum())
    risk_count = int((results_df["status"] == "RISK").sum())
    average_risk_score = round(float(results_df["risk_score"].mean() if total_operations else 0.0), 2)
    document_statuses = results_df.get("document_check_status", pd.Series(dtype="object")).fillna("").astype(str).str.upper()
    matched_count = int(results_df["matched_document"].notna().sum())
    not_provided_count = int((document_statuses == "NOT_PROVIDED").sum())
    missing_count = int((document_statuses == "MISSING").sum())
    mismatch_count = int((document_statuses == "MISMATCH").sum())
    documents_expected_count = max(total_operations - not_provided_count, 0)
    if documents_expected_count > 0:
        document_coverage = f"{matched_count}/{documents_expected_count}"
        document_coverage_percent: float | None = round((matched_count / documents_expected_count * 100), 2)
        document_scope_note = (
            f"для автоматической сверки были доступны документы по {documents_expected_count} операциям; "
            f"без найденного документа осталось {missing_count}, с расхождениями — {mismatch_count}"
        )
    else:
        document_coverage = "документы не предоставлены"
        document_coverage_percent = None
        document_scope_note = "первичные документы не подавались на вход, поэтому документальное покрытие не оценивалось"

    reason_summary = build_reason_summary(results_df)
    top_reason_text = (
        ", ".join(reason_summary["Причина"].head(3).tolist())
        if not reason_summary.empty
        else "существенные отклонения не выявлены"
    )

    counterparty_summary = build_counterparty_summary(results_df)
    problematic_counterparties_text = _build_problematic_counterparties_text(counterparty_summary)
    audit_recommendation = _build_audit_recommendation(ok_count, warning_count, risk_count)
    priority_review_focus = _build_priority_review_focus(
        results_df,
        {
            "documents_expected_count": documents_expected_count,
            "document_missing_count": missing_count,
        },
    )

    return {
        "total_operations": total_operations,
        "ok_count": ok_count,
        "warning_count": warning_count,
        "risk_count": risk_count,
        "average_risk_score": average_risk_score,
        "documents_uploaded": len(documents),
        "matched_operations": matched_count,
        "document_coverage": document_coverage,
        "document_coverage_percent": document_coverage_percent,
        "documents_expected_count": documents_expected_count,
        "document_not_provided_count": not_provided_count,
        "document_missing_count": missing_count,
        "document_mismatch_count": mismatch_count,
        "document_scope_note": document_scope_note,
        "priority_review_focus": priority_review_focus,
        "top_reason_text": top_reason_text,
        "problematic_counterparties_text": problematic_counterparties_text,
        "audit_recommendation": audit_recommendation,
    }


def build_report_tables(results_df: pd.DataFrame, summary: dict[str, object]) -> dict[str, pd.DataFrame]:
    return {
        "risk_register": build_risk_register(results_df),
        "reason_summary": build_reason_summary(results_df),
        "counterparty_summary": build_counterparty_summary(results_df),
        "document_reconciliation": build_document_reconciliation(results_df),
    }


def build_audit_conclusion(
    summary: dict[str, object],
    results_df: pd.DataFrame,
    report_tables: dict[str, pd.DataFrame] | None = None,
) -> dict[str, object]:
    main_reasons = summary.get("top_reason_text", "существенные отклонения не выявлены")
    problematic_counterparties = summary.get("problematic_counterparties_text", "не выявлены")
    recommendation = summary.get("audit_recommendation", "Продолжить выборочную проверку операций.")
    ai_comment = summary.get("dataset_comment") or recommendation
    report_tables = report_tables or build_report_tables(results_df, summary)
    risk_register = report_tables.get("risk_register", build_risk_register(results_df))
    top_risk_operations = []
    if not risk_register.empty:
        top_risk_operations = risk_register.head(5).to_dict(orient="records")
    coverage_quality = _build_document_coverage_quality(
        summary.get("document_coverage_percent"),
        summary.get("documents_expected_count"),
    )
    priority_review_focus = _build_priority_review_focus(results_df, summary)
    short_text = _build_short_audit_conclusion(summary, main_reasons, problematic_counterparties, coverage_quality)

    conclusion_text = (
        f"Проанализировано {summary['total_operations']} операций: "
        f"OK — {summary['ok_count']}, WARNING — {summary['warning_count']}, RISK — {summary['risk_count']}. "
        f"Основные причины отклонений: {main_reasons}. "
        f"Проблемные контрагенты: {problematic_counterparties}. "
        f"Документальное покрытие: {summary['document_coverage']}. "
        f"Качество покрытия — {coverage_quality}. "
        f"Краткий вывод: {short_text} "
        f"Рекомендация аудитору: {recommendation}"
    )

    return {
        "total_operations": summary["total_operations"],
        "ok_count": summary["ok_count"],
        "warning_count": summary["warning_count"],
        "risk_count": summary["risk_count"],
        "main_reasons": main_reasons,
        "problematic_counterparties": problematic_counterparties,
        "document_coverage": summary["document_coverage"],
        "document_coverage_quality": coverage_quality,
        "document_scope_note": summary.get("document_scope_note", ""),
        "recommendation": recommendation,
        "ai_comment": ai_comment,
        "short_text": short_text,
        "priority_review_focus": priority_review_focus,
        "top_risk_operations": top_risk_operations,
        "text": conclusion_text,
    }


def build_risk_register(results_df: pd.DataFrame) -> pd.DataFrame:
    risk_rows = results_df.loc[results_df["status"].isin(["WARNING", "RISK"])].copy()
    if risk_rows.empty:
        return pd.DataFrame(
            columns=[
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
            ]
        )

    risk_rows["Дата"] = risk_rows["operation_date_display"]
    risk_rows["Сумма"] = risk_rows["amount_display"]
    register = risk_rows[
        [
            "operation_id",
            "Дата",
            "document_number",
            "counterparty",
            "Сумма",
            "status",
            "risk_score",
            "priority",
            "confidence",
            "risk_category",
            "primary_risk_driver",
            "reason_details",
            "recommended_action",
            "document_check_status",
            "ai_comment",
        ]
    ].rename(
        columns={
            "operation_id": "ID операции",
            "document_number": "Номер документа",
            "counterparty": "Контрагент",
            "status": "Статус",
            "risk_score": "Риск-балл",
            "priority": "Приоритет проверки",
            "confidence": "Уверенность",
            "risk_category": "Категория риска",
            "primary_risk_driver": "Ведущий фактор риска",
            "reason_details": "Причина отклонения",
            "recommended_action": "Рекомендуемое действие",
            "document_check_status": "Статус сверки документа",
            "ai_comment": "Краткий AI-комментарий",
        }
    )
    return register.sort_values(["Риск-балл", "ID операции"], ascending=[False, True]).reset_index(drop=True)


def build_reason_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    total_operations = len(results_df)
    exploded_rows: list[dict[str, object]] = []
    for _, row in results_df.iterrows():
        codes = [item.strip() for item in str(row.get("reason_codes", "")).split("|") if item.strip()]
        for code in codes:
            exploded_rows.append(
                {
                    "code": code,
                    "risk_score": row.get("risk_score", 0),
                }
            )

    if not exploded_rows:
        return pd.DataFrame(
            columns=[
                "Причина",
                "Количество случаев",
                "Доля от общего числа операций, %",
                "Средний риск по причине",
            ]
        )

    exploded = pd.DataFrame(exploded_rows)
    summary = (
        exploded.groupby("code", dropna=False)
        .agg(
            count=("code", "size"),
            avg_risk=("risk_score", "mean"),
        )
        .reset_index()
    )
    summary["share"] = summary["count"].map(lambda value: round((value / total_operations * 100) if total_operations else 0.0, 2))
    summary["Причина"] = summary["code"].map(reason_code_to_label)
    return (
        summary.rename(
            columns={
                "count": "Количество случаев",
                "share": "Доля от общего числа операций, %",
                "avg_risk": "Средний риск по причине",
            }
        )[
            [
                "Причина",
                "Количество случаев",
                "Доля от общего числа операций, %",
                "Средний риск по причине",
            ]
        ]
        .sort_values(["Количество случаев", "Средний риск по причине"], ascending=[False, False])
        .reset_index(drop=True)
        .assign(**{"Средний риск по причине": lambda frame: frame["Средний риск по причине"].round(2)})
    )


def build_counterparty_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame(
            columns=[
                "Контрагент",
                "Количество операций",
                "Общая сумма операций",
                "Количество WARNING",
                "Количество RISK",
                "Средний риск",
                "Количество операций без документа",
                "Количество операций с расхождением документа",
            ]
        )

    enriched = results_df.copy()
    enriched["counterparty_report"] = enriched["counterparty"].fillna("не указан")
    enriched["amount_numeric"] = pd.to_numeric(enriched["amount"], errors="coerce").fillna(0.0)
    enriched["warning_flag"] = enriched["status"] == "WARNING"
    enriched["risk_flag"] = enriched["status"] == "RISK"
    enriched["no_document_flag"] = enriched["document_check_status"].isin(["MISSING", "NOT_PROVIDED"])
    enriched["document_mismatch_flag"] = enriched["document_check_status"] == "MISMATCH"

    grouped = (
        enriched.groupby("counterparty_report", dropna=False)
        .agg(
            operation_count=("operation_id", "size"),
            total_amount=("amount_numeric", "sum"),
            warning_count=("warning_flag", "sum"),
            risk_count=("risk_flag", "sum"),
            average_risk=("risk_score", "mean"),
            no_document_count=("no_document_flag", "sum"),
            mismatch_count=("document_mismatch_flag", "sum"),
        )
        .reset_index()
        .rename(
            columns={
                "counterparty_report": "Контрагент",
                "operation_count": "Количество операций",
                "total_amount": "Общая сумма операций",
                "warning_count": "Количество WARNING",
                "risk_count": "Количество RISK",
                "average_risk": "Средний риск",
                "no_document_count": "Количество операций без документа",
                "mismatch_count": "Количество операций с расхождением документа",
            }
        )
    )
    grouped["Общая сумма операций"] = grouped["Общая сумма операций"].round(2)
    grouped["Средний риск"] = grouped["Средний риск"].round(2)
    return grouped.sort_values(
        ["Количество RISK", "Количество WARNING", "Средний риск", "Общая сумма операций"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def build_document_reconciliation(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame(
            columns=[
                "ID операции",
                "Номер документа",
                "Найденный документ",
                "Статус сверки",
                "Совпадение суммы",
                "Совпадение даты",
                "Совпадение контрагента",
            ]
        )

    rows: list[dict[str, object]] = []
    for _, row in results_df.iterrows():
        amount_match = _build_amount_match_flag(row)
        date_match = _build_date_match_flag(row)
        counterparty_match = _build_counterparty_match_flag(row)
        rows.append(
            {
                "ID операции": row["operation_id"],
                "Номер документа": row.get("document_number"),
                "Найденный документ": row.get("matched_document"),
                "Статус сверки": _normalize_document_check_status(row.get("document_check_status")),
                "Совпадение суммы": amount_match,
                "Совпадение даты": date_match,
                "Совпадение контрагента": counterparty_match,
            }
        )

    return pd.DataFrame(rows).sort_values(["Статус сверки", "ID операции"]).reset_index(drop=True)


def summarize_reasons(results_df: pd.DataFrame) -> pd.DataFrame:
    reason_summary = build_reason_summary(results_df)
    if reason_summary.empty:
        return pd.DataFrame(columns=["Причина", "Количество"])
    return reason_summary[["Причина", "Количество случаев"]].rename(columns={"Количество случаев": "Количество"})


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
            "priority",
            "confidence",
            "risk_category",
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
            "priority": "Приоритет",
            "confidence": "Уверенность",
            "risk_category": "Категория риска",
            "document_check_status": "Сверка с документом",
            "reason_details": "Причины",
            "ai_comment": "AI-комментарий",
        }
    )


def export_results_csv(results_df: pd.DataFrame) -> bytes:
    return export_frame_csv(results_df)


def export_frame_csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")


def save_report_bundle(
    results_df: pd.DataFrame,
    summary: dict[str, object],
    target_dir: Path,
    report_tables: dict[str, pd.DataFrame] | None = None,
    audit_conclusion: dict[str, object] | None = None,
) -> dict[str, str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    report_tables = report_tables or build_report_tables(results_df, summary)
    audit_conclusion = audit_conclusion or build_audit_conclusion(summary, results_df)

    file_map: dict[str, str] = {}

    csv_path = target_dir / "audit_results.csv"
    json_path = target_dir / "audit_summary.json"
    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    summary_payload = {
        **summary,
        "audit_conclusion": audit_conclusion,
    }
    json_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    file_map["csv"] = str(csv_path)
    file_map["json"] = str(json_path)

    report_file_names = {
        "risk_register": "risk_register.csv",
        "reason_summary": "reason_summary.csv",
        "counterparty_summary": "counterparty_summary.csv",
        "document_reconciliation": "document_reconciliation.csv",
    }
    for key, frame in report_tables.items():
        if key not in report_file_names:
            continue
        path = target_dir / report_file_names[key]
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        file_map[key] = str(path)

    return file_map


def _build_problematic_counterparties_text(counterparty_summary: pd.DataFrame) -> str:
    if counterparty_summary.empty:
        return "не выявлены"

    problematic = counterparty_summary.loc[
        (counterparty_summary["Количество WARNING"] > 0) | (counterparty_summary["Количество RISK"] > 0)
    ].copy()
    if problematic.empty:
        return "не выявлены"

    problematic = problematic.sort_values(
        ["Количество RISK", "Количество WARNING", "Средний риск"],
        ascending=[False, False, False],
    ).head(3)
    return ", ".join(problematic["Контрагент"].astype(str).tolist())


def _build_audit_recommendation(ok_count: int, warning_count: int, risk_count: int) -> str:
    if risk_count > 0:
        return "Провести первоочередную ручную проверку операций со статусом RISK и подтвердить их первичными документами."
    if warning_count > 0:
        return "Провести выборочную ручную проверку операций со статусом WARNING и уточнить проблемные реквизиты."
    if ok_count > 0:
        return "Существенных отклонений не обнаружено, можно сохранить результаты как подтверждение внутреннего контроля."
    return "Данных для рекомендаций недостаточно."


def _build_short_audit_conclusion(
    summary: dict[str, object],
    main_reasons: str,
    problematic_counterparties: str,
    coverage_quality: str,
) -> str:
    if int(summary.get("risk_count", 0) or 0) > 0:
        return (
            f"Выявлены операции с повышенным риском; основные проблемы связаны с причинами: {main_reasons}. "
            f"Наибольшее внимание требуется по контрагентам: {problematic_counterparties}. "
            f"Документальное покрытие оценивается как {coverage_quality}."
        )
    if int(summary.get("warning_count", 0) or 0) > 0:
        return (
            f"Критичных операций не выявлено, однако имеются предупреждения, связанные с причинами: {main_reasons}. "
            f"Документальное покрытие оценивается как {coverage_quality}."
        )
    return "Существенных отклонений по анализируемой выборке не выявлено."


def _build_priority_review_focus(results_df: pd.DataFrame, summary: dict[str, object]) -> list[dict[str, object]]:
    def count_reason(code: str) -> int:
        return int(
            results_df["reason_codes"]
            .fillna("")
            .astype(str)
            .map(lambda value: code in {item.strip() for item in value.split("|") if item.strip()})
            .sum()
        )

    missing_description_count = int(
        results_df.get("description", pd.Series(dtype="object"))
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    focus_items = [
        {"label": "Операции без описания", "count": missing_description_count},
        {"label": "Крупные суммы", "count": count_reason("amount_outlier")},
        {"label": "Дубликаты номера документа", "count": count_reason("duplicate_document_number")},
        {"label": "Редкие контрагенты", "count": count_reason("atypical_counterparty")},
    ]
    if int(summary.get("documents_expected_count", 0) or 0) > 0:
        focus_items.append(
            {
                "label": "Операции без найденного документа",
                "count": int(summary.get("document_missing_count", 0) or 0),
            }
        )

    ranked = [item for item in focus_items if int(item["count"]) > 0]
    ranked.sort(key=lambda item: (-int(item["count"]), item["label"]))
    return ranked[:5]


def _build_document_coverage_quality(document_coverage_percent: object, documents_expected_count: object | None = None) -> str:
    try:
        expected = int(documents_expected_count) if documents_expected_count is not None else None
    except (TypeError, ValueError):
        expected = None
    if expected == 0:
        return "не оценивалось"

    try:
        coverage = float(document_coverage_percent)
    except (TypeError, ValueError):
        coverage = -1.0

    if coverage < 0:
        return "не оценивалось"

    if coverage >= 80:
        return "высокое"
    if coverage >= 50:
        return "умеренное"
    return "ограниченное"


def _normalize_document_check_status(status: object) -> str:
    status_text = str(status or "").strip().upper()
    if status_text in {"OK", "PARTIAL"}:
        return "MATCH"
    if status_text in {"MISMATCH", "MISSING", "NOT_PROVIDED"}:
        return status_text
    return status_text or "NOT_PROVIDED"


def _build_amount_match_flag(row: pd.Series) -> str:
    amount = row.get("amount")
    matched_amount = row.get("matched_document_amount")
    if row.get("matched_document") is None:
        return "Н/Д"
    if amount is None or matched_amount is None or pd.isna(amount) or pd.isna(matched_amount):
        return "Н/Д"
    tolerance = max(1.0, float(amount) * 0.02)
    return "Да" if abs(float(amount) - float(matched_amount)) <= tolerance else "Нет"


def _build_date_match_flag(row: pd.Series) -> str:
    operation_date = row.get("operation_date")
    matched_date = pd.to_datetime(row.get("matched_document_date"), errors="coerce")
    if row.get("matched_document") is None:
        return "Н/Д"
    if operation_date is None or pd.isna(operation_date) or pd.isna(matched_date):
        return "Н/Д"
    return "Да" if pd.Timestamp(operation_date).normalize() == matched_date.normalize() else "Нет"


def _build_counterparty_match_flag(row: pd.Series) -> str:
    counterparty = row.get("counterparty")
    matched_counterparty = row.get("matched_document_counterparty")
    if row.get("matched_document") is None:
        return "Н/Д"
    if not counterparty or not matched_counterparty:
        return "Н/Д"
    return "Да" if text_similarity(counterparty, matched_counterparty) >= 0.65 else "Нет"
