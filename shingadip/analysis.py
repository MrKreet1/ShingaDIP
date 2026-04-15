from __future__ import annotations

import json
import math
import re
from difflib import SequenceMatcher

import pandas as pd
from sklearn.ensemble import IsolationForest

from shingadip.config import RISK_WEIGHTS, SUSPICIOUS_DESCRIPTION_KEYWORDS
from shingadip.documents import DocumentExtraction


RISK_SCORE_COMPONENT_LABELS = {
    "score_missing_fields": "Пропущенные обязательные поля",
    "score_duplicate": "Дубликаты номера документа",
    "score_description": "Подозрительное описание",
    "score_amount": "Нетипично большая сумма",
    "score_document": "Документарные расхождения",
    "score_ml": "ML-анализ аномалий",
    "score_counterparty": "Нетипичный контрагент",
    "score_compound_bonus": "Комбинация факторов риска",
    "score_threshold_adjustment": "Пороговая корректировка статуса",
    "score_cap_adjustment": "Ограничение итогового балла",
}


def normalize_token(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"[^a-zа-я0-9]+", "", str(value).lower())


def text_similarity(left: object, right: object) -> float:
    left_token = normalize_token(left)
    right_token = normalize_token(right)
    if not left_token or not right_token:
        return 0.0
    return SequenceMatcher(a=left_token, b=right_token).ratio()


def detect_ml_anomalies(operations_df: pd.DataFrame) -> pd.DataFrame:
    scored = operations_df.copy()
    scored["ml_anomaly_flag"] = False
    scored["ml_anomaly_strength"] = 0.0

    if len(scored) < 5:
        return scored

    amount_mean = scored["amount"].dropna().mean()
    if math.isnan(amount_mean):
        amount_mean = 0.0

    features = pd.DataFrame(
        {
            "amount": scored["amount"].fillna(amount_mean),
            "vat": scored["vat"].fillna(0.0),
            "description_length": scored["description"].fillna("").map(len),
            "document_number_length": scored["document_number"].fillna("").map(len),
            "missing_count": scored["missing_required_fields"].map(len),
            "counterparty_frequency": scored["counterparty"].map(scored["counterparty"].value_counts(dropna=False)).fillna(0),
            "account_frequency": scored["account"].map(scored["account"].value_counts(dropna=False)).fillna(0),
            "month": scored["operation_date"].map(lambda value: value.month if value is not None else 0),
            "weekday": scored["operation_date"].map(lambda value: value.dayofweek if value is not None else 0),
        }
    )

    if features.nunique().sum() <= len(features.columns):
        return scored

    if len(scored) >= 1000:
        contamination = 0.01
    elif len(scored) >= 250:
        contamination = 0.015
    elif len(scored) >= 75:
        contamination = 0.02
    else:
        contamination = min(0.08, max(0.03, 2 / len(scored)))
    model = IsolationForest(random_state=42, contamination=contamination)
    predictions = model.fit_predict(features)
    anomaly_scores = -model.score_samples(features)
    normalized = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min() + 1e-9)

    scored["ml_anomaly_flag"] = predictions == -1
    scored["ml_anomaly_strength"] = normalized
    return scored


def match_document(operation_row: pd.Series, documents: list[DocumentExtraction]) -> tuple[DocumentExtraction | None, float]:
    if not documents:
        return None, 0.0

    best_document = None
    best_score = 0.0
    op_doc_number = normalize_token(operation_row.get("document_number"))
    if op_doc_number:
        exact_candidates = [
            document for document in documents if normalize_token(document.document_number) == op_doc_number
        ]
        if exact_candidates:
            documents_to_consider = exact_candidates
        else:
            fallback_candidates = [
                document for document in documents if not normalize_token(document.document_number)
            ]
            if not fallback_candidates:
                return None, 0.0
            documents_to_consider = fallback_candidates
    else:
        documents_to_consider = documents

    for document in documents_to_consider:
        score = 0.0
        doc_number = normalize_token(document.document_number)
        if op_doc_number and doc_number and op_doc_number == doc_number:
            score += 5.0
        elif op_doc_number and doc_number and (op_doc_number in doc_number or doc_number in op_doc_number):
            score += 3.0

        if operation_row.get("amount") is not None and document.amount is not None:
            difference = abs(operation_row["amount"] - document.amount)
            tolerance = max(1.0, operation_row["amount"] * 0.02)
            if difference <= tolerance:
                score += 3.0
            elif difference <= max(10.0, operation_row["amount"] * 0.05):
                score += 1.2

        if operation_row.get("operation_date") is not None and document.document_date is not None:
            day_difference = abs((operation_row["operation_date"] - document.document_date).days)
            if day_difference == 0:
                score += 2.0
            elif day_difference <= 3:
                score += 0.8

        similarity = text_similarity(operation_row.get("counterparty"), document.counterparty)
        if similarity >= 0.9:
            score += 2.0
        elif similarity >= 0.7:
            score += 1.0

        if score > best_score:
            best_score = score
            best_document = document

    if best_score < 3.0:
        return None, 0.0
    return best_document, best_score


def analyze_operations(
    operations_df: pd.DataFrame,
    documents: list[DocumentExtraction] | None = None,
) -> pd.DataFrame:
    documents = documents or []
    has_uploaded_documents = bool(documents)
    scored = detect_ml_anomalies(operations_df)
    duplicate_counts = scored["document_number"].fillna("").map(normalize_token).value_counts().to_dict()
    counterparty_counts = scored["counterparty"].fillna("не указано").value_counts(dropna=False).to_dict()

    amounts = scored["amount"].dropna()
    if len(amounts) >= 4:
        q1 = amounts.quantile(0.25)
        q3 = amounts.quantile(0.75)
        iqr = q3 - q1
        high_amount_threshold = q3 + 1.5 * iqr
    elif len(amounts) > 1:
        high_amount_threshold = amounts.mean() + 2 * amounts.std(ddof=0)
    else:
        high_amount_threshold = float(amounts.max()) if not amounts.empty else float("inf")

    results: list[dict[str, object]] = []
    for _, row in scored.iterrows():
        matched_document, match_score = match_document(row, documents)
        reasons: list[str] = []
        reason_codes: list[str] = []
        risk_score = 0
        score_breakdown = {key: 0 for key in RISK_SCORE_COMPONENT_LABELS}
        document_check_status = "NOT_PROVIDED" if not has_uploaded_documents else "OK"

        missing_fields = row.get("missing_required_fields", [])
        if missing_fields:
            reason_codes.append("missing_required_fields")
            missing_names = ", ".join(_translate_field_name(field) for field in missing_fields)
            reasons.append(f"Отсутствуют обязательные поля: {missing_names}.")
            missing_score = RISK_WEIGHTS["missing_required_fields"] + min(len(missing_fields) * 2, 8)
            risk_score += missing_score
            score_breakdown["score_missing_fields"] += missing_score

        document_key = normalize_token(row.get("document_number"))
        if document_key and duplicate_counts.get(document_key, 0) > 1:
            reason_codes.append("duplicate_document_number")
            reasons.append("Номер документа встречается в таблице более одного раза.")
            duplicate_score = RISK_WEIGHTS["duplicate_document_number"]
            risk_score += duplicate_score
            score_breakdown["score_duplicate"] += duplicate_score

        description = (row.get("description") or "").lower()
        if any(keyword in description for keyword in SUSPICIOUS_DESCRIPTION_KEYWORDS):
            reason_codes.append("suspicious_description")
            reasons.append("Описание операции содержит нетипичные или рискованные формулировки.")
            description_score = RISK_WEIGHTS["suspicious_description"]
            risk_score += description_score
            score_breakdown["score_description"] += description_score

        if row.get("amount") is not None and row["amount"] > high_amount_threshold:
            reason_codes.append("amount_outlier")
            reasons.append("Сумма операции значительно выше типового диапазона.")
            amount_score = RISK_WEIGHTS["amount_outlier"]
            risk_score += amount_score
            score_breakdown["score_amount"] += amount_score

        if matched_document is None and has_uploaded_documents:
            reason_codes.append("no_primary_document")
            reasons.append("Не найден подходящий первичный документ для сверки.")
            missing_doc_score = RISK_WEIGHTS["no_primary_document"]
            risk_score += missing_doc_score
            score_breakdown["score_document"] += missing_doc_score
            document_check_status = "MISSING"
        elif matched_document is not None:
            if matched_document.amount is None or matched_document.document_date is None:
                reason_codes.append("document_incomplete")
                reasons.append("Документ загружен, но извлеченные реквизиты неполные.")
                incomplete_score = RISK_WEIGHTS["document_incomplete"]
                risk_score += incomplete_score
                score_breakdown["score_document"] += incomplete_score
                document_check_status = "PARTIAL"

            if row.get("amount") is not None and matched_document.amount is not None:
                amount_difference = abs(row["amount"] - matched_document.amount)
                tolerance = max(1.0, row["amount"] * 0.02)
                if amount_difference > tolerance:
                    reason_codes.append("document_amount_mismatch")
                    reasons.append("Сумма операции не совпадает с суммой в первичном документе.")
                    mismatch_score = RISK_WEIGHTS["document_amount_mismatch"]
                    risk_score += mismatch_score
                    score_breakdown["score_document"] += mismatch_score
                    document_check_status = "MISMATCH"

            if row.get("operation_date") is not None and matched_document.document_date is not None:
                if row["operation_date"] != matched_document.document_date:
                    reason_codes.append("document_date_mismatch")
                    reasons.append("Дата операции не совпадает с датой первичного документа.")
                    date_score = RISK_WEIGHTS["document_date_mismatch"]
                    risk_score += date_score
                    score_breakdown["score_document"] += date_score
                    document_check_status = "MISMATCH"

            similarity = text_similarity(row.get("counterparty"), matched_document.counterparty)
            if matched_document.counterparty and similarity < 0.65:
                reason_codes.append("document_counterparty_mismatch")
                reasons.append("Контрагент в учете отличается от контрагента в документе.")
                counterparty_mismatch_score = RISK_WEIGHTS["document_counterparty_mismatch"]
                risk_score += counterparty_mismatch_score
                score_breakdown["score_document"] += counterparty_mismatch_score
                document_check_status = "MISMATCH"

        if bool(row.get("ml_anomaly_flag")):
            reason_codes.append("ml_anomaly")
            reasons.append("Модель аномалий отметила операцию как нетипичную относительно остальных записей.")
            ml_score = int(RISK_WEIGHTS["ml_anomaly"] * float(row.get("ml_anomaly_strength", 0.0) or 0.0) + 4)
            risk_score += ml_score
            score_breakdown["score_ml"] += ml_score

        counterparty = row.get("counterparty")
        rare_counterparty_support = {
            "amount_outlier",
            "missing_required_fields",
            "no_primary_document",
            "document_amount_mismatch",
            "document_date_mismatch",
            "document_counterparty_mismatch",
            "ml_anomaly",
        }
        if (
            counterparty
            and len(scored) >= 20
            and counterparty_counts.get(counterparty, 0) == 1
            and rare_counterparty_support.intersection(reason_codes)
        ):
            reason_codes.append("atypical_counterparty")
            reasons.append("Контрагент встречается редко и усиливает общий риск операции.")
            counterparty_score = RISK_WEIGHTS["atypical_counterparty"]
            risk_score += counterparty_score
            score_breakdown["score_counterparty"] += counterparty_score

        compound_risk_factors = {
            "missing_required_fields",
            "amount_outlier",
            "no_primary_document",
            "document_amount_mismatch",
            "document_date_mismatch",
            "document_counterparty_mismatch",
            "ml_anomaly",
            "atypical_counterparty",
        }
        compound_count = len(compound_risk_factors.intersection(reason_codes))
        if compound_count >= 4:
            risk_score += 16
            score_breakdown["score_compound_bonus"] += 16
        elif compound_count >= 3:
            risk_score += 10
            score_breakdown["score_compound_bonus"] += 10

        pre_threshold_score = risk_score
        if "document_amount_mismatch" in reason_codes:
            risk_score = max(risk_score, 45)
        elif "no_primary_document" in reason_codes:
            risk_score = max(risk_score, 30)
        elif {
            "document_date_mismatch",
            "document_counterparty_mismatch",
            "duplicate_document_number",
            "missing_required_fields",
        }.intersection(reason_codes):
            risk_score = max(risk_score, 30)
        score_breakdown["score_threshold_adjustment"] += max(risk_score - pre_threshold_score, 0)

        rounded_score = int(round(risk_score))
        final_score = min(rounded_score, 100)
        score_breakdown["score_cap_adjustment"] += final_score - rounded_score
        risk_score = final_score
        if risk_score >= 70:
            status = "RISK"
        elif risk_score >= 30:
            status = "WARNING"
        else:
            status = "OK"

        if not reasons:
            reasons = ["Существенных отклонений не обнаружено."]
            document_check_status = "OK" if matched_document is not None else document_check_status

        results.append(
            {
                **row.to_dict(),
                "status": status,
                "risk_score": risk_score,
                "reason_codes": " | ".join(reason_codes),
                "reason_details": " ".join(reasons),
                "matched_document": matched_document.file_name if matched_document else None,
                "matched_document_number": matched_document.document_number if matched_document else None,
                "matched_document_date": matched_document.document_date.strftime("%Y-%m-%d")
                if matched_document and matched_document.document_date is not None
                else None,
                "matched_document_amount": matched_document.amount if matched_document else None,
                "matched_document_counterparty": matched_document.counterparty if matched_document else None,
                "matched_document_confidence": round(match_score, 2) if matched_document else 0.0,
                "document_check_status": document_check_status,
                **score_breakdown,
                "risk_score_breakdown_json": json.dumps(score_breakdown, ensure_ascii=False),
                "risk_score_top_contributors": _build_top_score_contributors(score_breakdown),
            }
        )

    result_df = pd.DataFrame(results)
    return result_df.sort_values(["risk_score", "operation_id"], ascending=[False, True]).reset_index(drop=True)


def _translate_field_name(field_name: str) -> str:
    translations = {
        "operation_date": "дата операции",
        "document_number": "номер документа",
        "counterparty": "контрагент",
        "amount": "сумма",
        "description": "описание операции",
    }
    return translations.get(field_name, field_name)


def _build_top_score_contributors(score_breakdown: dict[str, int]) -> str:
    ranked = [
        (RISK_SCORE_COMPONENT_LABELS[key], value)
        for key, value in score_breakdown.items()
        if value and key != "score_cap_adjustment"
    ]
    ranked.sort(key=lambda item: (-int(item[1]), item[0]))
    if not ranked:
        return "существенные вклады не выявлены"
    return "; ".join(f"{label}: {value}" for label, value in ranked[:3])
