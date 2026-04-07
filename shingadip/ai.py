from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

import pandas as pd

from shingadip.config import (
    DEFAULT_DOCUMENT_ANALYSIS_MODE,
    DEFAULT_DOCUMENT_ENDPOINT,
    DEFAULT_LIGHTONOCR_ENDPOINT,
    DEFAULT_TEXT_ENDPOINT,
    LIGHTONOCR_MODEL_CANDIDATES,
    LIGHTONOCR_MODEL_ID,
    RISK_WEIGHTS,
    TEXT_MODEL_ID,
    VISION_MODEL_ID,
)


RISK_FACTOR_LABELS = {
    "missing_required_fields": "отсутствие обязательных реквизитов",
    "duplicate_document_number": "дублирование номера документа",
    "amount_outlier": "нетипично высокая сумма",
    "suspicious_description": "подозрительное описание операции",
    "atypical_counterparty": "редкий или нетипичный контрагент",
    "no_primary_document": "отсутствие подтверждающего документа",
    "document_amount_mismatch": "расхождение суммы с документом",
    "document_date_mismatch": "расхождение даты с документом",
    "document_counterparty_mismatch": "расхождение контрагента с документом",
    "document_incomplete": "неполные реквизиты документа",
    "ml_anomaly": "нетипичность по модели аномалий",
}

RISK_CATEGORY_LABELS = {
    "document": "Документарный риск",
    "amount": "Риск суммы",
    "counterparty": "Риск контрагента",
    "description": "Риск описания",
    "duplicate": "Риск дублирования",
    "anomaly": "Аномальный риск",
    "data_quality": "Риск качества реквизитов",
}

RISK_CATEGORY_MAP = {
    "missing_required_fields": "data_quality",
    "duplicate_document_number": "duplicate",
    "amount_outlier": "amount",
    "suspicious_description": "description",
    "atypical_counterparty": "counterparty",
    "no_primary_document": "document",
    "document_amount_mismatch": "document",
    "document_date_mismatch": "document",
    "document_counterparty_mismatch": "document",
    "document_incomplete": "document",
    "ml_anomaly": "anomaly",
}

ACTION_SEEDS = {
    "document_amount_mismatch": "Сверить сумму в учете, первичном документе и платежных регистрах.",
    "no_primary_document": "Запросить первичный документ и проверить документальное обоснование операции.",
    "document_counterparty_mismatch": "Проверить правильность указания контрагента в учете и документе.",
    "document_date_mismatch": "Проверить дату признания операции и дату первичного документа.",
    "duplicate_document_number": "Проверить, не произошло ли двойного отражения одной и той же операции.",
    "missing_required_fields": "Уточнить отсутствующие реквизиты и подтвердить хозяйственное содержание операции.",
    "amount_outlier": "Проверить экономическое основание нетипично крупной суммы.",
    "suspicious_description": "Уточнить содержание операции и проверить корректность описания.",
    "atypical_counterparty": "Проверить историю взаимодействия с контрагентом и основания выбора поставщика.",
    "ml_anomaly": "Провести выборочную ручную проверку, так как операция нетипична для выборки.",
}

PRIORITY_ORDER = [
    "document_amount_mismatch",
    "no_primary_document",
    "document_counterparty_mismatch",
    "document_date_mismatch",
    "missing_required_fields",
    "duplicate_document_number",
    "amount_outlier",
    "suspicious_description",
    "atypical_counterparty",
    "ml_anomaly",
    "document_incomplete",
]


@dataclass(slots=True)
class AISettings:
    use_lm_studio: bool = False
    endpoint: str = DEFAULT_TEXT_ENDPOINT
    model: str = TEXT_MODEL_ID
    timeout_seconds: int = 20
    max_rows: int = 8
    use_document_model: bool = False
    document_analysis_mode: str = DEFAULT_DOCUMENT_ANALYSIS_MODE
    document_endpoint: str = DEFAULT_DOCUMENT_ENDPOINT
    document_model: str = VISION_MODEL_ID
    lightonocr_endpoint: str = DEFAULT_LIGHTONOCR_ENDPOINT
    lightonocr_model: str = LIGHTONOCR_MODEL_ID
    max_document_ai_calls: int = 10


def discover_openai_models(endpoint: str, timeout_seconds: int = 5) -> tuple[list[str], str | None]:
    models_url = _models_url_from_chat_endpoint(endpoint)
    try:
        with urllib.request.urlopen(models_url, timeout=timeout_seconds) as response:
            raw_body = response.read().decode("utf-8")
        decoded = json.loads(raw_body)
        models = [item.get("id") for item in decoded.get("data", []) if item.get("id")]
        return models, None
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, ValueError) as exc:
        return [], str(exc)


def discover_lm_studio_models(endpoint: str, timeout_seconds: int = 5) -> tuple[list[str], str | None]:
    return discover_openai_models(endpoint, timeout_seconds=timeout_seconds)


def resolve_model_identifier(
    available_models: list[str],
    preferred: str,
    *,
    fallbacks: list[str] | None = None,
    contains_patterns: list[str] | None = None,
) -> str:
    if not available_models:
        return preferred

    normalized_map = {model.lower(): model for model in available_models}
    direct_candidates = [preferred, *(fallbacks or [])]
    for candidate in direct_candidates:
        resolved = normalized_map.get(candidate.lower())
        if resolved:
            return resolved

    patterns = [item.lower() for item in (contains_patterns or []) if item]
    for model in available_models:
        lowered = model.lower()
        if any(pattern in lowered for pattern in patterns):
            return model

    return preferred


def resolve_lightonocr_model_identifier(available_models: list[str]) -> str:
    return resolve_model_identifier(
        available_models,
        LIGHTONOCR_MODEL_ID,
        fallbacks=LIGHTONOCR_MODEL_CANDIDATES,
        contains_patterns=["lightonocr"],
    )


def request_openai_chat_completion(
    messages: list[dict[str, object]],
    *,
    endpoint: str,
    model: str,
    timeout_seconds: int,
    temperature: float = 0.2,
    max_tokens: int | None = None,
) -> str | None:
    payload: dict[str, object] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw_body = response.read().decode("utf-8")
        decoded = json.loads(raw_body)
        return decoded["choices"][0]["message"]["content"]
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, ValueError):
        return None


def request_lm_studio_completion(
    messages: list[dict[str, object]],
    settings: AISettings,
    *,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int | None = None,
) -> str | None:
    return request_openai_chat_completion(
        messages,
        endpoint=settings.endpoint,
        model=model or settings.model,
        timeout_seconds=settings.timeout_seconds,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def request_document_model_completion(
    messages: list[dict[str, object]],
    settings: AISettings,
    *,
    backend: str,
    temperature: float = 0.2,
    max_tokens: int | None = None,
) -> str | None:
    if backend == "lightonocr":
        endpoint = settings.lightonocr_endpoint
        model = settings.lightonocr_model
    else:
        endpoint = settings.document_endpoint
        model = settings.document_model

    return request_openai_chat_completion(
        messages,
        endpoint=endpoint,
        model=model,
        timeout_seconds=settings.timeout_seconds,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def extract_json_object(raw_text: str) -> dict[str, object] | None:
    cleaned = raw_text.strip()
    if not cleaned:
        return None

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def generate_row_commentary(results_df: pd.DataFrame, settings: AISettings) -> pd.DataFrame:
    enriched = apply_machine_audit_layer(results_df)
    short_comments: list[str] = []
    full_comments: list[str] = []
    actions: list[str] = []
    llm_budget = settings.max_rows

    for _, row in enriched.iterrows():
        short_comment, full_comment, action = generate_template_commentary(row)
        if settings.use_lm_studio and row["status"] != "OK" and llm_budget > 0:
            llm_short, llm_full, llm_action = try_lm_studio_comment(row, settings)
            if llm_short and llm_full:
                short_comment = llm_short
                full_comment = llm_full
                action = llm_action or action
                llm_budget -= 1
        short_comments.append(short_comment)
        full_comments.append(full_comment)
        actions.append(action)

    enriched["short_ai_comment"] = short_comments
    enriched["full_ai_comment"] = full_comments
    enriched["ai_comment"] = short_comments
    enriched["recommended_action"] = actions
    return enriched


def apply_machine_audit_layer(results_df: pd.DataFrame) -> pd.DataFrame:
    enriched = results_df.copy()
    priorities: list[str] = []
    confidences: list[str] = []
    categories: list[str] = []
    top_factors: list[str] = []
    dominant_factors: list[str] = []
    primary_drivers: list[str] = []
    action_seeds: list[str] = []
    payloads: list[str] = []
    flags_payloads: list[str] = []

    for _, row in enriched.iterrows():
        payload = build_row_interpretation_payload(row)
        priorities.append(payload["priority"])
        confidences.append(payload["confidence"])
        categories.append(payload["risk_category"])
        top_factors.append("; ".join(payload["top_risk_factors"]))
        dominant_factors.append("; ".join(payload["dominant_risk_factors"]))
        primary_drivers.append(payload["primary_risk_driver"])
        action_seeds.append(payload["recommended_action_seed"])
        payloads.append(json.dumps(payload, ensure_ascii=False))
        flags_payloads.append(json.dumps(payload["flags"], ensure_ascii=False))

    enriched["priority"] = priorities
    enriched["confidence"] = confidences
    enriched["risk_category"] = categories
    enriched["top_risk_factors"] = top_factors
    enriched["dominant_risk_factors"] = dominant_factors
    enriched["primary_risk_driver"] = primary_drivers
    enriched["recommended_action_seed"] = action_seeds
    enriched["machine_flags_json"] = flags_payloads
    enriched["machine_payload_json"] = payloads
    return enriched


def build_row_interpretation_payload(row: pd.Series) -> dict[str, object]:
    reason_codes = _extract_reason_codes(row)
    ranked_codes = _rank_reason_codes(reason_codes)
    top_codes = ranked_codes[:3]
    dominant_codes = ranked_codes[:2]
    missing_fields = _extract_missing_fields(row)
    flags = {
        "missing_description": "description" in missing_fields or not str(row.get("description") or "").strip(),
        "missing_required_fields": bool(missing_fields),
        "high_amount": "amount_outlier" in reason_codes,
        "rare_counterparty": "atypical_counterparty" in reason_codes,
        "document_missing": "no_primary_document" in reason_codes or str(row.get("document_check_status")) == "MISSING",
        "document_mismatch": str(row.get("document_check_status")) == "MISMATCH",
        "document_not_provided": str(row.get("document_check_status")) == "NOT_PROVIDED",
        "anomaly_model_flag": "ml_anomaly" in reason_codes,
        "duplicate_document": "duplicate_document_number" in reason_codes,
        "amount_mismatch": "document_amount_mismatch" in reason_codes,
        "date_mismatch": "document_date_mismatch" in reason_codes,
        "counterparty_mismatch": "document_counterparty_mismatch" in reason_codes,
        "suspicious_description": "suspicious_description" in reason_codes,
        "document_incomplete": "document_incomplete" in reason_codes,
    }

    payload = {
        "operation_id": row.get("operation_id"),
        "status": row.get("status"),
        "risk_score": int(row.get("risk_score", 0) or 0),
        "priority": _build_priority(row, reason_codes, flags),
        "confidence": _build_confidence(row, reason_codes, flags),
        "risk_category": _build_risk_category(reason_codes, flags),
        "amount": _safe_numeric_value(row.get("amount")),
        "amount_display": row.get("amount_display"),
        "counterparty": row.get("counterparty"),
        "document_number": row.get("document_number"),
        "document_check_status": row.get("document_check_status"),
        "top_risk_factors": [_reason_label(code) for code in top_codes],
        "dominant_risk_factors": [_reason_label(code) for code in dominant_codes],
        "primary_risk_driver": _reason_label(top_codes[0]) if top_codes else "существенные отклонения не выявлены",
        "flags": flags,
        "recommended_action_seed": _build_recommended_action_seed(reason_codes, row),
    }
    return payload


def generate_template_commentary(row: pd.Series) -> tuple[str, str, str]:
    payload = _payload_from_row(row)
    recommended_action = payload["recommended_action_seed"]

    if payload["status"] == "OK":
        short_comment = (
            "Операция не содержит существенных признаков риска и не требует дополнительной эскалации."
        )
        full_comment = (
            f"Операция оценена как низкорисковая. Приоритет проверки — {payload['priority']}, "
            f"уверенность оценки — {payload['confidence']}. "
            f"{_build_document_status_sentence(payload)}"
        )
        return short_comment, full_comment, recommended_action

    risk_level_text = {
        "HIGH": "высокий приоритет проверки",
        "MEDIUM": "средний приоритет проверки",
        "LOW": "низкий приоритет проверки",
    }.get(payload["priority"], "средний приоритет проверки")

    dominant = payload["dominant_risk_factors"] or payload["top_risk_factors"]
    dominant_text = ", ".join(dominant[:2]) if dominant else "существенные факторы риска"
    additional_factors = payload["top_risk_factors"][2:]
    additional_text = ""
    if additional_factors:
        additional_text = f" Дополнительно влияние оказали {', '.join(additional_factors)}."

    short_comment = (
        f"Операция имеет {risk_level_text} из-за сочетания факторов: {dominant_text}."
    )
    full_comment = (
        f"Операция классифицирована как {payload['status']} с риск-баллом {payload['risk_score']}. "
        f"Приоритет проверки — {payload['priority']}, уверенность оценки — {payload['confidence']}. "
        f"Основной вклад в риск внесли {dominant_text}. "
        f"{_build_importance_sentence(payload)}"
        f"{additional_text} "
        f"Категория риска: {payload['risk_category']}."
    ).strip()
    return short_comment, full_comment, recommended_action


def try_lm_studio_comment(row: pd.Series, settings: AISettings) -> tuple[str | None, str | None, str | None]:
    payload = _payload_from_row(row)
    content = request_lm_studio_completion(
        [
            {
                "role": "system",
                "content": (
                    "Ты помощник по бухгалтерскому учету и аудиту. "
                    "Формируй деловой, краткий и профессиональный комментарий по операции. "
                    "Не повторяй причины дословно. Не перечисляй флаги механически. "
                    "Сначала дай вывод о риске, затем объясни, какие факторы на него повлияли, "
                    "затем укажи рекомендуемое действие. Не придумывай факты, которых нет во входных данных. "
                    "Пиши так, как будто это часть внутреннего аудиторского заключения. "
                    "Верни строго JSON с полями short_comment, full_comment, recommended_action."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Проанализируй операцию на основе уже рассчитанных факторов риска. "
                    "LLM не должна менять статус, риск-балл или машинные выводы, а только интерпретировать их.\n"
                    f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
                ),
            },
        ],
        settings,
        temperature=0.15,
        max_tokens=700,
    )
    parsed = extract_json_object(content or "")
    if not parsed:
        return None, None, None
    short_comment = _clean_output_text(parsed.get("short_comment"))
    full_comment = _clean_output_text(parsed.get("full_comment"))
    recommended_action = _clean_output_text(parsed.get("recommended_action"))
    if not short_comment or not full_comment:
        return None, None, None
    return short_comment, full_comment, recommended_action


def generate_dataset_conclusion(
    summary: dict[str, object],
    report_tables: dict[str, pd.DataFrame],
    settings: AISettings,
) -> str:
    dataset_payload = build_dataset_interpretation_payload(summary, report_tables)
    template = build_dataset_commentary_template(dataset_payload)
    if not settings.use_lm_studio:
        return template

    content = request_lm_studio_completion(
        [
            {
                "role": "system",
                "content": (
                    "Ты помощник внутреннего аудитора. "
                    "Сформируй итоговое заключение по набору операций деловым языком. "
                    "Не пересчитывай риски и не придумывай факты. "
                    "Структура ответа: 1. Итоговое заключение 2. Ключевые проблемы "
                    "3. Самые рискованные операции 4. Проблемные контрагенты "
                    "5. Документальное покрытие 6. Рекомендация."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(dataset_payload, ensure_ascii=False, indent=2),
            },
        ],
        settings,
        temperature=0.15,
        max_tokens=900,
    )
    if not content:
        return template
    return content.strip()


def build_dataset_interpretation_payload(
    summary: dict[str, object],
    report_tables: dict[str, pd.DataFrame],
) -> dict[str, object]:
    risk_register = report_tables.get("risk_register", pd.DataFrame())
    reason_summary = report_tables.get("reason_summary", pd.DataFrame())
    counterparty_summary = report_tables.get("counterparty_summary", pd.DataFrame())

    top_risk_operations = []
    if not risk_register.empty:
        for _, row in risk_register.head(5).iterrows():
            top_risk_operations.append(
                {
                    "operation_id": row.get("ID операции"),
                    "document_number": row.get("Номер документа"),
                    "counterparty": row.get("Контрагент"),
                    "amount": row.get("Сумма"),
                    "status": row.get("Статус"),
                    "risk_score": row.get("Риск-балл"),
                    "priority": row.get("Приоритет проверки"),
                    "risk_category": row.get("Категория риска"),
                    "main_driver": row.get("Ведущий фактор риска"),
                    "recommended_action": row.get("Рекомендуемое действие"),
                }
            )

    top_reasons = []
    if not reason_summary.empty:
        top_reasons = reason_summary.head(5).to_dict(orient="records")

    counterparties = []
    if not counterparty_summary.empty:
        counterparties = counterparty_summary.head(5).to_dict(orient="records")

    return {
        "total_operations": summary["total_operations"],
        "ok_count": summary["ok_count"],
        "warning_count": summary["warning_count"],
        "risk_count": summary["risk_count"],
        "average_risk_score": summary["average_risk_score"],
        "document_coverage": summary["document_coverage"],
        "document_coverage_percent": summary.get("document_coverage_percent"),
        "documents_expected_count": summary.get("documents_expected_count", 0),
        "document_not_provided_count": summary.get("document_not_provided_count", 0),
        "document_missing_count": summary.get("document_missing_count", 0),
        "document_mismatch_count": summary.get("document_mismatch_count", 0),
        "document_scope_note": summary.get("document_scope_note", ""),
        "top_reasons": top_reasons,
        "problematic_counterparties": summary.get("problematic_counterparties_text", "не выявлены"),
        "top_risk_operations": top_risk_operations,
        "counterparty_focus": counterparties,
        "priority_review_focus": summary.get("priority_review_focus", []),
        "recommendation_seed": summary.get("audit_recommendation"),
    }


def build_dataset_commentary_template(dataset_payload: dict[str, object]) -> str:
    top_reasons = dataset_payload.get("top_reasons", [])
    top_risk_operations = dataset_payload.get("top_risk_operations", [])
    priority_review_focus = dataset_payload.get("priority_review_focus", [])
    main_reasons_text = (
        ", ".join(item["Причина"] for item in top_reasons[:3] if item.get("Причина"))
        if top_reasons
        else "существенные отклонения не выявлены"
    )

    if top_risk_operations:
        top_risk_lines = []
        for item in top_risk_operations[:5]:
            top_risk_lines.append(
                f"- {item.get('operation_id')}: {item.get('counterparty')} | {item.get('amount')} | "
                f"{item.get('main_driver')} | {item.get('recommended_action')}"
            )
        top_risk_text = "\n".join(top_risk_lines)
    else:
        top_risk_text = "- Операции с повышенным риском не выявлены."

    try:
        coverage_percent = float(dataset_payload.get("document_coverage_percent"))
    except (TypeError, ValueError):
        coverage_percent = None

    if int(dataset_payload.get("documents_expected_count", 0) or 0) <= 0:
        coverage_text = (
            "Первичные документы не подавались на вход, поэтому автоматическая оценка "
            "документального покрытия и полноты сверки не выполнялась."
        )
    elif coverage_percent is not None and coverage_percent >= 80:
        coverage_text = (
            f"Покрытие документами составляет {dataset_payload['document_coverage']} "
            f"({coverage_percent}%), поэтому его можно оценить как высокое."
        )
    elif coverage_percent is not None and coverage_percent >= 50:
        coverage_text = (
            f"Покрытие документами составляет {dataset_payload['document_coverage']} "
            f"({coverage_percent}%), поэтому его можно оценить как умеренное."
        )
    else:
        coverage_text = (
            f"Покрытие документами составляет {dataset_payload['document_coverage']}; "
            "документальное покрытие остается ограниченным."
        )

    review_focus_text = ""
    if priority_review_focus:
        review_focus_text = "\n".join(
            f"- {item.get('label')}: {item.get('count')}" for item in priority_review_focus[:5]
        )
    else:
        review_focus_text = "- Дополнительных приоритетных блоков для ручной проверки не выявлено."

    return (
        "1. Итоговое заключение\n"
        f"Проанализировано {dataset_payload['total_operations']} операций: "
        f"OK — {dataset_payload['ok_count']}, WARNING — {dataset_payload['warning_count']}, "
        f"RISK — {dataset_payload['risk_count']}. Основные риски связаны с факторами: {main_reasons_text}.\n\n"
        "2. Ключевые проблемы\n"
        f"Наиболее заметные отклонения сосредоточены в блоках, связанных с документальным подтверждением, "
        f"нетипичными параметрами операций и качеством заполнения реквизитов. "
        f"Средний риск по выборке составляет {dataset_payload['average_risk_score']}.\n\n"
        "3. Самые рискованные операции\n"
        f"{top_risk_text}\n\n"
        "4. Проблемные контрагенты\n"
        f"Наибольшее число отклонений связано с контрагентами: {dataset_payload.get('problematic_counterparties', 'не выявлены')}.\n\n"
        "5. Документальное покрытие\n"
        f"{coverage_text}\n\n"
        "6. Что проверять в первую очередь\n"
        f"{review_focus_text}\n\n"
        "7. Рекомендация\n"
        f"{dataset_payload.get('recommendation_seed', 'Продолжить выборочную проверку операций с повышенным риском.')}"
    )


def _payload_from_row(row: pd.Series) -> dict[str, object]:
    payload_json = row.get("machine_payload_json")
    if isinstance(payload_json, str) and payload_json.strip():
        try:
            parsed = json.loads(payload_json)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return build_row_interpretation_payload(row)


def _extract_reason_codes(row: pd.Series) -> list[str]:
    raw_codes = str(row.get("reason_codes", "") or "")
    return [item.strip() for item in raw_codes.split("|") if item.strip()]


def _extract_missing_fields(row: pd.Series) -> list[str]:
    value = row.get("missing_required_fields", [])
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [item.strip() for item in text.strip("[]").replace("'", "").split(",") if item.strip()]


def _rank_reason_codes(reason_codes: list[str]) -> list[str]:
    def key(code: str) -> tuple[int, int]:
        return (-int(RISK_WEIGHTS.get(code, 0)), PRIORITY_ORDER.index(code) if code in PRIORITY_ORDER else 999)

    return sorted(reason_codes, key=key)


def _reason_label(code: str) -> str:
    return RISK_FACTOR_LABELS.get(code, code)


def _build_priority(row: pd.Series, reason_codes: list[str], flags: dict[str, bool]) -> str:
    risk_score = int(row.get("risk_score", 0) or 0)
    if (
        risk_score >= 70
        or "document_amount_mismatch" in reason_codes
        or (flags["document_missing"] and flags["high_amount"])
        or len(reason_codes) >= 4
    ):
        return "HIGH"
    if risk_score >= 30 or reason_codes:
        return "MEDIUM"
    return "LOW"


def _build_confidence(row: pd.Series, reason_codes: list[str], flags: dict[str, bool]) -> str:
    status = str(row.get("status", "") or "")
    if status == "OK":
        if str(row.get("document_check_status")) in {"OK", "PARTIAL"}:
            return "HIGH"
        if str(row.get("document_check_status")) == "NOT_PROVIDED":
            return "MEDIUM"
        return "MEDIUM"
    if len(reason_codes) >= 3 or ("ml_anomaly" in reason_codes and len(reason_codes) >= 2):
        return "HIGH"
    if len(reason_codes) >= 1:
        return "MEDIUM"
    return "LOW"


def _build_risk_category(reason_codes: list[str], flags: dict[str, bool]) -> str:
    categories: list[str] = []
    for code in reason_codes:
        category_code = RISK_CATEGORY_MAP.get(code)
        if not category_code:
            continue
        label = RISK_CATEGORY_LABELS[category_code]
        if label not in categories:
            categories.append(label)

    if flags["missing_description"] and RISK_CATEGORY_LABELS["description"] not in categories:
        categories.append(RISK_CATEGORY_LABELS["description"])

    if not categories:
        return "Низкий риск"
    if len(categories) == 1:
        return categories[0]
    if len(categories) == 2:
        return f"{categories[0]} и {categories[1]}"
    return "Комбинированный риск"


def _build_recommended_action_seed(reason_codes: list[str], row: pd.Series) -> str:
    for code in PRIORITY_ORDER:
        if code in reason_codes:
            return ACTION_SEEDS.get(code, "Провести дополнительную ручную проверку операции.")
    if str(row.get("status")) == "OK":
        return "Сохранить результат контроля и оставить операцию без дополнительной эскалации."
    return "Провести дополнительную ручную проверку операции."


def _build_document_status_sentence(payload: dict[str, object]) -> str:
    document_status = str(payload.get("document_check_status") or "")
    if document_status == "MISSING":
        return "Подтверждающий документ для автоматической сверки не найден."
    if document_status == "MISMATCH":
        return "При автоматической сверке выявлены расхождения с документом."
    if document_status == "NOT_PROVIDED":
        return "Документы для автоматической сверки не были загружены, поэтому оценка основана на учетных данных."
    if document_status == "PARTIAL":
        return "Документ найден, но извлеченные реквизиты неполные."
    return "Документальная сверка не выявила существенных отклонений."


def _build_importance_sentence(payload: dict[str, object]) -> str:
    document_status_sentence = _build_document_status_sentence(payload)
    category = str(payload.get("risk_category") or "").lower()
    if "документарный" in category:
        impact = "Это повышает вероятность ошибки оформления либо недостаточного документального обоснования."
    elif "суммы" in category:
        impact = "Это требует подтверждения экономической обоснованности и корректности отраженной суммы."
    elif "контрагента" in category:
        impact = "Это требует дополнительной проверки правомерности выбора контрагента и корректности реквизитов."
    elif "описания" in category:
        impact = "Это снижает прозрачность хозяйственного содержания операции и усложняет внутренний контроль."
    else:
        impact = "Сочетание факторов повышает вероятность ошибки учета и требует дополнительной проверки."
    return f"{document_status_sentence} {impact}"


def _clean_output_text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _safe_numeric_value(value: object) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _models_url_from_chat_endpoint(endpoint: str) -> str:
    cleaned = endpoint.strip().rstrip("/")
    if cleaned.endswith("/chat/completions"):
        cleaned = cleaned[: -len("/chat/completions")] + "/models"
    elif cleaned.endswith("/completions"):
        cleaned = cleaned[: -len("/completions")] + "/models"
    elif cleaned.endswith("/v1"):
        cleaned = cleaned + "/models"
    else:
        parts = urlsplit(cleaned)
        path = parts.path.rstrip("/")
        if path.endswith("/chat"):
            path = path[: -len("/chat")] + "/models"
        elif path.endswith("/v1/models"):
            path = path
        elif "/v1/" not in path:
            path = path + "/v1/models"
        else:
            path = path.rsplit("/", 1)[0] + "/models"
        cleaned = urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))
    return cleaned
