from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

import pandas as pd


@dataclass(slots=True)
class AISettings:
    use_lm_studio: bool = False
    endpoint: str = "http://127.0.0.1:8080/v1/chat/completions"
    model: str = "qwen2.5-7b-instruct"
    timeout_seconds: int = 20
    max_rows: int = 8
    use_vision_for_documents: bool = False
    vision_model: str = "qwen2.5-vl-7b-instruct"
    max_document_ai_calls: int = 10


def discover_lm_studio_models(endpoint: str, timeout_seconds: int = 5) -> tuple[list[str], str | None]:
    models_url = _models_url_from_chat_endpoint(endpoint)
    try:
        with urllib.request.urlopen(models_url, timeout=timeout_seconds) as response:
            raw_body = response.read().decode("utf-8")
        decoded = json.loads(raw_body)
        models = [item.get("id") for item in decoded.get("data", []) if item.get("id")]
        return models, None
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, ValueError) as exc:
        return [], str(exc)


def request_lm_studio_completion(
    messages: list[dict[str, object]],
    settings: AISettings,
    *,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int | None = None,
) -> str | None:
    payload: dict[str, object] = {
        "model": model or settings.model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    request = urllib.request.Request(
        settings.endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=settings.timeout_seconds) as response:
            raw_body = response.read().decode("utf-8")
        decoded = json.loads(raw_body)
        return decoded["choices"][0]["message"]["content"]
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, ValueError):
        return None


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
    enriched = results_df.copy()
    comments: list[str] = []
    actions: list[str] = []
    llm_budget = settings.max_rows

    for _, row in enriched.iterrows():
        comment, action = generate_template_comment(row)
        if settings.use_lm_studio and row["status"] != "OK" and llm_budget > 0:
            llm_comment, llm_action = try_lm_studio_comment(row, settings)
            if llm_comment:
                comment = llm_comment
                action = llm_action or action
                llm_budget -= 1
        comments.append(comment)
        actions.append(action)

    enriched["ai_comment"] = comments
    enriched["recommended_action"] = actions
    return enriched


def generate_template_comment(row: pd.Series) -> tuple[str, str]:
    status = row["status"]
    if status == "OK":
        return (
            "Операция выглядит согласованной с учетом доступных данных. Существенных признаков повышенного риска не обнаружено.",
            "Сохранить результат проверки и оставить операцию в выборке без дополнительной эскалации.",
        )

    reasons = row["reason_details"]
    if "Сумма операции не совпадает" in reasons:
        action = "Сверить сумму по первичному документу, регистру учета и проводке, затем проверить корректность ручного ввода."
    elif "Номер документа встречается" in reasons:
        action = "Проверить, не было ли повторного отражения одной и той же операции или дублирования загрузки."
    elif "Не найден подходящий первичный документ" in reasons:
        action = "Запросить подтверждающий документ и проверить комплектность первичной документации."
    elif "Дата операции не совпадает" in reasons:
        action = "Проверить дату признания операции и дату документа, а также основание для расхождения."
    else:
        action = "Провести выборочную ручную проверку реквизитов, основания операции и маршрута согласования."

    comment = (
        f"Операция получила статус {status} с риск-баллом {row['risk_score']}. "
        f"Основные причины: {reasons}"
    )
    return comment, action


def try_lm_studio_comment(row: pd.Series, settings: AISettings) -> tuple[str | None, str | None]:
    content = request_lm_studio_completion(
        [
            {
                "role": "system",
                "content": "Ты внутренний аудитор. Отвечай строго JSON-объектом.",
            },
            {
                "role": "user",
                "content": (
                    "Ты помощник аудитора. Верни JSON с полями comment и action. "
                    "Дай краткое пояснение на русском языке без лишней воды.\n"
                    f"Статус: {row['status']}\n"
                    f"Риск-балл: {row['risk_score']}\n"
                    f"Документ: {row.get('document_number')}\n"
                    f"Контрагент: {row.get('counterparty')}\n"
                    f"Сумма: {row.get('amount')}\n"
                    f"Причины: {row.get('reason_details')}\n"
                    f"Сопоставление с документом: {row.get('document_check_status')}\n"
                ),
            },
        ],
        settings,
        temperature=0.2,
        max_tokens=500,
    )
    parsed = extract_json_object(content or "")
    if not parsed:
        return None, None
    return parsed.get("comment"), parsed.get("action")


def generate_dataset_conclusion(summary: dict[str, object], settings: AISettings) -> str:
    template = (
        f"Проанализировано {summary['total_operations']} операций. "
        f"Высокий риск присвоен {summary['risk_count']} операциям, предупреждения получены по {summary['warning_count']} операциям. "
        f"Наиболее частые причины отклонений: {summary['top_reason_text']}. "
        "Рекомендуется вручную проверить операции со статусами WARNING и RISK и подтвердить их первичными документами."
    )
    if not settings.use_lm_studio:
        return template

    content = request_lm_studio_completion(
        [
            {
                "role": "system",
                "content": "Ты аудитор. Дай краткое итоговое заключение на русском языке в 2-3 предложениях.",
            },
            {
                "role": "user",
                "content": json.dumps(summary, ensure_ascii=False),
            },
        ],
        settings,
        temperature=0.2,
        max_tokens=400,
    )
    if not content:
        return template
    return content.strip()


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
