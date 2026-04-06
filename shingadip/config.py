from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = PROJECT_ROOT / "workspace"
REPORT_DIR = WORKSPACE_DIR / "reports"
DEMO_DATA_DIR = PROJECT_ROOT / "sample_data"

SUPPORTED_DOCUMENT_TYPES = {".pdf", ".png", ".jpg", ".jpeg"}

CANONICAL_FIELDS = [
    "operation_date",
    "document_number",
    "document_type",
    "counterparty",
    "amount",
    "currency",
    "vat",
    "account",
    "description",
    "responsible_employee",
]

COLUMN_ALIASES: dict[str, list[str]] = {
    "operation_date": [
        "дата операции",
        "дата",
        "operation date",
        "date",
        "transaction date",
    ],
    "document_number": [
        "номер документа",
        "номер",
        "document number",
        "doc number",
        "invoice number",
        "document no",
    ],
    "document_type": [
        "тип документа",
        "document type",
        "type",
        "вид документа",
    ],
    "counterparty": [
        "контрагент",
        "продавец",
        "поставщик",
        "counterparty",
        "vendor",
        "supplier",
        "seller",
    ],
    "amount": [
        "сумма",
        "amount",
        "total",
        "итого",
    ],
    "currency": [
        "валюта",
        "currency",
    ],
    "vat": [
        "ндс",
        "vat",
        "tax",
    ],
    "account": [
        "счет учета",
        "счет",
        "account",
        "gl account",
    ],
    "description": [
        "описание операции",
        "описание",
        "назначение",
        "description",
        "purpose",
        "comment",
    ],
    "responsible_employee": [
        "ответственный сотрудник",
        "ответственный",
        "employee",
        "responsible employee",
        "owner",
    ],
}

REQUIRED_FIELDS = [
    "operation_date",
    "document_number",
    "counterparty",
    "amount",
    "description",
]

SUSPICIOUS_DESCRIPTION_KEYWORDS = [
    "срочно",
    "корректировка",
    "вручную",
    "прочее",
    "разное",
    "неизвест",
    "без договора",
    "налич",
    "manual",
    "urgent",
    "misc",
]

RISK_WEIGHTS = {
    "missing_required_fields": 18,
    "duplicate_document_number": 22,
    "amount_outlier": 20,
    "suspicious_description": 12,
    "atypical_counterparty": 10,
    "no_primary_document": 15,
    "document_amount_mismatch": 28,
    "document_date_mismatch": 18,
    "document_counterparty_mismatch": 18,
    "document_incomplete": 8,
    "ml_anomaly": 14,
}


def ensure_workspace() -> None:
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

