from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = PROJECT_ROOT / "workspace"
REPORT_DIR = WORKSPACE_DIR / "reports"
DEMO_DATA_DIR = PROJECT_ROOT / "sample_data"

DEFAULT_TEXT_ENDPOINT = "http://127.0.0.1:8080/v1/chat/completions"
DEFAULT_DOCUMENT_ENDPOINT = DEFAULT_TEXT_ENDPOINT
DEFAULT_LIGHTONOCR_ENDPOINT = DEFAULT_TEXT_ENDPOINT

TEXT_MODEL_ID = "qwen2.5-7b-instruct"
VISION_MODEL_ID = "qwen2.5-vl-7b-instruct"
LIGHTONOCR_MODEL_ID = "lightonocr-2-1b"
LIGHTONOCR_MODEL_CANDIDATES = [
    "lightonocr-2-1b",
    "lightonai/LightOnOCR-2-1B",
]

DEFAULT_DOCUMENT_ANALYSIS_MODE = "auto"
DOCUMENT_ANALYSIS_MODE_OPTIONS = [
    ("auto", "Auto"),
    ("pdf_text", "PDF text"),
    ("tesseract", "Tesseract"),
    ("vision_model", "Vision model"),
    ("lightonocr", "LightOnOCR"),
]
DOCUMENT_AI_MODES = {"vision_model", "lightonocr"}
LIGHTONOCR_RENDER_DPI = 200
LIGHTONOCR_TARGET_MAX_DIMENSION = 1540
LIGHTONOCR_MAX_PAGES = 3

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
    "duplicate_document_number": 18,
    "amount_outlier": 20,
    "suspicious_description": 10,
    "atypical_counterparty": 6,
    "no_primary_document": 8,
    "document_amount_mismatch": 28,
    "document_date_mismatch": 14,
    "document_counterparty_mismatch": 14,
    "document_incomplete": 6,
    "ml_anomaly": 10,
}


def ensure_workspace() -> None:
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
