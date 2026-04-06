from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
import pytesseract
from pypdf import PdfReader

from shingadip.ai import AISettings, extract_json_object, request_lm_studio_completion
from shingadip.config import SUPPORTED_DOCUMENT_TYPES
from shingadip.data_processing import clean_text_value, parse_date_value, parse_numeric_value, save_uploaded_file


@dataclass(slots=True)
class DocumentExtraction:
    file_name: str
    stored_path: str
    extraction_method: str
    extracted_text: str
    document_number: str | None
    document_date: pd.Timestamp | None
    counterparty: str | None
    amount: float | None
    currency: str | None
    description: str | None
    warnings: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_record(self) -> dict[str, object]:
        return {
            "file_name": self.file_name,
            "stored_path": self.stored_path,
            "extraction_method": self.extraction_method,
            "document_number": self.document_number,
            "document_date": self.document_date.strftime("%Y-%m-%d") if self.document_date is not None else None,
            "counterparty": self.counterparty,
            "amount": self.amount,
            "currency": self.currency,
            "description": self.description,
            "confidence": round(self.confidence, 2),
            "warnings": "; ".join(self.warnings) if self.warnings else "",
            "extracted_text": self.extracted_text,
        }


@dataclass(slots=True)
class LLMDocumentResult:
    fields: dict[str, object]
    extracted_text: str
    extraction_method: str
    warnings: list[str] = field(default_factory=list)


def extract_documents(
    sources: list[Any],
    target_dir: Path,
    ai_settings: AISettings | None = None,
) -> list[DocumentExtraction]:
    extracted: list[DocumentExtraction] = []
    remaining_ai_calls = 0
    if ai_settings and ai_settings.use_vision_for_documents:
        remaining_ai_calls = ai_settings.max_document_ai_calls

    for source in sources:
        stored_path = save_uploaded_file(source, target_dir)
        if stored_path.suffix.lower() not in SUPPORTED_DOCUMENT_TYPES:
            continue

        document_ai_settings = ai_settings if remaining_ai_calls > 0 else None
        extracted.append(extract_document(stored_path, document_ai_settings))
        if document_ai_settings is not None:
            remaining_ai_calls -= 1
    return extracted


def extract_document(path: Path, ai_settings: AISettings | None = None) -> DocumentExtraction:
    warnings: list[str] = []
    suffix = path.suffix.lower()
    base_text = _extract_text_from_pdf(path) if suffix == ".pdf" else ""

    llm_result: LLMDocumentResult | None = None
    if ai_settings and ai_settings.use_vision_for_documents:
        llm_result = _extract_document_with_lm_studio(path, ai_settings, base_text)
        if llm_result:
            supplemental_text = clean_text_value(base_text) or clean_text_value(llm_result.extracted_text) or ""
            if supplemental_text:
                supplemental_fields = _parse_document_fields(supplemental_text, path.name)
                llm_result = LLMDocumentResult(
                    fields=_merge_document_fields(llm_result.fields, supplemental_fields),
                    extracted_text=clean_text_value(llm_result.extracted_text) or supplemental_text,
                    extraction_method=llm_result.extraction_method,
                    warnings=llm_result.warnings,
                )
            warnings.extend(llm_result.warnings)
            llm_confidence = _field_coverage(llm_result.fields)
            if llm_confidence >= 0.5 and clean_text_value(llm_result.extracted_text):
                return _build_document_extraction(
                    path=path,
                    extraction_method=llm_result.extraction_method,
                    extracted_text=llm_result.extracted_text,
                    fields=llm_result.fields,
                    warnings=warnings,
                )

    if suffix == ".pdf":
        extraction_method = "pdf_text"
        text = base_text
    else:
        extraction_method = "ocr_image"
        text = _extract_text_from_image(path, warnings)

    if not text.strip():
        warnings.append("Не удалось извлечь текст из документа.")

    fallback_fields = _parse_document_fields(text, path.name)
    if llm_result:
        merged_fields = _merge_document_fields(llm_result.fields, fallback_fields)
        merged_text = clean_text_value(llm_result.extracted_text) or text
        merged_method = (
            llm_result.extraction_method
            if _field_coverage(llm_result.fields) >= _field_coverage(fallback_fields)
            else f"{llm_result.extraction_method}+{extraction_method}"
        )
        return _build_document_extraction(
            path=path,
            extraction_method=merged_method,
            extracted_text=merged_text,
            fields=merged_fields,
            warnings=warnings,
        )

    return _build_document_extraction(
        path=path,
        extraction_method=extraction_method,
        extracted_text=text,
        fields=fallback_fields,
        warnings=warnings,
    )


def _build_document_extraction(
    path: Path,
    extraction_method: str,
    extracted_text: str,
    fields: dict[str, object],
    warnings: list[str],
) -> DocumentExtraction:
    confidence = _field_coverage(fields)
    return DocumentExtraction(
        file_name=path.name,
        stored_path=str(path),
        extraction_method=extraction_method,
        extracted_text=extracted_text,
        document_number=fields["document_number"],
        document_date=fields["document_date"],
        counterparty=fields["counterparty"],
        amount=fields["amount"],
        currency=fields["currency"],
        description=fields["description"],
        warnings=warnings,
        confidence=confidence,
    )


def _extract_document_with_lm_studio(
    path: Path,
    ai_settings: AISettings,
    source_text: str,
) -> LLMDocumentResult | None:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        if not source_text.strip():
            return None
        prompt = (
            "Extract key accounting document fields from the following text. "
            "Return only valid JSON with keys: "
            "document_number, document_date, counterparty, amount, currency, description, extracted_text. "
            "Use YYYY-MM-DD for document_date when possible. Use null for missing values. "
            "extracted_text must contain a compact transcription of important lines.\n\n"
            f"FILE: {path.name}\n"
            f"TEXT:\n{source_text[:7000]}"
        )
        messages: list[dict[str, object]] = [
            {
                "role": "system",
                "content": (
                    "You extract structured fields from invoices, acts, receipts, and similar accounting documents. "
                    "Answer strictly with a JSON object."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
        extraction_method = "lm_studio_text_document"
    else:
        prompt = (
            "Extract key accounting document fields from this image. "
            "Return only valid JSON with keys: "
            "document_number, document_date, counterparty, amount, currency, description, extracted_text. "
            "Use YYYY-MM-DD for document_date when possible. Use null for missing values. "
            "extracted_text must contain a compact transcription of important visible text."
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a multimodal accounting document extraction module. "
                    "Read the document image carefully and answer strictly with a JSON object."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _image_to_data_url(path)}},
                ],
            },
        ]
        extraction_method = "lm_studio_vision_document"

    content = request_lm_studio_completion(
        messages,
        ai_settings,
        model=ai_settings.vision_model,
        temperature=0.1,
        max_tokens=900,
    )
    if not content:
        return None

    parsed = extract_json_object(content)
    if not parsed:
        return LLMDocumentResult(
            fields=_empty_document_fields(),
            extracted_text="",
            extraction_method=extraction_method,
            warnings=["Vision-модель не вернула корректный JSON, применен резервный разбор."],
        )

    fields, extracted_text = _normalize_llm_document_payload(parsed, source_text)
    return LLMDocumentResult(
        fields=fields,
        extracted_text=extracted_text,
        extraction_method=extraction_method,
    )


def _empty_document_fields() -> dict[str, object]:
    return {
        "document_number": None,
        "document_date": None,
        "counterparty": None,
        "amount": None,
        "currency": None,
        "description": None,
    }


def _normalize_llm_document_payload(
    payload: dict[str, object],
    fallback_text: str,
) -> tuple[dict[str, object], str]:
    document_number = _pick_payload_value(
        payload,
        "document_number",
        "doc_number",
        "invoice_number",
        "number",
    )
    document_date = _pick_payload_value(payload, "document_date", "date", "invoice_date")
    counterparty = _pick_payload_value(payload, "counterparty", "seller", "vendor", "supplier")
    amount_raw = _pick_payload_value(payload, "amount", "sum", "total")
    currency_raw = _pick_payload_value(payload, "currency")
    description = _pick_payload_value(payload, "description", "purpose", "summary")
    extracted_text = _pick_payload_value(
        payload,
        "extracted_text",
        "document_text",
        "text",
        "transcription",
        "ocr_text",
    )

    amount_text = _as_text(amount_raw)
    currency_text = _as_text(currency_raw)
    if amount_text and not currency_text:
        matched_currency = re.search(r"\b(KZT|USD|EUR|RUB|₸)\b", amount_text, flags=re.IGNORECASE)
        if matched_currency:
            currency_text = matched_currency.group(1)

    fields = {
        "document_number": clean_text_value(_as_text(document_number)),
        "document_date": parse_date_value(_as_text(document_date)),
        "counterparty": clean_text_value(_as_text(counterparty)),
        "amount": parse_numeric_value(amount_text),
        "currency": _normalize_currency(currency_text),
        "description": clean_text_value(_as_text(description)),
    }
    normalized_text = clean_text_value(_as_text(extracted_text)) or clean_text_value(fallback_text) or ""
    return fields, normalized_text


def _pick_payload_value(payload: dict[str, object], *keys: str) -> object | None:
    for key in keys:
        if key in payload:
            value = payload[key]
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            return value
    return None


def _as_text(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return " ".join(str(item) for item in value if item is not None)
    if isinstance(value, dict):
        return " ".join(f"{key}: {item}" for key, item in value.items())
    return str(value)


def _normalize_currency(value: str | None) -> str | None:
    cleaned = clean_text_value(value)
    if not cleaned:
        return None
    return cleaned.upper().replace("₸", "KZT")


def _merge_document_fields(primary: dict[str, object], fallback: dict[str, object]) -> dict[str, object]:
    merged = {}
    for key in _empty_document_fields():
        primary_value = primary.get(key)
        merged[key] = primary_value if primary_value not in (None, "") else fallback.get(key)
    return merged


def _field_coverage(fields: dict[str, object]) -> float:
    meaningful = sum(1 for value in fields.values() if value not in (None, ""))
    return meaningful / max(len(fields), 1)


def _extract_text_from_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages[:5]:
        page_text = page.extract_text() or ""
        if page_text.strip():
            parts.append(page_text)
    return "\n".join(parts).strip()


def _extract_text_from_image(path: Path, warnings: list[str]) -> str:
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        warnings.append("Tesseract OCR не найден в системе.")
        return ""

    try:
        image = Image.open(path)
        text = pytesseract.image_to_string(image, lang="rus+eng")
        return text.strip()
    except Exception as exc:  # pragma: no cover - depends on local OCR setup
        warnings.append(f"OCR завершился ошибкой: {exc}")
        return ""


def _image_to_data_url(path: Path) -> str:
    image = Image.open(path).convert("RGB")
    image.thumbnail((1600, 1600))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=88, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _parse_document_fields(text: str, file_name: str) -> dict[str, object]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    joined = "\n".join(lines)
    file_stem = Path(file_name).stem

    document_number = _extract_by_patterns(
        joined,
        [
            r"(?:document number|document no|invoice number|doc number|номер документа|номер|№)\s*[:#]?\s*([A-Za-zА-Яа-я0-9/_\-]+)",
            r"\b([A-Z]{2,5}-\d{3,6})\b",
        ],
    )
    if not document_number:
        fallback_number = _extract_by_patterns(file_stem, [r"([A-Za-z]{2,5}[_\-]?\d{3,6})"])
        document_number = fallback_number.replace("_", "-") if fallback_number else None

    date_raw = _extract_by_patterns(
        joined,
        [
            r"(?:document date|date|дата документа|дата)\s*[:#]?\s*([0-9]{4}-[0-9]{2}-[0-9]{2})",
            r"(?:document date|date|дата документа|дата)\s*[:#]?\s*([0-9]{2}[./][0-9]{2}[./][0-9]{4})",
            r"\b([0-9]{4}-[0-9]{2}-[0-9]{2})\b",
            r"\b([0-9]{2}[./][0-9]{2}[./][0-9]{4})\b",
        ],
    )
    document_date = parse_date_value(date_raw)

    counterparty = _extract_by_patterns(
        joined,
        [
            r"(?:seller|vendor|supplier|counterparty|продавец|поставщик|контрагент)\s*[:#]?\s*(.+?)(?=\s+(?:amount|total|sum|итого|сумма|description|purpose|назначение|описание|document date|date|дата)\b|$)",
        ],
    )
    if counterparty:
        counterparty = clean_text_value(counterparty.splitlines()[0])

    amount_text = _extract_by_patterns(
        joined,
        [
            r"(?:amount|total|sum|итого|сумма)\s*[:#]?\s*([0-9\s,.\-]+(?:\s*(?:KZT|USD|EUR|RUB|₸))?)",
            r"\b([0-9]{2,}(?:[ ,.][0-9]{3})*(?:[.,][0-9]{1,2})?)\s*(KZT|USD|EUR|RUB|₸)?\b",
        ],
        with_groups=True,
    )
    amount = None
    currency = None
    if isinstance(amount_text, tuple):
        amount = parse_numeric_value(amount_text[0])
        currency = clean_text_value(amount_text[1]) if len(amount_text) > 1 else None
    elif isinstance(amount_text, str):
        amount = parse_numeric_value(amount_text)

    if currency:
        currency = currency.upper().replace("₸", "KZT")

    if not currency:
        currency = _extract_by_patterns(joined, [r"\b(KZT|USD|EUR|RUB|₸)\b"])
        if currency:
            currency = currency.upper().replace("₸", "KZT")

    description = _extract_by_patterns(
        joined,
        [
            r"(?:description|purpose|назначение|описание)\s*[:#]?\s*(.+)",
        ],
    )
    if description:
        description = clean_text_value(description.splitlines()[0])

    return {
        "document_number": clean_text_value(document_number),
        "document_date": document_date,
        "counterparty": clean_text_value(counterparty),
        "amount": amount,
        "currency": currency,
        "description": description,
    }


def _extract_by_patterns(text: str, patterns: list[str], with_groups: bool = False) -> str | tuple[str, ...] | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if not match:
            continue
        if with_groups:
            return tuple(group for group in match.groups() if group is not None)
        if match.groups():
            return match.group(1).strip()
        return match.group(0).strip()
    return None
