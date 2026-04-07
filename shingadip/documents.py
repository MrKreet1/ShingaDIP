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

from shingadip.ai import AISettings, extract_json_object, request_document_model_completion
from shingadip.config import (
    DOCUMENT_AI_MODES,
    LIGHTONOCR_MAX_PAGES,
    LIGHTONOCR_RENDER_DPI,
    LIGHTONOCR_TARGET_MAX_DIMENSION,
    SUPPORTED_DOCUMENT_TYPES,
)
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
class DocumentParseResult:
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
    remaining_ai_calls = ai_settings.max_document_ai_calls if _can_use_document_ai(ai_settings) else 0

    for source in sources:
        stored_path = save_uploaded_file(source, target_dir)
        if stored_path.suffix.lower() not in SUPPORTED_DOCUMENT_TYPES:
            continue

        document_ai_settings = ai_settings if remaining_ai_calls > 0 else None
        extracted_document = extract_document(stored_path, document_ai_settings)
        extracted.append(extracted_document)
        if document_ai_settings is not None and extracted_document.extraction_method.startswith(
            ("lightonocr", "vision_model")
        ):
            remaining_ai_calls -= 1
    return extracted


def extract_document(path: Path, ai_settings: AISettings | None = None) -> DocumentExtraction:
    warnings: list[str] = []
    suffix = path.suffix.lower()
    requested_mode = _requested_document_mode(ai_settings)

    if requested_mode in DOCUMENT_AI_MODES and not _can_use_document_ai(ai_settings):
        warnings.append("AI-режим документов отключен, будет использован резервный способ извлечения.")

    try:
        base_text = _extract_text_from_pdf(path) if suffix == ".pdf" else ""
    except Exception as exc:
        base_text = ""
        warnings.append(f"Не удалось извлечь текстовый слой PDF: {exc}")

    strategy_order = _build_strategy_order(path, requested_mode, ai_settings, base_text)
    best_result: DocumentParseResult | None = None
    best_score = -1.0

    for strategy in strategy_order:
        result = _run_strategy(strategy, path, ai_settings, base_text)
        if result is None:
            continue

        warnings.extend(result.warnings)
        score = _result_score(result)
        if score > best_score:
            best_result = result
            best_score = score

        if strategy != "fallback" and _result_is_usable(result):
            return _build_document_extraction(
                path=path,
                extraction_method=result.extraction_method,
                extracted_text=result.extracted_text,
                fields=result.fields,
                warnings=_deduplicate_warnings(warnings),
            )

    if best_result is not None:
        return _build_document_extraction(
            path=path,
            extraction_method=best_result.extraction_method,
            extracted_text=best_result.extracted_text,
            fields=best_result.fields,
            warnings=_deduplicate_warnings(warnings),
        )

    warnings.append("Не удалось извлечь текст и реквизиты из документа.")
    return _build_document_extraction(
        path=path,
        extraction_method="fallback",
        extracted_text=clean_text_value(base_text) or "",
        fields=_parse_document_fields(base_text, path.name) if base_text else _empty_document_fields(),
        warnings=_deduplicate_warnings(warnings),
    )


def _can_use_document_ai(ai_settings: AISettings | None) -> bool:
    return bool(ai_settings and ai_settings.use_document_model and ai_settings.max_document_ai_calls > 0)


def _requested_document_mode(ai_settings: AISettings | None) -> str:
    if ai_settings is None:
        return "auto"
    return (clean_text_value(ai_settings.document_analysis_mode) or "auto").lower()


def _document_mode_can_call_ai(mode: str) -> bool:
    return mode in {"auto", *DOCUMENT_AI_MODES}


def _build_strategy_order(
    path: Path,
    requested_mode: str,
    ai_settings: AISettings | None,
    base_text: str,
) -> list[str]:
    suffix = path.suffix.lower()
    ai_enabled = _can_use_document_ai(ai_settings)
    base_text_usable = _has_meaningful_text(base_text)

    if requested_mode == "pdf_text":
        return ["pdf_text", "lightonocr", "vision_model", "tesseract", "fallback"] if ai_enabled else [
            "pdf_text",
            "tesseract",
            "fallback",
        ]

    if requested_mode == "tesseract":
        return ["tesseract", "pdf_text", "lightonocr", "vision_model", "fallback"] if ai_enabled else [
            "tesseract",
            "pdf_text",
            "fallback",
        ]

    if requested_mode == "vision_model":
        return ["vision_model", "pdf_text", "tesseract", "fallback"] if ai_enabled else [
            "pdf_text",
            "tesseract",
            "fallback",
        ]

    if requested_mode == "lightonocr":
        return ["lightonocr", "pdf_text", "tesseract", "vision_model", "fallback"] if ai_enabled else [
            "pdf_text",
            "tesseract",
            "fallback",
        ]

    if suffix == ".pdf" and base_text_usable:
        return ["pdf_text", "vision_model", "lightonocr", "tesseract", "fallback"] if ai_enabled else [
            "pdf_text",
            "tesseract",
            "fallback",
        ]

    if suffix == ".pdf":
        return ["lightonocr", "vision_model", "pdf_text", "tesseract", "fallback"] if ai_enabled else [
            "pdf_text",
            "tesseract",
            "fallback",
        ]

    return ["lightonocr", "vision_model", "tesseract", "fallback"] if ai_enabled else ["tesseract", "fallback"]


def _run_strategy(
    strategy: str,
    path: Path,
    ai_settings: AISettings | None,
    base_text: str,
) -> DocumentParseResult | None:
    if strategy == "pdf_text":
        return _extract_with_pdf_text(path, base_text)
    if strategy == "tesseract":
        return _extract_with_tesseract(path)
    if strategy == "vision_model":
        return _extract_with_vision_model(path, ai_settings, base_text)
    if strategy == "lightonocr":
        return _extract_with_lightonocr(path, ai_settings)
    if strategy == "fallback":
        fallback_text = clean_text_value(base_text) or ""
        return DocumentParseResult(
            fields=_parse_document_fields(fallback_text, path.name) if fallback_text else _empty_document_fields(),
            extracted_text=fallback_text,
            extraction_method="fallback",
            warnings=["Использован резервный режим без AI/OCR."],
        )
    return None


def _extract_with_pdf_text(path: Path, base_text: str) -> DocumentParseResult:
    if path.suffix.lower() != ".pdf":
        return DocumentParseResult(
            fields=_empty_document_fields(),
            extracted_text="",
            extraction_method="pdf_text",
            warnings=["PDF text extraction неприменим к изображению."],
        )

    text = clean_text_value(base_text) or ""
    warnings: list[str] = []
    if not text:
        warnings.append("В PDF не найден пригодный текстовый слой.")
    return DocumentParseResult(
        fields=_parse_document_fields(text, path.name) if text else _empty_document_fields(),
        extracted_text=text,
        extraction_method="pdf_text",
        warnings=warnings,
    )


def _extract_with_tesseract(path: Path) -> DocumentParseResult:
    warnings: list[str] = []
    images = _load_document_images(path, warnings, max_pages=LIGHTONOCR_MAX_PAGES)
    if not images:
        return DocumentParseResult(
            fields=_empty_document_fields(),
            extracted_text="",
            extraction_method="tesseract",
            warnings=warnings,
        )

    page_texts = []
    for image in images:
        text = _run_tesseract(image, warnings)
        if text:
            page_texts.append(text)

    combined_text = clean_text_value("\n\n".join(page_texts)) or ""
    if not combined_text:
        warnings.append("Tesseract OCR не вернул пригодный текст.")

    return DocumentParseResult(
        fields=_parse_document_fields(combined_text, path.name) if combined_text else _empty_document_fields(),
        extracted_text=combined_text,
        extraction_method="tesseract",
        warnings=warnings,
    )


def _extract_with_lightonocr(path: Path, ai_settings: AISettings | None) -> DocumentParseResult:
    if not _can_use_document_ai(ai_settings):
        return DocumentParseResult(
            fields=_empty_document_fields(),
            extracted_text="",
            extraction_method="lightonocr",
            warnings=["LightOnOCR недоступен: AI-режим документов выключен."],
        )

    warnings: list[str] = []
    images = _load_document_images(path, warnings, max_pages=LIGHTONOCR_MAX_PAGES)
    if not images:
        warnings.append("LightOnOCR пропущен: не удалось подготовить изображение документа.")
        return DocumentParseResult(
            fields=_empty_document_fields(),
            extracted_text="",
            extraction_method="lightonocr",
            warnings=warnings,
        )

    page_texts: list[str] = []
    for page_index, image in enumerate(images, start=1):
        messages = [
            {
                "role": "system",
                "content": "You are an OCR engine. Return only the document transcription as plain text.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Transcribe this accounting document page into clean plain text. "
                            "Preserve line breaks, dates, numbers, totals, invoice identifiers, and counterparties. "
                            "Return plain text only."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": _pil_image_to_data_url(image, image_format="PNG")},
                    },
                ],
            },
        ]
        content = request_document_model_completion(
            messages,
            ai_settings,
            backend="lightonocr",
            temperature=0.0,
            max_tokens=2400,
        )
        text = clean_text_value(content) or ""
        if text:
            page_texts.append(text)
        else:
            warnings.append(f"LightOnOCR не вернул текст для страницы {page_index}.")

    combined_text = clean_text_value("\n\n".join(page_texts)) or ""
    if not combined_text:
        warnings.append("LightOnOCR недоступен или вернул пустой результат, используется fallback.")

    return DocumentParseResult(
        fields=_parse_document_fields(combined_text, path.name) if combined_text else _empty_document_fields(),
        extracted_text=combined_text,
        extraction_method="lightonocr",
        warnings=warnings,
    )


def _extract_with_vision_model(
    path: Path,
    ai_settings: AISettings | None,
    base_text: str,
) -> DocumentParseResult:
    if not _can_use_document_ai(ai_settings):
        return DocumentParseResult(
            fields=_empty_document_fields(),
            extracted_text="",
            extraction_method="vision_model",
            warnings=["Vision-модель документов выключена."],
        )

    suffix = path.suffix.lower()
    warnings: list[str] = []
    source_text = clean_text_value(base_text) or ""

    if suffix == ".pdf" and source_text:
        messages: list[dict[str, object]] = [
            {
                "role": "system",
                "content": (
                    "You extract structured fields from invoices, receipts, acts, and similar accounting documents. "
                    "Answer strictly with a JSON object."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Extract key accounting document fields from the following text. "
                    "Return only valid JSON with keys: "
                    "document_number, document_date, counterparty, amount, currency, description, extracted_text. "
                    "Use YYYY-MM-DD for document_date when possible. Use null for missing values.\n\n"
                    f"FILE: {path.name}\n"
                    f"TEXT:\n{source_text[:7000]}"
                ),
            },
        ]
        extraction_method = "vision_model_text"
    else:
        images = _load_document_images(path, warnings, max_pages=1)
        if not images:
            warnings.append("Vision-модель пропущена: не удалось подготовить изображение документа.")
            return DocumentParseResult(
                fields=_empty_document_fields(),
                extracted_text="",
                extraction_method="vision_model",
                warnings=warnings,
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a multimodal accounting document extraction module. "
                    "Return only valid JSON."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extract key accounting document fields from this image. "
                            "Return only valid JSON with keys: "
                            "document_number, document_date, counterparty, amount, currency, description, extracted_text. "
                            "Use YYYY-MM-DD for document_date when possible. Use null for missing values."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": _pil_image_to_data_url(images[0], image_format="JPEG")},
                    },
                ],
            },
        ]
        extraction_method = "vision_model_image"

    content = request_document_model_completion(
        messages,
        ai_settings,
        backend="vision_model",
        temperature=0.1,
        max_tokens=900,
    )
    if not content:
        warnings.append("Vision-модель недоступна или не вернула ответ, используется fallback.")
        return DocumentParseResult(
            fields=_empty_document_fields(),
            extracted_text="",
            extraction_method=extraction_method,
            warnings=warnings,
        )

    parsed = extract_json_object(content)
    if parsed:
        fields, extracted_text = _normalize_llm_document_payload(parsed, source_text)
        supplemental_fields = _parse_document_fields(extracted_text or source_text, path.name)
        fields = _merge_document_fields(fields, supplemental_fields)
        return DocumentParseResult(
            fields=fields,
            extracted_text=extracted_text,
            extraction_method=extraction_method,
            warnings=warnings,
        )

    plain_text = clean_text_value(content) or source_text
    warnings.append("Vision-модель не вернула JSON, реквизиты выделены из обычного текста.")
    return DocumentParseResult(
        fields=_parse_document_fields(plain_text, path.name) if plain_text else _empty_document_fields(),
        extracted_text=plain_text,
        extraction_method=extraction_method,
        warnings=warnings,
    )


def _load_document_images(path: Path, warnings: list[str], max_pages: int) -> list[Image.Image]:
    if path.suffix.lower() == ".pdf":
        return _render_pdf_pages(path, warnings, max_pages=max_pages)

    try:
        with Image.open(path) as image:
            prepared = image.convert("RGB")
            prepared.thumbnail((LIGHTONOCR_TARGET_MAX_DIMENSION, LIGHTONOCR_TARGET_MAX_DIMENSION))
            return [prepared.copy()]
    except Exception as exc:
        warnings.append(f"Не удалось открыть изображение документа: {exc}")
        return []


def _render_pdf_pages(path: Path, warnings: list[str], max_pages: int) -> list[Image.Image]:
    try:
        import pypdfium2 as pdfium
    except Exception:
        warnings.append("Для OCR PDF не найден модуль pypdfium2.")
        return []

    images: list[Image.Image] = []
    scale = LIGHTONOCR_RENDER_DPI / 72.0
    try:
        pdf = pdfium.PdfDocument(str(path))
        page_limit = min(len(pdf), max_pages)
        for page_index in range(page_limit):
            page = pdf[page_index]
            rendered = page.render(scale=scale).to_pil().convert("RGB")
            rendered.thumbnail((LIGHTONOCR_TARGET_MAX_DIMENSION, LIGHTONOCR_TARGET_MAX_DIMENSION))
            images.append(rendered.copy())
    except Exception as exc:
        warnings.append(f"Не удалось преобразовать PDF в изображения: {exc}")
        return []

    return images


def _run_tesseract(image: Image.Image, warnings: list[str]) -> str:
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        warnings.append("Tesseract OCR не найден в системе.")
        return ""

    try:
        text = pytesseract.image_to_string(image, lang="rus+eng")
        return clean_text_value(text) or ""
    except Exception as exc:  # pragma: no cover - depends on local OCR setup
        warnings.append(f"OCR завершился ошибкой: {exc}")
        return ""


def _result_is_usable(result: DocumentParseResult) -> bool:
    return _has_meaningful_text(result.extracted_text) or _field_coverage(result.fields) >= 0.33


def _result_score(result: DocumentParseResult | None) -> float:
    if result is None:
        return -1.0
    text_length = len(clean_text_value(result.extracted_text) or "")
    return round(_field_coverage(result.fields) * 100 + min(text_length / 20, 25), 2)


def _has_meaningful_text(text: str) -> bool:
    cleaned = clean_text_value(text) or ""
    if len(cleaned) < 40:
        return False
    return bool(re.search(r"[A-Za-zА-Яа-я0-9]", cleaned))


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
        extracted_text=clean_text_value(extracted_text) or "",
        document_number=fields["document_number"],
        document_date=fields["document_date"],
        counterparty=fields["counterparty"],
        amount=fields["amount"],
        currency=fields["currency"],
        description=fields["description"],
        warnings=warnings,
        confidence=confidence,
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


def _pil_image_to_data_url(image: Image.Image, image_format: str) -> str:
    buffer = BytesIO()
    image.save(buffer, format=image_format, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime_type = "image/png" if image_format.upper() == "PNG" else "image/jpeg"
    return f"data:{mime_type};base64,{encoded}"


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


def _deduplicate_warnings(items: list[str]) -> list[str]:
    deduplicated: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = clean_text_value(item)
        if not cleaned or cleaned in seen:
            continue
        deduplicated.append(cleaned)
        seen.add(cleaned)
    return deduplicated
