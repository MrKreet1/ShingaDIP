from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
import pytesseract
from pypdf import PdfReader

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


def extract_documents(sources: list[Any], target_dir: Path) -> list[DocumentExtraction]:
    extracted: list[DocumentExtraction] = []
    for source in sources:
        stored_path = save_uploaded_file(source, target_dir)
        if stored_path.suffix.lower() not in SUPPORTED_DOCUMENT_TYPES:
            continue
        extracted.append(extract_document(stored_path))
    return extracted


def extract_document(path: Path) -> DocumentExtraction:
    warnings: list[str] = []
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        extraction_method = "pdf_text"
        text = _extract_text_from_pdf(path)
    else:
        extraction_method = "ocr_image"
        text = _extract_text_from_image(path, warnings)

    if not text.strip():
        warnings.append("Не удалось извлечь текст из документа.")

    fields = _parse_document_fields(text, path.name)
    confidence = sum(1 for value in fields.values() if value not in (None, "")) / max(len(fields), 1)

    return DocumentExtraction(
        file_name=path.name,
        stored_path=str(path),
        extraction_method=extraction_method,
        extracted_text=text,
        document_number=fields["document_number"],
        document_date=fields["document_date"],
        counterparty=fields["counterparty"],
        amount=fields["amount"],
        currency=fields["currency"],
        description=fields["description"],
        warnings=warnings,
        confidence=confidence,
    )


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
    except Exception as exc:
        warnings.append(f"OCR завершился ошибкой: {exc}")
        return ""


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
            r"(?:seller|vendor|supplier|counterparty|продавец|поставщик|контрагент)\s*[:#]?\s*(.+)",
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

