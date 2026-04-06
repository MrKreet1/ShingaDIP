from __future__ import annotations

import re
import shutil
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

from shingadip.config import CANONICAL_FIELDS, COLUMN_ALIASES, REQUIRED_FIELDS, WORKSPACE_DIR


def normalize_key(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def clean_text_value(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return re.sub(r"\s+", " ", text)


def parse_numeric_value(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None

    text = str(value).strip().replace("\xa0", " ")
    if not text:
        return None

    text = re.sub(r"[A-Za-zА-Яа-я$€₸%]", "", text)
    text = text.replace(" ", "")
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".")

    text = re.sub(r"[^0-9.\-]", "", text)
    if not text or text in {"-", ".", "-."}:
        return None

    try:
        return float(text)
    except ValueError:
        return None


def parse_date_value(value: Any) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.normalize()

    text = str(value).strip()
    if not text:
        return None

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        parsed = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
    else:
        parsed = pd.to_datetime(text, dayfirst=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.normalize()


def _read_csv_from_bytes(payload: bytes) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return pd.read_csv(BytesIO(payload), sep=None, engine="python", encoding=encoding)
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Не удалось прочитать CSV-файл: {last_error}")


def read_operations_file(source: Any) -> pd.DataFrame:
    suffix = Path(getattr(source, "name", source)).suffix.lower()
    if hasattr(source, "getvalue"):
        payload = source.getvalue()
        if suffix == ".csv":
            df = _read_csv_from_bytes(payload)
        elif suffix == ".xlsx":
            df = pd.read_excel(BytesIO(payload))
        else:
            raise ValueError("Поддерживаются только файлы CSV и XLSX.")
    else:
        source_path = Path(source)
        if suffix == ".csv":
            payload = source_path.read_bytes()
            df = _read_csv_from_bytes(payload)
        elif suffix == ".xlsx":
            df = pd.read_excel(source_path)
        else:
            raise ValueError("Поддерживаются только файлы CSV и XLSX.")

    return standardize_operations(df)


def standardize_operations(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Файл операций пустой.")

    normalized_columns = {column: normalize_key(column) for column in df.columns}
    prepared = pd.DataFrame(index=df.index)
    used_columns: set[str] = set()

    for canonical in CANONICAL_FIELDS:
        aliases = {normalize_key(canonical), *(normalize_key(item) for item in COLUMN_ALIASES.get(canonical, []))}
        matched = [column for column, normalized in normalized_columns.items() if normalized in aliases]
        prepared[canonical] = pd.NA
        if matched:
            series = df[matched[0]]
            for column in matched[1:]:
                series = series.where(series.notna() & (series.astype(str).str.strip() != ""), df[column])
            prepared[canonical] = series
            used_columns.update(matched)

    extra_columns = [column for column in df.columns if column not in used_columns]
    for column in extra_columns:
        prepared[column] = df[column]

    prepared["operation_date"] = prepared["operation_date"].apply(parse_date_value)
    prepared["amount"] = prepared["amount"].apply(parse_numeric_value)
    prepared["vat"] = prepared["vat"].apply(parse_numeric_value)
    prepared["currency"] = prepared["currency"].apply(clean_text_value)
    prepared["currency"] = prepared["currency"].map(lambda value: value.upper() if value else None)

    text_columns = [
        "document_number",
        "document_type",
        "counterparty",
        "account",
        "description",
        "responsible_employee",
    ]
    for column in text_columns:
        prepared[column] = prepared[column].apply(clean_text_value)

    prepared["operation_id"] = [f"OP-{index + 1:04d}" for index in range(len(prepared))]
    prepared["source_row"] = [index + 2 for index in range(len(prepared))]
    prepared["missing_required_fields"] = prepared.apply(find_missing_required_fields, axis=1)
    prepared["operation_date_display"] = prepared["operation_date"].map(
        lambda value: value.strftime("%Y-%m-%d") if value is not None else "не указана"
    )
    prepared["amount_display"] = prepared["amount"].map(format_amount)
    return prepared


def find_missing_required_fields(row: pd.Series) -> list[str]:
    missing = []
    for field in REQUIRED_FIELDS:
        value = row.get(field)
        if value is None or pd.isna(value) or (isinstance(value, str) and not value.strip()):
            missing.append(field)
    return missing


def format_amount(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "не указана"
    return f"{value:,.2f}".replace(",", " ")


def safe_filename(name: str) -> str:
    sanitized = re.sub(r"[^\w.\-]+", "_", name, flags=re.UNICODE)
    return sanitized.strip("._") or "uploaded_file"


def save_uploaded_file(source: Any, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(source, "getvalue"):
        file_name = safe_filename(source.name)
        target_path = target_dir / file_name
        target_path.write_bytes(source.getvalue())
        return target_path

    source_path = Path(source)
    target_path = target_dir / safe_filename(source_path.name)
    if source_path.resolve() != target_path.resolve():
        shutil.copy2(source_path, target_path)
    return target_path


def prepare_run_directory() -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = WORKSPACE_DIR / f"run_{run_id}"
    (run_dir / "documents").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    return run_dir
