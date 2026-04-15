from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

from shingadip.config import CANONICAL_FIELDS, COLUMN_ALIASES, REQUIRED_FIELDS, WORKSPACE_DIR


IMPORT_MODE_LABELS = {
    "auto": "Авто",
    "operations_table": "Обычная таблица операций",
    "one_c_export": "Выгрузка 1С",
    "osv_summary": "ОСВ",
}

IMPORT_MODE_OPTIONS = [(key, label) for key, label in IMPORT_MODE_LABELS.items()]

ONE_C_COLUMN_HINTS: dict[str, list[str]] = {
    "operation_date": [
        "период",
        "дата",
        "дата операции",
    ],
    "document_number": [
        "регистратор",
        "документ",
        "основание",
        "номер документа",
    ],
    "counterparty": [
        "контрагент",
        "субконто",
        "субконто дт",
        "субконто кт",
        "контрагент дт",
        "контрагент кт",
        "партнер",
    ],
    "amount": [
        "сумма",
        "сумма операции",
    ],
    "account": [
        "счет",
        "счет дт",
        "счет кт",
        "счетдт",
        "счеткт",
    ],
    "description": [
        "содержание",
        "операция",
        "комментарий",
        "описание операции",
    ],
}

SERVICE_ROW_PATTERNS = (
    "итого",
    "всего",
    "оборотно-сальдовая ведомость",
    "выводимые данные",
    "данные бухгалтерского учета",
    "организация",
)


@dataclass(slots=True)
class OperationsImportResult:
    dataframe: pd.DataFrame
    detected_mode: str
    resolved_mode: str
    display_name: str
    warnings: list[str] = field(default_factory=list)
    raw_preview: pd.DataFrame = field(default_factory=pd.DataFrame)
    normalized_preview: pd.DataFrame = field(default_factory=pd.DataFrame)
    available_columns: list[str] = field(default_factory=list)
    suggested_mapping: dict[str, str] = field(default_factory=dict)
    applied_mapping: dict[str, str] = field(default_factory=dict)
    header_row_index: int | None = None
    source_is_summary: bool = False
    source_note: str = ""
    period_label: str | None = None

    def to_state(self) -> dict[str, object]:
        return {
            "detected_mode": self.detected_mode,
            "resolved_mode": self.resolved_mode,
            "display_name": self.display_name,
            "warnings": list(self.warnings),
            "raw_preview": self.raw_preview.copy(),
            "normalized_preview": self.normalized_preview.copy(),
            "available_columns": list(self.available_columns),
            "suggested_mapping": dict(self.suggested_mapping),
            "applied_mapping": dict(self.applied_mapping),
            "header_row_index": self.header_row_index,
            "source_is_summary": self.source_is_summary,
            "source_note": self.source_note,
            "period_label": self.period_label,
        }


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


def get_import_mode_label(mode: str) -> str:
    return IMPORT_MODE_LABELS.get(mode, mode)


def _read_csv_from_bytes(payload: bytes, *, header: int | None = 0) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return pd.read_csv(BytesIO(payload), sep=None, engine="python", encoding=encoding, header=header)
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Не удалось прочитать CSV-файл: {last_error}")


def _read_excel_from_bytes(payload: bytes, suffix: str, *, header: int | None = 0) -> pd.DataFrame:
    engine = "xlrd" if suffix == ".xls" else "openpyxl"
    return pd.read_excel(BytesIO(payload), engine=engine, header=header)


def _read_excel_from_path(source_path: Path, *, header: int | None = 0) -> pd.DataFrame:
    suffix = source_path.suffix.lower()
    engine = "xlrd" if suffix == ".xls" else "openpyxl"
    return pd.read_excel(source_path, engine=engine, header=header)


def read_operations_file(
    source: Any,
    import_mode: str = "auto",
    column_mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    return load_operations_source(source, import_mode=import_mode, column_mapping=column_mapping).dataframe


def load_operations_source(
    source: Any,
    import_mode: str = "auto",
    column_mapping: dict[str, str] | None = None,
) -> OperationsImportResult:
    raw_df = _read_raw_operations_table(source)
    if raw_df.empty:
        raise ValueError("Файл операций пустой.")

    source_name = Path(getattr(source, "name", source)).name
    cleaned_raw = _cleanup_raw_frame(raw_df)
    detected_mode = detect_import_mode(cleaned_raw, source_name)
    resolved_mode = detected_mode if import_mode == "auto" else import_mode
    warnings: list[str] = []
    header_row_index: int | None = None
    period_label: str | None = None
    source_is_summary = False
    source_note = ""

    if resolved_mode == "osv_summary":
        parsed_df, header_row_index, period_label = _parse_osv_summary(cleaned_raw, source_name)
        suggested_mapping: dict[str, str] = {}
        applied_mapping: dict[str, str] = {}
        warnings.append(
            "Файл распознан как оборотно-сальдовая ведомость. Анализ выполняется по агрегированным строкам счетов, а не по первичным операциям."
        )
        source_is_summary = True
        source_note = (
            "Для ОСВ проект формирует псевдо-операции по строкам счетов. Поле 'Контрагент' заполняется наименованием счета,"
            " чтобы сводный источник можно было проанализировать теми же правилами, что и обычную таблицу."
        )
        standardized = standardize_operations(parsed_df)
    else:
        prepared_df, header_row_index = _prepare_tabular_source(cleaned_raw, resolved_mode)
        suggested_mapping = _suggest_column_mapping(prepared_df.columns, resolved_mode)
        applied_mapping = dict(suggested_mapping)
        for canonical, column_name in (column_mapping or {}).items():
            if column_name and column_name in prepared_df.columns:
                applied_mapping[canonical] = column_name
        standardized = standardize_operations(prepared_df, column_mapping=applied_mapping)
        if resolved_mode == "one_c_export":
            warnings.append("Файл распознан как выгрузка 1С. Служебные строки удалены, а поля счета и контрагента нормализованы.")
            source_note = "Импорт 1С объединяет типовые поля 'Регистратор', 'Счет Дт/Кт', 'Субконто' и 'Содержание' в унифицированную структуру."
        else:
            source_note = "Файл обработан как обычная таблица операций."

    standardized["import_detected_mode"] = detected_mode
    standardized["import_resolved_mode"] = resolved_mode
    standardized["import_source_is_summary"] = source_is_summary
    standardized["import_source_note"] = source_note
    if period_label:
        standardized["import_period_label"] = period_label

    raw_preview = _build_preview_frame(cleaned_raw)
    normalized_preview = _build_preview_frame(
        standardized[
            [
                column
                for column in [
                    "operation_date_display",
                    "document_number",
                    "counterparty",
                    "amount_display",
                    "account",
                    "description",
                ]
                if column in standardized.columns
            ]
        ]
    )

    return OperationsImportResult(
        dataframe=standardized,
        detected_mode=detected_mode,
        resolved_mode=resolved_mode,
        display_name=get_import_mode_label(resolved_mode),
        warnings=warnings,
        raw_preview=raw_preview,
        normalized_preview=normalized_preview,
        available_columns=list(parsed_df.columns if resolved_mode == "osv_summary" else prepared_df.columns),
        suggested_mapping=suggested_mapping,
        applied_mapping=applied_mapping,
        header_row_index=header_row_index,
        source_is_summary=source_is_summary,
        source_note=source_note,
        period_label=period_label,
    )


def detect_import_mode(raw_df: pd.DataFrame, source_name: str = "") -> str:
    text = _collect_frame_text(raw_df, max_rows=12, max_cols=8)
    normalized_name = normalize_key(source_name)
    if "оборотно-сальдовая ведомость" in text or ("осв" in normalized_name and "счет, наименование" in text):
        return "osv_summary"

    operations_header_row, operations_score = _find_header_row(raw_df, _operations_header_aliases())
    one_c_header_row, one_c_score = _find_header_row(raw_df, _one_c_header_aliases())
    _ = operations_header_row, one_c_header_row

    if one_c_score >= 3 and (one_c_score > operations_score or _contains_one_c_markers(text)):
        return "one_c_export"
    return "operations_table"


def standardize_operations(
    df: pd.DataFrame,
    column_mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Файл операций пустой.")

    column_mapping = column_mapping or {}
    normalized_columns = {column: normalize_key(column) for column in df.columns}
    prepared = pd.DataFrame(index=df.index)
    used_columns: set[str] = set()

    for canonical in CANONICAL_FIELDS:
        prepared[canonical] = pd.NA
        explicit_column = column_mapping.get(canonical)
        matched: list[str] = []
        if explicit_column and explicit_column in df.columns:
            matched = [explicit_column]
        else:
            aliases = {
                normalize_key(canonical),
                *(normalize_key(item) for item in COLUMN_ALIASES.get(canonical, [])),
            }
            matched = [column for column, normalized in normalized_columns.items() if normalized in aliases]

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
    if "__source_row_number" in prepared.columns:
        prepared["source_row"] = (
            pd.to_numeric(prepared["__source_row_number"], errors="coerce")
            .fillna(pd.Series(range(2, len(prepared) + 2), index=prepared.index))
            .astype(int)
        )
    else:
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


def _read_raw_operations_table(source: Any) -> pd.DataFrame:
    suffix = Path(getattr(source, "name", source)).suffix.lower()
    if hasattr(source, "getvalue"):
        payload = source.getvalue()
        if suffix == ".csv":
            return _read_csv_from_bytes(payload, header=None)
        if suffix in {".xls", ".xlsx"}:
            return _read_excel_from_bytes(payload, suffix, header=None)
        raise ValueError("Поддерживаются только файлы CSV, XLS и XLSX.")

    source_path = Path(source)
    if suffix == ".csv":
        return _read_csv_from_bytes(source_path.read_bytes(), header=None)
    if suffix in {".xls", ".xlsx"}:
        return _read_excel_from_path(source_path, header=None)
    raise ValueError("Поддерживаются только файлы CSV, XLS и XLSX.")


def _cleanup_raw_frame(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.dropna(how="all")
    cleaned = cleaned.loc[:, cleaned.notna().any(axis=0)]
    cleaned = cleaned.reset_index(drop=True)
    cleaned.columns = range(cleaned.shape[1])
    return cleaned


def _collect_frame_text(frame: pd.DataFrame, *, max_rows: int, max_cols: int) -> str:
    text_values: list[str] = []
    preview = frame.iloc[:max_rows, :max_cols]
    for value in preview.to_numpy().ravel():
        text = clean_text_value(value)
        if text:
            text_values.append(text.lower())
    return " ".join(text_values)


def _operations_header_aliases() -> dict[str, set[str]]:
    aliases: dict[str, set[str]] = {}
    for canonical in CANONICAL_FIELDS:
        aliases[canonical] = {
            normalize_key(canonical),
            *(normalize_key(item) for item in COLUMN_ALIASES.get(canonical, [])),
        }
    return aliases


def _one_c_header_aliases() -> dict[str, set[str]]:
    aliases = _operations_header_aliases()
    for canonical, items in ONE_C_COLUMN_HINTS.items():
        aliases.setdefault(canonical, set()).update(normalize_key(item) for item in items)
    return aliases


def _find_header_row(frame: pd.DataFrame, aliases: dict[str, set[str]], search_rows: int = 20) -> tuple[int | None, int]:
    best_index: int | None = None
    best_score = 0
    for index in range(min(len(frame), search_rows)):
        normalized_values = {normalize_key(value) for value in frame.iloc[index].tolist() if clean_text_value(value)}
        score = sum(1 for alias_set in aliases.values() if normalized_values.intersection(alias_set))
        if score > best_score:
            best_index = index
            best_score = score
    return best_index, best_score


def _contains_one_c_markers(text: str) -> bool:
    markers = ("регистратор", "счет дт", "счет кт", "субконто", "содержание")
    return sum(1 for marker in markers if marker in text) >= 2


def _prepare_tabular_source(raw_df: pd.DataFrame, source_mode: str) -> tuple[pd.DataFrame, int]:
    aliases = _one_c_header_aliases() if source_mode == "one_c_export" else _operations_header_aliases()
    header_row_index, score = _find_header_row(raw_df, aliases)
    if header_row_index is None or score < 2:
        if source_mode == "operations_table":
            header_row_index = 0
        else:
            raise ValueError("Не удалось распознать строку заголовков в таблице операций.")

    raw_header = raw_df.iloc[header_row_index].tolist()
    header = _make_unique_headers(raw_header)
    body = raw_df.iloc[header_row_index + 1 :].copy()
    body.columns = header
    body["__source_row_number"] = range(header_row_index + 2, header_row_index + 2 + len(body))
    body = body.dropna(how="all")
    body = body.loc[~body.apply(_is_service_row, axis=1)].reset_index(drop=True)
    if source_mode == "one_c_export":
        body = _normalize_one_c_export_columns(body)
    return body, header_row_index


def _make_unique_headers(raw_header: list[Any]) -> list[str]:
    headers: list[str] = []
    seen: dict[str, int] = {}
    for index, value in enumerate(raw_header):
        text = clean_text_value(value) or f"column_{index + 1}"
        count = seen.get(text, 0)
        headers.append(text if count == 0 else f"{text} ({count + 1})")
        seen[text] = count + 1
    return headers


def _is_service_row(row: pd.Series) -> bool:
    text_parts = [normalize_key(value) for value in row.tolist()[:4] if clean_text_value(value)]
    if not text_parts:
        return True
    joined = " ".join(text_parts)
    return any(joined.startswith(pattern) for pattern in SERVICE_ROW_PATTERNS)


def _normalize_one_c_export_columns(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    _ensure_derived_column(enriched, "Дата операции", _find_matching_columns(enriched.columns, ONE_C_COLUMN_HINTS["operation_date"]))
    _ensure_derived_column(enriched, "Номер документа", _find_matching_columns(enriched.columns, ONE_C_COLUMN_HINTS["document_number"]))
    _ensure_derived_column(enriched, "Контрагент", _find_matching_columns(enriched.columns, ONE_C_COLUMN_HINTS["counterparty"]))
    _ensure_derived_column(enriched, "Сумма", _find_matching_columns(enriched.columns, ONE_C_COLUMN_HINTS["amount"]))
    _ensure_derived_column(enriched, "Описание операции", _find_matching_columns(enriched.columns, ONE_C_COLUMN_HINTS["description"]))

    account_columns = _find_matching_columns(enriched.columns, ONE_C_COLUMN_HINTS["account"])
    if account_columns:
        if len(account_columns) == 1:
            enriched["Счет учета"] = enriched[account_columns[0]]
        else:
            account_left = enriched[account_columns[0]].apply(clean_text_value)
            account_right = enriched[account_columns[1]].apply(clean_text_value)
            enriched["Счет учета"] = account_left.combine(account_right, lambda left, right: " / ".join(item for item in [left, right] if item))
    return enriched


def _find_matching_columns(columns: pd.Index, aliases: list[str]) -> list[str]:
    normalized_aliases = {normalize_key(alias) for alias in aliases}
    matches = [column for column in columns if normalize_key(column) in normalized_aliases]
    if matches:
        return matches
    partial_matches: list[str] = []
    for column in columns:
        normalized = normalize_key(column)
        if any(alias in normalized for alias in normalized_aliases):
            partial_matches.append(column)
    return partial_matches


def _ensure_derived_column(frame: pd.DataFrame, target_column: str, source_columns: list[str]) -> None:
    if not source_columns:
        return
    values = frame[source_columns[0]].apply(clean_text_value)
    for column in source_columns[1:]:
        values = values.where(values.notna() & (values.astype(str).str.strip() != ""), frame[column].apply(clean_text_value))
    frame[target_column] = values


def _parse_osv_summary(raw_df: pd.DataFrame, source_name: str) -> tuple[pd.DataFrame, int, str | None]:
    header_row_index = _find_osv_header_row(raw_df)
    if header_row_index is None:
        raise ValueError("Не удалось распознать структуру оборотно-сальдовой ведомости.")

    period_label = _extract_osv_period_label(raw_df)
    period_date = _extract_osv_period_date(period_label)
    header_secondary = raw_df.iloc[header_row_index + 1]
    numeric_indices = [index for index, value in enumerate(header_secondary.tolist()) if clean_text_value(value)]
    selected_indices = [0, *numeric_indices]
    rows = raw_df.iloc[header_row_index + 2 :, selected_indices].copy()
    expected_columns = [
        "account_name_raw",
        "opening_debit",
        "opening_credit",
        "turnover_debit",
        "turnover_credit",
        "closing_debit",
        "closing_credit",
    ]
    if len(rows.columns) < len(expected_columns):
        for filler_index in range(len(rows.columns), len(expected_columns)):
            rows[f"__empty_{filler_index}"] = None
    rows = rows.iloc[:, : len(expected_columns)]
    rows.columns = expected_columns
    rows["__source_row_number"] = range(header_row_index + 3, header_row_index + 3 + len(rows))
    rows = rows.dropna(how="all")
    rows = rows.loc[~rows.apply(_is_service_row, axis=1)].reset_index(drop=True)

    records: list[dict[str, object]] = []
    for _, row in rows.iterrows():
        raw_account = clean_text_value(row.get("account_name_raw"))
        if not raw_account:
            continue
        account_code, account_name = _split_osv_account(raw_account)
        opening_debit = parse_numeric_value(row.get("opening_debit"))
        opening_credit = parse_numeric_value(row.get("opening_credit"))
        turnover_debit = parse_numeric_value(row.get("turnover_debit"))
        turnover_credit = parse_numeric_value(row.get("turnover_credit"))
        closing_debit = parse_numeric_value(row.get("closing_debit"))
        closing_credit = parse_numeric_value(row.get("closing_credit"))
        amount = max(
            item or 0.0
            for item in [
                turnover_debit,
                turnover_credit,
                closing_debit,
                closing_credit,
                opening_debit,
                opening_credit,
            ]
        )
        if amount <= 0:
            continue

        records.append(
            {
                "Дата операции": period_date,
                "Номер документа": f"OSV-{account_code or len(records) + 1}",
                "Тип документа": "ОСВ",
                "Контрагент": account_name or raw_account,
                "Сумма": amount,
                "Валюта": "KZT",
                "НДС": None,
                "Счет учета": account_code,
                "Описание операции": f"ОСВ: {account_name or raw_account}",
                "Ответственный сотрудник": None,
                "osv_account_code": account_code,
                "osv_account_name": account_name or raw_account,
                "osv_opening_debit": opening_debit,
                "osv_opening_credit": opening_credit,
                "osv_turnover_debit": turnover_debit,
                "osv_turnover_credit": turnover_credit,
                "osv_closing_debit": closing_debit,
                "osv_closing_credit": closing_credit,
                "__source_row_number": row["__source_row_number"],
                "import_origin_file": source_name,
            }
        )

    if not records:
        raise ValueError("ОСВ распознана, но в ней не найдено строк счетов с числовыми оборотами.")

    return pd.DataFrame(records), header_row_index, period_label


def _find_osv_header_row(raw_df: pd.DataFrame) -> int | None:
    for index in range(min(len(raw_df), 20)):
        row_text = " ".join(normalize_key(value) for value in raw_df.iloc[index].tolist() if clean_text_value(value))
        if "счет, наименование" in row_text and "сальдо на начало периода" in row_text:
            return index
    return None


def _extract_osv_period_label(raw_df: pd.DataFrame) -> str | None:
    preview = raw_df.iloc[:6, :2]
    for value in preview.to_numpy().ravel():
        text = clean_text_value(value)
        if not text:
            continue
        match = re.search(r"за\s+(.+)", text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip().rstrip(".")
    return None


def _extract_osv_period_date(period_label: str | None) -> pd.Timestamp | None:
    if not period_label:
        return None
    year_match = re.search(r"(20\d{2})", period_label)
    if year_match:
        return pd.Timestamp(f"{year_match.group(1)}-12-31")
    return None


def _split_osv_account(raw_account: str) -> tuple[str | None, str]:
    match = re.match(r"^\s*([\d.]+)\s*,\s*(.+)$", raw_account)
    if not match:
        return None, raw_account
    return match.group(1).strip(), match.group(2).strip()


def _suggest_column_mapping(columns: pd.Index, source_mode: str) -> dict[str, str]:
    aliases = _one_c_header_aliases() if source_mode == "one_c_export" else _operations_header_aliases()
    mapping: dict[str, str] = {}
    preferred_one_c_columns = {
        "operation_date": "Дата операции",
        "document_number": "Номер документа",
        "counterparty": "Контрагент",
        "amount": "Сумма",
        "description": "Описание операции",
        "account": "Счет учета",
    }
    for canonical, alias_set in aliases.items():
        preferred_column = preferred_one_c_columns.get(canonical)
        if source_mode == "one_c_export" and preferred_column and preferred_column in columns:
            mapping[canonical] = preferred_column
            continue
        for column in columns:
            if normalize_key(column) in alias_set:
                mapping[canonical] = column
                break
    return mapping


def _build_preview_frame(frame: pd.DataFrame, limit_rows: int = 8) -> pd.DataFrame:
    preview = frame.head(limit_rows).copy()
    preview.columns = [str(column) for column in preview.columns]
    return preview.fillna("")
