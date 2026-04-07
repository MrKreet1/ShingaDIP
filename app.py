from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from shingadip.ai import (
    AISettings,
    discover_lm_studio_models,
    generate_dataset_conclusion,
    generate_row_commentary,
    resolve_lightonocr_model_identifier,
)
from shingadip.analysis import analyze_operations
from shingadip.config import (
    DEFAULT_DOCUMENT_ANALYSIS_MODE,
    DEFAULT_LIGHTONOCR_ENDPOINT,
    DEFAULT_TEXT_ENDPOINT,
    DEMO_DATA_DIR,
    DOCUMENT_AI_MODES,
    DOCUMENT_ANALYSIS_MODE_OPTIONS,
    LIGHTONOCR_MODEL_ID,
    SUPPORTED_DOCUMENT_TYPES,
    TEXT_MODEL_ID,
    VISION_MODEL_ID,
    ensure_workspace,
)
from shingadip.data_processing import prepare_run_directory, read_operations_file
from shingadip.documents import extract_documents
from shingadip.reporting import (
    build_audit_conclusion,
    build_report_tables,
    build_summary,
    export_frame_csv,
    export_results_csv,
    save_report_bundle,
    to_display_frame,
)

st.set_page_config(
    page_title="AI-аудит бухгалтерских операций",
    page_icon=None,
    layout="wide",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f4efe7;
            --paper: #fffaf2;
            --accent: #0f6b5c;
            --accent-soft: #d7ece6;
            --warn: #a25f18;
            --risk: #9c2f2f;
            --text: #1f2a2a;
            --muted: #6b6a63;
            --border: rgba(15, 107, 92, 0.16);
        }
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(15, 107, 92, 0.08), transparent 30%),
                radial-gradient(circle at bottom left, rgba(162, 95, 24, 0.08), transparent 30%),
                var(--bg);
        }
        [data-testid="stAppViewContainer"] {
            background: transparent;
        }
        [data-testid="stAppViewContainer"] > .main {
            background: transparent;
        }
        .block-container {
            max-width: 1560px;
            padding-top: 1.8rem;
            padding-bottom: 3rem;
        }
        [data-testid="stHeader"] {
            background: transparent !important;
            border: none !important;
        }
        [data-testid="stDecoration"] {
            display: none;
        }
        html, body, [class*="css"]  {
            font-family: "Segoe UI", "Trebuchet MS", sans-serif;
        }
        .stApp,
        .stApp p,
        .stApp span,
        .stApp label,
        .stApp div {
            color: var(--text);
        }
        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(251, 246, 238, 0.98), rgba(226, 239, 235, 0.94)) !important;
            border-right: 1px solid rgba(15, 107, 92, 0.14);
        }
        [data-testid="stSidebar"] * {
            color: var(--text) !important;
        }
        [data-testid="stSidebar"] .block-container {
            padding-top: 2rem;
            padding-bottom: 1.5rem;
        }
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] small {
            color: var(--text) !important;
        }
        [data-testid="stSidebar"] [data-testid="stCheckbox"] {
            padding: 0.2rem 0 0.4rem 0;
        }
        [data-testid="stSidebar"] [data-testid="stCheckbox"] label {
            font-weight: 500;
        }
        [data-testid="stSidebar"] [data-baseweb="input"],
        [data-testid="stSidebar"] input {
            background: rgba(255, 250, 242, 0.95) !important;
            border: 1px solid rgba(15, 107, 92, 0.2) !important;
            border-radius: 12px !important;
            color: var(--text) !important;
        }
        [data-testid="stSidebar"] [data-baseweb="input"] {
            box-shadow: none !important;
        }
        [data-testid="stSidebar"] [data-baseweb="input"] input::placeholder {
            color: #6b6a63 !important;
        }
        [data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
            background: var(--accent) !important;
        }
        [data-testid="stSidebar"] [data-testid="stTickBar"] {
            background: rgba(15, 107, 92, 0.18) !important;
        }
        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp h4,
        .stApp h5,
        .stApp h6,
        .stMarkdown h1,
        .stMarkdown h2,
        .stMarkdown h3,
        [data-testid="stHeadingWithActionElements"] h1,
        [data-testid="stHeadingWithActionElements"] h2,
        [data-testid="stHeadingWithActionElements"] h3 {
            font-family: Georgia, "Times New Roman", serif;
            letter-spacing: 0.02em;
            color: var(--text) !important;
        }
        .hero {
            background: linear-gradient(140deg, rgba(255,250,242,0.95), rgba(215,236,230,0.95));
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 30px 32px;
            margin-bottom: 28px;
            box-shadow: 0 18px 42px rgba(31, 42, 42, 0.08);
        }
        .hero-title {
            font-size: clamp(2.2rem, 3.8vw, 3.4rem);
            margin: 0 0 10px 0;
            color: #122b29 !important;
            line-height: 1.1;
        }
        .hero-subtitle {
            color: var(--muted) !important;
            margin: 0;
            font-size: 1.08rem;
            max-width: 62rem;
        }
        .section-card {
            background: rgba(255,250,242,0.82);
            border: 1px solid rgba(15,107,92,0.1);
            border-radius: 18px;
            padding: 16px 18px;
            box-shadow: 0 8px 24px rgba(31,42,42,0.04);
        }
        .stApp [data-testid="stHeadingWithActionElements"] {
            margin-top: 0.35rem;
        }
        .conclusion-box {
            background: linear-gradient(145deg, rgba(255,250,242,0.98), rgba(215,236,230,0.55));
            border-left: 6px solid var(--accent);
            padding: 18px 20px;
            border-radius: 16px;
        }
        .status-ok {
            color: #1f6f43;
            font-weight: 600;
        }
        .status-warning {
            color: var(--warn);
            font-weight: 600;
        }
        .status-risk {
            color: var(--risk);
            font-weight: 700;
        }
        [data-testid="stFileUploader"] > label,
        .stSelectbox > label,
        .stTextArea > label,
        .stCaption,
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"] {
            color: var(--text) !important;
        }
        [data-testid="stFileUploaderDropzone"] {
            background: rgba(255,250,242,0.9);
            border: 1px dashed rgba(15, 107, 92, 0.25);
            border-radius: 16px;
            min-height: 68px;
        }
        [data-testid="stFileUploaderDropzone"] * {
            color: var(--text) !important;
        }
        [data-testid="stFileUploaderDropzone"] button,
        [data-testid="stFileUploaderDropzone"] [data-testid="baseButton-secondary"] {
            background: #142b28 !important;
            color: #fffaf2 !important;
            border: 1px solid rgba(255, 250, 242, 0.08) !important;
            border-radius: 10px !important;
        }
        [data-testid="stFileUploaderDropzone"] button:hover,
        [data-testid="stFileUploaderDropzone"] [data-testid="baseButton-secondary"]:hover {
            background: #0f6b5c !important;
            color: #fffaf2 !important;
        }
        [data-testid="stBaseButton-primary"] {
            background: linear-gradient(135deg, #0f6b5c, #147d6c) !important;
            color: #fffaf2 !important;
            border: none !important;
            box-shadow: 0 10px 24px rgba(15, 107, 92, 0.2);
        }
        [data-testid="stBaseButton-primary"]:hover {
            background: linear-gradient(135deg, #115f53, #0f6b5c) !important;
            color: #fffaf2 !important;
        }
        .stDownloadButton > button,
        .stButton > button {
            border-radius: 12px;
            font-weight: 600;
            min-height: 2.9rem;
        }
        [data-testid="stAlert"] {
            background: rgba(215,236,230,0.72);
            color: var(--text) !important;
            border: 1px solid rgba(15, 107, 92, 0.15);
            border-radius: 14px;
        }
        [data-testid="stAlert"] * {
            color: var(--text) !important;
        }
        [data-testid="stMetric"] {
            background: rgba(255, 250, 242, 0.78);
            border: 1px solid rgba(15, 107, 92, 0.12);
            border-radius: 16px;
            padding: 0.85rem 1rem;
        }
        [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }
        [data-baseweb="tab"] {
            background: rgba(255,250,242,0.88);
            border-radius: 12px 12px 0 0;
            color: var(--text) !important;
        }
        [data-baseweb="tab"][aria-selected="true"] {
            background: rgba(215,236,230,0.95);
        }
        .launch-note {
            margin-top: 0.7rem;
            margin-bottom: 1rem;
            color: var(--muted);
            font-size: 0.96rem;
        }
        @media (max-width: 960px) {
            .block-container {
                padding-top: 1rem;
            }
            .hero {
                padding: 24px 22px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1 class="hero-title">AI-аудит бухгалтерских операций</h1>
            <p class="hero-subtitle">
                Демонстрационный прототип для загрузки операций, проверки первичных документов,
                поиска аномалий и формирования комментариев в стиле помощника аудитора.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_run_block() -> bool:
    action_col, _ = st.columns((0.4, 0.6))
    with action_col:
        run_clicked = st.button("Запустить анализ", type="primary", use_container_width=True)
    st.markdown(
        """
        <div class="launch-note">
            Можно использовать собственные файлы или демонстрационный набор, уже подготовленный в проекте.
        </div>
        """,
        unsafe_allow_html=True,
    )
    return run_clicked


def demo_operation_path() -> Path:
    return DEMO_DATA_DIR / "operations_demo.csv"


def demo_document_paths() -> list[Path]:
    documents_dir = DEMO_DATA_DIR / "documents"
    if not documents_dir.exists():
        return []
    return sorted(path for path in documents_dir.iterdir() if path.suffix.lower() in SUPPORTED_DOCUMENT_TYPES)


def status_css_class(status: str) -> str:
    if status == "OK":
        return "status-ok"
    if status == "WARNING":
        return "status-warning"
    return "status-risk"


def display_value(value: object, fallback: str = "не найдено") -> str:
    if value is None:
        return fallback
    try:
        if pd.isna(value):
            return fallback
    except TypeError:
        pass
    text = str(value).strip()
    return text or fallback


def _style_status_cell(value: object) -> str:
    text = str(value or "").strip().upper()
    if text == "RISK":
        return "background-color: #f8d7da; color: #7a1f28; font-weight: 700; border-radius: 8px;"
    if text == "WARNING":
        return "background-color: #fff0cf; color: #8a5a00; font-weight: 700; border-radius: 8px;"
    if text == "OK":
        return "background-color: #dff3e7; color: #1f6f43; font-weight: 700; border-radius: 8px;"
    return ""


def _style_priority_cell(value: object) -> str:
    text = str(value or "").strip().upper()
    if text == "HIGH":
        return "background-color: #f9d8d8; color: #8b1e1e; font-weight: 700; border-radius: 8px;"
    if text == "MEDIUM":
        return "background-color: #fff1d6; color: #8c5a00; font-weight: 700; border-radius: 8px;"
    if text == "LOW":
        return "background-color: #dff1ea; color: #1b6a5c; font-weight: 700; border-radius: 8px;"
    return ""


def style_priority_report(frame: pd.DataFrame):
    styler = frame.style
    if "Статус" in frame.columns:
        styler = styler.applymap(_style_status_cell, subset=["Статус"])
    if "Приоритет проверки" in frame.columns:
        styler = styler.applymap(_style_priority_cell, subset=["Приоритет проверки"])
    return styler


def render_exportable_report(
    title: str,
    frame: pd.DataFrame,
    *,
    empty_message: str,
    download_label: str,
    file_name: str,
    key: str,
) -> None:
    st.subheader(title)
    if frame.empty:
        st.info(empty_message)
        return
    st.dataframe(frame, use_container_width=True, hide_index=True)
    st.download_button(
        download_label,
        data=export_frame_csv(frame),
        file_name=file_name,
        mime="text/csv",
        key=key,
    )


def render_sidebar() -> dict[str, object]:
    st.sidebar.header("Настройки анализа")
    use_demo_data = st.sidebar.checkbox("Использовать демонстрационный набор", value=True)
    include_demo_docs = st.sidebar.checkbox("Подключить демонстрационные документы", value=True)
    use_text_llm = st.sidebar.checkbox("Использовать модель для текста", value=False)
    use_document_model = st.sidebar.checkbox("Использовать модель для документов", value=False)
    text_endpoint = st.sidebar.text_input(
        "Endpoint текстовой модели",
        value=DEFAULT_TEXT_ENDPOINT,
        disabled=not use_text_llm,
    )

    available_document_modes = [
        item for item in DOCUMENT_ANALYSIS_MODE_OPTIONS if use_document_model or item[0] not in DOCUMENT_AI_MODES
    ]
    mode_to_label = {value: label for value, label in available_document_modes}
    label_to_mode = {label: value for value, label in available_document_modes}
    default_mode = DEFAULT_DOCUMENT_ANALYSIS_MODE if DEFAULT_DOCUMENT_ANALYSIS_MODE in mode_to_label else "pdf_text"
    selected_mode_label = st.sidebar.selectbox(
        "Режим анализа документов",
        options=list(label_to_mode.keys()),
        index=list(label_to_mode.values()).index(default_mode),
    )
    document_mode = label_to_mode[selected_mode_label]

    document_endpoint = st.sidebar.text_input(
        "Endpoint vision-модели документов",
        value=DEFAULT_TEXT_ENDPOINT,
        disabled=not (use_document_model and document_mode in {"auto", "vision_model"}),
    )
    lightonocr_endpoint = st.sidebar.text_input(
        "Endpoint LightOnOCR",
        value=DEFAULT_LIGHTONOCR_ENDPOINT,
        disabled=not (use_document_model and document_mode in {"auto", "lightonocr"}),
    )

    max_document_ai_calls = st.sidebar.slider(
        "Максимум AI-разборов документов",
        min_value=1,
        max_value=20,
        value=10,
        disabled=not (use_document_model and document_mode in {"auto", "vision_model", "lightonocr"}),
    )
    max_llm_rows = st.sidebar.slider(
        "Максимум LLM-пояснений за запуск",
        min_value=1,
        max_value=20,
        value=8,
        disabled=not use_text_llm,
    )
    resolved_lightonocr_model = LIGHTONOCR_MODEL_ID

    if use_text_llm:
        text_models, text_error = discover_lm_studio_models(text_endpoint)
        st.sidebar.caption(f"Текстовая модель зафиксирована: `{TEXT_MODEL_ID}`")
        if text_error:
            st.sidebar.warning(
                "Не удалось получить список текстовых моделей. Проверьте endpoint и локальный сервер."
            )
        elif text_models and TEXT_MODEL_ID not in text_models:
            st.sidebar.warning(f"Модель `{TEXT_MODEL_ID}` не найдена на endpoint текстовой модели.")

    if use_document_model:
        st.sidebar.caption(f"Режим документов: `{selected_mode_label}`")
        if document_mode in {"auto", "vision_model"}:
            document_models, document_error = discover_lm_studio_models(document_endpoint)
            st.sidebar.caption(f"Vision-модель зафиксирована: `{VISION_MODEL_ID}`")
            if document_error:
                st.sidebar.warning("Не удалось получить список vision-моделей документов.")
            elif document_models and VISION_MODEL_ID not in document_models:
                st.sidebar.warning(f"Модель `{VISION_MODEL_ID}` не найдена на endpoint документов.")

        if document_mode in {"auto", "lightonocr"}:
            lighton_models, lighton_error = discover_lm_studio_models(lightonocr_endpoint)
            resolved_lightonocr_model = resolve_lightonocr_model_identifier(lighton_models)
            st.sidebar.caption(f"LightOnOCR зафиксирован: `{resolved_lightonocr_model}`")
            if lighton_error:
                st.sidebar.warning("Не удалось получить список моделей LightOnOCR.")
            elif lighton_models and resolved_lightonocr_model not in lighton_models:
                st.sidebar.warning("Не удалось автоматически подобрать модель LightOnOCR на endpoint.")
    else:
        st.sidebar.caption("Без AI-модели документы разбираются через `PDF text`, `Tesseract` и fallback.")

    st.sidebar.caption(
        "При недоступности OCR или локальной модели система автоматически продолжит работу с fallback-разбором."
    )
    return {
        "use_demo_data": use_demo_data,
        "include_demo_docs": include_demo_docs,
        "ai_settings": AISettings(
            use_lm_studio=use_text_llm,
            endpoint=text_endpoint,
            model=TEXT_MODEL_ID,
            max_rows=max_llm_rows,
            use_document_model=use_document_model,
            document_analysis_mode=document_mode,
            document_endpoint=document_endpoint,
            document_model=VISION_MODEL_ID,
            lightonocr_endpoint=lightonocr_endpoint,
            lightonocr_model=resolved_lightonocr_model,
            max_document_ai_calls=max_document_ai_calls,
        ),
    }


def render_uploads() -> tuple[object | None, list[object]]:
    operations_col, docs_col = st.columns((1.1, 1))
    with operations_col:
        st.subheader("Таблица операций")
        operations_file = st.file_uploader(
            "CSV или XLSX",
            type=["csv", "xlsx"],
            key="operations_uploader",
        )
    with docs_col:
        st.subheader("Первичные документы")
        documents = st.file_uploader(
            "PDF или изображения",
            type=[suffix.lstrip(".") for suffix in sorted(SUPPORTED_DOCUMENT_TYPES)],
            accept_multiple_files=True,
            key="documents_uploader",
        )
    return operations_file, documents or []


def run_analysis(
    operations_source: object,
    document_sources: list[object],
    ai_settings: AISettings,
    source_label: str,
) -> None:
    ensure_workspace()
    run_dir = prepare_run_directory()
    operations_df = read_operations_file(operations_source)
    extracted_documents = extract_documents(document_sources, run_dir / "documents", ai_settings)
    results_df = analyze_operations(operations_df, extracted_documents)
    results_df = generate_row_commentary(results_df, ai_settings)
    summary = build_summary(results_df, extracted_documents)
    report_tables = build_report_tables(results_df, summary)
    summary["dataset_comment"] = generate_dataset_conclusion(summary, report_tables, ai_settings)
    audit_conclusion = build_audit_conclusion(summary, results_df, report_tables)
    report_paths = save_report_bundle(
        results_df,
        summary,
        run_dir / "reports",
        report_tables=report_tables,
        audit_conclusion=audit_conclusion,
    )

    st.session_state["analysis_state"] = {
        "source_label": source_label,
        "run_dir": str(run_dir),
        "operations_df": operations_df,
        "documents_df": pd.DataFrame([doc.to_record() for doc in extracted_documents]),
        "results_df": results_df,
        "summary": summary,
        "report_tables": report_tables,
        "audit_conclusion": audit_conclusion,
        "report_paths": report_paths,
    }


def render_metrics(summary: dict[str, object]) -> None:
    metrics = st.columns(4)
    metrics[0].metric("Всего операций", summary["total_operations"])
    metrics[1].metric("OK", summary["ok_count"])
    metrics[2].metric("WARNING", summary["warning_count"])
    metrics[3].metric("RISK", summary["risk_count"])


def render_results(state: dict[str, object]) -> None:
    results_df: pd.DataFrame = state["results_df"]
    summary: dict[str, object] = state["summary"]
    documents_df: pd.DataFrame = state["documents_df"]
    report_tables: dict[str, pd.DataFrame] = state["report_tables"]
    audit_conclusion: dict[str, object] = state["audit_conclusion"]
    risk_register = report_tables["risk_register"]
    reason_summary = report_tables["reason_summary"]
    counterparty_summary = report_tables["counterparty_summary"]
    document_reconciliation = report_tables["document_reconciliation"]

    render_metrics(summary)
    st.caption(f"Источник данных: {state['source_label']} | Рабочая директория: {state['run_dir']}")

    tabs = st.tabs(
        [
            "Результаты",
            "Реестр рисков",
            "Причины",
            "Контрагенты",
            "Сверка документов",
            "Детали операции",
            "Документы",
            "Итоговое заключение",
        ]
    )

    with tabs[0]:
        st.subheader("Сводка отклонений")
        chart_col, reasons_col = st.columns((1, 1.1))
        with chart_col:
            status_counts = pd.DataFrame(
                {
                    "status": ["OK", "WARNING", "RISK"],
                    "count": [summary["ok_count"], summary["warning_count"], summary["risk_count"]],
                }
            ).set_index("status")
            st.bar_chart(status_counts)
        with reasons_col:
            if reason_summary.empty:
                st.info("Существенные причины отклонений не выявлены.")
            else:
                st.dataframe(reason_summary.head(10), use_container_width=True, hide_index=True)

        st.subheader("Таблица результатов")
        display_df = to_display_frame(results_df)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Скачать полный результат CSV",
            data=export_results_csv(results_df),
            file_name="audit_results.csv",
            mime="text/csv",
            key="download_full_results",
        )

    with tabs[1]:
        render_exportable_report(
            "Реестр риск-операций",
            risk_register,
            empty_message="Операции со статусами WARNING и RISK не выявлены.",
            download_label="Скачать risk_register.csv",
            file_name="risk_register.csv",
            key="download_risk_register",
        )

    with tabs[2]:
        render_exportable_report(
            "Отчет по причинам отклонений",
            reason_summary,
            empty_message="Причины отклонений отсутствуют.",
            download_label="Скачать reason_summary.csv",
            file_name="reason_summary.csv",
            key="download_reason_summary",
        )

    with tabs[3]:
        render_exportable_report(
            "Отчет по контрагентам",
            counterparty_summary,
            empty_message="Нет данных для отчета по контрагентам.",
            download_label="Скачать counterparty_summary.csv",
            file_name="counterparty_summary.csv",
            key="download_counterparty_summary",
        )

    with tabs[4]:
        render_exportable_report(
            "Отчет по сверке документов",
            document_reconciliation,
            empty_message="Нет данных для сверки документов.",
            download_label="Скачать document_reconciliation.csv",
            file_name="document_reconciliation.csv",
            key="download_document_reconciliation",
        )

    with tabs[5]:
        st.subheader("Панель детализации")
        selectable = results_df.sort_values(["risk_score", "operation_id"], ascending=[False, True])
        labels = selectable.apply(
            lambda row: f"{row['operation_id']} | {row['document_number'] or 'без номера'} | {row['status']}",
            axis=1,
        )
        selected_label = st.selectbox("Выберите операцию", labels.tolist())
        selected_row = selectable.loc[labels == selected_label].iloc[0]
        detail_col, audit_col = st.columns((1, 1))
        with detail_col:
            st.markdown(
                f"""
                <div class="section-card">
                    <h3>Операция</h3>
                    <p><strong>ID:</strong> {selected_row['operation_id']}</p>
                    <p><strong>Дата:</strong> {selected_row['operation_date_display']}</p>
                    <p><strong>Документ:</strong> {selected_row['document_number'] or 'не указан'}</p>
                    <p><strong>Контрагент:</strong> {selected_row['counterparty'] or 'не указан'}</p>
                    <p><strong>Сумма:</strong> {selected_row['amount_display']}</p>
                    <p><strong>Статус:</strong> <span class="{status_css_class(selected_row['status'])}">{selected_row['status']}</span></p>
                    <p><strong>Риск-балл:</strong> {selected_row['risk_score']}</p>
                    <p><strong>Приоритет проверки:</strong> {display_value(selected_row.get('priority'))}</p>
                    <p><strong>Уверенность:</strong> {display_value(selected_row.get('confidence'))}</p>
                    <p><strong>Категория риска:</strong> {display_value(selected_row.get('risk_category'))}</p>
                    <p><strong>Ведущий фактор:</strong> {display_value(selected_row.get('primary_risk_driver'))}</p>
                    <p><strong>Ключевые факторы:</strong> {display_value(selected_row.get('top_risk_factors'))}</p>
                    <p><strong>Причины:</strong> {selected_row['reason_details']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with audit_col:
            st.markdown(
                f"""
                <div class="section-card">
                    <h3>Краткий вывод</h3>
                    <p>{display_value(selected_row.get('short_ai_comment'))}</p>
                    <h3>Подробный аудиторский комментарий</h3>
                    <p>{display_value(selected_row.get('full_ai_comment'))}</p>
                    <h3>Рекомендуемое действие</h3>
                    <p>{display_value(selected_row.get('recommended_action'))}</p>
                    <h3>Сопоставление с документом</h3>
                    <p><strong>Найденный документ:</strong> {selected_row['matched_document'] or 'не найден'}</p>
                    <p><strong>Статус сверки:</strong> {selected_row['document_check_status']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if selected_row.get("machine_payload_json"):
                st.text_area(
                    "Машинный слой интерпретации (JSON)",
                    display_value(selected_row.get("machine_payload_json"), fallback=""),
                    height=220,
                )

    with tabs[6]:
        st.subheader("Извлеченные документы")
        if documents_df.empty:
            st.info("Документы не были загружены.")
        else:
            st.dataframe(documents_df, use_container_width=True, hide_index=True)
            selected_doc_name = st.selectbox("Просмотр извлеченного текста", documents_df["file_name"].tolist())
            selected_doc = documents_df.loc[documents_df["file_name"] == selected_doc_name].iloc[0]
            doc_col, text_col = st.columns((0.95, 1.05))
            with doc_col:
                st.markdown(
                    f"""
                    <div class="section-card">
                        <h3>Карточка документа</h3>
                        <p><strong>Файл:</strong> {display_value(selected_doc['file_name'])}</p>
                        <p><strong>Способ извлечения:</strong> {display_value(selected_doc['extraction_method'])}</p>
                        <p><strong>Уверенность:</strong> {display_value(selected_doc['confidence'])}</p>
                        <p><strong>Номер документа:</strong> {display_value(selected_doc['document_number'])}</p>
                        <p><strong>Дата документа:</strong> {display_value(selected_doc['document_date'])}</p>
                        <p><strong>Контрагент:</strong> {display_value(selected_doc['counterparty'])}</p>
                        <p><strong>Сумма:</strong> {display_value(selected_doc['amount'])}</p>
                        <p><strong>Валюта:</strong> {display_value(selected_doc['currency'])}</p>
                        <p><strong>Описание:</strong> {display_value(selected_doc['description'])}</p>
                        <p><strong>Предупреждения:</strong> {display_value(selected_doc['warnings'], fallback='нет')}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with text_col:
                st.text_area("Извлеченный текст", display_value(selected_doc["extracted_text"], fallback=""), height=320)

    with tabs[7]:
        st.subheader("Итоговое заключение")
        st.markdown(
            f"""
            <div class="conclusion-box">
                <p><strong>Краткое заключение:</strong> {audit_conclusion['short_text']}</p>
                <p><strong>Всего операций:</strong> {audit_conclusion['total_operations']}</p>
                <p><strong>OK / WARNING / RISK:</strong> {audit_conclusion['ok_count']} / {audit_conclusion['warning_count']} / {audit_conclusion['risk_count']}</p>
                <p><strong>Основные причины:</strong> {audit_conclusion['main_reasons']}</p>
                <p><strong>Проблемные контрагенты:</strong> {audit_conclusion['problematic_counterparties']}</p>
                <p><strong>Покрытие документами:</strong> {display_value(summary['document_coverage'])}</p>
                <p><strong>Качество документального покрытия:</strong> {audit_conclusion['document_coverage_quality']}</p>
                <p><strong>Комментарий по документам:</strong> {display_value(audit_conclusion.get('document_scope_note'), fallback='нет дополнительного комментария')}</p>
                <p><strong>Средний риск:</strong> {summary['average_risk_score']}</p>
                <p><strong>Рекомендация аудитору:</strong> {audit_conclusion['recommendation']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        focus_col, top_col = st.columns((0.85, 1.15))
        with focus_col:
            st.subheader("Что проверять в первую очередь")
            priority_focus = audit_conclusion.get("priority_review_focus", [])
            if priority_focus:
                for item in priority_focus:
                    st.markdown(f"- **{item['label']}**: {item['count']}")
            else:
                st.info("Приоритетные блоки для дополнительной ручной проверки не выявлены.")
        with top_col:
            top_risk_operations = pd.DataFrame(audit_conclusion.get("top_risk_operations", []))
            if not top_risk_operations.empty:
                st.subheader("Топ-5 самых рискованных операций")
                preferred_columns = [
                    "ID операции",
                    "Контрагент",
                    "Сумма",
                    "Статус",
                    "Приоритет проверки",
                    "Категория риска",
                    "Ведущий фактор риска",
                    "Рекомендуемое действие",
                ]
                available_columns = [column for column in preferred_columns if column in top_risk_operations.columns]
                top_risk_display = top_risk_operations[available_columns]
                st.dataframe(style_priority_report(top_risk_display), use_container_width=True, hide_index=True)
            else:
                st.info("Операции с повышенным риском не выявлены.")
        with st.expander("Подробный AI-вывод аудитора", expanded=False):
            st.markdown(summary["dataset_comment"])
        st.caption(
            "Файлы отчетов сохранены локально: "
            f"{state['report_paths']['csv']}, "
            f"{state['report_paths']['risk_register']}, "
            f"{state['report_paths']['reason_summary']}, "
            f"{state['report_paths']['counterparty_summary']}, "
            f"{state['report_paths']['document_reconciliation']} "
            f"и {state['report_paths']['json']}"
        )


def main() -> None:
    inject_styles()
    settings = render_sidebar()
    render_header()
    operations_file, uploaded_docs = render_uploads()
    run_clicked = render_run_block()

    if run_clicked:
        operations_source: object | None = operations_file
        document_sources: list[object] = list(uploaded_docs)
        source_label = "Пользовательский файл"
        use_demo_operations = False

        if operations_source is None and settings["use_demo_data"]:
            operations_source = demo_operation_path()
            source_label = "Демонстрационный CSV"
            use_demo_operations = True

        if not document_sources and use_demo_operations and settings["include_demo_docs"]:
            document_sources = demo_document_paths()

        if operations_source is None:
            st.error("Загрузите файл операций или включите демонстрационный набор.")
        else:
            run_analysis(
                operations_source=operations_source,
                document_sources=document_sources,
                ai_settings=settings["ai_settings"],
                source_label=source_label,
            )
            st.success("Анализ завершен. Результаты доступны ниже.")

    state = st.session_state.get("analysis_state")
    if state:
        render_results(state)
    else:
        st.info(
            "После запуска анализа здесь появятся результаты проверки, карточка операции и итоговое заключение."
        )


if __name__ == "__main__":
    main()
