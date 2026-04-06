from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from shingadip.ai import (
    AISettings,
    discover_lm_studio_models,
    generate_dataset_conclusion,
    generate_row_commentary,
)
from shingadip.analysis import analyze_operations
from shingadip.config import DEMO_DATA_DIR, SUPPORTED_DOCUMENT_TYPES, ensure_workspace
from shingadip.data_processing import prepare_run_directory, read_operations_file
from shingadip.documents import extract_documents
from shingadip.reporting import (
    build_summary,
    export_results_csv,
    save_report_bundle,
    summarize_reasons,
    to_display_frame,
)


TEXT_MODEL_ID = "qwen2.5-7b-instruct"
VISION_MODEL_ID = "qwen2.5-vl-7b-instruct"


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


def render_sidebar() -> dict[str, object]:
    st.sidebar.header("Настройки анализа")
    use_demo_data = st.sidebar.checkbox("Использовать демонстрационный набор", value=True)
    include_demo_docs = st.sidebar.checkbox("Подключить демонстрационные документы", value=True)
    use_text_llm = st.sidebar.checkbox("Использовать модель для текста", value=False)
    use_document_llm = st.sidebar.checkbox("Использовать модель для документов", value=False)
    use_any_lm = use_text_llm or use_document_llm
    endpoint = st.sidebar.text_input(
        "Endpoint",
        value="http://127.0.0.1:8080/v1/chat/completions",
        disabled=not use_any_lm,
    )
    discovered_models: list[str] = []
    discovery_error: str | None = None
    if use_any_lm:
        discovered_models, discovery_error = discover_lm_studio_models(endpoint)

    if use_any_lm and discovery_error:
        st.sidebar.warning(
            "Не удалось получить список моделей из LM Studio. "
            "Проверьте endpoint и что локальный сервер запущен."
        )
    elif use_any_lm:
        st.sidebar.caption(f"LM Studio доступен. Найдено моделей: {len(discovered_models)}.")

    if use_text_llm:
        st.sidebar.caption(f"Текстовая модель зафиксирована: `{TEXT_MODEL_ID}`")
        if discovered_models and TEXT_MODEL_ID not in discovered_models:
            st.sidebar.warning(
                f"Модель `{TEXT_MODEL_ID}` не найдена среди загруженных в LM Studio."
            )

    use_vision_for_documents = use_document_llm
    if use_document_llm:
        st.sidebar.caption(f"Модель для документов зафиксирована: `{VISION_MODEL_ID}`")
        if discovered_models and VISION_MODEL_ID not in discovered_models:
            st.sidebar.warning(
                f"Модель `{VISION_MODEL_ID}` не найдена среди загруженных в LM Studio."
            )

    max_document_ai_calls = st.sidebar.slider(
        "Максимум AI-разборов документов",
        min_value=1,
        max_value=20,
        value=10,
        disabled=not use_document_llm,
    )
    max_llm_rows = st.sidebar.slider(
        "Максимум LLM-пояснений за запуск",
        min_value=1,
        max_value=20,
        value=8,
        disabled=not use_text_llm,
    )
    st.sidebar.caption(
        "При недоступности OCR или локальной модели система продолжит работу с офлайн-фолбэками."
    )
    return {
        "use_demo_data": use_demo_data,
        "include_demo_docs": include_demo_docs,
        "ai_settings": AISettings(
            use_lm_studio=use_text_llm,
            endpoint=endpoint,
            model=TEXT_MODEL_ID,
            max_rows=max_llm_rows,
            use_vision_for_documents=use_vision_for_documents,
            vision_model=VISION_MODEL_ID,
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
    summary["dataset_comment"] = generate_dataset_conclusion(summary, ai_settings)
    report_paths = save_report_bundle(results_df, summary, run_dir / "reports")

    st.session_state["analysis_state"] = {
        "source_label": source_label,
        "run_dir": str(run_dir),
        "operations_df": operations_df,
        "documents_df": pd.DataFrame([doc.to_record() for doc in extracted_documents]),
        "results_df": results_df,
        "summary": summary,
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

    render_metrics(summary)
    st.caption(f"Источник данных: {state['source_label']} | Рабочая директория: {state['run_dir']}")

    tabs = st.tabs(["Результаты", "Детали операции", "Документы", "Отчет"])

    with tabs[0]:
        st.subheader("Сводка отклонений")
        reasons_frame = summarize_reasons(results_df)
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
            if reasons_frame.empty:
                st.info("Существенные причины отклонений не выявлены.")
            else:
                st.dataframe(reasons_frame, use_container_width=True, hide_index=True)

        st.subheader("Таблица результатов")
        display_df = to_display_frame(results_df)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    with tabs[1]:
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
                    <p><strong>Причины:</strong> {selected_row['reason_details']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with audit_col:
            st.markdown(
                f"""
                <div class="section-card">
                    <h3>Комментарий аудитора</h3>
                    <p>{selected_row['ai_comment']}</p>
                    <h3>Рекомендуемое действие</h3>
                    <p>{selected_row['recommended_action']}</p>
                    <h3>Сопоставление с документом</h3>
                    <p><strong>Найденный документ:</strong> {selected_row['matched_document'] or 'не найден'}</p>
                    <p><strong>Статус сверки:</strong> {selected_row['document_check_status']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tabs[2]:
        st.subheader("Извлеченные документы")
        if documents_df.empty:
            st.info("Документы не были загружены.")
        else:
            st.dataframe(documents_df, use_container_width=True, hide_index=True)
            selected_doc_name = st.selectbox("Просмотр извлеченного текста", documents_df["file_name"].tolist())
            selected_doc = documents_df.loc[documents_df["file_name"] == selected_doc_name].iloc[0]
            st.text_area("Извлеченный текст", selected_doc["extracted_text"], height=260)

    with tabs[3]:
        st.subheader("Итоговое заключение")
        st.markdown(
            f"""
            <div class="conclusion-box">
                <p><strong>Покрытие документами:</strong> {summary['document_coverage']}</p>
                <p><strong>Средний риск:</strong> {summary['average_risk_score']}</p>
                <p><strong>Заключение:</strong> {summary['dataset_comment']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        csv_data = export_results_csv(results_df)
        st.download_button(
            "Скачать отчет CSV",
            data=csv_data,
            file_name="audit_results.csv",
            mime="text/csv",
        )
        st.caption(
            "Файлы отчета также сохранены локально: "
            f"{state['report_paths']['csv']} и {state['report_paths']['json']}"
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
