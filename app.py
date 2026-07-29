from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from finsense.analytics import apply_filters, monthly_summary_bytes  # noqa: E402
from finsense.anomalies import detect_anomalies  # noqa: E402
from finsense.data import (  # noqa: E402
    dataframe_to_csv_bytes,
    load_sample_transactions,
    template_bytes,
)
from finsense.forecasting import ForecastResult, run_forecast  # noqa: E402
from finsense.imports import (  # noqa: E402
    NONE_OPTION,
    detect_mapping,
    mapped_options,
    normalize_uploaded_transactions,
    read_uploaded_csv_bytes,
)
from finsense.insights import build_insights, build_recommendations  # noqa: E402
from finsense.reporting import cleaning_report_bytes, html_report  # noqa: E402
from finsense.scenarios import calculate_scenario  # noqa: E402
from finsense.scoring import financial_health_score  # noqa: E402
from finsense.ui import (  # noqa: E402
    configure_page,
    header,
    show_anomalies,
    show_eda,
    show_executive_overview,
    show_forecast,
    show_health_and_scenario,
    show_methodology,
    show_quality,
)

REQUIRED_FIELDS = ["date", "description", "amount", "debit", "credit"]
OPTIONAL_FIELDS = [
    "transaction_type",
    "category",
    "merchant",
    "payment_method",
    "currency",
    "transaction_id",
]


@st.cache_data(show_spinner=False)
def cached_sample() -> tuple[pd.DataFrame, object]:
    return load_sample_transactions()


@st.cache_data(show_spinner=False)
def cached_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    return detect_anomalies(df)


@st.cache_data(show_spinner=False)
def cached_forecast(df: pd.DataFrame, monthly_budget: float) -> ForecastResult:
    return run_forecast(df, monthly_budget)


def _mapping_select(
    label: str, columns: list[str], current: str | None, help_text: str | None = None
) -> str | None:
    options = [NONE_OPTION, *columns]
    index = options.index(current) if current in options else 0
    value = st.selectbox(label, options, index=index, help=help_text)
    return None if value == NONE_OPTION else value


def _show_upload_wizard() -> tuple[pd.DataFrame | None, object | None, str | None]:
    st.subheader("Analyze Your Data")
    st.caption(
        "Upload a CSV up to 10 MB. Processing happens in memory only; files are not saved. "
        "Use anonymized exports and remove account numbers where possible."
    )
    with st.expander("CSV fields and example", expanded=True):
        st.write(
            "FinSense accepts the template columns, debit/credit statements, signed amount statements, "
            "and amount plus transaction-type files."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "date": "2026-02-01",
                        "description": "UPI Swiggy dinner",
                        "debit": "840",
                        "credit": "",
                        "category": "",
                    }
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.download_button(
            "Download FinSense CSV template",
            template_bytes(),
            "transaction_template.csv",
            "text/csv",
        )

    uploaded = st.file_uploader("Upload file", type=["csv"], key="main_upload")
    if uploaded is None:
        st.info("Upload a CSV to begin. Demo metrics are hidden while upload mode is active.")
        return None, None, None

    content = uploaded.getvalue()
    parsed = read_uploaded_csv_bytes(content, uploaded.name)
    if parsed.error:
        st.error(parsed.error)
        return None, None, None

    raw = parsed.dataframe
    detected_format, detected_mapping = detect_mapping(raw)
    columns = [str(column) for column in raw.columns]
    st.success(f"Upload file complete: {uploaded.name}")

    steps = st.tabs(
        [
            "1 Upload file",
            "2 Detect columns",
            "3 Review mapping",
            "4 Preview cleaned data",
            "5 Analyze",
        ]
    )
    with steps[0]:
        st.write(f"File size: {parsed.size_bytes / 1024:.1f} KB")
        st.write(f"Encoding: {parsed.encoding}")
        st.caption(
            "Unknown columns are ignored. Balance columns are never used as transaction amounts."
        )
    with steps[1]:
        st.metric("Detected format", detected_format)
        st.write("Detected mapping")
        st.json(mapped_options(detected_mapping))
    with steps[2]:
        mapping = detected_mapping.copy()
        c1, c2 = st.columns(2)
        with c1:
            st.write("Required mappings")
            for field in REQUIRED_FIELDS:
                mapping[field] = _mapping_select(
                    field.replace("_", " ").title(),
                    columns,
                    mapping.get(field),
                    "Use amount for signed amounts or amount plus transaction type. Use debit/credit for bank statements.",
                )
        with c2:
            st.write("Optional mappings")
            for field in OPTIONAL_FIELDS:
                mapping[field] = _mapping_select(
                    field.replace("_", " ").title(), columns, mapping.get(field)
                )

    preview = normalize_uploaded_transactions(raw, mapping, detected_format)
    with steps[3]:
        if preview.validation_errors:
            for error in preview.validation_errors:
                st.error(error)
        metrics = st.columns(5)
        metrics[0].metric("Input Rows", preview.raw_rows)
        metrics[1].metric("Valid Rows", preview.report.output_rows)
        metrics[2].metric("Rejected Rows", preview.report.invalid_rows_rejected)
        metrics[3].metric("Missing Critical Values", preview.missing_critical_values)
        if not preview.cleaned.empty:
            metrics[4].metric(
                "Date Range",
                f"{preview.cleaned['date'].min():%Y-%m-%d} to {preview.cleaned['date'].max():%Y-%m-%d}",
            )
            st.dataframe(
                preview.cleaned.drop(columns=["transaction_id"], errors="ignore").head(10),
                use_container_width=True,
                hide_index=True,
            )
        else:
            metrics[4].metric("Date Range", "N/A")
    with steps[4]:
        st.write("Confirm this normalized data before opening the dashboards.")
        if st.button("Analyze this data", type="primary", disabled=not preview.is_ready):
            st.session_state["uploaded_cleaned"] = preview.cleaned
            st.session_state["uploaded_report"] = preview.report
            st.session_state["uploaded_filename"] = uploaded.name
            st.session_state["data_mode"] = "Upload My Transactions"
            st.rerun()
        if st.button("Change mapping"):
            st.session_state.pop("uploaded_cleaned", None)
            st.session_state.pop("uploaded_report", None)
            st.rerun()

    return None, None, None


def main() -> None:
    configure_page()
    if "data_mode" not in st.session_state:
        st.session_state["data_mode"] = "Explore Demo Data"

    mode = st.radio(
        "Choose how to start",
        ["Explore Demo Data", "Upload My Transactions"],
        horizontal=True,
        index=0 if st.session_state["data_mode"] == "Explore Demo Data" else 1,
    )
    st.session_state["data_mode"] = mode

    with st.sidebar:
        source_label_sidebar = (
            "Demo Mode"
            if mode == "Explore Demo Data"
            else st.session_state.get("uploaded_filename", "Upload not analyzed")
        )
        st.caption(f"Data source: {source_label_sidebar}")
        st.download_button(
            "Download CSV template", template_bytes(), "transaction_template.csv", "text/csv"
        )
        monthly_budget = st.number_input(
            "Monthly budget (INR)", min_value=1_000, max_value=1_000_000, value=110_000, step=5_000
        )
        monthly_savings_goal = st.number_input(
            "Monthly savings goal (INR)",
            min_value=0,
            max_value=1_000_000,
            value=35_000,
            step=5_000,
        )
        st.info("Uploaded data is processed in memory only and is not persisted.")

    source_label = "Built-in sample data"
    synthetic = True
    if mode == "Upload My Transactions":
        if st.button("Reset uploaded data"):
            for key in ["uploaded_cleaned", "uploaded_report", "uploaded_filename"]:
                st.session_state.pop(key, None)
            st.rerun()
        if "uploaded_cleaned" not in st.session_state:
            _show_upload_wizard()
            return
        cleaned = st.session_state["uploaded_cleaned"]
        report = st.session_state["uploaded_report"]
        source_label = (
            f"Your Uploaded Data: {st.session_state.get('uploaded_filename', 'uploaded.csv')}"
        )
        synthetic = False
    else:
        st.info("Demo Mode: exploring deterministic synthetic sample data.")
        cleaned, report = cached_sample()

    if cleaned.empty:
        st.error("No valid transactions are available after cleaning.")
        st.json(report.as_dict())
        return

    with st.sidebar:
        min_date = cleaned["date"].min().date()
        max_date = cleaned["date"].max().date()
        with st.expander("Filters", expanded=False):
            selected_range = st.date_input(
                "Date range", (min_date, max_date), min_value=min_date, max_value=max_date
            )
            categories = sorted(cleaned["category"].unique())
            selected_categories = st.multiselect(
                "Categories",
                categories,
                default=categories,
                placeholder="Choose categories",
                label_visibility="visible",
            )
        if st.button("Reset controls"):
            st.rerun()

    if isinstance(selected_range, tuple) and len(selected_range) == 2:
        start_date, end_date = selected_range
    else:
        start_date, end_date = min_date, max_date

    filtered = apply_filters(
        cleaned, pd.Timestamp(start_date), pd.Timestamp(end_date), selected_categories
    )
    if filtered.empty:
        st.warning("No transactions match the selected filters.")
        return

    anomalies = cached_anomalies(filtered)
    forecast = cached_forecast(filtered, monthly_budget)
    health = financial_health_score(filtered, monthly_budget, anomalies)

    expected_income_default = int(
        max(filtered[filtered["transaction_type"] == "income"]["amount"].mean(), 1)
    )

    forecast_value = (
        forecast.next_month_prediction
        if forecast.status == "ok"
        else filtered[filtered["transaction_type"] == "expense"]["amount"].sum()
    )
    insights = build_insights(filtered, monthly_budget, anomalies, forecast)
    recommendations = build_recommendations(
        filtered, monthly_budget, monthly_savings_goal, anomalies
    )

    cleaned_bytes = dataframe_to_csv_bytes(filtered)
    anomaly_bytes = dataframe_to_csv_bytes(anomalies)
    monthly_bytes = monthly_summary_bytes(filtered)
    model_bytes = (
        forecast.metrics.to_csv(index=False).encode("utf-8")
        if forecast.metrics is not None
        else b"model,MAE,RMSE,R2\n"
    )
    report_bytes = cleaning_report_bytes(report)
    header(source_label, (pd.Timestamp(start_date), pd.Timestamp(end_date)), synthetic)
    st.sidebar.metric("Filtered rows", len(filtered))

    page_names = [
        "Executive Overview",
        "Data Quality & ETL",
        "Exploratory Analysis",
        "Forecast & Budget Risk",
        "Anomaly Detection",
        "Financial Health & Scenario Planner",
        "Methodology",
    ]
    selected_page = st.selectbox("Page", page_names, label_visibility="collapsed")

    scenario = calculate_scenario(
        float(forecast_value or 0),
        float(expected_income_default),
        float(monthly_budget),
        float(monthly_savings_goal),
        0.0,
        0.0,
    )

    if selected_page == "Executive Overview":
        show_executive_overview(filtered, monthly_budget, health, anomalies, insights)
    elif selected_page == "Data Quality & ETL":
        show_quality(report, filtered, cleaned_bytes, report_bytes)
    elif selected_page == "Exploratory Analysis":
        show_eda(filtered, monthly_budget, monthly_bytes)
    elif selected_page == "Forecast & Budget Risk":
        show_forecast(forecast, monthly_budget, model_bytes)
    elif selected_page == "Anomaly Detection":
        show_anomalies(anomalies, anomaly_bytes)
    elif selected_page == "Financial Health & Scenario Planner":
        scenario = show_health_and_scenario(
            health,
            float(forecast_value or 0),
            float(expected_income_default),
            float(monthly_budget),
            float(monthly_savings_goal),
            recommendations,
        )
    else:
        show_methodology(source_label)

    html_bytes = html_report(
        filtered,
        monthly_budget,
        health,
        forecast,
        anomalies,
        insights,
        scenario,
        source_label,
    )
    st.sidebar.download_button(
        "Download HTML report", html_bytes, "finsense_financial_report.html", "text/html"
    )


if __name__ == "__main__":
    main()
