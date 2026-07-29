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
from finsense.etl import clean_transactions  # noqa: E402
from finsense.forecasting import ForecastResult, run_forecast  # noqa: E402
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


@st.cache_data(show_spinner=False)
def cached_sample() -> tuple[pd.DataFrame, object]:
    return load_sample_transactions()


@st.cache_data(show_spinner=False)
def cached_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    return detect_anomalies(df)


@st.cache_data(show_spinner=False)
def cached_forecast(df: pd.DataFrame, monthly_budget: float) -> ForecastResult:
    return run_forecast(df, monthly_budget)


def main() -> None:
    configure_page()

    with st.sidebar:
        source = st.radio("Data source", ["Built-in sample data", "Upload CSV"])
        uploaded = st.file_uploader(
            "Upload transaction CSV", type=["csv"], disabled=source == "Built-in sample data"
        )
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
    if source == "Upload CSV" and uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
            cleaned, report = clean_transactions(raw)
            source_label = uploaded.name
            synthetic = False
        except Exception as exc:  # noqa: BLE001
            st.error("The uploaded CSV could not be read or cleaned.")
            st.caption(str(exc))
            return
    else:
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
