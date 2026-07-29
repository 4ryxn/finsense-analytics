from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from finsense.analytics import apply_filters  # noqa: E402
from finsense.anomalies import detect_anomalies  # noqa: E402
from finsense.data import (  # noqa: E402
    dataframe_to_csv_bytes,
    load_sample_transactions,
    template_bytes,
)
from finsense.forecasting import run_forecast  # noqa: E402
from finsense.insights import build_insights  # noqa: E402
from finsense.ui import (  # noqa: E402
    configure_page,
    show_about,
    show_anomalies,
    show_eda,
    show_forecast,
    show_overview,
    show_quality,
)


def main() -> None:
    configure_page()
    st.title("FinSense Analytics")
    st.caption("Single-user Streamlit dashboard for personal-finance analytics and compact ML.")

    with st.sidebar:
        source = st.radio("Data source", ["Built-in sample data", "Upload CSV"])
        uploaded = st.file_uploader(
            "Upload transaction CSV", type=["csv"], disabled=source == "Built-in sample data"
        )
        monthly_budget = st.number_input(
            "Monthly budget (INR)", min_value=1_000, max_value=1_000_000, value=95_000, step=5_000
        )
        st.download_button(
            "Download CSV template", template_bytes(), "transaction_template.csv", "text/csv"
        )
        st.info("Uploaded data is processed in memory only and is not persisted.")

    if source == "Upload CSV" and uploaded is not None:
        raw = pd.read_csv(uploaded)
        from finsense.etl import clean_transactions

        cleaned, report = clean_transactions(raw)
        raw_rows = len(raw)
    else:
        cleaned, report = load_sample_transactions()
        raw_rows = report.input_rows

    if cleaned.empty:
        st.error("No valid transactions are available after cleaning.")
        st.json(report.as_dict())
        return

    with st.sidebar:
        min_date = cleaned["date"].min().date()
        max_date = cleaned["date"].max().date()
        selected_range = st.date_input(
            "Date range", (min_date, max_date), min_value=min_date, max_value=max_date
        )
        categories = sorted(cleaned["category"].unique())
        selected_categories = st.multiselect("Categories", categories, default=categories)
        if st.button("Reset filters"):
            st.rerun()

    if isinstance(selected_range, tuple) and len(selected_range) == 2:
        start_date, end_date = selected_range
    else:
        start_date, end_date = min_date, max_date

    filtered = apply_filters(
        cleaned, pd.Timestamp(start_date), pd.Timestamp(end_date), selected_categories
    )
    anomalies = detect_anomalies(filtered)
    forecast = run_forecast(filtered)
    cleaned_bytes = dataframe_to_csv_bytes(filtered)
    anomaly_bytes = dataframe_to_csv_bytes(anomalies)

    st.sidebar.metric("Filtered rows", len(filtered))
    for item in build_insights(filtered, monthly_budget, anomalies, forecast):
        st.sidebar.caption(item)

    tabs = st.tabs(
        [
            "Overview",
            "Data Quality",
            "Exploratory Analysis",
            "Expense Forecast",
            "Anomaly Detection",
            "About / Methodology",
        ]
    )
    with tabs[0]:
        show_overview(filtered, monthly_budget)
    with tabs[1]:
        show_quality(raw_rows, report, filtered, cleaned_bytes)
    with tabs[2]:
        show_eda(filtered, monthly_budget)
    with tabs[3]:
        show_forecast(forecast)
    with tabs[4]:
        show_anomalies(anomalies, anomaly_bytes)
    with tabs[5]:
        show_about()


if __name__ == "__main__":
    main()
