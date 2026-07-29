from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from finsense.analytics import (
    category_spending,
    daily_heatmap_data,
    kpis,
    merchant_spending,
    monthly_summary,
    rupee,
    spending_statistics,
    weekday_weekend,
)


def configure_page() -> None:
    st.set_page_config(page_title="FinSense Analytics", page_icon="₹", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
        [data-testid="stMetric"] {
            background: #102033;
            border: 1px solid rgba(16,185,129,.22);
            padding: 1rem;
            border-radius: 8px;
        }
        [data-testid="stSidebar"] {background: #07111f;}
        h1, h2, h3 {letter-spacing: 0;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_grid(df: pd.DataFrame, monthly_budget: float) -> None:
    values = kpis(df, monthly_budget)
    cols = st.columns(6)
    items = [
        ("Total Income", rupee(values["total_income"])),
        ("Total Expenses", rupee(values["total_expenses"])),
        ("Net Savings", rupee(values["net_savings"])),
        ("Savings Rate", f"{values['savings_rate']:.1f}%"),
        ("Avg Expense", rupee(values["average_expense_transaction"])),
        ("Budget Use", f"{values['budget_utilization']:.1f}%"),
    ]
    for col, (label, value) in zip(cols, items, strict=True):
        col.metric(label, value)


def chart_monthly(summary: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=summary["month"], y=summary["income"], name="Income", marker_color="#10b981")
    fig.add_bar(x=summary["month"], y=summary["expense"], name="Expense", marker_color="#38bdf8")
    fig.update_layout(
        barmode="group", yaxis_title="INR", height=380, margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig


def show_overview(df: pd.DataFrame, monthly_budget: float) -> None:
    st.subheader("Overview")
    metric_grid(df, monthly_budget)
    summary = monthly_summary(df)
    left, right = st.columns(2)
    left.plotly_chart(chart_monthly(summary), use_container_width=True)
    right.plotly_chart(
        px.line(summary, x="month", y="net", markers=True, title="Monthly Net Cash Flow"),
        use_container_width=True,
    )
    left.plotly_chart(
        px.pie(
            category_spending(df),
            names="category",
            values="amount",
            title="Category Spending Breakdown",
            hole=0.45,
        ),
        use_container_width=True,
    )
    right.dataframe(
        df.sort_values("date", ascending=False).head(12),
        use_container_width=True,
        hide_index=True,
    )


def show_quality(raw_rows: int, report: object, df: pd.DataFrame, csv_bytes: bytes) -> None:
    st.subheader("Data Quality")
    st.success("Validation passed") if getattr(report, "is_valid", False) else st.error(
        "Validation failed"
    )
    cols = st.columns(4)
    cols[0].metric("Input Rows", raw_rows)
    cols[1].metric("Output Rows", report.output_rows)
    cols[2].metric("Duplicates Removed", report.duplicates_removed)
    cols[3].metric("Invalid Rows Rejected", report.invalid_rows_rejected)
    left, right = st.columns(2)
    left.write("Missing values handled")
    left.json(report.missing_values_handled or {"none": 0})
    right.write("Rejected-row reasons")
    right.json(report.rejected_row_reasons or {"none": 0})
    st.write("Standardization counts")
    st.json(report.standardization_counts or {"none": 0})
    st.dataframe(df.head(100), use_container_width=True, hide_index=True)
    st.download_button(
        "Download cleaned CSV", csv_bytes, "finsense_cleaned_transactions.csv", "text/csv"
    )


def show_eda(df: pd.DataFrame, monthly_budget: float) -> None:
    st.subheader("Exploratory Analysis")
    stats = spending_statistics(df, monthly_budget)
    cols = st.columns(6)
    for col, key in zip(
        cols, ["mean", "median", "std", "p75", "p90", "month_over_month_growth"], strict=True
    ):
        value = f"{stats[key]:.1f}%" if key == "month_over_month_growth" else rupee(stats[key])
        col.metric(key.replace("_", " ").title(), value)

    expenses = df[df["transaction_type"] == "expense"]
    summary = monthly_summary(df)
    tabs = st.tabs(["Trends", "Categories", "Merchants", "Timing", "Distribution"])
    tabs[0].plotly_chart(
        px.line(summary, x="month", y="expense", markers=True, title="Monthly Spending Trend"),
        use_container_width=True,
    )
    tabs[0].plotly_chart(
        px.line(
            summary.assign(moving_average=summary["expense"].rolling(3, min_periods=1).mean()),
            x="month",
            y="moving_average",
            title="3-Month Moving Average",
        ),
        use_container_width=True,
    )
    tabs[1].plotly_chart(
        px.bar(category_spending(df), x="category", y="amount", title="Category-wise Spending"),
        use_container_width=True,
    )
    tabs[1].plotly_chart(
        px.box(expenses, x="category", y="amount", title="Category Boxplot"),
        use_container_width=True,
    )
    tabs[2].plotly_chart(
        px.bar(
            merchant_spending(df),
            x="merchant",
            y="amount",
            color="transactions",
            title="Merchant Analysis",
        ),
        use_container_width=True,
    )
    tabs[3].plotly_chart(
        px.bar(weekday_weekend(df), x="day_type", y="amount", title="Weekday vs Weekend Spending"),
        use_container_width=True,
    )
    heat = daily_heatmap_data(df)
    tabs[3].plotly_chart(
        px.density_heatmap(heat, x="month", y="weekday", z="amount", title="Spending Heatmap"),
        use_container_width=True,
    )
    tabs[4].plotly_chart(
        px.histogram(expenses, x="amount", nbins=40, title="Expense Amount Histogram"),
        use_container_width=True,
    )


def show_forecast(result: object) -> None:
    st.subheader("Expense Forecast")
    if result.status != "ok":
        st.info(result.message)
        return
    st.metric(
        f"Next Month Estimate ({result.next_month:%b %Y})", rupee(result.next_month_prediction)
    )
    st.caption(
        "Prediction is an analytical estimate, not guaranteed financial advice. No statistical confidence interval is claimed."
    )
    left, right = st.columns(2)
    left.dataframe(result.metrics, use_container_width=True, hide_index=True)
    right.plotly_chart(
        px.line(
            result.predictions,
            x="month",
            y=["actual", "predicted"],
            markers=True,
            title=f"Actual vs Predicted: {result.selected_model}",
        ),
        use_container_width=True,
    )
    if result.importance is not None and not result.importance.empty:
        st.plotly_chart(
            px.bar(
                result.importance,
                x="importance",
                y="feature",
                orientation="h",
                title="Permutation Importance",
            ),
            use_container_width=True,
        )


def show_anomalies(anomalies: pd.DataFrame, csv_bytes: bytes) -> None:
    st.subheader("Anomaly Detection")
    st.caption("Anomaly means unusual for this dataset, not fraudulent.")
    flagged = anomalies[anomalies["is_anomaly"]]
    st.metric("Flagged Transactions", len(flagged))
    st.dataframe(flagged, use_container_width=True, hide_index=True)
    st.download_button("Download anomaly CSV", csv_bytes, "finsense_anomalies.csv", "text/csv")


def show_about() -> None:
    st.subheader("About / Methodology")
    st.write(
        "FinSense Analytics is a compact single-user Streamlit dashboard for INR personal-finance exploration. "
        "CSV data is processed in memory and uploaded files are not persisted."
    )
    st.write(
        "Forecasting aggregates monthly expenses, creates lag and rolling features from prior months, uses chronological holdout validation, "
        "and compares a seasonal baseline with Linear Regression and Random Forest models using MAE, RMSE, and R-squared where valid."
    )
    st.write(
        "Anomaly detection uses a deterministic scikit-learn Isolation Forest pipeline over amount, time, category, merchant, and payment-method features."
    )
