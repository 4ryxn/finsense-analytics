from __future__ import annotations

from html import escape
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from finsense.analytics import (
    category_concentration,
    category_spending,
    compact_rupee,
    daily_heatmap_data,
    kpis,
    merchant_spending,
    monthly_summary,
    presentation_transactions,
    rupee,
    spending_statistics,
    weekday_weekend,
)
from finsense.scenarios import ScenarioResult, calculate_scenario
from finsense.scoring import HEALTH_SCORE_WEIGHTS, HealthScore


def _metric_rows(items: list[tuple[str, str, str]], columns: int = 4) -> None:
    for start in range(0, len(items), columns):
        cols = st.columns(columns)
        for col, (label, value, caption) in zip(cols, items[start : start + columns], strict=False):
            col.metric(label, value)
            if caption:
                col.caption(caption)


def _money_metric(label: str, value: float, caption_prefix: str = "Exact") -> tuple[str, str, str]:
    return label, compact_rupee(value), f"{caption_prefix}: {rupee(value)}"


def configure_page() -> None:
    st.set_page_config(page_title="FinSense Analytics", page_icon="₹", layout="wide")
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 8% 0%, rgba(34, 211, 238, .13), transparent 28rem),
                linear-gradient(135deg, #06111f 0%, #091827 48%, #07131f 100%);
            color: #e5f4ef;
        }
        .block-container {padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1500px;}
        [data-testid="stSidebar"] {background: #07111f;}
        [data-testid="stMetric"], div[data-testid="stExpander"] {
            background: rgba(16, 32, 51, .86);
            border: 1px solid rgba(56, 189, 248, .18);
            box-shadow: 0 16px 44px rgba(0, 0, 0, .18);
            border-radius: 8px;
            padding: .85rem;
        }
        .fs-header {
            border: 1px solid rgba(16,185,129,.25);
            background: linear-gradient(135deg, rgba(15, 118, 110, .30), rgba(14, 165, 233, .12));
            border-radius: 8px;
            padding: 1.2rem 1.35rem;
            margin-bottom: 1rem;
        }
        .fs-title {font-size: 2.1rem; font-weight: 760; margin: 0; letter-spacing: 0;}
        .fs-subtitle {color: #b9d7dd; margin-top: .25rem;}
        .fs-badge {
            display: inline-block;
            border: 1px solid rgba(125, 211, 252, .28);
            background: rgba(15, 23, 42, .46);
            border-radius: 999px;
            padding: .22rem .62rem;
            margin: .5rem .4rem 0 0;
            color: #dff8fb;
            font-size: .84rem;
        }
        h1, h2, h3 {letter-spacing: 0;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def header(
    source_label: str, date_range: tuple[pd.Timestamp, pd.Timestamp], synthetic: bool
) -> None:
    disclosure = "Synthetic sample data" if synthetic else "Uploaded CSV processed in memory"
    safe_source = escape(source_label)
    st.markdown(
        f"""
        <div class="fs-header">
            <div class="fs-title">FinSense Analytics</div>
            <div class="fs-subtitle">Portfolio-grade personal-finance analytics: ETL, EDA, forecasting, anomaly detection, scoring, scenarios, and reporting.</div>
            <span class="fs-badge">Source: {safe_source}</span>
            <span class="fs-badge">Range: {date_range[0]:%d %b %Y} to {date_range[1]:%d %b %Y}</span>
            <span class="fs-badge">{disclosure}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_badge(label: str, value: str) -> None:
    st.markdown(f'<span class="fs-badge">{label}: {value}</span>', unsafe_allow_html=True)


def metric_grid(
    df: pd.DataFrame, monthly_budget: float, health: HealthScore, anomaly_count: int
) -> None:
    values = kpis(df, monthly_budget)
    items = [
        _money_metric("Total Income", values["total_income"], "Income in selected filters"),
        _money_metric("Total Expenses", values["total_expenses"], "Expenses in selected filters"),
        _money_metric("Net Cash Flow", values["net_savings"], "Income minus expenses"),
        ("Savings Rate", f"{values['savings_rate']:.1f}%", "Net cash flow divided by income"),
        _money_metric(
            "Avg Monthly Expense",
            values["average_monthly_expense"],
            "Average across active months",
        ),
        (
            "Budget Use",
            f"{values['budget_utilization']:.1f}%",
            "Average monthly expense versus budget",
        ),
        ("Health Score", f"{health.score:.1f}", "Deterministic score from displayed metrics"),
        ("Unusual Txns", str(anomaly_count), "Isolation Forest unusual expenses"),
    ]
    _metric_rows(items, columns=4)


def chart_monthly(summary: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=summary["month"], y=summary["income"], name="Income", marker_color="#10b981")
    fig.add_bar(x=summary["month"], y=summary["expense"], name="Expense", marker_color="#38bdf8")
    fig.update_layout(
        barmode="group",
        yaxis_title="INR",
        height=390,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=35, b=10),
    )
    return fig


def show_executive_overview(
    df: pd.DataFrame,
    monthly_budget: float,
    health: HealthScore,
    anomalies: pd.DataFrame,
    insights: list[str],
) -> None:
    st.subheader("Executive Overview")
    metric_grid(
        df, monthly_budget, health, int(anomalies["is_anomaly"].sum()) if not anomalies.empty else 0
    )
    summary = monthly_summary(df)
    left, right = st.columns([1.35, 1])
    left.plotly_chart(chart_monthly(summary), use_container_width=True)
    savings = summary.assign(savings_rate=(summary["net"] / summary["income"] * 100).fillna(0))
    right.plotly_chart(
        px.line(savings, x="month", y="net", markers=True, title="Monthly Savings Trend"),
        use_container_width=True,
    )
    a, b, c = st.columns([1, 1, 1])
    a.plotly_chart(
        px.pie(
            category_spending(df),
            names="category",
            values="amount",
            title="Category Mix",
            hole=0.52,
        ),
        use_container_width=True,
    )
    budget_fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=kpis(df, monthly_budget)["budget_utilization"],
            number={"suffix": "%"},
            gauge={"axis": {"range": [0, 140]}, "bar": {"color": "#10b981"}},
            title={"text": "Budget Utilization"},
        )
    )
    budget_fig.update_layout(height=310, paper_bgcolor="rgba(0,0,0,0)")
    b.plotly_chart(budget_fig, use_container_width=True)
    c.write("Recent transactions")
    recent = presentation_transactions(df.sort_values("date", ascending=False).head(10))
    c.dataframe(recent, use_container_width=True, hide_index=True)
    st.write("Deterministic insight cards")
    for col, insight in zip(st.columns(min(4, max(len(insights), 1))), insights[:4], strict=False):
        col.info(insight)


def show_quality(report: Any, df: pd.DataFrame, csv_bytes: bytes, report_bytes: bytes) -> None:
    st.subheader("Data Quality & ETL")
    stages = ["Extract", "Validate", "Clean", "Transform", "Analyze"]
    cols = st.columns(len(stages))
    for col, stage in zip(cols, stages, strict=True):
        col.success(stage)
    status_badge("Validation", "Passed" if getattr(report, "is_valid", False) else "Needs review")
    _metric_rows(
        [
            ("Input Rows", str(report.input_rows), ""),
            ("Valid Output Rows", str(report.output_rows), ""),
            ("Rejected Rows", str(report.invalid_rows_rejected), ""),
            ("Duplicates Removed", str(report.duplicates_removed), ""),
            ("Start Date", f"{df['date'].min():%Y-%m-%d}", ""),
            ("End Date", f"{df['date'].max():%Y-%m-%d}", ""),
        ],
        columns=3,
    )
    left, right, third = st.columns(3)
    left.write("Missing-value handling")
    left.json(report.missing_values_handled or {"none": 0})
    right.write("Normalization counts")
    right.json(report.standardization_counts or {"none": 0})
    third.write("Validation issues")
    third.json(report.rejected_row_reasons or report.missing_required_columns or {"none": 0})
    st.dataframe(presentation_transactions(df.head(150)), use_container_width=True, hide_index=True)
    d1, d2 = st.columns(2)
    d1.download_button(
        "Download cleaned transactions", csv_bytes, "finsense_cleaned_transactions.csv", "text/csv"
    )
    d2.download_button(
        "Download cleaning report", report_bytes, "finsense_cleaning_report.csv", "text/csv"
    )


def show_eda(df: pd.DataFrame, monthly_budget: float, monthly_bytes: bytes) -> None:
    st.subheader("Exploratory Analysis")
    stats = spending_statistics(df, monthly_budget)
    values = [
        _money_metric("Mean", stats["mean"]),
        _money_metric("Median", stats["median"]),
        _money_metric("Std Dev", stats["std"]),
        _money_metric("P75", stats["p75"]),
        _money_metric("P90", stats["p90"]),
        ("MoM Growth", f"{stats['month_over_month_growth']:.1f}%", ""),
        ("Top-3 Concentration", f"{category_concentration(df):.1f}%", ""),
    ]
    _metric_rows(values, columns=4)
    expenses = df[df["transaction_type"] == "expense"]
    summary = monthly_summary(df)
    tabs = st.tabs(["Trends", "Categories", "Merchants", "Calendar patterns", "Statistics"])
    tabs[0].plotly_chart(
        px.line(
            summary,
            x="month",
            y=["income", "expense", "net"],
            markers=True,
            title="Monthly Income, Expense, and Savings",
        ),
        use_container_width=True,
    )
    tabs[0].plotly_chart(
        px.line(
            summary.assign(moving_average=summary["expense"].rolling(3, min_periods=1).mean()),
            x="month",
            y="moving_average",
            title="3-Month Expense Moving Average",
        ),
        use_container_width=True,
    )
    tabs[1].plotly_chart(
        px.bar(
            category_spending(df),
            x="category",
            y="amount",
            color="contribution",
            title="Category Distribution",
        ),
        use_container_width=True,
    )
    tabs[1].plotly_chart(
        px.box(expenses, x="category", y="amount", title="Transaction Box Plot by Category"),
        use_container_width=True,
    )
    tabs[2].plotly_chart(
        px.bar(
            merchant_spending(df, 15),
            x="merchant",
            y="amount",
            color="transactions",
            title="Top Merchants",
        ),
        use_container_width=True,
    )
    tabs[3].plotly_chart(
        px.bar(
            weekday_weekend(df), x="day_type", y="amount", title="Weekday versus Weekend Spending"
        ),
        use_container_width=True,
    )
    tabs[3].plotly_chart(
        px.density_heatmap(
            daily_heatmap_data(df),
            x="month",
            y="weekday",
            z="amount",
            title="Monthly Spending Heatmap",
        ),
        use_container_width=True,
    )
    tabs[4].plotly_chart(
        px.histogram(expenses, x="amount", nbins=45, title="Transaction Value Distribution"),
        use_container_width=True,
    )
    tabs[4].download_button(
        "Download monthly summary", monthly_bytes, "finsense_monthly_summary.csv", "text/csv"
    )


def show_forecast(result: Any, monthly_budget: float, model_bytes: bytes) -> None:
    st.subheader("Forecast & Budget Risk")
    if result.status != "ok":
        st.info(result.message)
        return
    uplift = result.model_uplift_pct * 100 if result.model_uplift_pct is not None else 0.0
    _metric_rows(
        [
            ("Selected Model", str(result.selected_model), result.model_uplift_message or ""),
            _money_metric("Next-Month Forecast", result.next_month_prediction),
            (
                "Prediction Range",
                f"{compact_rupee(result.prediction_lower)} to {compact_rupee(result.prediction_upper)}",
                f"Exact: {rupee(result.prediction_lower)} to {rupee(result.prediction_upper)}",
            ),
            _money_metric("Budget Gap", result.budget_gap or 0),
            ("Risk Label", str(result.risk_label), "Based on forecast range and selected budget"),
            (
                "Model Uplift vs Baseline",
                f"{uplift:.1f}%",
                "Validation MAE improvement threshold: 2.0%",
            ),
        ],
        columns=3,
    )
    st.info(result.model_uplift_message)
    if result.metrics is not None and (result.metrics["R2"] < 0).any():
        st.warning(
            "A negative R² means that model underperformed a mean-based reference on the validation period."
        )
    display_metrics = result.metrics.copy()
    display_metrics["model"] = display_metrics["model"].replace(
        {
            "Seasonal Baseline": "Seasonal baseline",
            "Linear Regression": "Linear regression",
            "Random Forest": "Random forest",
            "Gradient Boosting": "Gradient boosting",
        }
    )
    display_metrics[["MAE", "RMSE", "R2"]] = display_metrics[["MAE", "RMSE", "R2"]].round(2)
    display_metrics = display_metrics.rename(
        columns={"model": "Model", "MAE": "MAE", "RMSE": "RMSE", "R2": "R²"}
    )
    predictions = result.predictions.copy()
    predictions["month"] = pd.to_datetime(predictions["month"]).dt.strftime("%b %Y")
    left, right = st.columns(2)
    left.dataframe(display_metrics, use_container_width=True, hide_index=True)
    left.download_button(
        "Download model comparison", model_bytes, "finsense_model_comparison.csv", "text/csv"
    )
    right.plotly_chart(
        px.line(
            predictions,
            x="month",
            y=["actual", "predicted"],
            markers=True,
            title="Actual versus Predicted",
        ),
        use_container_width=True,
    )
    forecast_df = pd.DataFrame(
        {
            "Estimate": ["Lower estimate", "Point forecast", "Upper estimate", "Selected budget"],
            "Amount": [
                result.prediction_lower,
                result.next_month_prediction,
                result.prediction_upper,
                monthly_budget,
            ],
        }
    )
    budget_fig = px.bar(
        forecast_df,
        x="Estimate",
        y="Amount",
        color="Estimate",
        text=forecast_df["Amount"].map(compact_rupee),
        title=f"Budget versus Forecast for {result.next_month:%b %Y}",
    )
    budget_fig.update_traces(textposition="outside")
    budget_fig.update_layout(showlegend=False, height=380, margin=dict(l=10, r=10, t=48, b=10))
    st.plotly_chart(budget_fig, use_container_width=True)
    if result.importance is not None and not result.importance.empty:
        st.plotly_chart(
            px.bar(
                result.importance,
                x="importance",
                y="feature",
                orientation="h",
                title="Feature Importance",
            ),
            use_container_width=True,
        )
    with st.expander("Forecast methodology"):
        st.write(result.message)
        st.write(
            "The prediction range is derived from historical validation residuals, not from a statistical confidence interval."
        )


def show_anomalies(anomalies: pd.DataFrame, csv_bytes: bytes) -> None:
    st.subheader("Anomaly Detection")
    st.caption("Anomalies are unusual patterns in this dataset, not proof of fraud.")
    flagged = anomalies[anomalies["is_anomaly"]] if not anomalies.empty else anomalies
    cols = st.columns(3)
    cols[0].metric("Transactions Scored", len(anomalies))
    cols[1].metric("Anomaly Count", len(flagged))
    cols[2].metric(
        "Anomaly Rate", f"{len(flagged) / len(anomalies) * 100:.1f}%" if len(anomalies) else "0.0%"
    )
    left, right = st.columns(2)
    left.plotly_chart(
        px.histogram(
            anomalies, x="anomaly_score", color="severity", title="Anomaly Score Distribution"
        ),
        use_container_width=True,
    )
    right.plotly_chart(
        px.bar(
            flagged.groupby("category", as_index=False)["amount"].sum(),
            x="category",
            y="amount",
            title="Anomaly Category Distribution",
        ),
        use_container_width=True,
    )
    st.plotly_chart(
        px.scatter(
            flagged,
            x="date",
            y="amount",
            color="severity",
            hover_data=["merchant", "category", "explanation"],
            title="Unusual Transactions Over Time",
        ),
        use_container_width=True,
    )
    display_flagged = flagged.sort_values("anomaly_score").drop(
        columns=["transaction_id"], errors="ignore"
    )
    if not display_flagged.empty:
        display_flagged = display_flagged.copy()
        display_flagged["date"] = display_flagged["date"].dt.strftime("%Y-%m-%d")
        display_flagged["amount"] = display_flagged["amount"].map(rupee)
        display_flagged = display_flagged.rename(
            columns={
                "date": "Date",
                "category": "Category",
                "merchant": "Merchant",
                "amount": "Amount",
                "payment_method": "Payment Method",
                "anomaly_score": "Anomaly Score",
                "severity": "Severity",
                "is_anomaly": "Is Anomaly",
                "explanation": "Explanation",
            }
        )
    st.dataframe(display_flagged, use_container_width=True, hide_index=True)
    st.download_button("Download anomalies CSV", csv_bytes, "finsense_anomalies.csv", "text/csv")


def show_health_and_scenario(
    health: HealthScore,
    current_forecast: float,
    default_income: float,
    monthly_budget: float,
    monthly_savings_goal: float,
    recommendations: list[str],
) -> ScenarioResult:
    st.subheader("Financial Health & Scenario Planner")
    st.caption("Educational analysis only; this is not professional financial advice.")
    left, right = st.columns([1, 1.2])
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=health.score,
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#10b981"}},
            title={"text": health.band},
        )
    )
    gauge.update_layout(height=330, paper_bgcolor="rgba(0,0,0,0)")
    left.plotly_chart(gauge, use_container_width=True)
    right.write("Component scores")
    right.dataframe(
        pd.DataFrame(
            {
                "Component": [key.replace("_", " ").title() for key in health.components],
                "Score": list(health.components.values()),
                "Weight": [f"{HEALTH_SCORE_WEIGHTS[key]:.0%}" for key in health.components],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.write("Reasons and suggestions")
    for item in health.reasons + health.suggestions:
        st.info(item)
    st.write("Scenario inputs")
    i1, i2, i3 = st.columns(3)
    expected_income = i1.number_input(
        "Expected monthly income (INR)",
        min_value=0,
        max_value=2_000_000,
        value=int(default_income),
        step=5_000,
    )
    reduction_pct = i2.slider(
        "Planned expense reduction",
        0,
        50,
        8,
        help="Applies to forecasted recurring expenses.",
    )
    upcoming_expense = i3.number_input(
        "One-time upcoming expense (INR)",
        min_value=0,
        max_value=1_000_000,
        value=0,
        step=5_000,
    )
    scenario = calculate_scenario(
        current_forecast,
        float(expected_income),
        monthly_budget,
        monthly_savings_goal,
        float(reduction_pct),
        float(upcoming_expense),
    )
    current = calculate_scenario(
        current_forecast,
        float(expected_income),
        monthly_budget,
        monthly_savings_goal,
        0,
        0,
    )
    current_col, adjusted_col = st.columns(2)
    with current_col:
        st.write("Current Plan")
        _metric_rows(
            [
                _money_metric("Forecast Expense", current.current_forecast),
                _money_metric("Projected Cash Flow", current.projected_cash_flow),
                _money_metric("Budget Gap", current.budget_gap),
                ("Savings Goal Progress", f"{current.savings_goal_progress:.1f}%", ""),
            ],
            columns=2,
        )
    with adjusted_col:
        st.write("Adjusted Scenario")
        _metric_rows(
            [
                _money_metric("Adjusted Forecast", scenario.adjusted_forecast),
                _money_metric("Projected Cash Flow", scenario.projected_cash_flow),
                _money_metric("Budget Gap", scenario.budget_gap),
                ("Scenario Risk", scenario.risk_level, "Includes reduction and one-time expense"),
            ],
            columns=2,
        )
    st.write("Rule-based recommendations")
    for recommendation in recommendations[:5]:
        st.warning(recommendation)
    return scenario


def show_methodology(source_label: str) -> None:
    st.subheader("Methodology")
    with st.expander("Data Science lifecycle", expanded=True):
        st.write(
            "Ingestion accepts the sample CSV or an uploaded CSV, validates the documented schema, cleans records, transforms monthly features, runs EDA, evaluates forecasts, detects anomalies, scores financial health, and produces downloadable reports."
        )
    with st.expander("Financial Health Score weights"):
        st.json(HEALTH_SCORE_WEIGHTS)
        st.write(
            "Bands: Strong >= 80, Stable >= 65, Watch >= 50, otherwise At Risk. The score is educational and deterministic, not financial advice."
        )
    with st.expander("Forecasting and anomaly detection"):
        st.write(
            "Forecasting uses prior-month features, chronological validation, MAE model selection, and residual-derived ranges. Isolation Forest scores unusual expense transactions with deterministic random state."
        )
    st.info(f"Current source: {source_label}. Uploaded files are never persisted by the app.")
