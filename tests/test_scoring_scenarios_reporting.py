from __future__ import annotations

import pandas as pd

from finsense.analytics import apply_filters
from finsense.anomalies import detect_anomalies
from finsense.forecasting import run_forecast
from finsense.insights import build_insights
from finsense.reporting import html_report
from finsense.scenarios import calculate_scenario
from finsense.scoring import financial_health_score
from tests.test_features_forecasting import make_monthly_data


def test_filtering_consistency() -> None:
    df = make_monthly_data(6)
    filtered = apply_filters(df, pd.Timestamp("2025-02-01"), pd.Timestamp("2025-03-31"), ["Food"])
    assert filtered["date"].min() >= pd.Timestamp("2025-02-01")
    assert filtered["date"].max() <= pd.Timestamp("2025-03-31")
    assert set(filtered["category"]) == {"Food"}


def test_financial_health_score_bounds_and_components() -> None:
    df = make_monthly_data(18)
    anomalies = detect_anomalies(df)
    health = financial_health_score(df, 70_000, anomalies)
    assert 0 <= health.score <= 100
    assert set(health.components) == {
        "savings_rate",
        "expense_stability",
        "budget_adherence",
        "income_expense_ratio",
        "anomaly_burden",
    }
    assert health.reasons
    assert health.suggestions


def test_scenario_calculation_risk_and_adjustment() -> None:
    scenario = calculate_scenario(100_000, 150_000, 90_000, 40_000, 10, 5_000)
    assert scenario.adjusted_forecast == 95_000
    assert scenario.projected_cash_flow == 55_000
    assert scenario.savings_goal_progress > 100
    assert scenario.risk_level == "Medium"


def test_html_report_escapes_user_content() -> None:
    df = make_monthly_data(18)
    anomalies = detect_anomalies(df)
    forecast = run_forecast(df, 70_000)
    health = financial_health_score(df, 70_000, anomalies)
    scenario = calculate_scenario(60_000, 120_000, 70_000, 30_000, 0)
    report = html_report(
        df,
        70_000,
        health,
        forecast,
        anomalies,
        build_insights(df, 70_000, anomalies, forecast),
        scenario,
        "<script>alert(1)</script>",
    ).decode("utf-8")
    assert "<script>alert(1)</script>" not in report
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in report
