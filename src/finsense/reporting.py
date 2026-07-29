from __future__ import annotations

from datetime import UTC, datetime
from html import escape
from typing import Any

import pandas as pd

from finsense.analytics import kpis, monthly_summary, rupee
from finsense.forecasting import ForecastResult
from finsense.scenarios import ScenarioResult
from finsense.scoring import HEALTH_SCORE_WEIGHTS, HealthScore


def cleaning_report_bytes(report: Any) -> bytes:
    return pd.DataFrame([report.as_dict()]).to_csv(index=False).encode("utf-8")


def html_report(
    df: pd.DataFrame,
    monthly_budget: float,
    health: HealthScore,
    forecast: ForecastResult,
    anomalies: pd.DataFrame,
    insights: list[str],
    scenario: ScenarioResult,
    source_label: str,
) -> bytes:
    metrics = kpis(df, monthly_budget)
    monthly = monthly_summary(df)
    latest_month = monthly["month"].max().strftime("%b %Y") if not monthly.empty else "N/A"
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    anomaly_count = int(anomalies["is_anomaly"].sum()) if not anomalies.empty else 0
    forecast_text = (
        f"{rupee(forecast.next_month_prediction or 0)} for {forecast.next_month:%b %Y}"
        if forecast.status == "ok" and forecast.next_month is not None
        else escape(forecast.message)
    )
    insight_items = "".join(f"<li>{escape(item)}</li>" for item in insights)
    component_items = "".join(
        f"<li>{escape(key.replace('_', ' ').title())}: {value:.1f}/100</li>"
        for key, value in health.components.items()
    )
    weights = ", ".join(f"{key} {weight:.0%}" for key, weight in HEALTH_SCORE_WEIGHTS.items())
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>FinSense Analytics Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; color: #0f172a; margin: 32px; }}
    h1, h2 {{ color: #0f766e; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
    .card {{ border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px; }}
    .muted {{ color: #475569; }}
  </style>
</head>
<body>
  <h1>FinSense Analytics Report</h1>
  <p class="muted">Generated {escape(generated_at)}. Source: {escape(source_label)}. Uploaded data, when used, is processed in memory only.</p>
  <p class="muted">Sample data is synthetic and non-sensitive.</p>
  <div class="grid">
    <div class="card"><strong>Total income</strong><br>{rupee(metrics["total_income"])}</div>
    <div class="card"><strong>Total expenses</strong><br>{rupee(metrics["total_expenses"])}</div>
    <div class="card"><strong>Net cash flow</strong><br>{rupee(metrics["net_savings"])}</div>
    <div class="card"><strong>Savings rate</strong><br>{metrics["savings_rate"]:.1f}%</div>
    <div class="card"><strong>Health score</strong><br>{health.score:.1f}/100 ({escape(health.band)})</div>
    <div class="card"><strong>Anomalies</strong><br>{anomaly_count}</div>
  </div>
  <h2>Forecast</h2>
  <p>Selected model: {escape(str(forecast.selected_model or "N/A"))}. Next-month forecast: {forecast_text}.</p>
  <p>Scenario risk: {escape(scenario.risk_level)}. Adjusted forecast: {rupee(scenario.adjusted_forecast)}. Budget gap: {rupee(scenario.budget_gap)}.</p>
  <h2>Financial Health</h2>
  <p>Weights: {escape(weights)}.</p>
  <ul>{component_items}</ul>
  <h2>Insights</h2>
  <ul>{insight_items}</ul>
  <h2>Methodology</h2>
  <p>Latest analyzed month: {escape(latest_month)}. Forecasting uses chronological validation and MAE model selection. Anomaly detection uses a deterministic Isolation Forest pipeline. Health scoring is deterministic and educational, not professional financial advice.</p>
</body>
</html>"""
    return html.encode("utf-8")
