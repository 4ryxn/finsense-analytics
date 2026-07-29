from __future__ import annotations

import pandas as pd

from finsense.anomalies import ANOMALY_COLUMNS, detect_anomalies
from finsense.forecasting import run_forecast
from finsense.insights import build_insights
from tests.test_features_forecasting import make_monthly_data


def test_anomaly_output_schema_and_determinism() -> None:
    df = make_monthly_data(18)
    unusual = df.iloc[[0]].copy()
    unusual["transaction_id"] = "unusual"
    unusual["date"] = pd.Timestamp("2026-06-29")
    unusual["transaction_type"] = "expense"
    unusual["category"] = "Healthcare"
    unusual["merchant"] = "City Clinic"
    unusual["amount"] = 125_000
    df.loc[len(df)] = unusual.iloc[0]

    first = detect_anomalies(df)
    second = detect_anomalies(df)
    assert list(first.columns) == ANOMALY_COLUMNS
    assert first["is_anomaly"].sum() >= 1
    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))


def test_rule_based_insights_use_calculations() -> None:
    df = make_monthly_data(18)
    anomalies = detect_anomalies(df)
    forecast = run_forecast(df)
    insights = build_insights(df, 60_000, anomalies, forecast)
    assert insights
    assert any("Savings rate" in item for item in insights)
    assert any("largest expense category" in item for item in insights)
