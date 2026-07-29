from __future__ import annotations

import pandas as pd

from finsense.features import FEATURE_COLUMNS, modeling_frame, monthly_features
from finsense.forecasting import run_forecast


def make_monthly_data(months: int = 18) -> pd.DataFrame:
    rows = []
    for i, month in enumerate(pd.date_range("2025-01-01", periods=months, freq="MS")):
        month = pd.Timestamp(month)
        rows.append(
            {
                "transaction_id": f"inc-{i}",
                "date": month + pd.DateOffset(days=27),
                "transaction_type": "income",
                "category": "Salary",
                "merchant": "Payroll",
                "amount": 100_000 + i * 500,
                "payment_method": "Bank Transfer",
            }
        )
        rows.append(
            {
                "transaction_id": f"exp-{i}",
                "date": month + pd.DateOffset(days=5),
                "transaction_type": "expense",
                "category": "Housing",
                "merchant": "Rent",
                "amount": 30_000 + i * 700,
                "payment_method": "Net Banking",
            }
        )
        rows.append(
            {
                "transaction_id": f"food-{i}",
                "date": month + pd.DateOffset(days=12),
                "transaction_type": "expense",
                "category": "Food",
                "merchant": "Grocery",
                "amount": 8_000 + (i % 3) * 500,
                "payment_method": "UPI",
            }
        )
    return pd.DataFrame(rows)


def test_monthly_features_include_required_columns() -> None:
    features = monthly_features(make_monthly_data())
    assert set(FEATURE_COLUMNS).issubset(features.columns)
    assert len(features) == 18
    assert features["monthly_expense"].iloc[0] == 38_000


def test_modeling_frame_prevents_target_leakage_with_lags() -> None:
    x, y = modeling_frame(make_monthly_data())
    assert "monthly_expense" not in x.columns
    assert len(x) == len(y)
    assert x["expense_lag_1"].iloc[0] != y.iloc[0]


def test_insufficient_forecast_history() -> None:
    result = run_forecast(make_monthly_data(6))
    assert result.status == "insufficient_history"


def test_forecast_is_deterministic() -> None:
    first = run_forecast(make_monthly_data(20))
    second = run_forecast(make_monthly_data(20))
    assert first.status == "ok"
    assert second.status == "ok"
    assert first.selected_model == second.selected_model
    assert first.next_month_prediction == second.next_month_prediction
    assert list(first.metrics["model"]) == list(second.metrics["model"])
