from __future__ import annotations

import pandas as pd

from finsense.features import RAW_FEATURE_COLUMNS, modeling_frame, monthly_features
from finsense.forecasting import _metrics, run_forecast


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


def make_flat_expense_data(months: int = 18) -> pd.DataFrame:
    rows = []
    for i, month in enumerate(pd.date_range("2025-01-01", periods=months, freq="MS")):
        month = pd.Timestamp(month)
        rows.extend(
            [
                {
                    "transaction_id": f"inc-flat-{i}",
                    "date": month + pd.DateOffset(days=27),
                    "transaction_type": "income",
                    "category": "Salary",
                    "merchant": "Payroll",
                    "amount": 100_000,
                    "payment_method": "Bank Transfer",
                },
                {
                    "transaction_id": f"exp-flat-{i}",
                    "date": month + pd.DateOffset(days=5),
                    "transaction_type": "expense",
                    "category": "Housing",
                    "merchant": "Rent",
                    "amount": 50_000,
                    "payment_method": "Net Banking",
                },
            ]
        )
    return pd.DataFrame(rows)


def test_monthly_features_include_required_columns() -> None:
    features = monthly_features(make_monthly_data())
    assert set(RAW_FEATURE_COLUMNS).issubset(features.columns)
    assert len(features) == 18
    assert features["monthly_expense"].iloc[0] == 38_000


def test_modeling_frame_prevents_target_leakage_with_lags() -> None:
    x, y = modeling_frame(make_monthly_data())
    assert "monthly_expense" not in x.columns
    assert len(x) == len(y)
    assert x["expense_lag_1"].iloc[0] != y.iloc[0]
    assert x["feature_month"].is_monotonic_increasing


def test_target_month_data_cannot_appear_in_features() -> None:
    df = make_monthly_data(8)
    monthly = monthly_features(df)
    x, y = modeling_frame(df)
    first_month = x["feature_month"].iloc[0]
    target_row = monthly[monthly["month"].eq(first_month)].iloc[0]
    prior_row = monthly[monthly["month"].eq(first_month - pd.offsets.MonthBegin(1))].iloc[0]
    assert y.iloc[0] == target_row["monthly_expense"]
    assert x["previous_month_income"].iloc[0] == prior_row["monthly_income"]
    assert x["previous_month_savings"].iloc[0] == prior_row["net_savings"]
    assert x["previous_month_savings_rate"].iloc[0] == prior_row["savings_rate"]
    assert (
        x["previous_month_average_transaction_value"].iloc[0]
        == prior_row["average_transaction_value"]
    )
    assert x["previous_month_expense_frequency"].iloc[0] == prior_row["expense_frequency"]
    assert x["previous_month_expense_growth"].iloc[0] == prior_row["expense_growth"]
    assert x["expense_lag_1"].iloc[0] == prior_row["monthly_expense"]
    assert x["expense_lag_1"].iloc[0] != target_row["monthly_expense"]


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


def test_forecast_prediction_range_and_gradient_boosting() -> None:
    result = run_forecast(make_monthly_data(24), monthly_budget=75_000)
    assert result.status == "ok"
    assert result.prediction_lower is not None
    assert result.prediction_upper is not None
    assert result.prediction_lower <= result.next_month_prediction <= result.prediction_upper
    assert "Gradient Boosting" in set(result.metrics["model"])
    assert result.risk_label in {"Low", "Medium", "High", "Unknown"}


def test_baseline_selected_when_ml_uplift_below_threshold() -> None:
    result = run_forecast(make_flat_expense_data(24), monthly_budget=75_000)
    baseline_mae = result.metrics.loc[result.metrics["model"].eq("Seasonal Baseline"), "MAE"].iloc[
        0
    ]
    best_ml_mae = result.metrics.loc[~result.metrics["model"].eq("Seasonal Baseline"), "MAE"].min()
    if best_ml_mae < baseline_mae:
        assert (baseline_mae - best_ml_mae) / baseline_mae < 0.02
    assert result.selected_model == "Seasonal Baseline"


def test_metric_calculations_are_correct() -> None:
    values = _metrics(pd.Series([100.0, 200.0, 300.0]), pd.Series([110.0, 190.0, 330.0]))
    assert values["MAE"] == 50 / 3
    assert round(values["RMSE"], 6) == round(((100 + 100 + 900) / 3) ** 0.5, 6)
    assert round(values["R2"], 6) == 0.945
