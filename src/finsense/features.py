from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "previous_month_income",
    "previous_month_savings",
    "previous_month_savings_rate",
    "previous_month_average_transaction_value",
    "previous_month_expense_frequency",
    "previous_month_weekend_spending_ratio",
    "previous_month_merchant_diversity",
    "previous_month_category_diversity",
    "previous_month_expense_growth",
    "expense_lag_1",
    "expense_lag_2",
    "expense_lag_3",
    "rolling_expense_3m",
    "rolling_expense_6m",
    "month_sin",
    "month_cos",
]

RAW_FEATURE_COLUMNS = [
    "monthly_income",
    "net_savings",
    "savings_rate",
    "average_transaction_value",
    "expense_frequency",
    "weekend_spending_ratio",
    "merchant_diversity",
    "category_diversity",
    "expense_growth",
    "monthly_expense",
    "expense_lag_1",
    "expense_lag_2",
    "rolling_expense_3m",
    "rolling_expense_6m",
    "month_sin",
    "month_cos",
]

DISPLAY_FEATURE_LABELS = {
    "previous_month_income": "Previous-month income",
    "previous_month_savings": "Previous-month savings",
    "previous_month_savings_rate": "Previous-month savings rate",
    "previous_month_average_transaction_value": "Previous-month average transaction",
    "previous_month_expense_frequency": "Previous-month expense frequency",
    "previous_month_weekend_spending_ratio": "Previous-month weekend spending ratio",
    "previous_month_merchant_diversity": "Previous-month merchant diversity",
    "previous_month_category_diversity": "Previous-month category diversity",
    "previous_month_expense_growth": "Previous-month expense growth",
    "expense_lag_1": "Expense lag 1",
    "expense_lag_2": "Expense lag 2",
    "expense_lag_3": "Expense lag 3",
    "rolling_expense_3m": "Three-month rolling expense",
    "rolling_expense_6m": "Six-month rolling expense",
    "month_sin": "Forecast month sine",
    "month_cos": "Forecast month cosine",
}


def monthly_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["month", "monthly_expense", *FEATURE_COLUMNS])

    working = df.copy()
    working["month"] = working["date"].dt.to_period("M").dt.to_timestamp()
    expenses = working[working["transaction_type"] == "expense"]
    income = working[working["transaction_type"] == "income"]

    monthly = pd.DataFrame({"month": sorted(working["month"].unique())})
    expense_agg = expenses.groupby("month").agg(
        monthly_expense=("amount", "sum"),
        average_transaction_value=("amount", "mean"),
        expense_frequency=("amount", "size"),
        merchant_diversity=("merchant", "nunique"),
        category_diversity=("category", "nunique"),
    )
    income_agg = income.groupby("month").agg(monthly_income=("amount", "sum"))

    monthly = monthly.merge(expense_agg, on="month", how="left").merge(
        income_agg, on="month", how="left"
    )
    fill_zero = [
        "monthly_expense",
        "average_transaction_value",
        "expense_frequency",
        "merchant_diversity",
        "category_diversity",
        "monthly_income",
    ]
    monthly[fill_zero] = monthly[fill_zero].fillna(0.0)
    monthly["net_savings"] = monthly["monthly_income"] - monthly["monthly_expense"]
    monthly["savings_rate"] = np.where(
        monthly["monthly_income"] > 0,
        monthly["net_savings"] / monthly["monthly_income"] * 100,
        0.0,
    )

    weekend = expenses.assign(is_weekend=expenses["date"].dt.weekday >= 5)
    weekend_ratio = weekend.groupby("month")["is_weekend"].mean().rename("weekend_spending_ratio")
    monthly = monthly.merge(weekend_ratio, on="month", how="left")
    monthly["weekend_spending_ratio"] = monthly["weekend_spending_ratio"].fillna(0.0)

    monthly = monthly.sort_values("month").reset_index(drop=True)
    monthly["expense_growth"] = (
        monthly["monthly_expense"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    )
    for lag in [1, 2, 3]:
        monthly[f"expense_lag_{lag}"] = monthly["monthly_expense"].shift(lag)
    monthly["rolling_expense_3m"] = (
        monthly["monthly_expense"].shift(1).rolling(3, min_periods=1).mean()
    )
    monthly["rolling_expense_6m"] = (
        monthly["monthly_expense"].shift(1).rolling(6, min_periods=1).mean()
    )
    month_number = monthly["month"].dt.month
    monthly["month_sin"] = np.sin(2 * np.pi * month_number / 12)
    monthly["month_cos"] = np.cos(2 * np.pi * month_number / 12)
    return monthly.fillna(0.0)


def modeling_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features = monthly_features(df)
    if len(features) <= 3:
        return pd.DataFrame(columns=FEATURE_COLUMNS), pd.Series(dtype=float, name="monthly_expense")
    predictors = pd.DataFrame(
        {
            "previous_month_income": features["monthly_income"].shift(1),
            "previous_month_savings": features["net_savings"].shift(1),
            "previous_month_savings_rate": features["savings_rate"].shift(1),
            "previous_month_average_transaction_value": features["average_transaction_value"].shift(
                1
            ),
            "previous_month_expense_frequency": features["expense_frequency"].shift(1),
            "previous_month_weekend_spending_ratio": features["weekend_spending_ratio"].shift(1),
            "previous_month_merchant_diversity": features["merchant_diversity"].shift(1),
            "previous_month_category_diversity": features["category_diversity"].shift(1),
            "previous_month_expense_growth": features["expense_growth"].shift(1),
            "expense_lag_1": features["monthly_expense"].shift(1),
            "expense_lag_2": features["monthly_expense"].shift(2),
            "expense_lag_3": features["monthly_expense"].shift(3),
            "rolling_expense_3m": features["monthly_expense"].shift(1).rolling(3).mean(),
            "rolling_expense_6m": features["monthly_expense"].shift(1).rolling(6).mean(),
            "month_sin": features["month_sin"],
            "month_cos": features["month_cos"],
        }
    )
    frame = features.iloc[3:].copy()
    x = predictors.iloc[3:].fillna(0.0).reset_index(drop=True)
    x["feature_month"] = frame["month"].reset_index(drop=True)
    y = frame["monthly_expense"].reset_index(drop=True)
    return x, y


def next_month_feature_row(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    last = monthly.iloc[-1]
    next_month = pd.Timestamp(last["month"]) + pd.offsets.MonthBegin(1)
    recent = monthly.tail(6)
    row = {
        "previous_month_income": float(last["monthly_income"]),
        "previous_month_savings": float(last["monthly_income"] - last["monthly_expense"]),
        "previous_month_savings_rate": float(last["savings_rate"]),
        "previous_month_average_transaction_value": float(last["average_transaction_value"]),
        "previous_month_expense_frequency": float(last["expense_frequency"]),
        "previous_month_weekend_spending_ratio": float(last["weekend_spending_ratio"]),
        "previous_month_merchant_diversity": float(last["merchant_diversity"]),
        "previous_month_category_diversity": float(last["category_diversity"]),
        "previous_month_expense_growth": float(
            monthly["monthly_expense"].pct_change().fillna(0).iloc[-1]
        ),
        "expense_lag_1": float(last["monthly_expense"]),
        "expense_lag_2": float(monthly["monthly_expense"].iloc[-2]) if len(monthly) >= 2 else 0.0,
        "expense_lag_3": float(monthly["monthly_expense"].iloc[-3]) if len(monthly) >= 3 else 0.0,
        "rolling_expense_3m": float(monthly["monthly_expense"].tail(3).mean()),
        "rolling_expense_6m": float(recent["monthly_expense"].mean()),
        "month_sin": float(np.sin(2 * np.pi * next_month.month / 12)),
        "month_cos": float(np.cos(2 * np.pi * next_month.month / 12)),
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)
