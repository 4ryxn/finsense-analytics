from __future__ import annotations

import numpy as np
import pandas as pd


def rupee(value: float | int) -> str:
    return f"₹{value:,.0f}"


def apply_filters(
    df: pd.DataFrame,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    categories: list[str] | None = None,
) -> pd.DataFrame:
    filtered = df.copy()
    if start_date is not None:
        filtered = filtered[filtered["date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        filtered = filtered[filtered["date"] <= pd.Timestamp(end_date)]
    if categories:
        filtered = filtered[filtered["category"].isin(categories)]
    return filtered.reset_index(drop=True)


def split_income_expense(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df[df["transaction_type"] == "income"], df[df["transaction_type"] == "expense"]


def kpis(df: pd.DataFrame, monthly_budget: float) -> dict[str, float]:
    income, expense = split_income_expense(df)
    total_income = float(income["amount"].sum())
    total_expense = float(expense["amount"].sum())
    net = total_income - total_expense
    month_count = max(int(df["date"].dt.to_period("M").nunique()), 1) if not df.empty else 1
    monthly_expense = total_expense / month_count
    return {
        "total_income": total_income,
        "total_expenses": total_expense,
        "net_savings": net,
        "savings_rate": (net / total_income * 100) if total_income else 0.0,
        "average_expense_transaction": float(expense["amount"].mean())
        if not expense.empty
        else 0.0,
        "budget_utilization": (monthly_expense / monthly_budget * 100) if monthly_budget else 0.0,
    }


def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["month", "income", "expense", "net"])
    summary = (
        df.assign(month=df["date"].dt.to_period("M").dt.to_timestamp())
        .pivot_table(
            index="month", columns="transaction_type", values="amount", aggfunc="sum", fill_value=0
        )
        .reset_index()
    )
    for col in ["income", "expense"]:
        if col not in summary:
            summary[col] = 0.0
    summary["net"] = summary["income"] - summary["expense"]
    return summary.sort_values("month")


def category_spending(df: pd.DataFrame) -> pd.DataFrame:
    expenses = df[df["transaction_type"] == "expense"]
    return (
        expenses.groupby("category", as_index=False)["amount"]
        .sum()
        .sort_values("amount", ascending=False)
    )


def merchant_spending(df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    expenses = df[df["transaction_type"] == "expense"]
    return (
        expenses.groupby("merchant", as_index=False)
        .agg(amount=("amount", "sum"), transactions=("amount", "size"))
        .sort_values("amount", ascending=False)
        .head(top_n)
    )


def spending_statistics(df: pd.DataFrame, monthly_budget: float) -> dict[str, float]:
    expense = df[df["transaction_type"] == "expense"]["amount"]
    summary = monthly_summary(df)
    monthly_expense = summary["expense"] if "expense" in summary else pd.Series(dtype=float)
    return {
        "mean": float(expense.mean()) if not expense.empty else 0.0,
        "median": float(expense.median()) if not expense.empty else 0.0,
        "std": float(expense.std(ddof=0)) if len(expense) > 1 else 0.0,
        "p75": float(expense.quantile(0.75)) if not expense.empty else 0.0,
        "p90": float(expense.quantile(0.90)) if not expense.empty else 0.0,
        "month_over_month_growth": float(monthly_expense.pct_change().iloc[-1] * 100)
        if len(monthly_expense) > 1 and monthly_expense.iloc[-2] > 0
        else 0.0,
        "moving_average_3m": float(monthly_expense.tail(3).mean())
        if not monthly_expense.empty
        else 0.0,
        "budget_utilization": (float(monthly_expense.mean()) / monthly_budget * 100)
        if monthly_budget
        else 0.0,
    }


def weekday_weekend(df: pd.DataFrame) -> pd.DataFrame:
    expenses = df[df["transaction_type"] == "expense"].copy()
    expenses["day_type"] = np.where(expenses["date"].dt.weekday >= 5, "Weekend", "Weekday")
    return expenses.groupby("day_type", as_index=False)["amount"].sum()


def daily_heatmap_data(df: pd.DataFrame) -> pd.DataFrame:
    expenses = df[df["transaction_type"] == "expense"].copy()
    if expenses.empty:
        return pd.DataFrame(columns=["weekday", "month", "amount"])
    expenses["weekday"] = expenses["date"].dt.day_name()
    expenses["month"] = expenses["date"].dt.month_name()
    return expenses.groupby(["weekday", "month"], as_index=False)["amount"].sum()
