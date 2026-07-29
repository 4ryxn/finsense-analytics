from __future__ import annotations

import pandas as pd

from finsense.analytics import category_spending, kpis, monthly_summary, rupee
from finsense.forecasting import ForecastResult


def build_insights(
    df: pd.DataFrame,
    monthly_budget: float,
    anomalies: pd.DataFrame | None = None,
    forecast: ForecastResult | None = None,
) -> list[str]:
    if df.empty:
        return ["No transactions match the current filters."]

    insights: list[str] = []
    metrics = kpis(df, monthly_budget)
    savings_rate = metrics["savings_rate"]
    if savings_rate >= 25:
        insights.append(f"Savings rate is strong at {savings_rate:.1f}% for the selected period.")
    elif savings_rate >= 0:
        insights.append(f"Savings rate is positive at {savings_rate:.1f}%, with room to improve.")
    else:
        insights.append(
            f"Expenses exceeded income by {rupee(abs(metrics['net_savings']))} in this view."
        )

    categories = category_spending(df)
    if not categories.empty:
        top = categories.iloc[0]
        share = top["amount"] / categories["amount"].sum() * 100
        insights.append(
            f"{top['category']} is the largest expense category at {share:.1f}% of spending."
        )

    monthly = monthly_summary(df)
    if len(monthly) >= 2 and monthly["expense"].iloc[-2] > 0:
        change = (monthly["expense"].iloc[-1] / monthly["expense"].iloc[-2] - 1) * 100
        direction = "increased" if change >= 0 else "decreased"
        insights.append(
            f"Latest monthly expenses {direction} by {abs(change):.1f}% versus the previous month."
        )

    if monthly_budget:
        utilization = metrics["budget_utilization"]
        if utilization > 100:
            insights.append(
                f"Average monthly expenses are {utilization:.1f}% of the selected budget."
            )
        else:
            insights.append(f"Average monthly budget utilization is {utilization:.1f}%.")

    if anomalies is not None and not anomalies.empty:
        count = int(anomalies["is_anomaly"].sum())
        if count:
            insights.append(
                f"{count} unusual expense transactions are flagged for review; unusual does not mean fraudulent."
            )

    if forecast and forecast.status == "ok" and forecast.next_month_prediction is not None:
        insights.append(
            f"The selected forecasting model estimates next-month expenses at {rupee(forecast.next_month_prediction)}."
        )

    return insights
