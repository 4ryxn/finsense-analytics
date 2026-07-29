from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from finsense.analytics import kpis, monthly_summary

HEALTH_SCORE_WEIGHTS = {
    "savings_rate": 0.30,
    "expense_stability": 0.20,
    "budget_adherence": 0.20,
    "income_expense_ratio": 0.20,
    "anomaly_burden": 0.10,
}


@dataclass(frozen=True)
class HealthScore:
    score: float
    band: str
    components: dict[str, float]
    reasons: list[str]
    suggestions: list[str]


def _clip_score(value: float) -> float:
    return float(np.clip(value, 0, 100))


def _band(score: float) -> str:
    if score >= 80:
        return "Strong"
    if score >= 65:
        return "Stable"
    if score >= 50:
        return "Watch"
    return "At Risk"


def financial_health_score(
    df: pd.DataFrame,
    monthly_budget: float,
    anomalies: pd.DataFrame | None = None,
) -> HealthScore:
    metrics = kpis(df, monthly_budget)
    monthly = monthly_summary(df)
    expenses = monthly["expense"] if "expense" in monthly else pd.Series(dtype=float)

    savings_rate = metrics["savings_rate"]
    savings_component = _clip_score((savings_rate + 5) / 35 * 100)

    if len(expenses) >= 3 and expenses.mean() > 0:
        coefficient_variation = expenses.std(ddof=0) / expenses.mean()
        stability_component = _clip_score(100 - coefficient_variation * 180)
    else:
        stability_component = 55.0

    budget_utilization = metrics["budget_utilization"]
    budget_component = _clip_score(120 - max(budget_utilization, 0))

    ratio = (
        metrics["total_income"] / metrics["total_expenses"] if metrics["total_expenses"] else 2.0
    )
    ratio_component = _clip_score((ratio - 0.75) / 0.75 * 100)

    anomaly_count = (
        int(anomalies["is_anomaly"].sum()) if anomalies is not None and not anomalies.empty else 0
    )
    scored_count = len(anomalies) if anomalies is not None else 0
    anomaly_rate = anomaly_count / scored_count if scored_count else 0.0
    anomaly_component = _clip_score(100 - anomaly_rate * 500)

    components = {
        "savings_rate": savings_component,
        "expense_stability": stability_component,
        "budget_adherence": budget_component,
        "income_expense_ratio": ratio_component,
        "anomaly_burden": anomaly_component,
    }
    score = sum(components[key] * HEALTH_SCORE_WEIGHTS[key] for key in components)

    reasons: list[str] = [
        f"Savings rate is {savings_rate:.1f}%.",
        f"Average monthly budget utilization is {budget_utilization:.1f}%.",
        f"Income-to-expense ratio is {ratio:.2f}.",
        f"Unusual expense rate is {anomaly_rate * 100:.1f}%.",
    ]
    suggestions: list[str] = []
    if savings_rate < 15:
        suggestions.append("Prioritize recurring expense review to improve savings rate.")
    if budget_utilization > 95:
        suggestions.append("Build a monthly buffer below the selected budget.")
    if stability_component < 65:
        suggestions.append("Investigate categories driving month-to-month volatility.")
    if anomaly_rate > 0.04:
        suggestions.append("Review flagged transactions before treating them as recurring spend.")
    if not suggestions:
        suggestions.append("Maintain current savings discipline and monitor seasonal spikes.")

    return HealthScore(
        score=round(_clip_score(score), 1),
        band=_band(score),
        components={key: round(value, 1) for key, value in components.items()},
        reasons=reasons,
        suggestions=suggestions,
    )
