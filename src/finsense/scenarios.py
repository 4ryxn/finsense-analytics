from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioResult:
    current_forecast: float
    adjusted_forecast: float
    expected_income: float
    projected_cash_flow: float
    projected_savings: float
    budget_gap: float
    savings_goal_progress: float
    risk_level: str


def calculate_scenario(
    current_forecast: float,
    expected_monthly_income: float,
    monthly_budget: float,
    monthly_savings_goal: float,
    expense_reduction_pct: float,
    one_time_expense: float = 0.0,
) -> ScenarioResult:
    reduction = max(0.0, min(expense_reduction_pct, 100.0)) / 100
    adjusted_forecast = max(0.0, current_forecast * (1 - reduction) + max(0.0, one_time_expense))
    projected_cash_flow = expected_monthly_income - adjusted_forecast
    projected_savings = max(0.0, projected_cash_flow)
    budget_gap = monthly_budget - adjusted_forecast
    savings_goal_progress = (
        projected_savings / monthly_savings_goal * 100 if monthly_savings_goal > 0 else 100.0
    )

    if budget_gap < 0 and savings_goal_progress < 75:
        risk = "High"
    elif budget_gap < 0 or savings_goal_progress < 90:
        risk = "Medium"
    else:
        risk = "Low"

    return ScenarioResult(
        current_forecast=round(current_forecast, 2),
        adjusted_forecast=round(adjusted_forecast, 2),
        expected_income=round(expected_monthly_income, 2),
        projected_cash_flow=round(projected_cash_flow, 2),
        projected_savings=round(projected_savings, 2),
        budget_gap=round(budget_gap, 2),
        savings_goal_progress=round(savings_goal_progress, 1),
        risk_level=risk,
    )
