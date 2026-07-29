from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from finsense.config import RANDOM_STATE
from finsense.features import (
    DISPLAY_FEATURE_LABELS,
    FEATURE_COLUMNS,
    modeling_frame,
    monthly_features,
    next_month_feature_row,
)

MIN_MONTHS_FOR_FORECAST = 12
ML_UPLIFT_THRESHOLD = 0.02


@dataclass(frozen=True)
class ForecastResult:
    status: str
    message: str
    next_month: pd.Timestamp | None = None
    next_month_prediction: float | None = None
    selected_model: str | None = None
    metrics: pd.DataFrame | None = None
    predictions: pd.DataFrame | None = None
    importance: pd.DataFrame | None = None
    baseline_mae: float | None = None
    model_uplift_pct: float | None = None
    model_uplift_message: str | None = None
    prediction_lower: float | None = None
    prediction_upper: float | None = None
    budget_gap: float | None = None
    risk_label: str | None = None


def _metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
    }


def _risk_label(budget_gap: float, upper: float, monthly_budget: float) -> str:
    if monthly_budget <= 0:
        return "Unknown"
    if budget_gap < 0 or upper > monthly_budget * 1.08:
        return "High"
    if upper > monthly_budget or budget_gap < monthly_budget * 0.08:
        return "Medium"
    return "Low"


def run_forecast(df: pd.DataFrame, monthly_budget: float = 0.0) -> ForecastResult:
    monthly = monthly_features(df)
    if len(monthly) < MIN_MONTHS_FOR_FORECAST:
        return ForecastResult(
            status="insufficient_history",
            message=f"Need at least {MIN_MONTHS_FOR_FORECAST} months of transactions for a defensible holdout forecast.",
        )

    x_all, y = modeling_frame(df)
    x = x_all[FEATURE_COLUMNS]
    feature_months = x_all["feature_month"]
    if len(x) < 8:
        return ForecastResult(
            status="insufficient_history",
            message="Need more post-lag monthly observations after feature engineering.",
        )

    holdout = max(3, min(6, len(x) // 4))
    x_train, x_test = x.iloc[:-holdout], x.iloc[-holdout:]
    y_train, y_test = y.iloc[:-holdout], y.iloc[-holdout:]
    test_months = feature_months.iloc[-holdout:]

    baseline_value = y_train.shift(1).dropna().tail(3).mean()
    if np.isnan(baseline_value):
        baseline_value = y_train.mean()
    baseline_pred = np.repeat(float(baseline_value), len(y_test))

    models: dict[str, Pipeline | RandomForestRegressor | GradientBoostingRegressor] = {
        "Linear Regression": Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=2,
            random_state=RANDOM_STATE,
        ),
    }

    rows = [{"model": "Seasonal Baseline", **_metrics(y_test, baseline_pred)}]
    fitted: dict[str, Pipeline | RandomForestRegressor] = {}
    test_predictions: dict[str, np.ndarray] = {"Seasonal Baseline": baseline_pred}

    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        rows.append({"model": name, **_metrics(y_test, pred)})
        fitted[name] = model
        test_predictions[name] = pred

    metrics = pd.DataFrame(rows).sort_values(["MAE", "RMSE"], ignore_index=True)
    baseline_mae = float(metrics.loc[metrics["model"].eq("Seasonal Baseline"), "MAE"].iloc[0])
    ml_metrics = metrics[metrics["model"].ne("Seasonal Baseline")].sort_values(
        ["MAE", "RMSE"], ignore_index=True
    )
    best_ml_model = str(ml_metrics.iloc[0]["model"])
    best_ml_mae = float(ml_metrics.iloc[0]["MAE"])
    model_uplift_pct = (baseline_mae - best_ml_mae) / baseline_mae if baseline_mae else 0.0
    selected_model = (
        best_ml_model
        if best_ml_mae < baseline_mae and model_uplift_pct >= ML_UPLIFT_THRESHOLD
        else "Seasonal Baseline"
    )
    if selected_model == "Seasonal Baseline" and best_ml_mae < baseline_mae:
        uplift_message = f"Best ML uplift was {model_uplift_pct * 100:.1f}%, below the 2.0% threshold, so the simpler seasonal baseline is preferred."
    elif selected_model == "Seasonal Baseline":
        uplift_message = "No ML model beat the seasonal baseline on validation MAE."
    else:
        uplift_message = f"{selected_model} improved validation MAE by {model_uplift_pct * 100:.1f}% versus the seasonal baseline."
    if selected_model == "Seasonal Baseline":
        next_prediction = float(monthly["monthly_expense"].tail(3).mean())
        importance_model = fitted[best_ml_model]
        importance_model.fit(x, y)
        perm = permutation_importance(
            importance_model,
            x_test,
            y_test,
            n_repeats=10,
            random_state=RANDOM_STATE,
            scoring="neg_mean_absolute_error",
        )
        importance = (
            pd.DataFrame(
                {
                    "feature": [DISPLAY_FEATURE_LABELS[name] for name in FEATURE_COLUMNS],
                    "importance": perm.importances_mean,
                }
            )
            .sort_values("importance", ascending=False)
            .head(10)
        )
    else:
        selected = fitted[selected_model]
        selected.fit(x, y)
        next_prediction = float(selected.predict(next_month_feature_row(monthly))[0])
        perm = permutation_importance(
            selected,
            x_test,
            y_test,
            n_repeats=10,
            random_state=RANDOM_STATE,
            scoring="neg_mean_absolute_error",
        )
        importance = (
            pd.DataFrame(
                {
                    "feature": [DISPLAY_FEATURE_LABELS[name] for name in FEATURE_COLUMNS],
                    "importance": perm.importances_mean,
                }
            )
            .sort_values("importance", ascending=False)
            .head(10)
        )

    pred_df = pd.DataFrame(
        {
            "month": test_months.to_numpy(),
            "actual": y_test.to_numpy(),
            "predicted": test_predictions[selected_model],
        }
    )
    residuals = y_test.to_numpy() - test_predictions[selected_model]
    residual_width = float(np.quantile(np.abs(residuals), 0.80)) if len(residuals) else 0.0
    prediction_lower = max(0.0, next_prediction - residual_width)
    prediction_upper = max(0.0, next_prediction + residual_width)
    budget_gap = monthly_budget - next_prediction if monthly_budget else None
    next_month = pd.Timestamp(monthly.iloc[-1]["month"]) + pd.offsets.MonthBegin(1)
    return ForecastResult(
        status="ok",
        message="Forecast uses strictly prior-month features, chronological holdout validation, a required seasonal baseline, and ML selection only when validation MAE improves by at least 2%.",
        next_month=next_month,
        next_month_prediction=max(0.0, next_prediction),
        selected_model=selected_model,
        metrics=metrics,
        predictions=pred_df,
        importance=importance,
        baseline_mae=baseline_mae,
        model_uplift_pct=model_uplift_pct,
        model_uplift_message=uplift_message,
        prediction_lower=prediction_lower,
        prediction_upper=prediction_upper,
        budget_gap=budget_gap,
        risk_label=_risk_label(budget_gap, prediction_upper, monthly_budget)
        if budget_gap is not None
        else "Unknown",
    )
