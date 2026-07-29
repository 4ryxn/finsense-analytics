from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from finsense.config import RANDOM_STATE
from finsense.features import (
    FEATURE_COLUMNS,
    modeling_frame,
    monthly_features,
    next_month_feature_row,
)

MIN_MONTHS_FOR_FORECAST = 12


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


def _metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
    }


def run_forecast(df: pd.DataFrame) -> ForecastResult:
    monthly = monthly_features(df)
    if len(monthly) < MIN_MONTHS_FOR_FORECAST:
        return ForecastResult(
            status="insufficient_history",
            message=f"Need at least {MIN_MONTHS_FOR_FORECAST} months of transactions for a defensible holdout forecast.",
        )

    x, y = modeling_frame(df)
    if len(x) < 8:
        return ForecastResult(
            status="insufficient_history",
            message="Need more post-lag monthly observations after feature engineering.",
        )

    holdout = max(3, min(6, len(x) // 4))
    x_train, x_test = x.iloc[:-holdout], x.iloc[-holdout:]
    y_train, y_test = y.iloc[:-holdout], y.iloc[-holdout:]

    baseline_value = y_train.shift(1).dropna().tail(3).mean()
    if np.isnan(baseline_value):
        baseline_value = y_train.mean()
    baseline_pred = np.repeat(float(baseline_value), len(y_test))

    models: dict[str, Pipeline | RandomForestRegressor] = {
        "Linear Regression": Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
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
    selected_model = str(metrics.iloc[0]["model"])
    if selected_model == "Seasonal Baseline":
        next_prediction = float(monthly["monthly_expense"].tail(3).mean())
        ml_metrics = metrics[metrics["model"].ne("Seasonal Baseline")]
        importance_model_name = str(ml_metrics.iloc[0]["model"])
        importance_model = fitted[importance_model_name]
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
            pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": perm.importances_mean})
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
            pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": perm.importances_mean})
            .sort_values("importance", ascending=False)
            .head(10)
        )

    pred_df = pd.DataFrame(
        {
            "month": monthly.iloc[-holdout:]["month"].to_numpy(),
            "actual": y_test.to_numpy(),
            "predicted": test_predictions[selected_model],
        }
    )
    next_month = pd.Timestamp(monthly.iloc[-1]["month"]) + pd.offsets.MonthBegin(1)
    return ForecastResult(
        status="ok",
        message="Forecast uses chronological holdout validation and compares a baseline with two compact scikit-learn models.",
        next_month=next_month,
        next_month_prediction=max(0.0, next_prediction),
        selected_model=selected_model,
        metrics=metrics,
        predictions=pred_df,
        importance=importance,
        baseline_mae=float(metrics.loc[metrics["model"].eq("Seasonal Baseline"), "MAE"].iloc[0]),
    )
