from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from finsense.config import RANDOM_STATE

ANOMALY_COLUMNS = [
    "transaction_id",
    "date",
    "category",
    "merchant",
    "amount",
    "payment_method",
    "anomaly_score",
    "is_anomaly",
    "explanation",
]


def _explain(row: pd.Series, expenses: pd.DataFrame) -> str:
    category_median = expenses.loc[expenses["category"].eq(row["category"]), "amount"].median()
    merchant_median = expenses.loc[expenses["merchant"].eq(row["merchant"]), "amount"].median()
    reasons: list[str] = []
    if pd.notna(category_median) and category_median > 0 and row["amount"] >= category_median * 2.5:
        reasons.append("much higher than category median")
    if pd.notna(merchant_median) and merchant_median > 0 and row["amount"] >= merchant_median * 2.5:
        reasons.append("unusual merchant/category amount")
    if row["date"].day <= 3 or row["date"].day >= 28:
        reasons.append("unusual timing near month boundary")
    return (
        "; ".join(reasons)
        if reasons
        else "unusual combination of amount, category, merchant, or timing"
    )


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.04) -> pd.DataFrame:
    expenses = df[df["transaction_type"] == "expense"].copy()
    if expenses.empty:
        return pd.DataFrame(columns=ANOMALY_COLUMNS)

    expenses["day"] = expenses["date"].dt.day
    expenses["month"] = expenses["date"].dt.month
    expenses["weekday"] = expenses["date"].dt.weekday
    expenses["log_amount"] = np.log1p(expenses["amount"])

    numeric = ["log_amount", "day", "month", "weekday"]
    categorical = ["category", "merchant", "payment_method"]
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore", min_frequency=2), categorical),
        ]
    )
    model = Pipeline(
        [
            ("prep", preprocessor),
            (
                "isolation_forest",
                IsolationForest(
                    n_estimators=200,
                    contamination=min(max(contamination, 0.01), 0.15),
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(expenses[numeric + categorical])
    expenses["anomaly_score"] = model.decision_function(expenses[numeric + categorical])
    expenses["is_anomaly"] = model.predict(expenses[numeric + categorical]) == -1
    expenses["explanation"] = expenses.apply(lambda row: _explain(row, expenses), axis=1)
    return expenses[ANOMALY_COLUMNS].sort_values(
        ["is_anomaly", "anomaly_score"], ascending=[False, True]
    )
