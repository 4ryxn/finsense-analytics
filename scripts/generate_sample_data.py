from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "sample_transactions.csv"
RNG = np.random.default_rng(42)


def txn_id(date: pd.Timestamp, kind: str, category: str, merchant: str, amount: float) -> str:
    raw = f"{date:%Y-%m-%d}|{kind}|{category}|{merchant}|{amount:.2f}"
    return "txn-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def add(
    rows: list[dict[str, object]],
    date: pd.Timestamp,
    kind: str,
    category: str,
    merchant: str,
    amount: float,
    method: str,
) -> None:
    rows.append(
        {
            "transaction_id": txn_id(date, kind, category, merchant, amount),
            "date": date.strftime("%Y-%m-%d"),
            "transaction_type": kind,
            "category": category,
            "merchant": merchant,
            "amount": round(float(amount), 2),
            "payment_method": method,
        }
    )


def month_date(month: pd.Timestamp, low: int, high: int) -> pd.Timestamp:
    month = pd.Timestamp(month)
    day = int(RNG.integers(low, high + 1))
    return month + pd.DateOffset(days=int(min(day, month.days_in_month) - 1))


def main() -> None:
    rows: list[dict[str, object]] = []
    months = pd.date_range("2023-07-01", "2026-06-01", freq="MS")
    merchants = {
        "Food": ["BigBasket", "Blinkit", "Star Bazaar", "Local Grocery", "Swiggy"],
        "Transport": ["Uber", "Ola", "Metro Card", "Indian Oil"],
        "Utilities": ["Tata Power", "Airtel Fiber", "Jio Mobile", "Mahanagar Gas"],
        "Shopping": ["Amazon India", "Myntra", "Croma", "Decathlon"],
        "Healthcare": ["Apollo Pharmacy", "Practo", "City Clinic"],
        "Entertainment": ["BookMyShow", "Netflix", "Spotify", "PVR Cinemas"],
        "Education": ["Coursera", "Udemy", "Kindle Store"],
        "Other": ["Society Office", "Stationery Hub", "Gift Store"],
    }
    methods = ["UPI", "Credit Card", "Debit Card", "Wallet", "Net Banking", "Cash"]

    for index, month in enumerate(months):
        month = pd.Timestamp(month)
        salary = 145_000 + index * 1_200 + RNG.normal(0, 1_500)
        add(
            rows,
            month + pd.DateOffset(days=27),
            "income",
            "Salary",
            "Northstar Payroll",
            salary,
            "Bank Transfer",
        )
        if RNG.random() < 0.35:
            add(
                rows,
                month_date(month, 5, 20),
                "income",
                "Freelance",
                "Independent Client",
                RNG.normal(22_000, 5_500),
                "Bank Transfer",
            )
        if month.month in [3, 9, 12]:
            add(
                rows,
                month_date(month, 10, 24),
                "income",
                "Investment",
                "Index Fund Dividend",
                RNG.normal(8_500, 1_200),
                "Net Banking",
            )

        add(
            rows,
            month + pd.DateOffset(days=4),
            "expense",
            "Housing",
            "Metro Homes Rent",
            42_000 + int(index // 12) * 2_500,
            "Net Banking",
        )
        category_plan = {
            "Food": (7, 1_100, 850),
            "Transport": (5, 750, 450),
            "Utilities": (3, 2_200, 800),
            "Shopping": (3, 3_400, 2_200),
            "Healthcare": (1, 1_600, 1_000),
            "Entertainment": (3, 1_400, 900),
            "Education": (1, 2_400, 1_300),
            "Other": (2, 1_200, 800),
        }
        seasonal = 1.0
        if month.month in [10, 11, 12]:
            seasonal = 1.22
        elif month.month in [4, 5]:
            seasonal = 1.12
        for category, (count, center, spread) in category_plan.items():
            for _ in range(count + int(RNG.integers(-1, 2))):
                merchant = str(RNG.choice(merchants[category]))
                method = str(RNG.choice(methods, p=[0.42, 0.28, 0.10, 0.10, 0.06, 0.04]))
                amount = max(120, RNG.normal(center * seasonal, spread))
                add(rows, month_date(month, 1, 27), "expense", category, merchant, amount, method)

    unusual = [
        ("2024-11-02", "Shopping", "Croma", 96_500, "Credit Card"),
        ("2025-05-29", "Healthcare", "City Clinic", 74_000, "UPI"),
        ("2026-02-01", "Other", "Home Repair Studio", 68_500, "Net Banking"),
    ]
    for date_text, category, merchant, amount, method in unusual:
        add(rows, pd.Timestamp(date_text), "expense", category, merchant, amount, method)

    df = pd.DataFrame(rows).sort_values(["date", "transaction_id"])
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Wrote {len(df)} rows to {DATA_PATH}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")


if __name__ == "__main__":
    main()
