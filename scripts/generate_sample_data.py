from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "sample_transactions.csv"
RANDOM_SEED = 42


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
    amount = max(float(amount), 80.0)
    rows.append(
        {
            "transaction_id": txn_id(date, kind, category, merchant, amount),
            "date": date.strftime("%Y-%m-%d"),
            "transaction_type": kind,
            "category": category,
            "merchant": merchant,
            "amount": round(amount, 2),
            "payment_method": method,
        }
    )


def month_date(
    rng: np.random.Generator,
    month: pd.Timestamp,
    low: int,
    high: int,
    prefer_weekend: bool = False,
) -> pd.Timestamp:
    month = pd.Timestamp(month)
    for _ in range(10):
        day = int(rng.integers(low, high + 1))
        date = month + pd.DateOffset(days=int(min(day, month.days_in_month) - 1))
        if not prefer_weekend or date.weekday() >= 5:
            return date
    return date


def seasonal_multiplier(month: pd.Timestamp, category: str) -> float:
    festive = 1.18 if month.month in [10, 11, 12] else 1.0
    summer = 1.10 if month.month in [4, 5] else 1.0
    monsoon = 1.08 if month.month in [7, 8] else 1.0
    if category in {"Shopping", "Dining", "Entertainment"}:
        return festive
    if category in {"Utilities", "Travel"}:
        return summer
    if category == "Transport":
        return monsoon
    return 1.0


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict[str, object]] = []
    months = pd.date_range("2022-07-01", "2026-06-01", freq="MS")
    methods = ["UPI", "Credit Card", "Debit Card", "Wallet", "Net Banking", "Cash"]
    method_probabilities = [0.43, 0.28, 0.09, 0.10, 0.07, 0.03]
    merchants = {
        "Groceries": ["BigBasket", "Blinkit", "Star Bazaar", "Nature Basket", "Local Grocery"],
        "Transport": ["Uber", "Ola", "Metro Card", "Indian Oil", "Rapido"],
        "Utilities": ["Tata Power", "Airtel Fiber", "Jio Mobile", "Mahanagar Gas", "BESCOM"],
        "Subscriptions": ["Netflix", "Spotify", "Amazon Prime", "Google One"],
        "Healthcare": ["Apollo Pharmacy", "Practo", "City Clinic", "HealthKart"],
        "Shopping": ["Amazon India", "Myntra", "Croma", "Decathlon", "Nykaa"],
        "Dining": ["Swiggy", "Zomato", "Blue Tokai", "Third Wave Coffee", "Barbeque Nation"],
        "Entertainment": ["BookMyShow", "PVR Cinemas", "Steam", "NCPA"],
        "Education": ["Coursera", "Udemy", "Kindle Store", "Skillshare"],
        "Travel": ["MakeMyTrip", "IRCTC", "Air India Express", "Oyo Rooms"],
        "Other": ["Society Office", "Stationery Hub", "Gift Store", "Laundry Point"],
    }
    plan = {
        "Groceries": (8, 1_450, 460),
        "Transport": (6, 820, 360),
        "Utilities": (3, 2_650, 620),
        "Shopping": (3, 3_400, 1_550),
        "Dining": (5, 1_250, 620),
        "Healthcare": (1, 1_900, 850),
        "Entertainment": (3, 1_250, 520),
        "Education": (1, 2_300, 900),
        "Travel": (1, 5_500, 2_400),
        "Other": (2, 1_100, 520),
    }

    for index, month in enumerate(months):
        month = pd.Timestamp(month)
        inflation = 1 + index * 0.006
        salary = 135_000 * (1 + index * 0.0048) + rng.normal(0, 1_700)
        add(
            rows,
            month + pd.DateOffset(days=27),
            "income",
            "Salary",
            "Northstar Payroll",
            salary,
            "Bank Transfer",
        )
        if month.month in [3, 9]:
            add(
                rows,
                month_date(rng, month, 20, 27),
                "income",
                "Bonus",
                "Northstar Payroll",
                salary * rng.uniform(0.18, 0.30),
                "Bank Transfer",
            )
        if rng.random() < 0.28:
            add(
                rows,
                month_date(rng, month, 6, 21),
                "income",
                "Freelance",
                "Independent Client",
                rng.normal(24_000, 5_000),
                "Bank Transfer",
            )

        rent = 39_500 * (1 + int(index / 12) * 0.07)
        add(
            rows,
            month + pd.DateOffset(days=4),
            "expense",
            "Housing",
            "Metro Homes Rent",
            rent,
            "Net Banking",
        )

        for merchant, amount in [
            ("Netflix", 649),
            ("Spotify", 119),
            ("Google One", 210),
            ("Amazon Prime", 299),
        ]:
            add(
                rows,
                month_date(rng, month, 2, 8),
                "expense",
                "Subscriptions",
                merchant,
                amount * inflation,
                "Credit Card",
            )

        for category, (base_count, center, spread) in plan.items():
            count = max(1, base_count + int(rng.integers(-1, 2)))
            for _ in range(count):
                prefer_weekend = category in {"Dining", "Entertainment", "Travel"}
                merchant = str(rng.choice(merchants[category]))
                method = str(rng.choice(methods, p=method_probabilities))
                base = center * inflation * seasonal_multiplier(month, category)
                amount = rng.normal(base, spread)
                add(
                    rows,
                    month_date(rng, month, 1, 27, prefer_weekend=prefer_weekend),
                    "expense",
                    category,
                    merchant,
                    amount,
                    method,
                )

    unusual = [
        ("2023-11-04", "Shopping", "Croma", 92_000, "Credit Card"),
        ("2024-05-30", "Travel", "MakeMyTrip", 118_000, "Credit Card"),
        ("2025-08-02", "Healthcare", "City Clinic", 76_000, "UPI"),
        ("2026-02-28", "Other", "Home Repair Studio", 84_000, "Net Banking"),
    ]
    for date_text, category, merchant, amount, method in unusual:
        add(rows, pd.Timestamp(date_text), "expense", category, merchant, amount, method)

    df = pd.DataFrame(rows).sort_values(["date", "transaction_id"]).reset_index(drop=True)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Wrote {len(df)} rows to {DATA_PATH}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")


if __name__ == "__main__":
    main()
