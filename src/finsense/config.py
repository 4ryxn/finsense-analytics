from __future__ import annotations

from pathlib import Path

RANDOM_STATE = 42
CURRENCY_SYMBOL = "₹"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DATA_PATH = DATA_DIR / "sample_transactions.csv"
TEMPLATE_PATH = DATA_DIR / "transaction_template.csv"

REQUIRED_COLUMNS = {
    "date",
    "transaction_type",
    "category",
    "merchant",
    "amount",
    "payment_method",
}

CANONICAL_CATEGORIES = {
    "salary": "Salary",
    "freelance": "Freelance",
    "investment": "Investment",
    "housing": "Housing",
    "food": "Food",
    "transport": "Transport",
    "utilities": "Utilities",
    "groceries": "Groceries",
    "shopping": "Shopping",
    "dining": "Dining",
    "subscriptions": "Subscriptions",
    "healthcare": "Healthcare",
    "entertainment": "Entertainment",
    "education": "Education",
    "travel": "Travel",
    "other": "Other",
}

PAYMENT_METHODS = {
    "upi": "UPI",
    "credit card": "Credit Card",
    "debit card": "Debit Card",
    "net banking": "Net Banking",
    "bank transfer": "Bank Transfer",
    "cash": "Cash",
    "wallet": "Wallet",
}
