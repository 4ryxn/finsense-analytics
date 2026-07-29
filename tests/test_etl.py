from __future__ import annotations

import pandas as pd

from finsense.etl import clean_transactions


def base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "transaction_type": " Expense ",
                "category": " food ",
                "merchant": "  local grocery ",
                "amount": "250.50",
                "payment_method": "upi",
                "transaction_id": "a",
            },
            {
                "date": "2026-01-01",
                "transaction_type": " Expense ",
                "category": " food ",
                "merchant": "  local grocery ",
                "amount": "250.50",
                "payment_method": "upi",
                "transaction_id": "a",
            },
            {
                "date": "bad-date",
                "transaction_type": "expense",
                "category": "Food",
                "merchant": "Cafe",
                "amount": "100",
                "payment_method": "UPI",
                "transaction_id": "bad-date",
            },
            {
                "date": "2026-01-02",
                "transaction_type": "income",
                "category": "Salary",
                "merchant": None,
                "amount": "-10",
                "payment_method": "bank transfer",
                "transaction_id": "bad-amount",
            },
        ]
    )


def test_required_column_validation() -> None:
    cleaned, report = clean_transactions(pd.DataFrame({"date": ["2026-01-01"]}))
    assert cleaned.empty
    assert "amount" in report.missing_required_columns
    assert report.invalid_rows_rejected == 1


def test_invalid_dates_amounts_duplicates_and_report_counts() -> None:
    cleaned, report = clean_transactions(base_frame())
    assert len(cleaned) == 1
    assert report.input_rows == 4
    assert report.output_rows == 1
    assert report.duplicates_removed == 1
    assert report.invalid_rows_rejected == 2
    assert report.rejected_row_reasons["invalid_date"] == 1
    assert report.rejected_row_reasons["invalid_amount"] == 1


def test_category_merchant_and_payment_normalization() -> None:
    cleaned, report = clean_transactions(base_frame())
    row = cleaned.iloc[0]
    assert row["category"] == "Food"
    assert row["merchant"] == "Local Grocery"
    assert row["payment_method"] == "UPI"
    assert report.standardization_counts


def test_generated_ids_are_deterministic_and_input_unchanged() -> None:
    raw = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "transaction_type": "expense",
                "category": "Food",
                "merchant": "Cafe",
                "amount": 500,
                "payment_method": "UPI",
            }
        ]
    )
    original_columns = list(raw.columns)
    one, _ = clean_transactions(raw)
    two, _ = clean_transactions(raw)
    assert list(raw.columns) == original_columns
    assert one["transaction_id"].iloc[0] == two["transaction_id"].iloc[0]
