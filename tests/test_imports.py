from __future__ import annotations

from pathlib import Path

import pandas as pd

from finsense.imports import (
    MAX_UPLOAD_BYTES,
    detect_mapping,
    infer_category,
    infer_merchant,
    normalize_uploaded_transactions,
    read_uploaded_csv_bytes,
)


def normalize(raw: pd.DataFrame) -> tuple[pd.DataFrame, object]:
    detected, mapping = detect_mapping(raw)
    preview = normalize_uploaded_transactions(raw, mapping, detected)
    assert preview.validation_errors == []
    return preview.cleaned, preview.report


def test_existing_finsense_format_imports() -> None:
    raw = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "transaction_type": "expense",
                "category": "Groceries",
                "merchant": "Local Grocery",
                "amount": 450,
                "payment_method": "UPI",
            }
        ]
    )
    cleaned, report = normalize(raw)
    assert report.output_rows == 1
    assert cleaned["category"].iloc[0] == "Groceries"


def test_debit_credit_format_imports() -> None:
    raw = pd.DataFrame(
        {
            "Value Date": ["2026-01-01", "2026-01-02"],
            "Narration": ["UPI Swiggy dinner", "Salary credit"],
            "Withdrawal": ["840", ""],
            "Deposit": ["", "150000"],
        }
    )
    cleaned, _ = normalize(raw)
    assert list(cleaned["transaction_type"].astype(str)) == ["expense", "income"]
    assert list(cleaned["amount"]) == [840, 150000]
    assert cleaned["category"].iloc[0] == "Dining"


def test_signed_amount_format_imports() -> None:
    raw = pd.DataFrame(
        {
            "transaction_date": ["2026-01-01", "2026-01-02"],
            "details": ["Amazon shopping", "Payroll salary"],
            "amount": [-2500, 120000],
        }
    )
    cleaned, _ = normalize(raw)
    assert list(cleaned["transaction_type"].astype(str)) == ["expense", "income"]
    assert list(cleaned["amount"]) == [2500, 120000]


def test_alternative_column_names_and_inference() -> None:
    raw = pd.DataFrame(
        {
            "value_date": ["2026-01-01"],
            "remarks": ["NEFT Airtel Fiber bill"],
            "debit_amount": [1800],
            "credit_amount": [None],
        }
    )
    cleaned, _ = normalize(raw)
    assert cleaned["merchant"].iloc[0] == "Neft Airtel Fiber Bill"
    assert cleaned["category"].iloc[0] == "Utilities"


def test_category_and_merchant_inference() -> None:
    assert infer_category("PVR Cinemas weekend", "expense") == "Entertainment"
    assert infer_category("mystery transaction", "expense") == "Other"
    assert infer_merchant("UPI 123456 BigBasket Grocery") == "Upi Bigbasket Grocery"


def test_mapping_validation_errors_for_incomplete_mapping() -> None:
    raw = pd.DataFrame({"date": ["2026-01-01"], "amount": [100]})
    detected, mapping = detect_mapping(raw)
    preview = normalize_uploaded_transactions(raw, mapping, detected)
    assert preview.validation_errors
    assert preview.report.output_rows == 0


def test_malformed_empty_and_oversized_csv_handling() -> None:
    empty = read_uploaded_csv_bytes(b"")
    assert empty.error == "CSV file is empty."
    header_only = read_uploaded_csv_bytes(b"date,description,amount\n")
    assert header_only.error == "CSV file has no transaction rows."
    oversized = read_uploaded_csv_bytes(b"x" * (MAX_UPLOAD_BYTES + 1))
    assert "larger than 10 MB" in str(oversized.error)
    malformed = read_uploaded_csv_bytes(b'bad,csv\n"unterminated')
    assert malformed.error is not None


def test_upload_processing_does_not_persist_content(tmp_path: Path) -> None:
    content = b"date,description,amount\n2026-01-01,Swiggy dinner,-500\n2026-01-02,Salary,1000\n"
    before = set(tmp_path.iterdir())
    parsed = read_uploaded_csv_bytes(content)
    detected, mapping = detect_mapping(parsed.dataframe)
    preview = normalize_uploaded_transactions(parsed.dataframe, mapping, detected)
    after = set(tmp_path.iterdir())
    assert before == after
    assert preview.cleaned["amount"].sum() == 1500
