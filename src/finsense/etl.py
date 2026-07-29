from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from finsense.config import CANONICAL_CATEGORIES, PAYMENT_METHODS, REQUIRED_COLUMNS


@dataclass(frozen=True)
class CleaningReport:
    input_rows: int
    output_rows: int
    duplicates_removed: int
    invalid_rows_rejected: int
    missing_values_handled: dict[str, int] = field(default_factory=dict)
    standardization_counts: dict[str, int] = field(default_factory=dict)
    rejected_row_reasons: dict[str, int] = field(default_factory=dict)
    missing_required_columns: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.missing_required_columns and self.output_rows > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_rows": self.input_rows,
            "output_rows": self.output_rows,
            "duplicates_removed": self.duplicates_removed,
            "invalid_rows_rejected": self.invalid_rows_rejected,
            "missing_values_handled": self.missing_values_handled,
            "standardization_counts": self.standardization_counts,
            "rejected_row_reasons": self.rejected_row_reasons,
            "missing_required_columns": self.missing_required_columns,
        }


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy(deep=True)
    clean.columns = (
        clean.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return clean


def validate_required_columns(df: pd.DataFrame) -> list[str]:
    return sorted(REQUIRED_COLUMNS.difference(df.columns))


def _title_words(value: object) -> str:
    text = str(value).strip()
    return " ".join(word.capitalize() for word in text.split()) if text else "Unknown"


def _normalize_category(value: object) -> str:
    key = str(value).strip().lower().replace("_", " ")
    key = " ".join(key.split())
    return CANONICAL_CATEGORIES.get(key, _title_words(key))


def _normalize_payment_method(value: object) -> str:
    key = str(value).strip().lower().replace("_", " ")
    key = " ".join(key.split())
    return PAYMENT_METHODS.get(key, _title_words(key))


def _deterministic_id(row: pd.Series) -> str:
    raw = "|".join(
        [
            row["date"].strftime("%Y-%m-%d"),
            row["transaction_type"],
            row["category"],
            row["merchant"],
            f"{float(row['amount']):.2f}",
            row["payment_method"],
        ]
    )
    return "txn-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    working = normalize_columns(df)
    input_rows = len(working)
    missing_required = validate_required_columns(working)
    if missing_required:
        report = CleaningReport(
            input_rows=input_rows,
            output_rows=0,
            duplicates_removed=0,
            invalid_rows_rejected=input_rows,
            rejected_row_reasons={"missing_required_columns": input_rows},
            missing_required_columns=missing_required,
        )
        return pd.DataFrame(), report

    if "transaction_id" not in working.columns:
        working["transaction_id"] = pd.NA

    original = working.copy(deep=True)
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working["amount"] = pd.to_numeric(working["amount"], errors="coerce")
    working["transaction_type"] = working["transaction_type"].astype(str).str.strip().str.lower()

    missing_merchants = int(
        working["merchant"].isna().sum() + (working["merchant"].astype(str).str.strip() == "").sum()
    )
    working["merchant"] = working["merchant"].fillna("Unknown").map(_title_words)
    working.loc[working["merchant"].eq(""), "merchant"] = "Unknown"

    category_before = original["category"].astype(str)
    method_before = original["payment_method"].astype(str)
    type_before = original["transaction_type"].astype(str)

    working["category"] = working["category"].map(_normalize_category)
    working["payment_method"] = working["payment_method"].map(_normalize_payment_method)

    invalid_date = working["date"].isna()
    invalid_amount = working["amount"].isna() | (working["amount"] <= 0)
    invalid_type = ~working["transaction_type"].isin(["income", "expense"])
    invalid_category = working["category"].eq("Unknown") | working["category"].eq("")
    reject_mask = invalid_date | invalid_amount | invalid_type | invalid_category

    rejected_reasons = {
        "invalid_date": int(invalid_date.sum()),
        "invalid_amount": int(invalid_amount.sum()),
        "invalid_transaction_type": int(invalid_type.sum()),
        "invalid_category": int(invalid_category.sum()),
    }
    rejected_reasons = {key: value for key, value in rejected_reasons.items() if value}

    valid = working.loc[~reject_mask].copy()
    invalid_rows = int(reject_mask.sum())

    before_dupes = len(valid)
    valid = valid.drop_duplicates()
    exact_dupes_removed = before_dupes - len(valid)

    id_dupes_removed = 0
    present_ids = valid["transaction_id"].notna() & valid["transaction_id"].astype(
        str
    ).str.strip().ne("")
    if present_ids.any():
        duplicate_ids = present_ids & valid.duplicated(subset=["transaction_id"], keep="first")
        id_dupes_removed = int(duplicate_ids.sum())
        valid = valid.loc[~duplicate_ids].copy()

    missing_id = valid["transaction_id"].isna() | valid["transaction_id"].astype(
        str
    ).str.strip().eq("")
    if missing_id.any():
        valid.loc[missing_id, "transaction_id"] = valid.loc[missing_id].apply(
            _deterministic_id, axis=1
        )

    valid["transaction_id"] = valid["transaction_id"].astype(str).str.strip()
    valid["transaction_type"] = valid["transaction_type"].astype("category")
    valid["category"] = valid["category"].astype(str)
    valid["merchant"] = valid["merchant"].astype(str)
    valid["payment_method"] = valid["payment_method"].astype(str)
    valid["amount"] = valid["amount"].round(2)

    valid = valid.sort_values(["date", "transaction_id"]).reset_index(drop=True)
    output_cols = [
        "transaction_id",
        "date",
        "transaction_type",
        "category",
        "merchant",
        "amount",
        "payment_method",
    ]
    valid = valid[output_cols]

    standardization_counts = {
        "transaction_type": int((type_before.str.strip().str.lower() != type_before).sum()),
        "category": int(
            (category_before.map(_normalize_category) != category_before.str.strip()).sum()
        ),
        "merchant": int(
            (
                original["merchant"].fillna("Unknown").map(_title_words)
                != original["merchant"].fillna("Unknown").astype(str)
            ).sum()
        ),
        "payment_method": int(
            (method_before.map(_normalize_payment_method) != method_before.str.strip()).sum()
        ),
    }
    standardization_counts = {key: value for key, value in standardization_counts.items() if value}

    report = CleaningReport(
        input_rows=input_rows,
        output_rows=len(valid),
        duplicates_removed=exact_dupes_removed + id_dupes_removed,
        invalid_rows_rejected=invalid_rows,
        missing_values_handled={"merchant": missing_merchants} if missing_merchants else {},
        standardization_counts=standardization_counts,
        rejected_row_reasons=rejected_reasons,
        missing_required_columns=missing_required,
    )
    return valid, report
