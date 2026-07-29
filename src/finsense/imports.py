from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO

import pandas as pd

from finsense.etl import CleaningReport, clean_transactions

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
NONE_OPTION = "-- Not mapped --"

COLUMN_SYNONYMS = {
    "date": ["date", "transaction_date", "txn_date", "value_date", "posted_date", "posting_date"],
    "description": ["description", "narration", "details", "remarks", "particulars", "memo"],
    "debit": ["debit", "withdrawal", "withdrawals", "debit_amount", "paid_out"],
    "credit": ["credit", "deposit", "deposits", "credit_amount", "paid_in"],
    "amount": ["amount", "transaction_amount", "amt", "value"],
    "transaction_type": ["transaction_type", "type", "dr_cr", "debit_credit"],
    "category": ["category", "expense_category"],
    "merchant": ["merchant", "payee", "counterparty", "beneficiary"],
    "payment_method": ["payment_method", "mode", "channel", "method"],
    "currency": ["currency", "ccy"],
    "transaction_id": ["transaction_id", "txn_id", "reference", "reference_no", "ref_no", "utr"],
}

CATEGORY_KEYWORDS = {
    "Income": ["salary", "payroll", "bonus", "interest", "dividend", "deposit", "credit"],
    "Housing": ["rent", "maintenance", "society", "mortgage"],
    "Groceries": ["grocery", "bigbasket", "blinkit", "supermarket", "mart", "star bazaar"],
    "Dining": ["swiggy", "zomato", "restaurant", "cafe", "coffee", "dining"],
    "Transportation": ["uber", "ola", "metro", "fuel", "petrol", "rapido", "transport"],
    "Utilities": ["electricity", "power", "airtel", "jio", "gas", "water", "utility"],
    "Subscriptions": ["netflix", "spotify", "prime", "subscription", "google one"],
    "Shopping": ["amazon", "myntra", "croma", "nykaa", "shopping", "store"],
    "Healthcare": ["pharmacy", "clinic", "hospital", "doctor", "health", "apollo"],
    "Education": ["coursera", "udemy", "school", "college", "tuition", "kindle"],
    "Travel": ["makemytrip", "flight", "hotel", "irctc", "travel", "air india"],
    "Entertainment": ["bookmyshow", "pvr", "cinema", "movie", "steam"],
    "Transfers": ["transfer", "neft", "imps", "rtgs", "upi", "wallet"],
}


@dataclass(frozen=True)
class UploadedCsv:
    dataframe: pd.DataFrame
    encoding: str
    size_bytes: int
    error: str | None = None


@dataclass(frozen=True)
class MappingValidation:
    is_valid: bool
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ImportPreview:
    detected_format: str
    mapping: dict[str, str | None]
    raw_rows: int
    cleaned: pd.DataFrame
    report: CleaningReport
    missing_critical_values: int
    validation_errors: list[str]

    @property
    def is_ready(self) -> bool:
        return not self.validation_errors and not self.cleaned.empty


def canonical_column_name(name: object) -> str:
    return str(name).strip().lower().replace("/", "_").replace("-", "_").replace(" ", "_")


def read_uploaded_csv_bytes(content: bytes, filename: str = "uploaded.csv") -> UploadedCsv:
    if len(content) > MAX_UPLOAD_BYTES:
        return UploadedCsv(pd.DataFrame(), "", len(content), "CSV file is larger than 10 MB.")
    if not content.strip():
        return UploadedCsv(pd.DataFrame(), "", len(content), "CSV file is empty.")

    for encoding in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
        try:
            df = pd.read_csv(BytesIO(content), encoding=encoding)
            if df.empty and not list(df.columns):
                return UploadedCsv(df, encoding, len(content), "CSV file has no columns.")
            if df.empty:
                return UploadedCsv(df, encoding, len(content), "CSV file has no transaction rows.")
            return UploadedCsv(df, encoding, len(content))
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
            continue
    return UploadedCsv(
        pd.DataFrame(), "", len(content), f"{filename} could not be parsed as a supported CSV."
    )


def detect_mapping(df: pd.DataFrame) -> tuple[str, dict[str, str | None]]:
    normalized = {canonical_column_name(column): str(column) for column in df.columns}
    mapping: dict[str, str | None] = {mapping_field: None for mapping_field in COLUMN_SYNONYMS}
    for mapping_field, synonyms in COLUMN_SYNONYMS.items():
        for synonym in synonyms:
            if synonym in normalized:
                mapping[mapping_field] = normalized[synonym]
                break

    columns = set(canonical_column_name(column) for column in df.columns)
    if {"date", "transaction_type", "category", "merchant", "amount", "payment_method"}.issubset(
        columns
    ):
        detected = "FinSense template"
    elif mapping["debit"] or mapping["credit"]:
        detected = "Debit/Credit bank statement"
    elif mapping["amount"] and mapping["transaction_type"]:
        detected = "Amount with transaction type"
    elif mapping["amount"]:
        detected = "Signed amount statement"
    else:
        detected = "Needs column mapping"
    return detected, mapping


def validate_mapping(mapping: dict[str, str | None]) -> MappingValidation:
    errors: list[str] = []
    if not mapping.get("date"):
        errors.append("Map a date column.")
    if not mapping.get("description") and not mapping.get("merchant"):
        errors.append("Map a transaction description or merchant column.")
    has_debit_credit = bool(mapping.get("debit") or mapping.get("credit"))
    has_amount = bool(mapping.get("amount"))
    if not has_debit_credit and not has_amount:
        errors.append("Map either an amount column or debit/credit columns.")
    assigned: dict[str, str] = {}
    compatible_duplicates = {frozenset({"description", "merchant"})}
    for mapping_field, column in mapping.items():
        if not column:
            continue
        if column in assigned:
            pair = frozenset({mapping_field, assigned[column]})
            if pair not in compatible_duplicates:
                errors.append(f"Column '{column}' is assigned to multiple incompatible fields.")
        assigned[column] = mapping_field
    return MappingValidation(not errors, errors)


def parse_money(series: pd.Series) -> pd.Series:
    text = (
        series.fillna("")
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace("INR", "", case=False, regex=False)
        .str.strip()
    )
    negative = text.str.match(r"^\(.*\)$")
    text = text.str.replace(r"[()]", "", regex=True)
    values = pd.to_numeric(text, errors="coerce")
    values = values.where(~negative, -values.abs())
    return values


def infer_merchant(description: object) -> str:
    text = str(description or "").strip()
    text = " ".join(text.replace("/", " ").replace("-", " ").split())
    tokens = [
        token
        for token in text.split()
        if not token.isdigit() and not token.lower().startswith(("acct", "account", "a/c", "xx"))
    ]
    merchant = " ".join(tokens[:5]).strip()
    return merchant.title() if merchant else "Unknown"


def infer_category(description: object, transaction_type: str) -> str:
    if transaction_type == "income":
        return "Income"
    text = str(description or "").lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if category == "Income":
            continue
        if any(keyword in text for keyword in keywords):
            return category
    return "Other"


def _type_from_amount(value: float) -> str:
    return "income" if value > 0 else "expense"


def normalize_uploaded_transactions(
    raw: pd.DataFrame, mapping: dict[str, str | None], detected_format: str
) -> ImportPreview:
    validation = validate_mapping(mapping)
    if not validation.is_valid:
        empty_report = CleaningReport(
            input_rows=len(raw),
            output_rows=0,
            duplicates_removed=0,
            invalid_rows_rejected=len(raw),
            rejected_row_reasons={"invalid_mapping": len(raw)},
        )
        return ImportPreview(
            detected_format,
            mapping,
            len(raw),
            pd.DataFrame(),
            empty_report,
            len(raw),
            validation.errors,
        )

    normalized = pd.DataFrame()
    normalized["date"] = raw[mapping["date"]]
    description_source = mapping.get("description") or mapping.get("merchant")
    descriptions = (
        raw[description_source].fillna("") if description_source else pd.Series([""] * len(raw))
    )

    if mapping.get("debit") or mapping.get("credit"):
        debit = (
            parse_money(raw[mapping["debit"]])
            if mapping.get("debit")
            else pd.Series([0] * len(raw))
        )
        credit = (
            parse_money(raw[mapping["credit"]])
            if mapping.get("credit")
            else pd.Series([0] * len(raw))
        )
        normalized["amount"] = debit.abs().where(debit.notna() & debit.ne(0), credit.abs())
        normalized["transaction_type"] = (
            credit.where(credit.notna(), 0).gt(0).map({True: "income", False: "expense"})
        )
    else:
        amount = parse_money(raw[mapping["amount"]])
        if not mapping.get("transaction_type") and not (amount.lt(0).any() and amount.gt(0).any()):
            empty_report = CleaningReport(
                input_rows=len(raw),
                output_rows=0,
                duplicates_removed=0,
                invalid_rows_rejected=len(raw),
                rejected_row_reasons={"unsupported_amount_format": len(raw)},
            )
            return ImportPreview(
                detected_format,
                mapping,
                len(raw),
                pd.DataFrame(),
                empty_report,
                int(amount.isna().sum()),
                [
                    "Amount-only files must use signed values with both income and expense signs, or map a transaction type column."
                ],
            )
        normalized["amount"] = amount.abs()
        if mapping.get("transaction_type"):
            tx_type = raw[mapping["transaction_type"]].astype(str).str.lower()
            normalized["transaction_type"] = tx_type.map(
                lambda value: (
                    "income"
                    if any(token in value for token in ["income", "credit", "deposit", "cr"])
                    else "expense"
                )
            )
        else:
            normalized["transaction_type"] = amount.map(_type_from_amount)

    normalized["merchant"] = (
        raw[mapping["merchant"]].fillna("").map(infer_merchant)
        if mapping.get("merchant")
        else descriptions.map(infer_merchant)
    )
    normalized["category"] = (
        raw[mapping["category"]].fillna("")
        if mapping.get("category")
        else [
            infer_category(description, tx_type)
            for description, tx_type in zip(
                descriptions, normalized["transaction_type"], strict=False
            )
        ]
    )
    normalized["payment_method"] = (
        raw[mapping["payment_method"]].fillna("Unknown")
        if mapping.get("payment_method")
        else "Unknown"
    )
    if mapping.get("transaction_id"):
        normalized["transaction_id"] = raw[mapping["transaction_id"]]

    missing_critical = int(
        pd.to_datetime(normalized["date"], errors="coerce").isna().sum()
        + pd.to_numeric(normalized["amount"], errors="coerce").isna().sum()
        + descriptions.astype(str).str.strip().eq("").sum()
    )
    cleaned, report = clean_transactions(normalized)
    return ImportPreview(
        detected_format=detected_format,
        mapping=mapping,
        raw_rows=len(raw),
        cleaned=cleaned,
        report=report,
        missing_critical_values=missing_critical,
        validation_errors=[],
    )


def mapped_options(mapping: dict[str, str | None]) -> dict[str, str | None]:
    return {key: value for key, value in mapping.items() if value}
