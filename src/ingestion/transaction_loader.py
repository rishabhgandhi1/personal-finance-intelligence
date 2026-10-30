from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from io import BytesIO
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database.models import ImportBatch, Transaction

REQUIRED_COLUMNS = {
    "transaction_id",
    "transaction_date",
    "description",
    "amount",
    "transaction_type",
}

OPTIONAL_COLUMNS = {
    "category",
    "payment_method",
    "account_type",
}

SUPPORTED_COLUMNS = REQUIRED_COLUMNS | OPTIONAL_COLUMNS

EXPENSE_CATEGORIES = {
    "Food",
    "Groceries",
    "Transport",
    "Shopping",
    "Utilities",
    "Healthcare",
    "Entertainment",
    "Education",
    "Housing",
    "Travel",
    "Personal Care",
    "Subscriptions",
    "Other",
}

SUPPORTED_CATEGORIES = EXPENSE_CATEGORIES | {"Income"}

CATEGORY_LOOKUP = {
    category.casefold(): category
    for category in SUPPORTED_CATEGORIES
}

MAX_AMOUNT = Decimal("9999999999.99")

DEFAULT_ACCOUNT_TYPE = "BANK_ACCOUNT"
DEFAULT_PAYMENT_METHOD = "UNKNOWN"


@dataclass
class IngestionResult:
    batch_id: int
    total_rows: int
    accepted_rows: int
    rejected_rows: int
    duplicate_rows: int
    errors: list[dict]


class CSVFormatError(ValueError):
    """Raised when a CSV file does not meet the expected format."""


def read_csv_source(source) -> pd.DataFrame:
    """Read a CSV file path, byte stream, or file-like object."""
    if isinstance(source, pd.DataFrame):
        return source.copy()

    if isinstance(source, (bytes, bytearray)):
        return pd.read_csv(BytesIO(source), dtype=object)
    if hasattr(source, "read"):
        return pd.read_csv(source, dtype=object)
    return pd.read_csv(
        Path(source),
        dtype=object,
    )


def validate_columns(df: pd.DataFrame) -> None:
    """Normalize column names and verify the required schema."""
    normalized = [
        str(column).strip().lower()
        for column in df.columns
    ]

    if len(normalized) != len(set(normalized)):
        raise CSVFormatError(
            "The CSV contains duplicate column names after normalization."
        )
    df.columns = normalized
    missing = REQUIRED_COLUMNS - set(df.columns)
    unexpected = set(df.columns) - SUPPORTED_COLUMNS
    if missing:
        raise CSVFormatError(
            f"Missing required columns: {sorted(missing)}"
        )
    if unexpected:
        raise CSVFormatError(
            f"Unsupported columns: {sorted(unexpected)}"
        )


def normalize_text(value) -> str | None:
    """Trim text and convert blank values to None."""
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None
    return text


def parse_amount(value) -> Decimal:
    """Parse a positive monetary amount without binary float arithmetic."""
    if value is None or pd.isna(value):
        raise ValueError("Amount is missing.")

    text = str(value).strip()
    if not text:
        raise ValueError("Amount is missing.")
    try:
        amount = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError("Amount is not a valid decimal number.") from exc
    if not amount.is_finite():
        raise ValueError("Amount must be finite.")
    if amount <= 0:
        raise ValueError("Amount must be greater than zero.")
    if amount > MAX_AMOUNT:
        raise ValueError("Amount exceeds the supported maximum.")

    exponent = amount.as_tuple().exponent
    if isinstance(exponent, int) and exponent < -2:
        raise ValueError("Amount must have at most two decimal places.")

    return amount.quantize(Decimal("0.01"))


def parse_transaction_date(value):
    """Parse a strict YYYY-MM-DD calendar date."""
    if value is None or pd.isna(value):
        raise ValueError("Transaction date is missing.")

    text = str(value).strip()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        raise ValueError(
            "Date must use YYYY-MM-DD format."
        )
    try:
        return pd.Timestamp(
            pd.to_datetime(
                text,
                format="%Y-%m-%d",
                errors="raise",
            )
        ).date()
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "Transaction date is not a valid calendar date."
        ) from exc


def normalize_transaction_type(value) -> str:
    """Normalize and validate the transaction type."""
    text = normalize_text(value)

    if text is None:
        raise ValueError("Transaction type is missing.")
    transaction_type = text.upper()
    if transaction_type not in {"INCOME", "EXPENSE"}:
        raise ValueError(
            "Transaction type must be INCOME or EXPENSE."
        )
    return transaction_type


def normalize_category(value, transaction_type: str) -> str | None:
    """Normalize category values and enforce type-category consistency."""
    category = normalize_text(value)

    if category is None:
        return None
    normalized_category = CATEGORY_LOOKUP.get(
        category.casefold()
    )
    if normalized_category is None:
        raise ValueError(
            f"Unsupported category: {category}"
        )
    if transaction_type == "INCOME":
        if normalized_category != "Income":
            raise ValueError(
                "Income transactions must use the Income category."
            )
    elif normalized_category == "Income":
        raise ValueError(
            "Expense transactions cannot use the Income category."
        )
    return normalized_category


def prepare_transaction(row: dict) -> dict:
    """Validate and normalize one transaction row."""
    transaction_id = normalize_text(
        row.get("transaction_id")
    )

    if transaction_id is None:
        raise ValueError("Transaction ID is missing.")
    description = normalize_text(
        row.get("description")
    )
    if description is None:
        raise ValueError("Description is missing.")
    transaction_type = normalize_transaction_type(
        row.get("transaction_type")
    )
    return {
        "transaction_id": transaction_id,
        "transaction_date": parse_transaction_date(
            row.get("transaction_date")
        ),
        "description": description,
        "amount": parse_amount(row.get("amount")),
        "transaction_type": transaction_type,
        "category": normalize_category(
            row.get("category"),
            transaction_type,
        ),
        "category_source": "original",
        "payment_method": (
            normalize_text(row.get("payment_method"))
            or DEFAULT_PAYMENT_METHOD
        ),
        "account_type": (
            normalize_text(row.get("account_type"))
            or DEFAULT_ACCOUNT_TYPE
        ),
    }


def ingest_transactions(
    source,
    session: Session,
    filename: str = "uploaded_transactions.csv",
) -> IngestionResult:
    """
    Validate a CSV and save valid transactions.

    Existing transaction IDs are never inserted again.
    Invalid and duplicate rows are reported separately.
    """
    df = read_csv_source(source)
    validate_columns(df)
    total_rows = len(df)
    errors = []
    accepted = []
    seen_ids = set()
    duplicate_rows = 0
    for row_number, row in enumerate(
        df.to_dict(orient="records"),
        start=2,
    ):
        try:
            transaction = prepare_transaction(row)
            transaction_id = transaction["transaction_id"]
            if transaction_id in seen_ids:
                duplicate_rows += 1
                errors.append(
                    {
                        "row": row_number,
                        "transaction_id": transaction_id,
                        "reason": "Duplicate transaction ID within the file.",
                        "status": "duplicate",
                    }
                )
                continue
            seen_ids.add(transaction_id)
            accepted.append(transaction)
        except (ValueError, TypeError, ArithmeticError) as exc:
            errors.append(
                {
                    "row": row_number,
                    "transaction_id": normalize_text(
                        row.get("transaction_id")
                    ),
                    "reason": str(exc),
                    "status": "invalid",
                }
            )
    # Query existing IDs in batches to avoid an oversized SQL query.
    existing_ids = set()
    candidate_ids = [
        item["transaction_id"]
        for item in accepted
    ]
    for start in range(0, len(candidate_ids), 500):
        id_chunk = candidate_ids[start:start + 500]
        existing_ids.update(
            session.scalars(
                select(Transaction.transaction_id).where(
                    Transaction.transaction_id.in_(id_chunk)
                )
            ).all()
        )
    new_transactions = []
    for item in accepted:
        transaction_id = item["transaction_id"]
        if transaction_id in existing_ids:
            duplicate_rows += 1
            errors.append(
                {
                    "row": None,
                    "transaction_id": transaction_id,
                    "reason": "Transaction ID already exists in the database.",
                    "status": "duplicate",
                }
            )
            continue
        new_transactions.append(item)
    batch = ImportBatch(
        filename=Path(filename).name[:255],
        total_rows=total_rows,
        accepted_rows=len(new_transactions),
        rejected_rows=sum(
            error["status"] == "invalid"
            for error in errors
        ),
    )
    try:
        session.add(batch)
        session.flush()
        for item in new_transactions:
            session.add(
                Transaction(
                    **item,
                    import_batch_id=batch.id,
                )
            )
        session.commit()
    except Exception:
        session.rollback()
        raise
    return IngestionResult(
        batch_id=batch.id,
        total_rows=total_rows,
        accepted_rows=len(new_transactions),
        rejected_rows=batch.rejected_rows,
        duplicate_rows=duplicate_rows,
        errors=errors,
    )