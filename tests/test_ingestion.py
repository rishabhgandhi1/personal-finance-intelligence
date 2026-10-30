from datetime import date
from decimal import Decimal

import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from src.database.connection import Base
from src.database.models import ImportBatch, Transaction
from src.ingestion.transaction_loader import (
    CSVFormatError,
    ingest_transactions,
)


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)

    TestSession = sessionmaker(
        bind=engine,
        expire_on_commit=False,
    )
    with TestSession() as session:
        yield session
    engine.dispose()


def make_row(**overrides):
    """Return a valid transaction row with optional overrides."""
    row = {
        "transaction_id": "TXN000001",
        "transaction_date": "2026-10-01",
        "description": "Grocery Store",
        "amount": "450.00",
        "transaction_type": "EXPENSE",
        "category": "Groceries",
        "payment_method": "UPI",
        "account_type": "BANK_ACCOUNT",
    }

    row.update(overrides)
    return row


def test_valid_transaction_is_saved(db_session):
    df = pd.DataFrame([make_row()])

    result = ingest_transactions(
        df,
        db_session,
        filename="valid.csv",
    )
    transaction = db_session.scalar(
        select(Transaction).where(
            Transaction.transaction_id == "TXN000001"
        )
    )
    assert result.total_rows == 1
    assert result.accepted_rows == 1
    assert result.rejected_rows == 0
    assert result.duplicate_rows == 0
    assert transaction is not None
    assert transaction.amount == Decimal("450.00")
    assert transaction.transaction_date == date(2026, 10, 1)


def test_invalid_amount_is_rejected(db_session):
    df = pd.DataFrame([
        make_row(amount="abc")
    ])

    result = ingest_transactions(
        df,
        db_session,
        filename="invalid_amount.csv",
    )
    assert result.accepted_rows == 0
    assert result.rejected_rows == 1
    assert result.errors[0]["status"] == "invalid"


def test_invalid_date_is_rejected(db_session):
    df = pd.DataFrame([
        make_row(transaction_date="2026-02-30")
    ])

    result = ingest_transactions(
        df,
        db_session,
        filename="invalid_date.csv",
    )
    assert result.accepted_rows == 0
    assert result.rejected_rows == 1


def test_duplicate_ids_within_file_are_detected(db_session):
    df = pd.DataFrame([
        make_row(),
        make_row(),
    ])

    result = ingest_transactions(
        df,
        db_session,
        filename="duplicates.csv",
    )
    assert result.total_rows == 2
    assert result.accepted_rows == 1
    assert result.duplicate_rows == 1
    assert result.rejected_rows == 0


def test_existing_transaction_is_not_inserted_twice(db_session):
    df = pd.DataFrame([
        make_row()
    ])

    first_result = ingest_transactions(
        df,
        db_session,
        filename="first.csv",
    )
    second_result = ingest_transactions(
        df,
        db_session,
        filename="second.csv",
    )
    assert first_result.accepted_rows == 1
    assert second_result.accepted_rows == 0
    assert second_result.duplicate_rows == 1
    transactions = db_session.scalars(
        select(Transaction)
    ).all()
    assert len(transactions) == 1


def test_missing_required_column_rejects_file(db_session):
    df = pd.DataFrame([
        {
            "transaction_date": "2026-10-01",
            "description": "Grocery Store",
            "amount": "450.00",
            "transaction_type": "EXPENSE",
        }
    ])

    with pytest.raises(CSVFormatError):
        ingest_transactions(
            df,
            db_session,
            filename="missing_column.csv",
        )
    batches = db_session.scalars(
        select(ImportBatch)
    ).all()
    assert len(batches) == 0


def test_unknown_category_is_rejected(db_session):
    df = pd.DataFrame([
        make_row(category="SomethingUnknown")
    ])

    result = ingest_transactions(
        df,
        db_session,
        filename="unknown_category.csv",
    )
    assert result.accepted_rows == 0
    assert result.rejected_rows == 1


def test_blank_category_is_allowed(db_session):
    df = pd.DataFrame([
        make_row(category=None)
    ])

    result = ingest_transactions(
        df,
        db_session,
        filename="uncategorized.csv",
    )
    assert result.accepted_rows == 1
    transaction = db_session.scalar(
        select(Transaction)
    )
    assert transaction.category is None
