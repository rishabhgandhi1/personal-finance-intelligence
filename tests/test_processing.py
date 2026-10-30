from decimal import Decimal

import pandas as pd
import pytest

from src.processing.transaction_processor import (
    engineer_transaction_features,
)


def make_transactions():
    """Create representative transaction data for testing."""

    return pd.DataFrame(
        [
            {
                "transaction_id": "TXN001",
                "transaction_date": "2026-10-30",
                "description": "Grocery Store",
                "amount": Decimal("850.00"),
                "transaction_type": "EXPENSE",
                "category": "Groceries",
            },
            {
                "transaction_id": "TXN002",
                "transaction_date": "2026-10-31",
                "description": "Monthly Salary",
                "amount": Decimal("30000.00"),
                "transaction_type": "INCOME",
                "category": "Income",
            },
        ]
    )


def test_calendar_features_are_created():
    df = engineer_transaction_features(make_transactions())

    first_transaction = df.iloc[0]
    second_transaction = df.iloc[1]

    assert first_transaction["year"] == 2026
    assert first_transaction["month"] == 10
    assert first_transaction["year_month"] == "2026-10"
    assert first_transaction["day_of_week"] == "Friday"
    assert first_transaction["is_weekend"] == False

    assert second_transaction["day_of_week"] == "Saturday"
    assert second_transaction["is_weekend"] == True


def test_signed_amounts_distinguish_income_and_expenses():
    df = engineer_transaction_features(make_transactions())

    assert df.iloc[0]["signed_amount"] == -850.0
    assert df.iloc[1]["signed_amount"] == 30000.0


def test_original_amounts_are_not_modified():
    original = make_transactions()
    original_copy = original.copy(deep=True)

    engineer_transaction_features(original)

    pd.testing.assert_frame_equal(original, original_copy)


def test_missing_required_column_is_rejected():
    df = make_transactions().drop(columns=["transaction_type"])

    with pytest.raises(ValueError, match="Missing required columns"):
        engineer_transaction_features(df)


def test_invalid_date_is_rejected():
    df = make_transactions()
    df.loc[0, "transaction_date"] = "not-a-date"

    with pytest.raises(ValueError, match="invalid dates"):
        engineer_transaction_features(df)


def test_invalid_amount_is_rejected():
    df = make_transactions()
    df.loc[0, "amount"] = "not-a-number"

    with pytest.raises(ValueError, match="finite numbers"):
        engineer_transaction_features(df)


def test_zero_amount_is_rejected():
    df = make_transactions()
    df.loc[0, "amount"] = 0

    with pytest.raises(ValueError, match="greater than zero"):
        engineer_transaction_features(df)


def test_invalid_transaction_type_is_rejected():
    df = make_transactions()
    df.loc[0, "transaction_type"] = "TRANSFER"

    with pytest.raises(ValueError, match="INCOME or EXPENSE"):
        engineer_transaction_features(df)


def test_transaction_type_is_normalized():
    df = make_transactions()
    df.loc[0, "transaction_type"] = " expense "

    processed = engineer_transaction_features(df)

    assert processed.iloc[0]["transaction_type"] == "EXPENSE"


def test_empty_dataframe_is_supported():
    df = pd.DataFrame(
        columns=[
            "transaction_date",
            "amount",
            "transaction_type",
        ]
    )

    processed = engineer_transaction_features(df)

    assert processed.empty
    assert "signed_amount" in processed.columns
    assert "year_month" in processed.columns