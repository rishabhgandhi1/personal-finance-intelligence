from datetime import date
from decimal import Decimal

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.connection import Base
from src.database.models import Transaction


def test_transaction_can_be_saved_and_retrieved():
    # Use an isolated in-memory database for testing.
    test_engine = create_engine("sqlite:///:memory:")

    Base.metadata.create_all(bind=test_engine)

    TestSession = sessionmaker(
        bind=test_engine,
        expire_on_commit=False,
    )

    with TestSession() as session:
        transaction = Transaction(
            transaction_id="TEST001",
            transaction_date=date(2026, 10, 1),
            description="Grocery shopping",
            amount=Decimal("450.00"),
            transaction_type="EXPENSE",
            category="Groceries",
            category_source="original",
            payment_method="UPI",
            account_type="BANK_ACCOUNT",
        )

        session.add(transaction)
        session.commit()

        saved_transaction = (
            session.query(Transaction)
            .filter_by(transaction_id="TEST001")
            .one()
        )

        assert saved_transaction.amount == Decimal("450.00")
        assert saved_transaction.transaction_type == "EXPENSE"
        assert saved_transaction.category == "Groceries"

    test_engine.dispose()