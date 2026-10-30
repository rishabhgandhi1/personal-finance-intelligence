from datetime import date, datetime, timezone
from decimal import Decimal

from sqlalchemy import (
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from src.database.connection import Base  # type: ignore[import-not-found]


def utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc)


class ImportBatch(Base):
    __tablename__ = "import_batches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    total_rows: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    accepted_rows: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    rejected_rows: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    imported_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    __table_args__ = (
        CheckConstraint("total_rows >= 0", name="ck_batch_total_nonnegative"),
        CheckConstraint("accepted_rows >= 0", name="ck_batch_accepted_nonnegative"),
        CheckConstraint("rejected_rows >= 0", name="ck_batch_rejected_nonnegative"),
    )


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )
    transaction_id: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
    )
    transaction_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
        index=True,
    )
    description: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )
    amount: Mapped[Decimal] = mapped_column(
        Numeric(12, 2),
        nullable=False,
    )
    transaction_type: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
    )
    category: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )
    category_source: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="original",
    )
    payment_method: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )
    account_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="BANK_ACCOUNT",
    )
    import_batch_id: Mapped[int | None] = mapped_column(
        ForeignKey("import_batches.id"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    __table_args__ = (
        CheckConstraint("amount > 0", name="ck_transaction_amount_positive"),
        CheckConstraint(
            "transaction_type IN ('INCOME', 'EXPENSE')",
            name="ck_transaction_type",
        ),
        CheckConstraint(
            "category_source IN ('original', 'user', 'model')",
            name="ck_category_source",
        ),
    )


class MonthlyForecast(Base):
    __tablename__ = "monthly_forecasts"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )
    forecast_month: Mapped[date] = mapped_column(
        Date,
        nullable=False,
        index=True,
    )
    predicted_expenses: Mapped[Decimal] = mapped_column(
        Numeric(12, 2),
        nullable=False,
    )
    lower_bound: Mapped[Decimal] = mapped_column(
        Numeric(12, 2),
        nullable=False,
    )
    upper_bound: Mapped[Decimal] = mapped_column(
        Numeric(12, 2),
        nullable=False,
    )
    model_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    __table_args__ = (
        CheckConstraint(
            "predicted_expenses >= 0",
            name="ck_forecast_prediction_nonnegative",
        ),
        CheckConstraint(
            "lower_bound >= 0",
            name="ck_forecast_lower_nonnegative",
        ),
        CheckConstraint(
            "upper_bound >= lower_bound",
            name="ck_forecast_bounds_order",
        ),
        UniqueConstraint(
            "forecast_month",
            "model_name",
            "generated_at",
            name="uq_forecast_run",
        ),
    )
