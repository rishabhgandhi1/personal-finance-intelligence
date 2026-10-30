from decimal import Decimal
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from dataclasses import dataclass
from src.database.models import Transaction

@dataclass
class ForecastEvaluation:
    """Store the results of monthly expense forecast evaluation."""

    actual: pd.Series
    predicted: pd.Series
    mae: float
    rmse: float

def get_total_income(db: Session) -> Decimal:
    """Calculate the total income across all transactions."""
    result = db.scalar(
        select(func.coalesce(func.sum(Transaction.amount), 0)).where(
            Transaction.transaction_type == "INCOME"
        )
    )

    return Decimal(str(result))


def get_total_expenses(db: Session) -> Decimal:
    """Calculate the total expenses across all transactions."""
    result = db.scalar(
        select(func.coalesce(func.sum(Transaction.amount), 0)).where(
            Transaction.transaction_type == "EXPENSE"
        )
    )

    return Decimal(str(result))


def get_net_balance(db: Session) -> Decimal:
    """Calculate net balance as total income minus total expenses."""
    return get_total_income(db) - get_total_expenses(db)

def get_category_wise_spending(db: Session) -> pd.DataFrame:
    """Calculate total expenses grouped by category."""

    statement = (
        select(
            Transaction.category,
            func.sum(Transaction.amount).label("total_spending"),
        )
        .where(Transaction.transaction_type == "EXPENSE")
        .group_by(Transaction.category)
        .order_by(func.sum(Transaction.amount).desc())
    )

    results = db.execute(statement).all()

    return pd.DataFrame(
        results,
        columns=["category", "total_spending"],
    )

def get_monthly_summary(db: Session) -> pd.DataFrame:
    """Calculate monthly income and expenses."""

    statement = select(
        Transaction.transaction_date,
        Transaction.amount,
        Transaction.transaction_type,
    ).order_by(Transaction.transaction_date)

    results = db.execute(statement).all()

    df = pd.DataFrame(
        results,
        columns=["transaction_date", "amount", "transaction_type"],
    )

    if df.empty:
        return pd.DataFrame(
            columns=["month", "income", "expenses"]
        )

    df["transaction_date"] = pd.to_datetime(df["transaction_date"])

    df["month"] = (
        df["transaction_date"]
        .dt.to_period("M")
        .astype(str)
    )

    summary = (
        df.pivot_table(
            index="month",
            columns="transaction_type",
            values="amount",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )

    summary.columns.name = None

    for column in ["INCOME", "EXPENSE"]:
        if column not in summary.columns:
            summary[column] = 0

    summary = summary.rename(
        columns={
            "INCOME": "income",
            "EXPENSE": "expenses",
        }
    )

    return summary[["month", "income", "expenses"]]

def forecast_monthly_expenses(
    monthly_expenses: pd.Series,
    forecast_periods: int = 1,
) -> pd.Series:
    """
    Forecast future monthly expenses using Holt's linear trend method.
    """

    if not isinstance(monthly_expenses, pd.Series):
        raise TypeError("monthly_expenses must be a Pandas Series.")

    if forecast_periods < 1:
        raise ValueError("forecast_periods must be at least 1.")

    expenses = pd.to_numeric(monthly_expenses, errors="coerce")

    if expenses.isna().any():
        raise ValueError("Monthly expenses cannot contain missing values.")

    if not np.isfinite(expenses.to_numpy(dtype=float)).all():
        raise ValueError("Monthly expenses must contain finite values.")

    if (expenses < 0).any():
        raise ValueError("Monthly expenses cannot be negative.")

    if len(expenses) < 3:
        raise ValueError("At least 3 months of data are required.")

    model = ExponentialSmoothing(
        expenses.astype(float),
        trend="add",
        seasonal=None,
        initialization_method="estimated",
    )

    fitted_model = model.fit(optimized=True)

    forecast = fitted_model.forecast(forecast_periods)

    return forecast.clip(lower=0)

def evaluate_monthly_expense_forecast(
    monthly_expenses: pd.Series,
    test_periods: int = 3,
) -> ForecastEvaluation:
    """
    Evaluate monthly expense forecasting using a chronological holdout.

    The model trains on earlier months and forecasts the final
    test_periods months.
    """
    if not isinstance(monthly_expenses, pd.Series):
        raise TypeError("monthly_expenses must be a Pandas Series.")

    if (
        isinstance(test_periods, bool)
        or not isinstance(test_periods, int)
        or test_periods < 1
    ):
        raise ValueError("test_periods must be a positive integer.")

    if test_periods >= len(monthly_expenses):
        raise ValueError(
            "Test periods must be fewer than the total number of months."
        )

    training_data = monthly_expenses.iloc[:-test_periods]
    actual = monthly_expenses.iloc[-test_periods:]

    predicted = forecast_monthly_expenses(
        training_data,
        forecast_periods=test_periods,
    )

    predicted.index = actual.index

    errors = actual.to_numpy(dtype=float) - predicted.to_numpy(dtype=float)

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    return ForecastEvaluation(
        actual=actual,
        predicted=predicted,
        mae=mae,
        rmse=rmse,
    )

def evaluate_last_observation_baseline(
    monthly_expenses: pd.Series,
    test_periods: int = 3,
) -> ForecastEvaluation:
    """
    Evaluate a baseline that predicts future expenses using
    the last observed training month's expenses.
    """
    if not isinstance(monthly_expenses, pd.Series):
        raise TypeError("monthly_expenses must be a Pandas Series.")

    if (
        isinstance(test_periods, bool)
        or not isinstance(test_periods, int)
        or test_periods < 1
    ):
        raise ValueError("test_periods must be a positive integer.")

    if len(monthly_expenses) - test_periods < 3:
        raise ValueError(
            "At least 3 training months are required."
        )

    training_data = pd.to_numeric(
        monthly_expenses.iloc[:-test_periods],
        errors="coerce",
    )

    actual = monthly_expenses.iloc[-test_periods:]

    if (
        training_data.isna().any()
        or not np.isfinite(training_data.to_numpy(dtype=float)).all()
        or (training_data < 0).any()
    ):
        raise ValueError("Training expenses must be finite and nonnegative.")

    last_observed_expense = float(training_data.iloc[-1])

    predicted = pd.Series(
        [last_observed_expense] * test_periods,
        index=actual.index,
        dtype=float,
    )

    actual_values = pd.to_numeric(actual, errors="coerce")

    if (
        actual_values.isna().any()
        or not np.isfinite(actual_values.to_numpy(dtype=float)).all()
        or (actual_values < 0).any()
    ):
        raise ValueError("Actual expenses must be finite and nonnegative.")

    errors = (
        actual_values.to_numpy(dtype=float)
        - predicted.to_numpy(dtype=float)
    )

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    return ForecastEvaluation(
        actual=actual_values,
        predicted=predicted,
        mae=mae,
        rmse=rmse,
    )