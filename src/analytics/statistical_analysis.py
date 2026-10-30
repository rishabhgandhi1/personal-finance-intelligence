import numpy as np
import pandas as pd
from scipy import stats


def calculate_spending_variability(
    monthly_expenses: pd.Series,
) -> dict:
    """
    Calculate descriptive statistics for monthly expenses.
    """

    if not isinstance(monthly_expenses, pd.Series):
        raise TypeError("monthly_expenses must be a Pandas Series.")

    expenses = pd.to_numeric(monthly_expenses, errors="coerce")

    if expenses.empty:
        raise ValueError("Monthly expenses cannot be empty.")

    if expenses.isna().any():
        raise ValueError("Monthly expenses cannot contain missing values.")

    if not np.isfinite(expenses.to_numpy(dtype=float)).all():
        raise ValueError("Monthly expenses must contain finite values.")

    if (expenses < 0).any():
        raise ValueError("Monthly expenses cannot be negative.")

    if len(expenses) < 2:
        raise ValueError("At least 2 months of data are required.")

    mean_expenses = float(expenses.mean())
    standard_deviation = float(expenses.std(ddof=1))

    coefficient_of_variation = (
        standard_deviation / mean_expenses * 100
        if mean_expenses > 0
        else None
    )

    return {
        "mean": mean_expenses,
        "median": float(expenses.median()),
        "standard_deviation": standard_deviation,
        "minimum": float(expenses.min()),
        "maximum": float(expenses.max()),
        "coefficient_of_variation": coefficient_of_variation,
    }

def compare_monthly_spending(
    monthly_expenses: pd.Series,
) -> dict:
    """
    Compare average spending between the first and last six months
    using Welch's independent two-sample t-test.
    """

    if not isinstance(monthly_expenses, pd.Series):
        raise TypeError("monthly_expenses must be a Pandas Series.")

    expenses = pd.to_numeric(monthly_expenses, errors="coerce")

    if len(expenses) != 12:
        raise ValueError("Exactly 12 months of data are required.")

    if expenses.isna().any():
        raise ValueError("Monthly expenses cannot contain missing values.")

    if not np.isfinite(expenses.to_numpy(dtype=float)).all():
        raise ValueError("Monthly expenses must contain finite values.")

    if (expenses < 0).any():
        raise ValueError("Monthly expenses cannot be negative.")

    first_six_months = expenses.iloc[:6]
    last_six_months = expenses.iloc[6:]

    test_result = stats.ttest_ind(
        first_six_months,
        last_six_months,
        equal_var=False,
        alternative="two-sided",
    )

    t_statistic = float(getattr(test_result, "statistic", np.nan))
    p_value = float(getattr(test_result, "pvalue", np.nan))

    return {
        "first_period_mean": float(first_six_months.mean()),
        "second_period_mean": float(last_six_months.mean()),
        "t_statistic": t_statistic,
        "p_value": p_value,
        "significance_level": 0.05,
        "statistically_significant": p_value < 0.05,
    }