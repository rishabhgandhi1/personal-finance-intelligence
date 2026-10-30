import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure


def plot_monthly_income_vs_expenses(
    monthly_summary: pd.DataFrame,
) -> Figure:
    """
    Visualize monthly income against monthly expenses.
    Args:
        monthly_summary: DataFrame containing month, income,
                         and expenses columns.
    Returns:
        Matplotlib Figure object.
    """
    required_columns = {"month", "income", "expenses"}
    if not isinstance(monthly_summary, pd.DataFrame):
        raise TypeError("monthly_summary must be a Pandas DataFrame.")
    if not required_columns.issubset(monthly_summary.columns):
        raise ValueError(
            "monthly_summary must contain month, income, and expenses."
        )
    if monthly_summary.empty:
        raise ValueError("Monthly summary cannot be empty.")

        # Convert monetary columns into numeric values.

    plot_data = monthly_summary.copy()

    for column in ["income", "expenses"]:

        plot_data[column] = pd.to_numeric(

            plot_data[column],

            errors="coerce",

        )

        if (

            plot_data[column].isna().any()

            or not np.isfinite(

                plot_data[column].to_numpy(dtype=float)

            ).all()

        ):

            raise ValueError(

                f"{column} must contain finite numeric values."

            )
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        monthly_summary["month"],
        monthly_summary["income"],
        marker="o",
        label="Income",
    )
    ax.plot(
        monthly_summary["month"],
        monthly_summary["expenses"],
        marker="o",
        label="Expenses",
    )
    ax.set_title("Monthly Income vs Expenses")
    ax.set_xlabel("Month")
    ax.set_ylabel("Amount (₹)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    fig.tight_layout()
    return fig
def plot_category_spending(
    category_summary: pd.DataFrame,
) -> Figure:
    """
    Visualize total expenses across different categories.
    """
    required_columns = {"category", "total_spending"}

    if not isinstance(category_summary, pd.DataFrame):
        raise TypeError("category_summary must be a Pandas DataFrame.")

    if not required_columns.issubset(category_summary.columns):
        raise ValueError(
            "category_summary must contain category and total_spending."
        )

    if category_summary.empty:
        raise ValueError("Category summary cannot be empty.")

    plot_data = category_summary.copy()

    plot_data["total_spending"] = pd.to_numeric(
        plot_data["total_spending"],
        errors="coerce",
    )

    if (
        plot_data["total_spending"].isna().any()
        or not np.isfinite(
            plot_data["total_spending"].to_numpy(dtype=float)
        ).all()
        or (plot_data["total_spending"] < 0).any()
    ):
        raise ValueError(
            "Total spending must contain finite nonnegative values."
        )

    fig, ax = plt.subplots(figsize=(12, 7))

    sns.barplot(
        data=plot_data,
        x="total_spending",
        y="category",
        ax=ax,
    )

    ax.set_title("Category-wise Expense Distribution")
    ax.set_xlabel("Total Spending (₹)")
    ax.set_ylabel("Expense Category")

    fig.tight_layout()

    return fig

def plot_monthly_spending_trend(monthly_summary: pd.DataFrame) -> Figure:
    """
    Plot monthly expenses as a trend line.

    Args:
        monthly_summary: DataFrame containing month and expenses columns.

    Returns:
        A Matplotlib Figure.
    """
    required_columns = {"month", "expenses"}

    if not isinstance(monthly_summary, pd.DataFrame):
        raise TypeError("monthly_summary must be a pandas DataFrame.")

    missing_columns = required_columns - set(monthly_summary.columns)

    if missing_columns:
        raise ValueError(
            f"monthly_summary must contain columns: {sorted(required_columns)}"
        )

    if monthly_summary.empty:
        raise ValueError("monthly_summary cannot be empty.")

    plot_data = monthly_summary.copy()

    plot_data["expenses"] = pd.to_numeric(
        plot_data["expenses"], errors="coerce"
    )

    if (
        plot_data["expenses"].isna().any()
        or not np.isfinite(plot_data["expenses"]).all()
        or (plot_data["expenses"] < 0).any()
    ):
        raise ValueError(
            "expenses must contain finite nonnegative values."
        )

    plot_data = plot_data.sort_values("month")

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.lineplot(
        data=plot_data,
        x="month",
        y="expenses",
        marker="o",
        ax=ax,
    )

    ax.set_title("Monthly Spending Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Expenses (₹)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    return fig