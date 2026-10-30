from datetime import date
from decimal import Decimal
from src.analytics.spending_analytics import (
    evaluate_monthly_expense_forecast,
    forecast_monthly_expenses,
)
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from src.analytics.visualization import plot_monthly_spending_trend
from src.analytics.spending_analytics import (
    evaluate_last_observation_baseline,
)
from src.analytics.statistical_analysis import (
    calculate_spending_variability,
)
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.database.connection import Base
from src.database.models import Transaction
from src.analytics.spending_analytics import (
    get_total_income,
    get_total_expenses,
    get_net_balance,
    get_category_wise_spending,
    get_monthly_summary,
)


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    Base.metadata.create_all(bind=engine)

    TestingSessionLocal = sessionmaker(
        bind=engine,
        autoflush=False,
        expire_on_commit=False,
    )

    session = TestingSessionLocal()

    transactions = [
        Transaction(
            transaction_id="TXN001",
            transaction_date=date(2026, 1, 5),
            description="Monthly Salary",
            amount=Decimal("50000.00"),
            transaction_type="INCOME",
            category="Income",
        ),
        Transaction(
            transaction_id="TXN002",
            transaction_date=date(2026, 1, 10),
            description="Grocery Shopping",
            amount=Decimal("5000.00"),
            transaction_type="EXPENSE",
            category="Groceries",
        ),
        Transaction(
            transaction_id="TXN003",
            transaction_date=date(2026, 1, 15),
            description="Restaurant",
            amount=Decimal("2000.00"),
            transaction_type="EXPENSE",
            category="Food",
        ),
        Transaction(
            transaction_id="TXN004",
            transaction_date=date(2026, 2, 5),
            description="Monthly Salary",
            amount=Decimal("50000.00"),
            transaction_type="INCOME",
            category="Income",
        ),
        Transaction(
            transaction_id="TXN005",
            transaction_date=date(2026, 2, 12),
            description="Grocery Shopping",
            amount=Decimal("3000.00"),
            transaction_type="EXPENSE",
            category="Groceries",
        ),
    ]

    session.add_all(transactions)
    session.commit()

    yield session

    session.close()
    engine.dispose()


def test_total_income(db_session):
    assert get_total_income(db_session) == Decimal("100000.00")


def test_total_expenses(db_session):
    assert get_total_expenses(db_session) == Decimal("10000.00")


def test_net_balance(db_session):
    assert get_net_balance(db_session) == Decimal("90000.00")


def test_category_wise_spending(db_session):
    result = get_category_wise_spending(db_session)

    assert result["category"].tolist() == ["Groceries", "Food"]

    assert result["total_spending"].tolist() == [
        Decimal("8000.00"),
        Decimal("2000.00"),
    ]


def test_monthly_summary(db_session):
    result = get_monthly_summary(db_session)

    assert result["month"].tolist() == ["2026-01", "2026-02"]

    assert result["income"].tolist() == [
        Decimal("50000.00"),
        Decimal("50000.00"),
    ]

    assert result["expenses"].tolist() == [
        Decimal("7000.00"),
        Decimal("3000.00"),
    ]

def test_forecast_monthly_expenses_linear_trend():
    expenses = pd.Series([20000, 22000, 24000, 26000, 28000])

    forecast = forecast_monthly_expenses(expenses)

    assert len(forecast) == 1
    assert forecast.iloc[0] == pytest.approx(30000, abs=1)


def test_evaluate_monthly_expense_forecast():
    expenses = pd.Series(
        [
            20000, 22000, 24000, 26000, 28000,
            30000, 32000, 34000, 36000, 38000,
        ]
    )

    result = evaluate_monthly_expense_forecast(
        expenses,
        test_periods=2,
    )

    assert len(result.actual) == 2
    assert len(result.predicted) == 2

    assert result.actual.tolist() == [36000, 38000]

    assert result.mae >= 0
    assert result.rmse >= 0


def test_forecast_rejects_insufficient_data():
    expenses = pd.Series([10000, 12000])

    with pytest.raises(ValueError):
        forecast_monthly_expenses(expenses)


def test_forecast_rejects_negative_expenses():
    expenses = pd.Series([10000, 12000, -5000])

    with pytest.raises(ValueError):
        forecast_monthly_expenses(expenses)


def test_forecast_evaluation_rejects_invalid_test_periods():
    expenses = pd.Series([10000, 12000, 14000, 16000])

    with pytest.raises(ValueError):
        evaluate_monthly_expense_forecast(expenses, test_periods=0)

    with pytest.raises(ValueError):
        evaluate_monthly_expense_forecast(expenses, test_periods=4)

def test_last_observation_baseline():
    expenses = pd.Series(
        [20000, 22000, 24000, 26000, 28000]
    )

    result = evaluate_last_observation_baseline(
        expenses,
        test_periods=2,
    )

    assert result.actual.tolist() == [26000, 28000]
    assert result.predicted.tolist() == [24000, 24000]

    assert result.mae == pytest.approx(3000)
    assert result.rmse == pytest.approx(3162.27766)


def test_baseline_rejects_invalid_test_periods():
    expenses = pd.Series([10000, 12000, 14000, 16000])

    with pytest.raises(ValueError):
        evaluate_last_observation_baseline(expenses, 0)

    with pytest.raises(ValueError):
        evaluate_last_observation_baseline(expenses, 4)


def test_baseline_rejects_insufficient_training_data():
    expenses = pd.Series([10000, 12000, 14000])

    with pytest.raises(ValueError):
        evaluate_last_observation_baseline(
            expenses,
            test_periods=1,
        )

def test_calculate_spending_variability():
    expenses = pd.Series([10000, 20000, 30000, 40000])

    result = calculate_spending_variability(expenses)

    assert result["mean"] == pytest.approx(25000)
    assert result["median"] == pytest.approx(25000)
    assert result["minimum"] == pytest.approx(10000)
    assert result["maximum"] == pytest.approx(40000)

    assert result["standard_deviation"] == pytest.approx(
        12909.944487
    )

    assert result["coefficient_of_variation"] == pytest.approx(
        51.63977795
    )


def test_spending_variability_rejects_empty_data():
    with pytest.raises(ValueError):
        calculate_spending_variability(pd.Series(dtype=float))


def test_spending_variability_rejects_negative_values():
    with pytest.raises(ValueError):
        calculate_spending_variability(
            pd.Series([10000, -5000, 20000])
        )


def test_spending_variability_requires_two_months():
    with pytest.raises(ValueError):
        calculate_spending_variability(pd.Series([10000]))

def test_compare_monthly_spending():
    from src.analytics.statistical_analysis import compare_monthly_spending

    expenses = pd.Series([
        20000, 22000, 21000, 23000, 22000, 24000,
        25000, 26000, 24000, 27000, 25000, 28000,
    ])

    result = compare_monthly_spending(expenses)

    assert result["first_period_mean"] == 22000
    assert result["second_period_mean"] == 25833.333333333332
    assert 0 <= result["p_value"] <= 1
    assert result["significance_level"] == 0.05
    assert isinstance(result["statistically_significant"], bool)


def test_compare_monthly_spending_requires_12_months():
    from src.analytics.statistical_analysis import compare_monthly_spending

    expenses = pd.Series([10000, 12000, 15000])

    with pytest.raises(ValueError, match="Exactly 12 months"):
        compare_monthly_spending(expenses)


def test_compare_monthly_spending_rejects_negative_values():
    from src.analytics.statistical_analysis import compare_monthly_spending

    expenses = pd.Series([10000] * 11 + [-5000])

    with pytest.raises(ValueError, match="cannot be negative"):
        compare_monthly_spending(expenses)

def test_plot_monthly_income_vs_expenses():
    from src.analytics.visualization import plot_monthly_income_vs_expenses
    import matplotlib.pyplot as plt
    monthly_summary = pd.DataFrame({
        "month": ["2026-01", "2026-02", "2026-03"],
        "income": [30000, 35000, 32000],
        "expenses": [20000, 25000, 22000],
    })
    fig = plot_monthly_income_vs_expenses(monthly_summary)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    assert len(fig.axes[0].lines) == 2
    plt.close(fig)
def test_plot_monthly_income_vs_expenses_rejects_empty_data():
    from src.analytics.visualization import plot_monthly_income_vs_expenses
    monthly_summary = pd.DataFrame(
        columns=["month", "income", "expenses"]
    )
    with pytest.raises(ValueError, match="cannot be empty"):
        plot_monthly_income_vs_expenses(monthly_summary)
def test_plot_monthly_income_vs_expenses_rejects_missing_columns():
    from src.analytics.visualization import plot_monthly_income_vs_expenses
    monthly_summary = pd.DataFrame({
        "month": ["2026-01"],
        "income": [30000],
    })
    with pytest.raises(ValueError, match="must contain"):
        plot_monthly_income_vs_expenses(monthly_summary)
def test_plot_monthly_income_vs_expenses_rejects_invalid_values():
    from src.analytics.visualization import plot_monthly_income_vs_expenses
    monthly_summary = pd.DataFrame({
        "month": ["2026-01"],
        "income": ["invalid"],
        "expenses": [20000],
    })
    with pytest.raises(ValueError, match="finite numeric values"):
        plot_monthly_income_vs_expenses(monthly_summary)

def test_plot_category_spending():
    from src.analytics.visualization import plot_category_spending
    import matplotlib.pyplot as plt

    category_summary = pd.DataFrame({
        "category": ["Housing", "Food", "Transport"],
        "total_spending": [50000, 20000, 15000],
    })

    fig = plot_category_spending(category_summary)

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    assert len(fig.axes[0].patches) == 3

    plt.close(fig)


def test_plot_category_spending_rejects_empty_data():
    from src.analytics.visualization import plot_category_spending

    category_summary = pd.DataFrame(
        columns=["category", "total_spending"]
    )

    with pytest.raises(ValueError, match="cannot be empty"):
        plot_category_spending(category_summary)


def test_plot_category_spending_rejects_missing_columns():
    from src.analytics.visualization import plot_category_spending

    category_summary = pd.DataFrame({
        "category": ["Food"],
    })

    with pytest.raises(ValueError, match="must contain"):
        plot_category_spending(category_summary)


def test_plot_category_spending_rejects_invalid_values():
    from src.analytics.visualization import plot_category_spending

    category_summary = pd.DataFrame({
        "category": ["Food"],
        "total_spending": [-1000],
    })

    with pytest.raises(ValueError, match="finite nonnegative values"):
        plot_category_spending(category_summary)

    def test_plot_monthly_spending_trend():
        from src.analytics.visualization import plot_monthly_spending_trend
        import matplotlib.pyplot as plt
    monthly_summary = pd.DataFrame({
        "month": ["2026-01", "2026-02", "2026-03"],
        "expenses": [25000, 30000, 28000],
    })

    fig = plot_monthly_spending_trend(monthly_summary)

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    assert len(fig.axes[0].lines) >= 1

    plt.close(fig)


def test_plot_monthly_spending_trend_rejects_empty_data():
    from src.analytics.visualization import plot_monthly_spending_trend

    monthly_summary = pd.DataFrame(
        columns=["month", "expenses"]
    )

    with pytest.raises(ValueError, match="cannot be empty"):
        plot_monthly_spending_trend(monthly_summary)


def test_plot_monthly_spending_trend_rejects_missing_columns():
    from src.analytics.visualization import plot_monthly_spending_trend

    monthly_summary = pd.DataFrame({
        "month": ["2026-01"],
    })

    with pytest.raises(ValueError, match="must contain"):
        plot_monthly_spending_trend(monthly_summary)


def test_plot_monthly_spending_trend_rejects_invalid_values():
    from src.analytics.visualization import plot_monthly_spending_trend

    monthly_summary = pd.DataFrame({
        "month": ["2026-01"],
        "expenses": [-1000],
    })

    with pytest.raises(ValueError, match="finite nonnegative values"):
        plot_monthly_spending_trend(monthly_summary)

def test_plot_monthly_spending_trend():
    from src.analytics.visualization import plot_monthly_spending_trend
    import matplotlib.pyplot as plt

    monthly_summary = pd.DataFrame({
        "month": ["2026-01", "2026-02", "2026-03"],
        "expenses": [25000, 30000, 28000],
    })

    fig = plot_monthly_spending_trend(monthly_summary)

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    assert len(fig.axes[0].lines) >= 1

    plt.close(fig)