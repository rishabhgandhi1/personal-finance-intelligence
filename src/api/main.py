from fastapi import FastAPI
from fastapi import Depends
from functools import lru_cache
from src.database.connection import get_db, SessionLocal
import pandas as pd
from src.analytics.spending_analytics import (
    forecast_monthly_expenses,
)
from fastapi import HTTPException
from pydantic import BaseModel, Field

from src.database.models import Transaction
from src.services.gemini_service import generate_financial_explanation
from src.ml.expense_categorizer import (
    train_categorizer,
    predict_categories,
)
from sqlalchemy import text
from sqlalchemy.orm import Session
from src.analytics.spending_analytics import (
    get_total_income,
    get_total_expenses,
    get_net_balance,
)
from src.analytics.spending_analytics import (
    get_total_income,
    get_total_expenses,
    get_net_balance,
    get_category_wise_spending,
    get_monthly_summary,
)
from src.analytics.spending_analytics import (
    get_category_wise_spending,
)
from src.database.connection import get_db

app = FastAPI(
    title="Personal Finance Intelligence API",
    description="REST API for personal finance analytics and machine learning.",
    version="1.0.0",
)
@app.get("/")
def root():
    """Return basic API information."""
    return {
        "message": "Personal Finance Intelligence API",
        "status": "running",
    }
class ExpensePredictionRequest(BaseModel):
    description: str = Field(
        min_length=2,
        max_length=500,
        description="Description of the expense",
    )
    
@lru_cache(maxsize=1)
def get_trained_model():
    """Train and cache the expense categorization model."""
    db = SessionLocal()

    try:
        transactions = (
            db.query(Transaction)
            .filter(
                Transaction.transaction_type == "EXPENSE",
                Transaction.category.isnot(None),
            )
            .all()
        )

        training_data = pd.DataFrame(
            [
                {
                    "description": transaction.description,
                    "transaction_type": transaction.transaction_type,
                    "category": transaction.category,
                }
                for transaction in transactions
            ]
        )

        return train_categorizer(training_data)

    finally:
        db.close()

@app.post("/predict/category")
def predict_expense_category(
    request: ExpensePredictionRequest,
):
    """Predict the category of a new expense description."""
    try:
        model = get_trained_model()

        predicted_category = predict_categories(
            model,
            [request.description],
        )[0]

        return {
            "description": request.description,
            "predicted_category": predicted_category,
            "model": "TF-IDF + Logistic Regression",
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error
@app.get("/forecast/monthly")

@app.get("/forecast/monthly")
def monthly_expense_forecast(
    periods: int = 1,
    db: Session = Depends(get_db),
):
    """Forecast future monthly expenses."""
    if periods < 1 or periods > 12:
        raise HTTPException(
            status_code=400,
            detail="Forecast periods must be between 1 and 12.",
        )

    monthly_summary = get_monthly_summary(db)

    if len(monthly_summary) < 3:
        raise HTTPException(
            status_code=400,
            detail="At least 3 months of transaction data are required.",
        )

    monthly_expenses = monthly_summary["expenses"]

    try:
        predictions = forecast_monthly_expenses(
            monthly_expenses,
            forecast_periods=periods,
        )

        return {
            "forecast": [
                {
                    "period": index + 1,
                    "predicted_expenses": round(float(value), 2),
                }
                for index, value in enumerate(predictions)
            ],
            "currency": "INR",
            "model": "Statsmodels Exponential Smoothing",
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error
@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    """Check whether the API can communicate with the database."""
    db.execute(text("SELECT 1"))

    return {
        "api": "healthy",
        "database": "connected",
    }
@app.get("/summary")
def financial_summary(db: Session = Depends(get_db)):
    """Return the user's overall financial summary."""
    total_income = get_total_income(db)
    total_expenses = get_total_expenses(db)
    net_balance = get_net_balance(db)

    return {
        "total_income": float(total_income),
        "total_expenses": float(total_expenses),
        "net_balance": float(net_balance),
        "currency": "INR",
    }
@app.get("/spending/categories")
def category_spending(db: Session = Depends(get_db)):
    """Return total spending grouped by expense category."""
    category_summary = get_category_wise_spending(db)

    return {
        "categories": [
            {
                "category": row["category"],
                "total_spending": float(row["total_spending"]),
            }
            for _, row in category_summary.iterrows()
        ],
        "currency": "INR",
    }
@app.get("/spending/monthly")
def monthly_spending(db: Session = Depends(get_db)):
    """Return monthly income and expense summaries."""
    monthly_summary = get_monthly_summary(db)

    return {
        "monthly_data": [
            {
                "month": str(row["month"]),
                "income": float(row["income"]),
                "expenses": float(row["expenses"]),
            }
            for _, row in monthly_summary.iterrows()
        ],
        "currency": "INR",
    }
@app.get("/explain/finances")
def explain_finances(db: Session = Depends(get_db)):
    """Generate a plain-English explanation of verified financial data."""

    financial_summary = {
        "total_income": float(get_total_income(db)),
        "total_expenses": float(get_total_expenses(db)),
        "net_balance": float(get_net_balance(db)),
        "currency": "INR",
    }

    try:
        explanation = generate_financial_explanation(
            financial_summary
        )

        return {
            "financial_summary": financial_summary,
            "explanation": explanation,
            "model": "Gemini",
        }

    except ValueError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="Financial explanation service is temporarily unavailable.",
        ) from error