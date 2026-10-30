from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Personal Finance Intelligence API",
        "status": "running",
    }


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "api": "healthy",
        "database": "connected",
    }


def test_summary_endpoint():
    response = client.get("/summary")

    assert response.status_code == 200
    data = response.json()
    assert "total_income" in data
    assert "total_expenses" in data
    assert "net_balance" in data
    assert data["currency"] == "INR"


def test_category_spending_endpoint():
    response = client.get("/spending/categories")

    assert response.status_code == 200
    data = response.json()
    assert "categories" in data
    assert data["currency"] == "INR"
    for category in data["categories"]:
        assert "category" in category
        assert "total_spending" in category


def test_monthly_spending_endpoint():
    response = client.get("/spending/monthly")

    assert response.status_code == 200
    data = response.json()
    assert "monthly_data" in data
    assert data["currency"] == "INR"
    for month in data["monthly_data"]:
        assert "month" in month
        assert "income" in month
        assert "expenses" in month


def test_monthly_expense_forecast_endpoint():
    response = client.get("/forecast/monthly?periods=3")

    assert response.status_code == 200
    data = response.json()
    assert "forecast" in data
    assert data["currency"] == "INR"
    assert data["model"] == "Statsmodels Exponential Smoothing"
    for period in data["forecast"]:
        assert "period" in period
        assert "predicted_expenses" in period


def test_expense_category_prediction():
    response = client.post(
        "/predict/category",
        json={"description": "Swiggy food order"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["description"] == "Swiggy food order"
    assert data["predicted_category"] == "Food"
    assert data["model"] == "TF-IDF + Logistic Regression"


def test_expense_prediction_rejects_short_description():
    response = client.post(
        "/predict/category",
        json={"description": "A"},
    )

    assert response.status_code == 422


def test_expense_prediction_rejects_empty_description():
    response = client.post(
        "/predict/category",
        json={"description": ""},
    )

    assert response.status_code == 422


def test_forecast_rejects_zero_periods():
    response = client.get("/forecast/monthly?periods=0")

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Forecast periods must be between 1 and 12."
    )


def test_forecast_rejects_more_than_twelve_periods():
    response = client.get("/forecast/monthly?periods=13")

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Forecast periods must be between 1 and 12."
    )