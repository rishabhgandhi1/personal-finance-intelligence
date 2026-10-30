import streamlit as st
import httpx
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Personal Finance Intelligence",
    page_icon="💰",
    layout="wide",
)
import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def get_api_data(endpoint):
    """Fetch data from the FastAPI backend."""
    try:
        response = httpx.get(
            f"{API_BASE_URL}{endpoint}",
            timeout=10,
        )
        response.raise_for_status()
        return response.json()

    except httpx.HTTPError as error:
        st.error(f"Unable to connect to the API: {error}")
        return None
# Dashboard heading
st.title("💰 Personal Finance Intelligence")
st.subheader("Understand your money. Improve your financial decisions.")

st.markdown(
    """
    Welcome to your personal finance dashboard.

    Track your income, understand your spending habits,
    categorize expenses using machine learning, and forecast
    your future expenses.
    """
)

st.divider()
# Financial Summary
st.header("Financial Overview")

summary = get_api_data("/summary")

if summary:
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Income",
        f"₹{summary['total_income']:,.2f}",
    )

    col2.metric(
        "Total Expenses",
        f"₹{summary['total_expenses']:,.2f}",
    )

    col3.metric(
        "Net Balance",
        f"₹{summary['net_balance']:,.2f}",
    )
    # Category-wise Spending
st.divider()
st.header("Category-wise Spending")

category_data = get_api_data("/spending/categories")

if category_data:
    categories = category_data["categories"]

    if categories:
        import pandas as pd

        category_df = pd.DataFrame(categories)

        category_df = category_df.rename(
            columns={
                "category": "Category",
                "total_spending": "Total Spending",
            }
        )

        category_df["Total Spending"] = category_df[
            "Total Spending"
        ].round(2)

        st.bar_chart(
            category_df,
            x="Category",
            y="Total Spending",
        )

        st.dataframe(
            category_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No spending data available.")
# Monthly Income vs Expenses
st.divider()
st.header("Monthly Income vs Expenses")

monthly_data = get_api_data("/spending/monthly")

if monthly_data:
    monthly_df = pd.DataFrame(monthly_data["monthly_data"])

    if not monthly_df.empty:
        chart_df = monthly_df.set_index("month")

        st.line_chart(
            chart_df[["income", "expenses"]]
        )

        st.dataframe(
            monthly_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No monthly data available.")
# Expense Category Prediction
st.divider()
st.header("Expense Category Prediction")

st.write(
    "Enter a transaction description to predict its expense category "
    "using machine learning."
)

description = st.text_input(
    "Transaction Description",
    placeholder="Example: Swiggy food order",
)

if st.button("Predict Category"):
    if description.strip():
        try:
            response = httpx.post(
                f"{API_BASE_URL}/predict/category",
                json={"description": description},
                timeout=30,
            )

            response.raise_for_status()
            prediction = response.json()

            st.success(
                f"Predicted Category: {prediction['predicted_category']}"
            )

            st.caption(f"Model: {prediction['model']}")

        except httpx.HTTPError as error:
            st.error(f"Prediction failed: {error}")
    else:
        st.warning("Please enter a transaction description.")

# Monthly Expense Forecast
st.divider()
st.header("Monthly Expense Forecast")

st.write(
    "Estimate your future expenses using statistical forecasting."
)

forecast_periods = st.number_input(
    "Number of months to forecast",
    min_value=1,
    max_value=12,
    value=3,
    step=1,
)

if st.button("Generate Forecast"):
    forecast_data = get_api_data(
        f"/forecast/monthly?periods={forecast_periods}"
    )

    if forecast_data:
        forecast_df = pd.DataFrame(forecast_data["forecast"])

        forecast_df["predicted_expenses"] = (
            forecast_df["predicted_expenses"].round(2)
        )

        forecast_df = forecast_df.rename(
            columns={
                "period": "Forecast Period",
                "predicted_expenses": "Predicted Expenses (₹)",
            }
        )

        st.subheader("Forecast Results")

        st.bar_chart(
            forecast_df,
            x="Forecast Period",
            y="Predicted Expenses (₹)",
        )

        st.dataframe(
            forecast_df,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(f"Forecasting model: {forecast_data['model']}")
st.divider()

st.header("🤖 AI Financial Insights")

st.write(
    "Get an easy-to-understand explanation of your financial "
    "situation and practical budgeting suggestions using Gemini AI."
)

if st.button("Generate AI Financial Insights"):
    with st.spinner("Analyzing your financial data with Gemini..."):
        try:
            response = httpx.get(
                f"{API_BASE_URL}/explain/finances",
                timeout=60,
            )

            response.raise_for_status()
            explanation_data = response.json()
            st.subheader("Your Financial Analysis")
            st.markdown(explanation_data["explanation"])
            st.caption(
                f"Explanation generated using {explanation_data['model']}"
            )
        except httpx.HTTPStatusError as error:
            st.error(
                f"Financial analysis failed: "
                f"{error.response.status_code} - "
                f"{error.response.text}"
            )
        except httpx.HTTPError as error:
            st.error(f"Unable to connect to the API: {error}")