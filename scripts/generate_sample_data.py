from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "sample" / "transactions.csv"

SEED = 42
NUM_TRANSACTIONS = 1500

rng = np.random.default_rng(SEED)

CATEGORY_DESCRIPTIONS = {
    "Food": [
        "Swiggy Order",
        "Zomato Order",
        "Restaurant Payment",
        "Cafe Payment",
    ],
    "Groceries": [
        "Grocery Store",
        "Supermarket",
        "Vegetable Market",
        "Monthly Groceries",
    ],
    "Transport": [
        "Uber Ride",
        "Ola Ride",
        "Metro Recharge",
        "Bus Ticket",
        "Fuel Station",
    ],
    "Shopping": [
        "Clothing Store",
        "Online Shopping",
        "Electronics Store",
        "Footwear Store",
    ],
    "Utilities": [
        "Electricity Bill",
        "Internet Bill",
        "Mobile Recharge",
        "Water Bill",
    ],
    "Healthcare": [
        "Pharmacy",
        "Medical Store",
        "Clinic Payment",
        "Diagnostic Center",
    ],
    "Entertainment": [
        "Movie Tickets",
        "Gaming Purchase",
        "Concert Ticket",
        "Entertainment Venue",
    ],
    "Education": [
        "Online Course",
        "Book Store",
        "Course Fee",
        "Study Materials",
    ],
    "Housing": [
        "Monthly Rent",
        "Maintenance Payment",
        "Home Repair",
    ],
    "Travel": [
        "Hotel Booking",
        "Train Ticket",
        "Flight Booking",
        "Travel Booking",
    ],
    "Personal Care": [
        "Salon Payment",
        "Personal Care Store",
        "Grooming Products",
    ],
    "Subscriptions": [
        "Music Subscription",
        "Video Subscription",
        "Cloud Storage",
    ],
}

EXPENSE_RANGES = {
    "Food": (100, 1200),
    "Groceries": (200, 3500),
    "Transport": (30, 1500),
    "Shopping": (200, 8000),
    "Utilities": (200, 3000),
    "Healthcare": (100, 5000),
    "Entertainment": (150, 2500),
    "Education": (200, 10000),
    "Housing": (3000, 18000),
    "Travel": (500, 15000),
    "Personal Care": (100, 2000),
    "Subscriptions": (99, 999),
}

PAYMENT_METHODS = [
    "UPI",
    "CARD",
    "CASH",
    "NET_BANKING",
]

TRANSACTION_COLUMNS = [
    "transaction_id",
    "transaction_date",
    "description",
    "amount",
    "transaction_type",
    "category",
    "payment_method",
    "account_type",
]


def generate_transactions() -> pd.DataFrame:
    """Generate deterministic synthetic personal finance transactions."""
    start_date = pd.Timestamp("2025-11-01")
    end_date = pd.Timestamp("2026-10-31")

    dates = pd.date_range(start_date, end_date, freq="D")
    records = []
    for index in range(NUM_TRANSACTIONS):
        transaction_date = pd.Timestamp(rng.choice(dates.to_numpy()))
        # Generate salary income at the beginning of each month.
        if index % 25 == 0:
            category = "Income"
            transaction_type = "INCOME"
            description = "Monthly Salary"
            amount = 45000.00
            payment_method = "NET_BANKING"
        else:
            category = str(
                rng.choice(
                    list(CATEGORY_DESCRIPTIONS.keys()),
                    p=[
                        0.15,
                        0.18,
                        0.12,
                        0.10,
                        0.08,
                        0.05,
                        0.06,
                        0.04,
                        0.05,
                        0.04,
                        0.04,
                        0.09,
                    ],
                )
            )
            transaction_type = "EXPENSE"
            description = str(rng.choice(CATEGORY_DESCRIPTIONS[category]))
            low, high = EXPENSE_RANGES[category]
            amount = round(float(rng.uniform(low, high)), 2)
            payment_method = str(rng.choice(PAYMENT_METHODS))
        records.append(
            {
                "transaction_id": f"TXN{index + 1:06d}",
                "transaction_date": transaction_date.strftime("%Y-%m-%d"),
                "description": description,
                "amount": amount,
                "transaction_type": transaction_type,
                "category": category,
                "payment_method": payment_method,
                "account_type": "BANK_ACCOUNT",
            }
        )
    df = pd.DataFrame(records, columns=TRANSACTION_COLUMNS)
    # Shuffle rows so transactions aren't grouped by generation order.
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    return df


if __name__ == "__main__":
    transactions = generate_transactions()

    print(f"Generated {len(transactions)} transactions.")
    print(f"Saved to: {OUTPUT_PATH}")
    print("\nTransaction type counts:")
    print(transactions["transaction_type"].value_counts())
    print("\nCategory counts:")
    print(transactions["category"].value_counts())
    print("\nFirst five transactions:")
    print(transactions.head())