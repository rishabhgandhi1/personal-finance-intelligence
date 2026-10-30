import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "transaction_date",
    "amount",
    "transaction_type",
}


def engineer_transaction_features(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate transaction data and derive analytical features.

    The input DataFrame is not modified.

    Required columns:
        transaction_date
        amount
        transaction_type

    Returns:
        A copy of the DataFrame with normalized values and
        additional features for financial analytics and ML.

    Raises:
        TypeError: If the input is not a Pandas DataFrame.
        ValueError: If required columns or valid values are missing.
    """

    # 1. Validate the input type.
    if not isinstance(transactions, pd.DataFrame):
        raise TypeError(
            "Transactions must be provided as a Pandas DataFrame."
        )

    # 2. Check that the required columns exist.
    missing_columns = REQUIRED_COLUMNS - set(transactions.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # 3. Work on a copy to avoid modifying the original data.
    df = transactions.copy()

    if df.empty:
        # Return an empty DataFrame with the expected feature columns.
        df["transaction_date"] = pd.to_datetime(
            df["transaction_date"]
        )
        df["amount"] = pd.to_numeric(df["amount"])
        df["transaction_type"] = df["transaction_type"].astype("string")

        df["year"] = pd.Series(index=df.index, dtype="int64")
        df["month"] = pd.Series(index=df.index, dtype="int64")
        df["year_month"] = pd.Series(index=df.index, dtype="string")
        df["day_of_week"] = pd.Series(index=df.index, dtype="string")
        df["is_weekend"] = pd.Series(index=df.index, dtype="bool")
        df["signed_amount"] = pd.Series(index=df.index, dtype="float64")

        return df

    # 4. Normalize transaction dates.
    df["transaction_date"] = pd.to_datetime(
    df["transaction_date"],
    format="%Y-%m-%d",
    errors="coerce",
)

    if df["transaction_date"].isna().any():
        raise ValueError(
            "Transaction data contains missing or invalid dates."
        )

    # 5. Convert amounts into a numeric format for analysis.
    df["amount"] = pd.to_numeric(
        df["amount"],
        errors="coerce",
    )

    amounts = df["amount"].to_numpy(dtype=float)

    if not np.isfinite(amounts).all():
        raise ValueError(
            "Transaction amounts must be valid finite numbers."
        )

    if (df["amount"] <= 0).any():
        raise ValueError(
            "Transaction amounts must be greater than zero."
        )

    # 6. Normalize and validate transaction types.
    df["transaction_type"] = (
        df["transaction_type"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    valid_types = {"INCOME", "EXPENSE"}

    if (
        df["transaction_type"].isna().any()
        or not df["transaction_type"].isin(valid_types).all()
    ):
        raise ValueError(
            "Transaction type must be INCOME or EXPENSE."
        )

    # 7. Derive calendar features.
    df["year"] = df["transaction_date"].dt.year
    df["month"] = df["transaction_date"].dt.month

    df["year_month"] = (
        df["transaction_date"]
        .dt.to_period("M")
        .astype(str)
    )

    df["day_of_week"] = df["transaction_date"].dt.day_name()

    df["is_weekend"] = (
        df["transaction_date"].dt.day_of_week >= 5
    )

    # 8. Calculate signed amounts for cash-flow analysis.
    #
    # Stored transaction amounts remain positive.
    # Expenses become negative in this derived analytical column.
    # Income remains positive.
    df["signed_amount"] = np.where(
        df["transaction_type"] == "EXPENSE",
        -df["amount"],
        df["amount"],
    )

    return df