from dataclasses import dataclass
from typing import Any, cast

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
REQUIRED_COLUMNS = {
    "description",
    "transaction_type",
    "category",
}
@dataclass
class CategorizationEvaluation:
    """Results from evaluating the expense categorization model."""
    accuracy: float
    report: dict[str, Any]
    training_rows: int
    testing_rows: int
def prepare_labeled_expenses(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Select valid, labeled expense transactions for model training."""
    if not isinstance(transactions, pd.DataFrame):
        raise TypeError(
            "Transactions must be provided as a Pandas DataFrame."
        )
    missing_columns = REQUIRED_COLUMNS - set(transactions.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )
    df = transactions.copy()
    df["transaction_type"] = (
        df["transaction_type"]
        .astype("string")
        .str.strip()
        .str.upper()
    )
    df["description"] = (
        df["description"]
        .astype("string")
        .str.strip()
    )
    df["category"] = (
        df["category"]
        .astype("string")
        .str.strip()
    )
    valid_rows = (
        df["transaction_type"].eq("EXPENSE")
        & df["description"].notna()
        & df["description"].ne("")
        & df["category"].notna()
        & df["category"].ne("")
    )
    df = df.loc[
        valid_rows,
        ["description", "category"],
    ].copy()
    if df.empty:
        raise ValueError(
            "No labeled expense transactions are available for training."
        )
    return df
def build_categorization_pipeline() -> Pipeline:
    """Create the text-processing and classification pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    strip_accents="unicode",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )
def evaluate_categorizer(
    transactions: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> CategorizationEvaluation:
    """Train on one subset and evaluate on held-out transactions."""
    labeled = prepare_labeled_expenses(transactions)
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    class_counts = labeled["category"].value_counts()
    if len(class_counts) < 2:
        raise ValueError(
            "At least two expense categories are required for training."
        )
    if (class_counts < 2).any():
        raise ValueError(
            "Each category needs at least two labeled transactions "
            "for stratified evaluation."
        )
    test_rows = max(
        int(round(len(labeled) * test_size)),
        len(class_counts),
    )
    if test_rows >= len(labeled):
        raise ValueError(
            "Not enough labeled transactions for a train/test split."
        )
    train_data, test_data = train_test_split(
        labeled,
        test_size=test_rows,
        random_state=random_state,
        stratify=labeled["category"],
    )
    model = build_categorization_pipeline()
    model.fit(
        train_data["description"],
        train_data["category"],
    )
    predictions = model.predict(test_data["description"])
    return CategorizationEvaluation(
        accuracy=float(
            accuracy_score(test_data["category"], predictions)
        ),
        report=cast(
            dict[str, Any],
            classification_report(
                test_data["category"],
                predictions,
                output_dict=True,
                zero_division=0,
            ),
        ),
        training_rows=len(train_data),
        testing_rows=len(test_data),
    )


def evaluate_on_unseen_descriptions(
    transactions: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> CategorizationEvaluation:
    """Evaluate on descriptions excluded entirely from model training."""
    labeled = prepare_labeled_expenses(transactions)
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    # Each unique description belongs to exactly one split.
    groups = labeled["description"].str.strip().str.casefold()
    unique_descriptions = groups.nunique()
    if unique_descriptions < 2:
        raise ValueError(
            "At least two unique descriptions are required."
        )
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_indices, test_indices = next(
        splitter.split(
            labeled,
            labeled["category"],
            groups=groups,
        )
    )
    train_data = labeled.iloc[train_indices]
    test_data = labeled.iloc[test_indices]
    if train_data["category"].nunique() < 2:
        raise ValueError(
            "Training data must contain at least two categories."
        )
    model = build_categorization_pipeline()
    model.fit(
        train_data["description"],
        train_data["category"],
    )
    predictions = model.predict(test_data["description"])
    return CategorizationEvaluation(
        accuracy=float(
            accuracy_score(test_data["category"], predictions)
        ),
        report=cast(
            dict[str, Any],
            classification_report(
                test_data["category"],
                predictions,
                output_dict=True,
                zero_division=0,
            ),
        ),
        training_rows=len(train_data),
        testing_rows=len(test_data),
    )

def train_categorizer(
    transactions: pd.DataFrame,
) -> Pipeline:
    """Train the categorization model on all available labeled expenses."""
    labeled = prepare_labeled_expenses(transactions)
    if labeled["category"].nunique() < 2:
        raise ValueError(
            "At least two expense categories are required for training."
        )
    model = build_categorization_pipeline()
    model.fit(
        labeled["description"],
        labeled["category"],
    )
    return model
def predict_categories(
    model: Pipeline,
    descriptions: list[str],
) -> list[str]:
    """Predict categories for transaction descriptions."""
    if not descriptions:
        return []
    if any(
        not isinstance(description, str)
        or not description.strip()
        for description in descriptions
    ):
        raise ValueError(
            "Every description must be a non-empty string."
        )
    predictions = model.predict(descriptions)
    return [str(category) for category in predictions]