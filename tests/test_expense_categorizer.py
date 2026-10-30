import pandas as pd
import pytest
from src.ml.expense_categorizer import (
    evaluate_categorizer,
    evaluate_on_unseen_descriptions,
    predict_categories,
    prepare_labeled_expenses,
    train_categorizer,
)
from src.ml.expense_categorizer import (
    evaluate_categorizer,
    predict_categories,
    prepare_labeled_expenses,
    train_categorizer,
)
@pytest.fixture
def sample_transactions():
    return pd.DataFrame(
        [
            {
                "description": "Swiggy food order",
                "transaction_type": "EXPENSE",
                "category": "Food",
            },
            {
                "description": "Restaurant dinner",
                "transaction_type": "EXPENSE",
                "category": "Food",
            },
            {
                "description": "Cafe lunch",
                "transaction_type": "EXPENSE",
                "category": "Food",
            },
            {
                "description": "Vegetable market",
                "transaction_type": "EXPENSE",
                "category": "Groceries",
            },
            {
                "description": "Grocery store purchase",
                "transaction_type": "EXPENSE",
                "category": "Groceries",
            },
            {
                "description": "Fresh vegetables",
                "transaction_type": "EXPENSE",
                "category": "Groceries",
            },
            {
                "description": "Monthly salary",
                "transaction_type": "INCOME",
                "category": "Income",
            },
            {
                "description": "Unknown purchase",
                "transaction_type": "EXPENSE",
                "category": None,
            },
        ]
    )
def test_only_labeled_expenses_are_selected(sample_transactions):
    result = prepare_labeled_expenses(sample_transactions)
    assert len(result) == 6
    assert set(result["category"]) == {"Food", "Groceries"}
def test_missing_required_column_is_rejected():
    transactions = pd.DataFrame(
        {
            "description": ["Cafe lunch"],
            "transaction_type": ["EXPENSE"],
        }
    )
    with pytest.raises(ValueError, match="Missing required columns"):
        prepare_labeled_expenses(transactions)
def test_model_can_be_trained(sample_transactions):
    model = train_categorizer(sample_transactions)
    predictions = model.predict(
        ["Swiggy food order", "Vegetable market"]
    )
    assert len(predictions) == 2
    assert set(predictions).issubset({"Food", "Groceries"})
def test_predictions_reject_blank_descriptions(sample_transactions):
    model = train_categorizer(sample_transactions)
    with pytest.raises(ValueError, match="non-empty string"):
        predict_categories(model, [""])
def test_empty_prediction_input_returns_empty_list(sample_transactions):
    model = train_categorizer(sample_transactions)
    assert predict_categories(model, []) == []
def test_evaluation_returns_metrics(sample_transactions):
    result = evaluate_categorizer(sample_transactions)
    assert 0.0 <= result.accuracy <= 1.0
    assert result.training_rows + result.testing_rows == 6
    assert result.report
def test_unseen_description_evaluation_returns_valid_metrics(
    sample_transactions,
):
    result = evaluate_on_unseen_descriptions(sample_transactions)

    assert 0.0 <= result.accuracy <= 1.0
    assert result.training_rows > 0
    assert result.testing_rows > 0
    assert result.training_rows + result.testing_rows == 6