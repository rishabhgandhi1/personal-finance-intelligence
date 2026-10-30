import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.expense_categorizer import (
    evaluate_categorizer,
    evaluate_on_unseen_descriptions,
)
from src.ml.experiment_tracking import (
    log_categorization_experiment,
)

DATA_PATH = PROJECT_ROOT / "data" / "sample" / "transactions.csv"


def main():
    transactions = pd.read_csv(DATA_PATH)

    experiments = [
        (
            "random_stratified_split",
            evaluate_categorizer,
        ),
        (
            "unseen_description_split",
            evaluate_on_unseen_descriptions,
        ),
    ]

    for experiment_name, evaluation_function in experiments:
        evaluation = evaluation_function(transactions)

        run_id = log_categorization_experiment(
            evaluation=evaluation,
            evaluation_method=experiment_name,
        )

        print(f"\nExperiment: {experiment_name}")
        print(f"Accuracy: {evaluation.accuracy:.4f}")
        print(f"Training rows: {evaluation.training_rows}")
        print(f"Testing rows: {evaluation.testing_rows}")
        print(f"MLflow Run ID: {run_id}")


if __name__ == "__main__":
    main()