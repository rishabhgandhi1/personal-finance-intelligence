from pathlib import Path

import mlflow

from src.ml.expense_categorizer import (
    CategorizationEvaluation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MLFLOW_DATABASE = PROJECT_ROOT / "mlflow.db"

MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DATABASE.as_posix()}"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

EXPERIMENT_NAME = "Expense Categorization"


def log_categorization_experiment(
    evaluation: CategorizationEvaluation,
    evaluation_method: str,
) -> str:
    """
    Log expense categorization evaluation results to MLflow.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(
        run_name=evaluation_method.replace("_", " ").title()
    ) as run:

        mlflow.log_params({
            "algorithm": "LogisticRegression",
            "vectorizer": "TfidfVectorizer",
            "ngram_range": "(1, 2)",
            "max_iter": 1000,
            "random_state": 42,
            "evaluation_method": evaluation_method,
        })

        mlflow.log_metrics({
            "accuracy": evaluation.accuracy,
            "training_rows": evaluation.training_rows,
            "testing_rows": evaluation.testing_rows,
        })

        mlflow.log_dict(
            evaluation.report,
            "classification_report.json",
        )

        return run.info.run_id