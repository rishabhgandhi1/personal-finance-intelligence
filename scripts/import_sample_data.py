from pathlib import Path

from src.database.connection import SessionLocal
from src.database.init_db import init_db
from src.ingestion.transaction_loader import ingest_transactions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = PROJECT_ROOT / "data" / "sample" / "transactions.csv"


def main() -> None:
    """Initialize the database and import the sample transactions."""
    if not SAMPLE_CSV.exists():
        raise FileNotFoundError(
            f"Sample transaction file not found: {SAMPLE_CSV}"
        )

    init_db()

    session = SessionLocal()

    try:
        result = ingest_transactions(
            source=SAMPLE_CSV,
            session=session,
            filename=SAMPLE_CSV.name,
        )

        print("\nSample data import completed.")
        print(f"Batch ID: {result.batch_id}")
        print(f"Total rows: {result.total_rows}")
        print(f"Accepted rows: {result.accepted_rows}")
        print(f"Rejected rows: {result.rejected_rows}")
        print(f"Duplicate rows: {result.duplicate_rows}")

        if result.errors:
            print("\nImport issues:")
            for error in result.errors[:10]:
                print(error)

    finally:
        session.close()


if __name__ == "__main__":
    main()