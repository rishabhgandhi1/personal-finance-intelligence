from src.database.connection import Base, engine
from src.database.models import ImportBatch, MonthlyForecast, Transaction  # noqa: F401
def init_db() -> None:
    """Create all database tables that do not already exist."""
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully.")
if __name__ == "__main__":
    init_db()