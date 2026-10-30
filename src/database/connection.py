import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_DATABASE_URL = (
    f"sqlite:///{(PROJECT_ROOT / 'data' / 'personal_finance.db').as_posix()}"
)

DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)

if DATABASE_URL.startswith("sqlite:///") and ":memory:" not in DATABASE_URL:
    database_path = DATABASE_URL.removeprefix("sqlite:///")

    if not database_path.startswith("/"):
        database_path = str(PROJECT_ROOT / database_path)

    Path(database_path).parent.mkdir(parents=True, exist_ok=True)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""


engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
    if DATABASE_URL.startswith("sqlite")
    else {},
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
)


def get_db():
    """Provide a database session and close it after use."""
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()
