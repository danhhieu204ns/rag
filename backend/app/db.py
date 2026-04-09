from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .core.settings import settings


class Base(DeclarativeBase):
    """Base class for ORM models."""


DATABASE_URL = f"sqlite:///{settings.database_path}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)



def get_db() -> Generator[Session, None, None]:
    """Yield one database session per request."""

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



def init_db() -> None:
    """Create all configured database tables."""

    from . import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
