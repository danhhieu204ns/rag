from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, text
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


def _ensure_document_chunk_columns() -> None:
    """Apply lightweight sqlite migrations for newly added chunk source columns."""

    required_columns = {
        "source_page": "INTEGER",
        "source_kind": "VARCHAR(64)",
        "source_metadata_json": "TEXT",
    }

    with engine.begin() as connection:
        table_info = connection.execute(text("PRAGMA table_info(document_chunks)"))
        existing_columns = {str(row[1]) for row in table_info.fetchall()}

        for column_name, column_type in required_columns.items():
            if column_name in existing_columns:
                continue
            connection.execute(
                text(f"ALTER TABLE document_chunks ADD COLUMN {column_name} {column_type}")
            )



def get_db() -> Generator[Session, None, None]:
    """Yield one database session per request."""

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



def init_db() -> None:
    """Create all configured database tables and seed default admin if needed."""

    from . import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _seed_admin()


def _seed_admin() -> None:
    """Create default admin account on first run if none exists."""

    from .models import AdminUser
    from .core.security import hash_password

    db = SessionLocal()
    try:
        exists = db.query(AdminUser).first()
        if exists:
            return

        admin = AdminUser(
            username=settings.admin_default_username,
            hashed_password=hash_password(settings.admin_default_password),
            is_active=True,
        )
        db.add(admin)
        db.commit()
        print(
            f"[init_db] Default admin created — "
            f"username: {settings.admin_default_username}"
        )
    finally:
        db.close()
