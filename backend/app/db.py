from __future__ import annotations

from collections.abc import Generator
from typing import Any

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .core.settings import settings


class Base(DeclarativeBase):
    """Base class for ORM models."""


DATABASE_URL = f"sqlite:///{settings.database_path}"

engine = create_engine(
    DATABASE_URL,
    connect_args={
        "check_same_thread": False,
        "timeout": 60,
    },
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragmas(dbapi_connection: Any, _: Any) -> None:
    """Tune SQLite for concurrent read/write workloads used by FastAPI handlers."""

    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA busy_timeout=60000")
    cursor.close()

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



def _migrate_users_table() -> None:
    """Migrate admin_users table to users and add role column if necessary."""
    with engine.begin() as connection:
        # Check if admin_users exists
        table_info = connection.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='admin_users'")).fetchone()
        if table_info:
            # Rename table
            connection.execute(text("ALTER TABLE admin_users RENAME TO users"))
            
        # Check if users table exists and if it needs the role column
        table_info = connection.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")).fetchone()
        if table_info:
            columns_info = connection.execute(text("PRAGMA table_info(users)")).fetchall()
            existing_columns = {str(row[1]) for row in columns_info}
            if "role" not in existing_columns:
                connection.execute(text("ALTER TABLE users ADD COLUMN role VARCHAR(20) DEFAULT 'user' NOT NULL"))
                # Existing users in the old admin_users table were admins
                connection.execute(text("UPDATE users SET role = 'admin'"))

def init_db() -> None:
    """Create all configured database tables and seed default admin if needed."""

    from . import models  # noqa: F401

    _migrate_users_table()
    Base.metadata.create_all(bind=engine)
    _seed_admin()


def _seed_admin() -> None:
    """Create default admin account on first run if none exists."""

    from .models import User
    from .core.security import hash_password

    db = SessionLocal()
    try:
        exists = db.query(User).first()
        if exists:
            return

        admin = User(
            username=settings.admin_default_username,
            hashed_password=hash_password(settings.admin_default_password),
            role="admin",
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
