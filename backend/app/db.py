"""Database engine and session management (SQLite for local dev)."""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from src.config.settings import get_settings


def _get_db_url() -> str:
    settings = get_settings()
    url = settings.database_url
    # Ensure directory exists for SQLite files
    if url.startswith("sqlite:///"):
        db_path = url.replace("sqlite:///", "")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return url


engine = create_engine(
    _get_db_url(),
    connect_args={"check_same_thread": False},  # SQLite only
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency that yields a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _add_missing_columns() -> list[str]:
    """Additive, non-destructive migration for databases created by an older version.

    ``create_all`` never alters existing tables, so columns added to the ORM models later
    would otherwise raise "no such column" against a pre-existing SQLite file.  Only nullable
    columns are added; nothing is dropped or rewritten.
    """
    from sqlalchemy import inspect, text

    added: list[str] = []
    insp = inspect(engine)
    for table in Base.metadata.sorted_tables:
        if not insp.has_table(table.name):
            continue
        existing = {c["name"] for c in insp.get_columns(table.name)}
        for col in table.columns:
            if col.name in existing:
                continue
            if not col.nullable or col.primary_key:
                raise RuntimeError(
                    f"Cannot auto-migrate: new column {table.name}.{col.name} is not nullable"
                )
            ddl_type = col.type.compile(dialect=engine.dialect)
            with engine.begin() as conn:
                conn.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" {ddl_type}'))
            added.append(f"{table.name}.{col.name}")
    return added


def init_db() -> None:
    """Create all tables and add columns introduced since the DB file was created."""
    from backend.app import models  # noqa: F401 — import to register tables
    Base.metadata.create_all(bind=engine)
    added = _add_missing_columns()
    if added:
        import logging
        logging.getLogger(__name__).info("Database migrated; added columns: %s", ", ".join(added))
