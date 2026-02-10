"""
Database Connection Management for the Vector Knowledge Base.

Provides engine creation, session management, and schema initialization
for the PostgreSQL + pgvector database. Uses a singleton engine pattern
consistent with execution/sources/database.py.

Usage:
    from execution.vector_db.connection import get_engine, get_session, init_db

    # Initialize schema (idempotent)
    init_db()

    # Use a session
    with get_session() as session:
        docs = session.query(Document).all()
"""

from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from execution.vector_db.models import Base

# Module-level singleton
_engine: Optional[Engine] = None
_session_factory: Optional[sessionmaker] = None


def get_engine(db_url: Optional[str] = None) -> Engine:
    """Get or create the singleton SQLAlchemy engine for the vector DB.

    Reads the database URL from VectorDBConfig if not provided.
    Uses pool_pre_ping for connection health checks.

    Args:
        db_url: Override database URL. If None, reads from config.

    Returns:
        SQLAlchemy Engine instance.
    """
    global _engine
    if _engine is not None:
        return _engine

    if db_url is None:
        from execution.config import config
        db_url = config.vector_db.DATABASE_URL

    _engine = create_engine(
        db_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )
    return _engine


@contextmanager
def get_session(db_url: Optional[str] = None) -> Generator[Session, None, None]:
    """Provide a transactional session scope.

    Yields a SQLAlchemy Session bound to the vector DB engine.
    Commits on clean exit, rolls back on exception.

    Args:
        db_url: Override database URL passed to get_engine.

    Yields:
        SQLAlchemy Session instance.
    """
    global _session_factory
    engine = get_engine(db_url)

    if _session_factory is None:
        _session_factory = sessionmaker(bind=engine)

    session = _session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(db_url: Optional[str] = None) -> None:
    """Create all tables defined in the vector DB models.

    Idempotent -- safe to call multiple times. Uses SQLAlchemy's
    create_all which only creates tables that don't already exist.

    Args:
        db_url: Override database URL passed to get_engine.
    """
    engine = get_engine(db_url)
    Base.metadata.create_all(engine)


def reset_engine() -> None:
    """Dispose and reset the engine singleton.

    Intended for testing to ensure a fresh connection pool.
    """
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
        _engine = None
    _session_factory = None
