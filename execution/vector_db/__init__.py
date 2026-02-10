"""
Vector Database Subsystem for the GhostWriter Knowledge Layer.

Provides PostgreSQL + pgvector storage for document embeddings,
semantic search, and knowledge chunk management. Uses SQLAlchemy ORM
with a dedicated Base (separate from the SQLite content database).

Usage:
    from execution.vector_db import (
        Base, Document, KnowledgeChunk, IngestionLog,
        get_engine, get_session, init_db,
        EmbeddingClient, TokenTracker,
    )
"""

from execution.vector_db.models import Base, Document, KnowledgeChunk, IngestionLog
from execution.vector_db.connection import get_engine, get_session, init_db
from execution.vector_db.embeddings import (
    EmbeddingClient,
    TokenBudgetExceeded,
    BatchEmbeddingFailed,
    EmbeddingError,
)
from execution.vector_db.token_tracking import TokenTracker

__all__ = [
    "Base",
    "Document",
    "KnowledgeChunk",
    "IngestionLog",
    "get_engine",
    "get_session",
    "init_db",
    "EmbeddingClient",
    "TokenBudgetExceeded",
    "BatchEmbeddingFailed",
    "EmbeddingError",
    "TokenTracker",
]
