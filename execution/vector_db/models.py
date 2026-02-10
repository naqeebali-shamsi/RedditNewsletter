"""
SQLAlchemy ORM Models for the Vector Knowledge Base.

Defines the schema for documents, embedded knowledge chunks, and
ingestion logs. Uses a dedicated DeclarativeBase separate from the
existing SQLite content database (execution/sources/database.py).

All tables include tenant_id for multi-tenant isolation.
"""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Declarative base for the vector knowledge base (PostgreSQL)."""
    pass


class Document(Base):
    """A source document (email, paper, RSS item) before chunking.

    Attributes:
        id: Auto-incrementing primary key.
        tenant_id: Tenant identifier for multi-tenant isolation.
        source_type: Origin type (email, rss, paper, manual).
        source_id: Unique identifier within tenant + source_type.
        title: Document title.
        content: Full original text.
        url: Source URL if available.
        date_published: Original publication date.
        date_ingested: When the document was ingested.
        metadata_: Arbitrary JSON metadata (mapped to 'metadata' column).
        processing_status: Pipeline status (pending, chunked, embedded, failed).
        error_message: Error details if processing_status is 'failed'.
        chunks: Related KnowledgeChunk instances.
    """

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(50), nullable=False, default="default", index=True)
    source_type = Column(String(50), nullable=False)
    source_id = Column(String(255), nullable=False)
    title = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    url = Column(Text, nullable=True)
    date_published = Column(DateTime, nullable=True)
    date_ingested = Column(DateTime, default=datetime.utcnow)
    metadata_ = Column("metadata", JSONB, default=dict)
    processing_status = Column(String(20), default="pending")
    error_message = Column(Text, nullable=True)

    __table_args__ = (
        UniqueConstraint("tenant_id", "source_type", "source_id", name="uq_doc_tenant_source"),
    )

    chunks = relationship(
        "KnowledgeChunk",
        back_populates="document",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Document(id={self.id}, source_type={self.source_type!r}, title={self.title!r})>"


class KnowledgeChunk(Base):
    """An embedded chunk extracted from a document.

    Attributes:
        id: Auto-incrementing primary key.
        tenant_id: Tenant identifier for multi-tenant isolation.
        document_id: Foreign key to parent Document.
        chunk_index: Order within the parent document.
        content: Chunk text content.
        embedding: 1536-dimensional vector (null until embedded).
        topic_tags: AI-generated topic tags.
        entities: AI-extracted entities as [{type, value}].
        cited_by: Document IDs that cite this chunk.
        related_to: Related chunk IDs.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
        document: Parent Document relationship.
    """

    __tablename__ = "knowledge_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(50), nullable=False, default="default", index=True)
    document_id = Column(
        Integer,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(1536), nullable=True)
    topic_tags = Column(JSONB, default=list)
    entities = Column(JSONB, default=list)
    cited_by = Column(JSONB, default=list)
    related_to = Column(JSONB, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("ix_chunk_tenant_document", "tenant_id", "document_id"),
    )

    document = relationship("Document", back_populates="chunks")

    def __repr__(self) -> str:
        return f"<KnowledgeChunk(id={self.id}, document_id={self.document_id}, index={self.chunk_index})>"


class IngestionLog(Base):
    """Tracks ingestion runs for observability.

    Attributes:
        id: Auto-incrementing primary key.
        tenant_id: Tenant identifier.
        started_at: Run start timestamp.
        completed_at: Run completion timestamp.
        status: Run status (running, completed, failed, token_limit).
        documents_processed: Count of documents processed.
        chunks_created: Count of chunks created.
        embeddings_generated: Count of embeddings generated.
        tokens_used: Total embedding tokens consumed.
        errors: List of error details as JSON.
        source_type: Filter by source type (null = all sources).
    """

    __tablename__ = "ingestion_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(50), nullable=False, default="default")
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    status = Column(String(20), default="running")
    documents_processed = Column(Integer, default=0)
    chunks_created = Column(Integer, default=0)
    embeddings_generated = Column(Integer, default=0)
    tokens_used = Column(Integer, default=0)
    errors = Column(JSONB, default=list)
    source_type = Column(String(50), nullable=True)

    def __repr__(self) -> str:
        return f"<IngestionLog(id={self.id}, status={self.status!r}, docs={self.documents_processed})>"
