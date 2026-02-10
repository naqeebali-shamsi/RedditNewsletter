"""
Vector Database Subsystem for the GhostWriter Knowledge Layer.

Provides PostgreSQL + pgvector storage for document embeddings,
semantic search, and knowledge chunk management. Uses SQLAlchemy ORM
with a dedicated Base (separate from the SQLite content database).

Phase 1 - Vector DB Foundation:
    from execution.vector_db import (
        Base, Document, KnowledgeChunk, IngestionLog,
        get_engine, get_session, init_db,
        EmbeddingClient, TokenTracker,
        SemanticChunker, chunk_content, Chunk,
        AutoTagger, auto_tag, TagResult,
        IngestionPipeline, ingest_document, ingest_batch,
        create_hnsw_index, semantic_search, get_index_stats,
    )

Phase 2 - Retrieval Tools:
    from execution.vector_db import (
        MetadataFilter, build_filters,
        RecencyScorer,
        CitationExtractor, Citation,
        BM25Index,
        CrossEncoderReranker,
        HybridRetriever, RetrievalResult, hybrid_search,
    )
"""

# Phase 1: Vector DB Foundation
from execution.vector_db.models import Base, Document, KnowledgeChunk, IngestionLog
from execution.vector_db.connection import get_engine, get_session, init_db
from execution.vector_db.embeddings import (
    EmbeddingClient,
    embed_texts,
    batch_embed_texts,
    TokenBudgetExceeded,
    BatchEmbeddingFailed,
    EmbeddingError,
)
from execution.vector_db.token_tracking import TokenTracker
from execution.vector_db.chunking import SemanticChunker, chunk_content, Chunk
from execution.vector_db.tagging import AutoTagger, auto_tag, TagResult
from execution.vector_db.ingestion import IngestionPipeline, ingest_document, ingest_batch
from execution.vector_db.indexing import create_hnsw_index, semantic_search, get_index_stats

# Phase 2: Retrieval Tools
from execution.vector_db.metadata_filters import MetadataFilter, build_filters
from execution.vector_db.recency_scoring import RecencyScorer
from execution.vector_db.citations import CitationExtractor, Citation
from execution.vector_db.bm25_index import BM25Index
from execution.vector_db.reranking import CrossEncoderReranker
from execution.vector_db.retrieval import HybridRetriever, RetrievalResult, hybrid_search

__all__ = [
    # Phase 1 exports
    "Base",
    "Document",
    "KnowledgeChunk",
    "IngestionLog",
    "get_engine",
    "get_session",
    "init_db",
    "EmbeddingClient",
    "embed_texts",
    "batch_embed_texts",
    "TokenBudgetExceeded",
    "BatchEmbeddingFailed",
    "EmbeddingError",
    "TokenTracker",
    "SemanticChunker",
    "chunk_content",
    "Chunk",
    "AutoTagger",
    "auto_tag",
    "TagResult",
    "IngestionPipeline",
    "ingest_document",
    "ingest_batch",
    "create_hnsw_index",
    "semantic_search",
    "get_index_stats",
    # Phase 2 exports
    "MetadataFilter",
    "build_filters",
    "RecencyScorer",
    "CitationExtractor",
    "Citation",
    "BM25Index",
    "CrossEncoderReranker",
    "HybridRetriever",
    "RetrievalResult",
    "hybrid_search",
]
