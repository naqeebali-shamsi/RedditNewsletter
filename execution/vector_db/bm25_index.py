"""
BM25 Sparse Retrieval Index.

Implements fast BM25-based lexical search using the bm25s library
(500x faster than rank-bm25). Provides keyword matching for exact terms,
technical jargon, proper nouns, and acronyms that semantic search might miss.

Usage:
    from execution.vector_db.bm25_index import BM25Index

    # Build index from database
    index = BM25Index()
    count = index.build_index(tenant_id="default")

    # Search for keywords
    results = index.search("PostgreSQL vector indexing", top_k=50)
    # Returns: [{"id": chunk_id, "bm25_score": float, "rank": int}, ...]
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import bm25s

from execution.config import config
from execution.vector_db.connection import get_session
from execution.vector_db.models import KnowledgeChunk

logger = logging.getLogger(__name__)


class BM25Index:
    """BM25 sparse retrieval index for keyword-based search.

    Uses bm25s library for fast BM25 indexing and retrieval. Index is
    persisted to disk and loaded lazily. Supports multi-tenant isolation.
    """

    def __init__(self, index_dir: Optional[str] = None):
        """Initialize BM25 index with storage directory.

        Args:
            index_dir: Path to index storage directory. Defaults to
                .tmp/bm25_index/ from config.
        """
        if index_dir is None:
            self.index_dir = config.paths.TEMP_DIR / "bm25_index"
        else:
            self.index_dir = Path(index_dir)

        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.retriever = None
        self.chunk_ids = []
        self.is_loaded = False

    def build_index(self, tenant_id: str = "default") -> int:
        """Build BM25 index from all knowledge_chunks for given tenant.

        Queries database for all chunks, tokenizes with English stopwords,
        creates BM25 index, and saves to disk.

        Args:
            tenant_id: Tenant identifier for multi-tenant isolation.

        Returns:
            Number of documents indexed.
        """
        with get_session() as session:
            # Query all chunks for tenant with content
            chunks = session.query(KnowledgeChunk).filter(
                KnowledgeChunk.tenant_id == tenant_id,
                KnowledgeChunk.content.isnot(None)
            ).all()

            if not chunks:
                logger.warning(f"No chunks found for tenant {tenant_id}")
                return 0

            # Extract corpus and chunk IDs
            corpus = [chunk.content for chunk in chunks]
            self.chunk_ids = [chunk.id for chunk in chunks]

            # Tokenize corpus with English stopwords
            logger.info(f"Tokenizing {len(corpus)} documents for BM25 indexing...")
            corpus_tokens = bm25s.tokenize(corpus, stopwords="en")

            # Create and index BM25 retriever
            self.retriever = bm25s.BM25()
            self.retriever.index(corpus_tokens)

            # Save index to disk
            self.retriever.save(str(self.index_dir), corpus=corpus_tokens)

            # Save chunk IDs to JSON
            chunk_ids_path = self.index_dir / "chunk_ids.json"
            with open(chunk_ids_path, "w", encoding="utf-8") as f:
                json.dump(self.chunk_ids, f)

            self.is_loaded = True
            logger.info(f"BM25 index built: {len(chunks)} chunks for tenant {tenant_id}")

            return len(chunks)

    def load_index(self) -> bool:
        """Load saved BM25 index from disk.

        Returns:
            True if index loaded successfully, False if index not found.
        """
        try:
            # Load BM25 retriever
            self.retriever = bm25s.BM25.load(str(self.index_dir), load_corpus=False)

            # Load chunk IDs from JSON
            chunk_ids_path = self.index_dir / "chunk_ids.json"
            with open(chunk_ids_path, "r", encoding="utf-8") as f:
                self.chunk_ids = json.load(f)

            self.is_loaded = True
            logger.info(f"BM25 index loaded: {len(self.chunk_ids)} documents")
            return True

        except FileNotFoundError:
            logger.warning(f"BM25 index not found at {self.index_dir}")
            return False

    def search(self, query: str, top_k: int = 50) -> List[Dict]:
        """Search BM25 index for keyword matches.

        Args:
            query: Search query string.
            top_k: Number of results to return (default 50).

        Returns:
            List of dicts with keys: id (chunk_id), bm25_score (float), rank (int).
            Returns empty list if index not loaded or no matches.
        """
        # Lazy load index if not yet loaded
        if not self.is_loaded:
            if not self.load_index():
                logger.warning("Cannot search - BM25 index not available")
                return []

        if not self.chunk_ids:
            logger.warning("Cannot search - no documents in index")
            return []

        # Tokenize query
        query_tokens = bm25s.tokenize(query, stopwords="en")

        # Cap top_k to corpus size (bm25s requirement)
        k = min(top_k, len(self.chunk_ids))

        # Retrieve results
        results, scores = self.retriever.retrieve(query_tokens, k=k)

        # Build result list
        # results and scores are 2D arrays (one row per query)
        # We only have one query, so index into [0]
        result_list = []
        for i, (doc_idx, score) in enumerate(zip(results[0], scores[0])):
            # Filter out zero-score results (no match)
            if score > 0:
                result_list.append({
                    "id": self.chunk_ids[doc_idx],
                    "bm25_score": float(score),
                    "rank": i + 1
                })

        logger.debug(f"BM25 search returned {len(result_list)} results for query: {query[:50]}")
        return result_list

    def needs_rebuild(self, tenant_id: str = "default") -> bool:
        """Check if index is stale and needs rebuild.

        Compares chunk count in database vs saved index size.

        Args:
            tenant_id: Tenant identifier.

        Returns:
            True if index doesn't exist or chunk count differs.
        """
        # Check if index exists
        if not (self.index_dir / "chunk_ids.json").exists():
            return True

        # Load chunk IDs if not already loaded
        if not self.chunk_ids:
            try:
                chunk_ids_path = self.index_dir / "chunk_ids.json"
                with open(chunk_ids_path, "r", encoding="utf-8") as f:
                    self.chunk_ids = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                return True

        # Count chunks in database
        with get_session() as session:
            db_count = session.query(KnowledgeChunk).filter(
                KnowledgeChunk.tenant_id == tenant_id,
                KnowledgeChunk.content.isnot(None)
            ).count()

        index_count = len(self.chunk_ids)

        if db_count != index_count:
            logger.info(
                f"BM25 index stale: DB has {db_count} chunks, "
                f"index has {index_count}"
            )
            return True

        return False

    def clear_index(self) -> None:
        """Delete saved index files from disk and reset state.

        Used for testing and manual index management.
        """
        import shutil

        if self.index_dir.exists():
            shutil.rmtree(self.index_dir)
            logger.info(f"Cleared BM25 index at {self.index_dir}")

        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.retriever = None
        self.chunk_ids = []
        self.is_loaded = False
