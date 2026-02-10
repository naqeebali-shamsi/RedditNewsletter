"""
Hybrid Retrieval Orchestrator - Phase 2 Capstone Module.

Combines all Phase 2 retrieval modules into a single clean interface:
- BM25 sparse keyword search
- Vector semantic search with metadata filtering
- RRF (Reciprocal Rank Fusion) for merging results
- Recency scoring for trend-aware ranking
- CrossEncoder reranking for precision
- Citation extraction for source attribution

Usage:
    from execution.vector_db.retrieval import hybrid_search

    results = hybrid_search(
        "How does pgvector handle vector indexing?",
        top_k=10,
        source_types=["rss", "paper"],
        recency_months=3,
    )

    for result in results:
        print(f"{result.title} (score={result.fused_score:.3f})")
        if result.citations:
            print(f"  Citations: {len(result.citations)}")
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.sql import and_

from execution.config import config
from execution.vector_db.bm25_index import BM25Index
from execution.vector_db.citations import CitationExtractor
from execution.vector_db.connection import get_session
from execution.vector_db.embeddings import EmbeddingClient
from execution.vector_db.indexing import semantic_search
from execution.vector_db.metadata_filters import build_filters
from execution.vector_db.models import Document, KnowledgeChunk
from execution.vector_db.recency_scoring import RecencyScorer
from execution.vector_db.reranking import CrossEncoderReranker

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from hybrid retrieval with all scoring and metadata."""

    id: int  # chunk ID
    content: str  # chunk text
    title: Optional[str] = None  # document title
    url: Optional[str] = None  # document URL
    source_type: Optional[str] = None  # email, rss, paper
    date_published: Optional[datetime] = None  # document date
    topic_tags: Optional[list] = None  # chunk topic tags
    entities: Optional[list] = None  # chunk entities

    # Scoring
    rrf_score: float = 0.0  # RRF fusion score (0-1 normalized)
    rerank_score: Optional[float] = None  # CrossEncoder score (if reranking enabled)
    recency_score: Optional[float] = None  # Recency decay score (if trend query)
    fused_score: Optional[float] = None  # Final fused score (semantic + recency)

    # Citations (populated on demand)
    citations: Optional[list] = field(default=None)  # List[Citation] from citations module


class HybridRetriever:
    """Orchestrates hybrid retrieval: BM25 + Vector -> RRF -> Recency -> Rerank -> Citations."""

    def __init__(self, config_override: Optional[object] = None):
        """Initialize hybrid retriever with config.

        Args:
            config_override: Optional RetrievalConfig override (uses global config if None).
        """
        self.config = config_override if config_override else config.retrieval
        self.bm25_index = BM25Index()
        self.recency_scorer = RecencyScorer(half_life_days=self.config.RECENCY_HALF_LIFE_DAYS)
        self._reranker = None  # Lazy load

    def _ensure_reranker(self):
        """Lazy load CrossEncoder reranker."""
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(
                model_profile=self.config.RERANK_MODEL_PROFILE,
                timeout_ms=self.config.RERANK_TIMEOUT_MS,
            )

    def _vector_search(
        self,
        query: str,
        tenant_id: str,
        top_k: int,
        filters: Optional[List] = None,
    ) -> List[Dict]:
        """Vector semantic search with optional metadata filters.

        If filters provided, applies them at SQL level (WHERE clause) before vector search.
        If no filters, delegates to existing semantic_search().

        Args:
            query: Query text to embed and search.
            tenant_id: Tenant ID for multi-tenant isolation.
            top_k: Number of results to return.
            filters: Optional list of SQLAlchemy filter conditions from build_filters().

        Returns:
            List of dicts with keys: id, content, title, url, source_type, date_published,
            topic_tags, entities, distance, rank.
        """
        if not filters:
            # No filters - use existing semantic_search() for backward compatibility
            results = semantic_search(query, tenant_id=tenant_id, limit=top_k)
            # Add rank (1-based) and convert distance to rank for consistency
            for i, result in enumerate(results):
                result["rank"] = i + 1
                result["date_published"] = None  # semantic_search doesn't return date

            # Enrich with date_published via follow-up query
            if results:
                chunk_ids = [r["id"] for r in results]
                with get_session() as session:
                    stmt = (
                        select(
                            KnowledgeChunk.id,
                            Document.date_published,
                        )
                        .join(Document)
                        .where(KnowledgeChunk.id.in_(chunk_ids))
                    )
                    rows = session.execute(stmt).fetchall()
                    date_map = {row[0]: row[1] for row in rows}

                for result in results:
                    result["date_published"] = date_map.get(result["id"])

            return results

        # Filters provided - custom SQL with WHERE clause filtering
        start = time.time()

        # Get query embedding
        embed_client = EmbeddingClient()
        query_embedding = embed_client.embed_text(query)

        # Build SQLAlchemy query with filters
        with get_session() as session:
            stmt = (
                select(
                    KnowledgeChunk.id,
                    KnowledgeChunk.content,
                    KnowledgeChunk.topic_tags,
                    KnowledgeChunk.entities,
                    Document.title,
                    Document.source_type,
                    Document.url,
                    Document.date_published,
                    KnowledgeChunk.embedding.cosine_distance(query_embedding).label("distance"),
                )
                .join(Document)
                .where(and_(*filters))  # Apply all metadata filters
                .order_by(KnowledgeChunk.embedding.cosine_distance(query_embedding))
                .limit(top_k)
            )

            rows = session.execute(stmt).fetchall()

        # Convert to dict format
        results = []
        for i, row in enumerate(rows):
            results.append({
                "id": row[0],
                "content": row[1],
                "topic_tags": row[2],
                "entities": row[3],
                "title": row[4],
                "source_type": row[5],
                "url": row[6],
                "date_published": row[7],
                "distance": float(row[8]),
                "rank": i + 1,
            })

        elapsed = time.time() - start
        logger.debug(
            "Vector search with filters: %d results in %.1fms",
            len(results),
            elapsed * 1000,
        )

        return results

    def _bm25_search(self, query: str, tenant_id: str, top_k: int) -> List[Dict]:
        """BM25 keyword search with graceful degradation.

        Args:
            query: Query text for BM25 search.
            tenant_id: Tenant ID (passed for future multi-tenant BM25 support).
            top_k: Number of results to return.

        Returns:
            List of dicts with keys: id, bm25_score, rank.
            Returns empty list if BM25 index not available.
        """
        start = time.time()

        try:
            # Try loading index if not already loaded
            if not self.bm25_index.is_loaded:
                loaded = self.bm25_index.load_index()
                if not loaded:
                    logger.warning(
                        "BM25 index not available - falling back to vector-only search"
                    )
                    return []

            results = self.bm25_index.search(query, top_k=top_k)
            elapsed = time.time() - start
            logger.debug("BM25 search: %d results in %.1fms", len(results), elapsed * 1000)
            return results

        except Exception as exc:
            logger.error("BM25 search failed: %s - falling back to vector-only", exc)
            return []

    def _enrich_bm25_results(self, bm25_results: List[Dict], tenant_id: str) -> List[Dict]:
        """Enrich BM25 results with metadata from database.

        BM25 results only have {id, bm25_score, rank}. This fetches content, title,
        url, source_type, date_published, topic_tags, entities in a single batch query.

        Args:
            bm25_results: List of BM25 results (only have id, bm25_score, rank).
            tenant_id: Tenant ID for filtering.

        Returns:
            Enriched results with all metadata fields.
        """
        if not bm25_results:
            return []

        start = time.time()
        chunk_ids = [r["id"] for r in bm25_results]

        with get_session() as session:
            stmt = (
                select(
                    KnowledgeChunk.id,
                    KnowledgeChunk.content,
                    KnowledgeChunk.topic_tags,
                    KnowledgeChunk.entities,
                    Document.title,
                    Document.source_type,
                    Document.url,
                    Document.date_published,
                )
                .join(Document)
                .where(
                    and_(
                        KnowledgeChunk.id.in_(chunk_ids),
                        KnowledgeChunk.tenant_id == tenant_id,
                    )
                )
            )
            rows = session.execute(stmt).fetchall()

        # Create lookup map
        metadata_map = {}
        for row in rows:
            metadata_map[row[0]] = {
                "content": row[1],
                "topic_tags": row[2],
                "entities": row[3],
                "title": row[4],
                "source_type": row[5],
                "url": row[6],
                "date_published": row[7],
            }

        # Merge with BM25 results
        enriched = []
        for bm25_result in bm25_results:
            chunk_id = bm25_result["id"]
            if chunk_id in metadata_map:
                enriched.append({**bm25_result, **metadata_map[chunk_id]})

        elapsed = time.time() - start
        logger.debug(
            "Enriched %d BM25 results in %.1fms",
            len(enriched),
            elapsed * 1000,
        )

        return enriched

    def _rrf_fusion(
        self,
        bm25_results: List[Dict],
        vector_results: List[Dict],
        k: int = 60,
        bm25_weight: float = 1.0,
        vector_weight: float = 1.0,
    ) -> List[Dict]:
        """Reciprocal Rank Fusion to merge BM25 and vector results.

        Formula: contribution = weight / (k + rank)
        Accumulates scores by chunk ID, then normalizes to 0-1 range.

        Args:
            bm25_results: BM25 results with 'rank' key.
            vector_results: Vector results with 'rank' key.
            k: RRF constant (higher = less emphasis on top ranks).
            bm25_weight: Weight multiplier for BM25 contributions.
            vector_weight: Weight multiplier for vector contributions.

        Returns:
            Fused results sorted by RRF score descending, with 'rrf_score' key (0-1 normalized).
        """
        start = time.time()

        # Accumulate scores by chunk ID
        scores = {}
        metadata = {}  # Prefer vector results for metadata (more complete)

        # Add vector contributions
        for result in vector_results:
            chunk_id = result["id"]
            rank = result["rank"]
            contribution = vector_weight / (k + rank)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + contribution
            metadata[chunk_id] = result  # Vector has more metadata

        # Add BM25 contributions
        for result in bm25_results:
            chunk_id = result["id"]
            rank = result["rank"]
            contribution = bm25_weight / (k + rank)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + contribution
            # Only update metadata if not already set (prefer vector)
            if chunk_id not in metadata:
                metadata[chunk_id] = result

        # Normalize scores to 0-1 range
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                for chunk_id in scores:
                    scores[chunk_id] /= max_score

        # Build fused results
        fused = []
        for chunk_id, rrf_score in scores.items():
            result = metadata[chunk_id].copy()
            result["rrf_score"] = rrf_score
            fused.append(result)

        # Sort by RRF score descending
        fused.sort(key=lambda x: x["rrf_score"], reverse=True)

        elapsed = time.time() - start
        logger.debug(
            "RRF fusion: %d unique results from %d BM25 + %d vector in %.1fms",
            len(fused),
            len(bm25_results),
            len(vector_results),
            elapsed * 1000,
        )

        return fused

    def search(
        self,
        query: str,
        tenant_id: str = "default",
        top_k: Optional[int] = None,
        source_types: Optional[List[str]] = None,
        topic_tags: Optional[List[str]] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        recency_months: Optional[int] = None,
        rerank: Optional[bool] = None,
        include_citations: bool = False,
    ) -> List[RetrievalResult]:
        """Main hybrid retrieval entry point.

        Pipeline:
        1. Build metadata filters
        2. Vector search (with filters)
        3. BM25 search
        4. Enrich BM25 results
        5. RRF fusion
        6. Recency scoring (if trend query)
        7. CrossEncoder reranking (if enabled)
        8. Citation extraction (if requested)

        Args:
            query: Query text.
            tenant_id: Tenant ID for multi-tenant isolation.
            top_k: Number of final results to return (uses config default if None).
            source_types: Filter by source types (e.g., ["rss", "paper"]).
            topic_tags: Filter by topic tags.
            date_range: Filter by date range (start, end).
            recency_months: Filter by last N months.
            rerank: Enable/disable reranking (uses config default if None).
            include_citations: Extract sentence-level citations.

        Returns:
            List of RetrievalResult objects sorted by relevance.
        """
        pipeline_start = time.time()
        top_k = top_k or self.config.DEFAULT_TOP_K
        rerank = rerank if rerank is not None else self.config.RERANK_ENABLED

        # Step 1: Build metadata filters
        filters = build_filters(
            tenant_id=tenant_id,
            date_range=date_range,
            source_types=source_types,
            topic_tags=topic_tags,
            recency_months=recency_months,
        )

        # Step 2: Vector search with filters
        t0 = time.time()
        vector_results = self._vector_search(
            query,
            tenant_id,
            self.config.VECTOR_TOP_K,
            filters,
        )
        vector_time = (time.time() - t0) * 1000

        # Step 3: BM25 search
        t0 = time.time()
        bm25_results = self._bm25_search(query, tenant_id, self.config.BM25_TOP_K)
        bm25_time = (time.time() - t0) * 1000

        # Step 4: Enrich BM25 results
        if bm25_results:
            bm25_results = self._enrich_bm25_results(bm25_results, tenant_id)

        # Step 5: RRF fusion
        t0 = time.time()
        fused_results = self._rrf_fusion(
            bm25_results,
            vector_results,
            k=self.config.RRF_K,
            bm25_weight=self.config.BM25_WEIGHT,
            vector_weight=self.config.VECTOR_WEIGHT,
        )
        fusion_time = (time.time() - t0) * 1000

        # Limit to fusion top-K before expensive operations
        fused_results = fused_results[:self.config.FUSION_TOP_K]

        # Step 6: Recency scoring
        t0 = time.time()
        fused_results = self.recency_scorer.score_results(
            query,
            fused_results,
            semantic_weight=self.config.RECENCY_SEMANTIC_WEIGHT,
            recency_weight=self.config.RECENCY_WEIGHT,
        )
        recency_time = (time.time() - t0) * 1000

        # Step 7: CrossEncoder reranking (optional)
        rerank_time = 0
        if rerank:
            t0 = time.time()
            self._ensure_reranker()
            try:
                fused_results = self._reranker.rerank(
                    query,
                    fused_results,
                    top_k=self.config.RERANK_TOP_K,
                )
            except Exception as exc:
                logger.error("Reranking failed: %s - continuing with RRF results", exc)
            rerank_time = (time.time() - t0) * 1000

        # Step 8: Limit to final top-K
        fused_results = fused_results[:top_k]

        # Step 9: Convert to RetrievalResult objects
        results = []
        for r in fused_results:
            results.append(
                RetrievalResult(
                    id=r["id"],
                    content=r["content"],
                    title=r.get("title"),
                    url=r.get("url"),
                    source_type=r.get("source_type"),
                    date_published=r.get("date_published"),
                    topic_tags=r.get("topic_tags"),
                    entities=r.get("entities"),
                    rrf_score=r.get("rrf_score", 0.0),
                    rerank_score=r.get("rerank_score"),
                    recency_score=r.get("recency_score"),
                    fused_score=r.get("fused_score"),
                )
            )

        # Step 10: Extract citations (if requested)
        citation_time = 0
        if include_citations:
            t0 = time.time()
            extractor = CitationExtractor()
            for result in results:
                # Create mock chunk dict for citation extractor
                chunk_dict = {
                    "id": result.id,
                    "content": result.content,
                    "title": result.title,
                    "url": result.url,
                    "date_published": result.date_published,
                }
                result.citations = extractor.extract_citations(chunk_dict)
            citation_time = (time.time() - t0) * 1000

        # Log timing
        total_time = (time.time() - pipeline_start) * 1000
        logger.info(
            "Hybrid search: vector=%.1fms, bm25=%.1fms, fusion=%.1fms, recency=%.1fms, rerank=%.1fms, citations=%.1fms, total=%.1fms",
            vector_time,
            bm25_time,
            fusion_time,
            recency_time,
            rerank_time,
            citation_time,
            total_time,
        )

        return results

    def ensure_bm25_index(self, tenant_id: str = "default") -> int:
        """Build/rebuild BM25 index for a tenant.

        Args:
            tenant_id: Tenant ID to build index for.

        Returns:
            Number of chunks indexed.
        """
        return self.bm25_index.build_index(tenant_id=tenant_id)


def hybrid_search(
    query: str,
    tenant_id: str = "default",
    top_k: int = 10,
    **kwargs
) -> List[RetrievalResult]:
    """Convenience function for one-shot hybrid search.

    Args:
        query: Query text.
        tenant_id: Tenant ID for multi-tenant isolation.
        top_k: Number of results to return.
        **kwargs: Additional arguments passed to HybridRetriever.search().

    Returns:
        List of RetrievalResult objects.

    Example:
        results = hybrid_search(
            "How does pgvector work?",
            top_k=5,
            source_types=["rss"],
            recency_months=3,
        )
    """
    retriever = HybridRetriever()
    return retriever.search(query, tenant_id=tenant_id, top_k=top_k, **kwargs)
