"""
HNSW Index Management and Semantic Search.

Manages the pgvector HNSW index on the knowledge_chunks embedding column
and provides end-to-end semantic search (embed query -> vector search -> results).

Usage:
    from execution.vector_db.indexing import create_hnsw_index, semantic_search

    create_hnsw_index()  # One-time setup after bulk data load
    results = semantic_search("How does pgvector work?")
"""

import logging
import time
from typing import Optional

from sqlalchemy import text

from execution.vector_db.connection import get_engine
from execution.vector_db.embeddings import EmbeddingClient

logger = logging.getLogger(__name__)


def create_hnsw_index(m: int = 16, ef_construction: int = 64) -> bool:
    """Create an HNSW index on knowledge_chunks.embedding.

    Uses vector_cosine_ops for cosine distance similarity search.
    Sets maintenance_work_mem to 4GB for efficient index builds.

    Args:
        m: Maximum number of connections per node (higher = better recall, more memory).
        ef_construction: Size of dynamic candidate list during build (higher = better recall, slower build).

    Returns:
        True on success, False on failure.
    """
    engine = get_engine()
    start = time.time()

    try:
        with engine.connect() as conn:
            conn.execute(text("SET maintenance_work_mem = '4GB'"))
            conn.execute(text(
                f"CREATE INDEX IF NOT EXISTS idx_chunks_hnsw "
                f"ON knowledge_chunks "
                f"USING hnsw (embedding vector_cosine_ops) "
                f"WITH (m = {m}, ef_construction = {ef_construction})"
            ))
            conn.commit()

        elapsed = time.time() - start
        logger.info("HNSW index created in %.1fs (m=%d, ef_construction=%d)", elapsed, m, ef_construction)
        return True

    except Exception as exc:
        logger.error("Failed to create HNSW index: %s", exc)
        return False


def drop_hnsw_index() -> bool:
    """Drop the HNSW index if it exists.

    Used for rebuilding the index with different parameters.

    Returns:
        True on success, False on failure.
    """
    engine = get_engine()

    try:
        with engine.connect() as conn:
            conn.execute(text("DROP INDEX IF EXISTS idx_chunks_hnsw"))
            conn.commit()
        logger.info("HNSW index dropped")
        return True
    except Exception as exc:
        logger.error("Failed to drop HNSW index: %s", exc)
        return False


def semantic_search(
    query_text: str,
    tenant_id: Optional[str] = None,
    limit: int = 10,
    source_type: Optional[str] = None,
) -> list[dict]:
    """End-to-end semantic search: embed query, find nearest neighbors.

    Args:
        query_text: Natural language query to search for.
        tenant_id: Filter by tenant. Defaults to config default.
        limit: Maximum number of results.
        source_type: Optional filter by document source type.

    Returns:
        List of result dicts sorted by distance (closest first).
        Each dict: {id, content, distance, title, source_type, url, topic_tags, entities}.
    """
    if tenant_id is None:
        from execution.config import config
        tenant_id = config.vector_db.DEFAULT_TENANT

    # Embed the query
    client = EmbeddingClient()
    query_embedding = client.embed_text(query_text)

    # Build SQL query
    engine = get_engine()
    embedding_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

    sql = """
        SELECT kc.id, kc.content, kc.topic_tags, kc.entities,
               d.title, d.source_type, d.url,
               kc.embedding <=> :query_embedding AS distance
        FROM knowledge_chunks kc
        JOIN documents d ON kc.document_id = d.id
        WHERE kc.tenant_id = :tenant_id
          AND kc.embedding IS NOT NULL
    """

    params = {
        "query_embedding": embedding_str,
        "tenant_id": tenant_id,
    }

    if source_type:
        sql += " AND d.source_type = :source_type"
        params["source_type"] = source_type

    sql += " ORDER BY kc.embedding <=> :query_embedding LIMIT :limit"
    params["limit"] = limit

    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()

    results = []
    for row in rows:
        results.append({
            "id": row[0],
            "content": row[1],
            "topic_tags": row[2],
            "entities": row[3],
            "title": row[4],
            "source_type": row[5],
            "url": row[6],
            "distance": float(row[7]),
        })

    logger.info(
        "Semantic search for '%s': %d results (tenant=%s)",
        query_text[:50],
        len(results),
        tenant_id,
    )
    return results


def get_index_stats() -> dict:
    """Return statistics about the HNSW index.

    Returns:
        Dict with keys: exists (bool), size_bytes (int), rows_indexed (int).
    """
    engine = get_engine()

    try:
        with engine.connect() as conn:
            # Check if index exists
            result = conn.execute(text(
                "SELECT indexname FROM pg_indexes "
                "WHERE tablename = 'knowledge_chunks' AND indexname = 'idx_chunks_hnsw'"
            )).fetchone()

            if not result:
                return {"exists": False, "size_bytes": 0, "rows_indexed": 0}

            # Get index size
            size_result = conn.execute(text(
                "SELECT pg_relation_size('idx_chunks_hnsw')"
            )).fetchone()
            size_bytes = size_result[0] if size_result else 0

            # Get row count
            count_result = conn.execute(text(
                "SELECT COUNT(*) FROM knowledge_chunks WHERE embedding IS NOT NULL"
            )).fetchone()
            rows = count_result[0] if count_result else 0

            return {
                "exists": True,
                "size_bytes": size_bytes,
                "rows_indexed": rows,
            }

    except Exception as exc:
        logger.error("Failed to get index stats: %s", exc)
        return {"exists": False, "size_bytes": 0, "rows_indexed": 0}
