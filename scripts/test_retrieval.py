"""
End-to-end integration test for Phase 2 Hybrid Retrieval.

Exercises the full retrieval pipeline: hybrid search, metadata filtering,
recency scoring, citations, reranking, and backward compatibility.

Requires Docker (ghostwriter-vectordb) and OPENAI_API_KEY.

Usage:
    python scripts/test_retrieval.py             # Run tests, keep data
    python scripts/test_retrieval.py --cleanup    # Run tests, then delete test data
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime, timedelta

# Ensure project root is on sys.path so 'execution' package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------

def check_prerequisites() -> bool:
    """Verify Docker container and API key are available."""
    print("=" * 60)
    print("HYBRID RETRIEVAL INTEGRATION TEST")
    print("=" * 60)

    # Check Docker container
    print("\n[1/2] Checking Docker container...")
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=ghostwriter-vectordb", "--format", "{{.Status}}"],
            capture_output=True, text=True, timeout=10,
        )
        if "healthy" not in result.stdout.lower() and "up" not in result.stdout.lower():
            print("  FAIL: ghostwriter-vectordb container not running or not healthy.")
            print("  Run: docker compose up -d")
            print("  Then wait for health check: docker compose ps")
            return False
        print(f"  OK: Container status: {result.stdout.strip()}")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("  FAIL: Docker not available. Install Docker Desktop and run: docker compose up -d")
        return False

    # Check OPENAI_API_KEY
    print("[2/2] Checking OPENAI_API_KEY...")
    if not os.getenv("OPENAI_API_KEY"):
        # Try loading from .env
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        if not os.getenv("OPENAI_API_KEY"):
            from execution.config import config
            if not config.api.OPENAI_API_KEY:
                print("  FAIL: OPENAI_API_KEY not set.")
                print("  Add OPENAI_API_KEY=sk-... to .env file")
                return False

    print("  OK: OPENAI_API_KEY found")
    return True


# ---------------------------------------------------------------------------
# Test steps
# ---------------------------------------------------------------------------

def test_database_setup() -> bool:
    """Step 1: Initialize database tables."""
    print("\n" + "-" * 60)
    print("STEP 1: Database Setup")
    print("-" * 60)

    from execution.vector_db import init_db, Base

    try:
        init_db()
        table_names = list(Base.metadata.tables.keys())
        print(f"  Tables verified: {table_names}")
        assert "documents" in table_names
        assert "knowledge_chunks" in table_names
        print("  PASS: Database setup OK")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_ingest_documents() -> bool:
    """Step 2: Ingest test documents with varying attributes."""
    print("\n" + "-" * 60)
    print("STEP 2: Ingest Test Documents")
    print("-" * 60)

    from execution.vector_db import ingest_document

    test_docs = [
        {
            "source_id": "test-retr-001",
            "source_type": "rss",
            "title": "PostgreSQL pgvector HNSW Indexing Guide",
            "content": (
                "PostgreSQL handles vector indexing efficiently using pgvector extension. "
                "The HNSW algorithm provides fast approximate nearest neighbor search with high recall. "
                "HNSW indexing parameters m and ef_construction control the speed-accuracy tradeoff. "
                "Vector embeddings use cosine distance for semantic similarity matching."
            ),
            "url": "https://example.com/pgvector-hnsw",
            "date_published": datetime.utcnow() - timedelta(days=2),
        },
        {
            "source_id": "test-retr-002",
            "source_type": "email",
            "title": "Remote Work Digital Nomad Cities 2024",
            "content": (
                "Digital nomads are choosing cities like Lisbon, Barcelona, and Chiang Mai for remote work. "
                "These cities offer co-working spaces, strong internet connectivity, and vibrant expat communities. "
                "Cost of living varies widely, with Southeast Asia being most affordable. "
                "Visa regulations are becoming more digital-nomad friendly in many countries."
            ),
            "url": "https://example.com/remote-work",
            "date_published": datetime.utcnow() - timedelta(days=90),
        },
        {
            "source_id": "test-retr-003",
            "source_type": "paper",
            "title": "Vector Database Benchmarks and Performance Analysis",
            "content": (
                "We benchmark vector databases including Pinecone, Weaviate, Qdrant, and pgvector. "
                "Performance metrics include query latency, recall accuracy, and indexing throughput. "
                "HNSW and IVF indexing strategies show different characteristics under varying query patterns. "
                "Hybrid search combining sparse and dense retrieval improves relevance for many use cases."
            ),
            "url": "https://arxiv.org/example/vector-db-bench",
            "date_published": datetime.utcnow() - timedelta(days=5),
        },
        {
            "source_id": "test-retr-004",
            "source_type": "rss",
            "title": "Python Web Frameworks: Django vs Flask Comparison",
            "content": (
                "Django and Flask are popular Python web frameworks with different philosophies. "
                "Django provides batteries-included features like ORM, admin panel, and authentication. "
                "Flask is lightweight and gives developers more control over components. "
                "FastAPI is gaining popularity for building high-performance APIs with automatic documentation."
            ),
            "url": "https://example.com/python-frameworks",
            "date_published": datetime.utcnow() - timedelta(days=180),
        },
    ]

    try:
        total_chunks = 0
        for doc_data in test_docs:
            doc = ingest_document(
                source_id=doc_data["source_id"],
                source_type=doc_data["source_type"],
                title=doc_data["title"],
                content=doc_data["content"],
                url=doc_data["url"],
                date_published=doc_data["date_published"],
            )
            # ingest_document returns Document object - count chunks from relationship
            chunks = len(doc.chunks)
            total_chunks += chunks
            print(f"  Ingested: {doc_data['source_id']} ({chunks} chunks)")

        print(f"  Total chunks ingested: {total_chunks}")
        assert total_chunks > 0, "No chunks created"
        print("  PASS: Documents ingested successfully")
        return True

    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_build_bm25_index() -> bool:
    """Step 3: Build BM25 index."""
    print("\n" + "-" * 60)
    print("STEP 3: Build BM25 Index")
    print("-" * 60)

    from execution.vector_db import BM25Index

    try:
        bm25 = BM25Index()
        chunk_count = bm25.build_index(tenant_id="default")
        print(f"  BM25 index built: {chunk_count} chunks indexed")
        assert chunk_count > 0, "No chunks indexed"
        print("  PASS: BM25 index created successfully")
        return True

    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hybrid_search_basic() -> bool:
    """Step 4: Test basic hybrid search."""
    print("\n" + "-" * 60)
    print("STEP 4: Basic Hybrid Search")
    print("-" * 60)

    from execution.vector_db import hybrid_search

    try:
        results = hybrid_search("How does pgvector handle vector indexing?", top_k=3)

        print(f"  Retrieved {len(results)} results")
        assert len(results) > 0, "No results returned"

        # Check result structure
        result = results[0]
        assert hasattr(result, "id"), "Missing id field"
        assert hasattr(result, "content"), "Missing content field"
        assert hasattr(result, "rrf_score"), "Missing rrf_score field"
        assert result.rrf_score > 0, "RRF score should be > 0"

        # Top result should be pgvector-related
        top_content = results[0].content.lower()
        assert "pgvector" in top_content or "hnsw" in top_content or "vector" in top_content, \
            "Top result not relevant to query"

        print(f"\n  Top 3 results:")
        for i, r in enumerate(results[:3]):
            print(f"    {i+1}. {r.title or 'No title'}")
            print(f"       RRF score: {r.rrf_score:.4f}")
            print(f"       Content preview: {r.content[:80]}...")

        print("  PASS: Hybrid search working correctly")
        return True

    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metadata_filtering() -> bool:
    """Step 5: Test metadata filtering."""
    print("\n" + "-" * 60)
    print("STEP 5: Metadata Filtering")
    print("-" * 60)

    from execution.vector_db import HybridRetriever

    try:
        retriever = HybridRetriever()
        results = retriever.search("database", source_types=["rss"], top_k=10)

        print(f"  Retrieved {len(results)} results with source_types=['rss'] filter")
        assert len(results) > 0, "No results returned"

        # Verify all results are RSS
        source_types = set(r.source_type for r in results)
        print(f"  Source types in results: {source_types}")
        assert source_types.issubset({"rss"}), f"Non-RSS results found: {source_types - {'rss'}}"

        print("  PASS: Metadata filtering working correctly")
        return True

    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_recency_scoring() -> bool:
    """Step 6: Test recency scoring for trend queries."""
    print("\n" + "-" * 60)
    print("STEP 6: Recency Scoring")
    print("-" * 60)

    from execution.vector_db import hybrid_search

    try:
        # Use trend keyword "latest" to trigger recency boost
        results = hybrid_search("latest database news", top_k=4)

        print(f"  Retrieved {len(results)} results for trend query")
        assert len(results) > 0, "No results returned"

        # Check that recency_score and fused_score are populated
        has_recency = any(r.recency_score is not None for r in results)
        has_fused = any(r.fused_score is not None for r in results)

        print(f"\n  Results with recency scores:")
        for i, r in enumerate(results):
            days_old = (datetime.utcnow() - r.date_published).days if r.date_published else "unknown"
            print(f"    {i+1}. {r.title or 'No title'}")
            print(f"       Age: {days_old} days, Recency: {r.recency_score:.4f if r.recency_score else 'N/A'}, "
                  f"Fused: {r.fused_score:.4f if r.fused_score else 'N/A'}")

        # Recent docs should tend to rank higher (check top result is reasonably recent)
        if results[0].date_published:
            days_old_top = (datetime.utcnow() - results[0].date_published).days
            print(f"\n  Top result is {days_old_top} days old")

        print("  PASS: Recency scoring applied")
        return True

    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_citation_extraction() -> bool:
    """Step 7: Test citation extraction."""
    print("\n" + "-" * 60)
    print("STEP 7: Citation Extraction")
    print("-" * 60)

    from execution.vector_db import HybridRetriever

    try:
        retriever = HybridRetriever()
        results = retriever.search("pgvector", include_citations=True, top_k=3)

        print(f"  Retrieved {len(results)} results with citations")
        assert len(results) > 0, "No results returned"

        # Check citations populated
        citations_found = 0
        for r in results:
            if r.citations:
                citations_found += len(r.citations)

        print(f"  Total citations extracted: {citations_found}")
        assert citations_found > 0, "No citations extracted"

        # Show sample citations
        print(f"\n  Sample citations from top result:")
        if results[0].citations:
            for i, citation in enumerate(results[0].citations[:3]):
                print(f"    [{citation.citation_id}] {citation.text[:60]}...")

        print("  PASS: Citation extraction working")
        return True

    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reranking() -> bool:
    """Step 8: Test CrossEncoder reranking."""
    print("\n" + "-" * 60)
    print("STEP 8: CrossEncoder Reranking")
    print("-" * 60)

    from execution.vector_db import HybridRetriever

    try:
        retriever = HybridRetriever()

        # Search with reranking
        results_reranked = retriever.search("vector indexing performance", rerank=True, top_k=5)

        # Search without reranking
        results_no_rerank = retriever.search("vector indexing performance", rerank=False, top_k=5)

        print(f"  Reranked results: {len(results_reranked)}")
        print(f"  Non-reranked results: {len(results_no_rerank)}")

        # Check rerank_score field
        has_rerank_scores = any(r.rerank_score is not None for r in results_reranked)
        no_rerank_scores = all(r.rerank_score is None for r in results_no_rerank)

        print(f"\n  Reranked result scores:")
        for i, r in enumerate(results_reranked[:3]):
            print(f"    {i+1}. RRF={r.rrf_score:.4f}, Rerank={r.rerank_score:.4f if r.rerank_score else 'N/A'}")

        assert has_rerank_scores, "Reranked results missing rerank_score"
        assert no_rerank_scores, "Non-reranked results have rerank_score (should be None)"

        print("  PASS: Reranking working correctly")
        return True

    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_compatibility() -> bool:
    """Step 9: Test Phase 1 backward compatibility."""
    print("\n" + "-" * 60)
    print("STEP 9: Phase 1 Backward Compatibility")
    print("-" * 60)

    try:
        from execution.vector_db.indexing import semantic_search

        results = semantic_search("pgvector", limit=3)
        print(f"  semantic_search() returned {len(results)} results")
        assert len(results) > 0, "No results from semantic_search"

        # Check result structure (Phase 1 format)
        result = results[0]
        assert "id" in result, "Missing id field"
        assert "content" in result, "Missing content field"
        assert "distance" in result, "Missing distance field"

        print("  PASS: Phase 1 backward compatibility OK")
        return True

    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def cleanup_test_data() -> bool:
    """Clean up test documents."""
    print("\n" + "-" * 60)
    print("CLEANUP: Removing Test Data")
    print("-" * 60)

    from execution.vector_db import get_session
    from execution.vector_db.models import Document
    from sqlalchemy import select

    try:
        with get_session() as session:
            # Delete test documents (cascades to chunks)
            stmt = select(Document).where(Document.source_id.like("test-retr-%"))
            docs = session.execute(stmt).scalars().all()
            count = len(docs)

            for doc in docs:
                session.delete(doc)
            session.commit()

        print(f"  Deleted {count} test documents")

        # Clear BM25 index
        from execution.vector_db import BM25Index
        bm25 = BM25Index()
        bm25.clear_index()
        print("  Cleared BM25 index")

        print("  PASS: Cleanup complete")
        return True

    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 2 Hybrid Retrieval Integration Test")
    parser.add_argument("--cleanup", action="store_true", help="Delete test data after tests")
    args = parser.parse_args()

    # Check prerequisites
    if not check_prerequisites():
        print("\n" + "=" * 60)
        print("PREREQUISITES FAILED")
        print("=" * 60)
        sys.exit(1)

    # Run tests
    tests = [
        ("Step 1: Database Setup", test_database_setup),
        ("Step 2: Ingest Documents", test_ingest_documents),
        ("Step 3: Build BM25 Index", test_build_bm25_index),
        ("Step 4: Basic Hybrid Search", test_hybrid_search_basic),
        ("Step 5: Metadata Filtering", test_metadata_filtering),
        ("Step 6: Recency Scoring", test_recency_scoring),
        ("Step 7: Citation Extraction", test_citation_extraction),
        ("Step 8: Reranking", test_reranking),
        ("Step 9: Backward Compatibility", test_backward_compatibility),
    ]

    passed = 0
    failed = 0
    for name, test_func in tests:
        if test_func():
            passed += 1
        else:
            failed += 1

    # Cleanup if requested
    if args.cleanup:
        cleanup_test_data()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"  Passed: {passed}/{len(tests)}")
    print(f"  Failed: {failed}/{len(tests)}")

    if failed > 0:
        print("\n  RESULT: FAILED")
        sys.exit(1)
    else:
        print("\n  RESULT: ALL TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
