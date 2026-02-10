"""
End-to-end integration test for the Vector Database Foundation.

Exercises the full pipeline: init -> ingest -> chunk -> tag -> embed ->
index -> search. Requires Docker (ghostwriter-vectordb) and OPENAI_API_KEY.

Usage:
    python scripts/test_vectordb.py             # Run tests, keep data
    python scripts/test_vectordb.py --cleanup    # Run tests, then delete test data
"""

import argparse
import os
import subprocess
import sys

# Ensure project root is on sys.path so 'execution' package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------

def check_prerequisites() -> bool:
    """Verify Docker container and API key are available."""
    print("=" * 60)
    print("VECTOR DB INTEGRATION TEST")
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
        print(f"  Tables created: {table_names}")
        assert "documents" in table_names, "documents table missing"
        assert "knowledge_chunks" in table_names, "knowledge_chunks table missing"
        assert "ingestion_logs" in table_names, "ingestion_logs table missing"
        print("  PASS: All tables created successfully")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_single_ingestion() -> dict:
    """Step 2: Ingest a single test document."""
    print("\n" + "-" * 60)
    print("STEP 2: Single Document Ingestion")
    print("-" * 60)

    from execution.vector_db.ingestion import ingest_document

    try:
        doc = ingest_document(
            title="Test: PostgreSQL Vector Search in 2026",
            content=(
                "PostgreSQL with pgvector has become the standard approach for vector "
                "similarity search in 2026. The HNSW indexing algorithm provides fast approximate "
                "nearest neighbor search, while halfvec storage cuts memory usage by 50%.\n\n"
                "OpenAI's text-embedding-3-small model generates 1536-dimensional vectors at "
                "$0.01 per million tokens via the Batch API. This makes semantic search affordable "
                "even for large knowledge bases.\n\n"
                "Key benefits include: native SQL integration, ACID compliance, and the ability "
                "to combine vector search with traditional filtering on metadata columns like "
                "source_type and tenant_id."
            ),
            source_type="rss",
            source_id="test-001",
            url="https://example.com/test",
        )

        chunk_count = len(doc.chunks) if doc.chunks else 0
        print(f"  Document ID: {doc.id}")
        print(f"  Chunks: {chunk_count}")
        print(f"  Status: {doc.processing_status}")
        assert doc.processing_status == "embedded", f"Expected 'embedded', got '{doc.processing_status}'"
        assert chunk_count > 0, "No chunks created"
        print("  PASS: Document ingested and embedded")
        return {"doc_id": doc.id, "chunks": chunk_count}
    except Exception as e:
        print(f"  FAIL: {e}")
        return {}


def test_second_ingestion() -> dict:
    """Step 3: Ingest a second document on a different topic."""
    print("\n" + "-" * 60)
    print("STEP 3: Second Document Ingestion (Different Topic)")
    print("-" * 60)

    from execution.vector_db.ingestion import ingest_document

    try:
        doc = ingest_document(
            title="Remote Work Trends for Digital Nomads",
            content=(
                "Remote work continues to reshape how digital professionals approach their "
                "careers in 2026. Coworking spaces have expanded globally, with Lisbon, "
                "Bangkok, and Medellin leading as top digital nomad destinations.\n\n"
                "New visa programs in Portugal, Spain, and Thailand now offer dedicated "
                "remote worker permits, making it easier than ever to work from abroad. "
                "Companies are increasingly offering location-flexible contracts.\n\n"
                "The rise of async-first communication tools has made timezone differences "
                "less of a barrier. Teams spread across 8+ timezones report productivity "
                "gains when they adopt structured documentation practices."
            ),
            source_type="email",
            source_id="test-002",
        )

        chunk_count = len(doc.chunks) if doc.chunks else 0
        print(f"  Document ID: {doc.id}")
        print(f"  Chunks: {chunk_count}")
        print(f"  Status: {doc.processing_status}")
        assert doc.processing_status == "embedded", f"Expected 'embedded', got '{doc.processing_status}'"
        print("  PASS: Second document ingested")
        return {"doc_id": doc.id, "chunks": chunk_count}
    except Exception as e:
        print(f"  FAIL: {e}")
        return {}


def test_duplicate_detection() -> bool:
    """Step 4: Verify duplicate documents are skipped."""
    print("\n" + "-" * 60)
    print("STEP 4: Duplicate Detection")
    print("-" * 60)

    from execution.vector_db.ingestion import ingest_document
    from execution.vector_db.connection import get_session
    from execution.vector_db.models import KnowledgeChunk

    try:
        # Count chunks before
        with get_session() as session:
            before_count = session.query(KnowledgeChunk).count()

        # Re-ingest the same document
        doc = ingest_document(
            title="Test: PostgreSQL Vector Search in 2026",
            content="This content should be ignored.",
            source_type="rss",
            source_id="test-001",
        )

        # Count chunks after
        with get_session() as session:
            after_count = session.query(KnowledgeChunk).count()

        print(f"  Status: {doc.processing_status}")
        print(f"  Chunks before: {before_count}, after: {after_count}")
        assert before_count == after_count, "Duplicate created new chunks!"
        print("  PASS: Duplicate correctly skipped")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_hnsw_index() -> bool:
    """Step 5: Create HNSW index."""
    print("\n" + "-" * 60)
    print("STEP 5: HNSW Index Creation")
    print("-" * 60)

    from execution.vector_db.indexing import create_hnsw_index, get_index_stats

    try:
        success = create_hnsw_index()
        assert success, "Index creation returned False"

        stats = get_index_stats()
        print(f"  Index exists: {stats['exists']}")
        print(f"  Size: {stats['size_bytes']} bytes")
        print(f"  Rows indexed: {stats['rows_indexed']}")
        assert stats["exists"], "Index not found after creation"
        print("  PASS: HNSW index created")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_semantic_search() -> bool:
    """Step 6: Semantic search queries."""
    print("\n" + "-" * 60)
    print("STEP 6: Semantic Search")
    print("-" * 60)

    from execution.vector_db.indexing import semantic_search

    all_passed = True

    # Query 1: Should match the pgvector document
    print("\n  Query 1: 'How does pgvector handle vector indexing?'")
    try:
        results = semantic_search("How does pgvector handle vector indexing?", limit=5)
        if results:
            for i, r in enumerate(results):
                print(f"    [{i+1}] distance={r['distance']:.4f} | {r['title'][:50]}")
            # First result should be the pgvector document
            assert "PostgreSQL" in results[0]["title"] or "pgvector" in results[0]["content"].lower(), \
                "Expected pgvector document as top result"
            print("  PASS: pgvector document ranked first")
        else:
            print("  FAIL: No results returned")
            all_passed = False
    except Exception as e:
        print(f"  FAIL: {e}")
        all_passed = False

    # Query 2: Should match the remote work document
    print("\n  Query 2: 'What are the best cities for remote workers?'")
    try:
        results = semantic_search("What are the best cities for remote workers?", limit=5)
        if results:
            for i, r in enumerate(results):
                print(f"    [{i+1}] distance={r['distance']:.4f} | {r['title'][:50]}")
            assert "Remote" in results[0]["title"] or "nomad" in results[0]["content"].lower(), \
                "Expected remote work document as top result"
            print("  PASS: Remote work document ranked first")
        else:
            print("  FAIL: No results returned")
            all_passed = False
    except Exception as e:
        print(f"  FAIL: {e}")
        all_passed = False

    return all_passed


def cleanup_test_data() -> None:
    """Delete test documents and their chunks."""
    print("\n" + "-" * 60)
    print("CLEANUP: Removing test data")
    print("-" * 60)

    from execution.vector_db.connection import get_session
    from execution.vector_db.models import Document, KnowledgeChunk
    from execution.vector_db.indexing import drop_hnsw_index
    from sqlalchemy import and_

    with get_session() as session:
        test_docs = session.query(Document).filter(
            Document.source_id.in_(["test-001", "test-002"])
        ).all()

        for doc in test_docs:
            session.query(KnowledgeChunk).filter(
                KnowledgeChunk.document_id == doc.id
            ).delete()
            session.delete(doc)

        print(f"  Deleted {len(test_docs)} test documents and their chunks")

    drop_hnsw_index()
    print("  HNSW index dropped")
    print("  DONE: Cleanup complete")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Vector DB end-to-end integration test")
    parser.add_argument("--cleanup", action="store_true", help="Delete test data after run")
    args = parser.parse_args()

    if not check_prerequisites():
        return 1

    results = {}

    # Step 1: Database setup
    if not test_database_setup():
        return 1

    # Step 2: Single document ingestion
    doc1 = test_single_ingestion()
    if not doc1:
        return 1
    results["doc1"] = doc1

    # Step 3: Second document
    doc2 = test_second_ingestion()
    if not doc2:
        return 1
    results["doc2"] = doc2

    # Step 4: Duplicate detection
    if not test_duplicate_detection():
        return 1

    # Step 5: HNSW index
    if not test_hnsw_index():
        return 1

    # Step 6: Semantic search
    if not test_semantic_search():
        return 1

    # Summary
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
    print(f"  Document 1: {results['doc1']['chunks']} chunks embedded")
    print(f"  Document 2: {results['doc2']['chunks']} chunks embedded")
    print("  Duplicate detection: working")
    print("  HNSW index: created")
    print("  Semantic search: returning relevant results")

    if args.cleanup:
        cleanup_test_data()

    return 0


if __name__ == "__main__":
    sys.exit(main())
