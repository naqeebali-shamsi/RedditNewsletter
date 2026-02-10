---
phase: 01-vector-db-foundation
plan: 01
subsystem: vector-database
tags: [postgresql, pgvector, sqlalchemy, docker, orm]

dependency_graph:
  requires: []
  provides:
    - pgvector PostgreSQL container via Docker Compose
    - SQLAlchemy ORM models (Document, KnowledgeChunk, IngestionLog)
    - Database connection management (engine, session, init_db)
    - VectorDBConfig in GhostWriterConfig
  affects:
    - 01-02 (embedding pipeline needs connection.py and models.py)
    - 01-03 (chunking needs Document and KnowledgeChunk models)
    - 01-04 (ingestion needs IngestionLog and full schema)

tech_stack:
  added:
    - pgvector (Vector column type for SQLAlchemy)
    - psycopg[binary] (PostgreSQL adapter for Python)
    - apscheduler (task scheduling for future ingestion)
  patterns:
    - Singleton engine pattern (matching existing sources/database.py)
    - Separate DeclarativeBase for PostgreSQL (isolated from SQLite)
    - Tenant isolation via tenant_id on all tables
    - Context manager session pattern with auto-commit/rollback

file_tracking:
  created:
    - docker-compose.yml
    - execution/vector_db/__init__.py
    - execution/vector_db/models.py
    - execution/vector_db/connection.py
    - execution/vector_db/init.sql
  modified:
    - execution/config.py
    - requirements.txt

decisions:
  - id: D-01-01-001
    decision: "Separate DeclarativeBase for vector DB"
    rationale: "PostgreSQL and SQLite are separate databases; sharing metadata would cause cross-DB conflicts"
  - id: D-01-01-002
    decision: "Synchronous SQLAlchemy (not async)"
    rationale: "Matches existing codebase pattern in sources/database.py; async adds complexity without benefit for current batch workloads"
  - id: D-01-01-003
    decision: "JSONB columns for flexible metadata"
    rationale: "topic_tags, entities, cited_by, related_to need schema flexibility; JSONB supports indexing if needed later"

metrics:
  duration: "3 min"
  completed: "2026-02-10"
---

# Phase 1 Plan 01: Docker + pgvector + Schema Foundation Summary

**One-liner:** PostgreSQL pgvector container with SQLAlchemy ORM models for documents, knowledge chunks (Vector(1536)), and ingestion logs with tenant isolation.

## What Was Built

### Docker Compose (docker-compose.yml)
- pgvector/pgvector:pg17 container (`ghostwriter-vectordb`)
- Persistent volume for data, init.sql mounted to docker-entrypoint-initdb.d
- Health check via `pg_isready`, 4GB shared memory for HNSW index builds
- Password configurable via `VECTORDB_PASSWORD` env var (default: dev_password)

### SQLAlchemy Models (execution/vector_db/models.py)
- **Document**: Source documents with tenant_id, source_type, processing_status, JSONB metadata
- **KnowledgeChunk**: Embedded chunks with Vector(1536) column, topic_tags, entities, citation tracking
- **IngestionLog**: Observability for ingestion runs (token usage, error tracking)
- UniqueConstraint on (tenant_id, source_type, source_id) prevents duplicate ingestion
- Cascade delete: removing a Document removes all its chunks

### Connection Management (execution/vector_db/connection.py)
- `get_engine()`: Singleton engine with pool_pre_ping, pool_size=5, max_overflow=10
- `get_session()`: Context manager yielding Session with auto-commit/rollback
- `init_db()`: Idempotent schema creation via Base.metadata.create_all
- `reset_engine()`: Testing utility to dispose and reset singleton

### Config Extension (execution/config.py)
- VectorDBConfig with VECTORDB_ env prefix
- DATABASE_URL, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS, DAILY_TOKEN_LIMIT
- Per-content-type chunk sizes (email=400, paper=512, rss=256)
- Added as `vector_db` field on GhostWriterConfig

### Dependencies (requirements.txt)
- pgvector>=0.3.0, psycopg[binary]>=3.1.0, apscheduler>=3.10.0

## Commits

| Commit | Description |
|--------|-------------|
| 5e93d77 | feat(01-01): Docker Compose + pgvector initialization |
| d363267 | feat(01-01): SQLAlchemy models, connection management, and config |

## Deviations from Plan

None -- plan executed exactly as written.

## Decisions Made

1. **Separate DeclarativeBase**: Vector DB uses its own `Base` class, completely isolated from the SQLite `MetaData` in `sources/database.py`. This prevents any cross-database conflicts.

2. **Synchronous SQLAlchemy**: Kept sync patterns matching the existing codebase. Async can be added later if needed for concurrent embedding operations.

3. **JSONB for flexible fields**: topic_tags, entities, cited_by, related_to use JSONB for schema flexibility. GIN indexes can be added later for query performance.

## Verification Results

- `docker compose config`: Validates without errors
- `from execution.vector_db.models import Base, KnowledgeChunk, Document, IngestionLog`: OK
- `from execution.vector_db.connection import get_engine, get_session, init_db`: OK
- `from execution.config import config; config.vector_db.DATABASE_URL`: Returns correct default
- `from execution.sources.database import get_engine, insert_content_items`: OK (existing SQLite unaffected)

## Next Phase Readiness

Plan 01-02 (embedding pipeline) can proceed -- it needs:
- `execution/vector_db/models.py` (Document, KnowledgeChunk) -- available
- `execution/vector_db/connection.py` (get_session, init_db) -- available
- `execution/config.py` (config.vector_db.EMBEDDING_MODEL) -- available
- Running PostgreSQL container -- requires `docker compose up -d` (first-time setup)
