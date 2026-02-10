# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-09)

**Core value:** The knowledge layer must reliably ingest, categorize, store, and retrieve information from diverse sources — so that every piece of content GhostWriter produces is backed by a deep, growing intelligence.
**Current focus:** Phase 1 complete. Ready for Phase 2.

## Current Position

Phase: 2 of 7 (Retrieval Tools)
Plan: 1 of 3 in current phase
Status: In progress — Plan 02-01 complete (retrieval utilities)
Last activity: 2026-02-10 — Completed 02-01-PLAN.md (metadata filters, recency scoring, citations)

Progress: [████████████░░] ~17% overall (5/estimated 30 plans)

## Phase Delivery Summaries

### Phase 1: Vector DB Foundation (COMPLETE)

**4 plans executed across 3 waves:**
- Wave 1 (Plan 01-01): Docker + pgvector + SQLAlchemy models + config
- Wave 2 (Plans 01-02, 01-03): OpenAI embedding pipeline + semantic chunking + AI tagging
- Wave 3 (Plan 01-04): Ingestion orchestrator + HNSW indexing + integration test

**17 commits total. Key artifacts:**

| Module | Purpose | Key Exports |
|--------|---------|-------------|
| `execution/vector_db/models.py` | SQLAlchemy ORM (Document, KnowledgeChunk, IngestionLog) | Base, Document, KnowledgeChunk |
| `execution/vector_db/connection.py` | Singleton engine, session management, init_db | get_engine, get_session, init_db |
| `execution/vector_db/embeddings.py` | OpenAI embedding client (sync + batch modes) | EmbeddingClient, embed_texts |
| `execution/vector_db/token_tracking.py` | Daily token budget tracking | TokenTracker, estimate_tokens |
| `execution/vector_db/chunking.py` | Content-type-aware semantic chunking | SemanticChunker, chunk_content |
| `execution/vector_db/tagging.py` | AI-powered topic classification + entity extraction | AutoTagger, auto_tag |
| `execution/vector_db/ingestion.py` | Full pipeline orchestrator (chunk→tag→embed→store) | IngestionPipeline, ingest_document |
| `execution/vector_db/indexing.py` | HNSW index + semantic search | create_hnsw_index, semantic_search |
| `docker-compose.yml` | pgvector/pgvector:pg17 on port 5433 | Container config |
| `scripts/test_vectordb.py` | End-to-end integration test | CLI test runner |

**Deferred:** Live embedding + semantic search test (OpenAI monthly budget exhausted). All code paths verified via import/instantiation/unit checks.

### Phase 2: Retrieval Tools (IN PROGRESS)

**1 plan executed (Wave 1):**
- Plan 02-01: Metadata filters + recency scoring + citation extraction

**3 commits total. Key artifacts:**

| Module | Purpose | Key Exports |
|--------|---------|-------------|
| `execution/vector_db/metadata_filters.py` | SQLAlchemy filter builders for scoped retrieval | MetadataFilter, build_filters |
| `execution/vector_db/recency_scoring.py` | Time decay functions for trend-aware ranking | RecencyScorer |
| `execution/vector_db/citations.py` | Sentence-level citation extraction | CitationExtractor, Citation |

## Performance Metrics

**Velocity:**
- Total plans completed: 4
- Total execution time: ~45 min (team-based parallel execution)
- Review cycles: 2 (Wave 1 + Wave 2), 1 blocker caught + fixed

**Team execution model:**
- 2 developers (parallel on Wave 2)
- 2 code reviewers
- 2 adversarial auditors
- 1 research agent (library alternatives)
- Multiple tester iterations

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Knowledge layer before output formats: Build the engine first, then applications around it — richer knowledge makes every output better
- pgvector + OpenAI embeddings: Research recommends PostgreSQL with pgvector over managed vector DBs for cost efficiency and Python integration
- Modular RAG with tool-based retrieval: RAG operates as LangGraph tools invoked by agents, not as separate service layer coupling to hot path
- Daily batch for Gmail ingestion: Simpler than real-time, sufficient for newsletter use case
- Both structured extraction + semantic chunks: Structured for precise facts, chunks for broad retrieval — covers both use cases
- pysbd over regex for sentence splitting: 97.92% accuracy, handles abbreviations/decimals/URLs (D-01-03-001)
- html2text + regex for email processing: Library for HTML conversion, hand-rolled for domain-specific boilerplate (D-01-03-002)
- Rule-based chunking first, LLM-powered later: Deterministic and testable foundation; LLM chunking deferred to Phase 3 (D-01-03-003)
- Docker port 5433 to avoid local PostgreSQL conflict (D-01-04-001)
- expire_on_commit=False for returning ORM objects from ingestion pipeline (D-01-04-002)
- SQLAlchemy JSONB operators over raw SQL for metadata filtering: Eliminates SQL injection, handles type casting (D-02-01-001)
- 14-day half-life default for recency decay: Research-validated (ArXiv 2509.19376), configurable per source type (D-02-01-002)
- Keyword heuristics for trend detection (not LLM): Fast, deterministic, extensible to LLM later (D-02-01-003)
- Citation ID format {chunk_id}.{sentence_idx}: Compact, preserves context, easy to parse (D-02-01-004)

### Pending Todos

- Run full integration test with live OpenAI embeddings when budget resets (18 days)
- Fix numpy/matplotlib version conflict (lexicalrichness depends on matplotlib compiled for numpy 1.x)
- Add GIN indexes on topic_tags and entities columns before Phase 02-03 hybrid retrieval load testing (Phase 2)
- Track LLM citation compliance rate in Phase 5 (% of claims with [citation_id])

### Blockers/Concerns

**Critical integration risks identified in research:**
- Breaking existing pipeline with poorly integrated RAG (prevention: feature flags per agent, async retrieval with timeout, circuit breaker)
- Hallucination amplification from bad chunking/retrieval (prevention: semantic chunking from day one, reranking with cross-encoder, metadata filtering)
- Gmail API quota exhaustion (prevention: batch API, exponential backoff, proactive token refresh)
- Semantic Scholar rate limits (prevention: batch endpoints, 1 req/sec enforcement, citation depth limiting)

**Phase-specific research flags:**
- Phase 2 (Retrieval Tools): GIN index performance needs validation at 100k+ chunks before hybrid retrieval
- Phase 2 (Retrieval Tools): Trend keyword coverage may miss queries — log classifications to identify gaps
- Phase 3 (Gmail Ingestion): Newsletter format diversity testing needed with Substack/Beehiiv/ConvertKit
- Phase 4 (Semantic Scholar): Citation graph traversal strategies and batch endpoint behavior need validation
- Phase 5 (Pipeline Integration): Agent-specific retrieval impact unknown — A/B testing required
- Phase 5 (Pipeline Integration): LLM citation compliance unknown — track [citation_id] usage in responses

## Session Continuity

Last session: 2026-02-10T16:31:20Z
Stopped at: Completed 02-01-PLAN.md (metadata filters, recency scoring, citations)
Resume file: None
Next: Phase 02-02 (BM25 + RRF Hybrid Search) or 02-03 (Hybrid Retrieval Orchestrator)
