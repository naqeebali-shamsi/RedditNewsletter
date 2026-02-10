# Phase 1: Vector DB Foundation - Context

**Gathered:** 2026-02-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Establish core RAG infrastructure with pgvector for semantic storage and retrieval. Includes: PostgreSQL + pgvector setup, OpenAI embedding pipeline, semantic chunking, AI auto-tagging, tenant-aware data model, and incremental re-indexing. Does NOT include: retrieval tools (Phase 2), source-specific ingestion (Phases 3-4), or pipeline integration (Phase 5).

</domain>

<decisions>
## Implementation Decisions

### Database hosting
- Docker PostgreSQL locally for development
- AWS (RDS or Aurora) for production deployment
- SQLite stays for existing content items (reddit_content.db) — pgvector is additive, not a replacement
- Budget: $10-50/month for managed production instance is acceptable

### Chunking strategy
- LLM-powered extraction and parsing for newsletters — send stripped HTML to a cheap/fast model (Haiku or Gemini Flash) for structured extraction
- LLM identifies natural semantic boundaries for chunking (not fixed-size splits)
- Chunks overlap 10-20% at boundaries to preserve context
- Structured extraction output: key claims, statistics, quotes, links
- Source-specific chunking strategies: Claude's discretion based on research findings

### Data model design
- Organization structure: Claude's discretion (flat collection with metadata vs separate collections)
- Metadata fields required: source_type, date_ingested, date_published, topic_tags (AI-generated)
- Author/source name NOT tracked as primary metadata (can add later if needed)
- Tenant-aware: tenant_id column on all tables — minimal prep, filter by it, no full namespace isolation yet
- Basic relationship tracking from the start: cited_by and related_to links between documents — foundation for Phase 6 knowledge graph

### Ingestion behavior
- Failure handling: retry 3 times with backoff, then skip failed item and continue — log all failures for review
- Cost guardrail: daily token limit on embedding API — stop ingestion when exceeded
- Observability: structured logs for detail + Streamlit dashboard widget for quick ingestion status
- Schedule: automatic daily cron + manual trigger button in dashboard for on-demand runs

### Claude's Discretion
- Exact chunk sizes per content type (research should inform this)
- Flat vs separate collection architecture for pgvector
- Docker Compose configuration for local dev environment
- Cron implementation (system cron vs APScheduler vs Celery Beat)
- Specific LLM model choice for extraction (Haiku vs Gemini Flash — cost/quality tradeoff)
- HNSW index parameters (ef_construction, m values)

</decisions>

<specifics>
## Specific Ideas

- User suggested using an LLM to parse newsletter HTML rather than rule-based extraction — "What if we set an LLM to parse it?" — this is the preferred approach
- SQLite coexistence is important: the existing content pipeline (reddit_content.db) must continue working unchanged
- Daily token limit should have a sensible default but be configurable

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-vector-db-foundation*
*Context gathered: 2026-02-10*
