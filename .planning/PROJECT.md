# GhostWriter — Knowledge-Powered Content Engine

## What This Is

GhostWriter is an AI-powered content engine that accumulates intelligence from multiple sources (newsletters, research papers, Reddit, HackerNews, RSS) into a searchable knowledge base ("mind palace"), then generates high-quality content from that rich foundation. Currently built for NomadCrew (https://nomadcrew.uk), architected for multi-tenancy later. The existing pipeline handles article generation with multi-agent quality gates; this milestone adds the knowledge layer that makes it truly intelligent.

## Core Value

The knowledge layer must reliably ingest, categorize, store, and retrieve information from diverse sources — so that every piece of content GhostWriter produces is backed by a deep, growing intelligence rather than single-session web searches.

## Requirements

### Validated

- ✓ Multi-agent article generation pipeline (writer, editor, critic, style enforcer) — existing
- ✓ LangGraph state machine with 6-phase workflow (RESEARCH → GENERATE → VERIFY → REVIEW → REVISE → APPROVE) — existing
- ✓ Multi-LLM provider routing with fallback (Gemini/Groq/Perplexity/OpenAI/Anthropic) — existing
- ✓ Adversarial review panel with quality scoring (threshold ≥ 7.0) — existing
- ✓ Post-generation fact verification with claim extraction — existing
- ✓ Tone-adaptive writing with 6 built-in presets + custom inference from samples — existing
- ✓ Content provenance tracking (C2PA, Schema.org, AI disclosure) — existing
- ✓ Internet pulse monitoring from Reddit, HackerNews, RSS, GitHub — existing
- ✓ Streamlit dashboard UI — existing
- ✓ Style enforcement with 5-dimension quantitative scoring — existing
- ✓ SQLite persistence for content items — existing
- ✓ Gmail source handler (basic implementation exists) — existing

### Active

- [ ] Vector database infrastructure for knowledge storage and semantic retrieval
- [ ] Gmail newsletter ingestion pipeline — OAuth connection, daily batch processing
- [ ] Structured extraction from newsletters (key claims, stats, quotes, links)
- [ ] Semantic chunking and embedding of newsletter content
- [ ] Smart categorization and metadata tagging for all ingested content
- [ ] Semantic Scholar API integration — citation support for article claims
- [ ] Semantic Scholar trend discovery — monitor new papers in specific fields
- [ ] Semantic Scholar deep research — abstracts, key findings, citation graphs
- [ ] RAG context injection — query knowledge base before writing, feed relevant context to writer agent
- [ ] Direct agent access — writer, fact checker, researcher can search the knowledge base
- [ ] Dedicated Research Agent — prepares briefings from knowledge base before article generation
- [ ] End-to-end proof — one enriched article demonstrating knowledge layer working

### Out of Scope

- New output formats (X threads, YouTube scripts, brand blog templates) — deferred to next milestone after knowledge layer is solid
- Multi-tenancy implementation — architecture for it, but don't build tenant isolation yet
- Real-time email ingestion — daily batch is sufficient for v1
- Full Semantic Scholar MCP server — direct API integration first, MCP wrapper later if useful
- Mobile app or native clients — web-only via Streamlit

## Context

**Existing Architecture:**
- 3-layer architecture: Directives (Markdown SOPs) → Orchestration (Python config/pipeline) → Execution (agents + sources)
- LangGraph StateGraph orchestrates the 6-phase pipeline with node timeouts and checkpoint persistence
- Multi-provider LLM routing: Gemini for research, Groq for writing/editing, Claude for ethics, GPT-4o for structure review
- SQLite for content persistence, no vector DB yet
- Gmail source handler exists in `execution/sources/gmail_source.py` but is basic
- Pydantic models for config and article state

**NomadCrew Brand:**
- Target audience: tech-savvy professionals, builders, and creators
- Content tone: Expert Pragmatist (informed, direct, no fluff)
- Goal: build community trust through high-quality, research-backed content

**Technical Environment:**
- Python 3.x, Windows development, cloud deployment OK
- Already using: Streamlit, Pydantic, SQLAlchemy, LangGraph, multiple LLM APIs
- Google OAuth credentials already set up (for Gmail)

## Constraints

- **Single-tenant first**: Build for NomadCrew, but use tenant-aware data models so multi-tenancy doesn't require a rewrite
- **Cloud OK**: Can use cloud services (AWS, managed vector DBs) for persistence and scheduling
- **Daily batch**: Newsletter ingestion runs as daily batch, not real-time push
- **API costs**: Semantic Scholar API is free but rate-limited; embedding costs scale with volume — need cost-aware architecture
- **Existing pipeline**: Knowledge layer integrates with the existing LangGraph pipeline, doesn't replace it

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Knowledge layer before output formats | "Build the engine first, then applications around it" — richer knowledge makes every output better | — Pending |
| Vector DB choice | Research needed — evaluating ChromaDB, Pinecone, Qdrant, Weaviate, AWS AgentCore | — Pending |
| Direct Semantic Scholar API (not MCP) | Simpler integration, MCP adds abstraction without clear benefit yet | — Pending |
| Daily batch for Gmail ingestion | Simpler than real-time, sufficient for newsletter use case | — Pending |
| Both structured extraction + semantic chunks | Structured for precise facts, chunks for broad retrieval — covers both use cases | — Pending |
| RAG + agent access + dedicated research agent | Maximum flexibility — context injection for passive enrichment, agent access for active research | — Pending |

---
*Last updated: 2026-02-09 after initialization*
