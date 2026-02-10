# Roadmap: GhostWriter Knowledge Layer

## Overview

This milestone transforms GhostWriter from a single-session content generator into an intelligence-accumulating engine. The journey moves from foundation (vector database, embeddings, semantic chunking) through data sources (Gmail newsletters, academic papers), pipeline integration (RAG context injection, agent tool access), to advanced capabilities (adaptive learning, context-aware suggestions). Each phase delivers a coherent, verifiable capability while preserving the existing working pipeline through feature flag isolation. The roadmap culminates in an end-to-end proof: one enriched article demonstrating newsletter insights + academic papers + fine-grained citations flowing through the complete knowledge-powered pipeline.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Vector DB Foundation** - pgvector infrastructure with embedding and chunking pipelines
- [ ] **Phase 2: Retrieval Tools** - Hybrid search, reranking, and metadata filtering for RAG
- [ ] **Phase 3: Gmail Newsletter Ingestion** - OAuth-connected email ingestion with structured extraction
- [ ] **Phase 4: Semantic Scholar Integration** - Academic paper ingestion with citation support
- [ ] **Phase 5: Pipeline Integration** - LangGraph tool-based RAG integration with feature flags
- [ ] **Phase 6: Advanced Features** - Adaptive learning, context-aware suggestions, knowledge graph
- [ ] **Phase 7: End-to-End Proof** - Enriched article demonstrating full knowledge pipeline

## Phase Details

### Phase 1: Vector DB Foundation
**Goal**: Establish core RAG infrastructure with pgvector for semantic storage and retrieval
**Depends on**: Nothing (first phase)
**Requirements**: INFRA-01, INFRA-02, INFRA-03, INFRA-04, INFRA-05, INFRA-06
**Success Criteria** (what must be TRUE):
  1. PostgreSQL with pgvector extension stores and queries vector embeddings with HNSW indexing
  2. OpenAI text-embedding-3-small converts text to 1536-dim vectors via batch API
  3. Semantic chunking pipeline preserves meaning across email, paper, and RSS content types
  4. AI auto-tagging classifies topics, extracts entities, and labels sources on ingestion
  5. Tenant-aware data model uses namespace isolation for future multi-tenancy without rewrite
**Plans**: TBD

Plans: (to be defined during plan-phase)

### Phase 2: Retrieval Tools
**Goal**: Build modular RAG retrieval layer with hybrid search and reranking for agent integration
**Depends on**: Phase 1
**Requirements**: RETR-01, RETR-02, RETR-03, RETR-04, RETR-05
**Success Criteria** (what must be TRUE):
  1. Hybrid search combines semantic vectors + BM25 keyword matching with RRF fusion
  2. Metadata filters scope searches by date range, source type, and topic tags
  3. Fine-grained citations provide sentence-level source attribution with clickable links
  4. Source recency scoring prioritizes documents less than 6 months old for trend content
  5. CrossEncoder reranking improves precision on top-K retrieval results
**Plans**: TBD

Plans: (to be defined during plan-phase)

### Phase 3: Gmail Newsletter Ingestion
**Goal**: Enable automated newsletter ingestion from Gmail with structured extraction and vectorization
**Depends on**: Phase 1, Phase 2
**Requirements**: GMAIL-01, GMAIL-02, GMAIL-03, GMAIL-04, GMAIL-05, GMAIL-06, GMAIL-07
**Success Criteria** (what must be TRUE):
  1. Gmail OAuth 2.0 connection handles token refresh and survives password changes and 6-month inactivity
  2. Daily batch processing fetches new emails since last sync, processes, embeds, and stores
  3. Structured extraction captures key claims, statistics, quotes, and links from newsletters
  4. Semantic chunking converts email body content into vector-searchable chunks
  5. HTML parsing handles diverse newsletter formats including Substack, Beehiiv, ConvertKit
  6. Duplicate detection skips already-ingested emails on re-sync
  7. Historical backfill processes existing inbox emails on first connection
**Plans**: TBD

Plans: (to be defined during plan-phase)

### Phase 4: Semantic Scholar Integration
**Goal**: Add academic paper ingestion with citation support and trend discovery capabilities
**Depends on**: Phase 1, Phase 2
**Requirements**: SCHOL-01, SCHOL-02, SCHOL-03, SCHOL-04, SCHOL-05, SCHOL-06
**Success Criteria** (what must be TRUE):
  1. Authenticated Semantic Scholar API integration respects rate limits with exponential backoff
  2. Citation support finds academic papers backing specific claims in articles
  3. Trend discovery monitors new papers in configured fields and surfaces article-worthy topics
  4. Deep research pulls abstracts, key findings, methodology, and citation graphs
  5. Paper metadata (title, authors, abstract, year, citation count) extracted and stored in vector DB
  6. Citation graph traversal finds related and cited-by papers for depth exploration
**Plans**: TBD

Plans: (to be defined during plan-phase)

### Phase 5: Pipeline Integration
**Goal**: Integrate RAG retrieval into LangGraph pipeline via tools without breaking existing workflow
**Depends on**: Phase 1, Phase 2, Phase 3, Phase 4
**Requirements**: PIPE-01, PIPE-02, PIPE-03, PIPE-04, PIPE-05, PIPE-06
**Success Criteria** (what must be TRUE):
  1. RAG context injection queries knowledge base before article writing and feeds relevant context to writer agent
  2. LangGraph ToolNode exposes search_knowledge(), verify_claim(), research_topic() tools to agents
  3. Dedicated Research Agent prepares knowledge briefings from mind palace before generation
  4. Cross-source synthesis combines newsletter insights, academic papers, and web sources into unified narratives
  5. Feature flag isolation allows RAG to toggle on/off without breaking existing pipeline
  6. Retrieval latency stays within 500ms budget to avoid degrading pipeline performance
**Plans**: TBD

Plans: (to be defined during plan-phase)

### Phase 6: Advanced Features
**Goal**: Add intelligent capabilities for adaptive learning and proactive knowledge surfacing
**Depends on**: Phase 5
**Requirements**: ADV-01, ADV-02, ADV-03
**Success Criteria** (what must be TRUE):
  1. Adaptive voice learning tracks user edits and refines brand voice model over time
  2. Context-aware suggestions proactively surface relevant knowledge during article writing (Copilot-style)
  3. Knowledge graph enables entity relationships and multi-hop reasoning beyond pure vector search
**Plans**: TBD

Plans: (to be defined during plan-phase)

### Phase 7: End-to-End Proof
**Goal**: Demonstrate complete knowledge pipeline with one enriched article
**Depends on**: Phase 5
**Requirements**: PROOF-01
**Success Criteria** (what must be TRUE):
  1. One article generated using knowledge from Gmail newsletters, Semantic Scholar papers, and existing sources
  2. Article contains fine-grained citations with sentence-level attribution to sources
  3. Full pipeline demonstrates research → retrieval → synthesis → generation → verification working end-to-end
**Plans**: TBD

Plans: (to be defined during plan-phase)

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Vector DB Foundation | 0/TBD | Not started | - |
| 2. Retrieval Tools | 0/TBD | Not started | - |
| 3. Gmail Newsletter Ingestion | 0/TBD | Not started | - |
| 4. Semantic Scholar Integration | 0/TBD | Not started | - |
| 5. Pipeline Integration | 0/TBD | Not started | - |
| 6. Advanced Features | 0/TBD | Not started | - |
| 7. End-to-End Proof | 0/TBD | Not started | - |

---
*Roadmap created: 2026-02-10*
*Last updated: 2026-02-10 (initial creation)*
