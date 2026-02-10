# Requirements: GhostWriter Knowledge Layer

**Defined:** 2026-02-09
**Core Value:** The knowledge layer must reliably ingest, categorize, store, and retrieve information from diverse sources — so that every piece of content GhostWriter produces is backed by a deep, growing intelligence.

## v1 Requirements

Requirements for the knowledge layer milestone. Each maps to roadmap phases.

### RAG Infrastructure

- [ ] **INFRA-01**: Vector database (pgvector) provisioned with HNSW indexing for semantic search
- [ ] **INFRA-02**: Embedding pipeline using OpenAI text-embedding-3-small with batch API support
- [ ] **INFRA-03**: Semantic chunking pipeline that preserves meaning across content types (emails, papers, RSS)
- [ ] **INFRA-04**: AI auto-tagging on ingestion — topic classification, entity extraction, source type labeling
- [ ] **INFRA-05**: Tenant-aware data model (namespace isolation) for future multi-tenancy without rewrite
- [ ] **INFRA-06**: Incremental re-indexing pipeline for knowledge base updates (not full re-embed)

### Search & Retrieval

- [ ] **RETR-01**: Hybrid search combining dense vectors (semantic) + sparse vectors (BM25) with RRF fusion
- [ ] **RETR-02**: Metadata filtering — user can scope searches by date range, source type, topic tags
- [ ] **RETR-03**: Fine-grained citations — sentence-level source attribution with clickable links to original
- [ ] **RETR-04**: Source recency scoring — prioritize recent sources for trend-sensitive content
- [ ] **RETR-05**: CrossEncoder reranking on top-K retrieval results for precision improvement

### Gmail Newsletter Ingestion

- [ ] **GMAIL-01**: Gmail OAuth 2.0 connection with token refresh handling (survives password changes, 6-month inactivity)
- [ ] **GMAIL-02**: Daily batch processing — fetch new emails since last sync, process, embed, store
- [ ] **GMAIL-03**: Structured extraction from newsletters — key claims, statistics, quotes, links
- [ ] **GMAIL-04**: Semantic chunking of email body content for vector storage
- [ ] **GMAIL-05**: HTML parsing across diverse newsletter formats (Substack, Beehiiv, ConvertKit, custom)
- [ ] **GMAIL-06**: Duplicate detection — skip already-ingested emails on re-sync
- [ ] **GMAIL-07**: Historical backfill — process existing inbox emails on first connection

### Semantic Scholar Integration

- [ ] **SCHOL-01**: Authenticated Semantic Scholar API integration with rate limiting and exponential backoff
- [ ] **SCHOL-02**: Citation support — find papers that back up specific claims in articles
- [ ] **SCHOL-03**: Trend discovery — monitor new papers in configured fields, surface article-worthy topics
- [ ] **SCHOL-04**: Deep research — pull abstracts, key findings, methodology, and citation graphs
- [ ] **SCHOL-05**: Paper metadata extraction and storage in vector DB (title, authors, abstract, year, citation count)
- [ ] **SCHOL-06**: Citation graph traversal — find related/cited-by papers for depth exploration

### Pipeline Integration

- [ ] **PIPE-01**: RAG context injection — query knowledge base for relevant context before article writing begins
- [ ] **PIPE-02**: Agent tool access — LangGraph ToolNode integration with search_knowledge(), verify_claim(), research_topic()
- [ ] **PIPE-03**: Dedicated Research Agent — prepares knowledge briefings from the mind palace before generation
- [ ] **PIPE-04**: Cross-source synthesis — combine newsletter insights + academic papers + web sources into unified narrative
- [ ] **PIPE-05**: Feature flag isolation — RAG integration can be toggled on/off without breaking existing pipeline
- [ ] **PIPE-06**: Retrieval latency budget — knowledge queries complete within 500ms to avoid degrading pipeline performance

### Advanced Features

- [ ] **ADV-01**: Adaptive voice learning — system tracks user edits and refines brand voice model over time
- [ ] **ADV-02**: Context-aware suggestions — proactively surface relevant knowledge during article writing (Copilot-style)
- [ ] **ADV-03**: Knowledge graph — entity relationships and multi-hop reasoning beyond pure vector search

### End-to-End Proof

- [ ] **PROOF-01**: One enriched article generated using knowledge from Gmail newsletters + Semantic Scholar papers + existing sources, with fine-grained citations, demonstrating the full pipeline working end-to-end

## v2 Requirements

Deferred to future milestones. Tracked but not in current roadmap.

### Output Formats

- **OUT-01**: X/Twitter thread generation from knowledge base insights
- **OUT-02**: YouTube Shorts script generation with hooks and CTAs
- **OUT-03**: Brand blog templates with SEO optimization
- **OUT-04**: LinkedIn post generation from article summaries

### Multi-Tenancy

- **TENANT-01**: Tenant registration and onboarding flow
- **TENANT-02**: Per-tenant API key management and billing
- **TENANT-03**: Cross-tenant data isolation enforcement and auditing
- **TENANT-04**: Per-tenant knowledge base quotas and usage tracking

### Additional Sources

- **SRC-01**: User can connect custom RSS feeds as knowledge sources
- **SRC-02**: PDF/document upload and ingestion
- **SRC-03**: YouTube transcript ingestion
- **SRC-04**: Slack/Discord channel monitoring

### Intelligence

- **INTEL-01**: Audio knowledge briefings (daily podcast-style summaries)
- **INTEL-02**: Visual knowledge maps (concept graphs, mind maps)
- **INTEL-03**: Multi-hop question answering across knowledge base
- **INTEL-04**: Custom knowledge personas (expert modes trained on subsets)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Real-time email push notifications | Daily batch sufficient for newsletter use case; real-time adds WebSocket complexity |
| MCP server for Semantic Scholar | Direct API integration simpler; MCP wrapper adds abstraction without clear benefit yet |
| Mobile app / native clients | Web-only via Streamlit for this milestone |
| Multi-tenant billing/payments | Architecture for tenancy, but no payment infrastructure yet |
| Video content processing | Storage/bandwidth costs, not core to knowledge text pipeline |
| Full Semantic Scholar MCP | Direct API first; MCP if useful after validation |
| ChromaDB or other embedded vector DB | pgvector recommended by research for scale + cost; no dual-DB approach |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| INFRA-01 | — | Pending |
| INFRA-02 | — | Pending |
| INFRA-03 | — | Pending |
| INFRA-04 | — | Pending |
| INFRA-05 | — | Pending |
| INFRA-06 | — | Pending |
| RETR-01 | — | Pending |
| RETR-02 | — | Pending |
| RETR-03 | — | Pending |
| RETR-04 | — | Pending |
| RETR-05 | — | Pending |
| GMAIL-01 | — | Pending |
| GMAIL-02 | — | Pending |
| GMAIL-03 | — | Pending |
| GMAIL-04 | — | Pending |
| GMAIL-05 | — | Pending |
| GMAIL-06 | — | Pending |
| GMAIL-07 | — | Pending |
| SCHOL-01 | — | Pending |
| SCHOL-02 | — | Pending |
| SCHOL-03 | — | Pending |
| SCHOL-04 | — | Pending |
| SCHOL-05 | — | Pending |
| SCHOL-06 | — | Pending |
| PIPE-01 | — | Pending |
| PIPE-02 | — | Pending |
| PIPE-03 | — | Pending |
| PIPE-04 | — | Pending |
| PIPE-05 | — | Pending |
| PIPE-06 | — | Pending |
| ADV-01 | — | Pending |
| ADV-02 | — | Pending |
| ADV-03 | — | Pending |
| PROOF-01 | — | Pending |

**Coverage:**
- v1 requirements: 34 total
- Mapped to phases: 0
- Unmapped: 34

---
*Requirements defined: 2026-02-09*
*Last updated: 2026-02-09 after initial definition*
