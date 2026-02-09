# Project Research Summary

**Project:** GhostWriter Knowledge Infrastructure
**Domain:** RAG-powered content generation with multi-source knowledge integration
**Researched:** 2026-02-09
**Confidence:** HIGH

## Executive Summary

GhostWriter is adding a knowledge layer (vector DB, Gmail newsletters, academic papers, RAG) to an existing working multi-agent content pipeline. The research reveals a critical architectural principle: **RAG must operate as tool-based retrieval invoked by agents, not as a separate service layer that couples to the hot path.** This preserves the existing 3-layer architecture (Directives → Orchestration → Execution) while adding structured knowledge access.

The recommended approach is **Modular RAG with hybrid search (semantic + BM25), pgvector + OpenAI embeddings, and LangGraph tool-based retrieval**. Start minimal: single agent (FactVerificationAgent), single source (Gmail newsletters), pgvector embedded database. This avoids the primary failure mode: breaking what already works. The current pipeline generates articles in ~3 seconds with deterministic quality gates. Adding RAG introduces 5-7 new failure points and potential 250-700ms latency accumulation. The key insight is **95% accuracy per layer = 81% reliability over 5 layers** (compounding failure is the enemy).

Critical risks center on integration, not technology: (1) breaking the existing pipeline with poorly integrated RAG, (2) hallucination amplification from bad chunking/retrieval, (3) Gmail API quota exhaustion silently breaking ingestion, (4) Semantic Scholar rate limits blocking research. Prevention requires feature flags per agent, async retrieval with timeouts, circuit breakers, and semantic chunking from day one. The architecture must prioritize resilience: if RAG fails, fall back to the working pipeline.

## Key Findings

### Recommended Stack

**Start with cost-efficiency and Python integration, defer specialized infrastructure until proven bottleneck.** PostgreSQL with pgvector handles tens of millions of vectors on single-node deployments (OpenAI uses PostgreSQL for 800M ChatGPT users). Don't introduce managed services until your existing database becomes a constraint.

**Core technologies:**
- **PostgreSQL + pgvector (0.8.0+)**: Vector storage and similarity search — Leverage existing SQL knowledge, eliminate new infrastructure, 60-75% lower costs than managed vector DBs. HNSW indexing in 0.8.0 delivers 9x faster queries. Proven at massive scale.
- **OpenAI text-embedding-3-small**: Convert text to 1536-dim vectors — Best cost/performance ratio at $0.02/1M tokens (standard) or $0.01/1M (batch). Supports dimension truncation. 8x cheaper storage than text-embedding-3-large, outperforms Cohere embed-v3 on nDCG benchmarks.
- **LangGraph + LangChain Tools**: RAG orchestration — Already in your stack. Don't introduce LlamaIndex unless you need specialized document parsing. Use LangChain's retriever abstractions. LangGraph excels at agentic workflows with state management.
- **Gmail API (google-api-python-client)**: Newsletter ingestion — Official Google library, OAuth 2.0, batch operations. Rate limits: 1.2M units/min (project), 15K units/min (per-user). Use history API for incremental sync.
- **Semantic Scholar SDK**: Academic paper integration — Official SDK handles rate limiting. Authenticated: 1 req/sec (scales with usage). Use bulk search (less resource-intensive than relevance search).

**Migration path:**
- **Phase 1 (MVP)**: pgvector + OpenAI embeddings + LangGraph → triggers at 10M vectors OR p95 latency >200ms
- **Phase 2 (Optimization)**: Enable pgvectorscale, consider read replicas, add Redis caching → triggers at 50M vectors OR costs >$500/month
- **Phase 3 (Scale)**: Evaluate Qdrant (self-hosted) or AWS Bedrock KB + S3 Vectors → triggers at 100M+ vectors OR need zero-ops

### Expected Features

**Must have (table stakes):**
- **Semantic search with hybrid fallback** — Users expect natural language queries. Hybrid search (semantic + BM25) is 48% better than pure vector (2026 standard).
- **Fine-grained citations** — Sentence-level source links with metadata. Credibility requirement; RAG without citations = hallucination risk.
- **Newsletter auto-ingestion** — Gmail API → vector DB. Trend detection differentiation (no competitor does newsletter → content pipeline).
- **Academic paper grounding** — Semantic Scholar API → depth differentiation for technical content.
- **Brand voice consistency** — Integrate with existing tone system. Generated content must match author voice.
- **Cross-source synthesis** — Combine newsletter trends + papers + web → unified narrative. NotebookLM does multi-doc; GhostWriter does multi-source-type.

**Should have (competitive):**
- **Adaptive voice learning** — Track user edits → refine prompts. Copy.ai has static training; adaptive = differentiation.
- **Context-aware suggestions** — Proactive knowledge surfacing during writing (Superhuman/Shortwave pattern). Workflow integration.
- **Source recency scoring** — Prioritize documents <6 months old for trend-sensitive content. Critical for tech domains.
- **Hallucination detection** — Flag unsupported claims against knowledge base. Complements existing adversarial review panel.

**Defer (v2+):**
- **Knowledge graphs** — Multi-hop reasoning. Complex infrastructure, low ROI for MVP. Defer to Milestone 3.
- **Audio briefings** — Daily podcast-style summaries. Nice-to-have, not core workflow.
- **Visual knowledge maps** — Mind maps, concept graphs. Fits future visual content milestone, not MVP.
- **Custom knowledge personas** — Different "expert" modes per knowledge subset. NotebookLM 2026 feature, but extends existing tone system (defer).

### Architecture Approach

**Modular RAG with tool-based retrieval preserves GhostWriter's working pipeline.** The knowledge layer exposes search/verify/research as LangGraph tools that agents invoke when needed, rather than pre-briefing all agents with context. This enables selective retrieval (FactResearcherAgent queries frequently, EditorAgent rarely), multiple retrieval types (broad search vs precise verification), and token efficiency (only retrieve when agent determines necessity).

**Major components:**
1. **RetrievalTools (LangGraph tools)** — Expose `search_knowledge()`, `verify_claim()`, `research_topic()` as agent-callable tools. Return ranked documents with metadata.
2. **VectorStore (pgvector)** — Store embeddings with namespace isolation: `{source_type}_{tenant_id}`. Hybrid index (dense vector + BM25 sparse + metadata filters).
3. **IngestionPipeline (batch: daily/hourly)** — EmailSource (Gmail) → PaperSource (Semantic Scholar) → RSSSource (existing) → chunk (semantic/recursive hybrid) → embed (batch 32 docs) → store.
4. **QueryTransformer (pre-retrieval)** — Rewrite queries for specificity, expand to semantic variants, HyDE (hypothetical document embeddings) for better recall.
5. **Reranker (post-retrieval)** — CrossEncoder model scores query-doc pairs, repositions most relevant docs at start/end (mitigates "lost in the middle").

**Data flow:**
- **Ingestion**: Source polling → extraction → chunking (512-1024 tokens, 10-20% overlap) → embedding (batch, local model) → storage (namespace per source)
- **Retrieval**: Agent tool call → query transformation → hybrid search (dense + sparse, RRF fusion) → reranking → top-5 results → agent synthesis

**Integration with existing pipeline:**
```
[research] ──> [tools] (research_topic)
    ↓
[writer] ──> [tools] (search_knowledge, optional)
    ↓
[fact_verify] ──> [tools] (verify_claim, frequent)
```
Tools run in parallel ToolNode, return to agent, no blocking on hot path. If vector search fails, agents fall back to existing behavior.

### Critical Pitfalls

1. **Breaking the existing pipeline with RAG integration** — RAG bolted to hot path without isolation causes 3-5x latency increase, quality gate scores drop, fact verification false negatives. **Prevention:** Feature flag RAG per agent, async retrieval with 500ms timeout, context quality gate (>0.7 similarity threshold), circuit breaker (disable RAG if >1s latency or >10% error rate). **Severity:** CRITICAL — can break entire working pipeline.

2. **Hallucination amplification through bad retrieval (garbage in, garbage out)** — Fixed-length chunking splits concepts mid-sentence, losing coherence. Retrieved irrelevant docs poison generation. 95% accuracy per layer → 81% over 5 layers. **Prevention:** Semantic chunking (preserve meaning), chunk context headers (`[Date | Title | Section]`), reranking with cross-encoder, metadata filtering (date/source/relevance >0.7), track retrieval precision@k (alert <70%). **Severity:** CRITICAL — hallucinations destroy trust.

3. **Multi-tenant data leakage in vector search** — Vector similarity is global; without namespace isolation, User A's query returns User B's private newsletters/papers. GDPR/CCPA violations. **Prevention:** Namespace-based isolation (`{source}_{tenant_id}`), mandatory tenant_id filter in wrapper function, cross-tenant integration tests, audit logs with tenant_id. Design for single-tenant first, add namespace parameter later (Phase 6). **Severity:** CRITICAL — data breach, legal liability.

4. **Gmail API quota exhaustion & OAuth token failures** — Rate limits (100 req/sec shared project-wide), OAuth refresh tokens silently invalidate after 6 months inactivity or 100 token limit per account. Newsletter ingestion stops silently. **Prevention:** Exponential backoff with retry, batch API (100 messages/request), proactive token refresh (<5min expiry), token limit monitoring, surface UI notification for `invalid_grant` errors. **Severity:** CRITICAL — OAuth failures permanent until re-auth.

5. **Semantic Scholar API rate limits & citation graph explosions** — Authenticated: 1 req/sec. Citation graphs grow exponentially (1 paper → 50 citations → 2,500 second-order). Naive fetching hits limits instantly. **Prevention:** Batch endpoints (50-100 IDs/request), exponential backoff required, citation depth limiting (1st-order only unless user requests), prioritize recent papers, field filtering, cache metadata (30-day TTL). **Severity:** CRITICAL — blocks research ingestion entirely.

## Implications for Roadmap

Based on research, suggested phase structure prioritizes **foundation → integration → scaling**, with clear triggers for complexity. Start minimal (single agent, single source, embedded DB), prove value, scale incrementally.

### Phase 1: Foundation (Vector DB + Embedding Setup)
**Rationale:** Establish core RAG infrastructure with simplest possible stack. Embedded pgvector avoids operational complexity, local embeddings avoid API costs. No agents yet — validate retrieval quality first.
**Delivers:** Working vector DB with sample data, embedding pipeline, similarity search API.
**Addresses:** Vector storage (STACK), semantic search (FEATURES), chunking strategy (PITFALLS #2).
**Avoids:** Wrong DB for scale (PITFALL #13) — start with pgvector, migrate later if needed. Embedding dimension mismatches (PITFALL #9) — document model version upfront.
**Research needed:** Minimal — pgvector + OpenAI embeddings are well-documented. Test chunking strategies (semantic vs fixed-length) with GhostWriter's actual newsletter content.

### Phase 2: RAG Integration Layer
**Rationale:** Add retrieval tools to pipeline without coupling to agents yet. Build the tool interface that agents will call. Enables A/B testing of retrieval quality.
**Delivers:** LangGraph tools (`search_knowledge`, `verify_claim`, `research_topic`), hybrid search (dense + BM25), reranking, query transformation.
**Uses:** LangGraph ToolNode (STACK), hybrid retrieval pattern (ARCHITECTURE), cross-encoder reranker (ARCHITECTURE).
**Implements:** RetrievalTools component (ARCHITECTURE), QueryTransformer (ARCHITECTURE).
**Avoids:** Hallucination amplification (PITFALL #2) — semantic chunking + reranking from start. Stale embeddings (PITFALL #10) — TTL on vectors, incremental re-indexing.
**Research needed:** Minimal — LangChain retrieval abstractions are standard. May need phase-specific research on optimal reranking thresholds.

### Phase 3: Agent Integration
**Rationale:** Connect retrieval tools to one agent first (FactVerificationAgent), measure impact, expand gradually. This avoids breaking the working pipeline.
**Delivers:** FactVerificationAgent with knowledge base verification, ArticleState extension for tracking queries/results, BaseAgent.call_llm_with_tools() method.
**Addresses:** Knowledge-backed fact checking (FEATURES), adaptive routing (ARCHITECTURE), tool-based retrieval (ARCHITECTURE).
**Avoids:** Breaking existing pipeline (PITFALL #1) — feature flag RAG, async with timeout, circuit breaker. Lost in the middle (PITFALL #7) — rerank docs, limit to top-5. RAG latency accumulation (PITFALL #12) — selective RAG, connection pool tuning.
**Research needed:** Yes — phase-specific research on agent-specific retrieval patterns. Which agents benefit from retrieval? What prompts optimize tool usage? A/B test required.

### Phase 4: Gmail Ingestion
**Rationale:** Newsletter intelligence is core differentiation. Build ingestion before scaling to other sources. Allows validation of newsletter → article pipeline.
**Delivers:** EmailSource with Gmail API integration, MIME parsing, HTML extraction (AI-powered), batch ingestion (daily), OAuth token management.
**Addresses:** Newsletter auto-ingestion (FEATURES), trend detection (FEATURES differentiator).
**Avoids:** Gmail API quota exhaustion (PITFALL #4) — batch API, exponential backoff, token refresh automation. HTML parsing inconsistencies (PITFALL #6) — AI extraction, encoding normalization, structure preservation.
**Research needed:** Yes — phase-specific research on Gmail API batch limits with realistic volumes (test 100, 1K, 10K emails). Newsletter format diversity testing required.

### Phase 5: Semantic Scholar Integration
**Rationale:** Academic grounding is second differentiation axis. Build after newsletter ingestion to learn from first source integration.
**Delivers:** PaperSource with Semantic Scholar API, batch endpoints, exponential backoff, citation depth limiting, metadata caching (30-day TTL).
**Addresses:** Academic paper grounding (FEATURES), depth differentiation (FEATURES).
**Avoids:** Rate limits & citation explosions (PITFALL #5) — batch endpoints (50-100 IDs), 1 req/sec respected, 1st-order citations only. Embedding costs spiraling (PITFALL #11) — content filtering, incremental indexing, cheaper models.
**Research needed:** Yes — phase-specific research on Semantic Scholar batch endpoint reliability with high-citation papers (>500 citations). Test rate limit enforcement patterns.

### Phase 6: Multi-Tenant Preparation
**Rationale:** Design for multi-tenancy early, implement late. By Phase 6, you have real usage patterns to inform isolation strategy.
**Delivers:** TenantAwareVectorStore wrapper, namespace-based collection naming (`{source}_{tenant_id}`), cross-tenant isolation tests, per-tenant quotas.
**Addresses:** Future scaling (not in FEATURES — design concern), namespace isolation (ARCHITECTURE).
**Avoids:** Multi-tenant data leakage (PITFALL #3) — namespace isolation, mandatory filters, audit logs. Over-engineering too early (PITFALL #14) — single-tenant first, add namespace parameter incrementally.
**Research needed:** Minimal — namespace patterns are well-documented. Integration tests validate isolation.

### Phase Ordering Rationale

- **Phase 1 before Phase 2**: Can't integrate retrieval without vector DB. Foundation must be solid before building tools.
- **Phase 2 before Phase 3**: Tools must exist before agents call them. Decoupling tools from agents enables independent testing.
- **Phase 3 before Phase 4/5**: Validate RAG integration with one agent before adding data sources. Derisk pipeline changes early.
- **Phase 4 before Phase 5**: Newsletter ingestion is simpler than paper ingestion (no citation graphs). Learn source integration patterns with easier source.
- **Phase 6 last**: Don't over-engineer multi-tenancy until you have users. Real usage informs isolation requirements.

**Dependency chain:**
```
Phase 1 (Vector DB) → Phase 2 (Tools) → Phase 3 (Agents)
                                            ↓
                                    Phase 4 (Gmail) ──→ Phase 5 (Papers)
                                                              ↓
                                                        Phase 6 (Multi-tenant)
```

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 3 (Agent Integration)**: Complex integration with existing LangGraph pipeline. Need phase-specific research on agent-specific retrieval patterns (which agents benefit? what prompts work?). A/B testing required.
- **Phase 4 (Gmail Ingestion)**: Newsletter format diversity is high. Need phase-specific research on parsing strategies across Substack, Beehiiv, ConvertKit, custom templates. Gmail API quota testing at realistic volumes required.
- **Phase 5 (Semantic Scholar)**: Citation graph traversal strategies unclear. Need phase-specific research on batch endpoint behavior with high-citation papers. Rate limit enforcement patterns need validation.

Phases with standard patterns (skip research-phase):
- **Phase 1 (Vector DB)**: pgvector + OpenAI embeddings are well-documented. Installation, configuration, similarity search are standard operations. No phase-specific research needed.
- **Phase 2 (Tools)**: LangChain retrieval abstractions and LangGraph ToolNode are mature. Hybrid search (BM25 + vector) has established patterns. No phase-specific research needed beyond implementation.
- **Phase 6 (Multi-tenant)**: Namespace isolation is standard vector DB pattern. Cross-tenant tests validate correctness. No novel research needed.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | **HIGH** | pgvector proven at massive scale (OpenAI 800M users), OpenAI embeddings widely adopted, LangGraph already in stack. Extensive benchmarks confirm performance. Official docs for Gmail/Semantic Scholar APIs. |
| Features | **HIGH** | Table stakes features have industry consensus (hybrid search, citations). Differentiators validated via competitive analysis (no competitor does newsletter → content). Anti-features backed by PKM/RAG practitioner lessons. |
| Architecture | **HIGH** | Modular RAG pattern is 2026 best practice for agentic workflows. Tool-based retrieval aligns with existing LangGraph usage. Component boundaries clear from research + understanding of GhostWriter pipeline. |
| Pitfalls | **HIGH** | Critical pitfalls sourced from official docs (Gmail/S2 APIs), production case studies, academic research (lost in the middle). Integration risks informed by understanding existing pipeline. Phase-specific warnings grounded in practitioner experience. |

**Overall confidence:** **HIGH**

Research methodology combined ecosystem discovery (2026 sources via WebSearch), official documentation (AWS, Google, Semantic Scholar via WebFetch), cross-referenced benchmarks/pricing, and architectural analysis of existing GhostWriter codebase. All critical recommendations verified with primary sources. Medium confidence areas (email parsing libraries, Semantic Scholar SDK maturity) noted explicitly.

### Gaps to Address

Areas where research was inconclusive or needs validation during implementation:

- **Optimal chunking strategy for newsletter content**: Newsletters vary widely (plain text, HTML with complex layouts, embedded links). Semantic chunking works for prose, but newsletter-specific patterns (sponsor sections, CTAs, multiple articles per email) need experimentation. **Handle during Phase 1** — test with 20+ diverse newsletters, measure retrieval precision@k for different chunking strategies (semantic, recursive, hybrid).

- **Gmail API quota consumption at scale**: Official docs specify rate limits (1.2M units/min project-wide, 15K units/min per-user), but actual consumption patterns depend on batch size, message size, attachment handling. Unclear if 10K newsletter ingestion takes 10 minutes or 60 minutes. **Handle during Phase 4** — load test with 100, 1K, 10K emails, measure quota consumption and latency. Adjust batch sizes accordingly.

- **Semantic Scholar batch endpoint reliability**: Docs say batch endpoints exist, but behavior with large batches (100+ paper IDs) unclear. Does it timeout? Throttle? Return partial results? **Handle during Phase 5** — test with papers having >500 citations, measure success rate and latency for batch fetching.

- **Agent-specific retrieval impact**: Unknown which agents actually benefit from RAG. WriterAgent likely improves with context. CriticAgent might get confused. EditorAgent uncertain. **Handle during Phase 3** — A/B test retrieval impact per agent. Measure quality_score, iterations_used, adversarial_panel scores with vs without retrieval. Only enable for agents with measurable improvement.

- **Hybrid search weight tuning**: Optimal balance between semantic search and BM25 keyword matching depends on content type (trend-focused newsletters favor recency/keywords, depth-focused papers favor semantics). Default RRF (Reciprocal Rank Fusion) weights may not match GhostWriter's content distribution. **Handle during Phase 2** — test semantic vs keyword weights with representative queries, measure recall@10 and precision@5. Tune weights based on content type (newsletters vs papers).

- **Multi-tenant overhead at scale**: Namespace filtering adds query latency, but magnitude unknown. Is it 5ms or 50ms per query? Pinecone/Weaviate docs claim "negligible" but real-world testing needed. **Handle during Phase 6** — benchmark query latency with namespace filtering vs global search. If overhead >20ms, consider architecture changes (separate indexes per tenant).

## Sources

### Primary (HIGH confidence)
- **STACK.md** — Technology stack research including vector databases (pgvector benchmarks, pricing, migration paths), embedding models (OpenAI/Cohere/nomic comparisons, cost analysis), RAG frameworks (LangGraph vs LlamaIndex tradeoffs), Gmail/Semantic Scholar APIs (rate limits, quota costs, authentication patterns). 45+ sources including official docs (AWS, Google, Semantic Scholar), benchmarks (pgvectorscale, TigerData), pricing calculators.
- **FEATURES.md** — Feature landscape research covering table stakes (semantic search, citations, document ingestion, voice consistency), differentiators (newsletter intelligence, academic integration, adaptive learning), anti-features (manual entry, keyword-only search, fixed chunking). 30+ sources including Notion AI, NotebookLM, Jasper, Copy.ai, Elicit, competitive positioning matrix.
- **ARCHITECTURE.md** — Architecture patterns research including Modular RAG (vs Naive/Advanced), hybrid search (dense + sparse + reranking), ingestion pipeline stages (fetch → extract → chunk → embed → store), LangGraph tool-based retrieval, multi-tenant namespace isolation. 25+ sources including RAG evolution (MarkTechPost, Arxiv), LangGraph tutorials, chunking strategies (Weaviate, Databricks), embedding models.
- **PITFALLS.md** — Domain pitfalls research covering 15 pitfalls (6 critical, 6 important, 3 moderate) with prevention strategies, detection methods, phase-specific warnings. Includes compounding failure analysis (95% per layer → 81% over 5 layers), latency accumulation (250-700ms), cost spiraling ($10/month → $2K/month), OAuth token failures. 35+ sources including production case studies, Gmail/S2 API official docs, security research (Weaviate multi-tenancy), cost analyses.

### Secondary (MEDIUM confidence)
- Email parsing libraries (mail-parser, email-reply-parser) — Functionality documented but less battle-tested than Gmail API itself. Verified via PyPI docs and GitHub repos (SpamScope, Zapier). English-only limitation of email-reply-parser reduces confidence for multilingual use cases.
- Semantic Scholar SDK maturity — Official SDK exists but ecosystem less mature than OpenAI/Google. Rate limit enforcement patterns may vary (community reports vs official docs). Verified via official API tutorial and GitHub issues.
- AWS Bedrock KB + S3 Vectors (future migration option) — Recently GA'd (Dec 2025), pricing confirmed but limited production case studies. Strong technical foundation but needs time-in-market validation. Verified via AWS official docs and blog posts.

### Tertiary (LOW confidence)
- None — all research backed by primary or secondary sources. No single-source claims or unverified inferences in recommendations.

---
*Research completed: 2026-02-09*
*Ready for roadmap: yes*
