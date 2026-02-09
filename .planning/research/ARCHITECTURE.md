# Architecture Patterns: Knowledge Layer Integration

**Project:** GhostWriter Knowledge Infrastructure
**Domain:** RAG + Vector DB + Multi-Agent Content Pipeline
**Researched:** 2026-02-09
**Confidence:** HIGH

## Executive Summary

Adding a knowledge layer to an existing multi-agent content pipeline requires careful architectural decisions around retrieval patterns, ingestion flows, and agent integration points. Based on 2026 industry patterns, the recommended approach is **Modular RAG with hybrid search, LangGraph tool-based retrieval, and namespace-isolated vector storage**.

**Key architectural principle:** The knowledge layer should operate as *retrieval tools* that agents invoke via LangGraph's tool-calling mechanism, not as a separate service layer. This preserves GhostWriter's existing 3-layer architecture (Directives → Orchestration → Execution) while adding structured knowledge access.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      EXISTING: LangGraph Pipeline                    │
│  ┌──────┐   ┌────────┐   ┌────────┐   ┌────────┐   ┌──────────┐  │
│  │Writer│──>│ Editor │──>│ Critic │──>│ Style  │──>│ Quality  │  │
│  │Agent │   │ Agent  │   │ Agent  │   │Enforcer│   │   Gate   │  │
│  └───┬──┘   └────────┘   └────────┘   └────────┘   └──────────┘  │
│      │                                                               │
│      └─────────> BaseAgent (LLM provider routing)                  │
└─────────────────────────────────────────────────────────────────────┘
                               ▲
                               │ (NEW: tool_calls)
                               │
┌──────────────────────────────┴───────────────────────────────────────┐
│                    NEW: Knowledge Layer (Execution)                   │
│                                                                        │
│  ┌────────────────────────┐          ┌─────────────────────────┐   │
│  │  Retrieval Tools       │          │  Ingestion Pipeline     │   │
│  │  ────────────────      │          │  ──────────────────     │   │
│  │  • search_knowledge()  │◄─────────┤  • EmailSource          │   │
│  │  • verify_claim()      │          │  • PaperSource          │   │
│  │  • research_topic()    │          │  • RSSSource            │   │
│  └───────────┬────────────┘          │  • WebSource            │   │
│              │                        └────────┬────────────────┘   │
│              │                                 │                     │
│              ▼                                 ▼                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Vector Database (Chroma/Qdrant)                 │   │
│  │  ─────────────────────────────────────────────────────       │   │
│  │  Collections (namespace isolation):                          │   │
│  │    • emails_default         (Gmail ingestion)                │   │
│  │    • papers_default         (Semantic Scholar)               │   │
│  │    • rss_default            (HN, RSS feeds)                  │   │
│  │    • web_default            (Scraped content)                │   │
│  │                                                               │   │
│  │  Index structure: Hybrid (dense + sparse + reranking)        │   │
│  └───────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────┘
```

## Component Boundaries

### 1. Knowledge Layer Components

| Component | Responsibility | Input | Output | Dependencies |
|-----------|---------------|-------|--------|--------------|
| **RetrievalTools** | Expose search/verify/research as LangGraph tools | Query string, filters | Ranked documents + metadata | VectorStore, EmbeddingService |
| **VectorStore** | Store/retrieve embeddings with metadata | Embedding vector, filters | Ranked results | ChromaDB/Qdrant |
| **EmbeddingService** | Generate embeddings for queries/documents | Text content | Embedding vector (384-1536 dim) | sentence-transformers/OpenAI API |
| **IngestionPipeline** | Extract → Chunk → Embed → Store | Raw content (email/PDF/RSS) | Stored chunks with metadata | Sources, Chunker, EmbeddingService |
| **QueryTransformer** | Rewrite/expand queries for better retrieval | Original query | Transformed queries (1-5 variants) | LLM (lightweight) |
| **Reranker** | Score and rerank retrieved results | Query + candidate docs | Reranked results | CrossEncoder model |

### 2. Integration with Existing Architecture

**BaseAgent Extension Pattern:**
```python
class BaseAgent:
    # EXISTING: LLM provider routing
    def call_llm(self, prompt, system_instruction=None):
        ...

    # NEW: Tool-based knowledge access
    def call_llm_with_tools(self, prompt, tools: List[Tool], system_instruction=None):
        """Call LLM with retrieval tools enabled."""
        # LangGraph ToolNode integration
        # Agent can invoke search_knowledge, verify_claim, research_topic
        ...
```

**Agent-Initiated Retrieval Pattern (Recommended):**
- Agents decide when retrieval is needed
- LLM receives tool definitions: `search_knowledge()`, `verify_claim()`, `research_topic()`
- LLM makes tool calls when needed
- ToolNode executes retrieval, returns results to agent
- Agent synthesizes retrieved context into response

**Alternative: Pre-Briefing Pattern (Not Recommended for GhostWriter):**
- System retrieves context before agent invocation
- Context injected into system prompt
- Pro: Simpler, no tool calling logic
- Con: Retrieves even when not needed, wastes tokens/cost

**Why Tool-Based for GhostWriter:**
1. **Selective retrieval**: FactResearcher needs frequent retrieval, Writer needs occasional retrieval, Editor rarely needs retrieval
2. **Multiple retrieval types**: `search_knowledge()` (broad), `verify_claim()` (precise fact-checking), `research_topic()` (deep background)
3. **Token efficiency**: Only retrieves when agent determines it's needed
4. **Existing pattern**: GhostWriter agents already use multi-provider routing; tool calls extend this pattern

### 3. Data Flow: Source → Embedding → Storage → Retrieval → Generation

```
INGESTION FLOW (Batch: Daily for Gmail, Hourly for RSS/HN)
───────────────────────────────────────────────────────────

1. SOURCE POLLING
   EmailSource.fetch() → Raw email (subject, body, attachments)
   PaperSource.fetch() → Semantic Scholar metadata + abstract
   RSSSource.fetch() → Article content from feeds
   WebSource.scrape() → HTML content

2. DOCUMENT EXTRACTION
   • Email: Parse MIME, extract body/attachments, detect metadata
   • PDF: Parseur/PyPDF2 → text extraction with layout preservation
   • HTML: BeautifulSoup → clean text, preserve structure

3. CHUNKING (Semantic/Recursive Hybrid)
   • Semantic chunking: Group by topic shifts (embedding similarity)
   • Recursive chunking: Split by section → paragraph → sentence
   • Target: 512-1024 tokens per chunk (with 10-20% overlap)
   • Metadata enrichment: Source, date, author, content_type, parent_doc_id

4. EMBEDDING GENERATION (Batch)
   • Model: sentence-transformers/all-mpnet-base-v2 (768-dim, local, free)
   • Fallback: OpenAI text-embedding-3-small (1536-dim, API, paid)
   • Batch size: 32 documents per embedding call (optimize for throughput)

5. VECTOR STORAGE
   • Store in collection: {source_type}_{tenant}
   • Index: Hybrid (dense vector + BM25 sparse + metadata filters)
   • Metadata: source_type, content_type, timestamp, author, url, parent_id

RETRIEVAL FLOW (Real-time during article generation)
─────────────────────────────────────────────────────

1. AGENT INVOKES RETRIEVAL TOOL
   WriterAgent: search_knowledge("Anthropic Claude prompt caching")
   FactVerificationAgent: verify_claim("Claude 4.6 supports 2M token context")

2. QUERY TRANSFORMATION (Pre-retrieval)
   • Query rewriting: Make query more specific
   • Query expansion: Generate 2-3 semantic variants
   • HyDE (optional): Generate hypothetical answer, embed that

   Example:
   Original: "Claude prompt caching"
   Rewritten: "How does Anthropic Claude implement prompt caching?"
   Expanded: ["Claude cache_control API", "prompt caching token costs",
              "ephemeral prompt caching implementation"]

3. HYBRID RETRIEVAL
   • Dense search: Embed query → cosine similarity (top 100)
   • Sparse search: BM25 keyword matching (top 100)
   • Fusion: RRF (Reciprocal Rank Fusion) → top 50 candidates
   • Metadata filter: date range, source_type, content_type

4. RERANKING (Post-retrieval)
   • Model: CrossEncoder (e.g., ms-marco-MiniLM-L-12-v2)
   • Input: (query, candidate_doc) pairs
   • Output: Relevance score per document
   • Select: Top 5-10 most relevant chunks

5. CONTEXT PACKAGING
   • Format: JSON with metadata
   • Include: chunk_text, source, date, relevance_score, parent_doc_id
   • Return to agent via ToolMessage

6. AGENT SYNTHESIS
   • Agent receives retrieved context
   • Decides: Use context, request more, or proceed without
   • Generates response using context + LLM knowledge
```

## RAG Architecture Patterns

### Pattern Comparison: Naive vs Advanced vs Modular RAG

| Dimension | Naive RAG | Advanced RAG | Modular RAG (Recommended) |
|-----------|-----------|--------------|---------------------------|
| **Retrieval** | Simple vector similarity | Hybrid search (dense + sparse) | Hybrid + multiple retrievers (vector, graph, keyword) |
| **Query Processing** | Embed query as-is | Query rewriting | Query transformation pipeline (rewrite, expand, HyDE) |
| **Post-Retrieval** | Return top-k | Reranking with CrossEncoder | Multi-stage reranking + filtering |
| **Agent Integration** | Fixed pipeline | Fixed pipeline with better retrieval | Tool-based, agent-initiated |
| **Memory** | Stateless | Stateless | Stateful (LangGraph checkpoints) |
| **Routing** | All queries → same path | All queries → same path | Adaptive routing (parametric vs retrieval vs web) |
| **Complexity** | Low | Medium | High |
| **GhostWriter Fit** | Too basic | Partial fit | **Best fit** |

**Why Modular RAG for GhostWriter:**
1. **Agents need different retrieval patterns**: FactResearcher (deep search), Writer (background context), Editor (style references)
2. **Multiple knowledge sources**: Emails, papers, RSS, web → different retrieval strategies
3. **Adaptive routing**: Simple queries skip retrieval, complex queries trigger deep search
4. **LangGraph alignment**: Modular RAG is built for StateGraph workflows
5. **Future extensibility**: Add graph RAG, web search, external APIs as modules

### Hybrid Search Architecture (Dense + Sparse + Reranking)

**Why Hybrid Outperforms Pure Vector Search:**
- Dense embeddings: Capture semantic meaning, but miss exact keyword matches
- Sparse embeddings (BM25): Capture exact terms, but miss semantics
- Hybrid: Combine both via RRF → **48% better performance** than dense-only (Pinecone 2025 benchmark)

**Implementation Strategy:**
```python
# Hybrid retrieval with RRF fusion
def hybrid_search(query: str, top_k: int = 10) -> List[Document]:
    # Dense retrieval
    query_embedding = embed_model.encode(query)
    dense_results = vector_store.similarity_search(
        query_embedding, top_k=100
    )

    # Sparse retrieval (BM25)
    sparse_results = bm25_index.search(query, top_k=100)

    # Reciprocal Rank Fusion
    fused_results = rrf_fusion(dense_results, sparse_results)

    # Reranking with CrossEncoder
    reranked = reranker.rerank(query, fused_results[:50])

    return reranked[:top_k]
```

**Vector Database Support for Hybrid Search:**
| Database | Dense | Sparse (BM25) | Hybrid API | Reranking |
|----------|-------|---------------|------------|-----------|
| **Chroma** | ✅ | ⚠️ Manual | ⚠️ Manual fusion | ⚠️ External |
| **Qdrant** | ✅ | ✅ Built-in | ✅ Built-in | ✅ Built-in |
| **Weaviate** | ✅ | ✅ Built-in | ✅ Built-in | ✅ Built-in |
| **Pinecone** | ✅ | ✅ Built-in | ✅ Built-in | ⚠️ External |

**Recommendation for GhostWriter:**
- **Phase 1 (MVP)**: Chroma (embedded, no server) + manual BM25 via `rank-bm25` library + CrossEncoder reranking
- **Phase 2 (Scale)**: Migrate to Qdrant (native hybrid, multi-tenancy, better performance)

## Ingestion Pipeline Architecture

### Pipeline Stages

```
┌─────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                        │
│                                                               │
│  1. FETCH                                                    │
│     ├─> EmailSource.fetch() → Gmail API                     │
│     ├─> PaperSource.fetch() → Semantic Scholar API          │
│     ├─> RSSSource.fetch() → feedparser                      │
│     └─> WebSource.scrape() → BeautifulSoup                  │
│                                                               │
│  2. EXTRACT (Content Parsing)                               │
│     ├─> Email: MIME parsing, attachment extraction          │
│     ├─> PDF: Parseur/PyPDF2/pdfplumber                      │
│     ├─> HTML: Trafilatura/BeautifulSoup                     │
│     └─> Plain text: Direct ingestion                        │
│                                                               │
│  3. NORMALIZE                                                │
│     ├─> Detect language (langdetect)                        │
│     ├─> Extract metadata (title, author, date, url)         │
│     ├─> Clean text (remove boilerplate, ads, navigation)    │
│     └─> Structure detection (headings, lists, code blocks)  │
│                                                               │
│  4. CHUNK (Semantic + Recursive Hybrid)                     │
│     ├─> Recursive: Section → Paragraph → Sentence           │
│     ├─> Semantic: Group by topic shifts (embed similarity)  │
│     ├─> Target: 512-1024 tokens/chunk, 10-20% overlap       │
│     └─> Metadata: parent_id, position, hierarchy_level      │
│                                                               │
│  5. ENRICH (Metadata Enhancement)                           │
│     ├─> Content type classification (tutorial, news, doc)   │
│     ├─> Entity extraction (NER: people, orgs, tech)         │
│     ├─> Keyword extraction (RAKE, YAKE, LLM-based)          │
│     └─> Summary generation (LLM: 2-sentence summary)        │
│                                                               │
│  6. EMBED (Batch Generation)                                │
│     ├─> Model: sentence-transformers/all-mpnet-base-v2      │
│     ├─> Batch: 32 chunks per API call                       │
│     ├─> Cache: Store embeddings with content hash           │
│     └─> Retry: Handle transient errors, skip bad chunks     │
│                                                               │
│  7. STORE (Vector DB + Metadata)                            │
│     ├─> Collection: {source_type}_{tenant}                  │
│     ├─> Vector: Dense embedding (768-dim)                   │
│     ├─> Metadata: source, date, author, content_type, etc.  │
│     └─> Index: HNSW for dense, inverted index for sparse    │
│                                                               │
│  8. VERIFY (Quality Check)                                  │
│     ├─> Test retrieval: Sample query → expect result        │
│     ├─> Embedding quality: Cosine similarity sanity check   │
│     └─> Log: Ingestion metrics (count, errors, duration)    │
└─────────────────────────────────────────────────────────────┘
```

### Chunking Strategy: Semantic + Recursive Hybrid

**Recursive Chunking (Structure-Based):**
- Split by document structure: sections → paragraphs → sentences
- Preserve hierarchy with metadata: `{parent_id, hierarchy_level, position}`
- Example:
  ```
  Document: "# Introduction\n\nText...\n\n## Background\n\nMore text..."
  Chunks:
    1. Heading: "Introduction" (level=1, parent=None)
    2. Paragraph: "Text..." (level=2, parent=chunk_1)
    3. Heading: "Background" (level=2, parent=chunk_1)
    4. Paragraph: "More text..." (level=3, parent=chunk_3)
  ```

**Semantic Chunking (Meaning-Based):**
- Embed small segments (3-5 sentences)
- Compute cosine similarity between consecutive segments
- Merge segments if similarity > threshold (e.g., 0.8)
- Split segments if similarity < threshold
- Example:
  ```
  Sentences 1-3: Discussing prompt caching (similarity: 0.9) → Merge into chunk_1
  Sentences 4-6: Discussing token limits (similarity: 0.5) → New chunk_2
  ```

**Hybrid Approach (Recommended):**
1. **Start with recursive**: Split document by structure
2. **Validate with semantic**: If chunk is too large (>1024 tokens), apply semantic splitting
3. **Add overlap**: Include 1-2 sentences from previous/next chunk for context
4. **Enrich metadata**: Store hierarchy, parent_id, topic_shift_score

**Implementation:**
```python
def hybrid_chunk(document: str, max_tokens: int = 1024, overlap_pct: float = 0.15):
    # Stage 1: Recursive split by structure
    recursive_chunks = recursive_split(document, separators=["\n\n", "\n", ". "])

    # Stage 2: Semantic validation and refinement
    final_chunks = []
    for chunk in recursive_chunks:
        if token_count(chunk) > max_tokens:
            # Too large → apply semantic split
            sub_chunks = semantic_split(chunk, max_tokens=max_tokens)
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(chunk)

    # Stage 3: Add overlap
    overlapped_chunks = add_overlap(final_chunks, overlap_pct=overlap_pct)

    return overlapped_chunks
```

**Why Hybrid for GhostWriter:**
- **Emails**: Semantic chunking (unstructured conversations)
- **Papers**: Recursive chunking (structured sections: abstract, intro, methods)
- **RSS articles**: Hybrid (some structure, but variable quality)
- **Code snippets**: Recursive with language-aware splitting (AST-based)

### Content Source Handling

| Source Type | Extraction Method | Chunking Strategy | Metadata | Update Frequency |
|-------------|-------------------|-------------------|----------|------------------|
| **Gmail** | Gmail API (MIME parsing) | Semantic (conversation flow) | sender, date, thread_id, labels | Daily batch |
| **Semantic Scholar** | API (JSON) | Recursive (abstract, sections) | authors, citations, venue, year | On-demand |
| **RSS/HN** | feedparser, web scraping | Hybrid | url, published_date, source | Hourly |
| **Web scrapes** | BeautifulSoup, Trafilatura | Hybrid | url, scraped_date, domain | As-needed |
| **Local files** | File I/O (PDF, DOCX, TXT) | Recursive | filename, modified_date | Manual trigger |

**Email Parsing Pipeline:**
```python
class EmailIngestionPipeline:
    def ingest_batch(self, emails: List[GmailMessage]) -> List[Document]:
        documents = []
        for email in emails:
            # 1. Extract
            subject = email.subject
            body = self.extract_body(email)  # Handle HTML/plain text
            attachments = self.extract_attachments(email)  # PDF, images

            # 2. Normalize
            clean_body = self.clean_text(body)  # Remove signatures, quotes

            # 3. Chunk
            chunks = self.semantic_chunk(
                f"Subject: {subject}\n\n{clean_body}",
                max_tokens=1024
            )

            # 4. Enrich
            for i, chunk in enumerate(chunks):
                metadata = {
                    "source": "gmail",
                    "content_type": "email",
                    "sender": email.sender,
                    "date": email.date,
                    "thread_id": email.thread_id,
                    "labels": email.labels,
                    "chunk_index": i,
                    "has_attachments": len(attachments) > 0
                }
                documents.append(Document(text=chunk, metadata=metadata))

            # Process attachments separately
            for att in attachments:
                if att.content_type == "application/pdf":
                    pdf_docs = self.ingest_pdf(att.content, metadata)
                    documents.extend(pdf_docs)

        return documents
```

## Multi-Tenant Data Architecture

### Namespace Isolation Strategy

**Design for Current Single-Tenant, Scale to Multi-Tenant:**

GhostWriter is currently single-tenant (one user). However, the vector DB should be structured to support multi-tenancy without migration.

**Recommended Structure:**
```
Collection naming: {source_type}_{tenant_id}

Examples:
  emails_default       (Current: single tenant)
  papers_default
  rss_default
  web_default

Future multi-tenant:
  emails_user_123
  emails_user_456
  papers_user_123
  papers_user_456
```

**Why Namespace/Collection-Level Isolation:**
1. **Physical isolation**: Each tenant's data in separate collection
2. **Performance**: Queries only scan one tenant's data
3. **Security**: No risk of cross-tenant data leakage
4. **Scalability**: Collections scale independently
5. **Cost**: Easy to track storage/compute per tenant

**Alternative Approaches (Not Recommended for GhostWriter):**

| Approach | Isolation | Performance | Security | Complexity | Verdict |
|----------|-----------|-------------|----------|------------|---------|
| **Single collection + metadata filter** | Logical | Slower (scans all) | Risk of filter bug | Low | ❌ Avoid |
| **Database-level isolation** | Physical | Fast | High | High (manage many DBs) | ❌ Overkill |
| **Collection-level (namespace)** | Physical | Fast | High | Medium | ✅ **Recommended** |
| **Partition-level** | Semi-physical | Medium | Medium | Medium | ⚠️ Milvus-specific |

**Implementation Pattern:**
```python
class TenantAwareVectorStore:
    def __init__(self, client, tenant_id: str = "default"):
        self.client = client
        self.tenant_id = tenant_id

    def get_collection_name(self, source_type: str) -> str:
        return f"{source_type}_{self.tenant_id}"

    def search(self, query: str, source_type: str, top_k: int = 10):
        collection = self.get_collection_name(source_type)
        return self.client.search(
            collection=collection,
            query_embedding=self.embed(query),
            top_k=top_k
        )

    def ingest(self, documents: List[Document], source_type: str):
        collection = self.get_collection_name(source_type)
        # Ensure collection exists
        self.client.create_collection_if_not_exists(collection)
        # Insert documents
        self.client.insert(collection=collection, documents=documents)
```

**Migration Path (Single → Multi-Tenant):**
1. **Phase 1 (Now)**: Use `tenant_id="default"` everywhere
2. **Phase 2**: Add `tenant_id` parameter to ingestion/retrieval functions
3. **Phase 3**: UI for tenant selection (if applicable)
4. **Phase 4**: Billing/quotas per tenant

### Vector Database Selection: Chroma vs Qdrant

| Criterion | Chroma | Qdrant | Verdict |
|-----------|--------|--------|---------|
| **Deployment** | Embedded (no server) | Server required | Chroma (easier MVP) |
| **Multi-tenancy** | Manual (collection per tenant) | Native (built-in) | Qdrant (better scale) |
| **Hybrid search** | Manual (BM25 + vector) | Native (built-in) | Qdrant (less code) |
| **Reranking** | External | Built-in | Qdrant |
| **Performance** | Good (<100K docs) | Excellent (millions) | Qdrant (scale) |
| **LangChain integration** | ✅ First-class | ✅ First-class | Tie |
| **Python API** | ✅ Simple | ✅ Rich | Tie |
| **Cost** | Free (open source) | Free (open source) | Tie |
| **Ops complexity** | None (embedded) | Medium (Docker/cloud) | Chroma (MVP) |

**Recommendation:**
- **Phase 1 (MVP)**: Chroma embedded
  - Reason: Zero ops, fast setup, sufficient for <100K documents
  - Trade-off: Manual hybrid search, manual multi-tenancy
- **Phase 2 (Scale)**: Migrate to Qdrant
  - Trigger: >100K documents, need native hybrid search, or multi-tenant
  - Migration: Export Chroma → Ingest to Qdrant (straightforward)

## Embedding Pipeline Architecture

### Batch vs Streaming Embedding

| Dimension | Batch | Streaming | Hybrid (Recommended) |
|-----------|-------|-----------|----------------------|
| **Latency** | High (wait for batch) | Low (immediate) | Low for priority, deferred for bulk |
| **Throughput** | High (parallel batches) | Low (per-doc) | High overall |
| **Cost** | Low (batch discounts) | High (per-call overhead) | Optimized |
| **Error handling** | Retry batch | Retry per-doc | Granular |
| **Use case** | Daily Gmail sync | Real-time ingestion | GhostWriter: Both |

**GhostWriter Hybrid Strategy:**
1. **Batch (Daily)**: Gmail sync, Semantic Scholar papers
   - Fetch all new items
   - Batch embed 32 docs at a time
   - Store in vector DB
   - Low latency requirement (can take 10 minutes)

2. **Streaming (On-Demand)**: User uploads document, research during article generation
   - Immediate embedding needed
   - Single-doc or small-batch (4-8 docs)
   - Higher priority than batch jobs

**Implementation:**
```python
class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-mpnet-base-v2", batch_size: int = 32):
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Batch embedding for daily sync jobs."""
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=True)
            embeddings.extend(batch_embeddings)
        return embeddings

    def embed_streaming(self, text: str) -> np.ndarray:
        """Single-doc embedding for real-time requests."""
        return self.model.encode([text])[0]
```

### Embedding Model Selection

**Requirements for GhostWriter:**
1. **Quality**: High semantic understanding (technical content, research papers)
2. **Cost**: Prefer local/free (thousands of documents monthly)
3. **Speed**: Fast for real-time queries (<100ms)
4. **Dimension**: 768-1536 (balance quality vs storage)

**Model Comparison:**

| Model | Dimension | Quality (MTEB) | Speed | Cost | Deployment | Verdict |
|-------|-----------|----------------|-------|------|------------|---------|
| **all-mpnet-base-v2** | 768 | 63.3 | Fast | Free | Local | ✅ **MVP** |
| **all-MiniLM-L6-v2** | 384 | 58.8 | Very fast | Free | Local | ⚠️ Lower quality |
| **OpenAI text-embedding-3-small** | 1536 | 62.3 | Medium | $0.02/1M tokens | API | ⚠️ Cost for scale |
| **OpenAI text-embedding-3-large** | 3072 | 64.6 | Slow | $0.13/1M tokens | API | ❌ Too expensive |
| **Mistral embed** | 1024 | 77.8 | Medium | $0.10/1M tokens | API | ⚠️ Cost |
| **Cohere embed-v3** | 1024 | 64.5 | Fast | $0.10/1M tokens | API | ⚠️ Cost |

**Recommendation:**
- **Primary**: `all-mpnet-base-v2` (sentence-transformers)
  - Best balance of quality, speed, cost (free)
  - 768-dim: Good storage efficiency
  - Local: No API latency, works offline
  - MTEB score: Competitive with OpenAI small model

- **Fallback**: `OpenAI text-embedding-3-small`
  - Use if: Need slightly better quality OR processing non-English
  - Cost: ~$20/month for 1M tokens (manageable)
  - 1536-dim: 2x storage vs mpnet, but not 2x better quality

**Model Selection Strategy:**
```python
class EmbeddingService:
    def __init__(self):
        # Primary: Local sentence-transformers
        self.local_model = SentenceTransformer("all-mpnet-base-v2")
        # Fallback: OpenAI API
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.use_fallback = False

    def embed(self, text: str) -> np.ndarray:
        try:
            if not self.use_fallback:
                return self.local_model.encode([text])[0]
        except Exception as e:
            logger.warning(f"Local embedding failed: {e}, using fallback")
            self.use_fallback = True

        # Fallback to OpenAI
        response = self.openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return np.array(response.data[0].embedding)
```

### Embedding Refresh Strategy

**When to Re-Embed:**
1. **Model upgrade**: Better embedding model released
2. **Content update**: Document edited/corrected
3. **Quality issue**: Low retrieval performance detected

**Refresh Approaches:**

| Approach | When to Use | Process | Downtime |
|----------|-------------|---------|----------|
| **Full re-embed** | Model upgrade | Re-embed all documents, replace collection | ✅ Can run in parallel |
| **Incremental** | Content updates | Re-embed modified docs, update in place | ❌ None |
| **Dual-index** | Testing new model | Run old + new models in parallel, compare | ❌ None |

**Implementation:**
```python
def refresh_embeddings(collection: str, new_model: str):
    """Re-embed all documents with a new model (parallel to prod)."""
    # 1. Create new collection with _v2 suffix
    new_collection = f"{collection}_v2"

    # 2. Fetch all documents from old collection
    documents = vector_store.get_all(collection)

    # 3. Re-embed with new model (batch)
    new_embeddings = embedding_service.embed_batch(
        [doc.text for doc in documents],
        model=new_model
    )

    # 4. Insert into new collection
    vector_store.insert(new_collection, documents, new_embeddings)

    # 5. Validate: Test retrieval quality
    test_queries = ["prompt caching", "LangGraph tools", "vector databases"]
    for query in test_queries:
        old_results = vector_store.search(collection, query, top_k=5)
        new_results = vector_store.search(new_collection, query, top_k=5)
        # Compare relevance, precision
        ...

    # 6. Swap: Rename collections (atomic)
    vector_store.rename(collection, f"{collection}_old")
    vector_store.rename(new_collection, collection)

    # 7. Cleanup: Delete old collection after grace period
    vector_store.delete(f"{collection}_old")
```

## Agent-Knowledge Integration Patterns

### LangGraph Tool-Based Retrieval

**Core Pattern:**
```python
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

# Define retrieval tools
@tool
def search_knowledge(query: str, source_type: str = "all", top_k: int = 5) -> str:
    """Search the knowledge base for relevant information.

    Args:
        query: Search query (natural language)
        source_type: Filter by source ("emails", "papers", "rss", "all")
        top_k: Number of results to return

    Returns:
        JSON string with search results and metadata
    """
    results = vector_store.hybrid_search(query, source_type, top_k)
    return json.dumps([{
        "text": doc.text,
        "source": doc.metadata["source"],
        "date": doc.metadata["date"],
        "url": doc.metadata.get("url"),
        "relevance": doc.score
    } for doc in results])

@tool
def verify_claim(claim: str) -> str:
    """Verify a factual claim against the knowledge base.

    Args:
        claim: Factual claim to verify (e.g., "Claude 4.6 supports 2M tokens")

    Returns:
        JSON with verification status, supporting evidence, confidence
    """
    # High-precision retrieval (reranking + threshold)
    results = vector_store.hybrid_search(claim, source_type="all", top_k=10)
    reranked = reranker.rerank(claim, results)

    # Confidence scoring
    if reranked[0].score > 0.9:
        status = "VERIFIED"
        confidence = "high"
    elif reranked[0].score > 0.7:
        status = "LIKELY"
        confidence = "medium"
    else:
        status = "UNVERIFIED"
        confidence = "low"

    return json.dumps({
        "status": status,
        "confidence": confidence,
        "evidence": [{
            "text": doc.text,
            "source": doc.metadata["source"],
            "relevance": doc.score
        } for doc in reranked[:3]]
    })

@tool
def research_topic(topic: str, depth: str = "medium") -> str:
    """Deep research on a topic across all knowledge sources.

    Args:
        topic: Topic to research (e.g., "LangGraph agentic RAG")
        depth: "quick" (5 docs), "medium" (15 docs), "deep" (30 docs)

    Returns:
        JSON with research findings grouped by source type
    """
    top_k = {"quick": 5, "medium": 15, "deep": 30}[depth]

    # Multi-query expansion
    expanded_queries = query_transformer.expand(topic)

    # Retrieve for each query variant
    all_results = []
    for query in expanded_queries:
        results = vector_store.hybrid_search(query, source_type="all", top_k=top_k)
        all_results.extend(results)

    # Deduplicate and rerank
    unique_results = deduplicate(all_results)
    reranked = reranker.rerank(topic, unique_results)

    # Group by source type
    by_source = defaultdict(list)
    for doc in reranked[:top_k]:
        by_source[doc.metadata["source"]].append({
            "text": doc.text,
            "date": doc.metadata["date"],
            "url": doc.metadata.get("url"),
            "relevance": doc.score
        })

    return json.dumps(by_source)

# Bind tools to agent
tools = [search_knowledge, verify_claim, research_topic]
tool_node = ToolNode(tools)
```

**LangGraph Integration:**
```python
from langgraph.graph import StateGraph, END
from execution.article_state import ArticleState

def create_writer_graph():
    workflow = StateGraph(ArticleState)

    # Nodes
    workflow.add_node("research", research_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("tools", tool_node)  # NEW: Tool execution node
    workflow.add_node("editor", editor_node)

    # Edges
    workflow.add_edge("research", "writer")

    # NEW: Conditional edge based on tool calls
    workflow.add_conditional_edges(
        "writer",
        should_use_tools,  # Function: does writer want to call a tool?
        {
            "tools": "tools",      # If yes → execute tool
            "editor": "editor",    # If no → proceed to editor
        }
    )
    workflow.add_edge("tools", "writer")  # Tool results back to writer
    workflow.add_edge("editor", END)

    return workflow.compile()

def should_use_tools(state: ArticleState) -> str:
    """Decide if the writer invoked a tool."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "editor"
```

### Agent-Specific Retrieval Patterns

| Agent | Retrieval Need | Tool Used | Frequency | Priority |
|-------|----------------|-----------|-----------|----------|
| **WriterAgent** | Background context, examples | `search_knowledge()` | Medium | Low |
| **FactResearcherAgent** | Deep research, sources | `research_topic()` | High | High |
| **FactVerificationAgent** | Claim verification | `verify_claim()` | High | Critical |
| **EditorAgent** | Style references, similar articles | `search_knowledge()` | Low | Low |
| **StyleEnforcerAgent** | Voice examples, tone guides | `search_knowledge()` | Low | Low |
| **CriticAgent** | Fact-checking, quality examples | `verify_claim()`, `search_knowledge()` | Medium | Medium |

**Per-Agent Tool Configuration:**
```python
class FactVerificationAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            role="Fact Verifier",
            persona="You verify factual claims using the knowledge base.",
            model="llama-3.3-70b-versatile"
        )
        # Bind only relevant tools
        self.tools = [verify_claim, search_knowledge]

    def verify_article(self, article: str) -> Dict:
        # Extract claims
        claims = self.extract_claims(article)

        # Verify each claim using tools
        results = []
        for claim in claims:
            # LLM decides: Use verify_claim tool or skip
            response = self.call_llm_with_tools(
                prompt=f"Verify this claim: {claim}",
                tools=self.tools
            )
            # Tool execution happens in LangGraph ToolNode
            results.append(response)

        return {"claims": claims, "results": results}
```

### Query Transformation in Agent Context

**Pre-Retrieval Optimization:**
```python
class QueryTransformer:
    def __init__(self, llm_client):
        self.llm = llm_client

    def rewrite(self, query: str) -> str:
        """Rewrite query to be more specific and retrieval-friendly."""
        prompt = f"""Rewrite this query to be more specific for knowledge base retrieval:

Original: {query}

Rewritten (be specific, add context, use technical terms):"""
        return self.llm.generate(prompt)

    def expand(self, query: str, num_variants: int = 3) -> List[str]:
        """Generate multiple query variants for broader recall."""
        prompt = f"""Generate {num_variants} semantic variations of this query:

Original: {query}

Variations (different phrasings, synonyms, related concepts):"""
        response = self.llm.generate(prompt)
        return [query] + response.split("\n")[:num_variants]

    def hyde(self, query: str) -> str:
        """Generate hypothetical document that answers the query."""
        prompt = f"""Write a short passage that answers this query:

Query: {query}

Passage (2-3 sentences, technical, informative):"""
        return self.llm.generate(prompt)
```

**Integration Example:**
```python
def search_knowledge_with_transformation(query: str, strategy: str = "rewrite"):
    transformer = QueryTransformer(llm_client)

    if strategy == "rewrite":
        # Single rewritten query
        transformed = transformer.rewrite(query)
        results = vector_store.hybrid_search(transformed, top_k=10)

    elif strategy == "expand":
        # Multiple query variants → deduplicate results
        variants = transformer.expand(query, num_variants=3)
        all_results = []
        for variant in variants:
            results = vector_store.hybrid_search(variant, top_k=10)
            all_results.extend(results)
        results = deduplicate(all_results)[:10]

    elif strategy == "hyde":
        # Generate hypothetical doc → embed → search
        hypo_doc = transformer.hyde(query)
        results = vector_store.hybrid_search(hypo_doc, top_k=10)

    # Rerank
    reranked = reranker.rerank(query, results)
    return reranked
```

## Build Order and Dependencies

### Phase 1: Foundation (MVP)

**Objective:** Basic RAG with Chroma + sentence-transformers

**Components to Build:**
1. **VectorStore** (Chroma embedded)
   - Initialize Chroma client
   - Create collections for each source type
   - Implement `insert()`, `search()`, `delete()`

2. **EmbeddingService** (sentence-transformers/all-mpnet-base-v2)
   - Load model locally
   - `embed_batch()` for ingestion
   - `embed_streaming()` for queries

3. **IngestionPipeline** (Basic: Email + RSS)
   - EmailSource: Gmail API integration
   - RSSSource: feedparser integration
   - Chunking: Semantic splitter (simple)
   - Metadata extraction: source, date, author

4. **RetrievalTools** (LangGraph tools)
   - `search_knowledge()` (dense vector search only)
   - `verify_claim()` (high-threshold retrieval)
   - Bind tools to FactVerificationAgent

5. **BaseAgent Extension**
   - `call_llm_with_tools()` method
   - LangGraph ToolNode integration
   - Tool call parsing and execution

**Dependencies:**
```
EmbeddingService (no deps)
  ↓
VectorStore (depends on EmbeddingService)
  ↓
IngestionPipeline (depends on VectorStore + EmbeddingService)
  ↓
RetrievalTools (depends on VectorStore)
  ↓
BaseAgent Extension (depends on RetrievalTools)
  ↓
Agent Integration (depends on BaseAgent Extension)
```

**Estimated Effort:** 2-3 weeks
**Deliverable:** FactVerificationAgent can verify claims using Gmail + RSS knowledge base

### Phase 2: Advanced Retrieval

**Objective:** Hybrid search + reranking + query transformation

**Components to Build:**
1. **Hybrid Search**
   - BM25 sparse retrieval (rank-bm25 library)
   - RRF fusion (dense + sparse)
   - Update `search_knowledge()` to use hybrid

2. **Reranker**
   - CrossEncoder model (ms-marco-MiniLM-L-12-v2)
   - Post-retrieval reranking
   - Top-k selection with score threshold

3. **QueryTransformer**
   - Query rewriting (LLM-based)
   - Query expansion (semantic variants)
   - HyDE (hypothetical document embeddings)

4. **Additional Sources**
   - PaperSource (Semantic Scholar API)
   - WebSource (BeautifulSoup scraper)

**Dependencies:**
```
Phase 1 (Foundation)
  ↓
BM25 Sparse Index (no new deps)
  ↓
Hybrid Search (depends on VectorStore + BM25)
  ↓
Reranker (depends on Hybrid Search)
  ↓
QueryTransformer (depends on LLM)
  ↓
Update RetrievalTools (depends on all above)
```

**Estimated Effort:** 2 weeks
**Deliverable:** 48% better retrieval performance, support for papers + web sources

### Phase 3: Multi-Tenancy & Scale

**Objective:** Qdrant migration, multi-tenant structure

**Components to Build:**
1. **Qdrant Migration**
   - Export from Chroma
   - Import to Qdrant (namespace per source)
   - Update VectorStore interface to support both

2. **Multi-Tenant Architecture**
   - TenantAwareVectorStore wrapper
   - Collection naming: {source}_{tenant}
   - Per-tenant quotas and isolation

3. **Advanced Ingestion**
   - PDF extraction (Parseur API)
   - Code chunking (AST-based)
   - Entity extraction (NER)
   - Keyword extraction (RAKE/YAKE)

4. **Observability**
   - Retrieval metrics (latency, recall, precision)
   - Ingestion metrics (count, errors, duration)
   - Cost tracking (embedding API calls)

**Dependencies:**
```
Phase 2 (Advanced Retrieval)
  ↓
Qdrant Setup (Docker/cloud)
  ↓
Data Migration (export/import scripts)
  ↓
Multi-Tenant VectorStore (depends on Qdrant)
  ↓
Update all agents to use new VectorStore
```

**Estimated Effort:** 2-3 weeks
**Deliverable:** Production-ready RAG with multi-tenant support, observability

### Phase 4: Agent Optimization

**Objective:** Per-agent retrieval strategies, adaptive RAG

**Components to Build:**
1. **Agent-Specific Tools**
   - WriterAgent: `get_writing_examples()`
   - CriticAgent: `find_similar_critiques()`
   - StyleEnforcerAgent: `get_style_references()`

2. **Adaptive Routing**
   - Router: Decide parametric vs retrieval vs web search
   - Confidence-based: Skip retrieval if LLM is confident
   - Cost-aware: Use cheaper retrieval for simple queries

3. **Memory Integration**
   - Store previous searches in ArticleState
   - Avoid redundant retrieval
   - Session-based caching

4. **Feedback Loop**
   - Track: Which retrievals were useful?
   - Learn: Improve query transformation based on feedback
   - Adapt: Adjust reranking weights

**Dependencies:**
```
Phase 3 (Multi-Tenancy & Scale)
  ↓
Agent-Specific Tools (depends on RetrievalTools)
  ↓
Adaptive Routing (depends on LangGraph state)
  ↓
Memory Integration (depends on ArticleState)
  ↓
Feedback Loop (depends on all above + logging)
```

**Estimated Effort:** 2-3 weeks
**Deliverable:** Intelligent retrieval that adapts to agent needs and user feedback

## Integration Points with Existing LangGraph Pipeline

### Current GhostWriter Pipeline

```
START
  ↓
[research] (FactResearcherAgent)
  ↓
[writer] (WriterAgent)
  ↓
[voice_validate] (Voice validation)
  ↓
[editor] (EditorAgent)
  ↓
[style_enforce] (StyleEnforcerAgent)
  ↓
[critic] (CriticAgent)
  ↓
[fact_verify] (FactVerificationAgent)
  ↓
[quality_gate] (Multi-model panel)
  ↓
END
```

### Integration Points for Knowledge Layer

| Pipeline Stage | Integration | Tool Used | Change Required |
|----------------|-------------|-----------|-----------------|
| **[research]** | NEW: Research from knowledge base | `research_topic()` | Add tool binding |
| **[writer]** | NEW: Optional context retrieval | `search_knowledge()` | Add conditional edge to ToolNode |
| **[editor]** | NEW: Style reference lookup | `search_knowledge()` | Add tool binding |
| **[critic]** | NEW: Quality example retrieval | `search_knowledge()` | Add tool binding |
| **[fact_verify]** | **CRITICAL: Claim verification** | `verify_claim()` | Already uses external sources; add knowledge base as source |
| **[quality_gate]** | No change | N/A | N/A |

### Updated Pipeline with Knowledge Layer

```
START
  ↓
[research] ──────> [tools] (research_topic)
  ↓                   ↓
  ←───────────────────┘
  ↓
[writer] ──────> [tools] (search_knowledge, optional)
  ↓                ↓
  ←────────────────┘
  ↓
[voice_validate]
  ↓
[editor] ──────> [tools] (search_knowledge, rare)
  ↓                ↓
  ←────────────────┘
  ↓
[style_enforce]
  ↓
[critic] ──────> [tools] (search_knowledge, optional)
  ↓                ↓
  ←────────────────┘
  ↓
[fact_verify] ──> [tools] (verify_claim, frequent)
  ↓                ↓
  ←────────────────┘
  ↓
[quality_gate]
  ↓
END
```

### Code Changes Required

**1. ArticleState Extension:**
```python
class ArticleState(BaseModel):
    # EXISTING fields...
    research_facts: List[Dict] = Field(default_factory=list)

    # NEW: Knowledge layer fields
    knowledge_queries: List[Dict] = Field(default_factory=list)  # Track queries
    retrieved_contexts: List[Dict] = Field(default_factory=list)  # Retrieved docs
    verified_claims: List[Dict] = Field(default_factory=list)  # Claim verification results
```

**2. BaseAgent Extension:**
```python
class BaseAgent:
    # EXISTING: LLM calling
    def call_llm(self, prompt, system_instruction=None, temperature=0.7):
        ...

    # NEW: Tool-based calling
    def call_llm_with_tools(self, prompt, tools: List[Tool], system_instruction=None):
        """Call LLM with retrieval tools enabled."""
        # Bind tools to model
        model_with_tools = self.client.bind_tools(tools)

        # Invoke
        response = model_with_tools.invoke(prompt)

        # If tool calls, execute and return combined result
        if response.tool_calls:
            tool_results = self.execute_tools(response.tool_calls)
            return {"response": response, "tool_results": tool_results}

        return {"response": response, "tool_results": []}
```

**3. FactVerificationAgent Update:**
```python
class FactVerificationAgent(BaseAgent):
    def __init__(self):
        super().__init__(...)
        # NEW: Bind knowledge base tools
        self.tools = [verify_claim, search_knowledge]

    def verify_article(self, article: str) -> Dict:
        claims = self.extract_claims(article)
        results = []

        for claim in claims:
            # EXISTING: Web search verification (Perplexity, Gemini)
            web_result = self.verify_with_web_search(claim)

            # NEW: Knowledge base verification
            kb_result = self.call_llm_with_tools(
                prompt=f"Verify: {claim}",
                tools=self.tools
            )

            # Combine: Web + Knowledge Base
            combined = self.combine_verification(web_result, kb_result)
            results.append(combined)

        return {"claims": claims, "results": results}
```

**4. LangGraph Workflow Update:**
```python
def create_article_graph():
    workflow = StateGraph(ArticleState)

    # EXISTING nodes
    workflow.add_node("research", research_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("editor", editor_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("fact_verify", fact_verify_node)
    workflow.add_node("quality_gate", quality_gate_node)

    # NEW: Tool execution node
    workflow.add_node("tools", tool_node)

    # EXISTING edges
    workflow.add_edge("research", "writer")
    workflow.add_edge("voice_validate", "editor")
    workflow.add_edge("style_enforce", "critic")
    workflow.add_edge("quality_gate", END)

    # NEW: Conditional edges for tool usage
    workflow.add_conditional_edges(
        "research",
        should_use_tools,
        {"tools": "tools", "writer": "writer"}
    )
    workflow.add_edge("tools", "research")  # Loop back after tool execution

    # Similar for writer, critic, fact_verify...

    return workflow.compile()
```

## Recommended Technology Stack

### Core Technologies

| Component | Technology | Version | Rationale |
|-----------|-----------|---------|-----------|
| **Vector Database (MVP)** | Chroma | 0.4+ | Embedded, zero-ops, LangChain first-class support |
| **Vector Database (Scale)** | Qdrant | 1.7+ | Native hybrid search, multi-tenancy, production-grade |
| **Embedding Model** | sentence-transformers | 2.3+ | all-mpnet-base-v2: Free, local, 768-dim, 63.3 MTEB |
| **Embedding Fallback** | OpenAI API | v1 | text-embedding-3-small: 1536-dim, 62.3 MTEB, $0.02/1M tokens |
| **Sparse Retrieval** | rank-bm25 | 0.2+ | BM25 implementation for hybrid search |
| **Reranker** | sentence-transformers | 2.3+ | CrossEncoder: ms-marco-MiniLM-L-12-v2 |
| **Chunking** | LangChain TextSplitters | 0.1+ | RecursiveCharacterTextSplitter + SemanticChunker |
| **Email Parsing** | Gmail API + email | stdlib | MIME parsing, attachment handling |
| **PDF Parsing** | pdfplumber + PyPDF2 | latest | Text extraction with layout preservation |
| **Web Scraping** | Trafilatura + BeautifulSoup | latest | Clean text extraction |
| **LangGraph Integration** | LangGraph | 0.1+ | ToolNode for tool execution |
| **Orchestration** | Python asyncio | stdlib | Batch embedding, concurrent ingestion |

### Installation

```bash
# Vector database
pip install chromadb  # MVP
pip install qdrant-client  # Scale

# Embeddings
pip install sentence-transformers
pip install openai  # Fallback

# Retrieval
pip install rank-bm25  # BM25 sparse search

# Ingestion
pip install pdfplumber pypdf2  # PDF
pip install trafilatura beautifulsoup4  # Web
pip install google-api-python-client  # Gmail

# LangChain/LangGraph
pip install langchain langgraph langchain-core

# Utilities
pip install nltk spacy  # Text processing
```

## Quality Metrics and Monitoring

### Retrieval Quality Metrics

| Metric | Target | Measurement | Alert Threshold |
|--------|--------|-------------|-----------------|
| **Latency (p95)** | <500ms | Query → results time | >1000ms |
| **Recall@10** | >0.8 | Relevant docs in top 10 | <0.6 |
| **Precision@5** | >0.9 | Relevant docs in top 5 | <0.7 |
| **MRR (Mean Reciprocal Rank)** | >0.85 | Position of first relevant doc | <0.7 |
| **NDCG@10** | >0.85 | Ranking quality | <0.7 |

### Ingestion Quality Metrics

| Metric | Target | Measurement | Alert Threshold |
|--------|--------|-------------|-----------------|
| **Success Rate** | >99% | Successful ingestions / total | <95% |
| **Processing Time** | <30 min | Batch ingestion duration | >60 min |
| **Chunk Quality** | >0.9 | Readable chunks / total | <0.8 |
| **Embedding Errors** | <1% | Failed embeddings / total | >5% |

### Monitoring Implementation

```python
class RetrievalMetrics:
    def __init__(self):
        self.query_latencies = []
        self.relevance_scores = []

    def log_query(self, query: str, results: List[Document], latency_ms: float):
        self.query_latencies.append(latency_ms)

        # Compute relevance (manual labels or user feedback)
        relevance = self.compute_relevance(query, results)
        self.relevance_scores.append(relevance)

        # Log to file/database
        logger.info({
            "query": query,
            "latency_ms": latency_ms,
            "num_results": len(results),
            "top_score": results[0].score if results else 0,
            "relevance": relevance
        })

    def report_metrics(self) -> Dict:
        return {
            "latency_p50": np.percentile(self.query_latencies, 50),
            "latency_p95": np.percentile(self.query_latencies, 95),
            "latency_p99": np.percentile(self.query_latencies, 99),
            "avg_relevance": np.mean(self.relevance_scores),
            "recall@10": self.compute_recall(k=10),
            "precision@5": self.compute_precision(k=5),
        }
```

## Potential Pitfalls and Mitigations

### Pitfall 1: Embedding Model Mismatch

**Problem:** Query and document embedded with different models → poor retrieval
**Prevention:**
- Store embedding model name with each document
- Version collections when changing models
- Validate: Query model matches document model

### Pitfall 2: Chunking Too Large/Small

**Problem:** Chunks >1024 tokens → poor embedding quality. Chunks <100 tokens → lack context
**Prevention:**
- Target: 512-1024 tokens per chunk
- Validate: Measure retrieval quality at different chunk sizes
- Tune: Experiment with chunk size per source type

### Pitfall 3: Stale Embeddings

**Problem:** Content updated but embeddings not refreshed → outdated results
**Prevention:**
- Track: Document modification time
- Trigger: Re-embed on update
- Periodic: Monthly full re-embed (catch missed updates)

### Pitfall 4: Cross-Tenant Data Leakage

**Problem:** Metadata filter bug → tenant A sees tenant B's data
**Prevention:**
- Physical isolation: Namespace per tenant (not metadata filter)
- Test: Unit test retrieval with multi-tenant data
- Audit: Log all queries with tenant_id

### Pitfall 5: Tool Calling Loops

**Problem:** Agent keeps calling `search_knowledge()` → infinite loop
**Prevention:**
- Max tool calls: Limit 5 calls per agent invocation
- Deduplication: Don't retrieve same query twice
- Confidence threshold: Stop if retrieval confidence low

### Pitfall 6: Retrieval Latency Cascade

**Problem:** Slow retrieval → agent timeout → pipeline failure
**Prevention:**
- Timeout: 2 seconds max per retrieval
- Caching: Cache frequent queries (Redis)
- Fallback: Return partial results if timeout

### Pitfall 7: Token Cost Explosion

**Problem:** Every agent retrieves → 10x token usage
**Prevention:**
- Selective retrieval: Only agents that need it
- Caching: Reuse retrieved context across agents
- Compression: Summarize long retrieved docs

## Sources

### RAG Architecture Patterns
- [Evolution of RAGs: Naive RAG, Advanced RAG, and Modular RAG Architectures - MarkTechPost](https://www.marktechpost.com/2024/04/01/evolution-of-rags-naive-rag-advanced-rag-and-modular-rag-architectures/)
- [Modular RAG: Transforming RAG Systems into LEGO-like Reconfigurable Frameworks](https://arxiv.org/html/2407.21059v1)
- [14 types of RAG (Retrieval-Augmented Generation)](https://www.meilisearch.com/blog/rag-types)
- [Advanced & Modular RAG | Pattern & Architectures | AI Technology Radar](https://ai-radar.aoe.com/architecture-pattern/rag/)

### Multi-Tenancy and Vector Databases
- [Multi-Tenancy in Vector Databases | Pinecone](https://www.pinecone.io/learn/series/vector-databases-in-production-for-busy-engineers/vector-database-multi-tenancy/)
- [Rethinking Vector Search at Scale: Weaviate's Native, Efficient and Optimized Multi-Tenancy | Weaviate](https://weaviate.io/blog/weaviate-multi-tenancy-architecture-explained)
- [Designing Multi-Tenancy RAG with Milvus: Best Practices for Scalable Enterprise Knowledge Bases - Milvus Blog](https://milvus.io/blog/build-multi-tenancy-rag-with-milvus-best-practices-part-one.md)
- [The 7 Best Vector Databases in 2026 | DataCamp](https://www.datacamp.com/blog/the-top-5-vector-databases)

### LangGraph RAG Integration
- [Build a custom RAG agent with LangGraph - Docs by LangChain](https://docs.langchain.com/oss/python/langgraph/agentic-rag)
- [Building Agentic RAG Systems with LangGraph: The 2026 Guide - Rahul kolekar](https://rahulkolekar.com/building-agentic-rag-systems-with-langgraph/)
- [Building a Multi-Agent RAG System with LangGraph | by Wesley Huber | Jan, 2026 | Medium](https://wesleybaxterhuber.medium.com/building-a-multi-agent-rag-system-with-langgraph-43071904b123)

### Chunking Strategies
- [Chunking Strategies to Improve Your RAG Performance | Weaviate](https://weaviate.io/blog/chunking-strategies-for-rag)
- [RAG Pipeline Deep Dive: Ingestion, Chunking, Embedding, and Vector Search | by Derrick Ryan Giggs | Jan, 2026 | Medium](https://medium.com/@derrickryangiggs/rag-pipeline-deep-dive-ingestion-chunking-embedding-and-vector-search-abd3c8bfc177)
- [The Ultimate Guide to Chunking Strategies for RAG Applications with Databricks](https://community.databricks.com/t5/technical-blog/the-ultimate-guide-to-chunking-strategies-for-rag-applications/ba-p/113089)

### Hybrid Search and Reranking
- [Hybrid Retrieval: Combining Sparse and Dense Methods for Effective Information Retrieval - Interactive | Michael Brenndoerfer](https://mbrenndoerfer.com/writing/hybrid-retrieval-combining-sparse-dense-methods-effective-information-retrieval)
- [Introducing cascading retrieval: Unifying dense and sparse with reranking | Pinecone](https://www.pinecone.io/blog/cascading-retrieval/)
- [Reranking in Hybrid Search - Qdrant](https://qdrant.tech/documentation/advanced-tutorials/reranking-hybrid-search/)
- [Dense vector + Sparse vector + Full text search + Tensor reranker = Best retrieval for RAG? | Infinity](https://infiniflow.org/blog/best-hybrid-search-solution)

### Embedding Models
- [13 Best Embedding Models in 2026: OpenAI vs Voyage AI vs Ollama | Complete Guide + Pricing & Performance](https://elephas.app/blog/best-embedding-models)
- [Top Embedding Models 2026: Complete In-Depth Guide](https://artsmart.ai/blog/top-embedding-models-in-2025/)
- [The Best Open-Source Embedding Models in 2026](https://www.bentoml.com/blog/a-guide-to-open-source-embedding-models)

### Query Transformation
- [RAG Query Augmentation | Expansion & Transformation](https://apxml.com/courses/optimizing-rag-for-production/chapter-2-advanced-retrieval-optimization/query-augmentation-rag)
- [Unlocking the Power of Query Transformation in Retrieval-Augmented Generation (RAG) | by Aditya Sharma | Medium](https://medium.com/@adityabbsharma/unlocking-the-power-of-query-transformation-in-retrieval-augmented-generation-rag-fbe461c354d6)
- [Raising the bar for RAG excellence: introducing generative query rewriting and new ranking model](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/raising-the-bar-for-rag-excellence-query-rewriting-and-new-semantic-ranker/4302729)

### Ingestion and Parsing
- [Data Ingestion for Beginners - Qdrant](https://qdrant.tech/documentation/data-ingestion-beginners/)
- [RAG Architecture: Best Practice → Vector Database Ingestion | by Shekhar Manna | Medium](https://medium.com/@shekhar.manna83/rag-architecture-best-practice-vector-database-ingestion-6a7aecaa5ae4)
- [Best API For PDF Data Extraction (2026) | Parseur®](https://parseur.com/blog/best-api-data-extraction)
