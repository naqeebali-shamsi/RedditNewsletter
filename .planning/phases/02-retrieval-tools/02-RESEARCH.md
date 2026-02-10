# Phase 2: Retrieval Tools - Research

**Researched:** 2026-02-10
**Domain:** Hybrid search (BM25 + pgvector), CrossEncoder reranking, and metadata filtering for RAG retrieval
**Confidence:** HIGH

## Summary

This research investigates advanced retrieval patterns for RAG systems, focusing on hybrid search that combines lexical (BM25) and semantic (vector) retrieval, CrossEncoder reranking for precision improvement, and metadata filtering for scoped retrieval. The standard 2026 approach uses Reciprocal Rank Fusion (RRF) with k=60 to merge BM25 and vector rankings without score normalization, bm25s (500x faster than rank-bm25) for Python-based BM25, or PostgreSQL-native BM25 via pg_search/pg_textsearch extensions for database-integrated solutions. CrossEncoder reranking with sentence-transformers (ms-marco-MiniLM-L6-v2) adds 150-250ms latency but improves accuracy by 10-15% in two-stage retrieval (retrieve 50-100, rerank to 10). Source recency scoring uses half-life decay functions (14-day default) to prioritize fresh content. Critical success factors include proper RRF parameter tuning, limiting CrossEncoder candidates to control latency, and implementing metadata filtering at the SQL layer for performance.

**Primary recommendation:** Use bm25s for Python-based BM25 (500x faster than rank-bm25), RRF fusion with k=60, sentence-transformers CrossEncoder for reranking (retrieve 50, rerank to 10), and SQLAlchemy JSONB filtering for metadata. Implement time decay with 14-day half-life for trend-sensitive queries. For PostgreSQL-native BM25, consider ParadeDB pg_search or pg_textsearch extensions but evaluate stability and deployment complexity.

## Standard Stack

The established libraries/tools for hybrid search and reranking:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| bm25s | Latest | Fast BM25 sparse retrieval | 500x faster than rank-bm25, scipy sparse matrices, 573 QPS throughput |
| sentence-transformers | Latest | CrossEncoder reranking | Industry standard for reranking, pre-trained MS-MARCO models, 2-stage retrieval |
| pgvector | 0.8.0+ | Vector similarity search | Existing Phase 1 foundation, HNSW indexing for semantic search |
| SQLAlchemy | 2.0+ | JSONB metadata filtering | ORM with PostgreSQL JSONB support for complex metadata queries |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| ParadeDB pg_search | Latest | PostgreSQL-native BM25 | Database-integrated BM25, transactional consistency, eliminates Python BM25 sync |
| pg_textsearch | Latest | PostgreSQL BM25 extension | Alternative to ParadeDB, MIT-licensed, Tiger Data backed |
| rank-bm25 | Latest | Simple BM25 implementation | Prototyping, small datasets (<10k docs), when bm25s isn't available |
| pysbd | Latest | Sentence boundary detection | Fine-grained citation splitting (already in Phase 1 for chunking) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| bm25s | rank-bm25 | rank-bm25 is simpler but 500x slower (2 QPS vs 573 QPS), acceptable for <10k documents |
| sentence-transformers CrossEncoder | LLM-based reranking | LLMs have higher latency (500-2000ms vs 150-250ms) and cost, but occasionally better on narrow tasks |
| RRF fusion | Linear weighted combination | Weighted combo can outperform RRF if tuned perfectly, but requires dataset-specific calibration and drifts over time |
| PostgreSQL-native BM25 | Python BM25 (bm25s) | Native BM25 eliminates sync overhead, but adds deployment complexity (extensions) and locks into PostgreSQL ecosystem |
| CrossEncoder | ColBERT | ColBERT has better accuracy but 10-20x higher storage (hundreds of vectors per doc), use for specialized domains |

**Installation:**
```bash
# Core libraries
pip install bm25s sentence-transformers sqlalchemy

# Optional: for stemming in bm25s
pip install PyStemmer

# PostgreSQL-native BM25 (choose one if going native route)
# ParadeDB (Docker)
docker run --name paradedb -e POSTGRES_PASSWORD=password paradedb/paradedb

# pg_textsearch (PostgreSQL extension)
# Requires PostgreSQL extension installation - see pg_textsearch docs
```

## Architecture Patterns

### Recommended Project Structure
```
execution/
├── vector_db/
│   ├── retrieval.py          # NEW: Hybrid retrieval orchestrator (BM25 + vector + RRF)
│   ├── reranking.py          # NEW: CrossEncoder reranking logic
│   ├── metadata_filters.py   # NEW: SQLAlchemy filter builders (date, source, tags)
│   ├── recency_scoring.py    # NEW: Time decay functions for source freshness
│   ├── indexing.py           # EXISTING: Extends semantic_search for hybrid integration
│   ├── models.py             # EXISTING: Add BM25 index tracking if using Python-based
│   └── ...                   # Phase 1 modules (embeddings, chunking, ingestion)
```

### Pattern 1: Two-Stage Retrieval with RRF Fusion
**What:** Retrieve candidates from BM25 and vector search separately, fuse with RRF, then rerank top-K with CrossEncoder
**When to use:** Production RAG with 10k+ documents requiring both keyword precision and semantic understanding
**Example:**
```python
# Source: ParadeDB hybrid search patterns + RRF best practices
from execution.vector_db.retrieval import HybridRetriever
from execution.vector_db.reranking import CrossEncoderReranker

retriever = HybridRetriever()
reranker = CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L6-v2")

# Stage 1: Hybrid retrieval with RRF fusion
query = "How does PostgreSQL handle vector indexing?"
hybrid_results = retriever.search(
    query=query,
    tenant_id="default",
    bm25_top_k=50,       # Retrieve 50 from BM25
    vector_top_k=50,     # Retrieve 50 from vector search
    rrf_k=60,            # RRF constant (industry standard)
    fusion_top_k=50      # Keep top 50 after RRF fusion
)

# Stage 2: CrossEncoder reranking
final_results = reranker.rerank(
    query=query,
    candidates=hybrid_results,
    top_k=10             # Final top 10 results
)
```

### Pattern 2: RRF Fusion in Python (Retrieval Layer)
**What:** Implement Reciprocal Rank Fusion to merge BM25 and vector rankings without score normalization
**When to use:** When using Python-based BM25 (bm25s or rank-bm25) alongside pgvector
**Example:**
```python
# Source: https://www.paradedb.com/learn/search-concepts/reciprocal-rank-fusion
# RRF formula: score(d) = sum(1 / (k + rank_r(d))) for each ranking source r
from typing import List, Dict

def reciprocal_rank_fusion(
    bm25_results: List[Dict],    # [{"id": "doc1", "rank": 1}, ...]
    vector_results: List[Dict],  # [{"id": "doc2", "rank": 1}, ...]
    k: int = 60,                 # RRF constant (default 60)
    bm25_weight: float = 1.0,
    vector_weight: float = 1.0
) -> List[Dict]:
    """Fuse BM25 and vector search results using RRF."""
    rrf_scores = {}

    # Add BM25 contributions
    for result in bm25_results:
        doc_id = result["id"]
        rank = result["rank"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + bm25_weight / (k + rank)

    # Add vector contributions
    for result in vector_results:
        doc_id = result["id"]
        rank = result["rank"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + vector_weight / (k + rank)

    # Sort by fused RRF score
    fused = [{"id": doc_id, "rrf_score": score} for doc_id, score in rrf_scores.items()]
    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused
```

### Pattern 3: BM25 with bm25s (Fast Python Implementation)
**What:** Use bm25s for fast BM25 indexing and retrieval in Python, 500x faster than rank-bm25
**When to use:** When not using PostgreSQL-native BM25, need fast lexical retrieval, or prototyping hybrid search
**Example:**
```python
# Source: https://github.com/xhluca/bm25s
import bm25s
from execution.vector_db.models import KnowledgeChunk, Document
from execution.vector_db.connection import get_session

class BM25Index:
    """Fast BM25 indexing using bm25s for lexical retrieval."""

    def __init__(self, stemmer: str = "en"):
        self.retriever = bm25s.BM25()
        self.stemmer = stemmer
        self.doc_ids = []

    def index_corpus(self, tenant_id: str):
        """Build BM25 index from knowledge_chunks table."""
        session = get_session()

        # Load all chunks for tenant
        chunks = session.query(KnowledgeChunk).filter(
            KnowledgeChunk.tenant_id == tenant_id,
            KnowledgeChunk.content.isnot(None)
        ).all()

        # Tokenize corpus
        corpus = [chunk.content for chunk in chunks]
        corpus_tokens = bm25s.tokenize(corpus, stopwords=self.stemmer)

        # Build index
        self.retriever.index(corpus_tokens)
        self.doc_ids = [chunk.id for chunk in chunks]

        # Optional: save index to disk
        self.retriever.save("bm25_index", corpus=corpus_tokens)

    def search(self, query: str, top_k: int = 50) -> List[Dict]:
        """Search BM25 index and return ranked results."""
        query_tokens = bm25s.tokenize(query, stopwords=self.stemmer)
        results, scores = self.retriever.retrieve(query_tokens, k=top_k)

        return [
            {"id": self.doc_ids[idx], "bm25_score": float(score), "rank": i + 1}
            for i, (idx, score) in enumerate(zip(results[0], scores[0]))
        ]
```

### Pattern 4: CrossEncoder Reranking with Sentence-Transformers
**What:** Rerank top-K candidates with CrossEncoder for precision improvement
**When to use:** After hybrid retrieval, before presenting results to LLM (typical: retrieve 50, rerank to 10)
**Example:**
```python
# Source: https://www.sbert.net/docs/cross_encoder/usage/usage.html
from sentence_transformers import CrossEncoder
from typing import List, Dict

class CrossEncoderReranker:
    """Rerank retrieval candidates using CrossEncoder for precision."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        """Rerank candidates and return top-K by relevance.

        Args:
            query: User query
            candidates: List of dicts with "id" and "content" keys
            top_k: Number of results to return after reranking

        Returns:
            Reranked results with "rerank_score" added
        """
        # Build query-document pairs
        pairs = [(query, candidate["content"]) for candidate in candidates]

        # Score all pairs (batched internally)
        scores = self.model.predict(pairs)

        # Add scores and sort
        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_k]
```

### Pattern 5: Metadata Filtering with SQLAlchemy JSONB
**What:** Filter retrieval by date range, source type, and topic tags using SQLAlchemy JSONB operators
**When to use:** Scoped searches (e.g., "papers from last 6 months about databases")
**Example:**
```python
# Source: SQLAlchemy JSONB documentation and PostgreSQL full-text search patterns
from sqlalchemy import select, and_, or_, func
from execution.vector_db.models import KnowledgeChunk, Document
from datetime import datetime, timedelta

class MetadataFilter:
    """Build SQLAlchemy filters for metadata-scoped retrieval."""

    @staticmethod
    def date_range(start_date: datetime, end_date: datetime):
        """Filter by document publication date range."""
        return and_(
            Document.date_published >= start_date,
            Document.date_published <= end_date
        )

    @staticmethod
    def source_types(types: List[str]):
        """Filter by source types (email, rss, paper)."""
        return Document.source_type.in_(types)

    @staticmethod
    def topic_tags(tags: List[str], match_any: bool = True):
        """Filter by topic tags in JSONB array.

        Args:
            tags: List of topic tags to match
            match_any: If True, match documents with ANY tag. If False, match ALL tags.
        """
        if match_any:
            # Match if topic_tags contains any of the specified tags
            conditions = [
                KnowledgeChunk.topic_tags.contains([tag]) for tag in tags
            ]
            return or_(*conditions)
        else:
            # Match if topic_tags contains all specified tags
            return KnowledgeChunk.topic_tags.contains(tags)

    @staticmethod
    def recency(months: int = 6):
        """Filter to documents published within last N months."""
        cutoff = datetime.utcnow() - timedelta(days=months * 30)
        return Document.date_published >= cutoff

# Usage in retrieval
def filtered_semantic_search(
    query: str,
    tenant_id: str,
    source_types: List[str] = None,
    topic_tags: List[str] = None,
    date_range: tuple = None,
    limit: int = 50
):
    """Semantic search with metadata filters."""
    from execution.vector_db.embeddings import EmbeddingClient
    from execution.vector_db.connection import get_session

    session = get_session()
    client = EmbeddingClient()
    query_embedding = client.embed_text(query)

    # Build query with filters
    stmt = (
        select(KnowledgeChunk, Document)
        .join(Document, KnowledgeChunk.document_id == Document.id)
        .where(KnowledgeChunk.tenant_id == tenant_id)
        .where(KnowledgeChunk.embedding.isnot(None))
    )

    # Apply metadata filters
    if source_types:
        stmt = stmt.where(MetadataFilter.source_types(source_types))
    if topic_tags:
        stmt = stmt.where(MetadataFilter.topic_tags(topic_tags))
    if date_range:
        stmt = stmt.where(MetadataFilter.date_range(*date_range))

    # Order by vector similarity
    stmt = stmt.order_by(
        KnowledgeChunk.embedding.cosine_distance(query_embedding)
    ).limit(limit)

    results = session.execute(stmt).all()
    return results
```

### Pattern 6: Source Recency Scoring with Time Decay
**What:** Prioritize recent sources using half-life decay function for trend-sensitive queries
**When to use:** Trend content, breaking news, rapidly evolving technical domains
**Example:**
```python
# Source: https://arxiv.org/html/2509.19376 (Solving Freshness in RAG)
from datetime import datetime, timedelta
import math

class RecencyScorer:
    """Apply time decay to retrieval scores for recency-aware ranking."""

    def __init__(self, half_life_days: int = 14):
        """Initialize with half-life parameter (default 14 days)."""
        self.half_life_days = half_life_days

    def time_decay(self, date_published: datetime) -> float:
        """Calculate decay factor based on document age.

        Uses exponential decay: score = 0.5 ^ (age_days / half_life_days)

        Returns:
            Float between 0 and 1 (1 = published today, 0.5 = half_life_days old)
        """
        if date_published is None:
            return 0.5  # Neutral score for unknown dates

        age_days = (datetime.utcnow() - date_published).days

        if age_days < 0:
            age_days = 0  # Handle future dates

        decay = math.pow(0.5, age_days / self.half_life_days)
        return decay

    def fused_score(
        self,
        semantic_score: float,
        date_published: datetime,
        semantic_weight: float = 0.7,
        recency_weight: float = 0.3
    ) -> float:
        """Combine semantic similarity with recency decay.

        Args:
            semantic_score: Cosine similarity or RRF score (0-1 normalized)
            date_published: Document publication date
            semantic_weight: Weight for semantic score (default 0.7)
            recency_weight: Weight for recency score (default 0.3)

        Returns:
            Fused score combining semantic relevance and recency
        """
        recency = self.time_decay(date_published)
        return (semantic_weight * semantic_score) + (recency_weight * recency)

# Usage in retrieval
def recency_aware_search(query: str, tenant_id: str, top_k: int = 10):
    """Hybrid search with recency scoring."""
    from execution.vector_db.retrieval import HybridRetriever

    retriever = HybridRetriever()
    recency_scorer = RecencyScorer(half_life_days=14)

    # Get hybrid results (RRF scores normalized 0-1)
    results = retriever.search(query, tenant_id, fusion_top_k=50)

    # Apply recency weighting
    for result in results:
        result["fused_score"] = recency_scorer.fused_score(
            semantic_score=result["rrf_score"],
            date_published=result["date_published"],
            semantic_weight=0.7,
            recency_weight=0.3
        )

    # Re-sort by fused score
    results.sort(key=lambda x: x["fused_score"], reverse=True)
    return results[:top_k]
```

### Pattern 7: Fine-Grained Citation Extraction
**What:** Extract sentence-level citations from chunks for source attribution
**When to use:** When LLM generates responses from retrieved chunks, provide clickable citations
**Example:**
```python
# Source: https://www.tensorlake.ai/blog/rag-citations
# Existing Phase 1: pysbd for sentence splitting
import pysbd
from typing import List, Dict

class CitationExtractor:
    """Extract sentence-level citations from retrieved chunks."""

    def __init__(self):
        self.segmenter = pysbd.Segmenter(language="en", clean=False)

    def extract_sentences(self, chunk: Dict) -> List[Dict]:
        """Split chunk into sentences with citation metadata.

        Args:
            chunk: Dict with keys: id, content, title, url, source_type

        Returns:
            List of sentence dicts with citation info
        """
        sentences = self.segmenter.segment(chunk["content"])

        return [
            {
                "text": sentence,
                "chunk_id": chunk["id"],
                "title": chunk["title"],
                "url": chunk["url"],
                "source_type": chunk["source_type"],
                "citation_id": f"{chunk['id']}.{i}"
            }
            for i, sentence in enumerate(sentences)
        ]

    def format_citation(self, sentence: Dict) -> str:
        """Format citation as markdown link.

        Returns:
            Markdown: [Title](URL)
        """
        title = sentence["title"] or "Source"
        url = sentence["url"] or "#"
        return f"[{title}]({url})"

# Usage in LLM prompt
def build_rag_prompt(query: str, retrieved_chunks: List[Dict]) -> str:
    """Build LLM prompt with fine-grained citation instructions."""
    extractor = CitationExtractor()

    # Extract sentences with citation metadata
    cited_sentences = []
    for chunk in retrieved_chunks:
        cited_sentences.extend(extractor.extract_sentences(chunk))

    # Build context with inline citation IDs
    context = "\n\n".join([
        f"[{sent['citation_id']}] {sent['text']}"
        for sent in cited_sentences
    ])

    prompt = f"""Answer the question using the provided context. Cite sources by including [citation_id] after each claim.

Context:
{context}

Question: {query}

Answer (include [citation_id] after each claim):"""

    return prompt, cited_sentences
```

### Anti-Patterns to Avoid
- **Score normalization for BM25+vector fusion:** BM25 scores are unbounded, vector scores are bounded (0-1). Linear combination requires perfect calibration per dataset and drifts over time. Use RRF instead.
- **Reranking all candidates:** CrossEncoders are slow (1800 docs/sec for MiniLM-L6). Reranking 500 candidates adds 300ms+ latency. Retrieve 50-100, rerank to 10-20.
- **Not tuning RRF k parameter:** k=60 is robust default, but k=20-40 gives top results more influence (better for precise queries), k=80-100 for broader queries. Test on sample queries.
- **Using rank-bm25 for production:** rank-bm25 is 500x slower than bm25s (2 QPS vs 573 QPS). Only acceptable for <10k documents or prototyping.
- **Metadata filtering in application layer:** Filtering after retrieval wastes vector search compute. Apply filters in SQL WHERE clause before vector similarity calculation.
- **Ignoring recency for trend content:** Semantic search alone can return stale results for "latest developments" queries. Add time decay for trend-sensitive domains.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| BM25 sparse retrieval | Custom TF-IDF or frequency counting | bm25s or PostgreSQL pg_search | BM25 has document length normalization, IDF weighting, and saturation that prevent keyword stuffing. Custom implementations miss edge cases and are 500x slower. |
| Score fusion for hybrid search | Linear weighted combination of BM25 and vector scores | Reciprocal Rank Fusion (RRF) with k=60 | BM25 scores are unbounded, vector scores are 0-1. Linear combo requires dataset-specific calibration and drifts over time. RRF uses ranks, not scores, eliminating normalization. |
| CrossEncoder reranking | Custom BERT fine-tuning or LLM scoring | sentence-transformers pre-trained MS-MARCO models | MS-MARCO models are trained on 500k+ query-document pairs. Custom training requires labeled data and weeks of compute. LLMs are 5-10x slower. |
| Sentence splitting for citations | Regex or NLTK sentence tokenizer | pysbd (already in Phase 1) | pysbd handles edge cases (Dr., Mr., decimals, abbreviations) that break regex. Already integrated in chunking pipeline. |
| Metadata filtering query builder | String concatenation SQL | SQLAlchemy JSONB operators | SQL injection risks, JSONB syntax complexity (contains, has_key), and type casting. SQLAlchemy handles escaping and type conversion. |
| Time decay functions | Linear decay or step functions | Exponential half-life decay (research-validated) | Linear decay undervalues recent content, step functions create ranking discontinuities. Half-life decay (14 days) is validated in RAG research. |

**Key insight:** Hybrid search fusion is the hardest problem to hand-roll. Score normalization across different retrieval methods creates dataset-specific tuning nightmares. RRF solves this by operating on ranks (1, 2, 3...) instead of scores, making it robust across datasets without calibration. CrossEncoder reranking looks simple (pass query+doc to BERT) but requires massive training data to work well - pre-trained models are 95% as good as proprietary alternatives.

## Common Pitfalls

### Pitfall 1: BM25-Vector Score Normalization Hell
**What goes wrong:** Linear combination of BM25 and vector scores produces poor rankings, requires constant retuning as corpus evolves
**Why it happens:** BM25 scores are unbounded (can be 0-100+), vector cosine similarity is bounded (0-1). Normalized linear combo requires dataset-specific alpha weights that drift over time.
**How to avoid:**
- Use RRF (Reciprocal Rank Fusion) which operates on ranks, not scores
- Set k=60 as default (empirically validated across datasets)
- Only tune k if precision on top-5 results is critical (lower k=20-40) or broader recall needed (higher k=80-100)
- If linear combo required, test on 100+ sample queries quarterly to detect drift
**Warning signs:** Hybrid search performs worse than vector-only on semantic queries, worse than BM25-only on keyword queries, requires monthly retuning

### Pitfall 2: CrossEncoder Latency Explosion
**What goes wrong:** Reranking 500 candidates with CrossEncoder adds 500-1000ms latency, makes search unusably slow
**Why it happens:** CrossEncoders process each query-document pair independently. MiniLM-L6 scores ~1800 doc pairs/sec on CPU. 500 candidates = 280ms minimum + overhead.
**How to avoid:**
- Limit reranking to top 50-100 candidates from hybrid retrieval (not all vector results)
- Use faster CrossEncoder models: TinyBERT-L2 (9000 docs/sec, 69.84 NDCG) for speed, MiniLM-L6 (1800 docs/sec, 74.30 NDCG) for accuracy
- Batch candidates in groups of 50 for GPU acceleration if available
- Cache reranking results for duplicate queries (5-10% hit rate typical)
- Set timeout: if reranking takes >300ms, return pre-reranked RRF results
**Warning signs:** P95 search latency >1000ms, CPU usage spikes during searches, users report "slow loading"

### Pitfall 3: Metadata Filtering After Retrieval
**What goes wrong:** Vector search retrieves 1000 candidates, filter to 50 by date range, wasting 95% of compute
**Why it happens:** Applying metadata filters in application layer after vector similarity calculation instead of SQL WHERE clause
**How to avoid:**
- Build SQLAlchemy filters BEFORE vector search query (in WHERE clause)
- Use partial indexes if filtering by common predicates (e.g., source_type, recent dates)
- Test with EXPLAIN ANALYZE to verify filters applied before vector distance calculation
- For complex filters (JSONB array containment), ensure JSONB GIN indexes exist
**Warning signs:** Vector search queries take same time regardless of filter selectivity, high CPU usage, EXPLAIN shows filter after ORDER BY distance

### Pitfall 4: RRF K Parameter Misconception
**What goes wrong:** Team spends days tuning RRF k parameter (10, 20, 40, 60, 80, 100) with minimal accuracy improvement
**Why it happens:** Believing k parameter is critical tuning knob when research shows k=60 is robust across datasets
**How to avoid:**
- Start with k=60 (default) - works well for 95% of use cases
- Only tune k if: (a) top-5 precision is critical business metric (try k=20-40 for more top-heavy weighting), or (b) recall at 50+ is critical (try k=80-100)
- Test on 100+ representative queries before changing from default
- Document why k!=60 with benchmark results showing improvement
**Warning signs:** Team debates k values without benchmark data, k changes monthly based on anecdotes, k tuning takes more time than RRF implementation

### Pitfall 5: Stale Results for Trend Queries
**What goes wrong:** User searches "latest AI developments" and gets articles from 12 months ago that are semantically relevant but outdated
**Why it happens:** Semantic search ignores temporal signal - old comprehensive articles score higher than recent brief mentions
**How to avoid:**
- Detect trend queries with heuristics (keywords: "latest", "recent", "new", "2026") or LLM classification
- Apply time decay with 14-day half-life for detected trend queries (0.5^(age_days/14))
- Fuse semantic score + recency: 70% semantic, 30% recency for trend queries
- Store date_published on Document model (already in Phase 1), ensure NOT NULL for ingestion
- Consider separate recency-weighted index or query path for trend queries
**Warning signs:** User complaints about "old results", high bounce rate on trend queries, manual date filtering in UI

### Pitfall 6: BM25 Index Sync Lag
**What goes wrong:** New documents appear in vector search but not BM25 search for hours, creating inconsistent hybrid results
**Why it happens:** Using Python-based BM25 (bm25s, rank-bm25) requires separate index rebuild, not synchronized with PostgreSQL ingestion
**How to avoid:**
- If using Python BM25: rebuild index immediately after ingestion (blocking) or schedule incremental rebuilds every 5-10 minutes
- Store BM25 index version/timestamp, validate both indexes are at same ingestion checkpoint before hybrid search
- Alternative: use PostgreSQL-native BM25 (pg_search, pg_textsearch) which auto-updates like pgvector indexes
- Log index staleness metrics: max(vector_last_update, bm25_last_update) - min(...)
**Warning signs:** Hybrid search returns fewer results than vector-only, new documents missing from keyword queries, reranking scores inconsistent

### Pitfall 7: Fine-Grained Citations Without Structure
**What goes wrong:** LLM generates citations but they're inconsistent formats, broken links, or point to entire documents instead of sentences
**Why it happens:** Prompt asks LLM to "cite sources" without structured citation format or sentence-level metadata
**How to avoid:**
- Extract sentences with citation IDs during retrieval (Pattern 7: CitationExtractor)
- Embed citation IDs in prompt context: `[chunk_id.sentence_idx] sentence text`
- Instruct LLM to include [citation_id] after each claim in structured format
- Parse LLM response for [citation_id] patterns, map to stored citation metadata (url, title, sentence)
- Validate citation IDs exist in retrieval results before presenting to user
**Warning signs:** Citations point to wrong documents, broken URLs, "Source: various" instead of specific links, user reports "can't find claimed info in source"

## Code Examples

Verified patterns from official sources:

### Hybrid Retrieval Orchestrator (BM25 + Vector + RRF)
```python
# Source: Hybrid search patterns from ParadeDB, OpenSearch RRF, bm25s
from execution.vector_db.indexing import semantic_search
from execution.vector_db.models import KnowledgeChunk, Document
from execution.vector_db.connection import get_session
import bm25s
from typing import List, Dict, Optional

class HybridRetriever:
    """Orchestrate hybrid retrieval with BM25, vector search, and RRF fusion."""

    def __init__(self):
        self.bm25_index = None  # Loaded on first search

    def _load_bm25_index(self, tenant_id: str):
        """Load or build BM25 index for tenant."""
        try:
            # Try loading saved index
            self.bm25_index = bm25s.BM25.load("bm25_index")
        except FileNotFoundError:
            # Build index from scratch
            session = get_session()
            chunks = session.query(KnowledgeChunk).filter(
                KnowledgeChunk.tenant_id == tenant_id
            ).all()

            corpus = [chunk.content for chunk in chunks]
            corpus_tokens = bm25s.tokenize(corpus, stopwords="en")

            self.bm25_index = bm25s.BM25()
            self.bm25_index.index(corpus_tokens)
            self.bm25_index.save("bm25_index", corpus=corpus_tokens)

            self.doc_ids = [chunk.id for chunk in chunks]

    def _bm25_search(self, query: str, tenant_id: str, top_k: int) -> List[Dict]:
        """BM25 lexical search."""
        if self.bm25_index is None:
            self._load_bm25_index(tenant_id)

        query_tokens = bm25s.tokenize(query, stopwords="en")
        results, scores = self.bm25_index.retrieve(query_tokens, k=top_k)

        return [
            {"id": self.doc_ids[idx], "bm25_score": float(score), "rank": i + 1}
            for i, (idx, score) in enumerate(zip(results[0], scores[0]))
        ]

    def _vector_search(self, query: str, tenant_id: str, top_k: int) -> List[Dict]:
        """Vector semantic search (uses existing Phase 1 function)."""
        results = semantic_search(
            query_text=query,
            tenant_id=tenant_id,
            limit=top_k
        )

        # Add rank
        for i, result in enumerate(results):
            result["rank"] = i + 1

        return results

    def _rrf_fusion(
        self,
        bm25_results: List[Dict],
        vector_results: List[Dict],
        k: int = 60,
        bm25_weight: float = 1.0,
        vector_weight: float = 1.0
    ) -> List[Dict]:
        """Reciprocal Rank Fusion to merge BM25 and vector rankings.

        RRF formula: score(d) = sum(w_r / (k + rank_r(d))) for each source r
        """
        rrf_scores = {}
        doc_metadata = {}  # Store full document info

        # BM25 contributions
        for result in bm25_results:
            doc_id = result["id"]
            rank = result["rank"]
            rrf_scores[doc_id] = bm25_weight / (k + rank)
            doc_metadata[doc_id] = result

        # Vector contributions
        for result in vector_results:
            doc_id = result["id"]
            rank = result["rank"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + vector_weight / (k + rank)

            # Merge metadata (prefer vector results which have more fields)
            if doc_id not in doc_metadata:
                doc_metadata[doc_id] = result

        # Build fused results with normalized scores
        max_score = max(rrf_scores.values()) if rrf_scores else 1.0
        fused = []
        for doc_id, rrf_score in rrf_scores.items():
            result = doc_metadata[doc_id].copy()
            result["rrf_score"] = rrf_score / max_score  # Normalize 0-1
            fused.append(result)

        fused.sort(key=lambda x: x["rrf_score"], reverse=True)
        return fused

    def search(
        self,
        query: str,
        tenant_id: str,
        bm25_top_k: int = 50,
        vector_top_k: int = 50,
        rrf_k: int = 60,
        fusion_top_k: int = 50,
        bm25_weight: float = 1.0,
        vector_weight: float = 1.0,
        metadata_filters: Optional[Dict] = None
    ) -> List[Dict]:
        """Hybrid search with BM25 + vector + RRF fusion.

        Args:
            query: User query string
            tenant_id: Tenant identifier
            bm25_top_k: Number of results from BM25 retrieval
            vector_top_k: Number of results from vector retrieval
            rrf_k: RRF constant (default 60, empirically optimal)
            fusion_top_k: Number of results after RRF fusion
            bm25_weight: Weight for BM25 in fusion (default 1.0)
            vector_weight: Weight for vector in fusion (default 1.0)
            metadata_filters: Optional dict with date_range, source_types, topic_tags

        Returns:
            Fused and ranked results with rrf_score
        """
        # Stage 1: Parallel retrieval
        bm25_results = self._bm25_search(query, tenant_id, bm25_top_k)
        vector_results = self._vector_search(query, tenant_id, vector_top_k)

        # Stage 2: RRF fusion
        fused_results = self._rrf_fusion(
            bm25_results,
            vector_results,
            k=rrf_k,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight
        )

        # Stage 3: Apply metadata filters if provided
        if metadata_filters:
            fused_results = self._apply_metadata_filters(fused_results, metadata_filters)

        return fused_results[:fusion_top_k]

    def _apply_metadata_filters(self, results: List[Dict], filters: Dict) -> List[Dict]:
        """Apply metadata filters to fused results."""
        # TODO: Implement based on Pattern 5 (MetadataFilter)
        # For now, passthrough
        return results
```

### PostgreSQL-Native Hybrid Search with RRF (Alternative to Python BM25)
```sql
-- Source: https://www.paradedb.com/blog/hybrid-search-in-postgresql-the-missing-manual
-- PostgreSQL-native hybrid search using pg_search (ParadeDB) or pg_textsearch
-- This eliminates Python BM25 sync lag and keeps all retrieval in database

-- Create BM25 index (ParadeDB pg_search syntax)
CREATE INDEX idx_chunks_bm25 ON knowledge_chunks
USING bm25 (
  id,
  content::pdb.simple('stemmer=english')
)
WITH (key_field=id);

-- Hybrid search query with RRF fusion
WITH bm25_results AS (
  SELECT id, content, document_id,
         ROW_NUMBER() OVER (ORDER BY pdb.score(id) DESC) AS rank
  FROM knowledge_chunks
  WHERE tenant_id = 'default'
    AND content ||| 'PostgreSQL vector search'
  LIMIT 50
),
vector_results AS (
  SELECT id, content, document_id,
         ROW_NUMBER() OVER (ORDER BY embedding <=> :query_embedding) AS rank
  FROM knowledge_chunks
  WHERE tenant_id = 'default'
    AND embedding IS NOT NULL
  LIMIT 50
),
rrf_fusion AS (
  -- BM25 contributions
  SELECT id, 1.0 / (60 + rank) AS rrf_score
  FROM bm25_results

  UNION ALL

  -- Vector contributions
  SELECT id, 1.0 / (60 + rank) AS rrf_score
  FROM vector_results
)
SELECT
  kc.id,
  kc.content,
  kc.topic_tags,
  kc.entities,
  d.title,
  d.source_type,
  d.url,
  d.date_published,
  SUM(rf.rrf_score) AS fused_score
FROM rrf_fusion rf
JOIN knowledge_chunks kc ON rf.id = kc.id
JOIN documents d ON kc.document_id = d.id
GROUP BY kc.id, kc.content, kc.topic_tags, kc.entities, d.title, d.source_type, d.url, d.date_published
ORDER BY fused_score DESC
LIMIT 50;
```

### CrossEncoder Reranking with Batch Processing
```python
# Source: https://www.sbert.net/docs/cross_encoder/usage/usage.html
from sentence_transformers import CrossEncoder
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    """Rerank retrieval candidates using CrossEncoder."""

    # Model selection by speed/accuracy tradeoff
    MODELS = {
        "fast": "cross-encoder/ms-marco-TinyBERT-L2-v2",      # 9000 docs/sec, 69.84 NDCG
        "balanced": "cross-encoder/ms-marco-MiniLM-L6-v2",    # 1800 docs/sec, 74.30 NDCG
        "accurate": "cross-encoder/ms-marco-MiniLM-L12-v2"    # 960 docs/sec, 74.31 NDCG
    }

    def __init__(self, model_profile: str = "balanced"):
        """Initialize CrossEncoder with model profile.

        Args:
            model_profile: "fast", "balanced", or "accurate"
        """
        model_name = self.MODELS.get(model_profile, self.MODELS["balanced"])
        self.model = CrossEncoder(model_name)
        self.model_profile = model_profile
        logger.info(f"Loaded CrossEncoder: {model_name} (profile: {model_profile})")

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 10,
        batch_size: int = 32
    ) -> List[Dict]:
        """Rerank candidates and return top-K by relevance.

        Args:
            query: User query
            candidates: List of dicts with "id" and "content" keys
            top_k: Number of results to return after reranking
            batch_size: Batch size for CrossEncoder scoring (default 32)

        Returns:
            Reranked results with "rerank_score" added
        """
        if not candidates:
            return []

        # Limit candidates to avoid latency explosion
        max_rerank = 100
        if len(candidates) > max_rerank:
            logger.warning(
                f"Limiting reranking from {len(candidates)} to {max_rerank} candidates"
            )
            candidates = candidates[:max_rerank]

        # Build query-document pairs
        pairs = [(query, candidate["content"][:2000]) for candidate in candidates]  # Truncate to 2000 chars

        # Score in batches (model handles batching internally)
        import time
        start = time.time()
        scores = self.model.predict(pairs, batch_size=batch_size)
        elapsed = time.time() - start

        logger.info(
            f"CrossEncoder reranked {len(candidates)} candidates in {elapsed*1000:.1f}ms "
            f"({len(candidates)/elapsed:.0f} docs/sec)"
        )

        # Add scores and sort
        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_k]

    def rank(self, query: str, passages: List[str]) -> List[Dict]:
        """Simplified interface matching sentence-transformers API.

        Returns:
            List of dicts with keys: corpus_id, score
        """
        scores = self.model.predict([(query, passage) for passage in passages])

        results = [
            {"corpus_id": i, "score": float(score)}
            for i, score in enumerate(scores)
        ]
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
```

### Recency-Aware Hybrid Retrieval with Time Decay
```python
# Source: https://arxiv.org/html/2509.19376 (Solving Freshness in RAG)
from datetime import datetime, timedelta
import math
from typing import List, Dict

class RecencyAwareRetriever:
    """Hybrid retrieval with time decay for trend-sensitive queries."""

    # Trend query keywords
    TREND_KEYWORDS = {
        "latest", "recent", "new", "breaking", "current", "today",
        "2026", "now", "upcoming", "emerging", "trending"
    }

    def __init__(self, half_life_days: int = 14):
        """Initialize with half-life parameter.

        Args:
            half_life_days: Time for recency score to decay to 50% (default 14 days)
        """
        self.half_life_days = half_life_days

    def is_trend_query(self, query: str) -> bool:
        """Detect if query is trend-sensitive."""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.TREND_KEYWORDS)

    def time_decay(self, date_published: datetime) -> float:
        """Calculate exponential decay based on document age.

        Formula: decay = 0.5 ^ (age_days / half_life_days)

        Returns:
            Float between 0 and 1 (1 = today, 0.5 = half_life_days ago)
        """
        if date_published is None:
            return 0.5  # Neutral for unknown dates

        age_days = (datetime.utcnow() - date_published).days
        age_days = max(0, age_days)  # Handle future dates

        decay = math.pow(0.5, age_days / self.half_life_days)
        return decay

    def apply_recency_boost(
        self,
        results: List[Dict],
        semantic_weight: float = 0.7,
        recency_weight: float = 0.3
    ) -> List[Dict]:
        """Apply time decay to search results.

        Args:
            results: List of search results with rrf_score and date_published
            semantic_weight: Weight for semantic score (default 0.7)
            recency_weight: Weight for recency score (default 0.3)

        Returns:
            Results with fused_score combining semantic + recency
        """
        for result in results:
            date_published = result.get("date_published")
            rrf_score = result.get("rrf_score", 0.5)

            recency = self.time_decay(date_published)
            result["recency_score"] = recency
            result["fused_score"] = (
                semantic_weight * rrf_score + recency_weight * recency
            )

        # Re-sort by fused score
        results.sort(key=lambda x: x["fused_score"], reverse=True)
        return results

    def search_with_recency(
        self,
        query: str,
        tenant_id: str,
        top_k: int = 10
    ) -> List[Dict]:
        """Hybrid search with automatic recency detection.

        Args:
            query: User query
            tenant_id: Tenant identifier
            top_k: Number of results to return

        Returns:
            Search results, recency-boosted if trend query detected
        """
        from execution.vector_db.retrieval import HybridRetriever

        retriever = HybridRetriever()

        # Get hybrid results
        results = retriever.search(query, tenant_id, fusion_top_k=50)

        # Apply recency boost if trend query
        if self.is_trend_query(query):
            results = self.apply_recency_boost(
                results,
                semantic_weight=0.7,
                recency_weight=0.3
            )

        return results[:top_k]
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| rank-bm25 for Python BM25 | bm25s (scipy sparse matrices) | Late 2025 | 500x performance improvement (2 QPS → 573 QPS), enables real-time hybrid search in Python without database extensions |
| Linear weighted combination for hybrid search | Reciprocal Rank Fusion (RRF) | Mainstream adoption 2024-2025 | Eliminates score normalization problem, robust across datasets without tuning, k=60 default works for 95% of use cases |
| Elasticsearch for BM25 + pgvector for vectors | PostgreSQL-native hybrid (pg_search, pg_textsearch) | 2025-2026 | Single database for BM25 + vector search, eliminates sync lag, ACID guarantees, but adds extension deployment complexity |
| LLM-based reranking (GPT-4, Claude) | CrossEncoder reranking (sentence-transformers) | Established 2024+ | 5-10x faster (150-250ms vs 1000-2000ms), 10-50x cheaper, matches/exceeds LLM accuracy on MS-MARCO benchmarks |
| Document-level citations | Sentence-level fine-grained citations | Active research 2025-2026 | Better source attribution, clickable citations, reduced hallucination verification effort, but requires structured prompt engineering |
| Static relevance ranking | Time-decay recency scoring (half-life 14 days) | Research published late 2025 | Perfect Latest@10 accuracy on trend queries, addresses temporal drift, essential for breaking news and fast-moving technical domains |

**Deprecated/outdated:**
- **rank-bm25 for production:** 500x slower than bm25s, only acceptable for <10k documents or prototyping
- **Linear score fusion:** Requires dataset-specific calibration, drifts over time, RRF is more robust
- **Document-level citations only:** Modern RAG systems provide sentence-level attribution for verifiability
- **Elasticsearch for small-scale hybrid search:** PostgreSQL-native BM25 extensions (pg_search, pg_textsearch) eliminate deployment complexity for <10M documents

## Open Questions

Things that couldn't be fully resolved:

1. **PostgreSQL-native BM25 extensions (pg_search, pg_textsearch) stability and deployment complexity**
   - What we know: ParadeDB pg_search and pg_textsearch provide native BM25 in PostgreSQL, eliminating Python sync lag and enabling transactional BM25+vector queries
   - What's unclear: Production stability (both are relatively new, 2024-2025 releases), upgrade path complexity, cloud provider support (AWS RDS, Azure, GCP don't natively support), Docker/Kubernetes deployment overhead
   - Recommendation: Start with Python-based bm25s for Phase 2 (simpler deployment, proven stability). Evaluate PostgreSQL-native BM25 in Phase 3-4 after testing in staging environment. Trade-off: Python BM25 has sync lag but simpler ops, native BM25 has perfect consistency but deployment complexity.

2. **Optimal RRF weighting for domain-specific queries (keyword-heavy vs semantic-heavy)**
   - What we know: RRF k=60 is robust default. Unweighted RRF (bm25_weight=1.0, vector_weight=1.0) works for balanced queries. Keyword queries (exact terms, names) benefit from higher BM25 weight, semantic queries (conceptual, paraphrased) benefit from higher vector weight.
   - What's unclear: Automatic query classification (keyword vs semantic) accuracy, optimal per-query-type weights, whether dynamic weighting is worth implementation complexity
   - Recommendation: Start with unweighted RRF (1.0, 1.0) and k=60 for all queries. Log query patterns and user satisfaction. In Phase 4-5, implement simple heuristic classifier (quotes → keyword, question words → semantic) and test with weights like (1.5, 0.5) for keyword, (0.5, 1.5) for semantic. Measure precision@10 improvement before productionizing.

3. **CrossEncoder model selection: speed vs accuracy tradeoff for GhostWriter domain**
   - What we know: TinyBERT-L2 (9000 docs/sec, 69.84 NDCG), MiniLM-L6 (1800 docs/sec, 74.30 NDCG), MiniLM-L12 (960 docs/sec, 74.31 NDCG). MiniLM-L12 vs L6 has minimal accuracy gain (0.01 NDCG) with 2x latency cost.
   - What's unclear: Whether GhostWriter's technical content (newsletters, papers, RSS) matches MS-MARCO training distribution (web search queries). Domain mismatch could favor different models or require fine-tuning.
   - Recommendation: Start with MiniLM-L6 (balanced profile). Run offline evaluation on 100 sample queries from production logs. Measure: MRR@10, P@10, user click-through rate on top result. If P@10 < 70%, test TinyBERT-L2 (faster but less accurate) and measure if speed improves user satisfaction. Only fine-tune if accuracy is <60% on representative queries.

4. **Recency scoring half-life parameter for different content types (breaking news vs evergreen)**
   - What we know: Research recommends 14-day half-life for general trend queries. Half-life controls decay rate: 14 days means content from 14 days ago has 50% recency score, 28 days = 25%, etc.
   - What's unclear: Optimal half-life for GhostWriter's mixed content. RSS news may need shorter (7 days), academic papers longer (30-60 days), email newsletters medium (14 days). Single global half-life may not fit all source types.
   - Recommendation: Start with 14-day half-life for all content (research-validated). Log date_published distribution per source_type. In Phase 4, implement per-source-type half-life: RSS=7 days, papers=60 days, email=14 days. A/B test on trend queries ("latest AI news" vs "database architecture patterns") to measure impact.

5. **Metadata filter performance at scale (JSONB array containment with GIN indexes)**
   - What we know: SQLAlchemy JSONB filtering works for topic_tags.contains([tag]), requires GIN index on topic_tags for performance. GIN indexes add storage overhead and slow inserts.
   - What's unclear: Performance at 100k-1M chunks with complex filters (multiple tags, date range, source type). Whether GIN index maintenance cost outweighs query benefit. PostgreSQL query planner's index selection behavior with hybrid BM25+vector+metadata filters.
   - Recommendation: Add GIN index on topic_tags in Phase 2: `CREATE INDEX idx_topic_tags_gin ON knowledge_chunks USING gin(topic_tags)`. Run EXPLAIN ANALYZE on filtered queries with 10k+ chunks. If query time >500ms or index not used, consider denormalizing: add has_tag_X boolean columns for top 20 most-filtered tags, use B-tree indexes. Monitor insert performance degradation.

6. **Fine-grained citation extraction reliability (sentence splitting edge cases)**
   - What we know: pysbd handles most sentence boundary edge cases (Dr., Mr., decimals). Citation IDs format: `chunk_id.sentence_idx`. LLM instructed to include [citation_id] after claims.
   - What's unclear: LLM citation compliance rate (how often does it actually cite?), citation accuracy (does it cite the right sentence?), handling of multi-sentence claims, citations in code blocks or bullet lists
   - Recommendation: Implement citation extraction (Pattern 7) in Phase 2. Log: citation_compliance_rate (% of claims with [citation_id]), citation_accuracy (does cited sentence support claim?). Manual review 50 generated articles. If compliance <80%, add structured output (JSON with {claim, citation_id} pairs) instead of inline citations. If accuracy <70%, add citation validation step (check cited sentence supports claim via entailment model).

## Sources

### Primary (HIGH confidence)
- [bm25s GitHub](https://github.com/xhluca/bm25s) - Fast BM25 implementation, 500x speedup, scipy sparse matrices, benchmarks
- [rank-bm25 GitHub](https://github.com/dorianbrown/rank_bm25) - Standard Python BM25 library, multiple variants (Okapi, BM25L, BM25+)
- [ParadeDB Hybrid Search Guide](https://www.paradedb.com/blog/hybrid-search-in-postgresql-the-missing-manual) - PostgreSQL-native BM25 with pg_search, RRF fusion in SQL, integration with pgvector
- [ParadeDB RRF Documentation](https://www.paradedb.com/learn/search-concepts/reciprocal-rank-fusion) - RRF formula, k parameter, weighted RRF, implementation examples
- [Sentence-Transformers CrossEncoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html) - CrossEncoder usage, reranking API, two-stage retrieval patterns
- [MS-MARCO CrossEncoder Models](https://www.sbert.net/docs/pretrained-models/ce-msmarco.html) - Model benchmarks, speed/accuracy tradeoffs, version 2 recommendations
- [Tiger Data pg_textsearch](https://www.tigerdata.com/blog/introducing-pg_textsearch-true-bm25-ranking-hybrid-retrieval-postgres) - PostgreSQL BM25 extension, IDF, term frequency saturation, installation
- [ArXiv: Solving Freshness in RAG](https://arxiv.org/html/2509.19376) - Recency scoring research, half-life decay (14 days), semantic-temporal fusion, Latest@10 accuracy
- [Tensorlake: Citation-Aware RAG](https://www.tensorlake.ai/blog/rag-citations) - Fine-grained citation implementation, spatial anchors, sentence-level attribution, LLM prompt patterns

### Secondary (MEDIUM confidence)
- [OpenSearch RRF Announcement](https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/) - RRF in OpenSearch 2.19, k parameter guidance, hybrid query patterns
- [Azure AI Search Hybrid Search](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking) - RRF scoring in Azure, multiple query fusion, production usage patterns
- [Medium: RRF Explained](https://medium.com/@mudassar.hakim/the-quiet-hero-of-rag-pipelines-reciprocal-rank-fusion-explained-1b83af68b997) - RRF intuition, k parameter tuning, RAG pipeline integration (Jan 2026)
- [Medium: CrossEncoder Reranking](https://medium.com/@rossashman/the-art-of-rag-part-3-reranking-with-cross-encoders-688a16b64669) - Two-stage retrieval, performance benefits, practical guide
- [Milvus: CrossEncoder Overhead](https://milvus.io/ai-quick-reference/what-is-the-overhead-of-using-a-crossencoder-for-reranking-results-compared-to-just-using-biencoder-embeddings-and-how-can-you-minimize-that-extra-cost-in-a-system) - Latency analysis, cost mitigation strategies, candidate limiting
- [LangChain Metadata Filtering](https://www.geeksforgeeks.org/artificial-intelligence/metadata-filtering-in-langchain/) - Metadata filtering patterns, self-query retriever, date range examples
- [Medium: SQLAlchemy JSONB](https://medium.com/@PostgradExpat/querying-jsonb-and-date-diffs-with-postgresql-sqlalchemy-and-python-c3f55d5e7d11) - JSONB querying in SQLAlchemy, date filtering, Python examples
- [Qdrant: Reranking in Hybrid Search](https://qdrant.tech/documentation/advanced-tutorials/reranking-hybrid-search/) - Hybrid search reranking patterns, fusion methods, implementation guide

### Tertiary (LOW confidence - flagged for validation)
- WebSearch results on optimal RRF k parameter per domain - general guidance (k=60 default), but domain-specific tuning needs validation
- WebSearch results on CrossEncoder model selection for technical content - MS-MARCO benchmarks are web search, technical content distribution may differ
- WebSearch results on per-source-type recency half-life - research shows 14-day default, but source-specific values need testing
- WebSearch results on LLM citation compliance rates - anecdotal evidence of 70-90% compliance, needs measurement in production

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - bm25s, sentence-transformers, RRF are well-established 2025-2026 patterns with official documentation and benchmarks
- Architecture patterns: HIGH - RRF formula, CrossEncoder usage, bm25s implementation verified through official docs and research papers
- PostgreSQL-native BM25: MEDIUM - pg_search and pg_textsearch are production-ready but relatively new (2024-2025), deployment complexity and cloud support needs validation
- Pitfalls: HIGH - Score normalization issues, CrossEncoder latency, metadata filtering performance documented in multiple vendor blogs and case studies
- Recency scoring: MEDIUM - Half-life decay (14 days) validated in ArXiv research, but per-source-type tuning needs testing
- Fine-grained citations: MEDIUM - Implementation patterns documented (Tensorlake blog), but LLM citation compliance and accuracy needs production validation

**Research date:** 2026-02-10
**Valid until:** 2026-03-12 (30 days - fast-moving domain with new libraries and research, monthly review recommended)
