---
phase: 01-vector-db-foundation
plan: 03
subsystem: content-processing
tags: [chunking, tagging, nlp, pysbd, html2text, llm]

dependency_graph:
  requires:
    - 01-01 (models.py for KnowledgeChunk, config.py for VectorDBConfig)
  provides:
    - SemanticChunker with content-type-aware splitting strategies
    - AutoTagger for LLM-powered topic classification and entity extraction
    - chunk_content() convenience function for pipeline integration
    - auto_tag() convenience function for pipeline integration
  affects:
    - 01-04 (ingestion orchestrator calls chunk_content -> auto_tag -> embed)

tech_stack:
  added:
    - pysbd (sentence boundary detection, 97.92% accuracy)
    - html2text (HTML-to-text conversion for email content)
  patterns:
    - Content-type routing (email/paper/rss/default strategies)
    - Singleton pattern for module-level convenience functions
    - BaseAgent reuse for LLM calls (multi-provider routing)
    - Graceful degradation (tagging returns empty result on failure)

file_tracking:
  created:
    - execution/vector_db/chunking.py
    - execution/vector_db/tagging.py
  modified:
    - requirements.txt

decisions:
  - id: D-01-03-001
    decision: "Use pysbd instead of regex for sentence splitting"
    rationale: "Researcher finding: pysbd handles abbreviations (Dr. Smith), decimals ($3.50), URLs with 97.92% accuracy. Pure Python, ~50KB, zero heavy deps."
  - id: D-01-03-002
    decision: "Use html2text for email HTML parsing"
    rationale: "Researcher finding: emails often arrive as HTML. html2text converts to clean markdown before boilerplate stripping. Keep hand-rolled regex for email-specific domain logic (unsubscribe, signatures)."
  - id: D-01-03-003
    decision: "Rule-based chunker, not LLM-powered"
    rationale: "LLM chunking is expensive per call and adds latency. Rule-based handles 90% of cases deterministically. LLM-powered boundary detection deferred to Phase 3 for newsletters."

metrics:
  duration: "4 min"
  completed: "2026-02-10"
---

# Phase 1 Plan 03: Semantic Chunking + Auto-Tagging Summary

**One-liner:** Rule-based semantic chunker with content-type strategies (email/paper/RSS) using pysbd sentence splitting, plus LLM-powered auto-tagger using BaseAgent for topic/entity extraction.

## What Was Built

### Semantic Chunking Pipeline (execution/vector_db/chunking.py)

**SemanticChunker class** with content-type-aware splitting:

| Content Type | Target Tokens | Overlap | Strategy |
|-------------|--------------|---------|----------|
| email | 400 | 12% | HTML->text via html2text, strip boilerplate, paragraph/sentence split |
| paper | 512 | 15% | Section header detection (markdown/#/ALL CAPS/numbered), per-section splitting |
| rss | 256 | 10% | Single-chunk if small, paragraph/sentence split if oversized |
| default | 400 | 15% | Paragraph boundaries, then sentence fallback |

**Key features:**
- `pysbd` for sentence splitting (handles "Dr. Smith", "$3.50", URLs)
- `html2text` for HTML email conversion before boilerplate stripping
- Email boilerplate stripping (unsubscribe, signatures, "View in browser", copyright)
- Small chunk merging (min 50 tokens)
- Word-boundary-aware overlap (doesn't split mid-word)
- Paper section detection (markdown headers, ALL CAPS, numbered sections)
- Deterministic: same input always produces same output

**Chunk dataclass:** text, chunk_index, token_count, topic_hint, metadata

### AI Auto-Tagging (execution/vector_db/tagging.py)

**AutoTagger class** using BaseAgent for LLM calls:
- Reuses existing multi-provider routing (Groq default: llama-3.3-70b-versatile)
- Content truncated to 4000 chars for speed/cost
- Structured JSON prompt requesting topics, entities, content type, confidence
- Three-layer parsing: direct JSON -> json_parser utility -> regex fallback

**TagResult dataclass:**
- `topic_tags`: 3-7 keywords (e.g., ["AI", "databases", "Python"])
- `entities`: [{"type": "ORG|PERSON|TECH|CONCEPT", "value": "name"}]
- `source_type_label`: newsletter/research/news/tutorial/opinion
- `confidence`: 0.0-1.0

**Batch processing:** `tag_batch()` processes items sequentially with per-item error isolation.

**Graceful degradation:** Returns empty TagResult on any failure -- tagging is enrichment, not critical path.

## Commits

| Commit | Description |
|--------|-------------|
| c4364f3 | feat(01-03): semantic chunking pipeline with content-type strategies |
| 0737f32 | feat(01-03): AI auto-tagging with topic classification and entity extraction |

## Deviations from Plan

### Researcher-Directed Changes

**1. [Researcher] pysbd instead of regex for sentence splitting**
- **Plan said:** Use regex (split on `. `, `! `, `? ` followed by capital letter)
- **Changed to:** `pysbd` library for sentence boundary detection
- **Rationale:** 97.92% accuracy, handles abbreviations/decimals/URLs. Lightweight pure Python.
- **Files:** execution/vector_db/chunking.py, requirements.txt

**2. [Researcher] html2text for email HTML parsing**
- **Plan said:** `_strip_email_boilerplate()` using regex only
- **Changed to:** `html2text` as first step, then hand-rolled regex for domain-specific boilerplate
- **Rationale:** Emails often arrive as HTML. html2text handles conversion cleanly; regex handles email-specific patterns no library covers.
- **Files:** execution/vector_db/chunking.py, requirements.txt

## Decisions Made

1. **pysbd over regex** (D-01-03-001): Better accuracy for edge cases with minimal dependency cost.

2. **html2text + regex combo** (D-01-03-002): Two-stage email processing -- library for HTML conversion, hand-rolled for domain-specific boilerplate.

3. **Rule-based chunking** (D-01-03-003): Deterministic, testable, no per-call cost. LLM-powered chunking deferred to Phase 3 for newsletter-specific enhancement.

## Verification Results

- `from execution.vector_db.chunking import chunk_content, SemanticChunker, Chunk`: OK
- `from execution.vector_db.tagging import AutoTagger, TagResult, auto_tag`: OK
- chunk_content produces correctly-ordered Chunks for all content types (email, paper, rss, default)
- Email boilerplate stripped ("Unsubscribe", "View in browser" removed)
- AutoTagger instantiates with correct model (llama-3.3-70b-versatile via Groq)
- JSON parsing handles plain JSON, markdown-fenced JSON, and regex fallback
- Config modules unmodified (VectorDBConfig chunk sizes confirmed)

## Next Phase Readiness

Plan 01-04 (ingestion orchestrator) can proceed -- it needs:
- `execution/vector_db/chunking.py` (chunk_content) -- available
- `execution/vector_db/tagging.py` (auto_tag) -- available
- `execution/vector_db/models.py` (Document, KnowledgeChunk) -- available (from 01-01)
- Embedding pipeline (from 01-02) -- needed for chunk -> tag -> embed flow
