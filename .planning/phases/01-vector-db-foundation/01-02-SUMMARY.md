---
phase: 01-vector-db-foundation
plan: 02
subsystem: embedding-pipeline
tags: [openai, embeddings, batch-api, token-tracking, cost-guardrails]

dependency_graph:
  requires:
    - 01-01 (models.py, config.py VectorDBConfig)
  provides:
    - EmbeddingClient with sync and batch embedding modes
    - TokenTracker with daily usage persistence and budget enforcement
    - Cost guardrail system blocking embedding when limit exceeded
  affects:
    - 01-03 (chunking may need to estimate tokens before embedding)
    - 01-04 (ingestion orchestrator calls EmbeddingClient)

tech_stack:
  added: []
  patterns:
    - Lazy OpenAI client initialization (defer API key validation)
    - Tenacity retry with exponential backoff for transient errors
    - JSON file-based daily token tracking with automatic reset
    - OpenAI Batch API polling pattern for 50% cost savings

file_tracking:
  created:
    - execution/vector_db/embeddings.py
    - execution/vector_db/token_tracking.py
  modified:
    - execution/vector_db/__init__.py

decisions:
  - id: D-01-02-001
    decision: "Lazy OpenAI client initialization"
    rationale: "OpenAI SDK raises on init if no API key set; lazy init allows import and instantiation without key, deferring validation to first actual API call"
  - id: D-01-02-002
    decision: "JSON file for token tracking (not database)"
    rationale: "Token tracking is operational metadata, not knowledge data. File-based is simpler, no DB dependency, and sufficient for single-instance daily counters"
  - id: D-01-02-003
    decision: "Synchronous OpenAI client (not async)"
    rationale: "Matches existing codebase pattern (base_agent.py, sources/database.py). Async adds complexity without benefit for current batch ingestion workloads"

metrics:
  duration: "3 min"
  completed: "2026-02-10"
---

# Phase 1 Plan 02: OpenAI Embedding Pipeline Summary

**One-liner:** OpenAI embedding client with sync/batch modes, daily token tracking, and cost guardrails using tenacity retry logic.

## What Was Built

### Token Tracking (execution/vector_db/token_tracking.py)
- **TokenTracker** class persisting daily usage to `{TEMP_DIR}/token_usage.json`
- Automatic daily reset when date changes (UTC)
- `check_budget()` blocks embedding when estimated tokens exceed remaining limit
- Warning at 10% remaining threshold, error log when limit exceeded
- `estimate_tokens()` helper at 1 token per 4 characters
- Module-level convenience functions: `check_daily_limit()`, `record_usage()`

### Embedding Client (execution/vector_db/embeddings.py)
- **EmbeddingClient** with lazy OpenAI client initialization
- **Sync mode** (`embed_texts`): Up to 100 texts per call, immediate response
  - Tenacity retry: 3 attempts, exponential backoff (2-30s) + jitter
  - Transient error detection: RateLimitError, APITimeoutError, APIConnectionError, InternalServerError
  - Text truncation at 8191-token limit (~32,764 chars)
  - Budget check before API call, actual token recording from response
- **Batch mode** (`batch_embed_texts`): JSONL upload, polling, result extraction
  - 50% cost savings via OpenAI Batch API (24h completion window)
  - Progress logging during polling
  - Results sorted by custom_id to maintain input order
  - Total token usage accumulated from all result objects
- **Custom exceptions**: TokenBudgetExceeded, BatchEmbeddingFailed, EmbeddingError
- Module-level convenience functions: `embed_texts()`, `batch_embed_texts()`

### Package Exports (execution/vector_db/__init__.py)
- Updated with EmbeddingClient, TokenTracker, and all custom exceptions

## Commits

| Commit | Description |
|--------|-------------|
| 89ffb64 | feat(01-02): token tracking and daily limit enforcement |
| 83c9c3e | feat(01-02): OpenAI embedding client with sync and batch modes |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Lazy OpenAI client initialization**
- **Found during:** Task 2 verification
- **Issue:** OpenAI SDK raises `OpenAIError` during `__init__` when no API key is set, preventing import-time instantiation
- **Fix:** Made `client` a lazy property that initializes on first use, allowing instantiation without API key
- **Files modified:** execution/vector_db/embeddings.py
- **Commit:** 83c9c3e

## Verification Results

- `from execution.vector_db.embeddings import EmbeddingClient, TokenBudgetExceeded, BatchEmbeddingFailed`: OK
- `EmbeddingClient().model` returns "text-embedding-3-small": OK
- `TokenTracker(daily_limit=1000).get_usage_today()` returns correct dict: OK
- `estimate_tokens(['hello world'])` returns 2: OK
- `check_daily_limit(100)` returns True: OK
- Existing `execution.sources.database` imports unaffected: OK

## Next Phase Readiness

Plan 01-03 (chunking + auto-tagging) can proceed -- it needs:
- `estimate_tokens()` from token_tracking -- available
- Config chunk sizes from `config.vector_db.CHUNK_SIZE_*` -- available

Plan 01-04 (ingestion orchestrator) can proceed -- it needs:
- `EmbeddingClient.embed_texts()` and `batch_embed_texts()` -- available
- `TokenTracker` for budget management -- available
