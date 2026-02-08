# Feature: Source Ingestion — Upgrade Plan

**Agent**: Engineering Upgrade Agent
**Date**: 2026-02-08
**Constraint**: No rewrites. Incremental fixes only.

---

## Upgrade 1: Add Source Health Check and Registration Warnings

**Priority**: HIGH
**Effort**: S
**Regression Risk**: None (additive)

### What to Change

**In `__init__.py:189-207`, replace silent `except ImportError: pass`:**

```python
# BEFORE (silent):
try:
    from . import reddit_source
except ImportError:
    pass

# AFTER (logged):
import logging
_logger = logging.getLogger("ghostwriter.sources")

_IMPORT_WARNINGS = []

try:
    from . import reddit_source
except ImportError as e:
    _IMPORT_WARNINGS.append(f"RedditSource unavailable: {e}")
    _logger.warning(f"RedditSource import failed: {e}")

try:
    from . import gmail_source
except ImportError as e:
    _IMPORT_WARNINGS.append(f"GmailSource unavailable: {e}")
    _logger.warning(f"GmailSource import failed: {e}")

# ... same for hackernews_source, rss_source
```

**Add health check to SourceManager:**

```python
def health_report(self) -> dict:
    """Check health of all registered and configured sources."""
    registered = SourceFactory.get_registered_sources()
    expected = [SourceType.REDDIT, SourceType.HACKERNEWS, SourceType.RSS]
    missing = [s for s in expected if s not in registered]

    return {
        "registered_sources": [s.value for s in registered],
        "expected_sources": [s.value for s in expected],
        "missing_sources": [s.value for s in missing],
        "import_warnings": _IMPORT_WARNINGS,
        "healthy": len(missing) == 0,
    }
```

Call `health_report()` at pipeline start and log warnings.

---

## Upgrade 2: Fix Reddit Dual-DB Default (Write to Unified Table)

**Priority**: CRITICAL (ship-blocking)
**Effort**: S
**Regression Risk**: Low

### What to Change

**In `reddit_source.py:425-431` (CLI main), write to unified table by default:**

```python
# BEFORE:
legacy_inserted = source.insert_to_legacy_db(result.items)
print(f"✓ Inserted {legacy_inserted} to legacy 'posts' table")

if args.unified:
    unified_inserted = source.insert_to_unified_db(result.items)

# AFTER:
legacy_inserted = source.insert_to_legacy_db(result.items)
print(f"✓ Inserted {legacy_inserted} to legacy 'posts' table")

# Always write to unified table (pulse_aggregator reads from here)
unified_inserted = source.insert_to_unified_db(result.items)
print(f"✓ Inserted {unified_inserted} to unified 'content_items' table")
```

Remove the `--unified` flag or keep it as a no-op for backward compatibility.

**Also**: HN and RSS CLI entry points only print to stdout — they need DB insert calls too:

```python
# In hackernews_source.py main() and rss_source.py main(), after fetch:
# Insert to unified DB (so pulse_aggregator can see results)
from .reddit_source import DB_PATH  # Or centralize DB_PATH to config
import sqlite3, json

conn = sqlite3.connect(DB_PATH)
for item in result.items:
    try:
        conn.execute(
            "INSERT INTO content_items (...) VALUES (...)",
            (item.source_type.value, item.source_id, ...)
        )
    except sqlite3.IntegrityError:
        pass
conn.commit()
conn.close()
```

**Better long-term**: Add `insert_to_db()` to `ContentSource` base class so all sources share the same DB write logic.

---

## Upgrade 3: Add Retry with Exponential Backoff

**Priority**: HIGH
**Effort**: S
**Regression Risk**: Low

### What to Change

Add `tenacity` to `requirements.txt`. Apply retry to network calls:

**In `reddit_source.py:170`:**

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
)
def _fetch_subreddit_rss(self, subreddit_name: str) -> List[Dict[str, Any]]:
    ...
```

**In `hackernews_source.py:94` and `117`:**

Apply same retry decorator to the top-stories fetch and individual item fetch.

**In `rss_source.py:196`:**

`feedparser.parse()` handles its own HTTP, but wrap with retry for network errors.

### Fallback Behavior

On final retry failure, return empty list for that source/subreddit (current behavior), but log the failure with attempt count for health monitoring.

---

## Upgrade 4: Add Source Failure Circuit Breaker

**Priority**: HIGH (ship-blocking per dossier)
**Effort**: M
**Regression Risk**: Low

### What to Change

Create a simple circuit breaker for SourceManager:

```python
# In __init__.py or new file execution/sources/circuit_breaker.py

from dataclasses import dataclass, field
from time import time

@dataclass
class SourceCircuit:
    """Circuit breaker state for a single source."""
    source_type: str
    failure_count: int = 0
    failure_threshold: int = 3
    last_failure: float = 0
    cooldown_seconds: float = 300  # 5 minutes
    state: str = "closed"  # closed, open, half-open

    def record_success(self):
        self.failure_count = 0
        self.state = "closed"

    def record_failure(self):
        self.failure_count += 1
        self.last_failure = time()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"

    def should_attempt(self) -> bool:
        if self.state == "closed":
            return True
        if self.state == "open":
            if time() - self.last_failure > self.cooldown_seconds:
                self.state = "half-open"
                return True
            return False
        return True  # half-open: try once
```

**Update `SourceManager.fetch_all()`:**

```python
def fetch_all(self, ...) -> Dict[SourceType, FetchResult]:
    results = {}
    for source_type, source in self._sources.items():
        circuit = self._circuits.get(source_type)
        if circuit and not circuit.should_attempt():
            results[source_type] = FetchResult(
                items=[], success=False,
                error_message=f"Circuit open (last {circuit.failure_count} attempts failed)"
            )
            continue

        try:
            result = source.fetch(...)
            results[source_type] = result
            if circuit:
                circuit.record_success()
        except Exception as e:
            results[source_type] = FetchResult(items=[], success=False, error_message=str(e))
            if circuit:
                circuit.record_failure()

    # Check minimum source threshold
    successful = sum(1 for r in results.values() if r.success)
    if successful == 0:
        raise AllSourcesFailedError(
            f"All {len(results)} sources failed. "
            f"Errors: {'; '.join(r.error_message for r in results.values() if r.error_message)}"
        )

    return results
```

---

## Upgrade 5: Fix Reddit Engagement Data (Optional OAuth2)

**Priority**: MEDIUM
**Effort**: M
**Regression Risk**: Low

### What to Change

**Phase 1 (no API key required):** In `pulse_aggregator.py`, adjust scoring to account for Reddit's missing engagement:

```python
# In cluster_topics(), replace simple engagement sum:
# BEFORE:
engagement += meta.get("score", 0) + meta.get("upvotes", 0)

# AFTER: Normalize engagement per source
source_type = item.get("source_type", "unknown")
raw_engagement = meta.get("score", 0) + meta.get("upvotes", 0)

if source_type == "reddit" and raw_engagement == 0:
    # Reddit RSS doesn't provide engagement; use mention count as proxy
    engagement += 5  # Default engagement weight for Reddit mentions
else:
    engagement += raw_engagement
```

**Phase 2 (with Reddit API credentials):** Add optional PRAW integration:

```python
# In reddit_source.py, add authenticated fallback:
def _fetch_subreddit_api(self, subreddit_name: str) -> List[Dict[str, Any]]:
    """Fetch via Reddit API (requires REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET)."""
    import praw
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=self.user_agent,
    )
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for post in subreddit.hot(limit=self.max_posts):
        posts.append({
            "subreddit": subreddit_name,
            "title": post.title,
            "url": f"https://reddit.com{post.permalink}",
            "author": str(post.author),
            "content": post.selftext[:2000],
            "timestamp": int(post.created_utc),
            "upvotes": post.score,
            "num_comments": post.num_comments,
        })
    return posts
```

Try API first, fall back to RSS if credentials not configured.

---

## Upgrade 6: Add Concurrent Source Fetching

**Priority**: MEDIUM
**Effort**: S
**Regression Risk**: Low

### What to Change

**In `__init__.py`, update `SourceManager.fetch_all()`:**

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def fetch_all(self, limit_per_source=None, since=None) -> Dict[SourceType, FetchResult]:
    results = {}

    def _fetch_one(source_type, source):
        try:
            return source_type, source.fetch(limit=limit_per_source, since=since)
        except Exception as e:
            return source_type, FetchResult(items=[], success=False, error_message=str(e))

    with ThreadPoolExecutor(max_workers=len(self._sources)) as executor:
        futures = {
            executor.submit(_fetch_one, st, s): st
            for st, s in self._sources.items()
        }
        for future in as_completed(futures):
            source_type, result = future.result()
            results[source_type] = result

    return results
```

This reduces total fetch time from sum(latencies) to max(latencies).

---

## Upgrade 7: Add Basic Semantic Deduplication

**Priority**: MEDIUM
**Effort**: M
**Regression Risk**: Low

### What to Change

Add title-based deduplication in pulse_aggregator before topic clustering:

```python
# In pulse_aggregator.py, after fetch_recent_items():

def deduplicate_items(items: List[Dict[str, Any]], similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
    """Group items with similar titles, merge source types."""
    from collections import defaultdict

    def tokenize(text: str) -> set:
        return set(re.findall(r"[a-z][a-z0-9]+", text.lower())) - STOPWORDS

    groups = []
    used = set()

    for i, item in enumerate(items):
        if i in used:
            continue

        group = [item]
        tokens_i = tokenize(item.get("title", ""))

        for j, other in enumerate(items[i+1:], start=i+1):
            if j in used:
                continue
            tokens_j = tokenize(other.get("title", ""))

            # Jaccard similarity
            if tokens_i and tokens_j:
                similarity = len(tokens_i & tokens_j) / len(tokens_i | tokens_j)
                if similarity >= similarity_threshold:
                    group.append(other)
                    used.add(j)

        # Merge group: keep highest-engagement item, merge source types
        best = max(group, key=lambda x: x.get("metadata", {}).get("score", 0))
        best["_cross_sources"] = list(set(g.get("source_type", "unknown") for g in group))
        best["_duplicate_count"] = len(group)
        groups.append(best)

    return groups
```

Apply before `cluster_topics()` to prevent inflation of cross-source scores.

---

## Upgrade 8: Move Trust Tiers to Configuration

**Priority**: LOW
**Effort**: S
**Regression Risk**: None

### What to Change

Move `SUBREDDIT_TIERS` from Python dict to config:

```python
# In execution/config.py, add:
class SourceConfig:
    """Source ingestion configuration."""
    SUBREDDIT_TIERS: Dict[str, str] = {
        "LocalLLaMA": "b", "LLMDevs": "b", "LanguageTechnology": "b",
        "MachineLearning": "c", "deeplearning": "c", "mlops": "b",
    }
    RSS_FEEDS: List[Dict[str, str]] = [
        {"name": "Lobsters", "url": "https://lobste.rs/rss"},
        {"name": "Dev.to", "url": "https://dev.to/feed"},
        {"name": "Hacker Noon", "url": "https://hackernoon.com/feed"},
    ]
    DEFAULT_FETCH_HOURS: int = 72
    CONCURRENT_FETCH: bool = True
```

Update `reddit_source.py` to read from `config.sources.SUBREDDIT_TIERS` instead of module-level dict.

---

## Implementation Order

```
1. [CRITICAL] Upgrade 2: Fix Reddit dual-DB default (pulse data completeness)
2. [HIGH]     Upgrade 1: Source health check + registration warnings (visibility)
3. [HIGH]     Upgrade 3: Add retry with exponential backoff (reliability)
4. [HIGH]     Upgrade 4: Add circuit breaker (resilience)
5. [MEDIUM]   Upgrade 5: Fix Reddit engagement (Phase 1 only — scoring normalization)
6. [MEDIUM]   Upgrade 6: Concurrent source fetching (performance)
7. [MEDIUM]   Upgrade 7: Semantic deduplication (data quality)
8. [LOW]      Upgrade 8: Move trust tiers to config (maintainability)
```

## Estimated Total Effort: 3-4 days for a single engineer

## Files Modified

| File | Changes |
|------|---------|
| `execution/sources/__init__.py` | Upgrades 1, 4, 6: Import warnings, circuit breaker, concurrent fetch |
| `execution/sources/reddit_source.py` | Upgrades 2, 3, 5: Unified DB default, retry, optional OAuth2 |
| `execution/sources/hackernews_source.py` | Upgrades 2, 3: DB insert in CLI, retry |
| `execution/sources/rss_source.py` | Upgrades 2, 3: DB insert in CLI, retry |
| `execution/pulse_aggregator.py` | Upgrades 5, 7: Engagement normalization, semantic dedup |
| `execution/config.py` | Upgrade 8: SourceConfig class |
| `requirements.txt` | Upgrade 3: Add `tenacity`; Upgrade 5 Phase 2: Add `praw` (optional) |

## Cross-Feature Dependencies

| Upgrade | Depends On |
|---------|------------|
| Upgrade 4 (circuit breaker) | Works independently, but should integrate with pipeline error handling (Feature 3 Upgrade 6) |
| Upgrade 5 Phase 2 (Reddit API) | Requires Reddit OAuth2 credentials in `.env` |
| Upgrade 7 (semantic dedup) | Independent, but benefits from engagement data fix (Upgrade 5) for scoring |
