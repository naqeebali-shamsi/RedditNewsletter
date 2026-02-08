# Feature: Source Ingestion — Current State

**Agent**: Feature Deconstruction Agent
**Date**: 2026-02-08
**Files Analyzed**:
- `execution/sources/__init__.py` (224 lines) — Factory, registry, SourceManager
- `execution/sources/base_source.py` (262 lines) — ABC, ContentItem, FetchResult, TrustTier
- `execution/sources/reddit_source.py` (440 lines) — Reddit RSS fetcher
- `execution/sources/hackernews_source.py` (287 lines) — HN Firebase API fetcher
- `execution/sources/rss_source.py` (376 lines) — Configurable RSS/Atom fetcher
- `execution/pulse_aggregator.py` (538 lines) — Cross-source topic clustering
- `app.py:1263-1271` — Dashboard topic fetching

---

## Architecture Summary

### Design Pattern: Strategy + Factory + Registry

```
ContentSource (ABC)       SourceFactory.create()
     │                          │
     ├── RedditSource           ├── @register_source(SourceType.REDDIT)
     ├── HackerNewsSource       ├── @register_source(SourceType.HACKERNEWS)
     ├── RSSSource              ├── @register_source(SourceType.RSS)
     └── GmailSource            └── @register_source(SourceType.GMAIL)
```

### Data Flow

```
Source.fetch() → List[ContentItem] → DB insert → pulse_aggregator reads DB → topic clustering → content angles
                                      ↑                                                              ↓
                              (dual tables!)                                              Dashboard reads pulse
```

### ContentItem (Normalized Output)

Defined in `base_source.py:46-134`. Fields: `source_type`, `source_id`, `title`, `content`, `url`, `author`, `timestamp`, `trust_tier`, `metadata`. Has `unique_key` property (`source_type:source_id`) and `to_dict()`/`from_dict()` for DB serialization.

### TrustTier Enum

```
A (Curated):    Auto-signal, skip evaluation (e.g., Simon Willison, Pragmatic Engineer)
B (Semi-trusted): Light evaluation (e.g., general Substack, LocalLLaMA)
C (Untrusted):  Full Signal/Noise evaluation (e.g., unknown senders, Reddit)
X (Blocked):    Never fetch (spam, off-topic)
```

---

## Critical Findings

### F1: Reddit RSS Returns Hardcoded Zero Engagement (HIGH)

**Location**: `reddit_source.py:191-192`

```python
post = {
    ...
    "upvotes": 0,
    "num_comments": 0,
}
```

Reddit RSS feeds don't include vote counts or comment counts. The source hardcodes `0` for both. This means:
- Pulse aggregator's engagement scoring sees ALL Reddit posts as 0 engagement
- No way to distinguish a 5000-upvote post from a 0-upvote post
- The cross-source engagement comparison (`meta.get("score", 0) + meta.get("upvotes", 0)`) is biased toward HN which provides real `score`

### F2: Dashboard Bypasses ContentSource Entirely (HIGH)

**Location**: `app.py:1263-1271`

```python
topic_agent = TopicResearchAgent()
topics = topic_agent.fetch_trending_topics()
selected = topic_agent.analyze_and_select(topics)
```

The dashboard uses `TopicResearchAgent` for content discovery, NOT the ContentSource framework. This is a parallel data path:
- ContentSource → SQLite → pulse_aggregator → topics (CLI/automation path)
- TopicResearchAgent → direct API calls → topic selection (dashboard path)

Two completely separate content discovery implementations that may return different results.

### F3: Dual Database Tables with No Consistency (HIGH)

**Location**: `reddit_source.py:279-372`

Reddit has TWO insert methods:
- `insert_to_legacy_db()` → `posts` table (line 279)
- `insert_to_unified_db()` → `content_items` table (line 326)

**Critical**: The CLI entry point (`main()` at line 379) writes to legacy table by DEFAULT. Unified table only with `--unified` flag (line 429). But `pulse_aggregator.py:71-79` reads from `content_items` only.

This means: **Reddit posts fetched via CLI don't appear in pulse aggregator** unless `--unified` is explicitly passed.

### F4: No Circuit Breaker or Minimum Source Threshold (HIGH)

**Location**: `__init__.py:142-172` (SourceManager.fetch_all)

```python
for source_type, source in self._sources.items():
    try:
        results[source_type] = source.fetch(...)
    except Exception as e:
        results[source_type] = FetchResult(items=[], success=False, error_message=str(e))
```

If ALL sources fail, `fetch_all()` returns a dict of empty results with `success=False`. There's no circuit breaker to:
- Detect "all sources failed" vs "some sources failed"
- Alert or escalate when source health degrades
- Prevent pipeline from proceeding with zero content

### F5: No Retry or Backoff on HTTP Requests (MEDIUM)

**Location**: `reddit_source.py:170`

```python
response = requests.get(rss_url, headers=headers, timeout=10)
```

Single `requests.get()` with 10-second timeout. No retry on transient failures (429, 503, network errors). HN source has the same pattern (line 94-98, 117-121), though it at least has a `fetch_delay` between sequential item fetches.

RSS source delegates to `feedparser.parse(feed_url)` which has its own timeout handling but no retry.

### F6: Source Registration Silently Swallows ImportError (__init__.py:189-207)

```python
try:
    from . import reddit_source
except ImportError:
    pass  # Silent

try:
    from . import gmail_source
except ImportError:
    pass  # Silent
```

If a source fails to import (missing dependency, syntax error), it silently disappears from the registry. `SourceFactory.get_registered_sources()` would return fewer sources than expected with no warning.

### F7: Hardcoded Trust Tiers in Python Dict (MEDIUM)

**Location**: `reddit_source.py:44-54`

```python
SUBREDDIT_TIERS = {
    "LocalLLaMA": TrustTier.B,
    "LLMDevs": TrustTier.B,
    ...
}
```

Trust tier assignment is hardcoded in source code. Adding a new subreddit requires editing Python, not configuration. HN and RSS sources hardcode `TrustTier.C` for all items (lines 193 and 268 respectively).

### F8: HN Content Capped at 500 Characters (MEDIUM)

**Location**: `hackernews_source.py:177`

```python
content = " ".join(content_parts)[:500]
```

HN self-posts (Ask HN, Show HN) can have significant text content that gets truncated to 500 characters. This caps the information available for topic clustering and keyword extraction.

### F9: No Semantic Deduplication (MEDIUM)

All sources rely on URL-based deduplication:
- Reddit: URL as `source_id`, `IntegrityError` on duplicate (line 318, 366)
- HN: `source_id = str(story_id)` (unique per item)
- RSS: `normalize_url()` + `seen_urls` set within fetch (line 214-216), plus `source_id` from entry ID

None check for semantic duplicates. The same story posted to Reddit, HN, and Lobsters appears as three separate items. This inflates topic scores in pulse_aggregator (one story counted as 3 cross-source mentions).

### F10: Pulse Aggregator Sentiment is Keyword-Based (LOW)

**Location**: `pulse_aggregator.py:265-305`

```python
positive_words = {"great", "awesome", "excellent", ...}
negative_words = {"bad", "terrible", "broken", ...}
```

Bag-of-words sentiment classification. "This great fix for a terrible bug" would be counted as neutral (1 positive word, 1 negative word). The code comment acknowledges this: "This is intentionally simple -- production would use a proper NLP model."

### F11: SourceManager.fetch_all Is Sequential (LOW)

**Location**: `__init__.py:159-170`

```python
for source_type, source in self._sources.items():
    results[source_type] = source.fetch(...)
```

Sources are fetched sequentially. With 3-4 sources, each with network I/O (timeouts up to 10-15 seconds), total fetch time could be 30-60 seconds. No concurrent fetching.

---

## Source Capability Matrix

| Capability | Reddit | HackerNews | RSS |
|-----------|--------|------------|-----|
| Engagement data | NO (hardcoded 0) | YES (score, descendants) | NO |
| URL normalization | NO | N/A | YES (UTM strip) |
| Within-fetch dedup | NO | N/A (unique IDs) | YES (seen_urls set) |
| Retry/backoff | NO | NO | NO |
| Rate limiting | NO | YES (fetch_delay) | NO |
| Content truncation | NO | YES (500 chars) | NO |
| Trust tier logic | Per-subreddit dict | Hardcoded C | Hardcoded C |
| DB write | Dual (legacy + unified) | ContentItem only | ContentItem only |
| Tags/categories | NO | NO | YES (RSS tags) |

## Entry Points

| Entry | Invokes | Writes To | Pulse Visible |
|-------|---------|-----------|---------------|
| CLI: `reddit_source.py` | RedditSource.fetch() | `posts` table (default) | NO (unless `--unified`) |
| CLI: `hackernews_source.py` | HackerNewsSource.fetch() | stdout only | NO |
| CLI: `rss_source.py` | RSSSource.fetch() | stdout only | NO |
| SourceManager.fetch_all() | All registered sources | Depends on caller | Depends on caller |
| Dashboard: `app.py` | TopicResearchAgent | Unknown | NO |
| `pulse_aggregator.py` | Reads `content_items` table | `pulse_daily` table | YES |
