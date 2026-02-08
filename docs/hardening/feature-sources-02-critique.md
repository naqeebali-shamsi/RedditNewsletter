# Feature: Source Ingestion — Research & Adversarial Critique

**Agents**: Technique Research + Red Team
**Date**: 2026-02-08

---

## Research: Industry Standard Patterns for Source Ingestion

### R1: Circuit Breaker Pattern (Michael Nygard, "Release It!")

**The problem**: If all sources fail, the pipeline gets an empty content list and either does nothing or produces low-quality output from cached data without alerting anyone.

**Industry solutions:**
- **Circuit Breaker**: After N consecutive failures, "open" the circuit (stop trying). After a cooldown period, "half-open" (try one request). On success, close the circuit.
- **Bulkhead**: Isolate source failures so one failing source doesn't cascade (e.g., slow timeout exhausts thread pool).
- **Python implementation**: `pybreaker` library, or custom with `dataclass` tracking failure counts per source.

**Pattern**: Each source gets an independent circuit breaker. When fetching, check circuit state first. On failure, increment counter. When open, skip source and return cached results. SourceManager exposes aggregate health: "2/4 sources healthy."

### R2: Exponential Backoff with Jitter (AWS Architecture Blog)

**The problem**: Reddit rate-limits at 10 req/min for unauthenticated RSS. A single 429 response wastes the entire subreddit fetch.

**Industry solutions:**
- **`tenacity` library**: Declarative retry with exponential backoff, jitter, and max attempts
- **`urllib3.Retry`**: Built into requests via HTTPAdapter
- **Redis-based rate limiter**: Token bucket per source for multi-process setups

**Pattern**:
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _fetch_subreddit_rss(self, subreddit_name):
    ...
```

### R3: Concurrent Source Fetching (asyncio / concurrent.futures)

**The problem**: Sequential fetching of 4 sources with 10-15s timeouts = 40-60s total latency.

**Industry solutions:**
- **`asyncio.gather()`**: Fetch all sources concurrently with `aiohttp`
- **`concurrent.futures.ThreadPoolExecutor`**: Thread-based parallelism for sync HTTP calls
- **`concurrent.futures.as_completed()`**: Process results as they arrive

**Pattern**: Wrap `SourceManager.fetch_all()` in `ThreadPoolExecutor` to fetch 4 sources in parallel. Total latency = max(individual latencies) instead of sum.

### R4: Source Health Monitoring (Prometheus/Datadog Pattern)

**The problem**: Zero visibility into source health. A source can fail for days before anyone notices.

**Industry solutions:**
- **Structured health metrics**: Success rate, latency percentile, items fetched per run
- **Prometheus counters**: `source_fetch_total{source="reddit",status="success|error"}`
- **Alerting rules**: "Alert if source success rate drops below 80% over 1 hour"

**Pattern**: Track per-source metrics in SQLite or JSON file. Expose `SourceManager.health_report()` that returns success rates, last successful fetch time, average latency.

### R5: Reddit API Authentication (Reddit OAuth2)

**The problem**: RSS-only fetching provides zero engagement data (upvotes, comments). This makes Reddit posts invisible to engagement-based scoring.

**Industry solutions:**
- **Reddit OAuth2 (script type)**: Free, 60 req/min, returns full JSON with scores
- **PRAW library**: Python wrapper for Reddit API with built-in rate limiting
- **Pushshift API**: Historical Reddit data (but may have restrictions)

**Pattern**: Use `praw` with OAuth2 script credentials. Falls back to RSS if credentials not configured. When authenticated: get actual upvotes, comments, awards, crosspost count.

### R6: Semantic Deduplication (Embedding-Based)

**The problem**: Same story on Reddit + HN + Lobsters = 3 items = artificially inflated cross-source score.

**Industry solutions:**
- **Title embedding similarity**: Compute embeddings (sentence-transformers), cluster by cosine similarity
- **MinHash/LSH**: Locality-sensitive hashing for near-duplicate detection at scale
- **Simple overlap**: Jaccard similarity on title tokens (no ML required)

**Pattern**: After fetching all sources, compute pairwise title similarity. Group items with >0.8 similarity as "same story." Pulse aggregator counts the group as 1 item with N source types.

---

## Red Team: Trust-Breaking Attack Scenarios

### Attack 1: "The Silent Source Death" (SEVERITY: HIGH)

**Vector**: Source registration swallows ImportError + no health monitoring
**Steps**:
1. `feedparser` package is accidentally removed during a dependency update
2. `__init__.py:204` catches `ImportError` silently — RSS source disappears from registry
3. `SourceManager.fetch_all()` only fetches Reddit + HN (2 of 3 sources)
4. Pulse aggregator sees fewer cross-source signals → topics score lower
5. Content angles shift toward Reddit/HN-only topics, missing Lobsters/Dev.to coverage
6. Nobody notices for days/weeks because there's no health check or alerting

**Likelihood**: MEDIUM (dependency changes are common)
**Impact**: Degraded content discovery. Missing entire source verticals silently.

### Attack 2: "The Engagement Blindness" (SEVERITY: HIGH)

**Vector**: Reddit RSS returns 0 engagement → pulse scoring biased
**Steps**:
1. A Reddit post with 5000 upvotes and 400 comments appears in r/MachineLearning
2. RSS feed reports it with `upvotes: 0, num_comments: 0`
3. The same topic appears on HN with `score: 150`
4. Pulse aggregator scores: HN engagement = 150, Reddit engagement = 0
5. Engagement-based scoring systematically underweights Reddit
6. Viral Reddit discussions are treated the same as 0-upvote noise posts
7. Topic selection is biased toward HN's scoring, not real community interest

**Likelihood**: HIGH (this is the current state, not a hypothetical)
**Impact**: Systematic bias in content discovery. Most popular Reddit topics ranked same as spam.

### Attack 3: "The Phantom Cross-Source Signal" (SEVERITY: MEDIUM)

**Vector**: No semantic dedup → same story inflates cross-source score
**Steps**:
1. OpenAI releases GPT-5. Story posted to Reddit, HN, Lobsters, Dev.to, and Hacker Noon RSS
2. Pulse aggregator sees 5 separate items with keyword "gpt" from 5 different source types
3. Cross-source score: `5 sources × 10 = 50 points` (maximum possible)
4. Meanwhile, a genuinely cross-cutting trend (e.g., "SQLite replacing Postgres for edge apps") appears on Reddit + HN only
5. Cross-source score: `2 sources × 10 = 20 points`
6. GPT-5 announcement ranks #1 purely due to duplication, not because it's a better content angle
7. GhostWriter writes about the most obvious topic (GPT-5) instead of the more interesting niche trend

**Likelihood**: HIGH (major announcements always appear on all sources)
**Impact**: Content angles biased toward viral announcements, missing nuanced cross-source trends.

### Attack 4: "The Stale Pulse" (SEVERITY: MEDIUM)

**Vector**: Reddit CLI doesn't write to unified DB by default
**Steps**:
1. Operator runs `python reddit_source.py` to fetch latest Reddit posts
2. Posts go to `posts` table (legacy). `content_items` table is NOT updated (no `--unified` flag)
3. Operator runs `python pulse_aggregator.py` to generate daily pulse
4. Pulse aggregator reads from `content_items` table — sees zero Reddit posts
5. Daily pulse is generated with only HN and RSS data (if those were fetched separately)
6. Content angles are based on incomplete data
7. Operator doesn't realize Reddit data is missing because pulse still runs successfully

**Likelihood**: HIGH (the default behavior is wrong; requires opt-in flag for correct behavior)
**Impact**: Pulse analysis based on incomplete data. Systematic Reddit blindspot.

### Attack 5: "The Rate Limit Cascade" (SEVERITY: MEDIUM)

**Vector**: No retry/backoff + no circuit breaker
**Steps**:
1. Automated script fetches Reddit every 30 minutes via cron
2. Reddit returns 429 (rate limit) for r/MachineLearning
3. Single `requests.get()` fails → exception → error logged → subreddit skipped
4. Next subreddit (r/deeplearning) also hits 429 (same rate limit window)
5. All subreddits fail. FetchResult has `success=False` with concatenated error messages
6. No retry attempted. No backoff. Full fetch window wasted.
7. Next cron run (30 min later) may also hit limits if the window hasn't reset

**Likelihood**: MEDIUM (depends on fetch frequency and Reddit's tolerance)
**Impact**: Intermittent data loss. Unreliable content pipeline.

### Attack 6: "The 500-Character HN Blackhole" (SEVERITY: LOW)

**Vector**: HN content truncation at 500 chars
**Steps**:
1. A detailed "Ask HN" post (2000+ chars) discusses a nuanced technical topic
2. `hackernews_source.py:177` truncates to 500 chars
3. Key technical keywords are in the truncated portion
4. Pulse aggregator extracts keywords from 500-char snippet only
5. Topic clustering misses the real subject of the discussion
6. Post gets clustered with wrong topic or not clustered at all

**Likelihood**: LOW (most HN stories are links, not self-posts)
**Impact**: Occasional missed topics from HN self-posts.

---

## Summary Verdict

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Architecture | B- | Good ABC/Factory pattern; clean ContentItem normalization |
| Reliability | D | No retry, no circuit breaker, no health monitoring |
| Data quality | D | Reddit engagement blind, no semantic dedup, HN truncation |
| Consistency | F | Dual DB tables, dashboard bypasses framework entirely |
| Observability | F | Silent ImportError swallowing, no source health metrics |
| Configuration | C | Hardcoded trust tiers, but config schema exists |
| Scalability | C | Sequential fetching, but adequate for current source count |
