# Pulse Monitoring System

## Purpose

Monitor the internet pulse across multiple technical communities to surface trending topics, emerging discussions, and content opportunities. The pulse system feeds the GhostWriter content pipeline with cross-source signals.

## Sources Monitored

| Source | Type | Default Interval | Notes |
|--------|------|-------------------|-------|
| Hacker News | `hackernews` | Every 30 min | Top 30 stories via Firebase API |
| RSS Feeds | `rss` | Every 60 min | Lobsters, Dev.to, Hacker Noon (configurable) |
| Reddit | `reddit` | Every 60 min | S+ and S tier subreddits |

All sources use the polymorphic `ContentSource` architecture in `execution/sources/`.

## How to Run

### Full Pulse Fetch

```bash
# Fetch from all pulse sources (Reddit + HackerNews + RSS)
python execution/fetch_all.py --pulse

# Fetch specific sources only
python execution/fetch_all.py --sources hackernews rss
```

### Daily Aggregation

```bash
# Default: last 24 hours, min 2 mentions
python execution/pulse_aggregator.py

# Custom window
python execution/pulse_aggregator.py --days 3 --min-mentions 3
```

### Standalone Source Fetch

```bash
# HackerNews only
python execution/sources/hackernews_source.py --max-stories 50 --hours 12

# RSS only
python execution/sources/rss_source.py --hours 48
```

## Interpreting pulse_daily Data

Each `pulse_daily` row contains JSON fields:

### top_topics
Array of topic objects, sorted by score descending:
```json
{
  "topic": "language model",
  "mentions": 8,
  "sources": ["hackernews", "reddit", "rss"],
  "cross_source_count": 3,
  "engagement": 1240,
  "score": 42.0,
  "sample_titles": ["..."]
}
```

**Score formula**: `(cross_source_count * 10) + (engagement * 0.1) + recency_bonus + mention_count`

### sentiment_summary
Basic positive/negative/neutral counts across all items:
```json
{"positive": 45, "negative": 12, "neutral": 83}
```

### content_angles
Suggested angles based on topic analysis:
- **HIGH PRIORITY**: Topics with 3+ cross-source mentions
- **MODERATE**: Topics with 2 cross-source mentions
- **VOLUME**: High single-source activity

### source_breakdown
Item counts per source type:
```json
{"reddit": 120, "hackernews": 30, "rss": 85}
```

## Configurable Source List

The `pulse_feeds` table stores feed configurations:

```sql
-- Add a new RSS feed
INSERT INTO pulse_feeds (name, source_type, url, fetch_interval_minutes)
VALUES ('TechCrunch', 'rss', 'https://techcrunch.com/feed/', 60);

-- Disable a feed
UPDATE pulse_feeds SET is_active = 0 WHERE name = 'Hacker Noon';

-- Change fetch interval
UPDATE pulse_feeds SET fetch_interval_minutes = 30 WHERE name = 'Lobsters';
```

## Escalation Rules

### Content Priority Signal
Topics with **3+ cross-source mentions** (e.g., appears on HN, Reddit, and an RSS feed simultaneously) get automatic content priority. These represent genuine industry trends rather than single-community hype.

### Pipeline Integration
1. `fetch_all.py --pulse` ingests from all sources
2. `pulse_aggregator.py` clusters and scores topics
3. Cross-source signals feed into `evaluate_posts` for content pipeline prioritization
4. High-priority topics become candidates for article/post generation

## Edge Cases

### Low Engagement Periods
- Weekends and holidays show reduced activity across all sources
- Lower the `--min-mentions` threshold during these periods (e.g., `--min-mentions 1`)
- Weekend content often has higher quality due to fewer but more dedicated contributors

### Hype Cycles
- Major product launches (GPT releases, Apple events) flood all sources simultaneously
- During hype events, raise `--min-mentions` to 5+ to find secondary signals beneath the noise
- Focus on counter-narrative or implementation-detail angles rather than the obvious headline

### Seasonal Trends
- Conference seasons (NeurIPS, ICML, KubeCon) produce topic clusters that last 1-2 weeks
- Year-end produces "best of" and prediction content -- high volume, lower signal
- Back-to-school and Q1 planning periods show increased interest in learning resources

### API/Feed Failures
- HackerNews Firebase API occasionally returns null items -- the source handles this gracefully
- RSS feeds may return stale data or fail to parse -- errors are logged per-feed, non-fatal
- If a source fails, the pulse still generates from remaining sources

## Technical Details

### Scripts
- `execution/sources/hackernews_source.py` -- HN Firebase API client
- `execution/sources/rss_source.py` -- Multi-feed RSS parser with URL normalization
- `execution/pulse_aggregator.py` -- Keyword clustering and daily aggregation
- `execution/fetch_all.py` -- Orchestrator with `--pulse` mode

### Database Tables
- `content_items` -- Unified content storage (all sources)
- `pulse_daily` -- Daily trend summaries
- `pulse_feeds` -- Feed configuration

### URL Normalization
RSS source strips UTM and tracking parameters for deduplication:
- `utm_source`, `utm_medium`, `utm_campaign`, `utm_term`, `utm_content`
- `ref`, `source`, `fbclid`, `gclid`, `mc_cid`, `mc_eid`

### Rate Limiting
- HackerNews: 0.1s delay between item fetches (configurable via `fetch_delay`)
- RSS: Standard HTTP with 15s timeout per feed
- Reddit: Uses RSS feeds (not API), standard rate limits apply
