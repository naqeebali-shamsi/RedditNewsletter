"""
RSS Content Source

Fetches articles from configurable RSS/Atom feeds (Lobsters, Dev.to, Hacker Noon, etc.)
and maps them to ContentItems for the unified content pipeline.

Usage:
    from execution.sources import SourceFactory, SourceType

    config = {
        "feeds": [
            {"name": "Lobsters", "url": "https://lobste.rs/rss"},
            {"name": "Dev.to", "url": "https://dev.to/feed"},
        ]
    }
    source = SourceFactory.create(SourceType.RSS, config)
    result = source.fetch()
"""

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlencode, parse_qs, urlunparse

try:
    import feedparser
except ImportError:
    feedparser = None

from .base_source import (
    ContentSource,
    ContentItem,
    FetchResult,
    SourceType,
    TrustTier,
)
from . import register_source

# Database path (shared with reddit_source)
DB_PATH = Path(__file__).parent.parent.parent / "reddit_content.db"

# Default feeds to monitor
DEFAULT_FEEDS = [
    {"name": "Lobsters", "url": "https://lobste.rs/rss"},
    {"name": "Dev.to", "url": "https://dev.to/feed"},
    {"name": "Hacker Noon", "url": "https://hackernoon.com/feed"},
]

# UTM and tracking parameters to strip for URL normalization
TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "ref", "source", "fbclid", "gclid", "mc_cid", "mc_eid",
}


def normalize_url(url: str) -> str:
    """
    Strip tracking/UTM parameters from a URL for deduplication.

    Args:
        url: Raw URL possibly containing tracking params

    Returns:
        Cleaned URL without tracking parameters
    """
    if not url:
        return url

    try:
        parsed = urlparse(url)
        query_params = parse_qs(parsed.query, keep_blank_values=False)

        # Remove tracking parameters
        cleaned_params = {
            k: v for k, v in query_params.items()
            if k.lower() not in TRACKING_PARAMS
        }

        # Rebuild URL without tracking params
        cleaned_query = urlencode(cleaned_params, doseq=True)
        cleaned = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path.rstrip("/"),
            parsed.params,
            cleaned_query,
            "",  # Drop fragment
        ))
        return cleaned
    except Exception:
        return url


@register_source(SourceType.RSS)
class RSSSource(ContentSource):
    """
    Content source for RSS/Atom feeds.

    Fetches from multiple configurable feeds and normalizes entries
    into ContentItems. URLs are normalized to strip tracking params
    for deduplication.

    Config options:
        feeds: List of {"name": str, "url": str} dicts (default: Lobsters, Dev.to, Hacker Noon)
        max_entries_per_feed: Max entries per feed (default: 50)
        timeout: Request timeout in seconds (default: 15)
    """

    source_type = SourceType.RSS

    DEFAULT_MAX_ENTRIES = 50
    DEFAULT_TIMEOUT = 15

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.feeds = self.config.get("feeds", DEFAULT_FEEDS)
        self.max_entries = self.config.get("max_entries_per_feed", self.DEFAULT_MAX_ENTRIES)
        self.timeout = self.config.get("timeout", self.DEFAULT_TIMEOUT)

    def _validate_config(self) -> None:
        """Validate RSS source configuration."""
        if feedparser is None:
            raise ImportError("feedparser library is required for RSSSource")

        feeds = self.config.get("feeds", DEFAULT_FEEDS)
        if not isinstance(feeds, list):
            raise ValueError("feeds must be a list of {name, url} dicts")

        for feed in feeds:
            if not isinstance(feed, dict) or "url" not in feed:
                raise ValueError(f"Each feed must have at least a 'url' key: {feed}")

    def fetch(
        self,
        limit: Optional[int] = None,
        since: Optional[int] = None,
    ) -> FetchResult:
        """
        Fetch entries from all configured RSS feeds.

        Args:
            limit: Max total entries across all feeds (None = use per-feed limit)
            since: Unix timestamp - only include entries newer than this

        Returns:
            FetchResult with ContentItems
        """
        all_items: List[ContentItem] = []
        errors: List[str] = []
        seen_urls: set = set()  # Track normalized URLs for dedup within this fetch

        for feed_config in self.feeds:
            feed_url = feed_config["url"]
            feed_name = feed_config.get("name", feed_url)

            try:
                items = self._fetch_feed(feed_config, since, seen_urls)
                all_items.extend(items)

                # Check global limit
                if limit and len(all_items) >= limit:
                    all_items = all_items[:limit]
                    break

            except Exception as e:
                errors.append(f"{feed_name}: {e}")

        error_msg = None
        if errors:
            error_msg = "; ".join(errors)

        return FetchResult(
            items=all_items,
            success=len(all_items) > 0 or len(errors) == 0,
            error_message=error_msg,
            items_fetched=len(all_items),
        )

    def _fetch_feed(
        self,
        feed_config: Dict[str, str],
        since: Optional[int],
        seen_urls: set,
    ) -> List[ContentItem]:
        """
        Parse a single RSS/Atom feed and return ContentItems.

        Args:
            feed_config: Feed configuration dict with 'url' and optional 'name'
            since: Unix timestamp filter
            seen_urls: Set of already-seen normalized URLs for dedup

        Returns:
            List of ContentItems from this feed
        """
        feed_url = feed_config["url"]
        feed_name = feed_config.get("name", feed_url)

        feed = feedparser.parse(feed_url)

        if feed.bozo and not feed.entries:
            raise ValueError(f"Feed parse error: {feed.bozo_exception}")

        items = []
        for entry in feed.entries[:self.max_entries]:
            # Parse timestamp
            published = entry.get("published_parsed") or entry.get("updated_parsed")
            timestamp = int(time.mktime(published)) if published else int(time.time())

            # Filter by time
            if since and timestamp < since:
                continue

            # URL deduplication
            raw_url = entry.get("link", "")
            normalized = normalize_url(raw_url)
            if normalized in seen_urls:
                continue
            seen_urls.add(normalized)

            # Generate stable source_id from entry ID or link
            source_id = entry.get("id") or entry.get("link", "")
            if not source_id:
                # Fallback: hash the title + feed URL
                source_id = hashlib.sha256(
                    f"{feed_url}:{entry.get('title', '')}".encode()
                ).hexdigest()[:16]

            # Extract tags
            tags = []
            for tag in entry.get("tags", []):
                term = tag.get("term", "")
                if term:
                    tags.append(term)

            raw_item = {
                "source_id": source_id,
                "title": entry.get("title", ""),
                "content": entry.get("summary", ""),
                "url": normalized,
                "author": entry.get("author", ""),
                "timestamp": timestamp,
                "feed_url": feed_url,
                "feed_name": feed_name,
                "tags": tags,
            }

            item = self.normalize(raw_item)
            items.append(item)

        return items

    def normalize(self, raw_item: Dict[str, Any]) -> ContentItem:
        """
        Convert raw RSS entry to ContentItem.

        Args:
            raw_item: Processed entry dict from _fetch_feed

        Returns:
            Normalized ContentItem
        """
        return ContentItem(
            source_type=SourceType.RSS,
            source_id=raw_item["source_id"],
            title=raw_item.get("title", ""),
            content=raw_item.get("content", ""),
            url=raw_item.get("url", ""),
            author=raw_item.get("author", ""),
            timestamp=raw_item.get("timestamp"),
            trust_tier=TrustTier.C,
            metadata={
                "feed_url": raw_item.get("feed_url", ""),
                "feed_name": raw_item.get("feed_name", ""),
                "tags": raw_item.get("tags", []),
            },
        )

    def get_trust_tier(self, raw_item: Any) -> TrustTier:
        """RSS feeds default to TrustTier.C (untrusted, full evaluation)."""
        return TrustTier.C

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        """Return JSON schema for RSS source configuration."""
        return {
            "type": "object",
            "properties": {
                "feeds": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Display name for the feed"},
                            "url": {"type": "string", "description": "RSS/Atom feed URL"},
                        },
                        "required": ["url"],
                    },
                    "description": "List of RSS feeds to monitor",
                    "default": DEFAULT_FEEDS,
                },
                "max_entries_per_feed": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 200,
                    "default": 50,
                    "description": "Maximum entries to fetch per feed",
                },
                "timeout": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 60,
                    "default": 15,
                    "description": "Request timeout in seconds",
                },
            },
            "required": [],
        }

    def insert_to_unified_db(self, items: List[ContentItem]) -> int:
        """
        Insert items to unified 'content_items' table.

        Args:
            items: List of ContentItems to insert

        Returns:
            Number of new items inserted
        """
        if not items:
            return 0

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        inserted = 0
        for item in items:
            try:
                cursor.execute(
                    """
                    INSERT INTO content_items (source_type, source_id, title, content,
                                              author, url, timestamp, trust_tier,
                                              metadata, retrieved_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.source_type.value,
                        item.source_id,
                        item.title,
                        item.content,
                        item.author,
                        item.url,
                        item.timestamp,
                        item.trust_tier.value,
                        json.dumps(item.metadata) if item.metadata else None,
                        item.retrieved_at,
                    ),
                )
                inserted += 1
            except sqlite3.IntegrityError:
                # Duplicate source_type + source_id, skip
                pass

        conn.commit()
        conn.close()
        return inserted


# =========================================================================
# CLI Entry Point
# =========================================================================

def main():
    """CLI entry point for standalone RSS fetching."""
    import argparse

    parser = argparse.ArgumentParser(description="Fetch articles from RSS feeds")
    parser.add_argument(
        "--feeds", nargs="+",
        help="Custom feed URLs to fetch (overrides defaults)",
    )
    parser.add_argument(
        "--max-entries", type=int, default=50,
        help="Max entries per feed (default: 50)",
    )
    parser.add_argument(
        "--hours", type=int, default=72,
        help="Only include entries from last N hours (default: 72)",
    )
    parser.add_argument(
        "--no-db", action="store_true",
        help="Skip writing to unified content_items table",
    )

    args = parser.parse_args()

    config = {"max_entries_per_feed": args.max_entries}
    if args.feeds:
        config["feeds"] = [{"name": url, "url": url} for url in args.feeds]

    since = int(time.time()) - (args.hours * 3600)

    source = RSSSource(config)
    feed_count = len(source.feeds)

    print(f"\n{'='*60}")
    print(f"Fetching from {feed_count} RSS feeds...")
    print(f"{'='*60}\n")

    result = source.fetch(since=since)

    if result.error_message:
        print(f"  [!] Warnings: {result.error_message}")

    print(f"  [+] Fetched {result.items_fetched} entries")

    for item in result.items[:5]:
        feed_name = item.metadata.get("feed_name", "?")
        print(f"  [{feed_name:>12}] {item.title[:60]}")

    if result.items_fetched > 5:
        print(f"  ... and {result.items_fetched - 5} more")

    # Write to unified DB by default (pulse_aggregator reads from here)
    if not args.no_db:
        unified_inserted = source.insert_to_unified_db(result.items)
        print(f"  [+] Inserted {unified_inserted} to unified 'content_items' table")

    print(f"\n{'='*60}")
    print("Done")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
