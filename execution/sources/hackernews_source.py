"""
HackerNews Content Source

Fetches top stories from Hacker News via the Firebase API.
Maps stories to ContentItems for the unified content pipeline.

API docs: https://github.com/HackerNews/API

Usage:
    from execution.sources import SourceFactory, SourceType

    config = {"max_stories": 30}
    source = SourceFactory.create(SourceType.HACKERNEWS, config)
    result = source.fetch()
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import requests
except ImportError:
    requests = None

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    _has_tenacity = True
except ImportError:
    _has_tenacity = False

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

# HackerNews Firebase API base URL
HN_API_BASE = "https://hacker-news.firebaseio.com/v0"


@register_source(SourceType.HACKERNEWS)
class HackerNewsSource(ContentSource):
    """
    Content source for Hacker News top stories.

    Uses the official HN Firebase API to fetch top stories,
    then retrieves individual item details.

    Config options:
        max_stories: Max stories to fetch (default: 30)
        fetch_delay: Seconds between item fetches (default: 0.1)
        timeout: Request timeout in seconds (default: 10)
    """

    source_type = SourceType.HACKERNEWS

    DEFAULT_MAX_STORIES = 30
    DEFAULT_FETCH_DELAY = 0.1
    DEFAULT_TIMEOUT = 10

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.max_stories = self.config.get("max_stories", self.DEFAULT_MAX_STORIES)
        self.fetch_delay = self.config.get("fetch_delay", self.DEFAULT_FETCH_DELAY)
        self.timeout = self.config.get("timeout", self.DEFAULT_TIMEOUT)

    def _validate_config(self) -> None:
        """Validate HackerNews source configuration."""
        if requests is None:
            raise ImportError("requests library is required for HackerNewsSource")

        max_stories = self.config.get("max_stories", self.DEFAULT_MAX_STORIES)
        if not isinstance(max_stories, int) or max_stories < 1:
            raise ValueError("max_stories must be a positive integer")

    def _fetch_top_story_ids(self) -> List[int]:
        """Fetch top story IDs from HN API. Retried if tenacity is available."""
        resp = requests.get(
            f"{HN_API_BASE}/topstories.json",
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def _fetch_item(self, story_id: int) -> Optional[Dict[str, Any]]:
        """Fetch a single HN item. Retried if tenacity is available."""
        item_resp = requests.get(
            f"{HN_API_BASE}/item/{story_id}.json",
            timeout=self.timeout,
        )
        item_resp.raise_for_status()
        return item_resp.json()

    def fetch(
        self,
        limit: Optional[int] = None,
        since: Optional[int] = None,
    ) -> FetchResult:
        """
        Fetch top stories from Hacker News.

        Args:
            limit: Override max_stories (None = use config default)
            since: Unix timestamp - only include stories newer than this

        Returns:
            FetchResult with ContentItems
        """
        max_items = limit or self.max_stories
        all_items: List[ContentItem] = []
        errors: List[str] = []

        try:
            # Step 1: Get top story IDs
            story_ids = self._fetch_top_story_ids()

            if not story_ids:
                return FetchResult(items=[], success=True, items_fetched=0)

            # Limit the number of IDs we fetch details for
            story_ids = story_ids[:max_items]

        except requests.RequestException as e:
            return FetchResult(
                items=[],
                success=False,
                error_message=f"Failed to fetch top stories list: {e}",
            )

        # Step 2: Fetch individual story details
        for story_id in story_ids:
            try:
                raw_item = self._fetch_item(story_id)

                if raw_item is None:
                    continue

                # Skip deleted or dead items
                if raw_item.get("deleted") or raw_item.get("dead"):
                    continue

                # Filter by timestamp if specified
                item_time = raw_item.get("time", 0)
                if since and item_time < since:
                    continue

                content_item = self.normalize(raw_item)
                all_items.append(content_item)

            except requests.RequestException as e:
                errors.append(f"item {story_id}: {e}")
            except (ValueError, KeyError) as e:
                errors.append(f"item {story_id} parse error: {e}")

            # Respectful delay between requests
            if self.fetch_delay > 0:
                time.sleep(self.fetch_delay)

        error_msg = None
        if errors:
            error_msg = f"{len(errors)} item fetch errors: {'; '.join(errors[:5])}"

        return FetchResult(
            items=all_items,
            success=len(all_items) > 0 or len(errors) == 0,
            error_message=error_msg,
            items_fetched=len(all_items),
        )

    def normalize(self, raw_item: Dict[str, Any]) -> ContentItem:
        """
        Convert raw HN item to ContentItem.

        Args:
            raw_item: Raw item dict from HN API

        Returns:
            Normalized ContentItem
        """
        story_id = raw_item.get("id", 0)
        title = raw_item.get("title", "")
        text = raw_item.get("text", "")

        # Build content: title + text, capped at 500 chars
        content_parts = [title]
        if text:
            content_parts.append(text)
        content = " ".join(content_parts)[:500]

        # Determine item type
        item_type = raw_item.get("type", "story")

        # HN URL for self-posts, external URL for link posts
        url = raw_item.get("url") or f"https://news.ycombinator.com/item?id={story_id}"

        return ContentItem(
            source_type=SourceType.HACKERNEWS,
            source_id=str(story_id),
            title=title,
            content=content,
            url=url,
            author=raw_item.get("by", ""),
            timestamp=raw_item.get("time"),
            trust_tier=TrustTier.C,
            metadata={
                "score": raw_item.get("score", 0),
                "descendants": raw_item.get("descendants", 0),
                "type": item_type,
            },
        )

    def get_trust_tier(self, raw_item: Any) -> TrustTier:
        """HN stories default to TrustTier.C (untrusted, full evaluation)."""
        return TrustTier.C

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        """Return JSON schema for HackerNews source configuration."""
        return {
            "type": "object",
            "properties": {
                "max_stories": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 500,
                    "default": 30,
                    "description": "Maximum number of top stories to fetch",
                },
                "fetch_delay": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 5,
                    "default": 0.1,
                    "description": "Delay in seconds between individual item fetches",
                },
                "timeout": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 60,
                    "default": 10,
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


# Apply retry decorator to network fetches if tenacity is available
if _has_tenacity:
    _retry_decorator = retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    )
    HackerNewsSource._fetch_top_story_ids = _retry_decorator(HackerNewsSource._fetch_top_story_ids)
    HackerNewsSource._fetch_item = _retry_decorator(HackerNewsSource._fetch_item)


# =========================================================================
# CLI Entry Point
# =========================================================================

def main():
    """CLI entry point for standalone HackerNews fetching."""
    import argparse

    parser = argparse.ArgumentParser(description="Fetch HackerNews top stories")
    parser.add_argument(
        "--max-stories", type=int, default=30,
        help="Max stories to fetch (default: 30)",
    )
    parser.add_argument(
        "--hours", type=int, default=24,
        help="Only include stories from last N hours (default: 24)",
    )
    parser.add_argument(
        "--no-db", action="store_true",
        help="Skip writing to unified content_items table",
    )

    args = parser.parse_args()

    config = {"max_stories": args.max_stories}
    since = int(time.time()) - (args.hours * 3600)

    source = HackerNewsSource(config)

    print(f"\n{'='*60}")
    print(f"Fetching top {args.max_stories} HackerNews stories...")
    print(f"{'='*60}\n")

    result = source.fetch(since=since)

    if result.error_message:
        print(f"  [!] Warnings: {result.error_message}")

    print(f"  [+] Fetched {result.items_fetched} stories")

    for item in result.items[:5]:
        score = item.metadata.get("score", 0)
        print(f"  [{score:>4}] {item.title[:70]}")

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
