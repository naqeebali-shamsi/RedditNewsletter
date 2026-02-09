"""
Reddit Content Source

Fetches posts from Reddit subreddits via RSS feeds.
Migrated from fetch_reddit.py to ContentSource architecture.

Usage:
    from execution.sources import SourceFactory, SourceType

    config = {
        "subreddits": ["LocalLLaMA", "LLMDevs"],
        "max_posts_per_subreddit": 100,
        "hours_lookback": 72
    }
    source = SourceFactory.create(SourceType.REDDIT, config)
    result = source.fetch()
"""

import feedparser
import logging
import os
import time
from typing import Any, Dict, List, Optional

try:
    import requests
except ImportError:
    requests = None

try:
    import praw
    _HAS_PRAW = True
except ImportError:
    _HAS_PRAW = False

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    _has_tenacity = True
except ImportError:
    _has_tenacity = False

logger = logging.getLogger(__name__)

from .base_source import (
    ContentSource,
    ContentItem,
    FetchResult,
    SourceType,
    TrustTier,
)
from . import register_source
from .database import insert_content_items, insert_legacy_posts

# Subreddit tier configuration
SUBREDDIT_TIERS = {
    # S+ tier: Highly focused, expert communities
    "LocalLLaMA": TrustTier.B,
    "LLMDevs": TrustTier.B,
    "LanguageTechnology": TrustTier.B,
    # S tier: Quality ML/AI communities
    "MachineLearning": TrustTier.C,
    "deeplearning": TrustTier.C,
    "mlops": TrustTier.B,
    "learnmachinelearning": TrustTier.C,
}

# Predefined subreddit lists
TIER_SP = ["LocalLLaMA", "LLMDevs", "LanguageTechnology"]
TIER_S = ["MachineLearning", "deeplearning", "mlops", "learnmachinelearning"]


@register_source(SourceType.REDDIT)
class RedditSource(ContentSource):
    """
    Content source for Reddit posts via RSS feeds.

    Config options:
        subreddits: List of subreddit names to fetch
        max_posts_per_subreddit: Max posts per subreddit (default: 100)
        hours_lookback: Only fetch posts from last N hours (default: 72)
        include_all_tiers: If True, include both S+ and S tier subreddits
        user_agent: Custom User-Agent string
    """

    source_type = SourceType.REDDIT

    # Default subreddits if none specified
    DEFAULT_SUBREDDITS = TIER_SP

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.subreddits = self.config.get("subreddits", self.DEFAULT_SUBREDDITS)
        self.max_posts = self.config.get("max_posts_per_subreddit", 100)
        self.hours_lookback = self.config.get("hours_lookback", 72)
        self.user_agent = self.config.get(
            "user_agent",
            "GhostWriter/1.0 (by /u/RedditNewsBot)"
        )

        # Include all tiers if requested
        if self.config.get("include_all_tiers"):
            self.subreddits = TIER_SP + TIER_S

    def _validate_config(self) -> None:
        """Validate Reddit source configuration."""
        if requests is None:
            raise ImportError("requests library is required for RedditSource")

        subreddits = self.config.get("subreddits", self.DEFAULT_SUBREDDITS)
        if not isinstance(subreddits, list):
            raise ValueError("subreddits must be a list")

        max_posts = self.config.get("max_posts_per_subreddit", 100)
        if not isinstance(max_posts, int) or max_posts < 1:
            raise ValueError("max_posts_per_subreddit must be a positive integer")

    def fetch(
        self,
        limit: Optional[int] = None,
        since: Optional[int] = None
    ) -> FetchResult:
        """
        Fetch posts from configured subreddits.

        Args:
            limit: Max total posts across all subreddits (None = use per-subreddit limit)
            since: Unix timestamp - only fetch posts newer than this

        Returns:
            FetchResult with ContentItems
        """
        all_items: List[ContentItem] = []
        errors: List[str] = []

        # Calculate time cutoff
        if since is None:
            since = int(time.time()) - (self.hours_lookback * 3600)

        # Try PRAW first when available and configured, fall back to RSS
        used_praw = False
        if self._praw_available():
            logger.info("PRAW available — fetching via Reddit API")
            per_sub_limit = limit or self.max_posts
            raw_posts = self._fetch_via_praw(self.subreddits, per_sub_limit)
            if raw_posts:
                used_praw = True
                for raw_post in raw_posts:
                    if raw_post.get("timestamp", 0) < since:
                        continue
                    item = self.normalize(raw_post)
                    all_items.append(item)
                    if limit and len(all_items) >= limit:
                        break

        if not used_praw:
            logger.info("Fetching via RSS feeds (PRAW not available or returned no data)")
            for subreddit in self.subreddits:
                try:
                    raw_posts = self._fetch_subreddit_rss(subreddit)
                    for raw_post in raw_posts:
                        # Filter by time
                        if raw_post.get("timestamp", 0) < since:
                            continue

                        item = self.normalize(raw_post)
                        all_items.append(item)

                        # Check global limit
                        if limit and len(all_items) >= limit:
                            break

                except Exception as e:
                    errors.append(f"{subreddit}: {str(e)}")

                # Check global limit
                if limit and len(all_items) >= limit:
                    break

        return FetchResult(
            items=all_items,
            success=len(errors) == 0,
            error_message="; ".join(errors) if errors else None,
            items_fetched=len(all_items),
        )

    def _praw_available(self) -> bool:
        """Check if PRAW is installed and Reddit API credentials are configured."""
        if not _HAS_PRAW:
            return False
        client_id = os.environ.get("REDDIT_CLIENT_ID")
        client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
        return bool(client_id and client_secret)

    def _fetch_via_praw(
        self, subreddits: List[str], limit: int
    ) -> List[Dict[str, Any]]:
        """
        Fetch posts from subreddits using the PRAW Reddit API client.

        Requires REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and optionally
        REDDIT_USER_AGENT environment variables.

        Args:
            subreddits: List of subreddit names to fetch from.
            limit: Maximum posts to fetch per subreddit.

        Returns:
            List of raw post dictionaries in the same format as RSS fetch.
        """
        client_id = os.environ.get("REDDIT_CLIENT_ID")
        client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
        user_agent = os.environ.get("REDDIT_USER_AGENT", self.user_agent)

        if not client_id or not client_secret:
            logger.warning("PRAW credentials missing; returning empty list")
            return []

        try:
            reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent,
            )
        except Exception as e:
            logger.error("Failed to create PRAW Reddit instance: %s", e)
            return []

        posts: List[Dict[str, Any]] = []
        for sub_name in subreddits:
            try:
                subreddit = reddit.subreddit(sub_name)
                for submission in subreddit.hot(limit=limit):
                    post = {
                        "subreddit": sub_name,
                        "title": submission.title or "",
                        "url": f"https://www.reddit.com{submission.permalink}",
                        "author": str(submission.author) if submission.author else "unknown",
                        "content": submission.selftext or "",
                        "timestamp": int(submission.created_utc),
                        "upvotes": submission.score,
                        "num_comments": submission.num_comments,
                    }
                    posts.append(post)
            except Exception as e:
                logger.error("PRAW error fetching r/%s: %s", sub_name, e)

        return posts

    def _fetch_subreddit_rss(self, subreddit_name: str) -> List[Dict[str, Any]]:
        """
        Fetch raw posts from a subreddit's RSS feed.
        Retries up to 3 times on network errors if tenacity is available.

        Args:
            subreddit_name: Name of subreddit (without r/)

        Returns:
            List of raw post dictionaries
        """
        headers = {"User-Agent": self.user_agent}
        rss_url = f"https://www.reddit.com/r/{subreddit_name}.rss"

        response = requests.get(rss_url, headers=headers, timeout=10)
        response.raise_for_status()

        feed = feedparser.parse(response.content)

        if feed.bozo:
            raise ValueError(f"Error parsing feed: {feed.bozo_exception}")

        posts = []
        for entry in feed.entries[: self.max_posts]:
            # Parse timestamp
            published_time = entry.get("published_parsed", time.gmtime())
            timestamp = int(time.mktime(published_time))

            post = {
                "subreddit": subreddit_name,
                "title": entry.get("title", ""),
                "url": entry.get("link", ""),
                "author": entry.get("author", "unknown"),
                "content": entry.get("summary", ""),
                "timestamp": timestamp,
                "upvotes": 0,
                "num_comments": 0,
            }
            posts.append(post)

        return posts

    def normalize(self, raw_item: Dict[str, Any]) -> ContentItem:
        """
        Convert raw Reddit post to ContentItem.

        Args:
            raw_item: Raw post dictionary from RSS

        Returns:
            Normalized ContentItem
        """
        subreddit = raw_item.get("subreddit", "unknown")

        return ContentItem(
            source_type=SourceType.REDDIT,
            source_id=raw_item.get("url", ""),  # URL is unique ID for Reddit
            title=raw_item.get("title", ""),
            content=raw_item.get("content", ""),
            url=raw_item.get("url", ""),
            author=raw_item.get("author", ""),
            timestamp=raw_item.get("timestamp"),
            trust_tier=self.get_trust_tier(raw_item),
            metadata={
                "subreddit": subreddit,
                "upvotes": raw_item.get("upvotes", 0),
                "num_comments": raw_item.get("num_comments", 0),
            },
        )

    def get_trust_tier(self, raw_item: Dict[str, Any]) -> TrustTier:
        """
        Determine trust tier based on subreddit.

        S+ tier subreddits get TrustTier.B (semi-trusted)
        S tier subreddits get TrustTier.C (untrusted, full evaluation)
        """
        subreddit = raw_item.get("subreddit", "")
        return SUBREDDIT_TIERS.get(subreddit, TrustTier.C)

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        """Return JSON schema for Reddit source configuration."""
        return {
            "type": "object",
            "properties": {
                "subreddits": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of subreddit names to fetch",
                    "default": TIER_SP,
                },
                "max_posts_per_subreddit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 100,
                    "description": "Maximum posts to fetch per subreddit",
                },
                "hours_lookback": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 168,  # 1 week
                    "default": 72,
                    "description": "Only fetch posts from last N hours",
                },
                "include_all_tiers": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include both S+ and S tier subreddits",
                },
                "user_agent": {
                    "type": "string",
                    "description": "Custom User-Agent string for Reddit API",
                },
            },
            "required": [],
        }

    # =========================================================================
    # Legacy Compatibility Methods
    # =========================================================================

    def insert_to_legacy_db(self, items: List[ContentItem]) -> int:
        """Insert items to legacy 'posts' table for backward compatibility."""
        return insert_legacy_posts(items)

    def insert_to_unified_db(self, items: List[ContentItem]) -> int:
        """Insert items to unified 'content_items' table."""
        return insert_content_items(items)


# Apply retry decorator to network fetch if tenacity is available
if _has_tenacity:
    RedditSource._fetch_subreddit_rss = retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    )(RedditSource._fetch_subreddit_rss)


# =========================================================================
# CLI Entry Point (for backward compatibility with fetch_reddit.py)
# =========================================================================

def main():
    """CLI entry point - mirrors original fetch_reddit.py interface."""
    import argparse

    parser = argparse.ArgumentParser(description="Fetch Reddit posts from RSS feeds")
    parser.add_argument("--subreddits", nargs="+", help="Specific subreddits to fetch")
    parser.add_argument(
        "--all", action="store_true", help="Fetch from all S+ and S tier subreddits"
    )
    parser.add_argument(
        "--max-posts", type=int, default=100, help="Max posts per subreddit (default: 100)"
    )
    parser.add_argument(
        "--hours", type=int, default=72, help="Only fetch posts from last N hours (default: 72)"
    )
    parser.add_argument(
        "--unified", action="store_true",
        help="(deprecated, now always enabled) Insert to unified content_items table"
    )

    args = parser.parse_args()

    # Build config
    config = {
        "max_posts_per_subreddit": args.max_posts,
        "hours_lookback": args.hours,
        "include_all_tiers": args.all,
    }

    if args.subreddits:
        config["subreddits"] = args.subreddits

    # Create source and fetch
    source = RedditSource(config)

    print(f"\n{'='*60}")
    print(f"Fetching from {len(source.subreddits)} subreddits...")
    print(f"{'='*60}\n")

    result = source.fetch()

    if result.error_message:
        print(f"⚠ Errors: {result.error_message}")

    print(f"\n✓ Fetched {result.items_fetched} posts")

    # Insert to legacy table
    legacy_inserted = source.insert_to_legacy_db(result.items)
    print(f"✓ Inserted {legacy_inserted} to legacy 'posts' table")

    # Always write to unified table (pulse_aggregator reads from here)
    unified_inserted = source.insert_to_unified_db(result.items)
    print(f"✓ Inserted {unified_inserted} to unified 'content_items' table")

    print(f"\n{'='*60}")
    print(f"✓ Done")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
