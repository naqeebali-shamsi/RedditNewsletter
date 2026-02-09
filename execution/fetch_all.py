#!/usr/bin/env python3
"""
Multi-Source Content Orchestrator

Fetches content from all configured sources (Reddit, Gmail, etc.)
and stores in the unified content_items table.

Usage:
    python fetch_all.py                    # Fetch from all enabled sources
    python fetch_all.py --sources reddit   # Fetch from specific sources
    python fetch_all.py --list             # List available sources
"""

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    wait_random,
    retry_if_exception_type,
)

from execution.utils.logging import get_logger

logger = get_logger("fetch")

# Import the source system
from sources import (
    ContentItem,
    FetchResult,
    SourceFactory,
    SourceManager,
    SourceType,
    TrustTier,
)

# Import specific sources to register them
from sources.reddit_source import RedditSource

# Try to import Gmail source (may not exist yet)
try:
    from sources.gmail_source import GmailSource
    GMAIL_AVAILABLE = True
except ImportError:
    GMAIL_AVAILABLE = False

# Try to import HackerNews source
try:
    from sources.hackernews_source import HackerNewsSource
    HACKERNEWS_AVAILABLE = True
except ImportError:
    HACKERNEWS_AVAILABLE = False

# Try to import RSS source
try:
    from sources.rss_source import RSSSource
    RSS_AVAILABLE = True
except ImportError:
    RSS_AVAILABLE = False

# Database path
DB_PATH = Path(__file__).parent.parent / "reddit_content.db"


def get_db_connection():
    """Get database connection."""
    return sqlite3.connect(DB_PATH)


def ensure_tables_exist():
    """Ensure the unified content tables exist."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Create content_items table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS content_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_type TEXT NOT NULL,
            source_id TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT,
            author TEXT,
            url TEXT,
            timestamp INTEGER,
            trust_tier TEXT DEFAULT 'c',
            metadata TEXT,
            retrieved_at INTEGER NOT NULL,
            UNIQUE(source_type, source_id)
        )
    """)

    # Create indexes
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_content_source ON content_items(source_type)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_content_timestamp ON content_items(timestamp)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_content_trust ON content_items(trust_tier)"
    )

    conn.commit()
    conn.close()


def insert_items(items: List[ContentItem]) -> Dict[str, int]:
    """
    Insert content items to database.

    Returns:
        Dict with counts: {"inserted": N, "skipped": N}
    """
    if not items:
        return {"inserted": 0, "skipped": 0}

    conn = get_db_connection()
    cursor = conn.cursor()

    inserted = 0
    skipped = 0

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
            skipped += 1

    conn.commit()
    conn.close()

    return {"inserted": inserted, "skipped": skipped}


def log_audit_event(event_type: str, source_type: Optional[str], details: Dict[str, Any]):
    """Log an audit event to SQLite.

    DEPRECATED: SQLite audit logging is retained for backwards compatibility.
    Prefer structlog via ``get_logger("fetch")`` for new code.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            INSERT INTO audit_log (event_type, source_type, details, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (event_type, source_type, json.dumps(details), int(time.time())),
        )
        conn.commit()
    except sqlite3.OperationalError:
        # audit_log table may not exist yet
        pass
    finally:
        conn.close()


def get_source_config(source_type: SourceType) -> Optional[Dict[str, Any]]:
    """Get stored configuration for a source."""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "SELECT config FROM source_configs WHERE source_type = ? AND is_enabled = 1",
            (source_type.value,),
        )
        row = cursor.fetchone()
        if row and row[0]:
            return json.loads(row[0])
    except sqlite3.OperationalError:
        # source_configs table may not exist yet
        pass
    finally:
        conn.close()

    return None


def get_enabled_sources() -> List[SourceType]:
    """Get list of enabled source types from database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    enabled = []
    try:
        cursor.execute(
            "SELECT source_type FROM source_configs WHERE is_enabled = 1"
        )
        for row in cursor.fetchall():
            try:
                enabled.append(SourceType(row[0]))
            except ValueError:
                pass
    except sqlite3.OperationalError:
        # Table doesn't exist yet, return defaults
        pass
    finally:
        conn.close()

    # Always include Reddit as default if nothing configured
    if not enabled:
        enabled = [SourceType.REDDIT]

    return enabled


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=30) + wait_random(0, 2),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    reraise=True,
)
def fetch_from_source(source_type: SourceType, config: Optional[Dict[str, Any]] = None) -> FetchResult:
    """
    Fetch content from a single source.

    Retries up to 3 times on transient connection errors.

    Args:
        source_type: Type of source to fetch from
        config: Optional config override (uses stored config if not provided)

    Returns:
        FetchResult with items
    """
    # Get config from database if not provided
    if config is None:
        config = get_source_config(source_type) or {}

    # Create source and fetch
    source = SourceFactory.create(source_type, config)
    return source.fetch()


def fetch_all_sources(
    sources: Optional[List[SourceType]] = None,
    limit_per_source: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Fetch content from all configured sources.

    Args:
        sources: Specific sources to fetch (None = all enabled)
        limit_per_source: Max items per source

    Returns:
        Summary dict with results per source
    """
    ensure_tables_exist()

    # Determine which sources to fetch
    if sources is None:
        sources = get_enabled_sources()

    # Filter to only registered sources
    available = SourceFactory.get_registered_sources()
    sources = [s for s in sources if s in available]

    results = {
        "sources": {},
        "total_fetched": 0,
        "total_inserted": 0,
        "total_skipped": 0,
        "errors": [],
    }

    for source_type in sources:
        print(f"\n[{source_type.value.upper()}]")

        try:
            # Fetch from source
            fetch_result = fetch_from_source(source_type)

            if not fetch_result.success:
                results["errors"].append(f"{source_type.value}: {fetch_result.error_message}")
                print(f"  [!] Error: {fetch_result.error_message}")
                continue

            print(f"  [+] Fetched {fetch_result.items_fetched} items")

            # Insert to database
            db_result = insert_items(fetch_result.items)
            print(f"  [+] Inserted {db_result['inserted']} new items")
            print(f"  [ ] Skipped {db_result['skipped']} duplicates")

            # Structured log (primary)
            logger.info(
                "fetch_complete",
                source_type=source_type.value,
                items_fetched=fetch_result.items_fetched,
                items_inserted=db_result["inserted"],
                items_skipped=db_result["skipped"],
            )

            # SQLite audit log (deprecated, kept for backwards compatibility)
            log_audit_event(
                "fetch",
                source_type.value,
                {
                    "items_fetched": fetch_result.items_fetched,
                    "items_inserted": db_result["inserted"],
                    "items_skipped": db_result["skipped"],
                },
            )

            # Update totals
            results["sources"][source_type.value] = {
                "fetched": fetch_result.items_fetched,
                "inserted": db_result["inserted"],
                "skipped": db_result["skipped"],
            }
            results["total_fetched"] += fetch_result.items_fetched
            results["total_inserted"] += db_result["inserted"]
            results["total_skipped"] += db_result["skipped"]

        except Exception as e:
            error_msg = f"{source_type.value}: {str(e)}"
            results["errors"].append(error_msg)
            logger.error("fetch_failed", source_type=source_type.value, error=str(e))
            print(f"  [!] Error: {e}")

    return results


def list_available_sources():
    """Print list of available and enabled sources."""
    print("\nAvailable Sources:")
    print("-" * 40)

    registered = SourceFactory.get_registered_sources()
    enabled = get_enabled_sources()

    for source_type in SourceType:
        status_parts = []

        if source_type in registered:
            status_parts.append("[+] registered")
        else:
            status_parts.append("[-] not implemented")

        if source_type in enabled:
            status_parts.append("enabled")
        else:
            status_parts.append("disabled")

        status = ", ".join(status_parts)
        print(f"  {source_type.value:10s} [{status}]")

    print()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch content from all configured sources"
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=[s.value for s in SourceType],
        help="Specific sources to fetch from",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available sources",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Max items per source",
    )
    parser.add_argument(
        "--pulse",
        action="store_true",
        help="Pulse mode: fetch from all sources for daily pulse capture",
    )

    args = parser.parse_args()

    if args.list:
        list_available_sources()
        return

    # Parse source types
    sources = None
    if args.sources:
        sources = [SourceType(s) for s in args.sources]

    # Pulse mode: add HackerNews and RSS to the fetch list
    if args.pulse:
        print("Pulse mode: fetching from all sources for daily pulse capture")
        if sources is None:
            sources = [SourceType.REDDIT]
        if HACKERNEWS_AVAILABLE and SourceType.HACKERNEWS not in sources:
            sources.append(SourceType.HACKERNEWS)
        if RSS_AVAILABLE and SourceType.RSS not in sources:
            sources.append(SourceType.RSS)

    print(f"\n{'='*60}")
    print("GhostWriter Content Fetch")
    print(f"{'='*60}")

    results = fetch_all_sources(sources, args.limit)

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Total fetched:  {results['total_fetched']}")
    print(f"  Total inserted: {results['total_inserted']}")
    print(f"  Total skipped:  {results['total_skipped']}")

    if results["errors"]:
        print(f"\nErrors ({len(results['errors'])}):")
        for error in results["errors"]:
            print(f"  [!] {error}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
