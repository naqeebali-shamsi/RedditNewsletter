"""
Shared Database Module (SQLAlchemy Core)

Centralizes all database operations for content sources.
Replaces raw sqlite3 code with SQLAlchemy Core for connection pooling,
type safety, and DRY table definitions.

Usage:
    from execution.sources.database import (
        get_engine, ensure_tables,
        insert_content_items, insert_legacy_posts,
        get_sender_trust_tier, upsert_newsletter_sender,
    )
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import sqlalchemy as sa
from sqlalchemy import (
    Column,
    Float,
    Integer,
    String,
    Text,
    Boolean,
    MetaData,
    Table,
    UniqueConstraint,
    Index,
    create_engine,
)

# Database path (shared across all sources)
DB_PATH = Path(__file__).parent.parent.parent / "reddit_content.db"

# Module-level singleton
_engine: Optional[sa.engine.Engine] = None

metadata = MetaData()

# ---------------------------------------------------------------------------
# Table Definitions
# ---------------------------------------------------------------------------

content_items = Table(
    "content_items",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("source_type", Text, nullable=False),
    Column("source_id", Text, nullable=False),
    Column("title", Text, nullable=False),
    Column("content", Text),
    Column("author", Text),
    Column("url", Text),
    Column("timestamp", Integer),
    Column("trust_tier", Text, server_default="c"),
    Column("metadata", Text),  # JSON blob
    Column("retrieved_at", Integer, nullable=False),
    UniqueConstraint("source_type", "source_id"),
)

Index("idx_content_source", content_items.c.source_type)
Index("idx_content_timestamp", content_items.c.timestamp)
Index("idx_content_trust", content_items.c.trust_tier)
Index("idx_content_retrieved", content_items.c.retrieved_at)

posts = Table(
    "posts",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("subreddit", Text, nullable=False),
    Column("title", Text, nullable=False),
    Column("url", Text, nullable=False, unique=True),
    Column("author", Text),
    Column("content", Text),
    Column("timestamp", Integer),
    Column("upvotes", Integer, server_default="0"),
    Column("num_comments", Integer, server_default="0"),
    Column("retrieved_at", Integer, nullable=False),
)

Index("idx_posts_subreddit", posts.c.subreddit)
Index("idx_posts_timestamp", posts.c.timestamp)

newsletter_senders = Table(
    "newsletter_senders",
    metadata,
    Column("email", Text, primary_key=True),
    Column("display_name", Text),
    Column("trust_tier", Text, server_default="b"),
    Column("is_active", Boolean, server_default="1"),
    Column("added_at", Integer, nullable=False),
    Column("notes", Text),
)

Index("idx_newsletter_active", newsletter_senders.c.is_active)

review_decisions = Table(
    "review_decisions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("article_id", String(50), nullable=False),
    Column("decision", String(20), nullable=False),  # approve/reject/edit
    Column("reviewer_notes", Text, default=""),
    Column("decided_at", String(50), nullable=False),
    Column("article_topic", String(500), default=""),
    Column("quality_score", Float, default=0.0),
)

Index("idx_review_article", review_decisions.c.article_id)
Index("idx_review_decided", review_decisions.c.decided_at)


# ---------------------------------------------------------------------------
# Engine Factory
# ---------------------------------------------------------------------------

def get_engine(db_path: Optional[Path] = None) -> sa.engine.Engine:
    """
    Get or create the singleton SQLAlchemy engine.

    Uses SQLite with QueuePool (default) and pool_pre_ping for thread safety.

    Args:
        db_path: Override database path (default: project-level reddit_content.db)

    Returns:
        SQLAlchemy Engine instance
    """
    global _engine
    if _engine is not None:
        return _engine

    path = db_path or DB_PATH
    url = f"sqlite:///{path}"
    _engine = create_engine(url, pool_pre_ping=True)
    # Auto-create tables on first use
    metadata.create_all(_engine)
    return _engine


def reset_engine() -> None:
    """Reset the engine singleton (for testing)."""
    global _engine
    if _engine is not None:
        _engine.dispose()
        _engine = None


# ---------------------------------------------------------------------------
# Content Items (unified table)
# ---------------------------------------------------------------------------

def insert_content_items(items: List[Any]) -> int:
    """
    Insert ContentItems to the unified content_items table.
    Silently skips duplicates (same source_type + source_id).

    Args:
        items: List of ContentItem dataclass instances

    Returns:
        Number of new items inserted
    """
    if not items:
        return 0

    engine = get_engine()
    inserted = 0

    with engine.begin() as conn:
        for item in items:
            try:
                conn.execute(
                    content_items.insert().values(
                        source_type=item.source_type.value,
                        source_id=item.source_id,
                        title=item.title,
                        content=item.content,
                        author=item.author,
                        url=item.url,
                        timestamp=item.timestamp,
                        trust_tier=item.trust_tier.value,
                        metadata=json.dumps(item.metadata) if item.metadata else None,
                        retrieved_at=item.retrieved_at,
                    )
                )
                inserted += 1
            except sa.exc.IntegrityError:
                pass

    return inserted


# ---------------------------------------------------------------------------
# Legacy Posts table (Reddit backward compat)
# ---------------------------------------------------------------------------

def insert_legacy_posts(items: List[Any]) -> int:
    """
    Insert ContentItems to the legacy 'posts' table for Reddit backward compat.
    Silently skips duplicates (same URL).

    Args:
        items: List of ContentItem dataclass instances

    Returns:
        Number of new posts inserted
    """
    if not items:
        return 0

    engine = get_engine()
    inserted = 0

    with engine.begin() as conn:
        for item in items:
            try:
                item_metadata = item.metadata or {}
                conn.execute(
                    posts.insert().values(
                        subreddit=item_metadata.get("subreddit", "unknown"),
                        title=item.title,
                        url=item.url,
                        author=item.author,
                        content=item.content,
                        timestamp=item.timestamp,
                        upvotes=item_metadata.get("upvotes", 0),
                        num_comments=item_metadata.get("num_comments", 0),
                        retrieved_at=item.retrieved_at,
                    )
                )
                inserted += 1
            except sa.exc.IntegrityError:
                pass

    return inserted


# ---------------------------------------------------------------------------
# Newsletter Senders (Gmail)
# ---------------------------------------------------------------------------

def get_sender_trust_tier(email: str) -> Optional[str]:
    """
    Look up trust tier for a newsletter sender.

    Args:
        email: Sender email address

    Returns:
        Trust tier string ('a', 'b', 'c', 'x') or None if not found
    """
    engine = get_engine()

    with engine.connect() as conn:
        row = conn.execute(
            sa.select(newsletter_senders.c.trust_tier).where(
                sa.and_(
                    newsletter_senders.c.email == email.lower(),
                    newsletter_senders.c.is_active == 1,
                )
            )
        ).fetchone()

    if row:
        return row[0]
    return None


def upsert_newsletter_sender(
    email: str,
    display_name: Optional[str] = None,
    trust_tier: str = "b",
    added_at: Optional[int] = None,
    notes: Optional[str] = None,
) -> bool:
    """
    Add or update a newsletter sender.
    Uses INSERT OR REPLACE for SQLite upsert semantics.

    Args:
        email: Sender email address
        display_name: Sender display name
        trust_tier: Trust tier value string
        added_at: Unix timestamp (defaults to now)
        notes: User notes

    Returns:
        True if operation succeeded
    """
    import time as _time

    engine = get_engine()

    if added_at is None:
        added_at = int(_time.time())

    try:
        with engine.begin() as conn:
            # Try insert first
            try:
                conn.execute(
                    newsletter_senders.insert().values(
                        email=email.lower(),
                        display_name=display_name,
                        trust_tier=trust_tier,
                        is_active=True,
                        added_at=added_at,
                        notes=notes,
                    )
                )
            except sa.exc.IntegrityError:
                # Already exists, update
                conn.execute(
                    newsletter_senders.update()
                    .where(newsletter_senders.c.email == email.lower())
                    .values(
                        display_name=sa.case(
                            (sa.literal(display_name is not None), display_name),
                            else_=newsletter_senders.c.display_name,
                        ) if display_name is not None else newsletter_senders.c.display_name,
                        trust_tier=trust_tier,
                        notes=sa.case(
                            (sa.literal(notes is not None), notes),
                            else_=newsletter_senders.c.notes,
                        ) if notes is not None else newsletter_senders.c.notes,
                    )
                )
        return True
    except sa.exc.SQLAlchemyError:
        return False


# ---------------------------------------------------------------------------
# Review Decisions
# ---------------------------------------------------------------------------

def save_review_decision(
    article_id: str,
    decision: str,
    notes: str = "",
    topic: str = "",
    quality_score: float = 0.0,
) -> None:
    """Save a review decision to the database."""
    from datetime import datetime, timezone

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            review_decisions.insert().values(
                article_id=article_id,
                decision=decision,
                reviewer_notes=notes,
                decided_at=datetime.now(timezone.utc).isoformat(),
                article_topic=topic,
                quality_score=quality_score,
            )
        )


def get_review_history(limit: int = 50) -> List[Dict]:
    """Get recent review decisions."""
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            review_decisions.select()
            .order_by(review_decisions.c.decided_at.desc())
            .limit(limit)
        )
        return [dict(row._mapping) for row in result]


def get_review_decision_for_article(article_id: str) -> Optional[Dict]:
    """Get the most recent review decision for a specific article."""
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            review_decisions.select()
            .where(review_decisions.c.article_id == article_id)
            .order_by(review_decisions.c.decided_at.desc())
            .limit(1)
        )
        row = result.fetchone()
        if row:
            return dict(row._mapping)
    return None


def get_decision_stats() -> Dict:
    """Get aggregate stats on review decisions."""
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(review_decisions.select())
        rows = [dict(r._mapping) for r in result]

    total = len(rows)
    if total == 0:
        return {"total": 0, "approved": 0, "rejected": 0, "edited": 0}

    approved = sum(1 for r in rows if r["decision"] == "approve")
    rejected = sum(1 for r in rows if r["decision"] == "reject")
    edited = sum(1 for r in rows if r["decision"] == "edit")
    avg_score = sum(r.get("quality_score", 0) for r in rows) / total

    return {
        "total": total,
        "approved": approved,
        "rejected": rejected,
        "edited": edited,
        "approval_rate": approved / total if total else 0,
        "avg_quality_score": avg_score,
    }
