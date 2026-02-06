"""
ContentSource Abstract Base Class

Provides the polymorphic abstraction for ingesting content from multiple sources
(Reddit, Gmail newsletters, GitHub, etc.) into the GhostWriter pipeline.

Design Pattern: Strategy + Factory
- Each source implements the ContentSource interface
- Factory creates appropriate source based on configuration
- ContentItem provides normalized output regardless of source
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import time


class SourceType(str, Enum):
    """Supported content source types."""
    REDDIT = "reddit"
    GMAIL = "gmail"
    GITHUB = "github"
    RSS = "rss"
    HACKERNEWS = "hackernews"
    MANUAL = "manual"


class TrustTier(str, Enum):
    """
    Trust tier determines evaluation behavior.

    A (Curated): Auto-signal, skip evaluation (e.g., Simon Willison, Pragmatic Engineer)
    B (Semi-trusted): Light evaluation (e.g., general Substack newsletters)
    C (Untrusted): Full Signal/Noise evaluation (e.g., unknown senders, Reddit)
    X (Blocked): Never fetch (spam, off-topic)
    """
    A = "a"  # Curated - highest trust
    B = "b"  # Semi-trusted
    C = "c"  # Untrusted - full evaluation
    X = "x"  # Blocked - never fetch


@dataclass
class ContentItem:
    """
    Normalized content item from any source.

    This is the unified format that all sources produce, enabling
    source-agnostic evaluation and draft generation.
    """
    # Required fields
    source_type: SourceType
    source_id: str  # Unique identifier within the source (e.g., Reddit post ID, email message ID)
    title: str

    # Content fields
    content: Optional[str] = None  # Full text content (body, comments summary, etc.)
    url: Optional[str] = None  # Link to original content
    author: Optional[str] = None  # Author name or email

    # Timestamps
    timestamp: Optional[int] = None  # Original content timestamp (Unix epoch)
    retrieved_at: int = field(default_factory=lambda: int(time.time()))

    # Trust and metadata
    trust_tier: TrustTier = TrustTier.C
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Database tracking (set after insertion)
    db_id: Optional[int] = None

    def __post_init__(self):
        """Validate and normalize fields."""
        # Ensure source_type is enum
        if isinstance(self.source_type, str):
            self.source_type = SourceType(self.source_type)

        # Ensure trust_tier is enum
        if isinstance(self.trust_tier, str):
            self.trust_tier = TrustTier(self.trust_tier)

    @property
    def unique_key(self) -> str:
        """Composite key for deduplication: source_type + source_id."""
        return f"{self.source_type.value}:{self.source_id}"

    @property
    def should_evaluate(self) -> bool:
        """Whether this item needs Signal/Noise evaluation."""
        return self.trust_tier not in (TrustTier.A, TrustTier.X)

    @property
    def is_auto_signal(self) -> bool:
        """Whether this item should be treated as a signal without evaluation."""
        return self.trust_tier == TrustTier.A

    @property
    def is_blocked(self) -> bool:
        """Whether this item should be blocked/ignored."""
        return self.trust_tier == TrustTier.X

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion."""
        return {
            "source_type": self.source_type.value,
            "source_id": self.source_id,
            "title": self.title,
            "content": self.content,
            "url": self.url,
            "author": self.author,
            "timestamp": self.timestamp,
            "retrieved_at": self.retrieved_at,
            "trust_tier": self.trust_tier.value,
            "metadata": self.metadata,  # Will be JSON-serialized by DB layer
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContentItem":
        """Create ContentItem from database row."""
        return cls(
            source_type=SourceType(data["source_type"]),
            source_id=data["source_id"],
            title=data["title"],
            content=data.get("content"),
            url=data.get("url"),
            author=data.get("author"),
            timestamp=data.get("timestamp"),
            retrieved_at=data.get("retrieved_at", int(time.time())),
            trust_tier=TrustTier(data.get("trust_tier", "c")),
            metadata=data.get("metadata", {}),
            db_id=data.get("id"),
        )


@dataclass
class FetchResult:
    """Result of a fetch operation."""
    items: List[ContentItem]
    success: bool = True
    error_message: Optional[str] = None
    items_fetched: int = 0
    items_new: int = 0  # Not already in database
    items_skipped: int = 0  # Blocked or duplicate

    def __post_init__(self):
        if self.items_fetched == 0:
            self.items_fetched = len(self.items)


class ContentSource(ABC):
    """
    Abstract base class for content sources.

    Each source (Reddit, Gmail, GitHub, etc.) implements this interface
    to provide standardized content ingestion.

    Usage:
        source = RedditSource(config)
        result = source.fetch()
        for item in result.items:
            # Process normalized ContentItem
    """

    source_type: SourceType  # Must be set by subclass

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize source with configuration.

        Args:
            config: Source-specific configuration dict
        """
        self.config = config or {}
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate source-specific configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    @abstractmethod
    def fetch(self, limit: Optional[int] = None, since: Optional[int] = None) -> FetchResult:
        """
        Fetch content items from the source.

        Args:
            limit: Maximum number of items to fetch (None = source default)
            since: Unix timestamp - only fetch items newer than this

        Returns:
            FetchResult with list of ContentItems and metadata
        """
        pass

    @abstractmethod
    def normalize(self, raw_item: Any) -> ContentItem:
        """
        Convert source-specific raw item to normalized ContentItem.

        Args:
            raw_item: Source-specific data structure (e.g., Reddit post dict, email object)

        Returns:
            Normalized ContentItem
        """
        pass

    @classmethod
    @abstractmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        """
        Return JSON schema for source configuration.

        Used by UI to render configuration forms.

        Returns:
            JSON schema dict with required/optional fields
        """
        pass

    def get_trust_tier(self, raw_item: Any) -> TrustTier:
        """
        Determine trust tier for a raw item.

        Override in subclasses for source-specific logic
        (e.g., Gmail checks sender against newsletter_senders table).

        Args:
            raw_item: Source-specific data structure

        Returns:
            TrustTier enum value
        """
        return TrustTier.C  # Default: untrusted, full evaluation

    def is_duplicate(self, source_id: str, db_conn: Any) -> bool:
        """
        Check if item already exists in database.

        Args:
            source_id: Source-specific unique identifier
            db_conn: Database connection

        Returns:
            True if item exists, False otherwise
        """
        cursor = db_conn.execute(
            "SELECT 1 FROM content_items WHERE source_type = ? AND source_id = ? LIMIT 1",
            (self.source_type.value, source_id)
        )
        return cursor.fetchone() is not None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(source_type={self.source_type.value})"
