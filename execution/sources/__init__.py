"""
Content Sources Package

Factory and registry for content source implementations.
Provides centralized access to all source types.

Usage:
    from execution.sources import SourceFactory, SourceType

    # Create specific source
    reddit = SourceFactory.create(SourceType.REDDIT, config={...})

    # Get all registered sources
    sources = SourceFactory.get_all_sources()

    # Fetch from all configured sources
    all_items = SourceFactory.fetch_all()
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Type

from .base_source import (
    ContentSource,
    ContentItem,
    FetchResult,
    SourceType,
    TrustTier,
)
from .circuit_breaker import SourceCircuit, AllSourcesFailedError

_logger = logging.getLogger("ghostwriter.sources")

# Registry of source implementations
_SOURCE_REGISTRY: Dict[SourceType, Type[ContentSource]] = {}


def register_source(source_type: SourceType):
    """
    Decorator to register a content source implementation.

    Usage:
        @register_source(SourceType.REDDIT)
        class RedditSource(ContentSource):
            ...
    """
    def decorator(cls: Type[ContentSource]) -> Type[ContentSource]:
        if not issubclass(cls, ContentSource):
            raise TypeError(f"{cls.__name__} must inherit from ContentSource")
        cls.source_type = source_type
        _SOURCE_REGISTRY[source_type] = cls
        return cls
    return decorator


class SourceFactory:
    """
    Factory for creating and managing content sources.

    Provides centralized source creation, configuration validation,
    and multi-source orchestration.
    """

    @classmethod
    def create(
        cls,
        source_type: SourceType,
        config: Optional[Dict[str, Any]] = None
    ) -> ContentSource:
        """
        Create a content source instance.

        Args:
            source_type: Type of source to create
            config: Source-specific configuration

        Returns:
            Configured ContentSource instance

        Raises:
            ValueError: If source type is not registered
        """
        if isinstance(source_type, str):
            source_type = SourceType(source_type)

        if source_type not in _SOURCE_REGISTRY:
            registered = [s.value for s in _SOURCE_REGISTRY.keys()]
            raise ValueError(
                f"Unknown source type: {source_type.value}. "
                f"Registered sources: {registered}"
            )

        source_class = _SOURCE_REGISTRY[source_type]
        return source_class(config)

    @classmethod
    def get_registered_sources(cls) -> List[SourceType]:
        """Get list of all registered source types."""
        return list(_SOURCE_REGISTRY.keys())

    @classmethod
    def get_source_class(cls, source_type: SourceType) -> Type[ContentSource]:
        """Get the class for a source type."""
        if source_type not in _SOURCE_REGISTRY:
            raise ValueError(f"Unknown source type: {source_type.value}")
        return _SOURCE_REGISTRY[source_type]

    @classmethod
    def get_config_schema(cls, source_type: SourceType) -> Dict[str, Any]:
        """Get configuration schema for a source type."""
        source_class = cls.get_source_class(source_type)
        return source_class.get_config_schema()

    @classmethod
    def is_source_available(cls, source_type: SourceType) -> bool:
        """Check if a source type is registered and available."""
        return source_type in _SOURCE_REGISTRY


class SourceManager:
    """
    Manages multiple content sources for coordinated fetching.

    Used by fetch_all.py to orchestrate multi-source content ingestion.
    Includes circuit breaker per source and concurrent fetching.
    """

    def __init__(self):
        self._sources: Dict[SourceType, ContentSource] = {}
        self._circuits: Dict[SourceType, SourceCircuit] = {}

    def add_source(self, source: ContentSource) -> None:
        """Add a configured source to the manager."""
        self._sources[source.source_type] = source
        self._circuits[source.source_type] = SourceCircuit(
            source_type=source.source_type.value
        )

    def remove_source(self, source_type: SourceType) -> None:
        """Remove a source from the manager."""
        self._sources.pop(source_type, None)
        self._circuits.pop(source_type, None)

    def get_source(self, source_type: SourceType) -> Optional[ContentSource]:
        """Get a specific source if configured."""
        return self._sources.get(source_type)

    def get_all_sources(self) -> List[ContentSource]:
        """Get all configured sources."""
        return list(self._sources.values())

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
            "circuit_states": {
                st.value: {"state": c.state, "failures": c.failure_count}
                for st, c in self._circuits.items()
            },
            "healthy": len(missing) == 0,
        }

    def fetch_all(
        self,
        limit_per_source: Optional[int] = None,
        since: Optional[int] = None
    ) -> Dict[SourceType, FetchResult]:
        """
        Fetch content from all configured sources concurrently.

        Uses ThreadPoolExecutor for parallel fetching and circuit breakers
        to skip sources that have been consistently failing.

        Args:
            limit_per_source: Max items per source (None = source default)
            since: Unix timestamp - only fetch items newer than this

        Returns:
            Dict mapping source type to FetchResult

        Raises:
            AllSourcesFailedError: If every configured source fails
        """
        results: Dict[SourceType, FetchResult] = {}

        # Check circuit breakers first and filter to attemptable sources
        attemptable: Dict[SourceType, ContentSource] = {}
        for source_type, source in self._sources.items():
            circuit = self._circuits.get(source_type)
            if circuit and not circuit.should_attempt():
                results[source_type] = FetchResult(
                    items=[],
                    success=False,
                    error_message=(
                        f"Circuit open (last {circuit.failure_count} attempts failed)"
                    ),
                )
                continue
            attemptable[source_type] = source

        def _fetch_one(source_type, source):
            try:
                return source_type, source.fetch(
                    limit=limit_per_source, since=since
                )
            except Exception as e:
                return source_type, FetchResult(
                    items=[], success=False, error_message=str(e)
                )

        # Fetch concurrently
        if attemptable:
            with ThreadPoolExecutor(max_workers=len(attemptable)) as executor:
                futures = {
                    executor.submit(_fetch_one, st, s): st
                    for st, s in attemptable.items()
                }
                for future in as_completed(futures):
                    source_type, result = future.result()
                    results[source_type] = result

                    # Update circuit breaker
                    circuit = self._circuits.get(source_type)
                    if circuit:
                        if result.success:
                            circuit.record_success()
                        else:
                            circuit.record_failure()

        # Check minimum source threshold
        successful = sum(1 for r in results.values() if r.success)
        if successful == 0 and len(results) > 0:
            error_details = "; ".join(
                f"{st.value}: {r.error_message}"
                for st, r in results.items()
                if r.error_message
            )
            raise AllSourcesFailedError(
                f"All {len(results)} sources failed. Errors: {error_details}"
            )

        return results

    def get_total_items(self, results: Dict[SourceType, FetchResult]) -> int:
        """Count total items across all fetch results."""
        return sum(r.items_fetched for r in results.values())

    def get_all_items(self, results: Dict[SourceType, FetchResult]) -> List[ContentItem]:
        """Flatten all items from fetch results into single list."""
        items = []
        for result in results.values():
            if result.success:
                items.extend(result.items)
        return items


# Auto-import available sources to register them
# Each source module uses @register_source decorator
_IMPORT_WARNINGS: List[str] = []

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

try:
    from . import hackernews_source
except ImportError as e:
    _IMPORT_WARNINGS.append(f"HackerNewsSource unavailable: {e}")
    _logger.warning(f"HackerNewsSource import failed: {e}")

try:
    from . import rss_source
except ImportError as e:
    _IMPORT_WARNINGS.append(f"RSSSource unavailable: {e}")
    _logger.warning(f"RSSSource import failed: {e}")


# Export public API
__all__ = [
    # Core classes
    "ContentSource",
    "ContentItem",
    "FetchResult",
    # Enums
    "SourceType",
    "TrustTier",
    # Factory/Registry
    "SourceFactory",
    "SourceManager",
    "register_source",
    # Error types
    "AllSourcesFailedError",
]
