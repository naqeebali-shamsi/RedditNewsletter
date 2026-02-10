"""
Metadata Filter Builders for Vector Knowledge Base Retrieval.

Provides SQLAlchemy filter condition builders for scoped retrieval:
- Date range filtering (Document.date_published)
- Source type filtering (Document.source_type)
- Topic tag filtering (KnowledgeChunk.topic_tags JSONB)
- Recency shortcuts (last N months)
- Entity filtering (KnowledgeChunk.entities JSONB)

All methods return SQLAlchemy filter conditions (not query results).
These conditions are composed into WHERE clauses for pre-retrieval
filtering at the SQL layer, ensuring efficient scoped search.

Usage:
    from execution.vector_db.metadata_filters import MetadataFilter, build_filters

    # Individual filters
    date_filter = MetadataFilter.date_range(start_date, end_date)
    source_filter = MetadataFilter.source_types(['email', 'rss'])

    # Composed filters
    filters = build_filters(
        tenant_id='default',
        source_types=['email', 'rss'],
        recency_months=6
    )

    # Apply to query
    stmt = select(KnowledgeChunk).where(*filters)
"""

from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from sqlalchemy import and_, or_, func
from execution.vector_db.models import KnowledgeChunk, Document


class MetadataFilter:
    """Static filter builders for metadata-scoped retrieval.

    All methods return SQLAlchemy filter conditions that can be composed
    into WHERE clauses. No database queries are executed by this class.
    """

    @staticmethod
    def date_range(
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ):
        """Filter by document publication date range.

        Args:
            start_date: Earliest publication date (inclusive). None for open start.
            end_date: Latest publication date (inclusive). None for open end.

        Returns:
            SQLAlchemy condition on Document.date_published.
            Returns None if both dates are None (no filtering).

        Example:
            # Last 30 days
            filter = MetadataFilter.date_range(
                start_date=datetime.utcnow() - timedelta(days=30),
                end_date=datetime.utcnow()
            )
        """
        conditions = []

        if start_date is not None:
            conditions.append(Document.date_published >= start_date)

        if end_date is not None:
            conditions.append(Document.date_published <= end_date)

        if not conditions:
            return None

        if len(conditions) == 1:
            return conditions[0]

        return and_(*conditions)

    @staticmethod
    def source_types(types: List[str]):
        """Filter by source types.

        Args:
            types: List of source types (email, rss, paper, manual).
                   Must be non-empty.

        Returns:
            SQLAlchemy condition on Document.source_type.

        Raises:
            ValueError: If types list is empty.

        Example:
            filter = MetadataFilter.source_types(['email', 'rss'])
        """
        if not types:
            raise ValueError("source_types: types list cannot be empty")

        return Document.source_type.in_(types)

    @staticmethod
    def topic_tags(tags: List[str], match_any: bool = True):
        """Filter by topic tags in JSONB array.

        Args:
            tags: List of topic tags to match.
            match_any: If True, match chunks with ANY of the tags (OR).
                      If False, match chunks with ALL tags (AND).

        Returns:
            SQLAlchemy condition on KnowledgeChunk.topic_tags JSONB column.

        Raises:
            ValueError: If tags list is empty.

        Example:
            # Match chunks tagged with 'ai' OR 'ml'
            filter = MetadataFilter.topic_tags(['ai', 'ml'], match_any=True)

            # Match chunks tagged with BOTH 'ai' AND 'ml'
            filter = MetadataFilter.topic_tags(['ai', 'ml'], match_any=False)
        """
        if not tags:
            raise ValueError("topic_tags: tags list cannot be empty")

        if match_any:
            # OR: chunk has at least one of the tags
            # Use contains operator for each tag separately
            conditions = [
                KnowledgeChunk.topic_tags.contains([tag])
                for tag in tags
            ]
            return or_(*conditions)
        else:
            # AND: chunk has all tags
            # PostgreSQL JSONB @> operator checks if array contains all elements
            return KnowledgeChunk.topic_tags.contains(tags)

    @staticmethod
    def recency(months: int = 6):
        """Filter to documents published within last N months.

        Shorthand for date_range with open end (until now).

        Args:
            months: Number of months to look back (default 6).

        Returns:
            SQLAlchemy condition on Document.date_published.

        Example:
            # Documents from last 3 months
            filter = MetadataFilter.recency(months=3)
        """
        cutoff = datetime.utcnow() - timedelta(days=months * 30)
        return Document.date_published >= cutoff

    @staticmethod
    def entities(entity_values: List[str], match_any: bool = True):
        """Filter by entity values in JSONB entities array.

        Searches KnowledgeChunk.entities for objects with matching 'value' field.
        Example entities structure: [{"type": "person", "value": "Sam Altman"}, ...]

        Args:
            entity_values: List of entity values to search for.
            match_any: If True, match chunks with ANY entity value (OR).
                      If False, match chunks with ALL entity values (AND).

        Returns:
            SQLAlchemy condition on KnowledgeChunk.entities JSONB column.

        Raises:
            ValueError: If entity_values list is empty.

        Example:
            # Match chunks mentioning "Sam Altman" OR "Elon Musk"
            filter = MetadataFilter.entities(['Sam Altman', 'Elon Musk'])
        """
        if not entity_values:
            raise ValueError("entities: entity_values list cannot be empty")

        # Use PostgreSQL JSONB @> containment operator
        # Check if entities array contains object with matching value
        conditions = []
        for value in entity_values:
            # Search for {"value": "..."} in the entities array
            # Use JSONB containment: entities @> [{"value": "Sam Altman"}]
            conditions.append(
                KnowledgeChunk.entities.contains([{"value": value}])
            )

        if match_any:
            return or_(*conditions)
        else:
            return and_(*conditions)


def build_filters(
    tenant_id: str,
    date_range: Optional[Tuple[datetime, datetime]] = None,
    source_types: Optional[List[str]] = None,
    topic_tags: Optional[List[str]] = None,
    topic_match_any: bool = True,
    recency_months: Optional[int] = None,
    entity_values: Optional[List[str]] = None,
    entity_match_any: bool = True,
) -> List:
    """Build composed list of SQLAlchemy filter conditions.

    Convenience function that composes multiple metadata filters with
    mandatory tenant_id and embedding existence checks.

    Args:
        tenant_id: Tenant identifier (required).
        date_range: Optional tuple of (start_date, end_date).
        source_types: Optional list of source types to filter.
        topic_tags: Optional list of topic tags to filter.
        topic_match_any: If True, match ANY topic tag (OR). Default True.
        recency_months: Optional shorthand for date filtering (last N months).
        entity_values: Optional list of entity values to filter.
        entity_match_any: If True, match ANY entity value (OR). Default True.

    Returns:
        List of SQLAlchemy conditions suitable for .where(*conditions).

    Note:
        Always includes tenant_id filter and embedding existence check.
        If both date_range and recency_months are provided, date_range takes precedence.

    Example:
        filters = build_filters(
            tenant_id='default',
            source_types=['email', 'rss'],
            topic_tags=['ai', 'ml'],
            recency_months=6
        )

        stmt = select(KnowledgeChunk).where(*filters)
    """
    conditions = [
        KnowledgeChunk.tenant_id == tenant_id,
        KnowledgeChunk.embedding.isnot(None)
    ]

    # Date filtering (date_range takes precedence over recency_months)
    if date_range is not None:
        start_date, end_date = date_range
        date_filter = MetadataFilter.date_range(start_date, end_date)
        if date_filter is not None:
            conditions.append(date_filter)
    elif recency_months is not None:
        conditions.append(MetadataFilter.recency(recency_months))

    # Source type filtering
    if source_types:
        conditions.append(MetadataFilter.source_types(source_types))

    # Topic tag filtering
    if topic_tags:
        conditions.append(MetadataFilter.topic_tags(topic_tags, match_any=topic_match_any))

    # Entity filtering
    if entity_values:
        conditions.append(MetadataFilter.entities(entity_values, match_any=entity_match_any))

    return conditions
