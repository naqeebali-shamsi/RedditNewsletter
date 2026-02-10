"""
Recency Scoring with Time Decay for Trend-Aware Retrieval.

Implements exponential half-life decay to prioritize recent documents
for trend-sensitive queries. Combines semantic relevance with recency
using weighted fusion.

Key concepts:
- Time decay: Exponential half-life decay (default 14 days)
- Trend detection: Keyword-based heuristics for trend queries
- Recency boost: Fuses semantic + recency scores with configurable weights

Usage:
    from execution.vector_db.recency_scoring import RecencyScorer

    scorer = RecencyScorer(half_life_days=14)

    # Check if query needs recency boost
    if scorer.is_trend_query("latest AI developments"):
        # Apply recency boost to search results
        boosted = scorer.apply_recency_boost(results)

    # Or use convenience method
    results = scorer.score_results(query, results)
"""

import math
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional


# Trend query keywords for automatic recency detection
TREND_KEYWORDS = frozenset({
    "latest", "recent", "new", "breaking", "current", "today",
    "now", "upcoming", "emerging", "trending", "this week",
    "this month", "this year"
})

# Year pattern for detecting year-specific queries (e.g., "2026 AI trends")
YEAR_PATTERN = re.compile(r"\b202\d\b")

# Default half-life for time decay (14 days per research recommendation)
DEFAULT_HALF_LIFE_DAYS = 14


class RecencyScorer:
    """Apply time decay to search results for recency-aware ranking.

    Uses exponential half-life decay to prioritize recent documents
    in trend-sensitive queries. Automatically detects trend queries
    via keyword heuristics and year patterns.

    Attributes:
        half_life_days: Time for recency score to decay to 50%.
    """

    def __init__(self, half_life_days: int = DEFAULT_HALF_LIFE_DAYS):
        """Initialize RecencyScorer with half-life parameter.

        Args:
            half_life_days: Number of days for recency score to decay to 50%.
                           Default 14 days (research-validated for trend queries).

        Example:
            # Standard trend scoring (14-day half-life)
            scorer = RecencyScorer()

            # Faster decay for breaking news (7-day half-life)
            news_scorer = RecencyScorer(half_life_days=7)

            # Slower decay for evergreen content (30-day half-life)
            evergreen_scorer = RecencyScorer(half_life_days=30)
        """
        self.half_life_days = half_life_days

    def time_decay(self, date_published: Optional[datetime]) -> float:
        """Calculate exponential decay factor based on document age.

        Formula: decay = 0.5 ^ (age_days / half_life_days)

        Args:
            date_published: Document publication date. None returns neutral 0.5.

        Returns:
            Float between 0 and 1, where:
            - 1.0 = published today
            - 0.5 = published half_life_days ago
            - 0.25 = published 2*half_life_days ago
            - 0.5 = unknown date (neutral)

        Example:
            scorer = RecencyScorer(half_life_days=14)
            scorer.time_decay(datetime.utcnow())  # ~1.0 (today)
            scorer.time_decay(datetime.utcnow() - timedelta(days=14))  # ~0.5
            scorer.time_decay(datetime.utcnow() - timedelta(days=28))  # ~0.25
            scorer.time_decay(None)  # 0.5 (neutral)
        """
        if date_published is None:
            return 0.5  # Neutral score for unknown dates

        age_days = (datetime.utcnow() - date_published).days

        # Handle future dates (clamp age to 0)
        if age_days < 0:
            age_days = 0

        # Exponential decay: 0.5 ^ (age / half_life)
        decay = math.pow(0.5, age_days / self.half_life_days)

        return decay

    def is_trend_query(self, query: str) -> bool:
        """Detect if query is trend-sensitive via keyword heuristics.

        Checks for:
        - Trend keywords (latest, recent, new, breaking, etc.)
        - Year patterns (2026, 2025, etc.)

        Args:
            query: User query string.

        Returns:
            True if query appears to be trend-sensitive.

        Example:
            scorer = RecencyScorer()
            scorer.is_trend_query("latest AI developments")  # True
            scorer.is_trend_query("2026 tech trends")  # True
            scorer.is_trend_query("database architecture patterns")  # False
        """
        query_lower = query.lower()

        # Check for trend keywords
        if any(keyword in query_lower for keyword in TREND_KEYWORDS):
            return True

        # Check for year patterns (e.g., "2026")
        if YEAR_PATTERN.search(query):
            return True

        return False

    def apply_recency_boost(
        self,
        results: List[Dict],
        semantic_weight: float = 0.7,
        recency_weight: float = 0.3
    ) -> List[Dict]:
        """Apply time decay to search results and compute fused scores.

        Adds 'recency_score' and 'fused_score' keys to each result dict.
        Re-sorts results by fused_score descending.

        Formula: fused_score = semantic_weight * rrf_score + recency_weight * recency_score

        Args:
            results: List of result dicts with 'rrf_score' and 'date_published' keys.
            semantic_weight: Weight for semantic/RRF score (default 0.7).
            recency_weight: Weight for recency score (default 0.3).

        Returns:
            Results with added recency_score and fused_score, sorted by fused_score.

        Note:
            Modifies result dicts in-place. Missing 'rrf_score' defaults to 0.5.
            Missing 'date_published' gets neutral recency (0.5).

        Example:
            results = [
                {'rrf_score': 0.9, 'date_published': datetime(2025, 12, 1)},
                {'rrf_score': 0.7, 'date_published': datetime(2026, 2, 1)},
            ]
            boosted = scorer.apply_recency_boost(results)
            # Recent doc (0.7 semantic) may now rank higher than old doc (0.9 semantic)
        """
        for result in results:
            # Get semantic score (default to neutral if missing)
            rrf_score = result.get("rrf_score", 0.5)

            # Calculate recency score
            date_published = result.get("date_published")
            recency_score = self.time_decay(date_published)

            # Compute fused score
            fused_score = (semantic_weight * rrf_score) + (recency_weight * recency_score)

            # Add scores to result
            result["recency_score"] = recency_score
            result["fused_score"] = fused_score

        # Re-sort by fused score descending
        results.sort(key=lambda x: x["fused_score"], reverse=True)

        return results

    def score_results(
        self,
        query: str,
        results: List[Dict],
        semantic_weight: float = 0.7,
        recency_weight: float = 0.3
    ) -> List[Dict]:
        """Convenience method: apply recency boost only for trend queries.

        Automatically detects if query is trend-sensitive via is_trend_query().
        If trend query: applies recency boost and re-sorts.
        If not trend query: returns results unchanged.

        Args:
            query: User query string.
            results: List of result dicts with 'rrf_score' and 'date_published'.
            semantic_weight: Weight for semantic/RRF score (default 0.7).
            recency_weight: Weight for recency score (default 0.3).

        Returns:
            Results with optional recency boost applied, sorted by relevance.

        Example:
            scorer = RecencyScorer()

            # Trend query → recency boost applied
            results = scorer.score_results("latest AI news", results)

            # Non-trend query → results unchanged
            results = scorer.score_results("database architecture", results)
        """
        if self.is_trend_query(query):
            return self.apply_recency_boost(
                results,
                semantic_weight=semantic_weight,
                recency_weight=recency_weight
            )
        else:
            # Not a trend query - return results unchanged
            return results
