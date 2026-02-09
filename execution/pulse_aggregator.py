#!/usr/bin/env python3
"""
Pulse Aggregator

Queries content_items from the last N hours, clusters by keyword frequency,
scores topics by cross-source presence and engagement, and writes a daily
pulse summary to the pulse_daily table.

Usage:
    python pulse_aggregator.py                    # Default: last 24 hours, min 2 mentions
    python pulse_aggregator.py --days 3           # Last 3 days
    python pulse_aggregator.py --min-mentions 3   # Require 3+ mentions
"""

import argparse
import json
import re
import sqlite3
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Optional: scikit-learn for TF-IDF keyword extraction
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

# Optional: VADER for rule-based sentiment analysis
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
    _HAS_VADER = True
except ImportError:
    _vader = None
    _HAS_VADER = False

# Database path
DB_PATH = Path(__file__).parent.parent / "reddit_content.db"

# Common English stopwords to filter out of keyword extraction
STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "had", "has", "have", "he", "her", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "just", "my", "no", "not", "of", "on", "or", "our",
    "out", "so", "some", "than", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "to", "too", "up", "us", "very", "was",
    "we", "what", "when", "which", "who", "why", "will", "with", "would",
    "you", "your", "about", "after", "all", "also", "any", "back", "been",
    "before", "being", "between", "both", "can", "could", "did", "do", "does",
    "done", "down", "each", "even", "every", "few", "first", "get", "going",
    "got", "here", "him", "https", "http", "www", "com", "org", "new", "now",
    "one", "only", "other", "over", "own", "more", "most", "much", "like",
    "make", "made", "many", "may", "me", "might", "must", "need", "never",
    "next", "off", "old", "once", "open", "part", "per", "put", "really",
    "right", "said", "same", "see", "should", "show", "since", "still",
    "such", "take", "tell", "think", "through", "time", "under", "use",
    "used", "using", "want", "way", "well", "went", "were", "while",
    "work", "year", "yet",
}

# Minimum word length to consider
MIN_WORD_LENGTH = 3


def get_db_connection() -> sqlite3.Connection:
    """Get database connection with row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_recent_items(conn: sqlite3.Connection, since_timestamp: int) -> List[Dict[str, Any]]:
    """
    Fetch content_items from the database since a given timestamp.

    Args:
        conn: Database connection
        since_timestamp: Unix timestamp cutoff

    Returns:
        List of content item dicts
    """
    cursor = conn.execute(
        """
        SELECT id, source_type, source_id, title, content, author,
               url, timestamp, trust_tier, metadata, retrieved_at
        FROM content_items
        WHERE retrieved_at >= ? OR timestamp >= ?
        ORDER BY timestamp DESC
        """,
        (since_timestamp, since_timestamp),
    )

    items = []
    for row in cursor.fetchall():
        item = dict(row)
        # Parse metadata JSON
        if item.get("metadata"):
            try:
                item["metadata"] = json.loads(item["metadata"])
            except (json.JSONDecodeError, TypeError):
                item["metadata"] = {}
        else:
            item["metadata"] = {}
        items.append(item)

    return items


def _extract_keywords_simple(text: str) -> List[str]:
    """
    Extract meaningful keywords from text by tokenizing and filtering stopwords.

    Simple fallback used when scikit-learn is not installed.

    Args:
        text: Input text to extract keywords from

    Returns:
        List of lowercase keywords
    """
    if not text:
        return []

    # Lowercase and split on non-alphanumeric chars
    words = re.findall(r"[a-z][a-z0-9]+", text.lower())

    # Filter stopwords and short words
    return [w for w in words if w not in STOPWORDS and len(w) >= MIN_WORD_LENGTH]


def extract_keywords(text: str) -> List[str]:
    """Extract keywords from text. Delegates to _extract_keywords_simple."""
    return _extract_keywords_simple(text)


def extract_keywords_tfidf(documents: List[str], max_features: int = 200) -> List[Tuple[str, float]]:
    """Extract keywords using TF-IDF vectorization.

    Returns list of (keyword, tfidf_score) sorted by score descending.
    Falls back to simple frequency counting if sklearn not installed.

    Args:
        documents: List of text documents to analyze
        max_features: Maximum number of features for TF-IDF

    Returns:
        List of (keyword, score) tuples sorted by score descending
    """
    if not _HAS_SKLEARN or len(documents) < 2:
        # Fallback: frequency-based scoring from simple extraction
        all_words: Counter = Counter()
        for doc in documents:
            all_words.update(_extract_keywords_simple(doc))
        total = max(sum(all_words.values()), 1)
        scored = [(word, count / total) for word, count in all_words.most_common(max_features)]
        return scored

    try:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
        )
        tfidf_matrix = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()

        # Get mean TF-IDF score across all documents for each feature
        mean_scores = tfidf_matrix.mean(axis=0).A1
        scored = list(zip(feature_names, mean_scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
    except ValueError:
        # Empty vocabulary (all terms filtered by min_df/max_df on small corpora)
        all_words: Counter = Counter()
        for doc in documents:
            all_words.update(_extract_keywords_simple(doc))
        total = max(sum(all_words.values()), 1)
        return [(word, count / total) for word, count in all_words.most_common(max_features)]


def extract_bigrams(words: List[str]) -> List[str]:
    """
    Extract bigrams (two-word phrases) from a list of words.

    Args:
        words: List of keywords

    Returns:
        List of bigram strings (e.g., "machine learning")
    """
    if len(words) < 2:
        return []
    return [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]


def cluster_topics(
    items: List[Dict[str, Any]],
    min_mentions: int = 2,
) -> List[Dict[str, Any]]:
    """
    Cluster items into topic groups using keyword and bigram frequency.

    Scoring algorithm:
    - Cross-source bonus: topics appearing in 2+ source types score higher
    - Engagement bonus: aggregate score/upvotes from metadata
    - Recency bonus: newer items get a slight boost

    Args:
        items: List of content item dicts
        min_mentions: Minimum mentions to include a topic

    Returns:
        Sorted list of topic dicts with scores
    """
    # Track keywords/bigrams -> source types and items
    keyword_sources: Dict[str, Set[str]] = defaultdict(set)
    keyword_items: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    keyword_count: Counter = Counter()
    bigram_sources: Dict[str, Set[str]] = defaultdict(set)
    bigram_items: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    bigram_count: Counter = Counter()

    now = int(time.time())

    for item in items:
        source_type = item.get("source_type", "unknown")
        text = f"{item.get('title', '')} {item.get('content', '')}"
        words = extract_keywords(text)

        # Count unigrams
        seen_words = set()
        for word in words:
            if word not in seen_words:
                keyword_sources[word].add(source_type)
                keyword_items[word].append(item)
                keyword_count[word] += 1
                seen_words.add(word)

        # Count bigrams
        bigrams = extract_bigrams(words)
        seen_bigrams = set()
        for bigram in bigrams:
            if bigram not in seen_bigrams:
                bigram_sources[bigram].add(source_type)
                bigram_items[bigram].append(item)
                bigram_count[bigram] += 1
                seen_bigrams.add(bigram)

    # Score and rank topics (prefer bigrams for specificity, fall back to unigrams)
    topics = []

    # Process bigrams first (more specific)
    for bigram, count in bigram_count.most_common(50):
        if count < min_mentions:
            continue

        sources = bigram_sources[bigram]
        related = bigram_items[bigram]

        # Cross-source score: more source types = higher priority
        cross_source_score = len(sources) * 10

        # Engagement score from metadata
        engagement = 0
        for item in related:
            meta = item.get("metadata", {})
            engagement += meta.get("score", 0) + meta.get("upvotes", 0)

        # Recency bonus (items from last 6 hours get extra weight)
        recency_bonus = sum(
            2 for item in related
            if (item.get("timestamp") or 0) > now - 21600
        )

        total_score = cross_source_score + (engagement * 0.1) + recency_bonus + count

        topics.append({
            "topic": bigram,
            "mentions": count,
            "sources": sorted(sources),
            "cross_source_count": len(sources),
            "engagement": engagement,
            "score": round(total_score, 2),
            "sample_titles": [item.get("title", "")[:100] for item in related[:3]],
        })

    # Add top unigrams that aren't already covered by bigrams
    bigram_words = set()
    for t in topics:
        bigram_words.update(t["topic"].split())

    for word, count in keyword_count.most_common(30):
        if count < min_mentions or word in bigram_words:
            continue

        sources = keyword_sources[word]
        related = keyword_items[word]

        cross_source_score = len(sources) * 10
        engagement = sum(
            item.get("metadata", {}).get("score", 0) +
            item.get("metadata", {}).get("upvotes", 0)
            for item in related
        )
        recency_bonus = sum(
            2 for item in related
            if (item.get("timestamp") or 0) > now - 21600
        )

        total_score = cross_source_score + (engagement * 0.1) + recency_bonus + count

        topics.append({
            "topic": word,
            "mentions": count,
            "sources": sorted(sources),
            "cross_source_count": len(sources),
            "engagement": engagement,
            "score": round(total_score, 2),
            "sample_titles": [item.get("title", "")[:100] for item in related[:3]],
        })

    # Sort by score descending
    topics.sort(key=lambda t: t["score"], reverse=True)

    return topics[:20]  # Top 20 topics


def analyze_sentiment(text: str) -> Dict[str, float]:
    """Analyze sentiment using VADER (handles negation, intensifiers, slang).

    Returns dict with 'compound' (-1 to 1), 'pos', 'neg', 'neu' scores.
    Falls back to simple keyword matching if VADER not installed.

    Args:
        text: Text to analyze

    Returns:
        Dict with sentiment scores
    """
    if not text:
        return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}
    if _HAS_VADER and _vader:
        return _vader.polarity_scores(text)
    return _analyze_sentiment_simple(text)


def _analyze_sentiment_simple(text: str) -> Dict[str, float]:
    """Simple keyword-based sentiment fallback.

    Returns VADER-compatible dict with compound, pos, neg, neu scores
    so callers don't need to branch on which implementation ran.

    Args:
        text: Text to analyze

    Returns:
        Dict with sentiment scores (compound, pos, neg, neu)
    """
    positive_words = {
        "great", "awesome", "excellent", "amazing", "good", "best", "love",
        "improvement", "impressive", "breakthrough", "solved", "success",
        "faster", "better", "efficient", "powerful", "release", "launched",
    }
    negative_words = {
        "bad", "terrible", "broken", "bug", "issue", "problem", "error",
        "fail", "crash", "slow", "vulnerability", "breach", "hack", "risk",
        "deprecated", "removed", "worst", "warning", "critical",
    }

    words = set(re.findall(r"[a-z]+", text.lower()))
    pos_count = len(words & positive_words)
    neg_count = len(words & negative_words)
    total = max(pos_count + neg_count, 1)

    pos_score = pos_count / total if total > 0 else 0.0
    neg_score = neg_count / total if total > 0 else 0.0
    neu_score = 1.0 - pos_score - neg_score

    # Approximate a compound score in [-1, 1]
    compound = (pos_count - neg_count) / total if total > 0 else 0.0

    return {"compound": compound, "pos": pos_score, "neg": neg_score, "neu": max(neu_score, 0.0)}


def compute_sentiment_summary(items: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Classify items as positive, negative, or neutral using VADER sentiment.

    Uses VADER's compound score when available, falls back to simple keyword
    matching otherwise. Thresholds: compound >= 0.05 positive, <= -0.05 negative.

    Args:
        items: List of content item dicts

    Returns:
        Dict with positive, negative, neutral counts
    """
    counts = {"positive": 0, "negative": 0, "neutral": 0}

    for item in items:
        text = f"{item.get('title', '')} {item.get('content', '')}"
        scores = analyze_sentiment(text)
        compound = scores.get("compound", 0.0)

        if compound >= 0.05:
            counts["positive"] += 1
        elif compound <= -0.05:
            counts["negative"] += 1
        else:
            counts["neutral"] += 1

    return counts


def compute_source_breakdown(items: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Count items per source type.

    Args:
        items: List of content item dicts

    Returns:
        Dict mapping source_type -> count
    """
    breakdown: Counter = Counter()
    for item in items:
        breakdown[item.get("source_type", "unknown")] += 1
    return dict(breakdown)


def generate_content_angles(topics: List[Dict[str, Any]]) -> List[str]:
    """
    Suggest content angles based on top topics.

    Prioritizes cross-source topics (appearing in 3+ sources) and
    high-engagement topics.

    Args:
        topics: Ranked topic list from cluster_topics

    Returns:
        List of suggested content angle strings
    """
    angles = []

    for topic in topics[:10]:
        cross = topic["cross_source_count"]
        mentions = topic["mentions"]
        name = topic["topic"]

        if cross >= 3:
            angles.append(
                f"HIGH PRIORITY: '{name}' trending across {cross} sources "
                f"({mentions} mentions) - strong cross-platform signal"
            )
        elif cross >= 2:
            angles.append(
                f"MODERATE: '{name}' appearing in {cross} sources "
                f"({mentions} mentions) - worth monitoring"
            )
        elif mentions >= 5:
            angles.append(
                f"VOLUME: '{name}' has {mentions} mentions in single source "
                f"- high activity, verify breadth"
            )

    if not angles:
        angles.append("No strong cross-source signals detected in this period")

    return angles


def save_pulse_daily(
    conn: sqlite3.Connection,
    date_str: str,
    topics: List[Dict[str, Any]],
    sentiment: Dict[str, int],
    angles: List[str],
    source_breakdown: Dict[str, int],
) -> int:
    """
    Insert or replace a pulse_daily entry.

    Args:
        conn: Database connection
        date_str: Date string (YYYY-MM-DD)
        topics: Ranked topic list
        sentiment: Sentiment counts
        angles: Content angle suggestions
        source_breakdown: Items per source type

    Returns:
        Row ID of inserted/updated entry
    """
    cursor = conn.execute(
        """
        INSERT OR REPLACE INTO pulse_daily
            (date, top_topics, sentiment_summary, content_angles, source_breakdown, generated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            date_str,
            json.dumps(topics),
            json.dumps(sentiment),
            json.dumps(angles),
            json.dumps(source_breakdown),
            int(time.time()),
        ),
    )
    conn.commit()
    return cursor.lastrowid


def ensure_pulse_tables(conn: sqlite3.Connection) -> None:
    """Create pulse tables if they don't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pulse_daily (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            top_topics TEXT,
            sentiment_summary TEXT,
            content_angles TEXT,
            source_breakdown TEXT,
            generated_at INTEGER NOT NULL,
            UNIQUE(date)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pulse_feeds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            source_type TEXT NOT NULL,
            url TEXT NOT NULL,
            fetch_interval_minutes INTEGER DEFAULT 60,
            is_active BOOLEAN DEFAULT 1,
            last_fetched_at INTEGER,
            UNIQUE(url)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pulse_date ON pulse_daily(date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_feeds_active ON pulse_feeds(is_active)")
    conn.commit()


def run_pulse(days: int = 1, min_mentions: int = 2) -> Dict[str, Any]:
    """
    Run the full pulse aggregation pipeline.

    Args:
        days: Number of days to look back
        min_mentions: Minimum mentions to include a topic

    Returns:
        Summary dict of the pulse run
    """
    conn = get_db_connection()
    ensure_pulse_tables(conn)

    since_timestamp = int(time.time()) - (days * 86400)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Fetch recent items
    items = fetch_recent_items(conn, since_timestamp)

    if not items:
        print("  [!] No content items found in the specified period")
        conn.close()
        return {"items_analyzed": 0, "topics_found": 0, "date": date_str}

    # Run analysis
    topics = cluster_topics(items, min_mentions=min_mentions)
    sentiment = compute_sentiment_summary(items)
    source_breakdown = compute_source_breakdown(items)
    angles = generate_content_angles(topics)

    # Save to database
    row_id = save_pulse_daily(conn, date_str, topics, sentiment, angles, source_breakdown)
    conn.close()

    return {
        "date": date_str,
        "items_analyzed": len(items),
        "topics_found": len(topics),
        "sentiment": sentiment,
        "source_breakdown": source_breakdown,
        "top_topics": [t["topic"] for t in topics[:5]],
        "content_angles": angles[:3],
        "pulse_daily_id": row_id,
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Aggregate content items into daily pulse summary"
    )
    parser.add_argument(
        "--days", type=int, default=1,
        help="Number of days to look back (default: 1)",
    )
    parser.add_argument(
        "--min-mentions", type=int, default=2,
        help="Minimum mentions to include a topic (default: 2)",
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("Pulse Aggregator")
    print(f"{'='*60}")
    print(f"  Lookback: {args.days} day(s)")
    print(f"  Min mentions: {args.min_mentions}")

    summary = run_pulse(days=args.days, min_mentions=args.min_mentions)

    print(f"\n  Items analyzed: {summary['items_analyzed']}")
    print(f"  Topics found:   {summary['topics_found']}")

    if summary.get("sentiment"):
        s = summary["sentiment"]
        print(f"  Sentiment:      +{s.get('positive', 0)} / "
              f"-{s.get('negative', 0)} / "
              f"~{s.get('neutral', 0)}")

    if summary.get("source_breakdown"):
        print(f"  Sources:        {summary['source_breakdown']}")

    if summary.get("top_topics"):
        print(f"\n  Top topics:")
        for topic in summary["top_topics"]:
            print(f"    - {topic}")

    if summary.get("content_angles"):
        print(f"\n  Content angles:")
        for angle in summary["content_angles"]:
            print(f"    > {angle}")

    print(f"\n{'='*60}")
    print("Done")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
