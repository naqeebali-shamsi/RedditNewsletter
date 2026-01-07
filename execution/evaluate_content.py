#!/usr/bin/env python3
"""
Unified Content Evaluator

Evaluates content from any source (Reddit, Gmail, GitHub) using LLM-based
Signal vs Noise classification. Respects trust tiers:

- Tier A (Curated): Auto-signal, skip evaluation
- Tier B (Semi-trusted): Light evaluation
- Tier C (Untrusted): Full evaluation
- Tier X (Blocked): Skip entirely

Usage:
    python evaluate_content.py                    # Evaluate all pending
    python evaluate_content.py --source reddit   # Evaluate specific source
    python evaluate_content.py --limit 50        # Limit evaluations
    python evaluate_content.py --use-llm         # Use actual LLM (requires API key)
"""

import argparse
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Try to import LLM client
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Database path
DB_PATH = Path(__file__).parent.parent / "reddit_content.db"
DIRECTIVE_PATH = Path(__file__).parent.parent / "directives" / "market_strategy.md"


def get_db_connection():
    """Get database connection."""
    return sqlite3.connect(DB_PATH)


def load_market_strategy() -> str:
    """Load the market strategy directive for evaluation context."""
    if not DIRECTIVE_PATH.exists():
        return ""
    with open(DIRECTIVE_PATH, "r") as f:
        return f.read()


def get_unevaluated_content(
    source_type: Optional[str] = None,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """
    Fetch content items that haven't been evaluated yet.

    Args:
        source_type: Filter by source type (None = all sources)
        limit: Maximum items to return

    Returns:
        List of content item dicts
    """
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = """
        SELECT c.id, c.source_type, c.source_id, c.title, c.content,
               c.author, c.url, c.trust_tier, c.metadata, c.timestamp
        FROM content_items c
        LEFT JOIN evaluations_v2 e ON c.id = e.content_id
        WHERE e.id IS NULL
    """
    params = []

    if source_type:
        query += " AND c.source_type = ?"
        params.append(source_type)

    # Exclude blocked content (Tier X)
    query += " AND c.trust_tier != 'x'"

    query += " ORDER BY c.timestamp DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_tier_a_content(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Get Tier A (curated) content that needs auto-signaling.

    Returns:
        List of content items with trust_tier = 'a'
    """
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT c.id, c.source_type, c.source_id, c.title, c.content,
               c.author, c.url, c.trust_tier, c.metadata, c.timestamp
        FROM content_items c
        LEFT JOIN evaluations_v2 e ON c.id = e.content_id
        WHERE e.id IS NULL AND c.trust_tier = 'a'
        ORDER BY c.timestamp DESC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def evaluate_with_heuristics(
    title: str,
    content: str,
    source_type: str,
    trust_tier: str
) -> Tuple[bool, float, str]:
    """
    Simple heuristic-based evaluation (no API required).

    Returns:
        (is_signal, relevance_score, reasoning)
    """
    signal_keywords = [
        'production', 'deployment', 'llmops', 'mlops', 'quantization',
        'inference', 'fine-tuning', 'rag', 'benchmark', 'optimization',
        'architecture', 'framework', 'cost', 'scaling', 'performance',
        'llm', 'model', 'gpu', 'batch', 'latency', 'throughput',
        'embedding', 'vector', 'chunking', 'retrieval', 'agent',
        'prompt engineering', 'context window', 'token', 'api'
    ]

    noise_keywords = [
        'chatgpt', 'openai ceo', 'elon musk', 'agi', 'sentient',
        'will ai', 'should we', 'unpopular opinion', 'hot take',
        'conspiracy', 'skynet', 'terminator', 'superintelligence doom'
    ]

    text = (title + ' ' + (content or '')).lower()

    signal_count = sum(1 for kw in signal_keywords if kw in text)
    noise_count = sum(1 for kw in noise_keywords if kw in text)

    # Adjust thresholds based on trust tier
    if trust_tier == 'b':
        # Semi-trusted: lower threshold
        signal_threshold = 1
    else:
        # Untrusted: higher threshold
        signal_threshold = 2

    # Calculate score
    if signal_count >= signal_threshold and noise_count == 0:
        is_signal = True
        score = min(1.0, 0.5 + (signal_count * 0.1))
        reasoning = f"Contains {signal_count} technical keywords. Trust tier: {trust_tier.upper()}."
    elif noise_count > 0:
        is_signal = False
        score = max(0.0, 0.3 - (noise_count * 0.1))
        reasoning = f"Contains {noise_count} noise keywords. Likely non-technical."
    elif signal_count > 0:
        is_signal = trust_tier == 'b'  # Semi-trusted gets benefit of doubt
        score = 0.4 + (signal_count * 0.05)
        reasoning = f"Some technical content ({signal_count} keywords). Trust tier: {trust_tier.upper()}."
    else:
        is_signal = False
        score = 0.2
        reasoning = "No clear technical signals detected."

    return is_signal, score, reasoning


def evaluate_with_llm(
    title: str,
    content: str,
    source_type: str,
    trust_tier: str,
    market_strategy: str
) -> Tuple[bool, float, str]:
    """
    LLM-based evaluation using Groq or Gemini.

    Returns:
        (is_signal, relevance_score, reasoning)
    """
    prompt = f"""You are evaluating content for a technical AI/ML newsletter.

MARKET STRATEGY CONTEXT:
{market_strategy[:2000] if market_strategy else "Focus on practical AI engineering content."}

CONTENT TO EVALUATE:
Source: {source_type}
Trust Tier: {trust_tier} (a=curated, b=semi-trusted, c=untrusted)
Title: {title}
Content: {(content or '')[:1500]}

TASK:
Determine if this content is SIGNAL (valuable for AI engineers) or NOISE (speculative/off-topic).

SIGNAL criteria:
- Practical techniques, tutorials, or case studies
- Production deployment insights
- Performance optimization
- New tools/frameworks with technical depth
- Real-world implementation experiences

NOISE criteria:
- Pure speculation about AGI/singularity
- Celebrity/drama focused
- Marketing fluff without substance
- Off-topic content

Respond in this exact JSON format:
{{"is_signal": true/false, "score": 0.0-1.0, "reasoning": "Brief explanation"}}
"""

    # Try Groq first
    if GROQ_AVAILABLE and os.getenv("GROQ_API_KEY"):
        try:
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200,
            )
            result = json.loads(response.choices[0].message.content)
            return result["is_signal"], result.get("score", 0.5), result.get("reasoning", "")
        except Exception as e:
            print(f"    [!] Groq error: {e}, falling back to heuristics")

    # Try Gemini
    if GEMINI_AVAILABLE and os.getenv("GOOGLE_API_KEY"):
        try:
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            # Extract JSON from response
            text = response.text
            if "{" in text and "}" in text:
                json_str = text[text.find("{"):text.rfind("}")+1]
                result = json.loads(json_str)
                return result["is_signal"], result.get("score", 0.5), result.get("reasoning", "")
        except Exception as e:
            print(f"    [!] Gemini error: {e}, falling back to heuristics")

    # Fallback to heuristics
    return evaluate_with_heuristics(title, content, source_type, trust_tier)


def save_evaluation(
    content_id: int,
    is_signal: bool,
    relevance_score: float,
    reasoning: str
) -> None:
    """Save evaluation result to database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO evaluations_v2 (content_id, is_signal, relevance_score, reasoning, evaluated_at)
        VALUES (?, ?, ?, ?, ?)
    """, (content_id, is_signal, relevance_score, reasoning, int(time.time())))

    conn.commit()
    conn.close()


def auto_signal_tier_a(items: List[Dict[str, Any]]) -> int:
    """
    Auto-mark Tier A content as signal without evaluation.

    Returns:
        Number of items marked as signal
    """
    count = 0
    for item in items:
        save_evaluation(
            content_id=item["id"],
            is_signal=True,
            relevance_score=1.0,
            reasoning=f"Auto-signal: Tier A curated source ({item['source_type']})"
        )
        count += 1
    return count


def log_audit_event(event_type: str, source_type: Optional[str], details: Dict[str, Any]):
    """Log an audit event."""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT INTO audit_log (event_type, source_type, details, created_at)
            VALUES (?, ?, ?, ?)
        """, (event_type, source_type, json.dumps(details), int(time.time())))
        conn.commit()
    except sqlite3.OperationalError:
        pass
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate content from all sources for Signal vs Noise"
    )
    parser.add_argument(
        "--source",
        choices=["reddit", "gmail", "github", "rss", "manual"],
        help="Filter by source type",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of items to evaluate (default: 50)",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM for evaluation (requires API key)",
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("GhostWriter Content Evaluator")
    print(f"{'='*60}\n")

    # Load market strategy
    print("Loading market strategy...")
    market_strategy = load_market_strategy()
    if market_strategy:
        print(f"  [+] Loaded strategy ({len(market_strategy)} chars)")
    else:
        print("  [ ] No strategy file found, using defaults")

    # First, handle Tier A auto-signaling
    print("\nProcessing Tier A (curated) content...")
    tier_a_items = get_tier_a_content(args.limit)
    if tier_a_items:
        auto_count = auto_signal_tier_a(tier_a_items)
        print(f"  [+] Auto-signaled {auto_count} Tier A items")
    else:
        print("  [ ] No Tier A content pending")

    # Fetch unevaluated content (Tier B and C)
    print(f"\nFetching unevaluated content...")
    if args.source:
        print(f"  Filtering: source_type = {args.source}")

    items = get_unevaluated_content(args.source, args.limit)
    print(f"  [+] Found {len(items)} items to evaluate")

    if not items:
        print("\nNo content to evaluate. Run fetch_all.py first.\n")
        return

    # Evaluate each item
    signal_count = 0
    noise_count = 0
    evaluation_method = "LLM" if args.use_llm else "Heuristics"
    print(f"\nEvaluating with: {evaluation_method}\n")

    for i, item in enumerate(items, 1):
        source = item["source_type"]
        title = item["title"][:55] + "..." if len(item["title"]) > 55 else item["title"]
        trust = item["trust_tier"].upper()

        print(f"[{i}/{len(items)}] [{source}/{trust}] {title}")

        if args.use_llm:
            is_signal, score, reasoning = evaluate_with_llm(
                item["title"],
                item["content"],
                item["source_type"],
                item["trust_tier"],
                market_strategy,
            )
        else:
            is_signal, score, reasoning = evaluate_with_heuristics(
                item["title"],
                item["content"],
                item["source_type"],
                item["trust_tier"],
            )

        save_evaluation(item["id"], is_signal, score, reasoning)

        if is_signal:
            signal_count += 1
            print(f"    [+] SIGNAL (score: {score:.2f}): {reasoning[:60]}")
        else:
            noise_count += 1
            print(f"    [-] NOISE  (score: {score:.2f}): {reasoning[:60]}")

    # Log audit event
    log_audit_event(
        "evaluate",
        args.source,
        {
            "items_evaluated": len(items),
            "signals": signal_count,
            "noise": noise_count,
            "method": evaluation_method,
        },
    )

    # Summary
    total = len(items)
    print(f"\n{'='*60}")
    print("Evaluation Complete")
    print(f"  Total evaluated: {total}")
    if total > 0:
        print(f"  Signal: {signal_count} ({signal_count/total*100:.1f}%)")
        print(f"  Noise:  {noise_count} ({noise_count/total*100:.1f}%)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
