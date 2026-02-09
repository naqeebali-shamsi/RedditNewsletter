#!/usr/bin/env python3
"""Simple agent-based draft generation with quality gate.

For full pipeline with research, fact verification, specialist refinement,
and visual generation, use generate_medium_full.py instead.

This module provides a lighter-weight alternative that skips the research
and specialist passes but still runs WriterAgent + AdversarialPanel quality
gate loop. Use this when you want quick, quality-gated drafts without the
full multi-agent pipeline overhead.

Flow:
1. Fetch signal content from database
2. Writer Agent generates initial draft (using improved prompts)
3. Adversarial Panel reviews draft
4. If score < 7.0: Writer revises based on fix instructions
5. Loop until pass OR max iterations
6. Save approved drafts to database and files

Usage:
    python generate_drafts_v2.py --platform linkedin --limit 5
    python generate_drafts_v2.py --platform medium --limit 3 --max-iterations 5
    python generate_drafts_v2.py --unified --source reddit
"""

import json
import sqlite3
import argparse
import sys
from pathlib import Path
from execution.utils.datetime_utils import utc_now
from typing import Any, Dict, List, Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from execution.agents.writer import WriterAgent
from execution.agents.adversarial_panel import AdversarialPanelAgent
from execution.quality_gate import QualityGate, QualityGateResult

# Paths
DB_PATH = Path(__file__).parent.parent / "reddit_content.db"
DIRECTIVE_PATH = Path(__file__).parent.parent / "directives" / "market_strategy.md"
OUTPUT_DIR = Path(__file__).parent.parent / "drafts"


def get_db_connection():
    """Get database connection."""
    return sqlite3.connect(DB_PATH)


def get_signal_content_unified(
    limit: int = 10,
    source_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Fetch Signal content from the unified content_items table."""
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = """
        SELECT c.id, c.source_type, c.source_id, c.title, c.content,
               c.author, c.url, c.trust_tier, c.metadata,
               e.reasoning, e.relevance_score
        FROM content_items c
        JOIN evaluations_v2 e ON c.id = e.content_id
        WHERE e.is_signal = 1
    """
    params = []

    if source_type:
        query += " AND c.source_type = ?"
        params.append(source_type)

    query += " ORDER BY c.timestamp DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_signal_posts_legacy(limit: int = 10) -> List[Dict[str, Any]]:
    """Fetch signal posts from legacy posts/evaluations tables."""
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT p.id, p.subreddit, p.title, p.content, p.url, e.reasoning
        FROM posts p
        JOIN evaluations e ON p.id = e.post_id
        WHERE e.is_signal = 1
        ORDER BY p.timestamp DESC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    conn.close()

    # Convert to unified format
    items = []
    for row in rows:
        items.append({
            "id": row["id"],
            "source_type": "reddit",
            "title": row["title"],
            "content": row["content"],
            "url": row["url"],
            "reasoning": row["reasoning"],
            "metadata": json.dumps({"subreddit": row["subreddit"]})
        })

    return items


def save_draft_to_db(content_id: int, platform: str, draft_content: str, quality_score: float) -> int:
    """Save approved draft to database with quality score."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO drafts (post_id, platform, draft_content, generated_at, published)
        VALUES (?, ?, ?, ?, 0)
    """, (content_id, platform, draft_content, int(utc_now().timestamp())))

    draft_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return draft_id


def export_draft_to_file(
    draft_id: int,
    platform: str,
    draft_content: str,
    quality_result: QualityGateResult
) -> Path:
    """Export approved draft to file with quality metadata."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = utc_now().strftime("%Y%m%d_%H%M%S")
    status = "approved" if quality_result.passed else "escalated"
    filename = f"{platform}_{timestamp}_{draft_id}_{status}.md"
    filepath = OUTPUT_DIR / filename

    # Add quality metadata header
    header = f"""---
platform: {platform}
draft_id: {draft_id}
quality_score: {quality_result.final_score}/10
status: {"APPROVED" if quality_result.passed else "ESCALATED"}
iterations: {quality_result.iterations_used}/{quality_result.max_iterations}
generated: {quality_result.timestamp}
---

"""

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(header + draft_content)

    return filepath


def generate_with_quality_gate(
    item: Dict[str, Any],
    platform: str,
    writer: WriterAgent,
    gate: QualityGate,
    verbose: bool = True
) -> tuple[str, QualityGateResult]:
    """
    Generate draft with quality gate review loop.

    Returns:
        Tuple of (final_content, quality_result)
    """
    source_type = item.get("source_type", "external")

    # Prepare signal data for writer
    signal_data = {
        "title": item.get("title", ""),
        "content": item.get("content", ""),
        "url": item.get("url", ""),
        "reasoning": item.get("reasoning", ""),
        "author": item.get("author", ""),
        "metadata": item.get("metadata", "{}")
    }

    # Determine voice type
    voice_type = "external" if source_type in ["reddit", "gmail", "rss"] else "internal"

    if verbose:
        print(f"\n  ✍️  Generating initial {platform} draft...")

    # Generate initial draft
    if platform == "linkedin":
        initial_draft = writer.write_linkedin_post(signal_data, source_type=voice_type)
    else:
        initial_draft = writer.write_medium_article(signal_data, source_type=voice_type)

    if verbose:
        print(f"     Initial draft: {len(initial_draft)} chars")
        print(f"\n  🔍 Running quality gate...")

    # Run through quality gate
    result = gate.process(
        content=initial_draft,
        platform=platform,
        source_context=item.get("reasoning", "")
    )

    return result.final_content, result


def main():
    parser = argparse.ArgumentParser(
        description='Generate Quality-Gated Content Drafts (v2)'
    )
    parser.add_argument('--platform', choices=['linkedin', 'medium', 'both'],
                       default='both', help='Platform to generate for')
    parser.add_argument('--limit', type=int, default=5,
                       help='Number of items to process')
    parser.add_argument('--unified', action='store_true',
                       help='Use unified content_items table')
    parser.add_argument('--source', choices=['reddit', 'gmail', 'github', 'rss'],
                       help='Filter by source type (requires --unified)')
    parser.add_argument('--max-iterations', type=int, default=3,
                       help='Max quality gate iterations')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress verbose output')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Quality-Gated Draft Generation (v2)")
    print(f"{'='*60}")
    print(f"Platform: {args.platform}")
    print(f"Max Items: {args.limit}")
    print(f"Max Iterations: {args.max_iterations}")
    print(f"{'='*60}\n")

    # Initialize agents
    print("🤖 Initializing agents...")
    writer = WriterAgent()
    gate = QualityGate(
        max_iterations=args.max_iterations,
        verbose=not args.quiet
    )
    print("   ✓ Writer Agent ready")
    print("   ✓ Quality Gate ready (with Adversarial Panel)")

    # Fetch signal content
    if args.unified or args.source:
        print(f"\n📥 Fetching signal content (unified)...")
        items = get_signal_content_unified(args.limit, args.source)
    else:
        print(f"\n📥 Fetching signal content (legacy)...")
        items = get_signal_posts_legacy(args.limit)

    print(f"   Found {len(items)} signal items\n")

    if not items:
        print("⚠️  No signal content found. Run evaluation first.\n")
        return

    # Determine platforms
    platforms = ['linkedin', 'medium'] if args.platform == 'both' else [args.platform]

    # Statistics
    stats = {
        "total": 0,
        "approved": 0,
        "escalated": 0,
        "total_iterations": 0,
    }

    # Process each item
    for i, item in enumerate(items, 1):
        source = item.get("source_type", "unknown")
        title = item.get("title", "Untitled")
        title_short = title[:40] + "..." if len(title) > 40 else title

        print(f"\n{'─'*60}")
        print(f"[{i}/{len(items)}] [{source}] {title_short}")
        print(f"{'─'*60}")

        for platform in platforms:
            stats["total"] += 1

            # Generate with quality gate
            final_content, result = generate_with_quality_gate(
                item=item,
                platform=platform,
                writer=writer,
                gate=gate,
                verbose=not args.quiet
            )

            stats["total_iterations"] += result.iterations_used

            if result.passed:
                stats["approved"] += 1
                status_icon = "✅"
            else:
                stats["escalated"] += 1
                status_icon = "⚠️"

            # Save to database
            draft_id = save_draft_to_db(
                content_id=item["id"],
                platform=platform,
                draft_content=final_content,
                quality_score=result.final_score
            )

            # Export to file
            filepath = export_draft_to_file(
                draft_id=draft_id,
                platform=platform,
                draft_content=final_content,
                quality_result=result
            )

            print(f"\n  {status_icon} {platform.upper()}: Score {result.final_score}/10 ({result.iterations_used} iterations)")
            print(f"     📄 {filepath.name}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total Drafts: {stats['total']}")
    print(f"Approved: {stats['approved']} ({stats['approved']/stats['total']*100:.0f}%)" if stats['total'] > 0 else "Approved: 0")
    print(f"Escalated: {stats['escalated']} (need human review)")
    print(f"Avg Iterations: {stats['total_iterations']/stats['total']:.1f}" if stats['total'] > 0 else "Avg Iterations: N/A")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
