#!/usr/bin/env python3
"""DEPRECATED: Use generate_medium_full.py for full pipeline or generate_drafts_v2.py for simple drafts.

This module is the original template-based draft generator. It does NOT use LLM agents
and produces only hardcoded template output. All functionality here is superseded by:

- generate_drafts_v2.py: Agent-based drafts with quality gate (simple mode)
- generate_medium_full.py: Full article pipeline with research, verification, and visuals

This module is retained for backward compatibility. No new features should be added here.
Scheduled for removal in next major version.
"""

import warnings
warnings.warn(
    "generate_drafts is deprecated. Use generate_medium_full or generate_drafts_v2.",
    DeprecationWarning,
    stacklevel=2,
)

import json
import sqlite3
import argparse
import time
from pathlib import Path
from execution.utils.datetime_utils import utc_now
from typing import Any, Dict, List, Optional

# Paths
DB_PATH = Path(__file__).parent.parent / "reddit_content.db"
DIRECTIVE_PATH = Path(__file__).parent.parent / "directives" / "market_strategy.md"
OUTPUT_DIR = Path(__file__).parent.parent / ".tmp" / "drafts"


def get_db_connection():
    """Get database connection."""
    return sqlite3.connect(DB_PATH)


def load_market_strategy():
    """Load the market strategy directive."""
    if not DIRECTIVE_PATH.exists():
        return ""

    with open(DIRECTIVE_PATH, 'r') as f:
        return f.read()


def get_signal_content_unified(
    limit: int = 10,
    source_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Fetch Signal content from the unified content_items table.

    Returns:
        List of content dicts with all necessary fields
    """
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


def get_signal_posts(limit=10):
    """
    Fetch posts marked as Signal that don't have drafts yet.
    
    Returns:
        List of (post_id, subreddit, title, content, url, reasoning) tuples
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT p.id, p.subreddit, p.title, p.content, p.url, e.reasoning
        FROM posts p
        JOIN evaluations e ON p.id = e.post_id
        WHERE e.is_signal = 1
        ORDER BY p.timestamp DESC
        LIMIT ?
    """, (limit,))
    
    posts = cursor.fetchall()
    conn.close()
    
    return posts


def generate_linkedin_draft(post_id, subreddit, title, content, url, reasoning, market_strategy):
    """
    Generate a LinkedIn post draft.
    
    LinkedIn best practices:
    - Hook in first line
    - 1-3 paragraphs max
    - Call to action
    - Conversational tone but professional
    """
    
    # TODO: Replace with actual LLM API call
    # For now, use a template
    
    draft = f"""🚀 Interesting insight from r/{subreddit}:

{title}

Key takeaway: {reasoning}

This aligns with what I'm seeing in production AI systems - the gap between research and real-world deployment is still significant.

What's been your experience with this?

Source: {url}

#AIEngineering #MachineLearning #LLMOps #ProductionAI"""
    
    return draft


def generate_medium_draft(post_id, subreddit, title, content, url, reasoning, market_strategy):
    """
    Generate a Medium article draft.
    
    Medium best practices:
    - Long-form (800-2000 words)
    - Clear structure with headers
    - Deep dive into technical details
    - Personal insights
    """
    
    # TODO: Replace with actual LLM API call
    # For now, use a template
    
    draft = f"""# {title}

*Insights from the AI Engineering Community*

## Context

I've been following r/{subreddit} closely as part of my research into production AI engineering challenges. This post caught my attention because {reasoning.lower()}

## What the Community is Saying

{content[:500]}...

[Read full discussion on Reddit]({url})

## My Take

This highlights a critical challenge in AI engineering today: the gap between what works in research and what's reliable in production.

## Practical Implications

1. **For practitioners**: ...
2. **For teams**: ...
3. **For the industry**: ...

## Conclusion

As AI engineering matures, we need more conversations like this - focused on real production challenges, not just what's possible in a lab.

What's been your experience? Drop a comment below.

---

*This article is part of my ongoing research into AI engineering practices. Follow for more insights from the field.*
"""
    
    return draft


def save_draft_to_db(post_id, platform, draft_content):
    """Save draft to database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO drafts (post_id, platform, draft_content, generated_at, published)
        VALUES (?, ?, ?, ?, 0)
    """, (post_id, platform, draft_content, int(time.time())))
    
    draft_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return draft_id


def export_draft_to_file(draft_id, platform, draft_content):
    """Export draft to .tmp/drafts/ directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = utc_now().strftime("%Y%m%d_%H%M%S")
    filename = f"{platform}_{timestamp}_{draft_id}.txt"
    filepath = OUTPUT_DIR / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(draft_content)
    
    return filepath


def generate_unified_linkedin_draft(item: Dict[str, Any], market_strategy: str) -> str:
    """Generate LinkedIn draft from unified content item."""
    source_type = item["source_type"]
    title = item["title"]
    content = item.get("content", "")
    url = item.get("url", "")
    reasoning = item.get("reasoning", "")
    author = item.get("author", "")

    # Source-specific attribution
    if source_type == "reddit":
        metadata = json.loads(item.get("metadata", "{}") or "{}")
        source_label = f"r/{metadata.get('subreddit', 'unknown')}"
    elif source_type == "gmail":
        source_label = f"Newsletter: {author}"
    else:
        source_label = source_type.capitalize()

    draft = f"""Interesting insight from {source_label}:

{title}

Key takeaway: {reasoning}

This aligns with what I'm seeing in production AI systems - the gap between research and real-world deployment is still significant.

What's been your experience with this?

{f'Source: {url}' if url else ''}

#AIEngineering #MachineLearning #LLMOps #ProductionAI"""

    return draft


def generate_unified_medium_draft(item: Dict[str, Any], market_strategy: str) -> str:
    """Generate Medium draft from unified content item."""
    source_type = item["source_type"]
    title = item["title"]
    content = item.get("content", "")[:500]
    url = item.get("url", "")
    reasoning = item.get("reasoning", "")
    author = item.get("author", "")

    # Source-specific attribution
    if source_type == "reddit":
        metadata = json.loads(item.get("metadata", "{}") or "{}")
        source_label = f"r/{metadata.get('subreddit', 'unknown')}"
        community_type = "AI Engineering community on Reddit"
    elif source_type == "gmail":
        source_label = author
        community_type = f"the {author} newsletter"
    else:
        source_label = source_type.capitalize()
        community_type = f"the {source_type} community"

    draft = f"""# {title}

*Insights from {community_type}*

## Context

I've been following {source_label} closely as part of my research into production AI engineering challenges. This caught my attention because {reasoning.lower() if reasoning else 'it addresses a real-world challenge.'}

## What's Being Discussed

{content}...

{f'[Read more]({url})' if url else ''}

## My Take

This highlights a critical challenge in AI engineering today: the gap between what works in research and what's reliable in production.

## Practical Implications

1. **For practitioners**: ...
2. **For teams**: ...
3. **For the industry**: ...

## Conclusion

As AI engineering matures, we need more conversations like this - focused on real production challenges, not just what's possible in a lab.

What's been your experience? Drop a comment below.

---

*This article is part of my ongoing research into AI engineering practices. Follow for more insights from the field.*
"""

    return draft


def save_unified_draft_to_db(content_id: int, platform: str, draft_content: str) -> int:
    """Save draft to database (linked to content_items)."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Note: Using post_id column for backward compatibility
    # In a full migration, we'd add content_id column to drafts table
    cursor.execute("""
        INSERT INTO drafts (post_id, platform, draft_content, generated_at, published)
        VALUES (?, ?, ?, ?, 0)
    """, (content_id, platform, draft_content, int(time.time())))

    draft_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return draft_id


def main():
    parser = argparse.ArgumentParser(description='Generate content drafts from Signal posts')
    parser.add_argument('--platform', choices=['linkedin', 'medium', 'both'], default='both',
                       help='Platform to generate for (default: both)')
    parser.add_argument('--limit', type=int, default=10,
                       help='Number of posts to process (default: 10)')
    parser.add_argument('--unified', action='store_true',
                       help='Use unified content_items table instead of legacy posts')
    parser.add_argument('--source', choices=['reddit', 'gmail', 'github', 'rss', 'manual'],
                       help='Filter by source type (requires --unified)')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Generating Content Drafts")
    print(f"{'='*60}\n")

    # Load market strategy
    print("Loading market strategy directive...")
    market_strategy = load_market_strategy()
    if market_strategy:
        print(f"  [+] Loaded strategy")
    else:
        print(f"  [ ] No strategy file, using defaults")

    # Fetch signal content
    if args.unified or args.source:
        print(f"\nFetching Signal content from unified table...")
        if args.source:
            print(f"  Filtering: source_type = {args.source}")
        items = get_signal_content_unified(args.limit, args.source)
        print(f"  [+] Found {len(items)} Signal items\n")

        if not items:
            print("No Signal content found. Run evaluate_content.py first.\n")
            return

        # Generate drafts from unified content
        platforms = ['linkedin', 'medium'] if args.platform == 'both' else [args.platform]

        for i, item in enumerate(items, 1):
            source = item["source_type"]
            title = item["title"][:45] + "..." if len(item["title"]) > 45 else item["title"]
            print(f"[{i}/{len(items)}] [{source}] {title}")

            for platform in platforms:
                if platform == 'linkedin':
                    draft = generate_unified_linkedin_draft(item, market_strategy)
                else:
                    draft = generate_unified_medium_draft(item, market_strategy)

                # Save to DB
                draft_id = save_unified_draft_to_db(item["id"], platform, draft)

                # Export to file
                filepath = export_draft_to_file(draft_id, platform, draft)

                print(f"    [+] {platform.capitalize()}: {filepath.name}")

            print()

        total_drafts = len(items) * len(platforms)

    else:
        # Legacy mode: use posts/evaluations tables
        print(f"\nFetching up to {args.limit} Signal posts (legacy mode)...")
        posts = get_signal_posts(args.limit)
        print(f"  [+] Found {len(posts)} Signal posts\n")

        if not posts:
            print("No Signal posts found. Run evaluate_posts.py first.\n")
            return

        # Generate drafts
        platforms = ['linkedin', 'medium'] if args.platform == 'both' else [args.platform]

        for i, (post_id, subreddit, title, content, url, reasoning) in enumerate(posts, 1):
            print(f"[{i}/{len(posts)}] r/{subreddit}: {title[:50]}...")

            for platform in platforms:
                if platform == 'linkedin':
                    draft = generate_linkedin_draft(
                        post_id, subreddit, title, content, url, reasoning, market_strategy
                    )
                else:
                    draft = generate_medium_draft(
                        post_id, subreddit, title, content, url, reasoning, market_strategy
                    )

                # Save to DB
                draft_id = save_draft_to_db(post_id, platform, draft)

                # Export to file
                filepath = export_draft_to_file(draft_id, platform, draft)

                print(f"    [+] {platform.capitalize()}: {filepath.name}")

            print()

        total_drafts = len(posts) * len(platforms)

    print(f"{'='*60}")
    print(f"Draft Generation Complete")
    print(f"  Generated {total_drafts} drafts")
    print(f"  Location: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
