#!/usr/bin/env python3
"""
Generate LinkedIn/Medium drafts from Signal posts.

This script:
1. Fetches posts marked as "Signal" from the database
2. Uses an LLM to generate content drafts based on market strategy themes
3. Saves drafts to the database and exports to .tmp/drafts/

Usage:
    python generate_drafts.py --platform linkedin --limit 10
    python generate_drafts.py --platform medium --limit 5
"""

import sqlite3
import argparse
import time
from pathlib import Path
from datetime import datetime

# Paths
DB_PATH = Path(__file__).parent.parent / "reddit_content.db"
DIRECTIVE_PATH = Path(__file__).parent.parent / "directives" / "market_strategy.md"
OUTPUT_DIR = Path(__file__).parent.parent / ".tmp" / "drafts"


def load_market_strategy():
    """Load the market strategy directive."""
    if not DIRECTIVE_PATH.exists():
        raise FileNotFoundError(f"Market strategy not found: {DIRECTIVE_PATH}")
    
    with open(DIRECTIVE_PATH, 'r') as f:
        return f.read()


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
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{platform}_{timestamp}_{draft_id}.txt"
    filepath = OUTPUT_DIR / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(draft_content)
    
    return filepath


def main():
    parser = argparse.ArgumentParser(description='Generate content drafts from Signal posts')
    parser.add_argument('--platform', choices=['linkedin', 'medium', 'both'], default='both',
                       help='Platform to generate for (default: both)')
    parser.add_argument('--limit', type=int, default=10,
                       help='Number of posts to process (default: 10)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Generating Content Drafts")
    print(f"{'='*60}\n")
    
    # Load market strategy
    print("Loading market strategy directive...")
    market_strategy = load_market_strategy()
    print(f"✓ Loaded strategy\n")
    
    # Fetch signal posts
    print(f"Fetching up to {args.limit} Signal posts...")
    posts = get_signal_posts(args.limit)
    print(f"✓ Found {len(posts)} Signal posts\n")
    
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
            
            print(f"  ✓ {platform.capitalize()}: {filepath.name}")
        
        print()
    
    print(f"{'='*60}")
    print(f"✓ Draft Generation Complete")
    print(f"  Generated {len(posts) * len(platforms)} drafts")
    print(f"  Location: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
