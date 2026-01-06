#!/usr/bin/env python3
"""
Evaluate Reddit posts using an LLM to determine Signal vs Noise.

This script:
1. Fetches unevaluated posts from the database
2. Sends them to an LLM with the market strategy context
3. Stores the evaluation results back in the database

Usage:
    python evaluate_posts.py --limit 50
"""

import sqlite3
import argparse
import os
import time
from pathlib import Path
from datetime import datetime

# Database path
DB_PATH = Path(__file__).parent.parent / "reddit_content.db"
DIRECTIVE_PATH = Path(__file__).parent.parent / "directives" / "market_strategy.md"

# Placeholder for LLM API (will be configured via .env)
# For now, we'll use a simple heuristic + print statements


def load_market_strategy():
    """Load the market strategy directive."""
    if not DIRECTIVE_PATH.exists():
        raise FileNotFoundError(f"Market strategy not found: {DIRECTIVE_PATH}")
    
    with open(DIRECTIVE_PATH, 'r') as f:
        return f.read()


def get_unevaluated_posts(limit=50):
    """
    Fetch posts that haven't been evaluated yet.
    
    Returns:
        List of (post_id, subreddit, title, content) tuples
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT p.id, p.subreddit, p.title, p.content, p.url
        FROM posts p
        LEFT JOIN evaluations e ON p.id = e.post_id
        WHERE e.id IS NULL
        ORDER BY p.timestamp DESC
        LIMIT ?
    """, (limit,))
    
    posts = cursor.fetchall()
    conn.close()
    
    return posts


def evaluate_post_with_llm(post_id, subreddit, title, content, url, market_strategy):
    """
    Evaluate a single post using an LLM.
    
    Args:
        post_id: Database ID
        subreddit: Subreddit name
        title: Post title
        content: Post content
        url: Post URL
        market_strategy: The loaded market strategy text
    
    Returns:
        (is_signal: bool, reasoning: str)
    """
    
    # TODO: Replace this with actual LLM API call (OpenAI, Anthropic, etc.)
    # For now, use a simple heuristic based on keywords
    
    signal_keywords = [
        'production', 'deployment', 'llmops', 'mlops', 'quantization',
        'inference', 'fine-tuning', 'rag', 'benchmark', 'optimization',
        'architecture', 'framework', 'cost', 'scaling', 'performance',
        'llm', 'model', 'gpu', 'batch', 'latency', 'throughput'
    ]
    
    noise_keywords = [
        'chatgpt', 'openai ceo', 'elon musk', 'agi', 'sentient',
        'will ai', 'should we', 'unpopular opinion', 'hot take'
    ]
    
    text = (title + ' ' + content).lower()
    
    signal_count = sum(1 for kw in signal_keywords if kw in text)
    noise_count = sum(1 for kw in noise_keywords if kw in text)
    
    # Simple scoring
    if signal_count >= 2 and noise_count == 0:
        is_signal = True
        reasoning = f"Contains {signal_count} technical keywords indicating practical AI/ML engineering content."
    elif noise_count > 0:
        is_signal = False
        reasoning = f"Contains noise keywords ({noise_count} found). Likely speculative or non-technical."
    elif signal_count > 0:
        is_signal = True
        reasoning = f"Contains some technical content ({signal_count} keywords)."
    else:
        is_signal = False
        reasoning = "No clear technical or AI engineering signals detected."
    
    return is_signal, reasoning


def save_evaluation(post_id, is_signal, reasoning):
    """Save evaluation result to database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO evaluations (post_id, is_signal, reasoning, evaluated_at)
        VALUES (?, ?, ?, ?)
    """, (post_id, is_signal, reasoning, int(time.time())))
    
    conn.commit()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate Reddit posts for Signal vs Noise')
    parser.add_argument('--limit', type=int, default=50, help='Number of posts to evaluate (default: 50)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Evaluating Reddit Posts (Signal vs Noise)")
    print(f"{'='*60}\n")
    
    # Load market strategy
    print("Loading market strategy directive...")
    market_strategy = load_market_strategy()
    print(f"✓ Loaded strategy ({len(market_strategy)} chars)\n")
    
    # Fetch unevaluated posts
    print(f"Fetching up to {args.limit} unevaluated posts...")
    posts = get_unevaluated_posts(args.limit)
    print(f"✓ Found {len(posts)} posts to evaluate\n")
    
    if not posts:
        print("No posts to evaluate. Run fetch_reddit.py first.\n")
        return
    
    # Evaluate each post
    signal_count = 0
    noise_count = 0
    
    for i, (post_id, subreddit, title, content, url) in enumerate(posts, 1):
        print(f"[{i}/{len(posts)}] r/{subreddit}: {title[:60]}...")
        
        is_signal, reasoning = evaluate_post_with_llm(
            post_id, subreddit, title, content, url, market_strategy
        )
        
        save_evaluation(post_id, is_signal, reasoning)
        
        if is_signal:
            signal_count += 1
            print(f"  ✓ SIGNAL: {reasoning}\n")
        else:
            noise_count += 1
            print(f"  ✗ NOISE: {reasoning}\n")
    
    print(f"{'='*60}")
    print(f"✓ Evaluation Complete")
    print(f"  Signal: {signal_count} posts ({signal_count/len(posts)*100:.1f}%)")
    print(f"  Noise:  {noise_count} posts ({noise_count/len(posts)*100:.1f}%)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
