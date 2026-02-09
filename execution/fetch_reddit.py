#!/usr/bin/env python3
"""
Fetch Reddit posts from RSS feeds and store in database.

Usage:
    python fetch_reddit.py --subreddits LocalLLaMA LLMDevs LanguageTechnology
    python fetch_reddit.py --all  # Fetches from all S+ and S tier subreddits
"""

import feedparser
import requests
import sqlite3
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    wait_random,
    retry_if_exception_type,
)

# Database path
DB_PATH = Path(__file__).parent.parent / "reddit_content.db"

# S+ and S tier subreddits from market strategy
TIER_SP = ["LocalLLaMA", "LLMDevs", "LanguageTechnology"]
TIER_S = ["MachineLearning", "deeplearning", "mlops", "learnmachinelearning"]


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=30) + wait_random(0, 2),
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    reraise=True,
)
def _fetch_rss_request(url, headers):
    """HTTP request for subreddit RSS, retried on transient errors."""
    return requests.get(url, headers=headers, timeout=10)


def fetch_subreddit_rss(subreddit_name, max_posts=100):
    """
    Fetch posts from a subreddit's RSS feed.

    Args:
        subreddit_name: Name of the subreddit (without r/)
        max_posts: Maximum number of posts to fetch

    Returns:
        List of post dictionaries
    """
    # Reddit requires a custom User-Agent to avoid blocking
    # Format: appname/version (by /u/username)
    user_agent = "RedditNewsBot/1.0 (by /u/RedditNewsBot)"
    headers = {'User-Agent': user_agent}

    rss_url = f"https://www.reddit.com/r/{subreddit_name}.rss"
    print(f"Fetching {rss_url}...")

    try:
        response = _fetch_rss_request(rss_url, headers)
        print(f"  Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"  ✗ Failed to fetch: {response.status_code}")
            return []
            
        feed = feedparser.parse(response.content)
    except Exception as e:
        print(f"  ✗ Request failed: {e}")
        return []

    print(f"  Entries found: {len(feed.entries)}")
    
    if feed.bozo:
        print(f"  ✗ Error parsing feed: {feed.bozo_exception}")
        return []
    
    posts = []
    for entry in feed.entries[:max_posts]:
        # Parse timestamp
        published_time = entry.get('published_parsed', time.gmtime())
        timestamp = int(time.mktime(published_time))
        
        # Extract content (Reddit RSS includes HTML summary)
        content = entry.get('summary', '')
        
        post = {
            'subreddit': subreddit_name,
            'title': entry.get('title', ''),
            'url': entry.get('link', ''),
            'author': entry.get('author', 'unknown'),
            'content': content,
            'timestamp': timestamp,
            'upvotes': 0,  # RSS doesn't include upvotes
            'num_comments': 0,  # RSS doesn't include comment count
            'retrieved_at': int(time.time()),
            'source_type': 'external'  # Voice transformation: use observer voice
        }
        posts.append(post)
    
    print(f"  ✓ Fetched {len(posts)} posts")
    return posts


def filter_recent_posts(posts, hours=72):
    """Filter posts to only include those from last N hours."""
    cutoff = int(time.time()) - (hours * 3600)
    recent = [p for p in posts if p['timestamp'] >= cutoff]
    print(f"  Filtered to {len(recent)} posts from last {hours} hours")
    return recent


def insert_posts_to_db(posts):
    """
    Insert posts into database, ignoring duplicates.
    
    Returns:
        Number of new posts inserted
    """
    if not posts:
        return 0
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    inserted = 0
    for post in posts:
        try:
            cursor.execute("""
                INSERT INTO posts (subreddit, title, url, author, content, 
                                 timestamp, upvotes, num_comments, retrieved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                post['subreddit'], post['title'], post['url'], post['author'],
                post['content'], post['timestamp'], post['upvotes'],
                post['num_comments'], post['retrieved_at']
            ))
            inserted += 1
        except sqlite3.IntegrityError:
            # Duplicate URL, skip
            pass
    
    conn.commit()
    conn.close()
    
    print(f"  ✓ Inserted {inserted} new posts (skipped {len(posts) - inserted} duplicates)")
    return inserted


def main():
    parser = argparse.ArgumentParser(description='Fetch Reddit posts from RSS feeds')
    parser.add_argument('--subreddits', nargs='+', help='Specific subreddits to fetch')
    parser.add_argument('--all', action='store_true', help='Fetch from all S+ and S tier subreddits')
    parser.add_argument('--max-posts', type=int, default=100, help='Max posts per subreddit (default: 100)')
    parser.add_argument('--hours', type=int, default=72, help='Only fetch posts from last N hours (default: 72)')
    
    args = parser.parse_args()
    
    # Determine which subreddits to fetch
    if args.all:
        subreddits = TIER_SP + TIER_S
    elif args.subreddits:
        subreddits = args.subreddits
    else:
        # Default to S+ tier
        subreddits = TIER_SP
    
    print(f"\n{'='*60}")
    print(f"Fetching from {len(subreddits)} subreddits...")
    print(f"{'='*60}\n")
    
    total_inserted = 0
    for subreddit in subreddits:
        print(f"\n[{subreddit}]")
        posts = fetch_subreddit_rss(subreddit, args.max_posts)
        recent_posts = filter_recent_posts(posts, args.hours)
        inserted = insert_posts_to_db(recent_posts)
        total_inserted += inserted
    
    print(f"\n{'='*60}")
    print(f"✓ Total: {total_inserted} new posts added to database")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
