#!/usr/bin/env python3
"""
Fetch GitHub commits from configured repositories and store in database.

Usage:
    python fetch_github.py --repos microsoft/semantic-kernel langchain-ai/langchain
    python fetch_github.py --config  # Uses repos from GITHUB_REPOS env var
    python fetch_github.py --all     # Fetches from all default repos
"""

import requests
import sqlite3
import time
import argparse
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    wait_random,
    retry_if_exception_type,
)

# Load environment variables
load_dotenv()

# Database path
DB_PATH = Path(__file__).parent.parent / "reddit_content.db"

# Default repos (AI/ML focused for content generation)
DEFAULT_REPOS = [
    "microsoft/semantic-kernel",
    "langchain-ai/langchain",
    "run-llama/llama_index",
    "vllm-project/vllm",
    "huggingface/transformers",
]


def get_github_token():
    """Get GitHub token from environment."""
    return os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_PAT")


def get_configured_repos():
    """Get repos from environment variable (comma-separated)."""
    repos_str = os.getenv("GITHUB_REPOS", "")
    if repos_str:
        return [r.strip() for r in repos_str.split(",") if r.strip()]
    return DEFAULT_REPOS


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=30) + wait_random(0, 2),
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    reraise=True,
)
def _fetch_commit_details_request(url, headers):
    """HTTP request for commit details, retried on transient errors."""
    return requests.get(url, headers=headers, timeout=15)


def fetch_commit_details(owner, repo, sha, headers):
    """Fetch detailed commit info including files changed."""
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"

    try:
        response = _fetch_commit_details_request(url, headers)
        if response.status_code == 200:
            data = response.json()
            files = [f["filename"] for f in data.get("files", [])]
            return {
                "files": files[:20],  # Limit to 20 files
                "additions": data.get("stats", {}).get("additions", 0),
                "deletions": data.get("stats", {}).get("deletions", 0)
            }
    except Exception as e:
        print(f"    Warning: Could not fetch commit details: {e}")

    return {"files": [], "additions": 0, "deletions": 0}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=30) + wait_random(0, 2),
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    reraise=True,
)
def _fetch_repo_commits_request(url, headers, params):
    """HTTP request for repo commits list, retried on transient errors."""
    return requests.get(url, headers=headers, params=params, timeout=30)


def fetch_repo_commits(owner, repo, max_commits=50, since_hours=168):
    """
    Fetch recent commits from a GitHub repository.

    Args:
        owner: Repository owner (e.g., "microsoft")
        repo: Repository name (e.g., "semantic-kernel")
        max_commits: Maximum commits to fetch
        since_hours: Only fetch commits from last N hours (default 168 = 7 days)

    Returns:
        List of commit dictionaries
    """
    token = get_github_token()
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Calculate since date
    since_date = (datetime.utcnow() - timedelta(hours=since_hours)).isoformat() + "Z"

    url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    params = {
        "per_page": min(max_commits, 100),
        "since": since_date
    }

    print(f"  Fetching {url}...")

    try:
        response = _fetch_repo_commits_request(url, headers, params)
        print(f"  Status: {response.status_code}")

        if response.status_code == 403:
            remaining = response.headers.get('X-RateLimit-Remaining', 'unknown')
            print(f"  X Rate limited (remaining: {remaining}). Add GITHUB_TOKEN to .env for higher limits.")
            return []
        elif response.status_code == 404:
            print(f"  X Repository not found: {owner}/{repo}")
            return []
        elif response.status_code != 200:
            print(f"  X Failed: {response.status_code}")
            return []

        commits_data = response.json()

    except Exception as e:
        print(f"  X Request failed: {e}")
        return []

    if not commits_data:
        print(f"  No commits found in last {since_hours} hours")
        return []

    commits = []
    for i, commit in enumerate(commits_data[:max_commits]):
        # Get commit details (files changed)
        sha = commit["sha"]

        # Only fetch details for first 10 commits to avoid rate limits
        if i < 10:
            commit_detail = fetch_commit_details(owner, repo, sha, headers)
            time.sleep(0.2)  # Rate limiting between detail requests
        else:
            commit_detail = {"files": [], "additions": 0, "deletions": 0}

        commit_info = {
            "repo_owner": owner,
            "repo_name": repo,
            "commit_sha": sha,
            "author_name": commit["commit"]["author"]["name"],
            "author_email": commit["commit"]["author"]["email"],
            "commit_message": commit["commit"]["message"],
            "files_changed": json.dumps(commit_detail.get("files", [])),
            "additions": commit_detail.get("additions", 0),
            "deletions": commit_detail.get("deletions", 0),
            "committed_at": int(datetime.fromisoformat(
                commit["commit"]["author"]["date"].replace("Z", "+00:00")
            ).timestamp()),
            "retrieved_at": int(time.time()),
            "source_type": "internal"  # Voice transformation: use ownership voice
        }
        commits.append(commit_info)

    print(f"  + Fetched {len(commits)} commits")
    return commits


def insert_commits_to_db(commits):
    """Insert commits into database, ignoring duplicates."""
    if not commits:
        return 0

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    inserted = 0
    for commit in commits:
        try:
            cursor.execute("""
                INSERT INTO github_commits
                (repo_owner, repo_name, commit_sha, author_name, author_email,
                 commit_message, files_changed, additions, deletions,
                 committed_at, retrieved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                commit["repo_owner"], commit["repo_name"], commit["commit_sha"],
                commit["author_name"], commit["author_email"], commit["commit_message"],
                commit["files_changed"], commit["additions"], commit["deletions"],
                commit["committed_at"], commit["retrieved_at"]
            ))
            inserted += 1
        except sqlite3.IntegrityError:
            # Duplicate SHA, skip
            pass

    conn.commit()
    conn.close()

    print(f"  + Inserted {inserted} new commits (skipped {len(commits) - inserted} duplicates)")
    return inserted


def main():
    parser = argparse.ArgumentParser(description='Fetch GitHub commits from repositories')
    parser.add_argument('--repos', nargs='+', help='Specific repos (owner/name format)')
    parser.add_argument('--config', action='store_true', help='Use repos from GITHUB_REPOS env var')
    parser.add_argument('--all', action='store_true', help='Fetch from all default repos')
    parser.add_argument('--max-commits', type=int, default=50, help='Max commits per repo (default: 50)')
    parser.add_argument('--hours', type=int, default=168, help='Fetch commits from last N hours (default: 168 = 7 days)')

    args = parser.parse_args()

    # Check for token
    token = get_github_token()
    if token:
        print("GitHub token found - using authenticated requests (5000/hour limit)")
    else:
        print("No GITHUB_TOKEN found - using unauthenticated requests (60/hour limit)")
        print("Add GITHUB_TOKEN to .env for higher rate limits\n")

    # Determine which repos to fetch
    if args.repos:
        repos = [r.split("/") for r in args.repos if "/" in r]
    elif args.config:
        repo_list = get_configured_repos()
        repos = [r.split("/") for r in repo_list if "/" in r]
    elif args.all:
        repos = [r.split("/") for r in DEFAULT_REPOS]
    else:
        # Default: use first 3 default repos
        repos = [r.split("/") for r in DEFAULT_REPOS[:3]]

    if not repos:
        print("No valid repos specified. Use --repos owner/name or --all")
        return

    print(f"\n{'='*60}")
    print(f"Fetching from {len(repos)} repositories...")
    print(f"{'='*60}\n")

    total_inserted = 0
    for owner, repo in repos:
        print(f"\n[{owner}/{repo}]")
        commits = fetch_repo_commits(owner, repo, args.max_commits, args.hours)
        inserted = insert_commits_to_db(commits)
        total_inserted += inserted

        # Rate limiting between repos
        time.sleep(1)

    print(f"\n{'='*60}")
    print(f"+ Total: {total_inserted} new commits added to database")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
