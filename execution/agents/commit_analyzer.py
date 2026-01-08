"""
Commit Analysis Agent - Extracts actionable themes from GitHub commit activity.

Analyzes commit messages and file changes to identify:
1. Emerging patterns in AI/ML repos
2. Technical decisions worth discussing
3. Breaking changes that affect practitioners
"""

import json
import time
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional
from .base_agent import BaseAgent


# Database path
DB_PATH = Path(__file__).parent.parent.parent / "reddit_content.db"

# Repos optimized for AI Engineer content
AI_ENGINEERING_REPOS = [
    "microsoft/semantic-kernel",
    "langchain-ai/langchain",
    "run-llama/llama_index",
    "vllm-project/vllm",
    "huggingface/transformers",
    "openai/openai-python",
]


class CommitAnalysisAgent(BaseAgent):
    """
    Analyzes GitHub commits to extract content-worthy themes.

    Workflow:
        1. Receive batch of commits from database
        2. Group by pattern/theme
        3. Score themes for content potential
        4. Return top theme with suggested angle
    """

    def __init__(self):
        super().__init__(
            role="Open Source Intelligence Analyst",
            persona="""You are an expert at reading GitHub commit history to spot trends.

Your client is an AI Engineer building thought leadership. You help them:
- Identify what's ACTUALLY changing in production AI frameworks
- Spot patterns that practitioners care about (not just announcements)
- Find the "story behind the commits" - why are these changes happening?

You focus on:
1. TECHNICAL DEPTH: Changes that reveal architectural decisions
2. PRACTITIONER RELEVANCE: What would affect someone deploying these tools?
3. TIMING: Is this addressing a known pain point? Is it ahead of the curve?
4. UNIQUENESS: Can we offer insight others won't have?""",
            model="llama-3.3-70b-versatile"
        )

    def fetch_commits_from_db(self, limit: int = 100) -> List[Dict]:
        """
        Fetch recent commits from database.

        Args:
            limit: Maximum commits to fetch

        Returns:
            List of commit dicts
        """
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT repo_owner, repo_name, commit_message, files_changed,
                   additions, deletions, committed_at
            FROM github_commits
            ORDER BY committed_at DESC
            LIMIT ?
        """, (limit,))

        commits = [
            {
                "repo_owner": row[0],
                "repo_name": row[1],
                "commit_message": row[2],
                "files_changed": row[3],
                "additions": row[4],
                "deletions": row[5],
                "committed_at": row[6]
            }
            for row in cursor.fetchall()
        ]
        conn.close()

        return commits

    def analyze_commits(self, commits: List[Dict]) -> List[Dict]:
        """
        Analyze a batch of commits and extract themes.

        Args:
            commits: List of commit dicts with keys:
                     repo_owner, repo_name, commit_message, files_changed

        Returns:
            List of theme dicts with title, description, relevance_score, angle
        """
        if not commits:
            return []

        # Format commits for LLM
        commits_text = "\n".join([
            f"[{c['repo_owner']}/{c['repo_name']}] {c['commit_message'][:150]}"
            f" | Files: {str(c.get('files_changed', 'N/A'))[:80]}"
            for c in commits[:30]  # Limit for context window
        ])

        prompt = f"""Analyze these recent commits from AI/ML repositories and identify THEMES worth writing about.

COMMITS:
{commits_text}

Find 3-5 distinct themes. For each theme:
1. What pattern do you see across commits?
2. Why would an AI practitioner care?
3. What's the unique angle for content?

Output VALID JSON:
{{
    "themes": [
        {{
            "title": "Theme title (5-10 words)",
            "description": "What's happening (2-3 sentences)",
            "relevance_score": 0.0-1.0,
            "affected_repos": ["repo1", "repo2"],
            "content_angle": "Specific article angle",
            "practitioner_impact": "Why this matters for builders"
        }}
    ]
}}

Focus on TECHNICAL themes, not announcements. What would a Staff Engineer find interesting?"""

        response = self.call_llm(prompt, temperature=0.3)

        try:
            # Parse JSON response
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            result = json.loads(response.strip())
            return result.get("themes", [])

        except (json.JSONDecodeError, KeyError) as e:
            print(f"  [!] Failed to parse themes: {e}")
            return []

    def select_best_theme(self, themes: List[Dict]) -> Optional[Dict]:
        """
        Select the single best theme for content generation.

        Args:
            themes: List of theme dicts from analyze_commits()

        Returns:
            Best theme dict or None
        """
        if not themes:
            return None

        # Sort by relevance score
        sorted_themes = sorted(
            themes,
            key=lambda t: t.get("relevance_score", 0),
            reverse=True
        )

        # Return highest scoring theme
        return sorted_themes[0]

    def research_github_topics(self, commits: List[Dict]) -> Dict:
        """
        Full pipeline: analyze commits and return best topic.

        Args:
            commits: Raw commits from database

        Returns:
            Topic dict compatible with existing pipeline
        """
        print("\n" + "=" * 60)
        print("[Commit Analysis Agent] Analyzing GitHub activity...")
        print("=" * 60)

        # Analyze commits
        print(f"\n  [Commit Analysis] Processing {len(commits)} commits...")
        themes = self.analyze_commits(commits)

        if not themes:
            print("  [!] No themes extracted, using fallback")
            return {
                "title": "What the Latest AI Framework Commits Reveal About Production Patterns",
                "source": "github",
                "reasoning": "Fallback topic - no themes extracted from commits"
            }

        # Select best theme
        best = self.select_best_theme(themes)

        print(f"\n  [Commit Analysis] SELECTED THEME:")
        print(f"    Title: {best.get('title', 'N/A')}")
        print(f"    Score: {best.get('relevance_score', 0):.2f}")
        print(f"    Angle: {best.get('content_angle', 'N/A')}")

        return {
            "title": best.get("content_angle") or best.get("title"),
            "source": "github",
            "reasoning": best.get("description", ""),
            "angle": best.get("content_angle", ""),
            "recruiter_appeal": best.get("practitioner_impact", ""),
            "theme_data": best
        }

    def research_github_topics_from_db(self, limit: int = 100) -> Dict:
        """
        Convenience method: fetch commits from DB and analyze.

        Args:
            limit: Max commits to analyze

        Returns:
            Topic dict for content pipeline
        """
        commits = self.fetch_commits_from_db(limit)

        if not commits:
            print("  [!] No commits in database. Run fetch_github.py first.")
            return {
                "title": "Building Production-Ready AI Agents: Lessons from Open Source",
                "source": "github",
                "reasoning": "No commits found - run fetch_github.py to populate database"
            }

        return self.research_github_topics(commits)


def analyze_github_for_content() -> Dict:
    """
    Convenience function for one-click GitHub topic selection.

    Returns:
        Topic dict for content pipeline
    """
    agent = CommitAnalysisAgent()
    return agent.research_github_topics_from_db()


if __name__ == "__main__":
    # Test the agent
    print("Testing CommitAnalysisAgent...")
    result = analyze_github_for_content()
    print(f"\n\nFINAL TOPIC: {result['title']}")
    print(f"REASONING: {result.get('reasoning', 'N/A')}")
