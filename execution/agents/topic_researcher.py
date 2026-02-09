"""
Topic Research Agent - Automated Topic Selection for AI Engineer Positioning

This agent automatically:
1. Fetches trending topics from relevant subreddits
2. Analyzes topics for AI Engineer positioning value
3. Selects the best topic for Medium/LinkedIn content

Target: Passive recruiting visibility for AI Engineers in 2026
"""

import time
import json
from typing import List, Dict, Optional
from .base_agent import BaseAgent, LLMError
from execution.config import config
from execution.utils.json_parser import extract_json_from_llm


# Subreddits optimized for AI Engineer content
AI_ENGINEER_SUBREDDITS = [
    # Tier S+ - Core AI/ML Engineering
    "LocalLLaMA",
    "LLMDevs",
    "MachineLearning",

    # Tier S - Technical Depth
    "deeplearning",
    "mlops",
    "LanguageTechnology",

    # Tier A - Engineering Perspective
    "artificial",
    "programming",
    "ExperiencedDevs",

    # Tier B - Industry Trends
    "technology",
    "startups",
]


class TopicResearchAgent(BaseAgent):
    """
    Automated topic researcher for AI Engineer content positioning.

    Workflow:
        1. Fetch trending topics from curated subreddits
        2. Score topics based on positioning criteria
        3. Return the optimal topic for content generation
    """

    def __init__(self):
        super().__init__(
            role="Topic Research Strategist",
            persona="""You are a strategic content advisor for an AI Engineer building thought leadership.

Your client's goals:
- Position as a credible AI/ML Engineer on Medium and LinkedIn
- Create content that attracts recruiters passively (2026 recruiting is silent/passive)
- Build visibility through niche, consistent, high-value technical content
- Stand out as someone who BUILDS AI systems, not just talks about them

You evaluate topics based on:
1. RELEVANCE: Does it showcase AI engineering skills? (LLMs, MLOps, agents, deployment, etc.)
2. TRENDING: Is this topic hot right now? Will it get engagement?
3. RECRUITER APPEAL: Would a hiring manager find this impressive?
4. UNIQUE ANGLE: Can we offer a practitioner's perspective vs. generic takes?
5. LINKEDIN FIT: Will this work as a professional technical post?""",
            model=config.models.DEFAULT_FAST_MODEL
        )

    def fetch_trending_topics(self, max_per_sub: int = 8) -> List[Dict]:
        """
        Fetch trending topics from AI-relevant subreddits.

        Returns:
            List of topic dicts with title, subreddit, url
        """
        import feedparser
        import requests

        all_topics = []
        user_agent = "AIEngineerContentBot/1.0"

        print(f"\n[Topic Research] Fetching from {len(AI_ENGINEER_SUBREDDITS)} subreddits...")

        for subreddit in AI_ENGINEER_SUBREDDITS:
            try:
                rss_url = f"https://www.reddit.com/r/{subreddit}/hot.rss"
                response = requests.get(
                    rss_url,
                    headers={'User-Agent': user_agent},
                    timeout=10
                )

                if response.status_code != 200:
                    continue

                feed = feedparser.parse(response.content)

                for entry in feed.entries[:max_per_sub]:
                    title = entry.get('title', '')
                    # Skip pinned/meta posts
                    if not title or '[' in title[:3]:
                        continue

                    all_topics.append({
                        'title': title,
                        'subreddit': subreddit,
                        'url': entry.get('link', ''),
                    })

                time.sleep(0.3)  # Rate limiting

            except Exception as e:
                print(f"  [!] Failed r/{subreddit}: {str(e)[:40]}")
                continue

        # Deduplicate
        seen = set()
        unique = []
        for t in all_topics:
            if t['title'] not in seen:
                seen.add(t['title'])
                unique.append(t)

        print(f"  [Topic Research] Found {len(unique)} unique topics")
        return unique[:40]  # Cap at 40

    def analyze_and_select(self, topics: List[Dict]) -> Dict:
        """
        Use LLM to analyze topics and select the best one for AI Engineer positioning.

        Args:
            topics: List of topic dicts from fetch_trending_topics()

        Returns:
            Selected topic dict with reasoning
        """
        if not topics:
            return {
                'title': "Building Production-Ready AI Agents: Lessons from the Trenches",
                'subreddit': "generated",
                'reasoning': "Fallback topic - no Reddit topics available"
            }

        # Format topics for analysis
        topics_text = "\n".join([
            f"{i+1}. [{t['subreddit']}] {t['title']}"
            for i, t in enumerate(topics[:25])  # Limit for context
        ])

        prompt = f"""Analyze these trending Reddit topics and select THE BEST ONE for an AI Engineer building thought leadership on Medium/LinkedIn.

TRENDING TOPICS:
{topics_text}

SELECTION CRITERIA (rank importance):
1. Shows AI ENGINEERING depth (building, deploying, scaling AI - not just using ChatGPT)
2. Currently TRENDING with high engagement potential
3. RECRUITER MAGNET - would impress hiring managers at top AI companies
4. Allows UNIQUE PRACTITIONER ANGLE - not generic news coverage
5. Works on LINKEDIN - professional, technical, but accessible

Your response MUST be valid JSON with this exact structure:
{{
    "selected_number": <number 1-25>,
    "selected_title": "<exact title from list>",
    "reasoning": "<2-3 sentences on why this is optimal>",
    "angle": "<specific angle/hook for the article>",
    "recruiter_appeal": "<why a hiring manager would be impressed>"
}}

Think strategically. This content needs to position the author as someone companies WANT to hire."""

        try:
            response = self.call_llm(prompt, temperature=0.3)
        except LLMError as e:
            print(f"  [!] LLM call failed during topic analysis: {e}")
            # Fall through to fallback logic below
            response = None

        # Parse response
        if response:
            result = extract_json_from_llm(response)
            if result is not None:
                try:
                    # Get the actual topic
                    idx = result.get('selected_number', 1) - 1
                    if 0 <= idx < len(topics):
                        selected_topic = topics[idx]
                        selected_topic['reasoning'] = result.get('reasoning', '')
                        selected_topic['angle'] = result.get('angle', '')
                        selected_topic['recruiter_appeal'] = result.get('recruiter_appeal', '')
                        return selected_topic
                except (KeyError, TypeError) as e:
                    print(f"  [!] Parse error: {e}")
            else:
                print("  [!] Parse error: no valid JSON found")

        # Fallback to first AI-relevant topic
        for t in topics:
            if any(kw in t['title'].lower() for kw in ['llm', 'ai', 'ml', 'agent', 'model', 'gpt', 'claude']):
                t['reasoning'] = "Fallback selection - AI-relevant keyword match"
                return t

        return topics[0]

    def research_topic(self) -> Dict:
        """
        Full automated topic research pipeline.

        Returns:
            Best topic dict with title, reasoning, angle
        """
        print("\n" + "=" * 60)
        print("[Topic Research Agent] Starting automated research...")
        print("=" * 60)

        # Step 1: Fetch trending topics
        topics = self.fetch_trending_topics()

        if not topics:
            print("  [!] No topics fetched, using fallback")
            return {
                'title': "Why AI Engineers Are Building Agents Instead of Models in 2026",
                'subreddit': "generated",
                'reasoning': "Fallback topic - highly relevant to AI engineering trends"
            }

        # Step 2: LLM analysis and selection
        print("\n[Topic Research] Analyzing for AI Engineer positioning...")
        selected = self.analyze_and_select(topics)

        print(f"\n[Topic Research] SELECTED TOPIC:")
        print(f"  Title: {selected['title']}")
        print(f"  Source: r/{selected.get('subreddit', 'unknown')}")
        print(f"  Reasoning: {selected.get('reasoning', 'N/A')}")

        return selected


def auto_select_topic() -> str:
    """
    Convenience function for one-click topic selection.

    Returns:
        Selected topic title string
    """
    agent = TopicResearchAgent()
    result = agent.research_topic()
    return result['title']


if __name__ == "__main__":
    # Test the agent
    agent = TopicResearchAgent()
    topic = agent.research_topic()
    print(f"\n\nFINAL SELECTION: {topic['title']}")
