"""
Original Thought Agent - Generates authentic, original perspectives.

This agent is the KEY differentiator between:
- "Research regurgitation" (worthless)
- "Thought leadership" (valuable)

It takes research and asks: "What do WE think about this? What unique
angle can we provide? What's our contrarian take?"
"""

from .base_agent import BaseAgent
from typing import Dict, List, Optional
import asyncio
import json


class OriginalThoughtAgent(BaseAgent):
    """
    Generates original opinions, insights, and unique perspectives.

    This is NOT about summarizing what others said.
    This IS about adding genuine intellectual value.

    The agent operates in several modes:
    1. Contrarian Analysis - Challenge conventional wisdom
    2. Pattern Recognition - Connect dots others miss
    3. Future Prediction - Where is this heading?
    4. Experience Synthesis - Draw from practical experience
    5. First Principles - Break down to fundamentals, rebuild
    """

    THOUGHT_MODES = {
        "contrarian": {
            "name": "Contrarian Analysis",
            "prompt": """Challenge the conventional wisdom on this topic.

            Ask yourself:
            - What does everyone believe that might be wrong?
            - What's the unpopular opinion that has merit?
            - What are people afraid to say?
            - What assumption hasn't been questioned?

            Don't be contrarian for shock value. Be contrarian because
            you've thought deeper than the surface takes."""
        },
        "pattern": {
            "name": "Pattern Recognition",
            "prompt": """Connect dots that others are missing.

            Ask yourself:
            - What does this remind me of from another field?
            - What historical parallel exists?
            - What trend is this part of that nobody's naming?
            - What's the bigger picture everyone's missing?

            The best insights come from unexpected connections."""
        },
        "prediction": {
            "name": "Future Prediction",
            "prompt": """Where is this actually heading?

            Ask yourself:
            - If this trend continues, what happens in 2-5 years?
            - What's the second-order effect nobody's talking about?
            - Who wins and who loses from this change?
            - What will we look back and say "we should have seen this"?

            Make a specific, falsifiable prediction. Vague predictions are worthless."""
        },
        "experience": {
            "name": "Experience Synthesis",
            "prompt": """What practical wisdom applies here?

            Ask yourself:
            - What have practitioners learned that academics miss?
            - What works in theory but fails in practice (or vice versa)?
            - What's the "dirty secret" of this industry?
            - What would a 20-year veteran say about this?

            Ground your opinion in real-world experience, not theory."""
        },
        "first_principles": {
            "name": "First Principles Analysis",
            "prompt": """Break this down to fundamentals and rebuild.

            Ask yourself:
            - What are the underlying truths here?
            - If we started from scratch, would we build it this way?
            - What constraints are real vs. assumed?
            - What would a smart outsider ask about this?

            Strip away assumptions. Rebuild from bedrock truth."""
        }
    }

    # Opinion strength levels
    OPINION_STRENGTH = {
        "observation": "I've noticed that...",
        "hypothesis": "I believe that...",
        "conviction": "I'm convinced that...",
        "prediction": "I predict that...",
        "contrarian": "Against popular opinion, I think..."
    }

    # Static persona text — stable across calls, ideal for prompt caching
    PERSONA_TEXT = """You are a genuine thought leader, not a content mill.

Your job is to have REAL OPINIONS backed by REAL REASONING.

You are NOT:
- A summarizer of other people's ideas
- A "both sides" fence-sitter
- A generator of obvious takes
- A regurgitator of conventional wisdom

You ARE:
- An independent thinker with genuine perspectives
- Willing to take positions and defend them
- Able to say "I think X because Y" not just "some say X"
- Focused on adding intellectual value, not word count

Your opinions should be:
1. SPECIFIC - Not vague platitudes
2. DEFENSIBLE - You can explain your reasoning
3. ORIGINAL - Not just restating the consensus
4. USEFUL - Reader learns something actionable
5. HONEST - If you're uncertain, say so

Remember: A strong opinion weakly held beats no opinion at all."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        """
        Initialize with Claude for nuanced reasoning.

        Claude is chosen because:
        - Better at nuanced, thoughtful analysis
        - Less likely to give generic responses
        - Stronger at maintaining consistent voice
        """
        super().__init__(
            role="Original Thought Leader",
            persona=self.PERSONA_TEXT,
            model=model
        )
        # Pre-built static system prompt for caching
        self._static_system_prompt = (
            f"You are the {self.role}.\nPersona: {self.persona}\n"
        )

    def generate_original_angle(
        self,
        topic: str,
        research_summary: str,
        mode: str = "contrarian",
        target_audience: str = "technical professionals"
    ) -> Dict:
        """
        Generate an original angle on a researched topic.

        Args:
            topic: The subject matter
            research_summary: What the research found
            mode: Type of original thought (contrarian, pattern, etc.)
            target_audience: Who we're writing for

        Returns:
            Dict with original_angle, reasoning, confidence, caveats
        """
        mode_config = self.THOUGHT_MODES.get(mode, self.THOUGHT_MODES["contrarian"])

        prompt = f"""TOPIC: {topic}

RESEARCH SUMMARY:
{research_summary}

TARGET AUDIENCE: {target_audience}

YOUR TASK: {mode_config['prompt']}

Generate an ORIGINAL perspective on this topic. Not a summary. Not a "both sides" take.
A genuine opinion that adds value.

Return JSON:
{{
    "original_angle": "Your unique take in 1-2 sentences",
    "headline_hook": "A compelling headline that captures your angle",
    "core_argument": "Your main argument in 3-4 sentences",
    "supporting_points": [
        "Point 1 with specific reasoning",
        "Point 2 with specific reasoning",
        "Point 3 with specific reasoning"
    ],
    "potential_objections": [
        {{"objection": "What critics might say", "rebuttal": "Your response"}}
    ],
    "confidence_level": "low/medium/high",
    "caveats": ["What you're uncertain about or where you might be wrong"],
    "call_to_action": "What should the reader do with this insight?"
}}

Remember: Generic takes are worthless. What's YOUR genuine perspective?"""

        response = self.generate(prompt, expect_json=True,
                                 system_prompt=self._static_system_prompt)

        return {
            "mode": mode,
            "mode_name": mode_config["name"],
            "topic": topic,
            "output": response,
            "agent": "OriginalThoughtAgent"
        }

    def generate_opinion_spectrum(
        self,
        topic: str,
        research_summary: str
    ) -> Dict:
        """
        Generate multiple original angles across different modes in parallel.

        Uses asyncio.gather for ~3-4x speedup over sequential execution.
        Falls back to sequential if async is not available.
        """
        modes = ["contrarian", "pattern", "prediction", "first_principles"]

        async def _gather_angles():
            tasks = [
                self._generate_angle_async(topic, research_summary, mode)
                for mode in modes
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            angles = {}
            for mode, result in zip(modes, results):
                if isinstance(result, Exception):
                    angles[mode] = {
                        "mode": mode,
                        "mode_name": self.THOUGHT_MODES[mode]["name"],
                        "topic": topic,
                        "output": {"error": str(result),
                                   "angle": f"Failed to generate {mode} angle"},
                        "agent": "OriginalThoughtAgent",
                    }
                else:
                    angles[mode] = result
            return angles

        try:
            try:
                asyncio.get_running_loop()
                # Already in async context — cannot nest asyncio.run
                angles = self._generate_spectrum_sequential(
                    topic, research_summary, modes
                )
            except RuntimeError:
                # No running loop — safe to use asyncio.run
                angles = asyncio.run(_gather_angles())
        except Exception:
            # Ultimate fallback: sequential execution
            angles = self._generate_spectrum_sequential(
                topic, research_summary, modes
            )

        return {
            "topic": topic,
            "angles": angles,
            "recommendation": self._select_strongest_angle(angles),
        }

    async def _generate_angle_async(
        self, topic: str, research_summary: str, mode: str
    ) -> Dict:
        """Async wrapper for a single angle generation."""
        mode_config = self.THOUGHT_MODES.get(mode, self.THOUGHT_MODES["contrarian"])
        prompt = self._build_angle_prompt(topic, research_summary, mode_config)
        response = await self.generate_async(
            prompt, expect_json=True, system_prompt=self._static_system_prompt
        )
        return {
            "mode": mode,
            "mode_name": mode_config["name"],
            "topic": topic,
            "output": response,
            "agent": "OriginalThoughtAgent",
        }

    def _build_angle_prompt(
        self, topic: str, research_summary: str, mode_config: dict,
        target_audience: str = "technical professionals"
    ) -> str:
        """Build the prompt for a single angle generation."""
        return f"""TOPIC: {topic}

RESEARCH SUMMARY:
{research_summary}

TARGET AUDIENCE: {target_audience}

YOUR TASK: {mode_config['prompt']}

Generate an ORIGINAL perspective on this topic. Not a summary. Not a "both sides" take.
A genuine opinion that adds value.

Return JSON:
{{
    "original_angle": "Your unique take in 1-2 sentences",
    "headline_hook": "A compelling headline that captures your angle",
    "core_argument": "Your main argument in 3-4 sentences",
    "supporting_points": [
        "Point 1 with specific reasoning",
        "Point 2 with specific reasoning",
        "Point 3 with specific reasoning"
    ],
    "potential_objections": [
        {{"objection": "What critics might say", "rebuttal": "Your response"}}
    ],
    "confidence_level": "low/medium/high",
    "caveats": ["What you're uncertain about or where you might be wrong"],
    "call_to_action": "What should the reader do with this insight?"
}}

Remember: Generic takes are worthless. What's YOUR genuine perspective?"""

    def _generate_spectrum_sequential(
        self, topic: str, research_summary: str, modes: list
    ) -> Dict:
        """Sequential fallback for when async is not available."""
        angles = {}
        for mode in modes:
            try:
                angles[mode] = self.generate_original_angle(
                    topic=topic,
                    research_summary=research_summary,
                    mode=mode,
                )
            except Exception as e:
                angles[mode] = {
                    "mode": mode,
                    "mode_name": self.THOUGHT_MODES[mode]["name"],
                    "topic": topic,
                    "output": {"error": str(e),
                               "angle": f"Failed to generate {mode} angle"},
                    "agent": "OriginalThoughtAgent",
                }
        return angles

    def _select_strongest_angle(self, angles: Dict) -> str:
        """Select which angle is most promising."""
        # In practice, this could use another LLM call to evaluate
        # For now, return contrarian as default strong choice
        return "contrarian"

    def add_personal_insight(
        self,
        content: str,
        insight_type: str = "observation"
    ) -> str:
        """
        Add a personal insight/opinion to existing content.

        This injects authentic voice into otherwise generic content.
        """
        opener = self.OPINION_STRENGTH.get(insight_type, "I think that...")

        prompt = f"""CONTENT:
{content}

Add a genuine personal insight to this content. Start with: "{opener}"

The insight should:
1. Add value not present in the original
2. Show independent thinking
3. Be specific and defensible
4. Sound like a real human expert, not an AI

Return just the insight paragraph (2-4 sentences)."""

        return self.generate(prompt,
                             system_prompt=self._static_system_prompt)

    def challenge_assumptions(self, content: str) -> List[Dict]:
        """
        Identify and challenge hidden assumptions in content.

        Great for strengthening arguments by acknowledging weaknesses.
        """
        prompt = f"""CONTENT:
{content}

Identify 3-5 hidden assumptions in this content that should be questioned.

For each assumption, provide:
1. The assumption being made
2. Why it might be wrong
3. How to address it in the content

Return JSON array:
[
    {{
        "assumption": "What's being assumed",
        "challenge": "Why this might be wrong",
        "resolution": "How to handle this in the content"
    }}
]"""

        return self.generate(prompt, expect_json=True,
                             system_prompt=self._static_system_prompt)


# Convenience function
def generate_original_take(
    topic: str,
    research: str,
    mode: str = "contrarian"
) -> Dict:
    """Quick function to generate an original perspective."""
    agent = OriginalThoughtAgent()
    return agent.generate_original_angle(topic, research, mode)
