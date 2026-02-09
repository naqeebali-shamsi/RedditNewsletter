"""
Perplexity Research Agent - Uses Sonar Pro for grounded fact verification.

Perplexity's Sonar Pro is specialized for factual, grounded research:
- Built on Llama 3.3 70B, fine-tuned for factuality
- F-score 0.858 on SimpleQA benchmark (best in class)
- 1200 tokens/sec via Cerebras inference
- Returns numbered citations with URLs automatically

Why Perplexity for research:
- Purpose-built for grounded search (not a general LLM with search bolted on)
- Citations are first-class citizens in the response
- Optimized for factuality over creativity
- OpenAI-compatible API for easy integration
"""

import os
import json
import re
from typing import Dict, List, Optional
from openai import OpenAI
from execution.utils.json_parser import extract_json_from_llm
from execution.config import config
from execution.utils.research_templates import (
    generate_writer_constraints,
    generate_revision_instructions,
    FALLBACK_CONSTRAINTS_TEXT,
)


class PerplexityResearchAgent:
    """
    Research agent powered by Perplexity Sonar Pro.

    Perplexity excels at grounded search - every claim comes with citations.
    This makes it ideal for fact-checking technical content.
    """

    # Research-focused system prompt
    RESEARCHER_SYSTEM = """You are an elite technical fact-checker using real-time web search.

YOUR MISSION: Verify claims and gather REAL facts with citations.

PRINCIPLES:
1. SEARCH EVERYTHING - Use web search for ANY specific claim
2. CITE EVERYTHING - Every fact must have a source URL
3. TRUST OFFICIAL SOURCES - Official docs > blogs > forums
4. FLAG FAKE METRICS - "parameters per second" is NOT a real metric
5. ADMIT UNKNOWNS - "Could not verify" is valuable information

FOR HARDWARE SPECS:
- GPU TFLOPS, memory bandwidth → official manufacturer specs
- Model parameters → official announcements or papers
- Pricing → official pricing pages

FOR TECHNICAL CLAIMS:
- Performance numbers need benchmark sources
- "Studies show" needs actual study citation
- Round numbers (40%, 30%) are suspicious without source

OUTPUT FORMAT:
Return a JSON object with:
- verified_facts: Array of {fact, source_url, confidence}
- unverified_claims: Array of {claim, reason, recommendation}
- general_knowledge: Array of safe-to-state facts
- unknowns: Array of things you couldn't find"""

    FACTCHECKER_SYSTEM = """You are a ruthless technical fact-checker.

YOUR MISSION: Find FALSE, UNVERIFIABLE, or SUSPICIOUS claims in this draft.

WHAT TO VERIFY (search for each):
1. Hardware specs (GPU TFLOPS, memory, bandwidth) → official docs
2. Model specs (parameters, context windows) → papers/announcements
3. Performance claims (X% improvement) → need benchmark source
4. Metrics that don't exist ("parameters per second" = FAKE)
5. Vague appeals ("studies show", "experts agree") → need specific source

RED FLAGS:
- Round percentages without sources (40%, 30%, 25%)
- "Parameters per second" or similar fake metrics
- Trillion-scale claims without naming specific models
- Hardware specs that sound impossible

OUTPUT FORMAT:
Return a JSON object with:
- verified_claims: [{claim, source_url, confidence}]
- false_claims: [{claim, why_false, correction}]
- unverifiable_claims: [{claim, why_unverifiable}]
- suspicious_claims: [{claim, red_flag}]
- overall_accuracy_score: 0-100
- recommendation: "PASS" | "REVISE" | "REJECT"

Be ruthless. If you can't verify it, flag it."""

    def __init__(self, model: str = None):
        if model is None:
            model = config.models.RESEARCH_MODEL_FALLBACK
        """
        Initialize the Perplexity research agent.

        Args:
            model: Perplexity model to use. Options:
                   - sonar-pro: Best factuality (recommended)
                   - sonar: Faster, slightly less accurate
                   - sonar-reasoning: For complex multi-step research
        """
        api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            raise ValueError("PERPLEXITY_API_KEY environment variable required")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.perplexity.ai"
        )
        self.model = model

    def research_topic(self, topic: str, source_content: str = "") -> Dict:
        """
        Research a topic using Perplexity's grounded search.

        Args:
            topic: The topic to research
            source_content: Original source content to analyze

        Returns:
            Fact sheet with verified facts, sources, and writer constraints
        """
        user_prompt = f"""RESEARCH TASK:
Topic: {topic}

Source Content to Analyze:
{source_content[:3000] if source_content else "No source content provided"}

INSTRUCTIONS:
1. Identify ALL specific claims in the topic/source that need verification
2. Search the web to verify each claim
3. For hardware specs: Find official documentation
4. For model specs: Find official announcements or papers
5. For metrics/numbers: Find the original source
6. Flag anything that seems made up (like "parameters per second")

Return your findings as a JSON object with verified_facts, unverified_claims, general_knowledge, and unknowns."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.RESEARCHER_SYSTEM},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1  # Low temp for factual accuracy
            )

            result = self._parse_response(response.choices[0].message.content)

            # Extract citations if present in response
            if hasattr(response.choices[0].message, 'citations'):
                result["perplexity_citations"] = response.choices[0].message.citations

            # Generate writer constraints
            result["writer_constraints"] = self._generate_writer_constraints(result)

            return result

        except Exception as e:
            return {
                "error": str(e),
                "verified_facts": [],
                "unverified_claims": [],
                "general_knowledge": [],
                "unknowns": [f"Research failed: {e}"],
                "writer_constraints": self._generate_fallback_constraints()
            }

    def verify_draft(self, draft: str, topic: str = "") -> Dict:
        """
        Verify claims in a draft article against real sources.

        Args:
            draft: The article draft to fact-check
            topic: Topic context

        Returns:
            Verification results with corrections
        """
        user_prompt = f"""DRAFT TO VERIFY:
{draft[:12000]}

TOPIC CONTEXT: {topic}

INSTRUCTIONS:
1. Extract EVERY specific claim from this draft
2. Search the web to verify each claim
3. Check hardware specs against official documentation
4. Check model specs against papers/announcements
5. Flag any metrics that don't exist (like "parameters per second")
6. Flag vague appeals to authority without specific sources

Return your findings as a JSON object."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.FACTCHECKER_SYSTEM},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )

            result = self._parse_verification_response(response.choices[0].message.content)

            # Generate revision instructions if needed
            if result.get("recommendation") != "PASS":
                result["revision_instructions"] = self._generate_revision_instructions(result)

            return result

        except Exception as e:
            return {
                "error": str(e),
                "verified_claims": [],
                "false_claims": [],
                "unverifiable_claims": [],
                "suspicious_claims": [],
                "overall_accuracy_score": 0,
                "recommendation": "REJECT",
                "revision_instructions": f"Verification failed: {e}. Manual review required."
            }

    def _parse_response(self, text: str) -> Dict:
        """Parse the research response from Perplexity."""
        result = extract_json_from_llm(text)
        if result is not None:
            return result
        return {
            "verified_facts": [],
            "unverified_claims": [],
            "general_knowledge": [],
            "unknowns": [],
            "raw_response": text
        }

    def _parse_verification_response(self, text: str) -> Dict:
        """Parse the verification response from Perplexity."""
        result = extract_json_from_llm(text)
        if result is not None:
            return result
        return {
            "verified_claims": [],
            "false_claims": [],
            "unverifiable_claims": [],
            "suspicious_claims": [],
            "overall_accuracy_score": 0,
            "recommendation": "REJECT",
            "raw_response": text
        }

    def _generate_writer_constraints(self, fact_sheet: Dict) -> str:
        """Generate natural language constraints for the Writer."""
        return generate_writer_constraints(fact_sheet, provider_label="via Perplexity")

    def _generate_fallback_constraints(self) -> str:
        """Generate constraints when research fails."""
        return FALLBACK_CONSTRAINTS_TEXT

    def _generate_revision_instructions(self, verification: Dict) -> str:
        """Generate specific revision instructions based on what failed."""
        return generate_revision_instructions(verification)


# Convenience function for pipeline integration
def perplexity_research(topic: str, draft: str = "", source_content: str = "") -> Dict:
    """
    One-call function to research a topic and/or verify a draft using Perplexity.

    Usage in pipeline:
        # Before writing
        fact_sheet = perplexity_research(topic=topic, source_content=reddit_post)

        # After writing
        verification = perplexity_research(topic=topic, draft=article_draft)
    """
    agent = PerplexityResearchAgent()

    if draft:
        return agent.verify_draft(draft, topic)
    else:
        return agent.research_topic(topic, source_content)
