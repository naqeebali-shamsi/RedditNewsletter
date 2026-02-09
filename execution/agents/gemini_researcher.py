"""
Gemini Research Agent - Uses Google Search Grounding for REAL fact verification.

This agent leverages Gemini's native Google Search integration to:
1. Research topics with real-time web data
2. Verify claims in drafts against actual sources
3. Return grounded facts with citations

Why Gemini for research:
- Native Google Search integration (best search quality)
- Returns groundingMetadata with actual URLs
- Model autonomously decides when to search
- Citations are built into the response
"""

import os
import json
from typing import Dict, List, Optional
from google import genai
from google.genai import types
from execution.utils.json_parser import extract_json_from_llm
from execution.config import config
from execution.utils.research_templates import (
    generate_writer_constraints,
    generate_revision_instructions,
    FALLBACK_CONSTRAINTS_TEXT,
)


class GeminiResearchAgent:
    """
    Research agent powered by Gemini with Google Search Grounding.

    This is NOT a wrapper - it's a purpose-built research tool that uses
    Gemini's ability to search Google in real-time.
    """

    # Research-focused persona
    RESEARCHER_PERSONA = """You are an elite technical fact-checker and researcher.

YOUR MISSION: Verify claims and gather REAL facts using web search.

PRINCIPLES:
1. SEARCH AGGRESSIVELY - Use Google Search for ANY specific claim
2. TRUST OFFICIAL SOURCES - AMD.com > random blog for GPU specs
3. FLAG FAKE METRICS - "parameters per second" doesn't exist
4. CITE EVERYTHING - Every fact needs a URL
5. ADMIT UNKNOWNS - "Could not verify" is valuable

OUTPUT REQUIREMENTS:
- Return structured JSON with verified_facts and unverified_claims
- Include source URLs for every verified fact
- Explain WHY something couldn't be verified
"""

    # Fact-checker persona for draft review
    FACTCHECKER_PERSONA = """You are a ruthless technical fact-checker reviewing a draft article.

YOUR MISSION: Identify claims that are FALSE, UNVERIFIABLE, or SUSPICIOUS.

WHAT TO CHECK (search Google for each):
1. Hardware specs (GPU TFLOPS, memory, prices) - verify against official docs
2. Model specifications (parameters, context windows) - verify against papers
3. Performance claims (X% improvement, Y times faster) - need source
4. Made-up metrics ("parameters per second" = FAKE)
5. Vague appeals ("studies show", "experts agree") - need specific source

OUTPUT: Return JSON with:
- verified_claims: [{claim, source_url, confidence}]
- false_claims: [{claim, why_false, correction}]
- unverifiable_claims: [{claim, why_unverifiable}]
- suspicious_claims: [{claim, red_flag}]
"""

    def __init__(self, model: str = None):
        """
        Initialize the Gemini research agent.

        Args:
            model: Gemini model to use (must support google_search tool).
                   Defaults to config.models.GEMINI_RESEARCH_MODEL.
        """
        if model is None:
            model = config.models.GEMINI_RESEARCH_MODEL
        # Initialize Gemini client
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable required")

        self.client = genai.Client(api_key=api_key)
        self.model = model

        # Configure grounding tool
        self.grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )

    def research_topic(self, topic: str, source_content: str = "") -> Dict:
        """
        Research a topic using Google Search Grounding.

        Args:
            topic: The topic to research
            source_content: Original source content to analyze

        Returns:
            Fact sheet with verified facts, sources, and writer constraints
        """
        prompt = f"""{self.RESEARCHER_PERSONA}

RESEARCH TASK:
Topic: {topic}

Source Content to Analyze:
{source_content[:3000] if source_content else "No source content provided"}

INSTRUCTIONS:
1. Identify ALL specific claims in the topic/source that need verification
2. Use Google Search to verify each claim
3. For hardware specs: Find official documentation
4. For model specs: Find official announcements or papers
5. For metrics/numbers: Find the original source
6. Flag anything that seems made up (like "parameters per second")

Return a JSON object:
{{
    "verified_facts": [
        {{
            "fact": "The AMD MI50 has 13.4 TFLOPS FP32",
            "source": "AMD Official Specifications",
            "source_url": "https://...",
            "confidence": "high"
        }}
    ],
    "unverified_claims": [
        {{
            "claim": "10 trillion parameters per second",
            "reason": "This metric doesn't exist. Parameters are static weights, not processed per second.",
            "recommendation": "REMOVE - fabricated metric"
        }}
    ],
    "general_knowledge": [
        "GPUs are used for parallel processing"
    ],
    "unknowns": [
        "Exact pricing for the setup - couldn't find current prices"
    ],
    "search_queries_used": [
        "AMD MI50 specifications TFLOPS"
    ]
}}

Be thorough. Search for EVERY specific claim."""

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[self.grounding_tool],
                    temperature=0.1,  # Low temp for factual accuracy
                    response_mime_type="application/json",
                )
            )

            # Parse the response
            result = self._parse_research_response(response)

            # Add grounding metadata if available
            candidates = getattr(response, 'candidates', None) or []
            if candidates:
                candidate = candidates[0]
                grounding_metadata = getattr(candidate, 'grounding_metadata', None)
                if grounding_metadata is not None:
                    result["grounding_metadata"] = self._extract_grounding_metadata(
                        grounding_metadata
                    )

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
        prompt = f"""{self.FACTCHECKER_PERSONA}

DRAFT TO VERIFY:
{draft[:12000]}

TOPIC CONTEXT: {topic}

INSTRUCTIONS:
1. Extract EVERY specific claim from this draft
2. Search Google to verify each claim
3. Check hardware specs against official documentation
4. Check model specs against papers/announcements
5. Flag any metrics that don't exist (like "parameters per second")
6. Flag vague appeals to authority without sources

Return JSON:
{{
    "verified_claims": [
        {{"claim": "...", "source_url": "...", "confidence": "high/medium/low"}}
    ],
    "false_claims": [
        {{"claim": "...", "why_false": "...", "correction": "..."}}
    ],
    "unverifiable_claims": [
        {{"claim": "...", "why_unverifiable": "..."}}
    ],
    "suspicious_claims": [
        {{"claim": "...", "red_flag": "..."}}
    ],
    "overall_accuracy_score": 0-100,
    "recommendation": "PASS/REVISE/REJECT"
}}

Be ruthless. If you can't verify it, flag it."""

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[self.grounding_tool],
                    temperature=0.1,
                    response_mime_type="application/json",
                )
            )

            result = self._parse_verification_response(response)

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

    def _parse_research_response(self, response) -> Dict:
        """Parse the research response from Gemini."""
        text = getattr(response, 'text', None) or str(response)
        # Try native JSON mode first (response_mime_type="application/json")
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass
        # Fall back to extraction from LLM prose
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

    def _parse_verification_response(self, response) -> Dict:
        """Parse the verification response from Gemini."""
        text = getattr(response, 'text', None) or str(response)
        # Try native JSON mode first (response_mime_type="application/json")
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass
        # Fall back to extraction from LLM prose
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

    def _extract_grounding_metadata(self, metadata) -> Dict:
        """Extract useful info from Gemini's grounding metadata."""
        result = {
            "search_queries": [],
            "sources": []
        }

        queries = getattr(metadata, 'web_search_queries', None)
        if queries:
            result["search_queries"] = list(queries)

        chunks = getattr(metadata, 'grounding_chunks', None) or []
        for chunk in chunks:
            web = getattr(chunk, 'web', None)
            if web is not None:
                result["sources"].append({
                    "uri": getattr(web, 'uri', None),
                    "title": getattr(web, 'title', None),
                })

        return result

    def _generate_writer_constraints(self, fact_sheet: Dict) -> str:
        """Generate natural language constraints for the Writer."""
        return generate_writer_constraints(fact_sheet)

    def _generate_fallback_constraints(self) -> str:
        """Generate constraints when research fails."""
        return FALLBACK_CONSTRAINTS_TEXT

    def _generate_revision_instructions(self, verification: Dict) -> str:
        """Generate specific revision instructions based on what failed."""
        return generate_revision_instructions(verification)


# Convenience function for pipeline integration
def research_and_verify(topic: str, draft: str = "", source_content: str = "") -> Dict:
    """
    One-call function to research a topic and/or verify a draft.

    Usage in pipeline:
        # Before writing
        fact_sheet = research_and_verify(topic=topic, source_content=reddit_post)

        # After writing
        verification = research_and_verify(topic=topic, draft=article_draft)
    """
    agent = GeminiResearchAgent()

    if draft:
        # Verify an existing draft
        return agent.verify_draft(draft, topic)
    else:
        # Research a topic before writing
        return agent.research_topic(topic, source_content)
