"""
Fact Verification Agent - Post-generation claim verification for quality gate.

This agent is the GATEKEEPER for publication. It verifies claims in written content
AFTER generation, ensuring nothing gets published with unverified facts.

Architecture:
- Extracts claims from article content using LLM
- Verifies each claim using web search (Gemini + Perplexity fallback)
- Returns structured verification results for quality gate
- Blocks publication if unverified_claim_count > MAX_UNVERIFIED_CLAIMS

WSJ Four Showstoppers addressed:
- Attribution: Every factual claim must have a source
- Source Quality: Prefer official docs, papers, authoritative sources
- No Surprises: Flag claims that contradict established facts
"""

import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

# Import config for thresholds
try:
    from execution.config import config
except ImportError:
    config = None


class VerificationStatus(Enum):
    """Verification status for individual claims."""
    VERIFIED = "verified"          # Confirmed with 2+ sources
    PARTIALLY_VERIFIED = "partial" # Found 1 source
    UNVERIFIED = "unverified"      # Could not verify
    FALSE = "false"                # Contradicted by sources
    NOT_CHECKABLE = "not_checkable"  # Opinion/subjective


@dataclass
class Claim:
    """A factual claim extracted from content."""
    text: str
    claim_type: str  # "statistic", "technical_spec", "quote", "date", "general"
    line_number: Optional[int] = None
    context: str = ""


@dataclass
class VerificationResult:
    """Result of verifying a single claim."""
    claim: Claim
    status: VerificationStatus
    sources: List[Dict] = field(default_factory=list)
    explanation: str = ""
    correction: Optional[str] = None
    confidence: float = 0.0


@dataclass
class FactVerificationReport:
    """Complete verification report for quality gate."""
    claims: List[Claim]
    results: List[VerificationResult]
    verified_count: int = 0
    unverified_count: int = 0
    false_count: int = 0
    passes_quality_gate: bool = False
    summary: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "claims": [{"text": c.text, "type": c.claim_type, "line": c.line_number} for c in self.claims],
            "results": [
                {
                    "claim": r.claim.text,
                    "status": r.status.value,
                    "sources": r.sources,
                    "explanation": r.explanation,
                    "correction": r.correction,
                    "confidence": r.confidence
                } for r in self.results
            ],
            "verified_count": self.verified_count,
            "unverified_count": self.unverified_count,
            "false_count": self.false_count,
            "passes_quality_gate": self.passes_quality_gate,
            "summary": self.summary
        }


class FactVerificationAgent:
    """
    Post-generation fact verification for quality gate compliance.

    This agent:
    1. Extracts factual claims from article content
    2. Verifies each claim using web search
    3. Returns pass/fail for quality gate

    Quality Gate Rules (from config):
    - MAX_UNVERIFIED_CLAIMS: Maximum allowed unverified claims
    - MIN_VERIFIED_FACTS: Minimum verified facts required
    """

    # Claim extraction prompt
    CLAIM_EXTRACTOR_PROMPT = """You are a technical fact-checker extracting verifiable claims.

ARTICLE:
{content}

TASK: Extract ALL factual claims that can be verified. A claim is VERIFIABLE if:
1. It states a specific number, statistic, or metric
2. It names a specific product, model, or specification
3. It quotes or attributes something to a person/organization
4. It states a date, timeline, or historical fact
5. It makes a comparison with specific numbers

DO NOT extract:
- Opinions or subjective statements
- General knowledge everyone knows
- Rhetorical questions or hypotheticals

Return JSON:
{{
    "claims": [
        {{
            "text": "The exact claim text from the article",
            "type": "statistic|technical_spec|quote|date|comparison|general",
            "context": "The sentence or paragraph containing the claim",
            "why_verify": "Brief explanation of what needs verification"
        }}
    ]
}}

Extract 5-15 claims. Focus on the most impactful factual statements."""

    # Verification prompt
    VERIFICATION_PROMPT = """You are an elite fact-checker verifying claims against real sources.

CLAIM TO VERIFY:
{claim}

CONTEXT:
{context}

SEARCH RESULTS:
{search_results}

TASK: Determine if this claim is TRUE, FALSE, or UNVERIFIABLE based on the search results.

Return JSON:
{{
    "status": "verified|partial|unverified|false|not_checkable",
    "sources": [
        {{
            "url": "https://...",
            "title": "Source title",
            "relevance": "How this source supports/contradicts the claim"
        }}
    ],
    "explanation": "Detailed explanation of your verification",
    "correction": "If false, what is the correct information?",
    "confidence": 0.0-1.0
}}

RULES:
- "verified": 2+ authoritative sources confirm
- "partial": 1 source confirms, need more
- "unverified": No sources found to confirm
- "false": Sources contradict the claim
- "not_checkable": Claim is subjective/opinion"""

    def __init__(self):
        """Initialize the fact verification agent with available providers."""
        self.providers = []
        self._setup_providers()

        # Quality gate thresholds from config
        if config:
            self.max_unverified = config.quality.MAX_UNVERIFIED_CLAIMS
            self.min_verified = config.quality.MIN_VERIFIED_FACTS
        else:
            self.max_unverified = 1
            self.min_verified = 3

    def _setup_providers(self):
        """Setup available verification providers."""
        # Try Gemini first (best for grounded search)
        try:
            from execution.agents.gemini_researcher import GeminiResearchAgent
            gemini = GeminiResearchAgent()
            self.providers.append(("gemini", gemini))
            print("Gemini provider initialized")
        except Exception as e:
            print(f"Gemini provider not available: {e}")

        # Fallback to Perplexity
        try:
            from execution.agents.perplexity_researcher import PerplexityResearchAgent
            perplexity = PerplexityResearchAgent()
            self.providers.append(("perplexity", perplexity))
            print("Perplexity provider initialized")
        except Exception as e:
            print(f"Perplexity provider not available: {e}")

        # Groq as additional fallback (for claim extraction only, no web search)
        try:
            groq_key = os.getenv("GROQ_API_KEY")
            if groq_key:
                from groq import Groq
                groq_client = Groq(api_key=groq_key)
                # Create a simple wrapper object with client and model attributes
                class GroqWrapper:
                    def __init__(self, client):
                        self.client = client
                        self.model = "llama-3.3-70b-versatile"
                self.providers.append(("groq", GroqWrapper(groq_client)))
                print("Groq provider initialized (fallback for claim extraction)")
        except Exception as e:
            print(f"Groq provider not available: {e}")

        if not self.providers:
            raise RuntimeError("No verification providers available. Need Gemini, Perplexity, or Groq.")

    def check_health(self) -> dict:
        """Check which providers are available and their capabilities."""
        health = {
            "providers_available": len(self.providers),
            "has_web_search": False,
            "extraction_only_providers": [],
            "full_providers": [],
            "warnings": []
        }

        for name, provider in self.providers:
            if name in ("gemini", "perplexity"):
                health["full_providers"].append(name)
                health["has_web_search"] = True
            elif name == "groq":
                health["extraction_only_providers"].append(name)
                health["warnings"].append(
                    "Groq available for extraction only (no web search). "
                    "Verification quality may be degraded."
                )

        if not health["has_web_search"]:
            health["warnings"].append(
                "CRITICAL: No web search provider available. "
                "Verification will be unreliable."
            )

        if len(health["full_providers"]) < 2:
            health["warnings"].append(
                f"Only {len(health['full_providers'])} verification provider(s) available. "
                "Multi-source verification not possible."
            )

        for warning in health["warnings"]:
            print(f"[FactVerification Health] {warning}")

        return health

    def verify_article(self, content: str, topic: str = "") -> FactVerificationReport:
        """
        Verify all claims in an article.

        Args:
            content: The article content to verify
            topic: Optional topic context

        Returns:
            FactVerificationReport with pass/fail status
        """
        # Step 1: Extract claims
        claims = self._extract_claims(content)

        if not claims:
            return FactVerificationReport(
                claims=[],
                results=[],
                verified_count=0,
                unverified_count=0,
                false_count=0,
                passes_quality_gate=False,  # Fail-closed
                summary="NEEDS REVIEW: No verifiable claims could be extracted. "
                        "This may indicate extraction failure or highly abstract content. "
                        "Human review required."
            )

        # Step 1b: Validate claims against source (back-reference check)
        claims = self._validate_claims_against_source(claims, content)

        # Step 1c: Minimum claim count validation
        word_count = len(content.split())
        min_expected_claims = max(2, word_count // 300)  # ~1 claim per 300 words

        if len(claims) < min_expected_claims and word_count > 200:
            return FactVerificationReport(
                claims=claims,
                results=[],
                verified_count=0,
                unverified_count=0,
                false_count=0,
                passes_quality_gate=False,
                summary=f"NEEDS REVIEW: Only {len(claims)} claims extracted from "
                        f"{word_count}-word article (expected >= {min_expected_claims}). "
                        "Possible extraction failure."
            )

        # Step 2: Verify each claim
        results = []
        for claim in claims:
            result = self._verify_claim(claim, topic)
            results.append(result)

        # Step 3: Calculate counts
        verified = sum(1 for r in results if r.status == VerificationStatus.VERIFIED)
        partial = sum(1 for r in results if r.status == VerificationStatus.PARTIALLY_VERIFIED)
        unverified = sum(1 for r in results if r.status == VerificationStatus.UNVERIFIED)
        false_claims = sum(1 for r in results if r.status == VerificationStatus.FALSE)

        # Step 4: Quality gate decision
        passes = (
            unverified <= self.max_unverified and
            false_claims == 0 and
            (verified + partial) >= self.min_verified
        )

        # Generate summary
        summary = self._generate_summary(verified, partial, unverified, false_claims, passes)

        return FactVerificationReport(
            claims=claims,
            results=results,
            verified_count=verified + partial,
            unverified_count=unverified,
            false_count=false_claims,
            passes_quality_gate=passes,
            summary=summary
        )

    # Maximum chunk size for content splitting (chars, ~2000 words)
    MAX_CHUNK_SIZE = 12000

    def _extract_claims(self, content: str) -> List[Claim]:
        """Extract verifiable claims from content using LLM (tries all providers).

        For long content, splits into overlapping chunks at paragraph boundaries
        and deduplicates claims across chunks.
        """
        # Split long content into manageable chunks
        if len(content) <= self.MAX_CHUNK_SIZE:
            chunks = [content]
        else:
            chunks = self._split_into_chunks(content, self.MAX_CHUNK_SIZE, overlap=500)

        all_claims = []
        for chunk in chunks:
            chunk_claims = self._extract_claims_from_chunk(chunk)
            all_claims.extend(chunk_claims)

        # Deduplicate claims across chunks
        if len(chunks) > 1:
            all_claims = self._deduplicate_claims(all_claims)

        return all_claims

    def _extract_claims_from_chunk(self, content: str) -> List[Claim]:
        """Extract claims from a single content chunk using LLM providers."""
        prompt = self.CLAIM_EXTRACTOR_PROMPT.format(content=content)

        # Try each provider until one succeeds
        for provider_name, provider in self.providers:
            try:
                if provider_name == "gemini":
                    # Use Gemini directly
                    from google.genai import types
                    response = provider.client.models.generate_content(
                        model=provider.model,
                        contents=prompt,
                        config=types.GenerateContentConfig(temperature=0.1)
                    )
                    result_text = response.text
                elif provider_name == "perplexity":
                    # Use Perplexity's client directly for claim extraction (no web search needed)
                    response = provider.client.chat.completions.create(
                        model=provider.model,
                        messages=[
                            {"role": "system", "content": "You are a technical fact-checker extracting verifiable claims."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1
                    )
                    result_text = response.choices[0].message.content
                elif provider_name == "groq":
                    # Use Groq as fallback
                    response = provider.client.chat.completions.create(
                        model=provider.model,
                        messages=[
                            {"role": "system", "content": "You are a technical fact-checker extracting verifiable claims."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1
                    )
                    result_text = response.choices[0].message.content
                else:
                    continue

                # Parse JSON response
                claims = self._parse_claims(result_text)
                if claims:
                    return claims

            except Exception as e:
                print(f"Claim extraction with {provider_name} failed: {e}")
                continue

        print("All providers failed for claim extraction on chunk")
        return []

    def _split_into_chunks(self, content: str, max_size: int, overlap: int = 500) -> List[str]:
        """Split content into overlapping chunks at paragraph boundaries."""
        paragraphs = content.split("\n\n")
        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_len = len(para)
            if current_size + para_len > max_size and current_chunk:
                # Finalize current chunk
                chunks.append("\n\n".join(current_chunk))
                # Start new chunk with overlap from end of previous
                overlap_text = "\n\n".join(current_chunk)
                if len(overlap_text) > overlap:
                    # Keep last ~overlap chars worth of paragraphs
                    overlap_paras = []
                    overlap_size = 0
                    for p in reversed(current_chunk):
                        if overlap_size + len(p) > overlap:
                            break
                        overlap_paras.insert(0, p)
                        overlap_size += len(p)
                    current_chunk = overlap_paras
                    current_size = overlap_size
                else:
                    current_chunk = []
                    current_size = 0

            current_chunk.append(para)
            current_size += para_len

        # Don't forget the last chunk
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks if chunks else [content]

    def _deduplicate_claims(self, claims: List[Claim]) -> List[Claim]:
        """Remove near-duplicate claims by comparing significant words."""
        if not claims:
            return claims

        unique = []
        seen_signatures = []

        for claim in claims:
            # Create a signature from significant words
            words = set(w.lower() for w in claim.text.split() if len(w) > 4)
            if not words:
                unique.append(claim)
                continue

            # Check overlap with existing signatures
            is_duplicate = False
            for seen in seen_signatures:
                if not seen:
                    continue
                overlap = len(words & seen) / max(len(words), len(seen))
                if overlap >= 0.7:  # 70%+ word overlap = duplicate
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(claim)
                seen_signatures.append(words)

        if len(claims) != len(unique):
            print(f"  Deduplicated claims: {len(claims)} -> {len(unique)}")

        return unique

    # Common stop words excluded from back-reference overlap check
    _STOP_WORDS = frozenset({
        "about", "after", "again", "being", "between", "could", "during",
        "every", "first", "their", "there", "these", "thing", "think",
        "those", "through", "under", "using", "which", "while", "would",
        "other", "should", "still", "where", "before", "might", "never",
    })

    def _validate_claims_against_source(self, claims: List[Claim], content: str) -> List[Claim]:
        """Ensure extracted claims actually come from the article.

        Drops claims where fewer than 60% of significant words appear in
        the source text, which indicates the LLM hallucinated the claim
        during extraction.
        """
        validated = []
        content_lower = content.lower()

        for claim in claims:
            claim_words = set(claim.text.lower().split())
            significant_words = {
                w for w in claim_words
                if len(w) > 4 and w not in self._STOP_WORDS
            }

            if not significant_words:
                validated.append(claim)  # Short claims, give benefit of doubt
                continue

            # At least 60% of significant words should appear in source
            found = sum(1 for w in significant_words if w in content_lower)
            if found / len(significant_words) >= 0.6:
                validated.append(claim)
            else:
                print(f"  Dropping hallucinated claim: {claim.text[:60]}...")

        if len(claims) != len(validated):
            print(f"  Back-reference validation: {len(claims)} -> {len(validated)} claims")

        return validated

    def _parse_claims(self, response: str) -> List[Claim]:
        """Parse LLM response into Claim objects."""
        try:
            # Extract JSON from response
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            data = json.loads(response.strip())
            claims = []

            for item in data.get("claims", []):
                claims.append(Claim(
                    text=item.get("text", ""),
                    claim_type=item.get("type", "general"),
                    context=item.get("context", "")
                ))

            return claims

        except Exception as e:
            print(f"Failed to parse claims: {e}")
            return []

    def _verify_claim(self, claim: Claim, topic: str) -> VerificationResult:
        """Verify a single claim using available providers."""
        # Try each provider until one succeeds
        for provider_name, provider in self.providers:
            try:
                if provider_name == "gemini":
                    return self._verify_with_gemini(claim, topic, provider)
                else:
                    return self._verify_with_perplexity(claim, topic, provider)
            except Exception as e:
                print(f"Provider {provider_name} failed: {e}")
                continue

        # All providers failed
        return VerificationResult(
            claim=claim,
            status=VerificationStatus.UNVERIFIED,
            sources=[],
            explanation="All verification providers failed",
            confidence=0.0
        )

    def _verify_with_gemini(self, claim: Claim, topic: str, provider) -> VerificationResult:
        """Verify using Gemini with Google Search Grounding."""
        from google.genai import types

        # Use Gemini's verify_draft for grounded search
        search_prompt = f"""Verify this specific claim using Google Search:

CLAIM: {claim.text}
CONTEXT: {claim.context}
TOPIC: {topic}

Search for authoritative sources that confirm or contradict this claim.
Return your findings as JSON with status, sources, and explanation."""

        response = provider.client.models.generate_content(
            model=provider.model,
            contents=search_prompt,
            config=types.GenerateContentConfig(
                tools=[provider.grounding_tool],
                temperature=0.1
            )
        )

        return self._parse_verification_result(claim, response.text, response)

    def _verify_with_perplexity(self, claim: Claim, topic: str, provider) -> VerificationResult:
        """Verify using Perplexity search."""
        # Use verify_draft which has built-in web search
        claim_as_draft = f"""CLAIM TO VERIFY:
{claim.text}

CONTEXT:
{claim.context}

TOPIC:
{topic}

Please verify this specific claim using web search. Return JSON with status, sources, explanation, correction (if false), and confidence."""

        result = provider.verify_draft(claim_as_draft, topic)

        # Convert verify_draft response to our format
        return self._convert_perplexity_result(claim, result)

    def _convert_perplexity_result(self, claim: Claim, result: Dict) -> VerificationResult:
        """Convert Perplexity verify_draft response to VerificationResult."""
        # Map Perplexity recommendation to our status
        recommendation = result.get("recommendation", "REJECT")
        accuracy_score = result.get("overall_accuracy_score", 0)

        # Determine status from Perplexity's response
        if recommendation == "PASS" or accuracy_score >= 80:
            status = VerificationStatus.VERIFIED
        elif accuracy_score >= 50:
            status = VerificationStatus.PARTIALLY_VERIFIED
        elif result.get("false_claims"):
            status = VerificationStatus.FALSE
        else:
            status = VerificationStatus.UNVERIFIED

        # Extract sources from verified_claims
        sources = []
        for vc in result.get("verified_claims", []):
            if isinstance(vc, dict) and vc.get("source_url"):
                sources.append({
                    "url": vc.get("source_url"),
                    "title": vc.get("claim", "Verified claim"),
                    "relevance": "Perplexity verified"
                })

        # Get explanation and correction
        explanation = ""
        correction = None

        false_claims = result.get("false_claims", [])
        if false_claims:
            fc = false_claims[0] if false_claims else {}
            explanation = fc.get("why_false", "") if isinstance(fc, dict) else str(fc)
            correction = fc.get("correction") if isinstance(fc, dict) else None

        if not explanation:
            explanation = result.get("revision_instructions", result.get("raw_response", ""))[:500]

        return VerificationResult(
            claim=claim,
            status=status,
            sources=sources,
            explanation=explanation,
            correction=correction,
            confidence=accuracy_score / 100.0 if accuracy_score else 0.5
        )

    def _parse_verification_result(self, claim: Claim, response: str,
                                   gemini_response=None) -> VerificationResult:
        """Parse verification response into VerificationResult."""
        try:
            # Try to extract JSON
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            else:
                # Try to find JSON object
                start = response.find("{")
                end = response.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = response[start:end]
                else:
                    raise ValueError("No JSON found in response")

            data = json.loads(json_str.strip())

            # Map status string to enum
            status_map = {
                "verified": VerificationStatus.VERIFIED,
                "partial": VerificationStatus.PARTIALLY_VERIFIED,
                "unverified": VerificationStatus.UNVERIFIED,
                "false": VerificationStatus.FALSE,
                "not_checkable": VerificationStatus.NOT_CHECKABLE
            }
            status = status_map.get(data.get("status", "unverified"),
                                   VerificationStatus.UNVERIFIED)

            # Extract sources from response or Gemini grounding metadata
            sources = data.get("sources", [])
            if gemini_response and hasattr(gemini_response, 'candidates'):
                grounding_sources = self._extract_grounding_sources(gemini_response)
                sources.extend(grounding_sources)

            return VerificationResult(
                claim=claim,
                status=status,
                sources=sources,
                explanation=data.get("explanation", ""),
                correction=data.get("correction"),
                confidence=float(data.get("confidence", 0.5))
            )

        except Exception as e:
            print(f"Failed to parse verification result: {e}")
            # Fallback: analyze response text for keywords
            response_lower = response.lower()
            if "verified" in response_lower or "confirmed" in response_lower:
                status = VerificationStatus.PARTIALLY_VERIFIED
            elif "false" in response_lower or "incorrect" in response_lower:
                status = VerificationStatus.FALSE
            else:
                status = VerificationStatus.UNVERIFIED

            return VerificationResult(
                claim=claim,
                status=status,
                sources=[],
                explanation=response[:500],
                confidence=0.3
            )

    def _extract_grounding_sources(self, response) -> List[Dict]:
        """Extract sources from Gemini grounding metadata."""
        sources = []
        try:
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'grounding_metadata'):
                    metadata = candidate.grounding_metadata
                    if hasattr(metadata, 'grounding_chunks'):
                        for chunk in metadata.grounding_chunks:
                            if hasattr(chunk, 'web'):
                                sources.append({
                                    "url": chunk.web.uri,
                                    "title": chunk.web.title,
                                    "relevance": "Grounding source"
                                })
        except Exception as e:
            print(f"Failed to extract grounding sources: {e}")
        return sources

    def _generate_summary(self, verified: int, partial: int, unverified: int,
                          false_claims: int, passes: bool) -> str:
        """Generate human-readable verification summary."""
        total = verified + partial + unverified + false_claims

        if passes:
            emoji = "✅"
            status = "PASSED"
        else:
            emoji = "❌"
            status = "FAILED"

        lines = [
            f"{emoji} FACT VERIFICATION {status}",
            "",
            f"Total claims analyzed: {total}",
            f"  ✅ Verified: {verified}",
            f"  🔶 Partially verified: {partial}",
            f"  ⚠️  Unverified: {unverified}",
            f"  ❌ False: {false_claims}",
            "",
        ]

        if not passes:
            if false_claims > 0:
                lines.append(f"⚠️  BLOCKING: {false_claims} false claim(s) found")
            if unverified > self.max_unverified:
                lines.append(f"⚠️  BLOCKING: {unverified} unverified claims (max: {self.max_unverified})")
            if (verified + partial) < self.min_verified:
                lines.append(f"⚠️  BLOCKING: Only {verified + partial} verified facts (min: {self.min_verified})")

        return "\n".join(lines)


# Convenience function for pipeline integration
def verify_article_facts(content: str, topic: str = "") -> Tuple[bool, Dict]:
    """
    Verify article facts and return quality gate status.

    Returns:
        Tuple of (passes_quality_gate, verification_report_dict)
    """
    agent = FactVerificationAgent()
    report = agent.verify_article(content, topic)
    return report.passes_quality_gate, report.to_dict()


if __name__ == "__main__":
    # Test the agent
    import sys

    test_article = """
    # The Rise of Open Source AI

    Meta's Llama 3 70B model has achieved remarkable benchmarks, scoring 82% on MMLU,
    making it competitive with GPT-4. The model has 70 billion parameters and was
    trained on 15 trillion tokens of data.

    According to industry analysts, the AI market will reach $500 billion by 2027.
    OpenAI's GPT-4 has a context window of 128,000 tokens, while Claude 3 Opus
    can process up to 200,000 tokens.

    Studies show that 73% of enterprises are now using AI in production.
    """

    print("Testing Fact Verification Agent...")
    print("=" * 60)

    try:
        agent = FactVerificationAgent()
        report = agent.verify_article(test_article, "Open Source AI")

        print(report.summary)
        print()
        print("Detailed Results:")
        print("-" * 60)

        for result in report.results:
            status_emoji = {
                VerificationStatus.VERIFIED: "✅",
                VerificationStatus.PARTIALLY_VERIFIED: "🔶",
                VerificationStatus.UNVERIFIED: "⚠️",
                VerificationStatus.FALSE: "❌",
                VerificationStatus.NOT_CHECKABLE: "➖"
            }
            print(f"{status_emoji.get(result.status, '?')} {result.claim.text[:60]}...")
            print(f"   Status: {result.status.value}")
            print(f"   Explanation: {result.explanation[:100]}...")
            print()

    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)
