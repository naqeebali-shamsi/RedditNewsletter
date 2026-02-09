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
from execution.utils.json_parser import extract_json_from_llm
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None

# Import config for thresholds
try:
    from execution.config import config
except ImportError:
    config = None

# Comprehensive English stop words (based on NLTK's English stop words list).
# Used for claim deduplication and back-reference overlap checks.
# Hardcoded to avoid adding NLTK as a dependency.
STOP_WORDS = frozenset({
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her",
    "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs",
    "themselves", "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "until", "while", "of", "at", "by", "for", "with",
    "about", "against", "between", "through", "during", "before", "after", "above",
    "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s",
    "t", "can", "will", "just", "don", "should", "now", "d", "ll", "m", "o", "re",
    "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven",
    "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren",
    "won", "wouldn", "could", "would", "might", "shall", "may", "must", "need",
    "also", "however", "therefore", "thus", "hence", "moreover", "furthermore",
    "nevertheless", "nonetheless", "meanwhile", "otherwise", "instead", "although",
    "though", "yet", "still", "already", "even", "perhaps", "rather", "quite",
    "much", "many", "several", "often", "always", "never", "sometimes", "usually",
    "well", "really", "actually", "basically", "certainly", "definitely", "probably",
    "simply", "truly",
})

# Sentence boundary detection that handles abbreviations, decimals, and URLs.
# Common abbreviations that should NOT be treated as sentence endings.
_ABBREVIATIONS = frozenset({
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "vs", "etc",
    "approx", "dept", "est", "govt", "inc", "corp", "ltd",
    "assn", "vol", "no", "fig", "eq", "gen", "sgt", "col",
    "i.e", "e.g", "u.s", "u.k",
})

# Regex that finds candidate sentence breaks: punctuation + space + capital letter
_CANDIDATE_BREAK_RE = re.compile(r'([.!?])\s+(?=[A-Z])')


def _split_sentences(text: str) -> list:
    """Split text into sentences, respecting abbreviations and decimals.

    Uses a two-pass approach: first find candidate breaks with a simple regex,
    then filter out false positives from abbreviations and decimal numbers.
    """
    sentences = []
    last = 0

    for m in _CANDIDATE_BREAK_RE.finditer(text):
        break_pos = m.end()       # position right after the space
        dot_pos = m.start()       # position of the punctuation mark

        # Skip if preceded by a digit (decimal like 3.14, version like v2.0)
        if dot_pos > 0 and text[dot_pos - 1].isdigit() and m.group(1) == '.':
            continue

        # Skip if the word before the dot is an abbreviation
        word_before = text[max(0, dot_pos - 10):dot_pos].split()
        if word_before:
            token = word_before[-1].lower().rstrip('.')
            if token in _ABBREVIATIONS:
                continue

        sentence = text[last:break_pos].strip()
        if sentence:
            sentences.append(sentence)
        last = break_pos

    # Remaining text
    tail = text[last:].strip()
    if tail:
        sentences.append(tail)

    return sentences


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
                    "claim_type": r.claim.claim_type,
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


    def get_hyperlink_annotations(self) -> List[Dict]:
        """
        Generate hyperlink annotations for verified attributions.

        Returns a list of {text, url, type} for all verified person references,
        quotes, and research attributions that have source URLs.
        The pipeline can use this to inject markdown hyperlinks into the article.
        """
        annotations = []
        for result in self.results:
            if result.status not in (VerificationStatus.VERIFIED,
                                     VerificationStatus.PARTIALLY_VERIFIED):
                continue
            if not result.sources:
                continue

            url = None
            for src in result.sources:
                if isinstance(src, dict) and src.get("url"):
                    url = src["url"]
                    break

            if not url:
                continue

            if result.claim.claim_type == "person_reference":
                name = result.claim.text.replace("Person exists: ", "")
                annotations.append({"text": name, "url": url, "type": "person"})
            elif result.claim.claim_type == "research_attribution":
                institution = result.claim.text.replace("Research attribution: ", "")
                annotations.append({"text": institution, "url": url, "type": "research"})
            elif result.claim.claim_type == "direct_quote":
                annotations.append({
                    "text": result.claim.context[:60],
                    "url": url,
                    "type": "quote"
                })

        return annotations

    def get_fabrication_flags(self) -> List[Dict]:
        """
        Return all claims flagged as fabricated for revision instructions.
        """
        flags = []
        for result in self.results:
            if result.status == VerificationStatus.FALSE:
                flags.append({
                    "claim": result.claim.text,
                    "type": result.claim.claim_type,
                    "explanation": result.explanation,
                    "correction": result.correction,
                    "action": "REMOVE" if result.claim.claim_type in (
                        "person_reference", "direct_quote"
                    ) else "CORRECT"
                })
        return flags


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

    # --- Fabrication Detection: Regex patterns for high-risk hallucination targets ---
    # LLMs commonly fabricate: expert names with titles, direct quotes, institution research
    PERSON_PATTERNS = [
        # "Dr./Prof. FirstName LastName"
        re.compile(r'(?:Dr\.|Prof\.|Professor)\s+([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)'),
        # "FirstName LastName, a leading/renowned/noted ..."
        re.compile(
            r'([A-Z][a-z]+\s+[A-Z][a-z]+),\s+(?:a|an)\s+'
            r'(?:leading|renowned|noted|prominent|senior|chief|veteran|top|well-known|distinguished)\s+'
        ),
        # "According to FirstName LastName"
        re.compile(r'According to\s+([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)'),
    ]

    # Direct quotes: 30+ chars inside quotation marks (catches fabricated expert quotes)
    QUOTE_PATTERN = re.compile(r'["\u201c]([^"\u201d]{30,})["\u201d]')

    # Research attributions: "Research from Stanford and MIT shows..."
    RESEARCH_ATTRIBUTION_PATTERN = re.compile(
        r'(?:Research|Studies|Data|A\s+study|Survey|Report|Analysis)\s+'
        r'(?:from|by|at|conducted\s+at|published\s+by)\s+'
        r'([A-Z][a-zA-Z\s,&]+?)'
        r'(?:\s+(?:shows?|demonstrates?|reveals?|indicates?|found|suggests?|proves?))',
        re.IGNORECASE
    )

    # Specialized prompt for verifying person existence
    PERSON_VERIFY_PROMPT = """You are verifying whether a specific person exists with the claimed credentials.

PERSON NAME: {name}
ARTICLE CONTEXT: {context}
TOPIC: {topic}

TASK: Search for this EXACT person. Determine if they are REAL with matching credentials.

A person is VERIFIED only if:
- You find their official profile (university page, LinkedIn, Google Scholar, company page)
- Their field of expertise matches what the article claims
- They have published work or public presence in the relevant field

A person is FABRICATED (false) if:
- No person with that name works in this field
- The name exists but in a completely different field
- No public profile, publication, or mention can be found

Return JSON:
{{
    "status": "verified|false|unverified",
    "profile_url": "URL to their real profile, or null",
    "real_credentials": "Their actual title and affiliation if found",
    "explanation": "Why you believe this person is real or fabricated",
    "confidence": 0.0-1.0
}}

IMPORTANT: LLMs commonly fabricate expert names like "Dr. Sarah Chen" or "Prof. James Miller".
If you cannot find a SPECIFIC profile for this person in this field, mark as "false"."""

    # Specialized prompt for verifying direct quotes
    QUOTE_VERIFY_PROMPT = """You are verifying whether a specific quote was ever published or said.

QUOTE: "{quote_text}"
ATTRIBUTED TO: {attributed_to}
ARTICLE CONTEXT: {context}

TASK: Search for this EXACT quote text. Determine if it was ever published anywhere.

A quote is VERIFIED only if:
- You find the exact or near-exact text in a published source (article, interview, paper, talk)
- The attribution to the named person is confirmed by the source

A quote is FABRICATED (false) if:
- The exact quote text cannot be found published anywhere
- The person it's attributed to never said or wrote this
- The quote exists but was said by a different person

Return JSON:
{{
    "status": "verified|false|unverified",
    "source_url": "URL where the quote was found, or null",
    "actual_source": "Who actually said it if different, or null",
    "explanation": "Why you believe this quote is real or fabricated",
    "confidence": 0.0-1.0
}}

IMPORTANT: LLMs commonly fabricate plausible-sounding quotes attributed to real or fake experts.
If this exact quote cannot be found published ANYWHERE online, it is almost certainly fabricated."""

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
                        self.model = config.models.DEFAULT_WRITER_MODEL if config else "llama-3.3-70b-versatile"
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

        # Step 2: Verify each general claim
        results = []
        for claim in claims:
            result = self._verify_claim(claim, topic)
            results.append(result)

        # Step 2b: Fabrication risk scan — regex-based detection of persons,
        # quotes, and research attributions that LLMs commonly hallucinate
        print("\n--- Fabrication Risk Scan ---")
        fabrication_claims = self._scan_for_fabrication_risks(content)
        fabrication_results = []

        for fab_claim in fabrication_claims:
            if self._is_claim_already_covered(fab_claim, claims):
                print(f"  [Skip] Already covered: {fab_claim.text[:50]}")
                continue
            print(f"  [Verifying] {fab_claim.text[:60]}...")
            fab_result = self._verify_high_risk_claim(fab_claim, topic)
            fabrication_results.append(fab_result)
            claims.append(fab_claim)
            results.append(fab_result)
            status_label = fab_result.status.value.upper()
            print(f"    -> {status_label} (confidence: {fab_result.confidence:.1%})")

        # Step 2c: For person/quote claims, unverified = false (fail-closed)
        # If we can't confirm a named expert or quote exists, treat as fabricated
        for result in results:
            if (result.claim.claim_type in ("person_reference", "direct_quote")
                    and result.status == VerificationStatus.UNVERIFIED):
                print(f"  [Fail-Closed] Unverifiable {result.claim.claim_type} "
                      f"treated as FALSE: {result.claim.text[:50]}")
                result.status = VerificationStatus.FALSE
                result.explanation += (
                    " [FAIL-CLOSED: Unverifiable person/quote treated as fabricated. "
                    "All named experts and direct quotes MUST be traceable to a real source.]"
                )

        if fabrication_claims:
            fab_false = sum(1 for r in fabrication_results
                           if r.status == VerificationStatus.FALSE)
            fab_verified = sum(1 for r in fabrication_results
                              if r.status in (VerificationStatus.VERIFIED,
                                              VerificationStatus.PARTIALLY_VERIFIED))
            print(f"\n  Fabrication scan: {len(fabrication_claims)} high-risk items found, "
                  f"{fab_verified} verified, {fab_false} flagged as fabricated")
        else:
            print("  No high-risk fabrication patterns detected.")
        print("--- End Fabrication Scan ---\n")

        # Step 3: Calculate counts (includes fabrication scan results)
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

    # Reusable text splitter for content chunking (None if langchain not installed)
    _text_splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=12000, chunk_overlap=500,
            separators=["\n\n", "\n", ". ", " "],
        ) if RecursiveCharacterTextSplitter else None
    )

    def _split_into_chunks(self, content: str, max_size: int, overlap: int = 500) -> List[str]:
        """Split content into overlapping chunks. Uses langchain if available, else simple fallback."""
        if RecursiveCharacterTextSplitter is not None:
            splitter = self._text_splitter
            if max_size != 12000 or overlap != 500:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=max_size, chunk_overlap=overlap,
                    separators=["\n\n", "\n", ". ", " "],
                )
            chunks = splitter.split_text(content)
            return chunks if chunks else [content]
        # Fallback: simple character-based splitting with overlap
        if len(content) <= max_size:
            return [content]
        chunks = []
        start = 0
        while start < len(content):
            end = start + max_size
            chunks.append(content[start:end])
            start = end - overlap
        return chunks

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
    _STOP_WORDS = STOP_WORDS

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

    # ---------------------------------------------------------------
    # Fabrication Detection: Regex-based scan + targeted verification
    # ---------------------------------------------------------------

    @staticmethod
    def _extract_sentence_context(content: str, match_start: int, match_end: int) -> str:
        """Extract the sentence containing a regex match using proper boundary detection.

        Uses the module-level _split_sentences() so abbreviations like "Dr.",
        decimals like "3.14", and URLs are not treated as sentence boundaries.
        """
        # Find a reasonable window around the match to avoid splitting the entire document
        window_start = max(0, match_start - 500)
        window_end = min(len(content), match_end + 500)
        window = content[window_start:window_end]
        offset = match_start - window_start

        sentences = _split_sentences(window)
        # Find the sentence that contains the match position
        pos = 0
        for sentence in sentences:
            idx = window.find(sentence, pos)
            if idx == -1:
                continue
            sent_end = idx + len(sentence)
            if idx <= offset < sent_end:
                return sentence.strip()
            pos = sent_end

        # Fallback: return text around the match
        return content[max(0, match_start - 100):min(len(content), match_end + 100)].strip()

    def _scan_for_fabrication_risks(self, content: str) -> List[Claim]:
        """
        Regex-based scan for high-risk hallucination patterns.

        LLMs commonly fabricate:
        1. Expert names with impressive titles ("Dr. Emma Taylor, a leading AI researcher")
        2. Direct quotes attributed to fake experts
        3. "Research from {University}" without citing a real paper
        4. Specific company case studies with round numbers

        Returns claims that MUST be verified with stricter criteria.
        """
        high_risk_claims = []
        seen_names = set()
        seen_quotes = set()

        # 1. Detect person references
        for pattern in self.PERSON_PATTERNS:
            for match in pattern.finditer(content):
                name = match.group(1).strip().rstrip(',')
                if name in seen_names or len(name.split()) < 2:
                    continue
                seen_names.add(name)

                # Get sentence context using proper sentence boundary detection
                context = self._extract_sentence_context(content, match.start(), match.end())

                high_risk_claims.append(Claim(
                    text=f"Person exists: {name}",
                    claim_type="person_reference",
                    context=context
                ))
                print(f"  [Fabrication Scan] Detected person reference: {name}")

        # 2. Detect direct quotes (30+ chars in quotation marks)
        for match in self.QUOTE_PATTERN.finditer(content):
            quote_text = match.group(1).strip()
            # Normalize for dedup
            quote_sig = quote_text[:50].lower()
            if quote_sig in seen_quotes:
                continue
            seen_quotes.add(quote_sig)

            # Find attribution context (look before and after the quote)
            before_start = max(0, match.start() - 200)
            after_end = min(len(content), match.end() + 100)
            context = content[before_start:after_end].strip()

            # Try to extract who the quote is attributed to
            # Look in the immediate vicinity before the quote (same sentence)
            attributed_to = "unknown"
            before_quote = content[max(0, match.start() - 150):match.start()]
            for p_pattern in self.PERSON_PATTERNS:
                # Search the text right before the quote for the closest name
                attr_matches = list(p_pattern.finditer(before_quote))
                if attr_matches:
                    attributed_to = attr_matches[-1].group(1).strip().rstrip(',')
                    break

            high_risk_claims.append(Claim(
                text=f'Quote attributed to {attributed_to}: "{quote_text[:100]}"',
                claim_type="direct_quote",
                context=context
            ))
            print(f"  [Fabrication Scan] Detected direct quote attributed to: {attributed_to}")

        # 3. Detect research/study attributions
        for match in self.RESEARCH_ATTRIBUTION_PATTERN.finditer(content):
            institution = match.group(1).strip().rstrip(',')
            if len(institution) < 3:
                continue

            context = self._extract_sentence_context(content, match.start(), match.end())

            high_risk_claims.append(Claim(
                text=f"Research attribution: {institution}",
                claim_type="research_attribution",
                context=context
            ))
            print(f"  [Fabrication Scan] Detected research attribution: {institution}")

        return high_risk_claims

    @staticmethod
    def _word_similarity(text_a: str, text_b: str) -> float:
        """Compute containment similarity between two texts (ignoring stop words).

        Returns the fraction of the smaller word-set that overlaps with the larger,
        giving a length-independent similarity score between 0.0 and 1.0.
        """
        words_a = {w.lower() for w in text_a.split() if w.lower() not in STOP_WORDS}
        words_b = {w.lower() for w in text_b.split() if w.lower() not in STOP_WORDS}
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        return len(intersection) / min(len(words_a), len(words_b))

    def _is_claim_already_covered(self, new_claim: Claim, existing_claims: List[Claim]) -> bool:
        """Check if a fabrication-scan claim overlaps with an already-extracted claim."""
        combined_new = new_claim.text + " " + " ".join(new_claim.context.split()[:20])
        for existing in existing_claims:
            if self._word_similarity(combined_new, existing.text) >= 0.5:
                return True
        return False

    def _verify_high_risk_claim(self, claim: Claim, topic: str) -> VerificationResult:
        """
        Verify high-risk claims (persons, quotes, research) with TARGETED prompts.

        Unlike generic verification, these use specific search queries designed
        to detect LLM fabrication:
        - Person claims: "Does {name} exist as {role}?"
        - Quote claims: "Was this exact text ever published?"
        - Research claims: "Does this specific study exist?"
        """
        if claim.claim_type == "person_reference":
            name = claim.text.replace("Person exists: ", "")
            prompt = self.PERSON_VERIFY_PROMPT.format(
                name=name, context=claim.context, topic=topic
            )
        elif claim.claim_type == "direct_quote":
            # Extract quote text and attribution
            quote_match = re.search(r'"([^"]+)"', claim.text)
            quote_text = quote_match.group(1) if quote_match else claim.text
            attr_match = re.search(r'attributed to (\w[\w\s.]+?):', claim.text)
            attributed_to = attr_match.group(1) if attr_match else "unknown"
            prompt = self.QUOTE_VERIFY_PROMPT.format(
                quote_text=quote_text[:200],
                attributed_to=attributed_to,
                context=claim.context
            )
        elif claim.claim_type == "research_attribution":
            institution = claim.text.replace("Research attribution: ", "")
            prompt = f"""Verify this research attribution:

CLAIM: {claim.context}
INSTITUTION: {institution}
TOPIC: {topic}

Search for the SPECIFIC study or paper being referenced. Find:
- Paper title, authors, publication venue
- DOI or URL to the actual paper

Return JSON:
{{"status": "verified|false|unverified", "paper_url": "URL or null", "paper_title": "title or null", "explanation": "...", "confidence": 0.0-1.0}}

If the article vaguely says "Research from X shows..." without a specific paper, and you cannot find a matching study, mark as "unverified"."""
        else:
            return self._verify_claim(claim, topic)

        # Try each web-search-capable provider
        for provider_name, provider in self.providers:
            if provider_name == "groq":
                continue  # Groq has no web search — skip for fabrication detection

            try:
                if provider_name == "gemini":
                    from google.genai import types
                    response = provider.client.models.generate_content(
                        model=provider.model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            tools=[provider.grounding_tool],
                            temperature=0.1
                        )
                    )
                    return self._parse_verification_result(claim, response.text, response)

                elif provider_name == "perplexity":
                    response = provider.client.chat.completions.create(
                        model=provider.model,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a fact-checker detecting AI-fabricated content. "
                                    "Search the web to verify whether specific people, quotes, "
                                    "and research citations are real. Be ruthless — LLMs commonly "
                                    "fabricate expert names and quotes."
                                )
                            },
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1
                    )
                    return self._parse_verification_result(
                        claim, response.choices[0].message.content
                    )

            except Exception as e:
                print(f"  [Fabrication Check] {provider_name} failed for {claim.claim_type}: {e}")
                continue

        # All providers failed — fail-closed for high-risk claims
        print(f"  [Fabrication Check] ALL PROVIDERS FAILED for: {claim.text[:60]}")
        return VerificationResult(
            claim=claim,
            status=VerificationStatus.FALSE,
            sources=[],
            explanation="All verification providers failed. Treating as fabricated (fail-closed).",
            confidence=0.0
        )

    def _parse_claims(self, response: str) -> List[Claim]:
        """Parse LLM response into Claim objects."""
        try:
            data = extract_json_from_llm(response)
            if data is None:
                print("Failed to parse claims: no valid JSON found")
                return []

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
            data = extract_json_from_llm(response)
            if data is None:
                raise ValueError("No JSON found in response")

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
            f"  ❌ False/Fabricated: {false_claims}",
            "",
        ]

        if not passes:
            if false_claims > 0:
                lines.append(f"⚠️  BLOCKING: {false_claims} false/fabricated claim(s) found")
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

    result = report.to_dict()
    # Attach fabrication flags and hyperlink annotations
    result["fabrication_flags"] = report.get_fabrication_flags()
    result["hyperlink_annotations"] = report.get_hyperlink_annotations()

    return report.passes_quality_gate, result


def inject_hyperlinks(article: str, annotations: List[Dict]) -> str:
    """
    Post-process an article to inject markdown hyperlinks for verified attributions.

    Args:
        article: The article markdown text
        annotations: List of {text, url, type} from FactVerificationReport.get_hyperlink_annotations()

    Returns:
        Article with hyperlinks injected (e.g., "Dr. Smith" -> "[Dr. Smith](url)")
    """
    if not annotations:
        return article

    for annotation in annotations:
        text = annotation["text"]
        url = annotation["url"]
        # Only replace first occurrence to avoid double-linking
        linked = f"[{text}]({url})"
        if text in article and linked not in article:
            article = article.replace(text, linked, 1)

    return article


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
