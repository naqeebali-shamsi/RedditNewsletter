"""
Tone Inference Engine - Analyzes writing samples to extract tone profiles.

Combines deterministic text metrics (sentence structure, lexical diversity,
keyword detection) with LLM-powered style classification to produce a
ToneProfile from raw text or a URL.

Usage:
    from execution.tone_inference import ToneInferenceEngine

    engine = ToneInferenceEngine()
    profile = engine.infer_from_text_sync("Your sample text here...")
    print(profile.name)
    print(profile.to_writer_instructions())
"""

import asyncio
import re
import statistics
from typing import List, Optional

from execution.agents.base_agent import BaseAgent
from execution.config import config
from execution.tone_profiles import (
    ToneProfile,
    SentenceStyle,
    VocabularyPreferences,
)
from execution.utils.datetime_utils import utc_iso
from execution.utils.json_parser import extract_json_from_llm

# Optional NLP dependencies (same as style_enforcer.py)
try:
    from lexicalrichness import LexicalRichness
    HAS_LEXICAL = True
except ImportError:
    HAS_LEXICAL = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False


# ---------------------------------------------------------------------------
# Deterministic text analysis (reuses StyleEnforcerAgent logic)
# ---------------------------------------------------------------------------

def _tokenize_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    if HAS_NLTK:
        try:
            return sent_tokenize(text)
        except Exception:
            pass
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]


def _calculate_burstiness(text: str) -> dict:
    """Calculate sentence length variation metrics."""
    sentences = _tokenize_sentences(text)
    if len(sentences) < 3:
        return {"burstiness": 0.0, "avg_length": 0.0, "std_dev": 0.0, "count": 0}

    lengths = [len(s.split()) for s in sentences]
    mean_len = statistics.mean(lengths)
    std_dev = statistics.stdev(lengths) if len(lengths) > 1 else 0.0
    burstiness = std_dev / mean_len if mean_len > 0 else 0.0

    return {
        "burstiness": round(burstiness, 3),
        "avg_length": round(mean_len, 1),
        "std_dev": round(std_dev, 1),
        "count": len(sentences),
    }


def _calculate_lexical_diversity(text: str) -> dict:
    """Calculate lexical richness metrics."""
    if HAS_LEXICAL:
        try:
            lex = LexicalRichness(text)
            ttr = lex.ttr
            try:
                vocd = lex.vocd(ntokens=50, within_sample=100, iterations=3)
            except Exception:
                vocd = None
            return {"ttr": round(ttr, 3), "vocd": round(vocd, 1) if vocd else None}
        except Exception:
            pass

    words = text.lower().split()
    if not words:
        return {"ttr": None, "vocd": None}
    ttr = len(set(words)) / len(words)
    return {"ttr": round(ttr, 3), "vocd": None}


# War story / authenticity keywords (from StyleEnforcerAgent defaults)
_WAR_STORY_KEYWORDS = [
    "i built", "i broke", "we encountered", "pager duty",
    "context leak", "we built", "we broke", "i debugged",
    "i discovered", "we learned", "i spent", "we saw",
    "the gotcha", "in production", "i watched",
]

_WAR_STORY_RE = re.compile(
    "|".join(
        (r"\b" if re.match(r"\w", p) else "")
        + re.escape(p)
        + (r"\b" if re.search(r"\w$", p) else "")
        for p in _WAR_STORY_KEYWORDS
    ),
    re.IGNORECASE,
)

# Forbidden AI-tell phrases
_FORBIDDEN_PHRASES = [
    "in this post, we will explore",
    "furthermore, it is important",
    "in conclusion, we have seen",
    "it is worth mentioning",
    "as we can see from the above",
    "transitioning to",
    "let's dive in",
    "without further ado",
    "in today's fast-paced world",
    "it is important to note",
    "it goes without saying",
    "needless to say",
    "at the end of the day",
    "in this article",
    "as mentioned earlier",
    "moving forward",
]

_FORBIDDEN_RE = re.compile(
    "|".join(
        (r"\b" if re.match(r"\w", p) else "")
        + re.escape(p)
        + (r"\b" if re.search(r"\w$", p) else "")
        for p in _FORBIDDEN_PHRASES
    ),
    re.IGNORECASE,
)


def _detect_keywords(text: str) -> dict:
    """Detect war story keywords and forbidden phrases in text."""
    war_stories = [m.group() for m in _WAR_STORY_RE.finditer(text)]
    forbidden = [m.group() for m in _FORBIDDEN_RE.finditer(text)]
    return {"war_story_keywords": war_stories, "forbidden_found": forbidden}


def _analyze_text_deterministic(text: str) -> dict:
    """Run all deterministic analyses on a text sample.

    Returns a dict with burstiness, lexical diversity, and keyword data
    that can be passed to the LLM for context.
    """
    burstiness = _calculate_burstiness(text)
    lexical = _calculate_lexical_diversity(text)
    keywords = _detect_keywords(text)
    word_count = len(text.split())

    # Determine variance category from burstiness
    b = burstiness["burstiness"]
    if b >= 0.4:
        variance = "high"
    elif b >= 0.25:
        variance = "medium"
    else:
        variance = "low"

    return {
        "word_count": word_count,
        "burstiness": burstiness,
        "lexical": lexical,
        "keywords": keywords,
        "variance_category": variance,
    }


# ---------------------------------------------------------------------------
# LLM prompt for tone classification
# ---------------------------------------------------------------------------

_INFERENCE_SYSTEM_PROMPT = """You are a writing style analyst. You analyze text samples and classify their tone, voice, and style characteristics.

You will receive a text sample along with pre-computed metrics. Your job is to identify the subjective style dimensions that cannot be measured deterministically.

Respond ONLY with a JSON object in exactly this format (no markdown fences, no explanation):

{
    "formality_level": <float 0.0-1.0, 0=very casual, 1=very formal>,
    "technical_depth": <float 0.0-1.0, 0=layperson, 1=deep technical>,
    "personality": "<one of: witty, authoritative, conversational, provocative, analytical, inspirational, dry, empathetic>",
    "tone_descriptors": ["<adjective>", "<adjective>", "<adjective>"],
    "hook_style": "<one of: contrarian_challenge, contrarian_question, future_vision, problem_statement, personal_story, breaking_insight, myth_busting, shocking_stat>",
    "cta_style": "<one of: specific_question, challenge, resource_share, inspiration, community_discussion, call_to_action, none>",
    "preferred_words": ["<word>", "<word>", "<word>"],
    "avoided_words": ["<word>", "<word>", "<word>"],
    "jargon_level": "<one of: none, light, heavy>",
    "example_phrases": ["<representative phrase from text>", "<another>", "<another>"],
    "inferred_name": "<a 2-4 word name for this writing style>"
}"""


def _build_inference_prompt(text: str, metrics: dict) -> str:
    """Build the user prompt for the LLM tone classification call."""
    # Truncate very long samples to keep prompt size manageable
    sample = text[:3000] if len(text) > 3000 else text

    metrics_summary = (
        f"Pre-computed metrics:\n"
        f"- Word count: {metrics['word_count']}\n"
        f"- Avg sentence length: {metrics['burstiness']['avg_length']} words\n"
        f"- Burstiness ratio: {metrics['burstiness']['burstiness']}\n"
        f"- Sentence length variance: {metrics['variance_category']}\n"
        f"- TTR (lexical diversity): {metrics['lexical']['ttr']}\n"
        f"- War story keywords found: {', '.join(metrics['keywords']['war_story_keywords'][:5]) or 'none'}\n"
        f"- AI-tell phrases found: {', '.join(metrics['keywords']['forbidden_found'][:3]) or 'none'}\n"
    )

    return f"""Analyze this writing sample and classify its tone/style.

{metrics_summary}

--- BEGIN SAMPLE ---
{sample}
--- END SAMPLE ---

Respond with the JSON object only."""


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def _calculate_confidence(text: str) -> float:
    """Rate inference confidence based on sample characteristics."""
    word_count = len(text.split())
    if word_count < 100:
        return 0.3
    elif word_count < 500:
        return 0.6
    elif word_count < 1500:
        return 0.8
    else:
        return 0.95


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class ToneInferenceEngine:
    """Analyzes writing samples to extract tone profiles.

    Uses deterministic text metrics combined with LLM classification
    to produce a complete ToneProfile.
    """

    def __init__(self, model: str = None):
        """Initialize with a fast LLM model.

        Args:
            model: Override model name. Defaults to config.models.DEFAULT_FAST_MODEL.
        """
        self._model = model or config.models.DEFAULT_FAST_MODEL
        self._agent = BaseAgent(
            role="Tone Analyst",
            persona="Expert writing style analyst who classifies tone and voice.",
            model=self._model,
        )

    async def infer_from_text(self, sample_text: str) -> ToneProfile:
        """Analyze raw text and return an inferred ToneProfile.

        Steps:
        1. Calculate deterministic metrics (burstiness, lexical diversity, keywords)
        2. Send to LLM for subjective classification
        3. Merge deterministic + LLM results into ToneProfile
        4. Calculate confidence score

        Args:
            sample_text: The writing sample to analyze.

        Returns:
            A ToneProfile with source="inferred" and a confidence_score.
        """
        if not sample_text or len(sample_text.strip()) < 50:
            raise ValueError("Sample text must be at least 50 characters")

        # Step 1: Deterministic analysis
        metrics = _analyze_text_deterministic(sample_text)

        # Step 2: LLM classification
        prompt = _build_inference_prompt(sample_text, metrics)
        llm_response = await self._agent.call_llm_async(
            prompt,
            system_prompt=_INFERENCE_SYSTEM_PROMPT,
            temperature=0.3,
        )
        llm_data = extract_json_from_llm(llm_response, default={})

        # Step 3: Build ToneProfile from combined data
        confidence = _calculate_confidence(sample_text)
        profile = self._build_profile(metrics, llm_data, confidence)

        return profile

    async def infer_from_url(self, url: str) -> ToneProfile:
        """Fetch content from URL and analyze.

        Args:
            url: URL of a web page to analyze.

        Returns:
            A ToneProfile inferred from the page content.
        """
        import urllib.request
        from html.parser import HTMLParser

        # Simple HTML text extractor
        class _TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self._text_parts: List[str] = []
                self._skip = False

            def handle_starttag(self, tag, attrs):
                if tag in ("script", "style", "nav", "header", "footer"):
                    self._skip = True

            def handle_endtag(self, tag):
                if tag in ("script", "style", "nav", "header", "footer"):
                    self._skip = False

            def handle_data(self, data):
                if not self._skip:
                    stripped = data.strip()
                    if stripped:
                        self._text_parts.append(stripped)

            def get_text(self) -> str:
                return " ".join(self._text_parts)

        req = urllib.request.Request(url, headers={"User-Agent": "GhostWriter/2.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        extractor = _TextExtractor()
        extractor.feed(html)
        text = extractor.get_text()

        if len(text) < 50:
            raise ValueError(f"Not enough text content extracted from {url}")

        profile = await self.infer_from_text(text)
        # Record the source URL
        profile = profile.model_copy(update={"inferred_from": url, "updated_at": utc_iso()})
        return profile

    def infer_from_text_sync(self, sample_text: str) -> ToneProfile:
        """Synchronous wrapper for infer_from_text.

        For use in Streamlit UI and other sync contexts.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're inside an existing event loop (e.g. Jupyter/Streamlit)
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self.infer_from_text(sample_text))
                return future.result()
        else:
            return asyncio.run(self.infer_from_text(sample_text))

    def infer_from_url_sync(self, url: str) -> ToneProfile:
        """Synchronous wrapper for infer_from_url."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self.infer_from_url(url))
                return future.result()
        else:
            return asyncio.run(self.infer_from_url(url))

    def _build_profile(self, metrics: dict, llm_data: dict, confidence: float) -> ToneProfile:
        """Combine deterministic metrics and LLM classification into a ToneProfile."""
        burst = metrics["burstiness"]
        keywords = metrics["keywords"]

        # Name from LLM or fallback
        name = llm_data.get("inferred_name", "Inferred Style")

        # Description from tone descriptors
        descriptors = llm_data.get("tone_descriptors", [])
        desc_text = ", ".join(descriptors[:3]) if descriptors else "inferred style"
        description = f"Inferred profile: {desc_text}."

        # Sentence style from deterministic data
        sentence_style = SentenceStyle(
            avg_length_target=int(burst["avg_length"]) if burst["avg_length"] else 18,
            burstiness_target=burst["burstiness"] if burst["burstiness"] else 0.45,
            length_variance=metrics["variance_category"],
        )

        # Vocabulary from LLM
        vocabulary_preferences = VocabularyPreferences(
            preferred_words=llm_data.get("preferred_words", [])[:10],
            avoided_words=llm_data.get("avoided_words", [])[:10],
            jargon_level=llm_data.get("jargon_level", "light"),
        )

        # War story keywords: use detected ones if present, else empty
        war_stories = keywords.get("war_story_keywords", [])
        # Deduplicate while preserving order
        seen = set()
        unique_war_stories = []
        for kw in war_stories:
            lower = kw.lower()
            if lower not in seen:
                seen.add(lower)
                unique_war_stories.append(kw)

        # Forbidden phrases: combine global defaults with any detected
        forbidden = list(_FORBIDDEN_PHRASES)

        return ToneProfile(
            name=name,
            description=description,
            formality_level=llm_data.get("formality_level", 0.5),
            technical_depth=llm_data.get("technical_depth", 0.5),
            personality=llm_data.get("personality", "conversational"),
            sentence_style=sentence_style,
            vocabulary_preferences=vocabulary_preferences,
            hook_style=llm_data.get("hook_style", "problem_statement"),
            cta_style=llm_data.get("cta_style", "none"),
            example_phrases=llm_data.get("example_phrases", [])[:5],
            forbidden_phrases=forbidden,
            war_story_keywords=unique_war_stories,
            source="inferred",
            inferred_from=None,
            confidence_score=confidence,
            created_at=utc_iso(),
        )
