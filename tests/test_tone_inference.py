"""Tests for tone inference engine."""
import pytest
import json
from unittest.mock import patch, AsyncMock, MagicMock

from execution.tone_inference import (
    ToneInferenceEngine,
    _calculate_burstiness,
    _calculate_lexical_diversity,
    _detect_keywords,
    _analyze_text_deterministic,
    _calculate_confidence,
)
from execution.tone_profiles import ToneProfile


# ---------------------------------------------------------------------------
# Sample texts for testing
# ---------------------------------------------------------------------------

SHORT_TEXT = "This is short."

MEDIUM_TEXT = (
    "We burned $12K in GPU credits before we found the real bottleneck. "
    "The fix took 3 lines. Finding those 3 lines took 2 weeks. "
    "Everyone ships RAG. Nobody ships RAG that works in production. "
    "I built a prototype in a weekend. It broke on Monday. "
    "The gotcha was memory pressure under concurrent load. "
    "Here is a sentence of medium length for testing purposes. "
    "Short one. Another sentence to pad it out a bit more for testing. "
    "We encountered a race condition that only appeared at scale."
)

LONG_TEXT = MEDIUM_TEXT * 5  # ~400+ words

FORMAL_AI_TEXT = (
    "In this article, we will explore the fascinating world of artificial intelligence. "
    "Furthermore, it is important to note that machine learning has made great strides. "
    "Let's dive in to understand the key concepts. "
    "It is worth mentioning that deep learning is a subset of machine learning. "
    "As we can see from the above, the field is rapidly evolving. "
    "Moving forward, we should consider the implications of these advances."
)


# ---------------------------------------------------------------------------
# Deterministic analysis functions
# ---------------------------------------------------------------------------

class TestCalculateBurstiness:
    def test_returns_dict_keys(self):
        result = _calculate_burstiness(MEDIUM_TEXT)
        assert "burstiness" in result
        assert "avg_length" in result
        assert "std_dev" in result
        assert "count" in result

    def test_short_text_returns_zeros(self):
        result = _calculate_burstiness("Too short.")
        assert result["burstiness"] == 0.0
        assert result["count"] == 0

    def test_medium_text_has_positive_burstiness(self):
        result = _calculate_burstiness(MEDIUM_TEXT)
        assert result["burstiness"] > 0.0
        assert result["count"] >= 3

    def test_avg_length_is_reasonable(self):
        result = _calculate_burstiness(MEDIUM_TEXT)
        assert 5 <= result["avg_length"] <= 30

    def test_uniform_sentences_low_burstiness(self):
        uniform = "This has ten words in each sentence here now. " * 10
        result = _calculate_burstiness(uniform)
        assert result["burstiness"] < 0.2


class TestCalculateLexicalDiversity:
    def test_returns_dict_keys(self):
        result = _calculate_lexical_diversity(MEDIUM_TEXT)
        assert "ttr" in result

    def test_empty_text(self):
        result = _calculate_lexical_diversity("")
        assert result["ttr"] is None or result["ttr"] == 0

    def test_ttr_between_zero_and_one(self):
        result = _calculate_lexical_diversity(MEDIUM_TEXT)
        if result["ttr"] is not None:
            assert 0.0 <= result["ttr"] <= 1.0

    def test_repetitive_text_low_ttr(self):
        repetitive = "the the the the the the the the the the " * 10
        result = _calculate_lexical_diversity(repetitive)
        if result["ttr"] is not None:
            assert result["ttr"] < 0.2


class TestDetectKeywords:
    def test_detects_war_story_keywords(self):
        text = "I built a system and we encountered a bug in production."
        result = _detect_keywords(text)
        assert len(result["war_story_keywords"]) >= 2

    def test_detects_forbidden_phrases(self):
        result = _detect_keywords(FORMAL_AI_TEXT)
        assert len(result["forbidden_found"]) >= 3

    def test_no_keywords_in_clean_text(self):
        text = "The algorithm performs matrix multiplication efficiently."
        result = _detect_keywords(text)
        assert len(result["war_story_keywords"]) == 0
        assert len(result["forbidden_found"]) == 0


class TestAnalyzeTextDeterministic:
    def test_returns_all_sections(self):
        result = _analyze_text_deterministic(MEDIUM_TEXT)
        assert "word_count" in result
        assert "burstiness" in result
        assert "lexical" in result
        assert "keywords" in result
        assert "variance_category" in result

    def test_word_count_is_positive(self):
        result = _analyze_text_deterministic(MEDIUM_TEXT)
        assert result["word_count"] > 0

    def test_variance_category_values(self):
        result = _analyze_text_deterministic(MEDIUM_TEXT)
        assert result["variance_category"] in ("high", "medium", "low")

    def test_high_burstiness_maps_to_high_variance(self):
        # Text with very varied sentence lengths
        text = (
            "Short. "
            "This is a medium-length sentence with some extra words for padding. "
            "Tiny. "
            "This sentence is quite a bit longer with many additional words that go on and on and on to create high variance in length. "
            "Yes. "
            "Another medium sentence here with a few words."
        )
        result = _analyze_text_deterministic(text)
        # With such varied lengths, burstiness should be high
        assert result["burstiness"]["burstiness"] > 0.3


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

class TestCalculateConfidence:
    def test_short_text_low_confidence(self):
        text = " ".join(["word"] * 50)
        assert _calculate_confidence(text) == 0.3

    def test_medium_text_moderate_confidence(self):
        text = " ".join(["word"] * 300)
        assert _calculate_confidence(text) == 0.6

    def test_long_text_high_confidence(self):
        text = " ".join(["word"] * 1000)
        assert _calculate_confidence(text) == 0.8

    def test_very_long_text_highest_confidence(self):
        text = " ".join(["word"] * 2000)
        assert _calculate_confidence(text) == 0.95

    def test_boundary_100_words(self):
        text = " ".join(["word"] * 100)
        assert _calculate_confidence(text) == 0.6

    def test_boundary_500_words(self):
        text = " ".join(["word"] * 500)
        assert _calculate_confidence(text) == 0.8

    def test_boundary_1500_words(self):
        text = " ".join(["word"] * 1500)
        assert _calculate_confidence(text) == 0.95


# ---------------------------------------------------------------------------
# ToneInferenceEngine._build_profile
# ---------------------------------------------------------------------------

class TestBuildProfile:
    @pytest.fixture
    def engine(self):
        with patch("execution.tone_inference.BaseAgent"):
            return ToneInferenceEngine(model="test-model")

    def test_build_profile_with_full_llm_data(self, engine):
        metrics = _analyze_text_deterministic(MEDIUM_TEXT)
        llm_data = {
            "formality_level": 0.4,
            "technical_depth": 0.8,
            "personality": "witty",
            "tone_descriptors": ["bold", "technical", "direct"],
            "hook_style": "contrarian_challenge",
            "cta_style": "specific_question",
            "preferred_words": ["tradeoff", "shipped"],
            "avoided_words": ["leverage"],
            "jargon_level": "heavy",
            "example_phrases": ["We shipped it.", "The tradeoff was clear."],
            "inferred_name": "Bold Engineer",
        }
        profile = engine._build_profile(metrics, llm_data, 0.8)
        assert isinstance(profile, ToneProfile)
        assert profile.name == "Bold Engineer"
        assert profile.formality_level == 0.4
        assert profile.technical_depth == 0.8
        assert profile.source == "inferred"
        assert profile.confidence_score == 0.8
        assert "tradeoff" in profile.vocabulary_preferences.preferred_words

    def test_build_profile_with_empty_llm_data(self, engine):
        metrics = _analyze_text_deterministic(MEDIUM_TEXT)
        profile = engine._build_profile(metrics, {}, 0.6)
        assert isinstance(profile, ToneProfile)
        assert profile.name == "Inferred Style"
        assert profile.personality == "conversational"
        assert profile.source == "inferred"

    def test_build_profile_war_stories_deduplication(self, engine):
        text = "I built something and I built another thing."
        metrics = _analyze_text_deterministic(text)
        profile = engine._build_profile(metrics, {}, 0.5)
        # "I built" should appear only once even if detected twice
        lower_keywords = [kw.lower() for kw in profile.war_story_keywords]
        assert len(lower_keywords) == len(set(lower_keywords))

    def test_build_profile_forbidden_phrases_populated(self, engine):
        metrics = _analyze_text_deterministic(MEDIUM_TEXT)
        profile = engine._build_profile(metrics, {}, 0.5)
        assert len(profile.forbidden_phrases) > 0


# ---------------------------------------------------------------------------
# Async inference (mocked LLM)
# ---------------------------------------------------------------------------

class TestInferFromText:
    @pytest.fixture
    def mock_engine(self):
        with patch("execution.tone_inference.BaseAgent") as MockAgent:
            instance = MockAgent.return_value
            instance.call_llm_async = AsyncMock(return_value=json.dumps({
                "formality_level": 0.5,
                "technical_depth": 0.7,
                "personality": "authoritative",
                "tone_descriptors": ["sharp", "direct"],
                "hook_style": "contrarian_challenge",
                "cta_style": "challenge",
                "preferred_words": ["shipped"],
                "avoided_words": ["synergy"],
                "jargon_level": "light",
                "example_phrases": ["Ship it."],
                "inferred_name": "Sharp Analyst",
            }))
            engine = ToneInferenceEngine.__new__(ToneInferenceEngine)
            engine._model = "test-model"
            engine._agent = instance
            yield engine

    @pytest.mark.asyncio
    async def test_infer_from_text_returns_profile(self, mock_engine):
        profile = await mock_engine.infer_from_text(MEDIUM_TEXT)
        assert isinstance(profile, ToneProfile)
        assert profile.source == "inferred"

    @pytest.mark.asyncio
    async def test_infer_from_text_too_short_raises(self, mock_engine):
        with pytest.raises(ValueError, match="at least 50 characters"):
            await mock_engine.infer_from_text("Too short")

    @pytest.mark.asyncio
    async def test_infer_from_text_empty_raises(self, mock_engine):
        with pytest.raises(ValueError):
            await mock_engine.infer_from_text("")

    @pytest.mark.asyncio
    async def test_infer_calls_llm(self, mock_engine):
        await mock_engine.infer_from_text(MEDIUM_TEXT)
        mock_engine._agent.call_llm_async.assert_called_once()


# ---------------------------------------------------------------------------
# URL inference (mocked HTTP + LLM)
# ---------------------------------------------------------------------------

class TestInferFromUrl:
    @pytest.mark.asyncio
    async def test_infer_from_url_with_mocked_http(self):
        html_content = (
            "<html><body>"
            "<p>" + MEDIUM_TEXT + "</p>"
            "</body></html>"
        )
        mock_response = MagicMock()
        mock_response.read.return_value = html_content.encode("utf-8")
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("execution.tone_inference.BaseAgent") as MockAgent:
            instance = MockAgent.return_value
            instance.call_llm_async = AsyncMock(return_value=json.dumps({
                "formality_level": 0.5,
                "technical_depth": 0.5,
                "personality": "conversational",
                "hook_style": "problem_statement",
                "cta_style": "none",
                "inferred_name": "Web Style",
            }))
            engine = ToneInferenceEngine.__new__(ToneInferenceEngine)
            engine._model = "test-model"
            engine._agent = instance

            with patch("urllib.request.urlopen", return_value=mock_response):
                profile = await engine.infer_from_url("https://example.com/article")

            assert isinstance(profile, ToneProfile)
            assert profile.inferred_from == "https://example.com/article"
            assert profile.source == "inferred"

    @pytest.mark.asyncio
    async def test_infer_from_url_too_little_content(self):
        html_content = "<html><body><p>Hi</p></body></html>"
        mock_response = MagicMock()
        mock_response.read.return_value = html_content.encode("utf-8")
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("execution.tone_inference.BaseAgent") as MockAgent:
            engine = ToneInferenceEngine.__new__(ToneInferenceEngine)
            engine._model = "test-model"
            engine._agent = MockAgent.return_value

            with patch("urllib.request.urlopen", return_value=mock_response):
                with pytest.raises(ValueError, match="Not enough text"):
                    await engine.infer_from_url("https://example.com/empty")
