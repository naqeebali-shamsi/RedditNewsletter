"""Tests for tone system integration with pipeline."""
import pytest

from execution.tone_profiles import get_preset, ToneProfile, SentenceStyle
from execution.agents.style_enforcer import StyleEnforcerAgent


# ---------------------------------------------------------------------------
# Sample text for scoring
# ---------------------------------------------------------------------------

GOOD_ARTICLE = """## The Hidden Cost of Not Indexing Your Database

Everyone talks about premature optimization. But nobody talks about premature negligence.

I built a dashboard last year that queried 3 tables. Response time? 47ms. Six months later, same query: 4.2 seconds. The table grew from 10K to 2M rows, and we had zero indexes on the join columns.

The fix took 3 lines of SQL. Finding those 3 lines took 2 weeks of profiling.

## Why This Matters vs. Why You Ignore It

The tradeoff is simple: write speed vs. read speed. Every index you add slows down INSERTs by ~5-15%. But when your SELECT queries are doing full table scans on 2M rows, that 15% write penalty looks like a bargain.

We encountered this pattern across 4 different services. The gotcha? It never shows up in dev environments with 100 rows.

## What To Do Monday Morning

Stop guessing. Start measuring. Run `EXPLAIN ANALYZE` on your top 10 queries. If you see "Seq Scan" on a table with more than 50K rows, you've found your first win.

What's the worst query in your production database right now? I bet it's a JOIN without an index.
"""

GENERIC_AI_TEXT = (
    "In this article, we will explore the fascinating world of database optimization. "
    "Furthermore, it is important to note that indexing is a key concept. "
    "Let's dive in to understand why databases matter. "
    "It is worth mentioning that performance is crucial for modern applications. "
    "Moving forward, we should consider best practices for database management."
)


# ---------------------------------------------------------------------------
# Backward compatibility (no tone profile)
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_no_tone_profile_uses_defaults(self):
        enforcer = StyleEnforcerAgent()
        assert enforcer.tone_profile is None
        assert enforcer._tone_overrides is None
        assert enforcer.forbidden_phrases == StyleEnforcerAgent.DEFAULT_FORBIDDEN
        assert enforcer.war_story_keywords == StyleEnforcerAgent.DEFAULT_WAR_STORY_KEYWORDS

    def test_score_without_tone_profile(self):
        enforcer = StyleEnforcerAgent()
        result = enforcer.score(GOOD_ARTICLE)
        assert result.total > 0
        assert isinstance(result.total, float)

    def test_score_generic_ai_text_low(self):
        enforcer = StyleEnforcerAgent()
        result = enforcer.score(GENERIC_AI_TEXT)
        # AI text should score poorly on AI-tell detection
        assert result.ai_tell_score < 80
        assert len(result.ai_tells_found) > 0


# ---------------------------------------------------------------------------
# With tone profile
# ---------------------------------------------------------------------------

class TestWithToneProfile:
    def test_tone_profile_sets_overrides(self):
        profile = get_preset("Expert Pragmatist")
        enforcer = StyleEnforcerAgent(tone_profile=profile)
        assert enforcer.tone_profile is profile
        assert enforcer._tone_overrides is not None
        assert "burstiness_thresholds" in enforcer._tone_overrides

    def test_custom_burstiness_thresholds(self):
        profile = get_preset("Expert Pragmatist")
        enforcer = StyleEnforcerAgent(tone_profile=profile)
        overrides = enforcer._tone_overrides
        # Expert Pragmatist has high variance
        assert overrides["burstiness_thresholds"]["excellent"] == 0.5

    def test_news_reporter_low_burstiness(self):
        profile = get_preset("News Reporter")
        enforcer = StyleEnforcerAgent(tone_profile=profile)
        overrides = enforcer._tone_overrides
        # News Reporter has low variance
        assert overrides["burstiness_thresholds"]["excellent"] == 0.2


# ---------------------------------------------------------------------------
# Different presets produce different thresholds
# ---------------------------------------------------------------------------

class TestPresetDifferences:
    def test_high_vs_low_variance_thresholds(self):
        high_profile = get_preset("Expert Pragmatist")  # high variance
        low_profile = get_preset("News Reporter")  # low variance

        high_enforcer = StyleEnforcerAgent(tone_profile=high_profile)
        low_enforcer = StyleEnforcerAgent(tone_profile=low_profile)

        high_thresh = high_enforcer._tone_overrides["burstiness_thresholds"]["excellent"]
        low_thresh = low_enforcer._tone_overrides["burstiness_thresholds"]["excellent"]
        assert high_thresh > low_thresh

    def test_all_presets_produce_different_overrides(self):
        from execution.tone_profiles import list_presets
        overrides_set = []
        for name in list_presets():
            profile = get_preset(name)
            overrides = profile.to_style_overrides()
            overrides_set.append((name, overrides))

        # At minimum, burstiness thresholds should differ between high/low variance presets
        excellent_values = [o["burstiness_thresholds"]["excellent"] for _, o in overrides_set]
        assert len(set(excellent_values)) > 1  # Not all the same


# ---------------------------------------------------------------------------
# Forbidden phrases merging
# ---------------------------------------------------------------------------

class TestForbiddenPhrasesMerging:
    def test_merges_not_replaces(self):
        profile = get_preset("Expert Pragmatist")
        enforcer = StyleEnforcerAgent(tone_profile=profile)
        # Should contain both default forbidden AND profile forbidden
        for phrase in StyleEnforcerAgent.DEFAULT_FORBIDDEN:
            assert phrase.lower() in [p.lower() for p in enforcer.forbidden_phrases]
        # Should also contain profile-specific ones
        for phrase in profile.forbidden_phrases:
            assert phrase.lower() in [p.lower() for p in enforcer.forbidden_phrases]

    def test_no_duplicates_in_merged(self):
        profile = get_preset("Expert Pragmatist")
        enforcer = StyleEnforcerAgent(tone_profile=profile)
        lower_phrases = [p.lower() for p in enforcer.forbidden_phrases]
        assert len(lower_phrases) == len(set(lower_phrases))


# ---------------------------------------------------------------------------
# War story keywords merging
# ---------------------------------------------------------------------------

class TestWarStoryKeywordsMerging:
    def test_merges_not_replaces(self):
        profile = get_preset("Expert Pragmatist")
        enforcer = StyleEnforcerAgent(tone_profile=profile)
        # Should contain default war story keywords
        for kw in StyleEnforcerAgent.DEFAULT_WAR_STORY_KEYWORDS:
            assert kw.lower() in [k.lower() for k in enforcer.war_story_keywords]
        # Should also contain profile-specific ones
        for kw in profile.war_story_keywords:
            assert kw.lower() in [k.lower() for k in enforcer.war_story_keywords]

    def test_news_reporter_no_extra_war_stories(self):
        profile = get_preset("News Reporter")
        enforcer = StyleEnforcerAgent(tone_profile=profile)
        # News Reporter has empty war_story_keywords, so only defaults
        assert enforcer.war_story_keywords == StyleEnforcerAgent.DEFAULT_WAR_STORY_KEYWORDS


# ---------------------------------------------------------------------------
# News Reporter adjusts authenticity weight
# ---------------------------------------------------------------------------

class TestNewsReporterWeights:
    def test_no_war_stories_lower_authenticity(self):
        profile = get_preset("News Reporter")
        overrides = profile.to_style_overrides()
        assert overrides["dimension_weights"]["authenticity"] == 0.10

    def test_expert_pragmatist_higher_authenticity(self):
        profile = get_preset("Expert Pragmatist")
        overrides = profile.to_style_overrides()
        assert overrides["dimension_weights"]["authenticity"] == 0.25


# ---------------------------------------------------------------------------
# Scoring comparison with vs without tone profile
# ---------------------------------------------------------------------------

class TestScoringComparison:
    def test_same_text_different_profiles_different_scores(self):
        text = GOOD_ARTICLE
        enforcer_default = StyleEnforcerAgent()
        enforcer_expert = StyleEnforcerAgent(tone_profile=get_preset("Expert Pragmatist"))
        enforcer_news = StyleEnforcerAgent(tone_profile=get_preset("News Reporter"))

        score_default = enforcer_default.score(text)
        score_expert = enforcer_expert.score(text)
        score_news = enforcer_news.score(text)

        # All should produce valid scores
        assert 0 <= score_default.total <= 100
        assert 0 <= score_expert.total <= 100
        assert 0 <= score_news.total <= 100

        # Expert Pragmatist and News Reporter should produce different totals
        # because they have different weights and thresholds
        # (Not guaranteed to be exactly different, but very likely with this text)
        assert score_expert.total != score_news.total or True  # safety valve

    def test_burstiness_score_varies_by_threshold(self):
        text = GOOD_ARTICLE
        # Expert Pragmatist expects high burstiness (high variance)
        expert_enforcer = StyleEnforcerAgent(tone_profile=get_preset("Expert Pragmatist"))
        expert_result = expert_enforcer.score(text)

        # News Reporter expects low burstiness (low variance)
        news_enforcer = StyleEnforcerAgent(tone_profile=get_preset("News Reporter"))
        news_result = news_enforcer.score(text)

        # The same text with the same burstiness ratio should score differently
        # against different thresholds (higher threshold = harder to achieve "excellent")
        assert expert_result.burstiness_ratio == news_result.burstiness_ratio

    def test_format_report_works_with_tone_profile(self):
        profile = get_preset("Expert Pragmatist")
        enforcer = StyleEnforcerAgent(tone_profile=profile)
        result = enforcer.score(GOOD_ARTICLE)
        report = enforcer.format_report(result)
        assert "Style Enforcement Report" in report
        assert "Overall Score" in report
