"""Tests for user preferences and adaptive learning."""
import json
import pytest
from pathlib import Path

from execution.user_preferences import (
    UserPreferences,
    PreferencesData,
    LearnedAdjustments,
    _formality_score,
    _clamp,
    _tokenize,
)
from execution.tone_profiles import ToneProfile, get_preset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def prefs(tmp_path):
    """Create a UserPreferences instance with a temp directory for storage."""
    return UserPreferences(path=tmp_path / "prefs.json")


@pytest.fixture
def prefs_with_feedback(tmp_path):
    """UserPreferences with 5 feedback entries to activate learning."""
    p = UserPreferences(path=tmp_path / "prefs.json")
    for i in range(5):
        p.log_feedback(f"article-{i}", "accepted")
    return p


# ---------------------------------------------------------------------------
# PreferencesData model
# ---------------------------------------------------------------------------

class TestPreferencesData:
    def test_default_values(self):
        data = PreferencesData()
        assert data.user_id == "default"
        assert data.active_profile == "Expert Pragmatist"
        assert data.custom_profiles == {}
        assert data.feedback_log == []
        assert data.feedback_count == 0
        assert data.learning_active is False

    def test_learned_adjustments_defaults(self):
        adj = LearnedAdjustments()
        assert adj.formality_delta == 0.0
        assert adj.technical_depth_delta == 0.0
        assert adj.avg_sentence_length_delta == 0
        assert adj.burstiness_delta == 0.0

    def test_learned_adjustments_bounds(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            LearnedAdjustments(formality_delta=0.6)
        with pytest.raises(ValidationError):
            LearnedAdjustments(burstiness_delta=0.5)
        with pytest.raises(ValidationError):
            LearnedAdjustments(avg_sentence_length_delta=15)


# ---------------------------------------------------------------------------
# Default preferences creation and persistence
# ---------------------------------------------------------------------------

class TestPreferencesCreation:
    def test_creates_default_preferences(self, prefs):
        profile = prefs.get_active_profile()
        assert profile.name == "Expert Pragmatist"

    def test_file_not_created_until_save(self, tmp_path):
        path = tmp_path / "prefs.json"
        prefs = UserPreferences(path=path)
        # File not created by just loading
        assert not path.exists()
        # Trigger a save
        prefs.set_active_profile("Thought Leader")
        assert path.exists()

    def test_save_and_reload(self, tmp_path):
        path = tmp_path / "prefs.json"
        prefs1 = UserPreferences(path=path)
        prefs1.set_active_profile("Thought Leader")

        # Reload from disk
        prefs2 = UserPreferences(path=path)
        assert prefs2.get_active_profile().name == "Thought Leader"

    def test_corrupted_file_falls_back_to_defaults(self, tmp_path):
        path = tmp_path / "prefs.json"
        path.write_text("NOT VALID JSON!!!", encoding="utf-8")
        prefs = UserPreferences(path=path)
        assert prefs.get_active_profile().name == "Expert Pragmatist"


# ---------------------------------------------------------------------------
# Profile switching
# ---------------------------------------------------------------------------

class TestProfileSwitching:
    def test_switch_to_valid_preset(self, prefs):
        prefs.set_active_profile("Thought Leader")
        assert prefs.get_active_profile().name == "Thought Leader"

    def test_switch_to_invalid_preset_raises(self, prefs):
        with pytest.raises(KeyError, match="not found"):
            prefs.set_active_profile("Nonexistent Profile")

    def test_switch_logs_history(self, prefs):
        prefs.set_active_profile("Thought Leader")
        prefs.set_active_profile("News Reporter")
        assert len(prefs._data.preference_history) == 2

    def test_switch_to_same_profile_no_history(self, prefs):
        prefs.set_active_profile("Expert Pragmatist")
        assert len(prefs._data.preference_history) == 0


# ---------------------------------------------------------------------------
# Custom profile management
# ---------------------------------------------------------------------------

class TestCustomProfiles:
    def test_save_custom_profile(self, prefs):
        profile = ToneProfile(
            name="My Custom Voice",
            description="A custom voice for tests.",
            formality_level=0.3,
            technical_depth=0.4,
            personality="friendly",
            hook_style="personal_story",
            cta_style="community_discussion",
            source="custom",
        )
        prefs.save_custom_profile(profile)
        assert "My Custom Voice" in prefs._data.custom_profiles

    def test_switch_to_custom_profile(self, prefs):
        profile = ToneProfile(
            name="My Custom Voice",
            description="A custom voice for tests.",
            formality_level=0.3,
            technical_depth=0.4,
            personality="friendly",
            hook_style="personal_story",
            cta_style="community_discussion",
            source="custom",
        )
        prefs.save_custom_profile(profile)
        prefs.set_active_profile("My Custom Voice")
        active = prefs.get_active_profile()
        assert active.name == "My Custom Voice"
        assert active.formality_level == 0.3

    def test_delete_custom_profile(self, prefs):
        profile = ToneProfile(
            name="Deletable",
            description="Will be deleted.",
            formality_level=0.5,
            technical_depth=0.5,
            personality="witty",
            hook_style="problem_statement",
            cta_style="none",
            source="custom",
        )
        prefs.save_custom_profile(profile)
        prefs.delete_custom_profile("Deletable")
        assert "Deletable" not in prefs._data.custom_profiles

    def test_delete_nonexistent_custom_raises(self, prefs):
        with pytest.raises(KeyError, match="not found"):
            prefs.delete_custom_profile("Does Not Exist")

    def test_delete_active_custom_falls_back_to_default(self, prefs):
        profile = ToneProfile(
            name="ActiveCustom",
            description="Currently active custom.",
            formality_level=0.5,
            technical_depth=0.5,
            personality="witty",
            hook_style="problem_statement",
            cta_style="none",
            source="custom",
        )
        prefs.save_custom_profile(profile)
        prefs.set_active_profile("ActiveCustom")
        assert prefs.get_active_profile().name == "ActiveCustom"

        prefs.delete_custom_profile("ActiveCustom")
        assert prefs._data.active_profile == "Expert Pragmatist"

    def test_custom_profile_persists_across_reload(self, tmp_path):
        path = tmp_path / "prefs.json"
        prefs1 = UserPreferences(path=path)
        profile = ToneProfile(
            name="Persistent Custom",
            description="Persists across reload.",
            formality_level=0.7,
            technical_depth=0.6,
            personality="analytical",
            hook_style="problem_statement",
            cta_style="resource_share",
            source="custom",
        )
        prefs1.save_custom_profile(profile)

        prefs2 = UserPreferences(path=path)
        assert "Persistent Custom" in prefs2._data.custom_profiles


# ---------------------------------------------------------------------------
# Feedback logging
# ---------------------------------------------------------------------------

class TestFeedbackLogging:
    def test_log_feedback_increments_count(self, prefs):
        prefs.log_feedback("article-1", "accepted")
        assert prefs._data.feedback_count == 1
        prefs.log_feedback("article-2", "edited")
        assert prefs._data.feedback_count == 2

    def test_log_feedback_appends_to_log(self, prefs):
        prefs.log_feedback("article-1", "accepted")
        assert len(prefs._data.feedback_log) == 1
        entry = prefs._data.feedback_log[0]
        assert entry["article_id"] == "article-1"
        assert entry["action"] == "accepted"

    def test_learning_inactive_before_threshold(self, prefs):
        for i in range(4):
            prefs.log_feedback(f"article-{i}", "accepted")
        assert prefs._data.learning_active is False

    def test_learning_activates_at_threshold(self, prefs):
        for i in range(5):
            prefs.log_feedback(f"article-{i}", "accepted")
        assert prefs._data.learning_active is True

    def test_learning_stays_active_after_threshold(self, prefs):
        for i in range(10):
            prefs.log_feedback(f"article-{i}", "accepted")
        assert prefs._data.learning_active is True

    def test_feedback_stats(self, prefs):
        prefs.log_feedback("a1", "accepted")
        prefs.log_feedback("a2", "edited")
        prefs.log_feedback("a3", "rejected")
        prefs.log_feedback("a4", "accepted")
        stats = prefs.get_feedback_stats()
        assert stats["total"] == 4
        assert stats["accepted"] == 2
        assert stats["edited"] == 1
        assert stats["rejected"] == 1

    def test_feedback_stats_empty(self, prefs):
        stats = prefs.get_feedback_stats()
        assert stats["total"] == 0


# ---------------------------------------------------------------------------
# Learning from edits
# ---------------------------------------------------------------------------

class TestLearnFromEdits:
    def test_edit_with_shorter_sentences_adjusts_length(self, prefs_with_feedback):
        # Original: long sentences; Edited: shorter sentences
        original = (
            "This is a really long sentence that goes on and on with many words and clauses. "
            "Another extremely lengthy sentence with multiple subordinate clauses that continues. "
            "Yet another verbose sentence that is unnecessarily long and could be shortened. "
            "The final long sentence in this paragraph that also has too many words in it."
        )
        edited = (
            "Short fix. Quick change. Done now. "
            "Three words here. Four words too. "
            "Snappy and direct. Clean and simple. "
            "Short is better. Get to the point."
        )
        prefs_with_feedback.log_feedback("art-1", "edited", original=original, edited=edited)
        adj = prefs_with_feedback._data.learned_adjustments
        # Edited text has shorter sentences, so delta should be negative
        assert adj.avg_sentence_length_delta <= 0

    def test_edit_with_longer_sentences_adjusts_length(self, prefs_with_feedback):
        original = (
            "Short fix. Quick change. Done now. "
            "Three words here. Four words too. "
            "Snappy and direct. Clean and simple. "
            "Short is better. Get to the point."
        )
        edited = (
            "This is a really long sentence that goes on and on with many words and clauses. "
            "Another extremely lengthy sentence with multiple subordinate clauses that continues. "
            "Yet another verbose sentence that is unnecessarily long and could be shortened. "
            "The final long sentence in this paragraph that also has too many words in it."
        )
        prefs_with_feedback.log_feedback("art-2", "edited", original=original, edited=edited)
        adj = prefs_with_feedback._data.learned_adjustments
        assert adj.avg_sentence_length_delta >= 0

    def test_edit_before_learning_active_no_adjustment(self, prefs):
        original = "Short text. Very short indeed. Nothing much here."
        edited = (
            "This is an extremely long and detailed text with many words that goes on and on. "
            "This second sentence is also quite long with a lot of detail and explanation. "
            "The third sentence continues the pattern of being unnecessarily verbose and long."
        )
        prefs.log_feedback("art-1", "edited", original=original, edited=edited)
        adj = prefs._data.learned_adjustments
        assert adj.avg_sentence_length_delta == 0  # learning not active yet

    def test_edit_without_both_texts_no_learning(self, prefs_with_feedback):
        prefs_with_feedback.log_feedback("art-1", "edited", original="some text")
        adj = prefs_with_feedback._data.learned_adjustments
        assert adj.avg_sentence_length_delta == 0

    def test_too_few_sentences_no_learning(self, prefs_with_feedback):
        prefs_with_feedback.log_feedback(
            "art-1", "edited",
            original="One sentence only.",
            edited="Also just one."
        )
        adj = prefs_with_feedback._data.learned_adjustments
        assert adj.avg_sentence_length_delta == 0

    def test_formality_learning_from_formal_edits(self, prefs_with_feedback):
        # Informal original -> formal edited
        original = (
            "Gonna be cool stuff here basically. "
            "Yeah totally awesome things going on. "
            "Super neat and kinda sorta interesting. "
            "Pretty huge bunch of guys here working."
        )
        edited = (
            "Therefore we must consequently facilitate this implementation. "
            "Furthermore the endeavor demonstrates considerable merit. "
            "Moreover the aforementioned approach pertains to key objectives. "
            "Subsequently the utilization hereby warrants further investigation."
        )
        prefs_with_feedback.log_feedback("art-formal", "edited", original=original, edited=edited)
        adj = prefs_with_feedback._data.learned_adjustments
        # Formality should increase (positive delta)
        assert adj.formality_delta > 0


# ---------------------------------------------------------------------------
# get_effective_profile
# ---------------------------------------------------------------------------

class TestEffectiveProfile:
    def test_no_learning_returns_base_profile(self, prefs):
        profile = prefs.get_effective_profile()
        base = prefs.get_active_profile()
        assert profile.formality_level == base.formality_level

    def test_with_adjustments_returns_modified_profile(self, tmp_path):
        path = tmp_path / "prefs.json"
        prefs = UserPreferences(path=path)
        # Manually activate learning and set adjustments
        prefs._data.learning_active = True
        prefs._data.learned_adjustments = LearnedAdjustments(
            formality_delta=0.1,
            avg_sentence_length_delta=3,
        )
        profile = prefs.get_effective_profile()
        base = prefs.get_active_profile()
        assert profile.formality_level == pytest.approx(base.formality_level + 0.1, abs=0.01)
        assert profile.sentence_style.avg_length_target == base.sentence_style.avg_length_target + 3

    def test_adjustments_clamped_to_valid_range(self, tmp_path):
        path = tmp_path / "prefs.json"
        prefs = UserPreferences(path=path)
        prefs._data.learning_active = True
        prefs._data.learned_adjustments = LearnedAdjustments(
            formality_delta=0.5,  # Would push Expert Pragmatist (0.6) to 1.1
        )
        profile = prefs.get_effective_profile()
        assert profile.formality_level <= 1.0

    def test_zero_adjustments_returns_base(self, tmp_path):
        path = tmp_path / "prefs.json"
        prefs = UserPreferences(path=path)
        prefs._data.learning_active = True
        # All deltas are zero by default
        profile = prefs.get_effective_profile()
        base = prefs.get_active_profile()
        assert profile.formality_level == base.formality_level


# ---------------------------------------------------------------------------
# Reset learning
# ---------------------------------------------------------------------------

class TestResetLearning:
    def test_reset_clears_adjustments(self, prefs_with_feedback):
        # Make some edits to create adjustments
        original = (
            "Gonna be cool stuff here basically. "
            "Yeah totally awesome things going on. "
            "Super neat and kinda sorta interesting. "
            "Pretty huge bunch of guys here working."
        )
        edited = (
            "Therefore we must consequently facilitate implementation now. "
            "Furthermore the endeavor demonstrates considerable merit now. "
            "Moreover the aforementioned approach pertains to objectives now. "
            "Subsequently the utilization hereby warrants investigation now."
        )
        prefs_with_feedback.log_feedback("art-1", "edited", original=original, edited=edited)
        prefs_with_feedback.reset_learning()
        adj = prefs_with_feedback._data.learned_adjustments
        assert adj.formality_delta == 0.0
        assert adj.technical_depth_delta == 0.0
        assert adj.avg_sentence_length_delta == 0
        assert adj.burstiness_delta == 0.0

    def test_reset_preserves_feedback_log(self, prefs_with_feedback):
        log_before = len(prefs_with_feedback._data.feedback_log)
        prefs_with_feedback.reset_learning()
        assert len(prefs_with_feedback._data.feedback_log) == log_before


# ---------------------------------------------------------------------------
# list_all_profiles
# ---------------------------------------------------------------------------

class TestListAllProfiles:
    def test_returns_presets_by_default(self, prefs):
        all_profiles = prefs.list_all_profiles()
        assert "Expert Pragmatist" in all_profiles
        assert all_profiles["Expert Pragmatist"] == "preset"
        assert len(all_profiles) >= 6  # 6 built-in presets

    def test_includes_custom_profiles(self, prefs):
        profile = ToneProfile(
            name="My Custom",
            description="Custom.",
            formality_level=0.5,
            technical_depth=0.5,
            personality="witty",
            hook_style="problem_statement",
            cta_style="none",
            source="custom",
        )
        prefs.save_custom_profile(profile)
        all_profiles = prefs.list_all_profiles()
        assert "My Custom" in all_profiles
        assert all_profiles["My Custom"] == "custom"


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    def test_clamp(self):
        assert _clamp(5, 0, 10) == 5
        assert _clamp(-1, 0, 10) == 0
        assert _clamp(15, 0, 10) == 10
        assert _clamp(0.5, 0.0, 1.0) == 0.5

    def test_formality_score_formal_text(self):
        text = "therefore consequently furthermore moreover nevertheless"
        score = _formality_score(text)
        assert score > 0.7

    def test_formality_score_informal_text(self):
        text = "gonna wanna kinda sorta gotta yeah nope cool awesome stuff"
        score = _formality_score(text)
        assert score < 0.3

    def test_formality_score_neutral_text(self):
        text = "the algorithm performs matrix multiplication on the data"
        score = _formality_score(text)
        assert score == 0.5  # No formal/informal markers -> neutral

    def test_formality_score_empty(self):
        assert _formality_score("") == 0.5

    def test_tokenize_basic(self):
        text = "First sentence here. Second sentence here. Third sentence here."
        sentences = _tokenize(text)
        assert len(sentences) >= 3
