"""Tests for tone profile data model and preset management."""
import json
import pytest
from pathlib import Path
from pydantic import ValidationError

from execution.tone_profiles import (
    ToneProfile,
    SentenceStyle,
    VocabularyPreferences,
    TonePresetManager,
    list_presets,
    get_preset,
    get_default_profile,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def expert_profile():
    """Return the Expert Pragmatist preset."""
    return get_preset("Expert Pragmatist")


@pytest.fixture
def minimal_profile_data():
    """Minimal valid data for ToneProfile construction."""
    return {
        "name": "Test Profile",
        "description": "A test profile for unit tests.",
        "formality_level": 0.5,
        "technical_depth": 0.5,
        "personality": "conversational",
        "hook_style": "problem_statement",
        "cta_style": "none",
    }


@pytest.fixture
def custom_presets_file(tmp_path):
    """Create a temporary presets JSON file with one preset."""
    data = [
        {
            "name": "Custom Test",
            "description": "A custom test preset.",
            "formality_level": 0.4,
            "technical_depth": 0.6,
            "personality": "witty",
            "hook_style": "contrarian_question",
            "cta_style": "challenge",
            "source": "preset",
        }
    ]
    path = tmp_path / "test_presets.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# ToneProfile model validation
# ---------------------------------------------------------------------------

class TestToneProfileModel:
    def test_create_with_minimal_fields(self, minimal_profile_data):
        profile = ToneProfile(**minimal_profile_data)
        assert profile.name == "Test Profile"
        assert profile.formality_level == 0.5
        assert profile.source == "preset"

    def test_default_sentence_style(self, minimal_profile_data):
        profile = ToneProfile(**minimal_profile_data)
        assert profile.sentence_style.avg_length_target == 18
        assert profile.sentence_style.burstiness_target == 0.45
        assert profile.sentence_style.length_variance == "medium"

    def test_default_vocabulary_preferences(self, minimal_profile_data):
        profile = ToneProfile(**minimal_profile_data)
        assert profile.vocabulary_preferences.preferred_words == []
        assert profile.vocabulary_preferences.avoided_words == []
        assert profile.vocabulary_preferences.jargon_level == "light"

    def test_formality_level_validation_too_high(self, minimal_profile_data):
        minimal_profile_data["formality_level"] = 1.5
        with pytest.raises(ValidationError):
            ToneProfile(**minimal_profile_data)

    def test_formality_level_validation_too_low(self, minimal_profile_data):
        minimal_profile_data["formality_level"] = -0.1
        with pytest.raises(ValidationError):
            ToneProfile(**minimal_profile_data)

    def test_technical_depth_validation_too_high(self, minimal_profile_data):
        minimal_profile_data["technical_depth"] = 1.1
        with pytest.raises(ValidationError):
            ToneProfile(**minimal_profile_data)

    def test_formality_at_boundaries(self, minimal_profile_data):
        minimal_profile_data["formality_level"] = 0.0
        p = ToneProfile(**minimal_profile_data)
        assert p.formality_level == 0.0

        minimal_profile_data["formality_level"] = 1.0
        p = ToneProfile(**minimal_profile_data)
        assert p.formality_level == 1.0

    def test_invalid_source_value(self, minimal_profile_data):
        minimal_profile_data["source"] = "unknown"
        with pytest.raises(ValidationError):
            ToneProfile(**minimal_profile_data)

    def test_invalid_length_variance(self, minimal_profile_data):
        minimal_profile_data["sentence_style"] = {
            "avg_length_target": 18,
            "burstiness_target": 0.45,
            "length_variance": "extreme",
        }
        with pytest.raises(ValidationError):
            ToneProfile(**minimal_profile_data)

    def test_invalid_jargon_level(self, minimal_profile_data):
        minimal_profile_data["vocabulary_preferences"] = {
            "jargon_level": "moderate"
        }
        with pytest.raises(ValidationError):
            ToneProfile(**minimal_profile_data)

    def test_extra_fields_forbidden(self, minimal_profile_data):
        minimal_profile_data["unknown_field"] = "value"
        with pytest.raises(ValidationError):
            ToneProfile(**minimal_profile_data)

    def test_confidence_score_bounds(self, minimal_profile_data):
        minimal_profile_data["confidence_score"] = 1.5
        with pytest.raises(ValidationError):
            ToneProfile(**minimal_profile_data)

    def test_burstiness_target_bounds(self, minimal_profile_data):
        minimal_profile_data["sentence_style"] = {
            "burstiness_target": 1.5,
        }
        with pytest.raises(ValidationError):
            ToneProfile(**minimal_profile_data)

    def test_custom_profile_all_fields(self):
        profile = ToneProfile(
            name="Full Custom",
            description="All fields specified.",
            formality_level=0.8,
            technical_depth=0.9,
            personality="analytical",
            sentence_style=SentenceStyle(
                avg_length_target=20,
                burstiness_target=0.35,
                length_variance="low",
            ),
            vocabulary_preferences=VocabularyPreferences(
                preferred_words=["benchmark", "throughput"],
                avoided_words=["leverage", "synergy"],
                jargon_level="heavy",
            ),
            hook_style="shocking_stat",
            cta_style="resource_share",
            example_phrases=["Example one.", "Example two."],
            forbidden_phrases=["Don't say this"],
            war_story_keywords=["I debugged", "we shipped"],
            source="custom",
            inferred_from=None,
            confidence_score=0.9,
        )
        assert profile.name == "Full Custom"
        assert profile.sentence_style.length_variance == "low"
        assert len(profile.example_phrases) == 2
        assert profile.source == "custom"
        assert profile.confidence_score == 0.9


# ---------------------------------------------------------------------------
# to_writer_instructions()
# ---------------------------------------------------------------------------

class TestWriterInstructions:
    def test_returns_string(self, expert_profile):
        instructions = expert_profile.to_writer_instructions()
        assert isinstance(instructions, str)
        assert len(instructions) > 100

    def test_contains_profile_name(self, expert_profile):
        instructions = expert_profile.to_writer_instructions()
        assert "Expert Pragmatist" in instructions

    def test_formality_descriptor_formal(self, minimal_profile_data):
        minimal_profile_data["formality_level"] = 0.9
        p = ToneProfile(**minimal_profile_data)
        instructions = p.to_writer_instructions()
        assert "formal" in instructions.lower()

    def test_formality_descriptor_casual(self, minimal_profile_data):
        minimal_profile_data["formality_level"] = 0.2
        p = ToneProfile(**minimal_profile_data)
        instructions = p.to_writer_instructions()
        assert "informal" in instructions.lower() or "friend" in instructions.lower()

    def test_tech_depth_high(self, minimal_profile_data):
        minimal_profile_data["technical_depth"] = 0.9
        p = ToneProfile(**minimal_profile_data)
        instructions = p.to_writer_instructions()
        assert "deep technical" in instructions.lower() or "code" in instructions.lower()

    def test_tech_depth_low(self, minimal_profile_data):
        minimal_profile_data["technical_depth"] = 0.1
        p = ToneProfile(**minimal_profile_data)
        instructions = p.to_writer_instructions()
        assert "non-technical" in instructions.lower() or "business" in instructions.lower()

    def test_high_variance_instruction(self, minimal_profile_data):
        minimal_profile_data["sentence_style"] = {
            "avg_length_target": 18,
            "burstiness_target": 0.5,
            "length_variance": "high",
        }
        p = ToneProfile(**minimal_profile_data)
        instructions = p.to_writer_instructions()
        assert "dramatically" in instructions.lower() or "vary" in instructions.lower()

    def test_low_variance_instruction(self, minimal_profile_data):
        minimal_profile_data["sentence_style"] = {
            "avg_length_target": 20,
            "burstiness_target": 0.2,
            "length_variance": "low",
        }
        p = ToneProfile(**minimal_profile_data)
        instructions = p.to_writer_instructions()
        assert "consistent" in instructions.lower()

    def test_preferred_words_included(self):
        p = ToneProfile(
            name="Vocab Test",
            description="Test vocab.",
            formality_level=0.5,
            technical_depth=0.5,
            personality="witty",
            hook_style="problem_statement",
            cta_style="none",
            vocabulary_preferences=VocabularyPreferences(
                preferred_words=["tradeoff", "shipped"],
            ),
        )
        instructions = p.to_writer_instructions()
        assert "tradeoff" in instructions
        assert "shipped" in instructions

    def test_forbidden_phrases_section(self):
        p = ToneProfile(
            name="Forbidden Test",
            description="Test forbidden.",
            formality_level=0.5,
            technical_depth=0.5,
            personality="witty",
            hook_style="problem_statement",
            cta_style="none",
            forbidden_phrases=["Let's dive in", "Game-changer"],
        )
        instructions = p.to_writer_instructions()
        assert "FORBIDDEN" in instructions
        assert "Let's dive in" in instructions

    def test_example_phrases_section(self):
        p = ToneProfile(
            name="Example Test",
            description="Test examples.",
            formality_level=0.5,
            technical_depth=0.5,
            personality="witty",
            hook_style="problem_statement",
            cta_style="none",
            example_phrases=["We burned $12K in GPU credits."],
        )
        instructions = p.to_writer_instructions()
        assert "EXAMPLE PHRASES" in instructions
        assert "$12K" in instructions

    def test_hook_style_mapping(self, minimal_profile_data):
        for style in ["contrarian_challenge", "future_vision", "shocking_stat"]:
            minimal_profile_data["hook_style"] = style
            p = ToneProfile(**minimal_profile_data)
            instructions = p.to_writer_instructions()
            assert "OPENING:" in instructions

    def test_cta_style_mapping(self, minimal_profile_data):
        for style in ["specific_question", "challenge", "none"]:
            minimal_profile_data["cta_style"] = style
            p = ToneProfile(**minimal_profile_data)
            instructions = p.to_writer_instructions()
            assert "CLOSING:" in instructions


# ---------------------------------------------------------------------------
# to_style_overrides()
# ---------------------------------------------------------------------------

class TestStyleOverrides:
    def test_returns_dict(self, expert_profile):
        overrides = expert_profile.to_style_overrides()
        assert isinstance(overrides, dict)

    def test_has_expected_keys(self, expert_profile):
        overrides = expert_profile.to_style_overrides()
        assert "burstiness_thresholds" in overrides
        assert "avg_sentence_length_target" in overrides
        assert "dimension_weights" in overrides
        assert "war_story_keywords" in overrides
        assert "forbidden_phrases" in overrides

    def test_burstiness_thresholds_high_variance(self):
        p = ToneProfile(
            name="High",
            description="High variance.",
            formality_level=0.5,
            technical_depth=0.5,
            personality="witty",
            hook_style="problem_statement",
            cta_style="none",
            sentence_style=SentenceStyle(length_variance="high"),
        )
        overrides = p.to_style_overrides()
        assert overrides["burstiness_thresholds"]["excellent"] == 0.5

    def test_burstiness_thresholds_low_variance(self):
        p = ToneProfile(
            name="Low",
            description="Low variance.",
            formality_level=0.5,
            technical_depth=0.5,
            personality="witty",
            hook_style="problem_statement",
            cta_style="none",
            sentence_style=SentenceStyle(length_variance="low"),
        )
        overrides = p.to_style_overrides()
        assert overrides["burstiness_thresholds"]["excellent"] == 0.2

    def test_dimension_weights_sum_to_one(self, expert_profile):
        overrides = expert_profile.to_style_overrides()
        weights = overrides["dimension_weights"]
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01

    def test_no_war_stories_lowers_authenticity_weight(self):
        p = ToneProfile(
            name="No War",
            description="No war stories.",
            formality_level=0.5,
            technical_depth=0.5,
            personality="objective",
            hook_style="breaking_insight",
            cta_style="none",
            war_story_keywords=[],
        )
        overrides = p.to_style_overrides()
        assert overrides["dimension_weights"]["authenticity"] == 0.10

    def test_with_war_stories_higher_authenticity_weight(self, expert_profile):
        overrides = expert_profile.to_style_overrides()
        assert overrides["dimension_weights"]["authenticity"] == 0.25

    def test_formal_profile_higher_ai_tell_weight(self):
        p = ToneProfile(
            name="Formal",
            description="Very formal.",
            formality_level=0.9,
            technical_depth=0.5,
            personality="authoritative",
            hook_style="problem_statement",
            cta_style="none",
        )
        overrides = p.to_style_overrides()
        assert overrides["dimension_weights"]["ai_tell"] == 0.30


# ---------------------------------------------------------------------------
# merge_with_adjustments()
# ---------------------------------------------------------------------------

class TestMergeWithAdjustments:
    def test_returns_new_profile(self, expert_profile):
        merged = expert_profile.merge_with_adjustments({"formality_level": 0.3})
        assert merged is not expert_profile
        assert merged.formality_level == 0.3
        assert expert_profile.formality_level == 0.6  # original unchanged

    def test_marks_as_custom(self, expert_profile):
        merged = expert_profile.merge_with_adjustments({"formality_level": 0.3})
        assert merged.source == "custom"

    def test_sets_updated_at(self, expert_profile):
        merged = expert_profile.merge_with_adjustments({"formality_level": 0.3})
        assert merged.updated_at is not None

    def test_empty_adjustments_keeps_source(self, expert_profile):
        merged = expert_profile.merge_with_adjustments({})
        # Empty adjustments: no source change expected
        assert merged.source == expert_profile.source

    def test_nested_sentence_style_merge(self, expert_profile):
        merged = expert_profile.merge_with_adjustments({
            "sentence_style": {"avg_length_target": 25}
        })
        assert merged.sentence_style.avg_length_target == 25
        # Other sentence_style fields should be preserved
        assert merged.sentence_style.length_variance == expert_profile.sentence_style.length_variance

    def test_nested_vocabulary_merge(self, expert_profile):
        merged = expert_profile.merge_with_adjustments({
            "vocabulary_preferences": {"jargon_level": "none"}
        })
        assert merged.vocabulary_preferences.jargon_level == "none"
        # Preferred words should be preserved
        assert len(merged.vocabulary_preferences.preferred_words) == len(
            expert_profile.vocabulary_preferences.preferred_words
        )


# ---------------------------------------------------------------------------
# TonePresetManager
# ---------------------------------------------------------------------------

class TestTonePresetManager:
    def test_loads_default_presets(self):
        manager = TonePresetManager()
        names = manager.list_presets()
        assert len(names) == 6

    def test_expected_preset_names(self):
        manager = TonePresetManager()
        names = manager.list_presets()
        assert "Expert Pragmatist" in names
        assert "Thought Leader" in names
        assert "Technical Deep Dive" in names
        assert "Conversational Engineer" in names
        assert "News Reporter" in names
        assert "Contrarian Challenger" in names

    def test_get_preset_returns_tone_profile(self):
        manager = TonePresetManager()
        profile = manager.get_preset("Expert Pragmatist")
        assert isinstance(profile, ToneProfile)
        assert profile.name == "Expert Pragmatist"

    def test_get_preset_invalid_name(self):
        manager = TonePresetManager()
        with pytest.raises(KeyError, match="not found"):
            manager.get_preset("Nonexistent Preset")

    def test_get_default(self):
        manager = TonePresetManager()
        profile = manager.get_default()
        assert profile.name == "Expert Pragmatist"

    def test_custom_presets_file(self, custom_presets_file):
        manager = TonePresetManager(presets_path=custom_presets_file)
        names = manager.list_presets()
        assert names == ["Custom Test"]
        profile = manager.get_preset("Custom Test")
        assert profile.formality_level == 0.4

    def test_missing_presets_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            TonePresetManager(presets_path=str(tmp_path / "nonexistent.json"))

    def test_all_presets_produce_valid_instructions(self):
        manager = TonePresetManager()
        for name in manager.list_presets():
            profile = manager.get_preset(name)
            instructions = profile.to_writer_instructions()
            assert isinstance(instructions, str)
            assert len(instructions) > 50

    def test_all_presets_produce_valid_overrides(self):
        manager = TonePresetManager()
        for name in manager.list_presets():
            profile = manager.get_preset(name)
            overrides = profile.to_style_overrides()
            weights = overrides["dimension_weights"]
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.01, f"{name} weights sum to {total}"


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

class TestConvenienceFunctions:
    def test_list_presets(self):
        names = list_presets()
        assert isinstance(names, list)
        assert len(names) == 6

    def test_get_preset(self):
        profile = get_preset("Thought Leader")
        assert profile.name == "Thought Leader"

    def test_get_preset_invalid(self):
        with pytest.raises(KeyError):
            get_preset("Does Not Exist")

    def test_get_default_profile(self):
        profile = get_default_profile()
        assert profile.name == "Expert Pragmatist"
        assert isinstance(profile, ToneProfile)
