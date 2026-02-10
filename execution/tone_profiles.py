"""
Tone Profile System - Data model and preset management.

Provides the ToneProfile Pydantic model that captures all dimensions of
writing style, plus a TonePresetManager for loading built-in presets.

Usage:
    from execution.tone_profiles import get_preset, list_presets, ToneProfile

    profile = get_preset("Expert Pragmatist")
    instructions = profile.to_writer_instructions()
    overrides = profile.to_style_overrides()
"""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field, model_validator

from execution.utils.datetime_utils import utc_iso


class SentenceStyle(BaseModel):
    """Sentence-level style parameters."""
    avg_length_target: int = Field(default=18, description="Target average words per sentence")
    burstiness_target: float = Field(default=0.45, ge=0.0, le=1.0, description="Sentence length variance ratio")
    length_variance: str = Field(default="medium", pattern=r"^(high|medium|low)$", description="Variance category")


class VocabularyPreferences(BaseModel):
    """Vocabulary control parameters."""
    preferred_words: List[str] = Field(default_factory=list, description="Words to favor in output")
    avoided_words: List[str] = Field(default_factory=list, description="Words to avoid in output")
    jargon_level: str = Field(default="light", pattern=r"^(none|light|heavy)$", description="Technical jargon density")


class ToneProfile(BaseModel):
    """
    Complete tone/voice profile for content generation.

    Captures all dimensions of writing style: formality, technical depth,
    personality, sentence structure, vocabulary, and content patterns.
    Used by WriterAgent for prompt construction and StyleEnforcerAgent
    for scoring calibration.
    """
    model_config = {"extra": "forbid"}

    # Identity
    name: str = Field(description="Preset name or user-given name")
    description: str = Field(description="Human-readable description, 1-2 sentences")

    # Core dimensions
    formality_level: float = Field(ge=0.0, le=1.0, description="0=casual, 1=formal")
    technical_depth: float = Field(ge=0.0, le=1.0, description="0=layperson, 1=deep technical")
    personality: str = Field(description="e.g. witty, authoritative, conversational, provocative")

    # Structural style
    sentence_style: SentenceStyle = Field(default_factory=SentenceStyle)
    vocabulary_preferences: VocabularyPreferences = Field(default_factory=VocabularyPreferences)

    # Content patterns
    hook_style: str = Field(description="How to open pieces, e.g. contrarian_question, shocking_stat")
    cta_style: str = Field(description="How to close, e.g. specific_question, challenge, none")
    example_phrases: List[str] = Field(default_factory=list, description="3-5 representative phrases")
    forbidden_phrases: List[str] = Field(default_factory=list, description="Phrases to avoid (extends global list)")
    war_story_keywords: List[str] = Field(default_factory=list, description="Authenticity markers")

    # Provenance
    source: str = Field(default="preset", pattern=r"^(preset|inferred|custom)$")
    inferred_from: Optional[str] = Field(default=None, description="URL or filename if inferred")
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="For inferred profiles")
    created_at: str = Field(default_factory=utc_iso)
    updated_at: Optional[str] = Field(default=None)

    def to_writer_instructions(self) -> str:
        """Format the profile into natural language instructions for WriterAgent prompts."""
        # Formality descriptor
        if self.formality_level >= 0.8:
            formality_desc = "formal, polished prose"
        elif self.formality_level >= 0.5:
            formality_desc = "professional but approachable"
        elif self.formality_level >= 0.3:
            formality_desc = "casual and conversational"
        else:
            formality_desc = "very informal, like talking to a friend"

        # Technical depth descriptor
        if self.technical_depth >= 0.8:
            tech_desc = "deep technical detail with code examples and architecture decisions"
        elif self.technical_depth >= 0.5:
            tech_desc = "moderate technical detail accessible to mid-level engineers"
        elif self.technical_depth >= 0.3:
            tech_desc = "light technical context without deep implementation details"
        else:
            tech_desc = "non-technical, focusing on concepts and business impact"

        # Sentence style instruction
        ss = self.sentence_style
        if ss.length_variance == "high":
            rhythm_desc = "Vary sentence length dramatically - mix 5-word punches with 30-word explanations."
        elif ss.length_variance == "low":
            rhythm_desc = f"Keep sentences consistent, around {ss.avg_length_target} words each."
        else:
            rhythm_desc = f"Target ~{ss.avg_length_target} words per sentence with natural variation."

        # Vocabulary instruction
        vp = self.vocabulary_preferences
        vocab_lines = []
        if vp.preferred_words:
            vocab_lines.append(f"Favor these words/phrases: {', '.join(vp.preferred_words[:10])}")
        if vp.avoided_words:
            vocab_lines.append(f"Avoid these words/phrases: {', '.join(vp.avoided_words[:10])}")
        if vp.jargon_level == "heavy":
            vocab_lines.append("Use domain jargon freely - the audience is expert-level.")
        elif vp.jargon_level == "none":
            vocab_lines.append("Avoid all jargon - explain concepts in plain language.")
        vocab_section = "\n".join(f"- {line}" for line in vocab_lines) if vocab_lines else ""

        # Hook instruction
        hook_map = {
            "contrarian_challenge": "Open with a bold, contrarian claim that challenges conventional wisdom.",
            "contrarian_question": "Open with a provocative question that challenges assumptions.",
            "future_vision": "Open with a forward-looking vision of where things are heading.",
            "problem_statement": "Open with a clear, specific problem statement the reader recognizes.",
            "personal_story": "Open with a brief personal anecdote or moment.",
            "breaking_insight": "Open with the most newsworthy insight, lead-style.",
            "myth_busting": "Open by naming a popular belief and immediately dismantling it.",
            "shocking_stat": "Open with a surprising statistic that reframes the topic.",
        }
        hook_instruction = hook_map.get(self.hook_style, f"Open in a '{self.hook_style}' style.")

        # CTA instruction
        cta_map = {
            "specific_question": "End with a specific, answerable question about the reader's situation.",
            "challenge": "End with a direct challenge - dare the reader to try something.",
            "resource_share": "End by pointing to specific tools, repos, or resources.",
            "inspiration": "End with an inspiring, forward-looking statement.",
            "community_discussion": "End by inviting discussion with a low-barrier prompt.",
            "call_to_action": "End with a clear, specific action the reader should take.",
            "none": "End cleanly without an explicit call to action.",
        }
        cta_instruction = cta_map.get(self.cta_style, f"Close in a '{self.cta_style}' style.")

        # Example phrases section
        examples_section = ""
        if self.example_phrases:
            examples_section = "\n\nEXAMPLE PHRASES (match this voice):\n" + "\n".join(
                f'- "{phrase}"' for phrase in self.example_phrases
            )

        # Forbidden phrases section
        forbidden_section = ""
        if self.forbidden_phrases:
            forbidden_section = "\n\nFORBIDDEN PHRASES (never use):\n" + "\n".join(
                f'- "{phrase}"' for phrase in self.forbidden_phrases
            )

        instructions = f"""TONE PROFILE: {self.name}
{self.description}

VOICE & PERSONALITY: {self.personality}
- Formality: {formality_desc}
- Technical depth: {tech_desc}
- {rhythm_desc}

OPENING: {hook_instruction}
CLOSING: {cta_instruction}
{f'''
VOCABULARY:
{vocab_section}''' if vocab_section else ''}
{examples_section}
{forbidden_section}"""

        return instructions.strip()

    def to_style_overrides(self) -> dict:
        """Return overrides for StyleEnforcerAgent scoring parameters.

        Maps tone profile dimensions to the style enforcer's scoring
        thresholds and weights so content is judged against the right
        baseline for this tone.
        """
        ss = self.sentence_style

        # Burstiness target ranges based on variance setting
        burstiness_targets = {
            "high": {"excellent": 0.5, "good": 0.35, "acceptable": 0.25},
            "medium": {"excellent": 0.4, "good": 0.3, "acceptable": 0.2},
            "low": {"excellent": 0.2, "good": 0.15, "acceptable": 0.1},
        }
        burst = burstiness_targets.get(ss.length_variance, burstiness_targets["medium"])

        # Authenticity weight adjustment - profiles that don't use war stories
        # should have lower authenticity weight
        has_war_stories = len(self.war_story_keywords) > 0
        authenticity_weight = 0.25 if has_war_stories else 0.10

        # AI-tell weight increases for formal/professional profiles where
        # AI patterns are more noticeable
        ai_tell_weight = 0.25 if self.formality_level < 0.8 else 0.30

        # Redistribute remaining weight to other dimensions
        remaining = 1.0 - 0.20 - 0.15 - ai_tell_weight - authenticity_weight
        framework_weight = remaining

        return {
            "burstiness_thresholds": burst,
            "avg_sentence_length_target": ss.avg_length_target,
            "dimension_weights": {
                "burstiness": 0.20,
                "lexical_diversity": 0.15,
                "ai_tell": ai_tell_weight,
                "authenticity": authenticity_weight,
                "framework_compliance": framework_weight,
            },
            "war_story_keywords": self.war_story_keywords,
            "forbidden_phrases": self.forbidden_phrases,
        }

    def merge_with_adjustments(self, adjustments: dict) -> "ToneProfile":
        """Return a new profile with learned adjustments applied.

        Args:
            adjustments: Dict of field names to new values. Nested dicts
                         (sentence_style, vocabulary_preferences) are
                         merged shallowly rather than replaced entirely.

        Returns:
            A new ToneProfile with adjustments applied and updated_at set.
        """
        data = self.model_dump()

        for key, value in adjustments.items():
            if key in ("sentence_style", "vocabulary_preferences") and isinstance(value, dict):
                # Shallow merge for nested models
                if key in data and isinstance(data[key], dict):
                    data[key].update(value)
                else:
                    data[key] = value
            else:
                data[key] = value

        data["updated_at"] = utc_iso()
        # If adjustments change the profile, mark it as custom
        if adjustments:
            data["source"] = "custom"

        return ToneProfile(**data)


class TonePresetManager:
    """Loads and manages built-in tone presets from JSON."""

    DEFAULT_PRESETS_PATH = Path(__file__).parent / "tone_presets.json"
    DEFAULT_PRESET_NAME = "Expert Pragmatist"

    def __init__(self, presets_path: Optional[str] = None):
        """Load presets from a JSON file.

        Args:
            presets_path: Path to presets JSON. Defaults to execution/tone_presets.json.
        """
        path = Path(presets_path) if presets_path else self.DEFAULT_PRESETS_PATH
        if not path.exists():
            raise FileNotFoundError(f"Presets file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self._presets: Dict[str, ToneProfile] = {}
        for entry in raw:
            profile = ToneProfile(**entry)
            self._presets[profile.name] = profile

    def list_presets(self) -> List[str]:
        """Return all preset names."""
        return list(self._presets.keys())

    def get_preset(self, name: str) -> ToneProfile:
        """Return a preset by name.

        Raises:
            KeyError: If preset name not found.
        """
        if name not in self._presets:
            available = ", ".join(self._presets.keys())
            raise KeyError(f"Preset '{name}' not found. Available: {available}")
        return self._presets[name]

    def get_default(self) -> ToneProfile:
        """Return the default 'Expert Pragmatist' preset."""
        return self.get_preset(self.DEFAULT_PRESET_NAME)


# ---------------------------------------------------------------------------
# Convenience functions (module-level exports)
# ---------------------------------------------------------------------------

_manager: Optional[TonePresetManager] = None


def _get_manager() -> TonePresetManager:
    """Lazy-load the preset manager singleton."""
    global _manager
    if _manager is None:
        _manager = TonePresetManager()
    return _manager


def list_presets() -> List[str]:
    """Return all built-in preset names."""
    return _get_manager().list_presets()


def get_preset(name: str) -> ToneProfile:
    """Return a built-in preset by name."""
    return _get_manager().get_preset(name)


def get_default_profile() -> ToneProfile:
    """Return the default Expert Pragmatist profile."""
    return _get_manager().get_default()
