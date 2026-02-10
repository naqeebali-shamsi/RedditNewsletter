"""
User Preferences - Persists tone preferences with adaptive learning.

Manages the active tone profile, custom profiles, feedback logging,
and micro-adjustments learned from user edits over time.

Storage: JSON file at project root / execution / user_preferences.json

Usage:
    from execution.user_preferences import UserPreferences

    prefs = UserPreferences()
    profile = prefs.get_active_profile()
    prefs.log_feedback("article-123", "edited", original="...", edited="...")
"""

import json
import re
import statistics
import threading
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from execution.tone_profiles import ToneProfile, get_preset, list_presets
from execution.utils.datetime_utils import utc_iso

# Optional NLP dependencies
try:
    from nltk.tokenize import sent_tokenize
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False


# ---------------------------------------------------------------------------
# Pydantic models for preferences storage
# ---------------------------------------------------------------------------

class LearnedAdjustments(BaseModel):
    """Accumulated micro-adjustments from user feedback."""
    formality_delta: float = Field(default=0.0, ge=-0.5, le=0.5)
    technical_depth_delta: float = Field(default=0.0, ge=-0.5, le=0.5)
    avg_sentence_length_delta: int = Field(default=0, ge=-10, le=10)
    burstiness_delta: float = Field(default=0.0, ge=-0.3, le=0.3)


class FeedbackEntry(BaseModel):
    """A single feedback event."""
    article_id: str
    action: str  # accepted, edited, rejected, tone_adjustment
    timestamp: str = Field(default_factory=utc_iso)
    details: Optional[str] = None


class PreferencesData(BaseModel):
    """Root schema for user_preferences.json."""
    user_id: str = "default"
    active_profile: str = "Expert Pragmatist"
    custom_profiles: Dict[str, dict] = Field(default_factory=dict)
    preference_history: List[str] = Field(default_factory=list)
    feedback_log: List[dict] = Field(default_factory=list)
    learned_adjustments: LearnedAdjustments = Field(default_factory=LearnedAdjustments)
    feedback_count: int = 0
    learning_active: bool = False
    created_at: str = Field(default_factory=utc_iso)
    updated_at: str = Field(default_factory=utc_iso)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class UserPreferences:
    """Persists and manages user tone preferences with adaptive learning.

    Thread-safe: all file writes are guarded by a lock.
    """

    PREFERENCES_PATH = Path(__file__).parent.parent / "user_preferences.json"
    _LEARNING_THRESHOLD = 5  # Feedback count before learning activates
    _MAX_ADJUSTMENT = 0.1    # Max adjustment per learning cycle

    def __init__(self, path: Optional[Path] = None):
        self._path = path or self.PREFERENCES_PATH
        self._lock = threading.Lock()
        self._data: PreferencesData = self._load()

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def _load(self) -> PreferencesData:
        """Load preferences from disk, or create defaults."""
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                return PreferencesData(**raw)
            except (json.JSONDecodeError, Exception):
                pass
        return PreferencesData()

    def _save(self) -> None:
        """Persist current preferences to disk (must hold lock)."""
        self._data.updated_at = utc_iso()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._data.model_dump(), indent=2, default=str),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Profile management
    # ------------------------------------------------------------------

    def get_active_profile(self) -> ToneProfile:
        """Get the currently active tone profile.

        Looks up custom profiles first, then built-in presets.

        Returns:
            The active ToneProfile.

        Raises:
            KeyError: If the active profile name isn't found anywhere.
        """
        name = self._data.active_profile

        # Check custom profiles first
        if name in self._data.custom_profiles:
            return ToneProfile(**self._data.custom_profiles[name])

        # Fall back to built-in presets
        return get_preset(name)

    def set_active_profile(self, name: str) -> None:
        """Switch the active tone profile.

        Args:
            name: Profile name (must exist in presets or custom profiles).

        Raises:
            KeyError: If the name isn't found in presets or custom profiles.
        """
        # Validate the name exists somewhere
        if name not in self._data.custom_profiles and name not in list_presets():
            available = list(self._data.custom_profiles.keys()) + list_presets()
            raise KeyError(
                f"Profile '{name}' not found. Available: {', '.join(available)}"
            )

        with self._lock:
            # Log the switch
            if self._data.active_profile != name:
                self._data.preference_history.append(
                    f"{utc_iso()}: switched from '{self._data.active_profile}' to '{name}'"
                )
            self._data.active_profile = name
            self._save()

    def save_custom_profile(self, profile: ToneProfile) -> None:
        """Save a custom or inferred profile.

        Args:
            profile: The ToneProfile to persist. Its name is used as the key.
        """
        with self._lock:
            self._data.custom_profiles[profile.name] = profile.model_dump()
            self._save()

    def delete_custom_profile(self, name: str) -> None:
        """Delete a custom profile.

        Args:
            name: Profile name to delete.

        Raises:
            KeyError: If the name isn't in custom profiles.
        """
        if name not in self._data.custom_profiles:
            raise KeyError(f"Custom profile '{name}' not found")

        with self._lock:
            del self._data.custom_profiles[name]
            # If it was the active profile, fall back to default
            if self._data.active_profile == name:
                self._data.active_profile = "Expert Pragmatist"
            self._save()

    def list_all_profiles(self) -> Dict[str, str]:
        """List all available profiles (presets + custom).

        Returns:
            Dict mapping profile name to source ("preset" or "custom").
        """
        result = {}
        for name in list_presets():
            result[name] = "preset"
        for name in self._data.custom_profiles:
            result[name] = "custom"
        return result

    # ------------------------------------------------------------------
    # Feedback and learning
    # ------------------------------------------------------------------

    def log_feedback(
        self,
        article_id: str,
        action: str,
        original: Optional[str] = None,
        edited: Optional[str] = None,
    ) -> None:
        """Log user feedback on generated content.

        Args:
            article_id: ID of the article.
            action: One of 'accepted', 'edited', 'rejected', 'tone_adjustment'.
            original: Original generated text (for 'edited' action).
            edited: User-edited text (for 'edited' action).
        """
        entry = FeedbackEntry(article_id=article_id, action=action)

        with self._lock:
            self._data.feedback_log.append(entry.model_dump())
            self._data.feedback_count += 1

            # Activate learning after threshold
            if self._data.feedback_count >= self._LEARNING_THRESHOLD:
                self._data.learning_active = True

            # Learn from edits if we have both texts
            if action == "edited" and original and edited and self._data.learning_active:
                self._learn_from_edits(original, edited)

            self._save()

    def get_effective_profile(self) -> ToneProfile:
        """Get active profile with learned adjustments applied.

        If learning is active and adjustments have been accumulated,
        returns a modified copy of the active profile.
        """
        base = self.get_active_profile()

        if not self._data.learning_active:
            return base

        adj = self._data.learned_adjustments
        # Only apply if there are non-zero adjustments
        has_adjustments = any([
            adj.formality_delta != 0.0,
            adj.technical_depth_delta != 0.0,
            adj.avg_sentence_length_delta != 0,
            adj.burstiness_delta != 0.0,
        ])

        if not has_adjustments:
            return base

        return self._apply_adjustments(base, adj)

    def _learn_from_edits(self, original: str, edited: str) -> None:
        """Compare original vs edited content to learn preferences.

        Detects:
        - Formality changes (formal/informal word ratios)
        - Sentence length changes (average length shift)
        - Burstiness changes (variance shift)

        All analysis is deterministic (no LLM calls).
        Must hold self._lock when called.
        """
        # Sentence analysis
        orig_sentences = _tokenize(original)
        edit_sentences = _tokenize(edited)

        if len(orig_sentences) < 3 or len(edit_sentences) < 3:
            return

        orig_lengths = [len(s.split()) for s in orig_sentences]
        edit_lengths = [len(s.split()) for s in edit_sentences]

        orig_avg = statistics.mean(orig_lengths)
        edit_avg = statistics.mean(edit_lengths)

        orig_std = statistics.stdev(orig_lengths) if len(orig_lengths) > 1 else 0
        edit_std = statistics.stdev(edit_lengths) if len(edit_lengths) > 1 else 0

        orig_burst = orig_std / orig_avg if orig_avg > 0 else 0
        edit_burst = edit_std / edit_avg if edit_avg > 0 else 0

        # Formality heuristic: ratio of formal vs informal words
        orig_formality = _formality_score(original)
        edit_formality = _formality_score(edited)

        adj = self._data.learned_adjustments

        # Apply micro-adjustments (clamped to _MAX_ADJUSTMENT per cycle)
        cap = self._MAX_ADJUSTMENT

        # Sentence length delta
        length_diff = edit_avg - orig_avg
        if abs(length_diff) > 2:
            delta = max(-cap * 10, min(cap * 10, length_diff * 0.3))
            adj.avg_sentence_length_delta = _clamp(
                adj.avg_sentence_length_delta + int(delta), -10, 10
            )

        # Burstiness delta
        burst_diff = edit_burst - orig_burst
        if abs(burst_diff) > 0.05:
            delta = max(-cap, min(cap, burst_diff * 0.3))
            adj.burstiness_delta = _clamp(
                adj.burstiness_delta + delta, -0.3, 0.3
            )

        # Formality delta
        form_diff = edit_formality - orig_formality
        if abs(form_diff) > 0.05:
            delta = max(-cap, min(cap, form_diff * 0.3))
            adj.formality_delta = _clamp(
                adj.formality_delta + delta, -0.5, 0.5
            )

    def _apply_adjustments(self, profile: ToneProfile, adj: LearnedAdjustments) -> ToneProfile:
        """Apply learned adjustments to a profile.

        Returns a new ToneProfile with adjustments merged in.
        """
        adjustments = {}

        # Formality: clamp to [0, 1]
        if adj.formality_delta != 0.0:
            adjustments["formality_level"] = _clamp(
                profile.formality_level + adj.formality_delta, 0.0, 1.0
            )

        # Technical depth: clamp to [0, 1]
        if adj.technical_depth_delta != 0.0:
            adjustments["technical_depth"] = _clamp(
                profile.technical_depth + adj.technical_depth_delta, 0.0, 1.0
            )

        # Sentence style adjustments
        ss_updates = {}
        if adj.avg_sentence_length_delta != 0:
            new_len = max(5, profile.sentence_style.avg_length_target + adj.avg_sentence_length_delta)
            ss_updates["avg_length_target"] = new_len

        if adj.burstiness_delta != 0.0:
            new_burst = _clamp(
                profile.sentence_style.burstiness_target + adj.burstiness_delta,
                0.05, 0.95,
            )
            ss_updates["burstiness_target"] = round(new_burst, 3)

        if ss_updates:
            adjustments["sentence_style"] = ss_updates

        if not adjustments:
            return profile

        return profile.merge_with_adjustments(adjustments)

    def get_feedback_stats(self) -> dict:
        """Get summary statistics on feedback."""
        log = self._data.feedback_log
        total = len(log)
        if total == 0:
            return {"total": 0, "accepted": 0, "edited": 0, "rejected": 0}

        accepted = sum(1 for e in log if e.get("action") == "accepted")
        edited = sum(1 for e in log if e.get("action") == "edited")
        rejected = sum(1 for e in log if e.get("action") == "rejected")

        return {
            "total": total,
            "accepted": accepted,
            "edited": edited,
            "rejected": rejected,
            "acceptance_rate": accepted / total if total else 0,
            "learning_active": self._data.learning_active,
            "adjustments": self._data.learned_adjustments.model_dump(),
        }

    def reset_learning(self) -> None:
        """Reset all learned adjustments (keeps feedback log)."""
        with self._lock:
            self._data.learned_adjustments = LearnedAdjustments()
            self._save()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Split text into sentences (reuses style_enforcer logic)."""
    if HAS_NLTK:
        try:
            return sent_tokenize(text)
        except Exception:
            pass
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]


def _clamp(value, minimum, maximum):
    """Clamp a value between minimum and maximum."""
    return max(minimum, min(maximum, value))


# Formal vs informal word sets for formality heuristic
_FORMAL_WORDS = frozenset([
    "therefore", "consequently", "furthermore", "moreover", "nevertheless",
    "notwithstanding", "accordingly", "hereby", "wherein", "whereas",
    "henceforth", "subsequently", "aforementioned", "pertaining",
    "demonstrate", "facilitate", "implement", "utilize", "endeavor",
])

_INFORMAL_WORDS = frozenset([
    "gonna", "wanna", "kinda", "sorta", "gotta", "yeah", "nope",
    "cool", "awesome", "stuff", "things", "basically", "literally",
    "super", "totally", "pretty", "huge", "tons", "bunch", "guys",
])


def _formality_score(text: str) -> float:
    """Quick formality heuristic based on word choice.

    Returns a float 0.0 (very informal) to 1.0 (very formal).
    """
    words = text.lower().split()
    if not words:
        return 0.5

    formal_count = sum(1 for w in words if w in _FORMAL_WORDS)
    informal_count = sum(1 for w in words if w in _INFORMAL_WORDS)

    total_markers = formal_count + informal_count
    if total_markers == 0:
        return 0.5  # Neutral

    return formal_count / total_markers
